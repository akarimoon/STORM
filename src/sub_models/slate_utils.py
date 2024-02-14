from typing import Optional
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

from sub_models.attention_blocks import get_vector_mask
from sub_models.attention_blocks import PositionalEncoding1D


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    
    m = nn.Linear(in_features, out_features, bias)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0., gain=1.):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)
    
    def forward(self, q, k, v, attn_mask=None):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """

        B, T, d_q = q.shape
        _, S, _ = k.shape
        
        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)
        
        attn = torch.matmul(q*(d_q**(-0.5)), k.transpose(-1, -2))
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        
        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        output = self.output_dropout(self.proj_o(output))
        return output
    

class FlashCausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0., gain=1.):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)
    
    def forward(self, q, k, v, attn_mask=None):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """

        B, T, d_q = q.shape
        _, S, _ = k.shape
        
        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)
        
        if self.training:
            dropout = self.dropout
            is_causal = True
        else:
            dropout = 0.0
            is_causal = False

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=is_causal)
        output = output.transpose(1, 2).reshape(B, T, -1)
        output = self.output_dropout(self.proj_o(output))
        return output


class TransformerDecoderBlock(nn.Module):
    def __init__(self, max_len, d_model, num_heads, dropout=0., gain=1., is_first=False, use_flash_attn=False):
        super().__init__()
        
        self.is_first = is_first
        
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        if use_flash_attn:
            self.self_attn = FlashCausalMultiHeadAttention(d_model, num_heads, dropout, gain)
        else:
            self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        
        mask = torch.triu(torch.ones((max_len, max_len), dtype=torch.bool), diagonal=1)
        self.self_attn_mask = nn.Parameter(mask, requires_grad=False)
        
        self.encoder_decoder_attn_layer_norm = nn.LayerNorm(d_model)
        if use_flash_attn:
            self.encoder_decoder_attn = FlashCausalMultiHeadAttention(d_model, num_heads, dropout, gain)
        else:
            self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4*d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(4*d_model, d_model, gain=gain),
            nn.Dropout(dropout))
    
    def forward(self, input, encoder_output):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        T = input.shape[1]
        
        if self.is_first:
            input = self.self_attn_layer_norm(input)
            x = self.self_attn(input, input, input, self.self_attn_mask[:T, :T])
            input = input + x
        else:
            x = self.self_attn_layer_norm(input)
            x = self.self_attn(x, x, x, self.self_attn_mask[:T, :T])
            input = input + x
        
        x = self.encoder_decoder_attn_layer_norm(input)
        x = self.encoder_decoder_attn(x, encoder_output, encoder_output)
        input = input + x
        
        x = self.ffn_layer_norm(input)
        x = self.ffn(x)
        return input + x
    

class SLATETransformerDecoder(nn.Module):
    def __init__(self, feat_dim, num_slots, num_layers, num_heads, max_length, dropout=0., use_flash_attn=False):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_slots = num_slots
        
        gain = (3 * num_layers) ** (-0.5)
        self.layer_stack = nn.ModuleList(
            [TransformerDecoderBlock(max_length, feat_dim, num_heads, dropout, gain, is_first=True, use_flash_attn=use_flash_attn)] +
            [TransformerDecoderBlock(max_length, feat_dim, num_heads, dropout, gain, is_first=False, use_flash_attn=use_flash_attn)
                for _ in range(num_layers - 1)])
        
        self.layer_norm = nn.LayerNorm(feat_dim)

    def forward(self, input, encoder_output):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        for layer in self.layer_stack:
            input = layer(input, encoder_output)
        
        return self.layer_norm(input)
    

class SLATETransformerDecoderKVCache(nn.Module):
    def __init__(self, feat_dim, num_slots, num_layers, num_heads, max_length, dropout=0.):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_slots = num_slots
        
        gain = (3 * num_layers) ** (-0.5)
        self.layer_stack = nn.ModuleList(
            [TransformerDecoderBlock(max_length, feat_dim, num_heads, dropout, gain, is_first=True)] +
            [TransformerDecoderBlock(max_length, feat_dim, num_heads, dropout, gain, is_first=False)
                for _ in range(num_layers - 1)])
        
        self.layer_norm = nn.LayerNorm(feat_dim)

        self.position_encoding = PositionalEncoding1D(max_length=1+feat_dim, embed_dim=feat_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input, encoder_output):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        for layer in self.layer_stack:
            input = layer(input, encoder_output)
        
        return self.layer_norm(input)
    
    def reset_kv_cache_list(self, batch_size, dtype, device):
        '''
        Reset self.kv_cache_list
        '''
        self.kv_cache_list = []
        for layer in self.layer_stack:
            self.kv_cache_list.append(torch.zeros(size=(batch_size, 0, self.feat_dim), dtype=dtype, device=device))

    def forward_with_kv_cache(self, input, encoder_output):
        '''
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        '''
        assert input.shape[1] == 1

        input = self.position_encoding.forward_with_position(input, position=self.kv_cache_list[0].shape[1]//self.num_slots)
        input = self.dropout(input)

        for idx, layer in enumerate(self.layer_stack):
            self.kv_cache_list[idx] = torch.cat([self.kv_cache_list[idx], encoder_output], dim=1)
            input = layer(input, encoder_output)

        return self.layer_norm(input)


def resize_patches_to_image(patches: torch.Tensor, size: Optional[int] = None, 
                            scale_factor: Optional[float] = None, resize_mode: str = "bilinear") -> torch.Tensor:
    """Convert and resize a tensor of patches to image shape.

    This method requires that the patches can be converted to a square image.

    Args:
        patches: Patches to be converted of shape (..., C, P), where C is the number of channels and
            P the number of patches.
        size: Image size to resize to.
        scale_factor: Scale factor by which to resize the patches. Can be specified alternatively to
            `size`.
        resize_mode: Method to resize with. Valid options are "nearest", "nearest-exact", "bilinear",
            "bicubic".

    Returns:
        Tensor of shape (..., C, S, S) where S is the image size.
    """
    has_size = size is None
    has_scale = scale_factor is None
    if has_size == has_scale:
        raise ValueError("Exactly one of `size` or `scale_factor` must be specified.")

    n_channels = patches.shape[-2]
    n_patches = patches.shape[-1]
    patch_size_float = sqrt(n_patches)
    patch_size = int(sqrt(n_patches))
    if patch_size_float != patch_size:
        raise ValueError("The number of patches needs to be a perfect square.")

    image = torch.nn.functional.interpolate(
        patches.view(-1, n_channels, patch_size, patch_size),
        size=size,
        scale_factor=scale_factor,
        mode=resize_mode,
    )

    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])