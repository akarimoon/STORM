import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from sub_models.attention_blocks import get_vector_mask
from sub_models.attention_blocks import PositionalEncoding1D, AttentionBlock, AttentionBlockKVCache, OCPositionalEncoding1D, OCPositionalEncoding1D2Emb, OCRepeatedPositionalEncoding1D


class StochasticTransformer(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout):
        super().__init__()
        self.action_dim = action_dim

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlock(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)  # TODO: check if this is necessary

        self.head = nn.Linear(feat_dim, stoch_dim)

    def forward(self, samples, action, mask):
        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for enc_layer in self.layer_stack:
            feats, attn = enc_layer(feats, mask)

        feat = self.head(feats)
        return feat


class StochasticTransformerKVCache(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout):
        super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlockKVCache(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)  # TODO: check if this is necessary

    def forward(self, samples, action, mask):
        '''
        Normal forward pass
        '''
        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, mask)

        return feats

    def reset_kv_cache_list(self, batch_size, dtype, device):
        '''
        Reset self.kv_cache_list
        '''
        self.kv_cache_list = []
        for layer in self.layer_stack:
            self.kv_cache_list.append(torch.zeros(size=(batch_size, 0, self.feat_dim), dtype=dtype, device=device))

    def forward_with_kv_cache(self, samples, action):
        '''
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        '''
        assert samples.shape[1] == 1
        mask = get_vector_mask(self.kv_cache_list[0].shape[1]+1, samples.device)

        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding.forward_with_position(feats, position=self.kv_cache_list[0].shape[1])
        feats = self.layer_norm(feats)

        for idx, layer in enumerate(self.layer_stack):
            self.kv_cache_list[idx] = torch.cat([self.kv_cache_list[idx], feats], dim=1)
            feats, attn = layer(feats, self.kv_cache_list[idx], self.kv_cache_list[idx], mask)

        return feats
    

class OCStochasticTransformerKVCache(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_slots, num_layers, num_heads, max_length, dropout, 
                 emb_type, skip_connection=False):
        super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.num_slots = num_slots
        self.skip_connection = skip_connection

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        if emb_type == '1emb':
            self.position_encoding = OCPositionalEncoding1D(max_length=max_length, num_slots=num_slots, embed_dim=feat_dim)
        elif emb_type == '2emb':
            self.position_encoding = OCPositionalEncoding1D2Emb(max_length=max_length, num_slots=num_slots, embed_dim=feat_dim)
        elif emb_type == '1emb_repeat':
            self.position_encoding = OCRepeatedPositionalEncoding1D(max_length=max_length, num_slots=num_slots, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlockKVCache(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)  # TODO: check if this is necessary

    def forward(self, samples, action, mask):
        '''
        Normal forward pass

        samples: (B, L, N, D)
        action: (B, L, 1)
        '''
        B, L, N, D = samples.shape

        action = F.one_hot(action.long(), self.action_dim).float()
        action = repeat(action, "B L A -> B L N A", N=N)
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats) # B L N D -> B (L N) D
        feats = self.layer_norm(feats)

        out = feats
        for layer in self.layer_stack:
            out, attn = layer(out, feats, feats, mask)

        if self.skip_connection:
            out = out + feats
        else:
            out = out
        out = rearrange(out, "B (L N) D -> B L N D", L=L)
        return out

    def reset_kv_cache_list(self, batch_size, dtype, device):
        '''
        Reset self.kv_cache_list
        '''
        self.kv_cache_list = []
        for layer in self.layer_stack:
            self.kv_cache_list.append(torch.zeros(size=(batch_size, 0, self.feat_dim), dtype=dtype, device=device))

    def forward_with_kv_cache(self, samples, action):
        '''
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        '''
        assert samples.shape[1] == 1
        mask = get_vector_mask(self.kv_cache_list[0].shape[1]+self.num_slots, samples.device)

        B, L, N, D = samples.shape
        action = F.one_hot(action.long(), self.action_dim).float()
        action = repeat(action, "B L A -> B L N A", N=N)
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding.forward_with_position(feats, position=self.kv_cache_list[0].shape[1])
        feats = self.layer_norm(feats)

        out = feats
        for idx, layer in enumerate(self.layer_stack):
            self.kv_cache_list[idx] = torch.cat([self.kv_cache_list[idx], out], dim=1)
            out, attn = layer(out, self.kv_cache_list[idx], self.kv_cache_list[idx], mask)

        if self.skip_connection:
            out = out + feats
        else:
            out = out
        out = rearrange(out, "B (L N) D -> B L N D", L=L)
        return out