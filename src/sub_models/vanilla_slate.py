import math
from itertools import chain
from einops import rearrange, repeat, reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sub_models.oc_world_models import SlotAttention, SpatialBroadcastMLPDecoder, MSELoss, OneHotCategorical
from sub_models.world_models import CategoricalKLDivLossWithFreeBits
from sub_models.transformer_model import OCStochasticTransformerKVCache
from sub_models.attention_blocks import PositionalEncoding1D, PositionalEncoding2D, get_causal_mask_with_batch_length
from sub_models.transformer_utils import SLATETransformerDecoder, resize_patches_to_image
from utils import linear_warmup_exp_decay


def conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode="zeros",
    weight_init="xavier",
):
    m = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
    )
    if weight_init == "kaiming":
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.zeros_(m.bias)
    return m


def linear(in_features, out_features, bias=True, weight_init="xavier", gain=1.0):
    m = nn.Linear(in_features, out_features, bias)
    if weight_init == "kaiming":
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    if bias:
        nn.init.zeros_(m.bias)
    return m


def gru_cell(input_size, hidden_size, bias=True):
    m = nn.GRUCell(input_size, hidden_size, bias)
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    return m


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
    
    def forward(self, x):
        return F.relu(self.m(x))

class EncoderBN(nn.Module):
    def __init__(self, in_channels, stem_channels, final_feature_width, vocab_size) -> None:
        super().__init__()

        backbone = [
            Conv2dBlock(in_channels, stem_channels, 4, 4),
            Conv2dBlock(stem_channels, stem_channels, 1, 1),
            Conv2dBlock(stem_channels, stem_channels, 1, 1),
            Conv2dBlock(stem_channels, stem_channels, 1, 1),
            Conv2dBlock(stem_channels, stem_channels, 1, 1),
            Conv2dBlock(stem_channels, stem_channels, 1, 1),
            Conv2dBlock(stem_channels, stem_channels, 1, 1),
        ]

        backbone.append(conv2d(stem_channels, vocab_size, 1))

        self.backbone = nn.Sequential(*backbone)
        self.last_channels = vocab_size

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B L C H W -> (B L) C H W")
        x = self.backbone(x)
        x = F.log_softmax(x, dim=1)
        x = rearrange(x, "(B L) C H W -> B L C H W", B=batch_size)
        return x
    

class DecoderBN(nn.Module):
    def __init__(self, stoch_dim, last_channels, original_in_channels, stem_channels, final_feature_width, vocab_size) -> None:
        super().__init__()

        backbone = [
            Conv2dBlock(vocab_size, stem_channels, 1),
            Conv2dBlock(stem_channels, stem_channels, 3, 1, 1),
            Conv2dBlock(stem_channels, stem_channels, 1, 1),
            Conv2dBlock(stem_channels, stem_channels, 1, 1),
            Conv2dBlock(stem_channels, stem_channels * 2 * 2, 1),
            nn.PixelShuffle(2),
            Conv2dBlock(stem_channels, stem_channels, 3, 1, 1),
            Conv2dBlock(stem_channels, stem_channels, 1, 1),
            Conv2dBlock(stem_channels, stem_channels, 1, 1),
            Conv2dBlock(stem_channels, stem_channels * 2 * 2, 1),
            nn.PixelShuffle(2),
            conv2d(stem_channels, original_in_channels, 1),
        ]

        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        batch_size = sample.shape[0]
        sample = rearrange(sample, "B L C H W -> (B L) C H W")
        obs_hat = self.backbone(sample)
        obs_hat = rearrange(obs_hat, "(B L) C H W -> B L C H W", B=batch_size)
        return obs_hat
    

class SlotAttnCNNEncoder(nn.Module):
    def __init__(self, obs_size, obs_channels, hidden_size):
        super().__init__()
        self._encoder = nn.Sequential(
            Conv2dBlock(obs_channels, hidden_size, 5, 1, 2),
            Conv2dBlock(hidden_size, hidden_size, 5, 1, 2),
            Conv2dBlock(hidden_size, hidden_size, 5, 1, 2),
            conv2d(hidden_size, hidden_size, 5, 1, 2),
        )
        self.pos_embed = PositionalEncoding2D((obs_size, obs_size), hidden_size)

    def forward(self, sample):
        batch_size = sample.shape[0]
        sample = rearrange(sample, "B L C H W -> (B L) C H W")
        emb = self._encoder(sample)
        emb = self.pos_embed(emb)
        emb = rearrange(emb, "(B L) C H W -> B L (H W) C", B=batch_size) # B L K C
        return emb


class SlotAttention(nn.Module):
    def __init__(self, in_channels, slot_dim, num_slots, iters, eps=1e-8, hidden_dim=128) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim**-0.5

        # self.input_mlp = nn.Sequential(
        #     nn.LayerNorm(in_channels),
        #     linear(in_channels, in_channels, weight_init="kaiming"),
        #     nn.ReLU(),
        #     linear(in_channels, in_channels),
        # )

        self.slots_mu = nn.Parameter(torch.rand(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_log_sigma)

        self.to_q = linear(slot_dim, slot_dim, bias=False)
        self.to_k = linear(in_channels, slot_dim, bias=False)
        self.to_v = linear(in_channels, slot_dim, bias=False)

        self.gru = gru_cell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.mlp = nn.Sequential(
            linear(slot_dim, hidden_dim, weight_init="kaiming"),
            nn.ReLU(inplace=True),
            linear(hidden_dim, slot_dim),
        )
        # self.mlp = nn.Sequential(
        #     nn.Linear(slot_dim, slot_dim*4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(slot_dim*4, slot_dim),
        # )

        self.norm_input = nn.LayerNorm(in_channels)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

        self._init_params()

    def _init_params(self) -> None:
        for name, tensor in self.named_parameters():
            if name.endswith(".bias"):
                torch.nn.init.zeros_(tensor)
            elif len(tensor.shape) <= 1:
                pass  # silent
            else:
                nn.init.xavier_uniform_(tensor)
        torch.nn.init.zeros_(self.gru.bias_ih)
        torch.nn.init.zeros_(self.gru.bias_hh)
        torch.nn.init.orthogonal_(self.gru.weight_hh)

    def forward(self, inputs) -> torch.Tensor:
        b, n, d = inputs.shape

        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_log_sigma.expand(b, self.num_slots, -1).exp()
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bjd,bij->bid", v, attn)

            slots = self.gru(
                updates.reshape(-1, self.slot_dim), slots_prev.reshape(-1, self.slot_dim)
            )

            slots = slots.reshape(b, -1, self.slot_dim)
            slots = (slots + self.mlp(self.norm_pre_ff(slots)))

        return slots, dots.softmax(dim=1)


class SpatialBroadcastConvDecoder(nn.Module):
    def __init__(self, dec_input_dim, dec_hidden_layers, stoch_num_classes, stoch_dim) -> None:
        super().__init__()

        self.stoch_dim = stoch_dim
        self.stoch_num_classes = stoch_num_classes
        
        layers = []
        current_dim = dec_input_dim
    
        for dec_hidden_dim in dec_hidden_layers:
            layers.append(nn.Conv2d(current_dim, dec_hidden_dim, 3, stride=(1, 1), padding=1))
            layers.append(nn.ReLU(inplace=True))
            current_dim = dec_hidden_dim
        layers.append(nn.Conv2d(current_dim, stoch_dim+1, 3, stride=(1, 1), padding=1))
        
        self.layers = nn.Sequential(*layers)

        self.init_resolution = (16, 16)
        self.pos_embed = PositionalEncoding2D(self.init_resolution, dec_input_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z (BL N D)
        init_shape = z.shape[:-1]
        z = z.flatten(0, 1)
        z = z.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.init_resolution[0], self.init_resolution[1])

        # Simple learned additive embedding as in ViT
        z = self.pos_embed(z)
        out = self.layers(z)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = out.split([self.stoch_dim, 1], dim=-3)
        alpha = alpha.softmax(dim=-3)
        decoded_patches = rearrange(decoded_patches, "(BL K) C H W -> BL K (H W) C", BL=init_shape[0])
        alpha = rearrange(alpha, "(BL K) C H W -> BL K (H W) C", BL=init_shape[0])

        reconstruction = torch.sum(decoded_patches * alpha, dim=1)
        masks = alpha.squeeze(-1)
        masks_as_image = resize_patches_to_image(masks, size=64, resize_mode="bilinear")

        return reconstruction, masks, masks_as_image
    

class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, image_feat_dim, transformer_hidden_dim, stoch_num_classes, stoch_dim, sbd_hidden_dim, sbd_num_layers, post_type) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_type = post_type
        if post_type == 'broadcast':
            self.prior_head = SpatialBroadcastMLPDecoder(
                dec_input_dim=transformer_hidden_dim,
                dec_hidden_layers=[sbd_hidden_dim] * sbd_num_layers,
                stoch_num_classes=stoch_num_classes,
                stoch_dim=stoch_dim
            )
        elif post_type == 'autoregressive':
            self.prior_head = SLATETransformerDecoder(
                feat_dim=transformer_hidden_dim,
                num_slots=7,
                num_heads=4,
                num_layers=4,
                dropout=0.1,
                max_length=stoch_num_classes,
                use_flash_attn=True,
            )
        elif post_type == 'conv':
            self.prior_head = SpatialBroadcastConvDecoder(
                dec_input_dim=transformer_hidden_dim,
                dec_hidden_layers=[sbd_hidden_dim] * sbd_num_layers,
                stoch_num_classes=stoch_num_classes,
                stoch_dim=stoch_dim
            )

    def gumbel_softmax(self, logits, tau=1., hard=False, dim=-1):
        eps = torch.finfo(logits.dtype).tiny
        gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
        gumbels = (logits + gumbels) / tau
        y_soft = F.softmax(gumbels, dim)
        
        if hard:
            index = y_soft.argmax(dim, keepdim=True)
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft

    def forward_post(self, x, tau):
        batch_size = x.shape[0]
        x = rearrange(x, "B L C H W -> (B L) C H W")
        logits = self.gumbel_softmax(x, tau, hard=False, dim=1)
        hard_logits = self.gumbel_softmax(x, tau, hard=True, dim=1).detach()
        logits = rearrange(logits, "(B L) C H W -> B L C H W", B=batch_size)
        hard_logits = rearrange(hard_logits, "(B L) C H W -> B L (H W) C", B=batch_size) # B L K C
        return logits, hard_logits

    def forward_prior(self, x, z=None):
        batch_size = x.shape[0]
        x = rearrange(x, "B L N D -> (B L) N D")
        if self.post_type == 'broadcast':
            logits, _, mask_as_image = self.prior_head(x)
        elif self.post_type == 'autoregressive':
            z = rearrange(z, "B L K C -> (B L) K C")
            logits = self.prior_head(z[:, :-1], x)
            mask_as_image = None
        elif self.post_type == 'conv':
            logits, _, mask_as_image = self.prior_head(x)
        logits = rearrange(logits, "(B L) K C -> B L K C", B=batch_size)

        return logits, None, mask_as_image

class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs


class CELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits_hat, logits):
        loss = -(logits * F.log_softmax(logits_hat, dim=-1))
        loss = reduce(loss, "B L K C -> B L", "sum")
        return loss.mean()
    

class SLATE(nn.Module):
    def __init__(self, in_channels, action_dim, stem_channels, stoch_num_classes, stoch_dim, num_slots, slot_dim, sbd_hidden_dim, sbd_num_layers, vocab_size, post_type,
                 lr_vae, lr_sa, max_grad_norm_vae, max_grad_norm_sa, lr_warmup_steps, tau_anneal_steps) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.final_feature_width = 16
        self.stoch_num_classes = stoch_num_classes
        self.stoch_dim = stoch_dim
        self.stoch_flattened_dim = self.stoch_num_classes*self.stoch_dim
        self.use_amp = True
        self.tensor_dtype = torch.float16 if self.use_amp else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.post_type = post_type

        self.encoder = EncoderBN(
            in_channels=in_channels,
            stem_channels=stem_channels,
            final_feature_width=self.final_feature_width,
            vocab_size=vocab_size
        )
        self.sa_encoder = SlotAttnCNNEncoder(
            obs_size=64, 
            obs_channels=3, 
            hidden_size=stem_channels
        )
        self.slot_attn = SlotAttention(
            in_channels=stem_channels,
            slot_dim=slot_dim,
            num_slots=num_slots,
            iters=3,
        )
        self.dist_head = DistHead(
            image_feat_dim=self.encoder.last_channels*self.final_feature_width*self.final_feature_width,
            transformer_hidden_dim=slot_dim,
            stoch_num_classes=stoch_num_classes,
            sbd_hidden_dim=sbd_hidden_dim,
            sbd_num_layers=sbd_num_layers,
            stoch_dim=self.stoch_dim,
            post_type=post_type
        )
        self.image_decoder = DecoderBN(
            stoch_dim=self.stoch_flattened_dim,
            last_channels=self.encoder.last_channels,
            original_in_channels=in_channels,
            stem_channels=stem_channels,
            final_feature_width=self.final_feature_width,
            vocab_size=vocab_size
        )
        self.dict = OneHotDictionary(vocab_size, stoch_dim)
        if post_type == 'broadcast' or post_type == 'conv':
            self.pos_embed = nn.Sequential(
                PositionalEncoding1D(stoch_num_classes, stoch_dim, weight_init="trunc_normal"),
                nn.Dropout(0.1)
            )
        elif post_type == 'autoregressive':
            self.pos_embed = nn.Sequential(
                PositionalEncoding1D(1+stoch_num_classes, stoch_dim, weight_init="trunc_normal"),
                nn.Dropout(0.1)
            )
            self.bos = nn.Parameter(torch.Tensor(1, 1, self.stoch_dim))
            nn.init.xavier_uniform_(self.bos)
            self.slot_proj = linear(slot_dim, stoch_dim, bias=False)
        self.out = linear(stoch_dim, vocab_size, bias=False)

        self.mse_loss_func = MSELoss()
        self.ce_loss = CELoss()
        
        # self.optimizer_vae = torch.optim.Adam(list(self.encoder.parameters()) + list(self.image_decoder.parameters()), lr=lr_vae)
        # self.optimizer_sa = torch.optim.Adam(list(self.dist_head.parameters()) + list(self.slot_attn.parameters()), lr=lr_sa)
        self.optimizer_vae = torch.optim.Adam(self._get_vae_params(), lr=lr_vae)
        self.optimizer_sa = torch.optim.Adam(self._get_sa_params(), lr=lr_sa)
        self.optimizer_dec = torch.optim.Adam(self._get_dec_params(), lr=lr_vae)
        self.scheduler_vae = None
        # self.scheduler_sa = None
        # self.scheduler_vae = torch.optim.lr_scheduler.LambdaLR(self.optimizer_vae, lr_lambda=linear_warmup_exp_decay(lr_warmup_steps, 0.5, 250000))
        self.scheduler_sa = torch.optim.lr_scheduler.LambdaLR(self.optimizer_sa, lr_lambda=linear_warmup_exp_decay(lr_warmup_steps, 0.5, 250000))
        self.scheduler_dec = torch.optim.lr_scheduler.LambdaLR(self.optimizer_dec, lr_lambda=linear_warmup_exp_decay(lr_warmup_steps, 0.5, 250000))
        self.max_grad_norm_vae = max_grad_norm_vae
        self.max_grad_norm_sa = max_grad_norm_sa
        self.tau_anneal_steps = tau_anneal_steps

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.step = 0

    def _get_vae_params(self):
        return chain(
            self.encoder.parameters(),
            self.image_decoder.parameters(),
        )
    
    def _get_sa_params(self):
        if self.post_type == 'broadcast' or self.post_type == 'conv':
            return chain(
                self.sa_encoder.parameters(),
                self.slot_attn.parameters(),
            )
        elif self.post_type == 'autoregressive':
            return chain(
                self.sa_encoder.parameters(),
                self.slot_attn.parameters(),
                self.slot_proj.parameters(),
            )
    
    def _get_dec_params(self):
        if self.post_type == 'broadcast' or self.post_type == 'conv':
            return chain(
                self.dist_head.parameters(),
                self.dict.parameters(),
                self.pos_embed.parameters(),
                self.out.parameters(),
            )
        elif self.post_type == 'autoregressive':
            return chain(
                self.dist_head.parameters(),
                self.dict.parameters(),
                [self.bos],
                self.pos_embed.parameters(),
                self.out.parameters(),
            )

    def encode_obs(self, obs):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)
        return flattened_sample

    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs   
        return sample
        
    def cosine_anneal(self, start_value, final_value, start_step, final_step):
        assert start_value >= final_value
        assert start_step <= final_step
        
        if self.step < start_step:
            value = start_value
        elif self.step >= final_step:
            value = final_value
        else:
            a = 0.5 * (start_value - final_value)
            b = 0.5 * (start_value + final_value)
            progress = (self.step - start_step) / (final_step - start_step)
            value = a * math.cos(math.pi * progress) + b
        
        return value

    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

    def update(self, obs, action=None, reward=None, termination=None, logger=None):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        H, W = obs.shape[-2:]
        tau = self.cosine_anneal(1, 0.1, 0, self.tau_anneal_steps)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            # encoding
            dvae_embedding = self.encoder(obs)
            soft, hard = self.dist_head.forward_post(dvae_embedding, tau=tau)

            # slot attention
            embedding = self.sa_encoder(obs)
            embedding = rearrange(embedding, "B L K C -> (B L) K C")
            slots, attns = self.slot_attn(embedding)
            slots = rearrange(slots, "(B L) N D -> B L N D", B=batch_size)
            
            # decoding image
            obs_hat = self.image_decoder(soft)

            # slot space to logits
            hard = rearrange(hard, "B L K C -> (B L) K C")
            if self.post_type == 'broadcast' or self.post_type == 'conv':
                z_emb = None
            elif self.post_type == 'autoregressive':
                z_emb = self.dict(hard)
                bos = repeat(self.bos, "1 1 C -> (B L) 1 C", B=batch_size, L=batch_length)
                z_emb = torch.cat([bos, z_emb], dim=1)
                z_emb = self.pos_embed(z_emb)
                z_emb = rearrange(z_emb, "(B L) K C -> B L K C", B=batch_size)
            
                slots = self.slot_proj(slots)
            
            pred_logits, _, mask_as_image = self.dist_head.forward_prior(slots, z_emb)
            pred_logits = rearrange(pred_logits, "B L K C -> (B L) K C")
            pred_logits = self.out(pred_logits)

            # env loss
            reconstruction_loss = self.mse_loss_func(obs_hat, obs)

            # ce loss
            hard = rearrange(hard, "(B L) K C -> B L K C", B=batch_size)
            pred_logits = rearrange(pred_logits, "(B L) K C -> B L K C", B=batch_size)
            ce_loss = self.ce_loss(pred_logits, hard)
            
            total_loss = reconstruction_loss + ce_loss

            # visualize attention
                # mask_as_image = rearrange(mask_as_image, "(B L) N H W -> B L N 1 H W", B=batch_size)
                # obs_hat = torch.clamp(obs_hat, 0, 1)
                # attns = mask_as_image.repeat(1, 1, 1, 3, 1, 1) ### mask
            
            mask_as_image = rearrange(attns, 'B N (H W) -> B N H W', H=64, W=64)
            mask_as_image = rearrange(mask_as_image, "(B L) N H W -> B L N 1 H W", B=batch_size)
            obs_hat = torch.clamp(obs_hat, 0, 1)
            attns = obs_hat.unsqueeze(2).repeat(1, 1, self.num_slots, 1, 1, 1) * mask_as_image #+ 1. - mask_as_image ### color * mask
            # attns = mask_as_image.repeat(1, 1, 1, 3, 1, 1) ### mask

            self.step += 1

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer_vae)  # for clip grad
        self.scaler.unscale_(self.optimizer_sa)  # for clip grad
        self.scaler.unscale_(self.optimizer_dec)  # for clip grad
        norm_vae = torch.nn.utils.clip_grad_norm_(self._get_vae_params(), max_norm=self.max_grad_norm_vae, norm_type="inf")
        norm_sa = torch.nn.utils.clip_grad_norm_(self._get_sa_params(), max_norm=self.max_grad_norm_sa, norm_type="inf")
        norm_dec = torch.nn.utils.clip_grad_norm_(self._get_dec_params(), max_norm=self.max_grad_norm_vae, norm_type="inf")
        self.scaler.step(self.optimizer_vae)
        self.scaler.step(self.optimizer_sa)
        self.scaler.step(self.optimizer_dec)
        self.scaler.update()

        if self.scheduler_vae is not None:
            self.scheduler_vae.step()
        if self.scheduler_sa is not None:
            self.scheduler_sa.step()
        if self.scheduler_dec is not None:
            self.scheduler_dec.step()
            
        self.optimizer_vae.zero_grad(set_to_none=True)
        self.optimizer_sa.zero_grad(set_to_none=True)
        self.optimizer_dec.zero_grad(set_to_none=True)

        video = torch.cat([obs.unsqueeze(2), obs_hat.unsqueeze(2), attns], dim=2).cpu().detach() # B L N C H W
        video = (video * 255.).numpy().astype(np.uint8)

        logs = {
            "world_model/reconstruction_loss": reconstruction_loss.item(),
            "world_model/ce_loss": ce_loss.item(),
            "world_model/total_loss": total_loss.item(),
            "world_model/tau": tau,
            "world_model/lr_vae": self.optimizer_vae.param_groups[0]["lr"],
            "world_model/lr_sa": self.optimizer_sa.param_groups[0]["lr"],
            "world_model/lr_dec": self.optimizer_dec.param_groups[0]["lr"],
            "world_model/norm": norm_vae + norm_sa + norm_dec,
        }

        return logs, video