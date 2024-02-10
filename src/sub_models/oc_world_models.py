from collections import OrderedDict
from itertools import chain
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast
import wandb

from sub_models.functions_losses import SymLogTwoHotLoss
from sub_models.attention_blocks import get_causal_mask_with_batch_length, get_causal_mask, PositionalEncoding1D, PositionalEncoding2D
from sub_models.transformer_model import OCStochasticTransformerKVCache
from sub_models.transformer_utils import SLATETransformerDecoder, SLATETransformerDecoderKVCache, resize_patches_to_image
from utils import linear_warmup_exp_decay

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
    
    def forward(self, x):
        return F.relu(self.m(x))


class EncoderBN(nn.Module):
    def __init__(self, in_channels, stem_channels, vocab_size) -> None:
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

        backbone.append(nn.Conv2d(stem_channels, vocab_size, 1))

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
    def __init__(self, original_in_channels, stem_channels, vocab_size) -> None:
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
            nn.Conv2d(stem_channels, original_in_channels, 1),
        ]

        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        batch_size = sample.shape[0]
        sample = rearrange(sample, "B L C H W -> (B L) C H W")
        obs_hat = self.backbone(sample)
        obs_hat = rearrange(obs_hat, "(B L) C H W -> B L C H W", B=batch_size)
        return obs_hat


class SlotAttention(nn.Module):
    def __init__(self, in_channels, slot_dim, num_slots, iters, eps=1e-8, hidden_dim=128) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim**-0.5

        self.slots_mu = nn.Parameter(torch.rand(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))
        with torch.no_grad():
            limit = math.sqrt(6.0 / (1 + slot_dim))
            torch.nn.init.uniform_(self.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slots_log_sigma, -limit, limit)

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(in_channels, slot_dim, bias=False)
        self.to_v = nn.Linear(in_channels, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim),
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

    def _reset_slots(self):
        self.slots_mu = nn.Parameter(torch.rand((1, 1, self.slot_dim), device=self.slots_mu.device))
        self.slots_log_sigma = nn.Parameter(torch.randn((1, 1, self.slot_dim), device=self.slots_log_sigma.device))
        with torch.no_grad():
            limit = math.sqrt(6.0 / (1 + self.slot_dim))
            torch.nn.init.uniform_(self.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slots_log_sigma, -limit, limit)

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
    def __init__(self, transformer_hidden_dim, dec_hidden_dim, dec_num_layers, stoch_num_classes, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.prior_head = SpatialBroadcastConvDecoder(
            dec_input_dim=transformer_hidden_dim,
            dec_hidden_layers=[dec_hidden_dim] * dec_num_layers,
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

    def forward_prior(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B L N D -> (B L) N D")
        logits, _, mask_as_image = self.prior_head(x)
        logits = rearrange(logits, "(B L) K C -> B L K C", B=batch_size)

        return logits


class RewardDecoder(nn.Module):
    def __init__(self, num_classes, embedding_size, input_dim, hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat):
        feat = feat.sum(dim=2)
        feat = self.backbone(feat)
        reward = self.head(feat)
        return reward


class TerminationDecoder(nn.Module):
    def __init__(self, embedding_size, input_dim, hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat):
        feat = feat.sum(dim=2)
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination


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
    

class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L C H W -> B L", "sum")
        return loss.mean()


class CELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits_hat, logits):
        loss = -(logits * F.log_softmax(logits_hat, dim=-1))
        loss = reduce(loss, "B L K C -> B L", "sum")
        return loss.mean()


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div


class OCWorldModel(nn.Module):
    def __init__(self, in_channels, action_dim, stem_channels, stoch_num_classes, stoch_dim, num_slots, slot_dim, dec_hidden_dim, dec_num_layers, vocab_size,
                 transformer_hidden_dim, transformer_num_layers, transformer_num_heads, transformer_max_length, loss_type, agent_state_type,
                 lr_vae, lr_sa, lr_dec, lr_tf, max_grad_norm_vae, max_grad_norm_sa, max_grad_norm_dec, max_grad_norm_tf, 
                 lr_warmup_steps, tau_anneal_steps, coef_anneal_steps) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.final_feature_width = 16
        self.stoch_num_classes = stoch_num_classes
        self.stoch_dim = stoch_dim
        self.stoch_flattened_dim = self.stoch_num_classes*self.stoch_dim
        self.vocab_size = vocab_size
        self.transformer_hidden_dim = transformer_hidden_dim
        self.loss_type = loss_type
        self.agent_state_type = agent_state_type
        self.lr_warmup_steps = lr_warmup_steps
        self.tau_anneal_steps = tau_anneal_steps
        self.coef_anneal_steps = coef_anneal_steps
        self.use_amp = True
        self.tensor_dtype = torch.float16 if self.use_amp else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1

        self.encoder = EncoderBN(
            in_channels=in_channels,
            stem_channels=stem_channels,
            vocab_size=vocab_size
        )
        self.slot_attn = SlotAttention(
            in_channels=self.stoch_dim,
            slot_dim=slot_dim,
            num_slots=num_slots,
            iters=3,
        )
        self.storm_transformer = OCStochasticTransformerKVCache(
            stoch_dim=self.slot_dim,
            action_dim=action_dim,
            feat_dim=transformer_hidden_dim,
            num_slots=num_slots,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.1,
        )
        self.dist_head = DistHead(
            transformer_hidden_dim=slot_dim,
            dec_hidden_dim=dec_hidden_dim,
            dec_num_layers=dec_num_layers,
            stoch_num_classes=stoch_num_classes,
            stoch_dim=self.stoch_dim
        )
        self.image_decoder = DecoderBN(
            original_in_channels=in_channels,
            stem_channels=stem_channels,
            vocab_size=vocab_size
        )
        self.reward_decoder = RewardDecoder(
            num_classes=255,
            embedding_size=self.stoch_flattened_dim,
            input_dim=transformer_hidden_dim,
            hidden_dim=transformer_hidden_dim
        )
        self.termination_decoder = TerminationDecoder(
            embedding_size=self.stoch_flattened_dim,
            input_dim=transformer_hidden_dim,
            hidden_dim=transformer_hidden_dim
        )
        self.dict = OneHotDictionary(vocab_size, stoch_dim)
        self.pos_embed = nn.Sequential(
            PositionalEncoding1D(stoch_num_classes, stoch_dim, weight_init="trunc_normal"),
            nn.Dropout(0.1)
        )
        self.out = nn.Linear(stoch_dim, vocab_size, bias=False)
        if transformer_hidden_dim != slot_dim:
            self.slot_proj = nn.Sequential(
                nn.LayerNorm(transformer_hidden_dim),
                nn.Linear(transformer_hidden_dim, slot_dim, bias=False)
            )
        else:
            self.slot_proj = nn.Identity()

        self.mse_loss_func = MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)

        self.optimizer_vae = torch.optim.Adam(self._get_vae_params(), lr=lr_vae)
        self.optimizer_sa = torch.optim.Adam(self._get_sa_params(), lr=lr_sa)
        self.optimizer_dec = torch.optim.Adam(self._get_dec_params(), lr=lr_dec)
        self.optimizer_tf = torch.optim.Adam(self._get_tf_params(), lr=lr_tf)
        self.scheduler_vae = None
        self.scheduler_sa = None
        self.scheduler_dec = None
        self.scheduler_tf = None
        self.max_grad_norm_vae = max_grad_norm_vae
        self.max_grad_norm_sa = max_grad_norm_sa
        self.max_grad_norm_dec = max_grad_norm_dec
        self.max_grad_norm_tf = max_grad_norm_tf

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if agent_state_type == "latent":
            self.agent_input_dim = self.slot_dim
        elif agent_state_type == "hidden":
            self.agent_input_dim = self.transformer_hidden_dim
        else:
            self.agent_input_dim = self.slot_dim + self.transformer_hidden_dim

        self.step = 0

    def _get_vae_params(self):
        return chain(
            self.encoder.parameters(),
            self.image_decoder.parameters(),
        )
    
    def _get_sa_params(self):
        return chain(
            self.slot_attn.parameters(),
        )

    def _get_dec_params(self):
        return chain(
            self.dist_head.parameters(),
            self.dict.parameters(),
            self.out.parameters(),
            self.pos_embed.parameters(),
        )

    def _get_tf_params(self):
        if self.transformer_hidden_dim != self.slot_dim:
            return chain(
                self.storm_transformer.parameters(),
                self.slot_proj.parameters(),
            )
        else:
            return chain(
                self.storm_transformer.parameters(),
            )

    def load(self, path_to_checkpoint, device):
        def extract_state_dict(state_dict, module_name):
            return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})
        
        state_dict = torch.load(path_to_checkpoint, map_location=device)
        self.encoder.load_state_dict(extract_state_dict(state_dict, 'encoder'))
        self.slot_attn.load_state_dict(extract_state_dict(state_dict, 'slot_attn'))
        self.dist_head.load_state_dict(extract_state_dict(state_dict, 'dist_head'))
        self.image_decoder.load_state_dict(extract_state_dict(state_dict, 'image_decoder'))
        self.dict.load_state_dict(extract_state_dict(state_dict, 'dict'))
        self.pos_embed.load_state_dict(extract_state_dict(state_dict, 'pos_embed'))
        self.out.load_state_dict(extract_state_dict(state_dict, 'out'))

        self.slot_attn._reset_slots()

    def encode_obs(self, obs):
        batch_size = obs.shape[0]
        tau = 0.1

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            soft, post_logits = self.dist_head.forward_post(embedding, tau=tau)
            post_logits = rearrange(post_logits, "B L K C -> (B L) K C")
            sample = self.dict(post_logits)
            sample = rearrange(sample, "(B L) K C -> B L K C", B=batch_size)

        return sample
    
    def calc_slots(self, sample, return_attn=True):
        batch_size = sample.shape[0]
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            sample = rearrange(sample, "B L K C -> (B L) K C")
            sample = self.pos_embed(sample)

            slots, attns = self.slot_attn(sample)
            slots = rearrange(slots, "(B L) N D -> B L N D", B=batch_size)

        if not return_attn:
            return slots
        return slots, attns
    
    def calc_last_dist_feat(self, latent, action):
        batch_size, batch_length = latent.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            slots = self.calc_slots(latent, return_attn=False)
            temporal_mask = get_causal_mask_with_batch_length(batch_length, self.num_slots, latent.device)
            slots_hat = self.storm_transformer(slots, action, temporal_mask)
            last_slots_hat = slots_hat[:, -1:]
            last_slots_hat_ = self.slot_proj(last_slots_hat)
            prior_logits = self.dist_head.forward_prior(last_slots_hat_)
            prior_logits = rearrange(prior_logits, "B L K C -> (B L) K C")
            prior_logits = self.out(prior_logits)
            prior_logits = F.one_hot(prior_logits.argmax(dim=-1), self.vocab_size).float()
            sample = self.dict(prior_logits)
            sample = rearrange(sample, "(B L) K C -> B L K C", B=batch_size)
            pred_slots = self.calc_slots(sample, return_attn=False)

        return pred_slots, last_slots_hat
    
    def predict_next(self, last_sample, action, log_video=True):
        batch_size, batch_length = last_sample.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            slots, attns = self.calc_slots(last_sample)

            hidden_hat = self.storm_transformer.forward_with_kv_cache(slots, action)

            hidden_hat_proj = self.slot_proj(hidden_hat)
            prior_logits = self.dist_head.forward_prior(hidden_hat_proj)
            prior_logits = rearrange(prior_logits, "B L K C -> (B L) K C")
            prior_logits = self.out(prior_logits)
            prior_logits = F.one_hot(prior_logits.argmax(dim=-1), self.vocab_size).float()
    
            if log_video:
                z = rearrange(prior_logits, "(B L) (H W) C -> B L C H W", B=batch_size, H=self.final_feature_width, W=self.final_feature_width)
                obs_hat = self.image_decoder(z)
            else:
                obs_hat = None
    
            sample = self.dict(prior_logits)
            sample = rearrange(sample, "(B L) K C -> B L K C", B=batch_size)
            reward_hat = self.reward_decoder(hidden_hat)
            reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(hidden_hat)
            termination_hat = termination_hat > 0

            # visualize attn
            H, W = obs_hat.shape[-2:]
            attns = rearrange(attns, 'BL N (H W) -> BL N H W', H=self.final_feature_width, W=self.final_feature_width)
            mask_as_image = attns.repeat_interleave(H // self.final_feature_width, dim=-2).repeat_interleave(W // self.final_feature_width, dim=-1)
            mask_as_image = rearrange(mask_as_image, "(B L) N H W -> B L N 1 H W", B=batch_size)
            obs_hat = torch.clamp(obs_hat, 0, 1)
            attns = obs_hat.unsqueeze(2).repeat(1, 1, self.num_slots, 1, 1, 1) * mask_as_image #+ 1. - mask_as_image ### color * mask

        return obs_hat, reward_hat, termination_hat, sample, hidden_hat, attns
    
    def coef_warmup(self):
        multiplier = 1.0
        if self.coef_anneal_steps is not None and self.step < self.coef_anneal_steps:
            multiplier *= self.step / self.coef_anneal_steps
        return multiplier

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype, device):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_num_classes, self.stoch_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, self.num_slots, self.transformer_hidden_dim)
            scalar_size = (imagine_batch_size, imagine_batch_length)
            self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device=device)
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device=device)
            self.action_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)

    def imagine_data(self, agent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, log_video=False, logger=None):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=sample_obs.device)
        obs_hat_list = []
        attns_list = []

        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype, device=sample_obs.device)
        # context
        context_latent = self.encode_obs(sample_obs)
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_hidden, last_attns = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1]
            )
            obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])  # uniform sample vec_env
            attns_list.append(last_attns[::imagine_batch_size//16])  # uniform sample vec_env
        self.latent_buffer[:, 0:1] = last_latent
        self.hidden_buffer[:, 0:1] = last_hidden

        # imagine
        # imagine_length = random.randint(2, imagine_batch_length) if gt_for_inspection is None else imagine_batch_length # from TransDreamer: at least imagine 2 steps for TD target
        imagine_length = imagine_batch_length
        for i in range(imagine_length):
            if self.agent_state_type == "latent":
                state = self.calc_slots(self.latent_buffer[:, i:i+1], return_attn=False)
            elif self.agent_state_type == "hidden":
                state = self.hidden_buffer[:, i:i+1]
            else:
                state = torch.cat([self.calc_slots(self.latent_buffer[:, i:i+1], return_attn=False), self.hidden_buffer[:, i:i+1]], dim=-1)
            action = agent.sample(state)
            self.action_buffer[:, i:i+1] = action

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_hidden, last_attns = self.predict_next(
                self.latent_buffer[:, i:i+1],
                self.action_buffer[:, i:i+1],
            )

            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_hidden
            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_hidden
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
            obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])  # uniform sample vec_env
            attns_list.append(last_attns[::imagine_batch_size//16])  # uniform sample vec_env

        rollout = None
        if log_video:
            rollout = torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).unsqueeze(2).cpu().detach() # B L K C H W
            rollout = (rollout * 255.).numpy().astype(np.uint8)

        if self.agent_state_type == "latent":
            states = self.latent_buffer
        elif self.agent_state_type == "hidden":
            states = self.hidden_buffer
        else:
            states = torch.cat([self.calc_slots(self.latent_buffer, return_attn=False), self.hidden_buffer], dim=-1)

        return states, self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer, rollout

    def update(self, obs, action, reward, termination, logger=None):
        freeze_dict = True
        if freeze_dict:
            self.train()
            self.dict.eval()
        else:
            self.train()
        
        batch_size, batch_length = obs.shape[:2]
        H, W = obs.shape[-2:]
        tau = 0.1
        coef_ = self.coef_warmup()

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):            
            # encoding
            embedding = self.encoder(obs)
            soft, post_logits = self.dist_head.forward_post(embedding, tau=tau)

            # slot attention
            post_logits = rearrange(post_logits, "B L K C -> (B L) K C")
            z_emb = self.dict(post_logits)
            z_emb = self.pos_embed(z_emb)

            slots, attns = self.slot_attn(z_emb)
            slots = rearrange(slots, "(B L) N D -> B L N D", B=batch_size)
            
            # decoding image
            obs_hat = self.image_decoder(soft)
            
            # transformer
            temporal_mask = get_causal_mask_with_batch_length(batch_length, self.num_slots, soft.device)
            slots_hat = self.storm_transformer(slots, action, temporal_mask) # B L N D
            # decoding reward and termination with slots_hat
            reward_hat = self.reward_decoder(slots_hat)
            termination_hat = self.termination_decoder(slots_hat)

            # slot space to logits
            slots_hat = self.slot_proj(slots_hat)
            prior_logits = self.dist_head.forward_prior(slots_hat)
            prior_logits = rearrange(prior_logits, "B L K C -> (B L) K C")
            prior_logits = self.out(prior_logits)

            # env loss
            reconstruction_loss = self.mse_loss_func(obs_hat, obs)
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)

            post_logits = rearrange(post_logits, "(B L) K C -> B L K C", B=batch_size)
            prior_logits = rearrange(prior_logits, "(B L) K C -> B L K C", B=batch_size)
            if self.loss_type == "slate": # ce loss
                ce_loss = self.ce_loss(prior_logits[:, :-1], post_logits[:, 1:])
                total_loss = reconstruction_loss + coef_ * (ce_loss + reward_loss + termination_loss)
            elif self.loss_type == "storm": # dyn-rep loss
                dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
                representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
                total_loss = reconstruction_loss + reward_loss + termination_loss + 0.5*dynamics_loss + 0.1*representation_loss

            # visualize attention
            attns = rearrange(attns, 'BL N (H W) -> BL N H W', H=self.final_feature_width, W=self.final_feature_width)
            mask_as_image = attns.repeat_interleave(H // self.final_feature_width, dim=-2).repeat_interleave(W // self.final_feature_width, dim=-1)
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
        self.scaler.unscale_(self.optimizer_tf)  # for clip grad
        norm_vae, norm_sa, norm_dec = 0, 0, 0
        norm_vae = torch.nn.utils.clip_grad_norm_(self._get_vae_params(), max_norm=self.max_grad_norm_vae, norm_type="inf")
        norm_sa = torch.nn.utils.clip_grad_norm_(self._get_sa_params(), max_norm=self.max_grad_norm_sa, norm_type="inf")
        norm_dec = torch.nn.utils.clip_grad_norm_(self._get_dec_params(), max_norm=self.max_grad_norm_dec, norm_type="inf")
        norm_tf = torch.nn.utils.clip_grad_norm_(self._get_tf_params(), max_norm=self.max_grad_norm_tf)
        # print(norm_vae, norm_sa, norm_dec, norm_tf)
        self.scaler.step(self.optimizer_vae)
        self.scaler.step(self.optimizer_sa)
        self.scaler.step(self.optimizer_dec)
        self.scaler.step(self.optimizer_tf)
        self.scaler.update()

        if self.scheduler_vae is not None:
            self.scheduler_vae.step()
        if self.scheduler_sa is not None:
            self.scheduler_sa.step()
        if self.scheduler_dec is not None:
            self.scheduler_dec.step()
        if self.scheduler_tf is not None:
            self.scheduler_tf.step()
            
        self.optimizer_vae.zero_grad(set_to_none=True)
        self.optimizer_sa.zero_grad(set_to_none=True)
        self.optimizer_dec.zero_grad(set_to_none=True)
        self.optimizer_tf.zero_grad(set_to_none=True)

        video = torch.cat([obs.unsqueeze(2), obs_hat.unsqueeze(2), attns], dim=2).cpu().detach() # B L N C H W
        video = (video * 255.).numpy().astype(np.uint8)

        logs = {
            "world_model/reconstruction_loss": reconstruction_loss.item(),
            "world_model/reward_loss": reward_loss.item(),
            "world_model/termination_loss": termination_loss.item(),
            "world_model/total_loss": total_loss.item(),
            "world_model/coef": coef_,
            "world_model/lr_vae": self.optimizer_vae.param_groups[0]["lr"],
            "world_model/lr_sa": self.optimizer_sa.param_groups[0]["lr"],
            "world_model/lr_dec": self.optimizer_dec.param_groups[0]["lr"],
            "world_model/lr_tf": self.optimizer_tf.param_groups[0]["lr"],
            "world_model/norm": norm_vae + norm_sa + norm_dec + norm_tf,
        }
        if self.loss_type == "slate":
            logs.update({
                "world_model/ce_loss": ce_loss.item(),
            })
        elif self.loss_type == "storm":
            logs.update({
                "world_model/dynamics_loss": dynamics_loss.item(),
                "world_model/dynamics_real_kl_div": dynamics_real_kl_div.item(),
                "world_model/representation_loss": representation_loss.item(),
                "world_model/representation_real_kl_div": representation_real_kl_div.item(),
            })

        return logs, video
    
    def inspect_reconstruction(self, obs, tau=None):
        batch_size, batch_length = obs.shape[:2]
        H, W = obs.shape[-2:]
        tau = 0.1 if tau is None else tau

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(obs)
            soft, hard = self.dist_head.forward_post(embedding, tau=tau)

            # slot attention
            hard = rearrange(hard, "B L K C -> (B L) K C")
            z_emb = self.dict(hard)
            z_emb = self.pos_embed(z_emb)
            slots, attns = self.slot_attn(z_emb)
            slots = rearrange(slots, "(B L) N D -> B L N D", B=batch_size)

            # slot space to logits
            pred_logits = self.dist_head.forward_prior(slots)
            pred_logits = rearrange(pred_logits, "B L K C -> (B L) K C")
            pred_logits = self.out(pred_logits)
            pred_logits = F.one_hot(pred_logits.argmax(dim=-1), self.vocab_size).float()
            z = rearrange(pred_logits, "(B L) (H W) C -> B L C H W", B=batch_size, H=self.final_feature_width, W=self.final_feature_width)
            obs_hat = self.image_decoder(z)

            # visualize attention
            attns = rearrange(attns, 'B N (H W) -> B N H W', H=self.final_feature_width, W=self.final_feature_width)
            mask_as_image = attns.repeat_interleave(H // self.final_feature_width, dim=-2).repeat_interleave(W // self.final_feature_width, dim=-1)
            mask_as_image = rearrange(mask_as_image, "(B L) N H W -> B L N 1 H W", B=batch_size)
            obs_hat = torch.clamp(obs_hat, 0, 1)
            attns = obs_hat.unsqueeze(2).repeat(1, 1, self.num_slots, 1, 1, 1) * mask_as_image #+ 1. - mask_as_image ### color * mask
            # attns = mask_as_image.repeat(1, 1, 1, 3, 1, 1) ### mask

        video = torch.cat([obs.unsqueeze(2), obs_hat.unsqueeze(2), attns], dim=2).cpu().detach() # B L N C H W
        video = (video * 255.).numpy().astype(np.uint8)
        return video

    def inspect_rollout(self, sample_obs, sample_action, gt_obs, gt_action,
                        imagine_batch_size, imagine_batch_length, logger=None):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=sample_obs.device)
        obs_hat_list = []
        attns_list = []

        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype, device=sample_obs.device)
        # context
        context_latent = self.encode_obs(sample_obs)
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_hidden, last_attns = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1]
            )
            obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])  # uniform sample vec_env
            attns_list.append(last_attns[::imagine_batch_size//16])  # uniform sample vec_env
        self.latent_buffer[:, 0:1] = last_latent
        self.hidden_buffer[:, 0:1] = last_hidden

        # imagine
        imagine_length = imagine_batch_length
        for i in range(imagine_length):
            self.action_buffer[:, i:i+1] = gt_action[:, i:i+1]

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_hidden, last_attns = self.predict_next(
                self.latent_buffer[:, i:i+1],
                self.action_buffer[:, i:i+1],
            )

            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_hidden
            obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])  # uniform sample vec_env
            attns_list.append(last_attns[::imagine_batch_size//16])  # uniform sample vec_env

        sample_obs = sample_obs[::imagine_batch_size//16]
        gt_obs = gt_obs[::imagine_batch_size//16]
        obs = torch.cat([sample_obs, gt_obs], dim=1)
        obs_hat_list = torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1)
        attns_list = torch.cat(attns_list, dim=1)
        rollout = torch.cat([obs.unsqueeze(2), obs_hat_list.unsqueeze(2), attns_list], dim=2).cpu().detach() # B L K C H W
        rollout = (rollout * 255.).numpy().astype(np.uint8)
        return rollout