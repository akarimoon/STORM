from collections import OrderedDict
from itertools import chain
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal, MultivariateNormal, Categorical
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast
import wandb

from sub_models.functions_losses import SymLogTwoHotLoss, MSELoss, CELoss, KLLoss, CategoricalKLDivLossWithFreeBits
from sub_models.quantizer import FSQ
from sub_models.attention_blocks import PositionalEncoding1D, PositionalEncoding2D
from sub_models.attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask, get_causal_mask_with_batch_length, get_causal_mask, get_causal_sparse_mask_with_batch_length, get_causal_sparse_mask
from sub_models.transformer_model import OCStochasticTransformerKVCache
from sub_models.dynamics_model import OP3PhysicsNetwork
from sub_models.slate_utils import resize_patches_to_image
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


class SpatialBroadcastMLPDecoder(nn.Module):
    def __init__(self, dec_input_dim, dec_hidden_layers, stoch_num_classes, stoch_dim, pos_emb_type="2d") -> None:
        super().__init__()

        self.stoch_dim = stoch_dim
        self.stoch_num_classes = stoch_num_classes
        self.pos_emb_type = pos_emb_type
        
        layers = []
        current_dim = dec_input_dim
    
        for dec_hidden_dim in dec_hidden_layers:
            layers.append(nn.Linear(current_dim, dec_hidden_dim))
            nn.init.zeros_(layers[-1].bias)
            layers.append(nn.ReLU(inplace=True))
            current_dim = dec_hidden_dim

        layers.append(nn.Linear(current_dim, stoch_dim+1))
        nn.init.zeros_(layers[-1].bias)
        
        self.layers = nn.Sequential(*layers)

        if pos_emb_type == "1d":
            self.pos_embed = nn.Parameter(torch.randn(1, stoch_num_classes, dec_input_dim) * 0.02)
        elif pos_emb_type == "2d":    
            self.init_resolution = (16, 16)
            self.pos_embed = PositionalEncoding2D(self.init_resolution, dec_input_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z (BL N D)
        init_shape = z.shape[:-1]
        z = z.flatten(0, -2)

        if self.pos_emb_type == "1d":
            z = z.unsqueeze(1).expand(-1, self.stoch_num_classes, -1)
            # Simple learned additive embedding as in ViT
            z = z + self.pos_embed
        elif self.pos_emb_type == "2d":
            z = z.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.init_resolution[0], self.init_resolution[1])
            # Simple learned additive embedding as in ViT
            z = self.pos_embed(z)
            z = rearrange(z, "BLN D H W -> BLN (H W) D")
        out = self.layers(z)
        out = out.unflatten(0, init_shape)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = out.split([self.stoch_dim, 1], dim=-1)
        alpha = alpha.softmax(dim=-3)

        reconstruction = torch.sum(decoded_patches * alpha, dim=-3)
        masks = alpha.squeeze(-1)
        masks_as_image = resize_patches_to_image(masks, size=64, resize_mode="bilinear")

        return reconstruction, masks, masks_as_image
    

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
    def __init__(self, transformer_hidden_dim, dec_input_dim, dec_hidden_dim, dec_num_layers, stoch_num_classes, stoch_dim, 
                 sbd_target, post_type='broadcast') -> None:
        super().__init__()
        self.dec_input_dim = dec_input_dim
        self.stoch_dim = stoch_dim
        self.sbd_target = sbd_target
        self.post_type = post_type
        self.quantizer = FSQ(levels=[7, 5, 5, 5, 5], dim=64) # as proposed in the paper
        if post_type == 'mlp':
            self.prior_head = SpatialBroadcastMLPDecoder(
                dec_input_dim=dec_input_dim,
                dec_hidden_layers=[dec_hidden_dim] * dec_num_layers,
                stoch_num_classes=stoch_num_classes,
                stoch_dim=stoch_dim if sbd_target != 'z_q' else self.quantizer.effective_codebook_dim
            )
        elif post_type == 'conv':
            self.prior_head = SpatialBroadcastConvDecoder(
                dec_input_dim=dec_input_dim,
                dec_hidden_layers=[dec_hidden_dim] * dec_num_layers,
                stoch_num_classes=stoch_num_classes,
                stoch_dim=stoch_dim if sbd_target != 'z_q' else self.quantizer.effective_codebook_dim
            )
        self.out = nn.Linear(stoch_dim, self.quantizer.codebook_size, bias=False) if stoch_dim != self.quantizer.codebook_size else nn.Identity()

    def forward_post(self, x):
        batch_size = x.shape[0]
        H, W = x.shape[-2:]
        x = rearrange(x, "B L C H W -> B L (H W) C")
        out, codes, indices = self.quantizer(x)
        soft = rearrange(out, "B L (H W) C -> B L C H W", H=H, W=W)
        codes = rearrange(codes, "B L (H W) C -> B L C H W", H=H, W=W)
        hard = F.one_hot(indices, self.quantizer.codebook_size).float() #.detach()
        return soft, codes, hard

    def forward_prior(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B L N D -> (B L) N D")
        logits, _, mask_as_image = self.prior_head(x)

        if self.sbd_target == 'onehot':
            logits = self.out(logits)
            sample = self.quantizer.indices_to_codes(logits.argmax(dim=-1))
        elif self.sbd_target == 'soft':
            sample = logits
        elif self.sbd_target == 'z_q':
            sample = self.quantizer.project_out(self.quantizer.quantize(logits))
        logits = rearrange(logits, "(B L) K C -> B L K C", B=batch_size)
        sample = rearrange(sample, "(B L) K C -> B L K C", B=batch_size)
        mask_as_image = rearrange(mask_as_image, "(B L) N H W -> B L N 1 H W", B=batch_size)
        return logits, sample, mask_as_image
    

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
        feat = feat.sum(dim=-2)
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
        feat = feat.sum(dim=-2)
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination


class OCQuantizedReconstructor(nn.Module):
    def __init__(self, in_channels, action_dim, stem_channels, stoch_num_classes, stoch_dim, num_slots, slot_dim, dec_hidden_dim, dec_num_layers, vocab_size, sbd_target,
                 transformer_hidden_dim, transformer_num_layers, transformer_num_heads, transformer_max_length, emb_type, skip_connection, stochastic_slots, separate_dist_head,
                 post_type, mask_type, loss_config, agent_state_type, vis_attn_type, imagine_with,
                 lr_vae, lr_sa, lr_dec, lr_tf, lr_rt, max_grad_norm_vae, max_grad_norm_sa, max_grad_norm_dec, max_grad_norm_tf, max_grad_norm_rt,
                 coef_dyn, coef_stat, coef_recon) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.final_feature_width = 16
        self.stoch_num_classes = stoch_num_classes
        self.stoch_dim = stoch_dim
        self.stoch_flattened_dim = self.stoch_num_classes*self.stoch_dim
        self.vocab_size = vocab_size
        self.sbd_target = sbd_target
        self.transformer_hidden_dim = transformer_hidden_dim
        self.stochastic_slots = stochastic_slots
        self.mask_type = mask_type
        self.loss_config = loss_config
        self.vis_attn_type = vis_attn_type
        self.agent_state_type = agent_state_type
        self.imagine_with = imagine_with
        assert imagine_with in ["latent", "hidden"], "imagine_with must be either 'latent' or 'hidden'"
        self.coef_dyn = coef_dyn
        self.coef_stat = coef_stat
        self.coef_recon = coef_recon

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
        self.dist_head = DistHead(
            transformer_hidden_dim=slot_dim,
            dec_input_dim=slot_dim,
            dec_hidden_dim=dec_hidden_dim,
            dec_num_layers=dec_num_layers,
            stoch_num_classes=stoch_num_classes,
            stoch_dim=self.stoch_dim,
            sbd_target=sbd_target,
            post_type=post_type,
        )
        self.image_decoder = DecoderBN(
            original_in_channels=in_channels,
            stem_channels=stem_channels,
            vocab_size=vocab_size
        )
        self.codebook_size = self.dist_head.quantizer.codebook_size
        self.pos_embed = nn.Sequential(
            PositionalEncoding1D(self.codebook_size, stoch_dim, weight_init="trunc_normal"),
            nn.Dropout(0.1)
        )

        self.mse_loss_func = MSELoss()
        self.ce_loss = CELoss()
        self.kl_loss_func = KLLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)

        self.optimizer_vae = torch.optim.Adam(self._get_vae_params(), lr=lr_vae)
        self.optimizer_sa = torch.optim.Adam(self._get_sa_params(), lr=lr_sa)
        self.optimizer_dec = torch.optim.Adam(self._get_dec_params(), lr=lr_dec)
        self.scheduler_vae = None
        self.scheduler_sa = None
        self.scheduler_dec = None
        self.max_grad_norm_vae = max_grad_norm_vae
        self.max_grad_norm_sa = max_grad_norm_sa
        self.max_grad_norm_dec = max_grad_norm_dec

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
            self.pos_embed.parameters(),
        )

    def load(self, path_to_checkpoint, device):
        def extract_state_dict(state_dict, module_name):
            return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})
        
        state_dict = torch.load(path_to_checkpoint, map_location=device)
        self.encoder.load_state_dict(extract_state_dict(state_dict, 'encoder'))
        self.slot_attn.load_state_dict(extract_state_dict(state_dict, 'slot_attn'))
        self.dist_head.load_state_dict(extract_state_dict(state_dict, 'dist_head'))
        self.image_decoder.load_state_dict(extract_state_dict(state_dict, 'image_decoder'))
        self.pos_embed.load_state_dict(extract_state_dict(state_dict, 'pos_embed'))
        self.out.load_state_dict(extract_state_dict(state_dict, 'out'))

        # self.slot_attn._reset_slots()

    def coef_warmup(self, steps):
        multiplier = 1.0
        if steps is not None and self.step < steps:
            multiplier *= self.step / steps
        return multiplier
    
    def _softplus_to_std(self, softplus):
        softplus = torch.min(softplus, torch.ones_like(softplus)*80)
        return torch.sqrt(torch.log(1 + softplus.exp()) + 1e-5)

    def sample(self, slots, dist="prior"):
        if self.stochastic_slots:
            if dist == "prior":
                mu = self.prior_slots_mu(slots)
                sigma = self.prior_slots_sigma(slots)
            else:
                mu = self.posterior_slots_mu(slots)
                sigma = self.posterior_slots_sigma(slots)

            if not self.training:
                return mu, None, None
            std = self._softplus_to_std(sigma)
            eps = torch.randn_like(mu).to(mu.device)
            return mu + std * eps, mu, std
        else:
            return slots, None, None

    def get_ce_loss(self, logits_hat, logits, soft, z_q):
        if self.sbd_target == "onehot":
            if self.loss_config.ce_type == "ce":
                return self.ce_loss(logits_hat, logits)
            elif self.loss_config.ce_type == "storm":
                ce_dyn_loss = self.ce_loss(logits_hat.detach(), logits)
                ce_rep_loss = self.ce_loss(logits_hat, logits.detach())
                return 0.5*ce_dyn_loss + 0.1*ce_rep_loss
        elif self.sbd_target in ["soft", "z_q"]:
            logits_hat = rearrange(logits_hat, "B L (H W) C -> B L C H W", H=self.final_feature_width)
            if self.loss_config.ce_type == "mse":
                if self.sbd_target == "soft":
                    return self.mse_loss_func(logits_hat, soft)
                elif self.sbd_target == "z_q":
                    return self.mse_loss_func(logits_hat, z_q)
            elif self.loss_config.ce_type == "storm":
                if self.sbd_target == "soft":
                    ce_loss_1 = self.mse_loss_func(logits_hat, soft.detach())
                    ce_loss_2 = self.mse_loss_func(logits_hat.detach(), soft)
                    return 0.9*ce_loss_1 + 0.1*ce_loss_2
                elif self.sbd_target == "z_q":
                    ce_loss_1 = self.mse_loss_func(logits_hat, z_q.detach())
                    ce_loss_2 = self.mse_loss_func(logits_hat.detach(), z_q)
                    return 0.9*ce_loss_1 + 0.1*ce_loss_2
    
    def update(self, obs, action, reward, termination, logger=None):
        self.train()
        
        batch_size, batch_length = obs.shape[:2]
        H, W = obs.shape[-2:]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):            
            # encoding
            embedding = self.encoder(obs)
            soft, z_q, post_logits = self.dist_head.forward_post(embedding)

            # slot attention
            codes = rearrange(soft, "B L C H W -> (B L) (H W) C")
            z_emb = self.pos_embed(codes)
            slots, attns = self.slot_attn(z_emb)
            slots = rearrange(slots, "(B L) N D -> B L N D", B=batch_size)
            slots, post_mu, post_sigma = self.sample(slots, dist="posterior")
            
            # decoding image
            obs_hat = self.image_decoder(soft)
            
            # slot (slots) space to logits
            recon_logits, recon_sample, mask_as_image = self.dist_head.forward_prior(slots)

            recon_sample = rearrange(recon_sample, "B L (H W) C -> B L C H W", B=batch_size, H=self.final_feature_width, W=self.final_feature_width)

            # env loss
            reconstruction_loss = self.mse_loss_func(obs_hat, obs)

            total_loss = self.coef_recon * reconstruction_loss
            
            if self.loss_config.recon_z:
                ce_recon_loss = self.get_ce_loss(recon_logits, post_logits, soft, z_q)
                total_loss += self.coef_stat * ce_recon_loss

            if self.loss_config.recon_x_from_slots:
                obs_hat_from_recon = self.image_decoder(recon_sample)
                recon_curr_loss = self.mse_loss_func(obs_hat_from_recon, obs)
                total_loss += self.coef_stat * recon_curr_loss
            
            # visualize attention
            if self.vis_attn_type == "sbd":
                obs_hat = torch.clamp(obs_hat, 0, 1)
                attns = obs_hat.unsqueeze(2).repeat(1, 1, self.num_slots, 1, 1, 1) * mask_as_image #+ 1. - mask_as_image ### color * mask
                # attns = mask_as_image.repeat(1, 1, 1, 3, 1, 1) ### mask
            elif self.vis_attn_type == "sa":
                attns = rearrange(attns, 'B N (H W) -> B N H W', H=self.final_feature_width, W=self.final_feature_width)
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
        norm_vae = torch.nn.utils.clip_grad_norm_(self._get_vae_params(), max_norm=self.max_grad_norm_vae, norm_type="inf")
        norm_sa = torch.nn.utils.clip_grad_norm_(self._get_sa_params(), max_norm=self.max_grad_norm_sa, norm_type="inf")
        norm_dec = torch.nn.utils.clip_grad_norm_(self._get_dec_params(), max_norm=self.max_grad_norm_dec, norm_type="inf")
        # print(norm_vae, norm_sa, norm_dec, norm_tf)
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

        tokens = torch.argmax(post_logits, dim=-1)
        logs = {
            "world_model/reconstruction_loss": reconstruction_loss.item(),
            "world_model/total_loss": total_loss.item(),
            "world_model/coef_recon": self.coef_recon,
            "world_model/coef_stat": self.coef_stat,
            "world_model/lr_vae": self.optimizer_vae.param_groups[0]["lr"],
            "world_model/lr_sa": self.optimizer_sa.param_groups[0]["lr"],
            "world_model/lr_dec": self.optimizer_dec.param_groups[0]["lr"],
            "world_model/norm": norm_vae + norm_sa + norm_dec,
            "tokens/token_usage": len(torch.unique(tokens)) / self.dist_head.quantizer.codebook_size,
            "tokens/token_hist": wandb.Histogram(tokens.cpu().numpy().flatten()),
        }

        if self.loss_config.recon_z:
            logs.update({
                "world_model/ce_recon_loss": ce_recon_loss.item(),
            })
        if self.loss_config.recon_x_from_slots:
            logs.update({
                "world_model/reconstruction_from_slots_loss": recon_curr_loss.item(),
            })
        
        return logs, video
    
    def inspect_reconstruction(self, obs, tau=None):
        batch_size, batch_length = obs.shape[:2]
        H, W = obs.shape[-2:]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(obs)
            soft, z_q, hard = self.dist_head.forward_post(embedding)

            # slot attention
            codes = rearrange(soft, "B L C H W -> (B L) (H W) C")
            z_emb = self.pos_embed(codes)
            slots, attns = self.slot_attn(z_emb)
            slots = rearrange(slots, "(B L) N D -> B L N D", B=batch_size)
            slots, _, _ = self.sample(slots, dist="posterior")

            # slot space to logits
            pred_logits, sample, mask_as_image = self.dist_head.forward_prior(slots)
            z = rearrange(sample, "B L (H W) C -> B L C H W", B=batch_size, H=self.final_feature_width, W=self.final_feature_width)
            obs_hat = self.image_decoder(z)

            # visualize attention
            if self.vis_attn_type == "sbd":
                obs_hat = torch.clamp(obs_hat, 0, 1)
                attns = obs_hat.unsqueeze(2).repeat(1, 1, self.num_slots, 1, 1, 1) * mask_as_image #+ 1. - mask_as_image ### color * mask
                # attns = mask_as_image.repeat(1, 1, 1, 3, 1, 1) ### mask
            elif self.vis_attn_type == "sa":
                attns = rearrange(attns, 'B N (H W) -> B N H W', H=self.final_feature_width, W=self.final_feature_width)
                mask_as_image = attns.repeat_interleave(H // self.final_feature_width, dim=-2).repeat_interleave(W // self.final_feature_width, dim=-1)
                mask_as_image = rearrange(mask_as_image, "(B L) N H W -> B L N 1 H W", B=batch_size)
                obs_hat = torch.clamp(obs_hat, 0, 1)
                attns = obs_hat.unsqueeze(2).repeat(1, 1, self.num_slots, 1, 1, 1) * mask_as_image #+ 1. - mask_as_image ### color * mask
                # attns = mask_as_image.repeat(1, 1, 1, 3, 1, 1) ### mask

        video = torch.cat([obs.unsqueeze(2), obs_hat.unsqueeze(2), attns], dim=2).cpu().detach() # B L N C H W
        video = (video * 255.).numpy().astype(np.uint8)
        return video


class OCQuantizedVideoPredictor(nn.Module):
    def __init__(self, in_channels, action_dim, stem_channels, stoch_num_classes, stoch_dim, num_slots, slot_dim, dec_hidden_dim, dec_num_layers, vocab_size, sbd_target,
                 transformer_hidden_dim, transformer_num_layers, transformer_num_heads, transformer_max_length, emb_type, skip_connection, stochastic_slots, separate_dist_head,
                 post_type, mask_type, loss_config, agent_state_type, vis_attn_type, imagine_with,
                 lr_vae, lr_sa, lr_dec, lr_tf, lr_rt, max_grad_norm_vae, max_grad_norm_sa, max_grad_norm_dec, max_grad_norm_tf, max_grad_norm_rt,
                 coef_dyn, coef_stat, coef_recon) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.final_feature_width = 16
        self.stoch_num_classes = stoch_num_classes
        self.stoch_dim = stoch_dim
        self.stoch_flattened_dim = self.stoch_num_classes*self.stoch_dim
        self.vocab_size = vocab_size
        self.sbd_target = sbd_target
        self.transformer_hidden_dim = transformer_hidden_dim
        self.stochastic_slots = stochastic_slots
        self.separate_dist_head = separate_dist_head
        self.mask_type = mask_type
        self.loss_config = loss_config
        if self.sbd_target == "onehot":
            assert self.loss_config.ce_type in ["ce", "storm"], "ce_type must be either 'ce' or 'storm' for onehot target"
        elif self.sbd_target in ["soft", "z_q"]:
            assert self.loss_config.ce_type in ["mse", "storm"], "ce_type must be either 'mse' or 'storm' for soft/z_q target"
        self.vis_attn_type = vis_attn_type
        self.agent_state_type = agent_state_type
        self.imagine_with = imagine_with
        assert imagine_with in ["latent", "hidden"], "imagine_with must be either 'latent' or 'hidden'"
        self.coef_dyn = coef_dyn
        self.coef_stat = coef_stat
        self.coef_recon = coef_recon

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
            emb_type=emb_type,
            skip_connection=skip_connection
        )
        self.dist_head = DistHead(
            transformer_hidden_dim=transformer_hidden_dim,
            dec_input_dim=slot_dim,
            dec_hidden_dim=dec_hidden_dim,
            dec_num_layers=dec_num_layers,
            stoch_num_classes=stoch_num_classes,
            stoch_dim=self.stoch_dim,
            sbd_target=sbd_target,
            post_type=post_type,
        )
        if separate_dist_head: # use separete dist head for reconstruction
            self.dist_head2 = DistHead(
                transformer_hidden_dim=slot_dim,
                dec_input_dim=slot_dim,
                dec_hidden_dim=dec_hidden_dim,
                dec_num_layers=dec_num_layers,
                stoch_num_classes=stoch_num_classes,
                stoch_dim=self.stoch_dim,
                sbd_target=sbd_target,
                post_type=post_type
            )
        self.image_decoder = DecoderBN(
            original_in_channels=in_channels,
            stem_channels=stem_channels,
            vocab_size=vocab_size
        )
        self.codebook_size = self.dist_head.quantizer.codebook_size
        self.pos_embed = nn.Sequential(
            PositionalEncoding1D(self.codebook_size, stoch_dim, weight_init="trunc_normal"),
            nn.Dropout(0.1)
        )
        if transformer_hidden_dim != slot_dim:
            self.slot_proj = nn.Sequential(
                nn.LayerNorm(transformer_hidden_dim),
                nn.Linear(transformer_hidden_dim, slot_dim)
            )
        else:
            self.slot_proj = nn.Identity()

        if stochastic_slots:
            self.posterior_slots_mu = nn.Linear(slot_dim, slot_dim)
            self.posterior_slots_sigma = nn.Linear(slot_dim, slot_dim)
            self.prior_slots_mu = nn.Linear(slot_dim, slot_dim)
            self.prior_slots_sigma = nn.Linear(slot_dim, slot_dim)

        self.mse_loss_func = MSELoss()
        self.ce_loss = CELoss()
        self.kl_loss_func = KLLoss()
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
        # self.scheduler_sa = torch.optim.lr_scheduler.LambdaLR(self.optimizer_sa, lr_lambda=linear_warmup_exp_decay(10000, None, None))
        # self.scheduler_dec = torch.optim.lr_scheduler.LambdaLR(self.optimizer_sa, lr_lambda=linear_warmup_exp_decay(10000, None, None))
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
        if self.separate_dist_head:
            return chain(
            self.dist_head.parameters(),
            self.pos_embed.parameters(),
            self.dist_head2.parameters(),
            )
        return chain(
            self.dist_head.parameters(),
            self.pos_embed.parameters(),
        )

    def _get_tf_params(self):
        return chain(
            self.storm_transformer.parameters(),
            self.slot_proj.parameters(),
        )

    def load(self, path_to_checkpoint, device):
        def extract_state_dict(state_dict, module_name):
            return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})
        
        state_dict = torch.load(path_to_checkpoint, map_location=device)
        self.encoder.load_state_dict(extract_state_dict(state_dict, 'encoder'))
        self.slot_attn.load_state_dict(extract_state_dict(state_dict, 'slot_attn'))
        self.storm_transformer.load_state_dict(extract_state_dict(state_dict, 'storm_transformer'))
        self.dist_head.load_state_dict(extract_state_dict(state_dict, 'dist_head'))
        self.image_decoder.load_state_dict(extract_state_dict(state_dict, 'image_decoder'))
        self.pos_embed.load_state_dict(extract_state_dict(state_dict, 'pos_embed'))

        # self.slot_attn._reset_slots()

    def encode_obs(self, obs):
        batch_size = obs.shape[0]
        tau = 0.1

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            soft, z_q, post_logits = self.dist_head.forward_post(embedding)
            sample = rearrange(soft, "B L C H W -> B L (H W) C")

        return sample
    
    def calc_slots(self, sample, return_attn=True):
        batch_size = sample.shape[0]
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            sample = rearrange(sample, "B L K C -> (B L) K C")
            sample = self.pos_embed(sample)

            slots, attns = self.slot_attn(sample)
            slots = rearrange(slots, "(B L) N D -> B L N D", B=batch_size)
            slots, _, _ = self.sample(slots, dist="posterior")

        if not return_attn:
            return slots
        return slots, attns

    def get_mask_with_batch_length(self, batch_length, device):
        if self.mask_type == "block":
            mask = get_causal_mask_with_batch_length(batch_length, self.num_slots, device)
        elif self.mask_type == "subsequent":
            mask = get_subsequent_mask_with_batch_length(batch_length*self.num_slots, device)
        elif self.mask_type == "block-sparse":
            mask = get_causal_sparse_mask_with_batch_length(batch_length, self.num_slots, device)
        return mask
    
    def calc_last_dist_feat(self, latent, action):
        batch_size, batch_length = latent.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            slots = self.calc_slots(latent, return_attn=False)
            slots, _, _ = self.sample(slots, dist="posterior")
            temporal_mask = self.get_mask_with_batch_length(batch_length, device=latent.device)
            slots_hat = self.storm_transformer(slots, action, temporal_mask)
            last_slots_hat = slots_hat[:, -1:]
            last_slots_hat_proj = self.slot_proj(last_slots_hat)
            last_slots_hat_proj, _, _ = self.sample(last_slots_hat_proj, dist="prior")
            prior_logits, sample, mask_as_image = self.dist_head.forward_prior(last_slots_hat_proj)
            pred_slots = self.calc_slots(sample, return_attn=False)

        return pred_slots, last_slots_hat
    
    def predict_next(self, last_sample, last_hidden, action, log_video=True):
        batch_size, batch_length = last_sample.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            if last_hidden is not None and self.imagine_with == "hidden":
                slots, attns = self.dist_head.slot_proj(last_hidden), None
            else:
                slots, attns = self.calc_slots(last_sample)
                slots, _, _ = self.sample(slots, dist="posterior")

            hidden_hat = self.storm_transformer.forward_with_kv_cache(slots, action)

            hidden_hat_proj = self.slot_proj(hidden_hat)
            hidden_hat_proj, _, _ = self.sample(hidden_hat_proj, dist="prior")
            prior_logits, sample, mask_as_image = self.dist_head.forward_prior(hidden_hat_proj)
            # sample = self.dist_head.quantizer.indices_to_codes(Categorical(logits=prior_logits).sample())
    
            if log_video:
                z = rearrange(sample, "B L (H W) C -> B L C H W", B=batch_size, H=self.final_feature_width, W=self.final_feature_width)
                obs_hat = self.image_decoder(z)
            else:
                obs_hat = None

            # visualize attn
            H, W = obs_hat.shape[-2:]
            if self.vis_attn_type == "sbd":
                obs_hat = torch.clamp(obs_hat, 0, 1)
                attns = obs_hat.unsqueeze(2).repeat(1, 1, self.num_slots, 1, 1, 1) * mask_as_image #+ 1. - mask_as_image ### color * mask
                # attns = mask_as_image.repeat(1, 1, 1, 3, 1, 1) ### mask
            elif self.vis_attn_type == "sa":
                attns = rearrange(attns, 'B N (H W) -> B N H W', H=self.final_feature_width, W=self.final_feature_width)
                mask_as_image = attns.repeat_interleave(H // self.final_feature_width, dim=-2).repeat_interleave(W // self.final_feature_width, dim=-1)
                mask_as_image = rearrange(mask_as_image, "(B L) N H W -> B L N 1 H W", B=batch_size)
                obs_hat = torch.clamp(obs_hat, 0, 1)
                attns = obs_hat.unsqueeze(2).repeat(1, 1, self.num_slots, 1, 1, 1) * mask_as_image #+ 1. - mask_as_image ### color * mask
                # attns = mask_as_image.repeat(1, 1, 1, 3, 1, 1) ### mask

        return obs_hat, sample, hidden_hat, attns
    
    def _softplus_to_std(self, softplus):
        softplus = torch.min(softplus, torch.ones_like(softplus)*80)
        return torch.sqrt(torch.log(1 + softplus.exp()) + 1e-5)

    def sample(self, slots, dist="prior"):
        if self.stochastic_slots:
            if dist == "prior":
                mu = self.prior_slots_mu(slots)
                sigma = self.prior_slots_sigma(slots)
            else:
                mu = self.posterior_slots_mu(slots)
                sigma = self.posterior_slots_sigma(slots)

            if not self.training:
                return mu, None, None
            std = self._softplus_to_std(sigma)
            eps = torch.randn_like(mu).to(mu.device)
            return mu + std * eps, mu, std
        else:
            return slots, None, None
            
    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype, device, silent=False):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            if not silent:
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

    def get_ce_loss(self, logits_hat, logits, soft, z_q):
        if self.sbd_target == "onehot":
            if self.loss_config.ce_type == "ce":
                return self.ce_loss(logits_hat, logits)
            elif self.loss_config.ce_type == "storm": #TODO: recon_logits' STORM loss
                ce_dyn_loss = self.ce_loss(logits_hat.detach(), logits)
                ce_rep_loss = self.ce_loss(logits_hat, logits.detach())
                return 0.5*ce_dyn_loss + 0.1*ce_rep_loss
        elif self.sbd_target in ["soft", "z_q"]:
            logits_hat = rearrange(logits_hat, "B L (H W) C -> B L C H W", H=self.final_feature_width)
            if self.loss_config.ce_type == "mse":
                if self.sbd_target == "soft":
                    return self.mse_loss_func(logits_hat, soft)
                elif self.sbd_target == "z_q":
                    return self.mse_loss_func(logits_hat, z_q)
            elif self.loss_config.ce_type == "storm":
                if self.sbd_target == "soft":
                    ce_loss_1 = self.mse_loss_func(logits_hat, soft.detach())
                    ce_loss_2 = self.mse_loss_func(logits_hat.detach(), soft)
                    return 0.9*ce_loss_1 + 0.1*ce_loss_2
                elif self.sbd_target == "z_q":
                    ce_loss_1 = self.mse_loss_func(logits_hat, z_q.detach())
                    ce_loss_2 = self.mse_loss_func(logits_hat.detach(), z_q)
                    return 0.9*ce_loss_1 + 0.1*ce_loss_2

    def update(self, obs, action, reward, termination, logger=None):
        self.train()
        
        batch_size, batch_length = obs.shape[:2]
        H, W = obs.shape[-2:]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):            
            # encoding
            embedding = self.encoder(obs)
            soft, z_q, post_logits = self.dist_head.forward_post(embedding)

            # slot attention
            codes = rearrange(soft, "B L C H W -> (B L) (H W) C")
            z_emb = self.pos_embed(codes)
            slots, attns = self.slot_attn(z_emb)
            slots = rearrange(slots, "(B L) N D -> B L N D", B=batch_size)
            slots, post_mu, post_sigma = self.sample(slots, dist="posterior")
            
            # decoding image
            obs_hat = self.image_decoder(soft)
            
            # transformer
            temporal_mask = self.get_mask_with_batch_length(batch_length, device=soft.device)
            slots_hat = self.storm_transformer(slots, action, temporal_mask) # B L N D

            # slot (slots_hat) space to logits
            slots_hat = self.slot_proj(slots_hat)
            slots_hat, prior_mu, prior_sigma = self.sample(slots_hat, dist="prior")
            prior_logits, prior_sample, mask_as_image = self.dist_head.forward_prior(slots_hat)

            # slot (slots) space to logits
            if self.separate_dist_head:
                recon_logits, recon_sample, _ = self.dist_head2.forward_prior(slots)
            else:
                recon_logits, recon_sample, _ = self.dist_head.forward_prior(slots)

            prior_sample = rearrange(prior_sample, "B L (H W) C -> B L C H W", B=batch_size, H=self.final_feature_width, W=self.final_feature_width)
            recon_sample = rearrange(recon_sample, "B L (H W) C -> B L C H W", B=batch_size, H=self.final_feature_width, W=self.final_feature_width)

            total_loss = 0.
            # env loss
            reconstruction_loss = self.mse_loss_func(obs_hat, obs)
            total_loss += self.coef_recon * reconstruction_loss
            
            ce_loss = self.get_ce_loss(prior_logits[:, :-1], post_logits[:, 1:], soft[:, 1:], z_q[:, 1:])
            total_loss += self.coef_dyn * ce_loss

            if self.loss_config.recon_z:
                ce_recon_loss = self.get_ce_loss(recon_logits, post_logits, soft, z_q)
                total_loss += self.coef_stat * ce_recon_loss

            if self.loss_config.recon_x_from_slots:
                obs_hat_from_recon = self.image_decoder(recon_sample)
                obs_hat_from_prior = self.image_decoder(prior_sample)
                recon_curr_loss = self.mse_loss_func(obs_hat_from_recon, obs)
                recon_next_loss = self.mse_loss_func(obs_hat_from_prior[:, :-1], obs[:, 1:])
                total_loss += self.coef_stat * recon_curr_loss + self.coef_dyn * recon_next_loss
            
            if self.stochastic_slots:
                # kl_loss = self.kl_loss_func(prior_mu, prior_sigma, post_mu, post_sigma)
                kl_loss = self.kl_loss_func(torch.zeros_like(prior_mu), torch.ones_like(prior_sigma), post_mu, post_sigma)
                total_loss += kl_loss

            # visualize attention
            if self.vis_attn_type == "sbd":
                obs_hat = torch.clamp(obs_hat, 0, 1)
                attns = obs_hat.unsqueeze(2).repeat(1, 1, self.num_slots, 1, 1, 1) * mask_as_image #+ 1. - mask_as_image ### color * mask
                # attns = mask_as_image.repeat(1, 1, 1, 3, 1, 1) ### mask
            elif self.vis_attn_type == "sa":
                attns = rearrange(attns, 'B N (H W) -> B N H W', H=self.final_feature_width, W=self.final_feature_width)
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

        tokens = torch.argmax(post_logits, dim=-1)
        prior_tokens = torch.argmax(prior_logits, dim=-1)
        logs = {
            "world_model/reconstruction_loss": reconstruction_loss.item(),
            "world_model/total_loss": total_loss.item(),
            "world_model/coef_recon": self.coef_recon,
            "world_model/coef_dyn": self.coef_dyn,
            "world_model/coef_stat": self.coef_stat,
            "world_model/lr_vae": self.optimizer_vae.param_groups[0]["lr"],
            "world_model/lr_sa": self.optimizer_sa.param_groups[0]["lr"],
            "world_model/lr_dec": self.optimizer_dec.param_groups[0]["lr"],
            "world_model/lr_tf": self.optimizer_tf.param_groups[0]["lr"],
            "world_model/norm": norm_vae + norm_sa + norm_dec + norm_tf,
            "tokens/token_usage": len(torch.unique(tokens)) / self.dist_head.quantizer.codebook_size,
            "tokens/token_usage_prior": len(torch.unique(prior_tokens)) / self.dist_head.quantizer.codebook_size,
            "tokens/token_hist": wandb.Histogram(tokens.cpu().numpy().flatten()),
        }

        logs.update({
            "world_model/ce_loss": ce_loss.item(),
        })
        if self.loss_config.recon_z:
            logs.update({
                "world_model/ce_recon_loss": ce_recon_loss.item(),
            })
        if self.loss_config.recon_x_from_slots:
            logs.update({
                "world_model/reconstruction_from_slots_loss": recon_curr_loss.item(),
                "world_model/reconstruction_from_next_slots_loss": recon_next_loss.item(),
            })
        
        if self.stochastic_slots:
            logs.update({
                "world_model/kl_loss": kl_loss.item(),
            })

        return logs, video
    
    def inspect_reconstruction(self, obs, tau=None):
        batch_size, batch_length = obs.shape[:2]
        H, W = obs.shape[-2:]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(obs)
            soft, z_q, hard = self.dist_head.forward_post(embedding)

            # slot attention
            codes = rearrange(soft, "B L C H W -> (B L) (H W) C")
            z_emb = self.pos_embed(codes)
            slots, attns = self.slot_attn(z_emb)
            slots = rearrange(slots, "(B L) N D -> B L N D", B=batch_size)
            slots, _, _ = self.sample(slots, dist="posterior")

            # slot space to logits
            if self.separate_dist_head:
                pred_logits, sample, mask_as_image = self.dist_head2.forward_prior(slots)
            else:
                pred_logits, sample, mask_as_image = self.dist_head.forward_prior(slots)
            z = rearrange(sample, "B L (H W) C -> B L C H W", B=batch_size, H=self.final_feature_width, W=self.final_feature_width)
            obs_hat = self.image_decoder(z)

            # visualize attention
            if self.vis_attn_type == "sbd":
                obs_hat = torch.clamp(obs_hat, 0, 1)
                attns = obs_hat.unsqueeze(2).repeat(1, 1, self.num_slots, 1, 1, 1) * mask_as_image #+ 1. - mask_as_image ### color * mask
                # attns = mask_as_image.repeat(1, 1, 1, 3, 1, 1) ### mask
            elif self.vis_attn_type == "sa":
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
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=sample_obs.device, silent=True)
        obs_hat_list = []
        attns_list = []

        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype, device=sample_obs.device)
        # context
        context_latent = self.encode_obs(sample_obs)
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_latent, last_hidden, last_attns = self.predict_next(
                context_latent[:, i:i+1],
                None,
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

            last_obs_hat, last_latent, last_hidden, last_attns = self.predict_next(
                self.latent_buffer[:, i:i+1],
                self.hidden_buffer[:, i:i+1],
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