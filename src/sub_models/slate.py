from collections import OrderedDict
import math
from itertools import chain
from einops import rearrange, repeat, reduce
import numpy as np
import torch
from torch.distributions import OneHotCategorical, RelaxedOneHotCategorical
import torch.nn as nn
import torch.nn.functional as F
import wandb

from sub_models.oc_world_models import EncoderBN, DecoderBN, SlotAttention, MSELoss, CELoss, OneHotDictionary
from sub_models.world_models import CategoricalKLDivLossWithFreeBits
from sub_models.transformer_model import OCStochasticTransformerKVCache
from sub_models.attention_blocks import PositionalEncoding1D, PositionalEncoding2D, get_causal_mask_with_batch_length
from sub_models.slate_utils import SLATETransformerDecoder, resize_patches_to_image
from utils import linear_warmup_exp_decay


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
    def __init__(self, image_feat_dim, transformer_hidden_dim, dec_hidden_dim, dec_num_layers, stoch_num_classes, stoch_dim, post_type='broadcast') -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_type = post_type
        if post_type == 'mlp':
            self.prior_head = SpatialBroadcastMLPDecoder(
                dec_input_dim=transformer_hidden_dim,
                dec_hidden_layers=[dec_hidden_dim] * dec_num_layers,
                stoch_num_classes=stoch_num_classes,
                stoch_dim=stoch_dim
            )
        elif post_type == 'conv':
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

        return logits, None, mask_as_image


class SLATE(nn.Module):
    def __init__(self, in_channels, action_dim, stem_channels, stoch_num_classes, stoch_dim, num_slots, slot_dim, dec_hidden_dim, dec_num_layers, vocab_size, post_type,
                 loss_type, lr_vae, lr_sa, lr_dec, max_grad_norm_vae, max_grad_norm_sa, max_grad_norm_dec, lr_warmup_steps, tau_anneal_steps, vis_attn_type,
                 enable_dict_reset=False) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.final_feature_width = 16
        self.stoch_num_classes = stoch_num_classes
        self.stoch_dim = stoch_dim
        self.stoch_flattened_dim = self.stoch_num_classes*self.stoch_dim
        self.vocab_size = vocab_size
        self.use_amp = True
        self.tensor_dtype = torch.float16 if self.use_amp else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.loss_type = loss_type
        self.vis_attn_type = vis_attn_type

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
            image_feat_dim=self.encoder.last_channels*self.final_feature_width*self.final_feature_width,
            transformer_hidden_dim=slot_dim,
            dec_hidden_dim=dec_hidden_dim,
            dec_num_layers=dec_num_layers,
            stoch_num_classes=stoch_num_classes,
            stoch_dim=self.stoch_dim,
            post_type=post_type
        )
        self.image_decoder = DecoderBN(
            original_in_channels=in_channels,
            stem_channels=stem_channels,
            vocab_size=vocab_size
        )
        self.dict = OneHotDictionary(vocab_size, stoch_dim, enable_reset=enable_dict_reset)
        self.pos_embed = nn.Sequential(
            PositionalEncoding1D(stoch_num_classes, stoch_dim, weight_init="trunc_normal"),
            nn.Dropout(0.1)
        )
        self.out = nn.Linear(stoch_dim, vocab_size, bias=False)

        self.mse_loss_func = MSELoss()
        self.ce_loss = CELoss()
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        
        self.optimizer_vae = torch.optim.Adam(self._get_vae_params(), lr=lr_vae)
        self.optimizer_sa = torch.optim.Adam(self._get_sa_params(), lr=lr_sa)
        self.optimizer_dec = torch.optim.Adam(self._get_dec_params(), lr=lr_dec)
        self.scheduler_vae = None
        self.scheduler_sa = torch.optim.lr_scheduler.LambdaLR(self.optimizer_sa, lr_lambda=linear_warmup_exp_decay(lr_warmup_steps, 0.5, 250000))
        self.scheduler_dec = torch.optim.lr_scheduler.LambdaLR(self.optimizer_dec, lr_lambda=linear_warmup_exp_decay(lr_warmup_steps, 0.5, 250000))
        self.max_grad_norm_vae = max_grad_norm_vae
        self.max_grad_norm_sa = max_grad_norm_sa
        self.max_grad_norm_dec = max_grad_norm_dec
        self.tau_anneal_steps = tau_anneal_steps

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

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
    
    def inspect(self, obs, tau=None):
        batch_size, batch_length = obs.shape[:2]
        H, W = obs.shape[-2:]
        tau = self.cosine_anneal(1, 0.1, 0, self.tau_anneal_steps) if tau is None else tau

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
            pred_logits, _, mask_as_image = self.dist_head.forward_prior(slots)
            pred_logits = rearrange(pred_logits, "B L K C -> (B L) K C")
            pred_logits = self.out(pred_logits)
            pred_logits = F.one_hot(pred_logits.argmax(dim=-1), self.vocab_size).float()
            z = rearrange(pred_logits, "(B L) (H W) C -> B L C H W", B=batch_size, H=self.final_feature_width, W=self.final_feature_width)
            obs_hat = self.image_decoder(z)

            # visualize attention
            if self.vis_attn_type == 'sbd':
                mask_as_image = rearrange(mask_as_image, "(B L) N H W -> B L N 1 H W", B=batch_size)
                obs_hat = torch.clamp(obs_hat, 0, 1)
                attns = obs_hat.unsqueeze(2).repeat(1, 1, self.num_slots, 1, 1, 1) * mask_as_image #+ 1. - mask_as_image ### color * mask
                # attns = mask_as_image.repeat(1, 1, 1, 3, 1, 1) ### mask
            elif self.vis_attn_type == 'sa':
                attns = rearrange(attns, 'B N (H W) -> B N H W', H=self.final_feature_width, W=self.final_feature_width)
                mask_as_image = attns.repeat_interleave(H // self.final_feature_width, dim=-2).repeat_interleave(W // self.final_feature_width, dim=-1)
                mask_as_image = rearrange(mask_as_image, "(B L) N H W -> B L N 1 H W", B=batch_size)
                obs_hat = torch.clamp(obs_hat, 0, 1)
                attns = obs_hat.unsqueeze(2).repeat(1, 1, self.num_slots, 1, 1, 1) * mask_as_image #+ 1. - mask_as_image ### color * mask
            # attns = mask_as_image.repeat(1, 1, 1, 3, 1, 1) ### mask

        video = torch.cat([obs.unsqueeze(2), obs_hat.unsqueeze(2), attns], dim=2).cpu().detach() # B L N C H W
        video = (video * 255.).numpy().astype(np.uint8)

        return video


    def update(self, obs, action=None, reward=None, termination=None, logger=None):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        H, W = obs.shape[-2:]
        tau = self.cosine_anneal(1, 0.1, 0, self.tau_anneal_steps)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(obs)
            soft, hard = self.dist_head.forward_post(embedding, tau=tau)

            # slot attention
            hard = rearrange(hard, "B L K C -> (B L) K C")
            tokens = torch.argmax(hard, dim=-1)
            z_emb = self.dict(hard)
            z_emb = self.pos_embed(z_emb)
            slots, attns = self.slot_attn(z_emb)
            slots = rearrange(slots, "(B L) N D -> B L N D", B=batch_size)
            
            # decoding image
            obs_hat = self.image_decoder(soft)

            # slot space to logits
            pred_logits, _, mask_as_image = self.dist_head.forward_prior(slots)
            pred_logits = rearrange(pred_logits, "B L K C -> (B L) K C")
            pred_logits = self.out(pred_logits)

            # env loss
            reconstruction_loss = self.mse_loss_func(obs_hat, obs)

            # ce loss
            hard = rearrange(hard, "(B L) K C -> B L K C", B=batch_size)
            pred_logits = rearrange(pred_logits, "(B L) K C -> B L K C", B=batch_size)
            if self.loss_type == "slate":
                ce_loss = self.ce_loss(pred_logits, hard)
                total_loss = reconstruction_loss + ce_loss
            elif self.loss_type == "storm": # dyn-rep loss
                dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(pred_logits.detach(), hard)
                representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(pred_logits, hard.detach())
                total_loss = reconstruction_loss + 0.5*dynamics_loss + 0.1*representation_loss

            # visualize attention
            if self.vis_attn_type == 'sbd':
                mask_as_image = rearrange(mask_as_image, "(B L) N H W -> B L N 1 H W", B=batch_size)
                obs_hat = torch.clamp(obs_hat, 0, 1)
                attns = obs_hat.unsqueeze(2).repeat(1, 1, self.num_slots, 1, 1, 1) * mask_as_image #+ 1. - mask_as_image ### color * mask
                # attns = mask_as_image.repeat(1, 1, 1, 3, 1, 1) ### mask
            elif self.vis_attn_type == 'sa':
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
        # print(norm_vae, norm_sa, norm_dec)
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
            "world_model/total_loss": total_loss.item(),
            "world_model/tau": tau,
            "world_model/lr_vae": self.optimizer_vae.param_groups[0]["lr"],
            "world_model/lr_sa": self.optimizer_sa.param_groups[0]["lr"],
            "world_model/lr_dec": self.optimizer_dec.param_groups[0]["lr"],
            "world_model/norm": norm_vae + norm_sa + norm_dec,
            "tokens/token_usage": len(torch.unique(tokens)) / self.vocab_size,
            "tokens/token_hist": wandb.Histogram(tokens.cpu().numpy().flatten()),
            "tokens/variance": self.dict.dictionary.weight.var().item(),
            "tokens/dist_from_init": self.dict.dist_from_init.mean().item(),
            "tokens/dist_hist": wandb.Histogram(self.dict.dist_from_init.cpu().numpy().flatten()),
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
    

class SLATEWorldModel(SLATE):
    def __init__(self, in_channels, action_dim, stem_channels, stoch_num_classes, stoch_dim, num_slots, slot_dim, dec_hidden_dim, dec_num_layers, vocab_size, post_type,
                 transformer_hidden_dim, transformer_num_layers, transformer_num_heads, transformer_max_length, loss_type,
                 lr_vae, lr_sa, lr_tf, max_grad_norm_vae, max_grad_norm_sa, max_grad_norm_tf, lr_warmup_steps, tau_anneal_steps) -> None:
        super().__init__(
            in_channels, action_dim, stem_channels, stoch_num_classes, stoch_dim, num_slots, slot_dim, dec_hidden_dim, dec_num_layers, vocab_size, post_type, loss_type,
            lr_vae, lr_sa, max_grad_norm_vae, max_grad_norm_sa, lr_warmup_steps, tau_anneal_steps
        )
        self.vocab_size = vocab_size
        self.transformer_hidden_dim = transformer_hidden_dim
        
        self.storm_transformer = OCStochasticTransformerKVCache(
            stoch_dim=self.slot_dim,
            action_dim=action_dim,
            feat_dim=transformer_hidden_dim,
            num_slots=num_slots,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.,
        )
        if transformer_hidden_dim != slot_dim:
            self.slot_proj = nn.Sequential(
                nn.LayerNorm(transformer_hidden_dim),
                nn.Linear(transformer_hidden_dim, slot_dim, bias=False)
            )
        else:
            self.slot_proj = nn.Identity()

        self.optimizer_vae = torch.optim.Adam(self._get_vae_params(), lr=lr_vae)
        self.optimizer_sa = torch.optim.Adam(self._get_sa_params(), lr=lr_sa)
        self.optimizer_dec = torch.optim.Adam(self._get_dec_params(), lr=0.0001)
        self.optimizer_tf = torch.optim.Adam(self._get_tf_params(), lr=lr_tf)
        self.scheduler_vae = None
        self.scheduler_sa = None
        self.scheduler_dec = None
        self.scheduler_tf = None
        # self.scheduler_tf = torch.optim.lr_scheduler.LambdaLR(self.optimizer_tf, lr_lambda=linear_warmup_exp_decay(lr_warmup_steps))
        self.max_grad_norm_tf = max_grad_norm_tf

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
    
    def calc_slots(self, sample):
        batch_size = sample.shape[0]
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            sample = rearrange(sample, "B L K C -> (B L) K C")
            sample = self.pos_embed(sample)

            slots, attns = self.slot_attn(sample)
            slots = rearrange(slots, "(B L) N D -> B L N D", B=batch_size)
        return slots, attns
    
    def predict_next(self, last_sample, action, log_video=True):
        batch_size, batch_length = last_sample.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            slots, attns = self.calc_slots(last_sample)

            hidden_hat = self.storm_transformer.forward_with_kv_cache(slots, action)

            hidden_hat_proj = self.slot_proj(hidden_hat)
            prior_logits, _, _ = self.dist_head.forward_prior(hidden_hat_proj)
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

            H, W = obs_hat.shape[-2:]
            attns = rearrange(attns, 'BL N (H W) -> BL N H W', H=self.final_feature_width, W=self.final_feature_width)
            mask_as_image = attns.repeat_interleave(H // self.final_feature_width, dim=-2).repeat_interleave(W // self.final_feature_width, dim=-1)
            mask_as_image = rearrange(mask_as_image, "(B L) N H W -> B L N 1 H W", B=batch_size)
            obs_hat = torch.clamp(obs_hat, 0, 1)
            attns = obs_hat.unsqueeze(2).repeat(1, 1, self.num_slots, 1, 1, 1) * mask_as_image #+ 1. - mask_as_image ### color * mask

        return obs_hat, sample, hidden_hat, attns

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
    
    def coef_warmup(self, warmup_steps, use_exp_decay=False, exp_decay_rate=0.5, exp_decay_steps=250000):
        multiplier = 1.0
        if warmup_steps is not None and self.step < warmup_steps:
            multiplier *= self.step / warmup_steps
        if use_exp_decay:
            multiplier *= exp_decay_rate ** (self.step / exp_decay_steps)
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

    def inspect_rollout(self, sample_obs, sample_action, gt_obs, gt_action,
                        imagine_batch_size, imagine_batch_length, logger=None):
        batch_size, batch_length, _, H, W = sample_obs.shape

        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=sample_obs.device)
        obs_hat_list = []
        attns_list = []

        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype, device=sample_obs.device)
        # context
        context_latent = self.encode_obs(sample_obs)
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_latent, last_hidden, last_attns = self.predict_next(
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
            self.action_buffer[:, i:i+1] = gt_action[:, i:i+1]

            last_obs_hat, last_latent, last_hidden, last_attns = self.predict_next(
                self.latent_buffer[:, i:i+1],
                self.action_buffer[:, i:i+1],
            )

            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_hidden
            obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])  # uniform sample vec_env
            attns_list.append(last_attns[::imagine_batch_size//16])  # uniform sample vec_env

        # if gt_for_inspection is not None:
        sample_obs = sample_obs[::imagine_batch_size//16]
        gt_obs = gt_obs[::imagine_batch_size//16]
        obs = torch.cat([sample_obs, gt_obs], dim=1)
        obs_hat_list = torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1)
        attns_list = torch.cat(attns_list, dim=1)
        rollout = torch.cat([obs.unsqueeze(2), obs_hat_list.unsqueeze(2), attns_list], dim=2).cpu().detach() # B L K C H W
        rollout = (rollout * 255.).numpy().astype(np.uint8)
        return rollout

    def update(self, obs, action, reward, termination, logger=None):
        train_only_tf = False
        freeze_dict = True
        if train_only_tf:
            self.eval()
            self.storm_transformer.train()
            self.slot_proj.train()
        elif freeze_dict:
            self.train()
            self.dict.eval()
        else:
            self.train()
        
        batch_size, batch_length = obs.shape[:2]
        H, W = obs.shape[-2:]
        tau = 0.1

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

            pred_logits, _, mask_as_image = self.dist_head.forward_prior(slots)
            pred_logits = rearrange(pred_logits, "B L K C -> (B L) K C")
            pred_logits = self.out(pred_logits)
            
            # transformer
            temporal_mask = get_causal_mask_with_batch_length(batch_length, self.num_slots, soft.device)
            slots_hat = self.storm_transformer(slots, action, temporal_mask) # B L N D

            # slot space to logits
            slots_hat = self.slot_proj(slots_hat)
            prior_logits, _, _ = self.dist_head.forward_prior(slots_hat)
            prior_logits = rearrange(prior_logits, "B L K C -> (B L) K C")
            prior_logits = self.out(prior_logits)

            # env loss
            reconstruction_loss = self.mse_loss_func(obs_hat, obs)

            pred_logits = rearrange(pred_logits, "(B L) K C -> B L K C", B=batch_size)
            post_logits = rearrange(post_logits, "(B L) K C -> B L K C", B=batch_size)
            prior_logits = rearrange(prior_logits, "(B L) K C -> B L K C", B=batch_size)
            if self.loss_type == "slate": # ce loss
                ce_loss = self.ce_loss(prior_logits[:, :-1], post_logits[:, 1:])
                total_loss = reconstruction_loss + ce_loss * self.coef_warmup(self.tau_anneal_steps, use_exp_decay=False)
            elif self.loss_type == "storm": # dyn-rep loss
                dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
                representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
                total_loss = reconstruction_loss + 0.5*dynamics_loss + 0.1*representation_loss
            elif self.loss_type == "slate+":
                ce_loss = self.ce_loss(prior_logits[:, :-1], post_logits[:, 1:])
                ce_loss_ = self.ce_loss(pred_logits, post_logits)
                lambda_ = self.coef_warmup(self.tau_anneal_steps, use_exp_decay=True)
                total_loss = reconstruction_loss + ce_loss * lambda_ + ce_loss_ * (1 - lambda_)

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
        if not train_only_tf:
            self.scaler.unscale_(self.optimizer_vae)  # for clip grad
            self.scaler.unscale_(self.optimizer_sa)  # for clip grad
            self.scaler.unscale_(self.optimizer_dec)  # for clip grad
        self.scaler.unscale_(self.optimizer_tf)  # for clip grad
        norm_vae, norm_sa, norm_dec = 0, 0, 0
        if not train_only_tf:
            norm_vae = torch.nn.utils.clip_grad_norm_(self._get_vae_params(), max_norm=self.max_grad_norm_vae, norm_type="inf")
            norm_sa = torch.nn.utils.clip_grad_norm_(self._get_sa_params(), max_norm=self.max_grad_norm_sa, norm_type="inf")
            norm_dec = torch.nn.utils.clip_grad_norm_(self._get_dec_params(), max_norm=self.max_grad_norm_vae, norm_type="inf")
        norm_tf = torch.nn.utils.clip_grad_norm_(self._get_tf_params(), max_norm=self.max_grad_norm_tf)
        # print(norm_vae, norm_sa, norm_dec, norm_tf)
        if not train_only_tf:
            self.scaler.step(self.optimizer_vae)
            self.scaler.step(self.optimizer_sa)
            self.scaler.step(self.optimizer_dec)
        self.scaler.step(self.optimizer_tf)
        self.scaler.update()

        if not train_only_tf:
            if self.scheduler_vae is not None:
                self.scheduler_vae.step()
            if self.scheduler_sa is not None:
                self.scheduler_sa.step()
            if self.scheduler_dec is not None:
                self.scheduler_dec.step()
        if self.scheduler_tf is not None:
            self.scheduler_tf.step()
            
        if not train_only_tf:
            self.optimizer_vae.zero_grad(set_to_none=True)
            self.optimizer_sa.zero_grad(set_to_none=True)
            self.optimizer_dec.zero_grad(set_to_none=True)
        self.optimizer_tf.zero_grad(set_to_none=True)

        video = torch.cat([obs.unsqueeze(2), obs_hat.unsqueeze(2), attns], dim=2).cpu().detach() # B L N C H W
        video = (video * 255.).numpy().astype(np.uint8)

        logs = {
            "world_model/reconstruction_loss": reconstruction_loss.item(),
            "world_model/total_loss": total_loss.item(),
            "world_model/tau": tau,
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
        elif self.loss_type == "slate+":
            logs.update({
                "world_model/ce_loss": ce_loss.item(),
            })

        return logs, video