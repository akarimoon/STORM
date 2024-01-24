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

class EncoderBN(nn.Module):
    def __init__(self, in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        feature_width = 64//2
        channels = stem_channels
        backbone.append(nn.BatchNorm2d(stem_channels))
        backbone.append(nn.ReLU(inplace=True))

        # layers
        while True:
            backbone.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels*2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels *= 2
            feature_width //= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))

            if feature_width == final_feature_width:
                break

        self.backbone = nn.Sequential(*backbone)
        self.last_channels = channels

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B L C H W -> (B L) C H W")
        x = self.backbone(x)
        x = rearrange(x, "(B L) C H W -> B L (C H W)", B=batch_size)
        return x


class DecoderBN(nn.Module):
    def __init__(self, stoch_dim, last_channels, original_in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(nn.Linear(stoch_dim, last_channels*final_feature_width*final_feature_width, bias=False))
        backbone.append(Rearrange('B L (C H W) -> (B L) C H W', C=last_channels, H=final_feature_width))
        backbone.append(nn.BatchNorm2d(last_channels))
        backbone.append(nn.ReLU(inplace=True))
        # residual_layer
        # backbone.append(ResidualStack(last_channels, 1, last_channels//4))
        # layers
        channels = last_channels
        feat_width = final_feature_width
        while True:
            if channels == stem_channels:
                break
            backbone.append(
                nn.ConvTranspose2d(
                    in_channels=channels,
                    out_channels=channels//2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels //= 2
            feat_width *= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))

        backbone.append(
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=original_in_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        batch_size = sample.shape[0]
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
    def __init__(self, dec_input_dim, dec_hidden_layers, stoch_num_classes, stoch_dim) -> None:
        super().__init__()

        self.stoch_dim = stoch_dim
        self.stoch_num_classes = stoch_num_classes
        
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

        self.pos_embed = nn.Parameter(torch.randn(1, stoch_num_classes, dec_input_dim) * 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z (BL N D)
        init_shape = z.shape[:-1]
        z = z.flatten(0, -2)
        z = z.unsqueeze(1).expand(-1, self.stoch_num_classes, -1)

        # Simple learned additive embedding as in ViT
        z = z + self.pos_embed
        out = self.layers(z)
        out = out.unflatten(0, init_shape)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = out.split([self.stoch_dim, 1], dim=-1)
        alpha = alpha.softmax(dim=-3)

        reconstruction = torch.sum(decoded_patches * alpha, dim=-3)
        masks = alpha.squeeze(-1)
        masks_as_image = resize_patches_to_image(masks, size=64, resize_mode="bilinear")

        return reconstruction, masks, masks_as_image


class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, image_feat_dim, transformer_hidden_dim, sbd_hidden_dim, stoch_num_classes, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_num_classes*stoch_dim)
        self.prior_head = SpatialBroadcastMLPDecoder(
            dec_input_dim=transformer_hidden_dim,
            dec_hidden_layers=[sbd_hidden_dim, sbd_hidden_dim, sbd_hidden_dim],
            stoch_num_classes=stoch_num_classes,
            stoch_dim=stoch_dim
        )

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", C=self.stoch_dim)
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B L N D -> (B L) N D")
        logits, _, mask_as_image = self.prior_head(x)
        logits = rearrange(logits, "(B L) K C -> B L K C", B=batch_size)
        logits = self.unimix(logits)
        return logits, None, mask_as_image


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
        feat = rearrange(feat, "B L K C -> B L (K C)")
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
        feat = rearrange(feat, "B L K C -> B L (K C)")
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L C H W -> B L", "sum")
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
    def __init__(self, in_channels, action_dim, stoch_num_classes, stoch_dim, num_slots, slot_dim,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads, sbd_hidden_dim, 
                 lr_storm, lr_sa, max_grad_norm_storm, max_grad_norm_sa) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.transformer_hidden_dim = transformer_hidden_dim
        self.final_feature_width = 4
        self.stoch_num_classes = stoch_num_classes
        self.stoch_dim = stoch_dim
        self.stoch_flattened_dim = self.stoch_num_classes*self.stoch_dim
        self.use_amp = True
        self.tensor_dtype = torch.float16 if self.use_amp else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1

        self.encoder = EncoderBN(
            in_channels=in_channels,
            stem_channels=32,
            final_feature_width=self.final_feature_width
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
            image_feat_dim=self.encoder.last_channels*self.final_feature_width*self.final_feature_width,
            transformer_hidden_dim=transformer_hidden_dim,
            sbd_hidden_dim=sbd_hidden_dim,
            stoch_num_classes=stoch_num_classes,
            stoch_dim=self.stoch_dim
        )
        self.image_decoder = DecoderBN(
            stoch_dim=self.stoch_flattened_dim,
            last_channels=self.encoder.last_channels,
            original_in_channels=in_channels,
            stem_channels=32,
            final_feature_width=self.final_feature_width
        )
        self.reward_decoder = RewardDecoder(
            num_classes=255,
            embedding_size=self.stoch_flattened_dim,
            input_dim=transformer_hidden_dim*self.num_slots,
            hidden_dim=transformer_hidden_dim
        )
        self.termination_decoder = TerminationDecoder(
            embedding_size=self.stoch_flattened_dim,
            input_dim=transformer_hidden_dim*self.num_slots,
            hidden_dim=transformer_hidden_dim
        )

        self.mse_loss_func = MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        
        self.optimizer_storm = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.storm_transformer.parameters()) + list(self.image_decoder.parameters()) + \
            list(self.dist_head.parameters()) + list(self.reward_decoder.parameters()) + list(self.termination_decoder.parameters()),
            lr=lr_storm
        )
        self.optimizer_sa = torch.optim.Adam(self.slot_attn.parameters(), lr=lr_sa)
        self.scheduler_storm = None
        self.scheduler_sa = torch.optim.lr_scheduler.LambdaLR(self.optimizer_sa, lr_lambda=linear_warmup_exp_decay(10000, 0.5, 100000))
        self.max_grad_norm_storm = max_grad_norm_storm
        self.max_grad_norm_sa = max_grad_norm_sa

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.agent_input_dim = self.slot_dim + self.transformer_hidden_dim

    def encode_obs(self, obs):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)
        return flattened_sample
    
    def calc_slots(self, latent, is_flat=False, return_z_emb=False):
        batch_size, batch_length = latent.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            if is_flat:
                latent = rearrange(latent, "B L (K C) -> B L K C", C=self.stoch_dim)

            z_emb = rearrange(latent, "B L K C -> (B L) K C")
            # bos = repeat(self.enc_bos, "1 1 C -> (B L) 1 C", B=batch_size, L=batch_length)
            # z_emb = torch.cat([bos, z_emb], dim=1)
            # z_emb = self.enc_position_encoding(z_emb)

            # slot attention
            slots, _ = self.slot_attn(z_emb)
            slots = rearrange(slots, "(B L) N D -> B L N D", B=batch_size)

        if return_z_emb:
            z_emb = rearrange(z_emb, "(B L) K C -> B L K C", B=batch_size)
            return slots, z_emb

        return slots

    def calc_last_dist_feat(self, latent, action):
        batch_size, batch_length = latent.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            slots, z_emb = self.calc_slots(latent, is_flat=True, return_z_emb=True)
            temporal_mask = get_causal_mask_with_batch_length(batch_length, self.num_slots, latent.device)
            hidden_hat = self.storm_transformer(slots, action, temporal_mask)
            last_hidden_hat = hidden_hat[:, -1:]

            prior_logits, _, _ = self.dist_head.forward_prior(last_hidden_hat)
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)

        return self.calc_slots(prior_flattened_sample, is_flat=True), last_hidden_hat

    def predict_next(self, last_flattened_sample, action, log_video=True):
        batch_size, batch_length = last_flattened_sample.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            slots = self.calc_slots(last_flattened_sample, is_flat=True)

            hidden_hat = self.storm_transformer.forward_with_kv_cache(slots, action)

            prior_logits, _, mask_as_image = self.dist_head.forward_prior(hidden_hat)
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
            if log_video:
                obs_hat = self.image_decoder(prior_flattened_sample)
            else:
                obs_hat = None
            
            reward_hat = self.reward_decoder(hidden_hat)
            reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(hidden_hat)
            termination_hat = termination_hat > 0

            mask_as_image = rearrange(mask_as_image, "(B L) N H W -> B L N 1 H W", B=batch_size)
            attns = obs_hat.unsqueeze(2) * mask_as_image + 1. - mask_as_image

        return obs_hat, reward_hat, termination_hat, prior_flattened_sample, hidden_hat, attns

    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample

    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype, device):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_flattened_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, self.num_slots, self.transformer_hidden_dim)
            scalar_size = (imagine_batch_size, imagine_batch_length)
            self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device=device)
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device=device)
            self.action_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)

    def imagine_data(self, agent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, log_video=False, logger=None, gt_for_inspection=None):

        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=sample_obs.device)
        obs_hat_list = []
        attns_list = []

        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype, device=sample_obs.device)
        # context
        context_latent = self.encode_obs(sample_obs)
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_hidden, _ = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1]
            )
        self.latent_buffer[:, 0:1] = last_latent
        self.hidden_buffer[:, 0:1] = last_hidden

        # imagine
        imagine_length = random.randint(2, imagine_batch_length) if gt_for_inspection is None else imagine_batch_length
        for i in range(imagine_length): # from TransDreamer: at least imagine 2 steps for TD target
            action = agent.sample(torch.cat([self.calc_slots(self.latent_buffer[:, i:i+1], is_flat=True), self.hidden_buffer[:, i:i+1]], dim=-1))
            self.action_buffer[:, i:i+1] = action

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_hidden, last_attns = self.predict_next(
                self.latent_buffer[:, i:i+1],
                self.action_buffer[:, i:i+1],
            )

            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_hidden
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
            obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])  # uniform sample vec_env

            # visualize attention
            if self.final_feature_width**2 == self.stoch_num_classes:
                attns_list.append(last_attns[::imagine_batch_size//16])  # uniform sample vec_env


        if gt_for_inspection is not None:
            gt_for_inspection = gt_for_inspection[::imagine_batch_size//16]
            obs_hat_list, attns_list = torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1), torch.cat(attns_list, dim=1)
            rollout = torch.cat([gt_for_inspection.unsqueeze(2), obs_hat_list.unsqueeze(2), attns_list], dim=2).cpu().detach() # B L K C H W
            rollout = (rollout * 255.).numpy().astype(np.uint8)
            return rollout

        rollout = None
        if log_video:
            rollout = torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).unsqueeze(2).cpu().detach() # B L K C H W
            rollout = (rollout * 255.).numpy().astype(np.uint8)

        return torch.cat([self.calc_slots(self.latent_buffer, is_flat=True), self.hidden_buffer], dim=-1), self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer, rollout

    def update(self, obs, action, reward, termination, logger=None):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        H, W = obs.shape[-2:]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)

            # slot attention
            z_emb = rearrange(sample, "B L K C -> (B L) K C")
            slots, _ = self.slot_attn(z_emb)
            slots = rearrange(slots, "(B L) N D -> B L N D", B=batch_size)
            
            # decoding image
            obs_hat = self.image_decoder(flattened_sample)

            # transformer
            temporal_mask = get_causal_mask_with_batch_length(batch_length, self.num_slots, flattened_sample.device)
            slots_hat = self.storm_transformer(slots, action, temporal_mask) # B L N D
            # decoding reward and termination with slots_hat
            reward_hat = self.reward_decoder(slots_hat)
            termination_hat = self.termination_decoder(slots_hat)

            # slot space to logits   
            prior_logits, _, mask_as_image = self.dist_head.forward_prior(slots_hat)

            # env loss
            reconstruction_loss = self.mse_loss_func(obs_hat, obs)
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)

            # dyn-rep loss
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
            representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
            
            total_loss = reconstruction_loss + reward_loss + termination_loss + 0.5*dynamics_loss + 0.1*representation_loss

            # visualize attention
            if self.final_feature_width**2 == self.stoch_num_classes:
                mask_as_image = rearrange(mask_as_image, "(B L) N H W -> B L N 1 H W", B=batch_size)
                attns = obs_hat.unsqueeze(2) * mask_as_image + 1. - mask_as_image

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer_storm)  # for clip grad
        self.scaler.unscale_(self.optimizer_sa)  # for clip grad
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.storm_transformer.parameters()) + list(self.image_decoder.parameters()) + \
            list(self.dist_head.parameters()) + list(self.reward_decoder.parameters()) + list(self.termination_decoder.parameters()), 
            max_norm=self.max_grad_norm_storm
        )
        torch.nn.utils.clip_grad_norm_(self.slot_attn.parameters(), max_norm=self.max_grad_norm_sa)
        self.scaler.step(self.optimizer_storm)
        self.scaler.step(self.optimizer_sa)
        self.scaler.update()
        self.optimizer_storm.zero_grad(set_to_none=True)
        self.optimizer_sa.zero_grad(set_to_none=True)

        if self.scheduler_storm is not None:
            self.scheduler_storm.step()
        if self.scheduler_sa is not None:
            self.scheduler_sa.step()

        if self.final_feature_width**2 == self.stoch_num_classes:
            video = torch.cat([obs.unsqueeze(2), torch.clamp(obs_hat, 0, 1).unsqueeze(2), attns], dim=2).cpu().detach() # B L N C H W
        else:
            video = torch.cat([obs.unsqueeze(2), torch.clamp(obs_hat, 0, 1).unsqueeze(2)], dim=2).cpu().detach() # B L N C H W
        video = (video * 255.).numpy().astype(np.uint8)

        logs = {
            "world_model/reconstruction_loss": reconstruction_loss.item(),
            "world_model/reward_loss": reward_loss.item(),
            "world_model/termination_loss": termination_loss.item(),
            "world_model/dynamics_loss": dynamics_loss.item(),
            "world_model/dynamics_real_kl_div": dynamics_real_kl_div.item(),
            "world_model/representation_loss": representation_loss.item(),
            "world_model/representation_real_kl_div": representation_real_kl_div.item(),
            "world_model/total_loss": total_loss.item(),
        }

        return logs, video