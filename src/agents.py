import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
from torch.cuda.amp import autocast

from sub_models.functions_losses import SymLogTwoHotLoss
from utils import EMAScalar, linear_warmup_exp_decay

def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage*len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per


def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=torch.float32):
    # Invert termination to have 0 if the episode ended and 1 otherwise
    inv_termination = (termination * -1) + 1

    batch_size, batch_length = rewards.shape[:2]
    # gae_step = torch.zeros((batch_size, ), dtype=dtype, device="cuda")
    gamma_return = torch.zeros((batch_size, batch_length+1), dtype=dtype, device=rewards.device)
    gamma_return[:, -1] = values[:, -1]
    for t in reversed(range(batch_length)):  # with last bootstrap
        gamma_return[:, t] = \
            rewards[:, t] + \
            gamma * inv_termination[:, t] * (1-lam) * values[:, t] + \
            gamma * inv_termination[:, t] * lam * gamma_return[:, t+1]
    return gamma_return[:, :-1]


class ActorCriticAgent(nn.Module):
    def __init__(self, feat_dim, num_layers, hidden_dim, action_dim, gamma, lambd, entropy_coef,
                 lr, max_grad_norm) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_amp = True
        self.tensor_dtype = torch.float16 if self.use_amp else torch.float32

        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)

        actor = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ]
        for i in range(num_layers - 1):
            actor.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
        self.actor = nn.Sequential(
            *actor,
            nn.Linear(hidden_dim, action_dim)
        )

        critic = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ]
        for i in range(num_layers - 1):
            critic.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])

        self.critic = nn.Sequential(
            *critic,
            nn.Linear(hidden_dim, 255)
        )
        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def load(self, path_to_checkpoint, device):
        state_dict = torch.load(path_to_checkpoint, map_location=device)
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))

    def policy(self, x):
        logits = self.actor(x)
        return logits

    def value(self, x):
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def slow_value(self, x):
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def get_logits_raw_value(self, x):
        logits = self.actor(x)
        raw_value = self.critic(x)
        return logits, raw_value

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        self.eval()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            logits = self.policy(latent)
            dist = distributions.Categorical(logits=logits)
            if greedy:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
        return action

    def sample_as_env_action(self, latent, greedy=False):
        action = self.sample(latent, greedy)
        return action.detach().cpu().squeeze(-1).numpy()

    def update(self, latent, action, old_logprob, old_value, reward, termination, logger=None):
        '''
        Update policy and value model
        '''
        self.train()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            logits, raw_value = self.get_logits_raw_value(latent)
            dist = distributions.Categorical(logits=logits[:, :-1])
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # decode value, calc lambda return
            slow_value = self.slow_value(latent)
            slow_lambda_return = calc_lambda_return(reward, slow_value, termination, self.gamma, self.lambd)
            value = self.symlog_twohot_loss.decode(raw_value)
            lambda_return = calc_lambda_return(reward, value, termination, self.gamma, self.lambd)

            # update value function with slow critic regularization
            value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach())
            slow_value_regularization_loss = self.symlog_twohot_loss(raw_value[:, :-1], slow_lambda_return.detach())

            lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound-lower_bound
            norm_ratio = torch.max(torch.ones(1).to(latent.device), S)  # max(1, S) in the paper
            norm_advantage = (lambda_return-value[:, :-1]) / norm_ratio
            policy_loss = -(log_prob * norm_advantage.detach()).mean()

            entropy_loss = entropy.mean()

            loss = policy_loss + value_loss + slow_value_regularization_loss - self.entropy_coef * entropy_loss

        # gradient descent
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        norm_ac = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
        # print(norm_ac)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)

        self.update_slow_critic()

        logs = {
            'actor_critic/policy_loss': policy_loss.item(),
            'actor_critic/value_loss': value_loss.item(),
            'actor_critic/entropy_loss': entropy_loss.item(),
            'actor_critic/raw_norm_ratio': S.item(),
            'actor_critic/norm_ratio': norm_ratio.item(),
            'actor_critic/total_loss': loss.item(),
            'actor_critic/norm_ac': norm_ac.item()
        }
        return logs


class TransformerWithCLS(nn.Module):
    def __init__(self, in_features, d_model, num_heads, num_layers, norm_first=False):
        super(TransformerWithCLS, self).__init__()
        self._linear = nn.Linear(in_features, d_model)
        self._cls_token = nn.Parameter(torch.zeros(d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads,
            batch_first=True,
            #norm_first=norm_first
        )
        self._trans = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B, L = state.shape[:2]
        state = rearrange(state, "B L N D -> (B L) N D")

        state = self._linear(state)
        state = torch.cat(
            [self._cls_token.repeat(B*L, 1, 1), state], dim=1
        )
        state = rearrange(state, "B N D -> N B D")

        feats = self._trans(state)[0]
        feats = rearrange(feats, "(B L) D -> B L D", B=B)

        return feats
    

class MLP(nn.Module):
    def __init__(self, in_features, d_model):
        super(MLP, self).__init__()
        self._linear = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B, L = state.shape[:2]
        state = rearrange(state, "B L N D -> (B L) N D")
        state = state.sum(dim=1)

        feats = self._linear(state)
        feats = rearrange(feats, "(B L) D -> B L D", B=B)

        return feats


class OCActorCriticAgent(ActorCriticAgent):
    def __init__(self, feat_dim, num_heads, num_layers, hidden_dim, mlp_hidden_dim, action_dim, gamma, lambd, entropy_coef,
                 lr, max_grad_norm) -> None:
        super().__init__(feat_dim, num_layers, hidden_dim, action_dim, gamma, lambd, entropy_coef, lr, max_grad_norm)

        shared_transformer = TransformerWithCLS(feat_dim, hidden_dim, num_heads, num_layers)
        # shared_transformer = MLP(feat_dim, hidden_dim)
        shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.actor = nn.Sequential(
            shared_transformer,
            shared_mlp,
            nn.Linear(mlp_hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            shared_transformer,
            shared_mlp,
            nn.Linear(mlp_hidden_dim, 255)
        )

        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.max_grad_norm, eps=1e-5)
        self.scheduler = None
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_warmup_exp_decay(15000))

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)