import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange


class OP3PhysicsNetwork(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_slots, num_layers, num_heads, max_length, dropout, 
                 emb_type, skip_connection=False):
        super().__init__()
        feat_dim = stoch_dim
        action_enc_dim = 32
        effect_dim = 32

        self.action_dim = action_dim
        self.feat_dim = stoch_dim
        self.num_slots = num_slots
        self.skip_connection = skip_connection

        self.inertia_encoder = nn.Sequential(
            nn.Linear(stoch_dim, feat_dim),
            nn.ELU(),
            nn.Linear(feat_dim, stoch_dim),
            nn.ELU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, feat_dim),
            nn.ELU(),
            nn.Linear(feat_dim, action_enc_dim),
            nn.ELU(),
        )
        self.action_effect_network = nn.Sequential(
            nn.Linear(stoch_dim+action_enc_dim, feat_dim),
            nn.ELU(),
            nn.Linear(feat_dim, stoch_dim),
            nn.ELU(),
        )
        self.action_attn_network = nn.Sequential(
            nn.Linear(stoch_dim+action_enc_dim, feat_dim),
            nn.ELU(),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid()
        )

        self.pairwise_encoder = nn.Sequential(
            nn.Linear(stoch_dim*2, feat_dim*2),
            nn.ELU(),
            nn.Linear(feat_dim*2, stoch_dim),
            nn.ELU(),
        )
        self.interaction_effect_network = nn.Sequential(
            nn.Linear(stoch_dim, feat_dim),
            nn.ELU(),
            nn.Linear(feat_dim, effect_dim),
            nn.ELU(),
        )
        self.interaction_attn_network = nn.Sequential(
            nn.Linear(stoch_dim, feat_dim),
            nn.ELU(),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid()
        )
        self.final_merge_network = nn.Sequential(
            nn.Linear(stoch_dim+effect_dim, feat_dim),
            nn.ELU(),
            nn.Linear(feat_dim, stoch_dim),
            nn.ELU(),
        )

        self.output_network = nn.Sequential(
            nn.Linear(stoch_dim, feat_dim),
            nn.ELU(),
            nn.Linear(feat_dim, stoch_dim),
        )

    def forward(self, samples, action, mask):
        '''
        Normal forward pass

        samples: (B, L, N, D)
        action: (B, L, 1)
        '''
        B, L, N, D = samples.shape

        action = F.one_hot(action.long(), self.action_dim).float()
        action = repeat(action, "B L A -> B L N A", N=N)

        samples = rearrange(samples, "B L N D -> (B N) L D")
        action = rearrange(action, "B L N A -> (B N) L A")

        outs = []
        for t in range(L):
            sample = samples[:, t]
            act = action[:, t]

            # Encode sample
            state_enc_flat = self.inertia_encoder(sample)

            # Encode action
            action_enc = self.action_encoder(act)

            state_enc_actions = torch.cat([state_enc_flat, action_enc], dim=-1)
            state_action_effect = self.action_effect_network(state_enc_actions)
            state_action_attn = self.action_attn_network(state_enc_actions)
            state_enc = (state_action_effect*state_action_attn).view(B, N, D)

            #Create array of all pairs
            pairs = []
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    pairs.append(torch.cat([state_enc[:, i], state_enc[:, j]], dim=-1))

            all_pairs = torch.stack(pairs, dim=1).view(B*N, N-1, -1)

            pairwise_interaction = self.pairwise_encoder(all_pairs)
            effect = self.interaction_effect_network(pairwise_interaction)
            attn = self.interaction_attn_network(pairwise_interaction)
            total_effect = (effect*attn).sum(dim=1)

            state_and_effect = torch.cat([state_enc.view(B*N, D), total_effect], dim=-1)
            aggregate_state = self.final_merge_network(state_and_effect)
            out = self.output_network(aggregate_state)
            outs.append(out)

        out = torch.stack(outs, dim=1)
        out = rearrange(out, "(B N) L D -> B L N D", N=N)

        return out

    def reset_kv_cache_list(self, batch_size, dtype, device):
        '''
        Reset self.kv_cache_list
        '''
        pass

    def forward_with_kv_cache(self, samples, action):
        '''
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        '''
        return self.forward(samples, action, None)