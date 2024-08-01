import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size, enable_reset=False):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)
        self.vocab_size = vocab_size

        self.usage_count = torch.zeros(vocab_size)
        self.step_count = 0
        self._thres = 0.1
        self.last_seen_x = None
        self.init_dict = self.dictionary.weight.data.clone()
        self.dist_from_init = torch.zeros(vocab_size)

        self.enable_reset = enable_reset

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size

        if self.training:
            self.last_seen_x = rearrange(token_embs, "B K C -> (B K) C")
            self.usage_count += F.one_hot(tokens, num_classes=self.vocab_size).float().sum(dim=(0,1)).detach().cpu()
            self.dist_from_init = torch.norm(self.dictionary.weight.data - self.init_dict.to(x.device), dim=-1)
            self.step_count += 1

            if self.enable_reset and self.step_count % 1000 == 0:
                self.reset_unused_tokens(x.device)

        return token_embs
    
    def reset_unused_tokens(self, device):
        target_ids = torch.where(self.usage_count / self.step_count < self._thres)[0]
        most_used_ids = torch.argsort(self.usage_count, descending=True)[1:]
        rand_ids = torch.tensor(np.random.choice(most_used_ids, len(target_ids), replace=False)).to(device)
        self.dictionary.weight.data[target_ids] = self.dictionary.weight.data[rand_ids] + torch.randn_like(self.dictionary.weight.data[rand_ids]) / 100


"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from einops import rearrange, pack, unpack
from typing import List, Optional
from torch.cuda.amp import autocast

# helper functions
def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers
def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

# main class
class FSQ(nn.Module):
    def __init__(self, levels: List[int], dim: Optional[int] = None, num_codebooks = 1, 
                 keep_num_codebooks_dim: Optional[bool] = None, scale: Optional[float] = None):
        super().__init__()
        _levels = torch.tensor(levels, dtype=torch.int64)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int64)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width
    
    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int64)
    
    def indices_to_codes(self, indices: torch.Tensor, project_out=True) -> torch.Tensor:
        """Inverse of `codes_to_indices`."""
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        return codes

    @autocast(enabled=False)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - codebook dim (= # of levels * # of codebooks)
        """
        z = self.project_in(z)
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        out = self.project_out(codes)

        return out, codes, indices
