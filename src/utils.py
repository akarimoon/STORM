import os
import random
from typing import Optional
import numpy as np
import torch
from einops import repeat
from contextlib import contextmanager
import time


def seed_np_torch(seed=20001118):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# class Logger():
#     def __init__(self, path) -> None:
#         self.writer = SummaryWriter(logdir=path, flush_secs=1)
#         self.tag_step = {}

#     def log(self, tag, value):
#         if tag not in self.tag_step:
#             self.tag_step[tag] = 0
#         else:
#             self.tag_step[tag] += 1
#         if "video" in tag:
#             self.writer.add_video(tag, value, self.tag_step[tag], fps=15)
#         elif "images" in tag:
#             self.writer.add_images(tag, value, self.tag_step[tag])
#         elif "hist" in tag:
#             self.writer.add_histogram(tag, value, self.tag_step[tag])
#         else:
#             self.writer.add_scalar(tag, value, self.tag_step[tag])


class EMAScalar():
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar

def linear_warmup_exp_decay(warmup_steps: Optional[int] = None, exp_decay_rate: Optional[float] = None, exp_decay_steps: Optional[int] = None):
    assert (exp_decay_steps is None) == (exp_decay_rate is None)
    use_exp_decay = exp_decay_rate is not None
    if warmup_steps is not None:
        assert warmup_steps > 0
    def lr_lambda(step):
        multiplier = 1.0
        if warmup_steps is not None and step < warmup_steps:
            multiplier *= step / warmup_steps
        if use_exp_decay:
            multiplier *= exp_decay_rate ** (step / exp_decay_steps)
        return multiplier
    return lr_lambda

def configure_optimizer(model, learning_rate, weight_decay, *blacklist_module_names):
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                no_decay.add(fpn)
            elif 'bias' in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer