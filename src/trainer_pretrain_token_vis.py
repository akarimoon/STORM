import argparse, copy, json, os, pickle, random, shutil, time
from collections import deque
from pathlib import Path

import colorama
import cv2
import h5py
import math
import numpy as np
from einops import rearrange, reduce, repeat
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import wandb

from utils import seed_np_torch
import env_wrapper
import envs
from agents import ActorCriticAgent
from sub_models.slate import SLATE
from replay_buffer import ReplayBuffer


class DataSet(Dataset):
    def __init__(self, data):
        self._data = data
        self._num_samples = data["obss"].shape[0]

    def __getitem__(self, index):
        if index >= self._max_size:
            index = np.random.randint(self._max_size)

        res = {}
        for key in self._data.keys():
            if key == "obss":
                res[key] = torch.Tensor(self._data[key][index]).permute(2,0,1) / 255.0
            elif key == "labels":
                res[key] = torch.LongTensor([self._data[key][index]])
            else:
                if key == "num_objs":
                    continue
                res[key] = torch.Tensor(self._data[key][index])
        return res
    
    def set_size(self, size):
        if size > 0:
            self._max_size = min(size, self._num_samples)
        else:
            self._max_size = self._num_samples

    def __len__(self):
        return self._num_samples


# To get data from files
def get_dataloaders(parent_dir, batch_size, num_workers, replace=False):
    datafile = parent_dir / 'datasets' / 'ocrl_precollected.hdf5'
    f = h5py.File(datafile, "r")
    train_dl = DataLoader(
        DataSet(f["TrainingSet"]), batch_size, num_workers=num_workers, shuffle=True
    )
    val_dl = DataLoader(DataSet(f["ValidationSet"]), batch_size)
    return train_dl, val_dl


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume=True,
            **cfg.wandb
        )

        if cfg.common.seed is not None:
            seed_np_torch(cfg.common.seed)

        self.cfg = cfg
        self.device = torch.device(cfg.common.device)

        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')

        if not cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)

        action_dim = 4

        train_dl, valid_dl = get_dataloaders(
            Path(hydra.utils.get_original_cwd()),
            cfg.training.batch_size,
            num_workers=2
        )
        self.train_dl = train_dl
        self.valid_dl = valid_dl

        # build world model and agent
        self.world_model = instantiate(cfg.world_model, action_dim=action_dim, enable_dict_reset=cfg.token_vis.enable_dict_reset).to(self.device)
            
        self.num_envs = 1

    def run(self) -> None:
        self.total_epochs = 0
        self.total_steps= 0

        #############################
        # start with small dataset
        self.start_num_episodes = -1
        if self.cfg.token_vis.start_with_small_dataset:
            self.start_num_episodes = self.cfg.token_vis.start_num_episodes
        train_dl, valid_dl = get_dataloaders(
            Path(hydra.utils.get_original_cwd()),
            self.cfg.training.batch_size,
            num_workers=2,
        )
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.train_dl.dataset.set_size(self.start_num_episodes)
        self.valid_dl.dataset.set_size(self.start_num_episodes)
        #############################

        # sample and train
        for _ in range(self.cfg.common.num_epochs):
            for idx, batch in enumerate(tqdm(self.train_dl)):
                obs = batch["obss"].to(self.device)
                self.train_world_model(obs)

                if self.total_steps % (self.cfg.training.inspect_every_steps//self.num_envs) == 0:
                    valid_batch = next(iter(self.valid_dl))
                    valid_obs = valid_batch["obss"].to(self.device)
                    self.inspect_world_model(valid_obs)

            self.save()
            self.total_epochs += 1
    
    def train_world_model(self, obs) -> None:
        start_time = time.time()

        logs, video = self.world_model.update(obs.unsqueeze(1))
        logs["duration/train_wm"] = time.time() - start_time

        self.log(logs)
        if self.total_steps % (self.cfg.training.save_every_steps//self.num_envs) == 0: # only save video once in a while
            # if video.shape[2] >= 3:
            wandb.log({"step": self.total_steps//self.num_envs, "image/reconstruction_slots": wandb.Video(rearrange(video[:4], 'B L N C H W -> L C (B H) (N W)'), fps=4)})
            # else:
            #     wandb.log({"step": self.total_steps//self.num_envs, "image/reconstruction": wandb.Video(rearrange(video[:4], 'B L N C H W -> L C (B H) (N W)'), fps=4)})

        if self.total_steps % (self.cfg.training.vis_every_steps//self.num_envs) == 0: # save img in media_dir
            rand_idx = np.random.randint(video.shape[0])
            full_plot = rearrange(torch.tensor(video[rand_idx]).float().div(255.).permute(1, 0, 2, 3, 4), 'N L C H W -> (N L) C H W')
            save_image(full_plot, self.media_dir / f"reconstruction_{self.total_steps//self.num_envs}.png", nrow=video.shape[1])

        self.total_steps += 1

        #############################
        # increase dataset
        if self.cfg.token_vis.increase_dataset and self.total_steps > self.cfg.token_vis.increase_after_steps:
            self.train_dl.dataset.set_size(self.start_num_episodes + self.cfg.token_vis.increase_per_steps * self.total_steps)
            self.valid_dl.dataset.set_size(self.start_num_episodes + self.cfg.token_vis.increase_per_steps * self.total_steps)
        #############################

    @torch.no_grad()
    def inspect_world_model(self, obs) -> None:
        self.world_model.eval()

        with torch.no_grad():
            video = self.world_model.inspect(obs.unsqueeze(1))

        wandb.log({"step": self.total_steps//self.num_envs, "image/reconstruction_using_hard": wandb.Video(rearrange(video[:4], 'B L N C H W -> L C (B H) (N W)'), fps=4)})

        rand_idx = np.random.randint(video.shape[0])
        full_plot = rearrange(torch.tensor(video[rand_idx]).float().div(255.).permute(1, 0, 2, 3, 4), 'N L C H W -> (N L) C H W')
        save_image(full_plot, self.media_dir / f"reconstruction_inspect_{self.total_steps//self.num_envs}.png", nrow=video.shape[1])

    @torch.no_grad()
    def save(self) -> None:
        print(colorama.Fore.GREEN + f"Saving model at epoch {self.total_epochs}" + colorama.Style.RESET_ALL)
        torch.save(self.world_model.state_dict(), self.ckpt_dir / f"world_model_ep{self.total_epochs}.pth")

    def log(self, logs) -> None:
        wandb.log({"step": self.total_steps//self.num_envs, **logs})