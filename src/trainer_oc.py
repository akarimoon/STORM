import argparse, copy, json, os, pickle, random, shutil, time
from collections import deque
from pathlib import Path

import colorama
import cv2
import numpy as np
from einops import rearrange
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import wandb

from utils import seed_np_torch
import env_wrapper
import envs
from agents import ActorCriticAgent
from sub_models.world_models import WorldModel
from replay_buffer import ReplayBuffer


def build_single_atari_env(env_name, image_size, seed):
    env = gymnasium.make(f"ALE/{env_name}-v5", full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfo(env)
    return env

def build_single_ocrl_env(env_config, image_size, seed, max_step=1000):
    env = getattr(envs, env_config.env)(env_config, seed)
    env = env_wrapper.OCRLMaxStepWrapper(env, max_step=max_step)
    return env

def build_vec_env(env_name, env_type, image_size, num_envs, seed, max_step=1000):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        if env_type == "atari":
            return lambda: build_single_atari_env(env_name, image_size, seed)
        elif env_type == "ocrl":
            return lambda: build_single_ocrl_env(env_name, image_size, seed, max_step=max_step)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


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

        # getting action_dim with dummy env
        if cfg.envs.env_type == "atari":
            dummy_env = build_single_atari_env(cfg.envs.env_name, cfg.common.image_size, seed=0)
        elif cfg.envs.env_type == "ocrl":
            dummy_env = build_single_ocrl_env(cfg.env_config, cfg.common.image_size, seed=0, max_step=cfg.envs.max_step)
        action_dim = dummy_env.action_space.n

        # build world model and agent
        self.world_model = instantiate(cfg.world_model, action_dim=action_dim).to(self.device)
        self.agent = instantiate(cfg.agent, action_dim=action_dim, feat_dim=self.world_model.agent_input_dim).to(self.device)
        
        if cfg.common.load_pretrained:
            path_to_checkpoint = Path(hydra.utils.get_original_cwd()) / "pretrained" / cfg.initialization.pretrained_ckpt
            print(colorama.Fore.MAGENTA + f"loading pretrained model from {path_to_checkpoint}" + colorama.Style.RESET_ALL)
            self.world_model.load(path_to_checkpoint, self.device)

        # build replay buffer
        self.replay_buffer = instantiate(cfg.replay_buffer, obs_shape=(cfg.common.image_size, cfg.common.image_size, 3), num_envs=cfg.envs.num_envs, device=self.device)

        # judge whether to load demonstration trajectory
        if cfg.training.use_demonstration:
            print(colorama.Fore.MAGENTA + f"loading demonstration trajectory from {cfg.training.trajectory_path}" + colorama.Style.RESET_ALL)
            self.replay_buffer.load_trajectory(path=cfg.training.trajectory_path)

        self.num_envs = cfg.envs.num_envs

    def run(self) -> None:
        self.total_steps = 0

        # build vec env, not useful in the Atari100k setting
        # but when the max_steps is large, you can use parallel envs to speed up
        if self.cfg.envs.env_type == "atari":
            self.vec_env = build_vec_env(self.cfg.envs.env_name, self.cfg.envs.env_type, self.cfg.common.image_size, num_envs=self.num_envs, 
                                         seed=self.cfg.common.seed)
        elif self.cfg.envs.env_type == "ocrl":
            self.vec_env = build_vec_env(self.cfg.env_config, self.cfg.envs.env_type, self.cfg.common.image_size, num_envs=self.num_envs,
                                        seed=self.cfg.common.seed, max_step=self.cfg.envs.max_step)
        print("Current env: " + colorama.Fore.YELLOW + f"{self.cfg.envs.env_name}" + colorama.Style.RESET_ALL)

        # reset envs and variables
        sum_reward = np.zeros(self.num_envs)
        current_obs, current_info = self.vec_env.reset()
        context_obs = deque(maxlen=16)
        context_action = deque(maxlen=16)

        # sample and train
        for _ in tqdm(range(self.cfg.common.max_steps//self.num_envs)):
            context_obs, context_action, sum_reward, current_obs, current_info = self.collect(context_obs, context_action, sum_reward, current_obs, current_info)

            to_log = []
            if self.replay_buffer.ready():
                if self.total_steps % (self.cfg.training.train_dynamics_every_steps//self.num_envs) == 0:
                    to_log += self.train_world_model()

                if self.total_steps % (self.cfg.training.train_agent_every_steps//self.num_envs) == 0 and self.total_steps*self.num_envs >= 0:
                    to_log += self.train_agent()

                if self.total_steps % (self.cfg.training.save_every_steps//self.num_envs) == 0:
                    self.save()

                if self.total_steps % (self.cfg.training.inspect_every_steps//self.num_envs) == 0:
                    to_log += self.inspect_reconstruction()
                    to_log += self.inspect_world_model()

                    to_log += self.eval()

            for logs in to_log:
                self.log(logs)
            self.total_steps += 1
        
        wandb.finish()

    @torch.no_grad()
    def collect(self, context_obs, context_action, sum_reward, current_obs, current_info):
        start_time = time.time()

        if self.replay_buffer.ready():
            self.world_model.eval()
            self.agent.eval()
            if len(context_action) == 0:
                action = self.vec_env.action_space.sample()
            else:
                context_latent = self.world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=1)
                model_context_action = torch.Tensor(model_context_action).to(self.device)
                prior_flattened_sample, last_dist_feat = self.world_model.calc_last_dist_feat(context_latent, model_context_action)
                if self.world_model.agent_state_type == "latent":
                    state = prior_flattened_sample
                elif self.world_model.agent_state_type == "hidden":
                    state = last_dist_feat
                else:
                    state = torch.cat([prior_flattened_sample, last_dist_feat], dim=-1)
                action = self.agent.sample_as_env_action(
                    state,
                    greedy=False
                )

            context_obs.append(rearrange(torch.Tensor(current_obs).to(self.device), "B H W C -> B 1 C H W")/255)
            context_action.append(action)
        else:
            action = self.vec_env.action_space.sample()

        obs, reward, done, truncated, info = self.vec_env.step(action)
        if self.cfg.envs.env_type == "atari":
            self.replay_buffer.append(current_obs, action, reward, np.logical_or(done, info["life_loss"]))
        elif self.cfg.envs.env_type == "ocrl":
            action = action.reshape(1,)
            self.replay_buffer.append(current_obs, action, reward, done)

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        
        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(self.num_envs):
                if done_flag[i]:
                    wandb.log({
                        "episode_steps": current_info["episode_frame_number"][i]//4,  # framskip=4
                        "sample/reward": sum_reward[i],
                        "replay_buffer/length": len(self.replay_buffer),
                    })
                    sum_reward[i] = 0
                    self.replay_buffer.end_episode()
                    current_obs, current_info = self.vec_env.reset()
                    context_obs = deque(maxlen=16)
                    context_action = deque(maxlen=16)

        # if self.replay_buffer.ready():
        #      wandb.log({"step": self.total_steps//self.num_envs, "duration/collect": time.time()-start_time})

        return context_obs, context_action, sum_reward, current_obs, current_info
    
    def train_world_model(self) -> None:
        start_time = time.time()

        obs, action, reward, termination = self.replay_buffer.sample(self.cfg.training.batch_size, self.cfg.training.demonstration_batch_size, self.cfg.training.batch_length,
            sample_from_start=True)
        logs, video = self.world_model.update(obs, action, reward, termination)
        logs["duration/train_wm"] = time.time() - start_time

        # self.log(logs)
        if self.total_steps % (self.cfg.training.save_every_steps//self.num_envs) == 0: # only save video once in a while
            if video.shape[2] >= 3:
                # wandb.log({"step": self.total_steps//self.num_envs, "video/reconstruction_slots": wandb.Video(rearrange(video[:4], 'B L N C H W -> L C (B H) (N W)'), fps=4)})
                logs["video/reconstruction_slots"] = wandb.Video(rearrange(video[:4], 'B L N C H W -> L C (B H) (N W)'), fps=4)
            else:
                # wandb.log({"step": self.total_steps//self.num_envs, "video/reconstruction": wandb.Video(rearrange(video[:4], 'B L N C H W -> L C (B H) (N W)'), fps=4)})
                logs["video/reconstruction"] = wandb.Video(rearrange(video[:4], 'B L N C H W -> L C (B H) (N W)'), fps=4)

        if self.total_steps % (self.cfg.training.vis_every_steps//self.num_envs) == 0: # save img in media_dir
            rand_idx = np.random.randint(video.shape[0])
            full_plot = rearrange(torch.tensor(video[rand_idx]).float().div(255.).permute(1, 0, 2, 3, 4), 'N L C H W -> (N L) C H W')
            save_image(full_plot, self.media_dir / f"reconstruction_{self.total_steps//self.num_envs}.png", nrow=video.shape[1], pad_value=1.0)

        return [logs]

    def train_agent(self) -> None:
        self.world_model.eval()
        self.agent.eval()

        start_time = time.time()
        with torch.no_grad():
            sample_obs, sample_action, sample_reward, sample_termination = self.replay_buffer.sample(
                self.cfg.training.imagine_batch_size, self.cfg.training.imagine_demonstration_batch_size, self.cfg.training.imagine_context_length,
                sample_from_start=False)
            imagine_latent, agent_action, imagine_reward, imagine_termination, video = self.world_model.imagine_data(
                self.agent, sample_obs, sample_action,
                imagine_batch_size=self.cfg.training.imagine_batch_size+self.cfg.training.imagine_demonstration_batch_size,
                imagine_batch_length=self.cfg.training.imagine_batch_length,
                log_video=True,
            )
        imagine_time = time.time() - start_time
        start_time = time.time()

        logs = self.agent.update(
            latent=imagine_latent,
            action=agent_action,
            old_logprob=None,
            old_value=None,
            reward=imagine_reward,
            termination=imagine_termination,
        )
        logs["duration/imagination"] = imagine_time
        logs["duration/train_agent"] = time.time() - start_time

        # self.log(logs)
        logs = {}
        if self.total_steps % (self.cfg.training.save_every_steps//self.num_envs) == 0: # only save video once in a while
            # wandb.log({"step": self.total_steps//self.num_envs, "video/rollout": wandb.Video(rearrange(video[:4], 'B L N C H W -> L C (B H) (N W)'), fps=4)})
            logs["video/rollout"] = wandb.Video(rearrange(video[:4], 'B L N C H W -> L C (B H) (N W)'), fps=4)

        if self.total_steps % (self.cfg.training.vis_every_steps//self.num_envs) == 0: # save img in media_dir
            rand_idx = np.random.randint(video.shape[0])
            full_plot = rearrange(torch.tensor(video[rand_idx]).float().div(255.).permute(1, 0, 2, 3, 4), 'N L C H W -> (N L) C H W')
            save_image(full_plot, self.media_dir / f"rollout_{self.total_steps//self.num_envs}.png", nrow=video.shape[1], pad_value=1.0)

        return [logs]

    @torch.no_grad()
    def eval(self) -> None:
        self.world_model.eval()
        self.agent.eval()

        self.test_env = build_vec_env(self.cfg.env_config, self.cfg.envs.env_type, self.cfg.common.image_size, num_envs=self.num_envs,
                                        seed=self.cfg.common.seed, max_step=self.cfg.evaluation.max_step)
        
        all_rewards = []
        for _ in range(self.cfg.evaluation.num_episodes):
            current_obs, info = self.test_env.reset()
            context_obs = deque(maxlen=16)
            context_action = deque(maxlen=16)
            context_obs.append(rearrange(torch.Tensor(current_obs).to(self.device), "B H W C -> B 1 C H W")/255)
            context_action.append(self.test_env.action_space.sample())

            done = False
            rewards = 0
            while not done:
                context_latent = self.world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=1)
                model_context_action = torch.Tensor(model_context_action).to(self.device)
                prior_flattened_sample, last_dist_feat = self.world_model.calc_last_dist_feat(context_latent, model_context_action)
                if self.world_model.agent_state_type == "latent":
                    state = prior_flattened_sample
                elif self.world_model.agent_state_type == "hidden":
                    state = last_dist_feat
                else:
                    state = torch.cat([prior_flattened_sample, last_dist_feat], dim=-1)
                action = self.agent.sample_as_env_action(state, greedy=False)
                
                current_obs, reward, done, truncated, info = self.test_env.step(action)
                context_obs.append(rearrange(torch.Tensor(current_obs).to(self.device), "B H W C -> B 1 C H W")/255)
                context_action.append(action)

                rewards += reward
            all_rewards.append(rewards)
        return [{"eval/mean_reward": np.mean(all_rewards), "eval/std_reward": np.std(all_rewards)}]

    @torch.no_grad()
    def inspect_reconstruction(self) -> None:
        self.world_model.eval()
        with torch.no_grad():
            sample_obs, sample_action, sample_reward, sample_termination = self.replay_buffer.sample(
                self.cfg.training.imagine_batch_size, self.cfg.training.imagine_demonstration_batch_size, self.cfg.training.imagine_context_length)
            video = self.world_model.inspect_reconstruction(sample_obs, tau=0.1)

        logs = {}
        if video.shape[2] >= 3:
            logs = {"video/reconstruction_using_hard": wandb.Video(rearrange(video[:4], 'B L N C H W -> L C (B H) (N W)'), fps=4)}

        rand_idx = np.random.randint(video.shape[0])
        full_plot = rearrange(torch.tensor(video[rand_idx]).float().div(255.).permute(1, 0, 2, 3, 4), 'N L C H W -> (N L) C H W')
        save_image(full_plot, self.media_dir / f"reconstruction_inspect_{self.total_steps//self.num_envs}.png", nrow=video.shape[1], pad_value=1.0)

        return [logs]

    @torch.no_grad()
    def inspect_world_model(self) -> None:
        self.world_model.eval()
        self.agent.eval()

        with torch.no_grad():
            sample_obs, sample_action, sample_reward, sample_termination = self.replay_buffer.sample(
                self.cfg.training.inspect_batch_size, 0, self.cfg.training.inspect_context_length+self.cfg.training.inspect_batch_length)
            context_obs, context_action = sample_obs[:, :self.cfg.training.inspect_context_length], sample_action[:, :self.cfg.training.inspect_context_length]
            gt_obs, gt_action = sample_obs[:, self.cfg.training.inspect_context_length:], sample_action[:, self.cfg.training.inspect_context_length:]
            video = self.world_model.inspect_rollout(
                context_obs, context_action, gt_obs, gt_action,
                imagine_batch_size=self.cfg.training.inspect_batch_size,
                imagine_batch_length=self.cfg.training.inspect_batch_length,
            )

        logs = {}
        if video.shape[2] >= 3:
            logs = {"video/rollout_slots_with_gt": wandb.Video(rearrange(video[:4], 'B L N C H W -> L C (B H) (N W)'), fps=4)}
        else:
            logs = {"video/rollout_with_gt": wandb.Video(rearrange(video[:4], 'B L N C H W -> L C (B H) (N W)'), fps=4)}

        rand_idx = np.random.randint(video.shape[0])
        full_plot = rearrange(torch.tensor(video[rand_idx]).float().div(255.).permute(1, 0, 2, 3, 4), 'N L C H W -> (N L) C H W')
        save_image(full_plot, self.media_dir / f"rollout_inspect_{self.total_steps//self.num_envs}.png", nrow=video.shape[1], pad_value=1.0)

        return [logs]

    @torch.no_grad()
    def save(self) -> None:
        print(colorama.Fore.GREEN + f"Saving model at total steps {self.total_steps}" + colorama.Style.RESET_ALL)
        torch.save(self.world_model.state_dict(), self.ckpt_dir / f"world_model.pth")
        torch.save(self.agent.state_dict(), self.ckpt_dir / "agent.pth")

    def log(self, logs) -> None:
        wandb.log({"step": self.total_steps//self.num_envs, **logs})