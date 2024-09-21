import argparse, copy, json, os, pickle, random, shutil, time
from collections import deque
from pathlib import Path

import colorama
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from einops import rearrange
import hydra
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import wandb

from utils import seed_np_torch
import envs
from env_wrapper import build_single_env, build_vec_env
from agents import ActorCriticAgent
from sub_models.oc_world_models import OCWorldModel
from sub_models.ocq_world_models import OCQuantizedWorldModel
from replay_buffer import ReplayBuffer


def main(config_name, checkpoint_path):
    with initialize(version_base=None, config_path=os.path.join("..", checkpoint_path, "config")):
        cfg = compose(config_name=config_name)

    with initialize(version_base=None, config_path=os.path.join("..", "outputs/2024-08-17/09-05-45", "config")):
        cfg_storm = compose(config_name=config_name)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if cfg.common.seed is not None:
        seed_np_torch(cfg.common.seed)

    device = torch.device(cfg.common.device)

    dummy_env = build_single_env(cfg.envs, cfg.common.image_size, seed=0)
    action_dim = dummy_env.action_space.n

    # build world model and agent
    world_model = instantiate(cfg.world_model, action_dim=action_dim).to(device)
    
    wm_storm = instantiate(cfg_storm.world_model, action_dim=action_dim).to(device)
    agent = instantiate(cfg_storm.agent, action_dim=action_dim, feat_dim=wm_storm.agent_input_dim).to(device)
    
    if checkpoint_path is not None:
        path_to_checkpoint = checkpoint_path
        print(colorama.Fore.MAGENTA + f"loading pretrained model from {path_to_checkpoint}" + colorama.Style.RESET_ALL)
        world_model.load_state_dict(torch.load(os.path.join(path_to_checkpoint, "checkpoints", "world_model.pth"), map_location=device))
        agent.load_state_dict(torch.load(os.path.join("outputs/2024-08-17/09-05-45", "checkpoints", "agent.pth"), map_location=device))

    # build replay buffer
    replay_buffer = instantiate(cfg.replay_buffer, obs_shape=(cfg.common.image_size, cfg.common.image_size, 3), num_envs=cfg.envs.num_envs, device=device)
    num_envs = cfg.envs.num_envs

    world_model.eval()
    agent.eval()

    vec_env = build_vec_env(cfg.envs, cfg.common.image_size, num_envs, seed=cfg.common.seed)
    print("Current env: " + colorama.Fore.YELLOW + f"{cfg.envs.env_name}" + colorama.Style.RESET_ALL)

    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    final_rewards = []
    while not replay_buffer.ready():
        with torch.no_grad():
            if len(context_action) == 0:
                action = vec_env.action_space.sample()
            else:
                context_latent = wm_storm.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=1)
                model_context_action = torch.Tensor(model_context_action).to(device)
                prior_flattened_sample, last_dist_feat = wm_storm.calc_last_dist_feat(context_latent, model_context_action)
                if wm_storm.agent_state_type == "latent":
                    state = prior_flattened_sample
                elif wm_storm.agent_state_type == "hidden":
                    state = last_dist_feat
                else:
                    state = torch.cat([prior_flattened_sample, last_dist_feat], dim=-1)
                action = agent.sample_as_env_action(
                    state,
                    greedy=True # False or True?
                )

        context_obs.append(rearrange(torch.Tensor(current_obs).to(device), "B H W C -> B 1 C H W")/255)
        context_action.append(action)

        obs, reward, done, truncated, info = vec_env.step(action)
        if cfg.envs.env_type == "atari":
            replay_buffer.append(current_obs, action, reward, np.logical_or(done, info["life_loss"]))
        elif cfg.envs.env_type == "ocrl":
            action = action.reshape(1,)
            replay_buffer.append(current_obs, action, reward, done)

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        
        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    final_rewards.append(sum_reward[i])
                    sum_reward[i] = 0
                    replay_buffer.end_episode()
                    current_obs, current_info = vec_env.reset()
                    context_obs = deque(maxlen=16)
                    context_action = deque(maxlen=16)

    print(replay_buffer.get_stats())
    num_batches = 10
    reward_losses = 0.
    termination_losses = 0.
    with torch.no_grad():
        for _ in range(num_batches):
            sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(1, 0, cfg.training.inspect_context_length+cfg.training.inspect_batch_length)
            if sample_obs.shape[1] <= cfg.training.inspect_context_length + 1:
                continue
            context_obs_, context_action_ = sample_obs[:, :cfg.training.inspect_context_length], sample_action[:, :cfg.training.inspect_context_length]
            gt_obs, gt_action = sample_obs[:, cfg.training.inspect_context_length:], sample_action[:, cfg.training.inspect_context_length:]
            gt_reward, gt_termination = sample_reward[:, cfg.training.inspect_context_length:], sample_termination[:, cfg.training.inspect_context_length:]
            video = world_model.inspect_rollout(
                context_obs_, context_action_, gt_obs, gt_action,
                imagine_batch_size=1,
                imagine_batch_length=gt_obs.shape[1],
            )
            # print(gt_reward)
            # print(world_model.reward_hat_buffer)
            # print(gt_termination)
            # print(world_model.termination_hat_buffer)

            # calc mse between sample_reward and world_model.reward_hat_buffer (both are torch.Tensor)
            reward_loss = F.mse_loss(world_model.reward_hat_buffer, gt_reward)
            # calc confusion matrix between sample_termination and world_model.termination_hat_buffer (both are torch.Tensor)
            termination_loss = confusion_matrix(gt_termination.flatten().cpu().numpy(), world_model.termination_hat_buffer.flatten().cpu().numpy())
            reward_losses += reward_loss
            termination_losses += termination_loss

            # print(reward_loss)
            # print(termination_loss)
        
        reward_losses /= num_batches
        print(reward_losses)

        print(termination_losses)
        TN = termination_losses[0, 0]
        FP = termination_losses[0, 1]
        FN = termination_losses[1, 0]
        TP = termination_losses[1, 1]

        # Calculate metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1_score:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="trainer")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()

    main(args.config_name, args.checkpoint_path)