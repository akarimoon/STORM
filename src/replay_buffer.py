import numpy as np
import random
import unittest
import torch
from einops import rearrange
import copy
import pickle
import scipy


class ReplayBuffer():
    def __init__(self, obs_shape, num_envs, max_length=int(1E6), warmup_length=50000, store_on_gpu=False, device="cuda") -> None:
        self.store_on_gpu = store_on_gpu
        if store_on_gpu:
            self.obs_buffer = torch.empty((max_length//num_envs, num_envs, *obs_shape), dtype=torch.uint8, device=device, requires_grad=False)
            self.action_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=device, requires_grad=False)
            self.reward_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=device, requires_grad=False)
            self.termination_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=device, requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length//num_envs, num_envs, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.reward_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.termination_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)

        self.length = 0
        self.num_envs = num_envs
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None
        self.device = device

    def load_trajectory(self, path):
        buffer = pickle.load(open(path, "rb"))
        if self.store_on_gpu:
            self.external_buffer = {name: torch.from_numpy(buffer[name]).to(self.device) for name in buffer}
        else:
            self.external_buffer = buffer
        self.external_buffer_length = self.external_buffer["obs"].shape[0]

    def sample_external(self, batch_size, batch_length, to_device="cuda"):
        indexes = np.random.randint(0, self.external_buffer_length+1-batch_length, size=batch_size)
        if self.store_on_gpu:
            obs = torch.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
            action = torch.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
            reward = torch.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
            termination = torch.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
        else:
            obs = np.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
            action = np.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
            reward = np.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
            termination = np.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
        return obs, action, reward, termination

    def ready(self):
        return self.length * self.num_envs > self.warmup_length

    @torch.no_grad()
    def sample(self, batch_size, external_batch_size, batch_length, to_device="cuda"):
        if self.store_on_gpu:
            obs, action, reward, termination = [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs)
                    obs.append(torch.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    action.append(torch.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    reward.append(torch.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    termination.append(torch.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))

            if self.external_buffer_length is not None and external_batch_size > 0:
                external_obs, external_action, external_reward, external_termination = self.sample_external(
                    external_batch_size, batch_length, to_device)
                obs.append(external_obs)
                action.append(external_action)
                reward.append(external_reward)
                termination.append(external_termination)

            obs = torch.cat(obs, dim=0).float() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.cat(action, dim=0)
            reward = torch.cat(reward, dim=0)
            termination = torch.cat(termination, dim=0)
        else:
            obs, action, reward, termination = [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs)
                    obs.append(np.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    action.append(np.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    reward.append(np.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    termination.append(np.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))

            if self.external_buffer_length is not None and external_batch_size > 0:
                external_obs, external_action, external_reward, external_termination = self.sample_external(
                    external_batch_size, batch_length, to_device)
                obs.append(external_obs)
                action.append(external_action)
                reward.append(external_reward)
                termination.append(external_termination)

            obs = torch.from_numpy(np.concatenate(obs, axis=0)).float().to(self.device) / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.from_numpy(np.concatenate(action, axis=0)).to(self.device)
            reward = torch.from_numpy(np.concatenate(reward, axis=0)).to(self.device)
            termination = torch.from_numpy(np.concatenate(termination, axis=0)).to(self.device)

        return obs, action, reward, termination

    def append(self, obs, action, reward, termination):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % (self.max_length//self.num_envs)
        if self.store_on_gpu:
            self.obs_buffer[self.last_pointer] = torch.from_numpy(obs)
            self.action_buffer[self.last_pointer] = torch.from_numpy(action)
            self.reward_buffer[self.last_pointer] = torch.from_numpy(reward)
            self.termination_buffer[self.last_pointer] = torch.from_numpy(termination)
        else:
            self.obs_buffer[self.last_pointer] = obs
            self.action_buffer[self.last_pointer] = action
            self.reward_buffer[self.last_pointer] = reward
            self.termination_buffer[self.last_pointer] = termination

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length * self.num_envs


class OCRLReplayBuffer():
    def __init__(self, obs_shape, num_envs, max_length, max_episodes=int(1E6), warmup_length=50000, store_on_gpu=False, dreamsmooth=None, device="cuda") -> None:
        assert num_envs == 1, "OCRLReplayBuffer only supports num_envs=1"

        self.store_on_gpu = store_on_gpu
        if store_on_gpu:
            self.obs_buffer = torch.empty((max_episodes, max_length, *obs_shape), dtype=torch.uint8, device=device, requires_grad=False)
            self.action_buffer = torch.empty((max_episodes, max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.reward_buffer = torch.empty((max_episodes, max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.termination_buffer = torch.empty((max_episodes, max_length), dtype=torch.float32, device=device, requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_episodes, max_length, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty((max_episodes, max_length), dtype=np.float32)
            self.reward_buffer = np.empty((max_episodes, max_length), dtype=np.float32)
            self.termination_buffer = np.empty((max_episodes, max_length), dtype=np.float32)
        self.episode_length_buffer = np.zeros((max_episodes), dtype=np.int32)

        self.length = 0
        self.num_envs = num_envs
        self.episode_pointer = 0
        self.last_pointer = -1
        self.max_length = max_length
        self.max_episodes = max_episodes
        self.warmup_length = warmup_length
        self.external_buffer_length = None
        self.device = device
        self.dreamsmooth = dreamsmooth

    # def load_trajectory(self, path):
    #     buffer = pickle.load(open(path, "rb"))
    #     if self.store_on_gpu:
    #         self.external_buffer = {name: torch.from_numpy(buffer[name]).to(self.device) for name in buffer}
    #     else:
    #         self.external_buffer = buffer
    #     self.external_buffer_length = self.external_buffer["obs"].shape[0]

    # def sample_external(self, batch_size, batch_length, to_device="cuda"):
    #     indexes = np.random.randint(0, self.external_buffer_length+1-batch_length, size=batch_size)
    #     if self.store_on_gpu:
    #         obs = torch.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
    #         action = torch.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
    #         reward = torch.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
    #         termination = torch.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
    #     else:
    #         obs = np.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
    #         action = np.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
    #         reward = np.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
    #         termination = np.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
    #     return obs, action, reward, termination

    def ready(self):
        return self.length * self.num_envs > self.warmup_length
    
    def pad(self, x, padding_length_left, padding_length_right):
        pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
        return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

    def get_stats(self):
        collected_episodes = self.episode_pointer
        if self.store_on_gpu:
            collected_successful_episodes = torch.sum(self.reward_buffer[:collected_episodes].sum(1) > 0).item()
        else:
            collected_successful_episodes = np.sum(np.sum(self.reward_buffer[:collected_episodes], axis=1) > 0)
        return {
            "replay_buffer/num_episodes": collected_episodes,
            "replay_buffer/num_successful_episodes": collected_successful_episodes,
            "replay_buffer/success_rate": collected_successful_episodes / collected_episodes if collected_episodes > 0 else 0,
        }
    
    def smooth(self, reward):
        if self.dreamsmooth is None:
            return reward
        elif "gaussian" in self.dreamsmooth: # "gaussian_xx" where xx is the sigma
            sigma = float(self.dreamsmooth.split("_")[1])
            if self.store_on_gpu:
                device = reward.device
                return torch.tensor(scipy.ndimage.gaussian_filter1d(reward.cpu().numpy(), sigma=sigma, mode="nearest")).to(device)
            return scipy.ndimage.gaussian_filter1d(reward, sigma=sigma, mode="nearest")

    @torch.no_grad()
    def sample(self, batch_size, external_batch_size, batch_length, sample_from_start=True, to_device="cuda"):
        assert external_batch_size == 0, "OCRLReplayBuffer does not support external buffer"
        num_episodes = self.episode_pointer

        if self.store_on_gpu:
            obs, action, reward, termination = [], [], [], []
            if batch_size > 1:
                indexes = np.random.randint(0, num_episodes, size=batch_size)
                for id in indexes:
                    # start = np.random.randint(0, self.episode_length_buffer[id]+1-batch_length)
                    # obs.append(self.obs_buffer[id, start:start+batch_length])
                    # action.append(self.action_buffer[id, start:start+batch_length])
                    # reward.append(self.reward_buffer[id, start:start+batch_length])
                    # termination.append(self.termination_buffer[id, start:start+batch_length])

                    if sample_from_start:
                        start = np.random.randint(0, self.episode_length_buffer[id]-1)
                        stop = start + batch_length
                    else:
                        stop = np.random.randint(1, self.episode_length_buffer[id])
                        start = stop - batch_length
                    padding_length_right = max(0, stop - self.episode_length_buffer[id])
                    padding_length_left = max(0, -start)
                    start = max(0, start)
                    stop = min(self.episode_length_buffer[id], stop)

                    obs.append(self.pad(self.obs_buffer[id, start:stop], padding_length_left, padding_length_right))
                    action.append(self.pad(self.action_buffer[id, start:stop], padding_length_left, padding_length_right))
                    reward.append(self.pad(self.smooth(self.reward_buffer[id, start:stop]), padding_length_left, padding_length_right))
                    termination.append(self.pad(self.termination_buffer[id, start:stop], padding_length_left, padding_length_right))
            elif batch_size == 1:
                indexes = np.random.randint(0, num_episodes, size=batch_size)
                id = indexes[0]

                obs.append(self.obs_buffer[id, :self.episode_length_buffer[id]])
                action.append(self.action_buffer[id, :self.episode_length_buffer[id]])
                reward.append(self.smooth(self.reward_buffer[id, :self.episode_length_buffer[id]]))
                termination.append(self.termination_buffer[id, :self.episode_length_buffer[id]])

            obs = torch.stack(obs, dim=0).float() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.stack(action, dim=0)
            reward = torch.stack(reward, dim=0)
            termination = torch.stack(termination, dim=0)
        else:
            obs, action, reward, termination = [], [], [], []
            if batch_size > 1:
                indexes = np.random.randint(0, num_episodes, size=batch_size)
                for id in indexes:
                    start = np.random.randint(0, self.episode_length_buffer[id]+1-batch_length)
                    obs.append(self.obs_buffer[id, start:start+batch_length])
                    action.append(self.action_buffer[id, start:start+batch_length])
                    reward.append(self.reward_buffer[id, start:start+batch_length])
                    termination.append(self.termination_buffer[id, start:start+batch_length])
            elif batch_size == 1:
                indexes = np.random.randint(0, num_episodes, size=batch_size)
                id = indexes[0]

                obs.append(self.obs_buffer[id, :self.episode_length_buffer[id]])
                action.append(self.action_buffer[id, :self.episode_length_buffer[id]])
                reward.append(self.reward_buffer[id, :self.episode_length_buffer[id]])
                termination.append(self.termination_buffer[id, :self.episode_length_buffer[id]])

            obs = torch.from_numpy(np.array(obs, axis=0)).float().to(self.device) / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.from_numpy(np.array(action, axis=0)).to(self.device)
            reward = torch.from_numpy(np.array(reward, axis=0)).to(self.device)
            termination = torch.from_numpy(np.array(termination, axis=0)).to(self.device)

        return obs, action, reward, termination

    def append(self, obs, action, reward, termination):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        # self.episode_pointer = (self.episode_pointer + 1) % self.max_episodes
        self.last_pointer = (self.last_pointer + 1) % self.max_length
        if self.store_on_gpu:
            self.obs_buffer[self.episode_pointer, self.last_pointer] = torch.from_numpy(obs[0])
            self.action_buffer[self.episode_pointer, self.last_pointer] = torch.from_numpy(np.array(action[0]))
            self.reward_buffer[self.episode_pointer, self.last_pointer] = torch.from_numpy(np.array(reward[0]))
            self.termination_buffer[self.episode_pointer, self.last_pointer] = torch.from_numpy(np.array(termination[0]))
        else:
            self.obs_buffer[self.episode_pointer, self.last_pointer] = obs[0]
            self.action_buffer[self.episode_pointer, self.last_pointer] = action[0]
            self.reward_buffer[self.episode_pointer, self.last_pointer] = reward[0]
            self.termination_buffer[self.episode_pointer, self.last_pointer] = termination[0]

        if len(self) < self.max_length * self.max_episodes:
            self.length += 1

    def end_episode(self):
        self.episode_length_buffer[self.episode_pointer] = self.last_pointer + 1
        if self.last_pointer + 1 > 12:
            self.episode_pointer = (self.episode_pointer + 1) % self.max_episodes
        self.last_pointer = -1

    def __len__(self):
        return self.length * self.num_envs
