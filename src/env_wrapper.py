import argparse
from collections import deque

import numpy as np
import torch
from torchvision.utils import save_image
import cv2
from einops import rearrange
import gymnasium

import envs
from envs import TargetEnv

class LifeLossInfo(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives_info = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_lives_info = info["lives"]
        if current_lives_info < self.lives_info:
            info["life_loss"] = True
            self.lives_info = info["lives"]
        else:
            info["life_loss"] = False

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.lives_info = info["lives"]
        info["life_loss"] = False
        return observation, info


class SeedEnvWrapper(gymnasium.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.seed = seed
        self.env.action_space.seed(seed)

    def reset(self, **kwargs):
        kwargs["seed"] = self.seed
        obs, _ = self.env.reset(**kwargs)
        return obs, _

    def step(self, action):
        return self.env.step(action)


class MaxLast2FrameSkipWrapper(gymnasium.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs, _

    def step(self, action):
        total_reward = 0
        self.obs_buffer = deque(maxlen=2)
        for _ in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self.obs_buffer.append(obs)
            total_reward += reward
            if done or truncated:
                break
        if len(self.obs_buffer) == 1:
            obs = self.obs_buffer[0]
        else:
            obs = np.max(np.stack(self.obs_buffer), axis=0)
        return obs, total_reward, done, truncated, info
    

class OCRLMaxStepWrapper(gymnasium.Wrapper):
    def __init__(self, env, max_step):
        super().__init__(env)
        self.max_step = max_step
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        obs, _ = self.env.reset(**kwargs)
        return obs, _

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_step += 1
        # done = False
        if self.current_step >= self.max_step:
            done = True
        return obs, reward, done, truncated, info


def build_single_atari_env(env_name, image_size, seed):
    env = gymnasium.make(f"ALE/{env_name}-v5", full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = SeedEnvWrapper(env, seed=seed)
    env = MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = LifeLossInfo(env)
    return env

def build_single_ocrl_env(env_config, image_size, seed, max_step=1000):
    env = getattr(envs, env_config.env)(env_config, seed)
    env = OCRLMaxStepWrapper(env, max_step=max_step)
    return env

def build_single_env(cfg, image_size, seed):
    if cfg.env_type == "atari":
        return build_single_atari_env(cfg.env_name, image_size, seed)
    elif cfg.env_type == "ocrl":
        return build_single_ocrl_env(cfg.config, image_size, seed, max_step=cfg.max_step)

def build_vec_env(cfg, image_size, num_envs, seed):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator():
        if cfg.env_type == "atari":
            return lambda: build_single_atari_env(cfg.env_name, image_size, seed)
        elif cfg.env_type == "ocrl":
            return lambda: build_single_ocrl_env(cfg.config, image_size, seed, max_step=cfg.max_step)
    env_fns = [lambda_generator() for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


if __name__ == "__main__":
    def build_debug_atari_single_env(env_name, image_size):
        env = gymnasium.make(env_name, full_action_space=True, frameskip=1)
        from gymnasium.wrappers import AtariPreprocessing
        env = AtariPreprocessing(env, screen_size=image_size, grayscale_obs=False)
        return env

    def build_debug_atari_env(env_list, image_size, num_envs):
        # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
        assert num_envs % len(env_list) == 0
        env_fns = []
        vec_env_names = []
        for env_name in env_list:
            def lambda_generator(env_name, image_size):
                return lambda: build_debug_atari_single_env(env_name, image_size)
            env_fns += [lambda_generator(env_name, image_size) for i in range(num_envs//len(env_list))]
            vec_env_names += [env_name for i in range(num_envs//len(env_list))]
        vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
        return vec_env, vec_env_names

    def build_debug_ocrl_env(env_config, image_size, seed, max_step=1000):
        env = TargetEnv(env_config, seed)
        env = OCRLMaxStepWrapper(env, max_step=max_step)
        return env

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", type=str, default="atari", choices=["atari", "block"])
    args = parser.parse_args()

    if args.env_type == "atari":
        vec_env, vec_env_names = build_debug_atari_env(['ALE/Pong-v5', 'ALE/IceHockey-v5', 'ALE/Breakout-v5', 'ALE/Tennis-v5'], 64, num_envs=8)
        current_obs, _ = vec_env.reset()
        while True:
            action = vec_env.action_space.sample()
            obs, reward, done, truncated, info = vec_env.step(action)
            # done = done or truncated
            if done.any():
                print("---------")
                print(reward)
                print(info["episode_frame_number"])
            cv2.imshow("Pong", current_obs[0])
            cv2.imshow("IceHockey", current_obs[2])
            cv2.imshow("Breakout", current_obs[4])
            cv2.imshow("Tennis", current_obs[6])
            cv2.waitKey(40)
            current_obs = obs
    elif args.env_type == "block":
        import matplotlib.pyplot as plt
        from omegaconf import OmegaConf

        config = OmegaConf.load("config/ocrl/env_config/target-debug.yaml")
        env = build_debug_ocrl_env(config, 64, seed=0)

        def pad(x, padding_length_left, padding_length_right):
            pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
            return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

        all_episodes, episode_length = [], []
        max_episode_length = 50
        num_save = 10
        for _ in range(100):
            episode = []

            current_obs, current_info = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                if done:
                    episode_length.append(info["episode_frame_number"])
                episode.append(current_obs[0])
                current_obs = obs
            all_episodes.append(pad(torch.tensor(episode[:max_episode_length]), 0, max_episode_length - len(episode)))

        all_episodes = torch.stack(all_episodes)[:num_save] / 255.0
        all_episodes = rearrange(all_episodes, 'b t h w c -> (b t) c h w')
        save_image(all_episodes, "block_env.png", nrow=max_episode_length, pad_value=1.0)

        plt.hist(episode_length, bins=20)
        plt.savefig("episode_length.png")