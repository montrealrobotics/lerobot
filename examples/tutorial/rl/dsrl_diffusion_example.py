#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DSRL with a diffusion policy on RoboCasa.

Usage:
    python examples/tutorial/rl/dsrl_diffusion_example.py \\
        --policy_id lerobot/diffusion_robocasa \\
        --task CoffeePressButton \\
        --total_steps 500000
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch

from lerobot.envs.robocasa_env import RoboCasaEnv
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.rl.dsrl import train_dsrl
from lerobot.rl.dsrl.dsrl_config import DSRLConfig


def make_robocasa_env(task: str, camera_names: str = "robot0_agentview_center", seed: int = 0):
    return RoboCasaEnv(
        task_name=task,
        robot="PandaOmron",
        camera_name=camera_names,
        obs_type="pixels_agent_pos",
        seed=seed,
        observation_width=256,
        observation_height=256,
    )


def make_prepare_obs_fn(device: torch.device):
    """Raw (batched) RoboCasa obs → noise-actor observation dict.

    Under a ``gym.vector.VectorEnv`` the observation already carries a leading batch
    dimension, so no ``unsqueeze`` is needed.
    """

    def prepare_obs(obs: dict) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        agent_pos = obs.get("agent_pos")
        if agent_pos is not None:
            out["observation.state"] = torch.as_tensor(agent_pos, dtype=torch.float32, device=device)
        for cam_name, img in obs.get("pixels", {}).items():
            img_t = torch.as_tensor(img, dtype=torch.float32, device=device)  # (B, H, W, C)
            out[f"observation.image.{cam_name}"] = img_t.permute(0, 3, 1, 2)
        return out

    return prepare_obs


def make_generate_action_chunk_fn(
    frozen_policy: DiffusionPolicy, horizon: int, action_dim: int, device: torch.device
):
    image_keys = list(frozen_policy.config.image_features)

    def generate_action_chunk(stacked_obs: dict, noise: np.ndarray) -> torch.Tensor:
        noise_tensor = torch.from_numpy(noise).float().to(device).view(noise.shape[0], horizon, action_dim)
        batch = dict(stacked_obs)
        if image_keys:
            batch["observation.images"] = torch.stack([batch[k] for k in image_keys], dim=-4)
        return frozen_policy.diffusion.generate_actions(batch, noise=noise_tensor)

    return generate_action_chunk


def main():
    parser = argparse.ArgumentParser(description="DSRL with diffusion policy on RoboCasa")
    parser.add_argument("--policy_id", type=str, required=True, help="HF repo ID of pretrained diffusion policy")
    parser.add_argument("--task", type=str, default="CoffeePressButton", help="RoboCasa task name")
    parser.add_argument("--total_steps", type=int, default=500_000, help="Total environment steps")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs/dsrl_diffusion", help="Output directory")
    parser.add_argument("--log_freq", type=int, default=100, help="Log training stats every N SAC updates")
    parser.add_argument("--save_freq", type=int, default=10_000, help="Save checkpoint every N env steps")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading diffusion policy from {args.policy_id} ...")
    frozen_policy = DiffusionPolicy.from_pretrained(args.policy_id)
    frozen_policy.eval()
    frozen_policy.to(device)

    horizon = frozen_policy.config.horizon
    action_dim = frozen_policy.config.action_feature.shape[0]
    n_obs_steps = frozen_policy.config.n_obs_steps
    noise_dim = horizon * action_dim
    print(f"Diffusion policy: horizon={horizon}, action_dim={action_dim}, "
          f"n_obs_steps={n_obs_steps}, noise_dim={noise_dim}")

    env = gym.vector.SyncVectorEnv([lambda: make_robocasa_env(task=args.task, seed=args.seed)])

    prepare_obs_fn = make_prepare_obs_fn(device)
    generate_action_chunk_fn = make_generate_action_chunk_fn(frozen_policy, horizon, action_dim, device)

    train_dsrl(
        env=env,
        noise_dim=noise_dim,
        prepare_obs_fn=prepare_obs_fn,
        generate_action_chunk_fn=generate_action_chunk_fn,
        total_steps=args.total_steps,
        device=str(device),
        n_obs_steps=n_obs_steps,
        dsrl_config=DSRLConfig(log_freq=args.log_freq, save_freq=args.save_freq),
        seed=args.seed,
        output_dir=args.output_dir,
    )

    env.close()


if __name__ == "__main__":
    main()
