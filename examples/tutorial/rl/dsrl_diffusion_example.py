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

import numpy as np
import torch

from lerobot.envs.robocasa_env import RoboCasaEnv
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.rl.dsrl import train_dsrl


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


def make_diffusion_obs_to_policy_obs(device: torch.device):
    def obs_to_policy_obs(obs: dict) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        agent_pos = obs.get("agent_pos")
        if agent_pos is not None:
            out["observation.state"] = torch.from_numpy(agent_pos).float().unsqueeze(0).to(device)
        for cam_name, img in obs.get("pixels", {}).items():
            out[f"observation.image.{cam_name}"] = (
                torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)
            )
        return out
    return obs_to_policy_obs


def make_diffusion_noise_reshape_fn(horizon: int, action_dim: int):
    def reshape_noise(noise_np: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(noise_np).float().to(device).view(1, horizon, action_dim)
    return reshape_noise


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

    env = make_robocasa_env(task=args.task, seed=args.seed)

    obs_to_policy_obs = make_diffusion_obs_to_policy_obs(device)
    obs_to_frozen_obs = obs_to_policy_obs
    noise_reshape_fn = make_diffusion_noise_reshape_fn(horizon, action_dim)

    image_keys = list(frozen_policy.config.image_features)

    def action_fn(stacked_obs: dict, noise: torch.Tensor) -> torch.Tensor:
        batch = dict(stacked_obs)
        if image_keys:
            batch["observation.images"] = torch.stack([batch[k] for k in image_keys], dim=-4)
        return frozen_policy.diffusion.generate_actions(batch, noise=noise)

    train_dsrl(
        env=env,
        noise_dim=noise_dim,
        obs_to_policy_obs=obs_to_policy_obs,
        obs_to_frozen_obs=obs_to_frozen_obs,
        noise_reshape_fn=noise_reshape_fn,
        action_fn=action_fn,
        total_steps=args.total_steps,
        device=str(device),
        n_obs_steps=n_obs_steps,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
