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

"""DSRL with a trained diffusion policy on PushT.

Usage:
    python examples/tutorial/rl/dsrl_pusht_example.py \\
        --policy_path outputs/train/.../checkpoints/020000/pretrained_model \\
        --total_steps 100000 \\
        --eval_freq 5000 \\
        --wandb_enable
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from lerobot.envs.configs import PushtEnv
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.rl.dsrl import train_dsrl
from lerobot.rl.dsrl.dsrl_config import DSRLConfig
from lerobot.rl.dsrl.noise_actor import NoiseActorPolicy


def _make_single_env(env_cfg):
    import importlib

    import gymnasium as gym
    from gymnasium.envs.registration import registry as gym_registry

    if env_cfg.gym_id not in gym_registry:
        importlib.import_module(env_cfg.package_name)
    return gym.make(env_cfg.gym_id, disable_env_checker=env_cfg.disable_env_checker, **env_cfg.gym_kwargs)


def make_pusht_eval_fn(
    frozen_policy: DiffusionPolicy,
    obs_to_policy_obs,
    obs_to_frozen_obs,
    noise_reshape_fn,
    device: torch.device,
    num_episodes: int = 10,
    record_video: bool = True,
    video_max_steps: int = 200,
):
    from collections import deque

    def eval_fn(noise_actor: NoiseActorPolicy, step: int) -> dict[str, float]:
        noise_actor.eval()

        eval_env = _make_single_env(PushtEnv())
        n_obs_steps = frozen_policy.config.n_obs_steps

        total_return = 0.0
        total_steps = 0
        successes = 0
        episode_frames = []

        for _ in range(num_episodes):
            obs, _ = eval_env.reset()
            frozen_policy.reset()
            episode_return = 0.0
            episode_steps = 0
            frames = []

            obs_history: deque[dict[str, torch.Tensor]] = deque(maxlen=n_obs_steps)
            init_obs = obs_to_frozen_obs(obs)
            for _ in range(n_obs_steps):
                obs_history.append(init_obs)

            while True:
                if record_video and episode_steps < video_max_steps:
                    frames.append(obs["pixels"].copy())

                policy_obs = obs_to_policy_obs(obs)
                with torch.no_grad():
                    noise = noise_actor.select_action(policy_obs)
                noise_np = noise.squeeze(0).cpu().numpy()

                frozen_obs_single = obs_to_frozen_obs(obs)
                obs_history.append(frozen_obs_single)
                stacked_obs = {k: torch.stack([h[k] for h in obs_history], dim=1) for k in obs_history[0]}

                noise_tensor = noise_reshape_fn(noise_np, device)
                gen_batch = dict(stacked_obs)
                img_keys = list(frozen_policy.config.image_features)
                if img_keys:
                    gen_batch["observation.images"] = torch.stack(
                        [gen_batch[k] for k in img_keys], dim=-4
                    )
                with torch.no_grad():
                    action_chunk = frozen_policy.diffusion.generate_actions(gen_batch, noise=noise_tensor)
                action_np = action_chunk[:, 0, :].squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, _ = eval_env.step(action_np)
                done = bool(terminated) or bool(truncated)
                episode_return += float(reward)
                episode_steps += 1

                if done:
                    total_return += episode_return
                    total_steps += episode_steps
                    successes += 1 if episode_return > 0 else 0
                    break

            if frames:
                episode_frames.append((frames, episode_return))

        eval_env.close()
        noise_actor.train()

        if record_video and episode_frames:
            _log_eval_video(episode_frames, step)

        n = max(num_episodes, 1)
        return {
            "avg_return": total_return / n,
            "success_rate": successes / n,
            "avg_length": total_steps / n,
        }

    return eval_fn


def _log_eval_video(episode_frames: list[tuple[list, float]], step: int):
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    best_frames, best_return = max(episode_frames, key=lambda x: x[1])
    video = np.stack(best_frames).transpose(0, 3, 1, 2)
    wandb.log({"eval/video": wandb.Video(video, fps=10, format="mp4")}, step=step)
    print(f"[eval video @ step {step}] logged best episode (return={best_return:.2f}, frames={len(best_frames)})")


def make_pusht_obs_to_policy_obs(device: torch.device):
    """PushT raw env obs → policy-format dict.

    PushT env returns ``{"pixels": (H,W,3), "agent_pos": (2,)}``,
    policy expects ``{"observation.image": ..., "observation.state": ...}``.
    """

    def obs_to_policy_obs(obs: dict) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        if "agent_pos" in obs:
            out["observation.state"] = (
                torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device)
            )

        if "pixels" in obs:
            img = obs["pixels"]
            out["observation.image"] = (
                torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)
            )

        return out

    return obs_to_policy_obs


def make_pusht_noise_reshape_fn(horizon: int, action_dim: int):
    """Flat noise → (1, horizon, action_dim) tensor for diffusion policy."""

    def reshape_noise(noise_np: np.ndarray, device: torch.device) -> torch.Tensor:
        noise_tensor = torch.from_numpy(noise_np).float().to(device)
        return noise_tensor.view(1, horizon, action_dim)

    return reshape_noise


def main():
    parser = argparse.ArgumentParser(description="DSRL with diffusion policy on PushT")
    parser.add_argument(
        "--policy_path", type=str, required=True,
        help="Path to trained diffusion policy (local dir or HF repo id)",
    )
    parser.add_argument("--total_steps", type=int, default=100_000, help="Total environment steps")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs/dsrl_pusht", help="Output directory")
    parser.add_argument("--log_freq", type=int, default=100, help="Log training stats every N SAC updates")
    parser.add_argument("--save_freq", type=int, default=10_000, help="Save checkpoint every N env steps")
    # DSRL config
    parser.add_argument("--dsrl_image_resize", type=int, default=64, help="Resize images for compact encoder (0=no compact)")
    parser.add_argument("--dsrl_image_latent", type=int, default=64)
    parser.add_argument("--dsrl_state_latent", type=int, default=64)
    parser.add_argument("--dsrl_num_q", type=int, default=10, help="Number of Q-networks")
    # WandB
    parser.add_argument("--wandb_enable", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="lerobot-dsrl", help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name")
    # Eval
    parser.add_argument("--eval_freq", type=int, default=5_000, help="Run eval every N env steps (0=off)")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Episodes per eval")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading diffusion policy from {args.policy_path} ...")
    frozen_policy = DiffusionPolicy.from_pretrained(args.policy_path)
    frozen_policy.eval()
    frozen_policy.to(device)

    horizon = frozen_policy.config.horizon
    action_dim = frozen_policy.config.action_feature.shape[0]
    n_obs_steps = frozen_policy.config.n_obs_steps
    noise_dim = horizon * action_dim
    print(f"Diffusion policy: horizon={horizon}, action_dim={action_dim}, "
          f"n_obs_steps={n_obs_steps}, noise_dim={noise_dim}")

    env = _make_single_env(PushtEnv())

    obs_to_policy_obs = make_pusht_obs_to_policy_obs(device)
    obs_to_frozen_obs = obs_to_policy_obs
    noise_reshape_fn = make_pusht_noise_reshape_fn(horizon, action_dim)

    image_keys = list(frozen_policy.config.image_features)

    def action_fn(stacked_obs: dict, noise: torch.Tensor) -> torch.Tensor:
        batch = dict(stacked_obs)
        if image_keys:
            batch["observation.images"] = torch.stack([batch[k] for k in image_keys], dim=-4)
        return frozen_policy.diffusion.generate_actions(batch, noise=noise)

    wandb_kwargs = None
    if args.wandb_enable:
        wandb_kwargs = {
            "enable": True,
            "project": args.wandb_project,
            "name": args.wandb_name,
            "dir": args.output_dir,
        }

    eval_fn = None
    if args.eval_freq > 0:
        eval_fn = make_pusht_eval_fn(
            frozen_policy=frozen_policy,
            obs_to_policy_obs=obs_to_policy_obs,
            obs_to_frozen_obs=obs_to_frozen_obs,
            noise_reshape_fn=noise_reshape_fn,
            device=device,
            num_episodes=args.eval_episodes,
        )

    dsrl_cfg = DSRLConfig(
        image_resize_size=args.dsrl_image_resize if args.dsrl_image_resize > 0 else None,
        use_compact_encoder=args.dsrl_image_resize > 0,
        image_latent_dim=args.dsrl_image_latent,
        state_latent_dim=args.dsrl_state_latent,
        num_q_heads=args.dsrl_num_q,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
    )

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
        dsrl_config=dsrl_cfg,
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_kwargs=wandb_kwargs,
        eval_fn=eval_fn,
    )

    env.close()


if __name__ == "__main__":
    main()
