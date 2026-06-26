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
from pathlib import Path

import numpy as np
import torch

from lerobot.envs.configs import PushtEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.rl.dsrl import train_dsrl
from lerobot.rl.dsrl.dsrl_config import DSRLConfig
from lerobot.rl.dsrl.noise_actor import NoiseActorPolicy
from lerobot.utils.io_utils import write_video


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
    action_postprocess_fn,
    device: torch.device,
    num_episodes: int = 10,
    record_video: bool = True,
    num_videos: int = 3,
    video_fps: int = 10,
    video_dir: str | Path | None = None,
    success_bonus: float = 300.0,
):
    from collections import deque

    def eval_fn(noise_actor: NoiseActorPolicy, step: int) -> dict[str, float]:
        noise_actor.eval()

        eval_env = _make_single_env(PushtEnv())
        n_obs_steps = frozen_policy.config.n_obs_steps

        total_return = 0.0
        total_shaped_return = 0.0
        total_steps = 0
        total_max_coverage = 0.0
        successes = 0
        episode_frames = []

        for episode_idx in range(num_episodes):
            obs, _ = eval_env.reset()
            frozen_policy.reset()
            episode_return = 0.0
            episode_shaped_return = 0.0
            episode_steps = 0
            frames = []
            episode_success = False
            max_coverage = 0.0
            should_record_video = record_video and episode_idx < num_videos
            if should_record_video:
                frames.append(obs["pixels"].copy())

            obs_window: deque[dict[str, torch.Tensor]] = deque(maxlen=n_obs_steps)
            init_obs = obs_to_frozen_obs(obs)
            for _ in range(n_obs_steps):
                obs_window.append(init_obs)

            while True:
                policy_obs = obs_to_policy_obs(obs)
                with torch.no_grad():
                    noise = noise_actor.select_action(policy_obs)
                noise_np = noise.squeeze(0).cpu().numpy()

                stacked_obs = {k: torch.stack([h[k] for h in obs_window], dim=1) for k in obs_window[0]}

                noise_tensor = noise_reshape_fn(noise_np, device)
                gen_batch = dict(stacked_obs)
                img_keys = list(frozen_policy.config.image_features)
                if img_keys:
                    gen_batch["observation.images"] = torch.stack([gen_batch[k] for k in img_keys], dim=-4)
                with torch.no_grad():
                    action_chunk = frozen_policy.diffusion.generate_actions(gen_batch, noise=noise_tensor)
                action_chunk = action_postprocess_fn(action_chunk)

                done = False
                for action_np in action_chunk.squeeze(0).cpu().numpy():
                    obs, reward, terminated, truncated, info = eval_env.step(action_np)
                    raw_reward = float(reward)
                    step_success = bool(info.get("is_success", False))
                    shaped_reward = raw_reward + (success_bonus if step_success else 0.0)
                    episode_return += raw_reward
                    episode_shaped_return += shaped_reward
                    episode_steps += 1
                    episode_success = episode_success or step_success
                    max_coverage = max(max_coverage, float(info.get("coverage", 0.0)))
                    done = bool(terminated) or bool(truncated) or episode_success

                    if should_record_video:
                        frames.append(obs["pixels"].copy())

                    obs_window.append(obs_to_frozen_obs(obs))

                    if done:
                        break

                if done:
                    total_return += episode_return
                    total_shaped_return += episode_shaped_return
                    total_steps += episode_steps
                    total_max_coverage += max_coverage
                    successes += int(episode_success)
                    break

            if frames:
                episode_frames.append((frames, episode_return, episode_success, max_coverage))

        eval_env.close()
        noise_actor.train()

        if record_video and episode_frames:
            _log_eval_videos(
                episode_frames,
                step,
                fps=video_fps,
                video_dir=Path(video_dir)
                if video_dir is not None
                else Path("outputs/dsrl_pusht/eval_videos"),
            )

        n = max(num_episodes, 1)
        return {
            "avg_return": total_return / n,
            "avg_shaped_return": total_shaped_return / n,
            "success_rate": successes / n,
            "avg_length": total_steps / n,
            "avg_max_coverage": total_max_coverage / n,
        }

    return eval_fn


def _log_eval_videos(
    episode_frames: list[tuple[list, float, bool, float]], step: int, fps: int, video_dir: Path
):
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    video_dir.mkdir(parents=True, exist_ok=True)
    log_data = {}
    logged_videos = 0
    for i, (frames, _, _, _) in enumerate(episode_frames):
        if not frames:
            continue
        video_path = video_dir / f"eval_step_{step:08d}_episode_{i:02d}.mp4"
        try:
            write_video(video_path, frames, fps=fps)
        except ImportError as exc:
            print(f"[eval video] {exc}; skipping video logging.")
            return
        except Exception as exc:
            print(f"[eval video] failed to save {video_path}: {type(exc).__name__}: {exc}")
            continue
        log_data[f"eval/video_{i}"] = wandb.Video(str(video_path), fps=fps, format="mp4")
        logged_videos += 1

    if log_data:
        wandb.log(log_data, step=step)

    print(f"[eval video @ step {step}] logged first {logged_videos} eval videos")


def make_pusht_obs_to_policy_obs(device: torch.device, policy_preprocess_fn=None):
    """PushT raw env obs → policy-format dict.

    PushT env returns ``{"pixels": (H,W,3), "agent_pos": (2,)}``,
    policy expects ``{"observation.image": ..., "observation.state": ...}``.
    """

    def obs_to_policy_obs(obs: dict) -> dict[str, torch.Tensor]:
        policy_obs = preprocess_observation(obs)
        if policy_preprocess_fn is not None:
            return policy_preprocess_fn(policy_obs)
        return {key: value.to(device) for key, value in policy_obs.items()}

    return obs_to_policy_obs


def make_pusht_noise_reshape_fn(horizon: int, action_dim: int):
    """Flat noise → (1, horizon, action_dim) tensor for diffusion policy."""

    def reshape_noise(noise_np: np.ndarray, device: torch.device) -> torch.Tensor:
        noise_tensor = torch.from_numpy(noise_np).float().to(device)
        return noise_tensor.view(1, horizon, action_dim)

    return reshape_noise


def make_pusht_reward_fn(success_bonus: float):
    def reward_fn(reward: float, terminated: bool, truncated: bool, info: dict) -> float:
        del terminated, truncated
        return reward + (success_bonus if info.get("is_success", False) else 0.0)

    return reward_fn


def make_policy_processor_fns(frozen_policy: DiffusionPolicy, policy_path: str, device: torch.device):
    """Load saved policy processors when available."""
    try:
        from lerobot.policies.factory import make_pre_post_processors

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=frozen_policy.config,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
    except Exception as exc:
        print(
            "Warning: could not load policy processors; observations/actions will use only the "
            f"manual PushT conversion. ({type(exc).__name__}: {exc})"
        )

        def preprocess(policy_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {key: value.to(device) for key, value in policy_obs.items()}

        def identity(action_chunk: torch.Tensor) -> torch.Tensor:
            return action_chunk

        return preprocess, identity

    def preprocess(policy_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return preprocessor.process_observation(policy_obs)

    def postprocess(action_chunk: torch.Tensor) -> torch.Tensor:
        batch_size, chunk_size, action_dim = action_chunk.shape
        flat_actions = action_chunk.reshape(batch_size * chunk_size, action_dim)
        flat_actions = postprocessor(flat_actions)
        return flat_actions.reshape(batch_size, chunk_size, action_dim)

    print("Loaded policy processors for observation normalization and action unnormalization.")
    return preprocess, postprocess


def main():
    parser = argparse.ArgumentParser(description="DSRL with diffusion policy on PushT")
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Path to trained diffusion policy (local dir or HF repo id)",
    )
    parser.add_argument("--total_steps", type=int, default=100_000, help="Total environment steps")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs/dsrl_pusht", help="Output directory")
    parser.add_argument("--log_freq", type=int, default=100, help="Log training stats every N SAC updates")
    parser.add_argument("--save_freq", type=int, default=10_000, help="Save checkpoint every N env steps")
    # DSRL config
    parser.add_argument(
        "--dsrl_image_resize", type=int, default=64, help="Resize images for compact encoder (0=no compact)"
    )
    parser.add_argument("--dsrl_image_latent", type=int, default=64)
    parser.add_argument("--dsrl_state_latent", type=int, default=64)
    parser.add_argument(
        "--dsrl_hidden_dim", type=int, default=1024, help="Hidden dimension for actor/critic MLPs"
    )
    parser.add_argument("--dsrl_num_layers", type=int, default=3, help="Number of actor/critic MLP layers")
    parser.add_argument("--dsrl_num_q", type=int, default=2, help="Number of Q-networks")
    parser.add_argument(
        "--utd_ratio", type=int, default=20, help="SAC update-to-data ratio per DSRL macro step"
    )
    parser.add_argument("--target_entropy", type=float, default=0.0, help="SAC target entropy")
    parser.add_argument(
        "--success_bonus",
        type=float,
        default=300.0,
        help="Terminal reward bonus added when PushT reports success.",
    )
    # WandB
    parser.add_argument("--wandb_enable", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="lerobot-dsrl", help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name")
    # Eval
    parser.add_argument("--eval_freq", type=int, default=5_000, help="Run eval every N env steps (0=off)")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Episodes per eval")
    parser.add_argument("--eval_videos", type=int, default=3, help="Log videos for the first N eval episodes")
    args = parser.parse_args()
    if args.dsrl_num_layers < 1:
        raise ValueError("--dsrl_num_layers must be >= 1")

    device = torch.device(args.device)

    print(f"Loading diffusion policy from {args.policy_path} ...")
    frozen_policy = DiffusionPolicy.from_pretrained(args.policy_path)
    frozen_policy.eval()
    frozen_policy.to(device)

    horizon = frozen_policy.config.horizon
    action_dim = frozen_policy.config.action_feature.shape[0]
    n_obs_steps = frozen_policy.config.n_obs_steps
    noise_dim = horizon * action_dim
    print(
        f"Diffusion policy: horizon={horizon}, action_dim={action_dim}, "
        f"n_obs_steps={n_obs_steps}, noise_dim={noise_dim}"
    )

    env = _make_single_env(PushtEnv())

    policy_preprocess_fn, action_postprocess_fn = make_policy_processor_fns(
        frozen_policy, args.policy_path, device
    )
    obs_to_policy_obs = make_pusht_obs_to_policy_obs(device, policy_preprocess_fn=policy_preprocess_fn)
    obs_to_frozen_obs = obs_to_policy_obs
    noise_reshape_fn = make_pusht_noise_reshape_fn(horizon, action_dim)
    reward_fn = make_pusht_reward_fn(args.success_bonus)

    image_keys = list(frozen_policy.config.image_features)

    def action_fn(stacked_obs: dict, noise: torch.Tensor) -> torch.Tensor:
        batch = dict(stacked_obs)
        if image_keys:
            batch["observation.images"] = torch.stack([batch[k] for k in image_keys], dim=-4)
        action_chunk = frozen_policy.diffusion.generate_actions(batch, noise=noise)
        return action_postprocess_fn(action_chunk)

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
            action_postprocess_fn=action_postprocess_fn,
            device=device,
            num_episodes=args.eval_episodes,
            num_videos=args.eval_videos,
            video_fps=PushtEnv().fps,
            video_dir=Path(args.output_dir) / "eval_videos",
            success_bonus=args.success_bonus,
        )

    dsrl_cfg = DSRLConfig(
        image_resize_size=args.dsrl_image_resize if args.dsrl_image_resize > 0 else None,
        use_compact_encoder=args.dsrl_image_resize > 0,
        image_latent_dim=args.dsrl_image_latent,
        state_latent_dim=args.dsrl_state_latent,
        hidden_dims=tuple([args.dsrl_hidden_dim] * args.dsrl_num_layers),
        num_q_heads=args.dsrl_num_q,
        utd_ratio=args.utd_ratio,
        target_entropy=args.target_entropy,
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
        reward_fn=reward_fn,
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
