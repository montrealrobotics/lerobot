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

import gymnasium as gym
import numpy as np
import torch

from lerobot.envs.configs import PushtEnv
from lerobot.envs.factory import make_env
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.rl.dsrl import train_dsrl
from lerobot.rl.dsrl.dsrl_config import DSRLConfig
from lerobot.rl.dsrl.dsrl_env_wrapper import (
    DSRL_ACTION_CHUNK_RAW_REWARD,
    DSRL_ACTION_CHUNK_STEPS,
    DSRL_ACTION_CHUNK_SUCCESS,
    DSRLEnvWrapper,
)
from lerobot.rl.dsrl.noise_actor import NoiseActorPolicy
from lerobot.utils.io_utils import write_video


def _make_single_env(env_cfg):
    import importlib

    from gymnasium.envs.registration import registry as gym_registry

    if env_cfg.gym_id not in gym_registry:
        importlib.import_module(env_cfg.package_name)
    return gym.make(env_cfg.gym_id, disable_env_checker=env_cfg.disable_env_checker, **env_cfg.gym_kwargs)


def _make_collect_env(env_cfg, num_envs: int) -> gym.vector.VectorEnv:
    if num_envs < 1:
        raise ValueError("--collect_envs must be >= 1")
    envs = make_env(env_cfg, n_envs=num_envs, use_async_envs=False)
    return envs[env_cfg.type][0]


# ── Shared observation / action adapters (used by both training and eval) ──────────


def make_prepare_obs_fn(device: torch.device, policy_preprocess_fn=None):
    """PushT raw env obs → noise-actor observation dict.

    PushT env returns ``{"pixels": (H,W,3), "agent_pos": (2,)}``; the policy expects
    ``{"observation.image": ..., "observation.state": ...}``.
    """

    def prepare_obs(obs: dict) -> dict[str, torch.Tensor]:
        policy_obs = preprocess_observation(obs)
        if policy_preprocess_fn is not None:
            return policy_preprocess_fn(policy_obs)
        return {key: value.to(device) for key, value in policy_obs.items()}

    return prepare_obs


def make_generate_action_chunk_fn(
    frozen_policy: DiffusionPolicy,
    action_postprocess_fn,
    horizon: int,
    action_dim: int,
    device: torch.device,
    noise_chunk_size: int | None = None,
):
    """Bundle noise reshaping + frozen-policy chunk generation + action unnormalization.

    ``(stacked_obs, noise) -> action_chunk`` of shape ``(num_envs, n_action_steps, action_dim)``.

    If ``noise_chunk_size < horizon``, the actor only emits noise for ``noise_chunk_size``
    steps, which is then block-duplicated up to the diffusion ``horizon`` (the DSRL-paper
    trick for shrinking the SAC noise space: each noise vector governs a contiguous block of
    ``horizon // noise_chunk_size`` diffusion steps). ``None`` means full horizon (no
    reduction).
    """
    image_keys = list(frozen_policy.config.image_features)
    noise_chunk_size = noise_chunk_size or horizon

    def generate_action_chunk(stacked_obs: dict[str, torch.Tensor], noise: np.ndarray) -> torch.Tensor:
        noise_tensor = torch.from_numpy(noise).float().to(device)
        noise_tensor = noise_tensor.view(noise_tensor.shape[0], noise_chunk_size, action_dim)

        if noise_chunk_size != horizon:
            # Block-duplicate the compressed noise up to the full diffusion horizon.
            repeats = -(-horizon // noise_chunk_size)  # ceil
            noise_tensor = noise_tensor.repeat_interleave(repeats, dim=1)[:, :horizon]

        batch = dict(stacked_obs)
        if image_keys:
            batch["observation.images"] = torch.stack([batch[k] for k in image_keys], dim=-4)
        action_chunk = frozen_policy.diffusion.generate_actions(batch, noise=noise_tensor)
        return action_postprocess_fn(action_chunk)

    return generate_action_chunk


# Named primitive-step reward functions: (reward, terminated, truncated, info) -> reward.
# Summed over the action chunk by the env wrapper. Add an entry to introduce a new reward
# variant — no new CLI flag or branching needed; ``--reward_mode`` selects by key.


def _dense_reward(reward: float, terminated: bool, truncated: bool, info: dict) -> float:
    """PushT's native coverage reward, in [0, 1]."""
    del terminated, truncated, info
    return reward


def _sparse_reward(reward: float, terminated: bool, truncated: bool, info: dict) -> float:
    """LIBERO-style sparse reward: 1.0 on the success step (also termination), else 0.0."""
    del reward, terminated, truncated
    return 1.0 if info.get("is_success", False) else 0.0


REWARD_FNS = {"dense": _dense_reward, "sparse": _sparse_reward}


# Named macro-step (per action chunk) reward functions, which override the summed primitive
# reward. ``(raw_reward, shaped_reward, success, terminated, truncated, steps) -> reward``.


def _goal_reward(
    raw_reward: float, shaped_reward: float, success: bool, terminated: bool, truncated: bool, steps: int
) -> float:
    """DSRL's goal-reaching reward: -1 per action chunk until success, 0 on success.

    Together with bootstrapping being masked at the (terminal) success step, this bounds Q
    in [-1/(1-gamma), 0] and rewards reaching the goal in as few chunks as possible. This is
    the formulation the reference DSRL implementation trains LIBERO with; it ignores the
    env's own reward entirely.
    """
    del raw_reward, shaped_reward, terminated, truncated, steps
    return 0.0 if success else -1.0


MACRO_REWARD_FNS = {"goal": _goal_reward}


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


# ── Evaluation (reuses the DSRL env wrapper, no duplicated rollout logic) ───────────


class _FrameCapture(gym.Wrapper):
    """Records ``obs["pixels"]`` frames for the episodes flagged via :meth:`start`."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.frames: list[np.ndarray] = []
        self._recording = False

    def start(self, recording: bool):
        self.frames = []
        self._recording = recording

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._recording:
            self.frames.append(obs["pixels"].copy())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._recording:
            self.frames.append(obs["pixels"].copy())
        return obs, reward, terminated, truncated, info


def _info_scalar(info: dict, key: str, default=0.0):
    value = info.get(key, default)
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0]
    return value


def make_pusht_eval_fn(
    prepare_obs_fn,
    generate_action_chunk_fn,
    reward_fn,
    device: torch.device,
    n_obs_steps: int,
    noise_dim: int,
    macro_reward_fn=None,
    num_episodes: int = 10,
    record_video: bool = True,
    num_videos: int = 3,
    video_fps: int = 10,
    video_dir: str | Path | None = None,
):
    def eval_fn(noise_actor: NoiseActorPolicy, step: int) -> dict[str, float]:
        noise_actor.eval()

        frame_capture = _FrameCapture(_make_single_env(PushtEnv()))
        vec_env = gym.vector.SyncVectorEnv([lambda: frame_capture])
        dsrl_env = DSRLEnvWrapper(
            env=vec_env,
            noise_dim=noise_dim,
            prepare_obs_fn=prepare_obs_fn,
            generate_action_chunk_fn=generate_action_chunk_fn,
            device=str(device),
            n_obs_steps=n_obs_steps,
            reward_fn=reward_fn,
            macro_reward_fn=macro_reward_fn,
        )

        total_return = 0.0
        total_shaped_return = 0.0
        total_steps = 0
        total_max_coverage = 0.0
        successes = 0
        episode_frames: list[tuple[list, float, bool, float]] = []

        for episode_idx in range(num_episodes):
            frame_capture.start(record_video and episode_idx < num_videos)
            policy_obs, _ = dsrl_env.reset()

            episode_return = 0.0
            episode_shaped_return = 0.0
            episode_steps = 0
            episode_success = False
            max_coverage = 0.0
            done = False

            while not done:
                with torch.no_grad():
                    noise = noise_actor.select_action(policy_obs)
                noise_np = noise.cpu().numpy().reshape(1, noise_dim)
                policy_obs, shaped_reward, terminated, truncated, info = dsrl_env.step(noise_np)

                episode_shaped_return += float(shaped_reward[0])
                episode_return += float(info[DSRL_ACTION_CHUNK_RAW_REWARD][0])
                episode_steps += int(info[DSRL_ACTION_CHUNK_STEPS][0])
                episode_success = episode_success or bool(info[DSRL_ACTION_CHUNK_SUCCESS][0])
                max_coverage = max(max_coverage, float(_info_scalar(info, "coverage", 0.0)))
                done = bool(terminated[0]) or bool(truncated[0])

            total_return += episode_return
            total_shaped_return += episode_shaped_return
            total_steps += episode_steps
            total_max_coverage += max_coverage
            successes += int(episode_success)

            if frame_capture.frames:
                episode_frames.append((frame_capture.frames, episode_return, episode_success, max_coverage))

        dsrl_env.close()
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
    parser.add_argument(
        "--collect_envs",
        type=int,
        default=1,
        help="Number of synchronous PushT envs to batch for DSRL data collection.",
    )
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
    parser.add_argument("--dsrl_num_q", type=int, default=10, help="Number of Q-networks")
    parser.add_argument(
        "--critic_reduction",
        type=str,
        default="mean",
        choices=["mean", "min"],
        help="How the critic ensemble is reduced. DSRL uses 'mean' over a large ensemble; "
        "'min' is standard pessimistic SAC and underestimates Q badly at high UTD.",
    )
    parser.add_argument(
        "--noise_chunk_size",
        type=int,
        default=0,
        help="Compress the SAC noise space to this many steps, block-duplicated up to the "
        "diffusion horizon (DSRL-paper trick). 0 = full horizon (no reduction).",
    )
    parser.add_argument(
        "--exec_action_steps",
        type=int,
        default=0,
        help="Override how many predicted actions are executed per DSRL macro-step",
    )
    parser.add_argument(
        "--utd_ratio", type=int, default=20, help="SAC update-to-data ratio per DSRL macro step"
    )
    parser.add_argument(
        "--target_entropy",
        type=float,
        default=None,
        help="SAC target entropy. Default (unset) uses SAC's automatic -noise_dim/2, as DSRL does.",
    )
    parser.add_argument(
        "--use_backup_entropy",
        action="store_true",
        help="Include the entropy term in the critic's TD backup. Off by default: the reference "
        "DSRL critic omits it.",
    )
    parser.add_argument(
        "--primitive_discount",
        type=float,
        default=0.999,
        help="Per-primitive-step discount. Compounded over the executed chunk to give the "
        "macro-step discount SAC actually uses (primitive_discount ** exec_action_steps).",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="goal",
        choices=list(REWARD_FNS) + list(MACRO_REWARD_FNS),
        help="Named reward function. 'goal' (macro-step, DSRL's own) = -1 per action chunk until "
        "success; 'dense' = PushT coverage reward summed over the chunk; 'sparse' = 1.0 on the "
        "success step only.",
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

    if args.exec_action_steps > 0:
        max_exec = horizon - n_obs_steps + 1
        if args.exec_action_steps > max_exec:
            raise ValueError(
                f"--exec_action_steps ({args.exec_action_steps}) cannot exceed "
                f"horizon - n_obs_steps + 1 = {max_exec}"
            )
        print(
            f"Overriding executed action steps per macro-step: "
            f"{frozen_policy.config.n_action_steps} -> {args.exec_action_steps}"
        )
        frozen_policy.config.n_action_steps = args.exec_action_steps

    noise_chunk_size = args.noise_chunk_size if args.noise_chunk_size > 0 else horizon
    if noise_chunk_size > horizon:
        raise ValueError(f"--noise_chunk_size ({noise_chunk_size}) cannot exceed horizon ({horizon})")
    noise_dim = noise_chunk_size * action_dim
    print(
        f"Diffusion policy: horizon={horizon}, action_dim={action_dim}, "
        f"n_obs_steps={n_obs_steps}, noise_chunk_size={noise_chunk_size}, noise_dim={noise_dim}"
    )

    env = _make_collect_env(PushtEnv(), args.collect_envs)

    policy_preprocess_fn, action_postprocess_fn = make_policy_processor_fns(
        frozen_policy, args.policy_path, device
    )
    prepare_obs_fn = make_prepare_obs_fn(device, policy_preprocess_fn=policy_preprocess_fn)
    generate_action_chunk_fn = make_generate_action_chunk_fn(
        frozen_policy, action_postprocess_fn, horizon, action_dim, device, noise_chunk_size=noise_chunk_size
    )
    # A macro-step reward (e.g. 'goal') overrides the per-primitive-step reward, so only one
    # of the two is ever active.
    reward_fn = REWARD_FNS.get(args.reward_mode)
    macro_reward_fn = MACRO_REWARD_FNS.get(args.reward_mode)

    # SAC discounts per macro step (one action chunk), so compound the per-primitive-step
    # discount over the chunk to get an equivalent effective horizon.
    exec_action_steps = frozen_policy.config.n_action_steps
    discount = args.primitive_discount**exec_action_steps
    print(
        f"Discount: {args.primitive_discount} ** {exec_action_steps} executed steps = {discount:.4f} "
        f"per macro step"
    )

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
            prepare_obs_fn=prepare_obs_fn,
            generate_action_chunk_fn=generate_action_chunk_fn,
            reward_fn=reward_fn,
            macro_reward_fn=macro_reward_fn,
            device=device,
            n_obs_steps=n_obs_steps,
            noise_dim=noise_dim,
            num_episodes=args.eval_episodes,
            num_videos=args.eval_videos,
            video_fps=PushtEnv().fps,
            video_dir=Path(args.output_dir) / "eval_videos",
        )

    dsrl_cfg = DSRLConfig(
        image_resize_size=args.dsrl_image_resize if args.dsrl_image_resize > 0 else None,
        use_compact_encoder=args.dsrl_image_resize > 0,
        image_latent_dim=args.dsrl_image_latent,
        state_latent_dim=args.dsrl_state_latent,
        hidden_dims=tuple([args.dsrl_hidden_dim] * args.dsrl_num_layers),
        num_q_heads=args.dsrl_num_q,
        critic_reduction=args.critic_reduction,
        utd_ratio=args.utd_ratio,
        discount=discount,
        target_entropy=args.target_entropy,
        use_backup_entropy=args.use_backup_entropy,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
    )

    train_dsrl(
        env=env,
        noise_dim=noise_dim,
        prepare_obs_fn=prepare_obs_fn,
        generate_action_chunk_fn=generate_action_chunk_fn,
        reward_fn=reward_fn,
        macro_reward_fn=macro_reward_fn,
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
