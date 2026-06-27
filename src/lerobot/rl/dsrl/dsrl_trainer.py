#!/usr/bin/env python
"""DSRL training loop — shared between diffusion policy and pi0 variants."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import CriticNetworkConfig
from lerobot.rl.algorithms.sac import SACAlgorithm, SACAlgorithmConfig
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.dsrl.dsrl_config import DSRLConfig
from lerobot.rl.dsrl.dsrl_env_wrapper import (
    DSRL_ACTION_CHUNK_RAW_REWARD,
    DSRL_ACTION_CHUNK_STEPS,
    DSRLEnvWrapper,
)
from lerobot.rl.dsrl.noise_actor import (
    LightweightNoiseActorPolicy,
    NoiseActorPolicy,
    make_noise_actor_config,
)
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE


def train_dsrl(
    env,
    noise_dim: int,
    prepare_obs_fn: Callable[[dict], dict[str, torch.Tensor]],
    generate_action_chunk_fn: Callable[[dict[str, torch.Tensor], np.ndarray], torch.Tensor],
    *,
    prepare_frozen_obs_fn: Callable[[dict], dict[str, torch.Tensor]] | None = None,
    reward_fn: Callable[[float, bool, bool, dict], float] | None = None,
    total_steps: int = 500_000,
    device: str = "cuda",
    n_obs_steps: int = 1,
    output_dir: str | Path = "outputs/dsrl",
    seed: int = 0,
    wandb_kwargs: dict | None = None,
    dsrl_config: DSRLConfig | None = None,
    eval_fn: Callable[[NoiseActorPolicy, int], dict[str, float]] | None = None,
) -> NoiseActorPolicy:
    """Run DSRL training. Returns the trained NoiseActorPolicy.

    Args:
        env: A ``gym.vector.VectorEnv`` producing raw observations.
        noise_dim: Flat noise vector dimension.
        prepare_obs_fn: Raw env obs → noise-actor observation (batched tensors).
        generate_action_chunk_fn: ``(stacked_obs, noise) -> action_chunk`` for the frozen
            policy, returning ``(num_envs, n_action_steps, action_dim)``.
        prepare_frozen_obs_fn: Optional raw env obs → frozen-policy observation, when it
            differs from the noise-actor observation (e.g. VLAs needing language tokens).
        reward_fn: Optional primitive-step reward shaping function.
        dsrl_config: DSRL hyperparameters (encoders, Q-networks, training).
        wandb_kwargs: Optional ``{"enable": True, "project": "...", "name": "..."}``.
        eval_fn: Optional ``(noise_actor, step) -> {"metric": float}``.
    """
    dsrl_cfg = dsrl_config or DSRLConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # WandB
    wandb_run = None
    wandb_kwargs = wandb_kwargs or {}
    if wandb_kwargs.get("enable", False):
        try:
            import wandb

            wandb.init(
                project=wandb_kwargs.get("project", "lerobot-dsrl"),
                name=wandb_kwargs.get("name"),
                entity=wandb_kwargs.get("entity"),
                dir=str(wandb_kwargs.get("dir", output_dir)),
                config={
                    "noise_dim": noise_dim,
                    "total_steps": total_steps,
                    "buffer_capacity": dsrl_cfg.buffer_capacity,
                    "batch_size": dsrl_cfg.batch_size,
                    "utd_ratio": dsrl_cfg.utd_ratio,
                    "n_obs_steps": n_obs_steps,
                    "hidden_dims": list(dsrl_cfg.hidden_dims),
                    "num_q_heads": dsrl_cfg.num_q_heads,
                    "target_entropy": dsrl_cfg.target_entropy,
                    "seed": seed,
                    "collect_envs": getattr(env, "num_envs", 1),
                },
            )
            wandb_run = wandb
            print(f"WandB run: {wandb.run.get_url()}")
        except ImportError:
            print("wandb not installed — skipping WandB logging.")

    # Wrap environment. The wrapper presents a standard RL interface: reset()/step()
    # return noise-actor observations, so the loop below stays frozen-policy agnostic.
    dsrl_env = DSRLEnvWrapper(
        env=env,
        noise_dim=noise_dim,
        prepare_obs_fn=prepare_obs_fn,
        generate_action_chunk_fn=generate_action_chunk_fn,
        device=str(device),
        n_obs_steps=n_obs_steps,
        prepare_frozen_obs_fn=prepare_frozen_obs_fn,
        reward_fn=reward_fn,
    )
    num_collect_envs = dsrl_env.num_envs

    # Build noise actor
    dummy_policy_obs, _ = dsrl_env.reset()

    input_features: dict[str, PolicyFeature] = {}
    for key, tensor in dummy_policy_obs.items():
        shape = tuple(tensor.shape[1:]) if tensor.ndim > 1 else (1,)
        if key.startswith(OBS_IMAGE):
            feat_type = FeatureType.VISUAL
        elif key == OBS_STATE:
            feat_type = FeatureType.STATE
        else:
            feat_type = FeatureType.STATE
        input_features[key] = PolicyFeature(type=feat_type, shape=shape)

    noise_actor_cfg = make_noise_actor_config(
        noise_dim=noise_dim,
        input_features=input_features,
        device=str(device),
        actor_hidden_dims=list(dsrl_cfg.hidden_dims),
    )

    if dsrl_cfg.use_compact_encoder:
        noise_actor = LightweightNoiseActorPolicy(noise_actor_cfg, dsrl_config=dsrl_cfg)
    else:
        noise_actor = NoiseActorPolicy(noise_actor_cfg)
    noise_actor.train()
    noise_actor.to(device)

    # Build SAC algorithm
    algo_cfg = SACAlgorithmConfig.from_policy_config(noise_actor_cfg)
    algo_cfg.num_critics = dsrl_cfg.num_q_heads
    algo_cfg.actor_lr = dsrl_cfg.actor_lr
    algo_cfg.critic_lr = dsrl_cfg.critic_lr
    algo_cfg.temperature_lr = dsrl_cfg.temperature_lr
    algo_cfg.utd_ratio = dsrl_cfg.utd_ratio
    algo_cfg.discount = dsrl_cfg.discount
    algo_cfg.target_entropy = dsrl_cfg.target_entropy
    algo_cfg.critic_target_update_weight = dsrl_cfg.critic_target_update_weight
    algo_cfg.policy_update_freq = dsrl_cfg.policy_update_freq
    algo_cfg.critic_network_kwargs = CriticNetworkConfig(
        hidden_dims=list(dsrl_cfg.hidden_dims),
        activate_final=True,
    )
    algorithm = SACAlgorithm(policy=noise_actor, config=algo_cfg)
    algorithm.make_optimizers_and_scheduler()

    # Build replay buffer
    state_keys = list(input_features.keys())
    buffer = ReplayBuffer(
        capacity=dsrl_cfg.buffer_capacity,
        device=str(device),
        state_keys=state_keys,
        use_drq=True,
        storage_device="cpu",
    )

    # Training loop
    policy_obs, _ = dsrl_env.reset()
    episode_raw_rewards = np.zeros(num_collect_envs, dtype=np.float32)
    episode_shaped_rewards = np.zeros(num_collect_envs, dtype=np.float32)
    episode_steps = np.zeros(num_collect_envs, dtype=np.int64)
    episode_count = 0
    training_step = 0

    # Track rolling episode stats for WandB / console
    recent_raw_returns: deque[float] = deque(maxlen=10)
    recent_shaped_returns: deque[float] = deque(maxlen=10)

    print(
        f"Starting DSRL training for {total_steps} environment steps "
        f"(noise_dim={noise_dim}, collect_envs={num_collect_envs})"
    )

    env_step = 0
    while env_step < total_steps:
        with torch.no_grad():
            noise_tensor = noise_actor.select_action(policy_obs)
        noise_np = noise_tensor.cpu().numpy().reshape(num_collect_envs, noise_dim)

        next_policy_obs, shaped_reward, done, truncated, info = dsrl_env.step(noise_np)
        primitive_steps = _as_int_batch(info.get(DSRL_ACTION_CHUNK_STEPS, 1), num_collect_envs)
        shaped_rewards = _as_float_batch(shaped_reward, num_collect_envs)
        raw_rewards = _as_float_batch(info.get(DSRL_ACTION_CHUNK_RAW_REWARD, shaped_reward), num_collect_envs)
        dones = _as_bool_batch(done, num_collect_envs)
        truncateds = _as_bool_batch(truncated, num_collect_envs)
        collected_steps = int(primitive_steps.sum())
        env_step += collected_steps

        for env_idx in range(num_collect_envs):
            buffer.add(
                state=_slice_batch(policy_obs, env_idx),
                action=noise_tensor[env_idx : env_idx + 1],
                reward=float(shaped_rewards[env_idx]),
                next_state=_slice_batch(next_policy_obs, env_idx),
                done=bool(dones[env_idx]),
                truncated=bool(truncateds[env_idx]),
            )

        policy_obs = next_policy_obs
        episode_raw_rewards += raw_rewards
        episode_shaped_rewards += shaped_rewards
        episode_steps += primitive_steps

        finished = dones | truncateds
        for finished_env_idx in np.flatnonzero(finished):
            episode_count += 1
            episode_raw_reward = float(episode_raw_rewards[finished_env_idx])
            episode_shaped_reward = float(episode_shaped_rewards[finished_env_idx])
            finished_episode_steps = int(episode_steps[finished_env_idx])
            recent_raw_returns.append(episode_raw_reward)
            recent_shaped_returns.append(episode_shaped_reward)

            episode_metrics = {
                "episode/return": episode_shaped_reward,
                "episode/shaped_return": episode_shaped_reward,
                "episode/raw_return": episode_raw_reward,
                "episode/length": finished_episode_steps,
                "episode/count": episode_count,
                "episode/env_index": finished_env_idx,
                "episode/shaped_return_ma10": np.mean(recent_shaped_returns)
                if recent_shaped_returns
                else 0.0,
                "episode/raw_return_ma10": np.mean(recent_raw_returns) if recent_raw_returns else 0.0,
                "buffer/size": len(buffer),
            }

            if wandb_run is not None:
                wandb_run.log(episode_metrics, step=env_step)

            print(
                f"Episode {episode_count} | env={finished_env_idx} | steps={finished_episode_steps} | "
                f"shaped_return={episode_shaped_reward:.2f} | raw_return={episode_raw_reward:.2f} | "
                f"shaped_return_ma10={episode_metrics['episode/shaped_return_ma10']:.2f} | "
                f"buffer={len(buffer)}"
            )

        if finished.any():
            policy_obs, _ = dsrl_env.reset()
            episode_raw_rewards.fill(0.0)
            episode_shaped_rewards.fill(0.0)
            episode_steps.fill(0)

        # Train SAC
        if len(buffer) >= dsrl_cfg.min_buffer_size:
            batch_iterator = buffer.get_iterator(batch_size=dsrl_cfg.batch_size, async_prefetch=False)
            stats = algorithm.update(batch_iterator)
            training_step += 1

            if training_step % dsrl_cfg.log_freq == 0:
                log_dict = stats.to_log_dict()
                train_metrics = {f"train/{k}": v for k, v in log_dict.items()}
                train_metrics["train/step"] = training_step
                train_metrics["buffer/size"] = len(buffer)

                if wandb_run is not None:
                    wandb_run.log(train_metrics, step=env_step)

                print(
                    f"Step {env_step}/{total_steps} | "
                    f"critic_loss: {log_dict.get('loss_critic', 'N/A'):.4f} | "
                    f"actor_loss: {log_dict.get('loss_actor', 'N/A'):.4f} | "
                    f"buffer: {len(buffer)}"
                )

        # Periodic evaluation
        if (
            eval_fn is not None
            and dsrl_cfg.eval_freq > 0
            and env_step > 0
            and env_step % dsrl_cfg.eval_freq < max(collected_steps, 1)
        ):
            eval_metrics = eval_fn(noise_actor, env_step)
            if wandb_run is not None:
                wandb_run.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=env_step)
            print(f"[eval @ step {env_step}] " + " ".join(f"{k}={v:.3f}" for k, v in eval_metrics.items()))

        # Save checkpoint
        if env_step > 0 and env_step % dsrl_cfg.save_freq < max(collected_steps, 1):
            ckpt_dir = output_dir / f"step_{env_step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            noise_actor.save_pretrained(ckpt_dir)
            print(f"Checkpoint saved to {ckpt_dir}")

    # ── Final save ──────────────────────────────────────────────────────
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    noise_actor.save_pretrained(final_dir)
    print(f"Training complete. Final model saved to {final_dir}")

    if wandb_run is not None:
        wandb_run.finish()

    dsrl_env.close()
    return noise_actor


def _as_float_batch(value, batch_size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return np.full(batch_size, float(arr), dtype=np.float32)
    return arr.reshape(batch_size)


def _as_int_batch(value, batch_size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim == 0:
        return np.full(batch_size, int(arr), dtype=np.int64)
    return arr.reshape(batch_size)


def _as_bool_batch(value, batch_size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=bool)
    if arr.ndim == 0:
        return np.full(batch_size, bool(arr), dtype=bool)
    return arr.reshape(batch_size)


def _slice_batch(batch: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    return {key: value[index : index + 1] for key, value in batch.items()}
