#!/usr/bin/env python
"""DSRL training loop — shared between diffusion policy and pi0 variants."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.rl.algorithms.sac import SACAlgorithm, SACAlgorithmConfig
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.dsrl.dsrl_config import DSRLConfig
from lerobot.rl.dsrl.dsrl_env_wrapper import DSRLEnvWrapper
from lerobot.rl.dsrl.noise_actor import (
    LightweightNoiseActorPolicy,
    NoiseActorPolicy,
    make_noise_actor_config,
)
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE


def train_dsrl(
    env,
    noise_dim: int,
    obs_to_policy_obs: Callable[[dict], dict[str, torch.Tensor]],
    obs_to_frozen_obs: Callable[[dict], dict[str, torch.Tensor]],
    noise_reshape_fn: Callable[[np.ndarray, torch.device], torch.Tensor],
    action_fn: Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor],
    *,
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
        env: Base Gymnasium environment.
        noise_dim: Flat noise vector dimension.
        obs_to_policy_obs: Raw env obs → noise actor observation format.
        obs_to_frozen_obs: Raw env obs → frozen policy observation format.
        noise_reshape_fn: Flat numpy noise → policy-expected noise tensor.
        action_fn: (stacked_obs, noise_tensor) → action_chunk (B, N, action_dim).
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
                    "n_obs_steps": n_obs_steps,
                    "seed": seed,
                },
            )
            wandb_run = wandb
            print(f"WandB run: {wandb.run.get_url()}")
        except ImportError:
            print("wandb not installed — skipping WandB logging.")

    # Wrap environment
    dsrl_env = DSRLEnvWrapper(
        env=env,
        noise_dim=noise_dim,
        prepare_obs_fn=obs_to_frozen_obs,
        noise_reshape_fn=noise_reshape_fn,
        action_fn=action_fn,
        device=str(device),
        n_obs_steps=n_obs_steps,
    )

    # Build noise actor
    dummy_obs, _ = dsrl_env.reset()
    dummy_policy_obs = obs_to_policy_obs(dummy_obs)

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
    obs, _ = dsrl_env.reset()
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0
    training_step = 0

    # Track rolling episode stats for WandB / console
    recent_returns: deque[float] = deque(maxlen=10)

    print(f"Starting DSRL training for {total_steps} steps (noise_dim={noise_dim})")

    for env_step in range(total_steps):
        policy_obs = obs_to_policy_obs(obs)
        with torch.no_grad():
            noise_tensor = noise_actor.select_action(policy_obs)
        noise_np = noise_tensor.squeeze(0).cpu().numpy()

        next_obs, reward, done, truncated, _ = dsrl_env.step(noise_np)
        next_policy_obs = obs_to_policy_obs(next_obs)

        buffer.add(
            state=policy_obs,
            action=noise_tensor,
            reward=float(reward),
            next_state=next_policy_obs,
            done=bool(done),
            truncated=bool(truncated),
        )

        obs = next_obs
        episode_reward += reward
        episode_steps += 1

        if done or truncated:
            episode_count += 1
            recent_returns.append(episode_reward)

            episode_metrics = {
                "episode/return": episode_reward,
                "episode/length": episode_steps,
                "episode/count": episode_count,
                "episode/return_ma10": np.mean(recent_returns) if recent_returns else 0.0,
                "buffer/size": len(buffer),
            }

            if wandb_run is not None:
                wandb_run.log(episode_metrics, step=env_step)

            print(
                f"Episode {episode_count} | steps={episode_steps} | "
                f"return={episode_reward:.2f} | return_ma10={episode_metrics['episode/return_ma10']:.2f} | "
                f"buffer={len(buffer)}"
            )

            obs, _ = dsrl_env.reset()
            episode_reward = 0.0
            episode_steps = 0

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
        if eval_fn is not None and dsrl_cfg.eval_freq > 0 and env_step > 0 and env_step % dsrl_cfg.eval_freq == 0:
            eval_metrics = eval_fn(noise_actor, env_step)
            if wandb_run is not None:
                wandb_run.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=env_step)
            print(
                f"[eval @ step {env_step}] " + " ".join(f"{k}={v:.3f}" for k, v in eval_metrics.items())
            )

        # Save checkpoint
        if env_step > 0 and env_step % dsrl_cfg.save_freq == 0:
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
