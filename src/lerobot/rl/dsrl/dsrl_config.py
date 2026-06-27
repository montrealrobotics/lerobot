#!/usr/bin/env python
"""DSRL training configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DSRLConfig:
    """Configuration for DSRL (Diffusion Steering with RL) training.

    Controls the noise actor architecture (encoders, Q-networks) and
    training hyperparameters.
    """

    # Noise actor architecture
    image_latent_dim: int = 64
    state_latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (1024, 1024, 1024)

    # Encoder
    use_compact_encoder: bool = True
    image_resize_size: int | None = 64  # None = no resize, int = resize square

    # Q-network
    num_q_heads: int = 2

    # Training
    min_buffer_size: int = 1_000
    buffer_capacity: int = 100_000
    batch_size: int = 256
    utd_ratio: int = 20
    log_freq: int = 100
    save_freq: int = 10_000
    eval_freq: int = 0

    # SAC
    discount: float = 0.99
    target_entropy: float | None = 0.0
    critic_target_update_weight: float = 0.005
    policy_update_freq: int = 1

    # Optimizer
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
