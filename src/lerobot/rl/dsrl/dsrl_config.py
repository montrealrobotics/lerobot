#!/usr/bin/env python
"""DSRL training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class DSRLConfig:
    """Configuration for DSRL (Diffusion Steering with RL) training.

    Controls the noise actor architecture (encoders, Q-networks) and
    training hyperparameters.

    Defaults follow the reference DSRL implementation's LIBERO configuration: a large
    critic ensemble reduced by ``mean`` (rather than the pessimistic ``min`` of standard
    SAC), no entropy term in the TD backup, and an automatic ``-noise_dim / 2`` target
    entropy. Together these are what keep SAC stable at ``utd_ratio=20``.
    """

    # Noise actor architecture
    image_latent_dim: int = 64
    state_latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (128, 128, 128)

    # Encoder
    use_compact_encoder: bool = True
    image_resize_size: int | None = 64  # None = no resize, int = resize square

    # Q-network
    num_q_heads: int = 10
    # How the critic ensemble is reduced in the TD target and the actor loss. DSRL relies
    # on ``mean`` over a large ensemble; ``min`` over a small one badly underestimates Q
    # at high UTD.
    critic_reduction: Literal["min", "mean"] = "mean"

    # Training
    min_buffer_size: int = 1_000
    buffer_capacity: int = 100_000
    batch_size: int = 256
    utd_ratio: int = 20
    log_freq: int = 100
    save_freq: int = 10_000
    eval_freq: int = 0
    # Seed the replay buffer by sampling noise from the diffusion prior N(0, 1) until
    # ``min_buffer_size`` transitions are collected, instead of querying the untrained
    # (tanh-squashed, hence [-1, 1]-bounded) actor. Without this the warmup data is drawn
    # from a distribution the frozen policy was never trained to denoise.
    gaussian_warmup: bool = True

    # SAC
    # Discount applied per *macro* step (one action chunk). The reference implementation
    # discounts per primitive env step and compounds over the chunk, so the equivalent
    # value here is ``primitive_discount ** n_action_steps`` (e.g. ``0.999 ** 8``).
    discount: float = 0.99
    # ``None`` selects SAC's automatic ``-noise_dim / 2``, which is what DSRL uses.
    target_entropy: float | None = None
    # The reference DSRL critic omits the entropy term from the TD backup.
    use_backup_entropy: bool = False
    critic_target_update_weight: float = 0.005
    policy_update_freq: int = 1

    # Optimizer
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
