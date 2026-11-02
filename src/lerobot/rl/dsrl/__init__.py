"""DSRL (Diffusion Steering with Reinforcement Learning).

Trains a small RL policy in the noise space of a frozen diffusion policy.
"""

from .dsrl_config import DSRLConfig
from .dsrl_env_wrapper import DSRLEnvWrapper
from .dsrl_trainer import train_dsrl
from .noise_actor import LightweightNoiseActorPolicy, NoiseActorConfig, NoiseActorPolicy

__all__ = [
    "DSRLConfig",
    "DSRLEnvWrapper",
    "LightweightNoiseActorPolicy",
    "NoiseActorConfig",
    "NoiseActorPolicy",
    "train_dsrl",
]
