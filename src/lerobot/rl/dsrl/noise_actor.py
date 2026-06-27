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

"""Noise-space policy for DSRL.

A small Gaussian policy that outputs noise vectors instead of robot actions.
API-compatible with GaussianActorPolicy so it works with the existing SACAlgorithm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import PolicyFeature
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import (
    ActorNetworkConfig,
    GaussianActorConfig,
    PolicyConfig,
)
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import (
    MLP,
    GaussianActorObservationEncoder,
    Policy,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION

from .compact_encoders import CompactObservationEncoder
from .dsrl_config import DSRLConfig


@PreTrainedConfig.register_subclass("noise_actor")
@dataclass
class NoiseActorConfig(GaussianActorConfig):
    """Configuration for the noise-space actor.

    Extends GaussianActorConfig — the "action" dimension is actually the noise
    dimension. Use :func:`make_noise_actor_config` to construct one.
    """

    noise_dim: int = 160

    def __post_init__(self):
        super().__post_init__()
        if self.output_features is not None and ACTION in self.output_features:
            self.output_features[ACTION] = PolicyFeature(type="STATE", shape=(self.noise_dim,))


def make_noise_actor_config(
    noise_dim: int,
    input_features: dict[str, PolicyFeature],
    device: str = "cuda",
    actor_hidden_dims: list[int] | None = None,
    **kwargs,
) -> NoiseActorConfig:
    if actor_hidden_dims is None:
        actor_hidden_dims = [256, 256]

    output_features = {ACTION: PolicyFeature(type="STATE", shape=(noise_dim,))}

    return NoiseActorConfig(
        noise_dim=noise_dim,
        input_features=input_features,
        output_features=output_features,
        device=device,
        actor_network_kwargs=ActorNetworkConfig(hidden_dims=actor_hidden_dims),
        policy_kwargs=PolicyConfig(use_tanh_squash=True),
        num_discrete_actions=None,
        shared_encoder=True,
        freeze_vision_encoder=True,
        **kwargs,
    )


class NoiseActorPolicy(PreTrainedPolicy):
    """Small Gaussian policy that outputs noise vectors for DSRL.

    Drop-in replacement for GaussianActorPolicy in the SAC algorithm.
    """

    config_class = NoiseActorConfig
    name = "noise_actor"

    def __init__(self, config: NoiseActorConfig | None = None):
        super().__init__(config)
        config.validate_features()

        self.shared_encoder = config.shared_encoder
        self.encoder_critic = GaussianActorObservationEncoder(config)
        self.encoder_actor = (
            self.encoder_critic if self.shared_encoder else GaussianActorObservationEncoder(config)
        )

        self.actor = Policy(
            encoder=self.encoder_actor,
            network=MLP(input_dim=self.encoder_actor.output_dim, **asdict(config.actor_network_kwargs)),
            action_dim=config.noise_dim,
            encoder_is_shared=self.shared_encoder,
            **asdict(config.policy_kwargs),
        )

        self.discrete_critic = None

    def get_optim_params(self) -> dict:
        return {
            "actor": [
                p
                for n, p in self.actor.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
        }

    def reset(self):
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError(
            "NoiseActorPolicy does not support action chunking. Use select_action() for single noise vectors."
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        observations_features = None
        if self.shared_encoder and self.actor.encoder.has_images:
            observations_features = self.actor.encoder.get_cached_image_features(batch)
        actions, _, _ = self.actor(batch, observations_features)
        return actions

    def forward(self, batch: dict[str, Tensor | dict[str, Tensor]]) -> dict[str, Tensor]:
        observations = batch.get("state", batch)
        observation_features = batch.get("observation_feature") if isinstance(batch, dict) else None
        actions, log_probs, means = self.actor(observations, observation_features)
        return {"action": actions, "log_prob": log_probs, "action_mean": means}


class LightweightNoiseActorPolicy(NoiseActorPolicy):
    """Noise actor using compact encoders (~50K params vs ~11M for ResNet18).

    Uses CompactObservationEncoder for images+state and a small MLP actor head.
    """

    name = "lightweight_noise_actor"

    def __init__(self, config: NoiseActorConfig, dsrl_config: DSRLConfig | None = None):
        self.dsrl_config = dsrl_config or DSRLConfig()
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()

        image_keys = [k for k in config.input_features if k.startswith("observation.image")]
        state_key = "observation.state" if "observation.state" in config.input_features else None
        state_dim = config.input_features[state_key].shape[0] if state_key is not None else None

        encoder = CompactObservationEncoder(
            image_keys=image_keys,
            state_key=state_key,
            state_dim=state_dim,
            image_latent_dim=self.dsrl_config.image_latent_dim,
            state_latent_dim=self.dsrl_config.state_latent_dim,
            resize_size=self.dsrl_config.image_resize_size,
        )

        self.shared_encoder = True
        self.encoder_critic = encoder
        self.encoder_actor = encoder

        actor_hidden_dims = list(self.dsrl_config.hidden_dims)
        self.actor = Policy(
            encoder=self.encoder_actor,
            network=MLP(input_dim=encoder.output_dim, hidden_dims=actor_hidden_dims),
            action_dim=config.noise_dim,
            encoder_is_shared=True,
            **asdict(config.policy_kwargs),
        )

        self.discrete_critic = None
