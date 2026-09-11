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

"""Lightweight encoders for DSRL noise actor. ~13-50K params vs 11M for ResNet18."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


class _LightweightCNN(nn.Module):
    """Shared CNN backbone: 4 conv layers (32,32,32,32), stride-2 downsampling."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CompactObservationEncoder(nn.Module):
    """Replaces GaussianActorObservationEncoder with a lightweight CNN + MLP.

    Takes per-camera image keys and state key from a LeRobot observation dict,
    resizes images, stacks them, and encodes through a small CNN.
    """

    def __init__(
        self,
        image_keys: list[str],
        state_key: str | None = "observation.state",
        state_dim: int | None = None,
        image_latent_dim: int = 64,
        state_latent_dim: int = 64,
        resize_size: int | None = 64,
    ):
        super().__init__()
        self.image_keys = image_keys
        self.has_images = len(image_keys) > 0
        self.state_key = state_key
        self.has_state = state_key is not None
        self.resize_size = resize_size
        if self.has_state and state_dim is None:
            raise ValueError("state_dim must be provided when state_key is set")

        if self.has_images:
            num_images = len(image_keys)
            cnn_in = num_images * 3
            # After stride-2: H/2 × W/2 with 32 channels → flat = 32 * (H/2) * (W/2)
            r = resize_size if resize_size else 256
            flat_dim = 32 * (r // 2) * (r // 2)
            self.cnn = _LightweightCNN(cnn_in)
            self.image_proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, image_latent_dim),
                nn.LayerNorm(image_latent_dim),
                nn.Tanh(),
            )
        else:
            self.cnn = None
            self.image_proj = None

        if self.has_state:
            self.state_proj = nn.Sequential(
                nn.Linear(state_dim, state_latent_dim),
                nn.LayerNorm(state_latent_dim),
                nn.Tanh(),
            )
        else:
            self.state_proj = None

        self.output_dim = (image_latent_dim if self.has_images else 0) + (
            state_latent_dim if self.has_state else 0
        )
        self._init_weights()
        self._cached_features: dict[int, Tensor] = {}

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _resize(self, img: Tensor) -> Tensor:
        """Resize (B, C, H, W) → (B, C, R, R)."""
        if self.resize_size is None:
            return img
        # No-op when the caller already downsampled (observations are resized before being
        # stored in the replay buffer, so this is the common case).
        if img.shape[-2:] == (self.resize_size, self.resize_size):
            return img
        return F.interpolate(img, size=(self.resize_size, self.resize_size), mode="bilinear", align_corners=False)

    def forward(self, observations: dict[str, Tensor], cache: bool | Tensor = False, detach: bool = False) -> Tensor:
        if cache is not False and isinstance(cache, Tensor):
            features = cache.detach() if detach else cache
        else:
            features = self._encode(observations)
        return features

    def _encode(self, observations: dict[str, Tensor]) -> Tensor:
        features: list[Tensor] = []

        if self.has_images:
            imgs = []
            for key in self.image_keys:
                img = observations[key]
                if img.ndim == 4:
                    img = img.squeeze(1)
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                imgs.append(self._resize(img))
            stacked = torch.stack(imgs, dim=1)
            B, N, C, H, W = stacked.shape
            x = stacked.view(B, N * C, H, W)
            x = self.cnn(x)
            img_feat = self.image_proj(x)
            features.append(img_feat)

        if self.has_state:
            state = observations[self.state_key]
            if state.ndim == 3:
                state = state.squeeze(1)
            if state.ndim == 1:
                state = state.unsqueeze(0)
            features.append(self.state_proj(state))

        return torch.cat(features, dim=-1)

    def get_cached_image_features(self, observations: dict[str, Tensor]) -> Tensor | None:
        if not self.has_images:
            return None
        return self._encode(observations)
