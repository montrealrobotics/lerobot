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

"""Gym wrapper that translates noise actions → robot actions via a frozen diffusion policy."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch


class DSRLEnvWrapper(gym.Wrapper):
    """Wraps an environment so the RL agent sees a noise action space.

    Each step: noise → reshape → action_fn(stacked_obs, noise) → env.step(action).
    """

    def __init__(
        self,
        env: gym.Env,
        noise_dim: int,
        prepare_obs_fn: Callable[[dict], dict[str, torch.Tensor]],
        noise_reshape_fn: Callable[[np.ndarray, torch.device], torch.Tensor],
        action_fn: Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor],
        device: str = "cuda",
        n_obs_steps: int = 1,
        reset_fn: Callable[[], None] | None = None,
    ):
        super().__init__(env)
        self.noise_dim = noise_dim
        self.prepare_obs_fn = prepare_obs_fn
        self.noise_reshape_fn = noise_reshape_fn
        self.action_fn = action_fn
        self.device = torch.device(device)
        self.n_obs_steps = n_obs_steps
        self._reset_fn = reset_fn

        # Override action space to be noise space
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(noise_dim,),
            dtype=np.float32,
        )

        # Cache last raw observation for step()
        self._last_obs: dict | None = None

        # Observation history buffer for n_obs_steps > 1
        self._obs_history: deque[dict[str, torch.Tensor]] = deque(maxlen=n_obs_steps)

        # Action chunk cache — only call action_fn (expensive diffusion) when empty
        self._action_queue: list[np.ndarray] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs

        self._obs_history.clear()
        policy_obs = self.prepare_obs_fn(obs)
        for _ in range(self.n_obs_steps):
            self._obs_history.append(policy_obs)

        self._action_queue.clear()

        if self._reset_fn is not None:
            self._reset_fn()

        return obs, info

    def step(self, noise_action: np.ndarray):
        if self._last_obs is None:
            raise RuntimeError("Must call reset() before step().")

        policy_obs = self.prepare_obs_fn(self._last_obs)
        self._obs_history.append(policy_obs)

        if not self._action_queue:
            stacked_obs = self._stack_history()
            noise_tensor = self.noise_reshape_fn(noise_action, self.device)
            with torch.no_grad():
                action_chunk = self.action_fn(stacked_obs, noise_tensor)
            # action_chunk: (1, n_action_steps, action_dim) → list of (action_dim,) arrays
            self._action_queue = [a for a in action_chunk[0].cpu().numpy()]

        action = self._action_queue.pop(0)
        obs, reward, done, truncated, info = self.env.step(action)
        self._last_obs = obs
        return obs, reward, done, truncated, info

    def _stack_history(self) -> dict[str, torch.Tensor]:
        history_list = list(self._obs_history)
        stacked: dict[str, torch.Tensor] = {}
        for key in history_list[0]:
            stacked[key] = torch.stack([h[key] for h in history_list], dim=1)
        return stacked

