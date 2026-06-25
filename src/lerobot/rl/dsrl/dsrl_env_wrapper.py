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

"""Gym wrapper that translates noise actions to robot actions via a frozen diffusion policy."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch

from lerobot.rl.dsrl.env_utils import step_action_chunk


class DSRLEnvWrapper(gym.Wrapper):
    """Wraps an environment so the RL agent sees a noise action space.

    Each step maps one noise vector to one frozen-policy action chunk, then
    executes that whole chunk in the wrapped environment. The returned
    transition is therefore a macro transition: cumulative reward, final
    observation, and a terminal flag if the episode ended inside the chunk.
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
        reward_fn: Callable[[float, bool, bool, dict], float] | None = None,
    ):
        super().__init__(env)
        self.noise_dim = noise_dim
        self.prepare_obs_fn = prepare_obs_fn
        self.noise_reshape_fn = noise_reshape_fn
        self.action_fn = action_fn
        self.device = torch.device(device)
        self.n_obs_steps = n_obs_steps
        self._reset_fn = reset_fn
        self._reward_fn = reward_fn

        # Override action space to be noise space
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(noise_dim,),
            dtype=np.float32,
        )

        # Cache last raw observation for step()
        self._last_obs: dict | None = None

        # Rolling latest-N observation window for n_obs_steps > 1.
        self._obs_window: deque[dict[str, torch.Tensor]] = deque(maxlen=n_obs_steps)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs

        self._obs_window.clear()
        policy_obs = self.prepare_obs_fn(obs)
        for _ in range(self.n_obs_steps):
            self._obs_window.append(policy_obs)

        if self._reset_fn is not None:
            self._reset_fn()

        return obs, info

    def step(self, noise_action: np.ndarray):
        if self._last_obs is None:
            raise RuntimeError("Must call reset() before step().")

        stacked_obs = self._stack_history()
        noise_tensor = self.noise_reshape_fn(noise_action, self.device)
        with torch.no_grad():
            action_chunk = self.action_fn(stacked_obs, noise_tensor)

        if action_chunk.ndim != 3 or action_chunk.shape[0] != 1:
            raise ValueError(
                "action_fn must return a tensor shaped (1, n_action_steps, action_dim); "
                f"got {tuple(action_chunk.shape)}"
            )

        def update_obs_history(obs):
            self._last_obs = obs
            self._obs_window.append(self.prepare_obs_fn(obs))

        chunk_result = step_action_chunk(
            self.env,
            action_chunk[0].detach().cpu().numpy(),
            after_step=update_obs_history,
            reward_fn=self._reward_fn,
        )

        return (
            chunk_result.observation,
            chunk_result.reward,
            chunk_result.terminated,
            chunk_result.truncated,
            chunk_result.info,
        )

    def _stack_history(self) -> dict[str, torch.Tensor]:
        history_list = list(self._obs_window)
        stacked: dict[str, torch.Tensor] = {}
        for key in history_list[0]:
            stacked[key] = torch.stack([h[key] for h in history_list], dim=1)
        return stacked
