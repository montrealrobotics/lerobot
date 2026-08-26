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

"""Gym wrapper that turns a frozen diffusion policy into a noise-action environment.

The RL agent sees a *noise* action space. Each macro step maps one noise vector per
environment to one frozen-policy action chunk, executes that whole chunk in the wrapped
vector env, and returns a macro transition (cumulative reward, final observation, and a
terminal flag if the episode ended inside the chunk).

The wrapper presents a standard RL interface to the trainer: ``reset``/``step`` return
observations already converted to noise-actor (policy) format, so the training loop never
needs to know anything about the frozen policy. A single ``gym.vector.VectorEnv`` is always
used for construction and lifecycle; ``num_envs == 1`` is just a degenerate case.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

# Keys used to surface per-chunk bookkeeping through the gym ``info`` dict.
DSRL_ACTION_CHUNK_STEPS = "dsrl_action_chunk_steps"
DSRL_ACTION_CHUNK_INTERRUPTED = "dsrl_action_chunk_interrupted"
DSRL_ACTION_CHUNK_RAW_REWARD = "dsrl_action_chunk_raw_reward"
DSRL_ACTION_CHUNK_SHAPED_REWARD = "dsrl_action_chunk_shaped_reward"
DSRL_ACTION_CHUNK_SUCCESS = "dsrl_action_chunk_success"

PrepareObsFn = Callable[[dict], dict[str, Tensor]]
GenerateActionChunkFn = Callable[[dict[str, Tensor], np.ndarray], Tensor]
RewardFn = Callable[[float, bool, bool, dict], float]
# (raw_reward, shaped_reward, success, terminated, truncated, steps) -> macro reward, where
# the first two arguments are already accumulated over the chunk's primitive steps.
MacroRewardFn = Callable[[float, float, bool, bool, bool, int], float]


class DSRLEnvWrapper:
    """Vectorized DSRL macro-action wrapper around a ``gym.vector.VectorEnv``.

    Args:
        env: Synchronous (or async) gym vector env producing raw observations.
        noise_dim: Flat per-env noise vector dimension (the RL action space).
        prepare_obs_fn: Raw batched env obs → noise-actor observation (batched tensors).
            This is what ``reset``/``step`` return to the RL agent.
        generate_action_chunk_fn: ``(stacked_obs, noise) -> action_chunk`` where
            ``stacked_obs`` is the frozen-policy observation history stacked along ``dim=1``
            and the returned tensor is shaped ``(num_envs, n_action_steps, action_dim)``.
        device: Torch device the stacked observations live on.
        n_obs_steps: Number of past observations the frozen policy conditions on.
        prepare_frozen_obs_fn: Raw batched env obs → frozen-policy observation. Defaults to
            ``prepare_obs_fn`` when the frozen policy and noise actor share an observation
            format (e.g. diffusion policies). Supply a distinct transform for VLAs that need
            extra inputs such as language tokens (e.g. pi0).
        reward_fn: Optional primitive-step reward shaping ``(reward, terminated, truncated,
            info) -> reward`` applied per env. The result is summed over the chunk.
        macro_reward_fn: Optional macro-step reward, applied per env *after* the chunk has
            run and overriding the summed primitive reward. Use this for rewards that are
            only defined at the macro-step level — e.g. DSRL's goal-reaching reward of -1
            per chunk until success. Receives the chunk-accumulated raw and shaped rewards.
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        noise_dim: int,
        prepare_obs_fn: PrepareObsFn,
        generate_action_chunk_fn: GenerateActionChunkFn,
        device: str = "cuda",
        n_obs_steps: int = 1,
        prepare_frozen_obs_fn: PrepareObsFn | None = None,
        reward_fn: RewardFn | None = None,
        macro_reward_fn: MacroRewardFn | None = None,
    ):
        if not isinstance(env, gym.vector.VectorEnv):
            raise TypeError(
                "DSRLEnvWrapper requires a gym.vector.VectorEnv; wrap a single env in a "
                "SyncVectorEnv (num_envs=1) before passing it here."
            )

        self.env = env
        self.num_envs = env.num_envs
        self.noise_dim = noise_dim
        self.prepare_obs_fn = prepare_obs_fn
        self.generate_action_chunk_fn = generate_action_chunk_fn
        self.device = torch.device(device)
        self.n_obs_steps = n_obs_steps
        # When the frozen policy shares the noise actor's observation format we avoid a
        # redundant second conversion of every raw observation.
        self._prepare_frozen_obs_fn = prepare_frozen_obs_fn
        self._frozen_shares_obs = prepare_frozen_obs_fn is None
        self._reward_fn = reward_fn
        self._macro_reward_fn = macro_reward_fn

        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(noise_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_envs, noise_dim), dtype=np.float32
        )

        self._has_reset = False
        self._obs_window: deque[dict[str, Tensor]] = deque(maxlen=n_obs_steps)

    def reset(self, env_mask: np.ndarray | None = None, **kwargs) -> tuple[dict[str, Tensor], dict]:
        """Reset the env and return the noise-actor observation (current frame).

        Args:
            env_mask: Optional boolean mask of shape ``(num_envs,)`` selecting which
                sub-envs to reset. Envs left out keep their current observation and their
                observation history, so in-flight episodes survive. ``None`` resets all.
        """
        if env_mask is not None:
            env_mask = np.asarray(env_mask, dtype=bool).reshape(self.num_envs)
            if not env_mask.any():
                raise ValueError("env_mask must select at least one env to reset.")
            if env_mask.all():
                env_mask = None

        if env_mask is None:
            batched_obs, info = self.env.reset(**kwargs)
        else:
            options = dict(kwargs.pop("options", None) or {})
            options["reset_mask"] = env_mask
            batched_obs, info = self.env.reset(options=options, **kwargs)

        self._has_reset = True
        policy_obs, frozen_obs = self._prepare(batched_obs)

        if env_mask is None:
            # Prime the history with n_obs_steps copies of the initial frame. Each slot gets
            # its own dict so the per-env merge below can never alias across timesteps.
            self._obs_window.clear()
            for _ in range(self.n_obs_steps):
                self._obs_window.append({key: value.clone() for key, value in frozen_obs.items()})
        else:
            # Only the reset envs get their history overwritten with the fresh frame.
            index = torch.from_numpy(env_mask).to(self.device)
            merged: list[dict[str, Tensor]] = []
            for past_obs in self._obs_window:
                slot = {key: value.clone() for key, value in past_obs.items()}
                for key, value in frozen_obs.items():
                    slot[key][index] = value[index]
                merged.append(slot)
            self._obs_window.clear()
            self._obs_window.extend(merged)

        return policy_obs, info

    def step(self, noise_action: np.ndarray):
        """Execute one frozen-policy action chunk per env as a single macro step."""
        if not self._has_reset:
            raise RuntimeError("Must call reset() before step().")

        noise_action = np.asarray(noise_action, dtype=np.float32)
        if noise_action.shape != (self.num_envs, self.noise_dim):
            raise ValueError(
                f"noise_action must be shaped ({self.num_envs}, {self.noise_dim}); got {noise_action.shape}"
            )

        stacked_obs = self._stack_history()
        with torch.no_grad():
            action_chunk = self.generate_action_chunk_fn(stacked_obs, noise_action)
        if action_chunk.ndim != 3 or action_chunk.shape[0] != self.num_envs:
            raise ValueError(
                "generate_action_chunk_fn must return a tensor shaped "
                f"(num_envs, n_action_steps, action_dim); got {tuple(action_chunk.shape)}"
            )
        action_chunk_np = action_chunk.detach().cpu().numpy()

        raw_rewards = np.zeros(self.num_envs, dtype=np.float32)
        shaped_rewards = np.zeros(self.num_envs, dtype=np.float32)
        executed_steps = np.zeros(self.num_envs, dtype=np.int64)
        successes = np.zeros(self.num_envs, dtype=bool)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        info: dict = {}
        policy_obs: dict[str, Tensor] | None = None

        for action_step in range(action_chunk_np.shape[1]):
            observation, reward, terminated, truncated, info = self.env.step(action_chunk_np[:, action_step])
            reward = np.asarray(reward, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)

            raw_rewards += reward
            shaped_rewards += self._shape_rewards(reward, terminated, truncated, info)
            successes |= self._read_successes(info)
            executed_steps += 1

            policy_obs, frozen_obs = self._prepare(observation)
            self._obs_window.append(frozen_obs)
            if np.any(terminated | truncated):
                break

        if policy_obs is None:
            raise RuntimeError("Action chunk did not execute any primitive steps.")

        macro_rewards = shaped_rewards
        if self._macro_reward_fn is not None:
            macro_rewards = np.asarray(
                [
                    self._macro_reward_fn(
                        float(raw_rewards[idx]),
                        float(shaped_rewards[idx]),
                        bool(successes[idx]),
                        bool(terminated[idx]),
                        bool(truncated[idx]),
                        int(executed_steps[idx]),
                    )
                    for idx in range(self.num_envs)
                ],
                dtype=np.float32,
            )

        info = dict(info)
        info[DSRL_ACTION_CHUNK_STEPS] = executed_steps
        info[DSRL_ACTION_CHUNK_INTERRUPTED] = executed_steps < action_chunk_np.shape[1]
        info[DSRL_ACTION_CHUNK_RAW_REWARD] = raw_rewards
        info[DSRL_ACTION_CHUNK_SHAPED_REWARD] = shaped_rewards
        info[DSRL_ACTION_CHUNK_SUCCESS] = successes

        return policy_obs, macro_rewards, terminated, truncated, info

    def _read_successes(self, info: dict) -> np.ndarray:
        """Per-env ``is_success`` flags for one primitive step, defaulting to False."""
        successes = np.zeros(self.num_envs, dtype=bool)
        for idx in range(self.num_envs):
            successes[idx] = bool(_select_info(info, idx, self.num_envs).get("is_success", False))
        return successes

    def close(self):
        self.env.close()

    def _prepare(self, raw_obs: dict) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Convert a raw observation into (noise-actor obs, frozen-policy obs)."""
        policy_obs = self.prepare_obs_fn(raw_obs)
        if self._frozen_shares_obs:
            return policy_obs, policy_obs
        return policy_obs, self._prepare_frozen_obs_fn(raw_obs)

    def _stack_history(self) -> dict[str, Tensor]:
        history = list(self._obs_window)
        return {key: torch.stack([h[key] for h in history], dim=1) for key in history[0]}

    def _shape_rewards(
        self, rewards: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, info: dict
    ) -> np.ndarray:
        if self._reward_fn is None:
            return rewards
        return np.asarray(
            [
                self._reward_fn(
                    float(rewards[idx]),
                    bool(terminated[idx]),
                    bool(truncated[idx]),
                    _select_info(info, idx, self.num_envs),
                )
                for idx in range(self.num_envs)
            ],
            dtype=np.float32,
        )


def _select_info(info: dict, index: int, num_envs: int) -> dict:
    """Extract the per-env slice of a gym vector ``info`` dict."""
    selected = {}
    for key, value in info.items():
        if key.startswith("_"):
            continue
        mask = info.get(f"_{key}")
        if isinstance(mask, np.ndarray) and not bool(mask[index]):
            continue
        if isinstance(value, np.ndarray) and value.shape[:1] == (num_envs,):
            selected[key] = value[index]
        else:
            selected[key] = value
    return selected
