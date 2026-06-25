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

"""Environment utilities for DSRL macro-action execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

DSRL_ACTION_CHUNK_STEPS = "dsrl_action_chunk_steps"
DSRL_ACTION_CHUNK_INTERRUPTED = "dsrl_action_chunk_interrupted"


@dataclass(frozen=True)
class ActionChunkStepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict


def step_action_chunk(
    env: gym.Env,
    action_chunk: np.ndarray,
    after_step: Callable[[Any], None] | None = None,
) -> ActionChunkStepResult:
    """Execute a primitive-action chunk as one macro environment step."""
    if action_chunk.ndim != 2:
        raise ValueError(
            f"action_chunk must be shaped (n_action_steps, action_dim); got {action_chunk.shape}"
        )
    if action_chunk.shape[0] == 0:
        raise ValueError("action_chunk must contain at least one primitive action")

    reward_sum = 0.0
    terminated = False
    truncated = False
    info = {}
    obs = None
    executed_steps = 0

    for action in action_chunk:
        obs, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        executed_steps += 1
        if after_step is not None:
            after_step(obs)

        if terminated or truncated:
            break

    info = dict(info)
    info[DSRL_ACTION_CHUNK_STEPS] = executed_steps
    info[DSRL_ACTION_CHUNK_INTERRUPTED] = executed_steps < action_chunk.shape[0]

    return ActionChunkStepResult(
        observation=obs,
        reward=reward_sum,
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=info,
    )
