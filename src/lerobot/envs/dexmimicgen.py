#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.envs.utils import _LazyAsyncVectorEnv

ENV_ROBOTS: dict[str, list[str]] = {
    "SingleArmDrawerCleanup": ["PandaDexRH"],
    "TwoArmDrawerCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmBoxCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmThreading": ["Panda", "Panda"],
    "TwoArmThreePieceAssembly": ["Panda", "Panda"],
    "TwoArmTransport": ["Panda", "Panda"],
    "TwoArmLiftTray": ["PandaDexRH", "PandaDexLH"],
    "TwoArmCoffee": ["GR1FixedLowerBody"],
    "TwoArmPouring": ["GR1FixedLowerBody"],
    "TwoArmCanSortRandom": ["GR1ArmsOnly"],
}

DEFAULT_CAMERA_SIZES: dict[str, tuple[int, int]] = {
    "agentview": (256, 256),
    "robot0_eye_in_hand": (128, 128),
}
DEFAULT_MAX_EPISODE_STEPS = 400
STATE_LOW = -1000.0
STATE_HIGH = 1000.0


def _ensure_local_dexmimicgen(dexmimicgen_path: str | None) -> None:
    if dexmimicgen_path is None:
        return
    path = Path(dexmimicgen_path).expanduser()
    if path.exists():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    if isinstance(camera_name, str):
        cameras = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list | tuple)):
        cameras = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cameras:
        raise ValueError("camera_name resolved to an empty list.")
    return cameras


def _camera_size(camera: str, camera_sizes: Mapping[str, tuple[int, int]] | None) -> tuple[int, int]:
    if camera_sizes is not None and camera in camera_sizes:
        return camera_sizes[camera]
    return DEFAULT_CAMERA_SIZES.get(camera, (256, 256))


def _make_camera_dim_list(cameras: Sequence[str], camera_sizes: Mapping[str, tuple[int, int]]) -> list[int]:
    return [camera_sizes[camera][0] for camera in cameras]


def _load_controller_config(robots: list[str]) -> dict[str, Any]:
    try:
        from robosuite import load_composite_controller_config

        return load_composite_controller_config(robot=robots[0])
    except (ImportError, TypeError):
        from robosuite.controllers.composite.composite_controller_factory import (
            load_composite_controller_config,
        )

        return load_composite_controller_config(robot=robots[0])


class DexMimicGenEnv(gym.Env):
    """Gymnasium wrapper for local dexmimicgen robosuite environments."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task_name: str = "SingleArmDrawerCleanup",
        *,
        dexmimicgen_path: str | None = "~/dexmimicgen",
        robots: Sequence[str] | None = None,
        camera_name: str | Sequence[str] = "agentview,robot0_eye_in_hand",
        camera_name_mapping: dict[str, str] | None = None,
        camera_sizes: Mapping[str, tuple[int, int]] | None = None,
        obs_type: str = "pixels_agent_pos",
        state_mode: str = "joint_gripper",
        gripper_qpos_indices: Sequence[int] = (0, 2, 4, 6, 8, 11),
        render_mode: str = "rgb_array",
        control_freq: int = 20,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        seed: int | None = None,
        return_raw_obs: bool = False,
        **env_kwargs,
    ) -> None:
        super().__init__()
        _ensure_local_dexmimicgen(dexmimicgen_path)

        import dexmimicgen  # noqa: F401  # registers local robosuite environments
        import robosuite

        if task_name not in ENV_ROBOTS and robots is None:
            raise ValueError(
                f"Unknown dexmimicgen task '{task_name}'. Pass `robots` explicitly or choose one of "
                f"{list(ENV_ROBOTS)}."
            )

        self.task_name = task_name
        self.task = task_name
        self.obs_type = obs_type
        self.state_mode = state_mode
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._step_count = 0
        self.return_raw_obs = return_raw_obs
        self.gripper_qpos_indices = tuple(gripper_qpos_indices)
        self._last_rendered_images: dict[str, np.ndarray] | None = None

        self.camera_name = _parse_camera_names(camera_name)
        camera_sizes = {cam: _camera_size(cam, camera_sizes) for cam in self.camera_name}
        self.camera_sizes = dict(camera_sizes)
        self.camera_name_mapping = camera_name_mapping or {
            "agentview": "agentview",
            "robot0_eye_in_hand": "robot0_eye_in_hand",
        }

        robots = list(robots or ENV_ROBOTS[task_name])
        env_defaults = {
            "env_name": task_name,
            "robots": robots,
            "controller_configs": _load_controller_config(robots),
            "has_renderer": render_mode == "human",
            "has_offscreen_renderer": render_mode == "rgb_array",
            "ignore_done": False,
            "use_camera_obs": True,
            "control_freq": control_freq,
            "camera_names": self.camera_name,
            "camera_heights": _make_camera_dim_list(self.camera_name, camera_sizes),
            "camera_widths": [camera_sizes[cam][1] for cam in self.camera_name],
            "camera_depths": False,
            "seed": seed,
        }
        env_defaults.update(env_kwargs)
        if seed is not None:
            np.random.seed(seed)
        self._env = robosuite.make(**env_defaults)

        images = {}
        for cam in self.camera_name:
            height, width = camera_sizes[cam]
            images[self.camera_name_mapping.get(cam, cam)] = spaces.Box(
                low=0,
                high=255,
                shape=(height, width, 3),
                dtype=np.uint8,
            )

        state_dim = self._infer_state_dim()
        if obs_type == "pixels":
            self.observation_space = spaces.Dict({"pixels": spaces.Dict(images)})
        elif obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "agent_pos": spaces.Box(
                        low=STATE_LOW,
                        high=STATE_HIGH,
                        shape=(state_dim,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported obs_type '{obs_type}'. Use 'pixels' or 'pixels_agent_pos'.")

        action_low, action_high = self._env.action_spec
        self.action_dim = int(np.asarray(action_low).shape[0])
        self.action_space = spaces.Box(
            low=np.asarray(action_low, dtype=np.float32),
            high=np.asarray(action_high, dtype=np.float32),
            dtype=np.float32,
        )
        self.task_description = self._get_language_instruction()

    def _infer_state_dim(self) -> int:
        if self.state_mode == "joint_gripper":
            return 7 + len(self.gripper_qpos_indices)
        if self.state_mode == "eef_gripper":
            return 7 + len(self.gripper_qpos_indices)
        raise ValueError(f"Unsupported state_mode '{self.state_mode}'.")

    def _format_state(self, raw_obs: Mapping[str, np.ndarray]) -> np.ndarray:
        gripper = np.asarray(raw_obs["robot0_gripper_qpos"])[list(self.gripper_qpos_indices)]
        if self.state_mode == "joint_gripper":
            state = np.concatenate([raw_obs["robot0_joint_pos"], gripper], axis=-1)
        elif self.state_mode == "eef_gripper":
            state = np.concatenate([raw_obs["robot0_eef_pos"], raw_obs["robot0_eef_quat"], gripper], axis=-1)
        else:
            raise ValueError(f"Unsupported state_mode '{self.state_mode}'.")
        return np.asarray(state, dtype=np.float32)

    def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        if self.return_raw_obs:
            return raw_obs

        images = {}
        for cam in self.camera_name:
            key = f"{cam}_image"
            if key not in raw_obs:
                raise KeyError(f"Camera observation '{key}' not found. Available keys: {list(raw_obs)}")
            images[self.camera_name_mapping.get(cam, cam)] = np.asarray(raw_obs[key][::-1], dtype=np.uint8)

        if self.obs_type == "pixels":
            return {"pixels": images}
        return {"pixels": images, "agent_pos": self._format_state(raw_obs)}

    def _get_language_instruction(self) -> str:
        if hasattr(self._env, "get_task"):
            task = self._env.get_task()
            if isinstance(task, Mapping) and "language_instruction" in task:
                return str(task["language_instruction"])
        return f"Complete the {self.task_name} task"

    def reset(self, seed: int | None = None, **kwargs):
        super().reset(seed=seed)
        if seed is not None and hasattr(self._env, "set_rng"):
            self._env.set_rng(np.random.default_rng(seed))
        self._step_count = 0
        raw_obs = self._env.reset()
        obs = self._format_raw_obs(raw_obs)
        if "pixels" in obs:
            self._last_rendered_images = obs["pixels"]
        self.task_description = self._get_language_instruction()
        return obs, {"is_success": False, "task": self.task_name}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action)
        if action.ndim != 1:
            raise ValueError(f"Expected a 1-D action with shape ({self.action_dim},), got {action.shape}.")
        self._step_count += 1
        raw_obs, reward, done, info = self._env.step(action)
        obs = self._format_raw_obs(raw_obs)
        if "pixels" in obs:
            self._last_rendered_images = obs["pixels"]

        is_success = bool(self._env._check_success()) if hasattr(self._env, "_check_success") else bool(done)
        terminated = bool(done or is_success)
        truncated = self._step_count >= self.max_episode_steps
        info = dict(info)
        info.update({"task": self.task_name, "done": bool(done), "is_success": is_success})
        if terminated or truncated:
            info["final_info"] = {
                "task": self.task_name,
                "done": bool(done),
                "is_success": is_success,
            }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self._last_rendered_images is None:
            raise RuntimeError("render() called before reset().")
        return self._last_rendered_images[self.camera_name_mapping.get(self.camera_name[0], self.camera_name[0])]

    def close(self):
        self._env.close()


def _make_env_fns(
    *,
    task_name: str,
    n_envs: int,
    gym_kwargs: Mapping[str, Any],
    camera_names: list[str],
) -> list[Callable[[], DexMimicGenEnv]]:
    def _make_env(episode_index: int, **kwargs) -> DexMimicGenEnv:
        local_kwargs = dict(kwargs)
        seed = local_kwargs.pop("seed", episode_index)
        return DexMimicGenEnv(
            task_name=task_name,
            camera_name=camera_names,
            seed=seed,
            **local_kwargs,
        )

    return [partial(_make_env, episode_index, **gym_kwargs) for episode_index in range(n_envs)]


def create_dexmimicgen_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "",
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    gym_kwargs_camera_name = gym_kwargs.pop("camera_name", None)
    camera_name = camera_name if camera_name != "" else gym_kwargs_camera_name
    parsed_camera_names = _parse_camera_names(camera_name)

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    fns = _make_env_fns(
        task_name=task,
        n_envs=n_envs,
        gym_kwargs=gym_kwargs,
        camera_names=parsed_camera_names,
    )
    if env_cls is gym.vector.AsyncVectorEnv:
        out["dexmimicgen"][0] = _LazyAsyncVectorEnv(fns)
    else:
        out["dexmimicgen"][0] = env_cls(fns)
    return {suite: dict(task_map) for suite, task_map in out.items()}
