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

"""DSRL over a frozen pi05 policy on RoboCasa (CoffeePressButton, parallel gripper).

This is the RoboCasa analogue of the reference DSRL implementation's LIBERO experiment
(``~/dsrl_pi0``): a small SAC policy steers the noise input of a frozen, RoboCasa-finetuned
pi05 flow-matching policy. It mirrors the structure of the PushT diffusion example
(``dsrl_pusht_example.py``) but swaps the diffusion policy for pi05 and PushT for RoboCasa.

The frozen pi05 checkpoint is the output of ``examples/training/train_policy_casa_dex_pi05.py``.
We start with the *parallel-gripper* embodiment (``PandaOmron``, 7-DoF action) to validate the
pipeline before moving to the dexterous hand.

Two observation views are used:
  * the *noise actor* (small SAC policy) sees a compact, resized single-camera + state
    observation, exactly as in the reference implementation;
  * the *frozen pi05* policy sees its full, normalized, tokenized observation, built by
    reusing the checkpoint's own processor pipeline (the same path ``lerobot-eval`` takes).

Example usage:
    python examples/tutorial/rl/dsrl_pi05_robocasa_example.py \\
        --policy_path /path/to/pi05_robocasa/checkpoints/010000/pretrained_model \\
        --task CoffeePressButton \\
        --total_steps 500000 \\
        --eval_freq 10000 \\
        --wandb_enable
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

# RoboCasa's placement sampler JITs with numba ``cache=True``. A stale or partially-written
# numba cache surfaces as "EOFError: Ran out of input" at env creation. Isolating a fresh
# per-process cache dir sidesteps it. This must run before robocasa (and thus numba) is
# imported, and defers to a cache dir the caller already chose.
os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="numba_cache_"))

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.envs.configs import RoboCasaEnv
from lerobot.envs.factory import make_env
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.dsrl import train_dsrl
from lerobot.rl.dsrl.dsrl_config import DSRLConfig
from lerobot.rl.dsrl.dsrl_env_wrapper import (
    DSRL_ACTION_CHUNK_RAW_REWARD,
    DSRL_ACTION_CHUNK_STEPS,
    DSRL_ACTION_CHUNK_SUCCESS,
    DSRLEnvWrapper,
)
from lerobot.rl.dsrl.noise_actor import NoiseActorPolicy
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.io_utils import write_video

# Default cameras: one external view (agentview_left) + the wrist view (eye_in_hand). These
# are exactly the two image features the pi05 RoboCasa checkpoint was trained on. Providing a
# single external camera also makes the checkpoint's random-external-camera augmentation a
# deterministic no-op at inference.
DEFAULT_CAMERAS = "robot0_agentview_left,robot0_eye_in_hand"
# The reference DSRL feeds the SAC agent a single third-person camera + state in both LIBERO
# and ALOHA (never the wrist view). We match that by default; pass a comma-separated list to
# feed more views (e.g. add the wrist) for A/B testing.
DEFAULT_NOISE_ACTOR_CAMERAS = "observation.images.robot0_agentview_left"
DEFAULT_PROMPT = "Press the coffee machine button."


def _make_collect_env(
    env_cfg: RoboCasaEnv,
    num_envs: int,
    placement_bank: dict | None = None,
    seed: int = 0,
    kitchen_bank: dict | None = None,
    start_rejection: dict | None = None,
    scene_seeds: list[int] | None = None,
):
    """Collection vec env: sub-env ``i`` is one fixed kitchen, by construction seed.

    ``scene_seeds`` names that seed per sub-env; ``None`` means ``scene_seeds[i] == i``.

    Without a bank this is exactly ``make_env``. A placement bank moves the task object to that
    kitchen's next training placement on every reset; a kitchen bank makes each sub-env a pool of
    the bank's training kitchens, dealt round-robin, switching kitchen on every reset. Keeping
    ``num_envs`` slots either way keeps env steps per SAC update identical to the fixed-start
    runs. ``start_rejection`` wraps each raw env, inside any placement wrapper.
    """
    if num_envs < 1:
        raise ValueError("--collect_envs must be >= 1")
    scene_seeds = list(range(num_envs)) if scene_seeds is None else list(scene_seeds)
    if len(scene_seeds) != num_envs:
        raise ValueError(f"scene_seeds has {len(scene_seeds)} entries but --collect_envs is {num_envs}")

    def raw(s: int):
        return _wrap_start_rejection(_make_single_robocasa_env(env_cfg, seed=s), start_rejection)

    if kitchen_bank is not None:
        from lerobot.envs.robocasa_placement_bank import KitchenPoolEnv

        train_seeds = [k["scene_seed"] for k in kitchen_bank["train"]]
        slots = [train_seeds[i::num_envs] for i in range(num_envs)]
        for i, pool in enumerate(slots):
            print(f"Collect slot {i}: kitchen pool (construction seeds) {pool}")

        def pool_factory(i: int):
            return lambda: KitchenPoolEnv(raw, slots[i], shuffle=True, seed=seed + i)

        return gym.vector.SyncVectorEnv([pool_factory(i) for i in range(num_envs)])
    if placement_bank is None:
        if start_rejection is None and scene_seeds == list(range(num_envs)):
            envs = make_env(env_cfg, n_envs=num_envs, use_async_envs=False)
            return envs[env_cfg.type][0]
        return gym.vector.SyncVectorEnv([lambda s=s: raw(s) for s in scene_seeds])

    from lerobot.envs.robocasa_placement_bank import PlacementBankWrapper, scene_entries

    def factory(i: int, scene_seed: int):
        def _make():
            entries = scene_entries(placement_bank, scene_seed, "train")
            return PlacementBankWrapper(
                raw(scene_seed),
                placement_bank,
                scene_seed,
                entries,
                shuffle=True,
                seed=seed + i,
            )

        return _make

    # Same vector-env class make_env uses for a synchronous RoboCasa env.
    return gym.vector.SyncVectorEnv([factory(i, s) for i, s in enumerate(scene_seeds)])


def _wrap_start_rejection(env: gym.Env, start_rejection: dict | None) -> gym.Env:
    if start_rejection is None:
        return env
    from lerobot.envs.robocasa_placement_bank import StartPoseRejectionWrapper

    return StartPoseRejectionWrapper(env, **start_rejection)


def _register_literal_draccus_decoder() -> None:
    """Teach draccus to decode ``typing.Literal`` fields (passthrough; the dataclass's own
    ``__post_init__`` validates the value).

    ``PreTrainedConfig.from_pretrained`` decodes the saved config with draccus, which has no
    built-in ``Literal`` decoder. pi05 checkpoints happen to omit their ``Literal`` fields
    (defaults), but SmolVLA checkpoints save them (e.g. ``category_specific_action_proj_type``),
    so loading one raises "No decoding function for type typing.Literal". This registration is a
    no-op if the value is already a plain str/int.
    """
    import typing

    import draccus

    draccus.decode.register(typing.Literal, lambda raw_value, path=(): raw_value)


def load_frozen_policy(policy_path: str) -> PreTrainedPolicy:
    """Load any flow-matching VLA (pi05, SmolVLA, ...) by auto-detecting its config type.

    DSRL only needs the policy's ``predict_action_chunk(batch, noise=...)`` seam, which pi05
    and SmolVLA share, so the example is policy-agnostic: it dispatches on the checkpoint's
    registered config type.

    LoRA/PEFT adapter checkpoints (``adapter_config.json`` present) are loaded by rebuilding
    the finetuned policy on its base weights and applying the adapter — which brings both the
    LoRA deltas and any fully-trained modules saved via ``modules_to_save`` (e.g. the flow
    action expert in ``*_expert_full_ft`` runs) — then merging for frozen inference. This
    mirrors ``lerobot.policies.factory.make_policy``'s PEFT branch, reusing its ``get_policy_class``
    dispatch and the ``peft`` primitives.
    """
    import json

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    _register_literal_draccus_decoder()
    # Read the type first so get_policy_class imports the module, registering the config
    # subclass with PreTrainedConfig's choice registry before we decode.
    with open(os.path.join(policy_path, "config.json")) as f:
        policy_type = json.load(f)["type"]
    policy_cls = get_policy_class(policy_type)
    cfg = PreTrainedConfig.from_pretrained(policy_path)

    if not os.path.exists(os.path.join(policy_path, "adapter_config.json")):
        print(f"Loading '{cfg.type}' policy from {policy_path} ...")
        return policy_cls.from_pretrained(policy_path)

    from peft import PeftConfig, PeftModel

    peft_config = PeftConfig.from_pretrained(policy_path)
    base = peft_config.base_model_name_or_path
    if not base:
        raise ValueError(f"{policy_path}: adapter_config has no base_model_name_or_path to build on.")
    print(f"Loading '{cfg.type}' LoRA checkpoint (base={base}); applying + merging adapter ...")
    # Build the finetuned architecture (cfg) on the base weights, then layer the adapter on top.
    base_policy = policy_cls.from_pretrained(pretrained_name_or_path=base, config=cfg)
    peft_policy = PeftModel.from_pretrained(base_policy, policy_path, config=peft_config)
    return peft_policy.merge_and_unload()


# ── Observation adapters ────────────────────────────────────────────────────────────


def make_prepare_obs_fn(device: torch.device, noise_actor_cameras: list[str], resize_size: int | None = None):
    """Raw (batched) RoboCasa obs → compact noise-actor observation (image(s) + state).

    The small SAC policy conditions on the resized ``noise_actor_cameras`` and the robot
    state; the reference DSRL implementation uses a single third-person view. The heavy,
    normalized, tokenized observation the VLA needs is built separately by
    :func:`make_prepare_frozen_obs_fn`. Multiple cameras are stacked along the channel axis
    by the compact encoder.

    Images are downsampled to ``resize_size`` here — i.e. *before* they reach the replay
    buffer.
    """

    def prepare_obs(obs: dict) -> dict[str, torch.Tensor]:
        policy_obs = preprocess_observation(obs)
        out: dict[str, torch.Tensor] = {}
        if OBS_STATE in policy_obs:
            out[OBS_STATE] = policy_obs[OBS_STATE].to(device)
        for cam in noise_actor_cameras:
            if cam not in policy_obs:
                raise KeyError(
                    f"Noise-actor camera {cam!r} not in observation keys "
                    f"{sorted(k for k in policy_obs if k.startswith('observation.image'))}."
                )
            image = policy_obs[cam].to(device)
            if resize_size is not None and image.shape[-2:] != (resize_size, resize_size):
                image = F.interpolate(
                    image, size=(resize_size, resize_size), mode="bilinear", align_corners=False
                )
            out[cam] = image
        return out

    return prepare_obs


def make_prepare_frozen_obs_fn(
    env, preprocessor, device: torch.device, embodiment_id: int | None, prompt: str
):
    """Raw (batched) RoboCasa obs → fully-preprocessed pi05 batch.

    Mirrors the ``lerobot-eval`` conversion: :func:`preprocess_observation`, then inject the
    per-env task string and (optionally) the embodiment id, then run the checkpoint's own
    preprocessor pipeline (normalization, canonical image selection, state discretization,
    PaliGemma tokenization, device placement). Non-tensor entries (e.g. the raw ``task``
    strings) are dropped so the DSRL wrapper can stack the observation history.
    """

    def _tasks(batch_size: int) -> list[str]:
        # Prefer the env's per-episode language description; fall back to a fixed prompt.
        try:
            tasks = list(env.call("task_description"))
            tasks = [t if isinstance(t, str) and t else prompt for t in tasks]
            if len(tasks) == batch_size:
                return tasks
        except Exception:  # noqa: BLE001 — task description is best-effort; fall back to the prompt.
            pass
        return [prompt] * batch_size

    def prepare_frozen_obs(obs: dict) -> dict[str, torch.Tensor]:
        policy_obs = preprocess_observation(obs)
        batch_size = next(iter(policy_obs.values())).shape[0]

        if embodiment_id is not None:
            policy_obs["embodiment_id"] = torch.tensor([embodiment_id] * batch_size, device=device)
        policy_obs["task"] = _tasks(batch_size)

        processed = preprocessor(policy_obs)
        # Keep only tensors — the wrapper stacks the observation window with torch.stack.
        return {key: value for key, value in processed.items() if isinstance(value, torch.Tensor)}

    return prepare_frozen_obs


# ── Action generation (frozen pi05 with injected noise) ──────────────────────────────


def make_generate_action_chunk_fn(
    frozen_policy: PreTrainedPolicy,
    postprocessor,
    noise_chunk_size: int,
    full_chunk_size: int,
    max_action_dim: int,
    n_action_steps: int,
    device: torch.device,
):
    """Bundle noise reshaping + pi05 flow sampling + action unnormalization.

    ``(stacked_obs, noise) -> action_chunk`` of shape ``(num_envs, n_action_steps, action_dim)``.

    The SAC policy emits a compact noise vector ``(B, noise_chunk_size * max_action_dim)``
    which is reshaped to ``(B, noise_chunk_size, max_action_dim)`` and padded up to pi05's
    full flow-matching chunk by repeating the last noise step (the reference DSRL trick for
    shrinking the noise space). pi05 denoises the full chunk; we then execute only the first
    ``n_action_steps`` actions per DSRL macro-step.
    """

    comp_keys = ("embodiment_id", "normalization_id")

    def _unnormalize(action_chunk: torch.Tensor, comp_data: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, chunk, action_dim = action_chunk.shape
        flat = action_chunk.reshape(batch_size * chunk, action_dim)
        flat_comp = (
            {k: v.repeat_interleave(chunk, dim=0) for k, v in comp_data.items()} if comp_data else None
        )
        flat = postprocessor(flat, complementary_data=flat_comp or None)
        return flat.reshape(batch_size, chunk, action_dim)

    def generate_action_chunk(stacked_obs: dict[str, torch.Tensor], noise: np.ndarray) -> torch.Tensor:
        # Drop the single-step observation-history axis pi05 does not consume.
        batch = {key: value[:, 0] for key, value in stacked_obs.items()}

        batch_size = noise.shape[0]
        noise_tensor = torch.from_numpy(noise).float().to(device)
        noise_tensor = noise_tensor.view(batch_size, noise_chunk_size, max_action_dim)
        pad = full_chunk_size - noise_chunk_size
        if pad > 0:
            noise_tensor = torch.cat([noise_tensor, noise_tensor[:, -1:, :].repeat(1, pad, 1)], dim=1)

        action_chunk = frozen_policy.predict_action_chunk(batch, noise=noise_tensor)
        action_chunk = action_chunk[:, :n_action_steps]

        comp_data = {k: batch[k] for k in comp_keys if k in batch}
        return _unnormalize(action_chunk, comp_data)

    return generate_action_chunk


# ── Reward functions ─────────────────────────────────────────────────────────────────
# Primitive-step rewards (summed over the chunk).


def _dense_reward(reward: float, terminated: bool, truncated: bool, info: dict) -> float:
    del terminated, truncated, info
    return reward


REWARD_FNS = {"dense": _dense_reward}


# Macro-step (per action chunk) rewards, overriding the summed primitive reward.


def _goal_reward(
    raw_reward: float, shaped_reward: float, success: bool, terminated: bool, truncated: bool, steps: int
) -> float:
    """DSRL's goal-reaching reward: -1 per action chunk until success, 0 on success."""
    del raw_reward, shaped_reward, terminated, truncated, steps
    return 0.0 if success else -1.0


MACRO_REWARD_FNS = {"goal": _goal_reward}


# ── Evaluation ───────────────────────────────────────────────────────────────────────


def _make_single_robocasa_env(env_cfg: RoboCasaEnv, seed: int, ep_meta: dict | None = None):
    """Raw (non-vectorized) RoboCasa gym env, built with the same fields ``make_env`` uses.

    A given ``seed`` yields the same fixed kitchen, object instance and placement; only the arm
    reset noise varies per reset. ``ep_meta={"layout_ids": [L], "style_ids": [S]}`` pins the
    kitchen instead, turning ``seed`` into a sweep over object placement within that scene.
    """
    from lerobot.envs.robocasa_env import RoboCasaEnv as RoboCasaGymEnv

    return RoboCasaGymEnv(
        task_name=env_cfg.task,
        robot=env_cfg.robot,
        controller=env_cfg.controller,
        control_freq=env_cfg.fps,
        camera_name=env_cfg.camera_name,
        obs_type=env_cfg.obs_type,
        render_mode=env_cfg.render_mode,
        observation_width=env_cfg.observation_width,
        observation_height=env_cfg.observation_height,
        camera_name_mapping=env_cfg.camera_name_mapping,
        seed=seed,
        ep_meta=ep_meta,
    )


def _parse_eval_scenes(spec: str) -> list[tuple[int, int] | None]:
    """``"1:4,8:6"`` -> ``[(1, 4), (8, 6)]``; ``""`` -> ``[None]`` (seed picks the kitchen)."""
    spec = spec.strip()
    if not spec:
        return [None]
    scenes: list[tuple[int, int] | None] = []
    for chunk in spec.split(","):
        if not chunk.strip():
            continue
        layout, _, style = chunk.partition(":")
        if not style:
            raise ValueError(f"--eval_scenes entry {chunk!r} must look like 'layout:style', e.g. '1:4'")
        scenes.append((int(layout), int(style)))
    return scenes or [None]


from lerobot.envs.robocasa_placement_bank import (  # noqa: E402
    FIXED_RUN_KITCHEN_SEEDS,  # noqa: F401
    kitchen_eval_cells as _kitchen_eval_cells,
    placement_eval_cells as _placement_eval_cells,
)


class _RoboCasaFrameCapture(gym.Wrapper):
    """Records one camera's frames for the episodes flagged via :meth:`start`."""

    def __init__(self, env: gym.Env, camera: str):
        super().__init__(env)
        self.camera = camera
        self.frames: list[np.ndarray] = []
        self._recording = False

    def start(self, recording: bool):
        self.frames = []
        self._recording = recording

    def _grab(self, obs: dict):
        if not self._recording:
            return
        frame = obs.get("pixels", {}).get(self.camera)
        if frame is not None:
            self.frames.append(np.asarray(frame).copy())

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._grab(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._grab(obs)
        return obs, reward, terminated, truncated, info


def _log_eval_videos(
    labeled_frames: list[tuple[str, list[np.ndarray]]], step: int, fps: int, video_dir: Path
):
    """Write each kitchen's captured episode to mp4 and log to WandB (if a run is active).

    ``labeled_frames`` is a list of ``(label, frames)`` where ``label`` identifies the kitchen
    (e.g. ``"seed0"``), so each video is clearly attributable to its scene.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    video_dir.mkdir(parents=True, exist_ok=True)
    log_data = {}
    for label, frames in labeled_frames:
        if not frames:
            continue
        video_path = video_dir / f"eval_step_{step:08d}_{label}.mp4"
        try:
            write_video(video_path, frames, fps=fps)
        except Exception as exc:  # noqa: BLE001 — never let a video failure abort training.
            print(f"[eval video] failed to save {video_path}: {type(exc).__name__}: {exc}")
            continue
        log_data[f"eval/video_{label}"] = wandb.Video(str(video_path), fps=fps, format="mp4")

    if log_data:
        wandb.log(log_data, step=step)
        print(f"[eval video @ step {step}] logged {len(log_data)} eval videos")


def make_robocasa_eval_fn(
    env_cfg: RoboCasaEnv,
    make_frozen_obs_for_env,
    prepare_obs_fn,
    generate_action_chunk_fn,
    reward_fn,
    macro_reward_fn,
    device: torch.device,
    noise_dim: int,
    n_obs_steps: int,
    num_episodes: int = 10,
    eval_seeds: tuple[int, ...] = (0,),
    eval_scenes: list[tuple[int, int] | None] | None = None,
    num_videos: int = 3,
    video_camera: str = "robot0_agentview_left",
    video_dir: str | Path | None = None,
    placement_bank: dict | None = None,
    placement_cells: list[dict] | None = None,
    start_rejection: dict | None = None,
):
    """Roll out the noise actor greedily in each eval cell; report per-cell + aggregate.

    A cell is one start condition and runs ``num_episodes`` episodes. With ``eval_scenes=[None]``
    each seed in ``eval_seeds`` is a different kitchen (the historical behaviour); with explicit
    ``eval_scenes`` the kitchen is pinned and ``eval_seeds`` sweeps object placement within it;
    with ``placement_cells`` the cells come from a bank and metrics are also aggregated per group
    (``success_rate_heldout`` / ``_reference`` / ``_train``).

    The actor is evaluated deterministically, so episodes within a cell differ only by robocasa's
    arm-joint reset noise (0.02 rad) -- the number of cells is the effective sample size.
    """
    video_dir = Path(video_dir) if video_dir is not None else Path("outputs/dsrl_pi05_robocasa/eval_videos")
    video_fps = env_cfg.fps
    eval_seeds = tuple(eval_seeds)
    eval_scenes = list(eval_scenes) if eval_scenes else [None]

    def _cell_label(scene: tuple[int, int] | None, seed: int) -> str:
        return f"seed{seed}" if scene is None else f"L{scene[0]}S{scene[1]}_p{seed}"

    def _rollout_kitchen(
        noise_actor: NoiseActorPolicy,
        seed: int,
        record_video: bool,
        scene: tuple[int, int] | None = None,
        entry: dict | None = None,
    ):
        ep_meta = None if scene is None else {"layout_ids": [scene[0]], "style_ids": [scene[1]]}
        raw_env = _wrap_start_rejection(
            _make_single_robocasa_env(env_cfg, seed=seed, ep_meta=ep_meta), start_rejection
        )
        if entry is not None:
            from lerobot.envs.robocasa_placement_bank import PlacementBankWrapper

            raw_env = PlacementBankWrapper(raw_env, placement_bank, seed, [entry], shuffle=False)
        frame_capture = _RoboCasaFrameCapture(raw_env, video_camera)
        eval_env = gym.vector.SyncVectorEnv([lambda: frame_capture])
        dsrl_env = DSRLEnvWrapper(
            env=eval_env,
            noise_dim=noise_dim,
            prepare_obs_fn=prepare_obs_fn,
            generate_action_chunk_fn=generate_action_chunk_fn,
            device=str(device),
            n_obs_steps=n_obs_steps,
            prepare_frozen_obs_fn=make_frozen_obs_for_env(eval_env),
            reward_fn=reward_fn,
            macro_reward_fn=macro_reward_fn,
        )

        successes = 0
        ret = 0.0
        steps = 0
        video_frames: list[np.ndarray] | None = None

        for episode_idx in range(num_episodes):
            # Record only the first episode of a kitchen (one representative video per scene).
            frame_capture.start(record_video and episode_idx == 0)
            policy_obs, _ = dsrl_env.reset()
            done = False
            episode_return = 0.0
            episode_success = False

            while not done:
                with torch.no_grad():
                    noise = noise_actor.select_action(policy_obs, deterministic=True)
                noise_np = noise.cpu().numpy().reshape(1, noise_dim)
                policy_obs, reward, terminated, truncated, info = dsrl_env.step(noise_np)

                episode_return += float(info[DSRL_ACTION_CHUNK_RAW_REWARD][0])
                episode_success = episode_success or bool(info[DSRL_ACTION_CHUNK_SUCCESS][0])
                steps += int(info[DSRL_ACTION_CHUNK_STEPS][0])
                done = bool(terminated[0]) or bool(truncated[0])

            successes += int(episode_success)
            ret += episode_return
            if video_frames is None and frame_capture.frames:
                video_frames = frame_capture.frames

        dsrl_env.close()
        return successes, ret, steps, video_frames

    def eval_fn(noise_actor: NoiseActorPolicy, step: int) -> dict[str, float]:
        noise_actor.eval()

        total_successes = 0
        total_return = 0.0
        total_steps = 0
        metrics: dict[str, float] = {}
        labeled_frames: list[tuple[str, list[np.ndarray]]] = []

        if placement_cells is not None:
            cells = placement_cells
        else:
            cells = [
                {
                    "seed": seed,
                    "scene": scene,
                    "entry": None,
                    "label": _cell_label(scene, seed),
                    "group": None,
                }
                for scene in eval_scenes
                for seed in eval_seeds
            ]
        group_successes: dict[str, list[int]] = {}
        for i, cell in enumerate(cells):
            successes, ret, steps, frames = _rollout_kitchen(
                noise_actor,
                cell["seed"],
                record_video=i < num_videos,
                scene=cell["scene"],
                entry=cell["entry"],
            )
            total_successes += successes
            total_return += ret
            total_steps += steps
            metrics[f"success_rate_{cell['label']}"] = successes / max(num_episodes, 1)
            if cell["group"] is not None:
                group = group_successes.setdefault(cell["group"], [0, 0])
                group[0] += successes
                group[1] += num_episodes
            if frames:
                labeled_frames.append((cell["label"], frames))

        noise_actor.train()

        if labeled_frames:
            _log_eval_videos(labeled_frames, step, fps=video_fps, video_dir=video_dir)

        for group, (succ, n) in group_successes.items():
            metrics[f"success_rate_{group}"] = succ / max(n, 1)
        total_episodes = max(num_episodes * len(cells), 1)
        metrics["success_rate"] = total_successes / total_episodes
        metrics["avg_return"] = total_return / total_episodes
        metrics["avg_length"] = total_steps / total_episodes
        return metrics

    return eval_fn


# ── Eval-only (score a saved noise actor) ────────────────────────────────────────────


def _load_noise_actor(path: str, dsrl_cfg: DSRLConfig, device: torch.device) -> NoiseActorPolicy:
    """Rebuild a saved noise actor from a ``train_dsrl`` checkpoint dir.

    ``config.json`` carries ``noise_dim`` and ``input_features``, but ``dsrl_config`` is not saved:
    pass the same ``--dsrl_*`` flags the run trained with or the state dict will not fit.
    """
    from safetensors.torch import load_file

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.rl.dsrl.noise_actor import LightweightNoiseActorPolicy

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = str(device)
    actor = (
        LightweightNoiseActorPolicy(cfg, dsrl_config=dsrl_cfg)
        if dsrl_cfg.use_compact_encoder
        else NoiseActorPolicy(cfg)
    )
    state_dict = load_file(os.path.join(path, "model.safetensors"))
    # ``save_pretrained`` writes safetensors, which cannot store the same tensor twice. With
    # ``shared_encoder=True`` the encoder is reachable as ``actor.encoder.*``,
    # ``encoder_actor.*`` and ``encoder_critic.*``, so only one of those names survives in the
    # file and a strict load reports the aliases as missing. Load non-strict, then verify by
    # tensor identity that every *distinct* parameter really did receive a value — that still
    # catches a genuine architecture mismatch from wrong --dsrl_* flags.
    missing, unexpected = actor.load_state_dict(state_dict, strict=False)
    # Compare by storage pointer, not id(): state_dict() hands back a fresh detached Tensor per
    # call, so identity never matches, but aliases of one parameter share their storage.
    own = actor.state_dict()
    loaded_ptrs = {t.data_ptr() for name, t in own.items() if name in state_dict}
    truly_missing = sorted(name for name in missing if own[name].data_ptr() not in loaded_ptrs)
    if truly_missing or unexpected:
        raise RuntimeError(
            f"Noise-actor state dict does not match the rebuilt architecture "
            f"({len(truly_missing)} unloaded, {len(unexpected)} unexpected keys). The --dsrl_* "
            f"flags most likely differ from the training run (image_resize={dsrl_cfg.image_resize_size}, "
            f"hidden_dims={dsrl_cfg.hidden_dims}). First unloaded: {truly_missing[:3]}"
        )
    if missing:
        print(f"  ({len(missing)} shared-encoder alias keys resolved via shared storage)")
    actor.to(device)
    actor.eval()
    print(f"Loaded noise actor from {path} (noise_dim={cfg.noise_dim})")
    return actor


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from ``c`` successes in ``n`` rollouts (Chen et al., 2021)."""
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    from math import comb

    return 1.0 - comb(n - c, k) / comb(n, k)


def _run_steerability_eval(
    args,
    env_cfg: RoboCasaEnv,
    make_frozen_obs_for_env,
    prepare_obs_fn,
    generate_action_chunk_fn,
    reward_fn,
    macro_reward_fn,
    device: torch.device,
    noise_dim: int,
    n_action_steps: int,
    starts: list[tuple[int, str]],
    start_rejection: dict | None = None,
) -> None:
    """Best-of-K steerability of a frozen policy: random-noise rollouts, no actor, no training.

    Each chunk's noise is a fresh ``randn(noise_dim)`` -- the sampler DSRL's warmup uses -- pushed
    through the same wrapper and chunking as training. ``--noise_chunk_size`` sets the
    parameterization: 1 holds one vector across the chunk (the space the DSRL actor steers in),
    ``chunk_size`` draws fresh noise per timestep (vanilla pi05, the control). Episode bookkeeping
    mirrors dsrl_trainer. ``starts`` is ``[(construction_seed, group)]``; reports mean success and
    unbiased pass@k per start plus the mean over starts per group.
    """
    import json

    seeds = [seed for seed, _ in starts]
    group_of = dict(starts)
    per_start = args.steer_parallel_per_start
    quota = args.steer_rollouts_per_start
    factories = [
        lambda s=s: _wrap_start_rejection(_make_single_robocasa_env(env_cfg, seed=s), start_rejection)
        for s in seeds
        for _ in range(per_start)
    ]
    vec = gym.vector.SyncVectorEnv(factories)
    num_envs = len(factories)
    dsrl_env = DSRLEnvWrapper(
        env=vec,
        noise_dim=noise_dim,
        prepare_obs_fn=prepare_obs_fn,
        generate_action_chunk_fn=generate_action_chunk_fn,
        device=str(device),
        n_obs_steps=1,
        prepare_frozen_obs_fn=make_frozen_obs_for_env(vec),
        reward_fn=reward_fn,
        macro_reward_fn=macro_reward_fn,
    )
    print(
        f"Steerability eval: starts {starts} x {quota} rollouts, {per_start} parallel envs/start, "
        f"start rejection {start_rejection}, "
        f"noise_chunk_size={args.noise_chunk_size} (noise_dim={noise_dim}), exec {n_action_steps} steps/chunk"
    )

    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    results: dict[int, list[dict]] = {s: [] for s in seeds}
    ep_success = np.zeros(num_envs, dtype=bool)
    ep_steps = np.zeros(num_envs, dtype=np.int64)
    dsrl_env.reset()
    while any(len(results[s]) < quota for s in seeds):
        noise = torch.randn(num_envs, noise_dim, generator=rng).numpy()
        _, _, terminated, truncated, info = dsrl_env.step(noise)
        ep_success |= np.asarray(info[DSRL_ACTION_CHUNK_SUCCESS], dtype=bool)
        ep_steps += np.asarray(info[DSRL_ACTION_CHUNK_STEPS], dtype=np.int64)
        finished = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        for idx in np.flatnonzero(finished):
            seed = seeds[idx // per_start]
            if len(results[seed]) < quota:
                results[seed].append({"success": bool(ep_success[idx]), "steps": int(ep_steps[idx])})
                if len(results[seed]) % 8 == 0:
                    done = results[seed]
                    print(
                        f"  seed {seed}: {len(done)}/{quota} rollouts, success {np.mean([r['success'] for r in done]):.3f}"
                    )
        if finished.any():
            dsrl_env.reset(env_mask=finished)
            ep_success[finished] = False
            ep_steps[finished] = 0
    dsrl_env.close()

    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= quota]
    summary = {
        "policy_path": args.policy_path,
        "task": args.task,
        "robot": args.robot,
        "noise_chunk_size": args.noise_chunk_size,
        "noise_mode": "dsrl_constant_per_chunk"
        if args.noise_chunk_size == 1
        else f"per_step_chunk{args.noise_chunk_size}",
        "exec_action_steps": n_action_steps,
        "rollouts_per_start": quota,
        "seed": args.seed,
        "start_rejection": start_rejection,
        "starts": {},
        "groups": {},
    }
    print("\n==== steerability results ====")
    for seed in seeds:
        n = len(results[seed])
        c = sum(r["success"] for r in results[seed])
        entry = {
            "n": n,
            "successes": c,
            "mean_success": c / n,
            "pass_at_k": {k: _pass_at_k(n, c, k) for k in ks},
            "mean_success_steps": float(np.mean([r["steps"] for r in results[seed] if r["success"]]))
            if c
            else None,
            "group": group_of[seed],
            "rollouts": results[seed],
        }
        summary["starts"][seed] = entry
        print(
            f"  seed {seed} [{group_of[seed]}]: mean {entry['mean_success']:.3f} ({c}/{n}) | "
            + " ".join(f"pass@{k}={v:.3f}" for k, v in entry["pass_at_k"].items())
        )
    for group in dict.fromkeys(g for _, g in starts):
        members = [summary["starts"][seed] for seed, g in starts if g == group]
        agg = {
            "n_starts": len(members),
            "mean_success": float(np.mean([m["mean_success"] for m in members])),
            "pass_at_k": {k: float(np.mean([m["pass_at_k"][k] for m in members])) for k in ks},
            "starts_with_any_success": int(sum(m["successes"] > 0 for m in members)),
        }
        summary["groups"][group] = agg
        print(
            f"  GROUP {group} ({len(members)} starts, {agg['starts_with_any_success']} with any success): "
            f"mean {agg['mean_success']:.3f} | "
            + " ".join(f"pass@{k}={v:.3f}" for k, v in agg["pass_at_k"].items())
        )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"steerability_nc{args.noise_chunk_size}.json"
    path.write_text(json.dumps(summary, indent=1))
    print(f"wrote {path}")

    if args.wandb_enable:
        try:
            import wandb

            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                dir=args.output_dir,
                config={**vars(args), "mode": "steerability_eval"},
            )
            log = {}
            for group, e in summary["groups"].items():
                log[f"steer/{group}/mean_success"] = e["mean_success"]
                for k, v in e["pass_at_k"].items():
                    log[f"steer/{group}/pass@{k}"] = v
            for seed, e in summary["starts"].items():
                log[f"steer/seed{seed}/mean_success"] = e["mean_success"]
                for k, v in e["pass_at_k"].items():
                    log[f"steer/seed{seed}/pass@{k}"] = v
            run.log(log)
            run.finish()
        except ImportError:
            pass


def _run_eval_only(args, eval_fn, dsrl_cfg: DSRLConfig, device: torch.device, env) -> None:
    """Score one saved noise actor and print/log the per-cell metrics, then exit."""
    noise_actor = _load_noise_actor(args.noise_actor_path, dsrl_cfg, device)

    wandb_run = None
    if args.wandb_enable:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                dir=args.output_dir,
                config={**vars(args), "mode": "eval_only"},
            )
            wandb_run = wandb
        except ImportError:
            print("wandb not installed — skipping WandB logging.")

    metrics = eval_fn(noise_actor, step=0)
    print("\n==== eval_only results ====")
    print(f"noise_actor : {args.noise_actor_path}")
    print(f"frozen VLA  : {args.policy_path}")
    print(f"episodes/cell: {args.eval_episodes}")
    for key in sorted(metrics):
        print(f"  {key} = {metrics[key]:.4f}")
    if wandb_run is not None:
        wandb_run.log({f"eval/{k}": v for k, v in metrics.items()}, step=0)
        wandb_run.finish()
    env.close()


# ── Main ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DSRL over frozen pi05 on RoboCasa")
    parser.add_argument(
        "--policy_path", type=str, required=True, help="Path to the RoboCasa-finetuned pi05 checkpoint"
    )
    parser.add_argument("--task", type=str, default="CoffeePressButton", help="RoboCasa task name")
    parser.add_argument(
        "--robot",
        type=str,
        default="PandaOmron",
        help="RoboCasa robot. 'PandaOmron' = parallel gripper (start here); "
        "'PandaDexLeapRHOmron' = dexterous hand.",
    )
    parser.add_argument("--cameras", type=str, default=DEFAULT_CAMERAS, help="Comma-separated camera names")
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Path to a composite-controller .json. Must match what the policy was trained on: "
        "None = the robot's default (OSC_POSE); a *_joint_pos.json for joint-position policies "
        "(e.g. the XArm6 ScrewLightbulb checkpoints).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Env control rate (Hz). MUST match the dataset's collection/eval rate the policy was "
        "trained at (e.g. 30 for the ScrewLightbulb checkpoints, 20 for CoffeePressButton).",
    )
    parser.add_argument(
        "--noise_actor_cameras",
        type=str,
        default=DEFAULT_NOISE_ACTOR_CAMERAS,
        help="Comma-separated (preprocessed) image keys the small SAC policy conditions on. "
        "Default is a single third-person view (matches the reference); add the wrist view "
        "'observation.images.robot0_eye_in_hand' to A/B two cameras.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Fallback language prompt when the env exposes no per-episode description",
    )
    parser.add_argument("--total_steps", type=int, default=500_000, help="Total environment steps")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs/dsrl_pi05_robocasa")
    parser.add_argument(
        "--collect_envs",
        type=int,
        default=1,
        help="Number of synchronous collection envs. Each sub-env i is a distinct fixed kitchen "
        "(seed i), so this also sets how many kitchens training sees (0..collect_envs-1). Note: "
        "SAC updates are per macro-step regardless of env count, so raising this lowers effective "
        "UTD per transition — scale --utd_ratio or --total_steps up to compensate.",
    )
    parser.add_argument(
        "--collect_scene_seeds",
        type=str,
        default=None,
        help="Comma-separated construction seed per collection sub-env. Default: sub-env i = "
        "seed i. The lamp bank holds only seed 1, so lamp runs pass --collect_scene_seeds 1.",
    )
    parser.add_argument(
        "--noise_chunk_size",
        type=int,
        default=1,
        help="Number of distinct noise steps the SAC policy emits, padded to pi05's chunk_size "
        "by repeating the last step. Default 1 matches the reference (a single noise vector "
        "steering the whole chunk); increase for finer-grained noise control.",
    )
    parser.add_argument(
        "--exec_action_steps",
        type=int,
        default=0,
        help="Actions executed per DSRL macro-step. 0 = use the policy's n_action_steps.",
    )
    parser.add_argument(
        "--embodiment_id",
        type=int,
        default=-1,
        help="Embodiment id to inject. <0 uses the checkpoint's default_embodiment_id.",
    )
    # DSRL config
    parser.add_argument("--dsrl_image_resize", type=int, default=64, help="Compact-encoder image size")
    parser.add_argument("--dsrl_hidden_dim", type=int, default=128, help="Actor/critic MLP hidden dim")
    parser.add_argument("--dsrl_num_layers", type=int, default=3, help="Actor/critic MLP layers")
    parser.add_argument("--dsrl_num_q", type=int, default=10, help="Number of Q-networks")
    parser.add_argument("--critic_reduction", type=str, default="mean", choices=["mean", "min"])
    parser.add_argument("--utd_ratio", type=int, default=20, help="SAC update-to-data ratio")
    parser.add_argument(
        "--min_buffer_size",
        type=int,
        default=1_000,
        help="Number of chunk transitions to collect (with Gaussian warmup noise) before starting SAC updates.",
    )
    parser.add_argument(
        "--target_entropy",
        type=float,
        default=None,
        help="SAC target entropy. Default (unset) uses SAC's automatic -noise_dim/2.",
    )
    parser.add_argument("--use_backup_entropy", action="store_true", help="Include entropy in the TD backup")
    parser.add_argument(
        "--primitive_discount",
        type=float,
        default=0.999,
        help="Per-primitive-step discount, compounded over the executed chunk.",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="goal",
        choices=list(REWARD_FNS) + list(MACRO_REWARD_FNS),
        help="'goal' (DSRL macro-step -1/0) or 'dense' (env reward summed over the chunk).",
    )
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=10_000)
    # WandB
    parser.add_argument("--wandb_enable", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="lerobot-dsrl")
    parser.add_argument("--wandb_name", type=str, default=None)
    # Eval
    parser.add_argument("--eval_freq", type=int, default=10_000, help="Eval every N env steps (0=off)")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument(
        "--eval_videos", type=int, default=3, help="Log WandB videos for the first N eval episodes (0=off)"
    )
    parser.add_argument(
        "--eval_seeds",
        type=str,
        default="0",
        help="Comma-separated env seeds to evaluate in; each seed is one fixed kitchen. Training "
        "uses kitchens 0..collect_envs-1, so reuse those for in-distribution eval and add higher, "
        "unseen seeds (e.g. 100,101) to measure held-out generalization. num_episodes runs per "
        "kitchen, so eval cost scales with the number of seeds.",
    )
    parser.add_argument(
        "--video_camera",
        type=str,
        default="robot0_agentview_left",
        help="Raw camera name to record for eval videos (must be one of --cameras).",
    )
    parser.add_argument(
        "--eval_scenes",
        type=str,
        default="",
        help="Comma-separated 'layout:style' pairs to pin the eval kitchen(s), e.g. '1:4,8:6'. "
        "Empty (default) keeps the old behaviour where --eval_seeds picks the kitchen. When set, "
        "--eval_seeds instead sweeps object placement *within* each pinned scene, which is what "
        "you want for a seen-vs-unseen table.",
    )
    parser.add_argument(
        "--noise_actor_path",
        type=str,
        default=None,
        help="Evaluate this saved noise-actor checkpoint (a train_dsrl 'step_*' dir) instead of "
        "training. Requires --eval_only.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Run one evaluation pass and exit; no SAC training. Use with --noise_actor_path.",
    )
    parser.add_argument(
        "--placement_bank",
        type=str,
        default=None,
        help="Placement-bank JSON (scripts/build_lamp_placement_bank.py). Training: collect env i "
        "(construction seed i, its fixed kitchen) moves the task object to that kitchen's next "
        "*training* placement on every reset. Eval: cells come from --eval_placement_sets instead "
        "of --eval_seeds/--eval_scenes. Also usable with --eval_only to score pre-bank checkpoints "
        "on the same held-out placements.",
    )
    parser.add_argument(
        "--eval_placement_sets",
        type=str,
        default="heldout,reference,train:2",
        help="With --placement_bank: comma list of heldout | reference | train[:N]. 'reference' is "
        "the original fixed placement every pre-bank run trained on (held out of the bank). "
        "One cell per (kitchen, entry); --eval_episodes episodes each.",
    )
    parser.add_argument(
        "--kitchen_bank",
        type=str,
        default=None,
        help="Kitchen-bank JSON (scripts/build_coffee_kitchen_bank.py). Training: each collect slot "
        "is a pool of the bank's training kitchens and switches kitchen on every reset. Eval: "
        "cells come from --eval_kitchen_sets. Also usable with --eval_only to score fixed-kitchen "
        "checkpoints on the held-out kitchens.",
    )
    parser.add_argument(
        "--eval_kitchen_sets",
        type=str,
        default="heldout,reference",
        help="With --kitchen_bank: comma list of heldout | reference | train[:N]. 'reference' = "
        "construction seeds 0,1, the kitchens of the fixed-kitchen runs.",
    )
    parser.add_argument(
        "--reject_start_rot_deg",
        type=float,
        default=0.0,
        help="If > 0, re-reset (training AND eval envs) whenever the end effector starts more than "
        "this many degrees, or --reject_start_pos_cm, from the pose that kitchen settles into with "
        "RoboCasa's arm reset noise disabled. Removes starts where the noise jams the LEAP fingers "
        "into a wall cabinet and rotates the hand (up to ~68 deg in CoffeePressButton kitchens). "
        "0 = off (historical behaviour).",
    )
    parser.add_argument("--reject_start_pos_cm", type=float, default=3.0)
    parser.add_argument(
        "--steer_eval",
        action="store_true",
        help="Best-of-K steerability of the frozen policy: random-noise rollouts on the --eval_seeds starts, "
        "no actor, no training. --noise_chunk_size 1 = DSRL's noise space, chunk_size = vanilla pi05.",
    )
    parser.add_argument("--steer_rollouts_per_start", type=int, default=64)
    parser.add_argument("--steer_parallel_per_start", type=int, default=8)
    parser.add_argument("--reject_start_max_retries", type=int, default=10)
    args = parser.parse_args()
    start_rejection = None
    if args.reject_start_rot_deg > 0:
        start_rejection = {
            "max_rot_deg": args.reject_start_rot_deg,
            "max_pos_m": args.reject_start_pos_cm / 100.0,
            "max_retries": args.reject_start_max_retries,
        }
    if sum(bool(x) for x in (args.placement_bank, args.kitchen_bank, args.eval_scenes)) > 1:
        raise ValueError("--placement_bank, --kitchen_bank and --eval_scenes are mutually exclusive")
    if args.steer_eval and (args.eval_only or args.placement_bank or args.eval_scenes):
        raise ValueError("--steer_eval uses construction-seed starts (--eval_seeds, or a --kitchen_bank)")
    if args.eval_only and not args.noise_actor_path:
        raise ValueError("--eval_only requires --noise_actor_path")
    if args.dsrl_num_layers < 1:
        raise ValueError("--dsrl_num_layers must be >= 1")

    collect_scene_seeds = (
        [int(x) for x in args.collect_scene_seeds.split(",") if x.strip()]
        if args.collect_scene_seeds
        else list(range(args.collect_envs))
    )
    if len(collect_scene_seeds) != args.collect_envs:
        raise ValueError(
            f"--collect_scene_seeds has {len(collect_scene_seeds)} entries but --collect_envs is "
            f"{args.collect_envs}; give one construction seed per collection sub-env."
        )

    device = torch.device(args.device)

    # ── Load frozen VLA (pi05 / SmolVLA, auto-detected) + its processors ──
    frozen_policy = load_frozen_policy(args.policy_path)
    frozen_policy.eval()
    frozen_policy.to(device)

    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=frozen_policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    full_chunk_size = frozen_policy.config.chunk_size
    max_action_dim = frozen_policy.config.max_action_dim
    n_action_steps = (
        args.exec_action_steps if args.exec_action_steps > 0 else frozen_policy.config.n_action_steps
    )
    if n_action_steps > full_chunk_size:
        raise ValueError(f"exec_action_steps ({n_action_steps}) cannot exceed chunk_size ({full_chunk_size})")
    noise_chunk_size = args.noise_chunk_size
    if not (1 <= noise_chunk_size <= full_chunk_size):
        raise ValueError(f"--noise_chunk_size must be in [1, {full_chunk_size}]; got {noise_chunk_size}")
    noise_dim = noise_chunk_size * max_action_dim

    embodiment_id = (
        args.embodiment_id if args.embodiment_id >= 0 else frozen_policy.config.default_embodiment_id
    )

    print(
        f"{frozen_policy.config.type}: chunk_size={full_chunk_size}, max_action_dim={max_action_dim}, "
        f"n_action_steps(exec)={n_action_steps}, noise_chunk={noise_chunk_size}, noise_dim={noise_dim}, "
        f"embodiment_id={embodiment_id}"
    )

    # ── Environment ──────────────────────────────────────────────────────
    env_cfg = RoboCasaEnv(
        task=args.task,
        robot=args.robot,
        camera_name=args.cameras,
        controller=args.controller,
        fps=args.fps,
    )
    placement_bank = None
    if args.placement_bank:
        from lerobot.envs.robocasa_placement_bank import load_placement_bank

        placement_bank = load_placement_bank(args.placement_bank)
        if placement_bank["task"] != args.task or placement_bank["robot"] != args.robot:
            raise ValueError(
                f"Placement bank is for {placement_bank['task']}/{placement_bank['robot']}, "
                f"not {args.task}/{args.robot}"
            )
        missing = [s for s in collect_scene_seeds if s not in placement_bank["_scenes_by_seed"]]
        if missing:
            raise ValueError(
                f"Placement bank has no scenes for construction seeds {missing}; it holds "
                f"{sorted(placement_bank['_scenes_by_seed'])}. Set --collect_scene_seeds to seeds "
                "the bank covers."
            )
    kitchen_bank = None
    if args.kitchen_bank:
        import json

        with open(os.path.expanduser(args.kitchen_bank)) as f:
            kitchen_bank = json.load(f)
        if kitchen_bank.get("bank_type") != "kitchens":
            raise ValueError(
                f"{args.kitchen_bank} is not a kitchen bank (bank_type={kitchen_bank.get('bank_type')})"
            )
        if kitchen_bank["task"] != args.task or kitchen_bank["robot"] != args.robot:
            raise ValueError(
                f"Kitchen bank is for {kitchen_bank['task']}/{kitchen_bank['robot']}, not {args.task}/{args.robot}"
            )
        if len(kitchen_bank["train"]) < args.collect_envs:
            raise ValueError(
                f"Kitchen bank has fewer training kitchens than --collect_envs {args.collect_envs}"
            )
    env = (
        None
        if args.steer_eval
        else _make_collect_env(
            env_cfg,
            args.collect_envs,
            placement_bank=placement_bank,
            seed=args.seed,
            # --eval_only never steps the collect env (each eval cell builds its own), so do not pay
            # for a resident pool of every training kitchen just to close it again.
            kitchen_bank=None if args.eval_only else kitchen_bank,
            start_rejection=None if args.eval_only else start_rejection,
            scene_seeds=collect_scene_seeds,
        )
    )

    eval_seeds = tuple(int(s) for s in args.eval_seeds.split(",") if s.strip() != "")
    train_seeds = collect_scene_seeds
    if kitchen_bank is not None:
        pass  # pools are printed by _make_collect_env
    elif placement_bank is not None:
        for s in train_seeds:
            n_train = len(placement_bank["_scenes_by_seed"][s]["train"])
            print(f"Training kitchen seed {s}: object cycles through {n_train} banked placements per pass")
    else:
        held_out = [s for s in eval_seeds if s not in train_seeds]
        print(
            f"Training kitchens (seeds): {train_seeds} | eval kitchens: {list(eval_seeds)}"
            + (f" | held-out (unseen in training): {held_out}" if held_out else "")
        )

    # ── Adapters ─────────────────────────────────────────────────────────
    noise_actor_cameras = [c.strip() for c in args.noise_actor_cameras.split(",") if c.strip()]
    print(f"Noise actor conditions on {len(noise_actor_cameras)} camera(s): {noise_actor_cameras}")
    # Same size the compact encoder resizes to, so the buffer never stores full-res frames.
    obs_resize = args.dsrl_image_resize if args.dsrl_image_resize > 0 else None
    prepare_obs_fn = make_prepare_obs_fn(device, noise_actor_cameras, resize_size=obs_resize)

    def make_frozen_obs_for_env(target_env):
        return make_prepare_frozen_obs_fn(target_env, preprocessor, device, embodiment_id, args.prompt)

    prepare_frozen_obs_fn = make_frozen_obs_for_env(env)
    generate_action_chunk_fn = make_generate_action_chunk_fn(
        frozen_policy,
        postprocessor,
        noise_chunk_size,
        full_chunk_size,
        max_action_dim,
        n_action_steps,
        device,
    )
    reward_fn = REWARD_FNS.get(args.reward_mode)
    if args.steer_eval:
        _run_steerability_eval(
            args,
            env_cfg,
            make_frozen_obs_for_env,
            prepare_obs_fn,
            generate_action_chunk_fn,
            reward_fn,
            MACRO_REWARD_FNS.get(args.reward_mode),
            device,
            noise_dim,
            n_action_steps,
            starts=(
                [(k["scene_seed"], "train") for k in kitchen_bank["train"]]
                + [(k["scene_seed"], "heldout") for k in kitchen_bank["heldout"]]
                if kitchen_bank is not None
                else [(int(x), "fixed") for x in args.eval_seeds.split(",") if x.strip()]
            ),
            start_rejection=start_rejection,
        )
        return
    macro_reward_fn = MACRO_REWARD_FNS.get(args.reward_mode)

    discount = args.primitive_discount**n_action_steps
    print(f"Discount: {args.primitive_discount} ** {n_action_steps} = {discount:.4f} per macro step")

    # ── WandB / eval ─────────────────────────────────────────────────────
    wandb_kwargs = None
    if args.wandb_enable:
        wandb_kwargs = {
            "enable": True,
            "project": args.wandb_project,
            "name": args.wandb_name,
            "dir": args.output_dir,
        }

    eval_scenes = _parse_eval_scenes(args.eval_scenes)
    if eval_scenes != [None]:
        print(
            f"Eval kitchens PINNED to {[f'L{layout}S{style}' for layout, style in eval_scenes]}; "
            f"--eval_seeds {list(eval_seeds)} now sweeps object placement within each."
        )

    placement_cells = None
    if placement_bank is not None:
        # Eval kitchens = the bank's kitchens (normally the same seeds the collect envs use).
        bank_seeds = sorted(placement_bank["_scenes_by_seed"])
        placement_cells = _placement_eval_cells(placement_bank, args.eval_placement_sets, bank_seeds)
        print(
            f"Eval: {len(placement_cells)} placement cells ({args.eval_placement_sets}) x "
            f"{args.eval_episodes} episodes: {[c['label'] for c in placement_cells]}"
        )
    elif kitchen_bank is not None:
        # Kitchen cells reuse the same generic cell path: seed = kitchen, no scene pin, no entry.
        placement_cells = _kitchen_eval_cells(kitchen_bank, args.eval_kitchen_sets)
        print(
            f"Eval: {len(placement_cells)} kitchen cells ({args.eval_kitchen_sets}) x "
            f"{args.eval_episodes} episodes: {[c['label'] for c in placement_cells]}"
        )

    eval_fn = None
    if args.eval_freq > 0 or args.eval_only:
        eval_fn = make_robocasa_eval_fn(
            env_cfg=env_cfg,
            make_frozen_obs_for_env=make_frozen_obs_for_env,
            prepare_obs_fn=prepare_obs_fn,
            generate_action_chunk_fn=generate_action_chunk_fn,
            reward_fn=reward_fn,
            macro_reward_fn=macro_reward_fn,
            device=device,
            noise_dim=noise_dim,
            n_obs_steps=1,
            num_episodes=args.eval_episodes,
            eval_seeds=eval_seeds,
            eval_scenes=eval_scenes,
            num_videos=args.eval_videos,
            video_camera=args.video_camera,
            video_dir=Path(args.output_dir) / "eval_videos",
            placement_bank=placement_bank,
            placement_cells=placement_cells,
            start_rejection=start_rejection,
        )

    dsrl_cfg = DSRLConfig(
        image_resize_size=args.dsrl_image_resize if args.dsrl_image_resize > 0 else None,
        use_compact_encoder=args.dsrl_image_resize > 0,
        hidden_dims=tuple([args.dsrl_hidden_dim] * args.dsrl_num_layers),
        num_q_heads=args.dsrl_num_q,
        critic_reduction=args.critic_reduction,
        min_buffer_size=args.min_buffer_size,
        utd_ratio=args.utd_ratio,
        discount=discount,
        target_entropy=args.target_entropy,
        use_backup_entropy=args.use_backup_entropy,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
    )

    if args.eval_only:
        _run_eval_only(args, eval_fn, dsrl_cfg, device, env)
        return

    train_dsrl(
        env=env,
        noise_dim=noise_dim,
        prepare_obs_fn=prepare_obs_fn,
        generate_action_chunk_fn=generate_action_chunk_fn,
        prepare_frozen_obs_fn=prepare_frozen_obs_fn,
        reward_fn=reward_fn,
        macro_reward_fn=macro_reward_fn,
        total_steps=args.total_steps,
        device=str(device),
        n_obs_steps=1,
        dsrl_config=dsrl_cfg,
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_kwargs=wandb_kwargs,
        eval_fn=eval_fn,
    )

    env.close()


if __name__ == "__main__":
    main()
