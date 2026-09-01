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


def _make_collect_env(env_cfg: RoboCasaEnv, num_envs: int):
    if num_envs < 1:
        raise ValueError("--collect_envs must be >= 1")
    envs = make_env(env_cfg, n_envs=num_envs, use_async_envs=False)
    return envs[env_cfg.type][0]


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


def make_prepare_obs_fn(device: torch.device, noise_actor_cameras: list[str]):
    """Raw (batched) RoboCasa obs → compact noise-actor observation (image(s) + state).

    The small SAC policy conditions on the resized ``noise_actor_cameras`` and the robot
    state; the reference DSRL implementation uses a single third-person view. The heavy,
    normalized, tokenized observation the VLA needs is built separately by
    :func:`make_prepare_frozen_obs_fn`. Multiple cameras are stacked along the channel axis
    by the compact encoder.
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
            out[cam] = policy_obs[cam].to(device)
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


def _make_single_robocasa_env(env_cfg: RoboCasaEnv, seed: int):
    """Build the raw (non-vectorized) RoboCasa gym env, so eval can wrap it for frame capture.

    Constructed with the same fields ``make_env`` uses, so a given ``seed`` yields the same
    fixed kitchen (layout/style) as the training env with that seed. RoboCasa fixes the scene
    per env instance at construction; ``reset()`` only re-randomizes object/robot poses.
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
    num_videos: int = 3,
    video_camera: str = "robot0_agentview_left",
    video_dir: str | Path | None = None,
):
    """Roll out the noise actor greedily in each eval kitchen; report per-kitchen + aggregate.

    Each seed in ``eval_seeds`` is one fixed kitchen. Reuse training seeds
    (``0..collect_envs-1``) for in-distribution eval, and add higher, unseen seeds for
    held-out generalization. ``num_episodes`` episodes are run *per kitchen* (so total eval
    cost scales with ``len(eval_seeds) * num_episodes``), and one video is logged per kitchen.
    """
    video_dir = Path(video_dir) if video_dir is not None else Path("outputs/dsrl_pi05_robocasa/eval_videos")
    video_fps = env_cfg.fps
    eval_seeds = tuple(eval_seeds)

    def _rollout_kitchen(noise_actor: NoiseActorPolicy, seed: int, record_video: bool):
        frame_capture = _RoboCasaFrameCapture(_make_single_robocasa_env(env_cfg, seed=seed), video_camera)
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

        for i, seed in enumerate(eval_seeds):
            successes, ret, steps, frames = _rollout_kitchen(noise_actor, seed, record_video=i < num_videos)
            total_successes += successes
            total_return += ret
            total_steps += steps
            metrics[f"success_rate_seed{seed}"] = successes / max(num_episodes, 1)
            if frames:
                labeled_frames.append((f"seed{seed}", frames))

        noise_actor.train()

        if labeled_frames:
            _log_eval_videos(labeled_frames, step, fps=video_fps, video_dir=video_dir)

        total_episodes = max(num_episodes * len(eval_seeds), 1)
        metrics["success_rate"] = total_successes / total_episodes
        metrics["avg_return"] = total_return / total_episodes
        metrics["avg_length"] = total_steps / total_episodes
        return metrics

    return eval_fn


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
    args = parser.parse_args()
    if args.dsrl_num_layers < 1:
        raise ValueError("--dsrl_num_layers must be >= 1")

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
    env = _make_collect_env(env_cfg, args.collect_envs)

    eval_seeds = tuple(int(s) for s in args.eval_seeds.split(",") if s.strip() != "")
    train_seeds = list(range(args.collect_envs))
    held_out = [s for s in eval_seeds if s not in train_seeds]
    print(
        f"Training kitchens (seeds): {train_seeds} | eval kitchens: {list(eval_seeds)}"
        + (f" | held-out (unseen in training): {held_out}" if held_out else "")
    )

    # ── Adapters ─────────────────────────────────────────────────────────
    noise_actor_cameras = [c.strip() for c in args.noise_actor_cameras.split(",") if c.strip()]
    print(f"Noise actor conditions on {len(noise_actor_cameras)} camera(s): {noise_actor_cameras}")
    prepare_obs_fn = make_prepare_obs_fn(device, noise_actor_cameras)

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

    eval_fn = None
    if args.eval_freq > 0:
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
            num_videos=args.eval_videos,
            video_camera=args.video_camera,
            video_dir=Path(args.output_dir) / "eval_videos",
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
