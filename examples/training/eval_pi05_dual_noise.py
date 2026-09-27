#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Dual-noise SFT eval over a checkpoint ladder, on the DSRL eval cells.

Every checkpoint is scored twice, on the same start conditions its DSRL runs are scored on:

  iid          pi05's own prior: a fresh N(0,1) sample per chunk step. What SFT eval measures.
  duplicated   ONE N(0,1) vector repeated across the chunk. What DSRL steers.

Cells, cameras and start-pose rejection default to what dsrl_mila.sbatch / dsrl_drac.sbatch
pass, per task:

  lamp     pair bank, cells train,heldout (s1_start_0, s1_start_1, s1_mid)
  coffee   kitchen bank, cells heldout,reference, start-pose rejection 10 deg / 3 cm

One sub-env per cell, so episode i belongs to cell i % n_cells.

Usage:
    python examples/training/eval_pi05_dual_noise.py --run-dir <run> --task lamp \
        --steps 2000 4000 8000 16000 --episodes-per-cell 20

Results are merged into <run-dir>/dual_noise_dsrl_cells.json.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import sys
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import RoboCasaEnv
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.robocasa_env import RoboCasaEnv as RoboCasaGymEnv
from lerobot.envs.robocasa_placement_bank import (
    PlacementBankWrapper,
    StartPoseRejectionWrapper,
    kitchen_eval_cells,
    load_placement_bank,
    placement_eval_cells,
)
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.constants import CHECKPOINTS_DIR, PRETRAINED_MODEL_DIR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pi05_casa_experiments import TASKS, resolve_controller  # noqa: E402

NOISE_MODES = ("iid", "duplicated")
RESULTS_FILENAME = "dual_noise_dsrl_cells.json"

# The frozen policy's view inside DSRL. One external camera also makes the checkpoint's
# random-external-camera augmentation a no-op.
DSRL_CAMERAS = "robot0_agentview_left,robot0_eye_in_hand"
_BANKS = Path(os.environ.get("LEROBOT_PLACEMENT_BANKS", "~/scratch/lerobot/placement_banks")).expanduser()
# Mirrors the per-task invariants in dsrl_mila.sbatch / dsrl_drac.sbatch.
DSRL_EVAL = {
    "lamp": {
        "placement_bank": _BANKS / "screwlightbulb_xarm6_seed1_pair" / "bank.json",
        "eval_sets": "train,heldout",
        "reject_start_rot_deg": 0.0,
    },
    "coffee": {
        "kitchen_bank": _BANKS / "coffeepressbutton_pandadex_kitchens" / "bank.json",
        "eval_sets": "heldout,reference",
        "reject_start_rot_deg": 10.0,
    },
}

_MISSING = object()


def merge_result(out_path: Path, step: int, row: dict) -> dict:
    """Write one rung into the shared results file under a lock, so jobs scoring different rungs
    of the same run can run concurrently."""
    with open(out_path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = json.loads(out_path.read_text()) if out_path.exists() else {}
        results[str(step)] = row
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(results, indent=2, sort_keys=True))
        os.replace(tmp, out_path)
    return results


def _batch_size(batch: dict) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            return value.shape[0]
    raise ValueError("Could not infer batch size from the observation batch.")


@contextmanager
def duplicated_noise(policy):
    """Temporarily make `policy` denoise every chunk from ONE repeated noise vector.

    The action-queue logic mirrors `PI05Policy.select_action`; the only difference from
    the normal path is the `noise=` argument. Restores the original attributes on exit.
    """
    config = policy.config
    chunk_size, n_action_steps = config.chunk_size, config.n_action_steps
    max_action_dim = config.max_action_dim
    device = next(policy.parameters()).device
    queue: deque = deque(maxlen=n_action_steps)

    @torch.no_grad()
    def select_action(batch: dict, **kwargs) -> torch.Tensor:
        if len(queue) == 0:
            bsize = _batch_size(batch)
            # Same distribution and dtype as PI05FlowMatching.sample_noise, but one draw
            # per chunk instead of one per chunk step.
            noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=(bsize, 1, max_action_dim),
                dtype=torch.float32,
                device=device,
            ).expand(bsize, chunk_size, max_action_dim)
            actions = policy.predict_action_chunk(batch, noise=noise.contiguous())
            queue.extend(actions[:, :n_action_steps].transpose(0, 1))
        return queue.popleft()

    original_reset = policy.reset

    def reset() -> None:
        queue.clear()
        original_reset()

    # Save whatever was on the INSTANCE (usually nothing; the methods live on the class)
    # so the restore puts things back exactly as they were.
    saved = {name: policy.__dict__.get(name, _MISSING) for name in ("select_action", "reset")}
    policy.select_action = select_action
    policy.reset = reset
    try:
        yield policy
    finally:
        for name, value in saved.items():
            if value is _MISSING:
                policy.__dict__.pop(name, None)
            else:
                policy.__dict__[name] = value


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    """`outputs/train/<run>/checkpoints/<zero-padded step>/pretrained_model`.

    The zero-padding width depends on the run's total steps, so resolve by value
    rather than by reconstructing the name.
    """
    checkpoints = run_dir / CHECKPOINTS_DIR
    numbered = [d for d in checkpoints.iterdir() if d.is_dir() and d.name.isdigit()]
    matches = [d for d in numbered if int(d.name) == step]
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint for step {step} in {checkpoints}. "
            f"Available: {sorted(d.name for d in numbered)}"
        )
    return matches[0] / PRETRAINED_MODEL_DIR


def available_steps(run_dir: Path) -> list[int]:
    """Every checkpoint step present, ascending. save_freq is already the ladder spacing."""
    checkpoints = run_dir / CHECKPOINTS_DIR
    if not checkpoints.is_dir():
        raise FileNotFoundError(f"No {CHECKPOINTS_DIR}/ under {run_dir}")
    return sorted(int(d.name) for d in checkpoints.iterdir() if d.is_dir() and d.name.isdigit())


def build_env_cfg(task_spec, reference_checkpoint: Path, cameras: str) -> RoboCasaEnv:
    return RoboCasaEnv(
        task=task_spec.robocasa_task,
        robot=task_spec.robot,
        # Per-task, not per-robot: coffee is OSC_POSE deltas and lamp is absolute joint
        # targets, so deriving this from the robot silently scrambles the action vector.
        controller=resolve_controller(task_spec.controller_filename),
        fps=_fps_from_train_config(reference_checkpoint),
        camera_name=cameras,
    )


def build_cells(args, env_cfg: RoboCasaEnv) -> tuple[list[dict], dict]:
    if args.placement_bank:
        bank = load_placement_bank(args.placement_bank)
        cells = placement_eval_cells(bank, args.eval_sets, sorted(bank["_scenes_by_seed"]))
    else:
        bank = json.loads(Path(args.kitchen_bank).expanduser().read_text())
        cells = kitchen_eval_cells(bank, args.eval_sets)
    if bank["task"] != env_cfg.task or bank["robot"] != env_cfg.robot:
        raise ValueError(f"Bank is for {bank['task']}/{bank['robot']}, not {env_cfg.task}/{env_cfg.robot}")
    return cells, bank


def make_cell_vec_env(cells: list[dict], bank: dict, env_cfg: RoboCasaEnv, rejection: dict | None):
    """One sub-env per cell, built exactly as the DSRL eval builds them."""

    def factory(cell: dict):
        def _make():
            env = RoboCasaGymEnv(
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
                seed=cell["seed"],
            )
            if rejection is not None:
                env = StartPoseRejectionWrapper(env, **rejection)
            if cell["entry"] is not None:
                env = PlacementBankWrapper(env, bank, cell["seed"], [cell["entry"]], shuffle=False)
            return env

        return _make

    return gym.vector.SyncVectorEnv([factory(c) for c in cells])


def evaluate_step(
    pretrained_dir: Path,
    env_cfg: RoboCasaEnv,
    vec,
    cells: list[dict],
    episodes_per_cell: int,
    seed: int,
    videos_dir: Path | None,
    n_videos: int,
) -> dict[str, dict[str, float]]:
    """Score ONE checkpoint under both noise modes on the cell envs, reused across checkpoints."""
    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_dir)
    policy_cfg.pretrained_path = pretrained_dir

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_dir,
        preprocessor_overrides={"device_processor": {"device": str(policy_cfg.device)}},
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    labels = [c["label"] for c in cells]
    results: dict[str, dict[str, float]] = {}
    try:
        for mode in NOISE_MODES:
            # Same seed per mode, so the modes differ only in the noise parameterisation.
            set_seed(seed)
            noise_ctx = nullcontext(policy) if mode == "iid" else duplicated_noise(policy)
            with noise_ctx as evaluated, torch.no_grad():
                info = eval_policy(
                    env=vec,
                    policy=evaluated,
                    env_preprocessor=env_pre,
                    env_postprocessor=env_post,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=len(cells) * episodes_per_cell,
                    max_episodes_rendered=n_videos,
                    videos_dir=(videos_dir / mode) if videos_dir else None,
                    start_seed=seed,
                )
            succ = np.array([float(ep["success"]) for ep in info["per_episode"]]).reshape(
                episodes_per_cell, len(cells)
            )
            row = {
                "pc_success": 100.0 * float(succ.mean()),
                "n": int(succ.size),
                "eval_s": info["aggregated"]["eval_s"],
            }
            for i, label in enumerate(labels):
                row[f"pc_success_{label}"] = 100.0 * float(succ[:, i].mean())
            for group in dict.fromkeys(c["group"] for c in cells):
                idx = [i for i, c in enumerate(cells) if c["group"] == group]
                row[f"pc_success_{group}"] = 100.0 * float(succ[:, idx].mean())
                row[f"n_{group}"] = int(succ[:, idx].size)
            results[mode] = row
            logging.info("%s | %s -> %.1f%% success", pretrained_dir.parent.name, mode, row["pc_success"])
    finally:
        # A 4B policy per rung adds up; drop it before the next checkpoint loads.
        del policy, preprocessor, postprocessor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def _fps_from_train_config(pretrained_dir: Path) -> int:
    """Reuse the control_freq the run was trained/evaluated with."""
    train_config = pretrained_dir / "train_config.json"
    env = json.loads(train_config.read_text()).get("env")
    if isinstance(env, list):
        env = env[0] if env else None
    if isinstance(env, dict) and env.get("fps"):
        return int(env["fps"])
    raise ValueError(f"Could not read env.fps from {train_config}; refusing to guess the eval control rate.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="A training run's output_dir.")
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument(
        "--steps", type=int, nargs="+", default=None, help="Checkpoint steps. Default: every checkpoint."
    )
    parser.add_argument("--episodes-per-cell", type=int, default=20)
    bank = parser.add_mutually_exclusive_group()
    bank.add_argument("--placement-bank", type=Path, default=None, help="Default: the task's DSRL bank.")
    bank.add_argument("--kitchen-bank", type=Path, default=None, help="Default: the task's DSRL bank.")
    parser.add_argument("--eval-sets", default=None, help="Cell groups. Default: the task's DSRL eval sets.")
    parser.add_argument("--cameras", default=DSRL_CAMERAS)
    parser.add_argument("--reject-start-rot-deg", type=float, default=None, help="Default: per task.")
    parser.add_argument("--reject-start-pos-cm", type=float, default=3.0)
    parser.add_argument("--reject-start-max-retries", type=int, default=10)
    parser.add_argument("--n-videos", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--wandb-project", default="lerobot-dex-eval")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    defaults = DSRL_EVAL[args.task]
    if args.placement_bank is None and args.kitchen_bank is None:
        args.placement_bank = defaults.get("placement_bank")
        args.kitchen_bank = defaults.get("kitchen_bank")
    if args.eval_sets is None:
        args.eval_sets = defaults["eval_sets"]
    if args.reject_start_rot_deg is None:
        args.reject_start_rot_deg = defaults["reject_start_rot_deg"]
    return args


def init_wandb(args, task_spec, steps: list[int], labels: list[str]):
    """One wandb run per TRAINING run, stepped by checkpoint, or None when disabled."""
    if args.no_wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"dualnoise_dsrlcells_{args.run_dir.name}",
        job_type="dual_noise_eval",
        config={
            "run_dir": str(args.run_dir),
            "run_name": args.run_dir.name,
            "task": args.task,
            "robot": task_spec.robot,
            "robocasa_task": task_spec.robocasa_task,
            "bank": str(args.placement_bank or args.kitchen_bank),
            "eval_sets": args.eval_sets,
            "cells": labels,
            "cameras": args.cameras,
            "reject_start_rot_deg": args.reject_start_rot_deg,
            "episodes_per_cell": args.episodes_per_cell,
            "seed": args.seed,
            "steps": steps,
            "noise_modes": list(NOISE_MODES),
        },
        dir=str(args.run_dir),
    )


def main() -> None:
    init_logging()
    args = parse_args()
    register_third_party_plugins()
    task_spec = TASKS[args.task]

    # Ascending: wandb silently drops any step lower than the last one logged in a run.
    steps = sorted(args.steps) if args.steps is not None else available_steps(args.run_dir)
    env_cfg = build_env_cfg(task_spec, checkpoint_dir(args.run_dir, steps[0]), args.cameras)
    cells, bank = build_cells(args, env_cfg)
    labels = [c["label"] for c in cells]
    rejection = (
        {
            "max_rot_deg": args.reject_start_rot_deg,
            "max_pos_m": args.reject_start_pos_cm / 100.0,
            "max_retries": args.reject_start_max_retries,
        }
        if args.reject_start_rot_deg > 0
        else None
    )
    logging.info(
        "Scoring %d checkpoints %s on %d cells %s x %d episodes | cameras %s | rejection %s",
        len(steps), steps, len(cells), labels, args.episodes_per_cell, args.cameras, rejection,
    )

    out_path = args.run_dir / RESULTS_FILENAME
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    run = init_wandb(args, task_spec, steps, labels)

    # Built once for the whole ladder: identical for every checkpoint of a run.
    vec = make_cell_vec_env(cells, bank, env_cfg, rejection)
    try:
        for step in steps:
            pretrained_dir = checkpoint_dir(args.run_dir, step)
            videos_dir = None if args.no_videos else args.run_dir / "eval_dual_noise_dsrl_cells" / f"step_{step:07d}"
            row = evaluate_step(
                pretrained_dir=pretrained_dir,
                env_cfg=env_cfg,
                vec=vec,
                cells=cells,
                episodes_per_cell=args.episodes_per_cell,
                seed=args.seed,
                videos_dir=videos_dir,
                n_videos=0 if args.no_videos else args.n_videos,
            )
            results = merge_result(out_path, step, row)
            logging.info("Wrote %s", out_path)

            if run is not None:
                row = results[str(step)]
                payload = {f"{mode}/{k}": v for mode, m in row.items() for k, v in m.items()}
                # How much success is lost when ONE noise vector drives the chunk -- what DSRL steers.
                payload["gap/pc_success"] = row["iid"]["pc_success"] - row["duplicated"]["pc_success"]
                run.log(payload, step=step, commit=True)
    finally:
        vec.close()

    if run is not None:
        run.finish()

    groups = list(dict.fromkeys(c["group"] for c in cells))
    header = f"{'step':>8}  {'mode':>10}  {'all':>6}  " + "  ".join(f"{g:>9}" for g in groups)
    print("\n" + header)
    for step in sorted(results, key=int):
        for mode in NOISE_MODES:
            row = results[step][mode]
            cols = "  ".join(f"{row[f'pc_success_{g}']:>8.1f}%" for g in groups)
            print(f"{step:>8}  {mode:>10}  {row['pc_success']:>5.1f}%  {cols}")


if __name__ == "__main__":
    main()
