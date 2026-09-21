#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Dual-noise SFT eval over a checkpoint ladder: the checkpoint-selection metric for
the pi05 SFT-arm matrix (see train_pi05_casa_experiments.py).

Every checkpoint is scored twice in the same RoboCasa suite:

  iid          pi05's own flow-matching prior -- one fresh N(0,1) sample per
               chunk step, shape (B, chunk_size, max_action_dim). This is what
               `sample_actions` does when `noise=None` and what the in-loop
               training eval measures.

  duplicated   ONE N(0,1) vector per chunk, repeated across all chunk steps.
               This is the action distribution DSRL actually steers: the noise
               actor emits `noise_chunk_size` (default 1) vectors which
               `dsrl_pi05_robocasa_example.py` pads out by repeating. A policy
               that is only good under iid noise has a degenerate steering
               surface, and its DSRL numbers will not reflect its SFT numbers.

Why not in the training loop: `lerobot_train` calls `eval_policy_all` once per
`eval_freq`, and pi05's `select_action` has no noise seam (only
`predict_action_chunk(batch, noise=...)` does), so in-loop dual-noise needs a
patch to shared training code AND doubles an eval cost that already dominates
these runs. Post-hoc you score only the rungs you care about, in parallel, on
whatever GPU is free.

Usage (inside an allocation):
    python examples/training/eval_pi05_dual_noise.py \
        --run-dir /path/to/outputs/train/2026-09-18/12-00-00_sft_arms_coffee_arm1_full_ft_lrx1_seed42 \
        --task coffee --steps 8000 12000 16000 20000 --n-episodes 20

Results are appended to <run-dir>/dual_noise_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import RoboCasaEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.constants import CHECKPOINTS_DIR, PRETRAINED_MODEL_DIR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pi05_casa_experiments import TASKS, resolve_controller  # noqa: E402

NOISE_MODES = ("iid", "duplicated")

_MISSING = object()


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


def build_env_cfg(task_spec, reference_checkpoint: Path) -> RoboCasaEnv:
    return RoboCasaEnv(
        task=task_spec.robocasa_task,
        robot=task_spec.robot,
        # Per-task, not per-robot: coffee is OSC_POSE deltas and lamp is absolute joint
        # targets, so deriving this from the robot silently scrambles the action vector.
        controller=resolve_controller(task_spec.controller_filename),
        fps=_fps_from_train_config(reference_checkpoint),
        camera_name=task_spec.camera_name,
        scene_seeds=list(task_spec.eval_scene_seeds) if task_spec.eval_scene_seeds else None,
        placement_bank=task_spec.eval_placement_bank,
        placement_ids=list(task_spec.eval_placement_ids) if task_spec.eval_placement_ids else None,
    )


def evaluate_step(
    pretrained_dir: Path,
    task_spec,
    env_cfg: RoboCasaEnv,
    envs,
    n_episodes: int,
    seed: int,
    videos_dir: Path | None,
    n_videos: int,
) -> dict[str, dict[str, float]]:
    """Score ONE checkpoint under both noise modes, reusing already-built envs.

    The envs are created once per run and passed in: nothing about them depends on the
    checkpoint, rebuilding them costs ~40 s each time, and repeatedly spawning/closing
    async worker processes is a way to leak them across a 10-rung ladder.
    """
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
    env_processors = dict.fromkeys(envs, (env_pre, env_post))

    results: dict[str, dict[str, float]] = {}
    try:
        for mode in NOISE_MODES:
            # Identical seed per mode so both share episode/scene sampling and the only
            # difference is the noise parameterisation.
            set_seed(seed)
            noise_ctx = nullcontext(policy) if mode == "iid" else duplicated_noise(policy)
            with noise_ctx as evaluated, torch.no_grad():
                info = eval_policy_all(
                    envs=envs,
                    policy=evaluated,
                    env_processors=env_processors,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=n_episodes,
                    videos_dir=(videos_dir / mode) if videos_dir else None,
                    max_episodes_rendered=n_videos,
                    start_seed=seed,
                    close_envs_after_eval=False,  # reused by the next checkpoint
                )
            overall = info["overall"]
            results[mode] = {
                "pc_success": overall["pc_success"],
                "avg_sum_reward": overall["avg_sum_reward"],
                "eval_s": overall["eval_s"],
            }
            logging.info(
                "%s | %s -> %.1f%% success", pretrained_dir.parent.name, mode, overall["pc_success"]
            )
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="A training run's output_dir.")
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Checkpoint steps to score. Default: every checkpoint in the run.",
    )
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of parallel eval envs. Default: the task's eval_batch_size, which for "
        "lamp is one env per pinned placement so the sub-envs map 1:1 onto the bank.",
    )
    parser.add_argument("--n-videos", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--wandb-project", default="lerobot-dex-eval")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def init_wandb(args, task_spec, steps: list[int]):
    """One wandb run per TRAINING run, stepped by checkpoint, or None when disabled."""
    if args.no_wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        # Name it after the training run so an eval lines up with its curve in `lerobot-dex`.
        name=f"dualnoise_{args.run_dir.name}",
        job_type="dual_noise_eval",
        config={
            "run_dir": str(args.run_dir),
            "run_name": args.run_dir.name,
            "task": args.task,
            "robot": task_spec.robot,
            "robocasa_task": task_spec.robocasa_task,
            "controller": task_spec.controller_filename or "<robot default: OSC_POSE delta>",
            "n_episodes": args.n_episodes,
            "eval_batch_size": args.batch_size,
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

    if args.batch_size is None:
        args.batch_size = task_spec.eval_batch_size
    steps = args.steps if args.steps is not None else available_steps(args.run_dir)
    logging.info("Scoring %d checkpoints: %s", len(steps), steps)

    out_path = args.run_dir / "dual_noise_eval.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    run = init_wandb(args, task_spec, steps)

    # Built once for the whole ladder: identical for every checkpoint of a run.
    env_cfg = build_env_cfg(task_spec, checkpoint_dir(args.run_dir, steps[0]))
    envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=args.batch_size > 1)
    logging.info("Built eval envs: %s", {k: list(v) for k, v in envs.items()})

    for step in steps:
        pretrained_dir = checkpoint_dir(args.run_dir, step)
        videos_dir = None if args.no_videos else args.run_dir / "eval_dual_noise" / f"step_{step:07d}"
        results[str(step)] = evaluate_step(
            pretrained_dir=pretrained_dir,
            task_spec=task_spec,
            env_cfg=env_cfg,
            envs=envs,
            n_episodes=args.n_episodes,
            seed=args.seed,
            videos_dir=videos_dir,
            n_videos=0 if args.no_videos else args.n_videos,
        )
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
        logging.info("Wrote %s", out_path)

        if run is not None:
            row = results[str(step)]
            payload = {f"{mode}/{k}": v for mode, m in row.items() for k, v in m.items()}
            # The headline number for checkpoint selection: how much success is lost when the
            # chunk is driven by ONE noise vector, which is what DSRL actually steers.
            payload["gap/pc_success"] = row["iid"]["pc_success"] - row["duplicated"]["pc_success"]
            run.log(payload, step=step)

    for group in envs.values():
        for vec in group.values():
            vec.close()

    if run is not None:
        run.finish()

    print(f"\n{'step':>8}  {'iid':>8}  {'duplicated':>11}  {'gap':>7}")
    for step in sorted(results, key=int):
        row = results[step]
        iid, dup = row["iid"]["pc_success"], row["duplicated"]["pc_success"]
        print(f"{step:>8}  {iid:>7.1f}%  {dup:>10.1f}%  {iid - dup:>6.1f}")


if __name__ == "__main__":
    main()
