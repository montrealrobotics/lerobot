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

Results are appended to <run-dir>/dual_noise_eval.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
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


class DuplicatedNoisePolicy:
    """Wraps a pi05 policy so every chunk is denoised from a single repeated noise vector.

    Implements only what `eval_policy` touches: ``reset()``, ``eval()`` and
    ``select_action()``. The action-queue logic mirrors ``PI05Policy.select_action``;
    the only difference from the normal path is the ``noise`` argument.
    """

    def __init__(self, policy):
        self.policy = policy
        config = policy.config
        self.chunk_size = config.chunk_size
        self.n_action_steps = config.n_action_steps
        self.max_action_dim = config.max_action_dim
        self.device = next(policy.parameters()).device
        self._queue: deque = deque(maxlen=self.n_action_steps)

    def reset(self) -> None:
        self._queue.clear()
        self.policy.reset()

    def eval(self):
        self.policy.eval()
        return self

    @staticmethod
    def _batch_size(batch: dict) -> int:
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return value.shape[0]
        raise ValueError("Could not infer batch size from the observation batch.")

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        if len(self._queue) == 0:
            bsize = self._batch_size(batch)
            # Same distribution and dtype as PI05FlowMatching.sample_noise, but one
            # draw per episode-chunk instead of one per chunk step.
            noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=(bsize, 1, self.max_action_dim),
                dtype=torch.float32,
                device=self.device,
            ).expand(bsize, self.chunk_size, self.max_action_dim)
            actions = self.policy.predict_action_chunk(batch, noise=noise.contiguous())
            actions = actions[:, : self.n_action_steps]
            self._queue.extend(actions.transpose(0, 1))
        return self._queue.popleft()


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


def evaluate_step(
    pretrained_dir: Path,
    task_spec,
    n_episodes: int,
    batch_size: int,
    seed: int,
    videos_dir: Path | None,
    n_videos: int,
) -> dict[str, dict[str, float]]:
    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_dir)
    policy_cfg.pretrained_path = pretrained_dir

    env_cfg = RoboCasaEnv(
        task=task_spec.robocasa_task,
        robot=task_spec.robot,
        # Per-task, not per-robot: coffee is OSC_POSE deltas and lamp is absolute joint
        # targets, so deriving this from the robot silently scrambles the action vector.
        controller=resolve_controller(task_spec.controller_filename),
        fps=_fps_from_train_config(pretrained_dir),
        camera_name=task_spec.camera_name,
    )

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_dir,
        preprocessor_overrides={"device_processor": {"device": str(policy_cfg.device)}},
    )
    envs = make_env(env_cfg, n_envs=batch_size, use_async_envs=batch_size > 1)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    env_processors = dict.fromkeys(envs, (env_pre, env_post))

    results: dict[str, dict[str, float]] = {}
    for mode in NOISE_MODES:
        # Identical seed per mode so both share episode/scene sampling and the only
        # difference is the noise parameterisation.
        set_seed(seed)
        evaluated = policy if mode == "iid" else DuplicatedNoisePolicy(policy)
        with torch.no_grad():
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
                close_envs_after_eval=mode == NOISE_MODES[-1],
            )
        overall = info["overall"]
        results[mode] = {
            "pc_success": overall["pc_success"],
            "avg_sum_reward": overall["avg_sum_reward"],
            "eval_s": overall["eval_s"],
        }
        logging.info("%s | %s -> %.1f%% success", pretrained_dir.parent.name, mode, overall["pc_success"])
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
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="Checkpoint steps to score.")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-videos", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-videos", action="store_true")
    return parser.parse_args()


def main() -> None:
    init_logging()
    args = parse_args()
    register_third_party_plugins()
    task_spec = TASKS[args.task]

    out_path = args.run_dir / "dual_noise_eval.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for step in args.steps:
        pretrained_dir = checkpoint_dir(args.run_dir, step)
        videos_dir = None if args.no_videos else args.run_dir / "eval_dual_noise" / f"step_{step:07d}"
        results[str(step)] = evaluate_step(
            pretrained_dir=pretrained_dir,
            task_spec=task_spec,
            n_episodes=args.n_episodes,
            batch_size=args.batch_size,
            seed=args.seed,
            videos_dir=videos_dir,
            n_videos=0 if args.no_videos else args.n_videos,
        )
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
        logging.info("Wrote %s", out_path)

    print(f"\n{'step':>8}  {'iid':>8}  {'duplicated':>11}  {'gap':>7}")
    for step in sorted(results, key=int):
        row = results[step]
        iid, dup = row["iid"]["pc_success"], row["duplicated"]["pc_success"]
        print(f"{step:>8}  {iid:>7.1f}%  {dup:>10.1f}%  {iid - dup:>6.1f}")


if __name__ == "__main__":
    main()
