#!/usr/bin/env python
"""
Frozen-SFT success vs coffee-machine position, in one kitchen, under both noise modes.

Sets the training / held-out positions of the fixed-kitchen coffee DSRL bank, as
lamp_offset_calibration.sbatch set the lamp pair. Each cell slides the machine (and the mug under
it) with ``FixtureShiftWrapper``, along one axis of the robot base frame:

  --axis depth     toward the robot (the machine starts against the back wall), cm >= 0
  --axis lateral   sideways, + = the robot's left (the robot starts centred on it)

Robot, start pose and kitchen are identical across cells.

Scoring is ``eval_pi05_dual_noise.evaluate_step`` unchanged: same cameras, same start-pose
rejection, iid and duplicated noise, one sub-env per cell.

Usage:
    python scripts/eval_coffee_distance_calibration.py --run-dir <coffee sft run> --step 8000 \
        --scene-seed 1 --axis depth --shifts-cm 0 2 4 6 8 10 12 14 --episodes-per-cell 25 --output-dir <out>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import gymnasium as gym

from lerobot.envs.robocasa_env import RoboCasaEnv as RoboCasaGymEnv
from lerobot.envs.robocasa_placement_bank import FixtureShiftWrapper, StartPoseRejectionWrapper
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "training"))
from eval_pi05_dual_noise import (  # noqa: E402
    DSRL_CAMERAS,
    NOISE_MODES,
    TASKS,
    build_env_cfg,
    checkpoint_dir,
    evaluate_step,
)


def make_shift_vec_env(env_cfg, scene_seed: int, cells: list[dict], rejection: dict, axis: str):
    def factory(shift_m: float):
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
                seed=scene_seed,
            )
            kw = {"shift_m": shift_m} if axis == "depth" else {"lateral_m": shift_m}
            return FixtureShiftWrapper(StartPoseRejectionWrapper(env, **rejection), **kw)

        return _make

    return gym.vector.SyncVectorEnv([factory(c["shift_cm"] / 100.0) for c in cells])


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--scene-seed", type=int, default=1)
    parser.add_argument("--axis", choices=("depth", "lateral"), default="depth")
    parser.add_argument("--shifts-cm", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 14])
    parser.add_argument("--episodes-per-cell", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cameras", default=DSRL_CAMERAS)
    parser.add_argument("--reject-start-rot-deg", type=float, default=10.0)
    parser.add_argument("--reject-start-pos-cm", type=float, default=3.0)
    parser.add_argument("--reject-start-max-retries", type=int, default=20)
    parser.add_argument("--n-videos", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", choices=NOISE_MODES, default=list(NOISE_MODES))
    args = parser.parse_args()
    register_third_party_plugins()

    pretrained_dir = checkpoint_dir(args.run_dir, args.step)
    env_cfg = build_env_cfg(TASKS["coffee"], pretrained_dir, args.cameras)
    prefix = "shift" if args.axis == "depth" else "lateral"
    cells = [{"label": f"{prefix}_{s:g}cm", "group": "sweep", "shift_cm": s} for s in args.shifts_cm]
    rejection = {
        "max_rot_deg": args.reject_start_rot_deg,
        "max_pos_m": args.reject_start_pos_cm / 100.0,
        "max_retries": args.reject_start_max_retries,
    }
    suffix = "" if args.axis == "depth" else f"_{args.axis}"
    out_dir = args.output_dir / f"{args.run_dir.name}_step{args.step}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Scoring %s on seed %d, %s shifts %s cm", pretrained_dir, args.scene_seed, args.axis, args.shifts_cm
    )

    vec = make_shift_vec_env(env_cfg, args.scene_seed, cells, rejection, args.axis)
    try:
        geometry = [
            {"shift_cm": c["shift_cm"], "offset_world_m": e.offset().round(4).tolist()}
            for c, e in zip(cells, vec.envs, strict=True)
        ]
        row = evaluate_step(
            pretrained_dir=pretrained_dir,
            env_cfg=env_cfg,
            vec=vec,
            cells=cells,
            episodes_per_cell=args.episodes_per_cell,
            seed=args.seed,
            videos_dir=out_dir / "videos",
            n_videos=args.n_videos,
            modes=tuple(args.modes),
        )
        rejections = {
            c["label"]: {
                "resets": e.env.n_resets,
                "rejected": e.env.n_rejected,
                "exhausted": e.env.n_exhausted,
            }
            for c, e in zip(cells, vec.envs, strict=True)
        }
    finally:
        vec.close()

    result = {
        "run_dir": str(args.run_dir),
        "step": args.step,
        "scene_seed": args.scene_seed,
        "axis": args.axis,
        "episodes_per_cell": args.episodes_per_cell,
        "cameras": args.cameras,
        "start_rejection": rejection,
        "cells": geometry,
        "start_rejections": rejections,
        "results": row,
    }
    (out_dir / "calibration.json").write_text(json.dumps(result, indent=2))
    logging.info("Wrote %s", out_dir / "calibration.json")

    print(f"\n{args.axis + '_cm':>10}  " + "  ".join(f"{m:>10}" for m in args.modes))
    for c in cells:
        cols = "  ".join(f"{row[m][f'pc_success_{c["label"]}']:>9.0f}%" for m in args.modes)
        print(f"{c['shift_cm']:>10g}  {cols}")


if __name__ == "__main__":
    main()
