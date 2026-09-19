#!/usr/bin/env python
"""Score a frozen SFT policy (no DSRL) on the same bank cells the DSRL evals use.

Cells, kitchens, placements and start-pose handling match
``dsrl_pi05_robocasa_example.py --eval_only``; cameras are the two views pi05 receives inside
DSRL (with both agentview cameras present the checkpoint's random-external-camera augmentation
would swap in the right view, which DSRL never does). The policy samples its own i.i.d. flow
noise through lerobot's ``eval_policy``, which the DSRL harness cannot reproduce.

One sub-env per cell, so episode ``i`` belongs to cell ``i % n_cells``; envs are built once and
reused for every checkpoint in --steps.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="numba_cache_"))

import gymnasium as gym
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import make_env_pre_post_processors
from lerobot.envs.configs import RoboCasaEnv
from lerobot.envs.robocasa_env import RoboCasaEnv as RoboCasaGymEnv
from lerobot.envs.robocasa_placement_bank import (
    PlacementBankWrapper,
    StartPoseRejectionWrapper,
    kitchen_eval_cells,
    load_placement_bank,
    placement_eval_cells,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed

DSRL_CAMERAS = "robot0_agentview_left,robot0_eye_in_hand"


def _register_literal_draccus_decoder() -> None:
    import typing

    import draccus

    draccus.decode.register(typing.Literal, lambda raw_value, path=(): raw_value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sft_run", required=True, help="SFT training output dir (has checkpoints/)")
    parser.add_argument("--steps", required=True, help="Comma-separated SFT steps, e.g. 3000,7000")
    bank = parser.add_mutually_exclusive_group(required=True)
    bank.add_argument("--placement_bank", help="Lamp placement bank JSON")
    bank.add_argument("--kitchen_bank", help="Coffee kitchen bank JSON")
    parser.add_argument(
        "--eval_sets",
        default=None,
        help="Cell groups. Default: heldout,reference,train:2 (placement) / heldout,reference (kitchen)",
    )
    parser.add_argument("--episodes_per_cell", type=int, required=True)
    parser.add_argument("--cameras", default=DSRL_CAMERAS)
    parser.add_argument("--reject_start_rot_deg", type=float, default=0.0)
    parser.add_argument("--reject_start_pos_cm", type=float, default=3.0)
    parser.add_argument("--reject_start_max_retries", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--wandb_project", default="lerobot-dex")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    register_third_party_plugins()
    _register_literal_draccus_decoder()
    set_seed(args.seed)

    run_dir = Path(os.path.expanduser(args.sft_run))
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    model_dirs = {s: run_dir / "checkpoints" / f"{s:06d}" / "pretrained_model" for s in steps}
    for s, d in model_dirs.items():
        if not (d / "model.safetensors").exists():
            raise FileNotFoundError(f"SFT step {s}: no checkpoint at {d}")
    with open(model_dirs[steps[0]] / "train_config.json") as f:
        env_src = json.load(f)["env"]

    env_cfg = RoboCasaEnv(
        task=env_src["task"],
        robot=env_src["robot"],
        controller=env_src.get("controller"),
        fps=env_src["fps"],
        camera_name=args.cameras,
    )

    if args.placement_bank:
        bank_data = load_placement_bank(args.placement_bank)
        spec = args.eval_sets or "heldout,reference,train:2"
        cells = placement_eval_cells(bank_data, spec, sorted(bank_data["_scenes_by_seed"]))
    else:
        with open(os.path.expanduser(args.kitchen_bank)) as f:
            bank_data = json.load(f)
        spec = args.eval_sets or "heldout,reference"
        cells = kitchen_eval_cells(bank_data, spec)
    if bank_data["task"] != env_cfg.task or bank_data["robot"] != env_cfg.robot:
        raise ValueError(
            f"Bank is for {bank_data['task']}/{bank_data['robot']}, SFT run is {env_cfg.task}/{env_cfg.robot}"
        )

    rejection = None
    if args.reject_start_rot_deg > 0:
        rejection = {
            "max_rot_deg": args.reject_start_rot_deg,
            "max_pos_m": args.reject_start_pos_cm / 100.0,
            "max_retries": args.reject_start_max_retries,
        }

    def make_cell_env(cell: dict):
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
                env = PlacementBankWrapper(env, bank_data, cell["seed"], [cell["entry"]], shuffle=False)
            return env

        return _make

    labels = [c["label"] for c in cells]
    print(
        f"SFT-only baseline | {env_cfg.task} {env_cfg.robot} | run {run_dir.name} steps {steps}\n"
        f"  {len(cells)} cells ({spec}) x {args.episodes_per_cell} episodes | cameras {args.cameras} | "
        f"start rejection {rejection}\n  cells: {labels}"
    )
    vec = gym.vector.SyncVectorEnv([make_cell_env(c) for c in cells])

    out_dir = Path(os.path.expanduser(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            dir=str(out_dir),
            config={
                **vars(args),
                "task": env_cfg.task,
                "robot": env_cfg.robot,
                "cells": labels,
                "mode": "sft_only_bank_eval",
            },
        )

    results = []
    try:
        for step in steps:
            model_dir = model_dirs[step]
            policy_cfg = PreTrainedConfig.from_pretrained(model_dir)
            policy_cfg.pretrained_path = model_dir
            policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
            policy.eval()
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=model_dir,
                preprocessor_overrides={"device_processor": {"device": str(policy_cfg.device)}},
            )
            env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

            with torch.no_grad():
                info = eval_policy(
                    env=vec,
                    policy=policy,
                    env_preprocessor=env_pre,
                    env_postprocessor=env_post,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=len(cells) * args.episodes_per_cell,
                    max_episodes_rendered=0,
                    start_seed=args.seed,
                )
            succ = np.array([float(ep["success"]) for ep in info["per_episode"]]).reshape(
                args.episodes_per_cell, len(cells)
            )
            per_cell = dict(zip(labels, succ.mean(axis=0).tolist(), strict=True))
            metrics = {f"success_rate_{k}": v for k, v in per_cell.items()}
            for group in dict.fromkeys(c["group"] for c in cells):
                idx = [i for i, c in enumerate(cells) if c["group"] == group]
                metrics[f"success_rate_{group}"] = float(succ[:, idx].mean())
            metrics["success_rate"] = float(succ.mean())
            if rejection is not None:
                rej = [w if isinstance(w, StartPoseRejectionWrapper) else w.env for w in vec.envs]
                metrics["start_rejections"] = int(sum(r.n_rejected for r in rej))
            results.append({"sft_step": step, **metrics})

            print(f"\n==== SFT-only step {step} ====")
            for k in sorted(metrics):
                print(
                    f"  {k} = {metrics[k]:.4f}" if isinstance(metrics[k], float) else f"  {k} = {metrics[k]}"
                )
            if wandb_run is not None:
                wandb_run.log({f"sft_only/{k}": v for k, v in metrics.items()}, step=step)

            del policy
            torch.cuda.empty_cache()
    finally:
        vec.close()
        summary = out_dir / "sft_only_bank_eval.json"
        summary.write_text(
            json.dumps(
                {
                    "sft_run": str(run_dir),
                    "cells": labels,
                    "eval_sets": spec,
                    "episodes_per_cell": args.episodes_per_cell,
                    "cameras": args.cameras,
                    "start_rejection": rejection,
                    "results": results,
                },
                indent=1,
            )
        )
        print(f"wrote {summary}")
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
