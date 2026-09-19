#!/usr/bin/env python
"""Sweep every pi05 SFT checkpoint of a run over a pinned RoboCasa kitchen.

The training-time eval curve that checkpoints were picked from runs in whichever kitchens the
sub-env construction seeds pick, each contributing one object placement. This re-scores the whole
checkpoint sequence on one pinned kitchen over several placements, so the selection criterion can
be compared. Envs are built once and reused across checkpoints (construction dominates otherwise).
Results go to WandB with ``step`` = the SFT step, so the sweep overlays the training curve.

Example:
    python scripts/eval_sft_checkpoints_sweep.py \
        --run_dir ~/scratch/lerobot/outputs/train/2026-08-28/15-26-13_robocasa_pi05 \
        --layout_id 1 --style_id 4 --n_placements 10 --n_episodes 20
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path

# Must precede the robocasa (hence numba) import; see robocasa-env-gotchas.
os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="numba_cache_"))

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType  # noqa: F401  (keeps config registry import order stable)
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed


def _register_literal_draccus_decoder() -> None:
    """draccus has no ``Literal`` decoder; checkpoints that saved such a field fail to load."""
    import typing

    import draccus

    draccus.decode.register(typing.Literal, lambda raw_value, path=(): raw_value)


def discover_checkpoints(run_dir: Path, steps: str | None) -> list[tuple[int, Path]]:
    """Return ``[(step, pretrained_model_dir)]`` ascending, skipping the ``last`` symlink.

    ``last`` points at the final numbered checkpoint, so including it would double-count.
    """
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"No checkpoints/ under {run_dir}")

    wanted = None
    if steps:
        wanted = {int(s) for s in steps.replace(" ", "").split(",") if s}

    out = []
    for child in sorted(ckpt_root.iterdir()):
        if not child.is_dir() or child.is_symlink() or not child.name.isdigit():
            continue
        step = int(child.name)
        if wanted is not None and step not in wanted:
            continue
        model_dir = child / "pretrained_model"
        if (model_dir / "config.json").exists():
            out.append((step, model_dir))
    if not out:
        raise FileNotFoundError(f"No usable checkpoints found under {ckpt_root} (steps={steps!r})")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, required=True, help="SFT training output dir")
    parser.add_argument("--layout_id", type=int, required=True, help="RoboCasa layout to pin")
    parser.add_argument("--style_id", type=int, required=True, help="RoboCasa style to pin")
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated SFT steps to evaluate (e.g. '2000,4000,6000'). Default: every "
        "numbered checkpoint in the run.",
    )
    parser.add_argument(
        "--n_placements",
        type=int,
        default=10,
        help="Number of object placements = number of parallel envs. With the scene pinned, each "
        "sub-env's construction seed varies only object pose / robot base, so this is the axis "
        "that actually carries variance. Raise it in preference to --n_episodes.",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=20,
        help="Total episodes per checkpoint, spread over the placements. Episodes beyond the "
        "first per placement differ only by robocasa's 0.02 rad arm-joint reset noise.",
    )
    parser.add_argument("--use_async_envs", action="store_true", help="One process per placement")
    parser.add_argument("--n_videos", type=int, default=4, help="Eval videos to log per checkpoint")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="lerobot-dex")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    register_third_party_plugins()
    _register_literal_draccus_decoder()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = Path(os.path.expanduser(args.run_dir))
    checkpoints = discover_checkpoints(run_dir, args.steps)
    out_dir = Path(os.path.expanduser(args.output_dir)) if args.output_dir else run_dir / "eval_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The run's own saved config carries task/robot/controller/fps, so the sweep cannot drift
    # from how the policy was trained. Only the kitchen is overridden.
    with open(checkpoints[0][1] / "train_config.json") as f:
        train_cfg = json.load(f)
    from lerobot.envs.configs import RoboCasaEnv

    env_src = train_cfg["env"]
    env_cfg = RoboCasaEnv(
        task=env_src["task"],
        robot=env_src["robot"],
        controller=env_src.get("controller"),
        fps=env_src["fps"],
        camera_name=env_src["camera_name"],
        layout_id=args.layout_id,
        style_id=args.style_id,
    )
    print(
        f"Sweeping {len(checkpoints)} checkpoints of {run_dir.name}\n"
        f"  task={env_cfg.task} robot={env_cfg.robot} fps={env_cfg.fps}\n"
        f"  kitchen PINNED to layout={args.layout_id} style={args.style_id}\n"
        f"  {args.n_placements} placements x {args.n_episodes} total episodes per checkpoint\n"
        f"  steps: {[s for s, _ in checkpoints]}"
    )

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or f"sftsweep_{run_dir.name}_L{args.layout_id}S{args.style_id}",
                dir=str(out_dir),
                config={
                    **vars(args),
                    "task": env_cfg.task,
                    "robot": env_cfg.robot,
                    "sft_run": run_dir.name,
                    "mode": "sft_checkpoint_sweep",
                },
            )
            print(f"WandB run: {wandb.run.get_url()}")
        except ImportError:
            print("wandb not installed — results will only be written to disk.")

    # ── Build the envs once and reuse them for every checkpoint ──
    envs = make_env(env_cfg, n_envs=args.n_placements, use_async_envs=args.use_async_envs)

    results: list[dict] = []
    try:
        for step, model_dir in checkpoints:
            print(f"\n=== step {step}: {model_dir}")
            policy_cfg = PreTrainedConfig.from_pretrained(model_dir)
            policy_cfg.pretrained_path = model_dir
            # The SFT configs carry compile_model=True / gradient_checkpointing=True, both of
            # which are training settings. Left on, a sweep pays the compile cost once per
            # checkpoint AND leaks: dynamo's cache keeps each compiled module alive and CUDA-graph
            # private pools are not reclaimable by empty_cache(), so GPU use grows ~8 GB per
            # checkpoint until the next .to(device) OOMs. Inference does not need either.
            policy_cfg.compile_model = False
            policy_cfg.gradient_checkpointing = False
            policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
            policy.eval()

            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=model_dir,
                preprocessor_overrides={"device_processor": {"device": str(policy_cfg.device)}},
            )
            env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

            with torch.no_grad():
                info = eval_policy_all(
                    envs=envs,
                    policy=policy,
                    env_processors=dict.fromkeys(envs, (env_pre, env_post)),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=args.n_episodes,
                    max_episodes_rendered=args.n_videos,
                    videos_dir=out_dir / f"videos_L{args.layout_id}S{args.style_id}_step{step:06d}",
                    start_seed=args.seed,
                    max_parallel_tasks=env_cfg.max_parallel_tasks,
                    close_envs_after_eval=False,
                )
            overall = info["overall"]
            row = {"sft_step": step, **{k: v for k, v in overall.items() if isinstance(v, (int, float))}}
            results.append(row)
            print(f"  step {step}: " + " ".join(f"{k}={v}" for k, v in row.items() if k != "sft_step"))

            if wandb_run is not None:
                import wandb

                log = {f"eval_L{args.layout_id}S{args.style_id}/{k}": v for k, v in row.items()}
                paths = info.get("per_group", {})
                videos = []
                for group in paths.values():
                    videos += list(group.get("video_paths", []) or [])
                for i, vp in enumerate(videos[: args.n_videos]):
                    if os.path.exists(vp):
                        log[f"eval_L{args.layout_id}S{args.style_id}/video_{i}"] = wandb.Video(
                            vp, fps=env_cfg.fps, format="mp4"
                        )
                wandb.log(log, step=step)

            # Drop every reference to this checkpoint before loading the next one. `info` holds
            # per-episode tensors, and the processors hold normalization buffers; without this the
            # sweep accumulates one full policy per step on the GPU.
            del policy, preprocessor, postprocessor, env_pre, env_post, info
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                print(f"  gpu after cleanup: {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated")
    finally:
        close_envs(envs)
        summary = out_dir / f"sweep_L{args.layout_id}S{args.style_id}.json"
        with open(summary, "w") as f:
            json.dump(
                {
                    "run_dir": str(run_dir),
                    "layout_id": args.layout_id,
                    "style_id": args.style_id,
                    "n_placements": args.n_placements,
                    "n_episodes": args.n_episodes,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nWrote {summary}")
        if results:
            best = max(results, key=lambda r: r.get("pc_success", -1))
            print(f"argmax pc_success: step {best['sft_step']} at {best.get('pc_success')}")
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
