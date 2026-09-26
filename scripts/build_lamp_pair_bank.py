#!/usr/bin/env python
"""Build a two-start lamp bank: two training placements plus their midpoint, held out.

Takes the two entries from an existing bank, interpolates position and yaw to get the midpoint,
and runs the same feasibility checks on all three. Training = the pair; held out = the midpoint.

Example:
    python scripts/build_lamp_pair_bank.py \\
        --source_bank ~/scratch/lerobot/placement_banks/screwlightbulb_xarm6_seed1_calib/bank.json \\
        --pair s1_train_00,s1_train_06 --out $BANK/bank.json --render_dir $BANK/renders
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="numba_cache_"))

import numpy as np

_BUILDER = Path(__file__).resolve().parent / "build_lamp_placement_bank.py"
_spec = importlib.util.spec_from_file_location("lamp_bank_builder", _BUILDER)
bb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bb)


def yaw_to_quat(yaw_deg: float) -> np.ndarray:
    half = np.radians(yaw_deg) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def midpoint(a: dict, b: dict) -> dict:
    pos = (np.asarray(a["pos"]) + np.asarray(b["pos"])) / 2.0
    ya, yb = np.radians(a["yaw_deg"]), np.radians(b["yaw_deg"])
    yaw = np.degrees(np.arctan2((np.sin(ya) + np.sin(yb)) / 2, (np.cos(ya) + np.cos(yb)) / 2))
    return {"pos": pos.tolist(), "quat_wxyz": yaw_to_quat(yaw).tolist(), "yaw_deg": float(yaw)}


def find(scene: dict, entry_id: str) -> dict:
    pool = list(scene["train"]) + list(scene.get("heldout", []))
    ref = scene.get("reference_training_placement")
    if ref is not None:
        ref = {**ref, "id": ref.get("id", f"s{scene['scene_seed']}_reference")}
        pool.append(ref)
    for e in pool:
        if e.get("id") == entry_id:
            return e
    raise KeyError(f"{entry_id!r} not in bank scene {scene['scene_seed']}; have {[e.get('id') for e in pool]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_bank", required=True)
    parser.add_argument("--scene_seed", type=int, default=1)
    parser.add_argument("--pair", required=True, help="Two entry ids, comma-separated")
    parser.add_argument("--task", default="ScrewLightbulb")
    parser.add_argument("--robot", default="XArm6DexLeapRHOmron")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", required=True)
    parser.add_argument("--render_dir", default=None)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--allow_failed", action="store_true", help="Write the bank even if a check fails")
    args = parser.parse_args()

    ids = [x.strip() for x in args.pair.split(",") if x.strip()]
    if len(ids) != 2:
        raise ValueError(f"--pair needs exactly two ids, got {ids}")

    src = json.loads(Path(os.path.expanduser(args.source_bank)).read_text())
    scene_src = next(s for s in src["scenes"] if int(s["scene_seed"]) == args.scene_seed)
    a, b = (find(scene_src, i) for i in ids)
    mid = midpoint(a, b)

    sep = float(np.linalg.norm(np.asarray(a["pos"][:2]) - np.asarray(b["pos"][:2])))
    print(f"pair {ids[0]} / {ids[1]}: {sep * 100:.1f} cm apart, yaw {a['yaw_deg']:.1f} / {b['yaw_deg']:.1f}")
    print(f"midpoint: yaw {mid['yaw_deg']:.1f}")

    render = args.render_dir is not None
    scene = bb.LampScene(args.task, args.robot, args.fps, args.scene_seed, render, args.resolution)

    entries, failed = [], []
    for name, e in (("s1_start_0", a), ("s1_start_1", b), ("s1_mid", mid)):
        pos, quat = np.asarray(e["pos"]), np.asarray(e["quat_wxyz"])
        reasons, rest, _ = scene.evaluate(pos, quat)
        print(f"  {name:12s} {reasons or 'PASS'}  {bb._summary(rest)}")
        entry = {
            "id": name,
            "pos": pos.tolist(),
            "quat_wxyz": quat.tolist(),
            "yaw_deg": float(e["yaw_deg"]),
            "source_id": e.get("id"),
            "metrics": rest,
        }
        if reasons:
            entry["checks_failed"] = reasons
            failed.append(name)
        entries.append(entry)

    if failed and not args.allow_failed:
        scene.env.close()
        raise SystemExit(f"feasibility check failed for {failed}; inspect, then re-run with --allow_failed")

    train, heldout = entries[:2], entries[2:]
    ref = scene_src["reference_training_placement"]
    if render:
        bb._render_scene(
            scene, args.scene_seed, np.asarray(ref["pos"]), np.asarray(ref["quat_wxyz"]),
            train, heldout, Path(os.path.expanduser(args.render_dir)),
        )
    scene.env.close()

    bank = {
        "task": args.task,
        "robot": args.robot,
        "object_name": "lamp",
        "controller": bb.controller_path(args.robot),
        "fps": args.fps,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "quat_convention": src.get("quat_convention"),
        "how_to_apply": src.get("how_to_apply"),
        "checks": bb.CHECKS,
        "source_bank": str(args.source_bank),
        "pair_separation_m": sep,
        "scenes": [
            {
                "scene_seed": args.scene_seed,
                "layout_id": scene_src["layout_id"],
                "style_id": scene_src["style_id"],
                "support_fixture": scene_src.get("support_fixture"),
                "reference_training_placement": ref,
                "train": train,
                "heldout": heldout,
            }
        ],
    }
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
