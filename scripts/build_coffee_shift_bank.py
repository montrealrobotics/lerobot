#!/usr/bin/env python
"""Build a fixed-kitchen coffee bank whose entries slide the coffee machine, not an object.

One kitchen (construction seed), the robot centred on the machine as RoboCasa places it; each
entry offsets the machine (and the mug under it) by ``lateral_m`` along the robot's left (+) /
right (-) and ``shift_m`` toward the robot, applied by ``PlacementBankWrapper`` through
``FixtureShiftWrapper``. Groups mirror the lamp pair bank: two training offsets, the held-out
midpoint, and extrapolation cells one pair-spacing past each end.

Each entry is checked after a noise-free reset: the machine footprint stays on its counter, the
mug still rests under the dispenser, the robot does not touch the machine, and the button is in
the agentview_left frame.

Example:
    python scripts/build_coffee_shift_bank.py --scene_seed 1 \\
        --train_lateral_cm=-4,0 --heldout_lateral_cm=-2 --extrap_lateral_cm=-8,4 \\
        --out $BANKS/coffeepressbutton_pandadex_s1_lateral/bank.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="numba_cache_"))

import numpy as np

from lerobot.envs.robocasa_env import RoboCasaEnv
from lerobot.envs.robocasa_placement_bank import FixtureShiftWrapper

CAMERA = "robot0_agentview_left"
ROBOT_PREFIXES = ("robot0_", "gripper0_", "mobilebase0_")


def cm_list(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def entry_id(seed: int, lateral_cm: float, shift_cm: float) -> str:
    tag = f"s{seed}_lat{lateral_cm:+g}cm"
    return tag if shift_cm == 0 else f"{tag}_fwd{shift_cm:+g}cm"


def check(env: FixtureShiftWrapper, counter_box: tuple[np.ndarray, np.ndarray]) -> tuple[list[str], dict]:
    inner = env.unwrapped._env
    m, d = inner.sim.model._model, inner.sim.data._data
    base = m.body("robot0_base").id
    rot, p = d.xmat[base].reshape(3, 3), d.xpos[base]

    def rel(x):
        return rot.T @ (np.asarray(x) - p)

    machine = inner.coffee_machine
    corners = np.array(
        [rel(q + env.offset()) for q in machine.get_ext_sites(all_points=True, relative=False)]
    )
    lo, hi = corners.min(0), corners.max(0)
    clo, chi = counter_box
    reasons = []
    if lo[0] < clo[0] or hi[0] > chi[0] or lo[1] < clo[1] or hi[1] > chi[1]:
        reasons.append("machine_off_counter")

    machine_bodies = {i for i in range(m.nbody) if m.body(i).name.startswith(machine.naming_prefix)}
    touching = set()
    for i in range(d.ncon):
        b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
        for a, b in ((b1, b2), (b2, b1)):
            if a in machine_bodies and m.body(b).name.startswith(ROBOT_PREFIXES):
                touching.add(m.body(b).name)
    if touching:
        reasons.append(f"robot_touches_machine:{sorted(touching)}")

    pour = d.site_xpos[m.site(f"{machine.naming_prefix}receptacle_place_site").id]
    mug = d.xpos[m.body(inner.objects["obj"].root_body).id]
    mug_xy = float(np.linalg.norm(mug[:2] - pour[:2]))
    if mug_xy > 0.04:
        reasons.append(f"mug_not_under_dispenser:{mug_xy * 100:.1f}cm")

    btn = d.geom_xpos[m.geom(f"{machine.naming_prefix}start_button").id]
    cam = m.camera(CAMERA).id
    pc = d.cam_xmat[cam].reshape(3, 3).T @ (btn - d.cam_xpos[cam])
    f = 0.5 / math.tan(math.radians(m.cam_fovy[cam]) / 2)
    u, v = 0.5 + f * pc[0] / -pc[2], 0.5 - f * pc[1] / -pc[2]
    if not (0.05 <= u <= 0.95 and 0.05 <= v <= 0.95):
        reasons.append(f"button_out_of_frame:uv=({u:.2f},{v:.2f})")

    metrics = {
        "button_in_base_frame_m": np.round(rel(btn), 4).tolist(),
        "button_image_uv": [round(float(u), 3), round(float(v), 3)],
        "machine_footprint_base_m": {
            "x": [round(lo[0], 3), round(hi[0], 3)],
            "y": [round(lo[1], 3), round(hi[1], 3)],
        },
        "mug_to_dispenser_xy_m": round(mug_xy, 4),
    }
    return reasons, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene_seed", type=int, default=1)
    parser.add_argument("--task", default="CoffeePressButton")
    parser.add_argument("--robot", default="PandaDexLeapRHOmron")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--train_lateral_cm", required=True)
    parser.add_argument("--heldout_lateral_cm", default="")
    parser.add_argument("--extrap_lateral_cm", default="")
    parser.add_argument(
        "--shift_cm", type=float, default=0.0, help="Toward-robot offset applied to every entry"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow_failed", action="store_true")
    args = parser.parse_args()

    raw = RoboCasaEnv(
        task_name=args.task,
        robot=args.robot,
        control_freq=args.fps,
        camera_name=CAMERA,
        render_mode="none",
        return_raw_obs=True,
        seed=args.scene_seed,
    )
    inner = raw._env
    # Checks run on the noise-free start: robocasa's arm reset noise is what start-pose rejection
    # handles, separately, at run time.
    inner.robots[0].initialization_noise["magnitude"] = 0.0
    env = FixtureShiftWrapper(raw)
    m, d = inner.sim.model._model, inner.sim.data._data
    base = m.body("robot0_base").id
    rot, p = d.xmat[base].reshape(3, 3).copy(), d.xpos[base].copy()
    counter = np.array(
        [rot.T @ (q - p) for q in inner.counter.get_ext_sites(all_points=True, relative=False)]
    )
    counter_box = (counter.min(0), counter.max(0))

    groups: dict[str, list[dict]] = {}
    failed = []
    for group, spec in (
        ("train", args.train_lateral_cm),
        ("heldout", args.heldout_lateral_cm),
        ("extrap", args.extrap_lateral_cm),
    ):
        for lat in cm_list(spec):
            env.shift_m, env.lateral_m = args.shift_cm / 100.0, lat / 100.0
            env.reset()
            reasons, metrics = check(env, counter_box)
            e = {
                "id": entry_id(args.scene_seed, lat, args.shift_cm),
                "lateral_m": lat / 100.0,
                "shift_m": args.shift_cm / 100.0,
                "metrics": metrics,
            }
            if reasons:
                e["checks_failed"] = reasons
                failed.append(e["id"])
            groups.setdefault(group, []).append(e)
            print(
                f"  {group:8s} {e['id']:18s} {reasons or 'PASS'}  button {metrics['button_in_base_frame_m']}"
            )

    if failed and not args.allow_failed:
        raise SystemExit(f"feasibility check failed for {failed}; inspect, then re-run with --allow_failed")

    bank = {
        "task": args.task,
        "robot": args.robot,
        "fps": args.fps,
        "bank_type": "fixture_shift",
        "fixture_attr": "coffee_machine",
        "carried_objects": ["obj"],
        "object_name": None,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "how_to_apply": "PlacementBankWrapper(env built with seed=scene_seed, wrapped in start-pose rejection)",
        "axes": "lateral_m: + = robot base +y (robot's left); shift_m: + = toward the robot",
        "scenes": [
            {
                "scene_seed": args.scene_seed,
                "layout_id": int(inner.layout_id),
                "style_id": int(inner.style_id),
                "coffee_machine_fixture": inner.coffee_machine.name,
                **groups,
            }
        ],
    }
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, indent=1))
    print(f"wrote {out}")
    raw.close()


if __name__ == "__main__":
    main()
