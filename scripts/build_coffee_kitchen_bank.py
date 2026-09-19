#!/usr/bin/env python
"""Build a checked bank of CoffeePressButton kitchens (training + held-out).

Placement randomization does not apply to this task: RoboCasa centres the robot on the coffee
machine in every kitchen, so moving the machine moves the robot with it, and the mug does not
affect the button press. The unit of variation is therefore the kitchen, identified by its
construction seed exactly as the DSRL runs use it. Seeds 0,1 (the fixed-kitchen runs' kitchens)
are always in the training set.

Kitchens are rejected only when the mobile base spawns inside a fixture or the button is out of
frame. Fingertips touching cabinet trim at reset, and the arm occluding the button in
robot0_agentview_left, are recorded as flags: a first scan found them in 114/123 and 65/123
kitchens, including both kitchens where DSRL reached ~0.9.

``--render_dir`` renders every selected kitchen from robot0_agentview_left (needs EGL).
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

import mujoco
import numpy as np

from lerobot.envs.robocasa_env import RoboCasaEnv

CAMERA = "robot0_agentview_left"
ROBOT_PREFIXES = ("robot0_", "gripper0_", "mobilebase0_")
STYLE_MACHINE = {  # robocasa/models/assets/scenes/kitchen_styles/*.yaml
    0: "delonghi_espresso", 4: "delonghi_espresso", 5: "delonghi_espresso", 9: "delonghi_espresso",
    1: "nespresso", 7: "nespresso", 8: "nespresso", 10: "nespresso",
    2: "delonghi_espresso_2", 3: "delonghi_espresso_2", 6: "delonghi_espresso_2", 11: "delonghi_espresso_2",
}  # fmt: skip
# Layouts recoverable from the coffee SFT dataset's sim-state fingerprint (robocasa-sft-scene-coverage):
# 0,2,4,5,6,8,9 present; 1 vs 7 ambiguous; 3 absent. Styles are not recoverable.
SFT_LAYOUTS = {0: "seen", 2: "seen", 4: "seen", 5: "seen", 6: "seen", 8: "seen", 9: "seen",
               1: "ambiguous(1|7)", 7: "ambiguous(1|7)", 3: "absent"}  # fmt: skip
FRAME_MARGIN = 0.08


def seed_to_scene(seed: int) -> tuple[int, int]:
    """Verified robot- and task-invariant: the env rng's first draw picks the (layout, style)."""
    pairs = [(layout, style) for layout in range(10) for style in range(12)]
    return tuple(int(x) for x in np.random.default_rng(seed).choice(pairs))


def body_name(m, bid):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""


def check_kitchen(seed: int, robot: str, fps: int, n_resets: int, render: bool, resolution: int):
    env = RoboCasaEnv(
        task_name="CoffeePressButton",
        robot=robot,
        control_freq=fps,
        camera_name=CAMERA,
        render_mode="rgb_array" if render else "none",
        observation_width=resolution,
        observation_height=resolution,
        return_raw_obs=True,
        seed=seed,
    )
    inner = env._env
    m, d = inner.sim.model._model, inner.sim.data._data
    machine = inner.coffee_machine
    button = m.geom(f"{machine.naming_prefix}start_button").id

    contacts: dict[str, int] = {}
    image = None
    for k in range(n_resets):
        raw_obs, _ = env.reset()
        if k == 0 and render:
            image = np.ascontiguousarray(raw_obs[f"{CAMERA}_image"][::-1])
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = body_name(m, m.geom_bodyid[c.geom1]), body_name(m, m.geom_bodyid[c.geom2])
            ra, rb = a.startswith(ROBOT_PREFIXES), b.startswith(ROBOT_PREFIXES)
            if ra != rb and c.dist < 0.001:
                other = b if ra else a
                if "floor" in other:
                    continue
                key = f"{a if ra else b} <-> {other}"
                contacts[key] = contacts.get(key, 0) + 1

    btn = d.geom_xpos[button].copy()
    base_id = m.body("robot0_base").id
    rot, p = d.xmat[base_id].reshape(3, 3), d.xpos[base_id]
    btn_rel = rot.T @ (btn - p)

    cam = m.camera(CAMERA).id
    cam_pos, cam_rot = d.cam_xpos[cam], d.cam_xmat[cam].reshape(3, 3)
    pc = cam_rot.T @ (btn - cam_pos)
    u = v = float("nan")
    if pc[2] < 0:
        f = 0.5 / math.tan(math.radians(m.cam_fovy[cam]) / 2)
        u, v = 0.5 + f * pc[0] / -pc[2], 0.5 - f * pc[1] / -pc[2]
    ray = (btn - cam_pos) / np.linalg.norm(btn - cam_pos)
    geomid = np.array([-1], dtype=np.int32)
    mujoco.mj_ray(m, d, cam_pos + 0.05 * ray, ray, None, 1, -1, geomid)
    first = body_name(m, m.geom_bodyid[geomid[0]]) if geomid[0] >= 0 else ""

    reasons, flags = [], []
    if any(k.startswith("mobilebase0_") for k in contacts):
        reasons.append("mobile_base_in_fixture")
    if not (FRAME_MARGIN <= u <= 1 - FRAME_MARGIN and FRAME_MARGIN <= v <= 1 - FRAME_MARGIN):
        reasons.append(f"button_out_of_frame:uv=({u:.2f},{v:.2f})")
    if any(not k.startswith("mobilebase0_") for k in contacts):
        flags.append("hand_touches_env_at_reset")
    if not first.startswith(machine.naming_prefix):
        flags.append(f"button_occluded_in_agentview_left_by:{first}")

    info = {
        "scene_seed": seed,
        "layout_id": int(inner.layout_id),
        "style_id": int(inner.style_id),
        "coffee_machine_model": STYLE_MACHINE[int(inner.style_id)],
        "sft_layout": SFT_LAYOUTS[int(inner.layout_id)],
        "button_in_base_frame_m": np.round(btn_rel, 4).tolist(),
        "button_image_uv": [round(u, 3), round(v, 3)],
        "camera_ray_first_body": first,
        "robot_contacts_over_resets": {k: f"{n}/{n_resets}" for k, n in contacts.items()},
        "checks_failed": reasons,
        "flags": flags,
    }
    env.close()
    return info, image


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--robot", default="PandaDexLeapRHOmron")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--required_train_seeds", default="0,1", help="Kitchens of the existing DSRL runs")
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--n_heldout", type=int, default=2)
    parser.add_argument("--max_seed", type=int, default=200)
    parser.add_argument(
        "--exclude_seeds",
        default="",
        help="Construction seeds never to select (e.g. 2: L8S4, whose rendered start had the LEAP hand rotated by a wall cabinet)",
    )
    parser.add_argument("--n_resets", type=int, default=5, help="Resets per kitchen for the contact check")
    parser.add_argument("--out", required=True)
    parser.add_argument("--render_dir", default=None)
    parser.add_argument("--resolution", type=int, default=384)
    args = parser.parse_args()

    render = args.render_dir is not None
    required = [int(s) for s in args.required_train_seeds.split(",") if s.strip()]
    excluded = {int(s) for s in args.exclude_seeds.split(",") if s.strip()}
    # Balance machine models: fill every model to floor(n/3) first, then top up to ceil(n/3).
    caps = [args.n_train // 3, math.ceil(args.n_train / 3)]
    evaluated, images = {}, {}

    def evaluate(seed):
        if seed not in evaluated:
            info, img = check_kitchen(seed, args.robot, args.fps, args.n_resets, render, args.resolution)
            evaluated[seed], images[seed] = info, img
            flag = "PASS" if not info["checks_failed"] else info["checks_failed"]
            print(
                f"  seed {seed:3d} L{info['layout_id']}S{info['style_id']:<2} {info['coffee_machine_model']:20s} "
                f"btn(base)={info['button_in_base_frame_m']} -> {flag}  flags={info['flags']}"
            )
        return evaluated[seed]

    print("=== training kitchens")
    train, scenes, per_model = [], set(), {}
    for seed in required:
        info = evaluate(seed)
        if info["checks_failed"]:
            print(
                f"  !! required seed {seed} fails {info['checks_failed']} -- kept (the old runs trained here)"
            )
        train.append(info)
        scenes.add((info["layout_id"], info["style_id"]))
        per_model[info["coffee_machine_model"]] = per_model.get(info["coffee_machine_model"], 0) + 1
    for cap in caps:
        for seed in range(args.max_seed):
            if len(train) >= args.n_train:
                break
            if seed in required or seed in excluded or any(seed == t["scene_seed"] for t in train):
                continue
            scene = seed_to_scene(seed)
            model = STYLE_MACHINE[scene[1]]
            if scene in scenes or per_model.get(model, 0) >= cap:
                continue
            info = evaluate(seed)
            if info["checks_failed"]:
                continue
            train.append(info)
            scenes.add(scene)
            per_model[model] = per_model.get(model, 0) + 1
    train = train[: len(required)] + sorted(train[len(required) :], key=lambda t: t["scene_seed"])

    print("=== held-out kitchens")
    train_layouts = {t["layout_id"] for t in train}
    train_models = {t["coffee_machine_model"] for t in train}
    heldout, used_models = [], set()
    for strict_layout in (True, False):  # prefer layouts absent from training; relax only if needed
        for seed in range(args.max_seed):
            if len(heldout) >= args.n_heldout:
                break
            if seed in excluded or any(seed == t["scene_seed"] for t in train + heldout):
                continue
            scene = seed_to_scene(seed)
            model = STYLE_MACHINE[scene[1]]
            if scene in scenes or model not in train_models or model in used_models:
                continue
            if strict_layout and scene[0] in train_layouts:
                continue
            info = evaluate(seed)
            if info["checks_failed"]:
                continue
            info["heldout_layout_unseen_in_dsrl_training"] = scene[0] not in train_layouts
            heldout.append(info)
            used_models.add(model)
        if len(heldout) >= args.n_heldout:
            break

    print(
        f"\ntrain: {[(t['scene_seed'], t['layout_id'], t['style_id']) for t in train]}  per model {per_model}"
    )
    print(
        f"held-out: {[(h['scene_seed'], h['layout_id'], h['style_id'], h['coffee_machine_model']) for h in heldout]}"
    )

    if render:
        from PIL import Image, ImageDraw

        render_dir = Path(os.path.expanduser(args.render_dir))
        render_dir.mkdir(parents=True, exist_ok=True)
        tiles = []
        for group, items in (("TRAIN", train), ("HELD-OUT", heldout)):
            for info in items:
                img = Image.fromarray(images[info["scene_seed"]])
                img.save(render_dir / f"{group.lower()}_seed{info['scene_seed']}.png")
                draw = ImageDraw.Draw(img)
                label = (
                    f"{group} seed{info['scene_seed']} L{info['layout_id']}S{info['style_id']} "
                    f"{info['coffee_machine_model']}"
                )
                draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
                draw.text((6, 5), label, fill=(255, 60, 60) if group == "HELD-OUT" else (255, 255, 255))
                tiles.append(img)
        cols, (w, h) = 4, tiles[0].size
        sheet = Image.new("RGB", (cols * w, math.ceil(len(tiles) / cols) * h), (30, 30, 30))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * w, (i // cols) * h))
        sheet.save(render_dir / "kitchen_bank.png")
        print(f"rendered {len(tiles)} kitchens -> {render_dir / 'kitchen_bank.png'}")

    bank = {
        "task": "CoffeePressButton",
        "robot": args.robot,
        "fps": args.fps,
        "bank_type": "kitchens",
        "excluded_seeds": sorted(excluded),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "how_to_apply": "build RoboCasaEnv(seed=scene_seed) with no ep_meta -- the same construction as make_env",
        "train": train,
        "heldout": heldout,
        "all_evaluated": list(evaluated.values()),
    }
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
