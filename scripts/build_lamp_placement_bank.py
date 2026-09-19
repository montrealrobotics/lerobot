#!/usr/bin/env python
"""Build a feasibility-checked bank of ScrewLightbulb lamp placements.

Keeps each training kitchen (construction seeds 0,1) and its robot base fixed, and draws
alternative lamp poses from the task's own placement sampler. A candidate is applied by
overwriting ``env.object_placements["lamp"]`` before ``reset()``, so RoboCasa settles and welds
it as usual. The sampler only rejects bounding-box overlap with fixtures, so each candidate is
also checked in sim (see CHECKS): spawn collisions, settling, rest contacts, the screw joint,
overhead clearance for the hand, and visibility from robot0_agentview_left.

Accepted poses are split into a training bank and held-out poses with a minimum xy spacing.
``--render_dir`` renders every entry from robot0_agentview_left (needs EGL).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="numba_cache_"))

import mujoco
import numpy as np
import robosuite

from lerobot.envs.robocasa_env import RoboCasaEnv

CHECKS = {
    "spawn_contact_tol_m": 0.002,  # non-support contact closer than this at spawn = collision
    "support_penetration_max_m": 0.005,  # spawned sunk into the counter by more than this
    "settle_xy_drift_max_m": 0.02,
    "settle_tilt_max_deg": 5.0,
    "settle_z_drop_max_m": 0.03,  # spawn is ~1 cm above the counter, so ~1 cm drop is normal
    "screw_turns_max": 0.05,
    "overhead_clearance_min_m": 0.25,  # bulb centre to first geom straight up
    "frame_margin_frac": 0.08,  # bulb must project this far inside the image border
    "occlusion_tol_m": 0.06,  # camera ray may stop this short of the bulb centre (bulb radius)
}
CAMERA = "robot0_agentview_left"


def controller_path(robot: str) -> str | None:
    if "XArm6" not in robot:
        return None
    return os.path.join(
        os.path.dirname(robosuite.__file__),
        "controllers/config/robots/default_xarm6dexleaprhomron_joint_pos.json",
    )


def body_name(model, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""


class LampScene:
    """One fixed training kitchen (construction seed) plus helpers to test lamp poses."""

    def __init__(self, task, robot, fps, scene_seed, render, resolution):
        self.env = RoboCasaEnv(
            task_name=task,
            robot=robot,
            controller=controller_path(robot),
            control_freq=fps,
            camera_name=CAMERA,
            render_mode="rgb_array" if render else "none",
            observation_width=resolution,
            observation_height=resolution,
            return_raw_obs=True,
            seed=scene_seed,
            num_steps_wait=10,
        )
        self.inner = self.env._env
        self.model = self.inner.sim.model._model
        self.data = self.inner.sim.data._data
        self.lamp = self.inner.objects["lamp"]
        self.lamp_body = self.inner.obj_body_id["lamp"]
        self.lamp_prefix = self.lamp.naming_prefix
        self.support = self.inner.fixture_refs["counter"].name
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
        self.bulb_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lamp_bulb_center")
        self.root_joint = self.inner._get_obj_root_joint(self.lamp)
        self.reference = self.inner.object_placements["lamp"]  # the fixed training placement

    # ── placement plumbing ────────────────────────────────────────────────────────────
    def sample_candidate(self, rng: np.random.Generator):
        """One draw from the task's own sampler, with a private rng for reproducibility."""
        sampler = self.inner.placement_initializer
        for s in [sampler, *getattr(sampler, "samplers", {}).values()]:
            s.rng = rng
        placements = sampler.sample(placed_objects=self.inner.fxtr_placements)
        pos, quat, _ = placements["lamp"]
        return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)

    def apply_and_reset(self, pos, quat):
        obj = self.inner.object_placements["lamp"][2]
        self.inner.object_placements["lamp"] = (tuple(pos), tuple(quat), obj)
        raw_obs, _ = self.env.reset()
        return raw_obs

    # ── checks ────────────────────────────────────────────────────────────────────────
    def _is_lamp(self, geom_id: int) -> bool:
        return body_name(self.model, self.model.geom_bodyid[geom_id]).startswith(self.lamp_prefix)

    def _lamp_contacts(self):
        """[(partner_body_name, dist)] for every contact with exactly one lamp geom."""
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            a, b = self._is_lamp(c.geom1), self._is_lamp(c.geom2)
            if a == b:
                continue
            partner = c.geom2 if a else c.geom1
            out.append((body_name(self.model, self.model.geom_bodyid[partner]), float(c.dist)))
        return out

    def spawn_check(self, pos, quat):
        """Contacts at the exact spawn pose, before any settling."""
        qadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.root_joint)
        ]
        saved = self.data.qpos.copy()
        self.data.qpos[qadr : qadr + 7] = np.concatenate([pos, quat])
        mujoco.mj_forward(self.model, self.data)
        contacts = self._lamp_contacts()
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)

        collisions = [
            (n, d)
            for n, d in contacts
            if not n.startswith(self.support) and d < CHECKS["spawn_contact_tol_m"]
        ]
        sunk = [
            (n, d)
            for n, d in contacts
            if n.startswith(self.support) and d < -CHECKS["support_penetration_max_m"]
        ]
        return collisions, sunk

    def rest_metrics(self, spawn_pos):
        d, m = self.data, self.model
        root_pos = d.xpos[self.lamp_body].copy()
        z_axis = d.xmat[self.lamp_body].reshape(3, 3)[:, 2]
        bulb = d.site_xpos[self.bulb_site].copy()

        # overhead clearance: start above the bulb so the lamp's own geoms are not hit
        start = bulb + np.array([0.0, 0.0, 0.08])
        geomid = np.zeros(1, dtype=np.int32)
        hit = mujoco.mj_ray(m, d, start, np.array([0.0, 0.0, 1.0]), None, 1, -1, geomid)
        overhead = float("inf") if hit < 0 else 0.08 + float(hit)

        # projection into the camera image (MuJoCo cameras look down their local -z)
        cam_pos = d.cam_xpos[self.cam_id]
        cam_rot = d.cam_xmat[self.cam_id].reshape(3, 3)
        p = cam_rot.T @ (bulb - cam_pos)
        fovy = np.radians(m.cam_fovy[self.cam_id])
        in_front = p[2] < 0
        u = v = float("nan")
        if in_front:
            f = 0.5 / np.tan(fovy / 2)  # normalised focal length (image height = 1)
            u = 0.5 + f * p[0] / -p[2]
            v = 0.5 - f * p[1] / -p[2]  # 0 = top row, matching the flipped policy images

        # occlusion: does a ray from the camera reach the lamp first?
        to_bulb = bulb - cam_pos
        dist_to_bulb = float(np.linalg.norm(to_bulb))
        ray_dir = to_bulb / dist_to_bulb
        origin = cam_pos + 0.05 * ray_dir  # step off any robot geometry the camera sits in
        geomid[:] = -1
        hit = mujoco.mj_ray(m, d, origin, ray_dir, None, 1, -1, geomid)
        first_body = body_name(m, m.geom_bodyid[geomid[0]]) if geomid[0] >= 0 else ""
        occluded = not (
            first_body.startswith(self.lamp_prefix)
            or (hit >= 0 and 0.05 + hit >= dist_to_bulb - CHECKS["occlusion_tol_m"])
        )

        return {
            "rest_pos": root_pos.tolist(),
            "bulb_pos": bulb.tolist(),
            "xy_drift_m": float(np.linalg.norm(root_pos[:2] - spawn_pos[:2])),
            "z_drop_m": float(spawn_pos[2] - root_pos[2]),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(z_axis[2], -1, 1)))),
            "screw_turns": float(self.inner.get_screw_state()["turns"]),
            "rest_contact_bodies": sorted({n for n, _ in self._lamp_contacts()}),
            "overhead_clearance_m": overhead,
            "image_uv": [u, v],
            "camera_ray_first_body": first_body,
            "occluded": bool(occluded),
            "cam_to_bulb_m": dist_to_bulb,
        }

    def evaluate(self, pos, quat):
        collisions, sunk = self.spawn_check(pos, quat)
        raw_obs = self.apply_and_reset(pos, quat)
        rest = self.rest_metrics(pos)

        margin = CHECKS["frame_margin_frac"]
        u, v = rest["image_uv"]
        bad_rest = [n for n in rest["rest_contact_bodies"] if not n.startswith(self.support)]
        reasons = []
        if collisions:
            reasons.append(f"spawn_collision:{sorted({n for n, _ in collisions})}")
        if sunk:
            reasons.append("spawn_sunk_in_counter")
        if rest["xy_drift_m"] > CHECKS["settle_xy_drift_max_m"]:
            reasons.append(f"slid:{rest['xy_drift_m']:.3f}m")
        if rest["tilt_deg"] > CHECKS["settle_tilt_max_deg"]:
            reasons.append(f"tilted:{rest['tilt_deg']:.1f}deg")
        if rest["z_drop_m"] > CHECKS["settle_z_drop_max_m"]:
            reasons.append(f"fell:{rest['z_drop_m']:.3f}m")
        if bad_rest:
            reasons.append(f"rest_contact:{bad_rest}")
        if rest["screw_turns"] > CHECKS["screw_turns_max"]:
            reasons.append(f"prescrewed:{rest['screw_turns']:.3f}")
        if rest["overhead_clearance_m"] < CHECKS["overhead_clearance_min_m"]:
            reasons.append(f"low_clearance:{rest['overhead_clearance_m']:.3f}m")
        if not (margin <= u <= 1 - margin and margin <= v <= 1 - margin):
            reasons.append(f"out_of_frame:uv=({u:.2f},{v:.2f})")
        if rest["occluded"]:
            reasons.append(f"occluded_by:{rest['camera_ray_first_body']}")
        return reasons, rest, raw_obs


def yaw_deg(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def pick_spaced(pool, anchors, n, min_sep):
    """Greedy pick of n entries whose xy is at least min_sep from anchors and each other."""
    chosen = []
    for cand in pool:
        xy = np.asarray(cand["pos"][:2])
        if all(
            np.linalg.norm(xy - np.asarray(a[:2])) >= min_sep for a in anchors + [c["pos"] for c in chosen]
        ):
            chosen.append(cand)
            if len(chosen) == n:
                break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default="ScrewLightbulb")
    parser.add_argument("--robot", default="XArm6DexLeapRHOmron")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scene_seeds", default="0,1", help="Construction seeds = DSRL collect kitchens")
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--n_heldout", type=int, default=2)
    parser.add_argument("--n_candidates", type=int, default=60, help="Sampler draws evaluated per scene")
    parser.add_argument("--train_min_sep_m", type=float, default=0.04)
    parser.add_argument("--heldout_min_sep_m", type=float, default=0.06)
    parser.add_argument("--bank_seed", type=int, default=20260912)
    parser.add_argument("--out", required=True, help="Output bank JSON")
    parser.add_argument("--render_dir", default=None, help="Render bank entries here (needs EGL)")
    parser.add_argument("--resolution", type=int, default=384)
    args = parser.parse_args()

    render = args.render_dir is not None
    scenes_out = []
    for scene_seed in [int(s) for s in args.scene_seeds.split(",")]:
        scene = LampScene(args.task, args.robot, args.fps, scene_seed, render, args.resolution)
        inner = scene.inner
        print(
            f"\n=== scene seed {scene_seed}: layout={inner.layout_id} style={inner.style_id} "
            f"support={scene.support}"
        )

        ref_pos, ref_quat = np.asarray(scene.reference[0]), np.asarray(scene.reference[1])
        ref_reasons, ref_rest, _ = scene.evaluate(ref_pos, ref_quat)
        print(f"  reference (fixed training placement): {ref_reasons or 'PASS'}  {_summary(ref_rest)}")

        rng = np.random.default_rng([args.bank_seed, scene_seed])
        accepted, rejected = [], []
        for k in range(args.n_candidates):
            try:
                pos, quat = scene.sample_candidate(rng)
            except Exception as exc:  # noqa: BLE001 — RandomizationError: sampler found no spot
                rejected.append({"k": k, "reasons": [f"sampler:{type(exc).__name__}"]})
                continue
            reasons, rest, _ = scene.evaluate(pos, quat)
            entry = {
                "k": k,
                "pos": pos.tolist(),
                "quat_wxyz": quat.tolist(),
                "yaw_deg": yaw_deg(quat),
                "metrics": rest,
            }
            if reasons:
                entry["reasons"] = reasons
                rejected.append(entry)
            else:
                accepted.append(entry)
            print(f"  cand {k:02d}: {reasons if reasons else 'PASS'}  {_summary(rest)}")

        train = pick_spaced(accepted, [ref_pos.tolist()], args.n_train, args.train_min_sep_m)
        train_ids = {c["k"] for c in train}
        heldout = pick_spaced(
            [c for c in accepted if c["k"] not in train_ids],
            [ref_pos.tolist()] + [c["pos"] for c in train],
            args.n_heldout,
            args.heldout_min_sep_m,
        )
        for i, c in enumerate(train):
            c["id"] = f"s{scene_seed}_train_{i:02d}"
        for i, c in enumerate(heldout):
            c["id"] = f"s{scene_seed}_heldout_{i:02d}"

        reason_counts: dict[str, int] = {}
        for r in rejected:
            for reason in r["reasons"]:
                key = reason.split(":")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
        print(
            f"  -> {len(accepted)}/{args.n_candidates} passed; picked {len(train)} train, "
            f"{len(heldout)} held-out; rejections {reason_counts}"
        )
        if len(train) < args.n_train or len(heldout) < args.n_heldout:
            print("  !! bank is short -- raise --n_candidates or relax the spacing")

        if render:
            _render_scene(scene, scene_seed, ref_pos, ref_quat, train, heldout, Path(args.render_dir))

        scenes_out.append(
            {
                "scene_seed": scene_seed,
                "layout_id": int(inner.layout_id),
                "style_id": int(inner.style_id),
                "support_fixture": scene.support,
                "robot_base_pos": inner.sim.data.get_body_xpos("robot0_base").tolist(),
                "lamp_mjcf": inner.object_cfgs[0]["info"]["mjcf_path"],
                "reference_training_placement": {
                    "pos": ref_pos.tolist(),
                    "quat_wxyz": ref_quat.tolist(),
                    "yaw_deg": yaw_deg(ref_quat),
                    "checks_failed": ref_reasons,
                    "metrics": ref_rest,
                },
                "train": train,
                "heldout": heldout,
                "n_candidates": args.n_candidates,
                "n_accepted": len(accepted),
                "rejection_counts": reason_counts,
                "rejected": rejected,
            }
        )
        scene.env.close()

    bank = {
        "task": args.task,
        "robot": args.robot,
        "object_name": "lamp",
        "controller": controller_path(args.robot),
        "fps": args.fps,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "bank_seed": args.bank_seed,
        "quat_convention": "wxyz (MuJoCo free joint), as stored in env.object_placements",
        "how_to_apply": "env._env.object_placements['lamp'] = (pos, quat_wxyz, obj); env.reset()",
        "checks": CHECKS,
        "spacing_m": {"train": args.train_min_sep_m, "heldout": args.heldout_min_sep_m},
        "scenes": scenes_out,
    }
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, indent=1))
    print(f"\nwrote {out}")


def _summary(rest) -> str:
    u, v = rest["image_uv"]
    return (
        f"drift={rest['xy_drift_m'] * 100:.1f}cm tilt={rest['tilt_deg']:.1f}° "
        f"overhead={rest['overhead_clearance_m']:.2f}m uv=({u:.2f},{v:.2f}) "
        f"ray={rest['camera_ray_first_body'] or '-'}"
    )


def _render_scene(scene, scene_seed, ref_pos, ref_quat, train, heldout, render_dir: Path):
    from PIL import Image, ImageDraw

    render_dir.mkdir(parents=True, exist_ok=True)
    key = f"{CAMERA}_image"
    items = [("REF (fixed training)", ref_pos, ref_quat)]
    items += [(c["id"], np.asarray(c["pos"]), np.asarray(c["quat_wxyz"])) for c in train]
    items += [(c["id"] + " [HELD-OUT]", np.asarray(c["pos"]), np.asarray(c["quat_wxyz"])) for c in heldout]

    tiles = []
    for label, pos, quat in items:
        raw_obs = scene.apply_and_reset(pos, quat)
        img = Image.fromarray(np.ascontiguousarray(raw_obs[key][::-1]))
        img.save(render_dir / f"seed{scene_seed}_{label.split(' ')[0]}.png")
        draw = ImageDraw.Draw(img)
        color = (
            (255, 60, 60)
            if "HELD-OUT" in label
            else (255, 255, 0)
            if label.startswith("REF")
            else (255, 255, 255)
        )
        draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
        draw.text((6, 5), label, fill=color)
        tiles.append(img)

    cols = 5
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (30, 30, 30))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    path = render_dir / f"bank_seed{scene_seed}_L{scene.inner.layout_id}S{scene.inner.style_id}.png"
    sheet.save(path)
    print(f"  rendered {len(tiles)} placements -> {path}")


if __name__ == "__main__":
    main()
