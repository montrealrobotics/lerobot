"""Randomize RoboCasa start conditions that a construction seed would otherwise freeze.

``RoboCasaEnv`` uses ``hard_reset=False``, so the kitchen, the robot base and the task object's
placement are all sampled once at construction and restored on every reset. These wrappers vary
one of them per reset instead: a banked object placement, a pool of kitchens, or a re-roll of the
arm reset noise. Banks are JSON built by ``scripts/build_lamp_placement_bank.py`` and
``scripts/build_coffee_kitchen_bank.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


def load_placement_bank(path: str | os.PathLike) -> dict[str, Any]:
    bank = json.loads(Path(os.path.expanduser(str(path))).read_text())
    bank.setdefault("object_name", "lamp")  # banks written before the field existed are lamp banks
    bank["_scenes_by_seed"] = {int(s["scene_seed"]): s for s in bank["scenes"]}
    return bank


def scene_entries(bank: dict[str, Any], scene_seed: int, group: str) -> list[dict[str, Any]]:
    """Entries of one scene for ``group`` in {"train", "heldout", "reference"}."""
    try:
        scene = bank["_scenes_by_seed"][scene_seed]
    except KeyError as err:
        raise KeyError(
            f"Placement bank has no scene for construction seed {scene_seed}; "
            f"available: {sorted(bank['_scenes_by_seed'])}"
        ) from err
    if group == "reference":
        ref = dict(scene["reference_training_placement"])
        ref.setdefault("id", f"s{scene_seed}_reference")
        return [ref]
    return list(scene[group])


# ── Eval cells (shared by the DSRL example and scripts/eval_sft_on_banks.py) ──────────

# The two kitchens every pre-kitchen-bank DSRL run collected in (construction seeds 0 and 1).
FIXED_RUN_KITCHEN_SEEDS = (0, 1)


def kitchen_eval_cells(bank: dict, spec: str) -> list[dict]:
    """One cell per kitchen. ``spec`` groups: ``heldout``, ``reference`` (the fixed-kitchen runs'
    seeds 0,1, also in the bank's training set), ``train`` / ``train:N``.
    """
    cells = []
    for token in [t.strip() for t in spec.split(",") if t.strip()]:
        group, _, limit = token.partition(":")
        if group == "heldout":
            kitchens = bank["heldout"]
        elif group == "reference":
            kitchens = [k for k in bank["train"] if k["scene_seed"] in FIXED_RUN_KITCHEN_SEEDS]
        elif group == "train":
            kitchens = bank["train"]
        else:
            raise ValueError(f"--eval_kitchen_sets: unknown group {group!r} (heldout|reference|train[:N])")
        if limit:
            kitchens = kitchens[: int(limit)]
        for k in kitchens:
            label = f"{group}_seed{k['scene_seed']}_L{k['layout_id']}S{k['style_id']}"
            cells.append(
                {"seed": k["scene_seed"], "scene": None, "entry": None, "label": label, "group": group}
            )
    if not cells:
        raise ValueError(f"--eval_kitchen_sets {spec!r} selected no kitchens")
    return cells


def placement_eval_cells(bank: dict, spec: str, scene_seeds: list[int]) -> list[dict]:
    """One cell per (kitchen, entry). ``spec`` groups: ``heldout``, ``reference`` (the original
    fixed placement every pre-bank DSRL run trained on, itself held out of the bank), ``train`` /
    ``train:N``.
    """
    cells = []
    for token in [t.strip() for t in spec.split(",") if t.strip()]:
        group, _, limit = token.partition(":")
        if group not in ("heldout", "reference", "train"):
            raise ValueError(f"--eval_placement_sets: unknown group {group!r} (heldout|reference|train[:N])")
        for seed in scene_seeds:
            entries = scene_entries(bank, seed, group)
            if limit:
                entries = entries[: int(limit)]
            for entry in entries:
                cells.append(
                    {"seed": seed, "scene": None, "entry": entry, "label": entry["id"], "group": group}
                )
    if not cells:
        raise ValueError(f"--eval_placement_sets {spec!r} selected no placements")
    return cells


def _robocasa_inner(env: gym.Env):
    """The robosuite/RoboCasa kitchen env inside lerobot's ``RoboCasaEnv`` gym wrapper."""
    return env.unwrapped._env


class PlacementBankWrapper(gym.Wrapper):
    """On every ``reset()``, place the task object at the next bank entry.

    ``shuffle`` reshuffles the entries each pass rather than cycling in a fixed order.
    ``check_tol_m`` raises if the object comes to rest farther than this (xy) from the requested
    pose, i.e. the placement did not take or it slid.
    """

    def __init__(
        self,
        env: gym.Env,
        bank: dict[str, Any],
        scene_seed: int,
        entries: list[dict[str, Any]],
        shuffle: bool = True,
        seed: int = 0,
        check_tol_m: float = 0.03,
    ):
        super().__init__(env)
        if not entries:
            raise ValueError("PlacementBankWrapper needs at least one entry")
        self.bank = bank
        self.object_name = bank["object_name"]
        self.entries = entries
        self.shuffle = shuffle
        self.check_tol_m = check_tol_m
        self._rng = np.random.default_rng(seed)
        self._order: list[int] = []
        self.current: dict[str, Any] | None = None

        inner = _robocasa_inner(env)
        scene = bank["_scenes_by_seed"][scene_seed]
        got = (int(inner.layout_id), int(inner.style_id))
        want = (int(scene["layout_id"]), int(scene["style_id"]))
        if got != want:
            raise RuntimeError(
                f"Env built with seed {scene_seed} is kitchen L{got[0]}S{got[1]}, but the bank was "
                f"built for L{want[0]}S{want[1]}. The seed->kitchen mapping changed; rebuild the bank."
            )
        if self.object_name not in inner.object_placements:
            raise KeyError(
                f"Object {self.object_name!r} not in env.object_placements: {list(inner.object_placements)}"
            )

    def _next_entry(self) -> dict[str, Any]:
        if not self._order:
            idx = np.arange(len(self.entries))
            self._order = list(self._rng.permutation(idx) if self.shuffle else idx)
        return self.entries[self._order.pop(0)]

    def reset(self, **kwargs):
        entry = self._next_entry()
        inner = _robocasa_inner(self.env)
        obj = inner.object_placements[self.object_name][2]
        inner.object_placements[self.object_name] = (tuple(entry["pos"]), tuple(entry["quat_wxyz"]), obj)
        self.current = entry

        obs, info = self.env.reset(**kwargs)

        rest = inner.sim.data.body_xpos[inner.obj_body_id[self.object_name]]
        drift = float(np.linalg.norm(np.asarray(rest[:2]) - np.asarray(entry["pos"][:2])))
        if drift > self.check_tol_m:
            raise RuntimeError(
                f"Placement {entry.get('id')} did not take: {self.object_name} rests {drift * 100:.1f} cm "
                f"from the requested pose (tolerance {self.check_tol_m * 100:.0f} cm)."
            )
        return obs, info


class KitchenPoolEnv(gym.Env):
    """One collect slot that switches between several pre-built kitchens on reset.

    Rebuilding a kitchen takes far longer than an episode, so every kitchen stays resident
    (~3.4 GB each) and ``reset()`` hands control to the next one in a reshuffled order. Kitchens
    are built exactly as ``make_env`` builds a construction seed. The pool presents the active
    kitchen's gym API plus the attributes the DSRL wrapper reads via ``VectorEnv.call``.
    """

    def __init__(self, make_kitchen, seeds: list[int], shuffle: bool = True, seed: int = 0):
        super().__init__()
        if not seeds:
            raise ValueError("KitchenPoolEnv needs at least one kitchen seed")
        self.seeds = list(seeds)
        self.kitchens = [make_kitchen(s) for s in self.seeds]
        spaces = {(str(k.observation_space), str(k.action_space)) for k in self.kitchens}
        if len(spaces) != 1:
            raise ValueError(f"Kitchens in one pool must share observation/action spaces, got {spaces}")
        self.observation_space = self.kitchens[0].observation_space
        self.action_space = self.kitchens[0].action_space
        self.metadata = self.kitchens[0].metadata
        self.render_mode = getattr(self.kitchens[0], "render_mode", None)
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self._order: list[int] = []
        self._active = 0

    @property
    def active_seed(self) -> int:
        return self.seeds[self._active]

    @property
    def task_description(self):
        return self.kitchens[self._active].unwrapped.task_description

    @property
    def _max_episode_steps(self):
        return self.kitchens[self._active].unwrapped._max_episode_steps

    def reset(self, *, seed=None, options=None):
        if not self._order:
            idx = np.arange(len(self.kitchens))
            self._order = list(self._rng.permutation(idx) if self.shuffle else idx)
        self._active = int(self._order.pop(0))
        return self.kitchens[self._active].reset(seed=seed)

    def step(self, action):
        return self.kitchens[self._active].step(action)

    def render(self):
        return self.kitchens[self._active].render()

    def close(self):
        for k in self.kitchens:
            k.close()


def _eef_pose_in_base(inner) -> tuple[np.ndarray, np.ndarray]:
    """Right end-effector site position and rotation, expressed in the robot base frame."""
    model, data = inner.sim.model._model, inner.sim.data._data
    robot = inner.robots[0]
    base = model.body("robot0_base").id
    rot_base, pos_base = data.xmat[base].reshape(3, 3), data.xpos[base]
    site = robot.eef_site_id["right"]
    return rot_base.T @ (data.site_xpos[site] - pos_base), rot_base.T @ data.site_xmat[site].reshape(3, 3)


def _rotation_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    cos = (np.trace(rot_a.T @ rot_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class StartPoseRejectionWrapper(gym.Wrapper):
    """Re-reset when the arm reset noise leaves the hand jammed against the scene.

    RoboCasa perturbs the arm joints on every reset (``initialization_noise`` 0.02 rad); under a
    wall cabinet some draws jam the fingertips and rotate the hand by tens of degrees. The
    noise-free settled end-effector pose is measured once per kitchen; resets deviating from it
    by more than ``max_rot_deg`` / ``max_pos_m`` are redrawn. Ordinary noise (~2 cm / ~5 deg)
    passes. Wrap the raw env *inside* any placement wrapper so retries keep the same placement.
    """

    def __init__(
        self, env: gym.Env, max_rot_deg: float = 10.0, max_pos_m: float = 0.03, max_retries: int = 10
    ):
        super().__init__(env)
        self.max_rot_deg = max_rot_deg
        self.max_pos_m = max_pos_m
        self.max_retries = max_retries
        self.n_resets = 0
        self.n_rejected = 0
        self.n_exhausted = 0

        inner = env.unwrapped._env
        robot = inner.robots[0]
        magnitude = robot.initialization_noise["magnitude"]
        robot.initialization_noise["magnitude"] = 0.0
        try:
            env.reset()
            self.nominal_pos, self.nominal_rot = _eef_pose_in_base(inner)
        finally:
            robot.initialization_noise["magnitude"] = magnitude

    def deviation(self) -> tuple[float, float]:
        pos, rot = _eef_pose_in_base(self.env.unwrapped._env)
        return float(np.linalg.norm(pos - self.nominal_pos)), _rotation_angle_deg(self.nominal_rot, rot)

    def reset(self, **kwargs):
        self.n_resets += 1
        for _attempt in range(self.max_retries + 1):
            obs, info = self.env.reset(**kwargs)
            pos_dev, rot_dev = self.deviation()
            if pos_dev <= self.max_pos_m and rot_dev <= self.max_rot_deg:
                return obs, info
            self.n_rejected += 1
        self.n_exhausted += 1
        inner = self.env.unwrapped._env
        print(
            f"[StartPoseRejection] L{inner.layout_id}S{inner.style_id}: no clean start in "
            f"{self.max_retries + 1} resets (last: {pos_dev * 100:.1f} cm, {rot_dev:.1f} deg); using it anyway"
        )
        return obs, info
