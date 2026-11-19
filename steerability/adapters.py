"""Environment adapters.

`RobocasaAdapter` is the real one. It needs two hooks from your codebase (see README):

  --env-factory module:fn   fn(task, cell, seed) -> robosuite/robocasa env whose reset() always
                            lands in `cell` (bank start / kitchen) and applies the same start-pose
                            rejection as DSRL eval for coffee. It must return robosuite-format obs
                            dicts (no gym wrapper that rewrites observations), because after a
                            state restore obs are regenerated with env._get_observations().
  --obs-fn module:fn        fn(task, robosuite_obs_dict) -> batch dict (leading dim 1) that the
                            checkpoint's preprocessor accepts -- i.e. the same frame the frozen
                            pi05 sees inside DSRLEnvWrapper (2 cams + state + prompt).

`MockAdapter` + `MockPolicy` exercise every script without a GPU, MuJoCo or lerobot.
"""
from __future__ import annotations

import hashlib
import warnings

import os

import numpy as np

from common import TASK_SPECS, load_callable


class EnvAdapter:
    image_keys: list[str] = []
    third_person_key: str = ""
    wrist_key: str = ""

    def __init__(self, task: str):
        self.task = task
        self.spec = TASK_SPECS[task]

    # --- obs helpers shared by all adapters
    def snapshot_obs(self, raw_obs: dict) -> dict:
        return {k: np.array(v, copy=True) for k, v in raw_obs.items() if isinstance(v, (np.ndarray, float, int))}

    def perturb_obs(self, raw_obs: dict, kind: str, donor: dict | None = None) -> dict:
        o = {k: np.array(v, copy=True) for k, v in raw_obs.items()}
        if kind == "blank_third":
            o[self.third_person_key] = np.zeros_like(o[self.third_person_key])
        elif kind == "blank_wrist":
            o[self.wrist_key] = np.zeros_like(o[self.wrist_key])
        elif kind in ("shuffle_third", "swap_cell_third"):
            o[self.third_person_key] = np.array(donor[self.third_person_key], copy=True)
        elif kind == "swap_cell_state":
            for k, v in donor.items():
                if k not in self.image_keys:
                    o[k] = np.array(v, copy=True)
        else:
            raise ValueError(kind)
        return o


# ============================================================================ RoboCasa

try:
    import mujoco
    _MJ_FULL = mujoco.mjtState.mjSTATE_INTEGRATION
except ImportError:  # mock backend only
    mujoco = _MJ_FULL = None

def _unwrap(env):
    e = env
    for _ in range(20):
        if hasattr(e, "sim"):
            return e
        e = getattr(e, "env", None) or getattr(e, "unwrapped", None)
        if e is None:
            break
    raise RuntimeError("could not find a robosuite env with .sim inside the factory's env")


class RobocasaAdapter(EnvAdapter):
    image_keys = ["robot0_agentview_left_image", "robot0_eye_in_hand_image"]
    third_person_key = "robot0_agentview_left_image"
    wrist_key = "robot0_eye_in_hand_image"

    def __init__(self, task, env_factory: str, obs_fn: str):
        super().__init__(task)
        if not env_factory or not obs_fn:
            raise SystemExit("robocasa adapter needs --env-factory and --obs-fn (see README: 'Wiring')")
        self._factory = load_callable(env_factory)
        self.obs_fn = load_callable(obs_fn)
        self._warned_ctrl = False

    def make_env(self, cell, seed):
        return self._factory(task=self.task, cell=cell, seed=seed)

    def reset(self, env, seed):
        try:
            out = env.reset(seed=seed)
        except TypeError:
            np.random.seed(seed)
            out = env.reset()
        return out[0] if isinstance(out, tuple) else out

    def step(self, env, action):
        out = env.step(np.asarray(action, dtype=np.float64))
        if len(out) == 5:
            obs, _r, term, trunc, info = out
            done = bool(term or trunc)
        else:
            obs, _r, done, info = out
        raw = _unwrap(env)
        success = bool(info.get("is_success", info.get("success", False))) if isinstance(info, dict) else False
        if not success and hasattr(raw, "_check_success"):
            success = bool(raw._check_success())
        return obs, success, bool(done)

    def max_steps(self, env):
        return int(self.spec.get("max_steps") or getattr(_unwrap(env), "horizon", 1000))

    def get_state(self, env):
        """Full MuJoCo integration state (qpos/qvel plus act, ctrl, mocap, warmstart: sim.get_state
        drops these), the env's clocks, and the fixtures' Python-side flags (e.g. the coffee
        machine's sticky `_turned_on`, which `_check_success` reads)."""
        raw = _unwrap(env)
        m, d = raw.sim.model._model, raw.sim.data._data
        flat = np.empty(mujoco.mj_stateSize(m, _MJ_FULL))
        mujoco.mj_getState(m, d, flat, _MJ_FULL)
        fixtures = {name: {k: v for k, v in vars(f).items() if isinstance(v, bool)}
                    for name, f in getattr(raw, "fixtures", {}).items()}
        return dict(flat=flat, timestep=int(getattr(raw, "timestep", 0)),
                    cur_time=float(getattr(raw, "cur_time", 0.0)),
                    # model-side constraint data (the lamp base weld is written into model.eq_data at runtime) and
                    # the lamp task's "anchor on next step" flag: neither is part of the MuJoCo integration state
                    eq_data=np.array(raw.sim.model._model.eq_data),
                    needs_anchor=getattr(raw, "_needs_anchor", None),
                    gym_steps=getattr(getattr(env, "unwrapped", None), "_step_count", None),
                    fixtures=fixtures)

    def set_state(self, env, state):
        raw = _unwrap(env)
        m, d = raw.sim.model._model, raw.sim.data._data
        mujoco.mj_setState(m, d, np.asarray(state["flat"], dtype=np.float64), _MJ_FULL)
        if state.get("eq_data") is not None:
            m.eq_data[:] = state["eq_data"]
        raw.sim.forward()
        if hasattr(raw, "_needs_anchor"):  # lamp: base weld written into the model on the episode's first step
            eq_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, raw.objects["lamp"].naming_prefix + "base_anchor")
            if state.get("needs_anchor") is not None:
                raw._needs_anchor = state["needs_anchor"]
            elif eq_id != -1 and d.eq_active[eq_id]:
                # weld on in the snapshot: keep the model's weld data as this env first anchored it (re-pinning on
                # every restore adds float noise); anchor once if this env never has
                raw._needs_anchor = False
                if not getattr(raw, "_steer_anchored", False):
                    raw._anchor_base()
            else:  # snapshot taken before the episode's first step: anchor on the next step, as the original did
                raw._needs_anchor = bool(getattr(raw, "fix_base", True))
            raw._steer_anchored = True
        if hasattr(raw, "timestep"):
            raw.timestep = state["timestep"]
        if hasattr(raw, "cur_time"):
            raw.cur_time = state.get("cur_time", raw.cur_time)
        if hasattr(raw, "done"):
            raw.done = False
        base = getattr(env, "unwrapped", None)
        if state.get("gym_steps") is not None and hasattr(base, "_step_count"):
            base._step_count = state["gym_steps"]  # the gym wrapper's truncation counter
        fixtures = getattr(raw, "fixtures", {})
        for name, flags in state.get("fixtures", {}).items():
            for k, v in flags.items():
                setattr(fixtures[name], k, v)
            fixtures[name].update_state(raw)  # re-derive what the flags drive (e.g. coffee liquid site alpha)
        self._reset_controllers(raw)
        # robosuite observables keep sampling timers, delays and their last observed value outside the sim, so a
        # restore alone leaves each branch with whatever the previous branch left (measured 2026-10-05: lamp
        # policy rollouts from one restored state diverged by ~1 in ctrl while env-only replays matched to 1e-10).
        # Reset them all (Observable.reset() leaves `_sampled` set) so every branch starts in the same phase; the
        # forced update below then refills every value from the restored sim.
        for ob in getattr(raw, "_observables", {}).values():
            ob.reset()
            ob._sampled = False
        if os.environ.get("STEER_DOUBLE_RENDER") == "1":  # diagnostic: discard a possibly stale first render
            raw._get_observations(force_update=True)
            for ob in getattr(raw, "_observables", {}).values():
                ob.reset()
                ob._sampled = False
        return raw._get_observations(force_update=True)

    def _reset_controllers(self, raw):
        """Controllers cache the robot state and hold goals/slew setpoints that a sim restore does
        not touch. Refresh each part controller from the restored sim, then re-anchor its goal there
        (OSC keeps its configured goal-update mode). Verify with p2_branching.py --check-determinism."""
        ok = False
        for r in getattr(raw, "robots", []):
            comp = getattr(r, "composite_controller", None)
            if comp is None:
                continue
            comp.update_state()
            for c in comp.part_controllers.values():
                c.update(force=True)
                mode = getattr(c, "_goal_update_mode", None)
                if mode is not None:
                    c.reset_goal(goal_update_mode=mode)
                else:
                    c.reset_goal()
                ok = True
        if not ok and not self._warned_ctrl:
            warnings.warn("no controller reset method found; check determinism after restore")
            self._warned_ctrl = True

    def probe_targets(self, env):
        """EE pos/quat + positions of every free-joint body (lamp/bulb/mug...): generic targets
        for the linear probe, no task-specific object names needed."""
        raw = _unwrap(env)
        obs = raw._get_observations()
        parts = [np.asarray(obs.get("robot0_eef_pos", np.zeros(3))), np.asarray(obs.get("robot0_eef_quat", np.zeros(4)))]
        m, d = raw.sim.model, raw.sim.data
        for j in range(m.njnt):
            if int(m.jnt_type[j]) == 0:  # free joint
                adr = int(m.jnt_qposadr[j])
                parts.append(np.asarray(d.qpos[adr:adr + 3]))
        return np.concatenate(parts).astype(np.float32)


# ============================================================================ mock backend

def _cell_goal(task, cell):
    fixed = {"s1_start_0": (0.6, 0.15), "s1_start_1": (0.15, 0.6), "s1_mid": (0.4, 0.4)}
    if cell in fixed:
        return np.array(fixed[cell])
    h = int(hashlib.md5(f"{task}{cell}".encode()).hexdigest(), 16)
    return np.array([0.2 + 0.5 * ((h >> 3) % 100) / 100, 0.2 + 0.5 * ((h >> 11) % 100) / 100])


class MockEnv:
    def __init__(self, task, cell, seed):
        self.goal, self.horizon, self.cell = _cell_goal(task, cell), 192, cell
        self.rng = np.random.default_rng(seed)
        self.pos, self.timestep = np.zeros(2), 0

    def obs(self):
        img = np.zeros((8, 8, 3), np.uint8)
        img[..., 0], img[..., 1] = int(self.goal[0] * 255), int(self.goal[1] * 255)
        wrist = np.full((8, 8, 3), int(np.clip(self.pos[0], 0, 1) * 255), np.uint8)
        return {"agentview_image": img, "wrist_image": wrist, "state": np.r_[self.pos, self.timestep / 192.0].astype(np.float32)}


class MockAdapter(EnvAdapter):
    image_keys = ["agentview_image", "wrist_image"]
    third_person_key = "agentview_image"
    wrist_key = "wrist_image"
    obs_fn = None

    def make_env(self, cell, seed):
        return MockEnv(self.task, cell, seed)

    def reset(self, env, seed):
        env.rng = np.random.default_rng(seed)
        env.pos, env.timestep = env.rng.normal(0, 0.03, 2), 0
        return env.obs()

    def step(self, env, action):
        env.pos = env.pos + 0.02 * np.clip(np.asarray(action[:2]), -1, 1)
        env.timestep += 1
        return env.obs(), bool(np.linalg.norm(env.pos - env.goal) < 0.06), env.timestep >= env.horizon

    def max_steps(self, env):
        return env.horizon

    def get_state(self, env):
        return dict(flat=np.r_[env.pos], timestep=env.timestep)

    def set_state(self, env, state):
        env.pos, env.timestep = np.array(state["flat"][:2]), int(state["timestep"])
        return env.obs()

    def probe_targets(self, env):
        return np.r_[env.pos, env.goal].astype(np.float32)


class MockPolicy:
    """a[:2] = toward (1-mem)*goal_seen + mem*memorized_goal, plus steer*z[:2].

    steer -> how much noise moves the action (collapse when small).
    mem   -> ignores the image and heads for one memorized goal (the lamp 'start_1 only' failure).
    """

    def __init__(self, task, steer=0.6, mem=0.0, drift=0.0):
        self.task, self.steer, self.mem, self.drift = task, steer, mem, drift
        spec = TASK_SPECS[task]
        self.chunk_size, self.noise_dim, self.action_dim = spec["chunk_size"], spec["noise_dim"], spec["action_dim"]
        self.memorized = _cell_goal(task, spec["train_cells"][-1])
        r = np.random.default_rng(0)
        self.W0, self.Wd = r.normal(size=(5, 16)), r.normal(size=(5, 16))

    @classmethod
    def from_spec(cls, spec, task):
        kw = {}
        for part in spec.split(":", 1)[1].split(","):
            if part:
                k, v = part.split("=")
                kw[k] = float(v)
        return cls(task, **kw)

    def prepare(self, raw_obs):
        img = raw_obs["agentview_image"].astype(np.float32) / 255.0
        return dict(goal=img[0, 0, :2][None], state=np.asarray(raw_obs["state"], np.float32)[None])

    def chunk(self, batch, noise):
        B = noise.shape[0]
        goal = np.repeat(batch["goal"], B, 0) if batch["goal"].shape[0] == 1 else batch["goal"]
        pos = np.repeat(batch["state"], B, 0)[:, :2] if batch["state"].shape[0] == 1 else batch["state"][:, :2]
        tgt = (1 - self.mem) * goal + self.mem * self.memorized + np.array([0.07, -0.07])  # base policy misses
        d = np.clip(3.0 * (tgt - pos), -1, 1)
        a = np.zeros((B, self.chunk_size, self.action_dim), np.float32)
        a[:, :, :2] = d[:, None, :] + self.steer * noise[:, :, :2]
        a[:, :, 2:] = 0.1 * self.steer * noise[:, :, 2:self.action_dim]
        return a

    def to_env(self, chunk):
        return chunk

    def features(self, batch, module_suffixes):
        x = np.c_[batch["goal"], batch["state"]]
        W = (1 - self.drift) * self.W0 + self.drift * self.Wd
        f = np.tanh(x @ W)
        return {s: f * (1 + 0.1 * i) for i, s in enumerate(module_suffixes)}

    def release(self):
        pass


def make_adapter(args) -> EnvAdapter:
    if args.adapter == "mock":
        return MockAdapter(args.task)
    return RobocasaAdapter(args.task, args.env_factory, args.obs_fn)
