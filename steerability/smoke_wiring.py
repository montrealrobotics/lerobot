"""Smoke test for steerability_wiring.py on the real RoboCasa env (needs a GPU node for EGL)."""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from adapters import RobocasaAdapter, _unwrap  # noqa: E402
from steerability_wiring import _LANG, pi05_frame  # noqa: E402

CAM = "robot0_agentview_left_image"


def hold_action(task, env, obs):
    a = np.zeros(env.action_space.shape)
    if task == "lamp":  # absolute joint targets: hold current joints
        a[:6] = obs["robot0_joint_pos"]
    return a


def imgdiff(a, b):
    return int(np.abs(a.astype(int) - b.astype(int)).max())


for task, cell in [("lamp", "s1_mid"), ("coffee", "seed11_L1S4")]:
    t = time.time()
    ad = RobocasaAdapter(task, "steerability_wiring:make_cell_env", "steerability_wiring:pi05_frame")
    env = ad.make_env(cell, 0)
    obs = ad.reset(env, 0)
    raw = _unwrap(env)
    print(f"\n== {task} {cell}  built+reset {time.time() - t:.0f}s  lang: {_LANG.get(task)!r}")
    b = pi05_frame(task, obs)
    print({k: (tuple(v.shape), round(float(v.min()), 3), round(float(v.max()), 3)) if hasattr(v, "shape") else v
           for k, v in b.items()})
    print("max_steps", ad.max_steps(env), "| gym max_episode_steps", env.unwrapped.max_episode_steps,
          "| probe", ad.probe_targets(env).shape)

    # seeding
    j0 = ad.reset(env, 0)["robot0_joint_pos"].copy()
    j0b = ad.reset(env, 0)["robot0_joint_pos"].copy()
    j1 = ad.reset(env, 1)["robot0_joint_pos"].copy()
    print(f"reset seed0==seed0: {np.allclose(j0, j0b)}  seed0!=seed1: {not np.allclose(j0, j1)}")

    # save / restore
    obs = ad.reset(env, 0)
    img_ref = np.array(raw._get_observations(force_update=True)[CAM], copy=True)
    img_ref2 = np.array(raw._get_observations(force_update=True)[CAM], copy=True)
    qpos_ref = raw.sim.data.qpos.copy()
    st = ad.get_state(env)
    traj = []
    for _ in range(10):
        o, s, d = ad.step(env, hold_action(task, env, obs))
        traj.append(np.array(o[CAM], copy=True))
    o2 = ad.set_state(env, st)
    print(f"render repeatable: {imgdiff(img_ref, img_ref2)}  | after restore: qpos maxdiff "
          f"{np.abs(raw.sim.data.qpos - qpos_ref).max():.2e}, img maxdiff {imgdiff(o2[CAM], img_ref)} "
          f"(vs 10 steps later: {imgdiff(traj[-1], img_ref)})")

    # determinism of a replay from the restored state
    obs = o2
    traj2 = []
    for _ in range(10):
        o, s, d = ad.step(env, hold_action(task, env, obs))
        traj2.append(np.array(o[CAM], copy=True))
    print("replay after restore, img maxdiff per step:", [imgdiff(a, b) for a, b in zip(traj, traj2)])
    env.close()
