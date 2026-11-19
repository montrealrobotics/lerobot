"""Replay determinism after restore, as p2_branching --check-determinism measures it."""
import pickle, sys; sys.path.insert(0, ".")
import numpy as np
from adapters import RobocasaAdapter, _unwrap
CAM = "robot0_agentview_left_image"
task, cell = sys.argv[1], sys.argv[2]
ad = RobocasaAdapter(task, "steerability_wiring:make_cell_env", "steerability_wiring:pi05_frame")
env = ad.make_env(cell, 0); obs = ad.reset(env, 0); raw = _unwrap(env)
rng = np.random.default_rng(0)
base = np.zeros(env.action_space.shape)
if task == "lamp": base[:6] = obs["robot0_joint_pos"]
acts = [base + np.r_[0.02 * rng.standard_normal(6), 0.1 * rng.standard_normal(16), np.zeros(base.size - 22)] for _ in range(48)]
for a in acts[:16]: ad.step(env, a)
st = pickle.loads(pickle.dumps(ad.get_state(env)))   # states go through a pickle in collect_states
def run():
    imgs = []
    for a in acts:
        o, *_ = ad.step(env, a); imgs.append(np.array(o[CAM], float))
    return d_state(), imgs
def d_state(): return ad.get_state(env)["flat"]
def cmp(name, A, B):
    print(f"{name:34s} state maxdiff {np.abs(A[0]-B[0]).max():.2e}  img mean|diff| last {np.abs(A[1][-1]-B[1][-1]).mean():.3f}")
orig = run()
ad.set_state(env, st); r1 = run()
ad.set_state(env, st); r2 = run()
cmp("replay vs replay", r1, r2)
cmp("replay vs original", r1, orig)
if task == "coffee":
    cm = raw.coffee_machine; cm._turned_on = True
    ad.set_state(env, st)
    print("coffee _turned_on after restore:", cm._turned_on, "(saved:", st["fixtures"][cm.name].get("_turned_on"), ")")
print("gym step count restored:", env.unwrapped._step_count == st["gym_steps"])
