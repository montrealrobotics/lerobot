"""Is the observation right after a restore a function of the restored state alone?

For a few saved states: (a) restore twice in a row, no step between; (b) restore, step n with fixed actions,
restore again. Compares every observation key of the two restored observations (images included).
Run with STEER_NO_ANCHOR_FIX=1 to disable the lamp-weld handling in set_state.
"""
import argparse, json
import numpy as np
from adapters import make_adapter
from common import add_common_args, cells_for, load_states

def cmp(o1, o2):
    out = {}
    for k in o1:
        if k in o2 and hasattr(o1[k], "shape") and np.asarray(o1[k]).shape == np.asarray(o2[k]).shape:
            d = float(np.max(np.abs(np.asarray(o1[k], float) - np.asarray(o2[k], float)))) if np.asarray(o1[k]).size else 0.0
            if d: out[k] = d
    return out

p = add_common_args(argparse.ArgumentParser(), needs_policy=False)
p.add_argument("--states", required=True); p.add_argument("--n-steps", type=int, default=16)
a = p.parse_args()
ad = make_adapter(a)
S = [s for s in load_states(a.states)["states"] if s["cell"] == cells_for(a)[0]]
picks = [S[0], S[len(S) // 4], S[len(S) // 2]]
env = ad.make_env(cells_for(a)[0], a.seed)
for s in picks:
    o1 = ad.set_state(env, s["sim_state"]); o1 = {k: np.array(v) for k, v in o1.items() if hasattr(v, "shape")}
    o2 = ad.set_state(env, s["sim_state"]); o2 = {k: np.array(v) for k, v in o2.items() if hasattr(v, "shape")}
    print(json.dumps(dict(t=s["t"], test="restore_twice", diff=cmp(o1, o2))), flush=True)
    for _ in range(a.n_steps):
        ad.step(env, np.zeros(22))
    o3 = ad.set_state(env, s["sim_state"]); o3 = {k: np.array(v) for k, v in o3.items() if hasattr(v, "shape")}
    print(json.dumps(dict(t=s["t"], test="restore_step_restore", diff=cmp(o1, o3))), flush=True)
