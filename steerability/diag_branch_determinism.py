"""Determinism of restored branches on exactly the states and noises p2_branching uses.

For every state p2_branching would select (same --states-per-cell, --max-start-frac, noise seed), restore it
twice and run --n-chunks chunks with that state's first branching noise held fixed. Reports the final-state
difference per state and, when the two runs diverge, the first macro step and observation key that differ.

  python diag_branch_determinism.py --task lamp --arm arm2 --sft-step 4000 --checkpoint <ckpt> --out <dir> \
      --states <own states .pkl> --cells s1_start_0,s1_start_1 --env-factory ... --obs-fn ...
"""
import argparse
import json

import numpy as np

from adapters import make_adapter
from common import TASK_SPECS, NoiseStream, add_common_args, cells_for, load_policy, load_states, sample_noise
from p2_branching import select_states


def run(policy, adapter, env, state, z, n_chunks, spec):
    """Restore, then roll n_chunks chunks with z fixed; return per-macro obs and actions, and the final state."""
    obs = adapter.set_state(env, state["sim_state"])
    raw = __import__("adapters")._unwrap(env)
    again = raw._get_observations(force_update=True)  # second render of the same restored state, no step between
    rerender = {k: float(np.max(np.abs(np.asarray(obs[k], float) - np.asarray(again[k], float))))
                for k in obs if k.endswith("_image")}
    print("rerender_diff", json.dumps(rerender), flush=True)
    ns = NoiseStream(spec["noise_dim"], spec["chunk_size"], seed=0, first=z, after="fixed")
    trace = []
    for macro in range(n_chunks):
        acts = policy.to_env(policy.chunk(policy.prepare(obs), ns(macro)[None]))[0][: spec["n_action_steps"]]
        trace.append(({k: np.array(v) for k, v in obs.items() if hasattr(v, "shape")}, acts))
        for a in acts:
            obs, _, _ = adapter.step(env, a)
    return trace, adapter.get_state(env)["flat"]


def first_divergence(t1, t2):
    for macro, ((o1, a1), (o2, a2)) in enumerate(zip(t1, t2)):
        keys = [k for k in o1 if k in o2 and o1[k].shape == o2[k].shape and not np.array_equal(o1[k], o2[k])]
        if keys:
            worst = max(keys, key=lambda k: float(np.max(np.abs(o1[k].astype(float) - o2[k].astype(float)))))
            return dict(macro=macro, what="obs", keys=keys[:8], worst=worst,
                        worst_diff=float(np.max(np.abs(o1[worst].astype(float) - o2[worst].astype(float)))))
        if not np.array_equal(a1, a2):
            return dict(macro=macro, what="action", diff=float(np.max(np.abs(a1 - a2))))
    return None


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--states", required=True)
    p.add_argument("--states-per-cell", type=int, default=5)
    p.add_argument("--max-start-frac", type=float, default=0.6)
    p.add_argument("--noise-seed", type=int, default=4321)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n-chunks", type=int, default=6)
    args = p.parse_args()
    spec = TASK_SPECS[args.task]
    adapter = make_adapter(args)
    policy = load_policy(args, adapter)
    S = load_states(args.states)["states"]
    chosen = select_states(S, cells_for(args), args.states_per_cell,
                           max_t=int(args.max_start_frac * (args.max_steps or spec["max_steps"])))
    envs, rows = {}, []
    for si, s in chosen:
        env = envs.setdefault(s["cell"], adapter.make_env(s["cell"], args.seed))
        z = sample_noise("gaussian", args.k, spec["noise_dim"], np.random.default_rng(args.noise_seed + si))[0]
        (t1, f1), (t2, f2) = (run(policy, adapter, env, s, z, args.n_chunks, spec) for _ in range(2))
        row = dict(cell=s["cell"], t=int(s["t"]), max_state_diff=float(np.max(np.abs(f1 - f2))),
                   first_divergence=first_divergence(t1, t2))
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = f"{args.out}/branch_determinism_{args.task}_{args.arm}_{args.sft_step:06d}.json"
    json.dump(rows, open(out, "w"), indent=1)
    bad = [r for r in rows if r["max_state_diff"] > 1e-6]
    print(f"{len(rows) - len(bad)}/{len(rows)} states deterministic; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
