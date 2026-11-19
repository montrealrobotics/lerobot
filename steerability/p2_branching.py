"""Phase 2a -- branching rollouts: how much can choosing the noise improve success, state by state?

From saved sim states (the checkpoint's OWN rollouts are the natural choice), restore the state,
execute the first chunk with noise z_k (k = 1..K, duplicated), then continue until success or
horizon. Continuation:
  base   fresh duplicated N(0,1) noise every later chunk -- outcome estimates Q_base(s, z_k),
         exactly the quantity the DSRL critic learns (default)
  fixed  z_k held for the rest of the episode (deterministic; a 'constant steering' probe)
Continuation noise uses common random numbers: repeat r at state s uses the same seed for every
z_k, so differences between z's are not continuation luck.

Per state:  mean_p      success averaged over z           (what random steering gets)
            best_split  pick best z on repeat A, score it on repeat B, and vice versa
                        (unbiased; plain max over z is optimistic)
            gap         best_split - mean_p               (the steerability gap)
            frac_any    fraction of z with any success   (does good noise exist at all)
Per cell: means over states, plus gap_start (first saved state) and gap_max.
All z's and outcomes are stored for p2_smoothness / p2_transfer.

  python p2_branching.py --task lamp --arm arm1 --sft-step 4000 --checkpoint <ckpt> --out diag \
      --states diag/states/lamp_arm1_004000.pkl --env-factory ... --obs-fn ... --check-determinism
"""
import argparse

import numpy as np

from adapters import make_adapter
from common import (NoiseStream, TASK_SPECS, add_common_args, cells_for, load_policy, load_states,
                    nanmean, result_path, rollout, sample_noise, write_result)


def select_states(states, cells, per_cell, max_t=None):
    """per_cell states per cell, spread evenly over episode time; max_t drops late states that have too few
    steps left to succeed (e.g. t = 960 of a 990-step lamp episode)."""
    out = []
    for cell in cells:
        cs = sorted([(i, s) for i, s in enumerate(states) if s["cell"] == cell and (max_t is None or s["t"] <= max_t)],
                    key=lambda x: (x[1]["macro"], x[1]["rollout"]))
        if not cs:
            continue
        keep = np.unique(np.linspace(0, len(cs) - 1, min(per_cell, len(cs))).round().astype(int))
        out += [cs[i] for i in keep]
    return out


def branch(policy, adapter, env, state, z, cont, cont_seed, max_steps, spec):
    obs = adapter.set_state(env, state["sim_state"])
    ns = NoiseStream(spec["noise_dim"], spec["chunk_size"], seed=cont_seed, first=z, after=cont)
    r = rollout(policy, adapter, env, obs, ns, max_steps, spec["n_action_steps"], start_step=state["t"])
    return r


def check_determinism(policy, adapter, env, state, z, max_steps, spec):
    outs = []
    for _ in range(2):
        obs = adapter.set_state(env, state["sim_state"])
        ns = NoiseStream(spec["noise_dim"], spec["chunk_size"], seed=0, first=z, after="fixed")
        r = rollout(policy, adapter, env, obs, ns, min(max_steps, state["t"] + 3 * spec["n_action_steps"]),
                    spec["n_action_steps"], start_step=state["t"])
        outs.append((r.success, r.steps, adapter.get_state(env)["flat"]))
    diff = float(np.max(np.abs(outs[0][2] - outs[1][2])))
    return dict(same_outcome=outs[0][:2] == outs[1][:2], max_state_diff=diff)


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--states", required=True)
    p.add_argument("--states-per-cell", type=int, default=5)
    p.add_argument("--k", type=int, default=16, help="noise vectors per state")
    p.add_argument("--repeats", type=int, default=2, help=">=2 enables the unbiased best_split")
    p.add_argument("--continuation", choices=["base", "fixed"], default="base")
    p.add_argument("--noise-seed", type=int, default=4321)
    p.add_argument("--noise-kind", default="gaussian", help="'box' to restrict z to the actor's range")
    p.add_argument("--check-determinism", action="store_true")
    p.add_argument("--max-start-frac", type=float, default=0.6,
                   help="only branch from states with t <= this fraction of the episode cap")
    args = p.parse_args()

    spec = TASK_SPECS[args.task]
    adapter = make_adapter(args)
    policy = load_policy(args, adapter)
    S = load_states(args.states)["states"]
    chosen = select_states(S, cells_for(args), args.states_per_cell,
                           max_t=int(args.max_start_frac * (args.max_steps or spec["max_steps"])))
    R = 1 if args.continuation == "fixed" else args.repeats  # fixed is deterministic

    envs, details, extra = {}, [], {}
    for n, (si, s) in enumerate(chosen):
        env = envs.setdefault(s["cell"], adapter.make_env(s["cell"], args.seed))
        max_steps = args.max_steps or adapter.max_steps(env)
        Z = sample_noise(args.noise_kind, args.k, spec["noise_dim"], np.random.default_rng(args.noise_seed + si))
        if args.check_determinism and n == 0:
            extra["determinism"] = check_determinism(policy, adapter, env, s, Z[0], max_steps, spec)
            print("determinism:", extra["determinism"])
            if extra["determinism"]["max_state_diff"] > 1e-6:
                print("WARNING: restored rollouts diverge -- fix set_state/controller reset before trusting Phase 2")
        seeds = [args.noise_seed * 7 + si * 101 + r for r in range(R)]
        out = np.zeros((args.k, R))
        for k in range(args.k):
            for r in range(R):
                out[k, r] = branch(policy, adapter, env, s, Z[k], args.continuation, seeds[r], max_steps, spec).success
        pz = out.mean(1)
        if R >= 2:
            a, b = out[:, 0], out[:, 1]
            best_split = 0.5 * (b[np.argmax(a)] + a[np.argmax(b)])
        else:
            best_split = float(pz.max())  # deterministic continuation: max is exact
        mean_p = float(pz.mean())
        h = -(mean_p * np.log(mean_p) + (1 - mean_p) * np.log(1 - mean_p)) if 0 < mean_p < 1 else 0.0
        details.append(dict(state_index=si, cell=s["cell"], macro=s["macro"], t=s["t"], rollout=s["rollout"],
                            z=Z, outcomes=out, cont_seeds=seeds, mean_p=mean_p, best_split=float(best_split),
                            gap=float(best_split - mean_p), frac_any=float((pz > 0).mean()), entropy=float(h)))
        print(f"[{n + 1}/{len(chosen)}] {s['cell']} macro={s['macro']} mean_p={mean_p:.2f} best={best_split:.2f}")

    per_env = {}
    for cell in dict.fromkeys(d["cell"] for d in details):
        ds = sorted([d for d in details if d["cell"] == cell], key=lambda d: d["macro"])
        per_env[cell] = {k: nanmean([d[k] for d in ds]) for k in ("mean_p", "best_split", "gap", "frac_any", "entropy")}
        per_env[cell].update(gap_start=ds[0]["gap"], gap_max=max(d["gap"] for d in ds), n_states=len(ds))
        print(cell, per_env[cell])

    extra.update(continuation=args.continuation, repeats=R, states_file=args.states, details=details)
    write_result(result_path(args.out, "p2_branching", args), args, per_env, extra, phase="p2_branching")


if __name__ == "__main__":
    main()
