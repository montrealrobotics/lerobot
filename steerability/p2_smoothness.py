"""Phase 2b -- is the success region in noise space a basin SAC can climb, or isolated pockets?

Uses p2_branching output. At each state with mixed outcomes, takes the best noise z* and the
worst z_f found there and measures, with the same continuation mode and seeds as branching:
  retention_<s>   P(success) at z* + s * eps  (eps ~ N(0, I)), for each s in --sigmas
  basin_radius    largest s whose retention >= 0.5 * P(success at z*)
  interp_flips    outcome flips along the straight line z* -> z_f (--interp-points)
  interp_frac     fraction of the line that succeeds
Smooth landscapes: retention decays slowly, few flips. Rugged: retention collapses at small s,
many flips -> RL seed luck decides which pocket is found (the lamp split-seed checkpoints).

  python p2_smoothness.py --task lamp --arm arm1 --sft-step 4000 --checkpoint <ckpt> --out diag \
      --env-factory ... --obs-fn ...      # reads diag/p2_branching/lamp_arm1_004000.json
"""
import argparse
import json

import numpy as np

from adapters import make_adapter
from common import TASK_SPECS, add_common_args, load_policy, load_states, nanmean, result_path, write_result
from p2_branching import branch


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--branching", default=None, help="default: <out>/p2_branching/<tag>.json")
    p.add_argument("--sigmas", default="0.05,0.1,0.2,0.4,0.8")
    p.add_argument("--n-perturb", type=int, default=8)
    p.add_argument("--interp-points", type=int, default=9)
    p.add_argument("--max-states-per-cell", type=int, default=3)
    args = p.parse_args()

    spec = TASK_SPECS[args.task]
    bpath = args.branching or str(result_path(args.out, "p2_branching", args))
    B = json.load(open(bpath))
    S = load_states(B["states_file"])["states"]
    cont = B["continuation"]
    sigmas = [float(x) for x in args.sigmas.split(",")]
    adapter = make_adapter(args)
    policy = load_policy(args, adapter)

    envs, rows = {}, []
    by_cell = {}
    for d in B["details"]:
        pz = np.asarray(d["outcomes"]).mean(1)
        if 0 < pz.max() and pz.min() < 1:
            by_cell.setdefault(d["cell"], []).append(d)
    for cell, ds in by_cell.items():
        ds = sorted(ds, key=lambda d: -d["entropy"])[: args.max_states_per_cell]
        env = envs.setdefault(cell, adapter.make_env(cell, args.seed))
        for d in ds:
            s = S[d["state_index"]]
            max_steps = args.max_steps or adapter.max_steps(env)
            Z, out = np.asarray(d["z"], np.float32), np.asarray(d["outcomes"])
            pz = out.mean(1)
            zs, zf = Z[pz.argmax()], Z[pz.argmin()]
            seed = d["cont_seeds"][0]
            run = lambda z: float(branch(policy, adapter, env, s, z.astype(np.float32), cont, seed, max_steps, spec).success)  # noqa: E731
            p_star = run(zs)
            rng = np.random.default_rng(d["state_index"])
            row = dict(cell=cell, macro=d["macro"], p_star=p_star)
            radius = 0.0
            for sg in sigmas:
                ret = float(np.mean([run(zs + sg * rng.standard_normal(zs.shape)) for _ in range(args.n_perturb)]))
                row[f"retention_{sg:g}"] = ret
                if p_star > 0 and ret >= 0.5 * p_star:
                    radius = sg
            row["basin_radius"] = radius if p_star > 0 else np.nan
            path = [run((1 - a) * zs + a * zf) for a in np.linspace(0, 1, args.interp_points)]
            row["interp_flips"] = int(np.sum(np.abs(np.diff(path)) > 0))
            row["interp_frac"] = float(np.mean(path))
            rows.append(row)
            print(row)

    per_env = {}
    for cell in dict.fromkeys(r["cell"] for r in rows):
        rs = [r for r in rows if r["cell"] == cell]
        per_env[cell] = {k: nanmean([r[k] for r in rs]) for k in rs[0] if k not in ("cell", "macro")}
        per_env[cell]["n_states"] = len(rs)
    for d in B["details"]:  # cells with no mixed states: record why they are empty
        if d["cell"] not in per_env:
            per_env[d["cell"]] = dict(n_states=0, note="no_mixed_states")

    write_result(result_path(args.out, "p2_smoothness", args), args, per_env, dict(per_state=rows), phase="p2_smoothness")


if __name__ == "__main__":
    main()
