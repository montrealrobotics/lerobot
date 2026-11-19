"""Phase 1b -- does success survive when noise is restricted to what the DSRL actor can emit?

The actor is tanh-bounded to [-1,1]^32 but dual-noise eval samples N(0,1): ~32% of each
gaussian coordinate is out of the actor's reach. This re-runs duplicated-noise eval with
several noise sources, using the same reset seeds for every source (common random starts):

  gaussian          N(0,1)                     (= existing dual-noise 'duplicated')
  box               U[-1,1]                    (uniform over the actor's whole range)
  tanh_gaussian     tanh(N(0,1))               (an untrained tanh-gaussian actor)
  clipped_gaussian  clip(N(0,1), -1, 1)

Per cell: success_<kind>, box_retention = success_box / success_gaussian, and for the gaussian
episodes the out-of-box fraction of the noise used in successes vs failures (if successes need
|z|>1 coordinates, oob_frac_success > oob_frac_fail).

  python p1_box_reachability.py --task lamp --arm arm1 --sft-step 4000 --checkpoint <ckpt> \
      --out diag --env-factory ... --obs-fn ... --episodes 20
"""
import argparse

import numpy as np

from adapters import make_adapter
from common import (NOISE_KINDS, NoiseStream, TASK_SPECS, add_common_args, cells_for, load_policy,
                    result_path, rollout, write_result)


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--episodes", type=int, default=20, help="per cell per noise kind")
    p.add_argument("--kinds", default=",".join(NOISE_KINDS))
    args = p.parse_args()

    spec = TASK_SPECS[args.task]
    kinds = args.kinds.split(",")
    adapter = make_adapter(args)
    policy = load_policy(args, adapter)
    per_env = {}
    for cell in cells_for(args):
        env = adapter.make_env(cell, args.seed)
        res = {k: [] for k in kinds}
        oob = {True: [], False: []}
        for ep in range(args.episodes):
            reset_seed = args.seed * 10_000 + ep  # same start for every noise kind
            for ki, kind in enumerate(kinds):
                obs = adapter.reset(env, reset_seed)
                ns = NoiseStream(spec["noise_dim"], spec["chunk_size"], seed=reset_seed * 31 + ki,
                                 after="base", kind=kind)
                r = rollout(policy, adapter, env, obs, ns, args.max_steps or adapter.max_steps(env),
                            spec["n_action_steps"])
                res[kind].append(r.success)
                if kind == "gaussian" and r.noises:
                    oob[r.success].append(float(np.mean(np.abs(np.stack(r.noises)) > 1.0)))
        row = {f"success_{k}": float(np.mean(v)) for k, v in res.items()}
        g = row.get("success_gaussian", np.nan)
        if "success_box" in row:
            row["box_retention"] = row["success_box"] / g if g > 0 else np.nan
            row["box_minus_gaussian"] = row["success_box"] - g
        row["oob_frac_success"] = float(np.mean(oob[True])) if oob[True] else np.nan
        row["oob_frac_fail"] = float(np.mean(oob[False])) if oob[False] else np.nan
        row["n_episodes"] = args.episodes
        per_env[cell] = row
        print(cell, row)

    write_result(result_path(args.out, "p1_box_reachability", args), args, per_env, phase="p1_box_reachability")


if __name__ == "__main__":
    main()
