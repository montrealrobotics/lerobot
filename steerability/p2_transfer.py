"""Phase 2c -- does a noise that works in one cell still work in another?

Held-out DSRL success needs the steering learned in training cells to carry over. Using
p2_branching output (run it on ALL cells), for each target-cell state takes the source-cell
states at the nearest macro step, picks their top-T noises, and replays them at the target state
(same continuation mode and seeds as branching there).

  p_transfer       success of transplanted noises at the target
  transfer_lift    p_transfer - mean_p(target)                 (> 0: better than random noise)
  transfer_norm    lift / (best_split(target) - mean_p(target)) (1 = as good as the target's own best)

Default pairs: train cells -> each held-out cell, and train cell <-> train cell (for lamp the
start_0/start_1 pair is where the late checkpoints split). For coffee, kitchens differ in layout,
so macro-step matching is approximate: read it as 'same task phase', not 'same pose'.

  python p2_transfer.py --task lamp --arm arm1 --sft-step 4000 --checkpoint <ckpt> --out diag \
      --env-factory ... --obs-fn ...
"""
import argparse
import json

import numpy as np

from adapters import make_adapter
from common import TASK_SPECS, add_common_args, load_policy, load_states, nanmean, result_path, write_result
from p2_branching import branch


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--branching", default=None)
    p.add_argument("--top", type=int, default=3, help="noises transplanted per source state")
    args = p.parse_args()

    spec = TASK_SPECS[args.task]
    B = json.load(open(args.branching or str(result_path(args.out, "p2_branching", args))))
    S = load_states(B["states_file"])["states"]
    cont = B["continuation"]
    adapter = make_adapter(args)
    policy = load_policy(args, adapter)
    D = B["details"]
    cells = list(dict.fromkeys(d["cell"] for d in D))
    train = [c for c in spec["train_cells"] if c in cells]
    pairs = [(src, tgt) for tgt in cells for src in train if src != tgt]

    envs, rows = {}, []
    for src, tgt in pairs:
        env = envs.setdefault(tgt, adapter.make_env(tgt, args.seed))
        max_steps = args.max_steps or adapter.max_steps(env)
        for dt in [d for d in D if d["cell"] == tgt]:
            srcs = [d for d in D if d["cell"] == src]
            ds = min(srcs, key=lambda d: abs(d["macro"] - dt["macro"]))
            pz = np.asarray(ds["outcomes"]).mean(1)
            if pz.max() == 0:
                rows.append(dict(src=src, tgt=tgt, macro=dt["macro"], p_transfer=np.nan, note="source_has_no_success"))
                continue
            Zs = np.asarray(ds["z"], np.float32)[np.argsort(-pz)[: args.top]]
            s = S[dt["state_index"]]
            succ = [branch(policy, adapter, env, s, z, cont, seed, max_steps, spec).success
                    for z in Zs for seed in dt["cont_seeds"]]
            pt = float(np.mean(succ))
            denom = dt["best_split"] - dt["mean_p"]
            rows.append(dict(src=src, tgt=tgt, macro=dt["macro"], p_transfer=pt,
                             transfer_lift=pt - dt["mean_p"],
                             transfer_norm=(pt - dt["mean_p"]) / denom if denom > 0.05 else np.nan))
            print(rows[-1])

    per_env = {}
    for tgt in cells:
        rs = [r for r in rows if r["tgt"] == tgt]
        if not rs:
            continue
        row = {k: nanmean([r.get(k, np.nan) for r in rs]) for k in ("p_transfer", "transfer_lift", "transfer_norm")}
        for src in train:  # per-source breakdown, e.g. s1_start_1 -> s1_start_0
            sub = [r for r in rs if r["src"] == src]
            if sub:
                row[f"lift_from_{src}"] = nanmean([r.get("transfer_lift", np.nan) for r in sub])
        per_env[tgt] = row
        print(tgt, row)

    write_result(result_path(args.out, "p2_transfer", args), args, per_env, dict(pairs=rows), phase="p2_transfer")


if __name__ == "__main__":
    main()
