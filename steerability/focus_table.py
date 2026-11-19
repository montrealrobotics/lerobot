"""Per-checkpoint values of the Phase 1 diagnostics next to the baselines and DSRL outcomes, training cells only.

Rows are every DSRL checkpoint; the checkpoints dup gets wrong (arm0 over-performs, the 4k lock-in failures,
arm4@16k, arm2@16k) are flagged in the `focus` column so they can be read against the rest.

  python focus_table.py --table ~/scratch/diag/analysis/table.csv --dsrl-json dsrl.json --out ~/scratch/diag/analysis
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FOCUS = {
    ("lamp", "arm0", 2000): "arm0 > dup", ("lamp", "arm0", 4000): "arm0 > dup",
    ("lamp", "arm0", 8000): "arm0 > dup", ("lamp", "arm0", 16000): "arm0 > dup",
    ("lamp", "arm1", 4000): "4k lock-in (split)", ("lamp", "arm3", 4000): "4k lock-in (split)",
    ("lamp", "arm4", 4000): "4k lock-in (0/3)", ("lamp", "arm4", 16000): "arm4 late (partial)",
    ("lamp", "arm2", 16000): "never solved",
    ("coffee", "arm0", 4000): "arm0 > dup", ("coffee", "arm0", 8000): "arm0 > dup",
    ("coffee", "arm0", 14000): "arm0 > dup", ("coffee", "arm0", 18000): "arm0 > dup",
    ("coffee", "arm3", 18000): "late < dup", ("coffee", "arm2", 18000): "late < dup",
    ("coffee", "arm1", 14000): "late < dup", ("coffee", "arm4", 18000): "late < dup",
}
COLS = ["p1_noise_sensitivity.noise_share", "p1_noise_sensitivity.pair_spread", "p1_noise_sensitivity.part_ratio",
        "p1_noise_sensitivity.local_gain", "p1_noise_sensitivity.pair_gain",
        "p1_obs_reliance.rel_blank_third", "p1_obs_reliance.rel_swap_cell_state", "p1_obs_reliance.rel_swap_cell_third",
        "p1_obs_reliance.cka_vision_tower", "p1_obs_reliance.cka_language_model",
        "p1_obs_reliance.cell_acc_language_model"]


def seeds_solved(dsrl, task, arm, ck, train_envs):
    """Seeds whose mean training-cell success over the >=400k evals is >= 0.8."""
    n = 0
    seeds = dsrl[task][arm]["checkpoints"][str(ck)]["seeds"]
    for sd in seeds.values():
        steps = np.asarray(sd["env_steps"])
        v = np.nanmean([np.asarray(sd["per_env"][e], float)[steps >= 400_000] for e in train_envs], axis=0)
        n += float(np.nanmean(v)) >= 0.8
    return f"{n}/{len(seeds)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--dsrl-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = json.load(open(a.dsrl_json))
    t = pd.read_csv(a.table)
    t = t[t.split == "train"]
    cols = [c for c in COLS if c in t.columns]
    g = t.groupby(["task", "arm", "sft_step"]).agg(
        final=("final", "mean"), t80_k=("t80", lambda x: x.mean() / 1000), seed_std=("seed_std", "mean"),
        dup=("base.dup_train", "first"), iid=("base.iid_train", "first"), warmup=("base.warmup", "first"),
        **{c.split(".")[-1]: (c, "mean") for c in cols}).reset_index()
    g.insert(3, "solved", [seeds_solved(D["results"], r.task, r.arm, r.sft_step, D["meta"]["tasks"][r.task]["training_envs"])
                           for r in g.itertuples()])
    g.insert(3, "focus", [FOCUS.get((r.task, r.arm, r.sft_step), "") for r in g.itertuples()])
    out = Path(a.out).expanduser()
    g.to_csv(out / "focus_table.csv", index=False)
    lines = ["# Phase 1 diagnostics per checkpoint (training cells)", ""]
    for task, h in g.groupby("task", sort=False):
        lines += [f"## {task}", "", h.drop(columns="task").round(3).to_markdown(index=False), ""]
    (out / "focus_table.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
