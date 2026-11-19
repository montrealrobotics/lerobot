"""Compare a 1a result at a larger --micro-batch with the batch-16 result for the same checkpoint.

Criterion (Charlotte, 2026-10-04): every 1a metric within ~1% of the batch-16 value per state, and identical
rankings of states. Also prints the between-checkpoint gap for scale, speed and GPU memory.

  python compare_microbatch.py --ref ~/scratch/diag/p1_noise_sensitivity/lamp_arm4_016000.json \
      --test ~/scratch/diag/batchtest/mb64/p1_noise_sensitivity/lamp_arm4_016000.json \
      --neighbor ~/scratch/diag/p1_noise_sensitivity/lamp_arm4_008000.json
"""
import argparse
import json

import numpy as np
from scipy.stats import kendalltau


def load(p):
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--neighbor", help="another checkpoint's batch-16 result, for the between-checkpoint gap")
    ap.add_argument("--tol", type=float, default=0.01)
    a = ap.parse_args()
    R, T = load(a.ref), load(a.test)
    N = load(a.neighbor) if a.neighbor else None
    rs, ts = R["per_state"], T["per_state"]
    assert len(rs) == len(ts), (len(rs), len(ts))
    mt, mr = T["meta"], R["meta"]
    assert mt["states"]["sha1"] == mr["states"]["sha1"]
    keys = [k for k in rs[0] if isinstance(rs[0][k], (int, float)) and not isinstance(rs[0][k], bool)]
    ok = True
    print(f"micro-batch {mr['argv'].get('micro_batch', 16)} -> {mt['argv']['micro_batch']}, {len(rs)} states")
    print(f"{'metric':12s} {'max |rel diff|':>15s} {'median':>9s} {'kendall tau':>12s} {'rank swaps':>11s}"
          f" {'ckpt gap (median rel)':>22s}")
    for k in keys:
        r = np.array([s[k] for s in rs], float)
        t = np.array([s[k] for s in ts], float)
        rel = np.abs(t - r) / np.maximum(np.abs(r), 1e-12)
        tau = kendalltau(r, t)[0]
        swaps = int((np.argsort(np.argsort(r)) != np.argsort(np.argsort(t))).sum())
        gap = ""
        if N is not None:
            n = np.array([s[k] for s in N["per_state"]], float)
            gap = f"{np.median(np.abs(n - r) / np.maximum(np.abs(r), 1e-12)):.3f}"
        pass_k = np.nanmax(rel) <= a.tol and swaps == 0
        ok &= pass_k
        print(f"{k:12s} {np.nanmax(rel):15.5f} {np.nanmedian(rel):9.5f} {tau:12.5f} {swaps:11d} {gap:>22s}"
              f"  {'OK' if pass_k else 'FAIL'}")
    print("per-cell metrics:")
    for cell in R["per_env"]:
        diffs = {k: abs(T["per_env"][cell][k] - v) / max(abs(v), 1e-12) for k, v in R["per_env"][cell].items()
                 if isinstance(v, (int, float)) and k != "n_states"}
        print(f"  {cell}: max rel diff {max(diffs.values()):.5f} ({max(diffs, key=diffs.get)})")
    tr, tt = mr["timing"], mt["timing"]
    cr, ct = tr["total_s"] - tr.get("load_s", 0), tt["total_s"] - tt.get("load_s", 0)
    print(f"1a compute: {cr:.0f} s -> {ct:.0f} s  (speedup {cr / ct:.2f}x)")
    print(f"GPU peak: {mt.get('gpu')}")
    print("PASS" if ok else "FAIL (per-state tolerance or rankings)")


if __name__ == "__main__":
    main()
