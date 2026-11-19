"""Sanity checks on Phase 1 inputs and outputs, before releasing more jobs.

  python check_p1.py --out ~/scratch/diag states            reference sets: states per cell, success mix, bank
  python check_p1.py --out ~/scratch/diag results           1a/1c sanity + runtime breakdown, per result file
"""
import argparse
import collections
import json
import pickle
from pathlib import Path

import numpy as np

SEPT_LAMP_MD5 = {"cf7019dd418f0340b87b6bab18a3382b", "919c648884d68744e4cb0b841cb6d066"}  # _pair, _pair_dsrl
MIN_PER_CELL = 20


def check_states(out: Path):
    for f in sorted((out / "states").glob("ref_*.pkl")):
        if "unrecorded" in f.name:
            continue
        d = pickle.load(open(f, "rb"))
        m, S = d["meta"], d["states"]
        bank = m.get("bank") or {}
        ok_bank = bank.get("md5") in (SEPT_LAMP_MD5 if m["task"] == "lamp" else {"25d89adb859cdb2810d8ff4812a8a319"})
        print(f"== {f.name}: {m['task']} {m['arm']}@{m['sft_step']}, bank {bank.get('md5')} "
              f"({'OK' if ok_bank else 'NOT the expected bank'}), noise {m['noise_mode']}, every {m['record_every']}")
        c = collections.Counter((s["cell"], bool(s["rollout_success"])) for s in S)
        for cell, summ in m["summary"].items():
            ns, nf = c[(cell, True)], c[(cell, False)]
            flag = "OK" if ns + nf >= MIN_PER_CELL and ns and nf else "CHECK"
            print(f"  {cell:14s} rollouts {summ['n_rollouts']:2d}  success {summ['success_rate']:.2f}  "
                  f"states {ns + nf:4d} (from successes {ns}, failures {nf})  {flag}")


def check_results(out: Path):
    for phase in ("p1_obs_reliance", "p1_noise_sensitivity"):
        for f in sorted((out / phase).glob("*.json")):
            d = json.loads(f.read_text())
            m = d["meta"]
            t = {k: round(v, 1) for k, v in (m.get("timing") or {}).items()}
            print(f"== {phase} {f.stem}: bank {(m.get('bank') or {}).get('md5', '?')[:8]}, "
                  f"states {(m.get('states') or {}).get('sha1')}, noise_seed {m['argv'].get('noise_seed')}, timing {t}")
            if phase == "p1_noise_sensitivity":
                ps = d.get("per_state", [])
                spread = np.array([s.get("pair_spread", np.nan) for s in ps], float)
                share = [v.get("noise_share") for v in d["per_env"].values()]
                print(f"  per-state pair_spread: n={len(spread)} min {np.nanmin(spread):.3g} "
                      f"median {np.nanmedian(spread):.3g} max {np.nanmax(spread):.3g} cv {np.nanstd(spread) / np.nanmean(spread):.2f}"
                      f"  {'OK' if np.nanmin(spread) > 0 and np.nanstd(spread) > 0 else 'CHECK'}")
                print(f"  noise_share per cell: {[round(x, 3) if x is not None else None for x in share]}  "
                      f"{'OK' if all(x is not None and 0 <= x <= 1 for x in share) else 'CHECK'}")
            else:
                cell = next(iter(d["per_env"].values()))
                print("  " + "  ".join(f"{k} {cell[k]:.4f}" for k in sorted(cell) if k.startswith("cka_")))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("what", choices=["states", "results"])
    a = p.parse_args()
    (check_states if a.what == "states" else check_results)(Path(a.out).expanduser())
