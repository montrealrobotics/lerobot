#!/usr/bin/env python
"""Pick the DSRL checkpoint from a run's dual_noise_dsrl_cells.json.

Score is duplicated-noise success on the DSRL *training* cells (--group), smoothed over
--window adjacent rungs. Among rungs within one standard error of the best smoothed score, the
earliest wins. Held-out cells are shown but never used, so the held-out DSRL number stays
held out.

    python scripts/select_sft_checkpoint.py --run-dir <run>
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def num_frames(run_dir: Path) -> int | None:
    """dataset.num_frames as the trainer logged it."""
    for log in run_dir.glob("wandb/*/files/output.log"):
        m = re.search(r"dataset\.num_frames=(\d+)", log.read_text(errors="ignore"))
        if m:
            return int(m.group(1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--group", default="train", help="Cell group to select on")
    parser.add_argument("--window", type=int, default=3, help="Rungs to average over (odd)")
    parser.add_argument("--num-frames", type=int, default=None, help="Override for the epoch column")
    parser.add_argument("--batch-size", type=int, default=None, help="Override for the epoch column")
    args = parser.parse_args()

    results = json.loads((args.run_dir / "dual_noise_dsrl_cells.json").read_text())
    steps = sorted(results, key=int)
    if not steps:
        raise SystemExit("dual_noise_eval.json has no rungs")

    cfg_path = next(args.run_dir.glob("checkpoints/*/pretrained_model/train_config.json"), None)
    cfg = json.loads(cfg_path.read_text()) if cfg_path else {}
    batch = args.batch_size or cfg.get("batch_size")
    frames = args.num_frames or num_frames(args.run_dir)

    g = args.group
    rows = []
    for s in steps:
        dup, iid = results[s]["duplicated"], results[s]["iid"]
        if f"n_{g}" not in dup:
            raise SystemExit(f"step {s}: no n_{g} -- re-score with the current eval_pi05_dual_noise.py")
        n = dup[f"n_{g}"]
        rows.append(
            {
                "step": int(s),
                "n": n,
                "iid": iid[f"pc_success_{g}"],
                "dup": dup[f"pc_success_{g}"],
                "heldout": dup.get("pc_success_heldout"),
                "successes": round(dup[f"pc_success_{g}"] / 100 * n),
                "epochs": (int(s) * batch / frames) if batch and frames else None,
            }
        )

    half = args.window // 2
    for i, r in enumerate(rows):
        lo, hi = max(0, i - half), min(len(rows), i + half + 1)
        win = rows[lo:hi]
        n = sum(w["n"] for w in win)
        c = sum(w["successes"] for w in win)
        r["smoothed"] = 100 * c / n if n else float("nan")
        r["se"] = 100 * math.sqrt((c / n) * (1 - c / n) / n) if n else float("nan")
        r["ci"] = tuple(100 * x for x in wilson(c, n))

    best = max(rows, key=lambda r: r["smoothed"])
    threshold = best["smoothed"] - best["se"]
    pick = next(r for r in rows if r["smoothed"] >= threshold)

    print(f"selecting on duplicated-noise '{g}' cells; 'heldout' is informational only\n")
    print(
        f"{'step':>7} {'epochs':>7} {'iid':>7} {'dup':>7} {'gap':>6} {'smoothed':>9} {'95% CI':>15} "
        f"{'heldout':>8}"
    )
    for r in rows:
        ep = f"{r['epochs']:.1f}" if r["epochs"] is not None else "-"
        mark = " <-- pick" if r is pick else (" (max)" if r is best else "")
        ci = f"[{r['ci'][0]:.0f}, {r['ci'][1]:.0f}]"
        ho = f"{r['heldout']:.1f}%" if r["heldout"] is not None else "-"
        print(
            f"{r['step']:>7} {ep:>7} {r['iid']:>6.1f}% {r['dup']:>6.1f}% {r['iid'] - r['dup']:>5.1f} "
            f"{r['smoothed']:>8.1f}% {ci:>15} {ho:>8}{mark}"
        )
    print(f"\nselected step {pick['step']}  (window={args.window}, earliest within 1 SE of the max)")
    out = args.run_dir / "selected_checkpoint.json"
    out.write_text(
        json.dumps({"step": pick["step"], "group": g, "window": args.window, "rows": rows}, indent=1, default=list)
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
