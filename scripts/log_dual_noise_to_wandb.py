#!/usr/bin/env python
"""Log a run's dual_noise_dsrl_cells.json to ONE wandb run, every rung in step order.

Dual-noise jobs each open their own wandb run, so a ladder split across jobs is fragmented and
any rung logged below a job's previous step is dropped. The JSON is complete; this rebuilds the
wandb view from it.

    python scripts/log_dual_noise_to_wandb.py --run-dir <run> [--run-dir <run> ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--wandb-project", default="lerobot-dex-eval")
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    for run_dir in args.run_dir:
        results = json.loads((run_dir / "dual_noise_dsrl_cells.json").read_text())
        steps = sorted(results, key=int)
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"dualnoise_dsrlcells_{run_dir.name}_merged",
            job_type="dual_noise_eval_merged",
            config={"run_dir": str(run_dir), "run_name": run_dir.name, "steps": [int(s) for s in steps]},
            reinit="finish_previous",
        )
        for step in steps:
            row = results[step]
            payload = {f"{mode}/{k}": v for mode, m in row.items() for k, v in m.items()}
            payload["gap/pc_success"] = row["iid"]["pc_success"] - row["duplicated"]["pc_success"]
            run.log(payload, step=int(step))
        run.finish()
        print(f"{run_dir.name}: logged {len(steps)} rungs {[int(s) for s in steps]}")


if __name__ == "__main__":
    main()
