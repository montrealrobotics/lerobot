"""Swap the lamp arm4 DSRL records in dsrl_wandb_evals.json for the September-bank reruns.

The original lamp arm4 runs (forge, 2026-09-27) used the invalid 09-27 pair bank. The reruns
(`dsrl_arm4_lamp_<step>_s<seed>_sepbank`, wandb.ai chacloutier-4B/lerobot-dsrl) used `_pair_dsrl`,
pose-identical to Artur's `_pair`. Run IDs come from ~/scratch/claude_tmp/arm4_rerun/runs.tsv.

Writes:
  dsrl_wandb_evals.json            lamp/arm4 records replaced (previous file kept as *_oldbank_arm4.json)
  ../../diag/p0/history/<id>.csv   full history per rerun (same cache p0_wandb_signals.py uses)
  arm4_lamp_rerun_replacements.json  which rerun replaced which original run
"""

import csv
import json
import shutil
from pathlib import Path

import wandb

HERE = Path(__file__).parent
RUNS_TSV = Path.home() / "scratch/claude_tmp/arm4_rerun/runs.tsv"
HISTORY = HERE.parent.parent / "diag/p0/history"
PROJECT = "chacloutier-4B/lerobot-dsrl"
SOURCE = "chacloutier-4B/lerobot-dsrl"
SFT_RUN = "09-36-39_sft_arms_lamp_arm4_lora_vlm_full_expert_lrx1_seed42"
BANK = "/network/scratch/c/cloutiec/lerobot/placement_banks/screwlightbulb_xarm6_seed1_pair_dsrl/bank.json"
BANK_MD5 = "919c648884d68744e4cb0b841cb6d066"


def main():
    rows = [r for r in csv.DictReader(open(RUNS_TSV), delimiter="\t") if r["kind"] == "dsrl_full"]
    assert len(rows) == 12 and all(r["state"] == "COMPLETED" for r in rows), rows
    assert all(r["bank_md5"] == BANK_MD5 for r in rows)

    path = HERE / "dsrl_wandb_evals.json"
    backup = HERE / "dsrl_wandb_evals_oldbank_arm4.json"
    if not backup.exists():
        shutil.copy2(path, backup)
    records = json.loads(backup.read_text())
    old = {(r["checkpoint"], r["seed"]): r["id"] for r in records if r["task"] == "lamp" and r["arm"] == "arm4"}
    records = [r for r in records if not (r["task"] == "lamp" and r["arm"] == "arm4")]

    api = wandb.Api(timeout=120)
    HISTORY.mkdir(parents=True, exist_ok=True)
    replaced = []
    for row in rows:
        rid = row["wandb_run"].split("/")[-1]
        run = api.run(f"{PROJECT}/{rid}")
        ck, seed = int(row["rung"]), int(row["seed"])
        assert run.state == "finished", (rid, run.state)
        assert run.config.get("seed", seed) == seed
        hist = run.history(samples=100_000, pandas=True)
        hist.to_csv(HISTORY / f"{rid}.csv", index=False)
        eval_keys = sorted(k for k in hist.columns if k.startswith("eval/success_rate"))
        ev = hist[["_step", *eval_keys]].dropna(subset=["eval/success_rate"])
        evals = [{k: (int(v) if k == "_step" else (int(v) if float(v).is_integer() else float(v)))
                  for k, v in e.items()} for e in ev.to_dict("records")]
        records.append({
            "id": rid,
            "name": run.name,
            "state": run.state,
            "created": run.created_at if run.created_at.endswith("Z") else run.created_at + "Z",
            "task": "lamp",
            "arm": "arm4",
            "checkpoint": ck,
            "seed": seed,
            "sft_run": SFT_RUN,
            "policy_path": run.metadata["args"][run.metadata["args"].index("--policy_path") + 1],
            "host": run.metadata.get("host"),
            "evals": evals,
            "eval_keys": eval_keys,
            "source": SOURCE,
            "placement_bank": BANK,
            "placement_bank_md5": BANK_MD5,
        })
        replaced.append({"checkpoint": ck, "seed": seed, "new": f"{PROJECT}/{rid}", "new_job": row["job_id"],
                         "old": f"forge chacloutier-4B/lerobot-dsrl/{old.get((ck, seed))}",
                         "n_evals": len(evals), "last_eval_step": evals[-1]["_step"]})
        print(f"lamp arm4 {ck // 1000}k s{seed}: {old.get((ck, seed))} -> {rid}, {len(evals)} evals, last {evals[-1]['_step']}")

    path.write_text(json.dumps(records))
    (HERE / "arm4_lamp_rerun_replacements.json").write_text(json.dumps(
        {"bank": BANK, "bank_md5": BANK_MD5, "replacements": replaced}, indent=1))


if __name__ == "__main__":
    main()
