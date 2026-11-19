# Missing results

Checked 2026-09-30 against the ten SFT ladders (arms 0–4 × lamp, coffee): 111 DSRL runs in
`real-lab/lerobot-dsrl` (arms 0–3) and 30 in forge `chacloutier-4B/lerobot-dsrl` (arm4).

**Before RL (dual-noise): nothing missing.** All 10 ladders (lamp and coffee × arm0–4),
all 10 checkpoints (2k–20k), both noise modes (`iid`, `duplicated`).

**DSRL: all 120 runs exist and finished** (10 ladders × 4 checkpoints × 3 seeds). Arm4 is
complete. Four coffee arm0 runs never logged their final eval at 500k env steps; their last eval is at 475k:

| task | arm | SFT ckpt | seed | wandb run | noise actor to evaluate (rorqual) |
|---|---|---|---|---|---|
| coffee | arm0 | 4k | 0 | `jqz7oabk` | `/scratch/akur/lerobot/outputs/dsrl/dsrl_arm0_coffee_04000/step4000_seed0/step_500001` |
| coffee | arm0 | 4k | 1 | `ogvjozc4` | `/scratch/akur/lerobot/outputs/dsrl/dsrl_arm0_coffee_04000/step4000_seed1/step_500005` |
| coffee | arm0 | 4k | 2 | `35owx2rv` | `/scratch/akur/lerobot/outputs/dsrl/dsrl_arm0_coffee_04000/step4000_seed2/step_500011` |
| coffee | arm0 | 8k | 0 | `k4w3k8pd` | `/scratch/akur/lerobot/outputs/dsrl/dsrl_arm0_coffee_08000/step8000_seed0/step_500002` |

## To fill them

No retraining needed. For each row, run the DSRL eval once on the saved final actor, with
the same arguments the run used (coffee task args from `dsrl_task_args.sh`, the SFT
checkpoint `/scratch/akur/lerobot/outputs/train/2026-09-20/19-38-27_sft_arms_coffee_arm0_expert_only_lrx1_seed42/checkpoints/<004000|008000>/pretrained_model`,
the run's `--seed`, `--eval_episodes 5`), plus:

```
--eval_only --noise_actor_path <path from the table>
```

Record `eval/success_rate`, `eval/success_rate_reference`, `eval/success_rate_heldout` and
the four per-kitchen `eval/success_rate_{reference,heldout}_seed*` values. Then add each
as the 500000 entry for that seed in `results_export/dsrl.json`, or send them back and
rebuild with `results_export_work/build_export.py`.

## Not gaps, but know them

- The team's `dsrl_plots/dsrl_eval_history.csv` is missing coffee arm0 8k seed 1
  (wandb `y1ymthlb`, complete in wandb). The report table used 2 seeds there: held-out
  77%, which is 71% with all 3. The export includes it.
- Excluded on purpose: the old lamp arm1 checkpoint trained with state dropout 0.1
  (`2026-09-19/12-51-54_...arm1...`); failed or crashed duplicate DSRL attempts
  (11 coffee arm2, 2 lamp arm0 16k seed 0) that have complete reruns.
- Arm4 lamp has two finished SFT runs launched two minutes apart: `09-36-39_...lamp_arm4...`
  (forge `unt8hqrf`) and `09-38-40_...lamp_arm4...` (forge `5wlc2k8r`). Before-RL and DSRL both
  used `09-36-39`, so cite `unt8hqrf`.
- Arm4 before-RL used 20 episodes per environment; arms 0–3 used 25. Same eval otherwise.
- Stale before-RL runs from the old eval version (`dualnoise_<run>`, scored on the in-loop
  SFT environments) are still in both wandb projects: 8 in `real-lab/lerobot-dex-eval`, 2 in
  forge `chacloutier-4B/lerobot-dex-eval`. None are in the export.
