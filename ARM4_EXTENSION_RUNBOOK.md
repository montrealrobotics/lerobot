# Runbook — arm4 for the extrapolation extension

Arms 0–3 are done. arm4 needs the same four jobs, run on Mila from your own scratch. Why these cells:
the held-out cell of the original design sits _between_ the two training environments and scores the same
as them for every arm; cells _outside_ the training span are where the arms separate. General SFT/DSRL
runbook: [SFT_ARMS_RUNBOOK.md](SFT_ARMS_RUNBOOK.md).

|     | what                                                                                                   | jobs | GPU time     |
| --- | ------------------------------------------------------------------------------------------------------ | ---- | ------------ |
| A   | lamp, before RL: frozen SFT ladder on the 2 extrapolated lamp placements                               | 2    | ~2 × 3 h     |
| B   | lamp, after DSRL: your 12 existing **final** lamp noise actors, re-scored on train / held-out / extrap | 12   | ~12 × 30 min |
| C   | coffee, before RL: frozen SFT ladder on the 5 new coffee cells                                         | 2    | ~2 × 3.5 h   |
| D   | coffee, DSRL: **new** training runs on the lateral coffee setup, 4 checkpoints × 3 seeds               | 12   | ~12 × 14 h   |

No retraining for lamp: B scores the actors you already have.

## 0. Code

`git fetch && git checkout dsrl && git pull`. This runbook ships in the same commit as the code it needs, so if you can read it in your checkout you have that code. The commit adds:

- `FixtureShiftWrapper` and the `fixture_shift` / `extrap` bank support in `src/lerobot/envs/robocasa_placement_bank.py`
- `--results-name` and the noise-mode selection in `examples/training/eval_pi05_dual_noise.py`
- the `coffee_lat` task in `dsrl_mila.sbatch`; `LEROBOT_REPO` / `LEROBOT_SCRATCH` overrides in
  `dsrl_mila.sbatch`, `dual_noise_eval_mila.sbatch` and `lamp_extrap_eval.sbatch`; the controller path is now
  resolved from the installed robosuite, so no launcher edits are needed

## 1. Banks — and one check that decides whether B is valid

```bash
SHARED=/network/projects/real-g-grp/dex_vla_checkpoints/placement_banks
MINE=/network/scratch/c/cloutiec/lerobot/placement_banks
for b in screwlightbulb_xarm6_seed1_pair_extrap coffeepressbutton_pandadex_s1_lateral; do
  mkdir -p $MINE/$b && cp $SHARED/$b/bank.json $MINE/$b/
done
md5sum $MINE/screwlightbulb_xarm6_seed1_pair/bank.json   # the bank your arm4 lamp DSRL trained on
```

**That md5 must be `cf7019dd418f0340b87b6bab18a3382b`.** The extrapolated placements sit 7.4 cm past
_that_ pair's two training starts. If your lamp DSRL trained on a different pair (the rebuilt-from-default
issue from 2026-10-02), B measures something else — skip B and tell Artur. A, C and D are unaffected.

Expected: `_pair_extrap` = `46b844120912e60b7f210bf3656d54c0`, `_s1_lateral` = `98aa92b1e1a6583529899202e301ae56`.

## 2. Common settings

Run from your repo checkout, with the same venv you used for arm4 (numpy 2.2.5). Always partition **long**.

```bash
export LEROBOT_REPO=<your lerobot checkout>
export LEROBOT_SCRATCH=/network/scratch/c/cloutiec
S=$LEROBOT_SCRATCH/lerobot
LAMP=$S/outputs/train/2026-09-24/09-38-40_sft_arms_lamp_arm4_lora_vlm_full_expert_lrx1_seed42
COFFEE=$S/outputs/train/2026-09-23/22-49-49_sft_arms_coffee_arm4_lora_vlm_full_expert_lrx1_seed42
SB="--partition=long --export=ALL --output=$S/slurm_logs/%x-%j.out"
mkdir -p $S/slurm_logs
```

Check the two run dirs exist and hold all ten rungs (`ls $LAMP/checkpoints $COFFEE/checkpoints`, 002000–020000).

## 3. Jobs

**Job names matter** — the results export parses them. Keep them exactly as written.

### A. Lamp, before RL (frozen ladder, extrapolated cells)

```bash
LB="--placement-bank $S/placement_banks/screwlightbulb_xarm6_seed1_pair_extrap/bank.json --eval-sets extrap \
    --episodes-per-cell 20 --no-wandb --results-name dual_noise_lamp_extrap.json"
sbatch $SB --job-name=lamp_extrap_sft_arm4_a dual_noise_eval_mila.sbatch $LAMP lamp --steps 2000 4000 6000 8000 10000 $LB
sbatch $SB --job-name=lamp_extrap_sft_arm4_b dual_noise_eval_mila.sbatch $LAMP lamp --steps 12000 14000 16000 18000 20000 $LB
```

Writes `$LAMP/dual_noise_lamp_extrap.json` (both noise modes; the two jobs merge into it safely).

### B. Lamp, after DSRL (eval-only on your final actors)

```bash
export LAMP_EXTRAP_BANK=$S/placement_banks/screwlightbulb_xarm6_seed1_pair_extrap/bank.json   # read by the launcher
for STEP in 2000 4000 8000 16000; do for SD in 0 1 2; do
  sbatch $SB --job-name=lamp_extrap_dsrl_arm4_${STEP}_s$SD lamp_extrap_eval.sbatch dsrl \
    $LAMP/checkpoints/$(printf %06d $STEP)/pretrained_model \
    $S/outputs/dsrl/dsrl_arm4_lamp_${STEP}_s$SD/seed$SD/final
done; done
```

5 cells × 10 episodes each, deterministic actor. Results: `$S/eval/lamp_extrap/lamp_extrap_dsrl_arm4_<step>_s<seed>.log`.
Without `LAMP_EXTRAP_BANK` the launcher falls back to a 6-cell candidate bank — the numbers would not match arms 0–3.

### C. Coffee, before RL (frozen ladder, 5 lateral cells)

```bash
CB="--placement-bank $S/placement_banks/coffeepressbutton_pandadex_s1_lateral/bank.json --eval-sets train,heldout,extrap \
    --episodes-per-cell 25 --reject-start-max-retries 20 --no-wandb --results-name dual_noise_coffee_lateral.json"
sbatch $SB --job-name=dn_coffeelat_arm4_a dual_noise_eval_mila.sbatch $COFFEE coffee --steps 2000 4000 6000 8000 10000 $CB
sbatch $SB --job-name=dn_coffeelat_arm4_b dual_noise_eval_mila.sbatch $COFFEE coffee --steps 12000 14000 16000 18000 20000 $CB
```

### D. Coffee, DSRL training (new)

Smoke-test one first (~20 min), then launch all 12:

```bash
sbatch $SB --time=1:00:00 --job-name=dsrl_arm4_coffee_lat_smoke dsrl_mila.sbatch coffee_lat \
  $COFFEE/checkpoints/004000/pretrained_model 0 --total_steps 1000 --min_buffer_size 20 --eval_freq 800 --eval_episodes 1
# expect in the log: "Eval: 5 placement cells (train,heldout,extrap)" and an "[eval @ step ...]" line

for STEP in 4000 8000 14000 18000; do for SD in 0 1 2; do
  sbatch $SB --mem=48G --job-name=dsrl_arm4_coffee_lat_${STEP}_s$SD dsrl_mila.sbatch coffee_lat \
    $COFFEE/checkpoints/$(printf %06d $STEP)/pretrained_model $SD
done; done
```

`coffee_lat` = kitchen seed 1 (nespresso); the machine slides sideways between two training offsets
(−4 / 0 cm), held out −2 cm, extrapolated −8 / +4 cm; start-pose rejection with 20 retries. Everything
else (reward, SAC, 500k env steps, eval every 25k, 5 episodes per cell) is unchanged from the original
coffee runs. ~14 h each on an L40S; preemption-safe (`--resume`, requeue). wandb logs to your usual project.

## 4. Known failure

A bare `Aborted (core dumped)` / exit 134 right after the model loads is a native EGL abort on a bad node,
not your config. On rorqual it was always the same few nodes; on Mila we have not seen it. Resubmit; if it
repeats on the same node, add `--exclude=<node>`.

## 5. Send the results back

Into the shared drop folder (group `real-g-grp`; plain `cp` keeps it readable):

```bash
DROP=/network/projects/real-g-grp/dex_vla_checkpoints/arm4_results
mkdir -p $DROP/dn/$(basename $LAMP) $DROP/dn/$(basename $COFFEE)
cp $LAMP/dual_noise_lamp_extrap.json      $DROP/dn/$(basename $LAMP)/        # A
cp $COFFEE/dual_noise_coffee_lateral.json $DROP/dn/$(basename $COFFEE)/      # C
cp $S/eval/lamp_extrap/lamp_extrap_dsrl_arm4_*.log $DROP/lamp_extrap/        # B
cp $S/slurm_logs/dsrl_arm4_coffee_lat_[0-9]*_s?-*.out $DROP/coffee_lat_logs/ # D (the slurm logs hold the eval history)
```

Artur's `results_export_work/fetch_ext.sh` reads that folder; `build_extension.py` then fills arm4 into
`results_export/` and `make_figures_ext.py` adds it to every figure.
