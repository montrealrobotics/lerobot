# Steerability study: running log

Decisions and results, newest at the bottom of each section. The brief for this work is the
2026-10-03 task description (find a frozen-checkpoint metric that explains DSRL outcomes beyond dup).

## 2026-10-03: arm4 lamp reruns on the September bank

All 12 `dsrl_arm4_lamp_<step>_s<seed>_sepbank` runs (jobs 11046012–11046023) COMPLETED, exit 0,
each reached ≥500k steps with a final eval at 500k. The smoke run (11045859) and the dual-noise
eval (11045894, ended 2026-10-03 11:09) also completed. `~/scratch/claude_tmp/arm4_rerun/runs.tsv`
regenerated with final states and wandb IDs (backup `runs.tsv.bak_20261003`).

**Bank check.** The launcher passed `--placement_bank` twice: first its default (the invalid 09-27
`..._pair/bank.json`), then the explicit `..._pair_dsrl/bank.json` at the end. The script uses
argparse `store`, so the last value wins: every rerun used `_pair_dsrl` (md5 919c6488…). The wandb
config logs only the raw argv, so it shows both paths; don't misread it as the invalid bank.

| rung | seeds solved (train ≥0.8) | final train / heldout per seed | first train ≥0.8 |
|---|---|---|---|
| 2k | 3/3 | 1.0/1.0, 1.0/1.0, 1.0/1.0 | 150k, 300k, 200k |
| 4k | 0/3 | 0/0 ×3 | never |
| 8k | 0/3 | 0/0 ×3 | never |
| 16k | 0/3 | 0.5/0, 0.5/0, 0.1/0 | never |

wandb (chacloutier-4B/lerobot-dsrl): 2k 049brp9y h7oijfjv 9w29s3m7; 4k hypdplik 4s8s3mav ih6yiu4z;
8k srvgrydc wrlumb4i d3h802pd; 16k 1oh4f7qu gv9gbijo fu7ndv1p.

First look against dup (rerun dual-noise eval, 25 ep/cell): 2k dup high → all solve, as predicted.
4k dup still 28% but all three seeds fail, and 16k dup ~5% but two seeds hold one start (0.5).
So arm4 lamp does not simply follow dup; arm4@4k and arm4@16k are informative cases.

**Missing inputs.** `results_export_work/build_export.py` and `summary.json` are not on Mila
(searched both scratches, home and the shared project dir). Export formats are recoverable from the
existing `dsrl.json` / `dual_noise.json` / `warmup_success.csv`.

## 2026-10-03: arm4 lamp merge into the exports

Brief updated: Artur's export pipeline is in `export_tools/` (his `results_export_work/`), his finished
export in `export_artur/`.

1. **`dn/lamp_arm2.json` reconstructed** from `dual_noise.json` results.lamp.arm2 (original unreadable in
   Artur's folder; asked for). Same keys as `dn/lamp_arm{0,1,3}.json` except `eval_s` (unknown, omitted;
   `build_export.py` doesn't read it). Marked with a top-level `reconstructed_from` field, which the parser
   ignores (it only looks up step keys). The reproduction check is circular for this entry. When Artur's
   original arrives: replace it, rerun step 2.
2. **Reproduction check passed.** `build_export.py` with only `OUT` changed (→ `steerability/export_repro/`)
   reproduces `dsrl.json`, `dual_noise.json`, `summary.json` and `missing.json` byte for byte.
3. **Swap.** `export_tools/fetch_arm4_lamp_reruns.py` fetched the 12 rerun histories (wandb.ai
   chacloutier-4B/lerobot-dsrl; 20 evals each on the 25k grid, last at ~500k) into
   `dsrl_wandb_evals.json` (previous: `dsrl_wandb_evals_oldbank_arm4.json`) and cached full histories in
   `lerobot/diag/p0/history/<id>.csv` (note: the p0 cache is under `lerobot/diag/`, not `steerability/diag/`).
   Mapping old → new in `export_tools/arm4_lamp_rerun_replacements.json`:
   2k s0/s1/s2 w9wtw86b/enrey7u1/hpiu4584 → 049brp9y/h7oijfjv/9w29s3m7; 4k 5l6gl74q/8si7w1bx/iqqqpqcy →
   hypdplik/4s8s3mav/ih6yiu4z; 8k 0gjtolju/tegixvlk/cwkwwcb1 → srvgrydc/wrlumb4i/d3h802pd; 16k
   am1nwt30/30pedpgb/zepe5bc6 → 1oh4f7qu/gv9gbijo/fu7ndv1p (old: forge chacloutier-4B/lerobot-dsrl).
   `dn/lamp_arm4.json` ← `~/scratch/runs/lamp_arm4_sepbank/dual_noise_dsrl_cells.json` (25 ep/cell, like
   arms 0–3; the old one used 20); old kept as `dn/lamp_arm4_oldbank.json`.
4. **Regenerated** via `build_export.py` (`OUT` → `steerability/export_new/`, and the `meta.source` string
   now names wandb.ai for lamp arm4), installed into `steerability/`; previous files kept as
   `{dsrl,dual_noise,summary}_oldbank_arm4.json`, `warmup_success_oldbank_arm4.csv`. A structural diff
   against the previous exports shows changes only under `results.lamp.arm4` (+ that meta string).
   New md5s: dsrl e078d5f8…, dual_noise 0e0fe277…, summary a8974ec2…, warmup_success 2fbd00bc….
   `warmup_success.csv` definition (recovered; no producer script on Mila): mean `episode/success` over
   all collection episodes logged at `_step` ≤ 16000, pooled over the 3 seeds; `dual_noise_dup_train` =
   dup on the training cells. This reproduces all 40 previous rows exactly. Lamp arm4 new rows:
   2k 65 ep 0.523 / dup 0.42; 4k 56 ep 0.357 / 0.24; 8k unchanged (48, 0.021 / 0.0); 16k 48, 0.042 / 0.08.
   Note: the brief's arm4 dup numbers (40/28/4/5%) are all-cell; training-cell dup is 42/24/0/8%.
5. **wandb:** the 12 reruns are tagged `bank_pair_dsrl`, with a note giving the effective bank + md5 and
   the launcher quirk. The smoke run `utggbgcv` no longer exists in wandb (not tagged). Shared launcher
   not changed.

## 2026-10-03: analyze.py changes

The `analyze.py` here had no `--exclude` or `--baseline-csv` (no other copy on Mila), so it never
excluded lamp/arm4. Added (backup `analyze.py.bak_20261003`):
- `--baseline-csv` → `base.warmup`; `--exclude TASK/ARM ...` (default: none).
- `split` per env (train / heldout from dsrl.json meta); results per (task, split ∈ train, heldout, all).
- Checkpoint-level `base.dup_train`, `base.iid_train`. For heldout targets, per-env held-out dup/iid are
  dropped and the dup control is `base.dup_train`, so no held-out eval enters any predictor.
- `partial_given_warmup` next to `partial_given_dup`; LOAO uses the split's dup control.
- `final` now = mean over evals at ≥400k env steps (summary.json's definition) instead of the last 5
  evals; differs only for the 4 coffee arm0 runs with no 500k eval.

## 2026-10-03: baseline-only analysis (all 5 arms, arm4 lamp = reruns)

`python analyze.py --out ~/scratch/diag/baseline --dsrl-json dsrl.json --dual-noise-json dual_noise.json
--baseline-csv warmup_success.csv --boot 2000`. Spearman ρ [95% cluster-bootstrap CI]; units = (ckpt, env).

| task / cells | target | best baseline | ρ | beyond dup? |
|---|---|---|---|---|
| lamp train (n=40) | final | dup | 0.84 [0.67, 0.91] | warmup adds nothing (partial 0.07); dup adds beyond warmup (0.55) |
| lamp train | t80 | dup / iid | −0.66 / −0.64 | iid partial −0.55, but LOAO ΔR² −0.02 (doesn't transfer across arms) |
| lamp train | seed_std | none | all \|ρ\| ≤ 0.14 | **unexplained** |
| lamp heldout (n=20) | final | dup_train | 0.84 [0.67, 0.92] | — |
| coffee train (n=40) | final | warmup / dup | −0.42 / −0.39 (negative) | — |
| coffee train | t80 | iid | −0.69 [−0.84, −0.52] | partial −0.63, LOAO ΔR² +0.11 (holds out of sample) |
| coffee train | seed_std | none | \|ρ\| ≤ 0.15 | **unexplained** |
| coffee heldout (n=40) | final | none | all ρ 0.11–0.14, CIs span 0 | **unexplained**, the target that separates arms |
| coffee heldout | t80 | iid_train | −0.34 [−0.62, −0.01] | weak |
| coffee all (n=80) | final | dup 0.06, iid 0.26 | | matches the earlier pooled ρ 0.06 |

Lamp ρ(dup, final) = 0.84 is unchanged by the arm4 swap. Largest rank residuals (final vs dup, train
cells, per checkpoint): **under** dup: arm3@4k (dup 0.34 / warmup 0.33 → final 0.33, split seeds),
arm4@4k (0.24 / 0.36 → 0.01), arm1@4k (split); **over** dup: arm0@8k/16k (dup 0.28–0.32 → 1.0, 3/3).
Coffee: arm0 rungs (lowest dup 0.08–0.16 → final 0.93–0.97) over; late arm1/3/4 rungs (dup 0.5–0.8) under.
arm0 over-performs dup on both tasks; the 4k rungs of the LM-training arms under-perform it on lamp.

## 2026-10-03: prep for steps 2–3 (reference states, Phase 1a + 1c)

- **Checkpoints.** `find_checkpoints.py --search-roots <shared train/> <my outputs/train/>` →
  `~/scratch/diag/{checkpoints,manifest}.tsv`: all 40 rungs `ok`, all 10 (task, arm) in the manifest.
  lamp/arm0 resolved to Artur's original folder; the manifest now points at the shared copy instead.
- **Bank provenance in every result** (backups `common.py.bak_20261003`, `collect_states.py.bak_20261003`):
  `common.bank_info(args)` reads `<TASK>_BANK` from the `--env-factory` module and hashes it;
  `write_result` stores it as `meta.bank = {path, md5}` plus `meta.states = {path, sha1, bank, checkpoint}`
  for the state set used; `collect_states.py` stores `meta.bank` in the pickle. Mock chain
  (states → noise → obs) passes; against the real wiring: lamp `cf7019dd…`, coffee `25d89adb…`.
- **wandb.** Phase 1 scripts don't log, so each job uploads every result right after it's written with
  `upload_results.py` → `chacloutier-4B/lerobot-steerability`, name `p1{a,c}_<task>_<arm>_<step>_bank{pair,kitchens}`,
  `--bank` from the result's own `meta.bank`.
- **Jobs.** `~/scratch/diag/jobs/submit_p1ac.sh` (one body script per job in `jobs/bodies/`, logs in
  `jobs/logs/`). Not submitted yet; job list shown for approval.

## 2026-10-03: staged launch of steps 2–3 (budget ≤ ~15 GPU-h)

Approved plan: state sets + one Phase 1 job first, measure, then release the rest (show me if the full set
projects well above 15 GPU-h). Changes before launch (backups `*.bak_20261003`):
- `p1_obs_reliance.py`: vision tower added to the default feature modules (hooked on its first call per
  sample = the first camera, `robot0_agentview_left`), so CKA vs pi05_base covers SigLIP; arm0 must come
  out ~1 for vision tower and projector (free correctness check). The reference cache key now includes the
  module list; the cache write is atomic (tmp + rename). The cache is valid across arms because every
  arm's pre/postprocessor files are byte-identical within a task (checked lamp and coffee, all 5 arms).
- `common.py`: `meta.timing` in every result: `load_s` (checkpoint), `load_ref_s` + `ref_features_s`
  (pi05_base pass, only on a cache miss), `ref_cache_hit`, `features_s`, `total_s` (since `import common`).
- `submit_p1ac.sh`: `states` / `p1 <task/arm>...` stages; Phase 1 jobs are submitted **held**; 1c runs
  before 1a. Reference sets: 12 rollouts per cell (4 risked cells with no success at arm0's 14–30% dup),
  coffee records every macro step (episodes are ~10 macro steps).
- A `~/scratch/diag/states/ref_lamp.pkl` from 2026-10-02 18:48 was already there (arm0@16k, 4
  rollouts/cell, s1_mid 0 successes, no bank in meta; the wiring then pointed at `_pair_dsrl`). Renamed to
  `ref_lamp_20261002_unrecorded_bank.pkl`, not used.
- Submitted: `states_ref_lamp_bankpair` 11059751, `states_ref_coffee_bankkitchens` 11059752.
- States checks (`check_p1.py states`): **lamp OK**: `ref_lamp.pkl` bank cf7019dd (September `_pair`),
  12 rollouts/cell, success 0.25 / 0.25 / 0.33 (start_0 / start_1 / mid), 337 / 301 / 340 states, each from
  both successes (58 / 22 / 92) and failures. Job 29.5 min. **Coffee failed the mix check**: 12 rollouts/cell
  gave 0 successes on seed1_L4S8 and seed11_L1S4 (arm0@14k dup there is 8% / 4%). Kept as
  `ref_coffee_12roll.pkl` (unused); recollecting with 48 rollouts/cell, record every 2 (job 11060010,
  ~1.1 GPU-h). P(still no success): ~2% seed1, ~14% seed11. Coffee 12-rollout job: 16 min.
- Submitted + released `p1ac_lamp_arm0_bankpair` 11060011 and `p1ac_lamp_arm4_bankpair` 11060012
  (arm0 for the CKA check, arm4 for timing). The other 8 not submitted yet.
- **1c bug (fixed 2026-10-03 ~15:40):** the pi05_base reference pass failed in both lamp jobs at 2k
  ("All image features are missing"): `PI05Policy.from_pretrained("lerobot/pi05_base")` uses the base
  config's cameras (`base_0_rgb`, `left/right_wrist_0_rgb`), not ours. `Pi05Handle` now loads base
  weights with the evaluated checkpoint's config (`config=` from `processor_path`), as DSRL's LoRA
  loader does. The jobs start a fresh process per rung, so 4k/8k/16k pick up the fix; 1c @2k for
  lamp arm0 and arm4 must be rerun. 1a was unaffected (runs after 1c regardless).
- **Coffee reference set OK** (job 11060010, 33 min): `ref_coffee.pkl`, bank 25d89adb, 48 rollouts/cell, every 2.
  seed0 0.21 (226 states: 36 from successes / 190 failures), seed1 0.02 (238: 3 / 235, one successful
  rollout), seed11 0.04 (237: 7 / 230), seed14 0.12 (233: 23 / 210). Fixed from here on.
- 1a sanity on the first two results (lamp arm0@2k, arm4@2k): per-state pair_spread > 0 and varies
  (arm0 29.8–35.0, CV 0.03; arm4 23.5–31.3, CV 0.06); noise_share 0.90–0.92 (arm0), 0.85–0.86 (arm4);
  bank, state-set sha1 and noise_seed 1234 recorded. Timing: arm0 load 475 s + 1a ~817 s; arm4 load 136 s
  + 1a ~817 s.
- **1c fix confirmed on arm0@4k** (first rung after the config fix): CKA vs pi05_base = **1.0000 for the
  vision tower and the projector**, as required for arm0 (frozen). Timing: load 418 s, pi05_base load 209 s
  + features 22 s (cache miss, once per task), features 18 s, total 934 s. Reference cache written once
  (`ref_cache/lamp_41451d980474_lerobot_pi05_base_22ccb665.npz`).
- **Second 1c bug (fixed ~15:51):** no language-model features at all. pi05 runs the LM prefix pass as
  `language_model.forward(...)`, which bypasses forward hooks on `language_model`. The hook is now on
  `paligemma.model.language_model.norm` (final norm, called normally = LM last-layer output; metrics
  still named `*_language_model`). 1c now raises if any requested module captures no features. The
  module list changed, so the reference cache is rebuilt once under a new key. Lamp 1c to rerun: arm0 2k
  (failed), 4k (no LM); arm4 2k, 4k (failed); plus any rung that started before 15:51.
- Phase 1 sanity so far: arm0's per-cell obs reliance is low (blank third-person 0.12, swap-cell image
  0.03 of the noise spread; swap-cell proprio 0.26–0.36): noise moves arm0's action far more than the image does.
- **LM hook confirmed** (arm0@8k, first rung after 15:51): CKA vs pi05_base = 1.0 for vision tower, projector
  **and language model**: the arm0 correctness check passes on all three modules. arm4@16k (LoRA LM):
  vision 0.84, projector 0.77, LM 0.29. arm4@16k noise spread 3.5 (vs ~27 at 2k) and swap-cell proprio moves
  the action 4.1× the noise spread: the late arm4 checkpoint barely responds to noise (cf. its dup collapse).

### Runtime breakdown (L40S, lamp, 120 states for 1a / 90 for 1c)

| part | time | notes |
|---|---|---|
| checkpoint load, cold, shared project dir | 420–475 s | arm0 (projects FS) |
| checkpoint load, warm or my scratch | 133–138 s | 2nd load of the same rung (page cache) or arm4 |
| pi05_base reference pass (1c, once per task × module list) | 210–230 s | load ~190–210 s + features ~22 s |
| 1c compute | 285–292 s | 90 states × 8 noises × (1 + 5 perturbations) + features |
| 1a compute | 811–817 s | 120 states × 64 noises + local gains + noise_share; **dominates** |

Per rung, lamp: ~28 min (shared FS) / ~23 min (my scratch). Coffee has 4 cells (160 / 120 states): ~34 / ~29 min.
- Jobs done: `p1ac_lamp_arm4` 1:24:51 (6 uploads), `p1ac_lamp_arm0` 1:58:29 (7 uploads). GPU-h used so far
  in steps 2–3: ~4.7 (states 1.3 incl. the coffee recollection, lamp arm4 1.4, lamp arm0 2.0).
  Valid 1c: arm0 8k, 16k; arm4 16k. 1a: all 4 rungs for both.
- Early contrast at 16k (both low dup: arm0 0.28, arm4 0.08 train): 1a noise_share arm0 0.49–0.53 vs arm4
  0.06–0.10; median per-state pair_spread 16.2 vs 2.6. arm0@16k solves 3/3, arm4@16k 0/3 (two seeds hold
  one start). n = 2 checkpoints; just a lead.
- Projection for the rest (shown to Charlotte, awaiting choice): ~16.6 GPU-h for the other 8 ladders +
  ~0.75 for lamp 1c reruns → ~22 GPU-h total vs the 15 budget. Options: (1) run as is; (2) 1a at 20
  states/cell (~19 total, redo lamp 1a); (3) ~0.3 GPU-h test of a larger micro-batch first.

## 2026-10-04: micro-batch test (option 3) + lamp 1c reruns

Clarifications (Charlotte's questions):
- The "noise_share 0.85–0.92" in the first sanity report came from the only two results then: arm0@2k
  (0.915 / 0.898 / 0.904) and arm4@2k (0.858 / 0.852 / 0.845). Full lamp ladders: arm0 0.91 → 0.84 → 0.68 →
  0.51, arm4 0.85 → 0.63 → 0.28 → 0.08 (2k → 16k). Sanity checks now always name the rung.
- The single `ValueError: All image features are missing` in the arm0 log (lines 17–39) precedes the first
  written result (arm0 2k 1a, line 57): it is the 2k 1c (camera-name bug). arm0 4k 1c ran but had no LM
  features; it was never counted as valid. Valid lamp 1c before reruns: arm0 8k/16k, arm4 16k.

Setup:
- `common.py` (backup `common.py.bak_20261004`): `--micro-batch` (default 16 = the previous hard-coded
  value; results without the field ran at 16), recorded in `meta.argv.micro_batch`; `meta.gpu` = peak
  allocated/reserved GPU memory of the process.
- 1c draws 8 noises per state, so it runs as one batch of 8 for any micro-batch ≥ 8: 1c is unaffected.
  1a makes 2 calls × 64 noises per state: affected.
- 1c files without LM features moved to `p1_obs_reliance/invalid_no_lm/` (arm0 4k, arm4 8k).
- Jobs: `p1a_batchtest_lamp_arm4_bankpair` 11073167 (arm4@16k 1a at micro-batch 64 then 128, out
  `~/scratch/diag/batchtest/mb{64,128}/`, nvidia-smi every 5 s; not uploaded, test only);
  `p1c_rerun_lamp_bankpair` 11073168 (1c for arm0 2k/4k, arm4 2k/4k/8k; uploaded).
- Criterion: every 1a per-state metric within 1% of batch 16 and identical rankings
  (`compare_microbatch.py`). Pass → rerun lamp arm0/arm4 1a at the new size. Fail or speedup < 1.5× →
  option 1 (batch 16, everything as is, ≤ ~22 GPU-h approved). 1a stays at 40 states/cell either way.
- **Paused 2026-10-04 at Charlotte's request** (GPUs needed for other experiments): 11073167 and 11073168
  cancelled after 7 min, before writing any result. Nothing else queued. To resume: resubmit
  `jobs/bodies/p1a_batchtest_lamp_arm4.sh` and `jobs/bodies/p1c_rerun_lamp_bankpair.sh` through `job.sh`
  (sbatch lines as above), then apply the micro-batch decision rule.
- **Resumed 2026-10-04** (Charlotte): resubmitted as batch test 11073200, 1c rerun 11073201 (same bodies).
- **Micro-batch test result (11073200, 35 min, arm4@16k 1a)**: **fails both criteria → option 1 (batch 16).**
  Speedup 0.99× (1a compute 815 s at 16 → 826 s at 64, 821 s at 128): the GPU is already saturated.
  64 and 128 are identical because each 1a call has 64 noises (one batch either way). Per-state vs batch
  16: median |rel diff| 0.2–0.4%, max 1.7–6.6% (local_gain, hand_var largest), 33–47 rank changes over 120
  states (Kendall τ 0.988–0.994); per-cell means within 0.7%. Between-checkpoint gap (arm4 8k vs 16k,
  median per state) 25–760%, so the bf16 batch-shape effect is far below the signal, but it means the
  batch size must stay fixed: all results use 16. Peak GPU memory 24.9 / 46 GB (torch peak alloc 15.4 GB).
  Lamp arm0/arm4 1a stay as they are (batch 16).
- Submitted + released the other 8 ladders at batch 16 (1c first): lamp arm1/2/3 11073428–30,
  coffee arm0–4 11073431–35. Approved budget ≤ ~22 GPU-h total for steps 2–3; used before these ≈ 6.0
  (incl. 0.6 for the batch test, 0.2 for the cancelled pair); projection for the 8 ≈ 16.6 → ≈ 23 total.
- Lamp 1c reruns: arm0 2k/4k CKA 1.0 on all three modules (arm0 now 1.0 at every rung); arm4 2k LM 0.33,
  projector 0.82, vision 0.89 (LM drift already there at 2k; 16k LM 0.29).
- **Interpretation caveat, LM CKA:** lamp 2k LM-output CKA arm1 0.27, **arm2 (frozen LM) 0.26**, arm3 0.30.
  The hook measures the LM's output, which depends on the (trained) vision tower + projector upstream, so
  "CKA ≈ 1 for never-trained modules" holds only when everything upstream is frozen too (arm0). LM-output
  CKA tracks upstream representation drift, not LM weight change. Isolating the LM would need each
  checkpoint's LM fed pi05_base's projector outputs; not planned.

## 2026-10-05: Phase 1 (1a + 1c) complete, first analysis

All 80 results (1a + 1c × 40 checkpoints) validated: banks, one state set per task, noise seed 1234,
micro-batch 16, all three CKA modules, arm0 CKA 1.0 at every rung. 8 ladder jobs: 15.7 GPU-h; steps 2–3
total ≈ 22.4 GPU-h. `analyze.py --boot 300` → `~/scratch/diag/analysis/{table,correlations}.csv, report.md`;
`focus_table.py` → per-checkpoint values (`focus_table.md`); quick partials in `quick_correlations.csv`.

- **Coffee held-out final** (dup explains nothing): local_gain partial|dup 0.39, LOAO ΔR² +0.23; vision-tower
  CKA 0.37 / +0.27; noise_share 0.36 / +0.07. Raw ρ only 0.08–0.15 (CIs span 0): the signal appears only
  conditional on dup. The bootstrap CI covers raw ρ, not the partial; a CI on the partial is still needed.
  Lead, not a finding (n = 20 checkpoints, many tests).
- Lamp final: part_ratio / pair_spread / noise_share partial 0.23–0.32, LOAO ≈ 0. noise_share falls with SFT
  step in every arm, so it largely tracks training length.
- Seed reliability: nothing (|partial| ≤ 0.36, LOAO ≤ +0.11).
- **4k lock-in failures (arm1/3/4@4k) are not separated** from arm2@4k (3/3) by any 1a/1c diagnostic →
  points at learnability (Phase 2).
- arm0 keeps noise sensitivity far longer (coffee noise_share 0.56–0.98 vs 0.02–0.06 late arms 1–4; lamp
  16k 0.51 vs 0.07–0.17) and has the lowest image reliance; confounded with arm identity.
- LM-output CKA inherits upstream drift (arm2 frozen LM 0.26 ≈ arm1 0.27).
- Proposed next: bootstrap the partials; Phase 2 on ~7 lamp checkpoints (arm1/3/4@4k vs arm2@4k, arm0@4k,
  arm4@16k), cost to be shown first.

## 2026-10-05: checks on the coffee held-out lead (Charlotte's three checks)

- Partial-correlation bootstrap (2000 cluster resamples) on the *original* held-out analysis: local_gain
  [0.08, 0.71], vision CKA [0.01, 0.55], noise_share [0.09, 0.58]; superseded by the leakage check below.
- **Leakage found.** (a) The held-out rows used each held-out cell's own diagnostics, i.e. metrics computed
  on held-out states. (b) 1c's checkpoint-level metrics (CKA, probes, cell_acc) were pooled over all cells'
  states including held-out ones, and 1c's swap-cell perturbations could take a held-out cell as donor;
  1c ignored `--cells`. 1a computes per cell from that cell's states only, so its training-cell rows are clean.
  Pooling, for the record: features = mean over tokens of each module's first call per sample (vision tower:
  first camera), one vector per state; CKA = linear CKA over all pooled states vs pi05_base on the same
  states; probes = 5-fold ridge (λ by CV) from features to EE pose + object positions; cell_acc = 5-fold
  nearest-centroid over cells.
- Fixes: `p1_obs_reliance.py` honours `--cells` (filters states before everything; reference cache key now
  includes the cell set; backup `.bak_20261005`). `analyze.py` (backup `.bak_20261005`): held-out targets
  use each checkpoint's training-cell mean of every p1/p2 diagnostic; `--extra-diag-dirs` (later results
  replace earlier per phase/task/arm/step); `--controls` (e.g. `sft_step`, added to partials and the LOAO
  baseline); `--analysis-name`; report notes the 3-seed noise floor of `seed_std`. Training-cell results
  reproduce exactly; held-out results change.
- Coffee 1c on training cells only (seed0_L8S6, seed1_L4S8): jobs 11079773–77 → `~/scratch/diag/train_only/`,
  wandb `p1c_trainonly_coffee_<arm>_<step>_bankkitchens`. Estimate ~2–2.5 GPU-h (load-dominated; approval
  was ~1.6). Lamp not rerun: no lamp held-out lead to protect (lamp held-out is explained by dup).
- **Result, 1a part (no GPU needed): the coffee held-out lead disappears.** With training-cell 1a
  values: partial|dup 0.05–0.09 (CIs span 0), partial|dup+sft_step 0.10–0.17 (CIs span 0); arms 1–4 only:
  −0.05 to +0.19, LOAO all negative. The earlier 0.36–0.39 came from scoring held-out cells with their
  own-state diagnostics. Vision CKA part pending the training-cell 1c.

## 2026-10-05: Phase 2 plan for the lamp lock-in set (not submitted; shown for approval)

Checkpoints: failures F = arm4@4k (0/3), arm3@4k (1/3), arm1@4k (2/3); solvers S = arm2@4k (3/3),
arm0@4k (3/3); reference arm4@16k (0/3, two seeds hold one start). Training cells only
(s1_start_0, s1_start_1), September bank, each checkpoint's own states.

Per checkpoint (≈25 s per partial rollout, ~40 s per full lamp episode, 3–7 min per checkpoint load):
own states 2 cells × 4 rollouts (~10 min) → branching `--k 8 --states-per-cell 3 --repeats 2`
(2 × 3 × 8 × 2 = 96 rollouts, ~40 min) → smoothness on mixed-outcome states, trimmed
`--sigmas 0.1,0.3,0.8 --n-perturb 6` (≤ 6 states × 27 = 162 rollouts, ≤ ~70 min) → transfer between the
two starts (~15 min). ≈ 2.2–2.5 GPU-h per checkpoint, ≈ 13–15 GPU-h for 6. (Default smoothness, 49
rollouts per state, would add ~1.5 GPU-h per checkpoint.)

Pre-registered criteria (per checkpoint, means over states, state-bootstrap CIs):
- Learnability answer: all F below both S on basin_radius (or retention at σ = 0.3) and above on
  interp_flips, at comparable frac_any; ideally ordered arm4@4k < arm3@4k < arm1@4k.
- Existence/discovery answer: F below S on frac_any and the steerability gap (best_split − mean_p).
- No answer: F and S overlap on all of these (state-bootstrap CIs of F−S include 0) → next, 1b
  reachability (box/tanh noise) on the same set.
- arm4@16k: expected low frac_any (few good noises) with any good noise confined to one start (transfer ~0).
- Phase 2 pilot approved and submitted: lamp arm4@4k, job 11079871 (own states → branching → smoothness → transfer, training cells, 6 h limit).

## 2026-10-05: arm4 added to the extension export (lamp_extrap, coffee_lateral)

**`~/scratch/dexrobot/results_export 2/` holds a newer Artur export than `export_artur/`** (both have arm4
as an empty placeholder; main files identical, extension files differ only by additions):
- `dual_noise_lamp_extrap.json`: extra rungs arm1 6/10/12/14/18/20k, arm2 6/10/12/14/18/20k,
  arm3 6/10/12/14/18k (all arms 0–3 have the full 2–20k ladder there).
- `dual_noise_coffee_lateral.json`: extra rungs arm1 2/4/6/8/10k, arm2 20k.
- `dsrl_coffee_lateral.json`: every arm 0–3 run has full curves to 500k env steps (in `export_artur/`
  they stop at 125–350k, so its after-DSRL 400–500k window is empty), plus runs absent from `export_artur/`:
  arm1 8k/14k, arm2 4k/14k (3 seeds each). `summary_ext.json` coffee rows for arms 0–3 differ accordingly.

**Build.** `export_tools/fetch_ext.sh` and `build_extension.py` are now arm4-only (Artur's versions:
`*.orig`): fetch copies the drop folder `/network/projects/real-g-grp/dex_vla_checkpoints/arm4_results/`
(26 files) into `ext_raw_arm4/`; build makes arm4 only and splices it into a base export, arms 0–3
verbatim; lamp before-RL training/held-out cells from `steerability/dual_noise.json` (sepbank). Scripts glob
run dirs, nothing hardcoded the runbook's 09-38-40 (lamp run is 09-36-39). Our `ext_raw/` lacks arm1 (all),
arm2 coffee DSRL and other raw files, so arms 0–3 could not be rebuilt from raw.
- `export_ext_rebuilt/`: arm4 spliced on `export_artur/` (only arm4 differs from it; verified).
- **Used for the paper export: arm4 spliced on `results_export 2`'s own files** (built with BASE =
  that folder; arm4 entries byte-identical to `export_ext_rebuilt/`, arms 0–3 identical to the folder's).
  Copied there with the regenerated `dual_noise.json`/`dsrl.json`/`summary.json` (replacing Artur's
  originals, b152c0c1…); previous files kept as `*.pre_arm4`, `figures.pre_arm4/`. `make_paper_figures.py`:
  no "no data for arm4" notes (only the two step-0 placeholders remain). The other figure scripts
  (`make_figures.py`, `make_figures_ext.py`) were not rerun.
- Provenance checks: lamp extrap before-RL jobs 11073328/29 (`lamp_extrap_sft_arm4_a/b`) passed
  `--placement-bank .../screwlightbulb_xarm6_seed1_pair_extrap/bank.json` (md5 46b84412…); coffee lateral
  before-RL jobs 11073342/43 (`dn_coffeelat_arm4_a/b`) passed `.../coffeepressbutton_pandadex_s1_lateral/
  bank.json` (md5 98aa92b1…); both bank files last modified 10-04 14:55, before the jobs started. Lamp
  extrap DSRL logs score the `_sepbank` final actors on SFT run 09-36-39 over the 5 cells; coffee DSRL logs
  all use SFT run 22-49-49 and reach 500k.
- arm4 lamp after DSRL (train/held-out/extrap): 2k 1.00/1.00/0.85, 4k 0.02/0.00/0.03, 8k 0/0/0,
  16k 0.38/0.00/0.00. Coffee: 4k 0.95/0.97/0.91, 8k 0.97/0.93/0.63, 14k 0.94/0.95/0.33, 18k 0.82/0.85/0.34.

## 2026-10-05: coffee training-cell 1c result + Phase 2 pilot (arm4@4k)

- Coffee 1c on training cells only (jobs 11079773–77, 2.4 GPU-h total; 20/20 uploaded). Held-out final with
  training-cell 1c: all arms, CKA (vision / projector / LM) partial|dup ≈ 0.33 (CIs from −0.05 to +0.03 at the low
  end), LOAO +0.29 (vision, projector); **arms 1–4: every CI spans 0, LOAO negative for 11 of 12 rows** (LM CKA
  given dup+step: 0.39 [−0.13, 0.66], LOAO +0.16; one row among many). Verdict: arm0 identity (CKA ≡ 1),
  not a metric. **The coffee held-out lead is dead** (1a: leakage; 1c: arm0).
- Pilot p2 lamp arm4@4k (job 11079871): 1.49 GPU-h (own states ~11 min, branching 33 min, smoothness 35 min,
  transfer 10 min, ~19 s per branch rollout). Two problems, results not usable yet:
  1. **Determinism check failed**: max_state_diff 0.997 (same outcome) restoring the t=0 state twice and running
     3 chunks with fixed noise, on a freshly built env. The brief's "restore is deterministic" was apparently
     verified elsewhere (coffee?); lamp not. Diagnostic job 11080628 (`diag_determinism.py`): per state component,
     env-only replay vs policy vs repeated inference, fresh env vs after reset.
  2. State selection spreads states over the whole episode incl. t=960 of 990 (30 steps left, cannot
     succeed): 1 of 3 states per cell wasted. Needs a cap (e.g. t ≤ 0.6 × horizon) and more states per cell
     (3 gives 1–2 mixed states for smoothness; per-checkpoint precision too low for the F-vs-S comparison).

## 2026-10-05: Phase 2 determinism bug found and fixed

`diag_determinism.py` (jobs 11080628, 11080680, 11080723; ~6 min each), lamp arm4@4k own states, s1_start_0,
t = 0 and t = 96, fresh env and after reset:
- env-only replay of fixed actions from a restore: identical (≤ 1e-10) → the sim restore is exact.
- two inference calls on the identical restored obs: identical → pi05 inference is deterministic.
- policy rollouts (3 chunks, fixed noise) from one restore: diverged (ctrl 0.3–1.5, warmstart 0.3–0.9, qpos ~0.005).
  Re-seeding numpy/random/torch at each restore did not help. Observations after one identical step differed
  (joint_acc up to 1.7, joint_vel 0.04) with identical sim states.
- **Cause:** robosuite `Observable`s keep sampling timers, delay and last observed value outside the sim; a
  restore left each branch with the previous branch's observable phase (and `Observable.reset()` leaves
  `_sampled` set).
- **Fix** (`adapters.py`, backup `.bak_20261005`): `set_state` resets every observable (+ `_sampled = False`)
  before the forced observation refresh. After the fix every comparison is exactly 0 (all four conditions).
  Phase 1 is unaffected (no restores); the pilot's p2 results (2026-10-05, job 11079871) are invalid and must be
  rerun. The brief's "restore is deterministic" predates this and evidently missed it.
- `p2_branching.py` (backup `.bak_20261005`): `--max-start-frac` (default 0.6 of the episode cap) so no branch
  starts with too few steps left; 5 states/cell then spans t = 0…576 on lamp.

## 2026-10-05: Phase 2 launched on the lamp lock-in set (approved, ~15–18 GPU-h)

Invalid pilot outputs moved to `~/scratch/diag/p2_invalid_pilot_nondeterministic/` (their wandb runs
p2{a,b,c}_lamp_arm4_004000_bankpair from job 11079871 are superseded by the reruns of the same name).
Jobs (one per checkpoint, 6 h limit): arm4@4k, arm1@4k, arm3@4k, arm2@4k, arm0@4k, arm4@16k =  11082697 11082698 11082699 11082700 11082701 11082702.
Settings: training cells only; own states (4 rollouts/cell; arm4@4k reuses its pilot states, collection has
no restore); branching --k 8 --states-per-cell 5 --repeats 2 --max-start-frac 0.6 --check-determinism;
smoothness --sigmas 0.1,0.3,0.8 --n-perturb 6; transfer start_0 <-> start_1. Criteria as pre-registered above.
- Phase 2 jobs done (17.9 GPU-h, 3/3 uploads each, no tracebacks). **Determinism check fails for arm0@4k (max_state_diff 49.1) and arm2@4k (2.41)**, passes ~1e-10 for arm1@4k, arm3@4k, arm4@4k, arm4@16k. Solvers' results not usable until explained; diag jobs 11087370 11087371.
- Diag (11087370–71, original diag script, its own noise): arm0@4k all exact 0; arm2@4k exact at t=0 but policy
  rollouts from t=192 diverge (ctrl up to 0.88) with env replay, single inference and 1-step obs all exact.
- **Second restore bug (lamp task state):** the lamp base weld is activated by `ScrewLightbulb.step()` on the
  first step after a reset (`_needs_anchor`), writing the pose into `model.eq_data` (model side, not in the
  MuJoCo state). States saved at t=0 predate anchoring (weld off). In each env the first branch to step anchors;
  every later branch from a t=0 snapshot ran with the weld OFF (base free) and `_needs_anchor` cleared. Affects
  all t=0 branches of all checkpoints (1 in 5 states); shows up when the hand reaches the lamp early (solvers).
  Fix (`adapters.py`): `get_state` saves `eq_data` + `needs_anchor`; `set_state` restores them, and for older
  snapshots re-pins the weld at the restored pose if it was active, else re-arms `_needs_anchor`.
- The arm2 t=192 divergence is not explained by the weld. `diag_branch_determinism.py` runs the determinism
  check on exactly the states/noises branching uses (6 chunks) and reports the first macro/obs key that differs:
  jobs 11087688 (arm0@4k), 11087689 (arm2@4k), 11087690 (arm4@4k, control).
- `diag_branch_determinism.py` with the first weld fix (11087688–90): **9/10 states diverge for arm0, arm2 AND
  arm4 (control)**. At macro 0 (the restored obs itself) joint_acc differs ~1e-16 and, for t > 0 states, the
  camera images differ by up to 255.
- `diag_restore_render.py` (11088472, no policy loaded): restored obs are exactly identical (restore twice;
  restore → 16 steps → restore) **without** the weld handling; **with** the re-pin (`_anchor_base` + forward on
  every restore) joint_acc differs ~1e-15. No image differences either way. The policy does not see joint_acc
  (state = joint_pos + hand qpos), so images are what matters; they differ only when policy inference ran between
  restores → suspected stale offscreen render readback under GPU load.
- Revised weld fix: never re-pin when the snapshot's weld is on (anchor once per env if it never was); re-arm
  `_needs_anchor` for pre-anchor (t = 0) snapshots; `eq_data` / `needs_anchor` restored when the snapshot has
  them. `STEER_DOUBLE_RENDER=1` (diagnostic) discards the first render after a restore. Job 11088498 runs the
  branch check on arm4@4k with and without it, and records a second render of each restored state.
- Render test (11088498, arm4@4k, policy loaded): **rendering the same restored state twice in a row (no step)
  gives images differing by up to 255 for states at t ≥ 256** (identical at t = 0, 128); discarding the first
  render does not help (2/10 vs 4/10 states deterministic). Without the policy loaded (11088472) renders were
  identical. → with pi05 on the same GPU, EGL rendering is not a pure function of the state. Exact CRN branching is
  not achievable here without deeper work (may also matter for DSRL itself, which renders the same way).
  Decision: treat branches as a stochastic env. Branching stats stay valid Monte Carlo estimates (independent
  repeats; same procedure for every checkpoint), only noisier. The weld bug biased only t = 0 states.

## 2026-10-06: provisional Phase 2 answer (existing runs, t = 0 states excluded)

Failures F = arm4@4k, arm3@4k, arm1@4k (8 states each); solvers S = arm2@4k, arm0@4k (8 each). F − S, cluster
bootstrap (checkpoints, then states):
- mean_p (success under random noise from the checkpoint's own mid-episode states): F 0.22–0.25 vs S 0.38–0.39;
  F − S = −0.15 [−0.29, −0.02]. frac_any: F 0.38–0.45 vs S 0.59–0.62; −0.19 [−0.38, +0.02] (P(F<S) 0.97).
  All three F below both S on both, as pre-registered for "existence/discovery".
- gap (best_split − mean_p) ≈ 0 for all checkpoints: uninformative at k = 8 with stochastic branches.
- Learnability: basin_radius = 0.8 (largest σ) for every state with p* > 0 → grid saturated, uninformative;
  retention 0.1/0.3/0.8 and interp_flips: no F/S difference (only 5–6 usable states per group). No evidence for
  the learnability hypothesis, but the test has very little power.
- arm4@16k: frac_any 0.05, mean_p 0.02; the only state with any good noise is on s1_start_1, matching its DSRL
  outcome (two seeds hold start_1 only).
Caveats: own states (F's mid-episode states may simply be worse states: policy and noise effects are mixed);
only 2 solvers; stochastic rendering; 8 states per checkpoint.
