# pi05 SFT arms → DSRL steering

Working notes. How to run any of it: [SFT_ARMS_RUNBOOK.md](SFT_ARMS_RUNBOOK.md).

## The question

Stage 1 adapts `lerobot/pi05_base` to a dexterous embodiment (LEAP hand on a Franka or UFactory
arm) from a few dozen demonstrations. Stage 2 freezes that checkpoint and trains DSRL — a small
SAC policy steering the frozen policy's flow-matching **noise input** rather than its actions.

Claim under test: SFT recipes that look equivalent at the SFT endpoint can differ downstream, so
the arms are compared at the DSRL endpoint, not only at the SFT one.

| task | robot | action representation | fps | demos | frames |
|---|---|---|---|---|---|
| `CoffeePressButton` | `PandaDexLeapRHOmron` | OSC_POSE deltas (6 EE + 16 hand) | 20 | 56 | 7,928 |
| `ScrewLightbulb` | `XArm6DexLeapRHOmron` | absolute joint targets (6 arm + 16 hand) | 30 | 33 | 31,842 |

`ScrewLightbulb` only exists on the `lightbulb_task` branch of the robocasa checkout.

## Pipeline

| stage | script | output |
|---|---|---|
| 1. SFT | `examples/training/train_pi05_casa_experiments.py` | one run per (arm, task, lr-scale, seed); 20k steps, checkpoint every 2k |
| 2. Checkpoint selection | `examples/training/eval_pi05_dual_noise.py` | i.i.d. vs duplicated-noise success per rung |
| 3. DSRL | `examples/tutorial/rl/dsrl_pi05_robocasa_example.py` | noise actor trained over the frozen checkpoint |
| 4. Held-out scoring | `scripts/eval_sft_on_banks.py` | SFT-only baseline on the DSRL eval cells |

Stage 2 exists because the in-loop SFT eval samples i.i.d. noise, which is *not* the
distribution DSRL steers; see Checkpoint selection.

Launchers: `sft_arms.sbatch` (Alliance) and `sft_arms_mila.sbatch` (Mila), both
`sbatch <launcher> <arm> <task> [extra args]`.

## SFT arms

Stock pi05 architecture in every arm; they differ only in which parameters train and at what LR.
`--dry-run` asserts the partition before an arm is queued.

| arm | trains | frozen | trainable | peak LR |
|---|---|---|---|---|
| `arm0_expert_only` | action expert, time MLPs, action projections | **all** of PaliGemma (LM + SigLIP + projector) | 693M (16.7%) | 2.5e-5 |
| `arm1_full_ft` | everything | — | 4.14B (100%) | 2.5e-5 |
| `arm2_frozen_llm` | SigLIP, projector, expert, action projections | PaliGemma language model only | 1.11B (26.8%) | 2.5e-5 |
| `arm3_lora_r32` | LoRA r=32 on VLM-LM + expert; SigLIP, projector, action projections full-FT | base weights of LM + expert | 468M (10.2%) | 2.5e-4 LoRA / 2.5e-5 full-FT |
| `arm4_lora_vlm_full_expert` *(stretch)* | LoRA r=32 on VLM-LM; expert, SigLIP, projector, action projections full-FT | base weights of the LM | 1.15B (21.7%) | 2.5e-4 LoRA / 2.5e-5 full-FT |

arm1 → arm2 → arm0 is a nested ladder, so arm2 − arm0 is exactly SigLIP + the projector.

**Invariants** (never vary per arm, or the comparison is void): batch size 32; AdamW with
betas (0.9, 0.95), eps 1e-8, weight decay 0.01, grad clip 1.0; 20k steps, 1k warmup, cosine
to peak/10; chunk_size 32 / n_action_steps 16; colour jitter (brightness 0.3, contrast 0.4,
saturation 0.5, hue 0.08, always applied); state dropout p=0.2; save/eval every 2k.

**Knobs**: `--experiment`, `--task`, `--lr-scale {0.5,1,2}`, `--seed`.

## DSRL run design

### Observations

Noise actor: one third-person camera (`robot0_agentview_left`) at 64×64 plus the state vector,
through a compact conv encoder (image latent 64, state latent 64). Downsampling happens before
the replay buffer.

Frozen pi05: `robot0_agentview_left` + `robot0_eye_in_hand` through the checkpoint's own
processor pipeline. Supplying one external camera makes the checkpoint's random-external-camera
augmentation a no-op at inference.

### Action space

`Box(-1, 1, (noise_chunk_size × 32,))`, tanh-squashed. `noise_chunk_size=1` gives 32 numbers per
macro step, reshaped to `(1, 32)`, padded to pi05's 32-step chunk by repeating the last step, and
passed to `predict_action_chunk(batch, noise=...)`. The first `n_action_steps=16` actions run.

The actor is bounded to [-1,1] while the prior pi05 denoises is N(0,1); `gaussian_warmup` seeds
the buffer from the true prior.

### Reward and SAC

- Reward (`--reward_mode goal`): −1 per macro step until success, 0 on success.
- Discount 0.999 per primitive step, compounded over the chunk → ≈0.984 per macro step.
- SAC: 10 Q-heads reduced by `mean`, UTD 20, batch 256, buffer 100k, warmup 1,000 macro
  transitions, no backup entropy, target entropy −noise_dim/2, lr 3e-4.
- Budget: 500k primitive env steps.

`min_buffer_size` counts **macro** transitions and `total_steps` counts **primitive** steps, so
updates start after `min_buffer_size × n_action_steps` primitive steps.

### Train / test sets

**ScrewLightbulb — two-start bank**
`~/scratch/lerobot/placement_banks/screwlightbulb_xarm6_seed1_pair/bank.json`
One kitchen (construction seed 1 = layout 4 / style 8), one collection env, two fixed training
starts 7.4 cm apart (yaw −61.3° / −118.5°); `PlacementBankWrapper` alternates between them on
reset. Held out is their interpolated midpoint (yaw −89.9°). All three pass the bank's
feasibility checks. Built by `scripts/build_lamp_pair_bank.py`.

Frozen-policy baseline — arm0 step 16000, 50 episodes per cell, 2026-09-22:

| cell | success |
|---|---|
| `s1_start_0` | 10% |
| `s1_start_1` | 6% |
| `s1_mid` (held out) | 14% |

Two starts rather than a pool: over 17 placements 0–11.9 cm from the reference, success is 3.8%
overall and uncorrelated with offset (Pearson r = −0.21; ≤6 cm 6.7% vs >6 cm 3.2%, Fisher
p = 0.26), so most placements in a pool supply no reward. Historically one fixed start reached
`success_rate_ma10` 0.8–1.0 while a 20-placement pool reached 0.0–0.3.

**CoffeePressButton — kitchen bank**
`~/scratch/lerobot/placement_banks/coffeepressbutton_pandadex_kitchens/bank.json`
Placement randomization is meaningless here (the robot is always centred on the machine),
so the randomized axis is the kitchen. Train: seeds 0, 1, 4, 5, 6, 7, 8, 9, 10, 12 — i.e.
(8,6) (4,8) (7,3) (6,8) (4,5) (9,5) (7,2) (4,2) (7,9) (6,1), with the three coffee-machine
models balanced. Held out: seed 11 = (1,4) and seed 14 = (1,6). Seed 2 = (8,4) is excluded.

**Start-pose rejection, coffee only** (`--reject_start_rot_deg 10 --reject_start_pos_cm 3`).
RoboCasa's arm reset noise can jam the LEAP fingers into a wall cabinet and rotate the hand
by up to ~68°, in both training and held-out kitchens. Rejection re-resets until the end
effector starts within tolerance of the noise-free pose, so held-out numbers reflect
novelty rather than jammed starts. Not applied to lamp: the lamp kitchens' starts are
already consistent across the runs being compared.

### What gets reported

`--eval_placement_sets train,heldout` scores 3 cells — the two training starts and the midpoint —
aggregated as `success_rate_train` / `_heldout`, so the interpolation gap is logged directly.
`episode/success_rate_ma10` from the collection rollouts is a free on-policy training signal.

Cost: ~43 s per episode, so 3 cells × 5 episodes ≈ 11 min per pass; at `--eval_freq 25000`, ~3.6 h
over a 500k-step run against ~10 h of training.

### SFT-side eval

Lamp in-loop SFT eval is pinned to kitchen seed 1, sub-envs differing only in lamp placement
(`s1_reference`, `s1_train_04`) — a seed-1 number, not a cross-kitchen one. Score other kitchens
post-hoc with `scripts/eval_sft_on_banks.py`. Coffee keeps sub-env *i* = construction seed *i*.

`s1_train_04` is from the old wide bank (11.8 cm, 0% measured); repoint the in-loop eval at a
pair-bank placement.

### Checkpoint selection

Score every rung with `eval_pi05_dual_noise.py`: i.i.d. noise (fresh N(0,1) per chunk step,
what SFT eval measures) and duplicated noise (one vector repeated across the chunk, what DSRL
steers).

`scripts/select_sft_checkpoint.py` turns the ladder into one choice:

1. Score = duplicated-noise success.
2. Smooth over 3 adjacent rungs. A 20-episode rung has ~±9pp standard error, so argmax over a
   10-rung ladder mostly selects eval noise.
3. Among rungs within one standard error of the best smoothed score, take the **earliest** —
   fewest epochs, least memorization.

## DSRL run matrix

A **checkpoint ladder** per task, 3 RL seeds each. Only `--seed` varies within a rung: SAC init,
exploration noise, buffer order, bank shuffle.

| task | checkpoints |
|---|---|
| lamp | 2k, 4k, 8k, 16k |
| coffee | 4k, 8k, 14k, 18k |

12 runs per (arm, task), 24 per arm across both, at ~14 h each. The ladder measures
steerability against SFT training length directly, so `eval_pi05_dual_noise.py` becomes a
prediction to check against it rather than the gate that picks one rung.

### Differences from the pre-arm-matrix runs

Verified against `dsrl_pi05_robocasa_example.py` and the historical run configs, 2026-09-21.
Everything else matches `DSRLConfig` defaults.

| | historical | this study |
|---|---|---|
| `--collect_envs` | 2 | 1 (lamp) |
| `--utd_ratio` | 40 | 20 |
| `--collect_scene_seeds` | n/a | 1 (lamp) |
| lamp placements | 20 over 2 kitchens | 2 fixed starts in one kitchen, midpoint held out |

Gradient steps per transition is `utd_ratio / collect_envs`: 40/2 = 20 before, 20/1 = 20 now,
so effective UTD is unchanged.

`--collect_scene_seeds` was added for this study; without it a single-env run can only train in
construction seed 0, and the lamp bank holds only seed 1.

## Throughput

Measured 2026-09-22 on an L40S, lamp, UTD 20, `n_action_steps=16`
(`scripts/profile_dsrl_throughput.py`). Seconds per macro step:

| collect_envs | env | frozen | prepare | SAC | total | steps/s | h/500k | VRAM | RSS |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.464 | 0.208 | 0.027 | 0.434 | 1.136 | 14.1 | 9.9 | 8.8G | 18.1G |
| 2 | 0.882 | 0.231 | 0.037 | 0.425 | 1.578 | 20.3 | 6.8 | 9.0G | 21.3G |
| 4 | 1.976 | 0.310 | 0.060 | 0.443 | 2.791 | 22.9 | 6.1 | 9.2G | 21.6G |
| 8 | 3.963 | 0.474 | 0.096 | 0.439 | 4.974 | 25.7 | 5.4 | 9.6G | 24.1G |

Env stepping dominates and scales linearly (~0.49 s per sub-env): `SyncVectorEnv` steps sub-envs
sequentially. SAC is flat at ~0.43 s — one `update()` per iteration regardless of sub-env count.

Holding gradient steps per transition at 20 (`utd_ratio = 20 × collect_envs`) makes throughput
flat at ~16 steps/s for every row, so **parallel collection buys nothing at constant effective
UTD**. The levers are `AsyncVectorEnv` (not implemented), cheaper rendering, or a lower UTD.

Not VRAM-bound (9 GiB of 46). RSS above is with a 2k-transition buffer; a full 100k buffer of
64×64 float32 observations adds roughly 10 GiB. Obs preparation is 2.4% of a macro step, so the
redundant per-primitive-step `_prepare` in `DSRLEnvWrapper.step` is not worth fixing.

## Limitations

- **No LR sweep.** Every arm uses its default peak LR (2.5e-5 full-FT / 2.5e-4 LoRA), so arms
  are compared at one point in LR space. An arm could lose because its default LR suits it
  worse rather than because its parameter partition is worse.
- Lamp DSRL is a single-kitchen, ≤6 cm result. `scripts/eval_sft_on_banks.py` is the
  cross-kitchen instrument.
- No held-out imitation loss; SFT overfitting is inferred from train loss across runs.
- chunk_size 32 / n_action_steps 16 spans 1.60 s on coffee and 1.07 s on lamp.

## Status

- arm0 and arm1 lamp runs exist; arms 2–4 not yet run under the current invariants.
- arm1 lamp was at the floor (0/100 across every cell) against 6.0% for the historical
  expert-only checkpoint. arm0 recovers it — 20% / 0% / 20% in-loop at eval steps 5k / 10k / 15k
  — which points at the trainable set, not the LR or the augmentation.
- `eval_freq=5000` with `save_freq=2000` means no saved checkpoint carries an in-loop number.
  In-loop uses 4 cameras with the random-external swap live, the DSRL view uses 2, so in-loop
  numbers and bank numbers are not directly comparable.

## Open

1. Align `save_freq` and `eval_freq` so selected checkpoints carry in-loop numbers.
2. Coffee may need fewer than 3 RL seeds; unmeasured.
