# pi05 SFT arms → DSRL steering

Working notes for the study. Scope: how a pretrained VLA should be adapted to a new
embodiment when the endpoint is steering RL, not the SFT checkpoint itself.

How to run any of this — launchers, cluster setup, preflight, disk budget, codebase
gotchas — is in [SFT_ARMS_RUNBOOK.md](SFT_ARMS_RUNBOOK.md).

## The question

Stage 1 adapts `lerobot/pi05_base` to a dexterous embodiment (LEAP hand on a Franka or
UFactory arm) from a few dozen demonstrations. Stage 2 freezes that checkpoint and trains
DSRL — a small SAC policy that steers the frozen policy's flow-matching **noise input**
rather than its actions — on top of it.

The claim under test: **SFT recipes that look equivalent at the SFT endpoint can differ
downstream.** A checkpoint's success rate under its own i.i.d. noise prior says little
about how well its noise space can be steered, how stable RL on top of it is, or how it
generalizes to start conditions it never saw. So the arms are compared at the DSRL
endpoint, not only at the SFT one.

Two RoboCasa tasks, deliberately different in controller and control rate:

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
distribution DSRL steers. See the checkpoint-selection note below.

Launchers: `sft_arms.sbatch` (Alliance) and `sft_arms_mila.sbatch` (Mila), both
`sbatch <launcher> <arm> <task> [extra args]`.

## SFT arms

Every arm uses the stock pi05 architecture (no category-specific action projections) and
differs only in which parameters train and at what LR. Partitions are asserted by
`--dry-run` before any arm is queued.

| arm | trains | frozen | trainable | peak LR |
|---|---|---|---|---|
| `arm0_expert_only` | action expert, time MLPs, action projections | **all** of PaliGemma (LM + SigLIP + projector) | 693M (16.7%) | 2.5e-5 |
| `arm1_full_ft` | everything | — | 4.14B (100%) | 2.5e-5 |
| `arm2_frozen_llm` | SigLIP, projector, expert, action projections | PaliGemma language model only | 1.11B (26.8%) | 2.5e-5 |
| `arm3_lora_r32` | LoRA r=32 on VLM-LM + expert; SigLIP, projector, action projections full-FT | base weights of LM + expert | 468M (10.2%) | 2.5e-4 LoRA / 2.5e-5 full-FT |
| `arm4_lora_vlm_full_expert` *(stretch)* | LoRA r=32 on VLM-LM; expert, SigLIP, projector, action projections full-FT | base weights of the LM | 1.15B (21.7%) | 2.5e-4 LoRA / 2.5e-5 full-FT |

arm1 → arm2 → arm0 is a nested ladder: *freeze nothing* → *freeze the LM* → *freeze the LM
and the vision path*. arm2 − arm0 is therefore exactly SigLIP + the projector.

Describe the arms by composition, not by name. arm2's 1.11B is the second largest
trainable set here, so it is "all the adaptation, none of it in the language model", not
"less adaptation than LoRA". And 414.8M of arm3's 468M is the vision tower, so arm3 is
"LoRA'd LM + expert with vision full-FT exactly as in arm1", not "LoRA vs full FT".

**Invariants** (never vary per arm, or the comparison is void): batch size 32; AdamW with
betas (0.9, 0.95), eps 1e-8, weight decay 0.01, grad clip 1.0; 20k steps, 1k warmup, cosine
to peak/10; chunk_size 32 / n_action_steps 16; colour jitter (brightness 0.3, contrast 0.4,
saturation 0.5, hue 0.08, always applied); state dropout p=0.2; save/eval every 2k.

**Knobs**: `--experiment`, `--task`, `--lr-scale {0.5,1,2}`, `--seed`.

## DSRL run design

### What the RL agent sees

A **compact** observation, not the VLA's. One third-person camera
(`observation.images.robot0_agentview_left`) resized to 64×64, plus the proprioceptive
state vector, through a small conv encoder (image latent 64, state latent 64). This matches
the reference DSRL implementation, which never gives SAC the wrist view. Downsampling
happens before the replay buffer, so the buffer stores 64×64.

The frozen pi05 separately receives its full observation — `robot0_agentview_left` +
`robot0_eye_in_hand`, normalized and tokenized by the checkpoint's own processor pipeline.
Supplying exactly one external camera also makes the checkpoint's random-external-camera
augmentation a deterministic no-op at inference.

### What the RL agent emits

A **noise vector**, not an action. Action space is `Box(-1, 1, (noise_chunk_size × 32,))`,
tanh-squashed; with the default `noise_chunk_size=1` that is 32 numbers per macro step.
It is reshaped to `(1, 32)`, padded to pi05's full 32-step flow-matching chunk by repeating
the last step, and handed to `predict_action_chunk(batch, noise=...)`. pi05 denoises the
whole chunk; the first `n_action_steps=16` actions are executed.

Note the asymmetry worth stating in the paper: the actor is bounded to [-1,1] by the tanh,
while the prior the policy was trained to denoise is N(0,1). `gaussian_warmup` seeds the
replay buffer from the true prior so the first transitions are in-distribution.

Because chunk timing is set in steps, not seconds, one macro step is 0.80 s on coffee
(16 / 20 Hz) and 0.53 s on lamp (16 / 30 Hz). A noise vector therefore steers a different
span of time per task.

### Reward and SAC

- **Reward** (`--reward_mode goal`): −1 per macro step until success, 0 on success.
- **Discount**: 0.999 per primitive step, compounded over the executed chunk → ≈0.984 per
  macro step at `n_action_steps=16`.
- **SAC**: 10 Q-heads reduced by `mean` (not `min`), UTD 20, batch 256, buffer 100k
  transitions, warmup 1,000 macro transitions, no entropy in the TD backup, automatic
  target entropy −noise_dim/2, lr 3e-4 for actor/critic/temperature. The large ensemble +
  mean reduction + no backup entropy is what keeps SAC stable at UTD 20; `min` over a small
  ensemble badly underestimates Q at this UTD.
- **Budget**: 500k primitive env steps.

Units gotcha: `min_buffer_size` counts **macro** transitions (one per chunk) while
`total_steps` counts **primitive** env steps, so updates start after
`min_buffer_size × n_action_steps` primitive steps.

### Train / test sets

Start conditions come from feasibility-checked **banks**, so "held out" means a start the
DSRL agent never trained on, in the same kitchen.

**ScrewLightbulb — placement bank**
`~/scratch/lerobot/placement_banks/screwlightbulb_xarm6_seed1/bank.json`
One kitchen, construction seed 1 = layout 4 / style 8. Within it: 10 training lamp
placements (`s1_train_00..09`), 2 held-out (`s1_heldout_00/01`), plus `s1_reference`, the
scene's own native placement, which every pre-bank lightbulb run trained on.
`PlacementBankWrapper` re-applies a placement on every reset, so each cell is one fixed
start condition varying only by RoboCasa's 0.02 rad arm reset noise.

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

### SFT-side eval, and why it mirrors this

Lamp in-loop SFT eval is pinned to the same regime: both sub-envs are kitchen seed 1 and
differ only in lamp placement (`s1_reference`, `s1_train_04`). So **lamp in-loop success is
a seed-1 number, not a cross-kitchen generalization number** — score other kitchens
post-hoc with `scripts/eval_sft_on_banks.py`. Coffee keeps the default, sub-env *i* =
construction seed *i*.

### Checkpoint selection

By **dual-noise** eval (`eval_pi05_dual_noise.py`), not by in-loop success. Each rung is
scored twice: under i.i.d. noise (a fresh N(0,1) per chunk step, what SFT eval measures) and
under duplicated noise (one vector repeated across the chunk, what DSRL actually steers).
These rank checkpoints differently, and the duplicated-noise number is the one that
predicts DSRL.

## Status

- SFT arm matrix: arm0 added and dry-run verified (693,422,112 trainable; 201/4/4 tensors).
  arm1 lamp and coffee runs exist. arms 2–4 not yet run under the current invariants.
- **Lamp SFT is at the floor.** Measured 2026-09-20 over 50 episodes per protocol
  (`scripts/eval_lamp_protocol_ab.py`): the historical expert-only checkpoint scores 6.0%
  and the arm1 full-FT checkpoint 0/100 across every cell. This is a blocking issue — DSRL
  on top of a 0–6% policy has almost nothing to steer, so no arm-vs-arm claim on lamp is
  meaningful until lamp SFT is materially better. Coffee is not affected.
- Earlier DSRL results (pre-arm-matrix, on older checkpoints) are in the project notes, not
  here. The load-bearing one: lamp DSRL randomized over 2 kitchens × 20 placements never
  learned at a 500k budget, which is why the lamp bank here is narrowed to a single kitchen.

## Open — decide before running the DSRL half

1. **Lamp DSRL training pool**: all 10 seed-1 training placements, or a subset? The 20-pose
   version did not learn.
2. **Seeds per arm.** At the success rates lamp currently shows, one RL seed per arm cannot
   distinguish arms. Coffee may be fine with one.
3. **LR sweep transfer.** The plan selects `--lr-scale` on coffee and freezes it for lamp,
   but the tasks differ 4× in epochs and 1.5× in control rate. Sweeping lamp separately is
   the safer claim.
4. **No validation split.** Nothing in the pipeline measures held-out *imitation* loss, so
   "overfitting" is currently inferred from train loss across runs.
