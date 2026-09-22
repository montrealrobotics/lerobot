# Runbook — pi05 SFT arms

How to actually run the SFT half of the study. What the study *is*, and the DSRL half:
[SFT_ARMS_DSRL.md](SFT_ARMS_DSRL.md).

## Quickstart

```bash
# 1. Assert the arm's trainable set without burning a queue slot (~10 min, needs a GPU).
python examples/training/train_pi05_casa_experiments.py --experiment arm0_expert_only --task lamp --dry-run

# 2. Exercise the whole job path — dataloader, augmentation, optimizer step, checkpoint
#    write, env rollout, wandb log — in a few minutes.
sbatch --job-name=smoke sft_arms_mila.sbatch arm0_expert_only lamp --smoke-test

# 3. The real thing.
sbatch --job-name=sft_arm0_lamp   sft_arms_mila.sbatch arm0_expert_only lamp
sbatch --job-name=sft_arm1_coffee sft_arms.sbatch      arm1_full_ft     coffee --lr-scale 2.0 --seed 7
```

Anything after `<arm> <task>` is forwarded verbatim to the training script:
`--lr-scale`, `--seed`, `--steps`, `--eval-freq`, `--save-freq`, `--state-dropout-p`,
`--color-jitter-params` (with no values = augmentation off), `--dry-run`, `--smoke-test`.

## The two launchers

| | `sft_arms.sbatch` | `sft_arms_mila.sbatch` |
|---|---|---|
| cluster | Alliance | Mila |
| GPU | `h100:1`, 12 CPU, 48000M, 12 h | `l40s:1`, 8 CPU, 48G, 16 h |
| repo | `/project/6068036/akur/lerobot` | `/home/mila/a/artur.kuramshin/lerobot` |
| outputs | `~/scratch/lerobot/outputs/train` | `/network/scratch/a/artur.kuramshin/lerobot/outputs/train` |
| env | `module load StdEnv/2023 python/3.12.4`, `HF_HOME=/scratch/akur/hf_cache` | `source ~/.bashrc`, `uv pip install numpy==2.2.5` |
| disk check | scratch **quota** via `diskusage_report` | filesystem free space |

Two files because `#SBATCH` directives cannot be conditional and the clusters disagree on
every one of them. **Only cluster plumbing may differ between them** — both end in a single
`srun python examples/training/train_pi05_casa_experiments.py --experiment "$ARM" --task
"$TASK" "$@"`, so the fairness invariants live in the training script and cannot drift
between launchers.

**Which cluster to use:** prefer whichever cluster already holds the arm you are comparing
against — `arm1_full_ft` (lamp, seed 42, lr ×1) is on Alliance H100s.

## What the launchers do before training

**Preflight.** Each check below has a failure mode that otherwise surfaces minutes to hours
in, after the queue wait:

| check | why it fails late otherwise |
|---|---|
| `numpy == 2.2.5` | RoboCasa hard-asserts this exact version at import. The lockfile pins 2.2.6, and a bare `uv run` re-syncs and reverts it — use `uv run --no-sync`. |
| HF token present | `google/paligemma-3b-pt-224` is a **gated** repo. Without a token the run dies when processors are built, i.e. after pi05 has already loaded. |
| wandb credentials | `WANDB_API_KEY` or `~/.netrc`. |
| scratch space | see the disk budget below. |
| CUDA available | — |
| robocasa env registered | `ScrewLightbulb` only exists on the robocasa `lightbulb_task` branch. Missing, it surfaces at the first eval rather than at startup. robocasa registers its envs with robosuite **at import**, so the check imports it first. |

The HF one has a trap worth spelling out: `huggingface_hub` reads `$HF_HOME/token`, so a
token sitting in `~/.cache/huggingface` is **not** found once `HF_HOME` is set. Export
`HF_TOKEN` instead.

**Rendering.** `MUJOCO_GL=egl` / `PYOPENGL_PLATFORM=egl` for headless GPU rendering of the
eval rollouts; without them env creation falls back to software or fails on a compute node.

**Caches.** Inductor, numba and triton caches go to `$SLURM_TMPDIR`, per job. A shared cache
on the networked filesystem is slow, and concurrent jobs corrupt it — a stale numba cache
shows up as `EOFError: Ran out of input` at env creation.

## Disk budget and checkpoint pruning

MEASURED: one full-FT checkpoint is **23 GiB** — 8.7 GiB of weights plus 14.1 GiB of AdamW
state. The default ladder is `steps / save_freq` = 20 rungs, so a single `arm1_full_ft` run
would write ~460 GiB against a 1 TiB scratch quota. `arm0_expert_only` trains 693M
parameters rather than 4.14B, so its optimizer state is ~5.5 GiB and a rung is ~14 GiB.

The launchers run a background pruner that deletes `training_state/` from every checkpoint
except the newest; pruned rungs keep their full `pretrained_model/`. Budget
`rungs × 8.7 + 14.1` GiB.

- `SFT_PRUNE_OPTIMIZER_STATE=0` keeps everything.
- `SFT_STEPS` / `SFT_SAVE_FREQ` override the preflight projection only, not the run.
- The pruner matches the run directory on `*_${TASK}_${ARM}_*`. **Task must be in the
  pattern**: coffee and lamp jobs share an arm name, so matching on the arm alone lets one
  job's pruner latch onto the other job's directory and leave its own run unpruned.

## Implementation notes

Gotchas in the surrounding code, not in the study design.

**1. Parameter tree.** The real tree is

```
model.paligemma_with_expert.paligemma.model.{language_model, vision_tower, multi_modal_projector}...
model.paligemma_with_expert.gemma_expert.model.layers.<i>...
model.{action_in_proj, action_out_proj, time_mlp_in, time_mlp_out}
```

Note the extra `.model.` after `paligemma`. PEFT matches `target_modules` with
`re.fullmatch` and `modules_to_save` with `key.endswith(...)`, and raises only when **no**
entry matches — so one wrong path is silent and leaves SigLIP frozen, which is Ferchau's
worst configuration. `--dry-run` is the guard; run it before queueing an arm.

**2. `modules_to_save` is the only route for the full-FT islands.**
`wrap_with_peft` clears `requires_grad` on every parameter, and `save_checkpoint` writes only
adapters plus `modules_to_save`. So `full_training_modules` is the only way to keep SigLIP, the
projector and the action projections training *and* saved — re-enabling `requires_grad` by hand
trains them and then silently drops them at save time.

**3. Per-group LRs follow the schedule proportionally.**
`CosineDecayWithWarmupSchedulerConfig` builds a `LambdaLR` that multiplies each group's own
`initial_lr` by one shared lambda, so `peak_lr`/`decay_lr` enter only as their ratio. They
are set from the arm's largest group, and every group decays to 10% of its own base LR.

**4. Dead weights are expected**, identically in every arm: both `lm_head`s and layer 17 (the
last) of the VLM language model's `o_proj` / `gate_proj` / `up_proj` / `down_proj` plus its
post-attention layernorm never receive a gradient. In arm3/arm4 this shows as 8 LoRA parameters
with `grad=None`. Every `lora_A` also has zero gradient on the first backward, since `lora_B` is
zero-initialised. Not a bug and not a between-arm confound.

**5. draccus cannot decode `Literal`.** `PreTrainedConfig.from_pretrained` fails on configs
that explicitly saved such a field. Any script that loads a checkpoint registers a decoder
first:

```python
draccus.decode.register(typing.Literal, lambda raw_value, path=(): raw_value)
```

## After training

```bash
# Dual-noise checkpoint selection over the ladder (the selection metric — see SFT_ARMS_DSRL.md).
python examples/training/eval_pi05_dual_noise.py --run-dir <run> --task lamp \
    --steps 8000 12000 16000 20000 --n-episodes 20

# SFT-only success on the DSRL bank cells (the baseline DSRL numbers are compared against).
python scripts/eval_sft_on_banks.py --sft_run <run> --steps 20000 \
    --placement_bank ~/scratch/lerobot/placement_banks/screwlightbulb_xarm6_seed1_pair/bank.json \
    --episodes_per_cell 20 --output_dir <out>

# Re-score a checkpoint under both lamp eval protocols, split per sub-env.
python scripts/eval_lamp_protocol_ab.py --protocol legacy --n_episodes 50 \
    --checkpoint name=<run>/checkpoints/020000/pretrained_model --output_dir <out>
```

`eval_pi05_dual_noise.py` and `eval_sft_on_banks.py` both need the run's own
`train_config.json`, so point them at the run directory, not at a copied checkpoint.

## DSRL

Design: [SFT_ARMS_DSRL.md](SFT_ARMS_DSRL.md).

### 1. Lamp two-start bank

```bash
sbatch lamp_pair_bank.sbatch   # builds the bank + its 50-episode frozen-SFT baseline
```

Or directly, from the calibration bank:

```bash
BANK=~/scratch/lerobot/placement_banks/screwlightbulb_xarm6_seed1_pair
python scripts/build_lamp_pair_bank.py \
    --source_bank ~/scratch/lerobot/placement_banks/screwlightbulb_xarm6_seed1_calib/bank.json \
    --pair s1_train_00,s1_train_06 \
    --out $BANK/bank.json --render_dir $BANK/renders
```

Writes `train` = the two starts, `heldout` = their interpolated midpoint; refuses to write if
any of the three fails a feasibility check (`--allow_failed` overrides). Check the renders.

`scripts/build_lamp_placement_bank.py` builds a general pool instead. Success does not vary with
offset over 0–12 cm (measured 2026-09-22), so its `--max_offset_m` needs a reason.

### 2. Dual-noise scores over the ladder

```bash
python examples/training/eval_pi05_dual_noise.py --run-dir <run> --task lamp \
    --steps 8000 12000 16000 20000 --n-episodes 50
python scripts/select_sft_checkpoint.py --run-dir <run>
```

Writes `<run>/selected_checkpoint.json`. DSRL runs a ladder rather than one selected rung, so
this is the prediction to compare against, not the gate.

### 3. Launch DSRL — 4 checkpoints x 3 seeds

```bash
RUN=<sft_run>
for STEP in 2000 4000 8000 16000; do            # coffee: 4000 8000 14000 18000
  for S in 0 1 2; do
    sbatch --job-name=dsrl_arm0_lamp_${STEP}_s$S \
        dsrl_mila.sbatch lamp $RUN/checkpoints/$(printf %06d $STEP)/pretrained_model $S
  done
done
```

`dsrl_mila.sbatch <lamp|coffee> <policy_path> <seed> [extra args]` (Mila) and `dsrl_drac.sbatch`
(Alliance) hold the per-task invariants — robot, fps, controller, bank, eval sets, prompt,
`--collect_scene_seeds`; anything after the seed is forwarded. Override the GPU with
`sbatch --gres=gpu:rtx8000:1 ...` on Mila, `--gpus=...` on Alliance.

The two launchers differ only in cluster plumbing. Alliance additionally probes for a working
headless GL backend (`scripts/check_gl_backend.py`) because MIG slices cannot do EGL, and its
banks must be rsynced from Mila first.

Preemption is handled: on SIGTERM/SIGUSR1 the trainer writes `<output_dir>/resume_state.pt`
(actor, critics, targets, log_alpha, all three optimizers, the replay buffer, counters and RNG)
and exits 99, and the launcher requeues the job. Both launchers always pass `--resume`, so the
requeued job continues where it stopped; the state is deleted on normal completion. It is
~2.9 GiB at the 31k transitions a 500k-step run reaches, a few seconds to write.

`--resume_save_freq N` also dumps every N env steps, for crashes that send no signal. Off by
default. A 500k run is ~10 h training + ~3.6 h eval.

Smoke-test the path first (~15 min):

```bash
sbatch --job-name=dsrl_smoke --time=1:00:00 dsrl_mila.sbatch lamp $CKPT 0 \
    --total_steps 4000 --min_buffer_size 20 --eval_freq 2000 --eval_episodes 1 --save_freq 2000
```

### 4. Supporting jobs

```bash
sbatch dsrl_profile.sbatch             # macro-step time split + VRAM/RSS vs --collect_envs
sbatch lamp_offset_calibration.sbatch  # frozen-SFT success vs lamp offset
```

`eval_sft_on_banks.py --steps` needs an exact checkpoint dir: rungs are multiples of
`save_freq` (2000), which the `eval_freq` (5000) in-loop eval steps are not.
