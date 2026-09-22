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

**Which cluster to use:** an arm is only comparable to the arms it is contrasted against.
`arm1_full_ft` (lamp, seed 42, lr ×1) was trained on Alliance H100s, so running its
ablation on Mila L40S puts a GPU/driver/toolchain difference inside the contrast. Prefer
whichever cluster already holds the arm you are comparing to.

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
**except the newest**. Resume only ever reads `last`, and pruned rungs keep their full
`pretrained_model/`, so the ladder stays usable for eval and for dual-noise selection —
which is what the ladder exists for. Budget `rungs × 8.7 + 14.1` GiB.

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
`PreTrainedPolicy.wrap_with_peft` clears `requires_grad` on every parameter before calling
`get_peft_model`, so `full_training_modules` is the only way to keep SigLIP, the projector
and the action projections training — and the only way to get them into the checkpoint,
since `save_checkpoint` calls `PeftModel.save_pretrained`, which writes adapters plus
`modules_to_save` and nothing else. Re-enabling `requires_grad` by hand trains those
weights and then silently drops them at save time.

**3. Per-group LRs follow the schedule proportionally.**
`CosineDecayWithWarmupSchedulerConfig` builds a `LambdaLR` that multiplies each group's own
`initial_lr` by one shared lambda, so `peak_lr`/`decay_lr` enter only as their ratio. They
are set from the arm's largest group, and every group decays to 10% of its own base LR.

**4. Dead weights** (verified by forward+backward, identical in every arm — not a bug and
not a between-arm confound). `paligemma.lm_head`, `gemma_expert.lm_head`, and layer 17 (the
last of 18) of the VLM language model's `o_proj` / `gate_proj` / `up_proj` / `down_proj`
plus its post-attention layernorm never receive a gradient; layer 17's `q_proj` receives
exactly zero. pi05's suffix attends to the prefix's per-layer K/V, so the final prefix
layer's own output is never read, and pi05 never generates tokens. In arm3/arm4 this appears
as 8 LoRA parameters (4 modules × A,B at layer 17) with `grad=None`. Separately, every
`lora_A` has exactly zero gradient on the first backward because `lora_B` is
zero-initialised — standard LoRA, not a failure.

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
    --placement_bank ~/scratch/lerobot/placement_banks/screwlightbulb_xarm6_seed1/bank.json \
    --episodes_per_cell 20 --output_dir <out>

# Re-score a checkpoint under both lamp eval protocols, split per sub-env.
python scripts/eval_lamp_protocol_ab.py --protocol legacy --n_episodes 50 \
    --checkpoint name=<run>/checkpoints/020000/pretrained_model --output_dir <out>
```

`eval_pi05_dual_noise.py` and `eval_sft_on_banks.py` both need the run's own
`train_config.json`, so point them at the run directory, not at a copied checkpoint.

## DSRL

Design: [SFT_ARMS_DSRL.md](SFT_ARMS_DSRL.md).

### 1. Build the tight lamp placement bank (once)

```bash
BANK=~/scratch/lerobot/placement_banks/screwlightbulb_xarm6_seed1_tight
python scripts/build_lamp_placement_bank.py \
    --scene_seeds 1 --max_offset_m 0.06 \
    --n_train 10 --n_heldout 2 \
    --train_min_sep_m 0.015 --heldout_min_sep_m 0.025 \
    --n_candidates 400 \
    --out $BANK/bank.json --render_dir $BANK/renders
```

Yield inside the box is low, hence 400 candidates. The builder warns `!! bank is short` if it
cannot fill the quota. Check the renders before use.

### 2. Select the checkpoint

```bash
python examples/training/eval_pi05_dual_noise.py --run-dir <run> --task lamp \
    --steps 8000 12000 16000 20000 --n-episodes 50
python scripts/select_sft_checkpoint.py --run-dir <run>
```

Writes `<run>/selected_checkpoint.json`.

### 3. Launch DSRL — 3 seeds per frozen checkpoint

```bash
CKPT=<run>/checkpoints/<selected_step>/pretrained_model
for S in 0 1 2; do
  sbatch --job-name=dsrl_arm0_lamp_s$S dsrl_mila.sbatch lamp $CKPT $S
done
```

`dsrl_mila.sbatch <lamp|coffee> <policy_path> <seed> [extra args]` holds the per-task
invariants (robot, fps, controller, bank, eval sets, `--collect_scene_seeds`) so they cannot
drift between seeds or arms; anything after the seed is forwarded to the training script.
Override the GPU with `sbatch --gres=gpu:rtx8000:1 dsrl_mila.sbatch ...`.

### 4. Supporting jobs

```bash
sbatch dsrl_profile.sbatch             # macro-step time split + VRAM/RSS vs --collect_envs
sbatch lamp_offset_calibration.sbatch  # frozen-SFT success vs lamp offset, sets --max_offset_m
```
