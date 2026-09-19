#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
SFT experiment arms for: "What is the best way to adapt a pre-trained VLA to a
new embodiment (LEAP hand) in the low-data regime, when the endpoint is a
steering-RL (DSRL) pipeline?"

Unchanged pi05 architecture in every arm (no category-specific encoder/decoder).
Arms differ ONLY in which parameters train and at what learning rate.

ARMS AND HYPOTHESES
  arm1_full_ft            Baseline. Everything trains at the openpi default LR.
  arm2_frozen_llm         H1: does the pretrained LANGUAGE/semantic representation
                          need to move for a new embodiment? PaliGemma's language
                          model is frozen; SigLIP, the multimodal projector and the
                          whole action expert full-FT. This deliberately separates
                          "keep the LM fixed" from "keep the whole VLM fixed": the
                          latter also freezes SigLIP, which Ferchau Finding 3 makes
                          the single most damaging choice (ATP 0.14 frozen vs 0.74
                          full-FT), so a train_expert_only arm would confound the
                          two and would be predicted to lose for the wrong reason.
                          Requires PI05Config.freeze_llm, added for this study.
  arm3_lora_r32           H2: does constraining updates preserve pretrained
                          structure that pays off in steerability / RL stability
                          / generalization? Ferchau-matched config (LoRA r=32
                          uniform on VLM-LM + expert, SigLIP + projector
                          full-FT) which matches full-FT at the SFT endpoint --
                          so any downstream difference vs arm1 is pure hidden-
                          property signal.
  arm4_lora_vlm_full_expert  (stretch) Intermediate: LoRA'd VLM-LM, full-FT
                          expert. Run only if arms 1-3 are on track.

LEARNING-RATE POLICY (sources: openpi defaults; LWR = "LoRA Without Regret",
Schulman et al. / Thinking Machines blog 2025; Ferchau et al. arXiv:2607.10172)
  * Full-FT params: 2.5e-5 peak (openpi CosineDecaySchedule default).
  * LoRA params: 10x the full-FT LR => 2.5e-4 (LWR's central prescription;
    Ferchau et al. did NOT do this -- they reused one LR everywhere -- which
    their own limitations section partially concedes via the alpha/r issue).
  * r=32, alpha=32: at this rank LWR's fixed-alpha and Ferchau's alpha=r
    conventions coincide (alpha/r = 1), so rsLoRA scaling is not needed.
  * LoRA placement: ALL linear layers (attention + MLP) of the targeted
    components, per LWR ("apply to all layers, especially MLPs") and Ferchau
    (uniform allocation sufficient; asymmetric no better).
  * SigLIP + multimodal projector: FULL FT in every arm where they train at
    all, never LoRA (Ferchau Finding 3: SigLIP LoRA => ATP 0.43, frozen =>
    0.14, full-FT => 0.74).
  * Per-arm mini-sweep: run --lr-scale in {0.5, 1.0, 2.0} on ONE task (coffee),
    select by DUAL-NOISE SFT eval (iid + DSRL-style), freeze for the other
    task. State exactly this in the paper.

SCHEDULE / DURATION
  steps == decay_steps == 20_000, warmup 1_000, cosine to peak/10.
  Rationale: the checkpoint ladder needs a genuine "late" rung (~16-20k), and
  every arm must see the same LR-trajectory *shape*. NOTE: this differs from
  the old file (10k steps against a 30k decay horizon, ending mid-cosine). If
  arm1 reuses pre-existing full-FT checkpoints instead of retraining, declare
  the schedule mismatch in the paper or retrain arm1 under this schedule.

FAIRNESS INVARIANTS (do not change per-arm)
  batch_size=32 (LWR notes LoRA degrades at large batch; 32 is safe),
  optimizer family + betas/eps/weight decay, schedule shape, data, augmentation,
  chunk_size, save/eval cadence. The ONLY per-arm degrees of freedom are the
  trainable-parameter set and the LR policy above.

VERIFICATION NOTES (answers to the plumbing questions this file used to TODO)
  1. Module paths were read off `policies/pi05/modeling_pi05.py`. The real
     parameter tree is
         model.paligemma_with_expert.paligemma.model.{language_model,
             vision_tower, multi_modal_projector}...
         model.paligemma_with_expert.gemma_expert.model.layers.<i>...
         model.{action_in_proj, action_out_proj, time_mlp_in, time_mlp_out}
     i.e. there is an extra `.model.` between `paligemma` and
     `language_model`/`vision_tower`. PEFT matches `target_modules` with
     `re.fullmatch` on module keys and `modules_to_save` with
     `key.endswith(...)`, and `_set_trainable` only raises when *none* of the
     `modules_to_save` entries match -- so a single wrong path is silent and
     leaves SigLIP frozen (exactly Ferchau's worst config). `--dry-run`
     asserts against that.
  2. `PreTrainedPolicy.wrap_with_peft` sets `requires_grad=False` on every
     parameter before calling `get_peft_model`, so `full_training_modules`
     (PEFT `modules_to_save`) is the *only* way to keep SigLIP + projector +
     action projections training. It is also the only way to get them written
     into the checkpoint, because `save_checkpoint` calls
     `PeftModel.save_pretrained`, which stores adapters + `modules_to_save`
     and nothing else. Manually re-enabling `requires_grad` after wrapping
     would train those weights and then silently drop them at save time.
  3. Per-group LRs do follow the schedule proportionally:
     `CosineDecayWithWarmupSchedulerConfig` builds a `LambdaLR`, which
     multiplies each group's own `initial_lr` by the shared lambda. Its
     `peak_lr`/`decay_lr` only enter as the ratio `alpha = decay_lr/peak_lr`,
     so they are set from the arm's *largest* group and every group decays to
     10% of its own base LR.
  4. Optimizer hyperparameters are now taken from one place (`ADAM_*`,
     `WEIGHT_DECAY`) and pushed into both the policy preset (arms 1-2) and
     `NamedAdamWConfig` (arms 3-4), so the two paths cannot drift. Value is
     the lerobot pi05 preset's `optimizer_weight_decay=0.01`, matching the
     existing baseline runs, not openpi's ~0.
  5. Dual-noise eval is NOT run in-loop: `lerobot_train` calls
     `eval_policy_all` once per `eval_freq` and pi05's `select_action` has no
     noise seam (only `predict_action_chunk(batch, noise=...)` does). Doing it
     in-loop means patching `lerobot_train.py` and doubling an already
     dominant eval cost. Run it post-hoc over the checkpoint ladder with
     `examples/training/eval_pi05_dual_noise.py`, which is also cheaper
     because you only score the rungs you care about.
  6. Lamp constants filled in from the previously-run setup (commit 72dc9b12):
     dataset `akuramshin/robocasa_lightbulbscrew_dex_filtered`, robot
     `XArm6DexLeapRHOmron`, task string `ScrewLightbulb` -- confirmed against
     `class ScrewLightbulb(Kitchen)` in robocasa
     `environments/kitchen/single_stage/kitchen_lamp.py` on branch
     `lightbulb_task` (note the lowercase "b"; "ScrewLightBulb" does not exist).
  7. EMA: lerobot's training loop has no EMA of any kind (grep finds no
     `ema_decay`/`EMAModel`). openpi's 0.99 default is simply not in play, so
     all arms are non-EMA and there is nothing to match.

MEASURED, on lerobot/pi05_base (4,143,404,816 params), H100, synthetic batch:

                        trainable          resident   optimizer groups (peak LR)
  arm1_full_ft      4.14B (100.0%)            4.14B   one @ 2.5e-5
  arm2_frozen_llm   1.11B ( 26.8%)            4.14B   one @ 2.5e-5
  arm3_lora_r32      468M ( 10.2%)            4.61B   vlm_lora 39.2M @ 2.5e-4
                                                      expert_lora 14.0M @ 2.5e-4
                                                      vision 414.8M @ 2.5e-5
                                                      action_proj 0.07M @ 2.5e-5
  arm4              1.15B ( 21.7%)            5.29B   vlm_lora 39.2M @ 2.5e-4
                                                      expert 693M, vision 414.8M,
                                                      action_proj @ 2.5e-5

  Resident > 4.14B because PEFT's `modules_to_save` deep-copies each full-FT
  island (a frozen `original_module` plus the trainable copy): +0.47B for arm3,
  +1.15B for arm4. Budget GPU memory accordingly.

  NOTE on arm2's composition: "frozen LLM" sounds restrictive but is the second
  LARGEST trainable set here -- 1.11B (vision 414.8M + expert 693M + time MLPs +
  action projections), vs arm3's 468M. The frozen part is 3.04B of language model.
  So arm2 is not "less adaptation than LoRA"; it is "all the adaptation, none of
  it in the language model". Frame it that way.

  NOTE on arm3's composition: 414.8M of its 468M trainable parameters are the
  vision tower, i.e. ~89%. Arm3 is not "constrained updates" in aggregate -- it
  is "LoRA'd language model + expert, with vision full-FT exactly as in arm1".
  That is Ferchau's design and the intended contrast, but say it that way in the
  paper rather than "LoRA vs full fine-tuning".

DATA, measured from dataset metadata:

              fps   episodes   frames   state   action   epochs @ 20k x bs32
  coffee       20         56    7,928    (23,)    (22,)   ~81
  lamp         30         33   31,842    (22,)    (22,)   ~20

  * THE TWO TASKS WERE COLLECTED AT DIFFERENT CONTROL RATES (20 vs 30 Hz).
    Hardcoding `fps=30` for both -- which is what the old single-task script
    does, because it was last edited for lamp -- would run coffee eval at 1.5x
    its collection rate. `fps` is therefore read from dataset metadata.
  * CONSEQUENCE OF THE RATE DIFFERENCE (confirmed intentional by the recording
    setup, not a data bug): with chunk_size=32 / n_action_steps=16 fixed across
    tasks, a chunk spans 1.60s on coffee but 1.07s on lamp, and an executed
    segment 0.80s vs 0.53s. Within a task every arm sees the same horizon, so
    the arm comparison is unaffected -- but the same hyperparameters mean
    different things across tasks, and a DSRL noise vector steers a different
    span of time per task. Either say so, or equalise the horizon in seconds
    (e.g. chunk_size 32 @ 20Hz vs 48 @ 30Hz) if cross-task transfer of the
    selected LR is meant to be a real claim.
  * The state dims independently confirm the robot assignment: 23 = Panda(7) +
    LEAP(16) for coffee, 22 = XArm6(6) + LEAP(16) for lamp, matching
    `RoboCasaEnv._robot_dims`. This is why features are read from metadata
    rather than hardcoded to the padded 32.
  * WATCH THE EPOCH COUNT. 20k steps at batch 32 is ~81 epochs over coffee's
    7.9k frames. A 4.14B model at 81 epochs on 56 demonstrations will memorise;
    expect the arms to differ substantially in HOW FAST they overfit, which is
    arguably the interesting measurement but makes the checkpoint ladder and the
    LR sweep load-bearing rather than cosmetic. Also note lamp is 33 episodes,
    not the "~50 per task" in the project description.

EXACT TRAINABLE PARTITIONS (asserted by --dry-run, tensor counts):
  arm2_frozen_llm  vision_tower 437, multi_modal_projector 2, gemma_expert 201,
                   time_mlp 4, action_proj 4  (= 648, nothing else)
  arm3_lora_r32    lora_ 508, vision_tower 437, multi_modal_projector 2,
                   action_proj 4
  arm4             lora_ 252, gemma_expert 201, time_mlp 4, vision_tower 437,
                   multi_modal_projector 2, action_proj 4
  arm2 additionally asserts that nothing under `paligemma.model.language_model`
  or `paligemma.lm_head` has requires_grad, and a backward pass confirms the
  language model receives no gradient at all while SigLIP, the expert and the
  action projections all do.

DEAD WEIGHTS (verified by forward+backward, identical in every arm -- NOT a bug
and NOT a between-arm confound):
  `paligemma.lm_head`, `gemma_expert.lm_head`, and layer 17 (the last of 18) of
  the VLM language model's `o_proj`, `gate_proj`, `up_proj`, `down_proj` +
  post-attention layernorm never receive a gradient; layer 17's `q_proj` receives
  exactly zero. pi05's suffix attends to the prefix's per-layer K/V, so the final
  prefix layer's own output is never read, and pi05 never generates tokens. In
  arm3/arm4 this shows up as 8 LoRA parameters (4 modules x A,B at layer 17) with
  `grad=None` -- expected, and the same modules are dead under full-FT in arm1.
  Separately, EVERY `lora_A` has exactly zero gradient on the first backward
  because `lora_B` is zero-initialised; that is standard LoRA, not a failure.
"""

import argparse
import dataclasses
import datetime as dt
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import robosuite
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.configs.default import DatasetConfig, EvalConfig, PeftConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.configs import RoboCasaEnv
from lerobot.optim import (
    CosineDecayWithWarmupSchedulerConfig,
    NamedAdamWConfig,
    NamedParamGroupConfig,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts.lerobot_train import train
from lerobot.utils.import_utils import register_third_party_plugins

PI05_BASE_MODEL = "lerobot/pi05_base"


def _default_output_root() -> Path:
    """Scratch root for run outputs.

    Overridable with $LEROBOT_OUTPUT_ROOT. The Mila path is kept as the default
    when it exists; on any other cluster (e.g. Alliance, where it does not) we
    fall back to $HOME/scratch.
    """
    env_root = os.environ.get("LEROBOT_OUTPUT_ROOT")
    if env_root:
        return Path(env_root)
    mila_root = Path("/network/scratch/a/artur.kuramshin/lerobot/outputs/train")
    if mila_root.parent.parent.parent.exists():
        return mila_root
    return Path.home() / "scratch" / "lerobot" / "outputs" / "train"


SCRATCH_OUTPUT_ROOT = _default_output_root()

# ----------------------------------------------------------------------------
# Global training constants (fairness invariants -- identical across arms)
# ----------------------------------------------------------------------------
TOTAL_STEPS = 20_000
WARMUP_STEPS = 1_000
DECAY_STEPS = TOTAL_STEPS          # cosine completes; same trajectory shape everywhere
SAVE_FREQ = 1_000                  # dense checkpoint ladder for the selection study
EVAL_FREQ = 500
BATCH_SIZE = 32

FULL_FT_PEAK_LR = 2.5e-5           # openpi CosineDecaySchedule default
LORA_LR_MULT = 10.0                # LWR: LoRA optimal LR ~10x full-FT LR
LORA_PEAK_LR = FULL_FT_PEAK_LR * LORA_LR_MULT   # 2.5e-4
DECAY_FRACTION = 0.1               # decay to peak/10, mirroring openpi 2.5e-5 -> 2.5e-6

LORA_RANK = 32                     # Ferchau: saturation at r=32
LORA_ALPHA = 32                    # alpha == r => alpha/r = 1. MUST be explicit: PEFT's
                                   # LoraConfig defaults lora_alpha=8, which at r=32 would
                                   # silently scale every adapter by 0.25.

# Single source of truth for the optimizer, used by BOTH the policy preset (arms
# 1-2) and NamedAdamWConfig (arms 3-4) so the two code paths cannot drift.
# These are the lerobot pi05 preset defaults (PI05Config.optimizer_*).
ADAM_BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.01                # preset value; openpi's ~0 (1e-10) is NOT used
GRAD_CLIP_NORM = 1.0

# ----------------------------------------------------------------------------
# LoRA target patterns (PEFT matches `target_modules` with `re.fullmatch` on the
# module key, relative to the policy root).
#
#   VLM_LM_LINEAR_TARGETS : PaliGemma *language model* linears only. Anchored on
#                           `paligemma.model.language_model.` so it can never
#                           reach vision_tower (whose SigLIP attention uses
#                           q/k/v/out_proj) or multi_modal_projector.
#   EXPERT_LINEAR_TARGETS : gemma action expert linears + the time MLPs.
# ----------------------------------------------------------------------------
_LINEAR = r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
VLM_LM_LINEAR_TARGETS = rf".*paligemma\.model\.language_model\..*\.{_LINEAR}"
EXPERT_LINEAR_TARGETS = rf".*\.gemma_expert\..*\.{_LINEAR}|model\.(time_mlp_in|time_mlp_out)"

# Modules that FULL-FT whenever their arm trains them at all (never LoRA).
# These become PEFT `modules_to_save`, matched by `key.endswith(...)`.
# SigLIP + projector per Ferchau Finding 3; action projections are the
# pretrained pi05 pad-to-32 projections (unchanged architecture -- the old
# category-specific path is deliberately gone).
VISION_FULL_FT_MODULES = [
    "paligemma.model.vision_tower",
    "paligemma.model.multi_modal_projector",
]
ACTION_PROJ_MODULES = ["model.action_in_proj", "model.action_out_proj"]
TIME_MLP_MODULES = ["model.time_mlp_in", "model.time_mlp_out"]
EXPERT_MODULES = ["paligemma_with_expert.gemma_expert"]

# ----------------------------------------------------------------------------
# Tasks.
# Single dex dataset per task: the old dex+gripper mixture is a co-training
# axis that is OUT OF SCOPE for this matrix (it would confound the arm
# comparison). Keep it for a separate ablation if ever needed.
#
# NOTE ON `controller`: eval control mode MUST match the trained action
# representation. Both datasets are relabelled to absolute joint targets, so
# eval runs the robot's *_joint_pos composite controller. Leaving
# RoboCasaEnv.controller at None silently falls back to OSC_POSE and every
# rollout is garbage while training loss looks fine.
#
# NOTE ON `robocasa_task`: "ScrewLightbulb" (lowercase "b") matches
# `class ScrewLightbulb(Kitchen)` in robocasa's kitchen_lamp.py. That env only
# exists on the `lightbulb_task` branch of the robocasa checkout, so the lamp
# arms require it; `add_leap` does not define it.
# ----------------------------------------------------------------------------
WRIST_IMAGE_KEY = "observation.images.robot0_eye_in_hand"
EXTERNAL_IMAGE_KEY = "observation.images.robot0_agentview_left"
RANDOM_EXTERNAL_IMAGE_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
]

_ROBOSUITE_CONTROLLER_DIR = Path(robosuite.__file__).parent / "controllers/config/robots"
_JOINT_POS_CONTROLLERS = {
    "PandaDexLeapRHOmron": "default_pandadexleaprhomron_joint_pos.json",
    "XArm6DexLeapRHOmron": "default_xarm6dexleaprhomron_joint_pos.json",
}


def joint_pos_controller(robot: str) -> str:
    path = _ROBOSUITE_CONTROLLER_DIR / _JOINT_POS_CONTROLLERS[robot]
    if not path.exists():
        raise FileNotFoundError(f"Joint-position controller config not found for {robot}: {path}")
    return str(path)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset: str
    robot: str
    robocasa_task: str
    camera_name: str = (
        "robot0_agentview_left,robot0_agentview_right,robot0_eye_in_hand,robot0_agentview_center"
    )


TASKS: dict[str, TaskSpec] = {
    "coffee": TaskSpec(
        name="coffee",
        dataset="akuramshin/robocasa_coffeepressbutton_dex_augstyle",
        robot="PandaDexLeapRHOmron",
        robocasa_task="CoffeePressButton",
    ),
    "lamp": TaskSpec(
        name="lamp",
        dataset="akuramshin/robocasa_lightbulbscrew_dex_filtered",
        robot="XArm6DexLeapRHOmron",
        robocasa_task="ScrewLightbulb",
    ),
}

# ----------------------------------------------------------------------------
# Experiment arms
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    freeze_vision_encoder: bool
    train_expert_only: bool
    peft: PeftConfig | None
    freeze_llm: bool = False
    lr_groups: dict[str, NamedParamGroupConfig] = field(default_factory=dict)
    peak_lr: float = FULL_FT_PEAK_LR
    policy_optimizer_lr: float = FULL_FT_PEAK_LR
    # Substrings of parameter names that MUST / MUST NOT have requires_grad, asserted by --dry-run.
    expect_trainable: tuple[str, ...] = ()
    expect_frozen: tuple[str, ...] = ()
    # When set, --dry-run asserts the trainable set is EXACTLY the union of these
    # patterns: every trainable parameter must match one, so anything unexpected that
    # becomes trainable is a hard failure rather than something you notice in wandb.
    expect_trainable_only: tuple[str, ...] = ()
    notes: str = ""

    @property
    def decay_lr(self) -> float:
        return self.peak_lr * DECAY_FRACTION


EXPERIMENTS: dict[str, ExperimentSpec] = {
    # ------------------------------------------------------------------ arm 1
    "arm1_full_ft": ExperimentSpec(
        name="arm1_full_ft",
        freeze_vision_encoder=False,
        train_expert_only=False,
        peft=None,
        peak_lr=FULL_FT_PEAK_LR,
        policy_optimizer_lr=FULL_FT_PEAK_LR,
        expect_trainable=(
            "paligemma.model.vision_tower",
            "paligemma.model.multi_modal_projector",
            "paligemma.model.language_model",
            "gemma_expert",
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
        ),
        notes=(
            "Everything trainable at openpi-default LR (2.5e-5 peak). "
            "Uses the policy training preset. If reusing pre-existing full-FT "
            "checkpoints, declare the schedule mismatch (see header). "
            "LR sweep center; sweep via --lr-scale {0.5,1,2}."
        ),
    ),
    # ------------------------------------------------------------------ arm 2
    "arm2_frozen_llm": ExperimentSpec(
        name="arm2_frozen_llm",
        freeze_vision_encoder=False,   # SigLIP + projector full-FT
        train_expert_only=False,
        freeze_llm=True,               # PaliGemma's language model only
        peft=None,                     # no LoRA anywhere
        peak_lr=2.5e-5,
        policy_optimizer_lr=2.5e-5,
        expect_trainable=(
            "paligemma.model.vision_tower",
            "paligemma.model.multi_modal_projector",
            "gemma_expert",
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
        ),
        expect_frozen=("paligemma.model.language_model", "paligemma.lm_head"),
        # The exact partition requested for this arm: nothing outside these five
        # components may train.
        expect_trainable_only=(
            r"vision_tower",
            r"multi_modal_projector",
            r"gemma_expert",
            r"time_mlp_(in|out)",
            r"action_(in|out)_proj",
        ),
        notes=(
            "Freezes ONLY PaliGemma's language model; the vision tower, the "
            "multimodal projector and the whole action expert full-FT at the "
            "openpi default LR. Isolates 'keep the pretrained language/semantic "
            "representation fixed' from 'keep the whole VLM fixed' -- the latter "
            "also freezes SigLIP, which Ferchau Finding 3 says is the single "
            "most damaging choice (ATP 0.14 frozen vs 0.74 full-FT), so the old "
            "train_expert_only arm confounded the two. Needs PI05Config.freeze_llm, "
            "added for this study; --dry-run asserts the exact trainable set."
        ),
    ),
    # ------------------------------------------------------------------ arm 3
    "arm3_lora_r32": ExperimentSpec(
        name="arm3_lora_r32",
        freeze_vision_encoder=False,   # SigLIP trains (full-FT, not LoRA)
        train_expert_only=False,
        peft=PeftConfig(
            method_type="LORA",
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=rf"({VLM_LM_LINEAR_TARGETS})|({EXPERT_LINEAR_TARGETS})",
            # Everything here trains FULLY, outside LoRA. It is also the only
            # thing besides the adapters that PeftModel.save_pretrained writes.
            full_training_modules=VISION_FULL_FT_MODULES + ACTION_PROJ_MODULES,
        ),
        lr_groups={
            # All LoRA params at 10x full-FT LR (LWR). Uniform rank + uniform
            # LR across VLM-LM and expert (Ferchau: asymmetric no better).
            # Patterns are re.search'd against post-PEFT parameter names, which
            # carry a `base_model.model.` prefix and a `.lora_A.default.weight`
            # suffix.
            "vlm_lora": NamedParamGroupConfig(
                patterns=[r"paligemma\.model\.language_model.*lora_"],
                lr=LORA_PEAK_LR,
            ),
            "expert_lora": NamedParamGroupConfig(
                patterns=[r"gemma_expert.*lora_", r"time_mlp_(in|out).*lora_"],
                lr=LORA_PEAK_LR,
            ),
            # Full-FT islands at full-FT LR:
            "vision_full_ft": NamedParamGroupConfig(
                patterns=[r"vision_tower", r"multi_modal_projector"],
                lr=FULL_FT_PEAK_LR,
            ),
            "action_proj": NamedParamGroupConfig(
                patterns=[r"action_(in|out)_proj"],
                lr=FULL_FT_PEAK_LR,
            ),
        },
        peak_lr=LORA_PEAK_LR,                 # scheduler reference peak (largest group)
        policy_optimizer_lr=FULL_FT_PEAK_LR,  # NamedAdamW default for any unmatched group
        expect_trainable=(
            "paligemma.model.vision_tower",
            "paligemma.model.multi_modal_projector",
            "action_in_proj",
            "action_out_proj",
            "paligemma.model.language_model.layers.0.self_attn.q_proj.lora_A",
            "gemma_expert.model.layers.0.mlp.down_proj.lora_A",
        ),
        expect_trainable_only=(
            r"lora_",
            r"vision_tower",
            r"multi_modal_projector",
            r"action_(in|out)_proj",
        ),
        notes=(
            "H2 arm, Ferchau-matched: LoRA r=32/alpha=32 uniform on VLM-LM + "
            "expert (all linears incl. MLPs), SigLIP + projector full-FT, "
            "action projections full-FT. Known to MATCH full-FT at the SFT "
            "endpoint => downstream differences vs arm1 are hidden-property "
            "signal. LoRA LR = 2.5e-4 (10x rule; the LWR correction Ferchau "
            "did not apply). Sweep LoRA groups {1.25e-4, 2.5e-4, 5e-4} via "
            "--lr-scale; full-FT islands scale with it, acceptable."
        ),
    ),
    # ---------------------------------------------------------- arm 4 (stretch)
    "arm4_lora_vlm_full_expert": ExperimentSpec(
        name="arm4_lora_vlm_full_expert",
        freeze_vision_encoder=False,
        train_expert_only=False,
        peft=PeftConfig(
            method_type="LORA",
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=VLM_LM_LINEAR_TARGETS,
            # The expert trains fully, outside LoRA, plus the usual full-FT
            # islands. The time MLPs live on PI05FlowMatching, NOT under
            # gemma_expert, so they need their own entries or they stay frozen.
            # Listing gemma_expert here also excludes it from LoRA targeting
            # (PEFT refuses to adapt anything under modules_to_save), which is
            # what this arm wants.
            full_training_modules=(
                EXPERT_MODULES + TIME_MLP_MODULES + VISION_FULL_FT_MODULES + ACTION_PROJ_MODULES
            ),
        ),
        lr_groups={
            "vlm_lora": NamedParamGroupConfig(
                patterns=[r"paligemma\.model\.language_model.*lora_"],
                lr=LORA_PEAK_LR,
            ),
            "expert_full_ft": NamedParamGroupConfig(
                patterns=[r"gemma_expert", r"time_mlp_(in|out)"],
                lr=FULL_FT_PEAK_LR,
            ),
            "vision_full_ft": NamedParamGroupConfig(
                patterns=[r"vision_tower", r"multi_modal_projector"],
                lr=FULL_FT_PEAK_LR,
            ),
            "action_proj": NamedParamGroupConfig(
                patterns=[r"action_(in|out)_proj"],
                lr=FULL_FT_PEAK_LR,
            ),
        },
        peak_lr=LORA_PEAK_LR,
        policy_optimizer_lr=FULL_FT_PEAK_LR,
        expect_trainable=(
            "paligemma.model.vision_tower",
            "paligemma.model.multi_modal_projector",
            "gemma_expert.model.layers.0.mlp.down_proj",
            "time_mlp_in",
            "action_in_proj",
            "paligemma.model.language_model.layers.0.self_attn.q_proj.lora_A",
        ),
        expect_trainable_only=(
            r"lora_",
            r"gemma_expert",
            r"time_mlp_(in|out)",
            r"vision_tower",
            r"multi_modal_projector",
            r"action_(in|out)_proj",
        ),
        notes=(
            "STRETCH arm -- run only if arms 1-3 are on track. Intermediate "
            "between arm1 and arm3: constrained VLM-LM, unconstrained expert. "
            "NOTE: modules_to_save deep-copies the whole expert subtree, so "
            "this arm costs ~300M extra resident parameters vs arm3."
        ),
    ),
}


def make_timestamped_output_dir(root: Path, cfg: TrainPipelineConfig) -> Path:
    now = dt.datetime.now()
    job_name = cfg.job_name or cfg.trainable_config.type
    return root / f"{now:%Y-%m-%d}" / f"{now:%H-%M-%S}_{job_name}"


def _scaled(spec_lr: float, lr_scale: float) -> float:
    return spec_lr * lr_scale


def make_visual_feature(metadata: LeRobotDatasetMetadata, key: str) -> PolicyFeature:
    shape = tuple(metadata.features[key]["shape"])
    if len(shape) != 3:
        raise ValueError(f"Expected image feature '{key}' to have 3 dims, got {shape}.")
    if shape[0] == 3:
        chw_shape = shape
    elif shape[-1] == 3:
        chw_shape = (shape[-1], shape[0], shape[1])
    else:
        raise ValueError(f"Expected image feature '{key}' to be RGB, got {shape}.")
    return PolicyFeature(type=FeatureType.VISUAL, shape=chw_shape)


def make_config(
    experiment: str,
    task: str,
    lr_scale: float,
    seed: int,
    eval_freq: int = EVAL_FREQ,
    steps: int = TOTAL_STEPS,
) -> TrainPipelineConfig:
    spec = EXPERIMENTS[experiment]
    task_spec = TASKS[task]
    metadata = LeRobotDatasetMetadata(task_spec.dataset)
    action_dim = metadata.features["action"]["shape"][0]
    state_dim = metadata.features["observation.state"]["shape"][0]

    env = RoboCasaEnv(
        task=task_spec.robocasa_task,
        robot=task_spec.robot,
        # Absolute-joint-target policies must be evaluated with the joint-position
        # controller; the default (None -> OSC_POSE) would be a silent mismatch.
        controller=joint_pos_controller(task_spec.robot),
        # Eval control_freq must equal the dataset's collection rate (30 Hz for the
        # quest_rokoko dex datasets). Read it from metadata so it cannot drift.
        fps=metadata.fps,
        camera_name=task_spec.camera_name,
    )

    # Scale LRs for the per-arm mini-sweep.
    peak_lr = _scaled(spec.peak_lr, lr_scale)
    decay_lr = _scaled(spec.decay_lr, lr_scale)
    policy_lr = _scaled(spec.policy_optimizer_lr, lr_scale)
    lr_groups = {
        name: NamedParamGroupConfig(patterns=g.patterns, lr=_scaled(g.lr, lr_scale))
        for name, g in spec.lr_groups.items()
    }

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            # Plain string, not a 1-element list: a list routes through
            # MultiLeRobotDataset (extra dataset_index plumbing, common-key
            # intersection) for no benefit here.
            repo_id=task_spec.dataset,
            video_backend="pyav",
        ),
        env=env,
        job_name=f"sft_arms_{task_spec.name}_{spec.name}_lrx{lr_scale:g}_seed{seed}",
        resume=False,
        seed=seed,
        policy=PI05Config(
            pretrained_path=Path(PI05_BASE_MODEL),
            push_to_hub=False,
            dtype="bfloat16",
            gradient_checkpointing=True,
            compile_model=True,
            freeze_vision_encoder=spec.freeze_vision_encoder,
            train_expert_only=spec.train_expert_only,
            freeze_llm=spec.freeze_llm,
            chunk_size=32,
            n_action_steps=16,
            max_state_dim=max(32, state_dim),
            max_action_dim=max(32, action_dim),
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
                WRIST_IMAGE_KEY: make_visual_feature(metadata, WRIST_IMAGE_KEY),
                EXTERNAL_IMAGE_KEY: make_visual_feature(metadata, EXTERNAL_IMAGE_KEY),
            },
            random_external_camera_keys=RANDOM_EXTERNAL_IMAGE_KEYS,
            random_external_camera_output_key=EXTERNAL_IMAGE_KEY,
            random_external_camera_p=0.5,
            force_current_processor_config=True,
            # Unchanged pi05 architecture: category-specific path OFF for the
            # whole matrix (that axis is retired; padded 32-dim action space,
            # pretrained action projections finetuned).
            use_category_specific_action_proj=False,
            normalization_mapping={
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.QUANTILES,
                "ACTION": NormalizationMode.QUANTILES,
            },
            # Preset path (arms 1-2). Mirrored into NamedAdamWConfig below so
            # the two optimizer paths share one set of hyperparameters.
            optimizer_lr=policy_lr,
            optimizer_betas=ADAM_BETAS,
            optimizer_eps=ADAM_EPS,
            optimizer_weight_decay=WEIGHT_DECAY,
            optimizer_grad_clip_norm=GRAD_CLIP_NORM,
            scheduler_warmup_steps=WARMUP_STEPS,
            scheduler_decay_steps=DECAY_STEPS,
            scheduler_decay_lr=decay_lr,
        ),
        use_policy_training_preset=spec.peft is None,
        optimizer=(
            None
            if spec.peft is None
            else NamedAdamWConfig(
                lr=policy_lr,
                betas=ADAM_BETAS,
                eps=ADAM_EPS,
                weight_decay=WEIGHT_DECAY,
                grad_clip_norm=GRAD_CLIP_NORM,
                param_groups=lr_groups,
                # Any trainable parameter that no group claims is a plumbing bug,
                # not something to silently train at the default LR.
                fail_on_unmatched=True,
            )
        ),
        scheduler=(
            None
            if spec.peft is None
            # peak_lr/decay_lr only set the floor ratio (decay_lr/peak_lr); each
            # param group decays from its own base LR to 10% of it.
            else CosineDecayWithWarmupSchedulerConfig(
                peak_lr=peak_lr,
                decay_lr=decay_lr,
                num_warmup_steps=WARMUP_STEPS,
                num_decay_steps=DECAY_STEPS,
            )
        ),
        peft=spec.peft,
        wandb=WandBConfig(enable=True, project="lerobot-dex"),
        steps=steps,
        # In-loop eval is iid-noise only; dual-noise selection runs post-hoc over
        # the checkpoint ladder (see eval_pi05_dual_noise.py).
        eval_freq=eval_freq,
        save_freq=SAVE_FREQ,
        log_freq=50,
        tolerance_s=1e-3,
        eval=EvalConfig(n_episodes=20, n_videos=5, batch_size=2),
        batch_size=BATCH_SIZE,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
    )
    cfg.output_dir = make_timestamped_output_dir(SCRATCH_OUTPUT_ROOT, cfg)
    return cfg


# ----------------------------------------------------------------------------
# Dry run: build the policy exactly as lerobot_train does, then assert that the
# trainable set is the one this arm claims. Catches silent PEFT path typos
# (wrong module name => modules_to_save no-ops => SigLIP stays frozen) and
# unmatched optimizer groups, without burning a training job.
#
# This materialises pi05 (~3.3B params), so run it inside an allocation, e.g.
#   salloc --account=<acct> --time=0:30:00 --cpus-per-task=4 --mem=32G
# ----------------------------------------------------------------------------


def run_dry_run(cfg: TrainPipelineConfig, spec: ExperimentSpec) -> None:
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.policies.factory import make_policy
    from lerobot.utils.constants import IMAGENET_STATS

    cfg.validate()
    # Metadata only: make_policy needs `.features` and `.stats`, so there is no reason
    # to pull the dataset's video files just to inspect the trainable set.
    ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, root=cfg.dataset.root)
    if cfg.dataset.use_imagenet_stats:
        for key in ds_meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                ds_meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    # pi05 is CONSTRUCTED in float32 (~13GB for 3.3B params) and only cast to bfloat16
    # at the end of PaliGemmaWithExpertModel.__init__, so plain CPU construction needs
    # ~16GB of host RAM. Building inside a cuda device context moves that transient
    # peak onto the GPU, which lets the dry run fit in a small-RAM allocation. Real
    # training still constructs on CPU, so size those jobs for the full footprint.
    init_device = torch.device(device) if (device := cfg.policy.device) else torch.device("cpu")
    with torch.device(init_device):
        policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta, rename_map=cfg.rename_map)
    if cfg.peft is not None:
        policy = policy.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(cfg.peft))

    named = dict(policy.named_parameters())
    trainable = {n: p for n, p in named.items() if p.requires_grad}
    # PEFT rewrites names of `modules_to_save` entries as
    # `<module>.modules_to_save.<adapter>.<rest>`, which splits otherwise-contiguous
    # paths. Match expectations against the un-wrapped names too.
    unwrapped = {re.sub(r"\.modules_to_save\.[^.]+\.", ".", n) for n in trainable}
    n_total = sum(p.numel() for p in named.values())
    n_train = sum(p.numel() for p in trainable.values())
    print(f"\n[{spec.name}] trainable {n_train:,} / {n_total:,} params ({100 * n_train / n_total:.2f}%)")

    failures: list[str] = []

    # 1. LoRA must never touch the vision path.
    lora_in_vision = [
        n for n in trainable if "lora_" in n and ("vision_tower" in n or "multi_modal_projector" in n)
    ]
    if lora_in_vision:
        failures.append(f"LoRA applied inside the vision path: {lora_in_vision[:5]}")

    # 2. Arm-specific expectations.
    for needle in spec.expect_trainable:
        if not any(needle in n for n in trainable) and not any(needle in n for n in unwrapped):
            failures.append(f"expected trainable params matching '{needle}', found none")
    for needle in spec.expect_frozen:
        leaked = [n for n in trainable if needle in n]
        if leaked:
            failures.append(f"expected '{needle}' frozen, but {len(leaked)} params train, e.g. {leaked[:3]}")

    # 2b. Exact partition: nothing outside the declared components may train.
    if spec.expect_trainable_only:
        allowed = [re.compile(pat) for pat in spec.expect_trainable_only]
        stray = sorted(n for n in trainable if not any(pat.search(n) for pat in allowed))
        if stray:
            failures.append(
                f"{len(stray)} trainable params outside {list(spec.expect_trainable_only)}, "
                f"e.g. {stray[:4]}"
            )
        else:
            counts = {
                pat: sum(1 for n in trainable if re.search(pat, n)) for pat in spec.expect_trainable_only
            }
            print(f"    exact trainable partition OK: {counts}")

    # 3. Every trainable parameter must be claimed by an optimizer group
    #    (NamedAdamWConfig.fail_on_unmatched raises here for the PEFT arms).
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    # `group["lr"]` has already been multiplied by the step-0 warmup factor
    # (1/(warmup+1)), so report the group's base LR, which is the configured peak.
    base_lrs = scheduler.base_lrs if scheduler is not None else [g["lr"] for g in optimizer.param_groups]
    for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
        # The preset path (arms 1-2) hands AdamW every parameter, frozen ones included
        # -- they are inert because they never get a grad -- so count only trainables.
        n_params = sum(p.numel() for p in group["params"] if p.requires_grad)
        print(
            f"    group {group.get('name', '?'):16s} peak_lr={base_lr:.3e} "
            f"wd={group['weight_decay']:g} trainable_params={n_params:,}"
        )

    if failures:
        raise AssertionError("\n  - " + "\n  - ".join(failures))
    print(f"[{spec.name}] dry run OK\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="arm1_full_ft")
    parser.add_argument("--task", choices=tuple(TASKS), default="coffee")
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help="Multiply every LR in the arm by this factor. Mini-sweep {0.5,1,2} "
        "on coffee only; select by dual-noise SFT eval; freeze for lamp.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=EVAL_FREQ,
        help="In-loop (iid-noise) eval cadence. 0 disables in-loop eval; dual-noise "
        "checkpoint selection runs post-hoc either way.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build policy + optimizer and assert this arm's trainable set, then exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    register_third_party_plugins()
    cfg = make_config(
        args.experiment, args.task, args.lr_scale, args.seed, args.eval_freq, args.steps
    )
    if args.dry_run:
        run_dry_run(cfg, EXPERIMENTS[args.experiment])
    else:
        train(cfg)
