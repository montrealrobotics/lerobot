#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
SFT arms for: "How should a pretrained VLA be adapted to a new embodiment (LEAP hand)
in the low-data regime, when the endpoint is steering RL (DSRL)?"

Every arm uses the stock pi05 architecture (no category-specific action projections) and
differs ONLY in which parameters train and at what learning rate.

  Study design, DSRL half, open questions   SFT_ARMS_DSRL.md
  How to run it, cluster setup, gotchas     SFT_ARMS_RUNBOOK.md

  arm0_expert_only  All of PaliGemma frozen -- language model, SigLIP and the projector;
                    only the action expert, the time MLPs and the action projections
                    train. Against arm1 this isolates the trainable set, against arm2 it
                    isolates SigLIP.
  arm1_full_ft      Everything trains, at openpi's default LR.
  arm2_frozen_llm   Only PaliGemma's language model is frozen; SigLIP, the projector and
                    the whole expert full-FT. Tests whether the pretrained language /
                    semantic representation has to move for a new embodiment.
  arm3_lora_r32     LoRA r=32 on VLM-LM + expert, SigLIP + projector full-FT
                    (Ferchau-matched). Known to match full-FT at the SFT endpoint, so a
                    downstream difference vs arm1 is hidden-property signal.
  arm4_lora_vlm_full_expert   (stretch) LoRA'd VLM-LM, full-FT expert. Run only if
                    arms 0-3 are on track.

LEARNING RATES (openpi defaults; LWR = "LoRA Without Regret", Schulman et al. 2025;
Ferchau et al. arXiv:2607.10172)
  * Full-FT parameters 2.5e-5 peak; LoRA parameters 10x that, 2.5e-4 (LWR).
  * r=32 with alpha=32. alpha MUST be explicit: PEFT's LoraConfig defaults to 8, which
    at r=32 would silently scale every adapter by 0.25.
  * LoRA on ALL linears (attention + MLP) of the targeted components.
  * SigLIP + projector full-FT wherever they train at all, never LoRA (Ferchau Finding
    3: SigLIP frozen => ATP 0.14, LoRA => 0.43, full-FT => 0.74).
  * Per-arm mini-sweep: --lr-scale {0.5, 1, 2}, selected by dual-noise SFT eval.

SCHEDULE  steps == decay_steps == 20_000, warmup 1_000, cosine to peak/10.

FAIRNESS INVARIANTS (never vary per arm): batch_size 32, optimizer family and its
betas / eps / weight decay, schedule shape, data, augmentation, chunk_size, save and
eval cadence. The only per-arm degrees of freedom are the trainable set and the LR
policy above.

AUGMENTATION (invariant -- changing either invalidates every cross-arm comparison)
  * One always-applied torchvision ColorJitter with GR00T N1.5's finetuning defaults:
    brightness 0.3, contrast 0.4, saturation 0.5, hue 0.08.
  * State dropout p=0.2: one Bernoulli draw per sample, and when it fires the WHOLE
    normalized state vector is zeroed (GR00T semantics).
  Both are training-only; eval observations come from the env and never pass through
  them.

TRAINABLE PARTITIONS (--dry-run asserts these exactly, as tensor counts)
  arm0  gemma_expert 201, time_mlp 4, action_proj 4                          = 209
  arm2  the above + vision_tower 437, multi_modal_projector 2                = 648
  arm3  lora_ 508, vision_tower 437, multi_modal_projector 2, action_proj 4
  arm4  lora_ 252, gemma_expert 201, time_mlp 4, vision_tower 437,
        multi_modal_projector 2, action_proj 4

MEASURED on lerobot/pi05_base (4,143,404,816 params), H100, batch 32:

                      trainable        resident   optimizer groups (peak LR)
  arm0_expert_only     693M ( 16.7%)      4.14B   one @ 2.5e-5
  arm1_full_ft        4.14B (100.0%)      4.14B   one @ 2.5e-5
  arm2_frozen_llm     1.11B ( 26.8%)      4.14B   one @ 2.5e-5
  arm3_lora_r32        468M ( 10.2%)      4.61B   vlm_lora 39.2M @ 2.5e-4
                                                  expert_lora 14.0M @ 2.5e-4
                                                  vision 414.8M @ 2.5e-5
                                                  action_proj 0.07M @ 2.5e-5
  arm4                1.15B ( 21.7%)      5.29B   vlm_lora 39.2M @ 2.5e-4
                                                  expert 693M, vision 414.8M,
                                                  action_proj @ 2.5e-5

  Resident exceeds 4.14B on the PEFT arms because `modules_to_save` deep-copies each
  full-FT island (a frozen original plus the trainable copy): +0.47B for arm3, +1.15B
  for arm4. Budget GPU memory accordingly.

  Composition, worth describing precisely rather than by arm name: arm2's 1.11B is the
  second LARGEST trainable set here, so it is "all of the adaptation, none of it in the
  language model", not "less adaptation than LoRA". And 414.8M of arm3's 468M is the
  vision tower (~89%), so arm3 is "LoRA'd language model + expert with vision full-FT
  exactly as in arm1", not "LoRA vs full fine-tuning".

DATA (read from dataset metadata)

            fps   episodes   frames   state   action   epochs @ 20k x bs32
  coffee     20         56    7,928    (23,)    (22,)   ~81
  lamp       30         33   31,842    (22,)    (22,)   ~20

  * The two tasks were collected at DIFFERENT control rates, so `fps` comes from
    metadata and is never hardcoded. With chunk_size 32 / n_action_steps 16 fixed
    across tasks, a chunk spans 1.60 s on coffee but 1.07 s on lamp, and an executed
    segment 0.80 s vs 0.53 s -- so a DSRL noise vector steers a different span of time
    per task. Within a task every arm sees the same horizon, so the arm comparison is
    unaffected; cross-task transfer of a selected LR is the claim this weakens.
  * The state dims confirm the robot assignment: 23 = Panda(7) + LEAP(16) for coffee,
    22 = XArm6(6) + LEAP(16) for lamp.
  * ~81 epochs over 56 coffee demonstrations at 4.14B parameters will memorise. How
    fast each arm overfits is part of the measurement, which is what makes the
    checkpoint ladder and the LR sweep load-bearing rather than cosmetic.

Before queueing an arm, run --dry-run: PEFT matches `modules_to_save` with
`key.endswith(...)` and raises only when NO entry matches, so one wrong module path is
silent and leaves SigLIP frozen. That and the other codebase gotchas (parameter tree,
per-group LR semantics, dead weights) are in SFT_ARMS_RUNBOOK.md.
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
from lerobot.transforms import ImageTransformConfig, ImageTransformsConfig
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


def _default_lamp_placement_bank() -> str:
    """Seed-1 lamp placement bank, overridable with $LEROBOT_LAMP_PLACEMENT_BANK.

    Built by `scripts/build_lamp_placement_bank.py --scene_seeds 1`; see the LAMP EVAL
    note in the module docstring for why lamp eval is pinned to that one kitchen.
    """
    env_path = os.environ.get("LEROBOT_LAMP_PLACEMENT_BANK")
    if env_path:
        return env_path
    mila_root = Path("/network/scratch/a/artur.kuramshin/lerobot/placement_banks")
    root = mila_root if mila_root.parent.exists() else Path.home() / "scratch" / "lerobot" / "placement_banks"
    return str(root / "screwlightbulb_xarm6_seed1" / "bank.json")


LAMP_PLACEMENT_BANK = _default_lamp_placement_bank()

# The two lamp placements in-loop eval runs, one per eval sub-env, both in kitchen seed 1 (L4S8):
#   s1_reference -- the scene's own placement, i.e. what every fixed-placement lightbulb run so
#                   far has trained and evaluated on, so the numbers stay comparable.
#   s1_train_04  -- 12 cm away along the counter, drawn from the bank's DSRL training pool.
# Both are feasibility-checked (see the bank's `checks`); swap in any id from the bank.
LAMP_EVAL_PLACEMENT_IDS = ("s1_reference", "s1_train_04")

# ----------------------------------------------------------------------------
# Global training constants (fairness invariants -- identical across arms)
# ----------------------------------------------------------------------------
TOTAL_STEPS = 20_000
WARMUP_STEPS = 1_000
DECAY_STEPS = TOTAL_STEPS          # cosine completes; same trajectory shape everywhere
SAVE_FREQ = 2_000                  # dense checkpoint ladder for the selection study
EVAL_FREQ = 2_000
DEFAULT_EVAL_BATCH_SIZE = 2        # eval sub-envs when a task pins no explicit start conditions
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
# Augmentation (a FAIRNESS INVARIANT -- identical in every arm).
#
# GR00T N1.5's finetuning defaults. torchvision reads a float `b` as "sample the
# factor uniformly from [max(0,1-b), 1+b]", so these give brightness [0.7,1.3],
# contrast [0.6,1.4], saturation [0.5,1.5], hue [-0.08,0.08]. This is deliberately
# NOT lerobot's default ImageTransformsConfig, which samples a SUBSET of
# {brightness, contrast, saturation, hue, sharpness, affine} per frame as
# independent transforms; GR00T applies all four jointly, every frame.
COLOR_JITTER = {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}

# State dropout, GR00T N1.5 semantics: with this probability a training sample has its
# WHOLE normalized state replaced by zeros -- one draw per sample, all-or-nothing, not
# per-dimension noise -- so at p=0.2 one sample in five must act on vision alone. Stops
# the policy leaning on proprioception instead of vision. See StateDropoutProcessorStep.
STATE_DROPOUT_P = 0.2


def make_image_transforms(color_jitter: dict[str, float] | None) -> ImageTransformsConfig:
    """One always-applied ColorJitter, matching GR00T rather than lerobot's subset sampler."""
    if not color_jitter:
        return ImageTransformsConfig(enable=False)
    return ImageTransformsConfig(
        enable=True,
        max_num_transforms=1,   # with a single transform in `tfs`, this means "always apply it"
        random_order=False,
        tfs={
            "color_jitter": ImageTransformConfig(
                weight=1.0, type="ColorJitter", kwargs=dict(color_jitter)
            )
        },
    )

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
# Tasks. One dex dataset each -- the dex+gripper co-training mixture is a separate
# axis and is out of scope for this matrix.
#
# `controller`: the eval control mode MUST match the action representation the
# dataset stores (see TaskSpec.controller_filename). A mismatch is silent -- training
# loss looks fine and every rollout fails.
#
# `robocasa_task`: "ScrewLightbulb" (lowercase "b") only exists on the
# `lightbulb_task` branch of the robocasa checkout, which the lamp arms therefore
# require; `add_leap` does not define it.
# ----------------------------------------------------------------------------
WRIST_IMAGE_KEY = "observation.images.robot0_eye_in_hand"
EXTERNAL_IMAGE_KEY = "observation.images.robot0_agentview_left"
RANDOM_EXTERNAL_IMAGE_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
]

_ROBOSUITE_CONTROLLER_DIR = Path(robosuite.__file__).parent / "controllers/config/robots"
# Arm joints per robot, used to check the eval action space against the dataset.
_ARM_JOINTS = {"PandaDexLeapRHOmron": 7, "XArm6DexLeapRHOmron": 6}
LEAP_HAND_DIMS = 16


def resolve_controller(filename: str | None) -> str | None:
    """Absolute path to a robosuite composite-controller config, or None for the robot default."""
    if filename is None:
        return None
    path = _ROBOSUITE_CONTROLLER_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Controller config not found: {path}")
    return str(path)




@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset: str
    robot: str
    robocasa_task: str
    # Composite-controller config for EVAL. A property of how the DATASET was collected,
    # not of the robot, so it is stated per task rather than derived from the robot.
    # None = robosuite's default for the robot (OSC_POSE, delta end-effector).
    controller_filename: str | None = None
    # Arm dims the eval controller consumes: the arm's joint count for an absolute
    # JOINT_POSITION config, 6 (3 pos + 3 rot) for OSC_POSE.
    arm_action_dim: int = 6
    camera_name: str = (
        "robot0_agentview_left,robot0_agentview_right,robot0_eye_in_hand,robot0_agentview_center"
    )
    # In-loop eval start conditions, one per eval sub-env (the eval batch size is taken from
    # their length). None = the historical default: sub-env i is construction seed i, so the
    # sub-envs differ by kitchen AND by whatever object placement that seed happened to draw.
    eval_scene_seeds: tuple[int, ...] | None = None
    eval_placement_bank: str | None = None
    eval_placement_ids: tuple[str, ...] | None = None

    @property
    def eval_batch_size(self) -> int:
        return len(self.eval_scene_seeds) if self.eval_scene_seeds else DEFAULT_EVAL_BATCH_SIZE


TASKS: dict[str, TaskSpec] = {
    "coffee": TaskSpec(
        name="coffee",
        dataset="akuramshin/robocasa_coffeepressbutton_dex_augstyle",
        robot="PandaDexLeapRHOmron",
        robocasa_task="CoffeePressButton",
        # OSC_POSE deltas: the 22-dim action is 6 end-effector dims + 16 hand. A Panda
        # under absolute JOINT_POSITION would need 7 arm dims, so this data is NOT joint
        # targets, unlike lamp. None selects default_pandadexleaprhomron.json (OSC_POSE,
        # input_type=delta).
        controller_filename=None,
        arm_action_dim=6,
    ),
    "lamp": TaskSpec(
        name="lamp",
        dataset="akuramshin/robocasa_lightbulbscrew_dex_filtered",
        robot="XArm6DexLeapRHOmron",
        robocasa_task="ScrewLightbulb",
        # Absolute joint targets (6 XArm6 joints + 16 hand = 22), collected/relabelled that
        # way, so eval MUST use the joint-position controller rather than the OSC default.
        controller_filename="default_xarm6dexleaprhomron_joint_pos.json",
        arm_action_dim=6,
        # Both eval sub-envs are construction seed 1 (kitchen L4S8), differing only in where
        # the lamp stands -- the regime the downstream DSRL run trains and evaluates in. So
        # lamp in-loop success is a seed-1 number, NOT a cross-kitchen generalization number;
        # score other kitchens post-hoc with scripts/eval_sft_on_banks.py.
        eval_scene_seeds=(1, 1),
        eval_placement_bank=LAMP_PLACEMENT_BANK,
        eval_placement_ids=LAMP_EVAL_PLACEMENT_IDS,
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
    # ------------------------------------------------------------------ arm 0
    "arm0_expert_only": ExperimentSpec(
        name="arm0_expert_only",
        # train_expert_only freezes ALL of PaliGemma (LM + SigLIP + projector), so
        # freeze_vision_encoder is redundant -- see PI05Pytorch._set_requires_grad, where
        # the train_expert_only branch is taken and freeze_llm is skipped. Set anyway so a
        # diff against the 2026-08-31 run's train_config.json shows only real differences.
        freeze_vision_encoder=True,
        train_expert_only=True,
        peft=None,
        peak_lr=FULL_FT_PEAK_LR,
        policy_optimizer_lr=FULL_FT_PEAK_LR,
        expect_trainable=(
            "gemma_expert",
            "time_mlp_in",
            "action_in_proj",
            "action_out_proj",
        ),
        # Nothing in the VLM moves, including the vision path that arm2 leaves trainable.
        expect_frozen=(
            "paligemma.model.language_model",
            "paligemma.model.vision_tower",
            "paligemma.model.multi_modal_projector",
            "paligemma.lm_head",
        ),
        expect_trainable_only=(
            r"gemma_expert",
            r"time_mlp_(in|out)",
            r"action_(in|out)_proj",
        ),
        notes=(
            "Frozen VLM, expert-only. arm1 minus arm0 is the trainable set; arm2 minus "
            "arm0 is SigLIP. Smallest trainable set (693M) and cheapest arm to train, so "
            "it also bounds the overfitting hypothesis. Runs at the matrix LR, not the "
            "5e-5 the historical expert-only lamp run used; --lr-scale 2 reaches that "
            "peak, though not its flat shape (DECAY_FRACTION still applies)."
        ),
    ),
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
            "Everything trainable at the openpi-default LR (2.5e-5 peak), via the policy "
            "training preset. Centre of the LR sweep: --lr-scale {0.5,1,2}."
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
        # Nothing outside these five components may train.
        expect_trainable_only=(
            r"vision_tower",
            r"multi_modal_projector",
            r"gemma_expert",
            r"time_mlp_(in|out)",
            r"action_(in|out)_proj",
        ),
        notes=(
            "Freezes ONLY PaliGemma's language model; the vision tower, the multimodal "
            "projector and the whole action expert full-FT at the openpi default LR. "
            "Isolates 'keep the pretrained language/semantic representation fixed' from "
            "'keep the whole VLM fixed' (arm0), which also freezes SigLIP. Needs "
            "PI05Config.freeze_llm, added for this study."
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
            # All LoRA params at 10x the full-FT LR (LWR), uniform across VLM-LM and
            # expert. Patterns are re.search'd against post-PEFT parameter names, which
            # carry a `base_model.model.` prefix and a `.lora_A.default.weight` suffix.
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
            "Ferchau-matched: LoRA r=32/alpha=32 uniform on VLM-LM + expert (all linears "
            "incl. MLPs), SigLIP + projector + action projections full-FT. Matches "
            "full-FT at the SFT endpoint, so downstream differences vs arm1 are "
            "hidden-property signal. LoRA LR 2.5e-4 (the 10x LWR rule). --lr-scale "
            "sweeps the LoRA groups over {1.25e-4, 2.5e-4, 5e-4} and scales the full-FT "
            "islands with them."
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
    save_freq: int = SAVE_FREQ,
    color_jitter: dict[str, float] | None = None,
    state_dropout_p: float = STATE_DROPOUT_P,
) -> TrainPipelineConfig:
    spec = EXPERIMENTS[experiment]
    task_spec = TASKS[task]
    if color_jitter is None:
        color_jitter = COLOR_JITTER
    metadata = LeRobotDatasetMetadata(task_spec.dataset)
    action_dim = metadata.features["action"]["shape"][0]
    state_dim = metadata.features["observation.state"]["shape"][0]

    # The eval controller is only correct if it consumes exactly the action vector the
    # dataset stores. Training never notices a mismatch -- it only reads dataset actions --
    # so a wrong controller shows up solely as every rollout failing.
    expected_action_dim = task_spec.arm_action_dim + LEAP_HAND_DIMS
    if expected_action_dim != action_dim:
        raise ValueError(
            f"Task '{task_spec.name}': controller {task_spec.controller_filename or '<robot default>'} "
            f"consumes {task_spec.arm_action_dim} arm + {LEAP_HAND_DIMS} hand = {expected_action_dim} "
            f"dims, but dataset '{task_spec.dataset}' stores {action_dim}-dim actions. "
            f"({task_spec.robot} has {_ARM_JOINTS[task_spec.robot]} arm joints: an absolute "
            f"JOINT_POSITION config needs that many arm dims, OSC_POSE needs 6.)"
        )

    env = RoboCasaEnv(
        task=task_spec.robocasa_task,
        robot=task_spec.robot,
        # Absolute-joint-target policies must be evaluated with the joint-position
        # controller; the default (None -> OSC_POSE) would be a silent mismatch.
        controller=resolve_controller(task_spec.controller_filename),
        # Eval control_freq must equal the dataset's collection rate (30 Hz for the
        # quest_rokoko dex datasets). Read it from metadata so it cannot drift.
        fps=metadata.fps,
        camera_name=task_spec.camera_name,
        # Pin each eval sub-env to one kitchen + one banked object placement (lamp only;
        # None for coffee keeps the seed-per-sub-env default).
        scene_seeds=list(task_spec.eval_scene_seeds) if task_spec.eval_scene_seeds else None,
        placement_bank=task_spec.eval_placement_bank,
        placement_ids=list(task_spec.eval_placement_ids) if task_spec.eval_placement_ids else None,
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
            image_transforms=make_image_transforms(color_jitter),
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
            # MEASURED: compile_model=True uses compile_mode="max-autotune", whose
            # CUDA-graph private pools take ~40 GiB on top of the ~45 GiB this run needs,
            # which OOMs an 80 GiB H100 at batch_size=32. Off: 1.68 s/step, 45.0 GiB peak
            # (~9.3 h per 20k-step run). compile_mode="default" avoids the cudagraphs and
            # is the thing to try if step time becomes the bottleneck.
            compile_model=False,
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
            state_dropout_p=state_dropout_p,
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
        save_freq=save_freq,
        log_freq=50,
        tolerance_s=1e-3,
        # batch_size is the number of eval sub-envs, i.e. one per start condition in the task
        # spec; n_episodes is split over them (20 episodes = 10 per lamp placement).
        eval=EvalConfig(n_episodes=20, n_videos=5, batch_size=task_spec.eval_batch_size),
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


def apply_smoke_test_overrides(cfg: TrainPipelineConfig) -> TrainPipelineConfig:
    """Shrink a real run to a few steps so a batch job's whole path is exercised cheaply.

    Deliberately changes ONLY sizes and cadences, never which code runs: the dataloader
    (with colour jitter), state dropout, forward/backward, the optimizer step, a
    checkpoint write, an environment rollout and a wandb log all still happen. Anything
    that would crash a 12h job in its first minutes crashes here in a few.
    """
    cfg.steps = 6
    cfg.save_freq = 3
    cfg.eval_freq = 3
    cfg.log_freq = 1
    cfg.batch_size = 2
    cfg.num_workers = 2
    cfg.eval.n_episodes = 2
    cfg.eval.batch_size = 2
    cfg.eval.n_videos = 1
    cfg.job_name = f"SMOKETEST_{cfg.job_name}"
    cfg.output_dir = make_timestamped_output_dir(SCRATCH_OUTPUT_ROOT / "smoketest", cfg)
    return cfg


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
        "--color-jitter-params",
        nargs="*",
        metavar="NAME VALUE",
        default=None,
        help="Colour-jitter parameters as alternating name/value pairs, e.g. "
        "`--color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08` "
        f"(the default: {COLOR_JITTER}). Pass with no values to disable augmentation.",
    )
    parser.add_argument(
        "--state-dropout-p",
        type=float,
        default=STATE_DROPOUT_P,
        help="Probability that a training sample has its entire state zeroed, GR00T-style "
        f"(default {STATE_DROPOUT_P}; 0 disables).",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=SAVE_FREQ,
        help=f"Checkpoint cadence (default {SAVE_FREQ}). A full-FT checkpoint is 23 GiB "
        "(8.7 weights + 14.1 AdamW state), so steps/save_freq rungs is also a disk budget. "
        "Keep it identical across arms -- it is a declared fairness invariant.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build policy + optimizer and assert this arm's trainable set, then exit.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a few real training steps with a checkpoint, an eval rollout and a "
        "wandb log, into <output root>/smoketest/. For validating a batch job's "
        "environment before queueing 12h of it.",
    )
    args = parser.parse_args()
    args.color_jitter = parse_color_jitter(args.color_jitter_params)
    return args


_COLOR_JITTER_KEYS = ("brightness", "contrast", "saturation", "hue")


def parse_color_jitter(pairs: list[str] | None) -> dict[str, float] | None:
    """`["brightness", "0.3", "hue", "0.08"]` -> `{"brightness": 0.3, "hue": 0.08}`.

    None (flag absent) means "use COLOR_JITTER"; an empty list means "no augmentation".
    """
    if pairs is None:
        return None
    if len(pairs) == 0:
        return {}
    if len(pairs) % 2 != 0:
        raise ValueError(f"--color-jitter-params needs name/value pairs, got {len(pairs)} items: {pairs}")
    parsed: dict[str, float] = {}
    for name, value in zip(pairs[::2], pairs[1::2], strict=True):
        if name not in _COLOR_JITTER_KEYS:
            raise ValueError(f"Unknown colour-jitter parameter {name!r}; expected one of {_COLOR_JITTER_KEYS}.")
        parsed[name] = float(value)
    return parsed


if __name__ == "__main__":
    args = parse_args()
    register_third_party_plugins()
    cfg = make_config(
        args.experiment,
        args.task,
        args.lr_scale,
        args.seed,
        args.eval_freq,
        args.steps,
        save_freq=args.save_freq,
        color_jitter=args.color_jitter,
        state_dropout_p=args.state_dropout_p,
    )
    if args.smoke_test:
        cfg = apply_smoke_test_overrides(cfg)
    if args.dry_run:
        run_dry_run(cfg, EXPERIMENTS[args.experiment])
    else:
        train(cfg)
