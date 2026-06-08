#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Finetune SmolVLA on a mixture of RoboCasa dexterous-hand and gripper CoffeePressButton datasets.

This mirrors examples/training/train_policy_mixed_robocasa_pi05.py, but uses
lerobot/smolvla_base and SmolVLA's embodiment-specific action projections.

Usage:
    python examples/training/train_policy_mixed_robocasa_smolvla.py --experiment exp1_expert_lora
    python examples/training/train_policy_mixed_robocasa_smolvla.py --experiment exp2_vlm_expert_lora
"""

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

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
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_train import train
from lerobot.utils.import_utils import register_third_party_plugins

SMOLVLA_BASE_MODEL = "lerobot/smolvla_base"
SCRATCH_OUTPUT_ROOT = Path("/network/scratch/a/artur.kuramshin/lerobot/outputs/train")

DEX_EMBODIMENT_ID = 1
GRIPPER_EMBODIMENT_ID = 0
DEX_NORMALIZATION_ID = 0
GRIPPER_NORMALIZATION_ID = 1
MAX_DEX_STATE_DIM = 32
MAX_DEX_ACTION_DIM = 32

WRIST_IMAGE_KEY = "observation.images.robot0_eye_in_hand"
EXTERNAL_IMAGE_KEY = "observation.images.robot0_agentview_left"
RANDOM_EXTERNAL_IMAGE_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
]

DEX_DATASET = "akuramshin/robocasa_coffeepressbutton_dex_augstyle"
GRIPPER_DATASET = "akuramshin/robocasa_coffeepressbutton-kitchen_coffee"

EXPERT_LINEAR_TARGETS = (
    r"model\.vlm_with_expert\.lm_expert\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"|model\.(state_proj|action_time_mlp_in|action_time_mlp_out)"
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    freeze_vision_encoder: bool
    train_expert_only: bool
    peft: PeftConfig | None
    lr_groups: dict[str, NamedParamGroupConfig]
    peak_lr: float
    decay_lr: float
    policy_optimizer_lr: float
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 4
    persistent_workers: bool = True
    notes: str = ""


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "baseline_expert_full_ft": ExperimentSpec(
        name="baseline_expert_full_ft",
        freeze_vision_encoder=True,
        train_expert_only=True,
        peft=None,
        lr_groups={},
        peak_lr=5e-5,
        decay_lr=5e-6,
        policy_optimizer_lr=5e-5,
        batch_size=32,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=False,
        notes="Frozen VLM, full SmolVLA action-expert finetune, batch 32, 10K steps.",
    ),
    "exp1_expert_lora": ExperimentSpec(
        name="exp1_expert_lora",
        freeze_vision_encoder=True,
        train_expert_only=True,
        peft=PeftConfig(
            method_type="LORA",
            r=32,
            target_modules=EXPERT_LINEAR_TARGETS,
            full_training_modules=["model.action_in_proj", "model.action_out_proj"],
        ),
        lr_groups={
            "expert_lora": NamedParamGroupConfig(
                patterns=[r"lm_expert.*lora_", r"action_time_mlp_(in|out).*lora_", r"state_proj.*lora_"],
                lr=3e-5,
            ),
            "embodiment_mlp": NamedParamGroupConfig(
                patterns=[r"action_(in|out)_proj"],
                lr=3e-5,
            ),
        },
        peak_lr=3e-5,
        decay_lr=3e-6,
        policy_optimizer_lr=3e-5,
        notes="VLM frozen; SmolVLA expert/projection LoRA rank 32; embodiment MLP trained.",
    ),
    "exp2_vlm_expert_lora": ExperimentSpec(
        name="exp2_vlm_expert_lora",
        freeze_vision_encoder=False,
        train_expert_only=False,
        peft=PeftConfig(
            method_type="LORA",
            r=64,
            target_modules="all-linear",
            rank_pattern={
                r".*lm_expert.*": 32,
                r".*action_time_mlp_(in|out).*": 32,
                r".*state_proj.*": 32,
            },
            full_training_modules=["model.action_in_proj", "model.action_out_proj"],
        ),
        lr_groups={
            "vlm_lora": NamedParamGroupConfig(patterns=[r"vlm_with_expert\.vlm.*lora_"], lr=2e-4),
            "expert_lora": NamedParamGroupConfig(
                patterns=[r"lm_expert.*lora_", r"action_time_mlp_(in|out).*lora_", r"state_proj.*lora_"],
                lr=3e-5,
            ),
            "embodiment_mlp": NamedParamGroupConfig(
                patterns=[r"action_(in|out)_proj"],
                lr=3e-5,
            ),
        },
        peak_lr=2e-4,
        decay_lr=2e-5,
        policy_optimizer_lr=3e-5,
        notes="VLM LoRA rank 64 at 2e-4; expert/projection LoRA rank 32 at 3e-5; embodiment MLP trained.",
    ),
}


def make_timestamped_output_dir(root: Path, cfg: TrainPipelineConfig) -> Path:
    now = dt.datetime.now()
    if cfg.job_name is not None:
        job_name = cfg.job_name
    elif cfg.env is None:
        job_name = cfg.trainable_config.type
    else:
        env_configs = cfg.eval_env_configs
        env_tag = "_".join(ec.type for ec in env_configs)
        job_name = f"{env_tag}_{cfg.trainable_config.type}"
    return root / f"{now:%Y-%m-%d}" / f"{now:%H-%M-%S}_{job_name}"


def make_config(experiment: str = "exp1_expert_lora") -> TrainPipelineConfig:
    spec = EXPERIMENTS[experiment]
    dex_metadata = LeRobotDatasetMetadata(DEX_DATASET)

    def _make_visual_feature(key: str) -> PolicyFeature:
        shape = tuple(dex_metadata.features[key]["shape"])
        if len(shape) != 3:
            raise ValueError(f"Expected image feature '{key}' to have 3 dimensions, got {shape}.")
        if shape[0] == 3:
            chw_shape = shape
        elif shape[-1] == 3:
            chw_shape = (shape[-1], shape[0], shape[1])
        else:
            raise ValueError(f"Expected image feature '{key}' to be RGB, got {shape}.")
        return PolicyFeature(type=FeatureType.VISUAL, shape=chw_shape)

    dex_env = RoboCasaEnv(
        task="CoffeePressButton",
        robot="PandaDexLeapRHOmron",
        camera_name=(
            "robot0_agentview_left,robot0_agentview_right,robot0_eye_in_hand,robot0_agentview_center"
        ),
    )
    gripper_env = RoboCasaEnv(
        task="CoffeePressButton",
        robot="PandaOmron",
        camera_name=(
            "robot0_agentview_left,robot0_agentview_right,robot0_eye_in_hand,robot0_agentview_center"
        ),
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=[DEX_DATASET, GRIPPER_DATASET],
            embodiment_ids=[DEX_EMBODIMENT_ID, GRIPPER_EMBODIMENT_ID],
            normalization_ids=[DEX_NORMALIZATION_ID, GRIPPER_NORMALIZATION_ID],
            sampling_weights=[1.0, 1.0],
            video_backend="pyav",
        ),
        env=[dex_env, gripper_env],
        suite_metadata={
            "robocasa": {"normalization_id": DEX_NORMALIZATION_ID, "embodiment_id": DEX_EMBODIMENT_ID},
            "robocasa_1": {
                "normalization_id": GRIPPER_NORMALIZATION_ID,
                "embodiment_id": GRIPPER_EMBODIMENT_ID,
            },
        },
        job_name=f"mixed_robocasa_dex_gripper_smolvla_{spec.name}",
        resume=False,
        policy=SmolVLAConfig(
            pretrained_path=Path(SMOLVLA_BASE_MODEL),
            push_to_hub=False,
            load_vlm_weights=True,
            freeze_vision_encoder=spec.freeze_vision_encoder,
            train_expert_only=spec.train_expert_only,
            train_state_proj=True,
            force_current_processor_config=True,
            chunk_size=32,
            n_action_steps=16,
            max_state_dim=MAX_DEX_STATE_DIM,
            max_action_dim=MAX_DEX_ACTION_DIM,
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(MAX_DEX_STATE_DIM,)),
                WRIST_IMAGE_KEY: _make_visual_feature(WRIST_IMAGE_KEY),
                EXTERNAL_IMAGE_KEY: _make_visual_feature(EXTERNAL_IMAGE_KEY),
            },
            random_external_camera_keys=RANDOM_EXTERNAL_IMAGE_KEYS,
            random_external_camera_output_key=EXTERNAL_IMAGE_KEY,
            random_external_camera_p=0.5,
            use_category_specific_action_proj=True,
            category_specific_action_proj_type="mlp",
            max_num_embodiments=2,
            pretrained_action_proj_category=GRIPPER_EMBODIMENT_ID,
            default_embodiment_id=DEX_EMBODIMENT_ID,
            default_normalization_id=DEX_NORMALIZATION_ID,
            normalization_mapping={
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.MEAN_STD,
                "ACTION": NormalizationMode.MEAN_STD,
            },
            optimizer_lr=spec.policy_optimizer_lr,
            scheduler_warmup_steps=1_000,
            scheduler_decay_steps=30_000,
            scheduler_decay_lr=spec.decay_lr,
            optimizer_grad_clip_norm=1.0,
            num_steps=10,
            tokenizer_max_length=48,
            pad_language_to="max_length",
            prefix_length=0,
            num_expert_layers=0,
        ),
        use_policy_training_preset=spec.peft is None,
        optimizer=(
            None
            if spec.peft is None
            else NamedAdamWConfig(
                lr=spec.policy_optimizer_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.01,
                grad_clip_norm=1.0,
                param_groups=spec.lr_groups,
                fail_on_unmatched=True,
            )
        ),
        scheduler=(
            None
            if spec.peft is None
            else CosineDecayWithWarmupSchedulerConfig(
                peak_lr=spec.peak_lr,
                decay_lr=spec.decay_lr,
                num_warmup_steps=1_000,
                num_decay_steps=30_000,
            )
        ),
        peft=spec.peft,
        wandb=WandBConfig(enable=True, project="lerobot-dex"),
        steps=10_000,
        eval_freq=500,
        save_freq=1_000,
        log_freq=50,
        tolerance_s=1e-3,
        eval=EvalConfig(n_episodes=20, batch_size=2),
        batch_size=spec.batch_size,
        num_workers=spec.num_workers,
        prefetch_factor=spec.prefetch_factor,
        persistent_workers=spec.persistent_workers,
    )
    cfg.output_dir = make_timestamped_output_dir(SCRATCH_OUTPUT_ROOT, cfg)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENTS),
        default="exp1_expert_lora",
        help="Named mixed RoboCasa SmolVLA finetuning experiment to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    register_third_party_plugins()
    train(make_config(args.experiment))
