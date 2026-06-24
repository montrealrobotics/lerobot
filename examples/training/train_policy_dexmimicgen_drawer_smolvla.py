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
Finetune SmolVLA on the DexMimicGen bimanual_panda_hand.DrawerCleanup dataset,
with evaluation on the TwoArmDrawerCleanup DexMimicGen sim environment.

The dataset stores cameras under ``ego_view``, ``right_wrist_view``, and
``left_wrist_view``. A ``rename_map`` bridges these to the env-style names
``agentview``, ``robot0_eye_in_hand``, ``robot1_eye_in_hand`` so that
the policy sees the same observation keys during training (from the dataset)
and evaluation (from the sim env).

Usage:
    python examples/training/train_policy_dexmimicgen_drawer_smolvla.py --experiment baseline_expert_full_ft
    python examples/training/train_policy_dexmimicgen_drawer_smolvla.py --experiment exp1_expert_lora
    python examples/training/train_policy_dexmimicgen_drawer_smolvla.py --experiment exp2_vlm_expert_lora
    python examples/training/train_policy_dexmimicgen_drawer_smolvla.py --experiment exp3_vlm_lora_expert_full_ft
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
from lerobot.datasets.multi_dataset import resolve_dataset_repo_spec
from lerobot.envs.configs import DexMimicGenEnv
from lerobot.optim import (
    CosineDecayWithWarmupSchedulerConfig,
    NamedAdamWConfig,
    NamedParamGroupConfig,
)
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_train import train
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins

SMOLVLA_BASE_MODEL = "lerobot/smolvla_base"
DEXMIMICGEN_EMBODIMENT_ID = 30
SCRATCH_OUTPUT_ROOT = Path("/network/scratch/a/artur.kuramshin/lerobot/outputs/train")

# DexMimicGen bimanual_panda_hand.DrawerCleanup dataset
DATASET_NAME = "akuramshin/dexmimicgen"
DATASET_SUBSET = "bimanual_panda_hand.DrawerCleanup"
FULL_DATASET_ID = f"{DATASET_NAME}:{DATASET_SUBSET}"

# Dataset camera keys (these are the LeRobot-format keys stored on the Hub)
DATASET_CAMERA_KEYS = {
    "right_wrist_view": f"{OBS_IMAGES}.right_wrist_view",
    "left_wrist_view": f"{OBS_IMAGES}.left_wrist_view",
    "ego_view": f"{OBS_IMAGES}.ego_view",
}

# Policy / env camera keys — these are the canonical names the policy expects
# and the DexMimicGen eval env produces (after features_map).
POLICY_CAMERA_KEYS = {
    "robot0_eye_in_hand": f"{OBS_IMAGES}.robot0_eye_in_hand",
    "robot1_eye_in_hand": f"{OBS_IMAGES}.robot1_eye_in_hand",
    "agentview": f"{OBS_IMAGES}.agentview",
}

# Map dataset camera names → policy/env camera names
RENAME_MAP = {
    DATASET_CAMERA_KEYS["ego_view"]: POLICY_CAMERA_KEYS["agentview"],
    DATASET_CAMERA_KEYS["right_wrist_view"]: POLICY_CAMERA_KEYS["robot0_eye_in_hand"],
    DATASET_CAMERA_KEYS["left_wrist_view"]: POLICY_CAMERA_KEYS["robot1_eye_in_hand"],
}

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
        batch_size=16,
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
    "exp3_vlm_lora_expert_full_ft": ExperimentSpec(
        name="exp3_vlm_lora_expert_full_ft",
        freeze_vision_encoder=False,
        train_expert_only=False,
        batch_size=16,
        peft=PeftConfig(
            method_type="LORA",
            r=64,
            target_modules="all-linear",
            full_training_modules=[
                "model.vlm_with_expert.lm_expert",
                "model.state_proj",
                "model.action_in_proj",
                "model.action_out_proj",
                "model.action_time_mlp_in",
                "model.action_time_mlp_out",
            ],
        ),
        lr_groups={
            "vlm_lora": NamedParamGroupConfig(
                patterns=[r"vlm_with_expert\.vlm.*lora_", r"vision_model.*lora_"],
                lr=2e-4,
            ),
            "expert_full": NamedParamGroupConfig(
                patterns=[
                    r"lm_expert\.",
                    r"state_proj\.",
                    r"action_(in|out)_proj",
                    r"action_time_mlp_(in|out)\.",
                ],
                lr=5e-5,
            ),
        },
        peak_lr=2e-4,
        decay_lr=2e-5,
        policy_optimizer_lr=5e-5,
        notes="VLM LoRA rank 64 at 2e-4; action expert fully finetuned at 5e-5; embodiment MLP fully trained.",
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


def make_visual_feature(dataset_metadata: LeRobotDatasetMetadata, key: str) -> PolicyFeature:
    shape = tuple(dataset_metadata.features[key]["shape"])
    if len(shape) != 3:
        raise ValueError(f"Expected image feature '{key}' to have 3 dimensions, got {shape}.")

    if shape[0] == 3:
        chw_shape = shape
    elif shape[-1] == 3:
        chw_shape = (shape[-1], shape[0], shape[1])
    else:
        raise ValueError(f"Expected image feature '{key}' to be RGB, got {shape}.")

    return PolicyFeature(type=FeatureType.VISUAL, shape=chw_shape)


def make_env_config() -> DexMimicGenEnv:
    """Build a DexMimicGenEnv for eval on the TwoArmDrawerCleanup task.

    Uses three cameras: a shared agentview and one eye-in-hand per arm.
    The ``features_map`` maps the env's internal pixel keys to the
    LeRobot-style ``observation.images.*`` keys that the policy expects.
    """
    camera_name = "agentview,robot0_eye_in_hand,robot1_eye_in_hand"

    cfg = DexMimicGenEnv(
        task="TwoArmDrawerCleanup",
        dexmimicgen_path="~/dexmimicgen",
        camera_name=camera_name,
        fps=20,
        episode_length=400,
        obs_type="pixels_agent_pos",
        state_mode="full",
        action_dim=24,
        state_dim=40,
    )
    # Add the third (left-arm wrist) camera — the config's __post_init__ only
    # adds agentview and robot0_eye_in_hand by default.
    cfg.features["pixels/robot1_eye_in_hand_image"] = PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(cfg.wrist_observation_height, cfg.wrist_observation_width, 3),
    )
    cfg.features_map["pixels/robot1_eye_in_hand_image"] = POLICY_CAMERA_KEYS["robot1_eye_in_hand"]
    return cfg


def make_config(experiment: str = "baseline_expert_full_ft") -> TrainPipelineConfig:
    spec = EXPERIMENTS[experiment]
    # Resolve the repo_id:subfolder spec into a plain repo_id and local root path
    dataset_repo_id, dataset_root = resolve_dataset_repo_spec(FULL_DATASET_ID, revision="v3.0")
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    action_dim = dataset_metadata.features["action"]["shape"][0]
    state_dim = dataset_metadata.features["observation.state"]["shape"][0]

    # Build policy input features using env-style camera names (the canonical
    # keys after rename_map and env features_map are applied).
    input_features: dict[str, PolicyFeature] = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    for policy_key in POLICY_CAMERA_KEYS.values():
        dataset_key = next(k for k, v in RENAME_MAP.items() if v == policy_key)
        input_features[policy_key] = make_visual_feature(dataset_metadata, dataset_key)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=dataset_repo_id,
            root=str(dataset_root),
            video_backend="pyav",
        ),
        env=make_env_config(),
        job_name=f"dexmimicgen_drawer_smolvla_{spec.name}",
        resume=False,
        rename_map=RENAME_MAP,
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
            max_state_dim=max(32, state_dim),
            max_action_dim=max(32, action_dim),
            input_features=input_features,
            # No random camera selection — all three cameras are fixed inputs.
            random_external_camera_keys=[],
            random_external_camera_output_key=None,
            use_category_specific_action_proj=True,
            category_specific_action_proj_type="linear",
            max_num_embodiments=32,
            pretrained_action_proj_category=DEXMIMICGEN_EMBODIMENT_ID,
            default_embodiment_id=DEXMIMICGEN_EMBODIMENT_ID,
            default_normalization_id=0,
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
        default="baseline_expert_full_ft",
        help="Named DexMimicGen DrawerCleanup SmolVLA finetuning experiment to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    register_third_party_plugins()
    train(make_config(args.experiment))
