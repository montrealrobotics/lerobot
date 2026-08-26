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
Finetune pi05 on the RoboCasa dexterous-hand CoffeePressButton dataset.

This mirrors examples/training/train_policy_casa_dex.py for dataset/env selection, but
loads lerobot/pi05_base, freezes the VLM backbone, and trains the action expert plus a
fresh category-specific action encoder/decoder slot for the dexterous embodiment.

Usage:
    python examples/training/train_policy_casa_dex_pi05.py
"""

import datetime as dt
import os
from pathlib import Path

import robosuite

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.configs import RoboCasaEnv
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts.lerobot_train import train
from lerobot.utils.import_utils import register_third_party_plugins

PI05_BASE_MODEL = "lerobot/pi05_base"
ROBOCASA_DEX_EMBODIMENT_ID = 31
# Eval control mode must match the collected/trained action representation. This dataset is
# collected + relabeled to absolute joint targets, so evaluate with the JOINT_POSITION controller.
XARM6_JOINT_POS_CONTROLLER = os.path.join(
    os.path.dirname(robosuite.__file__),
    "controllers/config/robots/default_xarm6dexleaprhomron_joint_pos.json",
)
SCRATCH_OUTPUT_ROOT = Path("/network/scratch/a/artur.kuramshin/lerobot/outputs/train")
WRIST_IMAGE_KEY = "observation.images.robot0_eye_in_hand"
EXTERNAL_IMAGE_KEY = "observation.images.robot0_agentview_left"
RANDOM_EXTERNAL_IMAGE_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
]


def make_timestamped_output_dir(root: Path, cfg: TrainPipelineConfig) -> Path:
    now = dt.datetime.now()
    if cfg.job_name is not None:
        job_name = cfg.job_name
    elif cfg.env is None:
        job_name = cfg.trainable_config.type
    else:
        job_name = f"{cfg.env.type}_{cfg.trainable_config.type}"
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


def make_config() -> TrainPipelineConfig:
    # dataset_name = "akuramshin/robocasa_coffeepressbutton_dex_augstyle"
    dataset_name = "akuramshin/robocasa_coffeepressbutton-kitchen_coffee"
    dataset_metadata = LeRobotDatasetMetadata(dataset_name)
    action_dim = dataset_metadata.features["action"]["shape"][0]
    state_dim = dataset_metadata.features["observation.state"]["shape"][0]

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=dataset_name, video_backend="pyav"),
        env=RoboCasaEnv(
            task="CoffeePressButton",
            robot="XArm6DexLeapRHOmron",
            # robot="PandaDexLeapRHOmron",
            # robot="PandaOmron",
            controller=XARM6_JOINT_POS_CONTROLLER,  # joint-position eval to match trained actions
            camera_name=(
                "robot0_agentview_left,"
                "robot0_agentview_right,"
                "robot0_eye_in_hand,"
                "robot0_agentview_center"
            ),
        ),
        resume=False,
        policy=PI05Config(
            pretrained_path=Path(PI05_BASE_MODEL),
            push_to_hub=False,
            dtype="bfloat16",
            gradient_checkpointing=True,
            compile_model=True,
            freeze_vision_encoder=True,
            train_expert_only=True,
            chunk_size=32,
            n_action_steps=16,
            max_state_dim=max(32, state_dim),
            max_action_dim=max(32, action_dim),
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
                WRIST_IMAGE_KEY: make_visual_feature(dataset_metadata, WRIST_IMAGE_KEY),
                EXTERNAL_IMAGE_KEY: make_visual_feature(dataset_metadata, EXTERNAL_IMAGE_KEY),
            },
            random_external_camera_keys=RANDOM_EXTERNAL_IMAGE_KEYS,
            random_external_camera_output_key=EXTERNAL_IMAGE_KEY,
            random_external_camera_p=0.5,
            force_current_processor_config=True,
            use_category_specific_action_proj=True,
            max_num_embodiments=32,
            pretrained_action_proj_category=0,
            default_embodiment_id=ROBOCASA_DEX_EMBODIMENT_ID,
            normalization_mapping={
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.QUANTILES,
                "ACTION": NormalizationMode.QUANTILES,
            },
            optimizer_lr=5e-5,
            scheduler_warmup_steps=1_000,
            scheduler_decay_steps=30_000,
            scheduler_decay_lr=5e-5,
            optimizer_grad_clip_norm=1.0,
        ),
        wandb=WandBConfig(enable=True, project="lerobot-dex"),
        steps=10_000,
        eval_freq=500,
        save_freq=1_000,
        log_freq=50,
        eval=EvalConfig(n_episodes=20, n_videos=10, batch_size=5),
        batch_size=32,
        num_workers=4,
    )
    cfg.output_dir = make_timestamped_output_dir(SCRATCH_OUTPUT_ROOT, cfg)
    return cfg


if __name__ == "__main__":
    register_third_party_plugins()
    train(make_config())
