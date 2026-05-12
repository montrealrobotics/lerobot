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
Finetune pi05 on a mixture of DexMimicGen and RoboCasa dexterous simulation datasets.

The three DexMimicGen datasets share one category-specific action projection slot, and
the RoboCasa CoffeePressButton dataset uses a second slot.

Usage:
    python examples/training/train_policy_mixed_dex_pi05.py
"""

import datetime as dt
from pathlib import Path

from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import NormalizationMode
from lerobot.envs.configs import RoboCasaEnv
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts.lerobot_train import train
from lerobot.utils.import_utils import register_third_party_plugins

PI05_BASE_MODEL = "lerobot/pi05_base"
SCRATCH_OUTPUT_ROOT = Path("/network/scratch/a/artur.kuramshin/lerobot/outputs/train")

DEXMIMICGEN_EMBODIMENT_ID = 30
ROBOCASA_EMBODIMENT_ID = 31
DEXMIMICGEN_NORMALIZATION_ID = 0
ROBOCASA_NORMALIZATION_ID = 1
MAX_BIMANUAL_DEX_STATE_DIM = 64
MAX_BIMANUAL_DEX_ACTION_DIM = 64

DEXMIMICGEN_DATASETS = "akuramshin/dexmimicgen"
DATASET_REPO_IDS = [
    f"{DEXMIMICGEN_DATASETS}:bimanual_panda_hand.BoxCleanup",
    f"{DEXMIMICGEN_DATASETS}:bimanual_panda_hand.DrawerCleanup",
    f"{DEXMIMICGEN_DATASETS}:bimanual_panda_hand.LiftTray",
    "akuramshin/robocasa_coffeepressbutton_dex_augstyle",
]
DATASET_EMBODIMENT_IDS = [
    DEXMIMICGEN_EMBODIMENT_ID,
    DEXMIMICGEN_EMBODIMENT_ID,
    DEXMIMICGEN_EMBODIMENT_ID,
    ROBOCASA_EMBODIMENT_ID,
]
DATASET_NORMALIZATION_IDS = [
    DEXMIMICGEN_NORMALIZATION_ID,
    DEXMIMICGEN_NORMALIZATION_ID,
    DEXMIMICGEN_NORMALIZATION_ID,
    ROBOCASA_NORMALIZATION_ID,
]
DATASET_SAMPLING_WEIGHTS = [1.0, 1.0, 1.0, 3.0]


def make_timestamped_output_dir(root: Path, cfg: TrainPipelineConfig) -> Path:
    now = dt.datetime.now()
    if cfg.job_name is not None:
        job_name = cfg.job_name
    elif cfg.env is None:
        job_name = cfg.trainable_config.type
    else:
        job_name = f"{cfg.env.type}_{cfg.trainable_config.type}"
    return root / f"{now:%Y-%m-%d}" / f"{now:%H-%M-%S}_{job_name}"


def make_config() -> TrainPipelineConfig:
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=DATASET_REPO_IDS,
            embodiment_ids=DATASET_EMBODIMENT_IDS,
            normalization_ids=DATASET_NORMALIZATION_IDS,
            sampling_weights=DATASET_SAMPLING_WEIGHTS,
            video_backend="pyav",
        ),
        env=RoboCasaEnv(
            task="CoffeePressButton",
            robot="PandaDexLeapRHOmron",
            camera_name="robot0_agentview_right,robot0_eye_in_hand",
        ),
        job_name="mixed_dex_pi05",
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
            max_state_dim=MAX_BIMANUAL_DEX_STATE_DIM,
            max_action_dim=MAX_BIMANUAL_DEX_ACTION_DIM,
            force_current_processor_config=True,
            use_category_specific_action_proj=True,
            max_num_embodiments=32,
            pretrained_action_proj_category=0,
            default_embodiment_id=ROBOCASA_EMBODIMENT_ID,
            default_normalization_id=ROBOCASA_NORMALIZATION_ID,
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
        steps=20_000,
        eval_freq=500,
        save_freq=1_000,
        log_freq=50,
        tolerance_s=1e-3,
        eval=EvalConfig(n_episodes=20, batch_size=2),
        batch_size=64,
        num_workers=4,
    )
    cfg.output_dir = make_timestamped_output_dir(SCRATCH_OUTPUT_ROOT, cfg)
    return cfg


if __name__ == "__main__":
    register_third_party_plugins()
    train(make_config())
