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
Finetune pi05 on the DexMimicGen SingleArmDrawerCleanup dataset.

This mirrors examples/training/train_policy_casa_dex_pi05.py for dataset/env selection, but
targets the dexmimicgen SingleArmDrawerCleanup task with a PandaDexRH robot.

Usage:
    python examples/training/train_policy_casa_dex_pi05_dexmimicgen.py
"""

import datetime as dt
from pathlib import Path

from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.configs import DexMimicGenEnv
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts.lerobot_train import train
from lerobot.utils.import_utils import register_third_party_plugins

PI05_BASE_MODEL = "lerobot/pi05_base"
DEXMIMICGEN_EMBODIMENT_ID = 30
SCRATCH_OUTPUT_ROOT = Path("/network/scratch/a/artur.kuramshin/lerobot/outputs/train")
# DATASET_PATH = "/network/scratch/a/artur.kuramshin/huggingface/lerobot/dexmimicgen_single_arm_drawer_cleanup"
DATASET_ID = "dexmimicgen_single_arm_drawer_cleanup"


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
    dataset_metadata = LeRobotDatasetMetadata(DATASET_ID)
    action_dim = dataset_metadata.features["action"]["shape"][0]
    state_dim = dataset_metadata.features["observation.state"]["shape"][0]

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DATASET_ID, video_backend="pyav"),
        env=DexMimicGenEnv(
            task="SingleArmDrawerCleanup",
            camera_name="agentview,robot0_eye_in_hand",
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
            use_category_specific_action_proj=True,
            max_num_embodiments=32,
            pretrained_action_proj_category=0,
            default_embodiment_id=DEXMIMICGEN_EMBODIMENT_ID,
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
        eval=EvalConfig(n_episodes=20, batch_size=2),
        batch_size=32,
        num_workers=4,
    )
    cfg.output_dir = make_timestamped_output_dir(SCRATCH_OUTPUT_ROOT, cfg)
    return cfg


if __name__ == "__main__":
    register_third_party_plugins()
    train(make_config())
