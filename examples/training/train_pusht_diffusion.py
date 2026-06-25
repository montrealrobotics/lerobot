#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Train a Diffusion Policy on the PushT dataset with periodic online evaluation.

Uses the standard LeRobot training pipeline (``lerobot.scripts.lerobot_train.train``)
with WandB logging and PushT-v0 online evaluation.

Example usage:
    python examples/training/train_pusht_diffusion.py

This produces checkpoints at a timestamped directory under ``outputs/train/`` that
can be used directly by the DSRL Pusht example.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from lerobot.configs import FeatureType
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.configs import PushtEnv
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.scripts.lerobot_train import train
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.utils.import_utils import register_third_party_plugins

OUTPUT_ROOT = Path("/network/scratch/a/artur.kuramshin/lerobot/outputs/train")



def make_timestamped_output_dir(root: Path, cfg: TrainPipelineConfig) -> Path:
    """Create a timestamped output directory matching the pattern used by other training scripts."""
    now = dt.datetime.now()
    job_name = cfg.job_name or f"{cfg.env.type}_{cfg.trainable_config.type}"
    return root / f"{now:%Y-%m-%d}" / f"{now:%H-%M-%S}_{job_name}"


def make_config() -> TrainPipelineConfig:
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in features.items() if k not in output_features}

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/pusht"),
        env=PushtEnv(),
        policy=DiffusionConfig(
            input_features=input_features,
            output_features=output_features,
            push_to_hub=False,
            noise_scheduler_type="DDIM",
            num_inference_steps=100,
            resize_shape=(96,96),
            crop_shape=(84,84),
        ),
        wandb=WandBConfig(enable=True, project="lerobot-pusht"),
        eval=EvalConfig(n_episodes=15, batch_size=1),
        steps=200_000,
        eval_freq=10_000,
        save_freq=20_000,
        log_freq=100,
        batch_size=64,
        num_workers=4,
        seed=0,
    )
    cfg.output_dir = make_timestamped_output_dir(OUTPUT_ROOT, cfg)
    return cfg


if __name__ == "__main__":
    register_third_party_plugins()
    train(make_config())
