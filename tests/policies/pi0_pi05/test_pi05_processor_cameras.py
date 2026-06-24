#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import torch

from lerobot.policies.pi05.processor_pi05 import (
    PI05_CANONICAL_IMAGE_FEATURE_ALIASES,
    Pi05CanonicalImageObservationsProcessorStep,
)
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey


def test_pi05_canonical_image_processor_maps_env_camera_aliases():
    step = Pi05CanonicalImageObservationsProcessorStep(
        canonical_image_aliases=PI05_CANONICAL_IMAGE_FEATURE_ALIASES,
        image_shapes={
            "observation.images.right_wrist": (3, 256, 256),
            "observation.images.left_wrist": (3, 256, 256),
            "observation.images.external_one": (3, 256, 256),
        },
    )
    right_wrist = torch.ones(3, 64, 64)
    external = torch.ones(3, 32, 32) * 2

    processed = step(
        create_transition(
            observation={
                "observation.images.robot0_eye_in_hand": right_wrist,
                "observation.images.robot0_agentview_right": external,
            }
        )
    )
    observation = processed[TransitionKey.OBSERVATION]

    assert observation["observation.images.right_wrist"] is right_wrist
    assert observation["observation.images.external_one"] is external
    assert not observation["observation.images.right_wrist_is_pad"].item()
    assert not observation["observation.images.external_one_is_pad"].item()
    assert observation["observation.images.left_wrist_is_pad"].item()
    assert observation["observation.images.left_wrist"].shape == (3, 256, 256)
    assert torch.count_nonzero(observation["observation.images.left_wrist"]) == 0


def test_pi05_canonical_image_processor_preserves_existing_canonical_pad_flags():
    step = Pi05CanonicalImageObservationsProcessorStep(
        canonical_image_aliases=PI05_CANONICAL_IMAGE_FEATURE_ALIASES,
        image_shapes={"observation.images.left_wrist": (3, 256, 256)},
    )
    left_wrist = torch.zeros(3, 256, 256)

    processed = step(create_transition(observation={"observation.images.left_wrist": left_wrist}))

    observation = processed[TransitionKey.OBSERVATION]
    assert observation["observation.images.left_wrist"] is left_wrist
    assert "observation.images.left_wrist_is_pad" not in observation
