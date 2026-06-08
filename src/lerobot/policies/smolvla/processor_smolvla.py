#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    EnvTransition,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    RoutedNormalizerProcessorStep,
    RoutedUnnormalizerProcessorStep,
    TokenizerProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_smolvla import SmolVLAConfig


@ProcessorStepRegistry.register(name="smolvla_random_external_camera_processor_step")
@dataclass
class SmolVLARandomExternalCameraProcessorStep(ProcessorStep):
    """Select one external camera and expose it under a canonical observation key."""

    camera_keys: list[str]
    output_key: str
    p_first_camera: float = 0.5

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_keys": self.camera_keys,
            "output_key": self.output_key,
            "p_first_camera": self.p_first_camera,
        }

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.camera_keys:
            return transition

        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return transition

        available_keys = [key for key in self.camera_keys if key in observation]
        if not available_keys:
            return transition

        new_transition = transition.copy()
        new_observation = dict(observation)

        if len(available_keys) == 1:
            new_observation[self.output_key] = new_observation[available_keys[0]]
        else:
            new_observation[self.output_key] = self._select_camera(
                [new_observation[key] for key in available_keys]
            )

        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def _select_camera(self, images: list[Any]) -> Any:
        first = images[0]
        if len(images) == 2 and self.p_first_camera != 0.5:
            return self._select_two_cameras(images[0], images[1])

        if isinstance(first, torch.Tensor):
            if first.ndim >= 4:
                batch_size = first.shape[0]
                indices = torch.randint(len(images), (batch_size,), device=first.device)
                stacked = torch.stack(images, dim=0)
                return stacked[indices, torch.arange(batch_size, device=first.device)]
            index = torch.randint(len(images), (1,), device=first.device).item()
            return images[index]

        if isinstance(first, np.ndarray):
            if first.ndim >= 4:
                batch_size = first.shape[0]
                indices = np.random.randint(len(images), size=batch_size)
                stacked = np.stack(images, axis=0)
                return stacked[indices, np.arange(batch_size)]
            return images[np.random.randint(len(images))]

        return images[np.random.randint(len(images))]

    def _select_two_cameras(self, first: Any, second: Any) -> Any:
        if isinstance(first, torch.Tensor):
            if first.ndim >= 4:
                mask_shape = (first.shape[0], *([1] * (first.ndim - 1)))
                mask = torch.rand(mask_shape, device=first.device) < self.p_first_camera
                return torch.where(mask, first, second)
            return first if torch.rand((), device=first.device).item() < self.p_first_camera else second

        if isinstance(first, np.ndarray):
            if first.ndim >= 4:
                mask_shape = (first.shape[0], *([1] * (first.ndim - 1)))
                mask = np.random.random(mask_shape) < self.p_first_camera
                return np.where(mask, first, second)
            return first if np.random.random() < self.p_first_camera else second

        return first if np.random.random() < self.p_first_camera else second

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_stats_by_route: dict[int, dict[str, dict[str, torch.Tensor]]] | None = None,
    route_feature_shapes: dict[int, dict[str, tuple[int, ...]]] | None = None,
    rename_map: dict[str, str] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    normalizer_step: ProcessorStep
    unnormalizer_step: ProcessorStep
    if dataset_stats_by_route:
        normalizer_step = RoutedNormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats_by_route=dataset_stats_by_route,
            default_route=config.default_normalization_id,
            route_feature_shapes=route_feature_shapes,
        )
        unnormalizer_step = RoutedUnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats_by_route=dataset_stats_by_route,
            default_route=config.default_normalization_id,
            route_feature_shapes=route_feature_shapes,
        )
    else:
        normalizer_step = NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )
        unnormalizer_step = UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        )

    maybe_random_external_camera_step: ProcessorStep | None = (
        SmolVLARandomExternalCameraProcessorStep(
            camera_keys=config.random_external_camera_keys,
            output_key=config.random_external_camera_output_key,
            p_first_camera=config.random_external_camera_p,
        )
        if config.random_external_camera_keys and config.random_external_camera_output_key
        else None
    )

    input_steps = [
        RenameObservationsProcessorStep(rename_map=rename_map or {}),
        *([maybe_random_external_camera_step] if maybe_random_external_camera_step is not None else []),
        AddBatchDimensionProcessorStep(),
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
        normalizer_step,
    ]
    output_steps = [
        unnormalizer_step,
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
