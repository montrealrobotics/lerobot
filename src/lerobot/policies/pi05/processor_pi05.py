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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    RoutedNormalizerProcessorStep,
    RoutedUnnormalizerProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_pi05 import PI05Config

PI05_CANONICAL_IMAGE_FEATURE_ALIASES = {
    "observation.images.right_wrist": [
        "observation.images.right_wrist_view",
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_eye_in_hand_image",
    ],
    "observation.images.left_wrist": [
        "observation.images.left_wrist_view",
    ],
    "observation.images.external_one": [
        "observation.images.ego_view",
        "observation.images.robot0_agentview_right",
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_agentview_center",
        "observation.images.agentview",
        "observation.images.agentview_image",
    ],
}


@ProcessorStepRegistry.register(name="pi05_canonical_image_observations_processor_step")
@dataclass
class Pi05CanonicalImageObservationsProcessorStep(ProcessorStep):
    """Expose raw environment camera names under pi05's canonical image keys."""

    canonical_image_aliases: dict[str, list[str]]
    image_shapes: dict[str, tuple[int, ...]]

    def __post_init__(self):
        self.image_shapes = {key: tuple(shape) for key, shape in self.image_shapes.items()}

    def get_config(self) -> dict[str, Any]:
        return {
            "canonical_image_aliases": self.canonical_image_aliases,
            "image_shapes": self.image_shapes,
        }

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return transition

        new_transition = transition.copy()
        new_observation = dict(observation)

        for canonical_key, aliases in self.canonical_image_aliases.items():
            if canonical_key not in self.image_shapes:
                continue

            if canonical_key in new_observation:
                continue

            source_key = next((alias for alias in aliases if alias in new_observation), None)
            if source_key is not None:
                new_observation[canonical_key] = new_observation[source_key]
                new_observation[f"{canonical_key}_is_pad"] = self._make_pad_flag(
                    new_observation, False, source_key=source_key
                )
            else:
                new_observation[canonical_key] = self._make_empty_image(new_observation, canonical_key)
                new_observation[f"{canonical_key}_is_pad"] = self._make_pad_flag(new_observation, True)

        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def _reference_image(self, observation: dict[str, Any], source_key: str | None = None) -> Any | None:
        if source_key is not None:
            return observation.get(source_key)
        return next(
            (
                value
                for key, value in observation.items()
                if key.startswith(f"{OBS_IMAGES}.") and isinstance(value, torch.Tensor | np.ndarray)
            ),
            None,
        )

    def _make_empty_image(self, observation: dict[str, Any], canonical_key: str) -> Any:
        target_shape = self.image_shapes[canonical_key]
        reference = self._reference_image(observation)

        if isinstance(reference, torch.Tensor):
            shape = target_shape
            if reference.ndim == len(target_shape) + 1:
                shape = (reference.shape[0], *target_shape)
            return torch.zeros(shape, dtype=reference.dtype, device=reference.device)

        if isinstance(reference, np.ndarray):
            shape = target_shape
            if reference.ndim == len(target_shape) + 1:
                shape = (reference.shape[0], *target_shape)
            return np.zeros(shape, dtype=reference.dtype)

        return torch.zeros(target_shape, dtype=torch.float32)

    def _make_pad_flag(
        self, observation: dict[str, Any], is_pad: bool, source_key: str | None = None
    ) -> torch.Tensor:
        reference = self._reference_image(observation, source_key=source_key)
        if isinstance(reference, torch.Tensor) and reference.ndim >= 4:
            return torch.full((reference.shape[0],), is_pad, dtype=torch.bool, device=reference.device)
        if isinstance(reference, np.ndarray) and reference.ndim >= 4:
            return torch.full((reference.shape[0],), is_pad, dtype=torch.bool)
        return torch.tensor(is_pad)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="pi05_random_external_camera_processor_step")
@dataclass
class Pi05RandomExternalCameraProcessorStep(ProcessorStep):
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


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_stats_by_route: dict[int, dict[str, dict[str, torch.Tensor]]] | None = None,
    route_feature_shapes: dict[int, dict[str, tuple[int, ...]]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    # OpenPI order: raw → relative → normalize → model → unnormalize → absolute
    maybe_random_external_camera_step: ProcessorStep | None = (
        Pi05RandomExternalCameraProcessorStep(
            camera_keys=config.random_external_camera_keys,
            output_key=config.random_external_camera_output_key,
            p_first_camera=config.random_external_camera_p,
        )
        if config.random_external_camera_keys and config.random_external_camera_output_key
        else None
    )
    canonical_image_step = Pi05CanonicalImageObservationsProcessorStep(
        canonical_image_aliases=PI05_CANONICAL_IMAGE_FEATURE_ALIASES,
        image_shapes={
            key: feature.shape
            for key, feature in config.image_features.items()
            if key in PI05_CANONICAL_IMAGE_FEATURE_ALIASES
        },
    )
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
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        *([maybe_random_external_camera_step] if maybe_random_external_camera_step is not None else []),
        canonical_image_step,
        AddBatchDimensionProcessorStep(),
        relative_step,
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        normalizer_step,
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        unnormalizer_step,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
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
