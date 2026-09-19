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
Public API for lerobot configuration types and base config classes.

NOTE: TrainPipelineConfig, EvalPipelineConfig, and TrainRLServerPipelineConfig
are intentionally NOT re-exported here to avoid circular dependencies
(they import lerobot.envs and lerobot.policies at module level).
Import them directly: ``from lerobot.configs.train import TrainPipelineConfig``
"""

import typing
from collections.abc import Sequence
from typing import Any

from draccus.parsers.decoding import decode as _draccus_decode
from draccus.utils import DecodingError

from .dataset import DatasetRecordConfig
from .default import DatasetConfig, EvalConfig, PeftConfig, WandBConfig
from .policies import PreTrainedConfig
from .types import (
    FeatureType,
    NormalizationMode,
    PipelineFeatureType,
    PolicyFeature,
    RTCAttentionSchedule,
)


def _decode_literal(cls: Any, raw_value: Any, path: Sequence[str] = ()) -> Any:
    """Decode a ``typing.Literal[...]`` field.

    draccus 0.10 has no decoder for ``Literal``, so any config dataclass with a
    ``Literal`` field fails to round-trip: saving works (the value is a plain str)
    but ``from_pretrained`` raises ``DecodingError``. That breaks checkpoint reload,
    ``--resume`` and offline eval for every policy that uses one (currently pi05's
    and smolvla's ``category_specific_action_proj_type``).

    ``typing.get_origin(Literal["a", "b"])`` is ``typing.Literal``, and draccus
    dispatches on the origin, so one registration covers every ``Literal`` field.
    """
    allowed = typing.get_args(cls)
    if raw_value not in allowed:
        raise DecodingError(path, f"{raw_value!r} is not one of {allowed}")
    return raw_value


_draccus_decode.register(typing.Literal, _decode_literal, include_subclasses=True)


__all__ = [
    # Types
    "FeatureType",
    "NormalizationMode",
    "PipelineFeatureType",
    "PolicyFeature",
    "RTCAttentionSchedule",
    # Config classes
    "DatasetRecordConfig",
    "DatasetConfig",
    "EvalConfig",
    "PeftConfig",
    "PreTrainedConfig",
    "WandBConfig",
]
