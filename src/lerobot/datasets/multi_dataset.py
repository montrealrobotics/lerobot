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
import logging
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils
from huggingface_hub import snapshot_download

from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, HF_LEROBOT_HUB_CACHE, OBS_STATE

from .compute_stats import aggregate_stats
from .feature_utils import get_hf_features_from_features
from .lerobot_dataset import LeRobotDataset
from .video_utils import VideoFrame

logger = logging.getLogger(__name__)

NORMALIZED_FEATURE_DTYPES = {"timestamp": "float32"}
CANONICAL_IMAGE_FEATURE_ALIASES = {
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


def split_dataset_repo_spec(repo_spec: str) -> tuple[str, str | None]:
    """Split ``repo_id[:subfolder]`` dataset specs.

    Hugging Face repo ids contain exactly one slash, so a colon is unambiguous for the optional
    path-within-repo used by dataset collections such as GR00T-X Embodiment Sim.
    """
    repo_id, sep, subfolder = repo_spec.partition(":")
    if not sep:
        return repo_spec, None
    if not repo_id or not subfolder:
        raise ValueError(f"Invalid dataset repo spec '{repo_spec}'. Expected 'repo_id[:subfolder]'.")
    return repo_id, subfolder.strip("/")


def resolve_dataset_repo_spec(
    repo_spec: str,
    root: str | Path | None = None,
    revision: str | None = None,
) -> tuple[str, Path | None]:
    """Resolve a dataset spec to the repo id and local root passed to ``LeRobotDataset``."""
    repo_id, subfolder = split_dataset_repo_spec(repo_spec)

    if subfolder is None:
        if root is None:
            return repo_id, None
        return repo_id, Path(root) / repo_id

    base_root = Path(root) if root is not None else HF_LEROBOT_HOME
    local_root = base_root / repo_id / subfolder
    if local_root.exists():
        return repo_id, local_root

    if root is not None:
        repo_root = base_root / repo_id
        repo_root.mkdir(exist_ok=True, parents=True)
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=repo_root,
            allow_patterns=f"{subfolder}/**",
        )
        return repo_id, repo_root / subfolder

    snapshot_root = Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=HF_LEROBOT_HUB_CACHE,
            allow_patterns=f"{subfolder}/**",
        )
    )
    return repo_id, snapshot_root / subfolder


class MultiLeRobotDatasetMetadata:
    """Metadata view over a collection of LeRobot datasets."""

    def __init__(
        self,
        repo_ids: list[str],
        datasets_: list[LeRobotDataset],
        enabled_features: set[str],
        stats: dict,
        stats_by_normalization_id: dict[int, dict] | None = None,
        feature_shapes_by_normalization_id: dict[int, dict[str, tuple[int, ...]]] | None = None,
        feature_shapes: dict[str, tuple[int, ...]] | None = None,
        feature_overrides: dict[str, dict] | None = None,
    ):
        self.repo_id = repo_ids
        self.repo_ids = repo_ids
        self.root = [dataset.root for dataset in datasets_]
        self.revision = [dataset.revision for dataset in datasets_]
        self.stats = stats
        self.stats_by_normalization_id = stats_by_normalization_id or {}
        self.feature_shapes_by_normalization_id = feature_shapes_by_normalization_id or {}
        self._fps = datasets_[0].meta.fps
        self._robot_types = [dataset.meta.robot_type for dataset in datasets_]

        first_features = datasets_[0].meta.features
        self._features = {
            key: deepcopy(first_features[key]) for key in first_features if key in enabled_features
        }
        for key, feature in (feature_overrides or {}).items():
            if key in enabled_features:
                self._features[key] = deepcopy(feature)
        for key, shape in (feature_shapes or {}).items():
            if key in self._features:
                self._features[key]["shape"] = shape
        for key, dtype in NORMALIZED_FEATURE_DTYPES.items():
            if key in self._features:
                self._features[key]["dtype"] = dtype
        self.episodes = self._aggregate_episodes(datasets_)

    def _aggregate_episodes(self, datasets_: list[LeRobotDataset]) -> datasets.Dataset:
        rows = []
        frame_offset = 0
        episode_offset = 0
        for dataset_index, dataset in enumerate(datasets_):
            for episode in dataset.meta.episodes:
                row = dict(episode)
                row["episode_index"] = int(row["episode_index"]) + episode_offset
                row["dataset_from_index"] = int(row["dataset_from_index"]) + frame_offset
                row["dataset_to_index"] = int(row["dataset_to_index"]) + frame_offset
                row["source_dataset_index"] = dataset_index
                rows.append(row)
            frame_offset += dataset.num_frames
            episode_offset += dataset.num_episodes
        return datasets.Dataset.from_list(rows)

    @property
    def robot_type(self) -> str | None:
        robot_types = set(self._robot_types)
        return None if len(robot_types) != 1 else next(iter(robot_types))

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def features(self) -> dict[str, dict]:
        return self._features

    @property
    def image_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)

    @property
    def total_frames(self) -> int:
        return int(self.episodes[len(self.episodes) - 1]["dataset_to_index"]) if len(self.episodes) else 0

    @property
    def total_tasks(self) -> int:
        tasks = set()
        for episode in self.episodes:
            tasks.update(episode.get("tasks", []))
        return len(tasks)


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        revision: str | None = None,
        return_uint8: bool = False,
        embodiment_ids: list[int] | None = None,
        normalization_ids: list[int] | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else None
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        self.embodiment_ids = embodiment_ids
        if self.embodiment_ids is not None and len(self.embodiment_ids) != len(repo_ids):
            raise ValueError(
                f"Expected one embodiment id per dataset, got {len(self.embodiment_ids)} for {len(repo_ids)}."
            )
        self.normalization_ids = normalization_ids or list(range(len(repo_ids)))
        if len(self.normalization_ids) != len(repo_ids):
            raise ValueError(
                f"Expected one normalization id per dataset, got {len(self.normalization_ids)} "
                f"for {len(repo_ids)}."
            )
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = []
        for repo_spec in repo_ids:
            repo_id, dataset_root = resolve_dataset_repo_spec(repo_spec, root=self.root, revision=revision)
            dataset_episodes = None
            if episodes:
                dataset_episodes = episodes.get(repo_spec, episodes.get(repo_id))
            self._datasets.append(
                LeRobotDataset(
                    repo_id,
                    root=dataset_root,
                    episodes=dataset_episodes,
                    image_transforms=image_transforms,
                    delta_timestamps=delta_timestamps,
                    tolerance_s=self.tolerances_s[repo_spec],
                    download_videos=download_videos,
                    video_backend=video_backend,
                    revision=revision,
                    return_uint8=return_uint8,
                )
            )
        fps_values = {dataset.meta.fps for dataset in self._datasets}
        if len(fps_values) > 1:
            raise ValueError(
                f"Multi-dataset training requires matching FPS values, got {sorted(fps_values)}."
            )

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        raw_intersection_features = intersection_features
        self.canonical_image_features = self._get_canonical_image_features()
        intersection_features = raw_intersection_features | set(self.canonical_image_features)

        self.padded_feature_shapes = self._get_padded_feature_shapes(raw_intersection_features)
        self._validate_common_feature_shapes(raw_intersection_features)
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(raw_intersection_features)
            if extra_keys:
                logger.warning(
                    f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                    "other datasets."
                )
                self.disabled_features.update(extra_keys)

        self.delta_timestamps = delta_timestamps
        filtered_stats = [
            self._pad_stats(self._get_enabled_stats(dataset_index, raw_intersection_features))
            for dataset_index in range(len(self._datasets))
        ]
        self.stats = aggregate_stats(filtered_stats)
        self.stats_by_normalization_id = self._aggregate_stats_by_normalization_id(filtered_stats)
        self.feature_shapes_by_normalization_id = self._get_feature_shapes_by_normalization_id(
            raw_intersection_features
        )
        self.meta = MultiLeRobotDatasetMetadata(
            repo_ids=self.repo_ids,
            datasets_=self._datasets,
            enabled_features=intersection_features,
            stats=self.stats,
            stats_by_normalization_id=self.stats_by_normalization_id,
            feature_shapes_by_normalization_id=self.feature_shapes_by_normalization_id,
            feature_shapes=self.padded_feature_shapes,
            feature_overrides=self.canonical_image_features,
        )
        self.set_image_transforms(image_transforms)

    def _aggregate_stats_by_normalization_id(self, filtered_stats: list[dict[str, dict]]) -> dict[int, dict]:
        stats_by_id: dict[int, list[dict[str, dict]]] = {}
        for normalization_id, stats in zip(self.normalization_ids, filtered_stats, strict=True):
            stats_by_id.setdefault(int(normalization_id), []).append(stats)
        return {
            normalization_id: aggregate_stats(stats_list)
            for normalization_id, stats_list in stats_by_id.items()
        }

    def _get_feature_shapes_by_normalization_id(
        self, feature_keys: set[str]
    ) -> dict[int, dict[str, tuple[int, ...]]]:
        feature_shapes_by_id: dict[int, dict[str, tuple[int, ...]]] = {}
        for normalization_id in sorted(set(self.normalization_ids)):
            dataset_indices = [
                index for index, candidate_id in enumerate(self.normalization_ids) if candidate_id == normalization_id
            ]
            feature_shapes_by_id[int(normalization_id)] = {}
            for key in sorted(self.padded_feature_shapes.keys() & feature_keys):
                shapes = [
                    tuple(self._datasets[dataset_index].meta.features[key]["shape"])
                    for dataset_index in dataset_indices
                ]
                if shapes and all(len(shape) == 1 for shape in shapes):
                    feature_shapes_by_id[int(normalization_id)][key] = (max(shape[0] for shape in shapes),)
        return feature_shapes_by_id

    def _get_canonical_image_features(self) -> dict[str, dict]:
        self.dataset_canonical_image_sources: list[dict[str, str | None]] = []
        canonical_features = {}
        canonical_shapes: dict[str, list[tuple[int, ...]]] = {}

        for dataset in self._datasets:
            source_map = {}
            for canonical_key, aliases in CANONICAL_IMAGE_FEATURE_ALIASES.items():
                source_key = next((alias for alias in aliases if alias in dataset.meta.features), None)
                source_map[canonical_key] = source_key
                if source_key is not None:
                    if canonical_key not in canonical_features:
                        canonical_features[canonical_key] = deepcopy(dataset.meta.features[source_key])
                    canonical_shapes.setdefault(canonical_key, []).append(
                        tuple(dataset.meta.features[source_key]["shape"])
                    )
            self.dataset_canonical_image_sources.append(source_map)

        # Keep only canonical keys that exist in at least one dataset.
        for source_map in self.dataset_canonical_image_sources:
            for canonical_key in list(source_map):
                if canonical_key not in canonical_features:
                    del source_map[canonical_key]

        for canonical_key, shapes in canonical_shapes.items():
            canonical_features[canonical_key]["shape"] = self._get_max_image_feature_shape(shapes)

        self.canonical_image_item_shapes = {
            key: self._image_feature_shape_to_item_shape(feature["shape"])
            for key, feature in canonical_features.items()
        }
        return canonical_features

    def _get_max_image_feature_shape(self, shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        if not shapes:
            raise ValueError("Expected at least one image shape for canonical image feature.")
        if not all(len(shape) == 3 for shape in shapes):
            return tuple(max(dim_values) for dim_values in zip(*shapes, strict=True))

        if all(shape[-1] in {1, 3, 4} for shape in shapes):
            height = max(shape[0] for shape in shapes)
            width = max(shape[1] for shape in shapes)
            channels = max(shape[2] for shape in shapes)
            return (height, width, channels)

        return tuple(max(dim_values) for dim_values in zip(*shapes, strict=True))

    def _image_feature_shape_to_item_shape(self, feature_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(feature_shape) == 3 and feature_shape[-1] in {1, 3, 4}:
            return (feature_shape[-1], feature_shape[0], feature_shape[1])
        return feature_shape

    def _get_enabled_stats(self, dataset_index: int, raw_intersection_features: set[str]) -> dict[str, dict]:
        dataset = self._datasets[dataset_index]
        stats = {key: value for key, value in dataset.meta.stats.items() if key in raw_intersection_features}
        for canonical_key, source_key in self.dataset_canonical_image_sources[dataset_index].items():
            if source_key is not None and source_key in dataset.meta.stats:
                stats[canonical_key] = dataset.meta.stats[source_key]
        return stats

    def _get_padded_feature_shapes(self, feature_keys: set[str]) -> dict[str, tuple[int, ...]]:
        feature_shapes = {}
        for key in feature_keys:
            shapes = [tuple(dataset.meta.features[key]["shape"]) for dataset in self._datasets]
            dtypes = {dataset.meta.features[key]["dtype"] for dataset in self._datasets}
            if len(set(shapes)) > 1 and self._is_paddable_vector_feature(shapes, dtypes):
                feature_shapes[key] = (max(shape[0] for shape in shapes),)
        return feature_shapes

    def _is_paddable_vector_feature(self, shapes: list[tuple[int, ...]], dtypes: set[str]) -> bool:
        if not all(len(shape) == 1 for shape in shapes):
            return False
        if len(dtypes) != 1:
            return False
        try:
            dtype = np.dtype(next(iter(dtypes)))
        except TypeError:
            return False
        return np.issubdtype(dtype, np.number)

    def _validate_common_feature_shapes(self, feature_keys: set[str]) -> None:
        for key in sorted(feature_keys):
            signatures = {
                (
                    dataset.meta.features[key]["dtype"],
                    tuple(dataset.meta.features[key]["shape"]),
                )
                for dataset in self._datasets
            }
            if len(signatures) > 1:
                if self._can_normalize_feature_dtype(key, signatures):
                    logger.info(
                        "Normalizing multi-dataset feature '%s' to dtype %s from signatures %s.",
                        key,
                        NORMALIZED_FEATURE_DTYPES[key],
                        sorted(signatures),
                    )
                    continue
                if key in self.padded_feature_shapes:
                    dtypes = {signature[0] for signature in signatures}
                    if len(dtypes) == 1:
                        logger.info(
                            "Padding multi-dataset feature '%s' to shape %s from signatures %s.",
                            key,
                            self.padded_feature_shapes[key],
                            sorted(signatures),
                        )
                        continue
                raise ValueError(
                    f"Multi-dataset feature '{key}' has incompatible dtype/shape signatures: "
                    f"{sorted(signatures)}."
                )

    def _can_normalize_feature_dtype(self, key: str, signatures: set[tuple[str, tuple[int, ...]]]) -> bool:
        if key not in NORMALIZED_FEATURE_DTYPES:
            return False

        shapes = {shape for _, shape in signatures}
        dtypes = {dtype for dtype, _ in signatures}
        return len(shapes) == 1 and dtypes.issubset({"float32", "float64"})

    def _pad_stats(self, stats: dict[str, dict]) -> dict[str, dict]:
        padded_stats = deepcopy(stats)
        for key, target_shape in self.padded_feature_shapes.items():
            if key not in padded_stats:
                continue
            target_dim = target_shape[0]
            padded_stats[key] = {
                stat_name: self._pad_stat_value(stat_name, stat_value, target_dim)
                for stat_name, stat_value in padded_stats[key].items()
            }
        return padded_stats

    def _pad_stat_value(self, stat_name: str, stat_value, target_dim: int):
        if stat_name == "count":
            return stat_value

        value = np.asarray(stat_value)
        if value.ndim == 0 or value.shape[-1] >= target_dim:
            return stat_value

        pad_width = [(0, 0)] * value.ndim
        pad_width[-1] = (0, target_dim - value.shape[-1])
        pad_value = self._default_stat_pad_value(stat_name)
        return np.pad(value, pad_width, constant_values=pad_value)

    def _default_stat_pad_value(self, stat_name: str) -> float:
        if stat_name in {"std", "max", "q90", "q99"}:
            return 1.0
        if stat_name in {"min", "q01", "q10"}:
            return -1.0
        return 0.0

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        """Replace the transform for this dataset and its children."""
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self.image_transforms = image_transforms
        for dataset in getattr(self, "_datasets", []):
            dataset.set_image_transforms(self.image_transforms)

    def clear_image_transforms(self) -> None:
        """Remove the transform from this dataset and its children."""
        self.set_image_transforms(None)

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.fps

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return len(self._datasets[0].meta.video_keys) > 0

    @property
    def features(self) -> datasets.Features:
        return get_hf_features_from_features(self.meta.features)

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image | VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        item["normalization_id"] = torch.tensor(self.normalization_ids[dataset_idx])
        if self.embodiment_ids is not None:
            item["embodiment_id"] = torch.tensor(self.embodiment_ids[dataset_idx])
        self._add_canonical_image_items(item, dataset_idx)
        for key, target_shape in self.padded_feature_shapes.items():
            if key in item:
                item[key], item[f"{key}_dim_is_pad"] = self._pad_item_feature(item[key], target_shape[0])
        for key in NORMALIZED_FEATURE_DTYPES:
            if key in item:
                item[key] = item[key].to(dtype=torch.float32)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def _add_canonical_image_items(self, item: dict[str, torch.Tensor], dataset_idx: int) -> None:
        for canonical_key, source_key in self.dataset_canonical_image_sources[dataset_idx].items():
            if source_key is not None and source_key in item:
                item[canonical_key] = self._prepare_canonical_image_item(item[source_key], canonical_key)
                item[f"{canonical_key}_is_pad"] = torch.tensor(False)
            else:
                item[canonical_key] = self._make_empty_image_item(item, canonical_key)
                item[f"{canonical_key}_is_pad"] = torch.tensor(True)

    def _prepare_canonical_image_item(self, image: torch.Tensor, canonical_key: str) -> torch.Tensor:
        image = image.contiguous().clone()
        target_shape = self.canonical_image_item_shapes[canonical_key]
        if tuple(image.shape[-len(target_shape) :]) == target_shape:
            return image
        return self._pad_or_crop_last_dims(image, target_shape)

    def _pad_or_crop_last_dims(self, value: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
        slices = [slice(None)] * value.ndim
        for axis, target_dim in zip(
            range(value.ndim - len(target_shape), value.ndim), target_shape, strict=True
        ):
            slices[axis] = slice(0, min(value.shape[axis], target_dim))
        value = value[tuple(slices)]

        pad = []
        for current_dim, target_dim in zip(
            reversed(value.shape[-len(target_shape) :]), reversed(target_shape), strict=True
        ):
            pad.extend([0, max(0, target_dim - current_dim)])
        if any(pad):
            value = F.pad(value, pad)
        return value.contiguous()

    def _make_empty_image_item(self, item: dict[str, torch.Tensor], canonical_key: str) -> torch.Tensor:
        dtype = next(
            (
                value.dtype
                for key, value in item.items()
                if key in self.meta.camera_keys and isinstance(value, torch.Tensor)
            ),
            torch.uint8,
        )
        return torch.zeros(self.canonical_image_item_shapes[canonical_key], dtype=dtype)

    def _pad_item_feature(self, value: torch.Tensor, target_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        current_dim = value.shape[-1]
        dim_is_pad = torch.zeros(target_dim, dtype=torch.bool)
        if current_dim >= target_dim:
            return value[..., :target_dim], dim_is_pad

        padded = F.pad(value, (0, target_dim - current_dim))
        dim_is_pad[current_dim:] = True
        return padded, dim_is_pad

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
