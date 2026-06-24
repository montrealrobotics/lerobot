#!/usr/bin/env python

"""
Convert DexMimicGen HDF5 datasets to LeRobot format.

Example:
    python examples/port_datasets/port_dexmimicgen.py \
        --dataset-path /home/mila/a/artur.kuramshin/scratch/single_arm_clean.hdf5 \
        --repo-id your_hf_username/dexmimicgen_single_arm_drawer_cleanup
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import encode_video_frames
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_IMAGES, OBS_STATE

DEFAULT_GRIPPER_INDICES = (0, 2, 4, 6, 8, 11)
DEFAULT_TASK = "open and close the drawer"


def _natural_demo_key(name: str) -> tuple[str, int | str]:
    prefix, _, suffix = name.rpartition("_")
    return (prefix, int(suffix)) if suffix.isdigit() else ("", name)


def _read_json_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def _episode_group(file: h5py.File, ep_name: str) -> h5py.Group:
    return file.get("data", file)[ep_name]


def discover_dataset_properties(dataset_paths: list[str]) -> dict[str, Any]:
    image_shapes: dict[str, tuple[int, ...]] = {}
    obs_shapes: dict[str, tuple[int, ...]] = {}
    obs_dtypes: dict[str, str] = {}
    has_pca_actions = False
    fps = 20

    for path in dataset_paths:
        with h5py.File(Path(path).expanduser(), "r") as f:
            data = f.get("data", f)
            if "env_args" in data.attrs:
                env_args = _read_json_attr(data.attrs["env_args"])
                fps = int(env_args.get("env_kwargs", {}).get("control_freq", env_args.get("control_freq", fps)))

            demos = sorted(data.keys(), key=_natural_demo_key)
            if not demos:
                raise ValueError(f"No episodes found in {path}")

            for ep_name in demos[: min(10, len(demos))]:
                ep = _episode_group(f, ep_name)
                obs = ep["obs"]
                for key, value in obs.items():
                    shape = tuple(value.shape[1:])
                    if key.endswith("_image"):
                        image_shapes[key.removesuffix("_image")] = shape
                    else:
                        obs_shapes[key] = shape
                        obs_dtypes[key] = str(value.dtype)
                has_pca_actions = has_pca_actions or ("pca_actions" in ep)

            first_ep = _episode_group(f, demos[0])
            action_shape = tuple(first_ep["actions"].shape[1:])
            action_dtype = str(first_ep["actions"].dtype)
            pca_action_shape = tuple(first_ep["pca_actions"].shape[1:]) if "pca_actions" in first_ep else None
            pca_action_dtype = str(first_ep["pca_actions"].dtype) if "pca_actions" in first_ep else None

    return {
        "image_shapes": image_shapes,
        "obs_shapes": obs_shapes,
        "obs_dtypes": obs_dtypes,
        "action_shape": action_shape,
        "action_dtype": action_dtype,
        "has_pca_actions": has_pca_actions,
        "pca_action_shape": pca_action_shape,
        "pca_action_dtype": pca_action_dtype,
        "fps": fps,
    }


def create_lerobot_features(properties: dict[str, Any]) -> dict[str, Any]:
    features: dict[str, dict[str, Any]] = {}

    for camera_name, shape in sorted(properties["image_shapes"].items()):
        features[f"{OBS_IMAGES}.{camera_name}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }

    features[OBS_STATE] = {
        "dtype": "float32",
        "shape": (13,),
        "names": {
            "axes": [
                "joint_0",
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "gripper_0",
                "gripper_1",
                "gripper_2",
                "gripper_3",
                "gripper_4",
                "gripper_5",
            ]
        },
    }
    features[f"{OBS_STATE}.eef"] = {
        "dtype": "float32",
        "shape": (13,),
        "names": ["eef_pos_quat_gripper"],
    }
    features[f"{OBS_STATE}.joint"] = {
        "dtype": "float32",
        "shape": (13,),
        "names": ["joint_pos_gripper"],
    }
    features[f"{OBS_STATE}.eef_euler"] = {
        "dtype": "float32",
        "shape": (3,),
        "names": ["eef_euler_roll", "eef_euler_pitch", "eef_euler_yaw"],
    }
    features[f"{OBS_STATE}.eef_angle_axis"] = {
        "dtype": "float32",
        "shape": (3,),
        "names": ["eef_angle_axis_0", "eef_angle_axis_1", "eef_angle_axis_2"],
    }
    features[ACTION] = {
        "dtype": "float32",
        "shape": properties["action_shape"],
        "names": [ACTION],
    }
    if properties["has_pca_actions"]:
        features[f"{ACTION}.pca"] = {
            "dtype": "float32",
            "shape": properties["pca_action_shape"],
            "names": ["pca_action"],
        }
    return features


def _task_from_episode(ep: h5py.Group, default_task: str) -> str:
    if "task_description" in ep.attrs:
        value = ep.attrs["task_description"]
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    if "ep_meta" in ep.attrs:
        ep_meta = _read_json_attr(ep.attrs["ep_meta"])
        if isinstance(ep_meta, dict) and ep_meta.get("lang"):
            return str(ep_meta["lang"])
    return default_task


def _selected_gripper(obs: h5py.Group, key: str, t: int, gripper_indices: tuple[int, ...]) -> np.ndarray:
    return np.asarray(obs[key][t], dtype=np.float32)[list(gripper_indices)]


def add_episode(
    dataset: LeRobotDataset,
    ep: h5py.Group,
    properties: dict[str, Any],
    *,
    task: str,
    gripper_indices: tuple[int, ...],
) -> dict[str, Any]:
    obs = ep["obs"]
    actions = ep["actions"]
    num_steps = actions.shape[0]

    for t in range(num_steps):
        gripper = _selected_gripper(obs, "robot0_gripper_qpos", t, gripper_indices)
        joint_state = np.concatenate([np.asarray(obs["robot0_joint_pos"][t], dtype=np.float32), gripper])
        eef_pos = np.asarray(obs["robot0_eef_pos"][t], dtype=np.float32)
        eef_quat = np.asarray(obs["robot0_eef_quat"][t], dtype=np.float32)
        eef_state = np.concatenate([eef_pos, eef_quat, gripper])

        # Robosuite stores quaternions as [w, x, y, z]; scipy expects [x, y, z, w].
        rot = Rotation.from_quat(np.roll(eef_quat, -1))
        eef_euler = rot.as_euler("xyz", degrees=False).astype(np.float32)
        eef_angle_axis = rot.as_rotvec().astype(np.float32)

        frame = {
            OBS_STATE: joint_state,
            f"{OBS_STATE}.joint": joint_state,
            f"{OBS_STATE}.eef": eef_state,
            f"{OBS_STATE}.eef_euler": eef_euler,
            f"{OBS_STATE}.eef_angle_axis": eef_angle_axis,
            ACTION: np.asarray(actions[t], dtype=np.float32),
            "task": task,
        }
        if properties["has_pca_actions"]:
            frame[f"{ACTION}.pca"] = np.asarray(ep["pca_actions"][t], dtype=np.float32)

        for camera_name in properties["image_shapes"]:
            frame[f"{OBS_IMAGES}.{camera_name}"] = obs[f"{camera_name}_image"][t]

        dataset.add_frame(frame)

    dataset.save_episode()
    ep_meta = {}
    if "ep_meta" in ep.attrs:
        ep_meta = _read_json_attr(ep.attrs["ep_meta"])
    return {
        "num_samples": int(num_steps),
        "task": task,
        "ep_meta": ep_meta,
    }


def convert_dexmimicgen_to_lerobot(
    dataset_paths: list[str],
    repo_id: str,
    *,
    task: str = DEFAULT_TASK,
    robot_type: str = "PandaDexRH",
    push_to_hub: bool = False,
    private: bool = True,
    n_episodes: int = -1,
    gripper_indices: tuple[int, ...] = DEFAULT_GRIPPER_INDICES,
    video_codec: str | None = None,
    overwrite: bool = False,
) -> None:
    dataset_paths = [str(Path(path).expanduser()) for path in dataset_paths]
    properties = discover_dataset_properties(dataset_paths)

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    features = create_lerobot_features(properties)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=properties["fps"],
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    if video_codec is not None:
        def custom_encode_video(video_key: str, episode_index: int, dataset_ref=dataset):
            temp_path = Path(tempfile.mkdtemp(dir=dataset_ref.root)) / f"{video_key}_{episode_index:03d}.mp4"
            img_dir = dataset_ref._get_image_file_dir(episode_index, video_key)
            encode_video_frames(
                img_dir,
                temp_path,
                dataset_ref.fps,
                vcodec=video_codec,
                pix_fmt="yuv420p",
                g=2,
                crf=30,
                overwrite=True,
            )
            shutil.rmtree(img_dir)
            return temp_path

        dataset._encode_temporary_episode_video = custom_encode_video

    episode_metadata = []
    converted = 0
    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, "r") as f:
            data = f.get("data", f)
            for ep_name in sorted(data.keys(), key=_natural_demo_key):
                ep = _episode_group(f, ep_name)
                episode_task = _task_from_episode(ep, task)
                metadata = add_episode(
                    dataset,
                    ep,
                    properties,
                    task=episode_task,
                    gripper_indices=gripper_indices,
                )
                metadata.update({"source_path": dataset_path, "source_episode": ep_name})
                episode_metadata.append(metadata)
                converted += 1
                if n_episodes > 0 and converted >= n_episodes:
                    break
        if n_episodes > 0 and converted >= n_episodes:
            break

    dataset.finalize()
    meta_dir = output_path / "meta" / "episodes"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "dexmimicgen_episode_metadata.json", "w") as f:
        json.dump(episode_metadata, f, indent=2)

    if push_to_hub:
        dataset.push_to_hub(
            tags=["dexmimicgen", "robosuite", "panda"],
            private=private,
            push_videos=True,
            license="apache-2.0",
        )

    print(f"Converted {converted} episodes to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        nargs="+",
        default=["/home/mila/a/artur.kuramshin/scratch/single_arm_clean.hdf5"],
        help="One or more DexMimicGen HDF5 files.",
    )
    parser.add_argument("--repo-id", default="dexmimicgen_single_arm_drawer_cleanup")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--robot-type", default="PandaDexRH")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--public", action="store_true", help="Push as a public dataset instead of private.")
    parser.add_argument("--n-episodes", type=int, default=-1)
    parser.add_argument(
        "--gripper-indices",
        type=int,
        nargs="+",
        default=list(DEFAULT_GRIPPER_INDICES),
        help="Indices to select from robot0_gripper_qpos.",
    )
    parser.add_argument("--video-codec", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dexmimicgen_to_lerobot(
        dataset_paths=args.dataset_path,
        repo_id=args.repo_id,
        task=args.task,
        robot_type=args.robot_type,
        push_to_hub=args.push_to_hub,
        private=not args.public,
        n_episodes=args.n_episodes,
        gripper_indices=tuple(args.gripper_indices),
        video_codec=args.video_codec,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
