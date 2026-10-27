#!/usr/bin/env python
"""Check the GR00T study preprocessor's camera handling on real samples (no model needed).

Runs dataset samples -- which carry every recorded camera -- through the preprocessor up to and
including the pack step, and checks:
  * the packed `video` holds exactly the configured views (wrist + one external), never the extra
    dataset cameras;
  * the external view is the left camera for some samples and the right for others (pi05's
    random external-camera swap), and the wrist view is untouched;
  * a fixed observation with only the left camera (the DSRL view) passes through unchanged.

    python examples/training/groot_check_views.py --task coffee
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_groot_casa_experiments import make_config  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.groot.processor_groot import GrootN17PackInputsStep  # noqa: E402
from lerobot.processor import batch_to_transition  # noqa: E402
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

LEFT = "observation.images.robot0_agentview_left"
RIGHT = "observation.images.robot0_agentview_right"
WRIST = "observation.images.robot0_eye_in_hand"


def run_until_pack(pre, batch):
    transition = batch_to_transition(batch)
    for step in pre.steps:
        transition = step(transition)
        if isinstance(step, GrootN17PackInputsStep):
            return transition
    raise RuntimeError("no pack step in the preprocessor")


def to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    """(C,H,W) float [0,1] -> (H,W,C) uint8, as the pack step converts it."""
    return (img.permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="coffee")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    register_third_party_plugins()
    cfg = make_config("arm0_expert_only", args.task, seed=0)
    meta = LeRobotDatasetMetadata(cfg.dataset.repo_id)
    pre, _ = make_pre_post_processors(policy_cfg=cfg.policy, dataset_stats=meta.stats)
    pack = next(s for s in pre.steps if isinstance(s, GrootN17PackInputsStep))
    print("preprocessor:", [type(s).__name__ for s in pre.steps])
    print("pack.video_modality_keys:", pack.video_modality_keys, "| dataset cameras:", meta.camera_keys)

    ds = LeRobotDataset(cfg.dataset.repo_id, video_backend="pyav")
    idx = np.linspace(0, len(ds) - 1, args.n).astype(int)
    samples = [ds[int(i)] for i in idx]
    batch = {k: torch.stack([s[k] for s in samples]) for k in (LEFT, RIGHT, WRIST, "observation.state")}
    batch["task"] = [s["task"] for s in samples]

    out = run_until_pack(pre, batch)
    video = out[TransitionKey.OBSERVATION]["video"]  # (B, T, V, H, W, C) uint8
    video = np.asarray(video)
    print("packed video shape (B, T, V, H, W, C):", video.shape)
    assert video.shape[2] == 2, f"expected 2 views, got {video.shape[2]}"

    # Views are in sorted key order: robot0_agentview_left (external slot), robot0_eye_in_hand.
    picked = {"left": 0, "right": 0, "neither": 0}
    wrist_ok = 0
    for b in range(len(samples)):
        ext = video[b, 0, 0]
        left, right = to_uint8_hwc(batch[LEFT][b]), to_uint8_hwc(batch[RIGHT][b])
        d_left, d_right = np.abs(ext.astype(int) - left).mean(), np.abs(ext.astype(int) - right).mean()
        picked["left" if d_left < 0.5 else "right" if d_right < 0.5 else "neither"] += 1
        wrist_ok += int(np.abs(video[b, 0, 1].astype(int) - to_uint8_hwc(batch[WRIST][b])).mean() < 0.5)
    print(f"external view source over {len(samples)} samples: {picked}; wrist view intact: {wrist_ok}/{len(samples)}")
    assert picked["neither"] == 0 and picked["left"] > 0 and picked["right"] > 0, picked
    assert wrist_ok == len(samples)

    # DSRL view: only the left camera supplied -> the swap is a no-op.
    one = {LEFT: batch[LEFT][:4], WRIST: batch[WRIST][:4], "observation.state": batch["observation.state"][:4],
           "task": batch["task"][:4]}
    v1 = np.asarray(run_until_pack(pre, one)[TransitionKey.OBSERVATION]["video"])
    same = all(np.abs(v1[b, 0, 0].astype(int) - to_uint8_hwc(one[LEFT][b])).mean() < 0.5 for b in range(4))
    print("left-camera-only observation passes through unchanged:", same)
    assert same and v1.shape[2] == 2
    print("VIEWS OK")


if __name__ == "__main__":
    main()
