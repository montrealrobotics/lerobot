#!/usr/bin/env python
"""Phase-0 smoke test: GR00T N1.7 on our RoboCasa SFT datasets, before any training script exists.

For one task and one chunk size:
  1. builds GrootPolicy (nvidia/GR00T-N1.7-3B, embodiment 'new_embodiment') with our dataset's
     features, and the N1.7 pre/post processors fed our dataset stats;
  2. runs a training forward + backward on real dataset samples (loss must be finite);
  3. prints trainable / total parameters per top-level module, to define the SFT arms;
  4. noise sensitivity for DSRL: does noise in the PADDED action dims (>= the dataset's action
     dim) change the predicted real-dim actions? Same observation, same real-dim noise.

    python examples/training/groot_phase0_smoke.py --task lamp --chunk-size 40
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

DATASETS = {
    "lamp": "akuramshin/robocasa_lightbulbscrew_dex_filtered",
    "coffee": "akuramshin/robocasa_coffeepressbutton_dex_augstyle",
}
CAMERAS = ("observation.images.robot0_eye_in_hand", "observation.images.robot0_agentview_left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=DATASETS, required=True)
    ap.add_argument("--chunk-size", type=int, default=40)
    ap.add_argument("--n-action-steps", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = "cuda"
    report: dict = {"task": args.task, "chunk_size": args.chunk_size}

    meta = LeRobotDatasetMetadata(DATASETS[args.task])
    state_dim = meta.features[OBS_STATE]["shape"][0]
    action_dim = meta.features[ACTION]["shape"][0]
    print(f"dataset {meta.repo_id}: fps {meta.fps}, state {state_dim}, action {action_dim}, cameras {meta.camera_keys}")
    print("action stats keys:", sorted(meta.stats[ACTION]))
    report.update(state_dim=state_dim, action_dim=action_dim, action_stat_keys=sorted(meta.stats[ACTION]))

    def visual(key):
        shape = tuple(meta.features[key]["shape"])
        return PolicyFeature(type=FeatureType.VISUAL, shape=(shape[2], shape[0], shape[1]) if shape[-1] == 3 else shape)

    config = GrootConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
                        **{k: visual(k) for k in CAMERAS}},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        embodiment_tag="new_embodiment",
        device=dev,
    )
    policy = GrootPolicy(config).to(dev)
    pre, post = make_groot_pre_post_processors(config=policy.config, dataset_stats=meta.stats, dataset_meta=meta)
    print("preprocessor steps:", [type(s).__name__ for s in pre.steps])
    print("postprocessor steps:", [type(s).__name__ for s in post.steps])
    print("action_delta_indices:", policy.config.action_delta_indices)

    # Parameter breakdown by top-level module path (depth 3).
    groups = defaultdict(lambda: [0, 0])
    for name, p in policy.named_parameters():
        key = ".".join(name.split(".")[:4])
        groups[key][0] += p.numel()
        groups[key][1] += p.numel() if p.requires_grad else 0
    total = sum(v[0] for v in groups.values())
    print(f"\nparameters: total {total / 1e9:.3f}B, trainable {sum(v[1] for v in groups.values()) / 1e9:.3f}B")
    for k, (n, t) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
        if n > 1e6:
            print(f"  {k:70s} {n / 1e6:9.1f}M  trainable {t / 1e6:9.1f}M")
    report["params"] = {k: v for k, v in groups.items() if v[0] > 1e6}

    # Real samples, with the action chunk the config asks for.
    fps = meta.fps
    ds = LeRobotDataset(DATASETS[args.task], video_backend="pyav",
                        delta_timestamps={ACTION: [i / fps for i in policy.config.action_delta_indices]})
    idx = torch.linspace(0, len(ds) - 1, args.batch).long().tolist()
    samples = [ds[i] for i in idx]
    batch = {k: (torch.stack([s[k] for s in samples]).to(dev) if isinstance(samples[0][k], torch.Tensor) else [s[k] for s in samples])
             for k in samples[0] if k in (*CAMERAS, OBS_STATE, ACTION, "task")}
    print("batch:", {k: tuple(v.shape) if isinstance(v, torch.Tensor) else v[:1] for k, v in batch.items()})

    policy.train()
    processed = pre(deepcopy(batch))
    loss, metrics = policy.forward(processed)
    loss.backward()
    grads = sum(p.grad is not None for p in policy.parameters() if p.requires_grad)
    print(f"\ntrain forward: loss {loss.item():.4f} (finite {torch.isfinite(loss).item()}), params with grad {grads}")
    report["loss"] = loss.item()
    policy.zero_grad(set_to_none=True)

    # Noise sensitivity: one observation, real-dim noise fixed, padded dims varied.
    policy.eval()
    obs = {k: (v[:1] if isinstance(v, torch.Tensor) else v[:1]) for k, v in batch.items() if k != ACTION}
    obs_p = pre(deepcopy(obs))
    model = policy._groot_model
    horizon = model.action_head.action_horizon if hasattr(model, "action_head") else args.chunk_size
    max_dim = policy.config.max_action_dim
    print(f"noise shape: (1, {horizon}, {max_dim})")
    g = torch.Generator(device=dev).manual_seed(0)
    base = torch.randn(1, horizon, max_dim, device=dev, generator=g)
    variants = {"same": base.clone(), "pad_resampled": base.clone(), "pad_zero": base.clone(), "real_resampled": base.clone()}
    variants["pad_resampled"][..., action_dim:] = torch.randn(1, horizon, max_dim - action_dim, device=dev, generator=g)
    variants["pad_zero"][..., action_dim:] = 0
    variants["real_resampled"][..., :action_dim] = torch.randn(1, horizon, action_dim, device=dev, generator=g)
    outs = {}
    with torch.no_grad():
        for name, n in variants.items():
            outs[name] = policy.predict_action_chunk(deepcopy(obs_p), noise=n)[..., :action_dim].float()
    ref = outs["same"]
    print("\npredict_action_chunk output:", tuple(ref.shape))
    sens = {}
    for name, o in outs.items():
        d = (o - ref).abs()
        sens[name] = {"mean_abs_diff": d.mean().item(), "max_abs_diff": d.max().item()}
        print(f"  {name:15s} mean |Δ| {d.mean().item():.5f}   max |Δ| {d.max().item():.5f}")
    report["noise_sensitivity"] = sens
    real_scale = sens["real_resampled"]["mean_abs_diff"]
    if real_scale > 0:
        print(f"  padded-noise effect / real-noise effect: {sens['pad_resampled']['mean_abs_diff'] / real_scale:.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print("wrote", args.out)
    print("PHASE0 OK")


if __name__ == "__main__":
    sys.exit(main())
