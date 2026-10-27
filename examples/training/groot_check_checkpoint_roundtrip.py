#!/usr/bin/env python
"""Checkpoint round-trip for a GR00T study arm: save the way lerobot_train does, reload the way the
eval and DSRL scripts do, and require identical actions.

  1. build the arm's policy (+ LoRA wrapper for arm3), optimizer and processors, as lerobot_train;
  2. perturb weights so a reload that silently fell back to base weights would show: a LoRA B
     matrix (zero at init, so an un-loaded adapter is otherwise invisible), a full-FT island
     (vision tower / action encoder), and a DiT weight;
  3. predict an action chunk for real dataset observations with a fixed noise tensor;
  4. save_checkpoint(...) exactly as lerobot_train; then PreTrainedConfig.from_pretrained +
     make_policy + make_pre_post_processors(pretrained_path=...), as eval_pi05_dual_noise / DSRL;
  5. re-run 3 on the reloaded policy AND reloaded processors and compare.

    python examples/training/groot_check_checkpoint_roundtrip.py --experiment arm3_lora_r32 --task lamp
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import typing
from copy import deepcopy
from pathlib import Path

import draccus
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_groot_casa_experiments import EXPERIMENTS, make_config  # noqa: E402

from lerobot.common.train_utils import save_checkpoint  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.optim.factory import make_optimizer_and_scheduler  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE, PRETRAINED_MODEL_DIR  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

draccus.decode.register(typing.Literal, lambda raw_value, path=(): raw_value)
LEFT = "observation.images.robot0_agentview_left"
WRIST = "observation.images.robot0_eye_in_hand"


@torch.no_grad()
def predict(policy, pre, obs, noise):
    policy.eval()
    processed = pre(deepcopy(obs))
    return policy.predict_action_chunk(processed, noise=noise).float().cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="arm3_lora_r32")
    ap.add_argument("--task", default="lamp")
    ap.add_argument("--keep", action="store_true", help="Keep the checkpoint directory.")
    args = ap.parse_args()
    register_third_party_plugins()
    torch.manual_seed(0)

    cfg = make_config(args.experiment, args.task, seed=0)
    cfg.validate()
    meta = LeRobotDatasetMetadata(cfg.dataset.repo_id)
    policy = make_policy(cfg=cfg.policy, ds_meta=meta)
    if cfg.peft is not None:
        from dataclasses import asdict

        policy = policy.wrap_with_peft(peft_cli_overrides=asdict(cfg.peft))
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    pre, post = make_pre_post_processors(policy_cfg=cfg.policy, dataset_stats=meta.stats)
    device = next(policy.parameters()).device

    # 2. Perturb trainable weights (one per kind present in this arm).
    perturbed = []
    with torch.no_grad():
        for name, p in policy.named_parameters():
            if not p.requires_grad:
                continue
            kind = ("lora_B" if "lora_B" in name else "visual" if ".visual." in name
                    else "action_encoder" if "action_encoder" in name else "dit" if "transformer_blocks" in name else None)
            if kind and kind not in {k for k, _ in perturbed}:
                p.add_(0.05 * torch.randn_like(p))
                perturbed.append((kind, name))
    print("perturbed:", *[f"\n  {k:15s} {n}" for k, n in perturbed])

    # 3. Fixed observations (left camera + wrist, as DSRL supplies) and fixed noise.
    ds = LeRobotDataset(cfg.dataset.repo_id, video_backend="pyav")
    samples = [ds[i] for i in (0, len(ds) // 2)]
    obs = {k: torch.stack([s[k] for s in samples]).to(device) for k in (LEFT, WRIST, OBS_STATE)}
    obs["task"] = [s["task"] for s in samples]
    noise = torch.randn(2, 40, cfg.policy.max_action_dim, device=device)
    before = predict(policy, pre, obs, noise)

    # 4. Save exactly as lerobot_train, reload as the eval/DSRL scripts.
    tmp = Path(tempfile.mkdtemp(prefix=f"groot_roundtrip_{args.experiment}_", dir="/network/scratch/a/artur.kuramshin/tmp"))
    ckpt = tmp / "000001"
    save_checkpoint(ckpt, 1, cfg, policy, optimizer, scheduler, pre, post)
    pretrained = ckpt / PRETRAINED_MODEL_DIR
    print("saved:", sorted(p.name for p in pretrained.iterdir()))
    del policy
    torch.cuda.empty_cache()

    cfg2 = PreTrainedConfig.from_pretrained(pretrained)
    cfg2.pretrained_path = pretrained
    print(f"reload config: type={cfg2.type} use_peft={getattr(cfg2, 'use_peft', None)} "
          f"normalization_bounds={cfg2.normalization_bounds} state_dropout={cfg2.state_dropout_prob}")
    policy2 = make_policy(cfg=cfg2, ds_meta=meta)
    pre2, post2 = make_pre_post_processors(policy_cfg=cfg2, pretrained_path=pretrained)
    print("reloaded preprocessor:", [type(s).__name__ for s in pre2.steps])
    after = predict(policy2, pre2, obs, noise)

    # 5. Compare.
    diff = (after - before).abs()
    print(f"actions {tuple(before.shape)}: max |Δ| {diff.max().item():.3e}, mean |Δ| {diff.mean().item():.3e}")
    # Real actions too (post-processor), to cover un-normalization with the saved stats.
    a1 = post(before[:, :, : meta.features[ACTION]['shape'][0]].clone())
    a2 = post2(after[:, :, : meta.features[ACTION]['shape'][0]].clone())
    print(f"unnormalized actions: max |Δ| {(torch.as_tensor(a1) - torch.as_tensor(a2)).abs().max().item():.3e}")
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    assert diff.max().item() < 1e-4, "reloaded policy disagrees with the saved one"
    print(f"ROUNDTRIP OK ({args.experiment})")


if __name__ == "__main__":
    main()
