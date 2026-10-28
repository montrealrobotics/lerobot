#!/usr/bin/env python
"""Build a GR00T "SFT step 0" checkpoint: base nvidia/GR00T-N1.7-3B weights with an SFT run's
config and processors (cameras, quantile stats, normalization overrides).

Every arm starts fine-tuning from the same base weights, so one step-0 checkpoint per task serves
all arms. The policy is built exactly as training builds it before the first step (GrootPolicy from
`base_model_path`, our features and settings), then saved; processors and train_config.json are
copied from one of the run's checkpoints (they do not change across rungs). The trainable flags in
the copied config do not matter for evaluation.

    python examples/training/make_groot_step0_checkpoint.py <sft_run_dir> <out_run_dir>
writes <out_run_dir>/checkpoints/000000/pretrained_model/
"""

from __future__ import annotations

import shutil
import sys
import typing
from pathlib import Path

import draccus
import torch

draccus.decode.register(typing.Literal, lambda raw_value, path=(): raw_value)
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.groot.modeling_groot import GrootPolicy  # noqa: E402
from lerobot.utils.constants import CHECKPOINTS_DIR, PRETRAINED_MODEL_DIR  # noqa: E402


def main() -> None:
    run, out_run = Path(sys.argv[1]), Path(sys.argv[2])
    rungs = sorted(d for d in (run / CHECKPOINTS_DIR).iterdir() if d.name.isdigit())
    src = rungs[0] / PRETRAINED_MODEL_DIR
    cfg = PreTrainedConfig.from_pretrained(src)
    assert cfg.type == "groot", cfg.type
    cfg.pretrained_path = None
    if getattr(cfg, "use_peft", False):
        cfg.use_peft = False  # step 0 has no adapter: base weights only
    policy = GrootPolicy(cfg)  # loads the base model from cfg.base_model_path, as training does
    print(f"base model: {cfg.base_model_path}; params {sum(p.numel() for p in policy.parameters()) / 1e9:.3f}B")

    out = out_run / CHECKPOINTS_DIR / "000000" / PRETRAINED_MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    for f in src.iterdir():
        if f.name.startswith("policy_") or f.name == "train_config.json":
            shutil.copy2(f, out / f.name)
    print("wrote", out, sorted(p.name for p in out.iterdir()))

    # Sanity: the saved weights must be the base model's, untouched by fine-tuning.
    trained = rungs[-1] / PRETRAINED_MODEL_DIR / "model.safetensors"
    from safetensors import safe_open

    with safe_open(out / "model.safetensors", "pt") as a, safe_open(trained, "pt") as b:
        key = next(k for k in a.keys() if "action_head.model.transformer_blocks.0.attn1.to_q.weight" in k)
        diff = (a.get_tensor(key).float() - b.get_tensor(key).float()).abs().max().item()
    print(f"step-0 vs {rungs[-1].name} on {key}: max |delta| {diff:.3e} (should be > 0: fine-tuning moved it)")
    assert diff > 0
    print("STEP0 OK")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
