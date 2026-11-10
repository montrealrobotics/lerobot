#!/usr/bin/env python
"""Build a pi05 "SFT step 0" checkpoint and link it into SFT runs as checkpoints/000000.

    python examples/training/make_pi05_step0_checkpoint.py <out_dir> <sft_run_dir> [<sft_run_dir> ...]

writes <out_dir>/000000/pretrained_model/ and links <sft_run_dir>/checkpoints/000000 to it.
"""

from __future__ import annotations

import filecmp
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import CHECKPOINTS_DIR, PRETRAINED_MODEL_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pi05_casa_experiments import PI05_BASE_MODEL  # noqa: E402

STEP0 = "000000"


def rungs(run: Path) -> list[Path]:
    return sorted(d for d in (run / CHECKPOINTS_DIR).iterdir() if d.name.isdigit() and d.name != STEP0)


def main() -> None:
    out_root, runs = Path(sys.argv[1]), [Path(r) for r in sys.argv[2:]]
    src = rungs(runs[0])[0] / PRETRAINED_MODEL_DIR
    processor_files = [f.name for f in src.iterdir() if f.name.startswith("policy_")]
    for run in runs[1:]:
        other = rungs(run)[0] / PRETRAINED_MODEL_DIR
        _, mismatch, errors = filecmp.cmpfiles(src, other, processor_files, shallow=False)
        assert not mismatch and not errors, f"{run} has different processors: {mismatch + errors}"

    cfg = PreTrainedConfig.from_pretrained(src)
    assert cfg.type == "pi05", cfg.type
    policy = PI05Policy.from_pretrained(PI05_BASE_MODEL, config=cfg)

    out = out_root / STEP0 / PRETRAINED_MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    for name in [*processor_files, "train_config.json"]:
        shutil.copy2(src / name, out / name)
    print("wrote", out, sorted(p.name for p in out.iterdir()))

    with safe_open(out / "model.safetensors", "pt") as base:
        key = next(k for k in base.keys() if k.endswith("action_in_proj.weight"))
        base_weight = base.get_tensor(key).float()
    for run in runs:
        with safe_open(rungs(run)[-1] / PRETRAINED_MODEL_DIR / "model.safetensors", "pt") as trained:
            delta = (base_weight - trained.get_tensor(key).float()).abs().max().item()
        print(f"step 0 vs {run.name} last rung, {key}: max |delta| {delta:.3e}")
        assert delta > 0, "step 0 equals a fine-tuned rung"

        link = run / CHECKPOINTS_DIR / STEP0
        if link.is_symlink():
            link.unlink()
        link.symlink_to(out.parent.resolve(), target_is_directory=True)
        print("linked", link)
    print("STEP0 OK")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
