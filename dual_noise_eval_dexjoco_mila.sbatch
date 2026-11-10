#!/bin/bash
#   sbatch --job-name=dn_arm1_water_plant dual_noise_eval_dexjoco_mila.sbatch <run-dir> water_plant
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/network/scratch/a/artur.kuramshin/lerobot/slurm_logs/%x-%j.out

set -euo pipefail

RUN_DIR=${1:?usage: sbatch dual_noise_eval_dexjoco_mila.sbatch <run-dir> <task> [extra args...]}
TASK=${2:?usage: sbatch dual_noise_eval_dexjoco_mila.sbatch <run-dir> <task> [extra args...]}
shift 2

cd /network/scratch/a/artur.kuramshin/lerobot-sft-dexjoco
source dexjoco_env.sh

echo "=== preflight ($(date)) ==="
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) task=$TASK run_dir=$RUN_DIR extra=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - "$RUN_DIR" "$TASK" "$@" <<'PREFLIGHT'
import json, netrc, os, sys
from pathlib import Path

run_dir, task = Path(sys.argv[1]), sys.argv[2]
fail = []

from huggingface_hub import get_token
if not get_token():
    fail.append(f"no HF token visible (HF_HOME={os.environ.get('HF_HOME')})")

has_netrc = False
try:
    has_netrc = "api.wandb.ai" in netrc.netrc(os.path.expanduser("~/.netrc")).hosts
except Exception:
    pass
if "--no-wandb" not in sys.argv and not os.environ.get("WANDB_API_KEY") and not has_netrc:
    fail.append("no wandb credentials (set WANDB_API_KEY, run `wandb login`, or pass --no-wandb)")

ckpts = run_dir / "checkpoints"
steps = sorted(int(d.name) for d in ckpts.iterdir() if d.name.isdigit()) if ckpts.is_dir() else []
print(f"checkpoints: {steps}")
if not steps:
    fail.append(f"no numbered checkpoints under {ckpts}")
else:
    sys.path.insert(0, "examples/training")
    from train_pi05_casa_experiments import TASKS
    tc = ckpts / f"{steps[-1]:06d}" / "pretrained_model" / "train_config.json"
    trained_on = json.loads(tc.read_text())["env"]["task"]
    if trained_on != TASKS[task].dexjoco_task:
        fail.append(f"--task {task} but this run was trained on DexJoCo task {trained_on!r}")

import torch
if not torch.cuda.is_available():
    fail.append("CUDA not available")

if fail:
    print("\nPREFLIGHT FAILED:")
    for f in fail:
        print(f"  - {f}")
    sys.exit(1)
print("preflight OK")
PREFLIGHT

echo "=== dual-noise eval ($(date)) ==="
srun python examples/training/eval_pi05_dual_noise.py \
    --run-dir "$RUN_DIR" \
    --task "$TASK" \
    "$@"

echo "=== done ($(date)) ==="
