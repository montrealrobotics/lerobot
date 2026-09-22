#!/bin/bash
# Post-hoc dual-noise scoring of a finished SFT run's checkpoint ladder.
#
#   sbatch --job-name=dn_arm2_coffee dual_noise_eval.sbatch <run-dir> coffee
#   sbatch --job-name=dn_arm2_lamp   dual_noise_eval.sbatch <run-dir> lamp
#
# Extra args pass through to eval_pi05_dual_noise.py, e.g. --steps 16000 20000,
# --n-episodes 50, --no-wandb.
#SBATCH --partition main
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/network/scratch/a/artur.kuramshin/lerobot/slurm_logs/%x-%j.out

set -euo pipefail

RUN_DIR=${1:?usage: sbatch dual_noise_eval.sbatch <run-dir> <task> [extra args...]}
TASK=${2:?usage: sbatch dual_noise_eval.sbatch <run-dir> <task> [extra args...]}
shift 2

REPO=/home/mila/a/artur.kuramshin/lerobot
cd "$REPO"

source ~/.bashrc          # HF_HOME / HF_TOKEN live here on Mila
source .venv/bin/activate
# RoboCasa hard-asserts numpy==2.2.5 at import; the lockfile pins 2.2.6.
uv pip install --quiet numpy==2.2.5

export PYTHONUNBUFFERED=1
# NO HF_HOME override here: ~/.bashrc already points it at $SCRATCH/huggingface, which is where
# pi05_base and paligemma-3b-pt-224 are cached on Mila. The Alliance sibling sets it to
# /scratch/akur/hf_cache; that path does not exist on Mila and copying the line here makes every
# HF fetch die with PermissionError on /scratch (job 10883125).
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_${SLURM_JOB_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "=== preflight ($(date)) ==="
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) task=$TASK"
echo "run_dir=$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ -n "${SFT_MUJOCO_GL:-}" ]; then
    export MUJOCO_GL="$SFT_MUJOCO_GL"
    export PYOPENGL_PLATFORM="$SFT_MUJOCO_GL"
    echo "GL backend forced by SFT_MUJOCO_GL=$SFT_MUJOCO_GL"
else
    echo "probing headless GL backends (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
    if GL_EXPORTS=$(python scripts/check_gl_backend.py --emit-exports 2>/dev/null) && [ -n "$GL_EXPORTS" ]; then
        eval "$GL_EXPORTS"
        echo "GL backend: MUJOCO_GL=$MUJOCO_GL MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-n/a}"
    else
        echo "PREFLIGHT FAILED: no working headless GL backend on this node."
        echo "  Run 'python scripts/check_gl_backend.py' here for the per-device detail."
        echo "  On a MIG-partitioned cluster, check you got a FULL GPU (--gpus=h100:1)."
        exit 1
    fi
fi

python - "$RUN_DIR" "$TASK" <<'PREFLIGHT'
import json, os, sys
from pathlib import Path

run_dir, task = Path(sys.argv[1]), sys.argv[2]
fail = []

import numpy
if numpy.__version__ != "2.2.5":
    fail.append(f"numpy is {numpy.__version__}, RoboCasa asserts 2.2.5")

from huggingface_hub import get_token
if not get_token():
    fail.append(f"no HF token visible (HF_HOME={os.environ.get('HF_HOME')}); export HF_TOKEN")

import netrc
has_netrc = False
try:
    has_netrc = "api.wandb.ai" in netrc.netrc(os.path.expanduser("~/.netrc")).hosts
except Exception:
    pass
if not os.environ.get("WANDB_API_KEY") and not has_netrc:
    fail.append("no wandb credentials (set WANDB_API_KEY, run `wandb login`, or pass --no-wandb)")

# The run must exist, be readable, and carry the ladder we are about to score.
ckpts = run_dir / "checkpoints"
if not ckpts.is_dir():
    fail.append(f"no checkpoints/ under {run_dir}")
else:
    steps = sorted(int(d.name) for d in ckpts.iterdir() if d.is_dir() and d.name.isdigit())
    print(f"checkpoints found: {len(steps)} -> {steps}")
    if not steps:
        fail.append(f"{ckpts} has no numbered checkpoints")
    for step in steps:
        pm = ckpts / f"{step:06d}" / "pretrained_model"
        if not (pm / "model.safetensors").exists() and not list(pm.glob("*.safetensors")):
            fail.append(f"checkpoint {step} has no weights in {pm}")
            break

    # The run's OWN recorded task/robot must match the --task we were given.
    tc = ckpts / f"{steps[-1]:06d}" / "pretrained_model" / "train_config.json"
    if tc.exists():
        env = json.loads(tc.read_text()).get("env")
        env = env[0] if isinstance(env, list) and env else env
        if isinstance(env, dict):
            sys.path.insert(0, "examples/training")
            from train_pi05_casa_experiments import TASKS
            want = TASKS[task]
            print(f"train_config: task={env.get('task')} robot={env.get('robot')} fps={env.get('fps')}")
            if env.get("task") != want.robocasa_task:
                fail.append(
                    f"--task {task} expects robocasa task {want.robocasa_task!r} but this run "
                    f"was trained/evaluated on {env.get('task')!r}"
                )
            if env.get("robot") != want.robot:
                fail.append(
                    f"--task {task} expects robot {want.robot!r} but this run used {env.get('robot')!r}"
                )

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
