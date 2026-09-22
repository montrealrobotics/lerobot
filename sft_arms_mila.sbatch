#!/bin/bash
# SFT experiment arms for the pi05 embodiment-transfer study -- MILA cluster.
# Alliance sibling: sft_arms.sbatch. Runbook: SFT_ARMS_RUNBOOK.md, which also covers
# WHICH cluster an arm belongs on (an arm is only comparable to what it is contrasted
# against, and the existing arm1 lamp/coffee runs are on Alliance H100s).
#
#   sbatch --job-name=sft_arm0_lamp sft_arms_mila.sbatch arm0_expert_only lamp
#   sbatch --job-name=smoke sft_arms_mila.sbatch arm0_expert_only lamp --smoke-test
#
# ONLY cluster plumbing may differ between this file and its sibling: both end in the
# same single srun line forwarding "$@", so the fairness invariants live in the training
# script and cannot drift between launchers.
#
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/network/scratch/a/artur.kuramshin/lerobot/slurm_logs/%x-%j.out
#SBATCH --signal=B:TERM@300

set -euo pipefail

ARM=${1:?usage: sbatch sft_arms_mila.sbatch <arm> <task> [extra args...]}
TASK=${2:?usage: sbatch sft_arms_mila.sbatch <arm> <task> [extra args...]}
shift 2

REPO=/home/mila/a/artur.kuramshin/lerobot
OUTROOT=/network/scratch/a/artur.kuramshin/lerobot/outputs/train
cd "$REPO"

source ~/.bashrc          # HF_HOME / HF_TOKEN live here on Mila
source .venv/bin/activate
# RoboCasa hard-asserts numpy==2.2.5 at import; the lockfile pins 2.2.6.
uv pip install --quiet numpy==2.2.5

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_${SLURM_JOB_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR" "$(dirname "$OUTROOT")/slurm_logs"

echo "=== preflight ($(date)) ==="
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) arm=$ARM task=$TASK extra=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUTROOT="$OUTROOT" python - <<'PREFLIGHT'
import os, shutil, sys

fail = []

import numpy
if numpy.__version__ != "2.2.5":
    fail.append(f"numpy is {numpy.__version__}, RoboCasa asserts 2.2.5")

# google/paligemma-3b-pt-224 is a GATED repo; without a token the run dies late.
from huggingface_hub import get_token
if not get_token():
    fail.append(f"no HF token visible (HF_HOME={os.environ.get('HF_HOME')})")

import netrc
has_netrc = False
try:
    has_netrc = "api.wandb.ai" in netrc.netrc(os.path.expanduser("~/.netrc")).hosts
except Exception:
    pass
if not os.environ.get("WANDB_API_KEY") and not has_netrc:
    fail.append("no wandb credentials (set WANDB_API_KEY or run `wandb login`)")

# Mila scratch has no diskusage_report, so read the filesystem. Projection is full-FT
# sizing (23 GiB/rung), an upper bound for arm0's ~14 GiB. See SFT_ARMS_RUNBOOK.md.
out = os.environ["OUTROOT"]
os.makedirs(out, exist_ok=True)
sys.path.insert(0, "examples/training")
from train_pi05_casa_experiments import SAVE_FREQ as _SAVE_FREQ, TOTAL_STEPS as _TOTAL_STEPS
steps = int(os.environ.get("SFT_STEPS", _TOTAL_STEPS))
save_freq = int(os.environ.get("SFT_SAVE_FREQ", _SAVE_FREQ))
prune = os.environ.get("SFT_PRUNE_OPTIMIZER_STATE", "1") == "1"
rungs = steps // save_freq
projected = rungs * 8.7 + 14.1 if prune else rungs * 23.0
free_gb = shutil.disk_usage(out).free / 2**30
print(f"scratch free: {free_gb:.0f} GiB | projected checkpoints: {rungs} rungs, "
      f"prune={prune} -> {projected:.0f} GiB (upper bound, full-FT sizing)")
if projected > free_gb:
    fail.append(f"would write ~{projected:.0f} GiB but only {free_gb:.0f} GiB free")

import torch
if not torch.cuda.is_available():
    fail.append("CUDA not available")
else:
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")

if fail:
    print("\nPREFLIGHT FAILED:")
    for f in fail:
        print(f"  - {f}")
    sys.exit(1)
print("preflight OK")
PREFLIGHT

# ScrewLightbulb only exists on robocasa's `lightbulb_task` branch.
python - "$TASK" <<'ENVCHECK'
import sys
task = sys.argv[1]
sys.path.insert(0, "examples/training")
from train_pi05_casa_experiments import TASKS
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()
import robocasa  # noqa: F401
import robosuite  # noqa: F401
from robosuite.environments.base import REGISTERED_ENVS
name = TASKS[task].robocasa_task
if name not in REGISTERED_ENVS:
    raise SystemExit(
        f"robocasa env {name!r} is not registered even after importing robocasa. "
        f"ScrewLightbulb only exists on the robocasa `lightbulb_task` branch."
    )
print(f"robocasa env {name} registered OK")
ENVCHECK

# Drop AdamW state from every checkpoint but the newest: resume only ever reads `last`,
# and the pruned rungs keep pretrained_model/ so the checkpoint ladder stays usable for
# eval and for dual-noise selection. SFT_PRUNE_OPTIMIZER_STATE=0 keeps everything.
PRUNE=${SFT_PRUNE_OPTIMIZER_STATE:-1}
prune_loop () {
    local root="$1"
    while true; do
        sleep 300
        [ -d "$root" ] || continue
        local latest
        latest=$(find "$root" -maxdepth 1 -type d -regex '.*/[0-9]+' -printf '%f\n' 2>/dev/null | sort -n | tail -1)
        [ -n "$latest" ] || continue
        for d in "$root"/[0-9]*; do
            [ -d "$d" ] || continue
            [ "$(basename "$d")" = "$latest" ] && continue
            if [ -d "$d/training_state" ]; then
                rm -rf "$d/training_state"
                echo "[prune] dropped optimizer state from $(basename "$d") (keeping $latest)"
            fi
        done
    done
}

echo "=== training ($(date)) ==="
srun python examples/training/train_pi05_casa_experiments.py \
    --experiment "$ARM" \
    --task "$TASK" \
    "$@" &
TRAIN_PID=$!

if [ "$PRUNE" = "1" ]; then
    (
        for _ in $(seq 1 60); do
            sleep 30
            # TASK must be in the pattern: coffee and lamp jobs share an arm name.
            RUN=$(find "$OUTROOT" -maxdepth 2 -type d -newermt '-15 minutes' -name "*_${TASK}_${ARM}_*" 2>/dev/null | sort | tail -1)
            [ -n "$RUN" ] && break
        done
        if [ -n "${RUN:-}" ]; then
            echo "[prune] watching $RUN/checkpoints"
            prune_loop "$RUN/checkpoints"
        else
            echo "[prune] WARNING: could not locate the run directory; pruning disabled"
        fi
    ) &
    PRUNE_PID=$!
    trap 'kill $PRUNE_PID 2>/dev/null || true' EXIT
fi

wait $TRAIN_PID
TRAIN_RC=$?
echo "=== done rc=$TRAIN_RC ($(date)) ==="
exit $TRAIN_RC
