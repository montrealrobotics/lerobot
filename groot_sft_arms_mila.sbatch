#!/bin/bash
# GR00T N1.7 SFT arms (second VLA of the SFT-arms x DSRL study) -- MILA cluster.
# pi05 sibling: sft_arms_mila.sbatch. Training script: examples/training/train_groot_casa_experiments.py.
#
#   sbatch --job-name=groot_arm0_lamp groot_sft_arms_mila.sbatch arm0_expert_only lamp
#   sbatch --job-name=groot_smoke groot_sft_arms_mila.sbatch arm3_lora_r32 coffee --smoke-test
#   # arm1 trains all 3.1B parameters in fp32 with AdamW (~50 GiB before activations): 80 GiB GPU
#   sbatch --gres=gpu:a100l:1 --job-name=groot_arm1_lamp groot_sft_arms_mila.sbatch arm1_full_ft lamp
#
# ONLY cluster plumbing lives here; arms, invariants and parity overrides live in the training script.
# Everything runs through groot_env.sh, which isolates the GR00T venv (transformers 5.5) from the pi05
# one -- sbatch otherwise inherits whatever venv the submitting shell has active.
#
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/network/scratch/a/artur.kuramshin/lerobot/slurm_logs/%x-%j.out
#SBATCH --signal=B:TERM@300

set -euo pipefail

ARM=${1:?usage: sbatch groot_sft_arms_mila.sbatch <arm> <task> [extra args...]}
TASK=${2:?usage: sbatch groot_sft_arms_mila.sbatch <arm> <task> [extra args...]}
shift 2

GROOT_REPO=/home/mila/a/artur.kuramshin/lerobot-groot
ENV="$GROOT_REPO/groot_env.sh"
OUTROOT=/network/scratch/a/artur.kuramshin/lerobot/outputs/train
mkdir -p "$(dirname "$OUTROOT")/slurm_logs"
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_${SLURM_JOB_ID}"

echo "=== preflight ($(date)) ==="
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) arm=$ARM task=$TASK extra=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUTROOT="$OUTROOT" ARM="$ARM" "$ENV" python - <<'PREFLIGHT'
import os, shutil, sys

fail = []
import numpy
if numpy.__version__ != "2.2.5":
    fail.append(f"numpy is {numpy.__version__}, RoboCasa asserts 2.2.5")

# nvidia/Cosmos-Reason2-2B (GR00T's backbone) is GATED.
from huggingface_hub import get_token, model_info
if not get_token():
    fail.append(f"no HF token visible (HF_HOME={os.environ.get('HF_HOME')})")
else:
    try:
        model_info("nvidia/Cosmos-Reason2-2B")
    except Exception as e:  # noqa: BLE001
        fail.append(f"no access to gated nvidia/Cosmos-Reason2-2B: {type(e).__name__}")

import netrc
has_netrc = False
try:
    has_netrc = "api.wandb.ai" in netrc.netrc(os.path.expanduser("~/.netrc")).hosts
except Exception:
    pass
if not os.environ.get("WANDB_API_KEY") and not has_netrc:
    fail.append("no wandb credentials (set WANDB_API_KEY or run `wandb login`)")

# GR00T checkpoints are fp32: 12.6 GiB of weights per rung; AdamW state (2 x trainable fp32, up to
# 25 GiB for arm1) is kept only on the newest rung by the pruner below.
out = os.environ["OUTROOT"]
os.makedirs(out, exist_ok=True)
sys.path.insert(0, "examples/training")
from train_pi05_casa_experiments import SAVE_FREQ as _SAVE_FREQ, TOTAL_STEPS as _TOTAL_STEPS
steps = int(os.environ.get("SFT_STEPS", _TOTAL_STEPS))
save_freq = int(os.environ.get("SFT_SAVE_FREQ", _SAVE_FREQ))
prune = os.environ.get("SFT_PRUNE_OPTIMIZER_STATE", "1") == "1"
rungs = steps // save_freq
projected = rungs * 12.6 + 25.3 if prune else rungs * 37.9
free_gb = shutil.disk_usage(out).free / 2**30
print(f"scratch free: {free_gb:.0f} GiB | projected: {rungs} rungs, prune={prune} -> {projected:.0f} GiB (arm1 sizing)")
if projected > free_gb:
    fail.append(f"would write ~{projected:.0f} GiB but only {free_gb:.0f} GiB free")

import torch
if not torch.cuda.is_available():
    fail.append("CUDA not available")
else:
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | {total:.0f} GiB")
    if os.environ["ARM"] == "arm1_full_ft" and total < 70:
        fail.append(f"arm1_full_ft needs an 80 GiB GPU (got {total:.0f} GiB): sbatch --gres=gpu:a100l:1")

if fail:
    print("\nPREFLIGHT FAILED:")
    for f in fail:
        print(f"  - {f}")
    sys.exit(1)
print("preflight OK")
PREFLIGHT

# ScrewLightbulb only exists on robocasa's `lightbulb_task` branch.
"$ENV" python - "$TASK" <<'ENVCHECK'
import sys
sys.path.insert(0, "examples/training")
from train_pi05_casa_experiments import TASKS
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()
import robocasa  # noqa: F401
from robosuite.environments.base import REGISTERED_ENVS
name = TASKS[sys.argv[1]].robocasa_task
if name not in REGISTERED_ENVS:
    raise SystemExit(f"robocasa env {name!r} is not registered (ScrewLightbulb needs the lightbulb_task branch).")
print(f"robocasa env {name} registered OK")
ENVCHECK

# Drop AdamW state from every checkpoint but the newest (resume only reads `last`).
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
srun "$ENV" python examples/training/train_groot_casa_experiments.py \
    --experiment "$ARM" \
    --task "$TASK" \
    "$@" &
TRAIN_PID=$!

if [ "$PRUNE" = "1" ]; then
    (
        for _ in $(seq 1 60); do
            sleep 30
            # Run dirs are <date>/<time>_groot_sft_arms_<task>_<arm>_seed<S>; task + arm + "groot" in
            # the pattern so pi05 runs and the other task's runs are never touched.
            RUN=$(find "$OUTROOT" -maxdepth 2 -type d -newermt '-15 minutes' -name "*_groot_sft_arms_${TASK}_${ARM}_*" 2>/dev/null | sort | tail -1)
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

TRAIN_RC=0
wait $TRAIN_PID || TRAIN_RC=$?
echo "=== done rc=$TRAIN_RC ($(date)) ==="
exit $TRAIN_RC
