#!/bin/bash
# DSRL over a frozen pi05 SFT checkpoint -- ALLIANCE/DRAC. Mila sibling: dsrl_mila.sbatch.
# Design: SFT_ARMS_DSRL.md.
#
#   for S in 0 1 2; do
#     sbatch --job-name=dsrl_arm0_lamp_s$S dsrl_drac.sbatch lamp <ckpt>/pretrained_model $S
#   done
#
# Args after the seed are forwarded verbatim to the training script.
#
#SBATCH --account=def-kdarvish
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48000M
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@300
#SBATCH --output=/home/akur/scratch/lerobot/slurm_logs/%x-%j.out

# A 500k-step run is ~10 h of training plus ~3.6 h of eval. A job that hits the wall clock
# writes resume_state.pt and requeues (see the trap at the bottom), so no --total_steps
# trimming is needed to fit a cluster's cap.

set -euo pipefail

TASK=${1:?usage: sbatch dsrl_drac.sbatch <lamp|coffee> <policy_path> <seed> [extra args...]}
POLICY=${2:?usage: sbatch dsrl_drac.sbatch <lamp|coffee> <policy_path> <seed> [extra args...]}
SEED=${3:?usage: sbatch dsrl_drac.sbatch <lamp|coffee> <policy_path> <seed> [extra args...]}
shift 3

REPO=/project/6068036/akur/lerobot
SCRATCH=/home/akur/scratch
BANKS=$SCRATCH/lerobot/placement_banks
cd "$REPO"

module load StdEnv/2023 python/3.12.4
source .venv/bin/activate

export PYTHONUNBUFFERED=1
# Home is a 50G quota and pi05_base alone is ~8G.
export HF_HOME=/scratch/akur/hf_cache
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
# The NVIDIA driver caches compiled shaders under ~/.cache/nvidia/GLCache -- in $HOME, a
# filesystem shared by every job on every node. Concurrent EGL processes (three per DSRL
# pack, several eval workers per SFT job) race on those files, and a reader that catches a
# half-written shader makes the driver abort: SIGABRT, exit 134, no Python traceback. That
# matches the dsrl_arm0_coffee_04000 crashes on nibi and rorqual, whose cache files were
# being written inside the crash window. MuJoCo compiles a handful of shaders, so disabling
# the disk cache costs nothing measurable.
export __GL_SHADER_DISK_CACHE=0
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR" "$SCRATCH/lerobot/slurm_logs"

echo "=== $(date) job=${SLURM_JOB_ID:-none} node=$(hostname) task=$TASK seed=$SEED"
echo "policy=$POLICY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -d "$POLICY" ] || { echo "no such checkpoint: $POLICY" >&2; exit 2; }

# DSRL renders every collection step, so there is no eval-free mode to fall back to. MIG
# partitions cannot do EGL at all -- probe rather than assume.
if [ -n "${DSRL_MUJOCO_GL:-}" ]; then
    export MUJOCO_GL="$DSRL_MUJOCO_GL"
    export PYOPENGL_PLATFORM="$DSRL_MUJOCO_GL"
    echo "GL backend forced by DSRL_MUJOCO_GL=$DSRL_MUJOCO_GL"
elif GL_EXPORTS=$(python scripts/check_gl_backend.py --emit-exports 2>/dev/null) && [ -n "$GL_EXPORTS" ]; then
    eval "$GL_EXPORTS"
    echo "GL backend: MUJOCO_GL=$MUJOCO_GL MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-n/a}"
else
    echo "PREFLIGHT FAILED: no working headless GL backend on this node." >&2
    echo "  Run 'python scripts/check_gl_backend.py' here for per-device detail; check you" >&2
    echo "  got a FULL GPU (--gpus=h100:1), not a MIG slice." >&2
    exit 1
fi

python - <<'PREFLIGHT'
import os, sys
fail = []
import numpy
if numpy.__version__ != "2.2.5":
    fail.append(f"numpy is {numpy.__version__}, RoboCasa asserts 2.2.5")
from huggingface_hub import get_token
if not get_token():
    fail.append(f"no HF token visible (HF_HOME={os.environ.get('HF_HOME')}; export HF_TOKEN)")
import netrc
has_netrc = False
try:
    has_netrc = "api.wandb.ai" in netrc.netrc(os.path.expanduser("~/.netrc")).hosts
except Exception:
    pass
if not os.environ.get("WANDB_API_KEY") and not has_netrc:
    fail.append("no wandb credentials (set WANDB_API_KEY or run `wandb login`)")
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

# Per-task invariants (robot, fps, controller, bank, eval sets, prompt) -- shared with
# dsrl_drac_pack.sbatch so the two launchers cannot drift apart.
source "$REPO/dsrl_task_args.sh"

OUT=$SCRATCH/lerobot/outputs/dsrl/${SLURM_JOB_NAME:-dsrl_${TASK}}/seed${SEED}

# Preemption / wall-clock: the trainer writes <output_dir>/resume_state.pt on SIGTERM and exits
# 99. Slurm signals the step directly on preemption; the trap covers --signal=B:TERM@300, which
# only reaches the batch script. --open-mode=append keeps the requeued run's log.
# NOT via srun: `kill -TERM <srun pid>` makes srun CANCEL the step and SIGKILL the task
# (verified: sacct showed "CANCELLED ... 0:9"), so the trainer never sees SIGTERM. Running
# python as a direct child of the batch script means the trap reaches it, and on preemption
# Slurm's SIGTERM hits every process in the cgroup, including this one.
python examples/tutorial/rl/dsrl_pi05_robocasa_example.py \
    --policy_path "$POLICY" \
    "${TASK_ARGS[@]}" \
    --utd_ratio 20 \
    --total_steps 500000 \
    --eval_freq 25000 --eval_episodes 5 \
    --seed "$SEED" \
    --output_dir "$OUT" \
    --resume \
    --wandb_enable --wandb_name "${SLURM_JOB_NAME:-dsrl_${TASK}}_seed${SEED}" \
    "$@" &
PY_PID=$!
trap 'echo "[sbatch] signal received, forwarding to $PY_PID"; kill -TERM $PY_PID 2>/dev/null || true' TERM

# `wait` must not trip `set -e`, and it returns 128+signal when the trap interrupts it rather
# than when the child exits -- so wait a second time for the real status.
RC=0; wait $PY_PID || RC=$?
if [ "$RC" -gt 128 ]; then RC=0; wait $PY_PID || RC=$?; fi

if [ "$RC" = "99" ]; then
    echo "[sbatch] checkpointed at the signal; requeueing ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi
echo "=== done rc=$RC ($(date)) ==="
exit $RC
