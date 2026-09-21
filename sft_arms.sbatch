#!/bin/bash
# SFT experiment arms for the pi05 embodiment-transfer study.
#
#   sbatch --job-name=sft_arm1_coffee sft_arms.sbatch arm1_full_ft coffee
#   sbatch --job-name=sft_arm1_lamp   sft_arms.sbatch arm1_full_ft lamp
#
# One parameterized script rather than one file per arm x task: the matrix is 4 arms x
# 2 tasks x 3 lr-scales, and copies of a launcher drift apart in ways that silently
# break the fairness invariants the study depends on.
#
# Optional extra args are forwarded to the training script, e.g.
#   sbatch ... sft_arms.sbatch arm3_lora_r32 coffee --lr-scale 2.0 --seed 7
#
#SBATCH --account=def-kdarvish
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
# MEASURED budget for a 20k-step arm1 run (H100, batch 32, compile off), taken from the
# completed jobs 22285902/22285903 rather than from a smoke test:
#   training      20000 x 1.70 s                    = 9.4 h
#   in-loop eval  20 episodes @ eval bs 2, steady   = 1.5 min (coffee) / 7.4 min (lamp)
#   checkpoints   10 x ~60 s                        = 10 min
# Eval is far cheaper than an earlier smoke-test estimate suggested (that measurement was
# polluted by torch.compile warmup on a cold GPU), so eval_freq=2000 (10 evals) costs only
# ~15 min on coffee and ~74 min on lamp. Totals: ~9.8 h coffee, ~10.7 h lamp -> 12 h fits.
# Keep eval_freq identical across arms; it is a declared fairness invariant.
#SBATCH --time=12:00:00
#SBATCH --output=/home/akur/scratch/lerobot/slurm_logs/%x-%j.out
#SBATCH --signal=B:TERM@300

set -euo pipefail

ARM=${1:?usage: sbatch sft_arms.sbatch <arm> <task> [extra args...]}
TASK=${2:?usage: sbatch sft_arms.sbatch <arm> <task> [extra args...]}
shift 2

REPO=/project/6068036/akur/lerobot
cd "$REPO"

module load StdEnv/2023 python/3.12.4
source .venv/bin/activate

export PYTHONUNBUFFERED=1

# Where the pi05 weights, the PaliGemma tokenizer and the datasets are cached. Must be on
# scratch: home is a 50G quota and pi05_base alone is ~8G.
export HF_HOME=/scratch/akur/hf_cache

# Keep BLAS from oversubscribing the allocation; the dataloader already uses the cores.
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# torch.compile runs in max-autotune mode (PI05Config.compile_mode), which writes a large
# inductor cache. Keep it node-local and per-job: a shared cache on the networked filesystem
# is slow and concurrent jobs can corrupt it.
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_${SLURM_JOB_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR"

# ---------------------------------------------------------------------------
# Preflight. Every one of these has a failure mode that otherwise surfaces minutes
# to hours into the job, after the queue wait. Fail in seconds instead.
# ---------------------------------------------------------------------------
echo "=== preflight ($(date)) ==="
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) arm=$ARM task=$TASK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader


NEEDS_GL=1
case " $* " in
    *" --eval-freq 0 "*|*" --eval-freq=0 "*) NEEDS_GL=0 ;;
esac
[ "$NEEDS_GL" = "0" ] && echo "eval disabled (--eval-freq 0): no renderer required"

if [ -n "${SFT_MUJOCO_GL:-}" ]; then
    export MUJOCO_GL="$SFT_MUJOCO_GL"
    export PYOPENGL_PLATFORM="$SFT_MUJOCO_GL"
    echo "GL backend forced by SFT_MUJOCO_GL=$SFT_MUJOCO_GL"
else
    echo "probing headless GL backends (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
    if GL_EXPORTS=$(python scripts/check_gl_backend.py --emit-exports 2>/dev/null) && [ -n "$GL_EXPORTS" ]; then
        eval "$GL_EXPORTS"
        echo "GL backend: MUJOCO_GL=$MUJOCO_GL MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-n/a}"
    elif [ "$NEEDS_GL" = "0" ]; then
        echo "no working headless GL backend -- fine, this run has eval disabled."
    else
        echo "PREFLIGHT FAILED: no working headless GL backend on this node."
        echo "  Run 'python scripts/check_gl_backend.py' here for the per-device detail."
        echo "  On a MIG-partitioned cluster, check you got a FULL GPU (fir: --gpus=h100:1);"
        echo "  MIG cannot do OpenGL/EGL at all."
        echo "  To train here anyway WITHOUT rendering, pass --eval-freq 0 and score the"
        echo "  checkpoints on a cluster where EGL works."
        exit 1
    fi
fi

python - <<'PREFLIGHT'
import os, sys, shutil

fail = []

# RoboCasa hard-asserts this exact version at import time.
import numpy
if numpy.__version__ != "2.2.5":
    fail.append(f"numpy is {numpy.__version__}, RoboCasa asserts 2.2.5")

# pi05's tokenizer (google/paligemma-3b-pt-224) is a GATED repo. Without a token the run
# dies when processors are built, which is after the model has already been loaded.
from huggingface_hub import get_token
if not get_token():
    fail.append(f"no HF token visible (HF_HOME={os.environ.get('HF_HOME')}; "
                "huggingface_hub reads $HF_HOME/token, so a token in ~/.cache/huggingface "
                "is NOT found when HF_HOME is set -- export HF_TOKEN instead)")

# wandb resolves credentials from WANDB_API_KEY or ~/.netrc.
import netrc
has_netrc = False
try:
    has_netrc = "api.wandb.ai" in netrc.netrc(os.path.expanduser("~/.netrc")).hosts
except Exception:
    pass
if not os.environ.get("WANDB_API_KEY") and not has_netrc:
    fail.append("no wandb credentials (set WANDB_API_KEY or run `wandb login`)")

# Disk. MEASURED on this cluster: one full-FT checkpoint is 23 GiB -- 8.7 GiB of weights
# plus 14.1 GiB of AdamW state -- and the default ladder is steps/save_freq = 20 rungs,
# i.e. ~460 GiB for a single arm1 run. Scratch is a 1 TiB quota, so this does NOT fit
# alongside existing data and MUST be checked before burning a queue slot. The quota is
# what bites, not the filesystem, so read it from diskusage_report when available.
import re, subprocess
out = os.path.expanduser("~/scratch/lerobot/outputs/train")
os.makedirs(out, exist_ok=True)

sys.path.insert(0, "examples/training")
from train_pi05_casa_experiments import SAVE_FREQ as _SAVE_FREQ, TOTAL_STEPS as _TOTAL_STEPS
steps = int(os.environ.get("SFT_STEPS", _TOTAL_STEPS))
save_freq = int(os.environ.get("SFT_SAVE_FREQ", _SAVE_FREQ))
prune = os.environ.get("SFT_PRUNE_OPTIMIZER_STATE", "1") == "1"
rungs = steps // save_freq
# 8.7 GiB of weights + 14.1 GiB of AdamW state. With pruning only the newest rung keeps
# its optimizer state (all that `--resume` ever reads), so older rungs cost weights only.
projected = rungs * 8.7 + 14.1 if prune else rungs * 23.0

free_gb = None
try:
    report = subprocess.run(["diskusage_report"], capture_output=True, text=True, timeout=60).stdout
    m = re.search(r"/scratch[^\n]*?(\d+(?:\.\d+)?)([KMGT])iB/\s*(\d+(?:\.\d+)?)([KMGT])iB", report)
    if m:
        unit = {"K": 1 / 2**20, "M": 1 / 2**10, "G": 1, "T": 1024}
        used = float(m.group(1)) * unit[m.group(2)]
        quota = float(m.group(3)) * unit[m.group(4)]
        free_gb = quota - used
        print(f"scratch quota: {used:.0f}/{quota:.0f} GiB used -> {free_gb:.0f} GiB free")
except Exception as e:
    print(f"(diskusage_report unavailable: {e})")
if free_gb is None:
    free_gb = shutil.disk_usage(out).free / 2**30
    print(f"scratch filesystem free: {free_gb:.0f} GiB (quota not read)")

print(f"projected checkpoints: {rungs} rungs, prune_optimizer_state={prune} -> {projected:.0f} GiB")
if projected > free_gb:
    fail.append(
        f"this run would write ~{projected:.0f} GiB of checkpoints but only {free_gb:.0f} GiB "
        f"of scratch quota is free. Raise save_freq, prune training_state/ from old rungs "
        f"(14.1 of the 23 GiB per checkpoint is AdamW state, needed only to resume the "
        f"LATEST one), or clear old runs. Override the projection with SFT_STEPS/SFT_SAVE_FREQ."
    )

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

# The lamp task's environment only exists on one robocasa branch; a missing env would
# otherwise surface at the first eval, not at startup.
python - "$TASK" <<'ENVCHECK'
import sys
sys.argv[0] = "envcheck"
task = sys.argv[1]
sys.path.insert(0, "examples/training")
from train_pi05_casa_experiments import TASKS
# robocasa registers its environments with robosuite AT IMPORT, so REGISTERED_ENVS is
# just the robosuite base set until it (and any third-party plugins) are loaded.
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()
import robocasa  # noqa: F401  -- registers CoffeePressButton / ScrewLightbulb
import robosuite  # noqa: F401
from robosuite.environments.base import REGISTERED_ENVS
name = TASKS[task].robocasa_task
if name not in REGISTERED_ENVS:
    raise SystemExit(
        f"robocasa env {name!r} is not registered even after importing robocasa. "
        f"ScrewLightbulb only exists on the robocasa `lightbulb_task` branch. "
        f"{len(REGISTERED_ENVS)} envs registered, sample: {sorted(REGISTERED_ENVS)[:5]}"
    )
print(f"robocasa env {name} registered OK")
ENVCHECK

# ---------------------------------------------------------------------------
# Checkpoint pruning.
#
# A full-FT checkpoint is 23 GiB, of which 14.1 GiB is AdamW state. That state is only
# ever read to RESUME, and resume only ever reads the newest checkpoint (`last`), so
# keeping it on every rung of a 20-rung ladder wastes ~270 GiB per run against a 1 TiB
# scratch quota. This loop deletes `training_state/` from every checkpoint EXCEPT the
# newest one. The pruned rungs keep their full `pretrained_model/`, so they remain
# usable for eval and for the dual-noise checkpoint selection -- which is what the
# ladder exists for. Set SFT_PRUNE_OPTIMIZER_STATE=0 to keep everything.
# ---------------------------------------------------------------------------
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
    # The run directory is timestamped by the training script, so discover it once it exists.
    (
        OUTROOT=/home/akur/scratch/lerobot/outputs/train
        for _ in $(seq 1 60); do
            sleep 30
            # Must include TASK: both the coffee and lamp jobs share ARM=arm1_full_ft, so
            # matching on the arm alone lets one job's pruner latch onto the other job's
            # run directory -- leaving its own run unpruned and overrunning the quota.
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
