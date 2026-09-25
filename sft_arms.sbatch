#!/bin/bash
# SFT experiment arms for the pi05 embodiment-transfer study -- ALLIANCE cluster.
# Mila sibling: sft_arms_mila.sbatch. Runbook: SFT_ARMS_RUNBOOK.md.
#
#   sbatch --job-name=sft_arm1_coffee sft_arms.sbatch arm1_full_ft coffee
#   sbatch ... sft_arms.sbatch arm3_lora_r32 coffee --lr-scale 2.0 --seed 7
#
# Args after <arm> <task> are forwarded verbatim to the training script.
#
#SBATCH --account=def-kdarvish
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
# 64G is HEADROOM, not a diagnosed fix. Training sits at ~44 GB of the 46.88 GB that 48000M
# gives (94%), and the eval envs are built LAZILY -- _LazyAsyncVectorEnv creates the
# AsyncVectorEnv on first step -- so the first eval forkserver-spawns n_envs workers that
# each re-import robosuite/robocasa/MuJoCo and take an EGL context. Running that close to
# the cap is worth avoiding on its own.
# It is NOT established that memory killed job 22402972 (EOFError from a dead eval worker
# at its first eval): slurmstepd logged no oom_kill event, and job 22358170 completed 7
# evals of the same arm at a HIGHER peak (45.9 vs 44.0 GiB). That cause is still open.
#SBATCH --mem=64000M
#SBATCH --time=12:00:00
#SBATCH --output=/home/akur/scratch/lerobot/slurm_logs/%x-%j.out
#SBATCH --signal=B:TERM@300

set -euo pipefail

ARM=${1:?usage: sbatch sft_arms.sbatch <arm> <task> [extra args...]}
TASK=${2:?usage: sbatch sft_arms.sbatch <arm> <task> [extra args...]}
shift 2

# Cluster plumbing -- the ONLY thing that differs between clusters. #SBATCH directives
# cannot be conditional; on rorqual override them at submit time:
#   sbatch --account=<best FairShare from `sshare -U -u $USER`> \
#       --output=/scratch/$USER/lerobot/slurm_logs/%x-%j.out ... sft_arms.sbatch ...
case "${SLURM_CLUSTER_NAME:-$(hostname)}" in
  nibi*)
    REPO=/project/6068036/akur/lerobot
    OUTROOT=/home/akur/scratch/lerobot/outputs/train
    HF_HOME_DIR=/scratch/akur/hf_cache
    OFFLINE=0 ;;
  rorqual*)
    # Compute nodes have NO internet: pi05_base, the PaliGemma tokenizer and the
    # datasets come from a pre-warmed HF cache, and wandb logs offline (synced later from
    # a login node with scripts/sync_offline_wandb.sh). There is no ~/scratch symlink
    # here, so the training script's $HOME/scratch fallbacks must be overridden -- the
    # output root would otherwise land in the 50G home quota.
    REPO=/lustre09/project/6068036/akur/lerobot   # = ~/links/projects/def-gberseth/akur/lerobot
    OUTROOT=/scratch/$USER/lerobot/outputs/train
    HF_HOME_DIR=/scratch/$USER/hf_cache
    export LEROBOT_LAMP_PLACEMENT_BANK=/scratch/$USER/lerobot/placement_banks/screwlightbulb_xarm6_seed1/bank.json
    OFFLINE=1 ;;
  *) echo "unknown cluster '${SLURM_CLUSTER_NAME:-$(hostname)}' -- add it to the case above" >&2; exit 2 ;;
esac
[ -d "$REPO" ] || { echo "no repo at $REPO" >&2; exit 2; }
export LEROBOT_OUTPUT_ROOT=$OUTROOT
cd "$REPO"

module load StdEnv/2023 python/3.12.4
source .venv/bin/activate

export PYTHONUNBUFFERED=1

# Must be on scratch: home is a 50G quota and pi05_base alone is ~8G.
export HF_HOME=$HF_HOME_DIR
if [ "$OFFLINE" = "1" ]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export WANDB_MODE=offline
    echo "offline cluster: HF from cache at $HF_HOME, wandb logging offline"
fi

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

# Node-local, per-job: a shared cache on the networked filesystem is slow and concurrent
# jobs corrupt it (a stale numba cache surfaces as "EOFError: Ran out of input").
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_${SLURM_JOB_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR"

# Preflight -- each check otherwise fails hours in. See SFT_ARMS_RUNBOOK.md.
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

import numpy
if numpy.__version__ != "2.2.5":
    fail.append(f"numpy is {numpy.__version__}, RoboCasa asserts 2.2.5")

# google/paligemma-3b-pt-224 is a GATED repo; without a token the run dies late.
from huggingface_hub import get_token
if not get_token():
    fail.append(f"no HF token visible (HF_HOME={os.environ.get('HF_HOME')}; "
                "huggingface_hub reads $HF_HOME/token, so a token in ~/.cache/huggingface "
                "is NOT found when HF_HOME is set -- export HF_TOKEN instead)")

import netrc
has_netrc = False
try:
    has_netrc = "api.wandb.ai" in netrc.netrc(os.path.expanduser("~/.netrc")).hosts
except Exception:
    pass
if not os.environ.get("WANDB_API_KEY") and not has_netrc:
    fail.append("no wandb credentials (set WANDB_API_KEY or run `wandb login`)")

# A full-FT rung is 23 GiB and the default ladder is 20 rungs. The QUOTA is what bites
# here, not the filesystem, so prefer diskusage_report. Sizing: SFT_ARMS_RUNBOOK.md.
import re, subprocess
out = os.environ["LEROBOT_OUTPUT_ROOT"]
os.makedirs(out, exist_ok=True)

sys.path.insert(0, "examples/training")
from train_pi05_casa_experiments import SAVE_FREQ as _SAVE_FREQ, TOTAL_STEPS as _TOTAL_STEPS
steps = int(os.environ.get("SFT_STEPS", _TOTAL_STEPS))
save_freq = int(os.environ.get("SFT_SAVE_FREQ", _SAVE_FREQ))
prune = os.environ.get("SFT_PRUNE_OPTIMIZER_STATE", "1") == "1"
rungs = steps // save_freq
# With pruning only the newest rung keeps its optimizer state (14.1 of the 23 GiB).
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

# ScrewLightbulb only exists on robocasa's `lightbulb_task` branch.
python - "$TASK" <<'ENVCHECK'
import sys
sys.argv[0] = "envcheck"
task = sys.argv[1]
sys.path.insert(0, "examples/training")
from train_pi05_casa_experiments import TASKS
# robocasa registers its envs with robosuite AT IMPORT, so these must come first.
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

# Drop AdamW state from every checkpoint but the newest: resume only ever reads `last`,
# and pruned rungs keep pretrained_model/ so the ladder stays usable for eval and for
# dual-noise selection. SFT_PRUNE_OPTIMIZER_STATE=0 keeps everything.
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
    # The run dir is timestamped by the training script, so discover it once it exists.
    (
        for _ in $(seq 1 60); do
            sleep 30
            # TASK must be in the pattern: coffee and lamp jobs share an arm name, so
            # matching on the arm alone lets one job's pruner latch onto the other's run.
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

TRAIN_RC=0; wait $TRAIN_PID || TRAIN_RC=$?
# Sentinel for scripts/sync_offline_wandb.sh: lets it tell a finished run (sync once, mark
# synced) from a live one. Newest dir for THIS task+arm, same matching rule as the pruner.
if [ "$TRAIN_RC" = "0" ]; then
    RUN=$(find "$OUTROOT" -maxdepth 2 -type d -name "*_${TASK}_${ARM}_*" 2>/dev/null | sort | tail -1)
    [ -n "$RUN" ] && touch "$RUN/.sft_complete" && echo "marked complete: $RUN"
fi
echo "=== done rc=$TRAIN_RC ($(date)) ==="
exit $TRAIN_RC
