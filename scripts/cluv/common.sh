# Shared setup for the cluv job scripts in this directory -- sourced, never executed.
#
# cluv starts every job in the project checkout (sbatch --chdir), with the [tool.cluv.env]
# variables from pyproject.toml exported. Data comes from $DEXROBOT_DATA, the copy of Mila's
# /network/scratch/c/cloutiec/dexrobot_data that `cluv sync --sync-datasets` pushes to every
# cluster (`datasets_path` in pyproject.toml):
#   hf/                 HF_HOME: pi05_base, the PaliGemma tokenizer, the two LeRobot datasets
#   placement_banks/    the banks the Mila launchers read from $SCRATCH/lerobot/placement_banks
#   robocasa_assets/    objaverse, textures, generative_textures (symlinked into robocasa)
#   robocasa_fixtures/  fixture meshes git doesn't track (copied into robocasa once)
#   checkpoints/        SFT checkpoints for DSRL, by name (see dsrl.sh)
#
# Only cluster plumbing lives here. Task arguments, seeds and step counts stay in the training
# scripts and dsrl_task_args.sh, shared with the Mila launchers, so runs stay comparable.

set -euo pipefail

: "${SCRATCH:?SCRATCH is not set}"
export DEXROBOT_DATA=${DEXROBOT_DATA:-$SCRATCH/dexrobot_data}
[ -d "$DEXROBOT_DATA/hf" ] || {
    echo "no data at $DEXROBOT_DATA -- run 'cluv sync --sync-datasets <cluster>' from Mila" >&2
    exit 2
}

# The venv `cluv sync` built with `uv sync`. Activated directly rather than through `uv run`, so
# python is a direct child of the batch script and the SIGTERM trap in dsrl.sh reaches it.
# shellcheck disable=SC1091
source .venv/bin/activate

bash scripts/cluv/link_data.sh

# Everything the jobs read from the Hub is in $DEXROBOT_DATA/hf, so stay offline even on clusters
# with internet: no token needed (PaliGemma is gated), and no silent re-download of a newer
# pi05_base revision.
export HF_HOME=$DEXROBOT_DATA/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export BANKS=$DEXROBOT_DATA/placement_banks
export LEROBOT_LAMP_PLACEMENT_BANK=$BANKS/screwlightbulb_xarm6_seed1/bank.json
export LEROBOT_OUTPUT_ROOT=${LEROBOT_OUTPUT_ROOT:-$SCRATCH/lerobot/outputs/train}
# wandb defaults to ./wandb in the checkout, which is in $HOME (50G quota on DRAC).
export WANDB_DIR=${WANDB_DIR:-$SCRATCH/lerobot/wandb}
mkdir -p "$WANDB_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
# The NVIDIA driver caches compiled shaders under ~/.cache/nvidia/GLCache, on a filesystem shared
# by every node. Concurrent EGL processes race on it and the driver aborts (SIGABRT, exit 134).
export __GL_SHADER_DISK_CACHE=0
# Node-local, per-job caches: concurrent jobs corrupt shared ones ("EOFError: Ran out of input").
_TMP=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
export NUMBA_CACHE_DIR=$_TMP/numba_${SLURM_JOB_ID:-$$}
export TRITON_CACHE_DIR=$_TMP/triton_${SLURM_JOB_ID:-$$}
export TORCHINDUCTOR_CACHE_DIR=$_TMP/inductor_${SLURM_JOB_ID:-$$}
mkdir -p "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# cluv names jobs cluv-<name>; outputs and wandb runs use <name>.
JOB_NAME=${SLURM_JOB_NAME:-local}
JOB_NAME=${JOB_NAME#cluv-}

echo "=== $(date) cluster=${CLUV_CLUSTER:-?} job=${SLURM_JOB_ID:-none} node=$(hostname) commit=${GIT_COMMIT:-?}"
echo "data=$DEXROBOT_DATA wandb=${WANDB_MODE:-online}"

# Pick a headless GL backend by probing rather than assuming: MIG slices can't do EGL at all.
# Sets MUJOCO_GL / PYOPENGL_PLATFORM / MUJOCO_EGL_DEVICE_ID, or returns 1.
probe_gl () {
    local exports
    if [ -n "${FORCE_MUJOCO_GL:-}" ]; then
        export MUJOCO_GL=$FORCE_MUJOCO_GL PYOPENGL_PLATFORM=$FORCE_MUJOCO_GL
        echo "GL backend forced: MUJOCO_GL=$MUJOCO_GL"
        return 0
    fi
    if exports=$(python scripts/check_gl_backend.py --emit-exports 2>/dev/null) && [ -n "$exports" ]; then
        eval "$exports"
        echo "GL backend: MUJOCO_GL=$MUJOCO_GL MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-n/a}"
        return 0
    fi
    echo "no working headless GL backend on this node (python scripts/check_gl_backend.py for" >&2
    echo "details). Request a FULL GPU: --gpus=h100:1 (fir, nibi) or --gpus=a100:1 (narval)." >&2
    return 1
}

# numpy pin, CUDA, wandb credentials when logging online. Exits on failure.
preflight () {
    python - <<'PREFLIGHT'
import os, sys
fail = []
import numpy
if numpy.__version__ != "2.2.5":
    fail.append(f"numpy is {numpy.__version__}, RoboCasa asserts 2.2.5")
if os.environ.get("WANDB_MODE", "online") != "offline":
    import netrc
    try:
        has_netrc = "api.wandb.ai" in netrc.netrc(os.path.expanduser("~/.netrc")).hosts
    except Exception:
        has_netrc = False
    if not os.environ.get("WANDB_API_KEY") and not has_netrc:
        fail.append("wandb is online here but has no credentials: run `wandb login` on this "
                    "cluster's login node once")
import torch
if not torch.cuda.is_available():
    fail.append("CUDA not available (did you pass --gpus=...?)")
else:
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
if fail:
    print("PREFLIGHT FAILED:\n  - " + "\n  - ".join(fail))
    sys.exit(1)
print("preflight OK")
PREFLIGHT
}
