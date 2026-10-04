#!/bin/bash
# Coffee-machine position calibration (scripts/eval_coffee_distance_calibration.py) -- ALLIANCE/DRAC.
# Mila sibling: coffee_distance_calibration.sbatch. All args forward to the script.
#
#   sbatch --job-name=coffee_lat_arm1_4k coffee_calibration_drac.sbatch \
#       --run-dir <sft_run> --step 4000 --axis lateral --shifts-cm -8 -4 -2 0 4 --modes duplicated
#
# On rorqual override the output path: --output=/scratch/$USER/lerobot/slurm_logs/%x-%j.out
#SBATCH --account=def-kdarvish
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000M
#SBATCH --time=03:00:00
#SBATCH --output=/home/akur/scratch/lerobot/slurm_logs/%x-%j.out

set -euo pipefail

case "${SLURM_CLUSTER_NAME:-$(hostname)}" in
  nibi*)
    REPO=/project/6068036/akur/lerobot
    HF_HOME_DIR=/scratch/akur/hf_cache
    OFFLINE=0 ;;
  rorqual*)
    REPO=/lustre09/project/6068036/akur/lerobot
    HF_HOME_DIR=/scratch/$USER/hf_cache
    OFFLINE=1 ;;
  *) echo "unknown cluster '${SLURM_CLUSTER_NAME:-$(hostname)}'" >&2; exit 2 ;;
esac
cd "$REPO"

module load StdEnv/2023 python/3.12.4
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HOME=$HF_HOME_DIR
if [ "$OFFLINE" = "1" ]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export __GL_SHADER_DISK_CACHE=0
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR"

GL_EXPORTS=$(python scripts/check_gl_backend.py --emit-exports 2>/dev/null) && [ -n "$GL_EXPORTS" ] \
    || { echo "no working headless GL backend on $(hostname)" >&2; exit 1; }
eval "$GL_EXPORTS"

python scripts/eval_coffee_distance_calibration.py \
    --scene-seed 1 --episodes-per-cell 25 \
    --output-dir /scratch/$USER/lerobot/eval/coffee_distance_calibration \
    "$@"
