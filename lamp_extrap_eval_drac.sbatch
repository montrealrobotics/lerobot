#!/bin/bash
# Score one lamp DSRL rung's three saved noise actors (seeds 0-2) on the pair bank's train,
# heldout and extrap cells -- ALLIANCE/DRAC. Mila sibling: lamp_extrap_eval.sbatch.
#
#   sbatch --job-name=lamp_extrap_arm2_4000 lamp_extrap_eval_drac.sbatch <sft_run_dir> <arm> <step>
#
# Actors: /scratch/$USER/lerobot/outputs/dsrl/dsrl_<arm>_lamp_<step %05d>/step<step>_seed<S>/final
# Results: /scratch/$USER/lerobot/eval/lamp_extrap/dsrl_<arm>_<step>_s<S>.log
#
# On rorqual override the output path at submit time:
#   sbatch --output=/scratch/$USER/lerobot/slurm_logs/%x-%j.out lamp_extrap_eval_drac.sbatch ...
#SBATCH --account=def-kdarvish
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48000M
#SBATCH --time=03:00:00
#SBATCH --output=/home/akur/scratch/lerobot/slurm_logs/%x-%j.out

set -euo pipefail

RUN_DIR=${1:?usage: sbatch lamp_extrap_eval_drac.sbatch <sft_run_dir> <arm> <step>}
ARM=${2:?arm, e.g. arm2}
STEP=${3:?SFT step, e.g. 4000}

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

SCRATCH=/scratch/$USER/lerobot
BANKS=$SCRATCH/placement_banks
TASK=lamp
source "$REPO/dsrl_task_args.sh"   # sets TASK_ARGS for the pair bank; overridden below
BANK=$BANKS/screwlightbulb_xarm6_seed1_pair_extrap/bank.json
[ -f "$BANK" ] || { echo "no bank at $BANK -- rsync it from Mila" >&2; exit 2; }

POLICY=$RUN_DIR/checkpoints/$(printf %06d "$STEP")/pretrained_model
OUT=$SCRATCH/eval/lamp_extrap
mkdir -p "$OUT"
[ -d "$POLICY" ] || { echo "no checkpoint $POLICY" >&2; exit 2; }

for S in 0 1 2; do
    ACTOR=$SCRATCH/outputs/dsrl/dsrl_${ARM}_lamp_$(printf %05d "$STEP")/step${STEP}_seed${S}/final
    if [ ! -d "$ACTOR" ]; then echo "MISSING actor $ACTOR" | tee "$OUT/dsrl_${ARM}_${STEP}_s${S}.log"; continue; fi
    echo "=== $(date) $ARM $STEP seed $S  actor=$ACTOR"
    # Later flags override TASK_ARGS' bank and eval sets (argparse: last wins).
    python examples/tutorial/rl/dsrl_pi05_robocasa_example.py \
        --policy_path "$POLICY" "${TASK_ARGS[@]}" \
        --placement_bank "$BANK" --eval_placement_sets train,heldout,extrap \
        --eval_only --noise_actor_path "$ACTOR" --eval_episodes 10 \
        --output_dir "$OUT/dsrl_${ARM}_${STEP}_s${S}" \
        2>&1 | tee "$OUT/dsrl_${ARM}_${STEP}_s${S}.log"
done
echo "=== done $(date)"
