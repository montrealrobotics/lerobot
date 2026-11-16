#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
# DexJoCo pi05 SFT through cluv, on whichever cluster and allocation starts it first:
#   cluv submit first cluv_dexjoco_train.sh -- <arm> <task> <run name> [train flags]
# The run lives in $SCRATCH/dexjoco-runs/<run name>. A requeued job or the next `--chunking` chunk finds it there and
# resumes from its last checkpoint.
set -euo pipefail
ARM=$1 TASK=$2 NAME=$3
shift 3
source dexjoco_env.sh

RUN=$SCRATCH/dexjoco-runs/$NAME
run_dir=(--output-dir "$RUN")
if [ -d "$RUN/checkpoints/last/training_state" ]; then
    run_dir=(--resume-from "$RUN")
    echo "resuming $RUN from checkpoint $(basename "$(readlink -f "$RUN/checkpoints/last")")"
elif [ -e "$RUN" ]; then
    echo "$RUN has no resumable checkpoint yet; moving it aside and starting over"
    mv "$RUN" "$RUN.no-checkpoint-$(date +%s)"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

prune_loop "$RUN/checkpoints" &
trap 'kill $! 2>/dev/null || true' EXIT
python examples/training/train_pi05_casa_experiments.py --experiment "$ARM" --task "$TASK" "${run_dir[@]}" "$@"
