#!/bin/bash
#SBATCH --partition long
#SBATCH -c 12
#SBATCH --mem=64G                 # 16 resident kitchens (8 parallel rollouts x 2 starts)
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=steer_coffee

# ─────────────────────────────────────────────────────────────────────────────────────
# Best-of-K steerability of FROZEN CoffeePressButton SFT checkpoints (inference only).
#
# Every fixed-kitchen coffee DSRL run succeeded, so there is no failure to predict. The question
# here is sample efficiency: steps to first reach 0.8 success were 2k(cat=False) 70k, 4k(cat=True)
# 120k, 8k(cat=False) 200k, 8k(cat=True) 210k, while vanilla SFT mean success on the same kitchens
# (0.33 / 0.62 / 0.47 / 0.45) does not order them. Does success under DSRL's own noise?
#
# Starts and settings match the fixed-kitchen coffee DSRL runs exactly (their wandb argv): kitchens
# = construction seeds 0,1, script defaults for fps (20), controller and cameras, and NO start-pose
# rejection (those runs trained without it). Noise as in steerability_eval_lightbulb.sbatch:
#   MODE=dsrl  --noise_chunk_size 1   one randn vector held across the chunk (DSRL's noise space)
#   MODE=iid   --noise_chunk_size 32  fresh noise per timestep (vanilla pi05; control)
# Expect pass@64 ~= 1 everywhere (baseline success is 0.3-0.6); the signal is mean / low-k pass@k.
#
# Usage: sbatch steerability_eval_coffee.sbatch <cat_true_4k|cat_true_8k|cat_false_2k|cat_false_8k> <dsrl|iid>
# ─────────────────────────────────────────────────────────────────────────────────────

set -uo pipefail
ARM="${1:?arm}"
MODE="${2:?dsrl | iid}"

source ~/.bashrc
cd ~/lerobot
source .venv/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
if ! ls /cvmfs/ai.mila.quebec >/dev/null 2>&1; then
    echo "WARNING: /cvmfs unreadable on $(hostname); removing it from PATH"
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v cvmfs | paste -sd:)
fi
uv pip install --quiet numpy==2.2.5

TRAIN=~/scratch/lerobot/outputs/train
case "$ARM" in
    cat_true_4k)  SFT="$TRAIN/2026-09-04/08-14-09_robocasa_pi05/checkpoints/004000" ;;
    cat_true_8k)  SFT="$TRAIN/2026-09-04/08-14-09_robocasa_pi05/checkpoints/008000" ;;
    cat_false_2k) SFT="$TRAIN/2026-09-03/21-19-54_robocasa_pi05/checkpoints/002000" ;;
    cat_false_8k) SFT="$TRAIN/2026-09-03/21-19-54_robocasa_pi05/checkpoints/008000" ;;
    *) echo "unknown ARM=$ARM"; exit 1 ;;
esac
case "$MODE" in
    dsrl) NC=1 ;;
    iid)  NC=32 ;;
    *) echo "unknown MODE=$MODE"; exit 1 ;;
esac
[ -f "$SFT/pretrained_model/model.safetensors" ] || { echo "missing checkpoint $SFT"; exit 1; }

OUT=~/scratch/lerobot/outputs/steerability/coffee/${ARM}
mkdir -p "$OUT"
NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID} TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID} \
python examples/tutorial/rl/dsrl_pi05_robocasa_example.py --steer_eval \
    --policy_path "$SFT/pretrained_model" \
    --task CoffeePressButton \
    --robot PandaDexLeapRHOmron \
    --reward_mode goal \
    --eval_seeds 0,1 \
    --noise_chunk_size "$NC" \
    --steer_rollouts_per_start 64 \
    --steer_parallel_per_start 8 \
    --seed 0 \
    --output_dir "$OUT" \
    --wandb_enable \
    --wandb_project lerobot-dex \
    --wandb_name "steer_coffee_${ARM}_${MODE}" \
    2>&1 | tee "$OUT/steer_${MODE}.log"
