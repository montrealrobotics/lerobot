#!/bin/bash
#SBATCH --partition long
#SBATCH -c 12
#SBATCH --mem=64G                 # 12 resident kitchens (10 train + 2 held-out, 1 env each)
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=steer_coffee_kitchens

# ─────────────────────────────────────────────────────────────────────────────────────
# Best-of-K steerability of FROZEN CoffeePressButton SFT checkpoints over the KITCHEN BANK -- the
# start distribution the kitchen-randomized DSRL runs trained on, and where their held-out results
# vary while vanilla SFT mean success ranks them backwards.
#
# Starts: the bank's 10 training kitchens (group "train") + 2 held-out kitchens ("heldout"), with
# the same start-pose rejection those runs used. Noise per chunk from DSRL's warmup sampler:
#   MODE=dsrl  --noise_chunk_size 1   one vector held across the chunk (the space DSRL steers in)
#   MODE=iid   --noise_chunk_size 32  fresh noise per timestep (vanilla pi05; control)
# 32 rollouts per kitchen -> per-kitchen and per-group mean success and pass@k.
#
# steerability_eval_coffee.sbatch is the fixed-kitchen (seeds 0,1) variant of the same measurement.
#
# Usage: sbatch steerability_eval_coffee_kitchenbank.sbatch <cat_true_4k|cat_true_8k|cat_false_2k|cat_false_8k> <dsrl|iid>
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

OUT=~/scratch/lerobot/outputs/steerability/coffee_kitchenbank/${ARM}
mkdir -p "$OUT"
NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID} TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID} \
python examples/tutorial/rl/dsrl_pi05_robocasa_example.py --steer_eval \
    --policy_path "$SFT/pretrained_model" \
    --task CoffeePressButton \
    --robot PandaDexLeapRHOmron \
    --reward_mode goal \
    --kitchen_bank ~/scratch/lerobot/placement_banks/coffeepressbutton_pandadex_kitchens/bank.json \
    --reject_start_rot_deg 10 \
    --reject_start_pos_cm 3 \
    --noise_chunk_size "$NC" \
    --steer_rollouts_per_start 32 \
    --steer_parallel_per_start 1 \
    --seed 0 \
    --output_dir "$OUT" \
    --wandb_enable \
    --wandb_project lerobot-dex \
    --wandb_name "steer_coffee_kitchenbank_${ARM}_${MODE}" \
    2>&1 | tee "$OUT/steer_${MODE}.log"
