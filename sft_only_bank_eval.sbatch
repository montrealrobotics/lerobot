#!/bin/bash
#SBATCH --partition long
#SBATCH -c 8
#SBATCH --mem=56G                 # lamp: 10 resident kitchens (one per eval cell)
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=sft_only_bank_eval

# ─────────────────────────────────────────────────────────────────────────────────────
# Frozen-SFT pi05 (no DSRL) on the exact cells of the randomization experiments, for the frozen
# checkpoints every DSRL arm used. Answers whether sparse-reward DSRL has any successes to learn
# from on those starts, and whether DSRL beats or hurts the frozen policy there.
# Protocol matches the DSRL evals: lamp 5 episodes/cell, coffee 20 episodes/cell with start-pose
# rejection; cameras = agentview_left + eye_in_hand. See scripts/eval_sft_on_banks.py.
#
# Usage: sbatch sft_only_bank_eval.sbatch <lamp_cat_true|lamp_cat_false|coffee_cat_true|coffee_cat_false>
# ─────────────────────────────────────────────────────────────────────────────────────

set -uo pipefail
ARM="${1:?lamp_cat_true | lamp_cat_false | coffee_cat_true | coffee_cat_false}"

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
BANKS=~/scratch/lerobot/placement_banks
LAMP_BANK="--placement_bank $BANKS/screwlightbulb_xarm6_seed01/bank.json --episodes_per_cell 5"
COFFEE_BANK="--kitchen_bank $BANKS/coffeepressbutton_pandadex_kitchens/bank.json --episodes_per_cell 20 --reject_start_rot_deg 10 --reject_start_pos_cm 3"
case "$ARM" in
    lamp_cat_true)    RUN="$TRAIN/2026-08-28/15-26-13_robocasa_pi05"; STEPS="3000,7000"; BANK_ARGS="$LAMP_BANK" ;;
    lamp_cat_false)   RUN="$TRAIN/2026-08-31/18-26-31_robocasa_pi05"; STEPS="2000,8000"; BANK_ARGS="$LAMP_BANK" ;;
    coffee_cat_true)  RUN="$TRAIN/2026-09-04/08-14-09_robocasa_pi05"; STEPS="4000,8000"; BANK_ARGS="$COFFEE_BANK" ;;
    coffee_cat_false) RUN="$TRAIN/2026-09-03/21-19-54_robocasa_pi05"; STEPS="2000,8000"; BANK_ARGS="$COFFEE_BANK" ;;
    *) echo "unknown ARM=$ARM"; exit 1 ;;
esac

OUT=~/scratch/lerobot/outputs/sft_only_bank_eval/${ARM}
mkdir -p "$OUT"
# shellcheck disable=SC2086  # BANK_ARGS is intentionally word-split into flags
NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID} TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID} \
python scripts/eval_sft_on_banks.py --sft_run "$RUN" --steps "$STEPS" $BANK_ARGS \
    --output_dir "$OUT" --wandb_project lerobot-dex --wandb_name "sft_only_bankeval_${ARM}" \
    2>&1 | tee "$OUT/eval.log"
