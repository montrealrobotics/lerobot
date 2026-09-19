#!/bin/bash
#SBATCH --partition long
#SBATCH -c 8
#SBATCH --mem=56G                 # ~3.4 GB per resident RoboCasa kitchen x 10, plus torch
#SBATCH --gres=gpu:l40s:1         # ONE arm per job: the previous version ran both arms in one
#SBATCH --job-name=sft_sweep_seen
#SBATCH --time=12:00:00

# ─────────────────────────────────────────────────────────────────────────────────────
# Re-score ScrewLightbulb SFT checkpoints on a kitchen that IS in the SFT data, to check
# whether checkpoint selection would have landed somewhere else.
#
# (1,4) is one of the 14 (layout, style) pairs GUARANTEED present in the 33-episode
# filtered dataset (ground truth: regeneration_episodes.json). The original training-time
# curve instead used construction seeds 0,1 = (8,6) and (4,8), BOTH unseen in the SFT data,
# each contributing exactly one object placement -- a 2-placement estimate.
#
# 10 placements x 20 episodes: with the scene pinned, the per-sub-env seed varies object
# pose and robot base, the axis that carries the variance. Extra episodes per placement only
# resample robocasa's 0.02 rad arm-joint reset noise.
#
# ONE ARM PER JOB. The first attempt (job 10767066) backgrounded both arms in a single job
# and died OUT_OF_MEMORY at 47 min: 2 x 10 resident MuJoCo kitchens hit 67 GB RSS against a
# 64 GB request, and both pi05 processes shared one 44 GB GPU. Submit as:
#     sbatch --export=ALL,ARM=cat_true  sft_sweep_seen_env_lightbulb.sbatch
#     sbatch --export=ALL,ARM=cat_false sft_sweep_seen_env_lightbulb.sbatch
#
# NOTE: never put a comment line inside the backslash-continued command below -- bash ends
# the command at the '#', silently dropping every later flag and the redirect.
# ─────────────────────────────────────────────────────────────────────────────────────

set -u
# Positional args (sbatch forwards anything after the script name), with an ARM env fallback:
#   $1 = arm            cat_true | cat_false
#   $2 = steps          comma-separated SFT steps  (default: every 2k from 2k to 20k)
#   $3 = output suffix  appended to the output dir, so a top-up run cannot clobber the log or
#                       the summary JSON of a sweep that is still running
ARM="${1:-${ARM:-}}"
: "${ARM:?Set the arm: sbatch sft_sweep_seen_env_lightbulb.sbatch cat_true}"
STEPS="${2:-2000,4000,6000,8000,10000,12000,14000,16000,18000,20000}"
OUT_SUFFIX="${3:-}"

source ~/.bashrc
cd ~/lerobot
source .venv/bin/activate

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
# Reduce allocator fragmentation across the 10 sequential policy loads.
export PYTORCH_ALLOC_CONF=expandable_segments:True

# RoboCasa hard-asserts numpy==2.2.5; guard against a re-sync to the lockfile's 2.2.6.
uv pip install --quiet numpy==2.2.5

TRAIN=~/scratch/lerobot/outputs/train
case "$ARM" in
    cat_true)  RUN="$TRAIN/2026-08-28/15-26-13_robocasa_pi05" ;;
    cat_false) RUN="$TRAIN/2026-08-31/18-26-31_robocasa_pi05" ;;
    *) echo "Unknown ARM='$ARM' (expected cat_true or cat_false)"; exit 1 ;;
esac

LAYOUT=1
STYLE=4
OUTBASE=~/scratch/lerobot/outputs/sft_sweep_seen/lightbulb/${ARM}${OUT_SUFFIX}
mkdir -p "$OUTBASE"

echo "arm=$ARM  run=$RUN  kitchen=L${LAYOUT}S${STYLE}  steps=$STEPS  out=$OUTBASE"

python scripts/eval_sft_checkpoints_sweep.py \
    --run_dir "$RUN" \
    --layout_id $LAYOUT \
    --style_id $STYLE \
    --steps "$STEPS" \
    --n_placements 10 \
    --n_episodes 20 \
    --n_videos 4 \
    --output_dir "$OUTBASE" \
    --wandb_project lerobot-dex \
    --wandb_name "sftsweep_lightbulb_${ARM}${OUT_SUFFIX}_L${LAYOUT}S${STYLE}" \
    2>&1 | tee "$OUTBASE/sweep.log"

echo "sweep $ARM exited with ${PIPESTATUS[0]}"
