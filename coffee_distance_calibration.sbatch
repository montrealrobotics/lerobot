#!/bin/bash
#SBATCH --partition long
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=6:00:00
#SBATCH --job-name=coffee_dist_cal
#SBATCH --output=%x-%j.out

# Frozen-SFT success vs coffee-machine distance in kitchen seed 1 (L4S8, nespresso), iid and
# duplicated noise. Sets the train / held-out shifts of the fixed-kitchen coffee DSRL bank.
#   sbatch coffee_distance_calibration.sbatch <run_name> <step> [extra args]
#   sideways: ... <run_name> <step> --axis lateral --shifts-cm -8 -6 -4 -2 0 2 4 6 8
# One resident copy of the kitchen per shift (~3.4G each) -> 80G covers 9.

set -euo pipefail
source ~/.bashrc
cd ~/lerobot

uv pip install numpy==2.2.5
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$

SCRATCH=/network/scratch/a/artur.kuramshin
RUN=$SCRATCH/lerobot/outputs/train/$1
STEP=$2
shift 2
OUT=$SCRATCH/lerobot/eval/coffee_distance_calibration

uv run --no-sync python scripts/eval_coffee_distance_calibration.py \
  --run-dir $RUN --step $STEP \
  --scene-seed 1 --shifts-cm 0 2 4 6 8 10 12 14 \
  --episodes-per-cell 25 \
  --output-dir $OUT \
  "$@"
