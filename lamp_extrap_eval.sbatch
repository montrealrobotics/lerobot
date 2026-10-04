#!/bin/bash
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/network/scratch/a/artur.kuramshin/lerobot/slurm_logs/%x-%j.out

# Score lamp placements extrapolated past the DSRL training pair (bank built by
# build_lamp_pair_bank.py --extrapolate), to pick the extrapolation cell.
#
#   sbatch --job-name=lamp_extrap_sft  lamp_extrap_eval.sbatch frozen <sft_run> <step> [<step>...]
#   sbatch --job-name=lamp_extrap_X    lamp_extrap_eval.sbatch dsrl <frozen ckpt pretrained_model> <noise actor dir>
#
# frozen: iid + duplicated noise on the extrap cells only (train/mid are already in the run's
#         dual_noise_dsrl_cells.json); results -> <sft_run>/dual_noise_lamp_extrap.json
# dsrl:   the saved noise actor on train,heldout,extrap, the same protocol as the in-loop eval;
#         results -> the log and $OUT/<job-name>.log

set -euo pipefail
MODE=${1:?usage: lamp_extrap_eval.sbatch <frozen|dsrl> ...}
shift

REPO=${LEROBOT_REPO:-/home/mila/a/artur.kuramshin/lerobot}   # override per user: sbatch --export=ALL,LEROBOT_REPO=...
SCRATCH=${LEROBOT_SCRATCH:-/network/scratch/a/artur.kuramshin}
BANK=${LAMP_EXTRAP_BANK:-$SCRATCH/lerobot/placement_banks/screwlightbulb_xarm6_seed1_extrap_candidates/bank.json}
OUT=$SCRATCH/lerobot/eval/lamp_extrap
cd "$REPO"

source ~/.bashrc
source .venv/bin/activate
uv pip install --quiet numpy==2.2.5
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR" "$OUT" "$SCRATCH/lerobot/slurm_logs"

# Controller configs from whichever robosuite the venv imports (the checkout path differs per user).
ROBOSUITE_CONTROLLERS=$(python -c 'import os, robosuite; print(os.path.join(os.path.dirname(robosuite.__file__), "controllers/config/robots"))')
case "$MODE" in
  frozen)
    RUN=$SCRATCH/lerobot/outputs/train/$1
    shift
    python examples/training/eval_pi05_dual_noise.py \
      --run-dir "$RUN" --task lamp --steps "$@" \
      --placement-bank "$BANK" --eval-sets extrap \
      --episodes-per-cell 20 --no-wandb \
      --results-name dual_noise_lamp_extrap.json
    ;;
  dsrl)
    POLICY=${1:?frozen checkpoint}
    ACTOR=${2:?noise actor dir}
    python examples/tutorial/rl/dsrl_pi05_robocasa_example.py \
      --policy_path "$POLICY" \
      --task ScrewLightbulb --robot XArm6DexLeapRHOmron --fps 30 \
      --controller "$ROBOSUITE_CONTROLLERS/default_xarm6dexleaprhomron_joint_pos.json" \
      --collect_envs 1 --collect_scene_seeds 1 \
      --placement_bank "$BANK" \
      --eval_placement_sets train,heldout,extrap \
      --prompt "screw the light bulb into the lamp base" \
      --eval_only --noise_actor_path "$ACTOR" \
      --eval_episodes 10 \
      --output_dir "$OUT/${SLURM_JOB_NAME:-dsrl}" \
      2>&1 | tee "$OUT/${SLURM_JOB_NAME:-dsrl}.log"
    ;;
  *) echo "unknown mode '$MODE' (frozen|dsrl)" >&2; exit 2 ;;
esac
