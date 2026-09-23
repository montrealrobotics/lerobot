#!/bin/bash
# DSRL over a frozen pi05 SFT checkpoint -- MILA. Design: SFT_ARMS_DSRL.md.
#
#   for S in 0 1 2; do
#     sbatch --job-name=dsrl_arm0_lamp_s$S dsrl_mila.sbatch lamp <ckpt>/pretrained_model $S
#   done
#
# Override the GPU at submit time: sbatch --gres=gpu:rtx8000:1 dsrl_mila.sbatch ...
#
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@300
#SBATCH --output=/network/scratch/a/artur.kuramshin/lerobot/slurm_logs/%x-%j.out

set -euo pipefail

TASK=${1:?usage: sbatch dsrl_mila.sbatch <lamp|coffee> <policy_path> <seed> [extra args...]}
POLICY=${2:?usage: sbatch dsrl_mila.sbatch <lamp|coffee> <policy_path> <seed> [extra args...]}
SEED=${3:?usage: sbatch dsrl_mila.sbatch <lamp|coffee> <policy_path> <seed> [extra args...]}
shift 3

REPO=/home/mila/a/artur.kuramshin/lerobot
SCRATCH=/network/scratch/a/artur.kuramshin
BANKS=$SCRATCH/lerobot/placement_banks
cd "$REPO"

source ~/.bashrc
source .venv/bin/activate
uv pip install --quiet numpy==2.2.5

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR" "$SCRATCH/lerobot/slurm_logs"

# Task invariants. Controller and fps must match what the checkpoint was trained on: a wrong
# controller silently falls back to OSC_POSE and every rollout is garbage. --prompt is the
# dataset's task string, used only if the env exposes no ep_meta['lang'] (the script's own
# default is a coffee string, wrong for both tasks).
case "$TASK" in
  lamp)
    TASK_ARGS=(
      --task ScrewLightbulb --robot XArm6DexLeapRHOmron --fps 30
      --controller "$HOME/robosuite/robosuite/controllers/config/robots/default_xarm6dexleaprhomron_joint_pos.json"
      --collect_envs 1 --collect_scene_seeds 1
      --placement_bank "$BANKS/screwlightbulb_xarm6_seed1_pair/bank.json"
      --eval_placement_sets train,heldout
      --prompt "screw the light bulb into the lamp base"
    )
    ;;
  coffee)
    TASK_ARGS=(
      --task CoffeePressButton --robot PandaDexLeapRHOmron --fps 20
      --collect_envs 1
      --kitchen_bank "$BANKS/coffeepressbutton_pandadex_kitchens/bank.json"
      --eval_kitchen_sets heldout,reference
      --reject_start_rot_deg 10 --reject_start_pos_cm 3
      --prompt "press the button on the coffee machine to serve coffee"
    )
    ;;
  *) echo "unknown task '$TASK' (lamp|coffee)" >&2; exit 2 ;;
esac

OUT=$SCRATCH/lerobot/outputs/dsrl/${SLURM_JOB_NAME:-dsrl_${TASK}}/seed${SEED}

echo "=== $(date) job=${SLURM_JOB_ID:-none} node=$(hostname) task=$TASK seed=$SEED"
echo "policy=$POLICY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -d "$POLICY" ] || { echo "no such checkpoint: $POLICY" >&2; exit 2; }

# Preemption / wall-clock: the trainer writes <output_dir>/resume_state.pt on SIGTERM and exits
# 99. Slurm signals the step directly on preemption; the trap covers --signal=B:TERM@300, which
# only reaches the batch script. --open-mode=append keeps the requeued run's log.
# NOT via srun: `kill -TERM <srun pid>` makes srun CANCEL the step and SIGKILL the task
# (verified: sacct showed "CANCELLED ... 0:9"), so the trainer never sees SIGTERM. Running
# python as a direct child of the batch script means the trap reaches it, and on preemption
# Slurm's SIGTERM hits every process in the cgroup, including this one.
python examples/tutorial/rl/dsrl_pi05_robocasa_example.py \
    --policy_path "$POLICY" \
    "${TASK_ARGS[@]}" \
    --utd_ratio 20 \
    --total_steps 500000 \
    --eval_freq 25000 --eval_episodes 5 \
    --seed "$SEED" \
    --output_dir "$OUT" \
    --resume \
    --wandb_enable --wandb_name "${SLURM_JOB_NAME:-dsrl_${TASK}}_seed${SEED}" \
    "$@" &
PY_PID=$!
trap 'echo "[sbatch] signal received, forwarding to $PY_PID"; kill -TERM $PY_PID 2>/dev/null || true' TERM

# `wait` must not trip `set -e`, and it returns 128+signal when the trap interrupts it rather
# than when the child exits -- so wait a second time for the real status.
RC=0; wait $PY_PID || RC=$?
if [ "$RC" -gt 128 ]; then RC=0; wait $PY_PID || RC=$?; fi

if [ "$RC" = "99" ]; then
    echo "[sbatch] checkpointed at the signal; requeueing ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi
echo "=== done rc=$RC ($(date)) ==="
exit $RC
