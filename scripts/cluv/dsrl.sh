#!/bin/bash
# DSRL over a frozen pi05 SFT checkpoint, through cluv. Same run as dsrl_mila_cloutiec.sbatch and
# dsrl_drac.sbatch: task arguments come from dsrl_task_args.sh, and the hyperparameters below are
# copied from those launchers. Only cluster plumbing differs (scripts/cluv/common.sh).
#
#   cluv submit fir scripts/cluv/dsrl.sh --gpus=h100:1 --job-name=dsrl_fp32_1k_lamp_s0 \
#       -- lamp fp32_expert_only_lamp_001000 0
#
# <policy> is a pretrained_model directory: an absolute path on the cluster, or a name under
# $DEXROBOT_DATA/checkpoints/ (pushed from Mila with the rest of the data). Arguments after the
# seed go to the training script. Outputs: $SCRATCH/lerobot/outputs/dsrl/<job-name>/seed<seed>,
# as on Mila. Pass a full GPU: --gpus=h100:1 (fir, nibi) or --gpus=a100:1 (narval); on Mila,
# --gres=gpu:l40s:1 --partition=long.
#
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@300

# A 500k-step run is ~10 h of training plus ~3.6 h of eval. A job that hits the wall clock writes
# resume_state.pt and requeues (see the end of this script).

TASK=${1:?usage: dsrl.sh <lamp|coffee|coffee_lat> <policy> <seed> [extra args...]}
POLICY=${2:?usage: dsrl.sh <lamp|coffee|coffee_lat> <policy> <seed> [extra args...]}
SEED=${3:?usage: dsrl.sh <lamp|coffee|coffee_lat> <policy> <seed> [extra args...]}
shift 3

source scripts/cluv/common.sh

case "$POLICY" in
    /*) ;;
    *) POLICY=$DEXROBOT_DATA/checkpoints/$POLICY ;;
esac
[ -d "$POLICY" ] && [ -f "$POLICY/config.json" ] || { echo "no checkpoint at $POLICY" >&2; exit 2; }
echo "task=$TASK seed=$SEED policy=$POLICY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

probe_gl || exit 1
preflight

# Expects $TASK and $BANKS; sets $TASK_ARGS. Shared with the DRAC launchers.
source dsrl_task_args.sh

OUT=$SCRATCH/lerobot/outputs/dsrl/$JOB_NAME/seed$SEED
mkdir -p "$OUT"

STARTED=$(mktemp)  # timestamp: was resume state written during this job?
# Preemption / wall clock: the trainer writes <output_dir>/resume_state.pt on SIGTERM and exits
# 99. Not via srun: `kill -TERM <srun pid>` makes srun cancel the step and SIGKILL the task, so
# python must be a direct child of this script for the trap to reach it.
python examples/tutorial/rl/dsrl_pi05_robocasa_example.py \
    --policy_path "$POLICY" \
    "${TASK_ARGS[@]}" \
    --utd_ratio 20 \
    --total_steps 500000 \
    --eval_freq 25000 --eval_episodes 5 \
    --seed "$SEED" \
    --output_dir "$OUT" \
    --resume --resume_save_freq 100000 \
    --wandb_enable --wandb_name "${JOB_NAME}_seed${SEED}" \
    "$@" &
PY_PID=$!
trap 'echo "[job] signal received, forwarding to $PY_PID"; kill -TERM $PY_PID 2>/dev/null || true' TERM

# `wait` returns 128+signal when the trap interrupts it, so wait again for the real status.
RC=0; wait $PY_PID || RC=$?
if [ "$RC" -gt 128 ]; then RC=0; wait $PY_PID || RC=$?; fi

if [ "$RC" = "99" ]; then
    echo "[job] checkpointed at the signal; requeueing ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi
# SIGKILL (OOM, node failure) leaves no chance to checkpoint. Requeue only if the periodic dump
# advanced during this job, so a run that dies before its first dump can't loop forever.
if [ "$RC" = "137" ] && [ "$OUT/resume_state.pt" -nt "$STARTED" ]; then
    echo "[job] killed (rc=137) after a periodic resume dump; requeueing ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi
echo "=== done rc=$RC ($(date)) ==="
exit $RC
