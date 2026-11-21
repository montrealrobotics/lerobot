#!/bin/bash
# pi05 SFT experiment arm, through cluv. Same run as sft_arms_mila_cloutiec.sbatch and
# sft_arms.sbatch: everything after <arm> <task> goes to the training script, which holds the
# fairness invariants. Only cluster plumbing differs (scripts/cluv/common.sh).
#
#   cluv submit fir scripts/cluv/sft.sh --gpus=h100:1 --job-name=sft_fp32_arm0_lamp \
#       -- arm0_expert_only lamp --dtype float32 --save-steps 500 1000 2000 4000 8000 --stop-step 8000
#   cluv submit fir scripts/cluv/sft.sh --gpus=h100:1 --time=0:30:00 -- arm0_expert_only lamp --smoke-test
#
# Outputs: $SCRATCH/lerobot/outputs/train/<run>, as on Mila. Pass a full GPU (eval renders):
# --gpus=h100:1 (fir, nibi) or --gpus=a100:1 (narval); on Mila, --gres=gpu:a100:1 --partition=long.
#
#SBATCH --cpus-per-task=12
# Training peaks at ~46 GB, and the eval envs fork workers that each take an EGL context.
#SBATCH --mem=64G
#SBATCH --time=24:00:00

ARM=${1:?usage: sft.sh <arm> <task> [extra args...]}
TASK=${2:?usage: sft.sh <arm> <task> [extra args...]}
shift 2

source scripts/cluv/common.sh
echo "arm=$ARM task=$TASK extra=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

NEEDS_GL=1
case " $* " in *" --eval-freq 0 "*|*" --eval-freq=0 "*) NEEDS_GL=0 ;; esac
probe_gl || [ "$NEEDS_GL" = "0" ] || exit 1
preflight

# ScrewLightbulb only exists on robocasa's lightbulb_task branch.
python - "$TASK" <<'ENVCHECK'
import sys
sys.argv[0] = "envcheck"
task = sys.argv[1]
sys.path.insert(0, "examples/training")
from train_pi05_casa_experiments import TASKS
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()
import robocasa  # noqa: F401
from robosuite.environments.base import REGISTERED_ENVS
name = TASKS[task].robocasa_task
if name not in REGISTERED_ENVS:
    raise SystemExit(f"robocasa env {name!r} is not registered (needs robocasa lightbulb_task)")
print(f"robocasa env {name} registered OK")
ENVCHECK

# Drop AdamW state from every checkpoint but the newest: resume only reads the last one, and
# the pruned rungs keep pretrained_model/ for eval and DSRL. SFT_PRUNE_OPTIMIZER_STATE=0 keeps all.
OUTROOT=$LEROBOT_OUTPUT_ROOT
PRUNE=${SFT_PRUNE_OPTIMIZER_STATE:-1}
prune_loop () {
    local root="$1" latest d
    while true; do
        sleep 300
        [ -d "$root" ] || continue
        latest=$(find "$root" -maxdepth 1 -type d -regex '.*/[0-9]+' -printf '%f\n' 2>/dev/null | sort -n | tail -1)
        [ -n "$latest" ] || continue
        for d in "$root"/[0-9]*; do
            [ -d "$d" ] && [ "$(basename "$d")" != "$latest" ] && [ -d "$d/training_state" ] || continue
            rm -rf "$d/training_state"
            echo "[prune] dropped optimizer state from $(basename "$d") (keeping $latest)"
        done
    done
}

echo "=== training ($(date)) ==="
python examples/training/train_pi05_casa_experiments.py --experiment "$ARM" --task "$TASK" "$@" &
TRAIN_PID=$!

if [ "$PRUNE" = "1" ]; then
    (
        RUN=
        for _ in $(seq 1 60); do
            sleep 30
            # TASK must be in the pattern: coffee and lamp jobs share an arm name.
            RUN=$(find "$OUTROOT" -maxdepth 2 -type d -newermt '-15 minutes' -name "*_${TASK}_${ARM}_*" 2>/dev/null | sort | tail -1)
            [ -n "$RUN" ] && break
        done
        if [ -n "$RUN" ]; then
            echo "[prune] watching $RUN/checkpoints"
            prune_loop "$RUN/checkpoints"
        else
            echo "[prune] WARNING: could not locate the run directory; pruning disabled"
        fi
    ) &
    PRUNE_PID=$!
    trap 'kill $PRUNE_PID 2>/dev/null || true' EXIT
fi

TRAIN_RC=0; wait $TRAIN_PID || TRAIN_RC=$?
if [ "$TRAIN_RC" = "0" ]; then
    RUN=$(find "$OUTROOT" -maxdepth 2 -type d -name "*_${TASK}_${ARM}_*" 2>/dev/null | sort | tail -1)
    [ -n "$RUN" ] && touch "$RUN/.sft_complete" && echo "marked complete: $RUN"
fi
echo "=== done rc=$TRAIN_RC ($(date)) ==="
exit $TRAIN_RC
