#!/bin/bash
#   sbatch --job-name=sft_arm0_water_plant sft_arms_dexjoco_mila.sbatch arm0_expert_only water_plant
#   sbatch --job-name=smoke sft_arms_dexjoco_mila.sbatch arm0_expert_only water_plant --smoke-test
#
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/network/scratch/a/artur.kuramshin/lerobot/slurm_logs/%x-%j.out
#SBATCH --signal=B:TERM@300

set -euo pipefail

ARM=${1:?usage: sbatch sft_arms_dexjoco_mila.sbatch <arm> <task> [extra args...]}
TASK=${2:?usage: sbatch sft_arms_dexjoco_mila.sbatch <arm> <task> [extra args...]}
shift 2

REPO=/network/scratch/a/artur.kuramshin/lerobot-sft-dexjoco
OUTROOT=/network/scratch/a/artur.kuramshin/lerobot/outputs/train
cd "$REPO"
source dexjoco_env.sh

echo "=== preflight ($(date)) ==="
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) arm=$ARM task=$TASK extra=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUTROOT="$OUTROOT" python - <<'PREFLIGHT'
import os, shutil, sys

fail = []

# google/paligemma-3b-pt-224 is a GATED repo; without a token the run dies late.
from huggingface_hub import get_token
if not get_token():
    fail.append(f"no HF token visible (HF_HOME={os.environ.get('HF_HOME')})")

import netrc
has_netrc = False
try:
    has_netrc = "api.wandb.ai" in netrc.netrc(os.path.expanduser("~/.netrc")).hosts
except Exception:
    pass
if not os.environ.get("WANDB_API_KEY") and not has_netrc:
    fail.append("no wandb credentials (set WANDB_API_KEY or run `wandb login`)")

# Mila scratch has no diskusage_report, so read the filesystem. Projection is full-FT
# sizing (23 GiB/rung), an upper bound for arm0's ~14 GiB. See SFT_ARMS_RUNBOOK.md.
out = os.environ["OUTROOT"]
os.makedirs(out, exist_ok=True)
sys.path.insert(0, "examples/training")
from train_pi05_casa_experiments import SAVE_FREQ as _SAVE_FREQ, TOTAL_STEPS as _TOTAL_STEPS
steps = int(os.environ.get("SFT_STEPS", _TOTAL_STEPS))
save_freq = int(os.environ.get("SFT_SAVE_FREQ", _SAVE_FREQ))
prune = os.environ.get("SFT_PRUNE_OPTIMIZER_STATE", "1") == "1"
rungs = steps // save_freq
projected = rungs * 8.7 + 14.1 if prune else rungs * 23.0
free_gb = shutil.disk_usage(out).free / 2**30
print(f"scratch free: {free_gb:.0f} GiB | projected checkpoints: {rungs} rungs, "
      f"prune={prune} -> {projected:.0f} GiB (upper bound, full-FT sizing)")
if projected > free_gb:
    fail.append(f"would write ~{projected:.0f} GiB but only {free_gb:.0f} GiB free")

import torch
if not torch.cuda.is_available():
    fail.append("CUDA not available")
else:
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")

if fail:
    print("\nPREFLIGHT FAILED:")
    for f in fail:
        print(f"  - {f}")
    sys.exit(1)
print("preflight OK")
PREFLIGHT

python - "$TASK" <<'ENVCHECK'
import sys
task = sys.argv[1]
sys.path.insert(0, "examples/training")
from train_pi05_casa_experiments import TASKS, make_eval_env
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.factory import make_env
spec = TASKS[task]
if spec.dexjoco_task is None:
    raise SystemExit(f"task {task!r} is not a DexJoCo task; use sft_arms_mila.sbatch")
meta = LeRobotDatasetMetadata(spec.dataset, root=spec.dataset_root)
env_cfg = make_eval_env(spec, meta)
envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
env = next(iter(next(iter(envs.values())).values()))
obs, _ = env.reset(seed=0)
prompt = env.call("task_description")[0]
dataset_prompts = set(meta.tasks.index)
env.close()
print(f"dexjoco env {spec.dexjoco_task} OK: cameras={sorted(obs['pixels'])} prompt={prompt!r}")
if prompt not in dataset_prompts:
    raise SystemExit(f"eval prompt {prompt!r} is not a training prompt {sorted(dataset_prompts)}")
ENVCHECK

# Drop AdamW state from every checkpoint but the newest: resume only ever reads `last`,
# and the pruned rungs keep pretrained_model/ so the checkpoint ladder stays usable for
# eval and for dual-noise selection. SFT_PRUNE_OPTIMIZER_STATE=0 keeps everything.
PRUNE=${SFT_PRUNE_OPTIMIZER_STATE:-1}
prune_loop () {
    local root="$1"
    while true; do
        sleep 300
        [ -d "$root" ] || continue
        local last_target
        last_target=$(basename "$(readlink -f "$root/last")" 2>/dev/null)
        [ -n "$last_target" ] && [ "$last_target" != last ] || continue
        for d in "$root"/[0-9]*; do
            [ -d "$d" ] || continue
            [ "$(basename "$d")" = "$last_target" ] && continue
            if [ -d "$d/training_state" ]; then
                rm -rf "$d/training_state"
                echo "[prune] dropped optimizer state from $(basename "$d") (keeping $last_target)"
            fi
        done
    done
}

# Preemption requeues the job, which reruns this script from the top.
RUN_RECORD="$OUTROOT/.slurm_runs/${SLURM_JOB_ID:-local}"
RUN=""
RESUME=()
if [ -f "$RUN_RECORD" ]; then
    RUN=$(cat "$RUN_RECORD")
    if [ -d "$RUN/checkpoints/last/training_state" ]; then
        RESUME=(--resume-from "$RUN")
        echo "=== requeued: resuming $RUN from checkpoint $(basename "$(readlink -f "$RUN/checkpoints/last")") ==="
    else
        echo "=== requeued: $RUN has no resumable checkpoint yet, starting a new run ==="
        RUN=""
    fi
fi

echo "=== training ($(date)) ==="
srun python examples/training/train_pi05_casa_experiments.py \
    --experiment "$ARM" \
    --task "$TASK" \
    "${RESUME[@]}" \
    "$@" &
TRAIN_PID=$!

(
    if [ -z "$RUN" ]; then
        for _ in $(seq 1 60); do
            sleep 30
            # TASK must be in the pattern: coffee and lamp jobs share an arm name.
            RUN=$(find "$OUTROOT" -maxdepth 2 -type d -newermt '-15 minutes' -name "*_${TASK}_${ARM}_*" 2>/dev/null | sort | tail -1)
            [ -n "$RUN" ] && break
        done
    fi
    if [ -z "$RUN" ]; then
        echo "[run] WARNING: could not locate the run directory; no pruning, and a requeue restarts from scratch"
        exit 0
    fi
    mkdir -p "$(dirname "$RUN_RECORD")"
    echo "$RUN" > "$RUN_RECORD"
    if [ "$PRUNE" = "1" ]; then
        echo "[prune] watching $RUN/checkpoints"
        prune_loop "$RUN/checkpoints"
    fi
) &
WATCH_PID=$!
trap 'kill $WATCH_PID 2>/dev/null || true' EXIT

wait $TRAIN_PID
TRAIN_RC=$?
echo "=== done rc=$TRAIN_RC ($(date)) ==="
exit $TRAIN_RC
