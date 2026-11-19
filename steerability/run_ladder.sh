#!/usr/bin/env bash
# Run one phase over every (task, arm, DSRL checkpoint) in a manifest.
#
#   ./run_ladder.sh <phase> <manifest.tsv> <out_dir> [extra args for the script]
#
# phase:     states | noise | obs | box | branching | smoothness | transfer
#            (p0_wandb_signals.py and analyze.py run once, not per checkpoint;
#             extra args go to that phase's script only)
# manifest:  tab-separated  task  arm  run_dir   ('#' lines ignored)
#            run_dir may be 'mock:...' to use the mock backend
# env vars:  ENV_FACTORY, OBS_FN    wiring hooks (see README)
#            REF_STATES_<task>      reference state set for p1 (default <out>/states/ref_<task>.pkl)
#            DRY=1                  print commands instead of running them
#            LAUNCH="sbatch --wrap" prefix each command (e.g. to submit one job per checkpoint)
set -euo pipefail
PHASE=$1; MANIFEST=$2; OUT=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
declare -A LADDER=([lamp]="2000 4000 8000 16000" [coffee]="4000 8000 14000 18000")

run() {
  if [[ "${DRY:-0}" == 1 ]]; then echo "$*"; return; fi
  if [[ -n "${LAUNCH:-}" ]]; then $LAUNCH "$*"; else eval "$*"; fi
}

while IFS=$'\t' read -r TASK ARM RUN; do
  [[ -z "${TASK// }" || "$TASK" == \#* ]] && continue
  for STEP in ${LADDER[$TASK]}; do
    if [[ "$RUN" == mock:* ]]; then CKPT=$RUN; ENVARGS="--adapter mock"
    else CKPT="$RUN/checkpoints/$(printf %06d "$STEP")/pretrained_model"
         ENVARGS="--env-factory ${ENV_FACTORY:?set ENV_FACTORY} --obs-fn ${OBS_FN:?set OBS_FN}"; fi
    TAG="$TASK"_"$ARM"_$(printf %06d "$STEP")
    C="--task $TASK --arm $ARM --sft-step $STEP --checkpoint $CKPT --out $OUT $ENVARGS"
    REFVAR="REF_STATES_$TASK"; REF="${!REFVAR:-$OUT/states/ref_$TASK.pkl}"
    case $PHASE in
      states)     run python "$HERE/collect_states.py" $C "$@" ;;
      noise)      run python "$HERE/p1_noise_sensitivity.py" $C --states "$REF" "$@" ;;
      obs)        run python "$HERE/p1_obs_reliance.py" $C --states "$REF" "$@" ;;
      box)        run python "$HERE/p1_box_reachability.py" $C "$@" ;;
      branching)  run python "$HERE/p2_branching.py" $C --states "$OUT/states/$TAG.pkl" "$@" ;;
      smoothness) run python "$HERE/p2_smoothness.py" $C "$@" ;;
      transfer)   run python "$HERE/p2_transfer.py" $C "$@" ;;
      *) echo "unknown phase $PHASE" >&2; exit 2 ;;
    esac
  done
done < "$MANIFEST"
