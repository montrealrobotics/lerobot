#!/bin/bash
# Upload offline wandb runs to the server. Run on a LOGIN node: compute nodes on narval and
# rorqual have no internet, so dsrl_drac_pack.sbatch sets WANDB_MODE=offline there and each
# run writes <output_dir>/wandb/offline-run-*/ instead of streaming.
#
#   scripts/sync_offline_wandb.sh                 # finished DSRL + SFT runs not yet uploaded
#   scripts/sync_offline_wandb.sh --include-live  # also push runs still in progress
#   scripts/sync_offline_wandb.sh /scratch/$USER/lerobot/outputs/dsrl/dsrl_arm0_coffee_04000
#
# A run counts as finished when its launcher wrote a sentinel on success: .dsrl_complete
# (dsrl_drac_pack.sbatch) or .sft_complete (sft_arms.sbatch).
# Finished runs are marked synced by wandb, so re-running this skips them. Live runs are
# re-uploaded with --include-synced --append each time, which is what lets you watch
# progress mid-run; it re-reads the whole offline log, so it gets slower as runs grow.
set -euo pipefail

LIVE=0
ROOTS=()
for arg in "$@"; do
    case "$arg" in
        --include-live) LIVE=1 ;;
        *) ROOTS+=("$arg") ;;
    esac
done
[ ${#ROOTS[@]} -gt 0 ] || ROOTS=("/scratch/$USER/lerobot/outputs/dsrl" "/scratch/$USER/lerobot/outputs/train")

WANDB=$(command -v wandb || true)
[ -n "$WANDB" ] || WANDB=/scratch/$USER/envs/dex_vla/bin/wandb
[ -x "$WANDB" ] || { echo "no wandb executable found (activate the venv)" >&2; exit 2; }

done_n=0; live_n=0; skipped=0
while IFS= read -r offline_dir; do
    run_dir=$(dirname "$(dirname "$offline_dir")")   # <run>/wandb/offline-run-* -> <run>
    if [ -f "$run_dir/.dsrl_complete" ] || [ -f "$run_dir/.sft_complete" ]; then
        "$WANDB" sync "$offline_dir" --mark-synced && done_n=$((done_n + 1))
    elif [ "$LIVE" = "1" ]; then
        "$WANDB" sync "$offline_dir" --include-synced --append && live_n=$((live_n + 1))
    else
        skipped=$((skipped + 1))
    fi
done < <(find "${ROOTS[@]}" -type d -name 'offline-run-*' 2>/dev/null | sort)

echo "synced: $done_n finished, $live_n live; left for later: $skipped in-progress (use --include-live)"
