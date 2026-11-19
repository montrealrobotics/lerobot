#!/bin/bash
# Arm4-only fetch for the extension rebuild (our copy; Artur's original is fetch_ext.sh.orig).
# Reads only the arm4 drop folder and writes into ext_raw_arm4/, flattened to the names
# build_extension.py expects. Arms 0-3 are not fetched: their entries are taken unchanged from
# ../export_artur/ (our ext_raw/ copy lacks several of their raw files).
# Re-runnable. Then: python3 build_extension.py
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RAW=$HERE/ext_raw_arm4
DROP=/network/projects/real-g-grp/dex_vla_checkpoints/arm4_results
mkdir -p "$RAW"/{dn,lamp_dsrl,coffee_dsrl}

# Frozen dual-noise files, flattened to drop__<run name>__<file>. Run dirs are globbed, not
# hardcoded (the lamp one is 09-36-39_..., not the runbook's 09-38-40_...).
for f in "$DROP"/dn/*sft_arms_*arm4*/dual_noise_{lamp_extrap,coffee_lateral}.json; do
    [ -f "$f" ] || continue
    run=$(basename "$(dirname "$f")"); cp -u "$f" "$RAW/dn/drop__${run}__$(basename "$f")"
done

# Lamp DSRL eval-only logs (final _sepbank actors).
rsync -a "$DROP"/lamp_extrap/lamp_extrap_dsrl_arm4_*.log "$RAW/lamp_dsrl/"

# Coffee-lateral DSRL slurm logs (in-loop eval histories).
rsync -a "$DROP"/coffee_lat_logs/dsrl_arm4_coffee_lat_*.out "$RAW/coffee_dsrl/"
echo "fetched into $RAW: $(find "$RAW" -type f | wc -l) files"
