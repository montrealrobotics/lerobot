#!/bin/bash
# Arm4 lamp DSRL rerun on the regenerated September lamp bank (2026-10-02).
#
#   ./submit_dsrl_arm4.sh smoke   # one 2k seed-0 job, 4000 steps, 1 h
#   ./submit_dsrl_arm4.sh full    # 4 rungs x 3 seeds
#
# The original arm4 lamp runs (dsrl_arm4_lamp_<step>_s<seed>) used the 09-27 bank rebuild
# (screwlightbulb_xarm6_seed1_pair, md5 c9a65d35), whose starts differ from the bank arms 0-3
# trained on. They are kept but excluded from arm comparisons; see SFT_ARMS_DSRL.md.
# Reruns end in _sepbank, so their output dirs and wandb names never collide with the originals,
# and --resume cannot pick up an original run's resume_state.pt.
# The bank override works because dsrl_mila_cloutiec.sbatch forwards extra args AFTER its
# TASK_ARGS, and --placement_bank is a plain argparse option (last occurrence wins).
set -euo pipefail
MODE=${1:?usage: $0 smoke|full}

COFFEE_RUN=/network/scratch/c/cloutiec/lerobot/outputs/train/2026-09-23/22-49-49_sft_arms_coffee_arm4_lora_vlm_full_expert_lrx1_seed42
LAMP_RUN=/network/scratch/c/cloutiec/lerobot/outputs/train/2026-09-24/09-36-39_sft_arms_lamp_arm4_lora_vlm_full_expert_lrx1_seed42
SEP_BANK=/network/scratch/c/cloutiec/lerobot/placement_banks/screwlightbulb_xarm6_seed1_pair_dsrl/bank.json
SEP_BANK_MD5=919c648884d68744e4cb0b841cb6d066

cd /network/scratch/c/cloutiec/dexrobot/lerobot

got=$(md5sum "$SEP_BANK" | cut -d' ' -f1)
[ "$got" = "$SEP_BANK_MD5" ] || { echo "ABORT: $SEP_BANK md5 is $got, expected $SEP_BANK_MD5" >&2; exit 1; }
for STEP in 2000 4000 8000 16000; do
    CK=$LAMP_RUN/checkpoints/$(printf %06d $STEP)/pretrained_model
    [ -f "$CK/adapter_model.safetensors" ] && [ -f "$CK/config.json" ] || { echo "ABORT: missing checkpoint $CK" >&2; exit 1; }
done
echo "September bank md5 OK; all four lamp checkpoints present"

# Coffee: NOT rerun. Arm4 coffee DSRL already ran on the correct kitchen bank (matches the notes and
# arms 0-3), so only lamp needed redoing. Kept for reference.
# for STEP in 4000 8000 14000 18000; do
#     for S in 0 1 2; do
#         sbatch --job-name=dsrl_arm4_coffee_${STEP}_s${S} \
#             dsrl_mila_cloutiec.sbatch coffee \
#             $COFFEE_RUN/checkpoints/$(printf %06d $STEP)/pretrained_model \
#             $S
#     done
# done

submit() {  # <job-name> <step> <seed> [sbatch opts...] -- [extra trainer args...]
    local name=$1 step=$2 seed=$3; shift 3
    local sb=() ; while [ $# -gt 0 ] && [ "$1" != "--" ]; do sb+=("$1"); shift; done; shift || true
    sbatch --parsable --job-name="$name" "${sb[@]}" \
        dsrl_mila_cloutiec.sbatch lamp \
        $LAMP_RUN/checkpoints/$(printf %06d $step)/pretrained_model \
        $seed --placement_bank "$SEP_BANK" "$@"
}

case "$MODE" in
  smoke)
    id=$(submit dsrl_arm4_lamp_2000_s0_smoke_sepbank 2000 0 --time=1:00:00 -- \
        --total_steps 4000 --min_buffer_size 20 --eval_freq 2000 --eval_episodes 1 --save_freq 2000)
    echo "smoke dsrl_arm4_lamp_2000_s0_smoke_sepbank $id"
    ;;
  full)
    # Lamp: checkpoints 2k, 4k, 8k, 16k x 3 seeds
    for STEP in 2000 4000 8000 16000; do
        for S in 0 1 2; do
            id=$(submit dsrl_arm4_lamp_${STEP}_s${S}_sepbank $STEP $S --)
            echo "full dsrl_arm4_lamp_${STEP}_s${S}_sepbank $id"
        done
    done
    ;;
  *) echo "unknown mode '$MODE' (smoke|full)" >&2; exit 2 ;;
esac
squeue -u cloutiec
