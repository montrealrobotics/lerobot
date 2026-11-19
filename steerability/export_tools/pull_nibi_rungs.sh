#!/bin/bash
# Copy SFT rungs that exist only on nibi to Mila scratch (same relative paths), for the
# extension dual-noise ladders. rsync to a temp dir and move into place, so a failed transfer
# never leaves an empty pretrained_model/ that breaks the dual-noise preflight.
set -e
S=/scratch/akur/lerobot/outputs/train; D=/network/scratch/a/artur.kuramshin/lerobot/outputs/train
start=$(date +%s)
pull () {
  [ -f "$D/$1/train_config.json" ] && { echo "$1 already here"; return; }
  mkdir -p "$(dirname "$D/$1")"; rsync -a "nibi:$S/$1/" "$D/$1.partial/"; mv "$D/$1.partial" "$D/$1"
  echo "$1 done $(( $(date +%s)-start ))s"
}
for s in 006000 010000 012000 014000 018000; do pull 2026-09-20/21-06-12_sft_arms_lamp_arm3_lora_r32_lrx1_seed42/checkpoints/$s/pretrained_model; done
pull 2026-09-20/04-16-11_sft_arms_coffee_arm2_frozen_llm_lrx1_seed42/checkpoints/020000/pretrained_model
for s in 006000 010000 012000 014000 018000 020000; do pull 2026-09-20/04-16-48_sft_arms_lamp_arm2_frozen_llm_lrx1_seed42/checkpoints/$s/pretrained_model; done
echo ALL_DONE
