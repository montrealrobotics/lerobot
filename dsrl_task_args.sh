# Per-task DSRL invariants -- sourced, never executed directly.
#
# Sourced by dsrl_drac.sbatch (one run per GPU) and dsrl_drac_pack.sbatch (three runs per
# GPU). It lives in one file precisely because those two launchers would otherwise each
# carry their own copy of the robot / fps / controller / bank / eval-set definitions, and a
# drift between them silently makes packed runs incomparable to unpacked ones.
#
# Expects $TASK and $BANKS to be set; sets $BANK and $TASK_ARGS.

# Task invariants. Controller and fps must match what the checkpoint was trained on: a wrong
# controller silently falls back to OSC_POSE and every rollout is garbage. --prompt is the
# dataset's task string, used only if the env exposes no ep_meta['lang'] (the script's own
# default is a coffee string, wrong for both tasks).
# Resolved from whichever robosuite the venv imports, so the same file works on every
# cluster (the checkout lives at a different path on each). Needs the venv active.
ROBOSUITE_CONTROLLERS=$(python -c 'import os, robosuite; print(os.path.join(os.path.dirname(robosuite.__file__), "controllers/config/robots"))' 2>/dev/null)
[ -d "$ROBOSUITE_CONTROLLERS" ] || { echo "cannot resolve robosuite controller configs (venv active?)" >&2; exit 2; }

case "$TASK" in
  lamp)
    BANK=$BANKS/screwlightbulb_xarm6_seed1_pair/bank.json
    TASK_ARGS=(
      --task ScrewLightbulb --robot XArm6DexLeapRHOmron --fps 30
      --controller "$ROBOSUITE_CONTROLLERS/default_xarm6dexleaprhomron_joint_pos.json"
      --collect_envs 1 --collect_scene_seeds 1
      --placement_bank "$BANK"
      --eval_placement_sets train,heldout
      --prompt "screw the light bulb into the lamp base"
    )
    ;;
  coffee)
    BANK=$BANKS/coffeepressbutton_pandadex_kitchens/bank.json
    TASK_ARGS=(
      --task CoffeePressButton --robot PandaDexLeapRHOmron --fps 20
      --collect_envs 1
      --kitchen_bank "$BANK"
      --eval_kitchen_sets heldout,reference
      --reject_start_rot_deg 10 --reject_start_pos_cm 3
      --prompt "press the button on the coffee machine to serve coffee"
    )
    ;;
  *) echo "unknown task '$TASK' (lamp|coffee)" >&2; exit 2 ;;
esac

# The banks are built on Mila; rsync them over before the first run here.
[ -f "$BANK" ] || { echo "no bank at $BANK -- rsync it from Mila" >&2; exit 2; }
