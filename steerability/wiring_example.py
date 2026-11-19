"""Skeleton for the two hooks the RoboCasa adapter needs. Copy into your repo (e.g.
scripts/steerability_wiring.py), fill the TODOs from code you already have, and pass

    --env-factory scripts.steerability_wiring:make_cell_env
    --obs-fn      scripts.steerability_wiring:pi05_frame

Both must reproduce what the DSRL eval does, or the diagnostics measure a different setup.
"""
import numpy as np

LAMP_BANK = "~/scratch/lerobot/placement_banks/screwlightbulb_xarm6_seed1_pair/bank.json"
COFFEE_BANK = "~/scratch/lerobot/placement_banks/coffeepressbutton_pandadex_kitchens/bank.json"
PROMPTS = {"lamp": "TODO: same prompt string the DSRL launcher passes", "coffee": "TODO"}


def make_cell_env(task: str, cell: str, seed: int):
    """Return a robosuite/robocasa env whose every reset() lands in `cell`.

    Reuse the env construction in scripts/eval_sft_on_banks.py: same robot, controller, fps,
    cameras (robot0_agentview_left + robot0_eye_in_hand at the pi05 resolution), and the
    PlacementBankWrapper restricted to ONE cell instead of alternating.
    For coffee, keep the start-pose rejection (--reject_start_rot_deg 10 --reject_start_pos_cm 3).

    Requirements (checked by the adapter):
      * env.step / env.reset return robosuite obs dicts (no gym wrapper that rewrites obs);
      * the robosuite env with `.sim` is reachable through `.env` / `.unwrapped`.
    """
    raise NotImplementedError("wire to eval_sft_on_banks.py env construction")


def pi05_frame(task: str, obs: dict) -> dict:
    """robosuite obs dict -> batch dict (leading dim 1) for the checkpoint's preprocessor.

    Must equal the frame the frozen pi05 sees in DSRLEnvWrapper: two cameras (third-person
    robot0_agentview_left and wrist), the state vector in the training order, and the prompt.
    Typically:
        import torch
        img = lambda k: torch.from_numpy(np.ascontiguousarray(obs[k][::-1])).permute(2, 0, 1)[None].float() / 255
        return {
            "observation.images.<third_person_name>": img("robot0_agentview_left_image"),
            "observation.images.<wrist_name>":        img("robot0_eye_in_hand_image"),
            "observation.state": torch.from_numpy(<state vector as in training>)[None].float(),
            "task": [PROMPTS[task]],
        }
    Check the image flip convention against what training used.
    """
    raise NotImplementedError("wire to DSRLEnvWrapper's frozen-policy observation path")
