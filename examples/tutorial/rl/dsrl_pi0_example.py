#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DSRL with a pi0 (or pi05) VLA policy on RoboCasa.

Example usage:
    python examples/tutorial/rl/dsrl_pi0_example.py \
        --policy_id lerobot/pi0_robocasa \
        --task CoffeePressButton \
        --total_steps 500000

Notes:
  - The pi0 checkpoint must have been trained/fine-tuned on the target task(s).
  - Language tokens are automatically generated from the task name. For custom
    tasks, override ``make_pi0_obs_to_frozen_obs()``.
  - The noise dimension is a subset of the full pi0 noise space (by default
    5 chunks × max_action_dim) to keep the RL policy small. This follows the
    approach from the original DSRL paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lerobot.envs.robocasa_env import RoboCasaEnv
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.rl.dsrl import train_dsrl


# ── Helpers ────────────────────────────────────────────────────────────────


def make_robocasa_env(task: str, camera_names: str = "robot0_agentview_center", seed: int = 0) -> RoboCasaEnv:
    """Create a single RoboCasa environment for pi0."""
    return RoboCasaEnv(
        task_name=task,
        robot="PandaOmron",
        camera_name=camera_names,
        obs_type="pixels_agent_pos",
        seed=seed,
        observation_width=256,
        observation_height=256,
    )


def make_pi0_noise_reshape_fn(
    noise_chunk_size: int,
    full_chunk_size: int,
    max_action_dim: int,
):
    """Build noise reshape function for pi0.

    Pi0 expects noise of shape (batch, full_chunk_size, max_action_dim).
    The RL policy outputs a smaller noise (batch, noise_chunk_size, max_action_dim)
    which is padded by repeating the last step to match full_chunk_size.
    """

    def reshape_noise(noise_np: np.ndarray, device: torch.device) -> torch.Tensor:
        noise_tensor = torch.from_numpy(noise_np).float().to(device)
        # Reshape flat → (1, noise_chunk_size, max_action_dim)
        noise_tensor = noise_tensor.view(1, noise_chunk_size, max_action_dim)
        # Pad to full chunk size by repeating the last noise step
        pad_size = full_chunk_size - noise_chunk_size
        if pad_size > 0:
            last_step = noise_tensor[:, -1:, :]  # (1, 1, max_action_dim)
            padding = last_step.repeat(1, pad_size, 1)
            noise_tensor = torch.cat([noise_tensor, padding], dim=1)
        return noise_tensor

    return reshape_noise


def make_pi0_obs_to_policy_obs(device: torch.device):
    """Build function that converts RoboCasa raw obs → noise actor observation format.

    This is a simple image + state format that the small SAC policy can process.
    """

    def obs_to_policy_obs(obs: dict) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        agent_pos = obs.get("agent_pos")
        if agent_pos is not None:
            out["observation.state"] = torch.from_numpy(agent_pos).float().unsqueeze(0).to(device)

        pixels = obs.get("pixels", {})
        for cam_name, img in pixels.items():
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)
            out[f"observation.image.{cam_name}"] = img_tensor

        return out

    return obs_to_policy_obs


def make_pi0_obs_to_frozen_obs(
    device: torch.device,
    image_features: list[str],
    max_state_dim: int,
):
    """Build function that converts RoboCasa raw obs → pi0-format batch dict.

    Pi0 expects:
      - Images under keys matching ``image_features`` (e.g. "observation.image.robot0_agentview_center")
      - ``observation.state``: state vector
      - ``observation.language.tokens`` and ``observation.language.attention_mask``

    Note: Language tokens are loaded from a cached tokenizer (lazy init). If your
    pi0 checkpoint uses a different tokenizer, adjust ``_TOKENIZER_NAME`` below.
    """

    _TOKENIZER_NAME = "google/paligemma-3b-pt-224"
    _tokenizer = None

    def _get_tokenizer():
        nonlocal _tokenizer
        if _tokenizer is None:
            from transformers import AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
        return _tokenizer

    def obs_to_frozen_obs(obs: dict) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        # Images — map camera names to pi0 image feature keys
        pixels = obs.get("pixels", {})
        for cam_name, img in pixels.items():
            key = f"observation.image.{cam_name}"
            if key in image_features:
                img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)
                out[key] = img_tensor

        # State — pad to max_state_dim
        agent_pos = obs.get("agent_pos")
        if agent_pos is not None:
            state = torch.from_numpy(agent_pos).float().unsqueeze(0).to(device)
            if state.shape[-1] < max_state_dim:
                padding = torch.zeros(1, max_state_dim - state.shape[-1], device=device)
                state = torch.cat([state, padding], dim=-1)
            out["observation.state"] = state

        # Language tokens — encode a generic prompt
        tokenizer = _get_tokenizer()
        prompt = "Perform the task in the kitchen environment."
        tokenized = tokenizer(
            prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=32
        )
        out["observation.language.tokens"] = tokenized["input_ids"].to(device)
        out["observation.language.attention_mask"] = tokenized["attention_mask"].to(device)

        return out

    return obs_to_frozen_obs


def main():
    parser = argparse.ArgumentParser(description="DSRL with pi0 VLA on RoboCasa")
    parser.add_argument("--policy_id", type=str, required=True, help="HF repo ID of pretrained pi0/pi05 policy")
    parser.add_argument("--task", type=str, default="CoffeePressButton", help="RoboCasa task name")
    parser.add_argument("--total_steps", type=int, default=500_000, help="Total environment steps")
    parser.add_argument("--noise_chunk_size", type=int, default=5, help="Number of noise steps (subset of pi0 chunk_size)")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs/dsrl_pi0", help="Output directory")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load frozen pi0 (or pi05) policy ─────────────────────────────────
    print(f"Loading pi0 policy from {args.policy_id} ...")
    frozen_policy = PI0Policy.from_pretrained(args.policy_id)
    frozen_policy.eval()
    frozen_policy.to(device)

    # Extract dimensions from policy config
    full_chunk_size = frozen_policy.config.chunk_size
    max_action_dim = frozen_policy.config.max_action_dim
    noise_chunk_size = args.noise_chunk_size
    noise_dim = noise_chunk_size * max_action_dim

    print(
        f"Pi0 config: chunk_size={full_chunk_size}, max_action_dim={max_action_dim}, "
        f"noise_chunk={noise_chunk_size}, noise_dim={noise_dim}"
    )

    # ── Create environment ──────────────────────────────────────────────
    env = make_robocasa_env(task=args.task, seed=args.seed)

    # ── Observation / noise adapters ────────────────────────────────────
    obs_to_policy_obs = make_pi0_obs_to_policy_obs(device)
    obs_to_frozen_obs = make_pi0_obs_to_frozen_obs(
        device=device,
        image_features=frozen_policy.config.image_features,
        max_state_dim=frozen_policy.config.max_state_dim,
    )
    noise_reshape_fn = make_pi0_noise_reshape_fn(noise_chunk_size, full_chunk_size, max_action_dim)

    # ── Train ───────────────────────────────────────────────────────────
    train_dsrl(
        env=env,
        frozen_policy=frozen_policy,
        noise_dim=noise_dim,
        obs_to_policy_obs=obs_to_policy_obs,
        obs_to_frozen_obs=obs_to_frozen_obs,
        noise_reshape_fn=noise_reshape_fn,
        total_steps=args.total_steps,
        device=str(device),
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
