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

import gymnasium as gym
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


def make_prepare_obs_fn(device: torch.device):
    """Raw (batched) RoboCasa obs → compact noise-actor observation (image + state).

    This is what the small SAC policy sees. Under a vector env the observation already
    carries a leading batch dimension.
    """

    def prepare_obs(obs: dict) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        agent_pos = obs.get("agent_pos")
        if agent_pos is not None:
            out["observation.state"] = torch.as_tensor(agent_pos, dtype=torch.float32, device=device)
        for cam_name, img in obs.get("pixels", {}).items():
            img_t = torch.as_tensor(img, dtype=torch.float32, device=device)  # (B, H, W, C)
            out[f"observation.image.{cam_name}"] = img_t.permute(0, 3, 1, 2)
        return out

    return prepare_obs


def make_prepare_frozen_obs_fn(device: torch.device, image_features: list[str], max_state_dim: int):
    """Raw (batched) RoboCasa obs → pi0-format batch dict.

    Pi0 expects images under keys in ``image_features``, a state padded to
    ``max_state_dim``, and language tokens. Tokens come from a lazily-loaded tokenizer;
    adjust ``_TOKENIZER_NAME`` if your checkpoint uses a different one.
    """

    _TOKENIZER_NAME = "google/paligemma-3b-pt-224"
    _tokenizer = None

    def _get_tokenizer():
        nonlocal _tokenizer
        if _tokenizer is None:
            from transformers import AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
        return _tokenizer

    def prepare_frozen_obs(obs: dict) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        pixels = obs.get("pixels", {})
        batch_size = 1
        for cam_name, img in pixels.items():
            key = f"observation.image.{cam_name}"
            if key in image_features:
                img_t = torch.as_tensor(img, dtype=torch.float32, device=device)  # (B, H, W, C)
                out[key] = img_t.permute(0, 3, 1, 2)
                batch_size = img_t.shape[0]

        agent_pos = obs.get("agent_pos")
        if agent_pos is not None:
            state = torch.as_tensor(agent_pos, dtype=torch.float32, device=device)  # (B, D)
            batch_size = state.shape[0]
            if state.shape[-1] < max_state_dim:
                padding = torch.zeros(state.shape[0], max_state_dim - state.shape[-1], device=device)
                state = torch.cat([state, padding], dim=-1)
            out["observation.state"] = state

        # Language tokens — the same generic prompt for every env in the batch.
        tokenizer = _get_tokenizer()
        prompt = "Perform the task in the kitchen environment."
        tokenized = tokenizer(
            prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=32
        )
        out["observation.language.tokens"] = tokenized["input_ids"].to(device).repeat(batch_size, 1)
        out["observation.language.attention_mask"] = (
            tokenized["attention_mask"].to(device).repeat(batch_size, 1)
        )
        return out

    return prepare_frozen_obs


def make_generate_action_chunk_fn(
    frozen_policy: PI0Policy,
    noise_chunk_size: int,
    full_chunk_size: int,
    max_action_dim: int,
    device: torch.device,
):
    """Bundle noise padding + pi0 action generation.

    The RL policy emits a small noise ``(B, noise_chunk_size * max_action_dim)`` which is
    reshaped and padded (by repeating the last step) to pi0's full chunk size. The frozen
    observation history is stacked along ``dim=1``; with ``n_obs_steps=1`` we drop that axis
    to recover pi0's expected ``(B, ...)`` batch layout, then defer to
    ``PI0Policy.predict_action_chunk`` (which handles image/state/token preprocessing and
    forwards ``noise`` to the flow-matching sampler).

    NOTE: untested against real pi0 weights — pi0 wiring is left for later verification.
    """

    def generate_action_chunk(stacked_obs: dict, noise: np.ndarray) -> torch.Tensor:
        batch_size = noise.shape[0]
        noise_tensor = torch.from_numpy(noise).float().to(device)
        noise_tensor = noise_tensor.view(batch_size, noise_chunk_size, max_action_dim)
        pad_size = full_chunk_size - noise_chunk_size
        if pad_size > 0:
            padding = noise_tensor[:, -1:, :].repeat(1, pad_size, 1)
            noise_tensor = torch.cat([noise_tensor, padding], dim=1)

        # Drop the single-step history axis pi0 does not consume.
        batch = {key: value[:, 0] for key, value in stacked_obs.items()}
        return frozen_policy.predict_action_chunk(batch, noise=noise_tensor)

    return generate_action_chunk


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
    env = gym.vector.SyncVectorEnv([lambda: make_robocasa_env(task=args.task, seed=args.seed)])

    # ── Observation / noise adapters ────────────────────────────────────
    prepare_obs_fn = make_prepare_obs_fn(device)
    prepare_frozen_obs_fn = make_prepare_frozen_obs_fn(
        device=device,
        image_features=frozen_policy.config.image_features,
        max_state_dim=frozen_policy.config.max_state_dim,
    )
    generate_action_chunk_fn = make_generate_action_chunk_fn(
        frozen_policy, noise_chunk_size, full_chunk_size, max_action_dim, device
    )

    # ── Train ───────────────────────────────────────────────────────────
    train_dsrl(
        env=env,
        noise_dim=noise_dim,
        prepare_obs_fn=prepare_obs_fn,
        generate_action_chunk_fn=generate_action_chunk_fn,
        prepare_frozen_obs_fn=prepare_frozen_obs_fn,
        total_steps=args.total_steps,
        device=str(device),
        seed=args.seed,
        output_dir=args.output_dir,
    )

    env.close()


if __name__ == "__main__":
    main()
