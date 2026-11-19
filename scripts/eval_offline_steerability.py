"""Zero-rollout offline steerability metrics for the SFT-arms x DSRL checkpoints.

For every SFT checkpoint (5 recipes x 4 steps x 2 tasks) this computes, on a fixed set of
N_obs demonstration observations per task (same frames, same noises for every checkpoint):

  1. eRank / Tr(Sigma) / logdet(Sigma + eps I) of the decoded executed chunk (16 x 22) across M
     DSRL-style noises w ~ U[-1,1]^32 tiled over the 32-step chunk; total, arm (0:6), hand (6:22)
  2. hand/arm variance ratio, and the eRank of the instantaneous 16x16 finger covariance
  3. local Lipschitz gain ||f(w+d) - f(w)|| / ||d||, d ~ N(0, sigma_d^2 I) in w-space, and
     signal-to-chaos Tr(Sigma_total) / L_local
  4. flow-matching velocity MSE on the demo action chunk (total / arm / hand), with fixed (tau, eps)
  5. drift of the VLM prefix KV cache vs lerobot/pi05_base (cosine distance, relative L2)
  6. composites (computed in `analyze`, across the checkpoints of a task)

The VLM prefix runs once per observation; the M noises (plus the Lipschitz perturbations and the FM
samples) only go through the action expert, attending to the cached prefix KV. All action metrics
are in model space (quantile-normalized, before the postprocessor) so arm and hand dims are comparable.

  python scripts/eval_offline_steerability.py run --tasks lamp --recipes EXPERT-ONLY --steps 2000 --smoke
  python scripts/eval_offline_steerability.py run --tasks lamp coffee          # all checkpoints, resumable
  python scripts/eval_offline_steerability.py analyze                           # report + plots from the CSV
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results"
CHECKPOINTS_TSV = Path("/network/scratch/c/cloutiec/diag/checkpoints.tsv")  # from steerability/find_checkpoints.py
BASE_MODEL = "lerobot/pi05_base"

TASKS = {
    "lamp": dict(name="ScrewLightbulb", repo_id="akuramshin/robocasa_lightbulbscrew_dex_filtered", fps=30,
                 steps=[2000, 4000, 8000, 16000]),
    "coffee": dict(name="CoffeePressButton", repo_id="akuramshin/robocasa_coffeepressbutton_dex_augstyle", fps=20,
                   steps=[4000, 8000, 14000, 18000]),
}
TASK_BY_NAME = {v["name"]: k for k, v in TASKS.items()}
RECIPES = {"EXPERT-ONLY": "arm0", "FULL-FT": "arm1", "FROZEN-LM": "arm2", "LORA": "arm3", "LORA-LM": "arm4"}
# DSRL setup: chunk H=32, first 16 executed, noise dim 32 (= max_action_dim), action 22 = 6 arm + 16 LEAP hand
H, H_EXEC, NOISE_DIM, D_ACT, ARM = 32, 16, 32, 22, slice(0, 6)
HAND = slice(6, 22)
THIRD, WRIST = "observation.images.robot0_agentview_left", "observation.images.robot0_eye_in_hand"

# Success (%), as in the paper table: pre_* = before RL on the training cells (lamp: results_export/dual_noise.json;
# coffee: results_export/dual_noise_coffee_lateral.json); embedded so `analyze` needs no other input.
GROUND_TRUTH_DSRL = {
    "ScrewLightbulb": {
        ("EXPERT-ONLY", 2000):  {"pre_iid": 6,  "pre_rep": 46, "dsrl_train": 100, "dsrl_extrap": 98},
        ("EXPERT-ONLY", 4000):  {"pre_iid": 6,  "pre_rep": 54, "dsrl_train": 100, "dsrl_extrap": 73},
        ("EXPERT-ONLY", 8000):  {"pre_iid": 12, "pre_rep": 32, "dsrl_train": 100, "dsrl_extrap": 88},
        ("EXPERT-ONLY", 16000): {"pre_iid": 16, "pre_rep": 28, "dsrl_train": 100, "dsrl_extrap": 45},
        ("FROZEN-LM", 2000):    {"pre_iid": 0,  "pre_rep": 38, "dsrl_train": 100, "dsrl_extrap": 87},
        ("FROZEN-LM", 4000):    {"pre_iid": 0,  "pre_rep": 30, "dsrl_train": 99,  "dsrl_extrap": 70},
        ("FROZEN-LM", 8000):    {"pre_iid": 0,  "pre_rep": 6,  "dsrl_train": 0,   "dsrl_extrap": 0},
        ("FROZEN-LM", 16000):   {"pre_iid": 0,  "pre_rep": 0,  "dsrl_train": 0,   "dsrl_extrap": 0},
        ("FULL-FT", 2000):      {"pre_iid": 0,  "pre_rep": 40, "dsrl_train": 100, "dsrl_extrap": 73},
        ("FULL-FT", 4000):      {"pre_iid": 0,  "pre_rep": 32, "dsrl_train": 67,  "dsrl_extrap": 15},
        ("FULL-FT", 8000):      {"pre_iid": 0,  "pre_rep": 6,  "dsrl_train": 33,  "dsrl_extrap": 0},
        ("FULL-FT", 16000):     {"pre_iid": 0,  "pre_rep": 10, "dsrl_train": 46,  "dsrl_extrap": 0},
        ("LORA", 2000):         {"pre_iid": 6,  "pre_rep": 44, "dsrl_train": 100, "dsrl_extrap": 63},
        ("LORA", 4000):         {"pre_iid": 0,  "pre_rep": 34, "dsrl_train": 33,  "dsrl_extrap": 5},
        ("LORA", 8000):         {"pre_iid": 24, "pre_rep": 6,  "dsrl_train": 17,  "dsrl_extrap": 0},
        ("LORA", 16000):        {"pre_iid": 50, "pre_rep": 24, "dsrl_train": 50,  "dsrl_extrap": 0},
        ("LORA-LM", 2000):      {"pre_iid": 0,  "pre_rep": 42, "dsrl_train": 100, "dsrl_extrap": 85},
        ("LORA-LM", 4000):      {"pre_iid": 10, "pre_rep": 24, "dsrl_train": 2,   "dsrl_extrap": 3},
        ("LORA-LM", 8000):      {"pre_iid": 0,  "pre_rep": 0,  "dsrl_train": 0,   "dsrl_extrap": 0},
        ("LORA-LM", 16000):     {"pre_iid": 30, "pre_rep": 8,  "dsrl_train": 38,  "dsrl_extrap": 0},
    },
    "CoffeePressButton": {
        ("EXPERT-ONLY", 4000):  {"pre_iid": 32, "pre_rep": 6,  "dsrl_train": 100, "dsrl_extrap": 94},
        ("EXPERT-ONLY", 8000):  {"pre_iid": 24, "pre_rep": 6,  "dsrl_train": 99,  "dsrl_extrap": 87},
        ("EXPERT-ONLY", 14000): {"pre_iid": 28, "pre_rep": 12, "dsrl_train": 97,  "dsrl_extrap": 83},
        ("EXPERT-ONLY", 18000): {"pre_iid": 20, "pre_rep": 8,  "dsrl_train": 95,  "dsrl_extrap": 90},
        ("FROZEN-LM", 4000):    {"pre_iid": 46, "pre_rep": 18, "dsrl_train": 98,  "dsrl_extrap": 85},
        ("FROZEN-LM", 8000):    {"pre_iid": 60, "pre_rep": 42, "dsrl_train": 94,  "dsrl_extrap": 89},
        ("FROZEN-LM", 14000):   {"pre_iid": 72, "pre_rep": 52, "dsrl_train": 91,  "dsrl_extrap": 62},
        ("FROZEN-LM", 18000):   {"pre_iid": 78, "pre_rep": 64, "dsrl_train": 78,  "dsrl_extrap": 43},
        ("FULL-FT", 4000):      {"pre_iid": 40, "pre_rep": 18, "dsrl_train": 97,  "dsrl_extrap": 81},
        ("FULL-FT", 8000):      {"pre_iid": 72, "pre_rep": 30, "dsrl_train": 96,  "dsrl_extrap": 49},
        ("FULL-FT", 14000):     {"pre_iid": 80, "pre_rep": 54, "dsrl_train": 93,  "dsrl_extrap": 31},
        ("FULL-FT", 18000):     {"pre_iid": 48, "pre_rep": 44, "dsrl_train": 81,  "dsrl_extrap": 20},
        ("LORA", 4000):         {"pre_iid": 36, "pre_rep": 16, "dsrl_train": 95,  "dsrl_extrap": 83},
        ("LORA", 8000):         {"pre_iid": 72, "pre_rep": 28, "dsrl_train": 95,  "dsrl_extrap": 35},
        ("LORA", 14000):        {"pre_iid": 72, "pre_rep": 62, "dsrl_train": 65,  "dsrl_extrap": 17},
        ("LORA", 18000):        {"pre_iid": 66, "pre_rep": 76, "dsrl_train": 55,  "dsrl_extrap": 7},
        ("LORA-LM", 4000):      {"pre_iid": 82, "pre_rep": 18, "dsrl_train": 95,  "dsrl_extrap": 91},
        ("LORA-LM", 8000):      {"pre_iid": 74, "pre_rep": 40, "dsrl_train": 97,  "dsrl_extrap": 63},
        ("LORA-LM", 14000):     {"pre_iid": 74, "pre_rep": 58, "dsrl_train": 94,  "dsrl_extrap": 33},
        ("LORA-LM", 18000):     {"pre_iid": 88, "pre_rep": 72, "dsrl_train": 82,  "dsrl_extrap": 34},
    },
}
TARGETS = ["dsrl_train", "dsrl_extrap"]
BASELINES = ["pre_rep", "pre_iid"]  # rollout-based, shown in the report as the bar to beat


# ============================================================================ checkpoints & data

def checkpoint_paths(tsv: Path) -> dict:
    """(task, recipe, step) -> pretrained_model dir, from find_checkpoints.py's checkpoints.tsv."""
    arm_to_recipe = {v: k for k, v in RECIPES.items()}
    out = {}
    for line in tsv.read_text().splitlines():
        task, arm, step, status, path = line.split("\t")
        if status == "ok":
            out[(task, arm_to_recipe[arm], int(step))] = path
    return out


def dataset_root(repo_id: str) -> Path:
    snaps = Path.home() / ".cache/huggingface/lerobot/hub" / f"datasets--{repo_id.replace('/', '--')}" / "snapshots"
    roots = sorted(snaps.glob("*"))
    if not roots:
        raise FileNotFoundError(f"no local snapshot of {repo_id} under {snaps}")
    return roots[-1]


def load_observations(task: str, n_obs: int, seed: int, cache_dir: Path) -> list[dict]:
    """N_obs demo frames with their 32-step action chunk: one random frame from each of N_obs distinct
    episodes (with replacement only if the dataset has fewer episodes), restricted to frames whose full
    chunk lies inside the episode. Cached to disk so every job sees byte-identical inputs."""
    import torch

    cache = cache_dir / f"obs_{task}_n{n_obs}_seed{seed}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    spec = TASKS[task]
    ds = LeRobotDataset(spec["repo_id"], root=dataset_root(spec["repo_id"]), video_backend="pyav",
                        delta_timestamps={"action": [i / spec["fps"] for i in range(H)]})
    ep = ds.meta.episodes
    lengths, starts = np.asarray(ep["length"]), np.asarray(ep["dataset_from_index"])
    rng = np.random.default_rng(seed)
    eps = rng.choice(len(lengths), size=n_obs, replace=n_obs > len(lengths))
    obs = []
    for e in eps:
        f = int(rng.integers(0, lengths[e] - H + 1))
        it = ds[int(starts[e] + f)]
        assert not it["action_is_pad"].any()
        obs.append({THIRD: it[THIRD][None], WRIST: it[WRIST][None], "observation.state": it["observation.state"][None],
                    "task": [it["task"]], "action": it["action"][None],
                    "meta": dict(episode=int(e), frame=f, index=int(starts[e] + f))})
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".{os.getpid()}.tmp")
    torch.save(obs, tmp)
    os.replace(tmp, cache)
    return obs


def noise_sets(task: str, args) -> dict:
    """Common random numbers: identical for every checkpoint of a task (seeded by task + noise seed)."""
    rng = np.random.default_rng([args.noise_seed, list(TASKS).index(task)])
    out = []
    for _ in range(args.n_obs):
        if args.noise_kind == "box":
            W = rng.uniform(-1, 1, (args.m, NOISE_DIM))
        elif args.noise_kind == "tanh_gaussian":
            W = np.tanh(rng.standard_normal((args.m, NOISE_DIM)))
        else:
            W = np.clip(rng.standard_normal((args.m, NOISE_DIM)), -1, 1)
        delta = args.sigma_delta * rng.standard_normal((args.n_lip, NOISE_DIM))
        tau = rng.beta(1.5, 1.0, args.n_fm) * 0.999 + 0.001  # pi05's training time distribution
        eps = rng.standard_normal((args.n_fm, H, NOISE_DIM))
        out.append(dict(W=W.astype(np.float32), delta=delta.astype(np.float32), tau=tau.astype(np.float32),
                        eps=eps.astype(np.float32)))
    return out


# ============================================================================ model

def _sync():
    """torch.cuda.synchronize() when running on a GPU; no-op on CPU (timings are then wall-clock)."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_policy(path: str, device: str, config_from: str | None = None):
    """Checkpoint (incl. LoRA, merged) exactly as DSRL loads it; with config_from, the base weights
    under that checkpoint's config (its cameras), as the steerability 1c reference pass does."""
    import torch  # noqa: F401
    sys.path.insert(0, str(REPO / "steerability"))
    from common import _register_literal_decoder, load_frozen_policy

    _register_literal_decoder()
    if config_from is None:
        return load_frozen_policy(path).to(device).eval()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    return PI05Policy.from_pretrained(path, config=PreTrainedConfig.from_pretrained(config_from)).to(device).eval()


def make_preprocessor(path: str, device: str = "cuda"):
    """The checkpoint's preprocessor; its device step is pointed at `device` (saved configs say cuda)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    pre, _ = make_pre_post_processors(PreTrainedConfig.from_pretrained(path), pretrained_path=path,
                                      preprocessor_overrides={"device_processor": {"device": device}})
    return pre


def prepare(pre, ob: dict) -> dict:
    """Preprocessor on a copy of the frame (third-person camera only as agentview_left, as in DSRL);
    returns the batch with normalized `action` and language/state tokens."""
    return pre({k: (v.clone() if hasattr(v, "clone") else list(v)) for k, v in ob.items() if k != "meta"})


def prefix_kv(policy, batch):
    """VLM prefix pass with KV cache: the first half of PI05Pytorch.sample_actions."""
    import torch
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    m = policy.model
    images, img_masks = policy._preprocess_images(batch)
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    embs, pad, att = m.embed_prefix(images, img_masks, tokens, masks)
    att4 = m._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
    pos = torch.cumsum(pad, dim=1) - 1
    m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, kv = m.paligemma_with_expert.forward(attention_mask=att4, position_ids=pos, past_key_values=None,
                                            inputs_embeds=[embs, None], use_cache=True)
    n_lang = tokens.shape[1]
    return kv, pad, embs.shape[1] - n_lang


def expand_kv(kv, n: int):
    kv = copy.deepcopy(kv)
    kv.batch_repeat_interleave(n)
    return kv


def kv_layers(kv) -> list:
    if hasattr(kv, "layers"):
        return [(lay.keys, lay.values) for lay in kv.layers]
    return list(zip(kv.key_cache, kv.value_cache))


def velocity(policy, kv_b, pad, x, t, emb):
    """v_theta(x_t, t | o) through the action expert only, attending to the cached prefix."""
    B = x.shape[0]
    return policy.model.denoise_step(prefix_pad_masks=pad.expand(B, -1), past_key_values=kv_b, x_t=x,
                                     timestep=t, embodiment_ids=None if emb is None else emb.expand(B))


def decode(policy, kv, pad, noise, emb):
    """Euler integration of the flow from t=1 (noise) to t=0, as in sample_actions: (B, H, 32) -> (B, H, 32)."""
    import torch
    B, steps = noise.shape[0], policy.config.num_inference_steps
    kv_b, dt, x = expand_kv(kv, B), -1.0 / steps, noise
    for s in range(steps):
        t = torch.tensor(1.0 + s * dt, dtype=torch.float32, device=noise.device).expand(B)
        x = x + dt * velocity(policy, kv_b, pad, x, t, emb)
    return x


# ============================================================================ metrics

def spectrum_stats(X: np.ndarray, eps: float) -> dict:
    """X (M, d): Roy-Vetterli effective rank of the centered data, trace and logdet(Sigma + eps I)."""
    M, d = X.shape
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc.astype(np.float64), compute_uv=False)
    tot = s.sum()
    if tot <= 1e-12:
        return dict(erank=1.0, trace=0.0, logdet=d * math.log(eps))
    p = s / tot
    p = p[p > 0]
    lam = s ** 2 / max(M - 1, 1)  # nonzero eigenvalues of the sample covariance
    return dict(erank=float(np.exp(-(p * np.log(p)).sum())), trace=float(lam.sum()),
                logdet=float(np.log(lam + eps).sum() + (d - len(lam)) * math.log(eps)))


def action_metrics(A: np.ndarray, A_pert: np.ndarray, A_ref: np.ndarray, delta: np.ndarray, eps: float) -> dict:
    """A (M, H, 32) decoded chunks; A_pert/A_ref (n_lip, H, 32) chunks at w+delta and w; delta (n_lip, 32)."""
    E = A[:, :H_EXEC, :D_ACT]
    M = E.shape[0]
    r = {}
    for part, sl in (("total", slice(0, D_ACT)), ("arm", ARM), ("hand", HAND)):
        st = spectrum_stats(E[:, :, sl].reshape(M, -1), eps)
        r.update({f"erank_{part}": st["erank"], f"trace_{part}": st["trace"], f"logdet_{part}": st["logdet"]})
    r["hand_arm_trace_ratio"] = r["trace_hand"] / max(r["trace_arm"], 1e-12)
    r["hand_arm_perdim_ratio"] = (r["trace_hand"] / 16) / max(r["trace_arm"] / 6, 1e-12)
    r["finger_synergy_erank"] = float(np.mean([spectrum_stats(E[:, t, HAND], eps)["erank"] for t in range(H_EXEC)]))
    # local Lipschitz gain in w-space (delta lives in the 32-D repeated-noise space)
    dA = (A_pert[:, :H_EXEC, :D_ACT] - A_ref[:, :H_EXEC, :D_ACT])
    dn = np.linalg.norm(delta, axis=1)
    r["lipschitz_local"] = float(np.mean(np.linalg.norm(dA.reshape(len(dA), -1), axis=1) / dn))
    r["lipschitz_arm"] = float(np.mean(np.linalg.norm(dA[:, :, ARM].reshape(len(dA), -1), axis=1) / dn))
    r["lipschitz_hand"] = float(np.mean(np.linalg.norm(dA[:, :, HAND].reshape(len(dA), -1), axis=1) / dn))
    r["signal_to_chaos"] = r["trace_total"] / max(r["lipschitz_local"], 1e-12)
    return r


def kv_drift(kv_ft: list, kv_base, pad, n_img: int) -> dict:
    """Per-token drift of the prefix KV cache vs the base model, over non-padded tokens; means over layers.
    kv_base: tensor (L, 2, heads, T, d) on the same device."""
    import torch
    valid = pad[0].bool()
    img = torch.zeros_like(valid)
    img[:n_img] = True
    cos_l, rel_l, cos_img, cos_lang = [], [], [], []
    for li, (k, v) in enumerate(kv_ft):
        for j, x in enumerate((k, v)):
            a = x[0].float().permute(1, 0, 2).reshape(x.shape[2], -1)  # (T, heads*d)
            b = kv_base[li, j].float().permute(1, 0, 2).reshape(a.shape)
            cos = 1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)
            cos_l.append(cos[valid].mean().item())
            cos_img.append(cos[valid & img].mean().item())
            cos_lang.append(cos[valid & ~img].mean().item())
            rel_l.append(((a - b)[valid].norm() / b[valid].norm().clamp_min(1e-12)).item())
    L = len(kv_ft)
    return dict(kv_cos_dist=float(np.mean(cos_l)), kv_rel_l2=float(np.mean(rel_l)),
                kv_cos_dist_img=float(np.mean(cos_img)), kv_cos_dist_lang=float(np.mean(cos_lang)),
                kv_rel_l2_last=float(np.mean(rel_l[2 * (L - 1):])), kv_rel_l2_first=float(np.mean(rel_l[:2])))


# ============================================================================ run

def base_kv_cache(task, obs, ref_ckpt, args, cache_dir: Path):
    """Base pi05 prefix KV for every observation, fp16 on CPU; cached to disk per task (computed once)."""
    import torch
    path = cache_dir / f"base_kv_{task}_n{args.n_obs}_seed{args.obs_seed}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    t0 = time.time()
    base = load_policy(BASE_MODEL, args.device, config_from=ref_ckpt)
    pre = make_preprocessor(ref_ckpt, args.device)  # every arm's preprocessor is byte-identical within a task
    out = []
    with torch.no_grad():
        for ob in obs:
            kv, _, _ = prefix_kv(base, prepare(pre, ob))
            out.append(torch.stack([torch.stack([k[0], v[0]]) for k, v in kv_layers(kv)]).half().cpu())
    del base
    torch.cuda.empty_cache()
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(out, tmp)
    os.replace(tmp, path)
    print(f"[base] {task}: KV cache for {len(out)} obs in {time.time() - t0:.0f}s -> {path}", flush=True)
    return out


def eval_checkpoint(task, recipe, step, path, obs, noises, base_kv, args) -> tuple[dict, list]:
    import torch
    dev = args.device
    t0 = time.time()
    policy = load_policy(path, dev)
    pre = make_preprocessor(path, dev)
    t_load = time.time() - t0
    per_obs, t_prefix, t_decode = [], 0.0, 0.0
    with torch.no_grad():
        for i, (ob, nz) in enumerate(zip(obs, noises)):
            batch = prepare(pre, ob)
            emb = policy.prepare_embodiment_ids(batch)
            _sync()
            ta = time.time()
            kv, pad, n_img = prefix_kv(policy, batch)
            _sync()
            tb = time.time()
            # M box noises + n_lip perturbed copies of the first n_lip, tiled over the chunk: one batch
            W = torch.as_tensor(nz["W"], device=dev)
            Wp = W[: args.n_lip] + torch.as_tensor(nz["delta"], device=dev)
            noise = torch.cat([W, Wp])[:, None, :].expand(-1, H, -1).contiguous()
            A = decode(policy, kv, pad, noise, emb).float().cpu().numpy()
            # flow-matching velocity loss on the demo chunk, fixed (tau, eps)
            a = torch.nn.functional.pad(batch["action"].float(), (0, NOISE_DIM - D_ACT))  # (1, H, 32), normalized
            tau = torch.as_tensor(nz["tau"], device=dev)
            eps_ = torch.as_tensor(nz["eps"], device=dev)
            x_t = tau[:, None, None] * eps_ + (1 - tau[:, None, None]) * a
            v = velocity(policy, expand_kv(kv, len(tau)), pad, x_t, tau, emb).float()
            se = ((eps_ - a) - v)[..., :D_ACT] ** 2
            _sync()
            t_prefix += tb - ta
            t_decode += time.time() - tb
            r = action_metrics(A[: args.m], A[args.m:], A[: args.n_lip], nz["delta"], args.logdet_eps)
            r.update(fm_total=se.mean().item(), fm_arm=se[..., ARM].mean().item(), fm_hand=se[..., HAND].mean().item(),
                     fm_hand_exec=se[:, :H_EXEC, HAND].mean().item())
            r.update(kv_drift(kv_layers(kv), base_kv[i].to(dev), pad, n_img))
            r.update(ob["meta"])
            per_obs.append(r)
            if args.smoke and i == 0:
                smoke_checks(policy, batch, kv, pad, noise, A, emb, nz, args)
    del policy
    torch.cuda.empty_cache()
    keys = [k for k in per_obs[0] if k not in ("episode", "frame", "index")]
    row = dict(task=TASKS[task]["name"], recipe=recipe, step=step)
    row.update({k: float(np.mean([r[k] for r in per_obs])) for k in keys})
    row.update(n_obs=len(per_obs), m=args.m, n_lip=args.n_lip, sigma_delta=args.sigma_delta, n_fm=args.n_fm,
               noise_kind=args.noise_kind, load_s=round(t_load, 1), prefix_s=round(t_prefix, 2),
               expert_s=round(t_decode, 2), checkpoint=path)
    return row, per_obs


def smoke_checks(policy, batch, kv, pad, noise, A, emb, nz, args):
    """Shapes, slicing, and that the cached-prefix decode reproduces predict_action_chunk."""
    import torch
    print(f"[smoke] prefix tokens {pad.shape[1]} (valid {int(pad.sum())}), KV layers {len(kv_layers(kv))}, "
          f"layer0 K {tuple(kv_layers(kv)[0][0].shape)}", flush=True)
    print(f"[smoke] noise {tuple(noise.shape)} (tiled: {bool((noise == noise[:, :1]).all())}), decoded {A.shape}, "
          f"executed {A[:, :H_EXEC, :D_ACT].shape}, arm {A[:, :H_EXEC, ARM].shape}, hand {A[:, :H_EXEC, HAND].shape}",
          flush=True)
    ref = policy.predict_action_chunk(batch, noise=noise[:1]).float().cpu().numpy()  # B=1 reference path
    mine = decode(policy, kv, pad, noise[:1], emb).float().cpu().numpy()[..., : ref.shape[-1]]
    print(f"[smoke] predict_action_chunk {ref.shape} vs cached decode, B=1: max|diff| {np.abs(ref - mine).max():.2e}; "
          f"vs B={len(noise)} batch row 0: {np.abs(ref[0] - A[0, :, : ref.shape[-1]]).max():.2e} "
          f"(bf16 batch-shape effect); action range [{A.min():.2f}, {A.max():.2f}]", flush=True)
    for n in (1, 4, args.m + args.n_lip):
        z = noise[:1].expand(n, -1, -1).contiguous()
        _sync()
        t = time.time()
        decode(policy, kv, pad, z, emb)
        _sync()
        t1 = time.time()
        policy.predict_action_chunk({k: (v.expand(n, *v.shape[1:]) if hasattr(v, "expand") and v.ndim and v.shape[0] == 1
                                         else v) for k, v in batch.items()}, noise=z)
        _sync()
        print(f"[smoke] B={n}: cached-prefix decode {1e3 * (t1 - t):.0f} ms vs full predict_action_chunk "
              f"{1e3 * (time.time() - t1):.0f} ms", flush=True)


def append_row(csv_path: Path, row: dict):
    """Append under an exclusive lock (several jobs may write the same CSV); header from the first row."""
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        rows = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
        rows = [r for r in rows if (r["task"], r["recipe"], int(r["step"])) != (row["task"], row["recipe"], row["step"])]
        rows.append(row)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        tmp = csv_path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, csv_path)


def done_keys(csv_path: Path) -> set:
    import csv
    if not csv_path.exists():
        return set()
    return {(r["task"], r["recipe"], int(r["step"])) for r in csv.DictReader(open(csv_path))}


def cmd_run(args):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out)
    cache_dir, per_dir = out / "offline_steerability" / "cache", out / "offline_steerability" / "per_ckpt"
    per_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out / ("offline_steerability_smoke.csv" if args.smoke else "offline_steerability_metrics.csv")
    ckpts = checkpoint_paths(Path(args.checkpoints_tsv))
    done = set() if args.overwrite else done_keys(csv_path)
    for task in args.tasks:
        # explicit --steps may include non-DSRL SFT steps (no DSRL outcome; `analyze` ignores them)
        todo = [(r, s) for r in args.recipes for s in (args.steps or TASKS[task]["steps"])
                if (TASKS[task]["name"], r, s) not in done]
        if not todo:
            print(f"[{task}] nothing to do", flush=True)
            continue
        obs = load_observations(task, args.n_obs, args.obs_seed, cache_dir)
        noises = noise_sets(task, args)
        print(f"[{task}] {len(obs)} obs (episodes {[o['meta']['episode'] for o in obs]}), "
              f"{len(todo)} checkpoints to do", flush=True)
        ref_ckpt = ckpts[(task, todo[0][0], todo[0][1])]
        base_kv = base_kv_cache(task, obs, ref_ckpt, args, cache_dir)
        for recipe, step in todo:
            path = ckpts.get((task, recipe, step))
            if path is None:
                print(f"[{task}] {recipe}@{step}: no readable checkpoint, skipped", flush=True)
                continue
            t0 = time.time()
            row, per_obs = eval_checkpoint(task, recipe, step, path, obs, noises, base_kv, args)
            tag = f"{task}_{recipe}_{step:06d}" + ("_smoke" if args.smoke else "")
            (per_dir / f"{tag}.json").write_text(json.dumps(dict(summary=row, per_obs=per_obs), indent=1))
            append_row(csv_path, row)
            print(f"[{task}] {recipe}@{step} in {time.time() - t0:.0f}s (load {row['load_s']}s, prefix {row['prefix_s']}s, "
                  f"expert {row['expert_s']}s): " + ", ".join(
                      f"{k} {row[k]:.4g}" for k in ("erank_total", "erank_hand", "trace_total", "hand_arm_trace_ratio",
                                                    "lipschitz_local", "fm_hand", "kv_cos_dist")), flush=True)


# ============================================================================ analysis

# metric -> expected sign of its association with DSRL success (None = no prior); shown, not used for ranking
METRIC_GROUPS = {
    "1 latent-to-action rank": ["erank_total", "erank_arm", "erank_hand", "trace_total", "trace_arm", "trace_hand",
                                "logdet_total", "logdet_arm", "logdet_hand"],
    "2 finger vs arm": ["hand_arm_trace_ratio", "hand_arm_perdim_ratio", "finger_synergy_erank"],
    "3 smoothness": ["lipschitz_local", "lipschitz_arm", "lipschitz_hand", "signal_to_chaos"],
    "4 flow-matching loss": ["fm_total", "fm_arm", "fm_hand", "fm_hand_exec"],
    "5 KV drift vs base": ["kv_cos_dist", "kv_rel_l2", "kv_cos_dist_img", "kv_cos_dist_lang", "kv_rel_l2_first",
                           "kv_rel_l2_last"],
    "6 composite": ["erank_hand_over_fm_hand", "z_erank_total_minus_z_fm_hand", "z_erank_hand_minus_z_fm_hand",
                    "z_logdet_total_minus_z_fm_hand"],
    "baseline (rollouts)": BASELINES,
}


def add_composites(df):
    import pandas as pd
    z = lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0)  # noqa: E731  within task
    parts = []
    for _, g in df.groupby("task"):
        g = g.copy()
        g["erank_hand_over_fm_hand"] = g["erank_hand"] / g["fm_hand"]
        g["z_erank_total_minus_z_fm_hand"] = z(g["erank_total"]) - z(g["fm_hand"])
        g["z_erank_hand_minus_z_fm_hand"] = z(g["erank_hand"]) - z(g["fm_hand"])
        g["z_logdet_total_minus_z_fm_hand"] = z(g["logdet_total"]) - z(g["fm_hand"])
        parts.append(g)
    return pd.concat(parts)


def partial_spearman(x, y, c):
    """Spearman partial correlation of x and y given c (Pearson on rank residuals)."""
    from scipy.stats import rankdata
    rx, ry, rc = rankdata(x), rankdata(y), rankdata(c)
    res = lambda r: r - np.polyval(np.polyfit(rc, r, 1), rc)  # noqa: E731
    a, b = res(rx), res(ry)
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def within_group_rho(x, y, grp):
    """Spearman over ranks taken within each group (here: the 5 recipes at one SFT-step rung), pooled:
    does the metric order the recipes correctly at a fixed amount of training?"""
    from scipy.stats import rankdata, spearmanr
    rx, ry = np.empty(len(x)), np.empty(len(y))
    for gname in np.unique(grp):
        i = grp == gname
        rx[i], ry[i] = rankdata(x[i]), rankdata(y[i])
    return float(spearmanr(rx, ry)[0])


def correlations(df):
    import pandas as pd
    from scipy.stats import kendalltau, rankdata, spearmanr
    metrics = [m for ms in METRIC_GROUPS.values() for m in ms if m in df]
    group_of = {m: g for g, ms in METRIC_GROUPS.items() for m in ms}
    rows = []
    for scope, g in [*df.groupby("task"), ("pooled", df)]:
        for tgt in TARGETS:
            for m in metrics:
                if scope == "pooled":  # within-task ranks, so the tasks' different scales don't drive the result
                    x = df.groupby("task")[m].transform(lambda s: rankdata(s) / len(s)).to_numpy()
                    y = df.groupby("task")[tgt].transform(lambda s: rankdata(s) / len(s)).to_numpy()
                    c = df.groupby("task")["pre_rep"].transform(lambda s: rankdata(s) / len(s)).to_numpy()
                    st = df.groupby("task")["step"].transform(lambda s: rankdata(s)).to_numpy()
                    grp = (df["task"] + df.groupby("task")["step"].transform(lambda s: rankdata(s)).astype(str)).to_numpy()
                else:
                    x, y, c = g[m].to_numpy(float), g[tgt].to_numpy(float), g["pre_rep"].to_numpy(float)
                    st = g.groupby("task")["step"].transform(lambda s: rankdata(s)).to_numpy()
                    grp = st.astype(str)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 5 or np.ptp(x[ok]) == 0:
                    continue
                rho, p = spearmanr(x[ok], y[ok])
                tau, pt = kendalltau(x[ok], y[ok])
                rows.append(dict(scope=scope, target=tgt, metric=m, group=group_of[m], n=int(ok.sum()), spearman_rho=rho,
                                 spearman_p=p, kendall_tau=tau, kendall_p=pt,
                                 partial_rho_given_pre_rep=np.nan if m == "pre_rep" else partial_spearman(x[ok], y[ok], c[ok]),
                                 partial_rho_given_step=partial_spearman(x[ok], y[ok], st[ok]),
                                 within_step_rho=within_group_rho(x[ok], y[ok], grp[ok])))
    return pd.DataFrame(rows)


def _f(x):
    return "n/a" if not np.isfinite(x) else f"{x:+.2f}"


def fmt_table(c, scope, target, top=None):
    s = c[(c.scope == scope) & (c.target == target)].copy()
    s = s.reindex(s.spearman_rho.abs().sort_values(ascending=False).index)
    if top:
        s = s.head(top)
    lines = ["| rank | metric | group | Spearman ρ | p | Kendall τ | p | partial ρ given pre_rep | partial ρ given SFT step "
             "| within-step ρ |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(s.itertuples(), 1):
        name = f"**{r.metric}**" if r.group.startswith("baseline") else r.metric
        lines.append(f"| {i} | {name} | {r.group} | {r.spearman_rho:+.2f} | {r.spearman_p:.3g} | {r.kendall_tau:+.2f} | "
                     f"{r.kendall_p:.3g} | {_f(r.partial_rho_given_pre_rep)} | {_f(r.partial_rho_given_step)} | {_f(r.within_step_rho)} |")
    return "\n".join(lines)


PALETTE = {  # categorical slots 1, 2, 3, 7, 5 of the reference palette; marker shape is the secondary encoding
    "EXPERT-ONLY": ("#2a78d6", "o"), "FULL-FT": ("#eb6834", "s"), "FROZEN-LM": ("#1baf7a", "^"),
    "LORA": ("#4a3aa7", "D"), "LORA-LM": ("#e87ba4", "P"),
}


def scatter_plots(df, c, plot_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    tasks = [t for t in GROUND_TRUTH_DSRL if t in set(df.task)]
    metrics = [m for ms in METRIC_GROUPS.values() for m in ms if m in df and m not in BASELINES]
    for m in metrics:
        fig, axes = plt.subplots(len(TARGETS), len(tasks), figsize=(4.6 * len(tasks), 3.6 * len(TARGETS)), squeeze=False)
        for j, task in enumerate(tasks):
            g = df[df.task == task]
            steps = sorted(set(g.step))
            for i, tgt in enumerate(TARGETS):
                ax = axes[i, j]
                for recipe, (col, mk) in PALETTE.items():
                    h = g[g.recipe == recipe].sort_values("step")
                    if h.empty:
                        continue
                    ax.plot(h[m], h[tgt], color=col, lw=1, alpha=0.35, zorder=1)  # SFT-step trajectory per recipe
                    sizes = [36 + 26 * steps.index(s) for s in h.step]
                    ax.scatter(h[m], h[tgt], s=sizes, color=col, marker=mk, edgecolor="white", linewidth=1.2,
                               label=recipe, zorder=2)
                    for x, y, s in zip(h[m], h[tgt], h.step):
                        ax.annotate(f"{s // 1000}k", (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7,
                                    color="#52514e")
                r = c[(c.scope == task) & (c.target == tgt) & (c.metric == m)]
                rho = f"ρ = {r.spearman_rho.iloc[0]:+.2f} (p = {r.spearman_p.iloc[0]:.2g})" if len(r) else ""
                ax.set_title(f"{task}: {tgt}   {rho}", fontsize=9, color="#0b0b0b", loc="left")
                ax.set_xlabel(m)
                ax.set_ylabel(f"{tgt} success (%)")
                ax.set_ylim(-5, 105)
        axes[0, 0].legend(frameon=False, fontsize=7, loc="best", title="recipe (size = SFT step)", title_fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{m}.png", dpi=130)
        plt.close(fig)
    # overview: rho of every metric, per task x target (diverging blue <-> red, gray midpoint)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    all_m = [m for ms in METRIC_GROUPS.values() for m in ms if m in set(c.metric)]
    cols = [(s, t) for s in [*tasks, "pooled"] for t in TARGETS]
    R = np.full((len(all_m), len(cols)), np.nan)
    for a, m in enumerate(all_m):
        for b, (s, t) in enumerate(cols):
            r = c[(c.scope == s) & (c.target == t) & (c.metric == m)]
            if len(r):
                R[a, b] = r.spearman_rho.iloc[0]
    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 3, 0.28 * len(all_m) + 1.5))
    ax.imshow(R, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_yticks(range(len(all_m)), all_m)
    ax.set_xticks(range(len(cols)), [f"{s}\n{t}" for s, t in cols], fontsize=7)
    for a in range(R.shape[0]):
        for b in range(R.shape[1]):
            if np.isfinite(R[a, b]):
                ax.text(b, a, f"{R[a, b]:+.2f}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if abs(R[a, b]) > 0.6 else "#0b0b0b")
    ax.set_title("Spearman ρ vs post-DSRL success", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(plot_dir / "overview_spearman_heatmap.png", dpi=130)
    plt.close(fig)


def cmd_analyze(args):
    import pandas as pd
    out = Path(args.out)
    df = pd.read_csv(out / "offline_steerability_metrics.csv")
    gt = pd.DataFrame([dict(task=t, recipe=r, step=s, **v) for t, d in GROUND_TRUTH_DSRL.items() for (r, s), v in d.items()])
    df = add_composites(df.merge(gt, on=["task", "recipe", "step"], how="inner"))
    c = correlations(df)
    c.to_csv(out / "offline_steerability_correlations.csv", index=False)
    df.to_csv(out / "offline_steerability_metrics_with_gt.csv", index=False)
    scatter_plots(df, c, out / "plots")

    n_by_task = df.groupby("task").size().to_dict()
    meta = df.iloc[0]
    L = ["# Offline steerability metrics vs post-DSRL success", "",
         f"Checkpoints evaluated: {n_by_task} (of 20 per task). Per checkpoint: {int(meta.n_obs)} demo observations, "
         f"M = {int(meta.m)} DSRL noises w ~ {meta.noise_kind} [-1,1]^32 tiled over the 32-step chunk, "
         f"{int(meta.n_lip)} Lipschitz perturbations (σ_δ = {meta.sigma_delta}), {int(meta.n_fm)} flow-matching (τ, ε) "
         "samples; the same observations and noises for every checkpoint of a task.", "",
         "How to read this:",
         "- n = 20 per task, and about 30 metrics are tested against 2 targets in 3 scopes. With that many tests, "
         "|ρ| ≈ 0.45 (p ≈ 0.05) can happen by chance, so look for metrics that hold up in both tasks and in the pooled scope.",
         "- **Bold** rows are the rollout-based baselines (`pre_rep` = SFT success with repeated noise, "
         "`pre_iid` = with iid noise). An offline metric is only useful if it adds information beyond them. "
         "The last column, partial ρ given pre_rep, measures exactly that.",
         "- SFT step alone correlates negatively with DSRL success (more SFT, less steerable), and most metrics drift "
         "with step. Two controls separate a metric from training length: **partial ρ given SFT step**, and "
         "**within-step ρ**, which compares only the 5 recipes at the same step rung.",
         "- Most action metrics (flow-matching loss, trace, logdet, Lipschitz gain, eRank) move together, so read them "
         "as one factor: how strongly the decoded action still responds to the noise. eRank is scale-invariant, and "
         "it *rises* as the response collapses toward small isotropic jitter, so its negative sign is expected.",
         "- `pooled` ranks each metric and each target within its task first, so the two tasks' different scales "
         "don't create spurious correlation (n = 40).",
         "- The demo episodes were all used for SFT (neither dataset has a held-out split), so the flow-matching loss "
         "is measured on training data, not on validation data.", ""]
    best = c[~c.group.str.startswith("baseline")].copy()
    best["abs"] = best.spearman_rho.abs()
    L += ["## Headline: strongest offline metric per scope and target", "",
          "| scope | target | best offline metric | ρ | p | partial ρ given pre_rep | pre_rep ρ |", "|---|---|---|---:|---:|---:|---:|"]
    for (s, t), g in best.groupby(["scope", "target"], sort=False):
        r = g.sort_values("abs").iloc[-1]
        b = c[(c.scope == s) & (c.target == t) & (c.metric == "pre_rep")]
        L.append(f"| {s} | {t} | {r.metric} | {r.spearman_rho:+.2f} | {r.spearman_p:.2g} | "
                 f"{_f(r.partial_rho_given_pre_rep)} | {b.spearman_rho.iloc[0]:+.2f} |" if len(b) else "")
    L += ["", "![overview](plots/overview_spearman_heatmap.png)", ""]
    for scope in [*[t for t in GROUND_TRUTH_DSRL if t in n_by_task], "pooled"]:
        for tgt in TARGETS:
            L += [f"## {scope}: {tgt}", "", fmt_table(c, scope, tgt), ""]
    L += ["## Per-checkpoint values", "",
          df[["task", "recipe", "step", "dsrl_train", "dsrl_extrap", "pre_rep", "erank_total", "erank_hand",
              "hand_arm_trace_ratio", "lipschitz_local", "signal_to_chaos", "fm_hand", "kv_cos_dist",
              "z_erank_total_minus_z_fm_hand"]].sort_values(["task", "recipe", "step"])
          .to_markdown(index=False, floatfmt=".3g"), "",
          "Scatter plots: `plots/<metric>.png`. Color and marker shape show the recipe, marker size and the label "
          "show the SFT step, and lines connect each recipe's steps."]
    (out / "steerability_correlation_report.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out / 'steerability_correlation_report.md'}, {out / 'offline_steerability_correlations.csv'}, "
          f"{len(list((out / 'plots').glob('*.png')))} plots")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    r.add_argument("--recipes", nargs="+", default=list(RECIPES), choices=list(RECIPES))
    r.add_argument("--steps", nargs="+", type=int, default=None, help="default: the task's 4 DSRL rungs")
    r.add_argument("--n-obs", type=int, default=32)
    r.add_argument("--m", type=int, default=64, help="noise draws per observation")
    r.add_argument("--n-lip", type=int, default=16, help="noises (of the M) perturbed for the Lipschitz gain")
    r.add_argument("--sigma-delta", type=float, default=0.05)
    r.add_argument("--n-fm", type=int, default=16, help="(tau, eps) samples per observation for the FM loss")
    r.add_argument("--noise-kind", default="box", choices=["box", "tanh_gaussian", "clipped_gaussian"])
    r.add_argument("--logdet-eps", type=float, default=1e-4)
    r.add_argument("--obs-seed", type=int, default=0)
    r.add_argument("--noise-seed", type=int, default=1234)
    r.add_argument("--smoke", action="store_true", help="extra shape/consistency/timing checks; separate CSV")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--checkpoints-tsv", default=str(CHECKPOINTS_TSV))
    r.add_argument("--device", default="cuda")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    a = sub.add_parser("analyze")
    a.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    if args.cmd == "run":
        assert args.n_lip <= args.m
        cmd_run(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
