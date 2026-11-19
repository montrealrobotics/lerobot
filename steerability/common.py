"""Shared plumbing for the steerability diagnostics.

Everything here is numpy except `Pi05Handle`, which imports torch/lerobot lazily, so the
analysis code and the mock backend run on a machine without either.

Conventions
-----------
* A *noise vector* z has shape (noise_dim,) = (32,). "Duplicated" noise means z is repeated
  over all chunk_size steps of pi05's chunk -- exactly what DSRL with noise_chunk_size=1 does.
* `PolicyHandle.chunk(batch, noise)` takes noise of shape (B, chunk_size, noise_dim) and
  returns (B, chunk_size, action_dim) in the policy's *model* space (before the
  postprocessor). Geometry metrics are computed there so dims are comparable.
* `PolicyHandle.to_env(chunk)` applies the postprocessor -> env-space actions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import time
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TASK_SPECS = {
    "lamp": dict(
        robocasa_task="ScrewLightbulb", action_dim=22, arm_dims=6, chunk_size=32,
        n_action_steps=16, noise_dim=32, max_steps=990,  # DSRL episode cap, measured from wandb
        train_cells=["s1_start_0", "s1_start_1"], heldout_cells=["s1_mid"],
        dsrl_checkpoints=[2000, 4000, 8000, 16000],
    ),
    "coffee": dict(
        robocasa_task="CoffeePressButton", action_dim=22, arm_dims=6, chunk_size=32,
        n_action_steps=16, noise_dim=32, max_steps=156,  # DSRL episode cap, measured from wandb
        train_cells=["seed0_L8S6", "seed1_L4S8"], heldout_cells=["seed11_L1S4", "seed14_L1S6"],
        dsrl_checkpoints=[4000, 8000, 14000, 18000],
    ),
}

NOISE_KINDS = ("gaussian", "box", "tanh_gaussian", "clipped_gaussian")


# ----------------------------------------------------------------------------- CLI

def add_common_args(p: argparse.ArgumentParser, needs_policy: bool = True, needs_env: bool = True):
    p.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    p.add_argument("--arm", required=True, help="arm0..arm4 (label only)")
    p.add_argument("--sft-step", type=int, required=True, help="SFT checkpoint step (label only)")
    p.add_argument("--out", required=True, help="root output dir shared by all phases")
    if needs_policy:
        p.add_argument("--checkpoint", required=True,
                       help="pretrained_model dir, or 'mock:steer=0.6,drift=0.1' for the mock backend")
        p.add_argument("--device", default="cuda")
        p.add_argument("--micro-batch", type=int, default=16,
                       help="noises per forward pass; outputs agree across sizes only to bf16 precision, "
                            "so keep it fixed across checkpoints that are compared")
    if needs_env:
        p.add_argument("--adapter", default="robocasa", choices=["robocasa", "mock"])
        p.add_argument("--env-factory", default=None,
                       help="module:function(task, cell, seed) -> robosuite env pinned to that cell")
        p.add_argument("--obs-fn", default=None,
                       help="module:function(task, robosuite_obs_dict) -> batch dict for the pi05 preprocessor")
        p.add_argument("--max-steps", type=int, default=None, help="override env horizon (primitive steps)")
    p.add_argument("--cells", default=None, help="comma list; default = all train+heldout cells")
    p.add_argument("--seed", type=int, default=0)
    return p


def cells_for(args) -> list[str]:
    spec = TASK_SPECS[args.task]
    if args.cells:
        return [c.strip() for c in args.cells.split(",") if c.strip()]
    return spec["train_cells"] + spec["heldout_cells"]


def tag(args) -> str:
    return f"{args.task}_{args.arm}_{args.sft_step:06d}"


# ----------------------------------------------------------------------------- noise

def sample_noise(kind: str, n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """(n, dim) float32 noise vectors. `box` & friends are what a [-1,1] tanh actor can emit."""
    if kind == "gaussian":
        z = rng.standard_normal((n, dim))
    elif kind == "box":
        z = rng.uniform(-1.0, 1.0, (n, dim))
    elif kind == "tanh_gaussian":
        z = np.tanh(rng.standard_normal((n, dim)))
    elif kind == "clipped_gaussian":
        z = np.clip(rng.standard_normal((n, dim)), -1.0, 1.0)
    else:
        raise ValueError(kind)
    return z.astype(np.float32)


def dup(z: np.ndarray, chunk_size: int) -> np.ndarray:
    """(B, dim) -> (B, chunk_size, dim): one vector held across the chunk (DSRL's input)."""
    z = np.asarray(z, dtype=np.float32)
    if z.ndim == 1:
        z = z[None]
    return np.repeat(z[:, None, :], chunk_size, axis=1)


class NoiseStream:
    """Noise per macro step for a rollout.

    first:  optional fixed vector for macro step 0 (the noise under test in branching).
    after:  'fixed' keeps `first` forever; 'base' draws fresh duplicated gaussian noise;
            'iid' draws fresh per-step gaussian noise (the SFT policy's own sampling).
    """

    def __init__(self, dim, chunk_size, seed, first=None, after="base", kind="gaussian"):
        self.dim, self.T = dim, chunk_size
        self.rng = np.random.default_rng(seed)
        self.first = None if first is None else np.asarray(first, np.float32)
        self.after, self.kind = after, kind
        self.used: list[np.ndarray] = []

    def __call__(self, macro: int) -> np.ndarray:
        if self.first is not None and (macro == 0 or self.after == "fixed"):
            z = self.first
            out = dup(z, self.T)[0]
        elif self.after == "iid":
            z = None
            out = self.rng.standard_normal((self.T, self.dim)).astype(np.float32)
        else:
            z = sample_noise(self.kind, 1, self.dim, self.rng)[0]
            out = dup(z, self.T)[0]
        if z is not None:
            self.used.append(z)
        return out  # (T, dim)


# ----------------------------------------------------------------------------- policies

def _register_literal_decoder():
    import draccus  # noqa
    draccus.decode.register(typing.Literal, lambda raw_value, path=(): raw_value)


def load_frozen_policy(path: str):
    """Load a checkpoint exactly as DSRL does, by calling DSRL's own `load_frozen_policy`
    (examples/tutorial/rl/dsrl_pi05_robocasa_example.py): plain from_pretrained for full
    checkpoints; for LoRA checkpoints (adapter_config.json), base weights + adapter, merged.
    Override the script location with $DSRL_SCRIPT."""
    import importlib.util
    import os

    script = os.environ.get("DSRL_SCRIPT")
    if not script:
        import lerobot
        repo = Path(lerobot.__file__).resolve().parents[2]  # <repo>/src/lerobot -> <repo>
        script = str(repo / "examples/tutorial/rl/dsrl_pi05_robocasa_example.py")
    if not Path(script).exists():
        raise FileNotFoundError(f"DSRL script not found at {script}; set $DSRL_SCRIPT")
    spec = importlib.util.spec_from_file_location("_dsrl_example", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module-level code only; its main() is behind __main__
    return mod.load_frozen_policy(path)


def _repeat_batch(batch: dict, n: int):
    import torch
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == 1:
            out[k] = v.expand(n, *v.shape[1:]).contiguous()
        elif isinstance(v, list) and len(v) == 1:
            out[k] = v * n
        else:
            out[k] = v
    return out


class Pi05Handle:
    """Frozen pi05 checkpoint + its own pre/post-processors.

    processor_path lets a *different* model (e.g. lerobot/pi05_base, for representation
    drift) consume batches built by this checkpoint's preprocessor.
    """

    def __init__(self, path: str, device: str, obs_fn, task: str, processor_path: str | None = None,
                 micro_batch: int = 16):
        import torch
        _register_literal_decoder()
        from lerobot.policies.factory import make_pre_post_processors

        self.torch, self.device, self.task, self.obs_fn = torch, device, task, obs_fn
        is_lora = (Path(path) / "adapter_config.json").exists()
        if is_lora or Path(path).exists():
            self.policy = load_frozen_policy(path).to(device).eval()  # same loader as DSRL
        else:  # hub id such as lerobot/pi05_base (reference model for CKA)
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            # Base weights under the evaluated checkpoint's config, as DSRL's LoRA loader does: pi05_base's own
            # config names other cameras (base_0_rgb, ...), so it would see no images in our batches.
            cfg = None
            if processor_path:
                from lerobot.configs.policies import PreTrainedConfig
                cfg = PreTrainedConfig.from_pretrained(processor_path)
            self.policy = PI05Policy.from_pretrained(path, config=cfg).to(device).eval()
        self.is_lora = is_lora
        proc_src = processor_path or path
        if processor_path:
            from lerobot.configs.policies import PreTrainedConfig
            proc_cfg = PreTrainedConfig.from_pretrained(processor_path)
        else:
            proc_cfg = self.policy.config
        self.pre, self.post = make_pre_post_processors(proc_cfg, pretrained_path=proc_src)
        self.chunk_size = self.policy.config.chunk_size
        self.noise_dim = self.policy.config.max_action_dim
        self.action_dim = TASK_SPECS[task]["action_dim"]
        self.micro_batch = micro_batch

    def prepare(self, raw_obs):
        """Same as DSRL's make_prepare_frozen_obs_fn: frame, + embodiment_id (the checkpoint's
        default unless overridden), then the checkpoint's preprocessor."""
        with self.torch.no_grad():
            frame = self.obs_fn(self.task, raw_obs)
            emb = getattr(self.policy.config, "default_embodiment_id", None)
            if emb is not None and "embodiment_id" not in frame:
                frame["embodiment_id"] = self.torch.tensor([emb])
            return self.pre(frame)

    def chunk(self, batch, noise: np.ndarray) -> np.ndarray:
        torch = self.torch
        outs = []
        with torch.no_grad():
            for i in range(0, noise.shape[0], self.micro_batch):
                nz = torch.as_tensor(noise[i:i + self.micro_batch], device=self.device)
                b = _repeat_batch(batch, nz.shape[0])
                a = self.policy.predict_action_chunk(b, noise=nz)
                outs.append(a.float().cpu().numpy()[..., : self.action_dim])
        # complementary data DSRL's _unnormalize forwards to the postprocessor
        self._comp = {k: batch[k][:1] for k in ("embodiment_id", "normalization_id")
                      if k in batch and hasattr(batch[k], "shape")}
        return np.concatenate(outs, 0)

    def to_env(self, chunk: np.ndarray) -> np.ndarray:
        """Same as DSRL's _unnormalize: flatten to (B*T, D), forward embodiment/normalization ids
        as complementary data, reshape back."""
        torch = self.torch
        B, T, D = chunk.shape
        flat = torch.as_tensor(chunk.reshape(B * T, D))
        comp = getattr(self, "_comp", None) or {}
        flat_comp = {k: v.reshape(-1)[:1].to(flat.device).repeat(B * T) for k, v in comp.items()} or None
        with torch.no_grad():
            try:
                a = self.post(flat, complementary_data=flat_comp)
            except TypeError:  # older processor pipelines without complementary_data
                a = self.post(flat)
        a = a.float().cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
        return a.reshape(B, T, -1)[..., : self.action_dim]

    def features(self, batch, module_suffixes: list[str]) -> dict[str, np.ndarray]:
        """Mean-pooled output of each module's FIRST call during one sample (the prefix pass)."""
        torch = self.torch
        named = dict(self.policy.named_modules())
        grabbed, hooks = {}, []
        for suf in module_suffixes:
            hits = [n for n in named if n.endswith(suf)]
            if not hits:
                warnings.warn(f"no module ends with '{suf}'")
                continue
            name = min(hits, key=len)

            def hook(_m, _inp, out, name=suf):
                if name in grabbed:
                    return
                t = getattr(out, "last_hidden_state", None)
                if t is None:
                    t = out[0] if isinstance(out, (tuple, list)) else out
                grabbed[name] = t.float().mean(dim=1).cpu().numpy() if t.ndim == 3 else t.float().cpu().numpy()

            hooks.append(named[name].register_forward_hook(hook))
        try:
            rng = np.random.default_rng(0)
            self.chunk(batch, dup(sample_noise("gaussian", 1, self.noise_dim, rng), self.chunk_size))
        finally:
            for h in hooks:
                h.remove()
        return grabbed

    def release(self):
        del self.policy
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def load_policy(args, adapter, path: str | None = None, processor_path: str | None = None):
    key = "load_s" if path in (None, args.checkpoint) else "load_ref_s"
    path = path or args.checkpoint
    with timed(key):
        if path.startswith("mock:"):
            from adapters import MockPolicy
            return MockPolicy.from_spec(path, args.task)
        return Pi05Handle(path, args.device, adapter.obs_fn, args.task, processor_path=processor_path,
                          micro_batch=getattr(args, "micro_batch", 16))


# ----------------------------------------------------------------------------- rollouts

@dataclass
class Rollout:
    success: bool
    steps: int
    noises: list = field(default_factory=list)
    records: list = field(default_factory=list)


def rollout(policy, adapter, env, raw_obs, noise: NoiseStream, max_steps: int, n_action_steps: int,
            start_step: int = 0, record_every: int | None = None) -> Rollout:
    t, macro, records = start_step, 0, []
    while t < max_steps:
        if record_every and macro % record_every == 0:
            records.append(dict(macro=macro, t=t, sim_state=adapter.get_state(env),
                                raw_obs=adapter.snapshot_obs(raw_obs), probe=adapter.probe_targets(env)))
        nz = noise(macro)[None]  # (1, T, dim)
        acts = policy.to_env(policy.chunk(policy.prepare(raw_obs), nz))[0][:n_action_steps]
        for a in acts:
            raw_obs, success, done = adapter.step(env, a)
            t += 1
            if success:
                return Rollout(True, t, noise.used, records)
            if done or t >= max_steps:
                return Rollout(False, t, noise.used, records)
        macro += 1
    return Rollout(False, t, noise.used, records)


# ----------------------------------------------------------------------------- IO

def result_path(out: str, phase: str, args) -> Path:
    p = Path(out) / phase / f"{tag(args)}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def bank_info(args) -> dict | None:
    """Placement bank the env factory builds this task's cells from: {path, md5}. None for mock.

    The wiring module names it <TASK>_BANK (e.g. LAMP_BANK); a factory without one records path None."""
    spec = getattr(args, "env_factory", None)
    if getattr(args, "adapter", "robocasa") == "mock" or not spec:
        return None
    path = getattr(importlib.import_module(spec.partition(":")[0]), f"{args.task.upper()}_BANK", None)
    if path is None:
        return dict(path=None, md5=None)
    with open(path, "rb") as f:
        return dict(path=path, md5=hashlib.md5(f.read()).hexdigest())


def states_info(path: str | None) -> dict | None:
    """Provenance of a state set: file hash plus the bank and checkpoint it was collected with."""
    if not path:
        return None
    m = load_states(path).get("meta", {})
    return dict(path=str(Path(path).resolve()), sha1=file_hash(path), bank=m.get("bank"),
                checkpoint=m.get("checkpoint"))


_T0 = time.time()
TIMING: dict[str, float] = {}


@contextmanager
def timed(key: str):
    """Accumulate wall-clock seconds under TIMING[key]; write_result stores TIMING in meta.timing."""
    t = time.time()
    try:
        yield
    finally:
        TIMING[key] = TIMING.get(key, 0.0) + time.time() - t


def write_result(path: Path, args, per_env: dict, extra: dict | None = None, phase: str = ""):
    meta = dict(phase=phase, task=args.task, arm=args.arm, sft_step=args.sft_step,
                checkpoint=getattr(args, "checkpoint", None), written=time.strftime("%Y-%m-%d %H:%M:%S"),
                bank=bank_info(args), states=states_info(getattr(args, "states", None)),
                timing=dict(TIMING, total_s=time.time() - _T0),  # total_s: since `import common`
                gpu=_gpu_peak(),
                argv={k: v for k, v in vars(args).items() if _jsonable(v)})
    payload = dict(meta=meta, per_env=per_env)
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(_clean(payload), indent=1))
    print(f"wrote {path}")


def _gpu_peak() -> dict | None:
    """Peak GPU memory of this process, if torch was loaded and CUDA is in use."""
    import sys
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    return dict(peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
                total_gb=torch.cuda.get_device_properties(0).total_memory / 2**30,
                device=torch.cuda.get_device_name(0))


def _jsonable(v):
    return isinstance(v, (str, int, float, bool, type(None), list))


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def load_states(path: str) -> dict:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def file_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:12]


def load_callable(spec: str):
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def nanmean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")