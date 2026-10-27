#!/usr/bin/env python
"""SFT arms on GR00T N1.7 -- the second VLA of the pi05 SFT-arms x DSRL study.

Same question, same two tasks, same data and fairness invariants as
train_pi05_casa_experiments.py (whose task definitions, augmentation and schedule this script
imports rather than copies); only the model, its trainable partitions and its LR differ.

  arm0_expert_only   Whole VLM frozen (vision tower, merger, language model). The action head
                     trains: DiT, VL self-attention + layernorm, state/action encoders and the
                     action decoder. This is also GR00T's own default fine-tune.
  arm1_full_ft       Everything trains.
  arm2_frozen_llm    Only the language model is frozen; the vision tower (incl. its mergers, the
                     projector analogue) and the whole action head train.
  arm3_lora_r32      LoRA r=32 on every language-model linear and every action-transformer linear
                     (DiT incl. its adaLN, timestep encoder and output projections, VL
                     self-attention); vision tower + action encoders/decoders/position embedding
                     full-FT (`modules_to_save`); layernorms frozen. Mirrors pi05 arm3.

pi05's arm4 is dropped for GR00T (agreed 2026-10-25).

MEASURED partition (nvidia/GR00T-N1.7-3B, 3,144M parameters; language model truncated to its first
16 layers by select_layer): vision tower 407M (merger 25M + deepstack mergers 76M), language model
1,117M (embeddings 311M), action head 1,621M (DiT 1,083M, VL self-attention 201M, encoders/decoders
~330M). So arm0 trains 1,621M (51.5%) -- far more than pi05's arm0 (17%), because GR00T's action
head is half the model. --dry-run prints and asserts the exact partition of every arm.

DIFFERENCES FROM THE pi05 MATRIX, all deliberate:
  * Peak LR = GR00T's own fine-tune default 1e-4 (pi05 used openpi's 2.5e-5), LoRA at 10x; AdamW
    betas/eps/weight decay are GR00T's defaults too. The schedule shape, length, warmup, batch
    size and grad clip are the study's invariants, shared with pi05.
  * Horizon: GR00T's DiT works on 40 steps; we train on 32-step chunks (steps 32-39 and the 110
    padded action dims are masked out of the loss) and execute 16, as pi05.
OVERRIDDEN TO MATCH pi05 (GrootConfig fields added on this branch, baked into saved processors):
  * normalization_bounds="quantile": q01/q99 bounds, pi05's QUANTILES (GR00T default: min/max).
  * clip_normalized=False: pi05 does not clip normalized values (GR00T default: clip).
  * state_dropout_prob=0.2: pi05's state dropout (GR00T's new-embodiment default: 0).

    python examples/training/train_groot_casa_experiments.py --experiment arm0_expert_only --task lamp --dry-run

Always launch through ~/lerobot-groot/groot_env.sh: it isolates the GR00T venv (transformers 5.5)
from the pi05 one.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pi05_casa_experiments import (  # noqa: E402
    BATCH_SIZE,
    COLOR_JITTER,
    DECAY_FRACTION,
    DECAY_STEPS,
    EVAL_FREQ,
    EXTERNAL_IMAGE_KEY,
    GRAD_CLIP_NORM,
    LEAP_HAND_DIMS,
    RANDOM_EXTERNAL_IMAGE_KEYS,
    LORA_ALPHA,
    LORA_LR_MULT,
    LORA_RANK,
    SAVE_FREQ,
    SCRATCH_OUTPUT_ROOT,
    STATE_DROPOUT_P,
    TASKS,
    TOTAL_STEPS,
    WARMUP_STEPS,
    WRIST_IMAGE_KEY,
    _ARM_JOINTS,
    apply_resume,
    apply_smoke_test_overrides,
    make_image_transforms,
    make_timestamped_output_dir,
    make_visual_feature,
    parse_color_jitter,
    resolve_controller,
)

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.configs.default import DatasetConfig, EvalConfig, PeftConfig, WandBConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.envs.configs import RoboCasaEnv  # noqa: E402
from lerobot.optim import (  # noqa: E402
    CosineDecayWithWarmupSchedulerConfig,
    NamedAdamWConfig,
    NamedParamGroupConfig,
)
from lerobot.policies.groot.configuration_groot import GrootConfig  # noqa: E402
from lerobot.scripts.lerobot_train import train  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

GROOT_BASE_MODEL = "nvidia/GR00T-N1.7-3B"

# Isaac-GR00T N1.7 fine-tune defaults (configuration_groot.py); --dry-run asserts they still match.
GROOT_PEAK_LR = 1e-4
GROOT_LORA_PEAK_LR = GROOT_PEAK_LR * LORA_LR_MULT  # 1e-3
GROOT_ADAM_BETAS = (0.9, 0.999)
GROOT_ADAM_EPS = 1e-8
GROOT_WEIGHT_DECAY = 1e-5

CHUNK_SIZE = 32
N_ACTION_STEPS = 16

# ----------------------------------------------------------------------------
# Module paths. PEFT matches `target_modules` with re.fullmatch on module keys relative to the
# policy root, and `modules_to_save` with key.endswith(...). Every LoRA pattern is anchored under
# the language model or the action head, so it can never reach the vision tower.
# ----------------------------------------------------------------------------
_BACKBONE = r"_groot_model\.backbone\.model\.model"
LM_LINEAR_TARGETS = (
    rf"{_BACKBONE}\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
)
_ATTN_FF = r"attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2)"
HEAD_LINEAR_TARGETS = (
    r"_groot_model\.action_head\.("
    rf"model\.transformer_blocks\.\d+\.({_ATTN_FF}|norm1\.linear)"
    r"|model\.timestep_encoder\.timestep_embedder\.linear_(1|2)"
    r"|model\.proj_out_(1|2)"
    rf"|vl_self_attention\.transformer_blocks\.\d+\.({_ATTN_FF})"
    r")"
)
VISION_MODULES = ["backbone.model.model.visual"]
ACTION_IO_MODULES = [
    "action_head.state_encoder",
    "action_head.action_encoder",
    "action_head.action_decoder",
    "action_head.position_embedding",
]


@dataclass(frozen=True)
class GrootArmSpec:
    name: str
    tune_llm: bool
    tune_visual: bool
    tune_projector: bool = True        # state/action encoders, action decoder, position embedding
    tune_diffusion_model: bool = True  # DiT
    tune_vlln: bool = True             # VL layernorm + VL self-attention
    peft: PeftConfig | None = None
    lr_groups: dict[str, NamedParamGroupConfig] = field(default_factory=dict)
    peak_lr: float = GROOT_PEAK_LR
    # --dry-run: substrings that must have trainable params / must not, and (when set) regexes
    # whose union must be EXACTLY the trainable set.
    expect_trainable: tuple[str, ...] = ()
    expect_frozen: tuple[str, ...] = ()
    expect_trainable_only: tuple[str, ...] = ()
    notes: str = ""

    @property
    def decay_lr(self) -> float:
        return self.peak_lr * DECAY_FRACTION


_HEAD_PARTS = (
    "action_head.model.transformer_blocks",
    "action_head.vl_self_attention",
    "action_head.vlln",
    "action_head.state_encoder",
    "action_head.action_encoder",
    "action_head.action_decoder",
)

EXPERIMENTS: dict[str, GrootArmSpec] = {
    "arm0_expert_only": GrootArmSpec(
        name="arm0_expert_only",
        tune_llm=False,
        tune_visual=False,
        expect_trainable=_HEAD_PARTS,
        expect_frozen=("backbone.",),
        expect_trainable_only=(r"\.action_head\.",),
        notes="Frozen VLM, whole action head. GR00T's own default fine-tune.",
    ),
    "arm1_full_ft": GrootArmSpec(
        name="arm1_full_ft",
        tune_llm=True,
        tune_visual=True,
        expect_trainable=(".visual.blocks", ".visual.merger", ".language_model.layers", *_HEAD_PARTS),
        notes="Everything trains at GR00T's default LR.",
    ),
    "arm2_frozen_llm": GrootArmSpec(
        name="arm2_frozen_llm",
        tune_llm=False,
        tune_visual=True,
        expect_trainable=(".visual.blocks", ".visual.merger", ".visual.deepstack_merger_list", *_HEAD_PARTS),
        expect_frozen=(".language_model.",),
        expect_trainable_only=(r"\.visual\.", r"\.action_head\."),
        notes="Only the language model is frozen; vision tower + mergers and the action head train.",
    ),
    "arm3_lora_r32": GrootArmSpec(
        name="arm3_lora_r32",
        # LoRA freezes everything first; the tune flags only decide which frozen modules GR00T
        # forces into eval mode. The vision tower full-FTs, so keep it in train mode.
        tune_llm=False,
        tune_visual=True,
        peft=PeftConfig(
            method_type="LORA",
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=rf"({LM_LINEAR_TARGETS})|({HEAD_LINEAR_TARGETS})",
            full_training_modules=VISION_MODULES + ACTION_IO_MODULES,
        ),
        lr_groups={
            "lm_lora": NamedParamGroupConfig(patterns=[r"\.language_model\..*lora_"], lr=GROOT_LORA_PEAK_LR),
            "head_lora": NamedParamGroupConfig(patterns=[r"\.action_head\..*lora_"], lr=GROOT_LORA_PEAK_LR),
            "vision_full_ft": NamedParamGroupConfig(patterns=[r"\.visual\."], lr=GROOT_PEAK_LR),
            "action_io": NamedParamGroupConfig(
                patterns=[r"action_head\.(state_encoder|action_encoder|action_decoder|position_embedding)"],
                lr=GROOT_PEAK_LR,
            ),
        },
        peak_lr=GROOT_LORA_PEAK_LR,
        expect_trainable=(
            "language_model.layers.0.self_attn.q_proj.lora_A",
            "language_model.layers.0.mlp.down_proj.lora_A",
            "action_head.model.transformer_blocks.0.attn1.to_q.lora_A",
            "action_head.model.transformer_blocks.0.norm1.linear.lora_A",
            "action_head.vl_self_attention.transformer_blocks.0.ff.net.2.lora_A",
            "action_head.model.timestep_encoder.timestep_embedder.linear_1.lora_A",
            ".visual.blocks",
            ".visual.merger",
            "action_head.action_encoder",
            "action_head.action_decoder",
            "action_head.state_encoder",
            "action_head.position_embedding",
        ),
        expect_frozen=("vlln",),
        expect_trainable_only=(
            r"lora_",
            r"\.visual\.",
            r"action_head\.(state_encoder|action_encoder|action_decoder|position_embedding)",
        ),
        notes="LoRA r=32/alpha=32 on LM + action transformer; vision + action I/O full-FT; layernorms frozen.",
    ),
}


def make_config(
    experiment: str,
    task: str,
    seed: int,
    eval_freq: int = EVAL_FREQ,
    steps: int = TOTAL_STEPS,
    save_freq: int = SAVE_FREQ,
    color_jitter: dict[str, float] | None = None,
    state_dropout_p: float = STATE_DROPOUT_P,
) -> TrainPipelineConfig:
    spec = EXPERIMENTS[experiment]
    task_spec = TASKS[task]
    if color_jitter is None:
        color_jitter = COLOR_JITTER
    metadata = LeRobotDatasetMetadata(task_spec.dataset)
    action_dim = metadata.features[ACTION]["shape"][0]
    state_dim = metadata.features[OBS_STATE]["shape"][0]
    if task_spec.arm_action_dim + LEAP_HAND_DIMS != action_dim:
        raise ValueError(
            f"Task '{task}': controller expects {task_spec.arm_action_dim} arm + {LEAP_HAND_DIMS} hand dims, "
            f"dataset stores {action_dim} ({task_spec.robot} has {_ARM_JOINTS[task_spec.robot]} arm joints)."
        )

    env = RoboCasaEnv(
        task=task_spec.robocasa_task,
        robot=task_spec.robot,
        controller=resolve_controller(task_spec.controller_filename),
        fps=metadata.fps,
        camera_name=task_spec.camera_name,
        scene_seeds=list(task_spec.eval_scene_seeds) if task_spec.eval_scene_seeds else None,
        placement_bank=task_spec.eval_placement_bank,
        placement_ids=list(task_spec.eval_placement_ids) if task_spec.eval_placement_ids else None,
    )

    policy = GrootConfig(
        base_model_path=GROOT_BASE_MODEL,
        embodiment_tag="new_embodiment",
        push_to_hub=False,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            WRIST_IMAGE_KEY: make_visual_feature(metadata, WRIST_IMAGE_KEY),
            EXTERNAL_IMAGE_KEY: make_visual_feature(metadata, EXTERNAL_IMAGE_KEY),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        tune_llm=spec.tune_llm,
        tune_visual=spec.tune_visual,
        tune_projector=spec.tune_projector,
        tune_diffusion_model=spec.tune_diffusion_model,
        tune_vlln=spec.tune_vlln,
        use_relative_actions=False,
        normalization_bounds="quantile",
        clip_normalized=False,
        state_dropout_prob=state_dropout_p,
        # pi05's augmentation: per sample, the external view is the left or right agent-view camera.
        random_external_camera_keys=list(RANDOM_EXTERNAL_IMAGE_KEYS),
        random_external_camera_output_key=EXTERNAL_IMAGE_KEY,
        random_external_camera_p=0.5,
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=task_spec.dataset,
            video_backend="pyav",
            image_transforms=make_image_transforms(color_jitter),
        ),
        env=env,
        job_name=f"groot_sft_arms_{task}_{spec.name}_seed{seed}",
        resume=False,
        seed=seed,
        policy=policy,
        # One optimizer path for every arm (unlike pi05, where only the PEFT arms used it): GR00T's
        # preset would bring its own warmup-ratio/cosine-to-zero schedule, not the study's.
        use_policy_training_preset=False,
        optimizer=NamedAdamWConfig(
            lr=GROOT_PEAK_LR,
            betas=GROOT_ADAM_BETAS,
            eps=GROOT_ADAM_EPS,
            weight_decay=GROOT_WEIGHT_DECAY,
            grad_clip_norm=GRAD_CLIP_NORM,
            param_groups=dict(spec.lr_groups),
            fail_on_unmatched=bool(spec.lr_groups),
        ),
        scheduler=CosineDecayWithWarmupSchedulerConfig(
            peak_lr=spec.peak_lr,
            decay_lr=spec.decay_lr,
            num_warmup_steps=WARMUP_STEPS,
            num_decay_steps=DECAY_STEPS,
        ),
        peft=spec.peft,
        wandb=WandBConfig(enable=True, project="lerobot-dex"),
        steps=steps,
        eval_freq=eval_freq,
        save_freq=save_freq,
        log_freq=50,
        tolerance_s=1e-3,
        eval=EvalConfig(n_episodes=20, n_videos=5, batch_size=task_spec.eval_batch_size),
        batch_size=BATCH_SIZE,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
    )
    cfg.output_dir = make_timestamped_output_dir(SCRATCH_OUTPUT_ROOT, cfg)
    return cfg


# ----------------------------------------------------------------------------
# Dry run: build policy, PEFT wrapper, optimizer and processors exactly as lerobot_train does,
# then assert the arm's trainable set, its optimizer groups and the processor overrides.
# ----------------------------------------------------------------------------


def run_dry_run(cfg: TrainPipelineConfig, spec: GrootArmSpec) -> dict:
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.groot.processor_groot import GrootActionUnpackUnnormalizeStep, GrootN17PackInputsStep

    cfg.validate()
    failures: list[str] = []

    defaults = GrootConfig.__dataclass_fields__
    for name, ours in (
        ("optimizer_lr", GROOT_PEAK_LR),
        ("optimizer_betas", GROOT_ADAM_BETAS),
        ("optimizer_eps", GROOT_ADAM_EPS),
        ("optimizer_weight_decay", GROOT_WEIGHT_DECAY),
    ):
        upstream = defaults[name].default
        if isinstance(upstream, (list, tuple)):
            upstream = tuple(upstream)
        if upstream != ours:
            failures.append(f"GR00T default {name}={upstream} no longer matches this script's {ours}")

    ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, root=cfg.dataset.root)
    policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta, rename_map=cfg.rename_map)
    if cfg.peft is not None:
        policy = policy.wrap_with_peft(peft_cli_overrides=asdict(cfg.peft))

    named = dict(policy.named_parameters())
    trainable = {n: p for n, p in named.items() if p.requires_grad}
    unwrapped = {re.sub(r"\.modules_to_save\.[^.]+\.", ".", n) for n in trainable}
    n_total = sum(p.numel() for p in named.values())
    n_train = sum(p.numel() for p in trainable.values())
    # PEFT's modules_to_save keeps a frozen original next to each trainable copy, so the resident
    # count overstates the model; report the base model's size alongside.
    n_base = sum(p.numel() for n, p in named.items() if "lora_" not in n and ".modules_to_save." not in n)
    print(f"\n[{spec.name}] trainable {n_train / 1e6:,.1f}M / base model {n_base / 1e6:,.1f}M "
          f"({100 * n_train / n_base:.1f}%); resident {n_total / 1e6:,.1f}M")
    # The state/action encoders and the action decoder hold one weight slot per embodiment
    # (max_num_embodiments=32); fine-tuning only ever touches the 'new_embodiment' slot, so the
    # effective size of those modules is 1/32 of their parameter count.
    n_slots = 32
    unused = lambda items: sum(  # noqa: E731
        p.numel() * (n_slots - 1) // n_slots
        for n, p in items
        if re.search(r"action_head\.(state_encoder|action_encoder|action_decoder)\.", n) and p.shape[0] == n_slots
    )
    unused_base = unused((n, p) for n, p in named.items() if "lora_" not in n and ".modules_to_save." not in n)
    unused_train = unused(trainable.items())
    eff_train, eff_base = n_train - unused_train, n_base - unused_base
    print(f"    effective (one embodiment slot): trainable {eff_train / 1e6:,.1f}M / {eff_base / 1e6:,.1f}M "
          f"({100 * eff_train / eff_base:.1f}%)")

    by_part: dict[str, int] = {}
    for n, p in trainable.items():
        key = re.sub(r"\.modules_to_save\.[^.]+", "", n)
        for part in ("visual", "language_model", "action_head.model", "action_head.vl_self_attention",
                     "action_head.vlln", "action_head.state_encoder", "action_head.action_encoder",
                     "action_head.action_decoder", "action_head.position_embedding"):
            if part in key:
                tag = part + (" (lora)" if "lora_" in n else "")
                by_part[tag] = by_part.get(tag, 0) + p.numel()
                break
        else:
            by_part["OTHER"] = by_part.get("OTHER", 0) + p.numel()
    for k, v in sorted(by_part.items(), key=lambda kv: -kv[1]):
        print(f"    {k:42s} {v / 1e6:9.2f}M")

    lora_in_vision = [n for n in trainable if "lora_" in n and ".visual." in n]
    if lora_in_vision:
        failures.append(f"LoRA inside the vision tower: {lora_in_vision[:3]}")
    for needle in spec.expect_trainable:
        if not any(needle in n for n in trainable) and not any(needle in n for n in unwrapped):
            failures.append(f"expected trainable params matching '{needle}', found none")
    for needle in spec.expect_frozen:
        leaked = [n for n in trainable if needle in n]
        if leaked:
            failures.append(f"expected '{needle}' frozen, but {len(leaked)} params train, e.g. {leaked[:3]}")
    if spec.expect_trainable_only:
        allowed = [re.compile(pat) for pat in spec.expect_trainable_only]
        stray = sorted(n for n in trainable if not any(pat.search(n) for pat in allowed))
        if stray:
            failures.append(f"{len(stray)} trainable params outside {list(spec.expect_trainable_only)}, e.g. {stray[:4]}")
        else:
            print("    exact trainable partition OK:",
                  {pat: sum(1 for n in trainable if re.search(pat, n)) for pat in spec.expect_trainable_only})
    if cfg.peft is not None:
        n_lora_modules = len({n.split(".lora_")[0] for n in trainable if "lora_" in n})
        print(f"    LoRA-adapted linears: {n_lora_modules}")

    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    base_lrs = scheduler.base_lrs if scheduler is not None else [g["lr"] for g in optimizer.param_groups]
    covered = 0
    for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
        n_params = sum(p.numel() for p in group["params"] if p.requires_grad)
        covered += n_params
        print(f"    group {str(group.get('name', '?')):16s} peak_lr={base_lr:.2e} wd={group['weight_decay']:g} "
              f"trainable={n_params / 1e6:,.2f}M")
    if covered != n_train:
        failures.append(f"optimizer covers {covered:,} trainable params, policy has {n_train:,}")

    # Processors exactly as lerobot_train builds them (stats only, no dataset_meta).
    pre, post = make_pre_post_processors(policy_cfg=cfg.policy, dataset_stats=ds_meta.stats)
    pack = next(s for s in pre.steps if isinstance(s, GrootN17PackInputsStep))
    decode = next(s for s in post.steps if isinstance(s, GrootActionUnpackUnnormalizeStep))
    q01 = torch.as_tensor(ds_meta.stats[ACTION]["q01"]).float()
    pack_min = torch.as_tensor(pack.stats[ACTION]["min"]).float()[: q01.numel()]
    checks = {
        "pack.state_dropout_prob": (pack.state_dropout_prob, cfg.policy.state_dropout_prob),
        "pack.training": (pack.training, True),
        "pack.clip_outliers": (pack.clip_outliers, False),
        "decode.clip_normalized_action": (decode.clip_normalized_action, False),
        "action min == q01": (bool(torch.allclose(pack_min, q01)), True),
    }
    for name, (got, want) in checks.items():
        ok = got == want
        print(f"    processor {name:30s} {got!s:8s} {'OK' if ok else 'MISMATCH (want ' + str(want) + ')'}")
        if not ok:
            failures.append(f"processor {name} is {got}, expected {want}")

    if failures:
        raise AssertionError("\n  - " + "\n  - ".join(failures))
    print(f"[{spec.name}] dry run OK\n")
    return {"trainable": n_train, "base": n_base, "resident": n_total, "effective_trainable": eff_train,
            "effective_base": eff_base, "by_part": by_part}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="arm0_expert_only")
    parser.add_argument("--task", choices=tuple(TASKS), default="lamp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    parser.add_argument("--save-freq", type=int, default=SAVE_FREQ)
    parser.add_argument("--color-jitter-params", nargs="*", metavar="NAME VALUE", default=None)
    parser.add_argument("--state-dropout-p", type=float, default=STATE_DROPOUT_P)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build policy, optimizer and processors, assert the arm's partition, exit.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="A few real steps + checkpoint + eval rollout + wandb log, into <output root>/smoketest/.")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Run directory of an interrupted run; continues from checkpoints/last.")
    args = parser.parse_args()
    args.color_jitter = parse_color_jitter(args.color_jitter_params)
    return args


if __name__ == "__main__":
    args = parse_args()
    register_third_party_plugins()
    cfg = make_config(args.experiment, args.task, args.seed, args.eval_freq, args.steps,
                      save_freq=args.save_freq, color_jitter=args.color_jitter,
                      state_dropout_p=args.state_dropout_p)
    if args.smoke_test:
        cfg = apply_smoke_test_overrides(cfg)
        if args.steps != TOTAL_STEPS:  # e.g. extend a smoke run to test --resume-from
            cfg.steps = args.steps
    if args.resume_from:
        # pi05's resume (config_path on argv, use_peft instead of re-wrapping LoRA, original
        # output_dir). GR00T uses no optimizer preset, so its preset branch is a no-op here.
        cfg = apply_resume(cfg, args.resume_from)
    if args.dry_run:
        run_dry_run(cfg, EXPERIMENTS[args.experiment])
    else:
        train(cfg)
