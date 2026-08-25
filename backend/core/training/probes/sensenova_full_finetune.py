"""Phase U-2 exit smoke: a real SenseNova full fine-tune against the real checkpoint.

Two arms, one process each (``--arm``), run one at a time under the same VRAM
gate the Phase 1 / U-1 probes use.

* ``train``  -- a real ``FullParameterTrainer`` run on the generation half:
                finite loss for every step, the fused backward pass installed,
                stochastic rounding forced on and attached, the per-step
                updated-parameter census complete, and the UPDATE-NONZERO CENSUS
                (SENSENOVA_TRAINING_DESIGN.md 13.4 U-2-5) over the 294 generation
                decoder Linears -- how many of them the run actually moved,
                measured by comparing a per-Linear digest taken before the first
                step against the same digest after the last one.
                Then one checkpoint save, and the digests of what was saved.
* ``reload`` -- a FRESH process with no trainer: the saved checkpoint read back
                through the production reader (``load_sensenova_from_path``),
                with the per-Linear digests compared bit for bit against what the
                training process reported.

The two arms are separate processes because their host-RAM peaks do not overlap:
the trainer holds the dequantized half while the reader holds a whole second
model.

Nothing here claims quality. Phase 2b's exit criteria are "training is not
broken" only; the horizon is far too short for stochastic rounding's error to be
below the signal (6.3).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.training.probes.sensenova_real_checkpoint import (  # noqa: E402
    EXIT_SMOKE_HEIGHT,
    EXIT_SMOKE_STEPS,
    EXIT_SMOKE_WIDTH,
    EXPECTED_TARGETS,
    _ExitSmokeDataset,
    _apply_vram_gate,
    _cuda_memory,
    _require_repo_venv,
    _write_deterministic_smoke_image,
    trainer_exit_smoke_config,
)

RUN_NAME = "sensenova_u2_full_finetune_smoke"
# The LOWEST full-FT learning rate this repo names anywhere -- the setdefault in
# generate_full_finetune_config, which only a caller that omits the key reaches
# (the API sends 1e-4 and the UI 1e-5, so no product run gets it). Chosen for
# that reason: a census run at the smallest rate cannot be accused of having
# picked a value that makes weights move.
LEARNING_RATE = 1e-6
# Enough tensors to read an element-level moved fraction from without cloning
# 15 GiB of weights.
SAMPLED_LINEARS = 4


def _host_rss_bytes() -> int:
    import psutil

    return int(psutil.Process(os.getpid()).memory_info().rss)


def _host_peak_bytes() -> int:
    """Peak working set of THIS process, from the OS rather than a sampler."""
    import psutil

    info = psutil.Process(os.getpid()).memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def _linear_digest(weight: torch.Tensor) -> str:
    """SHA-256 of the weight's exact bytes.

    Deliberately not a float reduction: the first version of this probe summed
    in float64 and reported 137 of 294 Linears as changed by the save/reload
    round trip, because the training arm reduces on CUDA and the reload arm on
    CPU and the two accumulate in different orders. A byte digest is the only
    fingerprint that means the same thing on both sides, which is the whole
    point of comparing them.
    """
    import hashlib

    w = weight.detach().to("cpu").contiguous()
    return hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest()


def _gen_targets(transformer) -> list[tuple[str, Any]]:
    from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets

    return [
        (path, module)
        for path, _parent, _attr, module in iter_sensenova_lora_targets(
            transformer, branch="gen"
        )
    ]


def _digest_map(transformer) -> dict[str, list[float]]:
    return {path: _linear_digest(m.weight) for path, m in _gen_targets(transformer)}


# ---------------------------------------------------------------------------
# Arm 1 -- the run
# ---------------------------------------------------------------------------


def _run_train_arm(args: argparse.Namespace) -> dict[str, Any]:
    from core.training.full_parameter_trainer import FullParameterTrainer
    from core.training.ops.sensenova_ops import resolve_full_finetune_branch
    from core.training.ops.training_method import is_full_finetune

    vram_gate = _apply_vram_gate()
    config = trainer_exit_smoke_config()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    image_path = workdir / "training_image.png"
    output_dir = workdir / "full_finetune"
    _write_deterministic_smoke_image(image_path)

    train_config = dict(config["train_config"])
    train_config["sensenova_full_finetune_save_format"] = args.save_format

    torch.cuda.reset_peak_memory_stats()
    host_before_load = _host_rss_bytes()
    load_started = time.perf_counter()
    trainer = FullParameterTrainer(
        model_path=args.model_path,
        output_dir=str(output_dir),
        run_name=RUN_NAME,
        run_id=None,
        learning_rate=LEARNING_RATE,
        unet_lr=LEARNING_RATE,
        train_unet=True,
        train_text_encoder=False,
        device="cuda",
        weight_dtype="bf16",
        training_dtype="bf16",
        output_dtype="bf16",
        vae_dtype="bf16",
        mixed_precision=True,
        attention_backend="native",
        use_flash_attention=False,
        blocks_to_swap=0,
        train_config=train_config,
    )
    model_load_wall_time_s = time.perf_counter() - load_started
    host_after_load = _host_rss_bytes()
    model_resident = _cuda_memory()

    if not is_full_finetune(trainer):
        raise AssertionError("the trainer did not resolve as a full fine-tune")
    branch = resolve_full_finetune_branch(trainer)
    if branch != "gen":
        raise AssertionError(f"expected the gen branch, got {branch!r}")
    adapter_name = type(trainer.adapter).__name__
    if adapter_name != "SenseNovaFullParameterAdapter":
        raise AssertionError(
            f"the SD1.5 fallthrough was taken: adapter is {adapter_name}"
        )
    targets = _gen_targets(trainer.transformer)
    if len(targets) != EXPECTED_TARGETS:
        raise AssertionError(f"expected {EXPECTED_TARGETS} targets, got {len(targets)}")

    groups = trainer.setup_trainable_parameters()
    parameter_groups = [
        {"lr": g["lr"], "tensors": len(g["params"]),
         "elements": sum(p.numel() for p in g["params"])}
        for g in groups
    ]
    trainable_elements = sum(g["elements"] for g in parameter_groups)

    before = _digest_map(trainer.transformer)
    sampled = [path for path, _ in targets[:SAMPLED_LINEARS]]
    sampled_before = {
        path: dict(targets)[path].weight.detach().clone() for path in sampled
    }

    # Armed here, before train() builds the optimizer: it is the mechanism that
    # catches a hook that never fires, which no loss curve can show.
    trainer.optimizer_update_census = True

    losses: list[float] = []
    steps: list[int] = []

    def progress_callback(phase, step, total, epoch=0, loss=None):
        del total, epoch
        if phase != "training":
            return
        if loss is None or not math.isfinite(float(loss)):
            raise AssertionError(f"non-finite SenseNova full-FT loss: {loss!r}")
        steps.append(int(step))
        losses.append(float(loss))

    train = dict(config["train"])
    train.update({
        "num_epochs": 1,
        "optimizer_type": "adafactor",
        "sample_prompts": [],
        "sample_every_n_steps": 0,
        "sample_width": EXIT_SMOKE_WIDTH,
        "sample_height": EXIT_SMOKE_HEIGHT,
        "sample_seed": args.seed,
        "max_grad_norm": 0.0,
        "progress_callback": progress_callback,
        "run_id": None,
        "max_step_saves_to_keep": 1,
        "force_recache": False,
    })
    dataset = _ExitSmokeDataset(image_path, args.prompt)

    train_started = time.perf_counter()
    trainer.train(datasets=[dataset], **train)
    train_wall_time_s = time.perf_counter() - train_started

    if steps != list(range(1, EXIT_SMOKE_STEPS + 1)):
        raise AssertionError(f"expected {EXIT_SMOKE_STEPS} steps, got {steps}")

    after = _digest_map(trainer.transformer)
    moved = sorted(p for p in before if after[p] != before[p])
    unmoved = sorted(p for p in before if after[p] == before[p])
    sampled_moved_fraction = {}
    for path in sampled:
        now = dict(_gen_targets(trainer.transformer))[path].weight.detach()
        changed = (now != sampled_before[path]).sum().item()
        sampled_moved_fraction[path] = {
            "elements": int(now.numel()),
            "changed": int(changed),
            "fraction": float(changed) / float(now.numel()),
        }

    census = trainer._update_census
    checkpoint = output_dir / f"{RUN_NAME}_step_{EXIT_SMOKE_STEPS:06d}.safetensors"
    entry = checkpoint if checkpoint.is_file() else Path(str(checkpoint) + ".index.json")
    if not entry.is_file():
        raise AssertionError(f"the run saved neither {checkpoint} nor its shard index")

    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    written_bytes = sum(
        p.stat().st_size for p in entry.parent.glob(f"{entry.stem.split('.')[0]}*")
        if p.is_file()
    )
    return {
        "arm": "train",
        "adapter": adapter_name,
        "branch": branch,
        "targets": len(targets),
        "parameter_groups": parameter_groups,
        "trainable_elements": trainable_elements,
        "optimizer": "adafactor",
        "learning_rate": LEARNING_RATE,
        "stochastic_rounding": bool(trainer.optimizer_stochastic_rounding),
        "use_fused_backward": bool(getattr(trainer, "use_fused_backward", False)),
        "losses": losses,
        "steps": steps,
        "update_census": {
            "expected": census.expected_count if census else None,
            "steps_checked": census.steps_checked if census else None,
            "exempt": sorted(census.exempt) if census else None,
        },
        "moved_census": {
            "moved": len(moved),
            "unmoved": len(unmoved),
            "of": len(before),
            "unmoved_paths": unmoved[:10],
        },
        "sampled_element_moved_fraction": sampled_moved_fraction,
        "save_format_requested": args.save_format,
        "checkpoint_entry": str(entry),
        "checkpoint_bytes": written_bytes,
        "checkpoint_gib": written_bytes / 1024 ** 3,
        "post_train_digests": after,
        "vram": {
            "model_resident_gib": model_resident["allocated"] / 1024 ** 3,
            "peak_allocated_gib": peak_allocated / 1024 ** 3,
            "peak_reserved_gib": peak_reserved / 1024 ** 3,
            "gate_budget_gib": vram_gate.get("budget_bytes", 0) / 1024 ** 3,
        },
        "host_rss": {
            "before_load_gib": host_before_load / 1024 ** 3,
            "after_load_gib": host_after_load / 1024 ** 3,
            "peak_gib": _host_peak_bytes() / 1024 ** 3,
        },
        "wall_time_s": {
            "model_load": model_load_wall_time_s,
            "train_and_save": train_wall_time_s,
        },
    }


# ---------------------------------------------------------------------------
# Arm 2 -- the production reader, in a process that never held a trainer
# ---------------------------------------------------------------------------


def _run_reload_arm(args: argparse.Namespace) -> dict[str, Any]:
    import torch.nn as nn

    from core.models.ideogram4.vendor.int8_linear import Int8Linear
    from core.models.sensenova.loader import load_sensenova_from_path
    from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets

    expected = json.loads(Path(args.expect).read_text(encoding="utf-8"))
    entry = expected["checkpoint_entry"]
    started = time.perf_counter()
    components = load_sensenova_from_path(entry, torch_dtype=torch.bfloat16)
    load_wall_time_s = time.perf_counter() - started
    transformer = components["transformer"]

    gen = _gen_targets(transformer)
    und = [
        (path, module)
        for path, _p, _a, module in iter_sensenova_lora_targets(transformer, branch="und")
    ]
    gen_float = [p for p, m in gen if isinstance(getattr(m, "weight", None), nn.Parameter)
                 and m.weight.dtype.is_floating_point]
    und_int8 = [p for p, m in und if type(m) is Int8Linear]

    digests = {path: _linear_digest(m.weight) for path, m in gen}
    saved = expected["post_train_digests"]
    mismatched = sorted(p for p in saved if digests.get(p) != saved[p])

    return {
        "arm": "reload",
        "checkpoint_entry": entry,
        "load_wall_time_s": load_wall_time_s,
        "gen_linears": len(gen),
        "gen_float_materialized": len(gen_float),
        "und_linears": len(und),
        "und_still_int8": len(und_int8),
        "digest_matches": len(saved) - len(mismatched),
        "digest_mismatched": mismatched[:10],
        "digest_of": len(saved),
        "host_rss_peak_gib": _host_peak_bytes() / 1024 ** 3,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("train", "reload"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect", default=None,
                        help="the train arm's JSON, for --arm reload")
    parser.add_argument("--save-format", default="mixed")
    parser.add_argument("--prompt", default="a red square on a white background")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    _require_repo_venv()
    args = _parse_args()
    result = (_run_train_arm(args) if args.arm == "train"
              else _run_reload_arm(args))
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k != "post_train_digests"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
