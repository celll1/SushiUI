"""Phase U-1 exit smoke: the understanding-branch LoRA against the real checkpoint.

Every arm runs in its own short-lived process (``--arm``), one at a time, under
the shared VRAM gate imported from the Phase 1 probe.

* ``und_trainer``     -- 3-step trainer run with ``train_text_encoder=True``:
                         finite loss, 1764 saved tensors,
                         ``lora_targets=generation+understanding``, the und
                         gradient census (289 of 294 by name), the positive
                         differentiable-prefix assertion on the live path, and
                         a single break-direction check that the prefix
                         autocast is load-bearing.
* ``und_runtime``     -- fresh process, fresh model, the saved gen+und file:
                         588 applied / 588 restored, strength 0 bit-identical
                         to base, strength 1 different.
* ``mnt``             -- ``multi_noise_timesteps=2``: no freed-graph error, one
                         prefix rebuild per MNT iteration, und gradient nonzero
                         in every optimizer step.
* ``regression``      -- the UNCHANGED Phase 1 ``--smoke-arm trainer`` body run
                         against a chosen ``--library-root``; the driver runs it
                         once against this tree and once against a ``git
                         archive`` of the pre-U-1 commit and compares losses and
                         gradient SHA-256 bit-exactly.
* ``distill_runtime`` -- the shipped generation-only distillation LoRA still
                         applies 294 of 294 with strength 0 parity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "backend"


def _library_backend_root() -> Path:
    """The backend package root this process imports ``core.*`` from.

    ``--library-root`` has to be honoured BEFORE the first ``core`` import, so
    it is read straight off ``sys.argv`` rather than through argparse. This is
    what lets the regression arm run today's probe body against a checked-out
    older library tree without copying either one.
    """
    for index, value in enumerate(sys.argv):
        if value == "--library-root" and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1]).resolve() / "backend"
        if value.startswith("--library-root="):
            return Path(value.split("=", 1)[1]).resolve() / "backend"
    return BACKEND_ROOT


LIBRARY_BACKEND_ROOT = _library_backend_root()
if str(LIBRARY_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBRARY_BACKEND_ROOT))

from core.training.probes.sensenova_real_checkpoint import (  # noqa: E402
    EXIT_SMOKE_HEIGHT,
    EXIT_SMOKE_STEPS,
    EXIT_SMOKE_WIDTH,
    EXPECTED_LAYERS,
    EXPECTED_TARGETS,
    VRAM_GATE_FRACTION,
    _ExitSmokeDataset,
    _apply_vram_gate,
    _cuda_memory,
    _gradient_stats,
    _hash_named_tensors,
    _lora_layer_hash,
    _run_trainer_exit_smoke_arm,
    _write_deterministic_smoke_image,
    trainer_exit_smoke_config,
)


def _repo_venv_python() -> Path:
    """Always THIS repository's interpreter.

    Deliberately not the Phase 1 probe's copy: under ``--library-root`` that one
    derives its repository root from the exported older tree, which has no venv.
    """
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return (REPO_ROOT / "venv" / relative).resolve()


def _require_repo_venv() -> None:
    expected = os.path.normcase(str(_repo_venv_python()))
    actual = os.path.normcase(str(Path(sys.executable).resolve()))
    if actual != expected:
        raise RuntimeError(
            f"Run this probe with the repository virtualenv: {_repo_venv_python()}"
        )

EXPECTED_BOTH_TARGETS = EXPECTED_TARGETS * 2
EXPECTED_BOTH_TENSORS = EXPECTED_BOTH_TARGETS * 3
UND_RUN_NAME = "sensenova_u1_und_smoke"
MNT_RUN_NAME = "sensenova_u1_mnt_smoke"
MNT_VALUE = 2
MNT_TOTAL_STEPS = 4
PRE_U1_COMMIT = "3d837202"
DISTILL_LORA_PATH = REPO_ROOT / "lora" / "SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors"

# Same self-imposed abort line the U-0 probe used: 60% of the hard gate. An arm
# that crosses it is reported as a number, never silently continued into the
# shared GPU's headroom.
CEILING_FRACTION_OF_GATE = 0.60


def _budget(vram_gate: dict[str, Any]) -> dict[str, Any]:
    budget = int(vram_gate["budget_bytes"])
    return {
        **vram_gate,
        "ceiling_bytes": int(budget * CEILING_FRACTION_OF_GATE),
        "ceiling_fraction_of_gate": CEILING_FRACTION_OF_GATE,
    }


def _peak_report(vram_gate: dict[str, Any]) -> dict[str, Any]:
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    budget = int(vram_gate["budget_bytes"])
    return {
        "allocated": allocated,
        "reserved": reserved,
        "allocated_gib": allocated / 1024 ** 3,
        "reserved_gib": reserved / 1024 ** 3,
        "fraction_of_gate": allocated / budget,
        "over_soft_ceiling": allocated > int(vram_gate["ceiling_bytes"]),
    }


def _library_provenance() -> dict[str, str]:
    """Which tree the ``core`` package actually came from, recorded per arm."""
    from core.models.sensenova import sensenova_lora
    from core.training.ops import sensenova_ops

    return {
        "library_backend_root": str(LIBRARY_BACKEND_ROOT),
        "sensenova_ops": str(Path(sensenova_ops.__file__).resolve()),
        "sensenova_lora": str(Path(sensenova_lora.__file__).resolve()),
    }


def _seed_everything(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# Instrumentation (probe-side only; production behaviour is never altered)
# ---------------------------------------------------------------------------


class _PrefixAssertionCapture:
    """Record every live call to the positive differentiable-prefix assertion.

    The point of §13.3's assertion is that a prefix accidentally built under
    ``no_grad`` cannot pass unnoticed. Wrapping it (and then calling the
    original) is the only way to show it ran on the real training path with a
    full 42-of-42 ``grad_fn`` census rather than merely existing in the file.
    """

    def __init__(self):
        self.records: list[dict[str, Any]] = []
        self.build_calls = 0
        self._restore: list[Any] = []

    def __enter__(self) -> "_PrefixAssertionCapture":
        from core.training.ops import sensenova_ops

        original_assert = sensenova_ops._assert_prefix_cache_differentiable
        original_build = sensenova_ops._build_trainable_prefix
        capture = self

        def assert_differentiable(prefix_cache):
            layers = list(prefix_cache.layers)
            capture.records.append({
                "layers": len(layers),
                "keys_with_grad_fn": sum(
                    int(layer.keys.grad_fn is not None) for layer in layers
                ),
                "values_with_grad_fn": sum(
                    int(layer.values.grad_fn is not None) for layer in layers
                ),
            })
            return original_assert(prefix_cache)

        def build_trainable_prefix(trainer, transformer, inputs):
            capture.build_calls += 1
            return original_build(trainer, transformer, inputs)

        sensenova_ops._assert_prefix_cache_differentiable = assert_differentiable
        sensenova_ops._build_trainable_prefix = build_trainable_prefix
        self._restore = [
            lambda: setattr(
                sensenova_ops, "_assert_prefix_cache_differentiable", original_assert
            ),
            lambda: setattr(sensenova_ops, "_build_trainable_prefix", original_build),
        ]
        return self

    def __exit__(self, *exc_info) -> bool:
        for restore in self._restore:
            restore()
        return False

    def summary(self) -> dict[str, Any]:
        return {
            "assertion_calls": len(self.records),
            "prefix_builds": self.build_calls,
            "all_layers_differentiable": all(
                record["layers"] == EXPECTED_LAYERS
                and record["keys_with_grad_fn"] == EXPECTED_LAYERS
                and record["values_with_grad_fn"] == EXPECTED_LAYERS
                for record in self.records
            ),
            "distinct_censuses": sorted(
                {
                    (
                        record["layers"],
                        record["keys_with_grad_fn"],
                        record["values_with_grad_fn"],
                    )
                    for record in self.records
                }
            ),
        }


def _branch_layers(trainer) -> tuple[dict[str, Any], dict[str, Any]]:
    from core.training.adapters.base_adapter import (
        LORA_COMPONENT_TEXT_ENCODER_1,
        LORA_COMPONENT_UNET,
    )

    components = trainer.adapter.lora_components
    generation = {
        name: layer
        for name, layer in trainer.lora_layers.items()
        if components.get(name, LORA_COMPONENT_UNET) == LORA_COMPONENT_UNET
    }
    understanding = {
        name: layer
        for name, layer in trainer.lora_layers.items()
        if components.get(name) == LORA_COMPONENT_TEXT_ENCODER_1
    }
    return generation, understanding


def _dead_up_grad_targets(wrappers: dict[str, Any]) -> list[str]:
    """Targets whose ``lora_up`` gradient is absent or exactly zero.

    ``lora_up`` is the side that carries signal on the very first step (the
    zero-init lives on ``lora_up``'s weight, so ``lora_down``'s gradient is
    legitimately zero until step 2) -- the same choice U-0 made.
    """
    dead = []
    for name in sorted(wrappers):
        gradient = wrappers[name].lora_up.weight.grad
        if gradient is None or not bool(torch.count_nonzero(gradient)):
            dead.append(name)
    return dead


class _BranchGradCapture:
    """Per-optimizer-step gradient census, split by MoT branch."""

    def __init__(self, trainer):
        self._trainer = trainer
        self.records: list[dict[str, Any]] = []
        self._original = None

    def __enter__(self) -> "_BranchGradCapture":
        original = torch.optim.AdamW.step
        capture = self

        def step(optimizer_self, *args, **kwargs):
            generation, understanding = _branch_layers(capture._trainer)
            capture.records.append({
                "gen_up": _gradient_stats(generation, "lora_up"),
                "gen_down": _gradient_stats(generation, "lora_down"),
                "und_up": _gradient_stats(understanding, "lora_up"),
                "und_down": _gradient_stats(understanding, "lora_down"),
                "und_dead_up_targets": _dead_up_grad_targets(understanding),
                "gen_dead_up_targets": _dead_up_grad_targets(generation),
            })
            return original(optimizer_self, *args, **kwargs)

        torch.optim.AdamW.step = step
        self._original = original
        return self

    def __exit__(self, *exc_info) -> bool:
        torch.optim.AdamW.step = self._original
        return False


# ---------------------------------------------------------------------------
# Shared trainer driving
# ---------------------------------------------------------------------------


def _und_config(*, total_steps: int, mnt: int, reference: bool = False) -> dict[str, Any]:
    config = trainer_exit_smoke_config()
    config["train_config"]["multi_noise_timesteps"] = mnt
    config["train"]["multi_noise_timesteps"] = mnt
    config["train"]["total_steps"] = total_steps
    config["train"]["save_every_n_steps"] = total_steps
    config["train_config"]["use_reference_images"] = bool(reference)
    config["train"]["use_reference_images"] = bool(reference)
    return config


def _train_understanding(
    args: argparse.Namespace,
    *,
    run_name: str,
    output_subdir: str,
    total_steps: int,
    mnt: int,
) -> dict[str, Any]:
    """One real ``LoRATrainer`` run with the understanding branch armed."""
    from core.training.lora_trainer import LoRATrainer

    reference = bool(getattr(args, "reference", False))
    config = _und_config(total_steps=total_steps, mnt=mnt, reference=reference)
    workdir = Path(args.smoke_workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    image_path = workdir / "training_image.png"
    output_dir = workdir / output_subdir
    checkpoint_path = output_dir / f"{run_name}_step_{total_steps:06d}.safetensors"
    _write_deterministic_smoke_image(image_path)
    reference_paths: list[str] = []
    if reference:
        # Phase U-3: the same 289 census, with the understanding prefix carrying
        # spliced reference tokens. Not the target's geometry -- references do
        # not participate in bucketing (7.5 differential 4).
        reference_image = workdir / "reference_512x512.png"
        _write_deterministic_smoke_image(reference_image, 512, 512)
        reference_paths = [str(reference_image)]

    _seed_everything(args.seed)
    torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    trainer = LoRATrainer(
        model_path=args.model_path,
        output_dir=str(output_dir),
        run_name=run_name,
        run_id=None,
        learning_rate=args.unet_lr,
        text_encoder_1_lr=args.understanding_lr,
        train_text_encoder=True,
        device="cuda",
        train_config=dict(config["train_config"]),
        **dict(config["constructor"]),
    )
    model_load_wall_time_s = time.perf_counter() - load_started
    model_resident = _cuda_memory()

    generation, understanding = _branch_layers(trainer)
    if len(trainer.lora_layers) != EXPECTED_BOTH_TARGETS:
        raise AssertionError(
            f"expected {EXPECTED_BOTH_TARGETS} LoRA layers, got {len(trainer.lora_layers)}"
        )
    if len(generation) != EXPECTED_TARGETS or len(understanding) != EXPECTED_TARGETS:
        raise AssertionError(
            f"branch split is {len(generation)} gen / {len(understanding)} und"
        )

    parameter_groups = [
        {"lr": group["lr"], "tensors": len(group["params"])}
        for group in trainer.setup_trainable_parameters()
    ]

    losses: list[float] = []
    training_steps: list[int] = []

    def progress_callback(phase, step, total, epoch=0, loss=None):
        del total, epoch
        if phase != "training":
            return
        if loss is None or not math.isfinite(float(loss)):
            raise AssertionError(f"non-finite SenseNova und-smoke loss: {loss!r}")
        training_steps.append(int(step))
        losses.append(float(loss))

    train = dict(config["train"])
    train.update({
        "num_epochs": 1,
        "sample_prompts": [],
        "sample_guidance_scale": 1.0,
        "sample_steps": 1,
        "sample_width": EXIT_SMOKE_WIDTH,
        "sample_height": EXIT_SMOKE_HEIGHT,
        "sample_seed": args.seed,
        "max_grad_norm": 1.0,
        "progress_callback": progress_callback,
        "run_id": None,
        "max_step_saves_to_keep": 1,
        "force_recache": False,
    })
    dataset = _ExitSmokeDataset(
        image_path, args.prompt, reference_images=reference_paths
    )

    train_started = time.perf_counter()
    with _PrefixAssertionCapture() as prefix_capture:
        with _BranchGradCapture(trainer) as grad_capture:
            trainer.train(datasets=[dataset], **train)
    train_wall_time_s = time.perf_counter() - train_started

    if training_steps != list(range(1, total_steps + 1)):
        raise AssertionError(f"expected steps {list(range(1, total_steps + 1))}, got {training_steps}")
    if not checkpoint_path.is_file():
        raise AssertionError(f"trainer did not save {checkpoint_path.name}")

    lora_hash, lora_finite = _lora_layer_hash(trainer.lora_layers)
    if not lora_finite:
        raise AssertionError("trainer LoRA parameters contain a non-finite value")

    return {
        "trainer": trainer,
        "checkpoint_path": checkpoint_path,
        "config": config,
        "losses": losses,
        "training_steps": training_steps,
        "grad_records": grad_capture.records,
        "prefix_capture": prefix_capture,
        "parameter_groups": parameter_groups,
        "lora_parameter_sha256": lora_hash,
        "reference_conditioned": reference,
        "reference_image_paths": reference_paths,
        "model_resident": model_resident,
        "model_load_wall_time_s": model_load_wall_time_s,
        "train_wall_time_s": train_wall_time_s,
    }


def _inspect_saved_und_lora(path: Path, *, expected_step: int) -> dict[str, Any]:
    from safetensors import safe_open

    from core.models.sensenova.sensenova_lora import und_gradient_unreachable_paths

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        metadata = dict(handle.metadata() or {})
        tensors = [(key, handle.get_tensor(key)) for key in keys]
    if len(keys) != EXPECTED_BOTH_TENSORS:
        raise AssertionError(
            f"expected {EXPECTED_BOTH_TENSORS} LoRA tensors, got {len(keys)}"
        )
    targets = {
        key.rsplit(".lora_down.weight", 1)[0]
        for key in keys
        if key.endswith(".lora_down.weight")
    }
    if len(targets) != EXPECTED_BOTH_TARGETS:
        raise AssertionError(f"expected {EXPECTED_BOTH_TARGETS} targets, got {len(targets)}")
    if metadata.get("lora_targets") != "generation+understanding":
        raise AssertionError(
            f"metadata lora_targets={metadata.get('lora_targets')!r}"
        )
    if metadata.get("step") != str(expected_step):
        raise AssertionError(f"metadata step={metadata.get('step')!r}")
    all_hash, all_finite = _hash_named_tensors(tensors)
    parameter_hash, parameter_finite = _hash_named_tensors(
        (key, tensor) for key, tensor in tensors if key.endswith(".weight")
    )
    if not (all_finite and parameter_finite):
        raise AssertionError("saved LoRA contains a non-finite tensor")

    # The five structurally unreachable und adapters must still be PRESENT in
    # the file (enumeration keeps all 294) and still at their zero init.
    dead = sorted(und_gradient_unreachable_paths(EXPECTED_LAYERS))
    dead_state = {}
    lookup = dict(tensors)
    for name in dead:
        up = lookup.get(f"{name}.lora_up.weight")
        if up is None:
            raise AssertionError(f"the unreachable und target {name} is missing from the file")
        dead_state[name] = {
            "present": True,
            "lora_up_nonzero": int(torch.count_nonzero(up)),
        }
    return {
        "tensor_count": len(keys),
        "target_count": len(targets),
        "metadata": metadata,
        "tensor_sha256": all_hash,
        "parameter_sha256": parameter_hash,
        "finite": True,
        "unreachable_und_targets": dead_state,
    }


# ---------------------------------------------------------------------------
# Arm 1 -- understanding trainer smoke
# ---------------------------------------------------------------------------


def _autocast_break_check(trainer, prompt: str) -> dict[str, Any]:
    """One break-direction check that the prefix autocast is load-bearing.

    U-0 found by running it that ``LoRALinearLayer`` keeps fp32 adapters and
    needs an ambient autocast the prefix pass did not previously provide. This
    reproduces that failure exactly once, by calling the production
    ``encode_prompt`` with only the autocast wrapper removed, and restores the
    real function immediately afterwards.
    """
    from core.training.ops import sensenova_ops

    original = sensenova_ops._build_trainable_prefix

    def build_without_autocast(trainer_self, transformer, inputs):
        # The production body with ONLY the autocast wrapper removed, including
        # its ids-or-embeds entry (U-3): unpacking the inputs as a bare triple
        # here made this report `raised: True` for a ValueError from the probe
        # rather than the dtype mismatch it exists to reproduce.
        tokens, indexes, attention_mask = inputs[0], inputs[1], inputs[2]
        embeds = bool(inputs[3]) if len(inputs) > 3 else False
        return sensenova_ops.forward_und_prefix_layers(
            transformer.language_model.model,
            None if embeds else tokens,
            indexes,
            attention_mask,
            inputs_embeds=tokens if embeds else None,
            checkpoint_layers=bool(getattr(trainer_self, "gradient_checkpointing", True)),
        )

    sensenova_ops._build_trainable_prefix = build_without_autocast
    try:
        sensenova_ops.encode_prompt(trainer, prompt, requires_grad=True)
    except Exception as exc:  # noqa: BLE001 -- the failure IS the measurement
        outcome = {
            "raised": True,
            "exception_type": type(exc).__name__,
            "message": str(exc)[:400],
        }
    else:
        outcome = {"raised": False}
    finally:
        sensenova_ops._build_trainable_prefix = original
        torch.cuda.empty_cache()
    # "Something raised" is NOT the measurement, and treating it as one is a
    # defect this check has already had: after the U-3 inputs change the
    # stand-in above unpacked a 4-field inputs tuple as a triple, and the arm
    # reported a probe-side ValueError as a successful reproduction. The claim
    # is specifically the fp32-adapter dtype mismatch, so it is matched.
    expected_type = "RuntimeError"
    expected_fragment = "same dtype"
    outcome["expected_exception_type"] = expected_type
    outcome["expected_message_fragment"] = expected_fragment
    outcome["reproduced_the_dtype_mismatch"] = bool(
        outcome["raised"]
        and outcome.get("exception_type") == expected_type
        and expected_fragment in outcome.get("message", "")
    )
    if not outcome["reproduced_the_dtype_mismatch"]:
        raise AssertionError(
            "the autocast break check did not reproduce the U-0 failure it "
            f"exists to reproduce (expected {expected_type} containing "
            f"{expected_fragment!r}, got {outcome})"
        )
    return outcome


def _run_und_trainer_arm(args: argparse.Namespace) -> dict[str, Any]:
    from core.models.sensenova.sensenova_lora import und_gradient_unreachable_paths

    vram_gate = _budget(_apply_vram_gate())
    run = _train_understanding(
        args,
        run_name=UND_RUN_NAME,
        output_subdir="und_output",
        total_steps=EXIT_SMOKE_STEPS,
        mnt=1,
    )
    trainer = run["trainer"]
    saved = _inspect_saved_und_lora(run["checkpoint_path"], expected_step=EXIT_SMOKE_STEPS)
    if saved["parameter_sha256"] != run["lora_parameter_sha256"]:
        raise AssertionError("saved LoRA tensor hash differs from live trainer parameters")

    predicted_dead = sorted(und_gradient_unreachable_paths(EXPECTED_LAYERS))
    census = []
    for index, record in enumerate(run["grad_records"], start=1):
        census.append({
            "step": index,
            "gen_up_nonzero": record["gen_up"]["nonzero"],
            "gen_up_reached": record["gen_up"]["reached"],
            "gen_up_finite": record["gen_up"]["finite"],
            "gen_up_l2": record["gen_up"]["l2"],
            "gen_down_l2": record["gen_down"]["l2"],
            "und_up_nonzero": record["und_up"]["nonzero"],
            "und_up_reached": record["und_up"]["reached"],
            "und_up_finite": record["und_up"]["finite"],
            "und_up_l2": record["und_up"]["l2"],
            "und_down_l2": record["und_down"]["l2"],
            "und_dead_up_targets": record["und_dead_up_targets"],
            "und_dead_matches_prediction": record["und_dead_up_targets"] == predicted_dead,
            "gen_dead_up_targets": record["gen_dead_up_targets"],
        })
    # The five structurally unreachable adapters receive no ``.grad`` object at
    # all, so the reachable census is 289, not 294 with five zeros.
    expected_reached = EXPECTED_TARGETS - len(predicted_dead)
    for entry in census:
        if entry["gen_up_nonzero"] != EXPECTED_TARGETS:
            raise AssertionError(f"generation gradients have dead targets: {entry}")
        if (
            entry["und_up_reached"] != expected_reached
            or entry["und_up_finite"] != expected_reached
            or entry["und_up_nonzero"] != expected_reached
        ):
            raise AssertionError(
                f"und gradient census is not {expected_reached}-of-{EXPECTED_TARGETS}: {entry}"
            )
        if not entry["und_dead_matches_prediction"]:
            raise AssertionError(
                "und dead-gradient census does not match "
                f"und_gradient_unreachable_paths(): {entry['und_dead_up_targets']}"
            )

    peak = _peak_report(vram_gate)
    prefix_summary = run["prefix_capture"].summary()
    autocast_break = _autocast_break_check(trainer, args.prompt)

    result = {
        "arm": "und_trainer",
        "library": _library_provenance(),
        "seed": args.seed,
        "geometry": {"width": EXIT_SMOKE_WIDTH, "height": EXIT_SMOKE_HEIGHT, "batch": 1},
        "multi_noise_timesteps": 1,
        "reference_conditioned": run["reference_conditioned"],
        "reference_image_paths": run["reference_image_paths"],
        "lora_layers": len(trainer.lora_layers),
        "parameter_groups": run["parameter_groups"],
        "learning_rates": {
            "unet_lr": trainer.unet_lr,
            "text_encoder_1_lr": trainer.text_encoder_1_lr,
        },
        "training_steps": run["training_steps"],
        "losses": run["losses"],
        "losses_finite": all(math.isfinite(value) for value in run["losses"]),
        "checkpoint": {"name": run["checkpoint_path"].name, **saved},
        "lora_parameter_sha256": run["lora_parameter_sha256"],
        "predicted_unreachable_und_targets": predicted_dead,
        "und_expected_reachable_targets": expected_reached,
        "gradient_census": census,
        "prefix_assertion": prefix_summary,
        "autocast_break_check": autocast_break,
        "model_resident": run["model_resident"],
        "peak_memory": peak,
        "vram_gate": vram_gate,
        "model_load_wall_time_s": run["model_load_wall_time_s"],
        "wall_time_s": run["train_wall_time_s"],
    }
    try:
        trainer.writer.close()
    finally:
        trainer._db_executor.shutdown(wait=True)
    return result


# ---------------------------------------------------------------------------
# Arm 2 -- MNT > 1
# ---------------------------------------------------------------------------


def _run_mnt_arm(args: argparse.Namespace) -> dict[str, Any]:
    from core.models.sensenova.sensenova_lora import und_gradient_unreachable_paths

    vram_gate = _budget(_apply_vram_gate())
    run = _train_understanding(
        args,
        run_name=MNT_RUN_NAME,
        output_subdir="mnt_output",
        total_steps=MNT_TOTAL_STEPS,
        mnt=MNT_VALUE,
    )
    trainer = run["trainer"]
    predicted_dead = sorted(und_gradient_unreachable_paths(EXPECTED_LAYERS))
    census = []
    for index, record in enumerate(run["grad_records"], start=1):
        census.append({
            "optimizer_step": index,
            "und_up_nonzero": record["und_up"]["nonzero"],
            "und_up_reached": record["und_up"]["reached"],
            "und_up_l2": record["und_up"]["l2"],
            "gen_up_nonzero": record["gen_up"]["nonzero"],
            "gen_up_l2": record["gen_up"]["l2"],
            "und_dead_matches_prediction": record["und_dead_up_targets"] == predicted_dead,
        })
    expected_reached = EXPECTED_TARGETS - len(predicted_dead)
    for entry in census:
        if entry["und_up_nonzero"] != expected_reached:
            raise AssertionError(
                f"MNT iteration und gradient is not {expected_reached}-of-"
                f"{EXPECTED_TARGETS} nonzero: {entry}"
            )
        if not entry["und_dead_matches_prediction"]:
            raise AssertionError(f"MNT und dead census drifted: {entry}")

    peak = _peak_report(vram_gate)
    result = {
        "arm": "mnt",
        "library": _library_provenance(),
        "multi_noise_timesteps": MNT_VALUE,
        "total_steps": MNT_TOTAL_STEPS,
        "und_expected_reachable_targets": expected_reached,
        "optimizer_steps_observed": len(run["grad_records"]),
        "training_steps": run["training_steps"],
        "losses": run["losses"],
        "losses_finite": all(math.isfinite(value) for value in run["losses"]),
        "prefix_assertion": run["prefix_capture"].summary(),
        "gradient_census": census,
        "freed_graph_error": False,
        "checkpoint_saved": run["checkpoint_path"].name,
        "model_resident": run["model_resident"],
        "peak_memory": peak,
        "vram_gate": vram_gate,
        "model_load_wall_time_s": run["model_load_wall_time_s"],
        "wall_time_s": run["train_wall_time_s"],
    }
    try:
        trainer.writer.close()
    finally:
        trainer._db_executor.shutdown(wait=True)
    return result


# ---------------------------------------------------------------------------
# Arm 3 -- fresh runtime application
# ---------------------------------------------------------------------------


def _generation_args(args: argparse.Namespace, lora_path: Optional[str], strength: float):
    return SimpleNamespace(
        attn_backend="native",
        cfg_scale=args.smoke_cfg_scale,
        timestep_shift=args.smoke_timestep_shift,
        cfg_norm=args.smoke_cfg_norm,
        seed=args.seed,
        prompt=args.prompt,
        negative_prompt=None,
        output=None,
        lora=lora_path,
        lora_strength=strength,
        return_tensor=True,
    )


def _take_tensor(result: dict[str, Any]) -> torch.Tensor:
    tensor = result.pop("denoise_tensor", None)
    if not isinstance(tensor, torch.Tensor):
        raise AssertionError("runtime generation returned no denoise_tensor")
    return tensor


def _tensor_digest(tensor: torch.Tensor) -> str:
    digest, finite = _hash_named_tensors((("denoise", tensor),))
    if not finite:
        raise AssertionError("runtime denoise tensor is non-finite")
    return digest


def _runtime_lora_arm(
    args: argparse.Namespace,
    lora_path: Path,
    *,
    expected_modules: int,
    label: str,
) -> dict[str, Any]:
    from core.models.sensenova import sensenova_lora
    from core.models.sensenova import smoke as runtime_smoke

    vram_gate = _budget(_apply_vram_gate())
    _seed_everything(args.seed)

    raw, fmt, metadata = sensenova_lora.load_lora_safetensors(str(lora_path))
    grouped = sensenova_lora.normalise_lora_state_dict(raw)
    del raw

    model, _config, tokenizer = runtime_smoke._load_converted(
        args.model_path, torch.device("cuda"), torch.bfloat16
    )
    model.eval()
    model.requires_grad_(False)
    before_ids = {
        path: id(module)
        for path, _parent, _attr, module in sensenova_lora.iter_sensenova_lora_targets(
            model, branch="both"
        )
    }

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    base = runtime_smoke.run_generation(
        model, tokenizer, _generation_args(args, None, 0.0),
        EXIT_SMOKE_WIDTH, EXIT_SMOKE_HEIGHT, 1,
    )
    base_tensor = _take_tensor(base)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    zero = runtime_smoke.run_generation(
        model, tokenizer, _generation_args(args, str(lora_path), 0.0),
        EXIT_SMOKE_WIDTH, EXIT_SMOKE_HEIGHT, 1,
    )
    zero_tensor = _take_tensor(zero)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    one = runtime_smoke.run_generation(
        model, tokenizer, _generation_args(args, str(lora_path), 1.0),
        EXIT_SMOKE_WIDTH, EXIT_SMOKE_HEIGHT, 1,
    )
    one_tensor = _take_tensor(one)

    after_ids = {
        path: id(module)
        for path, _parent, _attr, module in sensenova_lora.iter_sensenova_lora_targets(
            model, branch="both"
        )
    }
    shortfall = sensenova_lora.check_lora_application(
        grouped, int(zero["lora_applied"]), metadata
    )
    strength0_equal = torch.equal(base_tensor, zero_tensor)
    strength1_equal = torch.equal(base_tensor, one_tensor)
    strength1_max_abs_delta = float(
        (one_tensor.float() - base_tensor.float()).abs().max()
    )

    if int(zero["lora_applied"]) != expected_modules:
        raise AssertionError(
            f"{label}: applied {zero['lora_applied']} module(s), expected {expected_modules}"
        )
    if int(zero["lora_restored"]) != expected_modules:
        raise AssertionError(
            f"{label}: restored {zero['lora_restored']} module(s), expected {expected_modules}"
        )
    if not strength0_equal:
        raise AssertionError(f"{label}: strength 0 changed the denoise tensor")
    if strength1_equal:
        raise AssertionError(f"{label}: strength 1 left the denoise tensor unchanged")
    if before_ids != after_ids:
        raise AssertionError(f"{label}: restore did not recover every module identity")
    if shortfall is not None:
        raise AssertionError(f"{label}: {shortfall}")

    peak = _peak_report(vram_gate)
    return {
        "arm": label,
        "library": _library_provenance(),
        "lora_path": str(lora_path),
        "lora_format": fmt,
        "lora_metadata": metadata,
        "modules_in_file": len(grouped),
        "lora_applied": int(zero["lora_applied"]),
        "lora_restored": int(zero["lora_restored"]),
        "strength1_applied": int(one["lora_applied"]),
        "strength1_restored": int(one["lora_restored"]),
        "expected_modules": expected_modules,
        "check_lora_application": shortfall,
        "base_denoise_sha256": _tensor_digest(base_tensor),
        "strength0_denoise_sha256": _tensor_digest(zero_tensor),
        "strength1_denoise_sha256": _tensor_digest(one_tensor),
        "strength0_equal_to_base": strength0_equal,
        "strength1_equal_to_base": strength1_equal,
        "strength1_max_abs_delta": strength1_max_abs_delta,
        "module_identity_restored": before_ids == after_ids,
        "settings": {
            "prompt": args.prompt,
            "seed": args.seed,
            "steps": 1,
            "width": EXIT_SMOKE_WIDTH,
            "height": EXIT_SMOKE_HEIGHT,
            "cfg_scale": args.smoke_cfg_scale,
            "timestep_shift": args.smoke_timestep_shift,
            "cfg_norm": args.smoke_cfg_norm,
            "attention_backend": "native",
            "deterministic_algorithms": True,
        },
        "peak_memory": peak,
        "vram_gate": vram_gate,
    }


def _run_und_runtime_arm(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke_lora_path is None:
        raise RuntimeError("the und_runtime arm requires --smoke-lora-path")
    return _runtime_lora_arm(
        args,
        Path(args.smoke_lora_path),
        expected_modules=EXPECTED_BOTH_TARGETS,
        label="und_runtime",
    )


def _run_distill_runtime_arm(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.distill_lora_path)
    if not path.is_file():
        raise RuntimeError(f"distillation LoRA not found: {path}")
    return _runtime_lora_arm(
        args, path, expected_modules=EXPECTED_TARGETS, label="distill_runtime"
    )


# ---------------------------------------------------------------------------
# Arm 4 -- train_text_encoder=false regression against the pre-U-1 library
# ---------------------------------------------------------------------------


def _run_regression_arm(args: argparse.Namespace) -> dict[str, Any]:
    """Run the UNCHANGED Phase 1 trainer arm against whichever library is on the path."""
    result = _run_trainer_exit_smoke_arm(
        SimpleNamespace(
            model_path=args.model_path,
            seed=args.seed,
            prompt=args.prompt,
            smoke_phase_eviction="off",
            smoke_workdir=args.smoke_workdir,
        )
    )
    result["library"] = _library_provenance()
    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_ARMS = {
    "und_trainer": _run_und_trainer_arm,
    "und_runtime": _run_und_runtime_arm,
    "mnt": _run_mnt_arm,
    "regression": _run_regression_arm,
    "distill_runtime": _run_distill_runtime_arm,
}


def _run_arm_subprocess(
    args: argparse.Namespace,
    arm: str,
    workdir: Path,
    *,
    lora_path: Path | None = None,
    library_root: Path | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    label = tag or arm
    result_path = workdir / f"{label}.json"
    cmd = [
        str(_repo_venv_python()),
        str(Path(__file__).resolve()),
        "--model-path", args.model_path,
        "--prompt", args.prompt,
        "--seed", str(args.seed),
        "--unet-lr", str(args.unet_lr),
        "--understanding-lr", str(args.understanding_lr),
        "--smoke-cfg-scale", str(args.smoke_cfg_scale),
        "--smoke-timestep-shift", str(args.smoke_timestep_shift),
        "--smoke-cfg-norm", args.smoke_cfg_norm,
        "--distill-lora-path", str(args.distill_lora_path),
        "--arm", arm,
        "--smoke-workdir", str(workdir / label),
        "--arm-json", str(result_path),
        *(["--reference"] if getattr(args, "reference", False) else []),
    ]
    if lora_path is not None:
        cmd.extend(("--smoke-lora-path", str(lora_path)))
    if library_root is not None:
        cmd.extend(("--library-root", str(library_root)))
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env, timeout=args.timeout_s, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"U-1 {label} arm exited with code {completed.returncode}")
    if not result_path.is_file():
        raise RuntimeError(f"U-1 {label} arm wrote no JSON result")
    with result_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _export_pre_u1_tree(destination: Path) -> dict[str, Any]:
    """Read-only ``git archive`` of the pre-U-1 backend, never a working-tree change."""
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / "pre_u1_backend.tar"
    with archive.open("wb") as handle:
        subprocess.run(
            ["git", "archive", PRE_U1_COMMIT, "backend"],
            cwd=str(REPO_ROOT), stdout=handle, check=True,
        )
    import tarfile

    with tarfile.open(archive) as handle:
        handle.extractall(destination)
    archive.unlink()
    resolved = subprocess.run(
        ["git", "rev-parse", PRE_U1_COMMIT],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
    ).stdout.strip()

    # Today's measurement body on BOTH sides: only the library may differ, or the
    # comparison would confound a library change with a probe change. (These two
    # happen to be identical at this commit; copying makes that a guarantee
    # rather than a coincidence.)
    import shutil

    measured = Path(__file__).resolve().parent / "sensenova_real_checkpoint.py"
    shutil.copyfile(
        measured,
        destination / "backend" / "core" / "training" / "probes" / measured.name,
    )
    return {
        "commit": resolved,
        "root": str(destination),
        "probe_body_copied_from": str(measured),
    }


def _compare_regression(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    def digests(record: dict[str, Any]) -> list[tuple[str, str]]:
        return [
            (entry["up"]["sha256"], entry["down"]["sha256"])
            for entry in record["grad_digests"]
        ]

    return {
        "losses_new": new["losses"],
        "losses_old": old["losses"],
        "losses_bit_exact": new["losses"] == old["losses"],
        "grad_sha256_bit_exact": digests(new) == digests(old),
        "grad_sha256_new": digests(new),
        "grad_sha256_old": digests(old),
        "lora_parameter_sha256_equal": (
            new["lora_parameter_sha256"] == old["lora_parameter_sha256"]
        ),
        "saved_tensor_sha256_equal": (
            new["checkpoint"]["tensor_sha256"] == old["checkpoint"]["tensor_sha256"]
        ),
        "saved_tensor_count_new": new["checkpoint"]["tensor_count"],
        "saved_tensor_count_old": old["checkpoint"]["tensor_count"],
        "new_library": new["library"],
        "old_library": old["library"],
    }


def _und_checkpoint_path(workdir: Path) -> Path | None:
    candidate = (
        workdir
        / "und_trainer"
        / "und_output"
        / f"{UND_RUN_NAME}_step_{EXIT_SMOKE_STEPS:06d}.safetensors"
    )
    return candidate if candidate.is_file() else None


class _KeptDirectory:
    """``TemporaryDirectory``'s context shape, without the deletion."""

    def __init__(self, path: Path):
        self._path = path

    def __enter__(self) -> str:
        return str(self._path)

    def __exit__(self, *exc_info) -> bool:
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="a red cube on a white table")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--unet-lr", type=float, default=1e-4)
    parser.add_argument("--understanding-lr", type=float, default=5e-5)
    parser.add_argument("--distill-lora-path", default=str(DISTILL_LORA_PATH))
    parser.add_argument("--reference", action="store_true",
                        help="give the training item a reference image, so the "
                             "understanding prefix is reference-conditioned "
                             "(Phase U-3)")
    parser.add_argument("--smoke-cfg-scale", type=float, default=None)
    parser.add_argument("--smoke-timestep-shift", type=float, default=None)
    parser.add_argument("--smoke-cfg-norm", default=None)
    parser.add_argument(
        "--arms",
        default="und_trainer,und_runtime,mnt,regression,distill_runtime",
    )
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--workdir",
        default=None,
        help="Keep arm artefacts here instead of a temporary directory.",
    )
    parser.add_argument("--arm", choices=sorted(_ARMS), default=None)
    parser.add_argument("--arm-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-workdir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-lora-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--library-root", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.arm is not None and (args.arm_json is None or args.smoke_workdir is None):
        parser.error("--arm requires --arm-json and --smoke-workdir")
    if any(
        value is None
        for value in (args.smoke_cfg_scale, args.smoke_timestep_shift, args.smoke_cfg_norm)
    ):
        from api.param_defaults import SENSENOVA_GENERATION_DEFAULTS

        if args.smoke_cfg_scale is None:
            args.smoke_cfg_scale = SENSENOVA_GENERATION_DEFAULTS["cfg_scale"]
        if args.smoke_timestep_shift is None:
            args.smoke_timestep_shift = SENSENOVA_GENERATION_DEFAULTS["timestep_shift"]
        if args.smoke_cfg_norm is None:
            args.smoke_cfg_norm = SENSENOVA_GENERATION_DEFAULTS["cfg_norm"]
    return args


def main() -> int:
    _require_repo_venv()
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the SenseNova U-1 exit smoke requires CUDA")

    if args.arm is not None:
        result = _ARMS[args.arm](args)
        path = Path(args.arm_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, default=str)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0

    requested = [name for name in args.arms.split(",") if name]
    results: dict[str, Any] = {}
    if args.workdir:
        keep = Path(args.workdir)
        keep.mkdir(parents=True, exist_ok=True)
        context: Any = _KeptDirectory(keep)
    else:
        context = tempfile.TemporaryDirectory(prefix="sensenova_u1_")
    with context as raw:
        workdir = Path(raw)
        und_lora_path: Path | None = None
        for arm in requested:
            if arm == "und_runtime":
                if und_lora_path is None:
                    # A previous invocation may have left it in a --workdir.
                    und_lora_path = _und_checkpoint_path(workdir)
                if und_lora_path is None or not und_lora_path.is_file():
                    raise RuntimeError(
                        "und_runtime needs the und_trainer arm's checkpoint; run "
                        "und_trainer first (optionally in an earlier invocation "
                        "sharing --workdir)"
                    )
                results[arm] = _run_arm_subprocess(
                    args, arm, workdir, lora_path=und_lora_path
                )
                continue
            if arm == "regression":
                pre_u1 = _export_pre_u1_tree(workdir / "pre_u1")
                new = _run_arm_subprocess(
                    args, "regression", workdir, tag="regression_new"
                )
                old = _run_arm_subprocess(
                    args,
                    "regression",
                    workdir,
                    tag="regression_pre_u1",
                    library_root=pre_u1["root"] and Path(pre_u1["root"]),
                )
                results[arm] = {
                    "pre_u1": pre_u1,
                    "new_tree": new,
                    "pre_u1_tree": old,
                    "comparison": _compare_regression(new, old),
                }
                continue
            results[arm] = _run_arm_subprocess(args, arm, workdir)
            if arm == "und_trainer":
                und_lora_path = _und_checkpoint_path(workdir)
                if und_lora_path is None:
                    raise RuntimeError("und_trainer arm produced no LoRA checkpoint")

        payload = {
            "probe": "sensenova_phase_u1_und_lora_exit_smoke",
            "checkpoint": Path(args.model_path).name,
            "vram_gate_fraction": VRAM_GATE_FRACTION,
            "arms": results,
        }
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
