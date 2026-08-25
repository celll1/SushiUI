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
    """Peak working set of THIS process, from the OS rather than a sampler.

    NOT a high-water mark of anything this process owns, and reported alongside
    ``_host_peak_commit_bytes`` for that reason. ``peak_wset`` counts RESIDENT
    mmap'd file pages, so a warm page cache over the 17.6 GiB base moves it, and
    Windows trims a working set under memory pressure, so it can fall. Two
    identical both-branch arms measured 61.67 and 51.97 GiB peak (26.07 vs 9.04
    GiB after load) -- the second followed a 32 GiB arm in the same session.
    Treat the LARGER as the bound.
    """
    import psutil

    info = psutil.Process(os.getpid()).memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def _host_peak_commit_bytes() -> int:
    """Peak commit charge (private, backed) of THIS process -- the reproducible one.

    Unlike the working set this counts only what the process has committed, so
    it is neither inflated by resident file mappings nor trimmable by the OS.
    This is the quantity a host-RAM budget should be written against. Falls back
    to the working-set peak where the platform has no such counter.
    """
    import psutil

    info = psutil.Process(os.getpid()).memory_info()
    return int(getattr(info, "peak_pagefile", 0)) or _host_peak_bytes()


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


def _gen_targets(transformer, branch: str = "gen") -> list[tuple[str, Any]]:
    from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets

    return [
        (path, module)
        for path, _parent, _attr, module in iter_sensenova_lora_targets(
            transformer, branch=branch
        )
    ]


def _digest_map(transformer, branch: str = "gen") -> dict[str, list[float]]:
    return {
        path: _linear_digest(m.weight) for path, m in _gen_targets(transformer, branch)
    }


def _halves(branch: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(trained halves, frozen halves) for a branch."""
    trained = ("gen", "und") if branch == "both" else (branch,)
    return trained, tuple(h for h in ("gen", "und") if h not in trained)


def _effective_format(branch: str, save_format: str) -> str:
    """The format the writer actually lands on (loader.py's own rule)."""
    _trained, frozen = _halves(branch)
    return "bf16" if (save_format == "mixed" and not frozen) else save_format


def expected_read_shape(branch: str, effective_format: str) -> dict[str, str]:
    """What the production reader must produce per half, by format (6.4).

    ``"int8"`` means every Linear of that half is an ``Int8Linear``; ``"float"``
    means every one holds a floating ``nn.Parameter``. The trained half is
    floating under every format except ``int8``; the frozen half keeps its int8
    codes only under ``mixed``, since that is the whole of what ``mixed`` means.

    A function rather than four lines inside the reload arm because the arm had
    the ``gen``-branch answer HARDCODED, which is why the two formats that had
    never been read back could not be.
    """
    trained, _frozen = _halves(branch)
    return {
        half: (
            "int8" if effective_format == "int8"
            else "float" if half in trained
            else ("int8" if effective_format == "mixed" else "float")
        )
        for half in ("gen", "und")
    }


def train_arm_failures(
    *,
    moved,
    unmoved,
    of: int,
    predicted_unmoved,
    steps,
) -> list[str]:
    """Every post-run verdict the train arm reaches, as data rather than raises.

    Collected instead of raised so the JSON is written first: these are FACTS
    ABOUT THE RUN, not preconditions, and a 25 GiB run that fails one of them
    should not also lose the numbers that say how it failed.

    The census clause is the U-2-5 criterion itself (13.4). It is NOT "everything
    moved": ``predicted_unmoved`` carries the paths a t2i loss structurally
    cannot reach, and equality against it is what makes the criterion fire on a
    dead hook while staying silent on the five layer-41 projections.
    """
    failures: list[str] = []
    if list(steps) != list(range(1, EXIT_SMOKE_STEPS + 1)):
        failures.append(f"expected {EXIT_SMOKE_STEPS} steps, got {list(steps)}")
    if sorted(unmoved) != sorted(predicted_unmoved):
        failures.append(
            f"update-nonzero census: {len(moved)} of {of} moved; the unmoved set "
            f"is {sorted(unmoved)} but und_gradient_unreachable_paths() predicts "
            f"{sorted(predicted_unmoved)}"
        )
    return failures


def u2_5_unmoved_expectation(paths, num_layers: int) -> list[str]:
    """The paths a t2i loss structurally cannot move, within an enumeration.

    The U-2-5 criterion is not "everything moved": the understanding half's
    layer-41 attention-onward projections receive nothing, because the prefix
    keeps ``past_key_values`` and discards ``last_hidden_state``. Intersected
    with ``paths`` rather than returned whole, so the generation enumeration
    (``*_mot_gen``) correctly expects none of them.
    """
    from core.models.sensenova.sensenova_lora import und_gradient_unreachable_paths

    unreachable = und_gradient_unreachable_paths(num_layers)
    return sorted(p for p in paths if p in unreachable)


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
    train_config["sensenova_mot_phase_eviction"] = bool(args.four_phase)
    train_config["sensenova_four_phase_eviction"] = bool(args.four_phase)
    train_understanding = args.branch in ("und", "both")

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
        train_unet=args.branch in ("gen", "both"),
        train_text_encoder=train_understanding,
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
    if branch != args.branch:
        raise AssertionError(f"expected the {args.branch} branch, got {branch!r}")
    if args.four_phase:
        if getattr(trainer, "sensenova_four_phase", None) is None:
            raise AssertionError("four-phase eviction was requested but not installed")
        if getattr(trainer, "sensenova_phase_evictor", None) is None:
            raise AssertionError("four-phase eviction requires the MoT evictor")
    adapter_name = type(trainer.adapter).__name__
    if adapter_name != "SenseNovaFullParameterAdapter":
        raise AssertionError(
            f"the SD1.5 fallthrough was taken: adapter is {adapter_name}"
        )
    targets = _gen_targets(trainer.transformer, branch)
    expected_targets = EXPECTED_TARGETS * (2 if branch == "both" else 1)
    if len(targets) != expected_targets:
        raise AssertionError(f"expected {expected_targets} targets, got {len(targets)}")

    groups = trainer.setup_trainable_parameters()
    parameter_groups = [
        {"lr": g["lr"], "tensors": len(g["params"]),
         "elements": sum(p.numel() for p in g["params"])}
        for g in groups
    ]
    trainable_elements = sum(g["elements"] for g in parameter_groups)

    before = _digest_map(trainer.transformer, branch)
    sampled = [path for path, _ in targets[:SAMPLED_LINEARS]]
    sampled_before = {
        path: dict(targets)[path].weight.detach().to("cpu", copy=True)
        for path in sampled
    }

    # Armed here, before train() builds the optimizer: it is the mechanism that
    # catches a hook that never fires, which no loss curve can show.
    trainer.optimizer_update_census = True

    losses: list[float] = []
    steps: list[int] = []
    # train()'s finally nulls sensenova_phase_evictor, so reading it afterwards
    # reports None whether four-phase ran or was never installed. Sample it while
    # the run is live instead.
    evictor_states: list[str] = []

    def progress_callback(phase, step, total, epoch=0, loss=None):
        del total, epoch
        if phase != "training":
            return
        live = getattr(trainer, "sensenova_phase_evictor", None)
        if live is not None:
            evictor_states.append(str(live.state))
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

    after = _digest_map(trainer.transformer, branch)
    moved = sorted(p for p in before if after[p] != before[p])
    unmoved = sorted(p for p in before if after[p] == before[p])
    sampled_moved_fraction = {}
    for path in sampled:
        # .cpu(): four-phase teardown normalizes every weight to host memory, so
        # the post-run tensor and the pre-run clone can sit on different devices.
        now = dict(_gen_targets(trainer.transformer, branch))[path].weight.detach().cpu()
        changed = (now != sampled_before[path]).sum().item()
        sampled_moved_fraction[path] = {
            "elements": int(now.numel()),
            "changed": int(changed),
            "fraction": float(changed) / float(now.numel()),
        }

    # THE U-2-5 CRITERION, as an assertion rather than a number to read off
    # (13.4): the set that did not move must be exactly the paths
    # und_gradient_unreachable_paths() predicts, intersected with this branch's
    # enumeration -- empty on `gen`, five on `und` and `both`. "Everything moved"
    # is the WRONG assertion here and is supposed to fail on those five.
    predicted_unmoved = u2_5_unmoved_expectation(
        before, len(trainer.transformer.language_model.model.layers)
    )
    failures = train_arm_failures(
        moved=moved, unmoved=unmoved, of=len(before),
        predicted_unmoved=predicted_unmoved, steps=steps,
    )

    census = trainer._update_census
    checkpoint = output_dir / f"{RUN_NAME}_step_{EXIT_SMOKE_STEPS:06d}.safetensors"
    entry = checkpoint if checkpoint.is_file() else Path(str(checkpoint) + ".index.json")
    # A post-run FACT about the run, like the census and the step list -- not a
    # precondition. Raising here discarded everything the run had just measured.
    saved = entry.is_file()
    if not saved:
        failures.append(
            f"the run saved neither {checkpoint} nor its shard index"
        )

    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    written_bytes = sum(
        p.stat().st_size for p in entry.parent.glob(f"{entry.stem.split('.')[0]}*")
        if p.is_file()
    ) if saved else 0
    return {
        "arm": "train",
        "four_phase_eviction": bool(args.four_phase),
        "evictor_states_during_run": evictor_states,
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
            "predicted_unmoved": predicted_unmoved,
            "expected_moved": len(before) - len(predicted_unmoved),
            "holds": not failures,
        },
        "sampled_element_moved_fraction": sampled_moved_fraction,
        "save_format_requested": args.save_format,
        "save_format_effective": _effective_format(branch, args.save_format),
        "checkpoint_entry": str(entry),
        "failures": failures,
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
            # The reproducible sibling: peak_wset is not comparable between
            # sessions, peak commit is.
            "peak_commit_gib": _host_peak_commit_bytes() / 1024 ** 3,
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

    # The FILE's own words, not the training arm's: the reader is what a later
    # run has, and the metadata is what it reads. Cross-checked against the
    # training arm below rather than trusted from either side alone.
    # Raised rather than defaulted: falling back to the training arm's own
    # answer makes the cross-check below compare a value with itself, which is
    # exactly the "the file's own words" property this block claims. The writer
    # always emits both keys, so absence means the file is not one of ours.
    metadata = components.get("metadata") or {}
    missing = [
        key for key in ("sensenova_trained_branch", "sensenova_save_format")
        if not metadata.get(key)
    ]
    if missing:
        raise AssertionError(
            f"the checkpoint at {entry} carries no {', '.join(missing)} in its "
            f"metadata, so the reload arm cannot read the branch and format from "
            f"the file itself. save_sensenova_full_finetune_checkpoint always "
            f"writes both."
        )
    branch = metadata["sensenova_trained_branch"]
    effective = metadata["sensenova_save_format"]
    trained_halves, frozen_halves = _halves(branch)

    def _targets(half: str) -> list[tuple[str, Any]]:
        return [
            (path, module)
            for path, _p, _a, module in iter_sensenova_lora_targets(
                transformer, branch=half
            )
        ]

    def _float_count(pairs) -> int:
        return sum(
            1 for _p, m in pairs
            if isinstance(getattr(m, "weight", None), nn.Parameter)
            and m.weight.dtype.is_floating_point
        )

    per_half = {}
    for half in ("gen", "und"):
        pairs = _targets(half)
        per_half[half] = {
            "linears": len(pairs),
            "float_materialized": _float_count(pairs),
            "int8": sum(1 for _p, m in pairs if type(m) is Int8Linear),
            "role": "trained" if half in trained_halves else "frozen",
        }

    want = expected_read_shape(branch, effective)
    failures: list[str] = []
    for half, stats in per_half.items():
        want_int8 = want[half] == "int8"
        got = stats["int8"] if want_int8 else stats["float_materialized"]
        if got != stats["linears"]:
            failures.append(
                f"{half} half ({stats['role']}) under save format {effective!r}: "
                f"expected all {stats['linears']} Linear(s) to be "
                f"{'Int8Linear' if want_int8 else 'floating nn.Parameter'}, got {got}"
            )

    saved = expected["post_train_digests"]
    if effective == "int8":
        # Requantization is lossy by construction, so byte equality is the wrong
        # question and its absence would mean nothing.
        digest_result = {"compared": False, "why": "int8 requantization is lossy"}
    else:
        digests: dict[str, str] = {}
        for half in trained_halves:
            digests.update({p: _linear_digest(m.weight) for p, m in _targets(half)})
        mismatched = sorted(p for p in saved if digests.get(p) != saved[p])
        digest_result = {
            "compared": True,
            "matches": len(saved) - len(mismatched),
            "of": len(saved),
            "mismatched": mismatched[:10],
        }
        if mismatched:
            failures.append(
                f"{len(mismatched)} of {len(saved)} trained-half weight(s) do not "
                f"match the training arm byte for byte (first: {mismatched[0]})"
            )

    # Both sides are now independent: the branch came from the file, this is the
    # training arm's. With the old fallback in place this compared a value with
    # itself whenever the metadata key was absent.
    for key, mine in (("branch", branch), ("save_format_effective", effective)):
        theirs = expected.get(key)
        if theirs is not None and theirs != mine:
            failures.append(
                f"checkpoint metadata says {key} {mine!r}, the training arm said "
                f"{theirs!r}"
            )

    return {
        "arm": "reload",
        "checkpoint_entry": entry,
        "load_wall_time_s": load_wall_time_s,
        "branch_from_metadata": branch,
        "save_format_effective": effective,
        "save_format_requested": metadata.get("sensenova_save_format_requested"),
        "per_half": per_half,
        "digests": digest_result,
        "failures": failures,
        "host_rss_peak_gib": _host_peak_bytes() / 1024 ** 3,
        "host_peak_commit_gib": _host_peak_commit_bytes() / 1024 ** 3,
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
    parser.add_argument("--branch", choices=("gen", "und", "both"), default="gen")
    parser.add_argument("--four-phase", action="store_true",
                        help="arm sensenova_four_phase_eviction (8.3.2)")
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
    # Written and printed FIRST: a criterion that fails is the measurement, and
    # a run that cost 25 GiB of writes should not also cost its own numbers.
    failures = result.get("failures") or []
    if failures:
        raise AssertionError("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
