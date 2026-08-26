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
* ``resume`` -- a FRESH process that CONTINUES the train arm's run from the
                checkpoint in its own output_dir: step/epoch position, Adafactor
                state, LR-scheduler position, the update census on the first step
                after the resume, and the trained half compared byte for byte
                against what the train arm held when it saved.

The two arms are separate processes because their host-RAM peaks do not overlap:
the trainer holds the dequantized half while the reader holds a whole second
model.

Nothing here claims quality. Phase 2b's exit criteria are "training is not
broken" only; the horizon is far too short for stochastic rounding's error to be
below the signal (6.3).

The train arm doubles as the MoT PHASE-SWAP INSTRUMENT: per-step wall time and
the evictor's own per-direction seconds and bytes, summarized over the
post-warmup steps, beside the run's allocated AND reserved CUDA peaks. Section
8.6 states this loop's transfer volume as arithmetic and ships
``sensenova_mot_overlap_transfer`` off because what it buys is unmeasured; these
flags (``--overlap-transfer``, ``--phase-eviction``, ``--vram-fraction``,
``--warmup-steps``, ``--label``) are how an arm measures it. They change nothing
for a caller that names none of them.
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
    VRAM_GATE_FRACTION,
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


class _StepPeakRecorder:
    """Separate the peak DURING a training step from the load-time high-water.

    Every measurement before this one reported a single process-wide
    ``max_memory_allocated``, in which the load high-water and the step are the
    same number: at 64px the step's activations are four image tokens, so the
    load peak dominated and ``model_resident == peak_allocated`` held. That
    equality is the reason the resolution at which the STEP becomes the peak was
    unknown -- it cannot be read off a statistic that never separated them.

    The window is callback-to-callback: ``progress_callback(phase="training")``
    fires after forward, backward and (under fused backward) the parameter
    updates, and before the ``should_step_optimizer`` block, so window N covers
    the tail of step N-1's optimizer block, the data load and the whole of step
    N. ``reset_peak_memory_stats`` resets the peak to the CURRENT allocation, so
    each window peak is bounded below by the resident model and the quantity
    that matters is ``peak - baseline``: what the step added on top.
    """

    def __init__(self):
        self.windows: list[dict[str, Any]] = []

    def _sample(self, label: str) -> None:
        torch.cuda.synchronize()
        self.windows.append({
            "label": label,
            "peak_allocated": int(torch.cuda.max_memory_allocated()),
            "peak_reserved": int(torch.cuda.max_memory_reserved()),
            "allocated_at_close": int(torch.cuda.memory_allocated()),
        })
        torch.cuda.reset_peak_memory_stats()

    def open_first_window(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def close_step(self, step: int) -> None:
        self._sample(f"step_{step}")

    def close_tail(self) -> None:
        self._sample("after_train")

    def summary(self, load_peak_allocated: int) -> dict[str, Any]:
        steps = [w for w in self.windows if w["label"].startswith("step_")]
        peak = max((w["peak_allocated"] for w in self.windows), default=0)
        step_peak = max((w["peak_allocated"] for w in steps), default=0)
        return {
            "windows": [
                {
                    "label": w["label"],
                    "peak_allocated_gib": w["peak_allocated"] / 1024 ** 3,
                    "peak_reserved_gib": w["peak_reserved"] / 1024 ** 3,
                    "allocated_at_close_gib": w["allocated_at_close"] / 1024 ** 3,
                }
                for w in self.windows
            ],
            "train_phase_peak_allocated_bytes": peak,
            "train_phase_peak_allocated_gib": peak / 1024 ** 3,
            "step_only_peak_allocated_bytes": step_peak,
            "step_only_peak_allocated_gib": step_peak / 1024 ** 3,
            "load_peak_allocated_bytes": load_peak_allocated,
            "load_peak_allocated_gib": load_peak_allocated / 1024 ** 3,
            "step_exceeds_load": bool(step_peak > load_peak_allocated),
            "step_minus_load_bytes": step_peak - load_peak_allocated,
            "step_minus_load_gib": (step_peak - load_peak_allocated) / 1024 ** 3,
            # Growth across steady-state windows: a monotone climb is allocator
            # fragmentation or a leak, which a three-step run cannot show.
            "first_step_peak_gib": (
                steps[0]["peak_allocated"] / 1024 ** 3 if steps else None
            ),
            "last_step_peak_gib": (
                steps[-1]["peak_allocated"] / 1024 ** 3 if steps else None
            ),
            "steady_state_drift_gib": (
                (steps[-1]["peak_allocated"] - steps[1]["peak_allocated"]) / 1024 ** 3
                if len(steps) > 2 else None
            ),
        }


def _clock() -> float:
    """A step-boundary timestamp with the device's queue drained first."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _stat(values: list[float]) -> dict[str, Any]:
    import statistics

    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
    }


# The evictor series base_trainer logs from its single per-step drain.
_SN_METRIC_KEYS = (
    "sn_d2h_s", "sn_h2d_s", "sn_d2h_gib", "sn_h2d_gib", "sn_swap_overlap",
    "sn_peak_alloc_gib", "sn_peak_resv_gib",
)


class _StepTransferRecorder:
    """Per-step swap cost, taken from the trainer's OWN drain rather than a second one.

    ``base_trainer`` calls ``drain_transfer_stats()`` exactly once per step and
    the drain RESETS the evictor's accumulators, so a probe that also drained
    would read whichever of the two ran second -- zeros for one side, and the
    trainer's own charted series silently emptied. This wraps
    ``trainer.log_extra_metric`` instead: the drain stays single, and what is
    recorded here is byte for byte what the trainer recorded.

    ``sn_peak_alloc_gib`` / ``sn_peak_resv_gib`` are the trainer's reading of the
    CUDA peak counters at the drain point, which ``_StepPeakRecorder`` resets
    once per step -- they are per-window, not process-wide, and the run-level
    peaks in the report come from that recorder instead.
    """

    def __init__(self):
        self._pending: dict[str, Any] = {}
        self.steps: list[dict[str, Any]] = []
        self._mark: float | None = None

    def install(self, trainer) -> None:
        real = trainer.log_extra_metric

        def recording(name, value):
            if name in _SN_METRIC_KEYS:
                self._pending[name] = value
            return real(name, value)

        trainer.log_extra_metric = recording

    def start(self) -> None:
        self._mark = _clock()

    def close_step(self, step: int) -> None:
        now = _clock()
        wall = None if self._mark is None else now - self._mark
        self._mark = now
        entry: dict[str, Any] = {"step": int(step), "wall_s": wall}
        entry.update({key: self._pending.get(key) for key in _SN_METRIC_KEYS})
        self._pending = {}
        self.steps.append(entry)

    def summary(self, warmup: int) -> dict[str, Any]:
        """Statistics over the POST-WARMUP steps only.

        The first swaps are not steady state: every pinned staging block is a
        fresh device-synchronizing ``cudaHostAlloc``, and torch's caching host
        allocator only converges once each distinct tensor size has been freed
        once. SENSENOVA_TRAINING_DESIGN.md 8.6 retracted a transfer number
        measured without this exclusion.
        """
        warmup = max(0, int(warmup))
        steady = self.steps[warmup:]
        notes: list[str] = []
        if not steady:
            notes.append(
                f"no steady-state steps: {len(self.steps)} step(s) ran and "
                f"{warmup} were excluded as warmup"
            )
        wall = [s["wall_s"] for s in steady]
        d2h_s = [s["sn_d2h_s"] for s in steady]
        h2d_s = [s["sn_h2d_s"] for s in steady]
        ratios = [
            (s["sn_d2h_s"] + s["sn_h2d_s"]) / s["wall_s"]
            for s in steady
            if s["wall_s"] and s["sn_d2h_s"] is not None and s["sn_h2d_s"] is not None
        ]
        overlap_flags = [
            s["sn_swap_overlap"] for s in steady if s["sn_swap_overlap"] is not None
        ]
        if steady and not any(s["sn_d2h_s"] is not None for s in steady):
            notes.append(
                "no evictor transfer series was logged: either the phase evictor "
                "is off or it never transitioned"
            )
        return {
            "warmup_steps": warmup,
            "steady_state_steps": len(steady),
            "step_wall_s": _stat(wall),
            "d2h_s": _stat(d2h_s),
            "h2d_s": _stat(h2d_s),
            "d2h_gib_per_step": _stat([s["sn_d2h_gib"] for s in steady]),
            "h2d_gib_per_step": _stat([s["sn_h2d_gib"] for s in steady]),
            # The transfer share of a step, read directly. Under overlap the two
            # seconds are concurrent CUDA-event times on two streams, so their
            # sum can exceed the step wall and this ratio can pass 1.0; that is
            # the unit changing, not a step spending more than its own time.
            "transfer_share_of_step": _stat(ratios),
            "overlap_active_all_steps": bool(overlap_flags) and all(
                float(v) == 1.0 for v in overlap_flags
            ),
            "overlap_active_any_step": any(float(v) == 1.0 for v in overlap_flags),
            "notes": notes,
        }


def train_arm_failures(
    *,
    moved,
    unmoved,
    of: int,
    predicted_unmoved,
    steps,
    expected_steps: int = EXIT_SMOKE_STEPS,
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
    if list(steps) != list(range(1, expected_steps + 1)):
        failures.append(f"expected {expected_steps} steps, got {list(steps)}")
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

    phase_eviction = _resolve_phase_eviction(args)
    vram_gate = _apply_vram_gate(_vram_fraction(args))
    config = trainer_exit_smoke_config()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    resolution = int(args.resolution)
    total_steps = int(args.steps)
    image_path = workdir / f"training_image_{resolution}.png"
    output_dir = workdir / "full_finetune"
    _write_deterministic_smoke_image(image_path, resolution, resolution)

    train_config = dict(config["train_config"])
    train_config["sensenova_full_finetune_save_format"] = args.save_format
    train_config["sensenova_mot_phase_eviction"] = phase_eviction
    train_config["sensenova_four_phase_eviction"] = bool(args.four_phase)
    train_config["use_reference_images"] = bool(args.reference)
    # Left unset when the caller named neither flag, so the installer reads
    # TRAINING_DEFAULTS (OFF) exactly as it did before this arm could ask.
    overlap_requested = getattr(args, "overlap_transfer", None)
    if overlap_requested is not None:
        train_config["sensenova_mot_overlap_transfer"] = bool(overlap_requested)
    train_understanding = args.branch in ("und", "both")

    reference_paths: list[str] = []
    if args.reference:
        # Deliberately a different geometry from the target: the reference does
        # not participate in bucketing at all (7.5 differential 4), so a smoke
        # that reused the target's square would not show that.
        reference_image = workdir / "reference_512x512.png"
        _write_deterministic_smoke_image(reference_image, 512, 512)
        reference_paths = [str(reference_image)]

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
    # The load-time high-water, read BEFORE anything resets it. Every earlier
    # measurement conflated this with the step peak.
    load_peak_allocated = int(torch.cuda.max_memory_allocated())
    load_peak_reserved = int(torch.cuda.max_memory_reserved())

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
    if phase_eviction and getattr(trainer, "sensenova_phase_evictor", None) is None:
        raise AssertionError("phase eviction was requested but no evictor was installed")
    if not phase_eviction and getattr(trainer, "sensenova_phase_evictor", None) is not None:
        raise AssertionError("phase eviction was refused but an evictor was installed")
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
    step_peaks = _StepPeakRecorder()
    transfers = _StepTransferRecorder()
    transfers.install(trainer)
    # What the evictor ACTUALLY ran, sampled live: the flag says what was asked
    # for, and a run that lost the overlap to a pin failure looks identical from
    # the config alone.
    overlap_observed: dict[str, Any] = {}

    def progress_callback(phase, step, total, epoch=0, loss=None):
        del total, epoch
        if phase != "training":
            return
        live = getattr(trainer, "sensenova_phase_evictor", None)
        if live is not None:
            evictor_states.append(str(live.state))
            overlap_observed["configured"] = bool(getattr(live, "_overlap", False))
            overlap_observed["downgraded"] = bool(
                getattr(live, "_overlap_downgraded", False)
            )
        if loss is None or not math.isfinite(float(loss)):
            raise AssertionError(f"non-finite SenseNova full-FT loss: {loss!r}")
        steps.append(int(step))
        losses.append(float(loss))
        # Before the peak recorder: its _sample resets the peak counters, and the
        # step wall must be closed on the step's own synchronize.
        transfers.close_step(int(step))
        step_peaks.close_step(int(step))

    train = dict(config["train"])
    train.update({
        "num_epochs": 1,
        "optimizer_type": "adafactor",
        "total_steps": total_steps,
        "base_resolutions": [resolution],
        # total_steps + 1, NOT 0: `base_trainer.train` computes
        # `global_step % save_every_n_steps` with no guard, so 0 -- the obvious
        # spelling of "never save" -- raises ZeroDivisionError at step 1 on
        # every architecture, and the emergency handler then writes a full
        # checkpoint anyway. Reported as a finding; routed around here.
        "save_every_n_steps": (total_steps + 1) if args.no_save else total_steps,
        "sample_prompts": [],
        "sample_every_n_steps": 0,
        "sample_width": resolution,
        "sample_height": resolution,
        "sample_seed": args.seed,
        "max_grad_norm": 0.0,
        "progress_callback": progress_callback,
        "run_id": None,
        "max_step_saves_to_keep": 1,
        "force_recache": False,
        "use_reference_images": bool(args.reference),
    })
    dataset = _ExitSmokeDataset(
        image_path, args.prompt, resolution, resolution,
        reference_images=reference_paths,
    )

    # What the step ACTUALLY ran at, from the tensor rather than from the flag
    # that was set: with bucketing off, base_resolutions only clamps DOWN, so a
    # resolution arm that forgot either half of the pair would silently run at
    # 64px and report a resolution it never used.
    from core.training.ops import sensenova_ops as _ops

    observed_shapes: list[list[int]] = []
    _original_train_step = _ops.train_step

    def _recording_train_step(trainer_, *, images, **kwargs):
        shape = [int(v) for v in images.shape]
        if shape not in observed_shapes:
            observed_shapes.append(shape)
        return _original_train_step(trainer_, images=images, **kwargs)

    _ops.train_step = _recording_train_step

    # Whether each prefix was actually reference-conditioned, read from the call
    # rather than from the flag that was set: `use_reference_images` arms the
    # route and per-item presence decides it (7.2 judgement 1), so a run that
    # armed the route and lost the item's paths would otherwise look identical.
    prefix_records: list[dict[str, Any]] = []
    _original_encode = _ops.encode_prompt

    def _recording_encode_prompt(trainer_, prompt, **kwargs):
        prefix = _original_encode(trainer_, prompt, **kwargs)
        prefix_records.append({
            "has_reference": bool(kwargs.get("reference_image_paths")),
            "requires_grad": bool(kwargs.get("requires_grad")),
            "prefix_seq_length": int(prefix.cache.get_seq_length()),
            "text_length": int(prefix.text_length),
            "kv_grad_fn": prefix.cache.layers[0].keys.grad_fn is not None,
            "kv_requires_grad": bool(prefix.cache.layers[0].keys.requires_grad),
        })
        return prefix

    _ops.encode_prompt = _recording_encode_prompt
    train_started = time.perf_counter()
    step_peaks.open_first_window()
    transfers.start()
    try:
        trainer.train(datasets=[dataset], **train)
    finally:
        _ops.train_step = _original_train_step
        _ops.encode_prompt = _original_encode
    step_peaks.close_tail()
    train_wall_time_s = time.perf_counter() - train_started
    if [s[-2:] for s in observed_shapes] != [[resolution, resolution]]:
        raise AssertionError(
            f"asked for {resolution}px but the training step saw {observed_shapes}"
        )

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
        expected_steps=total_steps,
    )
    if args.reference and not all(r["has_reference"] for r in prefix_records):
        failures.append(
            "a reference arm ran prefixes without references: "
            f"{sum(1 for r in prefix_records if not r['has_reference'])} of "
            f"{len(prefix_records)} were text-only"
        )
    if train_understanding and not all(
        r["kv_grad_fn"] or r["kv_requires_grad"] for r in prefix_records
    ):
        failures.append(
            "the understanding branch was trained but a prefix arrived with "
            "neither a grad_fn nor a grad-requiring boundary leaf"
        )

    census = trainer._update_census
    checkpoint = output_dir / f"{RUN_NAME}_step_{total_steps:06d}.safetensors"
    entry = checkpoint if checkpoint.is_file() else Path(str(checkpoint) + ".index.json")
    # A post-run FACT about the run, like the census and the step list -- not a
    # precondition. Raising here discarded everything the run had just measured.
    saved = entry.is_file()
    if not saved and not args.no_save:
        failures.append(
            f"the run saved neither {checkpoint} nor its shard index"
        )

    # The recorder resets the CUDA peak counters once per step, so the
    # process-wide peak has to be reassembled from the windows plus the load
    # high-water rather than read back from torch.
    _windows = step_peaks.summary(load_peak_allocated)
    peak_allocated = max(load_peak_allocated,
                         _windows["train_phase_peak_allocated_bytes"])
    peak_reserved = max(
        load_peak_reserved,
        max((w["peak_reserved"] for w in step_peaks.windows), default=0),
    )
    written_bytes = sum(
        p.stat().st_size for p in entry.parent.glob(f"{entry.stem.split('.')[0]}*")
        if p.is_file()
    ) if saved else 0
    transfer_summary = transfers.summary(int(getattr(args, "warmup_steps", 0)))
    return {
        "arm": "train",
        "label": getattr(args, "label", None),
        "resolution": resolution,
        "observed_step_image_shapes": observed_shapes,
        "requested_steps": total_steps,
        "saved_checkpoint": bool(saved),
        "four_phase_eviction": bool(args.four_phase),
        "phase_eviction": phase_eviction,
        "overlap_transfer_requested": overlap_requested,
        "overlap_transfer_observed": dict(overlap_observed),
        "vram_fraction": _vram_fraction(args),
        "transfer_per_step": transfers.steps,
        "transfer_steady_state": transfer_summary,
        "reference_conditioned": bool(args.reference),
        "reference_image_paths": reference_paths,
        "prefix_records": prefix_records,
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
            "model_resident_bytes": int(model_resident["allocated"]),
            "model_resident_gib": model_resident["allocated"] / 1024 ** 3,
            "peak_allocated_bytes": peak_allocated,
            "peak_allocated_gib": peak_allocated / 1024 ** 3,
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_gib": peak_reserved / 1024 ** 3,
            # RESERVED matters as much as allocated here. The overlap path's
            # destination-allocation rule exists so a side-stream copy can reuse
            # a block the default stream just freed; if it did not hold, the
            # allocator grows by a whole half and only this number shows it.
            "peak_reserved_minus_allocated_gib": (peak_reserved - peak_allocated) / 1024 ** 3,
            "load_peak_reserved_gib": load_peak_reserved / 1024 ** 3,
            "gate_budget_gib": vram_gate.get("budget_bytes", 0) / 1024 ** 3,
            "device_total_gib": vram_gate.get("device_total_bytes", 0) / 1024 ** 3,
        },
        "step_vs_load": step_peaks.summary(load_peak_allocated),
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


# ---------------------------------------------------------------------------
# Arm 3 -- a real resume: same output_dir, fresh process, no materialization
# ---------------------------------------------------------------------------


def _run_resume_arm(args: argparse.Namespace) -> dict[str, Any]:
    """Continue the train arm's run from its own checkpoint and report what carried.

    The train arm's ``--out`` JSON is passed as ``--expect``; this arm reuses its
    ``output_dir``, resumes with ``resume_from_checkpoint='latest'``, and records
    the four things a resume is supposed to carry (step/epoch position, optimizer
    state, scheduler position, and the per-step update census on the first step
    after the resume) plus the one this route adds: the trained half's weights
    compared BYTE FOR BYTE against what the train arm held when it saved.
    """
    from core.training.full_parameter_trainer import FullParameterTrainer
    from core.training.ops.sensenova_ops import resolve_full_finetune_branch

    expected = json.loads(Path(args.expect).read_text(encoding="utf-8"))
    vram_gate = _apply_vram_gate(_vram_fraction(args))
    config = trainer_exit_smoke_config()
    workdir = Path(args.workdir)
    resolution = int(expected["resolution"])
    first_steps = int(expected["requested_steps"])
    total_steps = first_steps + int(args.resume_extra_steps)
    image_path = workdir / f"training_image_{resolution}.png"
    output_dir = workdir / "full_finetune"
    _write_deterministic_smoke_image(image_path, resolution, resolution)

    train_config = dict(config["train_config"])
    train_config["sensenova_full_finetune_save_format"] = args.save_format
    train_config["sensenova_mot_phase_eviction"] = _resolve_phase_eviction(args)
    train_config["sensenova_four_phase_eviction"] = bool(args.four_phase)
    train_config["use_reference_images"] = False
    branch_arg = expected["branch"]

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    trainer = FullParameterTrainer(
        model_path=args.model_path,
        output_dir=str(output_dir),
        run_name=RUN_NAME,
        run_id=None,
        learning_rate=LEARNING_RATE,
        unet_lr=LEARNING_RATE,
        train_unet=branch_arg in ("gen", "both"),
        train_text_encoder=branch_arg in ("und", "both"),
        device="cuda",
        weight_dtype="bf16",
        training_dtype="bf16",
        output_dtype="bf16",
        vae_dtype="bf16",
        mixed_precision=True,
        attention_backend="native",
        use_flash_attention=False,
        blocks_to_swap=0,
        resume_from_checkpoint="latest",
        train_config=train_config,
    )
    model_load_wall_time_s = time.perf_counter() - load_started
    load_peak_allocated = int(torch.cuda.max_memory_allocated())
    branch = resolve_full_finetune_branch(trainer)

    failures: list[str] = []
    loaded_checkpoint = getattr(trainer, "_loaded_checkpoint_path", None)
    if not loaded_checkpoint:
        failures.append("the trainer did not resolve a checkpoint to resume from")
    resumed_format = getattr(trainer, "sensenova_resumed_save_format", None)
    if resumed_format is None:
        failures.append(
            "the resume went through the materializing route, not the "
            "resume-shaped acceptance (sensenova_resumed_save_format is None)"
        )

    # THE LOSSLESSNESS CLAIM: the tree the resume loaded, against the weights the
    # train arm held when it wrote the file. Taken BEFORE any step runs.
    saved_digests = expected["post_train_digests"]
    at_resume = _digest_map(trainer.transformer, branch)
    mismatched = sorted(p for p in saved_digests if at_resume.get(p) != saved_digests[p])
    if mismatched:
        failures.append(
            f"{len(mismatched)} of {len(saved_digests)} trained-half weight(s) "
            f"differ from what the train arm saved (first: {mismatched[0]})"
        )

    # The resume's own bookkeeping, captured at the point base_trainer restores it.
    restored: dict[str, Any] = {}
    _real_load_optimizer_state = trainer.load_optimizer_state
    _real_load_training_state = trainer.load_training_state

    def _capture_load_optimizer_state(step: int):
        scheduler = getattr(trainer, "lr_scheduler", None)
        before = len(getattr(trainer.optimizer, "state", {}) or {})
        ok = _real_load_optimizer_state(step)
        state = getattr(trainer.optimizer, "state", {}) or {}
        sample = None
        for entry in state.values():
            sample = sorted(str(k) for k in entry)
            break
        restored["optimizer"] = {
            "requested_step": int(step),
            "loaded": bool(ok),
            "param_states_before": before,
            "param_states_after": len(state),
            "first_param_state_keys": sample,
            # Adafactor's own step counter, the thing a fresh optimizer would
            # have at 0 and the thing its bias/decay schedule reads.
            "adafactor_step": next(
                (int(e["step"]) for e in state.values()
                 if isinstance(e.get("step"), (int, float))), None
            ),
        }
        restored["scheduler_at_resume"] = {
            "last_epoch": int(getattr(scheduler, "last_epoch", -1)),
            "last_lr": [float(v) for v in scheduler.get_last_lr()] if scheduler else None,
        }
        return ok

    def _capture_load_training_state(step: int):
        state = _real_load_training_state(step)
        restored["training_state"] = dict(state) if state else None
        return state

    trainer.load_optimizer_state = _capture_load_optimizer_state
    trainer.load_training_state = _capture_load_training_state
    trainer.optimizer_update_census = True

    before_resumed_steps = dict(at_resume)
    losses: list[float] = []
    steps: list[int] = []
    step_peaks = _StepPeakRecorder()

    def progress_callback(phase, step, total, epoch=0, loss=None):
        del total, epoch
        if phase != "training":
            return
        if loss is None or not math.isfinite(float(loss)):
            raise AssertionError(f"non-finite SenseNova resumed loss: {loss!r}")
        steps.append(int(step))
        losses.append(float(loss))
        step_peaks.close_step(int(step))

    train = dict(config["train"])
    train.update({
        "num_epochs": 1,
        "optimizer_type": "adafactor",
        "total_steps": total_steps,
        "base_resolutions": [resolution],
        "save_every_n_steps": total_steps,
        "sample_prompts": [],
        "sample_every_n_steps": 0,
        "sample_width": resolution,
        "sample_height": resolution,
        "sample_seed": args.seed,
        "max_grad_norm": 0.0,
        "progress_callback": progress_callback,
        "run_id": None,
        "max_step_saves_to_keep": 2,
        "force_recache": False,
        "use_reference_images": False,
        "resume_from_checkpoint": "latest",
    })
    dataset = _ExitSmokeDataset(image_path, args.prompt, resolution, resolution)

    train_started = time.perf_counter()
    step_peaks.open_first_window()
    trainer.train(datasets=[dataset], **train)
    step_peaks.close_tail()
    train_wall_time_s = time.perf_counter() - train_started

    after = _digest_map(trainer.transformer, branch)
    moved = sorted(p for p in before_resumed_steps if after[p] != before_resumed_steps[p])
    unmoved = sorted(p for p in before_resumed_steps if after[p] == before_resumed_steps[p])
    predicted_unmoved = u2_5_unmoved_expectation(
        before_resumed_steps, len(trainer.transformer.language_model.model.layers)
    )
    if sorted(unmoved) != sorted(predicted_unmoved):
        failures.append(
            f"post-resume update-nonzero census: {len(moved)} of "
            f"{len(before_resumed_steps)} moved; unmoved is {sorted(unmoved)[:10]} "
            f"but und_gradient_unreachable_paths() predicts {sorted(predicted_unmoved)}"
        )

    # THE STEP-POSITION CLAIM: the run continued, it did not restart.
    expected_steps = list(range(first_steps + 1, total_steps + 1))
    if steps != expected_steps:
        failures.append(
            f"expected the resumed run to report steps {expected_steps}, got {steps}"
        )
    optimizer_record = restored.get("optimizer") or {}
    if not optimizer_record.get("loaded"):
        failures.append("the Adafactor state was not restored from the checkpoint")
    if optimizer_record.get("param_states_after", 0) <= 0:
        failures.append("the restored optimizer holds no per-parameter state")
    scheduler_at_resume = restored.get("scheduler_at_resume") or {}
    if scheduler_at_resume.get("last_epoch") != first_steps:
        failures.append(
            f"the LR scheduler resumed at position "
            f"{scheduler_at_resume.get('last_epoch')}, expected {first_steps}"
        )
    census = trainer._update_census

    _windows = step_peaks.summary(load_peak_allocated)
    peak_allocated = max(load_peak_allocated,
                         _windows["train_phase_peak_allocated_bytes"])
    return {
        "arm": "resume",
        "resumed_from": str(loaded_checkpoint),
        "resumed_save_format": resumed_format,
        "branch": branch,
        "first_run_steps": first_steps,
        "total_steps": total_steps,
        "steps": steps,
        "losses": losses,
        "weights_identical_to_saved": {
            "matches": len(saved_digests) - len(mismatched),
            "of": len(saved_digests),
            "mismatched": mismatched[:10],
        },
        "restored": restored,
        "scheduler_after_run": {
            "last_epoch": int(getattr(trainer.lr_scheduler, "last_epoch", -1)),
            "last_lr": [float(v) for v in trainer.lr_scheduler.get_last_lr()],
        },
        "update_census": {
            "expected": census.expected_count if census else None,
            "steps_checked": census.steps_checked if census else None,
            "exempt": sorted(census.exempt) if census else None,
        },
        "moved_census": {
            "moved": len(moved),
            "unmoved": len(unmoved),
            "of": len(before_resumed_steps),
            "predicted_unmoved": predicted_unmoved,
        },
        "failures": failures,
        "vram": {
            "load_peak_allocated_gib": load_peak_allocated / 1024 ** 3,
            "peak_allocated_gib": peak_allocated / 1024 ** 3,
            "gate_budget_gib": vram_gate.get("budget_bytes", 0) / 1024 ** 3,
            "device_total_gib": vram_gate.get("device_total_bytes", 0) / 1024 ** 3,
        },
        "step_vs_load": _windows,
        "host_rss": {
            "peak_gib": _host_peak_bytes() / 1024 ** 3,
            "peak_commit_gib": _host_peak_commit_bytes() / 1024 ** 3,
        },
        "wall_time_s": {
            "model_load": model_load_wall_time_s,
            "train_and_save": train_wall_time_s,
        },
    }


def _vram_fraction(args: argparse.Namespace) -> float:
    fraction = getattr(args, "vram_fraction", None)
    return VRAM_GATE_FRACTION if fraction is None else float(fraction)


def _resolve_phase_eviction(args: argparse.Namespace) -> bool:
    """Eviction follows ``--four-phase`` unless the caller named it explicitly.

    Unset reproduces the pre-flag behaviour exactly. Explicitly off UNDER
    ``--four-phase`` is refused here rather than several frames deeper in
    ``assert_four_phase_contract``, since the split has nothing to evict.
    """
    requested = getattr(args, "phase_eviction", None)
    if requested is None:
        return bool(args.four_phase)
    if args.four_phase and not requested:
        raise ValueError(
            "--no-phase-eviction cannot be combined with --four-phase: the split "
            "exists so an evicted understanding half can still be trained, and "
            "without eviction it is a second backward and a recomputed forward "
            "for nothing"
        )
    return bool(requested)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("train", "reload", "resume"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect", default=None,
                        help="the train arm's JSON, for --arm reload / --arm resume")
    parser.add_argument("--resume-extra-steps", type=int, default=2,
                        help="steps to run AFTER the resume, for --arm resume")
    parser.add_argument("--save-format", default="mixed")
    parser.add_argument("--branch", choices=("gen", "und", "both"), default="gen")
    parser.add_argument("--four-phase", action="store_true",
                        help="arm sensenova_four_phase_eviction (8.3.2)")
    parser.add_argument("--phase-eviction", dest="phase_eviction",
                        action="store_true", default=None,
                        help="arm sensenova_mot_phase_eviction independently of "
                             "--four-phase; unset follows --four-phase")
    parser.add_argument("--no-phase-eviction", dest="phase_eviction",
                        action="store_false",
                        help="run with both halves resident, for the arm that "
                             "asks whether that fits at all")
    parser.add_argument("--overlap-transfer", dest="overlap_transfer",
                        action="store_true", default=None,
                        help="set sensenova_mot_overlap_transfer; unset leaves "
                             "TRAINING_DEFAULTS (off)")
    parser.add_argument("--no-overlap-transfer", dest="overlap_transfer",
                        action="store_false")
    parser.add_argument("--vram-fraction", type=float, default=VRAM_GATE_FRACTION,
                        help="per-process VRAM cap; raise it only for an arm "
                             "whose question is whether a configuration fits in "
                             "the whole card")
    parser.add_argument("--warmup-steps", type=int, default=3,
                        help="leading steps EXCLUDED from the steady-state "
                             "transfer statistics (the first pinned staging "
                             "allocations are cudaHostAlloc, not a copy)")
    parser.add_argument("--label", default=None,
                        help="echoed into the JSON so arm outputs self-identify")
    parser.add_argument("--reference", action="store_true",
                        help="give the item a reference image, so the prefix is "
                             "reference-conditioned (Phase U-3)")
    parser.add_argument("--resolution", type=int, default=EXIT_SMOKE_WIDTH,
                        help="square training resolution; sets BOTH the dataset "
                             "item dims and base_resolutions")
    parser.add_argument("--steps", type=int, default=EXIT_SMOKE_STEPS)
    parser.add_argument("--no-save", action="store_true",
                        help="skip the checkpoint save (a 25-32 GiB write and a "
                             "whole second host-resident state dict) when the "
                             "arm is measuring the step rather than the writer")
    parser.add_argument("--prompt", default="a red square on a white background")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _format_summary(result: dict[str, Any]) -> str:
    """A compact transfer/VRAM digest for whoever reads stdout.

    Tolerant of every key: an arm that has no transfer series still prints its
    identity rather than raising over a missing one.
    """
    def num(value, fmt="{:.4f}") -> str:
        return "n/a" if value is None else fmt.format(float(value))

    steady = result.get("transfer_steady_state") or {}
    vram = result.get("vram") or {}
    observed = result.get("overlap_transfer_observed") or {}
    wall = steady.get("step_wall_s") or {}
    d2h = steady.get("d2h_s") or {}
    h2d = steady.get("h2d_s") or {}
    share = steady.get("transfer_share_of_step") or {}
    lines = [
        "=" * 68,
        f"SENSENOVA SWAP MEASUREMENT  label={result.get('label')!r} "
        f"arm={result.get('arm')!r}",
        f"  branch={result.get('branch')!r} resolution={result.get('resolution')} "
        f"steps={result.get('requested_steps')} "
        f"warmup={steady.get('warmup_steps')} "
        f"steady={steady.get('steady_state_steps')}",
        f"  phase_eviction={result.get('phase_eviction')} "
        f"four_phase={result.get('four_phase_eviction')} "
        f"overlap_requested={result.get('overlap_transfer_requested')} "
        f"overlap_configured={observed.get('configured')} "
        f"overlap_downgraded={observed.get('downgraded')} "
        f"vram_fraction={result.get('vram_fraction')}",
        f"  overlap_active: all_steps={steady.get('overlap_active_all_steps')} "
        f"any_step={steady.get('overlap_active_any_step')}",
        f"  step wall  s: median {num(wall.get('median'))} "
        f"min {num(wall.get('min'))} max {num(wall.get('max'))}",
        f"  d2h       s: median {num(d2h.get('median'))} "
        f"min {num(d2h.get('min'))} max {num(d2h.get('max'))}",
        f"  h2d       s: median {num(h2d.get('median'))} "
        f"min {num(h2d.get('min'))} max {num(h2d.get('max'))}",
        f"  GiB/step   : d2h {num((steady.get('d2h_gib_per_step') or {}).get('median'), '{:.3f}')} "
        f"h2d {num((steady.get('h2d_gib_per_step') or {}).get('median'), '{:.3f}')}",
        f"  (d2h+h2d)/step_wall: median {num(share.get('median'))} "
        f"min {num(share.get('min'))} max {num(share.get('max'))}",
        f"  VRAM peak  : allocated {num(vram.get('peak_allocated_gib'), '{:.3f}')} GiB  "
        f"reserved {num(vram.get('peak_reserved_gib'), '{:.3f}')} GiB  "
        f"reserved-allocated {num(vram.get('peak_reserved_minus_allocated_gib'), '{:.3f}')} GiB",
        f"  host RSS   : peak {num((result.get('host_rss') or {}).get('peak_gib'), '{:.2f}')} GiB  "
        f"commit {num((result.get('host_rss') or {}).get('peak_commit_gib'), '{:.2f}')} GiB",
    ]
    for note in steady.get("notes") or []:
        lines.append(f"  NOTE: {note}")
    for failure in result.get("failures") or []:
        lines.append(f"  FAILURE: {failure}")
    lines.append("=" * 68)
    return "\n".join(lines)


def main() -> int:
    _require_repo_venv()
    args = _parse_args()
    result = {
        "train": _run_train_arm, "reload": _run_reload_arm, "resume": _run_resume_arm,
    }[args.arm](args)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k != "post_train_digests"}, indent=2))
    print(_format_summary(result))
    # Written and printed FIRST: a criterion that fails is the measurement, and
    # a run that cost 25 GiB of writes should not also cost its own numbers.
    failures = result.get("failures") or []
    if failures:
        raise AssertionError("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
