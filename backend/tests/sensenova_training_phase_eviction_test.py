import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models.sensenova import mot_cpu_staging, mot_phase_eviction
from core.models.sensenova.mot_cpu_staging import stage_modules_to_pinned_cpu
from core.models.sensenova.mot_weight_selector import select_mot_weight_modules
from core.training import sensenova_phase_eviction
from core.training.sensenova_phase_eviction import SenseNovaTrainingPhaseEvictor


class BufferWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.ones(2))
        self.register_buffer("scratch", torch.ones(1), persistent=False)


class Layer(nn.Module):
    def __init__(self, *, asymmetric=False, with_lora=False):
        super().__init__()
        self.proj = BufferWeight()
        self.proj_mot_gen = BufferWeight()
        if asymmetric:
            self.extra = nn.Linear(2, 2, bias=False)
        if with_lora:
            self.proj_mot_gen.lora_down = nn.Linear(2, 1, bias=False)
            self.proj_mot_gen.lora_up = nn.Linear(1, 2, bias=False)
        self.rotary_emb = nn.Module()
        self.rotary_emb.register_buffer("inv_freq", torch.ones(1), persistent=False)


def transformer(*, count=42, asymmetric=False, with_lora=False):
    layers = nn.ModuleList(
        [
            Layer(asymmetric=asymmetric and index == 0, with_lora=with_lora)
            for index in range(count)
        ]
    )
    root = nn.Module()
    root.language_model = nn.Module()
    root.language_model.model = nn.Module()
    root.language_model.model.layers = layers
    return root


def test_selector_includes_persistent_buffers_parameters_and_generation_lora():
    model = transformer(with_lora=True)
    selected = select_mot_weight_modules(model, require_exact_symmetry=True)

    assert len(selected.und_modules) == 42
    assert len(selected.gen_modules) == 42 * 3
    assert all("scratch" not in module.state_dict() for module in selected.gen_modules)
    assert not any(
        module is layer.rotary_emb
        for layer in model.language_model.model.layers
        for module in selected.gen_modules
    )


def test_selector_fails_closed_for_missing_or_asymmetric_halves():
    with pytest.raises(RuntimeError, match="exactly 42"):
        select_mot_weight_modules(transformer(count=41), require_exact_symmetry=True)
    with pytest.raises(RuntimeError, match="asymmetric"):
        select_mot_weight_modules(transformer(asymmetric=True), require_exact_symmetry=True)


def test_state_machine_orders_two_cycles_and_is_idempotent():
    events = []

    evictor = SenseNovaTrainingPhaseEvictor(transformer(), "cuda")
    gen_ids = {id(module) for module in evictor._gen_modules}
    events.clear()

    def d2h_identity(modules, *, warn_once, pageable=False):
        del warn_once, pageable
        items = tuple(modules)
        events.append(("d2h", "gen" if id(items[0]) in gen_ids else "und"))

    def h2d_identity(modules, device):
        del device
        items = tuple(modules)
        events.append(("h2d", "gen" if id(items[0]) in gen_ids else "und"))

    with patch(
        "core.training.sensenova_phase_eviction._move_modules_to_cpu", d2h_identity
    ), patch(
        "core.training.sensenova_phase_eviction._move_modules_to_device",
        h2d_identity,
    ):
        evictor.enter_prefix()
        evictor.enter_prefix()
        evictor.enter_denoise()
        evictor.enter_denoise()
        evictor.enter_prefix()
        evictor.enter_denoise()
        evictor.teardown()
        evictor.teardown()

    def swap(out, incoming):
        return [("d2h", out), ("h2d", incoming)] * 42

    assert events == (
        [("d2h", "gen")] * 42          # full -> prefix (one-sided)
        + swap("und", "gen")           # prefix -> denoise, interleaved
        + swap("gen", "und")           # denoise -> prefix, interleaved
        + swap("und", "gen")           # prefix -> denoise, interleaved
        + [("d2h", "gen")] * 42        # teardown normalizes both halves
        + [("d2h", "und")] * 42
    )


def test_parameter_objects_are_not_replaced():
    model = transformer(with_lora=True)
    evictor = SenseNovaTrainingPhaseEvictor(model, "cpu")
    parameters = list(model.language_model.model.layers.parameters())
    identities = [id(parameter) for parameter in parameters]
    evictor.enter_prefix()
    evictor.enter_denoise()
    parameters[0].grad = torch.ones_like(parameters[0])
    evictor.assert_generation_resident()
    assert [id(parameter) for parameter in parameters] == identities
    evictor.enter_prefix()
    with pytest.raises(RuntimeError, match="denoise state"):
        evictor.assert_generation_resident()


class Staged(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.register_buffer("scale", torch.full((2,), 3.0))
        self.register_buffer("scratch", torch.zeros(1), persistent=False)


def test_staging_lands_on_cpu_and_skips_non_persistent_buffers():
    module = Staged()
    parameter = module.weight
    scratch = module.scratch
    warn_once = {}

    stage_modules_to_pinned_cpu((module,), warn_once=warn_once)

    assert module.weight is parameter
    assert torch.equal(module.weight.data, torch.arange(4, dtype=torch.float32))
    assert torch.equal(module.scale, torch.full((2,), 3.0))
    assert module.weight.data.device.type == "cpu"
    assert module.scale.device.type == "cpu"
    assert module.scratch is scratch
    if torch.cuda.is_available():
        assert module.weight.data.is_pinned() and module.scale.is_pinned()
        assert not module.scratch.is_pinned()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory needs CUDA")
def test_staging_reuses_an_already_pinned_tensor_without_copying():
    module = Staged()
    warn_once = {}
    stage_modules_to_pinned_cpu((module,), warn_once=warn_once)
    pointers = (module.weight.data.data_ptr(), module.scale.data_ptr())

    stage_modules_to_pinned_cpu((module,), warn_once=warn_once)

    assert (module.weight.data.data_ptr(), module.scale.data_ptr()) == pointers
    assert warn_once == {}


def test_staging_falls_back_to_pageable_when_pinned_allocation_fails(capsys):
    modules = [Staged(), Staged()]
    warn_once = {}

    def no_pinned_memory(*args, **kwargs):
        raise RuntimeError("no pinned allocator")

    with patch.object(mot_cpu_staging.torch, "empty_like", no_pinned_memory):
        stage_modules_to_pinned_cpu(modules, warn_once=warn_once)

    for module in modules:
        assert module.weight.data.device.type == "cpu"
        assert not module.weight.data.is_pinned()
        assert torch.equal(module.weight.data, torch.arange(4, dtype=torch.float32))
        assert torch.equal(module.scale, torch.full((2,), 3.0))
    assert warn_once == {"pin_failed": True}
    assert capsys.readouterr().out.count("no pinned allocator") == 1


def test_generation_and_training_evictors_share_one_staging_implementation():
    assert not hasattr(mot_phase_eviction, "_pin_module_cpu_")
    assert (
        mot_phase_eviction.stage_modules_to_pinned_cpu
        is mot_cpu_staging.stage_modules_to_pinned_cpu
    )
    module = Staged()
    warn_once = {}
    with patch.object(mot_cpu_staging, "_stage_tensor", return_value=torch.zeros(1)):
        sensenova_phase_eviction._move_modules_to_cpu((module,), warn_once=warn_once)
    assert torch.equal(module.scale, torch.zeros(1))


# --------------------------------------------------------------------------
# transfer accounting (SENSENOVA_TRAINING_DESIGN.md 8.6 had arithmetic only)
# --------------------------------------------------------------------------

# Every selected module of ``transformer()`` owns one persistent buffer of 2
# float32 -> 8 B, and each half holds 42 of them.
_MODULE_BYTES = 8
_HALF_BYTES = 42 * _MODULE_BYTES


def _relocate_to_cpu(modules, *, warn_once, pageable=False):
    """Stand-in for ``_move_modules_to_cpu`` that really moves meta tensors to
    CPU (the real one allocates pinned host memory, which needs CUDA)."""
    del warn_once, pageable
    for module in modules:
        for parameter in module._parameters.values():
            if parameter is not None:
                parameter.data = torch.zeros_like(parameter.data, device="cpu")
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and name not in module._non_persistent_buffers_set:
                module._buffers[name] = torch.zeros_like(buffer, device="cpu")


def _accounted():
    return patch(
        "core.training.sensenova_phase_eviction._move_modules_to_cpu", _relocate_to_cpu
    )


def test_transfer_byte_counters_match_the_moved_half():
    """MUTANT: charging every tensor instead of the ones the operation will
    actually copy makes the one-sided ``full -> prefix`` report a nonzero h2d.
    """
    evictor = SenseNovaTrainingPhaseEvictor(transformer(), "meta")
    with _accounted():
        evictor.enter_prefix()
        assert (evictor.d2h_bytes, evictor.h2d_bytes) == (_HALF_BYTES, 0)
        evictor.enter_denoise()

    assert evictor.d2h_bytes == 2 * _HALF_BYTES   # gen evicted, then und
    assert evictor.h2d_bytes == _HALF_BYTES       # gen brought back


def test_both_directions_are_timed_into_separate_buckets():
    """MUTANT: accumulating both directions into one bucket, or reusing the d2h
    elapsed for the h2d, leaves one of these two totals at zero."""
    evictor = SenseNovaTrainingPhaseEvictor(transformer(), "meta")
    with _accounted():
        evictor.enter_prefix()
        assert evictor.d2h_seconds > 0.0 and evictor.h2d_seconds == 0.0
        evictor.enter_denoise()

    assert evictor.d2h_seconds > 0.0
    assert evictor.h2d_seconds > 0.0
    assert evictor.d2h_seconds != evictor.h2d_seconds


def test_drain_resets_so_consecutive_steps_do_not_double_count():
    """MUTANT: a drain that reads without resetting turns the per-step series
    into a run-cumulative one -- the second step below would report both."""
    evictor = SenseNovaTrainingPhaseEvictor(transformer(), "meta")
    with _accounted():
        evictor.enter_prefix()
        evictor.enter_denoise()
        first = evictor.drain_transfer_stats()
        assert evictor.drain_transfer_stats() == {
            "d2h_seconds": 0.0, "h2d_seconds": 0.0, "d2h_bytes": 0, "h2d_bytes": 0
        }
        evictor.enter_prefix()      # denoise -> prefix, a full swap
        evictor.enter_denoise()
        second = evictor.drain_transfer_stats()

    assert first["d2h_bytes"] == 2 * _HALF_BYTES and first["h2d_bytes"] == _HALF_BYTES
    assert second["d2h_bytes"] == 2 * _HALF_BYTES
    assert second["h2d_bytes"] == 2 * _HALF_BYTES


def test_a_non_cuda_device_never_synchronizes():
    """MUTANT: dropping the device guard in ``_sync`` makes every synthetic-tree
    test raise on a machine without CUDA, and needlessly sync on one with it."""
    evictor = SenseNovaTrainingPhaseEvictor(transformer(), "meta")
    calls = []
    with _accounted(), patch(
        "torch.cuda.synchronize", lambda *a, **k: calls.append(a)
    ):
        evictor.enter_prefix()
        evictor.enter_denoise()

    assert evictor._sync_device is None
    assert calls == []


def test_transfer_metrics_are_registered_for_the_chart():
    from core.training.metric_registry import EXTRA_METRIC_DEFS

    for name in ("sn_d2h_s", "sn_h2d_s", "sn_d2h_gib", "sn_h2d_gib"):
        assert EXTRA_METRIC_DEFS[name]["axis"] == "right"


def test_partial_transfer_failure_is_fatal_and_normalized_to_cpu():
    evictor = SenseNovaTrainingPhaseEvictor(transformer(), "cpu")
    calls = 0

    def fail_once(modules, *, warn_once, pageable=False):
        nonlocal calls
        del modules, warn_once, pageable
        calls += 1
        if calls == 1:
            raise RuntimeError("copy failed")

    with patch("core.training.sensenova_phase_eviction._move_modules_to_cpu", fail_once):
        with pytest.raises(RuntimeError, match="copy failed"):
            evictor.enter_prefix()
        assert evictor.state == "failed"
        assert calls == 85
        with pytest.raises(RuntimeError, match="failed transfer state"):
            evictor.enter_prefix()
        evictor.teardown()
        assert evictor.state == "closed"
