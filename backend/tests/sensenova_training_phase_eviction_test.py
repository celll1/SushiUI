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

    def d2h_identity(modules, *, warn_once):
        del warn_once
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

    assert events[:7] == [
        ("d2h", "gen"),
        ("d2h", "und"),
        ("h2d", "gen"),
        ("d2h", "gen"),
        ("h2d", "und"),
        ("d2h", "und"),
        ("h2d", "gen"),
    ]
    assert events[7:] == [("d2h", "gen")] * 42 + [("d2h", "und")] * 42


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


def test_partial_transfer_failure_is_fatal_and_normalized_to_cpu():
    evictor = SenseNovaTrainingPhaseEvictor(transformer(), "cpu")
    calls = 0

    def fail_once(modules, *, warn_once):
        nonlocal calls
        del modules, warn_once
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
