"""``sensenova_mot_pageable_staging``: an opt-in escape hatch from the pinned
host pool's stickiness (see ``sensenova_phase_eviction``'s PAGEABLE STAGING
note). No CUDA is touched anywhere in this file -- every real call below is
CPU-only or is patched, exactly like ``sensenova_mot_staging_highwater_test``.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/sensenova_mot_pageable_staging_test.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.arch_capabilities import (
    TRAINING_DECLARED_ARCHS,
    TRAINING_FEATURE_PARAMS,
    training_feature_unsupported_reason,
)
from api.param_defaults import TRAINING_DEFAULTS
from core.models.sensenova import mot_cpu_staging
from core.models.sensenova.mot_cpu_staging import (
    _stage_tensor,
    stage_modules_to_pinned_cpu,
)
from core.training import sensenova_phase_eviction
from core.training.sensenova_phase_eviction import (
    SenseNovaTrainingPhaseEvictor,
    install_training_phase_eviction,
)
from core.training.train_runner import _apply_sensenova_training_contract

from sensenova_mot_staging_highwater_test import (  # noqa: E402
    _HALF_BYTES, _LARGEST_MODULE_BYTES, Ledger, _instrumented, _transformer,
)


# ---------------------------------------------------------------------------
# _stage_tensor / stage_modules_to_pinned_cpu, pageable=True (no CUDA)
# ---------------------------------------------------------------------------


def test_pageable_stage_never_attempts_a_pinned_allocation():
    """The opt-in path must not even TRY to pin -- unlike the failure-path
    fallback, which pins first and only falls back to pageable on exception."""
    calls = []
    real_empty_like = torch.empty_like

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_empty_like(*args, **kwargs)

    tensor = torch.ones(4, 4)
    with patch("torch.empty_like", side_effect=spy):
        staged = _stage_tensor(tensor, {}, "unused", pageable=True)

    assert calls == []
    assert staged.device.type == "cpu"
    assert not staged.is_pinned()


def test_pageable_stage_short_circuits_an_already_cpu_tensor():
    """Matches the pinned short-circuit's zero-copy promise: a tensor already
    staged for the CURRENT mode is returned with no additional host copy
    (``.detach()`` always returns a fresh Python tensor object, so identity is
    not the right check -- shared storage, i.e. no ``.to("cpu")`` copy, is)."""
    tensor = torch.ones(4, 4)
    once = _stage_tensor(tensor, {}, "unused", pageable=True)
    twice = _stage_tensor(once, {}, "unused", pageable=True)
    assert twice.data_ptr() == once.data_ptr()


def test_stage_modules_to_pinned_cpu_pageable_moves_params_and_buffers():
    class _Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4, 4))
            self.register_buffer("scale", torch.ones(1))
            self.register_buffer("cache", torch.ones(2), persistent=False)

    module = _Module()
    stage_modules_to_pinned_cpu((module,), warn_once={}, pageable=True)
    assert module.weight.data.device.type == "cpu" and not module.weight.data.is_pinned()
    assert module.scale.device.type == "cpu" and not module.scale.is_pinned()


def test_negative_control_default_still_pins_by_attempted_call():
    """The shipped path, unchanged: pageable defaults False, so the pinned
    allocation IS attempted (whether it succeeds depends on CUDA, which this
    test does not require -- it only checks the attempt is made)."""
    calls = []
    real_empty_like = torch.empty_like

    def spy(*args, **kwargs):
        calls.append(kwargs.get("pin_memory"))
        return real_empty_like(*args, **kwargs)

    tensor = torch.ones(4, 4)
    with patch("torch.empty_like", side_effect=spy):
        _stage_tensor(tensor, {}, "unused")  # pageable omitted -> False

    assert calls == [True]


# ---------------------------------------------------------------------------
# Evictor threading: pageable_staging -> stage_modules_to_pinned_cpu(pageable=)
# ---------------------------------------------------------------------------


def _spy_move_to_cpu(captured):
    def spy(modules, *, warn_once, pageable=False):
        del warn_once
        captured.append(pageable)
    return spy


def _noop_move_to_device(modules, device):
    del modules, device


def test_evictor_threads_pageable_flag_into_every_stage_call():
    """Patches the same two seams ``sensenova_mot_staging_highwater_test``'s
    ``_instrumented`` does (``_move_modules_to_cpu``/``_move_modules_to_device``
    on ``sensenova_phase_eviction`` itself), so no real transfer -- pinned,
    pageable, or a real device -- ever runs."""
    captured = []
    evictor = SenseNovaTrainingPhaseEvictor(
        _transformer(), "meta", pageable_staging=True
    )
    with patch.multiple(
        sensenova_phase_eviction,
        _move_modules_to_cpu=_spy_move_to_cpu(captured),
        _move_modules_to_device=_noop_move_to_device,
    ):
        evictor.enter_prefix()
        evictor.enter_denoise()

    assert captured  # at least one d2h ran
    assert all(pageable is True for pageable in captured)


def test_negative_control_default_evictor_never_requests_pageable():
    captured = []
    evictor = SenseNovaTrainingPhaseEvictor(_transformer(), "meta")
    assert evictor._pageable is False
    with patch.multiple(
        sensenova_phase_eviction,
        _move_modules_to_cpu=_spy_move_to_cpu(captured),
        _move_modules_to_device=_noop_move_to_device,
    ):
        evictor.enter_prefix()
        evictor.enter_denoise()

    assert captured
    assert all(pageable is False for pageable in captured)


def test_pageable_staging_does_not_change_the_byte_accounting():
    """Pageable is a staging-MODE change, not a memory-accounting change: the
    ledger (which tracks bytes/location, not pin state) must show the same
    peaks as the pinned default (sensenova_mot_staging_highwater_test)."""
    evictor = SenseNovaTrainingPhaseEvictor(
        _transformer(), "meta", pageable_staging=True
    )
    ledger, instrumentation = _instrumented(evictor)
    with instrumentation:
        evictor.enter_prefix()
        # Mirrors sensenova_mot_staging_highwater_test.py's
        # test_device_residency_never_holds_both_halves: the "full" -> "prefix"
        # eviction starts with both halves already resident, so its own peak
        # is not the invariant under test here; reset before the swap that is.
        ledger.device_peak = ledger.device
        evictor.enter_denoise()

    assert ledger.host_peak == _HALF_BYTES + _LARGEST_MODULE_BYTES == 133_928
    assert ledger.device_peak == _HALF_BYTES


def test_module_already_staged_cpu_treats_any_cpu_tensor_as_staged_under_pageable():
    """A fast path, not a correctness fix: under pageable mode ANY cpu tensor
    counts as already staged, so ``_best_effort_cpu`` can skip a module
    outright instead of re-entering ``_stage_tensor(pageable=True)`` per
    tensor. Checking ``is_pinned()`` here (the pinned-mode condition) would
    make a never-pinned cpu tensor look UNstaged and cause a redundant
    ``_move_modules_to_cpu`` call on every sweep -- wasted work, not a pin,
    since ``_stage_tensor(pageable=True)`` still would not pin it (see the
    end-to-end test below)."""
    from core.training.sensenova_phase_eviction import _module_already_staged_cpu

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4, 4))

    module = _M()
    assert _module_already_staged_cpu(module, pageable=True) is True
    assert _module_already_staged_cpu(module, pageable=False) is False  # never pinned


def test_best_effort_cpu_never_pins_anything_under_pageable_staging():
    """End to end, with NO mocking: a full ``_best_effort_cpu()`` pass over a
    pageable-staging evictor must not pin a single tensor. If the "already
    staged" check above regressed to the pinned-mode condition, this would
    call ``_move_modules_to_cpu`` (pageable=True) on every never-pinned
    tensor, which is harmless on its own -- ``_stage_tensor(pageable=True)``
    still would not pin it -- so this also stands as the negative control
    that no code path here ever reaches ``torch.empty_like(pin_memory=True)``
    while ``pageable=True``, keeping this test free of any CUDA dependency."""
    evictor = SenseNovaTrainingPhaseEvictor(
        _transformer(), "meta", pageable_staging=True
    )
    error = evictor._best_effort_cpu()
    assert error is None
    for module in (*evictor._gen_modules, *evictor._und_modules):
        for parameter in module._parameters.values():
            if parameter is not None:
                assert not parameter.data.is_pinned()


# ---------------------------------------------------------------------------
# install_training_phase_eviction reads trainer.config, not a new attribute
# ---------------------------------------------------------------------------


def test_install_reads_pageable_staging_off_trainer_config():
    trainer = type("T", (), {})()
    trainer.transformer = _transformer()
    trainer.device = "meta"
    trainer.sensenova_four_phase_eviction = False
    trainer.config = {"sensenova_mot_pageable_staging": True}

    evictor = install_training_phase_eviction(trainer)
    assert evictor._pageable is True


def test_install_defaults_to_false_with_no_config_attribute_at_all():
    trainer = type("T", (), {})()
    trainer.transformer = _transformer()
    trainer.device = "meta"
    trainer.sensenova_four_phase_eviction = False
    # No .config at all -- getattr(trainer, "config", {}) must not raise.

    evictor = install_training_phase_eviction(trainer)
    assert evictor._pageable is False


# ---------------------------------------------------------------------------
# The contract: refused without sensenova_mot_phase_eviction
# ---------------------------------------------------------------------------


def _sensenova():
    from core.model_loader import ModelLoader
    return patch.object(ModelLoader, "detect_model_type", return_value="sensenova")


def _config(**overrides):
    config = {
        "batch_size": 1,
        "optimizer": "adafactor",
        "gradient_accumulation_steps": 1,
        "use_ema": False,
        "num_optimizer_groups": 0,
        "blocks_to_swap": 0,
        "block_swap_h2d_only": False,
        "train_unet": True,
        "train_text_encoder": False,
        "sensenova_mot_phase_eviction": False,
        "sensenova_four_phase_eviction": False,
    }
    config.update(overrides)
    return config


def test_pageable_staging_is_refused_without_eviction():
    with _sensenova():
        with pytest.raises(ValueError, match="requires sensenova_mot_phase_eviction"):
            _apply_sensenova_training_contract(
                "model", "lora",
                _config(sensenova_mot_pageable_staging=True), {"sample": {}})


def test_pageable_staging_is_accepted_alongside_eviction():
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "lora",
            _config(sensenova_mot_phase_eviction=True,
                    sensenova_mot_pageable_staging=True), {"sample": {}})


def test_pageable_staging_defaults_off_and_does_not_arm_the_refusal():
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "lora", _config(), {"sample": {}})


# ---------------------------------------------------------------------------
# Capability table parity (CLAUDE.md: 4ab32d65's openapi-parity fix)
# ---------------------------------------------------------------------------


def test_the_flag_is_declared_as_its_own_feature_not_folded_into_mot_eviction():
    """sensenova_mot_eviction's list is pinned exactly by
    sensenova_four_phase_ui_exposure_test; this flag must not be added to it."""
    assert TRAINING_FEATURE_PARAMS["sensenova_mot_eviction"] == [
        "sensenova_mot_phase_eviction", "sensenova_four_phase_eviction",
        "sensenova_four_phase_shared_prefix",
        "sensenova_four_phase_grad_reduction"]
    assert TRAINING_FEATURE_PARAMS["sensenova_mot_pageable_staging"] == [
        "sensenova_mot_pageable_staging"]


def test_the_mechanism_is_declared_absent_everywhere_but_sensenova():
    for arch in sorted(TRAINING_DECLARED_ARCHS - {"sensenova"}):
        assert training_feature_unsupported_reason(
            arch, "sensenova_mot_pageable_staging"), arch
    assert training_feature_unsupported_reason(
        "sensenova", "sensenova_mot_pageable_staging") is None


def test_the_openapi_entry_matches_the_arch_capabilities_claim():
    """CLAUDE.md: a description claiming 'accepted and warned by every other
    architecture' with no matching arch_capabilities warning was the exact
    4ab32d65 regression."""
    import yaml

    repo = Path(__file__).resolve().parents[2]
    spec = yaml.safe_load((repo / "openapi.yaml").read_text(encoding="utf-8"))
    props = spec["components"]["schemas"]["TrainingRunCreateRequest"]["properties"]
    description = props["sensenova_mot_pageable_staging"]["description"]
    assert "Accepted and warned by every other architecture" in description
    assert TRAINING_FEATURE_PARAMS["sensenova_mot_pageable_staging"] == [
        "sensenova_mot_pageable_staging"]
    for arch in sorted(TRAINING_DECLARED_ARCHS - {"sensenova"}):
        assert training_feature_unsupported_reason(
            arch, "sensenova_mot_pageable_staging"), arch


def test_the_default_matches_across_param_defaults_and_pydantic():
    from api.routes import TrainingRunCreateRequest

    assert TRAINING_DEFAULTS["sensenova_mot_pageable_staging"] is False
    assert (
        TrainingRunCreateRequest.model_fields["sensenova_mot_pageable_staging"].default
        is False
    )
