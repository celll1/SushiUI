"""Pinned-host high-water of the SenseNova MoT half swap.

See ``sensenova_phase_eviction``'s TRANSFER ORDER for the mechanism (the
batched order holds both halves pinned at once; the interleave does not).
These tests measure that in bytes on synthetic modules with a ledger rather
than the real allocator (which needs CUDA).
"""

import sys
import weakref
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models.sensenova import mot_cpu_staging, mot_phase_eviction
from core.models.sensenova.mot_weight_selector import (
    MotWeightSelection,
    select_mot_weight_modules,
)
from core.training import sensenova_phase_eviction
from core.training.sensenova_phase_eviction import SenseNovaTrainingPhaseEvictor


# --------------------------------------------------------------------------
# synthetic tree
# --------------------------------------------------------------------------


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(16, 16))   # 1024 B
        self.register_buffer("scale", torch.ones(1))     #    4 B
        self.register_buffer("cache", torch.ones(8), persistent=False)


class _Int8Like(nn.Module):
    """The frozen half's shape: quantized weight + scale, no Parameter."""

    def __init__(self):
        super().__init__()
        self.register_buffer("qweight", torch.ones(16, 16, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(1))


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(32, 16))   # 2048 B


class _Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(16))       #   64 B


_MODULE_BYTES = {"attn": 1028, "mlp": 2048, "norm": 64}
_HALF_BYTES = 42 * sum(_MODULE_BYTES.values())           # 131,880 B
_LARGEST_MODULE_BYTES = max(_MODULE_BYTES.values())      #     2048 B (mlp)


class _Layer(nn.Module):
    def __init__(self, *, with_gen_lora=False):
        super().__init__()
        for suffix in ("", "_mot_gen"):
            setattr(self, "attn" + suffix, _Attn())
            setattr(self, "mlp" + suffix, _Mlp())
            setattr(self, "norm" + suffix, _Norm())
        if with_gen_lora:
            self.attn_mot_gen.lora_down = nn.Linear(16, 2, bias=False)
            self.attn_mot_gen.lora_up = nn.Linear(2, 16, bias=False)
        self.rotary_emb = nn.Module()
        self.rotary_emb.register_buffer("inv_freq", torch.ones(4), persistent=False)


def _transformer(*, count=42, with_gen_lora=False, layer_cls=None):
    root = nn.Module()
    root.language_model = nn.Module()
    root.language_model.model = nn.Module()
    make = (lambda: layer_cls()) if layer_cls else (
        lambda: _Layer(with_gen_lora=with_gen_lora)
    )
    root.language_model.model.layers = nn.ModuleList([make() for _ in range(count)])
    return root


# --------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------


def _owned(module):
    for name, parameter in module._parameters.items():
        if parameter is not None:
            yield (id(module), "p", name), parameter.data.numel() * parameter.data.element_size()
    for name, buffer in module._buffers.items():
        if buffer is None or name in module._non_persistent_buffers_set:
            continue
        yield (id(module), "b", name), buffer.numel() * buffer.element_size()


class Ledger:
    """Byte-exact model of the two staging primitives.

    ``d2h`` charges host bytes (the pinned destination is allocated and written)
    and only then releases the device bytes; ``h2d`` allocates on the device and
    only then drops the host tensor, which is what ``parameter.data = ...``
    does. Peaks are therefore counted at the physically correct moment.
    """

    def __init__(self, evictor):
        self.location = {}
        for module in (*evictor._gen_modules, *evictor._und_modules):
            for key, nbytes in _owned(module):
                self.location[key] = ("device", nbytes)
        self.host = 0
        self.device = sum(nbytes for _, nbytes in self.location.values())
        self.host_peak = self.host
        self.device_peak = self.device
        self.events = []

    def _mark(self):
        self.host_peak = max(self.host_peak, self.host)
        self.device_peak = max(self.device_peak, self.device)

    def d2h(self, modules, *, warn_once=None, warn_message=None):
        del warn_once, warn_message
        for module in modules:
            for key, nbytes in _owned(module):
                where, _ = self.location[key]
                if where == "host":
                    continue
                self.host += nbytes
                self._mark()
                self.device -= nbytes
                self.location[key] = ("host", nbytes)
        self.events.append(("d2h", len(tuple(modules))))

    def h2d(self, modules, device=None, **kwargs):
        del device, kwargs
        for module in modules:
            for key, nbytes in _owned(module):
                where, _ = self.location[key]
                if where == "device":
                    continue
                self.device += nbytes
                self._mark()
                self.host -= nbytes
                self.location[key] = ("device", nbytes)
        self.events.append(("h2d", len(tuple(modules))))


def _instrumented(evictor):
    ledger = Ledger(evictor)
    return ledger, patch.multiple(
        sensenova_phase_eviction,
        _move_modules_to_cpu=ledger.d2h,
        _move_modules_to_device=ledger.h2d,
    )


# --------------------------------------------------------------------------
# (A) the release point the whole defect rests on
# --------------------------------------------------------------------------


def test_the_model_holds_the_staged_host_tensor_until_the_h2d_reassignment():
    """The whole defect: a staged half keeps the model's only reference to its
    host allocation, and that reference is dropped by ``parameter.data = ...``
    in ``_move_modules_to_device`` -- not by the outgoing half's staging. So the
    batched order, which defers every h2d, has both halves referenced at once.

    Shown on a buffer-only module, which is not a contrivance: the frozen half
    is ``Int8Linear``, which owns no Parameter at all. Buffers are released by
    plain reassignment, so a weakref is an exact instrument here. On the
    Parameter side the release is ``param.data = ...`` rebinding the storage,
    which is not observable without a second real device.
    """
    incoming, outgoing = _Int8Like(), _Int8Like()
    warn_once = {}
    mot_cpu_staging.stage_modules_to_pinned_cpu((incoming,), warn_once=warn_once)
    staged = weakref.ref(incoming.qweight)

    # The outgoing half stages: the incoming half still holds its host tensor.
    mot_cpu_staging.stage_modules_to_pinned_cpu((outgoing,), warn_once=warn_once)
    assert staged() is not None
    assert incoming.qweight.device.type == "cpu"

    sensenova_phase_eviction._move_modules_to_device((incoming,), "meta")

    assert staged() is None  # released only here
    assert incoming.qweight.device.type == "meta"
    assert incoming.scale.device.type == "meta"


# --------------------------------------------------------------------------
# (E) negative control: the shipped batched order, in bytes
# --------------------------------------------------------------------------


def _shipped_batched_swap(evictor, evicted):
    """The order this file replaces: every d2h, then every h2d."""
    if evicted == "generation":
        return (
            ("d2h", evictor._gen_modules, "generation"),
            ("h2d", evictor._und_modules, "understanding"),
        )
    return (
        ("d2h", evictor._und_modules, "understanding"),
        ("h2d", evictor._gen_modules, "generation"),
    )


def test_negative_control_shipped_order_pins_two_halves_at_once():
    evictor = SenseNovaTrainingPhaseEvictor(_transformer(), "meta")
    ledger, instrumentation = _instrumented(evictor)
    with instrumentation:
        evictor._transition(evictor._evict_plan("generation"), "prefix")
        baseline = ledger.host_peak
        evictor._transition(_shipped_batched_swap(evictor, "understanding"), "denoise")

    assert baseline == _HALF_BYTES == 131_880
    assert ledger.host_peak == 2 * _HALF_BYTES == 263_760
    assert ledger.device_peak == 2 * _HALF_BYTES  # the pre-eviction "full" state


def test_interleaved_order_pins_one_half_plus_one_module():
    evictor = SenseNovaTrainingPhaseEvictor(_transformer(), "meta")
    ledger, instrumentation = _instrumented(evictor)
    with instrumentation:
        evictor.enter_prefix()
        evictor.enter_denoise()

    assert ledger.host_peak == _HALF_BYTES + _LARGEST_MODULE_BYTES == 133_928
    assert ledger.host_peak < 2 * _HALF_BYTES


def test_every_two_sided_transition_holds_one_half_plus_one_module():
    evictor = SenseNovaTrainingPhaseEvictor(_transformer(), "meta", four_phase=True)
    ledger, instrumentation = _instrumented(evictor)
    ceiling = _HALF_BYTES + _LARGEST_MODULE_BYTES
    with instrumentation:
        evictor.enter_prefix()                    # full -> prefix (one-sided)
        assert ledger.host_peak == _HALF_BYTES
        evictor.enter_denoise()                   # prefix -> denoise
        assert ledger.host_peak == ceiling
        evictor.enter_und_backward()              # denoise -> und_backward
        assert ledger.host_peak == ceiling
        before_no_op = len(ledger.events)
        evictor.enter_prefix()                    # und_backward -> prefix, no-op
        assert len(ledger.events) == before_no_op
        assert evictor.state == "prefix"
        evictor.enter_denoise()
        evictor.enter_prefix()                    # denoise -> prefix
        assert evictor.state == "prefix"

    assert ledger.host_peak == ceiling
    assert ledger.host == _HALF_BYTES  # exactly one half staged at rest


def test_teardown_normalizes_both_halves_to_cpu_at_once():
    """MAJOR-1: teardown's job is to release the GPU-resident half entirely, so
    it necessarily ends with BOTH halves pinned on CPU at once -- the peak the
    interleave otherwise avoids. ``_best_effort_cpu``'s already-pinned skip
    saves the wasted work of re-entering the half that was already there; it
    does not and cannot lower this number, because that half was never double
    counted in the ledger to begin with (see ``Ledger.d2h``'s own
    already-host check)."""
    evictor = SenseNovaTrainingPhaseEvictor(_transformer(), "meta", four_phase=True)
    ledger, instrumentation = _instrumented(evictor)
    with instrumentation:
        evictor.enter_prefix()
        evictor.enter_denoise()
        evictor.teardown()

    assert ledger.host_peak == 2 * _HALF_BYTES == 263_760
    assert ledger.host == 2 * _HALF_BYTES
    assert ledger.device == 0
    assert evictor.state == "closed"


def test_device_residency_never_holds_both_halves():
    """The invariant the all-d2h-first order guaranteed, kept by the pairwise
    d2h-before-h2d: the paired modules carry identical tensor signatures, so the
    incoming half grows on device exactly as fast as the outgoing one shrinks."""
    evictor = SenseNovaTrainingPhaseEvictor(_transformer(), "meta", four_phase=True)
    ledger, instrumentation = _instrumented(evictor)
    with instrumentation:
        evictor.enter_prefix()
        after_eviction = ledger.device
        ledger.device_peak = ledger.device
        evictor.enter_denoise()
        evictor.enter_und_backward()
        evictor.enter_prefix()
        evictor.enter_denoise()

    assert after_eviction == _HALF_BYTES
    assert ledger.device_peak == _HALF_BYTES  # not one byte over a single half


# --------------------------------------------------------------------------
# (A) what the selector actually guarantees
# --------------------------------------------------------------------------


class _DupLayer(nn.Module):
    """Passes the SET-based symmetry check with 2 generation modules against 1
    understanding module: ``dup.leaf_mot_gen`` and ``dup_mot_gen.leaf`` both
    normalize to ``dup.leaf``, and a set collapses them."""

    def __init__(self):
        super().__init__()
        self.dup = nn.Module()
        self.dup.leaf = _Norm()
        self.dup.leaf_mot_gen = _Norm()
        self.dup_mot_gen = nn.Module()
        self.dup_mot_gen.leaf = _Norm()


def test_set_symmetry_alone_does_not_imply_pairability():
    root = _transformer(layer_cls=_DupLayer)
    # No exception from the signature-set comparison itself...
    selection = select_mot_weight_modules(root)
    assert len(selection.gen_modules) == 84 and len(selection.und_modules) == 42
    assert selection.pairs == ()  # no guarantee without require_exact_symmetry


def test_unpairable_tree_fails_loudly_instead_of_zipping_by_position():
    with pytest.raises(RuntimeError, match="not pairable"):
        select_mot_weight_modules(
            _transformer(layer_cls=_DupLayer), require_exact_symmetry=True
        )
    with pytest.raises(RuntimeError, match="not pairable"):
        SenseNovaTrainingPhaseEvictor(_transformer(layer_cls=_DupLayer), "meta")


def test_pairs_match_by_signature_not_by_index():
    selection = select_mot_weight_modules(_transformer(), require_exact_symmetry=True)
    assert len(selection.pairs) == 42 * 3
    assert selection.gen_unpaired == () and selection.und_unpaired == ()
    for gen_module, und_module in selection.pairs:
        assert type(gen_module) is type(und_module)
        gen_sizes = sorted(nbytes for _, nbytes in _owned(gen_module))
        und_sizes = sorted(nbytes for _, nbytes in _owned(und_module))
        assert gen_sizes == und_sizes  # what makes the swap byte-neutral


def test_generation_adapters_are_declared_extras_not_pairing_failures():
    selection = select_mot_weight_modules(
        _transformer(with_gen_lora=True), require_exact_symmetry=True
    )
    assert len(selection.pairs) == 42 * 3
    assert len(selection.gen_unpaired) == 42 * 2
    assert selection.und_unpaired == ()

    evictor = SenseNovaTrainingPhaseEvictor(_transformer(with_gen_lora=True), "meta")
    plan = evictor._swap_plan("generation")
    assert [entry[0] for entry in plan[:84]] == ["d2h"] * 84  # outgoing extras first
    assert [entry[0] for entry in plan[84:]] == ["d2h", "h2d"] * (42 * 3)
    plan = evictor._swap_plan("understanding")
    assert [entry[0] for entry in plan[-84:]] == ["h2d"] * 84  # incoming extras last


def test_partial_pairing_is_refused_at_construction():
    real = select_mot_weight_modules(_transformer(), require_exact_symmetry=True)
    doctored = MotWeightSelection(
        real.gen_modules, real.und_modules, real.pairs[:-1], (), ()
    )
    with patch.object(
        sensenova_phase_eviction, "select_mot_weight_modules", return_value=doctored
    ):
        with pytest.raises(RuntimeError, match="cannot interleave"):
            SenseNovaTrainingPhaseEvictor(_transformer(), "meta")


# --------------------------------------------------------------------------
# (C) the four-phase pre-flight survives the interleave
# --------------------------------------------------------------------------


def test_grad_check_still_pre_flights_the_whole_outgoing_half():
    """A gradient on the LAST module of the half must stop the swap before the
    first module moves -- otherwise the interleave strands a half-done swap."""
    evictor = SenseNovaTrainingPhaseEvictor(_transformer(), "meta", four_phase=True)
    ledger, instrumentation = _instrumented(evictor)
    with instrumentation:
        evictor.enter_prefix()
        evictor.enter_denoise()
        evictor._gen_modules[-1].weight.grad = torch.zeros(16)
        moved = len(ledger.events)
        with pytest.raises(RuntimeError, match="still holds a gradient"):
            evictor.enter_und_backward()

    # The only transfers after the refusal are the failure path's best-effort
    # normalize-to-CPU (both halves, one module at a time). No h2d ran, so no
    # module of the incoming half was brought in against a half-done swap.
    # ``_best_effort_cpu``'s already-pinned skip does not shrink this count:
    # this harness mocks the transfer functions entirely, so the modules'
    # real tensors are never actually pinned and the skip never fires here.
    normalize = len(evictor._gen_modules) + len(evictor._und_modules)
    assert ledger.events[moved:] == [("d2h", 1)] * normalize
    assert evictor.state == "failed"


def test_three_state_evictor_takes_no_gradient_check_under_the_interleave():
    evictor = SenseNovaTrainingPhaseEvictor(_transformer(), "meta")
    ledger, instrumentation = _instrumented(evictor)
    with instrumentation:
        evictor.enter_prefix()
        evictor._und_modules[0].weight.grad = torch.zeros(16, 16)
        evictor.enter_denoise()
    assert evictor.state == "denoise"


# --------------------------------------------------------------------------
# (E) the inference-side evictor: does it share this path?
# --------------------------------------------------------------------------


def test_inference_evictor_shares_the_staging_module():
    assert (
        mot_phase_eviction.stage_modules_to_pinned_cpu
        is mot_cpu_staging.stage_modules_to_pinned_cpu
    )


def test_inference_evictor_still_pins_two_halves_at_once(capsys, monkeypatch):
    """RECORDED, NOT FIXED as a scoped design decision, not because it is
    impossible. ``MotPhaseEvictor.on_phase`` stages the whole understanding
    half before loading any of the generation half, and the generation half
    has been pinned since the prefix notification, so its host high-water is
    two halves for the same reason the training path's was. It builds its
    selection WITHOUT ``require_exact_symmetry`` today, so there is no pairing
    guarantee available to it -- but that guarantee could be requested with a
    ``try``/fallback to today's batched order on failure, which would be a
    strict improvement for inference with no new failure mode (the batched
    order already IS its status quo). That fallback is unacceptable for
    TRAINING, where it would silently restore the 2x peak this change removes
    -- and ``install()`` (mot_phase_eviction.py) already catches every
    construction exception and returns None, both halves left GPU-resident,
    which is why raising out of a failed pairing attempt is not an option
    here either."""
    monkeypatch.setattr("api.generation_status.add_warning", lambda *a, **k: None)
    root = _transformer()
    evictor = mot_phase_eviction.MotPhaseEvictor(root, "meta")
    capsys.readouterr()

    class _Shim:
        _gen_modules = evictor._gen_modules
        _und_modules = evictor._und_modules

    ledger = Ledger(_Shim())
    for module in evictor._gen_modules:
        module.to = lambda device, non_blocking=False, _m=module: ledger.h2d((_m,))

    with patch.object(mot_phase_eviction, "stage_modules_to_pinned_cpu", ledger.d2h):
        evictor.on_phase("prefix")
        evictor.on_phase("denoise")

    assert ledger.host_peak == 2 * _HALF_BYTES == 263_760
    assert ledger.host == _HALF_BYTES
