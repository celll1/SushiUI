"""MiniMax-H3: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``MiniMaxH3LoRAAdapter`` (injection + ``save_checkpoint``) over a
3-block CPU stub and the REAL ``MiniMaxH3Mixin._load_lora_minimax_h3``.

Common composite algebra is covered in
``adapter_lycoris_roundtrip_cheap_test.py``. Two MiniMax-H3-specific properties
remain pinned here:

  * the BRANCH stays a ``MiniMaxH3LoRALinearLayer``. This architecture's forward
    runs without ``torch.autocast`` and needs that subclass's per-call
    activation cast; the composite drives it through ``forward_delta`` and never
    tests its class.
  * the scale-defining pair is the FUSED qkv stem's ``alpha``/rank, not the
    branch's own tensor rank after a compact split, so the branch carries that
    pair and a restrengthening recomputes the same number.

Its sibling ``minimax_h3_lora_apply_cheap_test.py`` owns the key codec, the
refusals and alpha precedence; ``minimax_h3_lora_conversion_test.py`` builds a
50-block stub at the real widths (tens of GB of host RAM) and must not be run
casually or imitated.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_lora_roundtrip_cheap_test.py -v
"""

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from lora_roundtrip_common import warning_codes, warning_probe

from core.adapters import (  # noqa: E402
    CompositeAdapterLayer, MiniMaxH3LoRALinearLayer,
)
from core.models.minimax_h3 import minimax_h3_lora as lora_mod  # noqa: E402
from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin  # noqa: E402
from core.training.adapters.minimax_h3_adapter import (  # noqa: E402
    DEFAULT_MINIMAX_H3_SCOPE, MiniMaxH3LoRAAdapter, _resolve_leaf,
    iter_minimax_h3_lora_targets,
)

_HIDDEN = 16
_INNER = 8
_FFN = 24
_N_BLOCKS = 3
_N_TARGETS = _N_BLOCKS * 6

RANK = 4
# alpha/rank = 1.5 and strength 0.7 give scale 1.05. A scale of exactly 1.0 --
# which alpha 8 / rank 4 / strength 0.5 produces -- makes every plausible
# reassociation of the strength folding identical in IEEE754 and the
# bit-identity gate below vacuous.
ALPHA = 6
RATIO = ALPHA / RANK
STRENGTH = 0.7
STRENGTH_B = 0.4  # the second LoRA's, so a shared scale shows up as a wrong sum


def _linear(fan_in, fan_out):
    layer = nn.Linear(fan_in, fan_out)
    nn.init.normal_(layer.weight, std=0.05)
    nn.init.normal_(layer.bias, std=0.05)
    return layer


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = _linear(_HIDDEN, _INNER)
        self.to_k = _linear(_HIDDEN, _INNER)
        self.to_v = _linear(_HIDDEN, _INNER)
        self.to_out = nn.ModuleList([_linear(_INNER, _HIDDEN)])


class _FFProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _linear(_HIDDEN, _FFN)


class _FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([_FFProj(), nn.Identity(),
                                  _linear(_FFN // 2, _HIDDEN)])


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()
        self.ff = _FF()


class _StubModule(nn.Module):
    def __init__(self, n_blocks=_N_BLOCKS):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])


def _Stub(n_blocks=_N_BLOCKS, seed=7):
    """A stub with SEEDED base weights.

    Gates 2 and 3 compare two independently built models, so unseeded bases turn
    a bit-identity claim about the branch arithmetic into a claim about nothing.
    """
    torch.manual_seed(seed)
    return _StubModule(n_blocks)


class _StubTrainer:
    def __init__(self, transformer):
        self.transformer = transformer


class _Backend(MiniMaxH3Mixin):
    """Just enough of the pipeline manager for the LoRA load/unload path."""

    def __init__(self, transformer, variant="fl2va"):
        self.minimax_h3_components = {"transformer": transformer, "variant": variant}


def target_paths(n_blocks=_N_BLOCKS):
    """The module paths the TRAINING side targets, straight from its own walker."""
    return {path for path, _parent, _attr, _cur
            in iter_minimax_h3_lora_targets(_Stub(n_blocks), DEFAULT_MINIMAX_H3_SCOPE)}


def wrapped_paths(model):
    """Target paths a GENERATION load covers, i.e. the composite roots."""
    return {name for name, module in model.named_modules()
            if isinstance(module, CompositeAdapterLayer)}


def sole_branch(composite):
    assert len(composite) == 1, f"expected one branch, got {composite.branch_names}"
    return composite.get_branch(composite.branch_names[0])


def train_and_save(tmp_path, name="minimax_h3.safetensors", seed=1234):
    """Write a native LoRA through the REAL adapter save path."""
    adapter = MiniMaxH3LoRAAdapter(_StubTrainer(_Stub()), lora_rank=RANK,
                                   lora_alpha=ALPHA)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert count == _N_TARGETS
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in layers.values():
            # lora_up initialises to zeros; a round trip over all-zero tensors
            # would pass even if the halves were transposed or swapped.
            for weight in (layer.lora_down.weight, layer.lora_up.weight):
                weight.copy_(torch.randn(weight.shape, generator=generator) * 0.2)
    out = tmp_path / name
    adapter.save_checkpoint(layers, step=1, epoch=0, output_path=out)
    return str(out)


def file_targets(path):
    raw, metadata = lora_mod.load_lora_safetensors(path)
    return lora_mod.normalise_lora_state_dict(raw, metadata)


def resolve(model, dotted):
    return _resolve_leaf(model, dotted)[2]


def analytic_delta(weights, x, strength):
    down, up = weights["down"], weights["up"]
    return float(weights["scale_ratio"]) * strength * (x @ down.T @ up.T)


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


@pytest.fixture
def resolve_by_path(monkeypatch):
    """``_resolve_lora_path`` as identity, so several files can be selected."""
    from core.extensions import lora_manager as lm

    monkeypatch.setattr(lm.lora_manager, "_resolve_lora_path", lambda p: p)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def test_minimax_h3_generation_covers_exactly_the_targets_the_trainer_wrapped(
        tmp_path, resolve_by_path):
    path = train_and_save(tmp_path)
    trained = target_paths()

    model = _Stub()
    backend = _Backend(model)
    assert backend._load_lora_minimax_h3(
        [{"path": path, "strength": STRENGTH}], {}) == _N_TARGETS
    assert wrapped_paths(model) == trained
    assert backend._minimax_h3_lora_wrapped_keys == trained
    assert any(p.endswith(".attn.to_out.0") for p in trained)
    assert any(p.endswith(".ff.net.2") for p in trained)


def test_minimax_h3_branches_keep_the_architectures_own_layer_class(
        tmp_path, resolve_by_path):
    """The composite must not normalise the branch to the stock layer: this
    architecture's forward runs WITHOUT autocast and needs the per-call cast."""
    path = train_and_save(tmp_path)
    model = _Stub()
    _Backend(model)._load_lora_minimax_h3([{"path": path, "strength": STRENGTH}], {})

    for target in sorted(target_paths()):
        branch = sole_branch(resolve(model, target))
        assert type(branch) is MiniMaxH3LoRALinearLayer, target


# ---------------------------------------------------------------------------
# The fused-QKV scale survives runtime strength changes
# ---------------------------------------------------------------------------

def test_minimax_h3_restrengthening_a_branch_reproduces_the_checkpoints_ratio(
        tmp_path, resolve_by_path):
    """The branch carries the scale-DEFINING pair, so ``set_strength`` recomputes
    ``alpha / scale_rank * strength`` -- the same number the load folded in. With
    the branch's own tensor rank instead, a compact qkv split would silently
    lose the fused stem's ratio."""
    path = train_and_save(tmp_path)
    model = _Stub()
    _Backend(model)._load_lora_minimax_h3([{"path": path, "strength": STRENGTH}], {})

    for module_path, weights in file_targets(path).items():
        composite = resolve(model, module_path)
        name = composite.branch_names[0]
        assert composite.get_strength(name) == STRENGTH, module_path
        composite.set_strength(name, STRENGTH_B)
        assert composite.get_branch(name).scale == \
            float(weights["scale_ratio"]) * STRENGTH_B, module_path


# ---------------------------------------------------------------------------
# Gates 1-3: the stack
# ---------------------------------------------------------------------------

def test_minimax_h3_the_same_file_selected_twice_is_two_branches(tmp_path, resolve_by_path):
    """Branch names are per REQUEST INDEX, so a duplicate selection doubles the
    delta instead of raising a duplicate-name error."""
    path = train_and_save(tmp_path)
    targets = file_targets(path)

    model = _Stub()
    _Backend(model)._load_lora_minimax_h3(
        [{"path": path, "strength": STRENGTH}, {"path": path, "strength": STRENGTH}], {})

    for module_path, weights in targets.items():
        composite = resolve(model, module_path)
        assert len(composite) == 2, composite.branch_names
        base = composite.original_module
        x = torch.randn(3, base.in_features)
        expected = base(x) + 2 * analytic_delta(weights, x, STRENGTH)
        assert torch.allclose(composite(x), expected, atol=1e-5), module_path


def test_minimax_h3_a_comfy_lora_stacks_onto_a_native_one(tmp_path, resolve_by_path,
                                                          warnings_seen):
    """The two key conventions resolve to the same vendored targets, so a fused
    ComfyUI qkv file must add branches beside a native file's."""
    native = train_and_save(tmp_path)
    raw = {}
    generator = torch.Generator().manual_seed(99)
    for block in range(_N_BLOCKS):
        stem = f"diffusion_model.blocks.{block}.attn.qkv_proj"
        raw[f"{stem}.lora_A.weight"] = torch.randn(RANK, _HIDDEN, generator=generator) * 0.2
        raw[f"{stem}.lora_B.weight"] = torch.randn(3 * _INNER, RANK, generator=generator) * 0.2
    comfy = tmp_path / "comfy.safetensors"
    save_file(raw, str(comfy))

    model = _Stub()
    _Backend(model)._load_lora_minimax_h3(
        [{"path": native, "strength": STRENGTH},
         {"path": str(comfy), "strength": STRENGTH_B}], {})

    qkv = {p for p in target_paths()
           if p.endswith((".attn.to_q", ".attn.to_k", ".attn.to_v"))}
    for module_path in sorted(target_paths()):
        composite = resolve(model, module_path)
        assert len(composite) == (2 if module_path in qkv else 1), module_path
    assert "lora_stacking_unsupported" not in warning_codes(warnings_seen)


# ---------------------------------------------------------------------------
# Gate 5: restore identity
# ---------------------------------------------------------------------------

def test_minimax_h3_a_leaked_wrapper_is_restored_before_the_next_load(
        tmp_path, resolve_by_path):
    """The load restores unconditionally at its top: without that, a composite
    that outlived its request would now SUM into the next one instead of being
    caught by the stacking refusal.

    Two leaks, both reachable: a generation killed between the wrap and its
    ``finally`` (no unload at all), and an unload whose restore RAISED, which
    leaves the wrappers installed with the bookkeeping still naming them.
    """
    path = train_and_save(tmp_path)

    reference = _Stub()
    _Backend(reference)._load_lora_minimax_h3([{"path": path, "strength": STRENGTH}], {})

    def check(backend, model):
        backend._load_lora_minimax_h3([{"path": path, "strength": STRENGTH}], {})
        for module_path in sorted(target_paths()):
            leaked = resolve(model, module_path)
            assert len(leaked) == 1, f"{module_path}: {leaked.branch_names}"
            x = torch.randn(3, leaked.original_module.in_features)
            assert torch.equal(leaked(x), resolve(reference, module_path)(x)), module_path

    no_finally = _Stub()
    backend = _Backend(no_finally)
    backend._load_lora_minimax_h3([{"path": path, "strength": STRENGTH}], {})
    check(backend, no_finally)

    failed_restore = _Stub()
    backend = _Backend(failed_restore)
    backend._load_lora_minimax_h3([{"path": path, "strength": STRENGTH}], {})

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    real_unload = backend._minimax_h3_lora_session.unload
    backend._minimax_h3_lora_session.unload = _boom
    try:
        with pytest.raises(RuntimeError, match="boom"):
            backend._unload_lora_minimax_h3()
    finally:
        backend._minimax_h3_lora_session.unload = real_unload
    check(backend, failed_restore)


def test_minimax_h3_two_loras_over_one_module_are_no_longer_refused(
        tmp_path, resolve_by_path, warnings_seen):
    path = train_and_save(tmp_path)
    model = _Stub()
    applied = _Backend(model)._load_lora_minimax_h3(
        [{"path": path, "strength": STRENGTH},
         {"path": path, "strength": STRENGTH_B}], {})
    assert applied == 2 * _N_TARGETS
    assert "lora_stacking_unsupported" not in warning_codes(warnings_seen)


def test_minimax_h3_unmatched_target_is_refused_atomically(tmp_path, resolve_by_path,
                                                         warnings_seen):
    from core.adapters import AdapterIncompatible

    path = train_and_save(tmp_path)
    saved = load_file(path)
    ghost = "lora_unet_transformer_blocks_40_attn_to_q"
    saved[f"{ghost}.lora_down.weight"] = torch.randn(RANK, _HIDDEN)
    saved[f"{ghost}.lora_up.weight"] = torch.randn(_INNER, RANK)
    extended = tmp_path / "extended.safetensors"
    save_file(saved, str(extended), metadata={"model_type": "minimax_h3"})

    model = _Stub()
    with pytest.raises(AdapterIncompatible) as excinfo:
        _Backend(model)._load_lora_minimax_h3(
            [{"path": str(extended), "strength": STRENGTH}], {})
    assert excinfo.value.code == "lora_partial"
    assert not wrapped_paths(model)
    assert "lora_partial" in warning_codes(warnings_seen)


# ---------------------------------------------------------------------------
# Quantizer reach, and block swap
# ---------------------------------------------------------------------------

def test_minimax_h3_wrapper_roots_are_countable_but_no_quantizer_reads_them(
        tmp_path, resolve_by_path):
    """Adoption makes ``lora_wrapped_count`` non-zero for this architecture for
    the first time (``MiniMaxH3LoRALinearLayer`` is deliberately absent from
    ``_ADAPTER_WRAPPER_CLASS_NAMES``). Both readers of that count are behind
    ``RUNTIME_INT8_ARCHS``, which MiniMax-H3 is not in, so nothing new refuses --
    and this pins that pair of facts together, since gaining the count without
    gaining a consumer is what makes the adoption safe here."""
    from core.models.common.int8_runtime_quantize import (
        RUNTIME_INT8_ARCHS, lora_wrapped_count,
    )

    assert "minimax_h3" not in RUNTIME_INT8_ARCHS

    path_a = train_and_save(tmp_path, seed=1234)
    path_b = train_and_save(tmp_path, name="second.safetensors", seed=4321)
    model = _Stub()
    assert lora_wrapped_count(model) == 0

    backend = _Backend(model)
    backend._load_lora_minimax_h3([{"path": path_a, "strength": STRENGTH},
                                   {"path": path_b, "strength": STRENGTH_B}], {})
    # ROOTS, not branches: one hidden slot per target however many branches.
    assert lora_wrapped_count(model) == _N_TARGETS
    assert sum(len(m) for m in model.modules()
               if isinstance(m, CompositeAdapterLayer)) == 2 * _N_TARGETS

    backend._unload_lora_minimax_h3()
    assert lora_wrapped_count(model) == 0


def test_minimax_h3_block_swap_sees_one_base_per_target_and_a_uniform_rename(
        tmp_path, resolve_by_path):
    """``TransformerBlockOffloader`` selects by ``endswith("Linear")`` plus a
    non-None weight and pairs blocks by module path. Neither the composite nor
    the branch class ends in ``Linear``, so the base is enrolled ONCE, at the
    same path the old wrapper put it; the branch weights are new paths, and what
    has to hold is that the per-block path set stays identical ACROSS blocks."""
    from core.memory_management.block_offloading import linear_weight_dtypes

    path_a = train_and_save(tmp_path, seed=1234)
    path_b = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    model = _Stub()
    backend = _Backend(model)
    blocks = model.transformer_blocks
    bare = [set(linear_weight_dtypes(b)) for b in blocks]
    assert len({frozenset(s) for s in bare}) == 1

    for configs, per_target in (([{"path": path_a, "strength": STRENGTH}], 1),
                                ([{"path": path_a, "strength": STRENGTH},
                                  {"path": path_b, "strength": STRENGTH_B}], 2)):
        backend._load_lora_minimax_h3(configs, {})
        sets = [set(linear_weight_dtypes(b)) for b in blocks]
        assert len({frozenset(s) for s in sets}) == 1, "blocks stopped pairing"
        for module_path in sorted(target_paths()):
            block_idx, rest = module_path.split(".")[1], module_path.split(".", 2)[2]
            assert f"{rest}.original_module" in sets[int(block_idx)]
            assert rest not in sets[int(block_idx)]
        assert len(sets[0]) == len(bare[0]) * (1 + 2 * per_target)
        seen = [id(m) for b in blocks for _n, m in b.named_modules()
                if m.__class__.__name__.endswith("Linear")
                and getattr(m, "weight", None) is not None]
        assert len(seen) == len(set(seen)), "a module is enrolled at two paths"

    backend._unload_lora_minimax_h3()
    assert [set(linear_weight_dtypes(b)) for b in blocks] == bare
