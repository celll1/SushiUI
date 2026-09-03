"""Cheap MiniMax-H3 inference-LoRA tests: 3-block stub, toy widths, no real files.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_lora_apply_cheap_test.py -v

The sibling ``minimax_h3_lora_conversion_test.py`` builds a 50-block stub at the
REAL hidden/inner sizes (tens of GB of host RAM) and needs the real checkpoints
outside the repo, so it cannot be run casually. This file needs neither: 3
blocks x 6 leaves = 18 modules at toy widths, well under a second, and it covers
the apply/refusal paths that file does not reach at all --

  * the native (SushiUI-trained) key round trip, driven through the REAL
    training adapter's save path. This is the headline regression: a
    self-trained LoRA used to normalise to ZERO targets, because the only key
    parser required the ``diffusion_model.`` prefix.
  * every refusal ``_load_lora_minimax_h3`` raises, and the warning code each
    one puts in the generation's ``warnings[]``.
  * alpha precedence on BOTH branches -- native consults file metadata, comfy
    deliberately does not (see the module docstring of ``minimax_h3_lora``).
"""

import os
import sys

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_h3 import minimax_h3_lora as lora_mod  # noqa: E402
from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin  # noqa: E402
from core.training.adapters.minimax_h3_adapter import (  # noqa: E402
    DEFAULT_MINIMAX_H3_SCOPE, MiniMaxH3LoRAAdapter, iter_minimax_h3_lora_targets,
)


# ---------------------------------------------------------------------------
# Stub transformer: the vendored module-tree NAMING the adapter walks
# (transformer_blocks.<i>.attn.{to_q,to_k,to_v,to_out.0} / ff.{net.0.proj,net.2}),
# at widths chosen to be tiny -- nothing here depends on the real sizes.
# ---------------------------------------------------------------------------

_HIDDEN = 16
_INNER = 8
_FFN = 24
_N_BLOCKS = 3
_N_TARGETS = _N_BLOCKS * 6


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(_HIDDEN, _INNER)
        self.to_k = nn.Linear(_HIDDEN, _INNER)
        self.to_v = nn.Linear(_HIDDEN, _INNER)
        self.to_out = nn.ModuleList([nn.Linear(_INNER, _HIDDEN)])


class _FFProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(_HIDDEN, _FFN)


class _FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([_FFProj(), nn.Identity(), nn.Linear(_FFN // 2, _HIDDEN)])


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()
        self.ff = _FF()


class _StubTransformer(nn.Module):
    def __init__(self, n_blocks=_N_BLOCKS):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])


class _StubTrainer:
    def __init__(self, transformer):
        self.transformer = transformer


class _StubBackend(MiniMaxH3Mixin):
    """Just enough of the pipeline manager for the LoRA load/unload path."""

    def __init__(self, transformer, variant="fl2va"):
        self.minimax_h3_components = {"transformer": transformer, "variant": variant}


def _expected_target_paths(n_blocks=_N_BLOCKS):
    """The module paths the TRAINING side targets, straight from its own walker."""
    return {
        path for path, _parent, _attr, _cur
        in iter_minimax_h3_lora_targets(_StubTransformer(n_blocks), DEFAULT_MINIMAX_H3_SCOPE)
    }


def _save_native_lora(tmp_path, name="cheap_native.safetensors", rank=4, alpha=8):
    """Write a LoRA through the REAL adapter save path. Returns (path, n_layers)."""
    adapter = MiniMaxH3LoRAAdapter(_StubTrainer(_StubTransformer()), lora_rank=rank,
                                   lora_alpha=alpha)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    with torch.no_grad():
        for layer in layers.values():
            # lora_up initialises to zeros; a round trip over all-zero tensors
            # would pass even if the halves were transposed or swapped.
            layer.lora_up.weight.normal_()
            layer.lora_down.weight.normal_()
    out = tmp_path / name
    adapter.save_checkpoint(layers, step=1, epoch=0, output_path=out)
    return str(out), count


@pytest.fixture
def captured_warnings(monkeypatch):
    """Intercept the generation-status channel the backend's warn() writes to."""
    import api.generation_status as status

    recorded = []
    monkeypatch.setattr(status, "add_warning",
                        lambda message, code=None: recorded.append((message, code)))
    return recorded


def _codes(recorded):
    return [code for _message, code in recorded]


# ---------------------------------------------------------------------------
# 1. Headline fix: a SushiUI-trained LoRA normalises to the adapter's own paths
# ---------------------------------------------------------------------------

def test_native_round_trip_matches_the_adapter_paths_exactly(tmp_path):
    path, count = _save_native_lora(tmp_path)
    assert count == _N_TARGETS

    raw, metadata = lora_mod.load_lora_safetensors(path)
    targets = lora_mod.normalise_lora_state_dict(raw, metadata)

    # Set EQUALITY, not "non-empty": the regression was 0 matched targets, and a
    # partial match would be just as wrong and much quieter.
    assert set(targets) == _expected_target_paths()
    for module_path, weights in targets.items():
        # alpha=8 / rank=4 written per key by the adapter.
        assert weights["scale_ratio"] == pytest.approx(2.0), module_path
        assert weights["down"].shape[0] == 4
        assert weights["up"].shape[1] == 4


def test_native_lora_applies_to_every_target_and_restores(tmp_path):
    path, _count = _save_native_lora(tmp_path)
    raw, metadata = lora_mod.load_lora_safetensors(path)
    targets = lora_mod.normalise_lora_state_dict(raw, metadata)

    transformer = _StubTransformer()
    originals, wrapped = {}, set()
    applied, missing = lora_mod.apply_lora_group(transformer, targets, 1.0, originals, wrapped)
    assert (applied, missing) == (_N_TARGETS, [])
    # scale is scale_ratio * strength, on the composite's sole branch.
    composite = transformer.transformer_blocks[0].attn.to_q
    assert composite.get_branch(composite.branch_names[0]).scale == pytest.approx(2.0)

    restored = lora_mod.restore_originals(transformer, originals, wrapped)
    assert restored == _N_TARGETS
    assert not wrapped
    assert isinstance(transformer.transformer_blocks[0].attn.to_q, nn.Linear)
    assert isinstance(transformer.transformer_blocks[0].attn.to_out[0], nn.Linear)


# ---------------------------------------------------------------------------
# 2. Refusals raised by _load_lora_minimax_h3, and their warning codes
# ---------------------------------------------------------------------------

def test_missing_file_refuses_and_warns(captured_warnings, monkeypatch):
    from core.extensions import lora_manager as lm

    monkeypatch.setattr(lm.lora_manager, "_resolve_lora_path", lambda p: None)
    backend = _StubBackend(_StubTransformer())
    with pytest.raises(FileNotFoundError):
        backend._load_lora_minimax_h3([{"path": "no_such_lora.safetensors"}], {})
    assert "lora_not_found" in _codes(captured_warnings)


def test_zero_matched_targets_refuses_and_warns(tmp_path, captured_warnings, monkeypatch):
    """A native LoRA for block indices this transformer does not have."""
    from core.extensions import lora_manager as lm

    raw = {}
    for leaf in ("attn_to_q", "ff_net_2"):
        stem = f"lora_unet_transformer_blocks_40_{leaf}"
        raw[f"{stem}.lora_down.weight"] = torch.randn(4, _HIDDEN)
        raw[f"{stem}.lora_up.weight"] = torch.randn(_HIDDEN, 4)
        raw[f"{stem}.alpha"] = torch.tensor(4.0)
    path = tmp_path / "far_block_lora.safetensors"
    save_file(raw, str(path), metadata={"model_type": "minimax_h3"})
    monkeypatch.setattr(lm.lora_manager, "_resolve_lora_path", lambda p: path)

    backend = _StubBackend(_StubTransformer())
    with pytest.raises(RuntimeError, match="0 of"):
        backend._load_lora_minimax_h3([{"path": str(path)}], {})
    assert "lora_incompatible" in _codes(captured_warnings)


def test_stacking_two_loras_on_the_same_targets_now_sums(tmp_path, captured_warnings,
                                                         monkeypatch):
    """MiniMax-H3 is on ``CompositeAdapterLayer``; the refusal this asserted is
    gone. The numerics of the stack live in
    ``minimax_h3_lora_roundtrip_cheap_test.py``."""
    from core.extensions import lora_manager as lm

    path, _count = _save_native_lora(tmp_path)
    monkeypatch.setattr(lm.lora_manager, "_resolve_lora_path", lambda p: path)

    transformer = _StubTransformer()
    backend = _StubBackend(transformer)
    applied = backend._load_lora_minimax_h3(
        [{"path": path, "strength": 1.0}, {"path": path, "strength": 1.0}], {})
    assert applied == 2 * _N_TARGETS
    assert transformer.transformer_blocks[0].attn.to_q.branch_names == (
        f"0:{os.path.basename(path)}", f"1:{os.path.basename(path)}")
    assert "lora_stacking_unsupported" not in _codes(captured_warnings)


def test_variant_mismatch_refusal_reaches_warnings(tmp_path, captured_warnings, monkeypatch):
    """fl2va/ref2va are indistinguishable by weights, so this refusal is the only
    guard -- and it must reach warnings[], not only the API error."""
    from core.extensions import lora_manager as lm

    raw = {
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": torch.randn(4, _HIDDEN),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": torch.randn(_INNER, 4),
    }
    path = tmp_path / "declared_fl2va.safetensors"
    save_file(raw, str(path), metadata={"base_model": "minimax_h3_fl2va_pruned_bf16"})
    monkeypatch.setattr(lm.lora_manager, "_resolve_lora_path", lambda p: path)

    backend = _StubBackend(_StubTransformer(), variant="ref2va")
    with pytest.raises(ValueError, match="Refusing to load this LoRA"):
        backend._load_lora_minimax_h3([{"path": str(path)}], {})
    assert "minimax_h3_lora_variant_mismatch" in _codes(captured_warnings)


def test_mixed_key_conventions_refuse():
    raw = {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.zeros(4, _HIDDEN),
        "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": torch.zeros(3 * _INNER, 4),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": torch.zeros(4, _HIDDEN),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": torch.zeros(_INNER, 4),
    }
    with pytest.raises(ValueError, match="mixes"):
        lora_mod.normalise_lora_state_dict(raw)


def test_native_stem_missing_a_half_raises():
    """An incomplete tensor group is a defect in a file this repo wrote, not
    something to drop quietly (module docstring, convention 2)."""
    raw = {
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": torch.zeros(4, _HIDDEN),
        "lora_unet_transformer_blocks_0_attn_to_q.alpha": torch.tensor(4.0),
    }
    with pytest.raises(ValueError, match="only one of lora_down/lora_up"):
        lora_mod.normalise_lora_state_dict(raw)


def test_unrecognised_native_leaf_raises():
    raw = {
        "lora_unet_transformer_blocks_0_attn_to_wat.lora_down.weight": torch.zeros(4, _HIDDEN),
        "lora_unet_transformer_blocks_0_attn_to_wat.lora_up.weight": torch.zeros(_INNER, 4),
    }
    with pytest.raises(ValueError, match="name no MiniMax-H3 LoRA target"):
        lora_mod.normalise_lora_state_dict(raw)


# ---------------------------------------------------------------------------
# 3. Alpha precedence: native consults file metadata, comfy must not
# ---------------------------------------------------------------------------

_RANK = 4


def _comfy_raw(with_alpha=None):
    raw = {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.ones(_RANK, _HIDDEN),
        "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": torch.ones(3 * _INNER, _RANK),
    }
    if with_alpha is not None:
        raw["diffusion_model.blocks.0.attn.qkv_proj.alpha"] = torch.tensor(float(with_alpha))
    return raw


def _native_raw(with_alpha=None):
    stem = "lora_unet_transformer_blocks_0_attn_to_q"
    raw = {
        f"{stem}.lora_down.weight": torch.ones(_RANK, _HIDDEN),
        f"{stem}.lora_up.weight": torch.ones(_INNER, _RANK),
    }
    if with_alpha is not None:
        raw[f"{stem}.alpha"] = torch.tensor(float(with_alpha))
    return raw


def _ratios(raw, metadata):
    return {v["scale_ratio"] for v in lora_mod.normalise_lora_state_dict(raw, metadata).values()}


@pytest.mark.parametrize("metadata", [
    {},
    {"lora_alpha": "8.0"},
    {"ss_network_alpha": "8.0"},
    {"ss_network_alpha": "Dynamic"},
])
def test_comfy_branch_ignores_file_metadata_alpha(metadata):
    """The real ``lightx2v_turbo_4step`` file carries no per-key alpha and bakes
    a flat multiplier into lora_B; honouring a numeric ss_network_alpha here
    would double-attenuate it. ``'Dynamic'`` is that file's literal value."""
    assert _ratios(_comfy_raw(), metadata) == {1.0}


def test_native_branch_uses_file_metadata_alpha_when_no_per_key_alpha():
    assert _ratios(_native_raw(), {}) == {1.0}
    assert _ratios(_native_raw(), {"lora_alpha": "8.0"}) == {2.0}
    assert _ratios(_native_raw(), {"ss_network_alpha": "8.0"}) == {2.0}
    # Non-numeric metadata alpha is not an alpha; fall through to rank.
    assert _ratios(_native_raw(), {"ss_network_alpha": "Dynamic"}) == {1.0}


def test_per_key_alpha_outranks_metadata_on_both_branches():
    assert _ratios(_comfy_raw(with_alpha=2.0), {"lora_alpha": "8.0"}) == {0.5}
    assert _ratios(_native_raw(with_alpha=2.0), {"lora_alpha": "8.0"}) == {0.5}


# ---------------------------------------------------------------------------
# 4. Numerical effect, restore identity and the model-reload guard
#    (this file's original three sections cover the key codec, the refusals and
#    alpha precedence; these are the assertions they do not reach)
# ---------------------------------------------------------------------------

_STRENGTH = 0.75


def _resolve(root, dotted):
    from core.training.adapters.minimax_h3_adapter import _resolve_leaf
    return _resolve_leaf(root, dotted)


def test_native_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    """alpha=8 / rank=4 -> 2.0, times strength. Only meaningful because
    ``_save_native_lora`` randomises lora_up out of its zero init."""
    path, _count = _save_native_lora(tmp_path)
    raw, metadata = lora_mod.load_lora_safetensors(path)
    targets = lora_mod.normalise_lora_state_dict(raw, metadata)

    transformer = _StubTransformer()
    originals, wrapped = {}, set()
    lora_mod.apply_lora_group(transformer, targets, _STRENGTH, originals, wrapped)

    for module_path, weights in targets.items():
        _parent, _attr, wrapper = _resolve(transformer, module_path)
        x = torch.randn(3, wrapper.original_module.in_features)
        base = wrapper.original_module(x)
        expected = base + 2.0 * _STRENGTH * (x @ weights["down"].T @ weights["up"].T)
        assert torch.allclose(wrapper(x), expected, atol=1e-5), module_path
        assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{module_path}: inert"


def test_restore_returns_the_identical_original_objects(tmp_path):
    path, _count = _save_native_lora(tmp_path)
    raw, metadata = lora_mod.load_lora_safetensors(path)
    targets = lora_mod.normalise_lora_state_dict(raw, metadata)

    transformer = _StubTransformer()
    before = {p: _resolve(transformer, p)[2] for p in _expected_target_paths()}
    originals, wrapped = {}, set()
    lora_mod.apply_lora_group(transformer, targets, 1.0, originals, wrapped)

    assert lora_mod.restore_originals(transformer, originals, wrapped) == _N_TARGETS
    for module_path, original in before.items():
        # id(), not tensor equality: a fresh Linear carrying the same weights
        # would pass an equality check and still have dropped every hook,
        # device placement and quantized buffer the real module carried.
        assert _resolve(transformer, module_path)[2] is original, module_path
    assert not wrapped
    assert lora_mod.restore_originals(transformer, originals, wrapped) == 0


def test_model_reload_never_splices_model_a_into_model_b(tmp_path, monkeypatch):
    from core.extensions import lora_manager as lm

    path, _count = _save_native_lora(tmp_path)
    monkeypatch.setattr(lm.lora_manager, "_resolve_lora_path", lambda p: path)

    model_a = _StubTransformer()
    backend = _StubBackend(model_a)
    backend._load_lora_minimax_h3([{"path": path, "strength": 1.0}], {})
    a_ids = ({id(m) for _n, m in model_a.named_modules()}
             | {id(m) for m in backend._minimax_h3_lora_original_modules.values()})

    model_b = _StubTransformer()
    b_ids_before = {id(m) for _n, m in model_b.named_modules()}
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    backend.minimax_h3_components = {"transformer": model_b, "variant": "fl2va"}
    assert backend._unload_lora_minimax_h3() == 0
    assert {id(m) for _n, m in model_b.named_modules()} == b_ids_before
