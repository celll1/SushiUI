"""MiniMax-H3 inference LoRA: the comfy->vendored conversion is a contract.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_lora_conversion_test.py -v

Exercises ``core.models.minimax_h3.minimax_h3_lora`` against two REAL LoRA
checkpoints (outside the repo, read-only, never moved to GPU):

    M:/model/minimax_h3/loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors
    M:/model/minimax_h3/loras/minimax_h3_fl2va_4step_lora.safetensors

Both files are skipped (not failed) when absent, since they live outside the
repository.

WHAT THIS FILE CANNOT CHECK
----------------------------
None of this runs the real 33B transformer or a GPU forward pass -- that is
explicitly out of scope (no GPU in this environment). The module-tree
resolution is checked against a STUB transformer built with the exact
in/out feature shapes the LoRA files themselves declare (the shapes are read
straight off the real, unquantized weights the loaded checkpoint would have
at those layers -- fc1 in=5376/out=28672, qkv in=5376/out=7168*3, etc. --
not invented), so path resolution and shape-matching are checked against the
real vendored module tree structure (``_resolve_leaf``,
``transformer_blocks``/``token_refiner.refiner_blocks``/``norm_out``/
``proj_out``/``audio_proj_out`` naming) without instantiating the real model.
"""

import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_h3 import minimax_h3_lora as lora_mod  # noqa: E402
from core.training.adapters.minimax_h3_adapter import _resolve_leaf  # noqa: E402


LORA_DIR = "M:/model/minimax_h3/loras"
F1_PATH = os.path.join(LORA_DIR, "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors")
F2_PATH = os.path.join(LORA_DIR, "minimax_h3_fl2va_4step_lora.safetensors")


def _require(path):
    if not os.path.isfile(path):
        pytest.skip(f"real LoRA file not present outside the repo: {path}")


# ---------------------------------------------------------------------------
# Stub transformer: exact vendored module-tree shape/naming, no real weights.
# ---------------------------------------------------------------------------

_HIDDEN = 5376
_INNER = 7168          # per-projection q/k/v output dim
_FFN_IN = 28672         # ff.net.0.proj out_features (SwiGLU, 2 * ffn_dim)
_FFN_OUT_IN = 14336      # ff.net.2 in_features (= ffn_dim)
_ADALN_IN = 8            # pruned/curve-variant time embed dim (per-block AdaLN)
_ADALN_OUT = 96768        # per-block adaln_proj.linear out
_NORM_OUT_OUT = 10752      # final_layer norm_out.linear out
_VIDEO_OUT = 96           # proj_out out
_AUDIO_OUT = 32           # audio_proj_out out


class _StubAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(_HIDDEN, _INNER, bias=False, dtype=torch.bfloat16)
        self.to_k = nn.Linear(_HIDDEN, _INNER, bias=False, dtype=torch.bfloat16)
        self.to_v = nn.Linear(_HIDDEN, _INNER, bias=False, dtype=torch.bfloat16)
        self.to_out = nn.ModuleList([nn.Linear(_INNER, _HIDDEN, bias=False, dtype=torch.bfloat16), nn.Dropout(0.0)])


class _StubFF(nn.Module):
    class _Proj(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(_HIDDEN, _FFN_IN, bias=False, dtype=torch.bfloat16)

    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([
            self._Proj(),
            nn.Dropout(0.0),
            nn.Linear(_FFN_OUT_IN, _HIDDEN, bias=False, dtype=torch.bfloat16),
        ])


class _StubAdaLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(_ADALN_IN, _ADALN_OUT, bias=True, dtype=torch.float32)


class _StubBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _StubAttn()
        self.ff = _StubFF()
        self.adaln_proj = _StubAdaLN()


class _StubNormOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(_ADALN_IN, _NORM_OUT_OUT, bias=True, dtype=torch.float32)


class _StubTokenRefiner(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.refiner_blocks = nn.ModuleList([_StubBlock() for _ in range(n)])


class _StubTransformer(nn.Module):
    """Vendored ``MiniMaxH3Transformer3DModel``'s naming, shapes only."""

    def __init__(self, num_blocks=50, num_refiner_blocks=2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_StubBlock() for _ in range(num_blocks)])
        self.token_refiner = _StubTokenRefiner(num_refiner_blocks)
        self.norm_out = _StubNormOut()
        self.proj_out = nn.Linear(_HIDDEN, _VIDEO_OUT, bias=True, dtype=torch.float32)
        self.audio_proj_out = nn.Linear(_HIDDEN, _AUDIO_OUT, bias=True, dtype=torch.float32)


def _make_stub():
    return _StubTransformer(num_blocks=50, num_refiner_blocks=2)


# ---------------------------------------------------------------------------
# 1. Target resolution + shape agreement against the stub module tree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [F1_PATH, F2_PATH])
def test_targets_resolve_and_shapes_match(path):
    _require(path)
    raw, metadata = lora_mod.load_lora_safetensors(path)
    targets = lora_mod.normalise_lora_state_dict(raw)
    assert targets, "no LoRA targets converted from a nonempty file"

    stub = _make_stub()
    unresolved = []
    shape_mismatches = []
    for module_path, weights in targets.items():
        resolved = _resolve_leaf(stub, module_path)
        if resolved is None:
            unresolved.append(module_path)
            continue
        _parent, _attr, module = resolved
        down, up = weights["down"], weights["up"]
        if down.shape[1] != module.in_features:
            shape_mismatches.append((module_path, "down", tuple(down.shape), module.in_features))
        if up.shape[0] != module.out_features:
            shape_mismatches.append((module_path, "up", tuple(up.shape), module.out_features))

    assert not unresolved, f"targets not present in the vendored module tree: {unresolved[:10]}"
    assert not shape_mismatches, f"down/up shape does not match target in/out_features: {shape_mismatches[:10]}"


@pytest.mark.parametrize("path", [F1_PATH, F2_PATH])
def test_apply_and_restore_round_trip(path):
    _require(path)
    raw, _metadata = lora_mod.load_lora_safetensors(path)
    targets = lora_mod.normalise_lora_state_dict(raw)

    stub = _make_stub()
    original_modules = {}
    for module_path in targets:
        _, _, module = _resolve_leaf(stub, module_path)
        original_modules[module_path] = module

    lora_originals = {}
    wrapped_keys = set()
    applied, missing = lora_mod.apply_lora_group(stub, targets, 1.0, lora_originals, wrapped_keys)
    assert not missing, f"apply_lora_group could not resolve: {missing[:10]}"
    assert applied == len(targets)

    for module_path in targets:
        _, _, module = _resolve_leaf(stub, module_path)
        assert module.__class__.__name__ == "MiniMaxH3LoRALinearLayer", module_path

    restored = lora_mod.restore_originals(stub, lora_originals, wrapped_keys)
    assert restored == len(targets)
    for module_path, original in original_modules.items():
        _, _, module = _resolve_leaf(stub, module_path)
        assert module is original, f"{module_path} did not restore to its true original"


# ---------------------------------------------------------------------------
# 2. qkv split: coverage/rank consistency + numeric exactness, both real
#    checkpoints -- F1's is block-diagonal (compact path), F2's is dense
#    (general shared-down fallback path); neither is a guess.
# ---------------------------------------------------------------------------

def _raw_qkv_stems(raw):
    stems = set()
    for key in raw:
        if key.startswith("diffusion_model.") and key.endswith(".attn.qkv_proj.lora_A.weight"):
            stems.add(key[len("diffusion_model."):-len(".lora_A.weight")])
    return sorted(stems)


@pytest.mark.parametrize("path", [F1_PATH, F2_PATH])
def test_qkv_split_ranks_are_internally_consistent(path):
    _require(path)
    raw, _metadata = lora_mod.load_lora_safetensors(path)
    stems = _raw_qkv_stems(raw)
    assert stems, "no fused qkv LoRA stems found"

    for stem in stems:
        down = raw[f"diffusion_model.{stem}.lora_A.weight"]
        up = raw[f"diffusion_model.{stem}.lora_B.weight"]
        parts, rank_total = lora_mod._split_qkv(stem, down, up)
        assert set(parts.keys()) == {"to_q", "to_k", "to_v"}
        assert rank_total == down.shape[0]
        for name, (d, u) in parts.items():
            assert d.shape[1] == _HIDDEN, (stem, name)
            assert u.shape[1] == d.shape[0], (stem, name, "down/up rank mismatch")
            assert u.shape[0] == _INNER, (stem, name)


@pytest.mark.parametrize("path", [F1_PATH, F2_PATH])
def test_qkv_split_reconstructs_exactly(path):
    """B @ A of the fused stem == vertical stack of the three split products,
    in the file's own dtype (no cast). Verifies the split loses nothing,
    whether it took the compact (block-diagonal) or the general (shared-down)
    path."""
    _require(path)
    raw, _metadata = lora_mod.load_lora_safetensors(path)
    stems = _raw_qkv_stems(raw)
    stem = stems[len(stems) // 2]  # a middle block, not the (often-degenerate) first

    down = raw[f"diffusion_model.{stem}.lora_A.weight"]
    up = raw[f"diffusion_model.{stem}.lora_B.weight"]
    fused = (up.to(torch.float64) @ down.to(torch.float64))

    parts, _rank_total = lora_mod._split_qkv(stem, down, up)
    split_stack = torch.cat(
        [(u.to(torch.float64) @ d.to(torch.float64)) for d, u in
         (parts["to_q"], parts["to_k"], parts["to_v"])],
        dim=0,
    )
    assert torch.allclose(fused, split_stack, atol=1e-6, rtol=1e-6)


def test_qkv_split_takes_the_compact_path_when_block_diagonal():
    """A synthetic, genuinely block-diagonal lora_B takes the COMPACT path:
    each component's down gets only its own (smaller) rank slice."""
    r = 6
    down = torch.randn(r, _HIDDEN)
    up = torch.zeros(3 * _INNER, r)
    up[:_INNER, 0:2] = torch.randn(_INNER, 2)
    up[_INNER:2 * _INNER, 2:4] = torch.randn(_INNER, 2)
    up[2 * _INNER:, 4:6] = torch.randn(_INNER, 2)
    parts, rank_total = lora_mod._split_qkv("fake.attn.qkv_proj", down, up)
    assert rank_total == r
    assert parts["to_q"][0].shape[0] == 2
    assert parts["to_k"][0].shape[0] == 2
    assert parts["to_v"][0].shape[0] == 2
    total_split_rank = sum(d.shape[0] for d, _u in parts.values())
    assert total_split_rank == r, "block-diagonal input must take the compact (not the 3r) path"


def test_qkv_split_falls_back_to_shared_down_when_not_block_diagonal():
    """A synthetic, deliberately NON-block-diagonal (dense) lora_B falls back
    to the general shared-``down`` split -- never refused, never guessed as
    an equal r // 3 partition -- and the reconstruction is still exact."""
    r = 6
    down = torch.randn(r, _HIDDEN, dtype=torch.float64)
    up = torch.zeros(3 * _INNER, r, dtype=torch.float64)
    # Overlapping / dense: to_q's active columns [0,4) overlap to_k's [2,6).
    up[:_INNER, 0:4] = torch.randn(_INNER, 4, dtype=torch.float64)
    up[_INNER:2 * _INNER, 2:6] = torch.randn(_INNER, 4, dtype=torch.float64)
    up[2 * _INNER:, 4:6] = torch.randn(_INNER, 2, dtype=torch.float64)

    parts, rank_total = lora_mod._split_qkv("fake.attn.qkv_proj", down, up)
    assert rank_total == r
    # The fallback path keeps the FULL rank for every component (nothing to
    # compact when the columns are not disjoint).
    for name in ("to_q", "to_k", "to_v"):
        assert parts[name][0].shape[0] == r, name

    fused = up @ down
    split_stack = torch.cat(
        [u @ d for d, u in (parts["to_q"], parts["to_k"], parts["to_v"])], dim=0,
    )
    assert torch.equal(fused, split_stack)


# ---------------------------------------------------------------------------
# 3. fc1 SwiGLU half swap: applied, and a negative control for reversal/removal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [F1_PATH, F2_PATH])
def test_fc1_swap_applied_and_reversible(path):
    _require(path)
    raw, _metadata = lora_mod.load_lora_safetensors(path)
    fc1_keys = [k for k in raw if k.endswith(".mlp.fc1.lora_B.weight")]
    assert fc1_keys

    raw_up = raw[fc1_keys[0]]
    converted = lora_mod._swap_fc1_halves(raw_up)

    # Negative control: a swap that was REMOVED (identity) or produced garbage
    # both fail this. Real gate/up halves of a trained LoRA are essentially
    # never bit-identical to each other, so "the swap changed nothing" is a
    # meaningful failure signal, not a coincidence.
    assert not torch.equal(converted, raw_up), (
        "fc1 half-swap produced no change -- looks like the swap was removed"
    )
    # Negative control for a REVERSED/incorrect permutation: swap is its own
    # exact inverse (chunk+cat of two halves), so swapping twice must recover
    # the original bit-for-bit. Any other row permutation would not.
    assert torch.equal(lora_mod._swap_fc1_halves(converted), raw_up), (
        "fc1 half-swap is not its own inverse -- the permutation is wrong"
    )

    # And end-to-end: the converted target's `up` for this stem is the SAME
    # swapped tensor the direct helper produces, and never the original.
    targets = lora_mod.normalise_lora_state_dict(raw)
    stem = fc1_keys[0][len("diffusion_model."):-len(".lora_B.weight")]
    from core.models.minimax_h3.loader import _rename_dit_key
    mapped = _rename_dit_key(stem + ".weight")[: -len(".weight")]
    assert torch.equal(targets[mapped]["up"], converted)
    assert not torch.equal(targets[mapped]["up"], raw_up)


# ---------------------------------------------------------------------------
# 4. Scale: F1 (no alpha) -> 1.0; F2 (real alphas) -> 1.0
# ---------------------------------------------------------------------------

def test_f1_no_alpha_resolves_to_scale_one():
    _require(F1_PATH)
    raw, metadata = lora_mod.load_lora_safetensors(F1_PATH)
    assert not any(k.endswith(".alpha") for k in raw), "F1 is expected to carry no alpha keys"
    targets = lora_mod.normalise_lora_state_dict(raw)
    assert targets
    for module_path, weights in targets.items():
        assert weights["scale_ratio"] == pytest.approx(1.0), (module_path, weights["scale_ratio"])


def test_f2_real_alphas_resolve_to_scale_one():
    _require(F2_PATH)
    raw, metadata = lora_mod.load_lora_safetensors(F2_PATH)
    assert any(k.endswith(".alpha") for k in raw), "F2 is expected to carry real alpha keys"
    targets = lora_mod.normalise_lora_state_dict(raw)
    assert targets
    for module_path, weights in targets.items():
        assert weights["scale_ratio"] == pytest.approx(1.0), (module_path, weights["scale_ratio"])


# ---------------------------------------------------------------------------
# 5. Variant guard
# ---------------------------------------------------------------------------

def test_variant_guard_refuses_explicit_mismatch():
    warnings = []

    def warn(message, code):
        warnings.append((message, code))

    metadata = {"base_model": "Comfy-Org/MiniMax-H3 minimax_h3_fl2va_pruned_bf16"}
    with pytest.raises(ValueError):
        lora_mod.check_variant_compatibility(metadata, "some_lora.safetensors", "ref2va", warn)
    # A matching variant must not raise or warn.
    lora_mod.check_variant_compatibility(metadata, "some_lora.safetensors", "fl2va", warn)
    assert not warnings


def test_variant_guard_warns_on_ambiguous_filename_fallback():
    warnings = []

    def warn(message, code):
        warnings.append((message, code))

    # No base_model metadata (F1's real case) -- fall back to the filename.
    lora_mod.check_variant_compatibility({}, "minimax_h3_ref2va_something.safetensors", "fl2va", warn)
    assert warnings and warnings[0][1] == "minimax_h3_lora_variant_ambiguous"


# ---------------------------------------------------------------------------
# 6. Rank variation across blocks (block-swap warning trigger)
# ---------------------------------------------------------------------------

def test_f1_qkv_rank_varies_across_blocks():
    """F1's ranks are documented as wildly unequal per block (31 distinct
    patterns across 52 stems) -- the rank-variation detector must see this."""
    _require(F1_PATH)
    raw, _metadata = lora_mod.load_lora_safetensors(F1_PATH)
    targets = lora_mod.normalise_lora_state_dict(raw)
    variation = lora_mod.detect_rank_variation(targets)
    assert variation.get("attn.to_q") is True
    assert variation.get("attn.to_k") is True
    assert variation.get("attn.to_v") is True


def test_uniform_rank_target_set_reports_no_variation():
    # Every block gets the identical rank -> no variation flagged.
    targets = {}
    for i in range(4):
        targets[f"transformer_blocks.{i}.attn.to_q"] = {
            "down": torch.zeros(8, _HIDDEN), "up": torch.zeros(_INNER, 8), "scale_ratio": 1.0,
        }
    variation = lora_mod.detect_rank_variation(targets)
    assert variation.get("attn.to_q") is False
