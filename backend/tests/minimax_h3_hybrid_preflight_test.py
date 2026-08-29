"""The MiniMax-H3 hybrid preflight refuses every incompatible pair, and the
block-range selector picks exactly the blocks it was asked for.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_hybrid_preflight_test.py -v

Contract: `docs/guides/MINIMAX_H3_HYBRID_LOADER_DESIGN.md`. Everything here is
HEADER-ONLY: the fixtures write safetensors files with a real struct-packed JSON
header and a zero-length
data section, which is exactly what the preflight reads and all it is allowed to
read.

The fixture writer is NOT new -- `_write_fake_h3_dit` / `_build_h3_tree` come
from `minimax_h3_model_listing_test.py`, which already builds the full fake tree
(`diffusion_models/ official/ text_encoders/ vae/`). Only the header content is
supplied per test.
"""

import os
import sys

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TESTS_DIR)
sys.path.insert(0, os.path.dirname(_TESTS_DIR))

from minimax_h3_model_listing_test import _build_h3_tree, _write_fake_h3_dit  # noqa: E402

from core.models.minimax_h3.hybrid_spec import (  # noqa: E402
    DEFAULT_BLOCK_RANGE_END,
    DEFAULT_BLOCK_RANGE_START,
    HEADER_SOURCE,
    PRESET_BLOCK_RANGE_ADALN,
    BlockRangeAdalnSelector,
    MiniMaxH3HybridPreflight,
    MiniMaxH3HybridRefusal,
    MiniMaxH3HybridSpec,
    compatibility_digest,
    preflight_minimax_h3_hybrid,
    validate_preset,
)


# ---------------------------------------------------------------------------
# header fixtures
# ---------------------------------------------------------------------------

def _t(dtype, shape):
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}


def _h3_header(
    *,
    num_blocks=6,
    adaln_bias=True,
    adaln_bias_blocks=None,
    weight_dtype="BF16",
    adaln_shape=(256, 8),
    quant_metadata=None,
    pruned=True,
    extra=None,
    drop=(),
):
    """A plausible pruned-variant H3 DiT header, small enough to read.

    `adaln_bias_blocks`, when given, limits which blocks carry an AdaLN bias --
    that is how the one-sided-bias case is built.
    """
    header = {
        "token_refiner.blocks.0.attn.qkv_proj.weight": _t(weight_dtype, [96, 32]),
        "video_patch_proj.weight": _t("F32", [64, 96]),
        "audio_patch_proj.weight": _t("F32", [64, 32]),
        "condition_proj.weight": _t(weight_dtype, [64, 48]),
        "final_layer.adaln_proj.linear.weight": _t(weight_dtype, list(adaln_shape)),
        "final_layer.video_out.weight": _t("F32", [96, 64]),
        "final_layer.audio_out.weight": _t("F32", [32, 64]),
    }
    if pruned:
        header["adaln_t_table"] = _t("F32", [64, adaln_shape[1]])
    else:
        # The full-modulation variant: a time_embedder MLP and no curve table.
        header["time_embedder.linear_1.weight"] = _t(weight_dtype, [64, 256])
        header["time_embedder.linear_2.weight"] = _t(weight_dtype, [64, 64])
    if adaln_bias:
        header["final_layer.adaln_proj.linear.bias"] = _t(weight_dtype, [adaln_shape[0]])
    for n in range(num_blocks):
        header[f"blocks.{n}.attn.qkv_proj.weight"] = _t(weight_dtype, [192, 64])
        header[f"blocks.{n}.mlp.fc1.weight"] = _t(weight_dtype, [256, 64])
        header[f"blocks.{n}.adaln_proj.linear.weight"] = _t(weight_dtype, list(adaln_shape))
        if adaln_bias and (adaln_bias_blocks is None or n in adaln_bias_blocks):
            header[f"blocks.{n}.adaln_proj.linear.bias"] = _t(weight_dtype, [adaln_shape[0]])
    for key in drop:
        header.pop(key, None)
    header.update(extra or {})
    metadata = {"format": "pt"}
    if quant_metadata is not None:
        metadata["_quantization_metadata"] = quant_metadata
    header["__metadata__"] = metadata
    return header


def _tree(tmp_path, *, base_header=None, overlay_header=None, name="h3"):
    """A fake H3 tree with both DiT partitions; returns (base_path, overlay_path)."""
    root = str(tmp_path / name)
    _build_h3_tree(root)
    base = os.path.join(root, "diffusion_models",
                        "minimax_h3_fl2va_pruned_fp8_scaled.safetensors")
    overlay = os.path.join(root, "diffusion_models",
                           "minimax_h3_ref2va_pruned_fp8_scaled.safetensors")
    _write_fake_h3_dit(base, header=base_header if base_header is not None else _h3_header())
    _write_fake_h3_dit(overlay,
                       header=overlay_header if overlay_header is not None else _h3_header())
    return base, overlay


def _run(base, overlay, **kwargs):
    kwargs.setdefault("block_range_start", 2)
    kwargs.setdefault("block_range_end", 3)
    return preflight_minimax_h3_hybrid(base, overlay, **kwargs)


def _refusal_code(base, overlay, **kwargs):
    with pytest.raises(MiniMaxH3HybridRefusal) as excinfo:
        _run(base, overlay, **kwargs)
    return excinfo.value.code


# ---------------------------------------------------------------------------
# the accepting case
# ---------------------------------------------------------------------------

def test_a_matching_pair_validates_and_gets_a_digest(tmp_path):
    base, overlay = _tree(tmp_path)
    result = _run(base, overlay)

    assert isinstance(result, MiniMaxH3HybridPreflight)
    assert isinstance(result.spec, MiniMaxH3HybridSpec)
    assert result.spec.validated
    assert result.spec.compatibility_digest.startswith("h3hybrid1:")
    assert result.spec.base_variant == "fl2va"
    assert result.spec.overlay_variant == "ref2va"
    assert result.spec.preset == PRESET_BLOCK_RANGE_ADALN
    assert result.quant_format == "unquantized"
    assert result.num_blocks == 6


def test_the_defaults_are_the_design_docs(tmp_path):
    assert (DEFAULT_BLOCK_RANGE_START, DEFAULT_BLOCK_RANGE_END) == (25, 49)
    spec = MiniMaxH3HybridSpec(base_dit_path="b", overlay_dit_path="o")
    assert spec.block_range_start == 25 and spec.block_range_end == 49
    assert spec.final_adaln_from_overlay is False
    assert spec.compatibility_digest is None and not spec.validated


# ---------------------------------------------------------------------------
# 4.2 -- the refusals
# ---------------------------------------------------------------------------

def test_a_missing_overlay_refuses_by_name(tmp_path):
    base, overlay = _tree(tmp_path)
    os.remove(overlay)
    assert _refusal_code(base, overlay) == "overlay_missing"
    assert _refusal_code(base, "") == "overlay_missing"


def test_different_trees_refuse(tmp_path):
    base, _ = _tree(tmp_path, name="tree_a")
    _, overlay = _tree(tmp_path, name="tree_b")
    assert _refusal_code(base, overlay) == "different_tree"


def test_reversed_variant_direction_refuses(tmp_path):
    base, overlay = _tree(tmp_path)
    # ref2va as the base and fl2va as the overlay is not the measured recipe.
    assert _refusal_code(overlay, base) == "variant_direction"
    # ...and so is fl2va onto itself.
    assert _refusal_code(base, base) == "variant_direction"


def test_a_non_h3_overlay_refuses(tmp_path):
    base, overlay = _tree(tmp_path)
    _write_fake_h3_dit(overlay, header={"some.other.arch.weight": _t("BF16", [4, 4])})
    assert _refusal_code(base, overlay) == "not_h3_checkpoint"


def test_key_set_mismatch_refuses(tmp_path):
    base, overlay = _tree(tmp_path, overlay_header=_h3_header(num_blocks=5))
    assert _refusal_code(base, overlay) == "key_set_mismatch"


def test_shape_mismatch_refuses(tmp_path):
    base, overlay = _tree(tmp_path, overlay_header=_h3_header(adaln_shape=(512, 8)))
    assert _refusal_code(base, overlay) == "shape_mismatch"


def test_dtype_mismatch_refuses(tmp_path):
    base, overlay = _tree(tmp_path, overlay_header=_h3_header(weight_dtype="F8_E4M3"))
    assert _refusal_code(base, overlay) == "dtype_mismatch"


def test_a_pruned_full_mix_refuses(tmp_path):
    """A pruned base with a full-modulation overlay is refused.

    The doc lists geometry as check 6, but a real pruned/full pair cannot reach
    it: the two variants differ in KEY NAMES (`adaln_t_table` vs
    `time_embedder.*`), so check 4 fires first. The refusal is real; its code is
    the key-set one, and this test records that rather than pretending check 6
    is what catches it. Check 6's reachable arm is the both-full case below.
    """
    base, overlay = _tree(tmp_path, overlay_header=_h3_header(pruned=False))
    assert _refusal_code(base, overlay) == "key_set_mismatch"


def test_two_full_modulation_checkpoints_refuse_as_out_of_scope(tmp_path):
    base, overlay = _tree(tmp_path,
                          base_header=_h3_header(pruned=False),
                          overlay_header=_h3_header(pruned=False))
    assert _refusal_code(base, overlay) == "geometry_unsupported"


def test_mixed_quantization_metadata_refuses(tmp_path):
    """Identical tensors, different declared quantization contract.

    This is the case checks 4/5 cannot see: `__metadata__` is not part of the
    key census.
    """
    fp8 = '{"layers": {"blocks.0.attn.qkv_proj": {"format": "float8_e4m3fn"}}}'
    other = ('{"layers": {"blocks.0.attn.qkv_proj": '
             '{"format": "float8_e4m3fn", "full_precision_matrix_mult": true}}}')
    base, overlay = _tree(tmp_path,
                          base_header=_h3_header(quant_metadata=fp8),
                          overlay_header=_h3_header(quant_metadata=other))
    assert _refusal_code(base, overlay) == "quant_metadata_mismatch"


def test_quantization_metadata_on_one_side_only_refuses(tmp_path):
    fp8 = '{"layers": {"blocks.0.attn.qkv_proj": {"format": "float8_e4m3fn"}}}'
    base, overlay = _tree(tmp_path, base_header=_h3_header(quant_metadata=fp8))
    assert _refusal_code(base, overlay) == "quant_metadata_mismatch"


def test_matching_quantization_metadata_passes(tmp_path):
    fp8 = '{"layers": {"blocks.0.attn.qkv_proj": {"format": "float8_e4m3fn"}}}'
    base, overlay = _tree(tmp_path,
                          base_header=_h3_header(weight_dtype="F8_E4M3", quant_metadata=fp8),
                          overlay_header=_h3_header(weight_dtype="F8_E4M3", quant_metadata=fp8))
    result = _run(base, overlay)
    assert result.quant_format == "fp8_scaled"


def test_an_invalid_w4a8_contract_refuses(tmp_path):
    """The loader's own W4A8 validator runs on BOTH files, header-only.

    A hybrid must not be able to accept a contract the single-file path refuses.
    """
    w4a8 = ('{"layers": {"blocks.0.attn.qkv_proj": '
            '{"format": "asym_w4a8_int8", "convrot": true}}}')
    # Declared W4A8 with no packed weight/sidecars at all -> the validator refuses.
    header = _h3_header(quant_metadata=w4a8)
    base, overlay = _tree(tmp_path, base_header=header, overlay_header=header)
    assert _refusal_code(base, overlay) == "w4a8_contract_invalid"


# ---------------------------------------------------------------------------
# 4.2 check 9 -- range boundaries and the bias
# ---------------------------------------------------------------------------

def test_an_empty_range_refuses(tmp_path):
    base, overlay = _tree(tmp_path)
    assert _refusal_code(base, overlay, block_range_start=4,
                         block_range_end=3) == "block_range_empty"


def test_an_out_of_range_end_refuses_naming_the_last_block(tmp_path):
    base, overlay = _tree(tmp_path)
    with pytest.raises(MiniMaxH3HybridRefusal) as excinfo:
        _run(base, overlay, block_range_start=2, block_range_end=6)
    assert excinfo.value.code == "block_range_out_of_range"
    assert "5" in excinfo.value.message
    assert _refusal_code(base, overlay, block_range_start=-1,
                         block_range_end=3) == "block_range_out_of_range"


def test_the_full_block_range_is_accepted_at_both_ends(tmp_path):
    base, overlay = _tree(tmp_path)
    result = _run(base, overlay, block_range_start=0, block_range_end=5)
    blocks = sorted(int(k.split(".")[1]) for k in result.overlay_keys
                    if k.endswith(".weight"))
    assert blocks == [0, 1, 2, 3, 4, 5]


def test_a_bias_present_on_one_side_only_refuses(tmp_path):
    base, overlay = _tree(tmp_path, overlay_header=_h3_header(adaln_bias=False))
    # The bias keys themselves make the key sets differ, so check 4 speaks first.
    assert _refusal_code(base, overlay) == "key_set_mismatch"


def test_a_bias_absent_from_both_sides_is_simply_not_overlaid(tmp_path):
    """Section 4.2 check 9: the bias is not assumed to exist."""
    header = _h3_header(adaln_bias=False)
    base, overlay = _tree(tmp_path, base_header=header, overlay_header=header)
    result = _run(base, overlay)
    assert result.overlay_bias_eligible is False
    assert result.overlay_keys == (
        "blocks.2.adaln_proj.linear.weight", "blocks.3.adaln_proj.linear.weight")
    assert result.selector.source_for("blocks.2.adaln_proj.linear.bias") == "base"


def test_a_bias_present_on_both_sides_is_overlaid_with_its_weight(tmp_path):
    base, overlay = _tree(tmp_path)
    result = _run(base, overlay)
    assert result.overlay_bias_eligible is True
    assert result.overlay_keys == (
        "blocks.2.adaln_proj.linear.bias", "blocks.2.adaln_proj.linear.weight",
        "blocks.3.adaln_proj.linear.bias", "blocks.3.adaln_proj.linear.weight")


def test_a_partially_present_bias_refuses(tmp_path):
    """Some selected blocks carry a bias and some do not: not applicable."""
    header = _h3_header(adaln_bias_blocks=(0, 1, 2))
    base, overlay = _tree(tmp_path, base_header=header, overlay_header=header)
    assert _refusal_code(base, overlay, block_range_start=2,
                         block_range_end=3) == "adaln_bias_partial"


def test_an_unclassified_adaln_sidecar_refuses(tmp_path):
    """Section 4.4 atomicity, as a guard: shipped files never hit this."""
    extra = {"blocks.2.adaln_proj.linear.mystery_scale": _t("F32", [8])}
    header = _h3_header(extra=extra)
    base, overlay = _tree(tmp_path, base_header=header, overlay_header=header)
    assert _refusal_code(base, overlay) == "adaln_sidecar_unknown"


def test_an_unclassified_sidecar_outside_the_range_does_not_refuse(tmp_path):
    """A future NVFP4-ish `.weight_scale_2` on block 5 must not kill range 2..3.

    Block 5's AdaLN is read from the base whatever sidecars it has, so there is
    nothing to decide and nothing to refuse.
    """
    extra = {"blocks.5.adaln_proj.linear.weight_scale_2": _t("F32", [8]),
             "final_layer.adaln_proj.linear.weight_scale_2": _t("F32", [8])}
    header = _h3_header(extra=extra)
    base, overlay = _tree(tmp_path, base_header=header, overlay_header=header)
    result = _run(base, overlay, block_range_start=2, block_range_end=3)
    assert "blocks.5.adaln_proj.linear.weight_scale_2" not in result.overlay_keys


# ---------------------------------------------------------------------------
# 4.3 -- the selector as a pure predicate
# ---------------------------------------------------------------------------

def test_the_selector_picks_exactly_the_requested_blocks_and_nothing_else():
    selector = BlockRangeAdalnSelector(block_range_start=25, block_range_end=49,
                                       overlay_bias=True)
    for n in (24, 50, 0, 99):
        assert selector.source_for(f"blocks.{n}.adaln_proj.linear.weight") == "base"
        assert selector.source_for(f"blocks.{n}.adaln_proj.linear.bias") == "base"
    for n in (25, 26, 37, 48, 49):
        assert selector.source_for(f"blocks.{n}.adaln_proj.linear.weight") == "overlay"
        assert selector.source_for(f"blocks.{n}.adaln_proj.linear.bias") == "overlay"
    # Everything that is not a block AdaLN projection stays on the base --
    # including the curve table and the in-range block's OTHER tensors.
    for key in ("adaln_t_table",
                "blocks.30.attn.qkv_proj.weight",
                "blocks.30.mlp.fc1.weight",
                "blocks.30.norm1.weight",
                "video_patch_proj.weight",
                "token_refiner.blocks.0.adaln_proj.linear.weight",
                "final_layer.video_out.weight"):
        assert selector.source_for(key) == "base", key


def test_the_final_adaln_toggle_is_separate_and_defaults_off():
    off = BlockRangeAdalnSelector(block_range_start=0, block_range_end=99, overlay_bias=True)
    assert off.final_adaln_from_overlay is False
    assert off.source_for("final_layer.adaln_proj.linear.weight") == "base"
    on = BlockRangeAdalnSelector(block_range_start=0, block_range_end=99,
                                 final_adaln_from_overlay=True, overlay_bias=True)
    assert on.source_for("final_layer.adaln_proj.linear.weight") == "overlay"
    assert on.source_for("final_layer.adaln_proj.linear.bias") == "overlay"
    assert on.source_for("adaln_t_table") == "base"


def test_the_final_adaln_toggle_reaches_the_preflight(tmp_path):
    base, overlay = _tree(tmp_path)
    result = _run(base, overlay, final_adaln_from_overlay=True)
    assert "final_layer.adaln_proj.linear.weight" in result.overlay_keys
    assert result.spec.recipe()["final_adaln_from_overlay"] is True
    assert "final_layer.adaln_proj.linear.weight" not in _run(base, overlay).overlay_keys


def test_quantization_sidecars_follow_their_weight():
    """Section 4.4: a selected weight's sidecars come from the same file.

    Not exercised by shipped checkpoints (`adaln_proj` is not quantized) -- this
    pins the rule for the export that changes that.
    """
    selector = BlockRangeAdalnSelector(block_range_start=2, block_range_end=3,
                                       overlay_bias=True)
    for suffix in (".weight", ".weight_scale", ".comfy_quant", ".weight_s_rel",
                   ".weight_s_channel", ".weight_codebook", ".weight_correction"):
        assert selector.source_for(f"blocks.2.adaln_proj.linear{suffix}") == "overlay", suffix
        assert selector.source_for(f"blocks.9.adaln_proj.linear{suffix}") == "base", suffix
    # `.input_scale` is dropped by loader policy, so it has no provenance.
    assert selector.source_for("blocks.2.adaln_proj.linear.input_scale") == "base"
    with pytest.raises(MiniMaxH3HybridRefusal) as excinfo:
        selector.source_for("blocks.2.adaln_proj.linear.mystery_scale")
    assert excinfo.value.code == "adaln_sidecar_unknown"


def test_source_for_is_total_for_every_key_the_selection_does_not_touch():
    """Only a key that WOULD be overlaid may raise.

    `overlay_keys()` runs `source_for` over every header key, and C3's reader
    will call it per key; an unknown sidecar on an out-of-range block is
    unambiguously base, so refusing it would kill the feature for ranges that
    never touch it -- and would make the reader need a try/except.
    """
    selector = BlockRangeAdalnSelector(block_range_start=2, block_range_end=3,
                                       overlay_bias=True)
    assert selector.source_for("blocks.5.adaln_proj.linear.weight_scale_2") == "base"
    assert selector.source_for("blocks.99.adaln_proj.linear.anything_at_all") == "base"
    # The final-AdaLN toggle is off, so its sidecars are equally untouched.
    assert selector.source_for("final_layer.adaln_proj.linear.weight_scale_2") == "base"
    # ...and running the whole header through it does not raise.
    keys = ["blocks.5.adaln_proj.linear.weight_scale_2",
            "blocks.2.adaln_proj.linear.weight", "adaln_t_table"]
    assert selector.overlay_keys(keys) == ["blocks.2.adaln_proj.linear.weight"]
    # With the toggle ON, the same final-layer sidecar becomes undecidable.
    with pytest.raises(MiniMaxH3HybridRefusal) as excinfo:
        BlockRangeAdalnSelector(2, 3, final_adaln_from_overlay=True).source_for(
            "final_layer.adaln_proj.linear.weight_scale_2")
    assert excinfo.value.code == "adaln_sidecar_unknown"


# ---------------------------------------------------------------------------
# 8 -- the non-MVP recipes are refused by name
# ---------------------------------------------------------------------------

def test_custom_glob_full_overlay_and_multiple_overlays_are_refused():
    for preset in ("custom_glob", "full_overlay", "all", "multi_overlay"):
        with pytest.raises(MiniMaxH3HybridRefusal) as excinfo:
            validate_preset(preset, "one.safetensors")
        assert excinfo.value.code == "preset_unsupported", preset
    with pytest.raises(MiniMaxH3HybridRefusal) as excinfo:
        validate_preset("adaln_by_similarity", "one.safetensors")
    assert excinfo.value.code == "preset_unknown"
    with pytest.raises(MiniMaxH3HybridRefusal) as excinfo:
        validate_preset(PRESET_BLOCK_RANGE_ADALN, ["a.safetensors", "b.safetensors"])
    assert excinfo.value.code == "multiple_overlays"


def test_a_refused_preset_never_reaches_the_header_reads(tmp_path):
    base, overlay = _tree(tmp_path)
    assert _refusal_code(base, overlay, preset="full_overlay") == "preset_unsupported"


# ---------------------------------------------------------------------------
# 4.2 -- the digest and the header-source contract
# ---------------------------------------------------------------------------

def test_the_digest_is_reproducible_and_recipe_independent(tmp_path):
    base, overlay = _tree(tmp_path)
    first = _run(base, overlay).spec.compatibility_digest
    second = _run(base, overlay).spec.compatibility_digest
    assert first == second
    # The RECIPE is not part of the contract digest (C4 composes identity as
    # digest + recipe), so changing the range must not change it.
    assert _run(base, overlay, block_range_start=0,
                block_range_end=5).spec.compatibility_digest == first
    assert _run(base, overlay,
                final_adaln_from_overlay=True).spec.compatibility_digest == first


def test_the_digest_changes_when_the_validated_contract_changes(tmp_path):
    base, overlay = _tree(tmp_path)
    baseline = _run(base, overlay).spec.compatibility_digest

    other = _h3_header(num_blocks=5)
    b2, o2 = _tree(tmp_path, base_header=other, overlay_header=other, name="h3_five")
    assert _run(b2, o2).spec.compatibility_digest != baseline

    fp8 = '{"layers": {"blocks.0.attn.qkv_proj": {"format": "float8_e4m3fn"}}}'
    quant = _h3_header(weight_dtype="F8_E4M3", quant_metadata=fp8)
    b3, o3 = _tree(tmp_path, base_header=quant, overlay_header=quant, name="h3_fp8")
    assert _run(b3, o3).spec.compatibility_digest != baseline


def test_the_digest_distinguishes_two_trees_with_identical_filenames(tmp_path):
    """The standard release filenames repeat across trees; the digest must not.

    Doc section 7 needs the digest to notice an overlay file that changed
    between preflight and the real read, so it has to identify the FILES.
    """
    base_a, overlay_a = _tree(tmp_path, name="tree_one")
    other = _h3_header(num_blocks=6, extra={"blocks.0.norm1.weight": _t("BF16", [64])})
    base_b, overlay_b = _tree(tmp_path, base_header=_h3_header(), overlay_header=other,
                              name="tree_two")
    assert os.path.basename(base_a) == os.path.basename(base_b)
    assert os.path.basename(overlay_a) == os.path.basename(overlay_b)
    # tree_two's overlay differs from its base, so it does not validate; what
    # this pins is that the OVERLAY's census reaches the digest at all.
    assert _refusal_code(base_b, overlay_b) == "key_set_mismatch"

    both = _tree(tmp_path, base_header=other, overlay_header=other, name="tree_three")
    assert _run(*both).spec.compatibility_digest != _run(base_a, overlay_a) \
        .spec.compatibility_digest


def test_a_changed_overlay_changes_the_digest(tmp_path):
    """Same base, an overlay whose contents moved -> a different digest."""
    base, overlay = _tree(tmp_path)
    before = _run(base, overlay).spec.compatibility_digest
    # Same key set/shape/dtype on both sides, but both files' censuses shift.
    moved = _h3_header(adaln_shape=(256, 8), extra={"blocks.0.norm1.weight": _t("BF16", [64])})
    base2, overlay2 = _tree(tmp_path, base_header=moved, overlay_header=moved, name="moved")
    assert _run(base2, overlay2).spec.compatibility_digest != before


def test_a_mixed_int8_plus_fp8_export_is_labelled_as_both(tmp_path):
    """This repo's own `--format int8` exporter emits a mixed file on purpose.

    Section 5.4 puts this label into generation metadata, so calling such a file
    plain `fp8_scaled` would be a false record.
    """
    mixed = _h3_header(extra={
        "blocks.0.attn.qkv_proj.weight": _t("I8", [192, 64]),
        "blocks.0.attn.qkv_proj.comfy_quant": _t("U8", [16]),
        "blocks.0.mlp.fc2.weight": _t("F8_E4M3", [64, 256]),
    })
    base, overlay = _tree(tmp_path, base_header=mixed, overlay_header=mixed)
    assert _run(base, overlay).quant_format == "fp8_scaled+int8_convrot"


def test_a_file_that_is_neither_variant_says_so(tmp_path):
    """Both an `adaln_t_table` AND `time_embedder.*`: the loader's own wording."""
    both = _h3_header(extra={"time_embedder.linear_1.weight": _t("BF16", [64, 256])})
    base, overlay = _tree(tmp_path, base_header=both, overlay_header=both)
    with pytest.raises(MiniMaxH3HybridRefusal) as excinfo:
        _run(base, overlay)
    assert excinfo.value.code == "geometry_contradictory"
    assert "neither" in excinfo.value.message


def test_the_digest_helper_refuses_malformed_metadata_in_the_taxonomy(tmp_path):
    """`compatibility_digest` is public; C4/C5 may call it without the preflight."""
    with pytest.raises(MiniMaxH3HybridRefusal) as excinfo:
        compatibility_digest(
            base_path="b.safetensors", overlay_path="o.safetensors",
            base_variant="fl2va", overlay_variant="ref2va",
            header={}, metadata={"_quantization_metadata": "{not json"},
            overlay_header={}, quant_format="unquantized", num_blocks=1)
    assert excinfo.value.code == "quant_metadata_malformed"


def test_every_header_consumer_is_handed_the_base_header(tmp_path):
    """Section 4.2's closing contract, encoded structurally rather than hoped for."""
    base, overlay = _tree(tmp_path)
    result = _run(base, overlay)
    assert HEADER_SOURCE == "base" and result.header_source == "base"

    from core.models.minimax_h3.loader import read_safetensors_header
    expected = read_safetensors_header(base)
    expected.pop("__metadata__", None)
    assert result.header == expected
    assert result.metadata == {"format": "pt"}

    # The overlay header is not reachable from the preflight result at all, so
    # no downstream consumer can pick the wrong one by accident.
    fields = set(vars(result))
    assert not any("overlay_header" in name for name in fields), fields
    assert fields == {
        "spec", "header", "metadata", "base_layout", "overlay_layout", "selector",
        "overlay_keys", "quant_format", "num_blocks", "overlay_bias_eligible",
        "header_source",
    }


def test_the_spec_serialises_in_the_design_docs_field_order(tmp_path):
    base, overlay = _tree(tmp_path)
    spec = _run(base, overlay).spec
    assert list(spec.as_dict()) == [
        "schema_version", "base_dit_path", "overlay_dit_path", "preset",
        "block_range_start", "block_range_end", "final_adaln_from_overlay",
        "base_variant", "overlay_variant", "compatibility_digest",
    ]


def test_provenance_carries_no_absolute_paths(tmp_path):
    base, overlay = _tree(tmp_path)
    provenance = _run(base, overlay).provenance()
    assert provenance["variant"] == "hybrid"
    assert provenance["base_file"] == os.path.basename(base)
    assert provenance["overlay_file"] == os.path.basename(overlay)
    blob = repr(provenance)
    assert os.path.dirname(base) not in blob
