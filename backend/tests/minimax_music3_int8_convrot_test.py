"""MiniMax Music 3 INT8 ConvRot -- design doc phase 13.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_int8_convrot_test.py -v

Two layers, matching the convention `minimax_h3_te_int8_convrot_test.py` and
`minimax_music3_gguf_q8_0_linear_test.py` already use for a quantization
phase:

* SYNTHETIC unit tests for the pure remap logic in
  `core.models.minimax_music3.convrot_remap` (marker validation, the sidecar
  splitter, and the two `apply_*_with_convrot` functions) -- no checkpoint
  needed, run everywhere.
* REAL-FILE tests against the two staged artifacts
  (`<MODEL_ROOT>/minimax-music3/diffusion_models/minimax_music3_dit_int8_convrot.
  safetensors` and `.../text_encoders/minimax_music3_text_encoder_pruned_
  int8_convrot.safetensors`), skipped cleanly when the snapshot is not
  present on this machine. These are the POSITIVE cases the design doc's
  audit history flags as missing when only refusal shapes are tested (see
  `minimax_music3_text_encoder_choice_test.py`'s own real-file section and
  its "regression this section exists to catch" note): a detector or a
  builder that only ever sees refusal fixtures can regress the ACCEPT path
  silently.
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model_root import model_path  # noqa: E402

from core.models.common.convrot_int8_linear import ConvRotInt8Linear  # noqa: E402
from core.models.minimax_music3 import convrot_remap, loader  # noqa: E402

# ---------------------------------------------------------------------------
# supported_int8_convrot_marker -- synthetic
# ---------------------------------------------------------------------------


def _marker_tensor(payload: dict) -> torch.Tensor:
    return torch.frombuffer(bytearray(json.dumps(payload).encode("utf-8")), dtype=torch.uint8).clone()


def test_supported_int8_convrot_marker_accepts_the_validated_contract():
    header = {
        "layer.weight": {"dtype": "I8", "shape": [512, 256]},
        "layer.weight_scale": {"dtype": "F32", "shape": [512, 1]},
    }
    marker = _marker_tensor({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
    config = convrot_remap.supported_int8_convrot_marker(
        "layer.comfy_quant", marker, header, path="fixture",
    )
    assert config == {"convrot_groupsize": 256, "marker_numel": marker.numel()}


def test_supported_int8_convrot_marker_rejects_a_different_format():
    header = {
        "layer.weight": {"dtype": "I8", "shape": [512, 256]},
        "layer.weight_scale": {"dtype": "F32", "shape": [512, 1]},
    }
    marker = _marker_tensor({"format": "nvfp4", "full_precision_matrix_mult": True})
    assert convrot_remap.supported_int8_convrot_marker(
        "layer.comfy_quant", marker, header, path="fixture",
    ) is None


def test_supported_int8_convrot_marker_rejects_a_different_groupsize():
    header = {
        "layer.weight": {"dtype": "I8", "shape": [512, 128]},
        "layer.weight_scale": {"dtype": "F32", "shape": [512, 1]},
    }
    marker = _marker_tensor({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 128})
    assert convrot_remap.supported_int8_convrot_marker(
        "layer.comfy_quant", marker, header, path="fixture",
    ) is None


def test_supported_int8_convrot_marker_rejects_k_not_divisible_by_256():
    header = {
        "layer.weight": {"dtype": "I8", "shape": [512, 300]},
        "layer.weight_scale": {"dtype": "F32", "shape": [512, 1]},
    }
    marker = _marker_tensor({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
    with pytest.raises(ValueError, match="not divisible by 256"):
        convrot_remap.supported_int8_convrot_marker("layer.comfy_quant", marker, header, path="fixture")


def test_supported_int8_convrot_marker_rejects_a_missing_scale():
    header = {"layer.weight": {"dtype": "I8", "shape": [512, 256]}}
    marker = _marker_tensor({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
    with pytest.raises(ValueError, match="missing weight or weight_scale"):
        convrot_remap.supported_int8_convrot_marker("layer.comfy_quant", marker, header, path="fixture")


# ---------------------------------------------------------------------------
# _split_convrot_sidecar -- synthetic, no checkpoint
# ---------------------------------------------------------------------------


def test_split_convrot_sidecar_equal_three_way():
    marker = _marker_tensor({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
    scale = torch.arange(6, dtype=torch.float32)  # [0, 1, 2, 3, 4, 5]
    dest_sizes = (("a.weight", -1), ("b.weight", -1), ("c.weight", -1))
    config = {"convrot_groupsize": 256, "marker_numel": marker.numel()}
    out = convrot_remap._split_convrot_sidecar(marker, scale, dest_sizes, config)
    assert set(out.keys()) == {"a", "b", "c"}
    torch.testing.assert_close(out["a"][0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(out["b"][0], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(out["c"][0], torch.tensor([4.0, 5.0]))
    seen_storage_ids = set()
    for dest in ("a", "b", "c"):
        torch.testing.assert_close(out[dest][1], marker)  # same VALUE
        assert out[dest][1] is not marker  # but a DISTINCT tensor object (cloned, F4)
        seen_storage_ids.add(out[dest][1].data_ptr())
        assert out[dest][2] == config
        assert out[dest][2] is not config  # but the config dict itself IS copied
    # No two destinations share the marker's storage either -- a future
    # `safetensors.save_file` must not see aliased tensors.
    assert len(seen_storage_ids) == 3


def test_split_convrot_sidecar_explicit_uneven_sizes():
    """The language model's GQA-uneven qkv split (q_dim != k_dim == v_dim)."""
    marker = _marker_tensor({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
    scale = torch.arange(10, dtype=torch.float32)
    dest_sizes = (("q.weight", 6), ("k.weight", 2), ("v.weight", 2))
    config = {"convrot_groupsize": 256, "marker_numel": marker.numel()}
    out = convrot_remap._split_convrot_sidecar(marker, scale, dest_sizes, config)
    assert out["q"][0].numel() == 6
    assert out["k"][0].numel() == 2
    assert out["v"][0].numel() == 2
    torch.testing.assert_close(out["q"][0], scale[:6])
    torch.testing.assert_close(out["k"][0], scale[6:8])
    torch.testing.assert_close(out["v"][0], scale[8:10])


def test_split_convrot_sidecar_rejects_mixed_explicit_and_equal_sizes():
    marker = _marker_tensor({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
    scale = torch.zeros(4)
    with pytest.raises(ValueError, match="mixes explicit and equal-split"):
        convrot_remap._split_convrot_sidecar(
            marker, scale, (("a.weight", 2), ("b.weight", -1)), {"convrot_groupsize": 256, "marker_numel": 1},
        )


def test_split_convrot_sidecar_rejects_wrong_total():
    marker = _marker_tensor({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
    scale = torch.zeros(4)
    with pytest.raises(ValueError, match="expected"):
        convrot_remap._split_convrot_sidecar(
            marker, scale, (("a.weight", 2), ("b.weight", 3)), {"convrot_groupsize": 256, "marker_numel": 1},
        )


# ---------------------------------------------------------------------------
# apply_flat_dit_state_dict_with_convrot -- synthetic (a tiny fused qkv AND a
# tiny non-fused layer, both "quantized"). K=8 here -- NOT a real 256-multiple
# -- deliberately: the 256 divisibility contract lives in
# `supported_int8_convrot_marker` (already covered above), not in the apply
# function itself, which only cares about ROW counts.
# ---------------------------------------------------------------------------


def _tiny_flat_dit_state_dict_with_convrot():
    """A minimal state dict shaped like `flat_remap`'s own fixture (see
    `minimax_music3_flat_dit_fixture.py`) but with ONE quantized fused qkv
    layer and ONE quantized non-fused layer (`self_attn.to_out`), each
    carrying int8 `.weight` + `.weight_scale` + `.comfy_quant`."""
    inner = 4
    marker = _marker_tensor({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
    state = {
        "diffusion_transformer.transformer.project_in.weight": torch.randn(inner, 6),
        "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight": torch.randint(
            -127, 128, (3 * inner, 8), dtype=torch.int8,
        ),
        "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight_scale": torch.arange(
            3 * inner, dtype=torch.float32,
        ).reshape(-1, 1) + 1.0,
        "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.comfy_quant": marker,
        "diffusion_transformer.transformer.layers.0.self_attn.to_out.weight": torch.randint(
            -127, 128, (inner, inner), dtype=torch.int8,
        ),
        "diffusion_transformer.transformer.layers.0.self_attn.to_out.weight_scale": torch.arange(
            inner, dtype=torch.float32,
        ).reshape(-1, 1) + 1.0,
        "diffusion_transformer.transformer.layers.0.self_attn.to_out.comfy_quant": marker,
    }
    convrot_source_layers = {
        "diffusion_transformer.transformer.layers.0.self_attn.to_qkv": {
            "convrot_groupsize": 256, "marker_numel": marker.numel(),
        },
        "diffusion_transformer.transformer.layers.0.self_attn.to_out": {
            "convrot_groupsize": 256, "marker_numel": marker.numel(),
        },
    }
    return state, convrot_source_layers


def test_apply_flat_dit_state_dict_with_convrot_places_sidecars_and_splits_the_fused_qkv():
    state, convrot_source_layers = _tiny_flat_dit_state_dict_with_convrot()
    remapped, dest_layer_configs = convrot_remap.apply_flat_dit_state_dict_with_convrot(
        state, convrot_source_layers,
    )
    transformer = remapped["transformer"]

    # The three split destinations, each with its OWN weight (int8 codes,
    # row-split -- exact, per the module docstring) and weight_scale, and the
    # SAME marker.
    for suffix, expected_scale in (
        ("to_q", torch.tensor([1.0, 2.0, 3.0, 4.0])),
        ("to_k", torch.tensor([5.0, 6.0, 7.0, 8.0])),
        ("to_v", torch.tensor([9.0, 10.0, 11.0, 12.0])),
    ):
        dest = f"transformer_blocks.0.attn.{suffix}"
        assert transformer[dest + ".weight"].dtype is torch.int8
        assert transformer[dest + ".weight"].shape == (4, 8)
        torch.testing.assert_close(transformer[dest + ".weight_scale"], expected_scale)
        assert dest in dest_layer_configs
        assert dest_layer_configs[dest] == {"convrot_groupsize": 256, "marker_numel": 72}

    # The non-fused destination.
    dest = "transformer_blocks.0.attn.to_out.0"
    assert transformer[dest + ".weight"].dtype is torch.int8
    torch.testing.assert_close(transformer[dest + ".weight_scale"], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert dest in dest_layer_configs


def test_apply_flat_dit_state_dict_with_convrot_is_a_no_op_pass_through_when_nothing_is_quantized():
    from tests.minimax_music3_flat_dit_fixture import write_tiny_flat_dit_and_official_tree
    from core.models.minimax_music3.flat_remap import apply_flat_dit_state_dict
    from core.models.common.single_file_format import read_state_dict

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fixture = write_tiny_flat_dit_and_official_tree(tmp)
        flat_state_dict, _metadata = read_state_dict(fixture["dit_path"])
        remapped, dest_layer_configs = convrot_remap.apply_flat_dit_state_dict_with_convrot(
            flat_state_dict, {},
        )
        assert dest_layer_configs == {}
        expected = apply_flat_dit_state_dict(flat_state_dict)
        assert set(remapped["transformer"].keys()) == set(expected["transformer"].keys())
        assert set(remapped["condition_encoder"].keys()) == set(expected["condition_encoder"].keys())


# ---------------------------------------------------------------------------
# Real staged files, header-only census + real structural builds + a
# tensor-level numeric proof against the BF16/FP16 sibling -- skipped cleanly
# when the snapshot is not present, matching
# `minimax_h3_te_int8_convrot_test.py::test_real_distribution_selects_
# int8_convrot_by_default`'s own convention.
# ---------------------------------------------------------------------------

_REAL_ROOT = model_path("minimax-music3")
_REAL_DIT_CONVROT = os.path.join(_REAL_ROOT, "diffusion_models", "minimax_music3_dit_int8_convrot.safetensors")
_REAL_DIT_FP16 = os.path.join(_REAL_ROOT, "diffusion_models", "minimax_music3_dit_fp16.safetensors")
_REAL_TE_CONVROT = os.path.join(
    _REAL_ROOT, "text_encoders", "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
)
_REAL_TE_BF16 = os.path.join(
    _REAL_ROOT, "text_encoders", "minimax_music3_text_encoder_pruned_bf16.safetensors",
)
_REAL_OFFICIAL = os.path.join(_REAL_ROOT, "official")


def _skip_unless_real_files_present(*paths):
    for path in paths:
        if not os.path.exists(path):
            pytest.skip(f"{path} not present on this machine")


def test_real_dit_convrot_header_census_matches_the_design_docs_own_count():
    """374 dense (non-sidecar) tensor names -- the SAME census the design
    doc's "Quantization" section states for the unquantized flat DiT -- with
    144 of the `.weight` entries additionally carrying a validated ConvRot
    `.comfy_quant` marker (36 layers x 4 quantized module kinds:
    self_attn.to_qkv, self_attn.to_out, ff.ff.0.proj, ff.ff.2)."""
    _skip_unless_real_files_present(_REAL_DIT_CONVROT)
    convrot_layers = loader._int8_convrot_source_layers(_REAL_DIT_CONVROT, label="flat DiT")
    assert len(convrot_layers) == 144

    header = loader.read_safetensors_header(_REAL_DIT_CONVROT)
    header.pop("__metadata__", None)
    dense_keys = [k for k in header if not (k.endswith(".comfy_quant") or k.endswith(".weight_scale"))]
    assert len(dense_keys) == 374

    from core.models.minimax_music3.flat_remap import plan_flat_dit_keys

    plan = plan_flat_dit_keys(dense_keys)
    assert plan.unrecognized == []


def test_real_dit_convrot_builds_structurally_with_the_right_swap_count():
    """A real load, not a structural assertion against fixtures: proves the
    ConvRot swap puts real ConvRotInt8Linear modules where the dense remap
    would have put plain nn.Linear, with no stranded meta tensor.

    Host RAM: the DiT file is 2.50 GB; ConvRot weights stay packed (int8),
    so the model's resident footprint is close to that, not the ~5 GB a
    bf16-expanded 36-layer DiT of this width would need. Expect a peak
    under ~6 GB for this one test. No GPU needed -- the build never leaves
    CPU/meta."""
    _skip_unless_real_files_present(_REAL_DIT_CONVROT, _REAL_OFFICIAL)
    transformer, _tcfg, condition_encoder, _ccfg = loader.build_transformer_and_condition_encoder_from_flat_dit(
        _REAL_DIT_CONVROT, _REAL_OFFICIAL, torch.bfloat16,
    )
    n_convrot = sum(1 for m in transformer.modules() if isinstance(m, ConvRotInt8Linear))
    # 144 source layers; the fused to_qkv (36 of the 144) each expand to 3
    # destination Linears, so 108 non-fused + 108 (36 x 3) fused-expanded = 216.
    assert n_convrot == 216
    stranded = [
        n for n, t in list(transformer.named_parameters()) + list(transformer.named_buffers())
        if getattr(t, "is_meta", False)
    ]
    assert stranded == []
    assert sum(1 for m in condition_encoder.modules() if isinstance(m, ConvRotInt8Linear)) == 0


def test_real_dit_convrot_dequantizes_to_the_fp16_sibling_within_int8_noise_floor():
    """Tensor-level A/B (design doc phase 13's own requirement): dequantize a
    handful of real ConvRot weights via comfy-kitchen's own op and compare
    against the flat FP16 sibling's plain weight for the identical layer --
    phase 9 already proved the flat FP16 file is bit-exact under
    `official.bfloat16().half()`, so this is effectively the BF16 A/B the
    design doc asks for, expressed at the weight level rather than at the
    decoded-audio level (see this test's module docstring / the design doc's
    own status entry for why the weight-level comparison was chosen over a
    full generation).

    Single-tensor reads only (`safe_open`), never a full state-dict load --
    this test's own host-RAM footprint is a few MB regardless of file size.
    """
    _skip_unless_real_files_present(_REAL_DIT_CONVROT, _REAL_DIT_FP16)
    try:
        import comfy_kitchen  # noqa: F401
    except ImportError:
        pytest.skip("comfy-kitchen is not installed")
    from safetensors import safe_open

    layers = [
        "diffusion_transformer.transformer.layers.0.self_attn.to_qkv",
        "diffusion_transformer.transformer.layers.0.self_attn.to_out",
        "diffusion_transformer.transformer.layers.0.ff.ff.0.proj",
        "diffusion_transformer.transformer.layers.35.ff.ff.2",
    ]
    with safe_open(_REAL_DIT_CONVROT, framework="pt", device="cpu") as cr, \
            safe_open(_REAL_DIT_FP16, framework="pt", device="cpu") as fp16:
        for base in layers:
            weight_i8 = cr.get_tensor(base + ".weight")
            scale = cr.get_tensor(base + ".weight_scale")
            dequantized = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
                weight_i8, scale.reshape(-1, 1), 256, 0,  # dtype_code 0 == float32
            )
            reference = fp16.get_tensor(base + ".weight").to(torch.float32)
            rel_rms = (dequantized - reference).pow(2).mean().sqrt() / reference.pow(2).mean().sqrt()
            # Measured ~0.8-0.94% across sampled layers (see the design doc's
            # phase-13 status entry) -- 2% is a generous ceiling, not the
            # measured figure, so this test does not silently drift.
            assert rel_rms < 0.02, f"{base}: rel_rms={rel_rms.item()}"


def test_real_pruned_te_convrot_header_census_matches_the_design_docs_own_count():
    """328 dense (non-sidecar) tensor names -- the SAME census the design doc
    states for the dense pruned text encoder -- with 160 of the `.weight`
    entries additionally carrying a validated ConvRot marker (144 in the
    language model's 36 layers + 16 in the depth decoder's 4 layers, 4
    quantized module kinds each: self_attn.qkv_proj, self_attn.o_proj,
    mlp.gate_up_proj, mlp.down_proj)."""
    _skip_unless_real_files_present(_REAL_TE_CONVROT, _REAL_OFFICIAL)
    convrot_layers = loader._int8_convrot_source_layers(_REAL_TE_CONVROT, label="pruned text encoder")
    assert len(convrot_layers) == 160

    header = loader.read_safetensors_header(_REAL_TE_CONVROT)
    header.pop("__metadata__", None)
    dense_keys = [k for k in header if not (k.endswith(".comfy_quant") or k.endswith(".weight_scale"))]
    assert len(dense_keys) == 328

    with open(os.path.join(_REAL_OFFICIAL, "language_model", "config.json"), encoding="utf-8") as fh:
        lm_config = json.load(fh)
    from core.models.minimax_music3.pruned_text_encoder_remap import plan_pruned_text_encoder_keys

    plan = plan_pruned_text_encoder_keys(dense_keys, lm_config)
    assert plan.unrecognized == []


def test_real_pruned_te_convrot_builds_structurally_with_the_right_swap_counts_and_vocab_view():
    """Host RAM: the TE file is 9.20 GB; ConvRot weights stay packed, so the
    resident footprint tracks the file size plus the language model's dense
    (BF16) norms/vocab tables, not a full BF16 expansion of the whole 8B
    stack. Expect a peak under ~12 GB. No GPU needed."""
    _skip_unless_real_files_present(_REAL_TE_CONVROT, _REAL_OFFICIAL)
    from core.models.minimax_music3.vocab_view import PrunedVocabView, resolve_vocab_view

    language_model, rvq_depth_decoder, _depth_cfg = (
        loader.build_language_model_and_depth_decoder_from_pruned_flat_text_encoder(
            _REAL_TE_CONVROT, _REAL_OFFICIAL, torch.bfloat16,
        )
    )
    n_lm = sum(1 for m in language_model.modules() if isinstance(m, ConvRotInt8Linear))
    n_depth = sum(1 for m in rvq_depth_decoder.modules() if isinstance(m, ConvRotInt8Linear))
    # 36 LM layers x (o_proj + qkv-split-into-3 + gate_up-split-into-2 + down_proj) = 36 x 7 = 252.
    assert n_lm == 252
    # 4 depth layers x 7 = 28.
    assert n_depth == 28
    assert hasattr(language_model, "lm_head_pruned")
    assert hasattr(language_model.model, "embed_tokens_audio")
    assert isinstance(resolve_vocab_view(language_model), PrunedVocabView)
    stranded = [
        n for n, t in list(language_model.named_parameters()) + list(language_model.named_buffers())
        if getattr(t, "is_meta", False)
    ]
    assert stranded == []


def test_real_pruned_te_convrot_dequantizes_to_the_bf16_sibling_within_int8_noise_floor():
    """Same tensor-level A/B as the DiT test above, for the text encoder --
    both the language model's and the RVQ depth decoder's quantized layers,
    since the file composes pruned-vocabulary AND ConvRot together."""
    _skip_unless_real_files_present(_REAL_TE_CONVROT, _REAL_TE_BF16)
    try:
        import comfy_kitchen  # noqa: F401
    except ImportError:
        pytest.skip("comfy-kitchen is not installed")
    from safetensors import safe_open

    layers = [
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.35.mlp.down_proj",
        "model.audio_decoder.layers.0.self_attn.qkv_proj",
        "model.audio_decoder.layers.3.mlp.down_proj",
    ]
    with safe_open(_REAL_TE_CONVROT, framework="pt", device="cpu") as cr, \
            safe_open(_REAL_TE_BF16, framework="pt", device="cpu") as bf16:
        for base in layers:
            weight_i8 = cr.get_tensor(base + ".weight")
            scale = cr.get_tensor(base + ".weight_scale")
            dequantized = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
                weight_i8, scale.reshape(-1, 1), 256, 0,
            )
            reference = bf16.get_tensor(base + ".weight").to(torch.float32)
            rel_rms = (dequantized - reference).pow(2).mean().sqrt() / reference.pow(2).mean().sqrt()
            assert rel_rms < 0.02, f"{base}: rel_rms={rel_rms.item()}"


def test_real_end_to_end_load_composes_both_convrot_files():
    """The DiT ConvRot file through the ordinary model-path DiT selection,
    and the pruned TE ConvRot file through `text_encoder_file` -- both
    reachable in the SAME load, proving the two independently-landed paths
    compose. Host RAM: sum of the two builds above, so expect a peak under
    ~18 GB. No GPU needed."""
    _skip_unless_real_files_present(_REAL_DIT_CONVROT, _REAL_TE_CONVROT, _REAL_OFFICIAL)
    result = loader.load_minimax_music3_from_path(
        _REAL_DIT_CONVROT, torch_dtype=torch.bfloat16, text_encoder_file=_REAL_TE_CONVROT,
    )
    assert result["type"] == "minimax_music3"
    assert os.path.normcase(result["dit_path"]) == os.path.normcase(_REAL_DIT_CONVROT)
    assert os.path.normcase(result["text_encoder_path"]) == os.path.normcase(_REAL_TE_CONVROT)
    assert result["text_encoder_origin"] == "selected_external"
    n_dit = sum(1 for m in result["transformer"].modules() if isinstance(m, ConvRotInt8Linear))
    n_lm = sum(1 for m in result["language_model"].modules() if isinstance(m, ConvRotInt8Linear))
    n_depth = sum(1 for m in result["rvq_depth_decoder"].modules() if isinstance(m, ConvRotInt8Linear))
    assert (n_dit, n_lm, n_depth) == (216, 252, 28)


def test_real_detector_selects_flat_pruned_for_the_convrot_text_encoder():
    _skip_unless_real_files_present(_REAL_TE_CONVROT)
    assert loader.detect_minimax_music3_text_encoder_source(_REAL_TE_CONVROT) == "flat_pruned"
