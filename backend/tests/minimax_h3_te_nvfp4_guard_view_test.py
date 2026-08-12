"""MiniMax-H3: the TE builder's guard view agrees with its swap view.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_te_nvfp4_guard_view_test.py -v

`minimax_h3_te_nvfp4_test.py` covers the marker validator and the header-side
probe on hand-built fixtures. What it could not see is the SECOND guard call,
inside `_build_text_encoder`: its view of the state dict dropped only the
`.comfy_quant` markers of the layers about to be swapped and left their
`.pre_quant_scale` siblings in, which the generic guard refuses outright --
so the real NVFP4/AWQ encoder was refused at load time by the very builder
that implements it. Synthetic key sets missed it because none of them carried
a `.pre_quant_scale` alongside a validated marker.

Header-only against the real file (no tensor reads beyond the ~50-byte
`.comfy_quant` markers the validator must decode, no model build), plus the
negative case that keeps the waiver narrow.
"""

import os
import sys

import pytest
import torch

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from core.models.common.quantized_checkpoint_guard import (  # noqa: E402
    UnsupportedQuantSemanticsError,
    refuse_unsupported_quant_semantics,
    unsupported_quant_semantics_report,
)
from core.models.minimax_h3.loader import (  # noqa: E402
    _HEADER_DTYPES,
    _h3_nvfp4_layers_from_markers,
    _int8_convrot_layers_from_markers,
    _rewrite_te_key,
    _te_guard_state_dict,
    read_safetensors_header,
)

REAL_ROOT = "M:/model/minimax_h3"
REAL_TE = os.path.join(
    REAL_ROOT, "text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
)


def test_real_nvfp4_awq_guard_view_contains_nothing_the_guard_refuses():
    """The real file, through the builder's own filter. Zero weight bytes read:
    every non-marker tensor is a zero-element dtype proxy, which is all the
    semantics guard looks at."""
    if not os.path.isfile(REAL_TE):
        pytest.skip(f"{REAL_TE} not present on this machine")
    from safetensors import safe_open

    header = read_safetensors_header(REAL_TE)
    header.pop("__metadata__", None)

    with safe_open(REAL_TE, framework="pt", device="cpu") as handle:
        nvfp4_source = _h3_nvfp4_layers_from_markers(handle, header, path=REAL_TE)
        convrot_source = _int8_convrot_layers_from_markers(handle, header, path=REAL_TE)
        state_dict = {
            _rewrite_te_key(key): (
                handle.get_tensor(key) if key.endswith(".comfy_quant")
                else torch.empty(0, dtype=_HEADER_DTYPES.get(
                    (entry or {}).get("dtype"), torch.float32))
            )
            for key, entry in header.items()
        }

    nvfp4_layer_configs = {_rewrite_te_key(k): dict(v) for k, v in nvfp4_source.items()}
    convrot_layer_configs = {_rewrite_te_key(k): dict(v) for k, v in convrot_source.items()}
    view = _te_guard_state_dict(
        state_dict,
        swappable_layer_configs={**convrot_layer_configs, **nvfp4_layer_configs},
        nvfp4_layer_configs=nvfp4_layer_configs,
    )

    assert nvfp4_layer_configs, "the real nvfp4_awq file validated no NVFP4 layer"
    in_file = [k for k in state_dict if k.endswith(".pre_quant_scale")]
    visible = [k for k in view if k.endswith(".pre_quant_scale")]
    hidden = len(in_file) - len(visible)
    assert in_file, "the real nvfp4_awq file carries no .pre_quant_scale at all"
    assert visible == [], (
        f"{len(visible)} .pre_quant_scale key(s) reach the semantics guard, which "
        f"refuses that suffix outright: {visible[:3]}")
    assert hidden == len(in_file)

    assert unsupported_quant_semantics_report(view) is None
    refuse_unsupported_quant_semantics(
        view, arch="MiniMax-H3", path=REAL_TE, label="text encoder")

    print(f"\n[guard view] validated NVFP4 Linears: {len(nvfp4_layer_configs)}, "
          f".pre_quant_scale hidden: {hidden}, left visible: {len(visible)}")


def test_pre_quant_scale_on_an_unvalidated_layer_is_still_refused():
    """The waiver is scoped to layers the marker validator confirmed. A
    `.pre_quant_scale` on anything else -- here a layer with no marker at all,
    which no module in this builder would apply one for -- must still refuse."""
    marker = torch.tensor(
        list(b'{"format": "nvfp4", "full_precision_matrix_mult": true}'), dtype=torch.uint8
    )
    validated = "model.language_model.layers.0.self_attn.o_proj"
    state_dict = {
        validated + ".weight": torch.empty(0, dtype=torch.uint8),
        validated + ".weight_scale": torch.empty(0, dtype=torch.float8_e4m3fn),
        validated + ".weight_scale_2": torch.empty(0, dtype=torch.float32),
        validated + ".comfy_quant": marker,
        validated + ".pre_quant_scale": torch.empty(0, dtype=torch.bfloat16),
        "model.language_model.layers.1.mlp.gate_proj.pre_quant_scale":
            torch.empty(0, dtype=torch.bfloat16),
    }
    nvfp4_layer_configs = {validated: {"has_pre_quant_scale": True}}
    view = _te_guard_state_dict(
        state_dict,
        swappable_layer_configs=dict(nvfp4_layer_configs),
        nvfp4_layer_configs=nvfp4_layer_configs,
    )
    assert validated + ".pre_quant_scale" not in view
    assert "model.language_model.layers.1.mlp.gate_proj.pre_quant_scale" in view
    with pytest.raises(UnsupportedQuantSemanticsError, match="pre_quant_scale"):
        refuse_unsupported_quant_semantics(
            view, arch="MiniMax-H3", path="fixture.safetensors", label="text encoder")


def test_a_convrot_layers_pre_quant_scale_is_not_waived():
    """A ConvRot layer's marker is hidden (its module keeps it), but
    `ConvRotInt8Linear` has no `pre_quant_scale` to apply, so one on such a
    layer must not ride the NVFP4 waiver."""
    marker = torch.tensor(
        list(b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'),
        dtype=torch.uint8,
    )
    layer = "model.language_model.layers.0.self_attn.o_proj"
    state_dict = {
        layer + ".weight": torch.empty(0, dtype=torch.int8),
        layer + ".weight_scale": torch.empty(0, dtype=torch.float32),
        layer + ".comfy_quant": marker,
        layer + ".pre_quant_scale": torch.empty(0, dtype=torch.bfloat16),
    }
    view = _te_guard_state_dict(
        state_dict,
        swappable_layer_configs={layer: {"convrot_groupsize": 256}},
        nvfp4_layer_configs={},
    )
    assert layer + ".comfy_quant" not in view
    with pytest.raises(UnsupportedQuantSemanticsError, match="pre_quant_scale"):
        refuse_unsupported_quant_semantics(
            view, arch="MiniMax-H3", path="fixture.safetensors", label="text encoder")
