"""SenseNova mixed ConvRot + plain-int8 load-path coverage.

No model and no GPU: everything here exercises the dict-filtering helper
and the debug ablation classifier directly against synthetic state dicts.
This is the counterpart to ``minimax_h3_int8_convrot_test.py`` for the
SenseNova loader's own three-dict split, which -- unlike H3's real files --
must actually cope with a MIXED ConvRot + plain-int8 checkpoint (see
``sensenova/loader.py``'s ``_sensenova_quant_dict_views`` docstring).
"""

import os
import sys

import pytest


BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from core.models.sensenova.loader import (  # noqa: E402
    _SENSENOVA_CONVROT_ABLATION_GROUPS,
    _sensenova_convrot_ablation_group,
    _sensenova_quant_dict_views,
)


def _fake_sd():
    """One ConvRot-markered layer, one plain-int8 layer -- the mixed-checkpoint case."""
    return {
        "language_model.model.layers.0.self_attn.q_proj.weight": "convrot-weight",
        "language_model.model.layers.0.self_attn.q_proj.weight_scale": "convrot-scale",
        "language_model.model.layers.0.self_attn.q_proj.comfy_quant": "convrot-marker",
        "language_model.model.layers.0.self_attn.k_proj.weight": "plain-weight",
        "language_model.model.layers.0.self_attn.k_proj.weight_scale": "plain-scale",
        "language_model.model.layers.0.self_attn.k_proj.comfy_quant": "plain-marker",
        "language_model.model.norm.weight": "unquantized",
    }


CONVROT_LAYER = "language_model.model.layers.0.self_attn.q_proj"
PLAIN_LAYER = "language_model.model.layers.0.self_attn.k_proj"


def test_convrot_layer_is_excluded_entirely_from_plain_sd():
    sd = _fake_sd()
    source_layers = {CONVROT_LAYER: {"convrot_groupsize": 256}}

    _guard_sd, plain_sd, _sd_for_load = _sensenova_quant_dict_views(sd, source_layers)

    # The ConvRot layer's weight, scale AND marker must all be absent --
    # leaving even the marker behind would make ``is_int8_state_dict`` refuse
    # the whole checkpoint on its own "convrot: true" declaration.
    assert f"{CONVROT_LAYER}.weight" not in plain_sd
    assert f"{CONVROT_LAYER}.weight_scale" not in plain_sd
    assert f"{CONVROT_LAYER}.comfy_quant" not in plain_sd
    # The plain layer and the unquantized tensor are untouched.
    assert f"{PLAIN_LAYER}.weight" in plain_sd
    assert f"{PLAIN_LAYER}.weight_scale" in plain_sd
    assert f"{PLAIN_LAYER}.comfy_quant" in plain_sd
    assert "language_model.model.norm.weight" in plain_sd


def test_convrot_layer_is_fully_present_in_sd_for_load():
    sd = _fake_sd()
    source_layers = {CONVROT_LAYER: {"convrot_groupsize": 256}}

    _guard_sd, _plain_sd, sd_for_load = _sensenova_quant_dict_views(sd, source_layers)

    # The actual tensors the model loads: the ConvRot layer keeps its
    # weight, scale AND marker (the marker becomes live module state).
    assert f"{CONVROT_LAYER}.weight" in sd_for_load
    assert f"{CONVROT_LAYER}.weight_scale" in sd_for_load
    assert f"{CONVROT_LAYER}.comfy_quant" in sd_for_load


def test_plain_layer_marker_is_dropped_from_sd_for_load():
    sd = _fake_sd()
    source_layers = {CONVROT_LAYER: {"convrot_groupsize": 256}}

    _guard_sd, _plain_sd, sd_for_load = _sensenova_quant_dict_views(sd, source_layers)

    # A plain layer's provenance marker has served its purpose at census
    # time and must not reach ``load_state_dict`` (``Int8Linear`` has no
    # ``comfy_quant`` buffer to receive it).
    assert f"{PLAIN_LAYER}.comfy_quant" not in sd_for_load
    # Its weight/scale are unaffected.
    assert f"{PLAIN_LAYER}.weight" in sd_for_load
    assert f"{PLAIN_LAYER}.weight_scale" in sd_for_load


def test_guard_sd_drops_only_the_validated_convrot_marker():
    sd = _fake_sd()
    source_layers = {CONVROT_LAYER: {"convrot_groupsize": 256}}

    guard_sd, _plain_sd, _sd_for_load = _sensenova_quant_dict_views(sd, source_layers)

    # The already-validated ConvRot marker must not be re-checked by
    # declared-semantics refusal.
    assert f"{CONVROT_LAYER}.comfy_quant" not in guard_sd
    # Every other key, including the plain layer's own (unvalidated) marker,
    # still reaches the refusal check.
    assert f"{PLAIN_LAYER}.comfy_quant" in guard_sd
    assert f"{CONVROT_LAYER}.weight" in guard_sd
    assert f"{PLAIN_LAYER}.weight" in guard_sd


def test_no_convrot_layers_is_a_pure_pass_through():
    sd = _fake_sd()

    guard_sd, plain_sd, sd_for_load = _sensenova_quant_dict_views(sd, {})

    # With no ConvRot source layers pre-validated, nothing is exempt: every
    # ``.comfy_quant`` marker still reaches declared-semantics refusal
    # (guard_sd keeps everything, since none of them was already accepted).
    assert guard_sd == sd
    # plain_sd keeps everything too (there is no ConvRot layer to exclude).
    assert plain_sd == sd
    # sd_for_load drops every ``.comfy_quant`` key: with no ConvRot layer to
    # retain as live module state, every provenance marker has served its
    # purpose and none should reach ``load_state_dict``.
    assert not any(key.endswith(".comfy_quant") for key in sd_for_load)
    assert f"{CONVROT_LAYER}.weight" in sd_for_load
    assert f"{PLAIN_LAYER}.weight" in sd_for_load


# --- Debug ablation group classifier -----------------------------------


@pytest.mark.parametrize(
    "module_path,expected",
    [
        ("language_model.model.layers.0.self_attn.o_proj_mot_gen", "gen_o_proj"),
        ("language_model.model.layers.0.self_attn.o_proj", "understanding_o_proj"),
        ("language_model.model.layers.0.self_attn.q_proj_mot_gen", "gen_attn_qkv"),
        ("language_model.model.layers.0.self_attn.k_proj_mot_gen", "gen_attn_qkv"),
        ("language_model.model.layers.0.self_attn.v_proj_mot_gen", "gen_attn_qkv"),
        ("language_model.model.layers.0.self_attn.q_proj", "understanding_attn_qkv"),
        ("language_model.model.layers.0.self_attn.k_proj", "understanding_attn_qkv"),
        ("language_model.model.layers.0.self_attn.v_proj", "understanding_attn_qkv"),
        ("language_model.model.layers.0.mlp_mot_gen.gate_proj", "gen_mlp"),
        ("language_model.model.layers.0.mlp.gate_proj", "understanding_mlp"),
        ("language_model.model.embed_tokens", None),
        ("fm_modules.fm_head.proj", None),
        ("language_model.model.norm", None),
    ],
)
def test_ablation_group_classifier(module_path, expected):
    assert _sensenova_convrot_ablation_group(module_path) == expected


def test_ablation_groups_tuple_matches_classifier_vocabulary():
    # Every group the classifier can return is a declared, valid group name
    # (the env-var parser validates requested groups against this tuple).
    assert set(_SENSENOVA_CONVROT_ABLATION_GROUPS) == {
        "gen_attn_qkv", "gen_o_proj", "gen_mlp",
        "understanding_attn_qkv", "understanding_o_proj", "understanding_mlp",
    }


# --- Debug ablation env-var parsing (unknown group name) ----------------


def test_ablation_unknown_group_name_raises(monkeypatch):
    import torch

    from core.models.sensenova.loader import _apply_sensenova_convrot_dequant_ablation

    monkeypatch.setenv("SUSHI_SENSENOVA_CONVROT_DEQUANT", "not_a_real_group")
    model = torch.nn.Module()
    with pytest.raises(ValueError, match="unknown group"):
        _apply_sensenova_convrot_dequant_ablation(model)


def test_ablation_unset_env_var_is_a_silent_no_op(monkeypatch):
    import torch

    from core.models.sensenova.loader import _apply_sensenova_convrot_dequant_ablation

    monkeypatch.delenv("SUSHI_SENSENOVA_CONVROT_DEQUANT", raising=False)
    model = torch.nn.Module()
    # Must not raise and must not touch anything -- the function's own
    # docstring promises a no-op when unset.
    _apply_sensenova_convrot_dequant_ablation(model)


def test_ablation_forces_only_the_requested_group(monkeypatch):
    import torch

    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    from core.models.sensenova.loader import _apply_sensenova_convrot_dequant_ablation

    marker_numel = 8

    def _layer():
        return ConvRotInt8Linear(
            256, 8, False, torch.bfloat16, convrot_groupsize=256, marker_numel=marker_numel,
        )

    # Nested under a "layer0" module so the dotted path has something before
    # "self_attn" -- the classifier's suffix checks require a preceding dot
    # (real paths look like "...layers.0.self_attn.o_proj").
    model = torch.nn.Module()
    model.layer0 = torch.nn.Module()
    model.layer0.self_attn = torch.nn.Module()
    setattr(model.layer0.self_attn, "o_proj", _layer())
    setattr(model.layer0.self_attn, "o_proj_mot_gen", _layer())

    monkeypatch.setenv("SUSHI_SENSENOVA_CONVROT_DEQUANT", "understanding_o_proj")
    _apply_sensenova_convrot_dequant_ablation(model)

    assert model.layer0.self_attn.o_proj._force_dequant is True
    assert model.layer0.self_attn.o_proj_mot_gen._force_dequant is False


def test_reshape_convrot_scales_touches_only_marker_validated_layers():
    """`[out, 1]` -> `(out,)` for ConvRot layers; plain int8 scales left alone.

    The narrow scope is what makes the reshape legitimate rather than the
    blanket squeeze ``quantized_checkpoint_guard``'s docstring warns about.
    """
    import torch

    from core.models.sensenova.loader import _reshape_convrot_scales

    convrot = "language_model.model.layers.0.self_attn.q_proj"
    plain = "language_model.model.layers.0.self_attn.k_proj"
    sd = {
        f"{convrot}.weight_scale": torch.zeros(8, 1),
        f"{plain}.weight_scale": torch.zeros(8, 1),
    }

    reshaped = _reshape_convrot_scales(sd, {convrot: {"convrot_groupsize": 256}})

    assert reshaped == 1
    assert tuple(sd[f"{convrot}.weight_scale"].shape) == (8,)
    assert tuple(sd[f"{plain}.weight_scale"].shape) == (8, 1)


def test_reshape_convrot_scales_is_idempotent():
    """An already-1-D scale (the marker validator accepts both) is left as-is."""
    import torch

    from core.models.sensenova.loader import _reshape_convrot_scales

    layer = "language_model.model.layers.0.self_attn.q_proj"
    sd = {f"{layer}.weight_scale": torch.zeros(8)}

    assert _reshape_convrot_scales(sd, {layer: {"convrot_groupsize": 256}}) == 0
    assert tuple(sd[f"{layer}.weight_scale"].shape) == (8,)
