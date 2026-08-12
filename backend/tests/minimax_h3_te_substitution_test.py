"""MiniMax-H3: applying a substituted text encoder's projection, and refusing what it cannot serve.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_te_substitution_test.py -v

P2 (``minimax_h3_converted_te_test``) proved a converted small encoder is
BUILT only when its trained projection is on disk. This file is about what
happens afterwards: the projection is actually applied at the one seam where
conditioning is born, the workflows a text-only encoder cannot serve are
refused at the route and again at the encode phase, and every generation that
used a substitution says so.

Synthetic tensors only; no model, no GPU, nothing large on disk.
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from api.error_handlers import ValidationError  # noqa: E402
from api.generation_status import (  # noqa: E402
    complete_generation, get_warnings, start_generation,
)
from api.generation_utils import resolve_minimax_h3_text_only_te_gate  # noqa: E402
from core.models.minimax_h3 import h3_pipeline_ops as ops  # noqa: E402
from core.models.minimax_h3.te_projection import (  # noqa: E402
    TE_SUBSTITUTION_WARNING_CODE,
    describe_te_substitution,
    load_te_projection,
    read_te_projection_spec,
)
from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin  # noqa: E402

# The real pairing whose agreement is recorded, so a test that reads the
# measured numbers reads the same strings production does.
ENCODER_4B = "qwen3vl_4b_heretic_tap24_bf16.safetensors"
PROJECTION_4B = "mmh3-4b-ClipProj-celeb-mlp.safetensors"

D_IN, TEXT_DIM = 6, 9


def _projection(tmp_path, *, d_in=D_IN, d_out=TEXT_DIM, name=PROJECTION_4B):
    """A tiny but structurally real projection (skip + 1-hidden-layer MLP)."""
    g = torch.Generator().manual_seed(7)
    tensors = {
        "W": torch.randn(d_in, d_out, generator=g),
        "mean_in": torch.randn(d_in, generator=g),
        "std_in": torch.rand(d_in, generator=g) + 0.5,
        "mean_out": torch.randn(d_out, generator=g),
        "std_out": torch.rand(d_out, generator=g) + 0.5,
        "sink_out": torch.randn(d_out, generator=g),
        "mlp.0.weight": torch.randn(4, d_in, generator=g),
        "mlp.0.bias": torch.randn(4, generator=g),
        "mlp.2.weight": torch.randn(d_out, 4, generator=g),
        "mlp.2.bias": torch.randn(d_out, generator=g),
    }
    path = str(tmp_path / name)
    save_file(tensors, path, metadata={"d_in": str(d_in), "d_out": str(d_out), "tap": "24",
                                       "mlp_hidden": "4", "mlp_depth": "1"})
    return load_te_projection(read_te_projection_spec(path))


def _components(tmp_path, *, projection=None, text_only=False, encoder=ENCODER_4B):
    return {
        "text_encoder_path": str(tmp_path / encoder),
        "te_text_only": text_only,
        "te_projection": projection,
        "transformer_config": {"text_dim": TEXT_DIM},
    }


class _Backend(MiniMaxH3Mixin):
    """The mixin alone: every method exercised here reads only its arguments."""


# ---------------------------------------------------------------------------
# 1. The projection, applied at the encode seam
# ---------------------------------------------------------------------------

def test_projection_preserves_the_token_count_and_produces_text_dim(tmp_path):
    """S is load-bearing geometry (media rows' rotary clock starts at S)."""
    projection = _projection(tmp_path)
    hidden = torch.randn(1, 11, D_IN, dtype=torch.bfloat16)

    projected = ops.project_prompt_embeds(hidden, projection, text_dim=TEXT_DIM, device="cpu")

    assert projected.shape == (1, 11, TEXT_DIM)
    assert projected.dtype == torch.bfloat16
    assert projected.device.type == "cpu"
    assert torch.isfinite(projected).all()


def test_projection_matches_the_reference_forward(tmp_path):
    """Same arithmetic as ``apply_te_projection``, only cast and moved."""
    from core.models.minimax_h3.te_projection import apply_te_projection

    projection = _projection(tmp_path)
    hidden = torch.randn(1, 5, D_IN)

    expected = apply_te_projection(hidden, projection).to("cpu", torch.bfloat16)
    got = ops.project_prompt_embeds(hidden, projection, text_dim=TEXT_DIM, device="cpu")
    assert torch.equal(got, expected)


def test_non_finite_conditioning_is_refused(tmp_path):
    """The same assertion the unprojected encoder output already gets."""
    projection = _projection(tmp_path)
    hidden = torch.randn(1, 4, D_IN)
    hidden[0, 2, 0] = float("inf")

    with pytest.raises(RuntimeError, match=r"non-finite conditioning"):
        ops.project_prompt_embeds(hidden, projection, text_dim=TEXT_DIM, device="cpu")


def test_a_projection_that_does_not_fit_the_dit_is_refused(tmp_path):
    """The output width is checked against the DiT's own text_dim."""
    projection = _projection(tmp_path, d_out=TEXT_DIM + 1, name="wrong-width.safetensors")
    hidden = torch.randn(1, 4, D_IN)

    with pytest.raises(RuntimeError, match=r"text_dim=9"):
        ops.project_prompt_embeds(hidden, projection, text_dim=TEXT_DIM, device="cpu")


# ---------------------------------------------------------------------------
# 2. The encode-phase wiring: applied, recorded, warned about
# ---------------------------------------------------------------------------

def test_the_encode_seam_projects_records_and_warns(tmp_path):
    projection = _projection(tmp_path)
    components = _components(tmp_path, projection=projection, text_only=True)
    params = {}
    hidden = torch.randn(1, 7, D_IN, dtype=torch.bfloat16)

    generation_id = start_generation("txt2vid")
    try:
        projected = _Backend()._minimax_h3_project_prompt_embeds(
            hidden, components, params, device="cpu")
        warnings = get_warnings(generation_id)
    finally:
        complete_generation(generation_id=generation_id)

    assert projected.shape == (1, 7, TEXT_DIM)
    assert params["text_encoder_file"] == ENCODER_4B
    assert params["clip_projection_file"] == PROJECTION_4B

    entries = [w for w in warnings if w["code"] == TE_SUBSTITUTION_WARNING_CODE]
    assert len(entries) == 1
    message = entries[0]["message"]
    assert message == describe_te_substitution(components["text_encoder_path"], projection["path"])
    assert ENCODER_4B in message and PROJECTION_4B in message
    # The measured agreement, from the one table (gate G0c).
    assert "mean-removed cosine 0.826" in message
    assert "qwen3vl_32b_minimax_h3_int8_convrot.safetensors" in message


def test_a_released_encoder_is_left_completely_alone(tmp_path):
    """No projection, no warning, and the same tensor object back."""
    components = _components(tmp_path, projection=None, text_only=False,
                             encoder="qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    params = {}
    hidden = torch.randn(1, 3, TEXT_DIM, dtype=torch.bfloat16)

    generation_id = start_generation("txt2vid")
    try:
        result = _Backend()._minimax_h3_project_prompt_embeds(
            hidden, components, params, device="cpu")
        warnings = get_warnings(generation_id)
    finally:
        complete_generation(generation_id=generation_id)

    assert result is hidden
    assert "clip_projection_file" not in params
    assert params["text_encoder_file"] == "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    assert not warnings


def test_an_unmeasured_pairing_says_so_rather_than_borrowing_a_number(tmp_path):
    projection = _projection(tmp_path, name="some-other-proj.safetensors")
    components = _components(tmp_path, projection=projection, text_only=True)

    generation_id = start_generation("txt2vid")
    try:
        _Backend()._minimax_h3_project_prompt_embeds(
            torch.randn(1, 2, D_IN), components, {}, device="cpu")
        warnings = get_warnings(generation_id)
    finally:
        complete_generation(generation_id=generation_id)

    assert "No agreement with a released encoder is recorded" in warnings[0]["message"]
    assert "cosine" not in warnings[0]["message"]


# ---------------------------------------------------------------------------
# 3. The refusals
# ---------------------------------------------------------------------------

def test_gate_passes_a_released_encoder_through(tmp_path):
    resolve_minimax_h3_text_only_te_gate(
        _components(tmp_path, projection=None, text_only=False),
        workflow="reference conditioning", has_vision_references=True)


def test_gate_refuses_references_for_the_vision_tower_reason(tmp_path):
    components = _components(tmp_path, projection=_projection(tmp_path), text_only=True)

    with pytest.raises(ValidationError) as excinfo:
        resolve_minimax_h3_text_only_te_gate(
            components, workflow="reference conditioning (/generate/ref2vid)",
            has_vision_references=True)

    detail = str(excinfo.value.detail)
    assert "deepstack" in detail and "get_rope_index" in detail
    assert ENCODER_4B in str(excinfo.value.message)


def test_gate_refuses_keyframes_naming_what_was_measured(tmp_path):
    components = _components(tmp_path, projection=_projection(tmp_path), text_only=True)

    with pytest.raises(ValidationError) as excinfo:
        resolve_minimax_h3_text_only_te_gate(
            components, workflow="keyframe/audio conditioning (/generate/img2vid)")

    detail = str(excinfo.value.detail)
    assert "prompt-only presentations" in detail
    assert "keyframe/audio conditioning (/generate/img2vid)" in detail
    assert "/generate/txt2vid" in detail


@pytest.mark.parametrize("route", ["generate_img2vid", "generate_ref2vid",
                                   "generate_outpaint_video", "generate_inpaint_video"])
def test_every_conditioned_video_route_runs_the_gate(route):
    """The refusal must arrive before the model load, not after it."""
    from api import routes

    source = inspect.getsource(getattr(routes, route))
    assert "resolve_minimax_h3_text_only_te_gate(" in source


def test_the_encode_phase_refuses_defensively(tmp_path):
    """A caller that bypasses the route hits the same decision."""
    backend = _Backend()
    backend.minimax_h3_components = _components(
        tmp_path, projection=_projection(tmp_path), text_only=True)
    backend.minimax_h3_components["variant"] = "fl2va"

    with pytest.raises(ValidationError) as excinfo:
        backend._generate_minimax_h3({"prompt": "a"}, keyframes=[("first", object())],
                                     label="img2vid")
    assert "text-only" in str(excinfo.value.message)


def test_the_encode_phase_lets_a_prompt_only_request_through(tmp_path):
    """t2va is what the substitution IS admitted for; the gate must not fire."""
    backend = _Backend()
    backend.minimax_h3_components = _components(
        tmp_path, projection=_projection(tmp_path), text_only=True)

    # Runs past the gate and dies later, in the geometry it was given no
    # components for -- any failure that is not the refusal proves it passed.
    with pytest.raises(Exception) as excinfo:
        backend._generate_minimax_h3({"prompt": "a"}, label="txt2vid")
    assert not isinstance(excinfo.value, ValidationError)


# ---------------------------------------------------------------------------
# 4. Component switching: the refusal, and the stale-projection failure mode
# ---------------------------------------------------------------------------

def test_switching_to_a_converted_encoder_is_still_refused(tmp_path):
    """P2's refusal, re-run here because P3 is what would have relaxed it.

    The switch entry point returns ``(model, config)`` and has nowhere to put a
    projection, so it must refuse rather than install an encoder whose hidden
    state nothing would map.
    """
    import json
    import struct

    from core.models.minimax_h3 import loader

    dims = {"hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "intermediate_size": 128, "rms_norm_eps": 1e-06,
            "rope_theta": 5000000.0, "mrope_section": [4, 2, 2], "vocab_size": 32}
    tap = 2
    shapes = {"model.embed_tokens.weight": [dims["vocab_size"], dims["hidden_size"]]}
    for layer in range(tap):
        for suffix, shape in {
            "input_layernorm.weight": [64], "post_attention_layernorm.weight": [64],
            "self_attn.q_proj.weight": [64, 64], "self_attn.k_proj.weight": [32, 64],
            "self_attn.v_proj.weight": [32, 64], "self_attn.o_proj.weight": [64, 64],
            "self_attn.q_norm.weight": [16], "self_attn.k_norm.weight": [16],
            "mlp.gate_proj.weight": [128, 64], "mlp.up_proj.weight": [128, 64],
            "mlp.down_proj.weight": [64, 128],
        }.items():
            shapes[f"model.layers.{layer}.{suffix}"] = shape
    header = {name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
              for name, shape in shapes.items()}
    header["__metadata__"] = {"minimax_h3_te": json.dumps(
        {"num_hidden_layers": tap, "modalities": "text", **dims})}
    blob = json.dumps(header).encode("utf-8")
    path = tmp_path / ENCODER_4B
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)

    with pytest.raises(ValueError) as excinfo:
        loader.build_minimax_h3_text_encoder(str(path), None)
    message = str(excinfo.value)
    assert ENCODER_4B in message
    assert "trained projection" in message and "component switching does not carry" in message


def test_a_stale_projection_is_refused_after_a_switch_back(tmp_path):
    """Switching converted -> released leaves `te_projection` behind.

    The switcher replaces the encoder in place and does not clear the paired
    projection, so the released encoder's 5120-wide hidden state would meet a
    projection expecting the converted width. `apply_te_projection`'s `d_in`
    guard fails closed rather than projecting garbage.
    """
    projection = _projection(tmp_path)
    components = _components(tmp_path, projection=projection, text_only=True)
    # What the switch does to the dict, and what it leaves alone.
    components["text_encoder_path"] = str(tmp_path / "qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    released_hidden = torch.randn(1, 4, TEXT_DIM)

    # The seam names the cause...
    with pytest.raises(RuntimeError, match=r"no longer belong to each other"):
        _Backend()._minimax_h3_project_prompt_embeds(
            released_hidden, components, {}, device="cpu")
    # ...and the projection's own guard fails closed regardless of the caller.
    from core.models.minimax_h3.te_projection import apply_te_projection

    with pytest.raises(ValueError, match=rf"takes d_in={D_IN} but the hidden state is {TEXT_DIM}-wide"):
        apply_te_projection(released_hidden, projection)
