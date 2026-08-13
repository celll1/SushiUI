"""MiniMax-H3: live component switching carries the text encoder's projection.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_te_switch_gates_test.py -v

A converted small Qwen3-VL conditions the DiT only through a projection trained
for its exact (width, tap) pair, so a switch that installs the encoder without
the projection -- or leaves a previous encoder's projection behind -- is a wrong
encode, not a degraded one. Everything here runs the REAL discovery and the real
pairing gates over header-sized files; only ``_build_text_encoder`` is stubbed,
because building the module is what ``minimax_h3_converted_te_test`` covers.
"""

import json
import os
import struct
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from core.models.minimax_h3 import loader  # noqa: E402
from core.models.components import component_switcher  # noqa: E402
from core.models.components.component_switcher import (  # noqa: E402
    ComponentSwitchFailed, switch_component,
)

TEXT_DIM = 128
CONVERTED_4B = "qwen3vl_4b_heretic_tap24_bf16.safetensors"
CONVERTED_8B = "qwen3vl_8b_instruct_tap24_bf16.safetensors"
RELEASED = "qwen3vl_32b_minimax_h3_bf16.safetensors"


def _converted_te(path, *, hidden=64, tap=2, vocab=32):
    """A converted small encoder at toy dims, with its own declaration."""
    heads, head_dim, kv_heads, ffn = 4, 16, 2, 128
    q, kv = heads * head_dim, kv_heads * head_dim
    shapes = {"model.embed_tokens.weight": (vocab, hidden)}
    for layer in range(tap):
        for suffix, shape in {
            "input_layernorm.weight": (hidden,),
            "post_attention_layernorm.weight": (hidden,),
            "self_attn.q_proj.weight": (q, hidden),
            "self_attn.k_proj.weight": (kv, hidden),
            "self_attn.v_proj.weight": (kv, hidden),
            "self_attn.o_proj.weight": (hidden, q),
            "self_attn.q_norm.weight": (head_dim,),
            "self_attn.k_norm.weight": (head_dim,),
            "mlp.gate_proj.weight": (ffn, hidden),
            "mlp.up_proj.weight": (ffn, hidden),
            "mlp.down_proj.weight": (hidden, ffn),
        }.items():
            shapes[f"model.layers.{layer}.{suffix}"] = shape
    declared = {
        "num_hidden_layers": tap, "modalities": "text", "source_size_label": "4B",
        "converter": "minimax_h3_te_gguf_convert", "hidden_size": hidden,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads, "head_dim": head_dim,
        "intermediate_size": ffn, "rms_norm_eps": 1e-06, "rope_theta": 5000000.0,
        "mrope_section": [4, 2, 2], "vocab_size": vocab,
    }
    save_file({name: torch.zeros(shape, dtype=torch.bfloat16) for name, shape in shapes.items()},
              str(path), metadata={"minimax_h3_te": json.dumps(declared)})
    return str(path)


def _header_only(path, keys, metadata=None):
    header = dict(keys)
    header["__metadata__"] = metadata or {"format": "pt"}
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
    return str(path)


_DENSE_SHAPES = {
    "self_attn.q_proj": [8192, 5120], "self_attn.k_proj": [1024, 5120],
    "self_attn.v_proj": [1024, 5120], "self_attn.o_proj": [5120, 8192],
    "mlp.gate_proj": [25600, 5120], "mlp.up_proj": [25600, 5120],
    "mlp.down_proj": [5120, 25600],
}


def _released_te(path):
    """The shipped 32B bf16 geometry, header only: it declares no dims."""
    keys = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [151936, 5120]}}
    for layer in range(50):
        for suffix, shape in _DENSE_SHAPES.items():
            keys[f"model.layers.{layer}.{suffix}.weight"] = {"dtype": "BF16", "shape": shape}
    return _header_only(path, keys)


def _projection(path, *, d_in=64, d_out=TEXT_DIM, tap=2, hidden=4):
    g = torch.Generator().manual_seed(3)
    tensors = {
        "W": torch.randn(d_in, d_out, generator=g),
        "mean_in": torch.randn(d_in, generator=g),
        "std_in": torch.rand(d_in, generator=g) + 0.5,
        "mean_out": torch.randn(d_out, generator=g),
        "std_out": torch.rand(d_out, generator=g) + 0.5,
        "sink_out": torch.randn(d_out, generator=g),
        "mlp.0.weight": torch.randn(hidden, d_in, generator=g),
        "mlp.0.bias": torch.randn(hidden, generator=g),
        "mlp.2.weight": torch.randn(d_out, hidden, generator=g),
        "mlp.2.bias": torch.randn(d_out, generator=g),
    }
    save_file(tensors, str(path), metadata={
        "d_in": str(d_in), "d_out": str(d_out), "tap": str(tap),
        "mlp_hidden": str(hidden), "mlp_depth": "1"})
    return str(path)


def _tree(tmp_path):
    """``<root>/{diffusion_models,text_encoders,clip_projections}`` with a DiT header."""
    for name in ("diffusion_models", "text_encoders", "clip_projections"):
        (tmp_path / name).mkdir()
    dit = _header_only(tmp_path / "diffusion_models" / "dit.safetensors",
                       {"condition_proj.weight": {"dtype": "BF16", "shape": [256, TEXT_DIM]}})
    return str(tmp_path), dit


class _Manager:
    def __init__(self, root, dit, components):
        self.current_model_info = {"type": "minimax_h3", "source": dit,
                                   "source_type": "safetensors"}
        self.current_model_info.update(loader.minimax_h3_te_model_info_fields(components))
        self.minimax_h3_components = components
        self.model_revision = 1
        self.component_revision = 1
        self.component_health = "ready"
        self._load_model_lock = threading.Lock()
        self.txt2img_pipeline = None
        self.vision_encoder = None


@pytest.fixture(autouse=True)
def _stub_module_build(monkeypatch):
    """Everything but the transformers module build is real."""
    built = []

    def build(te_path, official_dir):
        built.append(te_path)
        return SimpleNamespace(name=os.path.basename(te_path)), {"official": official_dir}

    monkeypatch.setattr(loader, "_build_text_encoder", build)
    monkeypatch.setattr(component_switcher, "_release_device_cache", lambda: None)
    import core.keep_hot as keep_hot
    monkeypatch.setattr(keep_hot, "clear_resident", lambda _manager: None)
    return built


def _components(root, dit, te_path, *, projection=None, text_only=False):
    return {
        "transformer": object(), "vae": object(), "audio_vae": object(),
        "text_encoder": object(), "text_encoder_config": {},
        "text_encoder_path": te_path, "text_encoder_origin": "architecture_default",
        "te_projection": projection, "te_text_only": text_only,
        "dit_path": dit, "official_dir": os.path.join(root, "official"),
    }


def _switch(manager, path):
    return switch_component(manager, "text_encoder", {
        "compatibility": "compatible", "switchable": True, "_path": path,
    }, manager.model_revision, manager.component_revision)


# ---------------------------------------------------------------------------
# The switch carries the projection, in both directions
# ---------------------------------------------------------------------------

def test_switching_to_a_converted_encoder_installs_its_resolved_projection(tmp_path):
    root, dit = _tree(tmp_path)
    released = _released_te(tmp_path / "text_encoders" / RELEASED)
    converted = _converted_te(tmp_path / "text_encoders" / CONVERTED_4B)
    projection = _projection(tmp_path / "clip_projections" / "mmh3-4b-clipproj-celeb-mlp.safetensors")
    manager = _Manager(root, dit, _components(root, dit, released))

    _switch(manager, converted)

    components = manager.minimax_h3_components
    assert components["text_encoder_path"] == converted
    assert components["te_projection"]["path"] == projection
    assert components["te_projection"]["spec"]["d_in"] == 64
    assert components["te_text_only"] is True
    assert manager.current_model_info["text_encoder_file"] == CONVERTED_4B
    assert manager.current_model_info["clip_projection_file"] == os.path.basename(projection)
    assert manager.current_model_info["te_text_only"] is True
    assert manager.component_health == "ready"


def test_switching_back_to_a_released_encoder_leaves_no_projection_behind(tmp_path):
    """The 32B's 5120-wide hidden state must not meet a 64-wide projection."""
    root, dit = _tree(tmp_path)
    released = _released_te(tmp_path / "text_encoders" / RELEASED)
    converted = _converted_te(tmp_path / "text_encoders" / CONVERTED_4B)
    _projection(tmp_path / "clip_projections" / "mmh3-4b-clipproj-celeb-mlp.safetensors")
    manager = _Manager(root, dit, _components(root, dit, released))

    _switch(manager, converted)
    assert manager.minimax_h3_components["te_projection"] is not None
    _switch(manager, released)

    components = manager.minimax_h3_components
    assert components["text_encoder_path"] == released
    assert components["te_projection"] is None
    assert components["te_text_only"] is False
    assert manager.current_model_info["clip_projection_file"] is None
    assert manager.current_model_info["te_text_only"] is False


# ---------------------------------------------------------------------------
# The three pairing gates, at switch time
# ---------------------------------------------------------------------------

def _bundle(tmp_path, te_path, dit, *, override=None):
    return loader.build_minimax_h3_text_encoder_bundle(
        te_path, None, root=str(tmp_path), dit_path=dit, projection_override=override)


def test_gate_refuses_a_projection_whose_d_in_is_another_encoders_width(tmp_path):
    root, dit = _tree(tmp_path)
    converted = _converted_te(tmp_path / "text_encoders" / CONVERTED_4B)
    other = _projection(tmp_path / "clip_projections" / "for-a-wider-encoder.safetensors", d_in=96)

    with pytest.raises(ValueError) as excinfo:
        _bundle(tmp_path, converted, dit, override=other)

    assert "d_in=96" in str(excinfo.value) and "hidden_size=64" in str(excinfo.value)


def test_gate_refuses_a_projection_whose_d_out_is_not_the_dits_conditioning(tmp_path):
    root, dit = _tree(tmp_path)
    converted = _converted_te(tmp_path / "text_encoders" / CONVERTED_4B)
    narrow = _projection(tmp_path / "clip_projections" / "for-another-dit.safetensors", d_out=64)

    with pytest.raises(ValueError) as excinfo:
        _bundle(tmp_path, converted, dit, override=narrow)

    assert "d_out=64" in str(excinfo.value) and f"text_dim={TEXT_DIM}" in str(excinfo.value)


def test_gate_refuses_a_projection_trained_on_another_tap(tmp_path):
    root, dit = _tree(tmp_path)
    converted = _converted_te(tmp_path / "text_encoders" / CONVERTED_4B)
    wrong_tap = _projection(tmp_path / "clip_projections" / "tap-24.safetensors", tap=24)

    with pytest.raises(ValueError) as excinfo:
        _bundle(tmp_path, converted, dit, override=wrong_tap)

    assert "tap=24" in str(excinfo.value) and "num_hidden_layers=2" in str(excinfo.value)


def test_a_converted_encoder_with_no_projection_is_refused_and_the_pair_restored(tmp_path):
    """The refusal names both widths, and the previous PAIR comes back whole."""
    root, dit = _tree(tmp_path)
    old = _converted_te(tmp_path / "text_encoders" / CONVERTED_4B)
    old_projection = _projection(
        tmp_path / "clip_projections" / "mmh3-4b-clipproj-celeb-mlp.safetensors")
    unpairable = _converted_te(tmp_path / "text_encoders" / CONVERTED_8B, hidden=96)
    manager = _Manager(root, dit, _components(
        root, dit, old,
        projection=loader.resolve_minimax_h3_te_projection(
            te_path=old, declared=loader._te_file_declaration(old), root=root, text_dim=TEXT_DIM),
        text_only=True))

    with pytest.raises(ComponentSwitchFailed) as excinfo:
        _switch(manager, unpairable)

    assert "d_in=96" in str(excinfo.value) and f"{TEXT_DIM}-wide" in str(excinfo.value)
    components = manager.minimax_h3_components
    assert components["text_encoder"] is not None
    assert components["text_encoder_path"] == old
    assert components["te_projection"]["path"] == old_projection
    assert components["te_text_only"] is True
    assert manager.current_model_info["clip_projection_file"] == os.path.basename(old_projection)
    assert manager.component_health == "ready"
    assert manager.component_revision == 1


def test_a_second_matching_projection_is_refused_rather_than_guessed(tmp_path):
    root, dit = _tree(tmp_path)
    released = _released_te(tmp_path / "text_encoders" / RELEASED)
    converted = _converted_te(tmp_path / "text_encoders" / CONVERTED_4B)
    _projection(tmp_path / "clip_projections" / "a.safetensors")
    _projection(tmp_path / "clip_projections" / "b.safetensors")
    manager = _Manager(root, dit, _components(root, dit, released))

    with pytest.raises(ComponentSwitchFailed) as excinfo:
        _switch(manager, converted)

    assert "Name one explicitly" in str(excinfo.value)
    components = manager.minimax_h3_components
    assert components["text_encoder_path"] == released
    assert components["te_projection"] is None
    assert components["te_text_only"] is False


# ---------------------------------------------------------------------------
# The listing the UI offers from
# ---------------------------------------------------------------------------

def test_listing_reports_the_pairing_a_switch_would_actually_form(tmp_path):
    root, dit = _tree(tmp_path)
    _released_te(tmp_path / "text_encoders" / RELEASED)
    _converted_te(tmp_path / "text_encoders" / CONVERTED_4B)
    _converted_te(tmp_path / "text_encoders" / CONVERTED_8B, hidden=96)
    projection = _projection(
        tmp_path / "clip_projections" / "mmh3-4b-clipproj-celeb-mlp.safetensors")

    choices = loader.describe_minimax_h3_text_encoder_choices(dit)
    entries = {os.path.basename(entry["path"]): entry for entry in choices["text_encoders"]}

    assert entries[RELEASED]["requires_projection"] is False
    assert entries[RELEASED]["projection"] is None
    paired = entries[CONVERTED_4B]
    assert paired["requires_projection"] is True
    assert paired["projection"] == projection
    unpaired = entries[CONVERTED_8B]
    assert unpaired["projection"] is None
    assert "d_in=96" in unpaired["projection_reason"]
