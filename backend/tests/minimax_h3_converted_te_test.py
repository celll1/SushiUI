"""MiniMax-H3: loading a converted small Qwen3-VL as the text encoder.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_converted_te_test.py -v

``te_gguf_convert`` produces a file with the shipped 32B's key names but its own
(2560/4096-wide, 24-block, text-only) geometry. Four things have to hold, and
each of them is a way this could load clean and be wrong:

* the module is built from the FILE's declared dims, not from the 32B's
  ``official/text_encoder/config.json``;
* the installed tensors are shape-checked, because ``load_state_dict(assign=True)``
  is not;
* a text-only file's absent vision tower is tolerated -- and only then;
* auto-selection never reaches such a file; only an explicit override does.

The model built here is a real ``Qwen3VLForConditionalGeneration`` at toy dims
(hidden 64, 2 layers, vocab 32), so the whole build path runs on a few hundred
kilobytes.
"""

import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from core.models.minimax_h3 import loader  # noqa: E402
from core.models.minimax_h3.loader import (  # noqa: E402
    MINIMAX_H3_TE_PATTERNS,
    detect_minimax_h3_layout,
)

BF16_NAME = MINIMAX_H3_TE_PATTERNS[1]
CONVERTED_NAME = "qwen3vl_4b_heretic_tap24_bf16.safetensors"

# Toy dims for the built model. `OFFICIAL_HIDDEN` deliberately differs from
# `DIMS["hidden_size"]` so "which config was used" is observable.
OFFICIAL_HIDDEN = 128
DIMS = {
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000.0,
    "mrope_section": [4, 2, 2],
    "vocab_size": 32,
}
TAP = 2


def _official_tree(tmp_path, hidden=OFFICIAL_HIDDEN):
    """A config-only tree shaped like MiniMax's, at toy dims."""
    official = tmp_path / "official"
    (official / "text_encoder").mkdir(parents=True)
    config = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "image_token_id": 5, "video_token_id": 6,
        "vision_start_token_id": 7, "vision_end_token_id": 8,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_vl_text",
            "attention_bias": False, "hidden_act": "silu",
            "hidden_size": hidden, "intermediate_size": 256,
            "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
            "num_hidden_layers": 2, "vocab_size": 32, "rms_norm_eps": 1e-06,
            "rope_theta": 5000000, "max_position_embeddings": 4096,
            "rope_scaling": {"mrope_interleaved": True, "mrope_section": [4, 2, 2],
                             "rope_type": "default"},
        },
        "vision_config": {
            "model_type": "qwen3_vl", "depth": 2, "hidden_size": 32,
            "intermediate_size": 32, "num_heads": 2, "in_channels": 3,
            "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
            "num_position_embeddings": 64, "out_hidden_size": hidden,
            "deepstack_visual_indexes": [0], "hidden_act": "gelu_pytorch_tanh",
        },
    }
    (official / "text_encoder" / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(official)


def _converted_tensors(dims=DIMS, tap=TAP, bad_key=None, bad_shape=None):
    q = dims["num_attention_heads"] * dims["head_dim"]
    kv = dims["num_key_value_heads"] * dims["head_dim"]
    h, ffn = dims["hidden_size"], dims["intermediate_size"]
    tensors = {"model.embed_tokens.weight": (dims["vocab_size"], h)}
    for layer in range(tap):
        for suffix, shape in {
            "input_layernorm.weight": (h,),
            "post_attention_layernorm.weight": (h,),
            "self_attn.q_proj.weight": (q, h),
            "self_attn.k_proj.weight": (kv, h),
            "self_attn.v_proj.weight": (kv, h),
            "self_attn.o_proj.weight": (h, q),
            "self_attn.q_norm.weight": (dims["head_dim"],),
            "self_attn.k_norm.weight": (dims["head_dim"],),
            "mlp.gate_proj.weight": (ffn, h),
            "mlp.up_proj.weight": (ffn, h),
            "mlp.down_proj.weight": (h, ffn),
        }.items():
            tensors[f"model.layers.{layer}.{suffix}"] = shape
    if bad_key is not None:
        tensors[bad_key] = bad_shape
    return {name: torch.zeros(shape, dtype=torch.bfloat16) for name, shape in tensors.items()}


def _write_converted_te(path, *, dims=DIMS, tap=TAP, declare_dims=True, text_only=True,
                        bad_key=None, bad_shape=None):
    declared = {"num_hidden_layers": tap,
                "output": f"unnormalized_hidden_after_layer_{tap}",
                "converter": "minimax_h3_te_gguf_convert", "tap": tap,
                "source_size_label": "4B"}
    if declare_dims:
        declared.update(dims)
    if text_only:
        declared["modalities"] = "text"
    save_file(_converted_tensors(dims, tap, bad_key, bad_shape), str(path),
              metadata={"minimax_h3_te": json.dumps(declared)})
    return str(path)


def _write_header(path, keys, metadata=None):
    """A safetensors file carrying only its JSON header; zero tensor bytes."""
    header = dict(keys)
    header["__metadata__"] = metadata or {"format": "pt"}
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)


def _converted_header_only(path, *, tap=TAP, dims=DIMS):
    keys = {name: {"dtype": "BF16", "shape": list(tensor.shape), "data_offsets": [0, 0]}
            for name, tensor in _converted_tensors(dims, tap).items()}
    declared = {"num_hidden_layers": tap, "modalities": "text", "source_size_label": "4B",
                "converter": "minimax_h3_te_gguf_convert", **dims}
    _write_header(path, keys, {"minimax_h3_te": json.dumps(declared)})
    return str(path)


def _drop(model):
    """Release the file mapping so the next build of the same path is allowed."""
    import gc

    del model
    gc.collect()


# ---------------------------------------------------------------------------
# Which config the module is built from
# ---------------------------------------------------------------------------

def test_declared_dims_build_the_module_at_the_files_own_width(tmp_path, capsys):
    official = _official_tree(tmp_path)
    te_path = _write_converted_te(tmp_path / CONVERTED_NAME)

    model, config = loader._build_text_encoder(te_path, official)
    try:
        assert config.text_config.hidden_size == DIMS["hidden_size"]
        assert config.text_config.num_hidden_layers == TAP
        assert config.text_config.intermediate_size == DIMS["intermediate_size"]
        assert config.text_config.num_attention_heads == DIMS["num_attention_heads"]
        installed = dict(model.named_parameters())
        assert tuple(installed["model.language_model.layers.0.self_attn.q_proj.weight"].shape) == (
            DIMS["num_attention_heads"] * DIMS["head_dim"], DIMS["hidden_size"])
        out = capsys.readouterr().out
        assert "geometry from the file's own minimax_h3_te metadata" in out
    finally:
        _drop(model)


def test_a_file_without_declared_dims_uses_the_official_config(tmp_path):
    """The shipped 32B spelling: geometry comes from ``official/``.

    A converted file put through that path does not fit it, and torch 2.10's
    ``load_state_dict(assign=True)`` says so ("size mismatch", measured) rather
    than installing a 64-wide tensor into a 128-wide skeleton. So this is the
    loud failure the declared-dims path exists to avoid, not a silent one.
    """
    official = _official_tree(tmp_path)
    te_path = _write_converted_te(tmp_path / "undeclared.safetensors", declare_dims=False)

    with pytest.raises(RuntimeError, match=r"size mismatch"):
        loader._build_text_encoder(te_path, official)


def test_official_config_is_used_verbatim_when_the_file_declares_no_dims(tmp_path):
    """Same path, a file that DOES match the 32B-shaped config: nothing is overridden."""
    official = _official_tree(tmp_path)
    official_dims = dict(DIMS, hidden_size=OFFICIAL_HIDDEN, intermediate_size=256,
                         num_attention_heads=8, num_key_value_heads=2)
    te_path = _write_converted_te(tmp_path / "official_shaped.safetensors",
                                  dims=official_dims, declare_dims=False)

    model, config = loader._build_text_encoder(te_path, official)
    try:
        assert config.text_config.hidden_size == OFFICIAL_HIDDEN
        assert config.text_config.num_attention_heads == 8
    finally:
        _drop(model)


def test_shape_verification_catches_a_mis_declared_file(tmp_path):
    """The declared dims say 64; one tensor is 48 wide."""
    official = _official_tree(tmp_path)
    te_path = _write_converted_te(
        tmp_path / "mis_declared.safetensors",
        bad_key="model.layers.0.mlp.up_proj.weight",
        bad_shape=(DIMS["intermediate_size"], 48))

    with pytest.raises(RuntimeError) as excinfo:
        loader._build_text_encoder(te_path, official)
    message = str(excinfo.value)
    assert "mlp.up_proj.weight" in message
    assert "file (128, 48)" in message and "model (128, 64)" in message


# ---------------------------------------------------------------------------
# Text-only tolerance
# ---------------------------------------------------------------------------

def test_text_only_tolerates_the_absent_vision_tower(tmp_path, capsys):
    official = _official_tree(tmp_path)
    te_path = _write_converted_te(tmp_path / "text_only.safetensors")

    model, _config = loader._build_text_encoder(te_path, official)
    try:
        vision_meta = [name for name, tensor in model.named_parameters()
                       if name.startswith("model.visual.") and tensor.is_meta]
        assert vision_meta, "the vision tower should still be present, on the meta device"
        assert not [name for name, tensor in model.named_parameters()
                    if tensor.is_meta and not name.startswith("model.visual.")]
        assert "vision-tower tensor(s) left on the meta device" in capsys.readouterr().out
    finally:
        _drop(model)


def test_a_file_that_does_not_declare_text_only_still_requires_the_vision_tower(tmp_path):
    official = _official_tree(tmp_path)
    te_path = _write_converted_te(tmp_path / "no_modalities.safetensors", text_only=False)

    with pytest.raises(RuntimeError, match=r"missing key\(s\)"):
        loader._build_text_encoder(te_path, official)


# ---------------------------------------------------------------------------
# Selection: a converted file is never auto-selected
# ---------------------------------------------------------------------------

def _tree(tmp_path, te_files):
    root = tmp_path / "tree"
    dit_dir = root / "diffusion_models"
    dit_dir.mkdir(parents=True)
    _write_header(str(dit_dir / "minimax_h3_fl2va_pruned_bf16.safetensors"), {
        "token_refiner.0.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 0]},
        "adaln_t_table": {"dtype": "F32", "shape": [1], "data_offsets": [0, 0]},
        "condition_proj.weight": {"dtype": "F32", "shape": [8, 5120], "data_offsets": [0, 0]},
    })
    vae_dir = root / "vae"
    vae_dir.mkdir()
    for name in ("minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"):
        _write_header(str(vae_dir / name), {
            "x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 0]}})
    te_dir = root / "text_encoders"
    te_dir.mkdir()
    for name in te_files:
        if name == CONVERTED_NAME:
            _converted_header_only(str(te_dir / name))
        else:
            _write_header(str(te_dir / name), {
                "model.layers.0.self_attn.q_proj.weight": {
                    "dtype": "BF16", "shape": [8, 8], "data_offsets": [0, 0]}})
    return root


def test_the_glob_fallback_never_selects_a_converted_encoder(tmp_path):
    """Deliverable 4: the only file on disk is a converted one, and it is NOT chosen."""
    root = _tree(tmp_path, [CONVERTED_NAME])
    layout = detect_minimax_h3_layout(str(root))
    assert layout is not None
    assert layout["text_encoder"] is None
    reason = layout["text_encoder_reason"]
    assert CONVERTED_NAME in reason and "explicit override" in reason


def test_a_shipped_file_still_wins_when_both_are_present(tmp_path):
    root = _tree(tmp_path, [CONVERTED_NAME, BF16_NAME])
    layout = detect_minimax_h3_layout(str(root))
    assert layout["text_encoder"] == os.path.join(str(root), "text_encoders", BF16_NAME)


def test_an_explicit_override_does_reach_a_converted_encoder(tmp_path):
    root = _tree(tmp_path, [CONVERTED_NAME])
    override = os.path.join(str(root), "text_encoders", CONVERTED_NAME)
    layout = detect_minimax_h3_layout(str(root), te_override=override)
    assert layout["text_encoder"] == override
    assert layout["text_encoder_reason"] == "explicit override"


def test_capability_predicate_rejects_a_converted_file_directly(tmp_path):
    path = tmp_path / CONVERTED_NAME
    _converted_header_only(str(path))
    assert loader._te_capability_accept(path) is False


# ---------------------------------------------------------------------------
# The candidate listing
# ---------------------------------------------------------------------------

def test_listing_shows_a_converted_file_as_selectable_but_not_default(tmp_path):
    path = tmp_path / CONVERTED_NAME
    _converted_header_only(str(path))
    entry = loader.inspect_minimax_h3_text_encoder_candidate(str(path))
    assert entry["compatible"] is True
    assert entry["variant"] == "converted_small"
    reason = entry["reason"]
    assert "Never the architecture default" in reason
    assert "clip_projections/" in reason
    assert f"hidden {DIMS['hidden_size']}" in reason


def test_listing_refuses_a_file_that_contradicts_its_own_declaration(tmp_path):
    path = tmp_path / "wrong_embed.safetensors"
    keys = {name: {"dtype": "BF16", "shape": list(tensor.shape), "data_offsets": [0, 0]}
            for name, tensor in _converted_tensors().items()}
    keys["model.embed_tokens.weight"]["shape"] = [99, DIMS["hidden_size"]]
    _write_header(str(path), keys, {"minimax_h3_te": json.dumps(
        {"num_hidden_layers": TAP, "modalities": "text", **DIMS})})
    entry = loader.inspect_minimax_h3_text_encoder_candidate(str(path))
    assert entry["compatible"] is False
    assert "its embedding is" in entry["reason"]


def test_switching_to_a_converted_encoder_is_refused_not_silently_unprojected(tmp_path):
    """The component-switch entry point cannot carry a projection (P3)."""
    path = tmp_path / CONVERTED_NAME
    _converted_header_only(str(path))
    with pytest.raises(ValueError, match=r"usable only with its trained projection"):
        loader.build_minimax_h3_text_encoder(str(path), None)


# ---------------------------------------------------------------------------
# The load path: pairing is resolved before the encoder is mapped
# ---------------------------------------------------------------------------

def _stub_components(monkeypatch, tmp_path, calls):
    official = _official_tree(tmp_path)
    for component in ("vae", "audio_vae"):
        directory = os.path.join(official, component)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as fh:
            fh.write("{}")
    monkeypatch.setattr(loader, "_build_text_encoder",
                        lambda *a: (calls.append(a) or (object(), object())))
    monkeypatch.setattr(loader, "_build_transformer", lambda *a: (object(), object()))
    monkeypatch.setattr(loader, "_build_video_vae", lambda *a: (object(), {}))
    monkeypatch.setattr(loader, "_build_audio_vae", lambda *a: (object(), {}))
    monkeypatch.setattr(loader, "_load_tokenizer_and_processor", lambda *a: (None, None))
    monkeypatch.setattr(loader, "_load_schedulers", lambda *a: (object(), object()))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return official


def _projection_file(root, *, d_in, d_out=5120, tap=TAP):
    directory = root / "clip_projections"
    directory.mkdir(exist_ok=True)
    path = directory / f"proj-{d_in}.safetensors"
    tensors = {
        "W": torch.zeros(d_in, d_out), "mean_in": torch.zeros(d_in), "std_in": torch.ones(d_in),
        "mean_out": torch.zeros(d_out), "std_out": torch.ones(d_out),
        "sink_out": torch.zeros(d_out),
        "mlp.0.weight": torch.zeros(2, d_in), "mlp.0.bias": torch.zeros(2),
        "mlp.2.weight": torch.zeros(d_out, 2), "mlp.2.bias": torch.zeros(d_out),
    }
    save_file(tensors, str(path), metadata={"d_in": str(d_in), "d_out": str(d_out),
                                            "tap": str(tap), "mlp_hidden": "2", "mlp_depth": "1"})
    return str(path)


def test_load_refuses_a_converted_encoder_with_no_projection_on_disk(tmp_path, monkeypatch):
    calls = []
    root = _tree(tmp_path, [CONVERTED_NAME])
    official = _stub_components(monkeypatch, tmp_path, calls)
    override = os.path.join(str(root), "text_encoders", CONVERTED_NAME)
    monkeypatch.setattr(loader, "_resolve_official_dir", lambda _root: official)

    with pytest.raises(FileNotFoundError) as excinfo:
        loader.load_minimax_h3_from_path(str(root), te_override=override)
    assert "Refusing to encode" in str(excinfo.value)
    assert not calls, "the 5-48 GiB encoder must not be mapped before the pairing is settled"


def test_load_pairs_the_projection_and_records_it(tmp_path, monkeypatch):
    calls = []
    root = _tree(tmp_path, [CONVERTED_NAME])
    official = _stub_components(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(loader, "_resolve_official_dir", lambda _root: official)
    projection_path = _projection_file(root, d_in=DIMS["hidden_size"])
    override = os.path.join(str(root), "text_encoders", CONVERTED_NAME)

    components = loader.load_minimax_h3_from_path(str(root), te_override=override)
    assert components["te_text_only"] is True
    assert components["te_projection"]["path"] == projection_path
    assert components["te_projection"]["spec"]["d_out"] == 5120
    assert calls, "the encoder should have been built once the pairing was settled"


def test_load_of_the_shipped_encoder_needs_no_projection(tmp_path, monkeypatch):
    calls = []
    root = _tree(tmp_path, [BF16_NAME])
    official = _stub_components(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(loader, "_resolve_official_dir", lambda _root: official)

    components = loader.load_minimax_h3_from_path(str(root))
    assert components["te_projection"] is None
    assert components["te_text_only"] is False
