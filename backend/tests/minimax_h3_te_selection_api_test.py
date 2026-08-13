"""MiniMax-H3's load-time text-encoder/projection choice, over the API.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_te_selection_api_test.py -v

`POST /models/load` gained `text_encoder_file` / `clip_projection_file`, and
`GET /models/minimax-h3/text-encoders` lists what they may be set to. Three of
the ways this could fail do so SILENTLY -- the requested encoder is not built
and the load reports success -- so each has its own test here:

* the DiT-only fast path recomputes both layouts WITHOUT the override, so the
  encoder compares equal and the request disappears;
* the same-model early return fires because the model id does not change with
  the encoder;
* `**kwargs` drops an unrecognised key at any of the four hops between the
  route and `load_minimax_h3_from_path`.

Everything here is header-only: no tensor bytes are read and no component is
built, so nothing loads a 5-48 GiB encoder.
"""

import asyncio
import inspect
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest  # noqa: E402

from core.model_loader import ModelLoader  # noqa: E402
from core.models.minimax_h3 import loader as h3_loader  # noqa: E402
from core.models.minimax_h3.loader import MINIMAX_H3_TE_PATTERNS  # noqa: E402

RELEASED_NAME = MINIMAX_H3_TE_PATTERNS[1]
# The exact basenames PUBLISHED_TE_SUBSTITUTIONS is keyed by; the dims below are
# toy, the names are what makes the fallback agreement resolvable.
CONVERTED_NAME = "qwen3vl_4b_heretic_tap24_bf16.safetensors"
PROJECTION_NAME = "mmh3-4b-clipproj-celeb-mlp.safetensors"

TEXT_DIM = 5120
CONVERTED_HIDDEN = 64
CONVERTED_TAP = 2
CONVERTED_DIMS = {
    "hidden_size": CONVERTED_HIDDEN,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000.0,
    "mrope_section": [4, 2, 2],
    "vocab_size": 32,
}


def _write_header(path, entries, metadata=None):
    """A safetensors file carrying only its JSON header; zero tensor bytes."""
    header = {name: {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}
              for name, (dtype, shape) in entries.items()}
    header["__metadata__"] = metadata or {"format": "pt"}
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
    return str(path)


def _released_te(path):
    """The released 32B geometry: 50 layers, 5120-wide, unquantized bf16."""
    entries = {"model.embed_tokens.weight": ("BF16", [151936, TEXT_DIM])}
    for layer in range(50):
        for suffix, shape in {
            "self_attn.q_proj": [8192, TEXT_DIM],
            "self_attn.k_proj": [1024, TEXT_DIM],
            "self_attn.v_proj": [1024, TEXT_DIM],
            "self_attn.o_proj": [TEXT_DIM, 8192],
            "mlp.gate_proj": [25600, TEXT_DIM],
            "mlp.up_proj": [25600, TEXT_DIM],
            "mlp.down_proj": [TEXT_DIM, 25600],
        }.items():
            entries[f"model.layers.{layer}.{suffix}.weight"] = ("BF16", shape)
    return _write_header(path, entries)


def _converted_te(path, dims=CONVERTED_DIMS, tap=CONVERTED_TAP):
    hidden, ffn = dims["hidden_size"], dims["intermediate_size"]
    q = dims["num_attention_heads"] * dims["head_dim"]
    kv = dims["num_key_value_heads"] * dims["head_dim"]
    entries = {"model.embed_tokens.weight": ("BF16", [dims["vocab_size"], hidden])}
    for layer in range(tap):
        for suffix, shape in {
            "self_attn.q_proj": [q, hidden],
            "self_attn.k_proj": [kv, hidden],
            "self_attn.v_proj": [kv, hidden],
            "self_attn.o_proj": [hidden, q],
            "mlp.gate_proj": [ffn, hidden],
            "mlp.up_proj": [ffn, hidden],
            "mlp.down_proj": [hidden, ffn],
        }.items():
            entries[f"model.layers.{layer}.{suffix}.weight"] = ("BF16", shape)
    declared = {"num_hidden_layers": tap, "modalities": "text", "source_size_label": "4B",
                "converter": "minimax_h3_te_gguf_convert", **dims}
    return _write_header(path, entries, {"minimax_h3_te": json.dumps(declared)})


def _projection(path, *, d_in=CONVERTED_HIDDEN, d_out=TEXT_DIM, tap=CONVERTED_TAP):
    entries = {
        "W": ("F32", [d_in, d_out]),
        "mean_in": ("F32", [d_in]), "std_in": ("F32", [d_in]),
        "mean_out": ("F32", [d_out]), "std_out": ("F32", [d_out]),
        "sink_out": ("F32", [d_out]),
        "mlp.0.weight": ("F32", [2, d_in]), "mlp.0.bias": ("F32", [2]),
        "mlp.2.weight": ("F32", [d_out, 2]), "mlp.2.bias": ("F32", [d_out]),
    }
    return _write_header(path, entries, {"d_in": str(d_in), "d_out": str(d_out),
                                         "tap": str(tap), "mlp_hidden": "2", "mlp_depth": "1"})


def _tree(tmp_path, *, converted=True, released=True, projection=True):
    """A MiniMax-H3 tree whose every file is header-only."""
    root = tmp_path / "minimax_h3"
    (root / "diffusion_models").mkdir(parents=True)
    for variant in ("fl2va", "ref2va"):
        _write_header(root / "diffusion_models" / f"minimax_h3_{variant}_pruned_bf16.safetensors", {
            "token_refiner.0.weight": ("F32", [1, 1]),
            "adaln_t_table": ("F32", [1]),
            "condition_proj.weight": ("F32", [8, TEXT_DIM]),
        })
    (root / "vae").mkdir()
    for name in ("minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"):
        _write_header(root / "vae" / name, {"x": ("F32", [1])})
    (root / "text_encoders").mkdir()
    if released:
        _released_te(root / "text_encoders" / RELEASED_NAME)
    if converted:
        _converted_te(root / "text_encoders" / CONVERTED_NAME)
    if projection:
        (root / "clip_projections").mkdir()
        _projection(root / "clip_projections" / PROJECTION_NAME)
    official = root / "official"
    official.mkdir()
    (official / "model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"}), encoding="utf-8")
    # The loader checks these exist before it maps anything.
    for component in ("vae", "audio_vae", "text_encoder"):
        (official / component).mkdir()
        (official / component / "config.json").write_text("{}", encoding="utf-8")
    return root


def _dit(root, variant="fl2va"):
    return str(root / "diffusion_models" / f"minimax_h3_{variant}_pruned_bf16.safetensors")


def _te(root, name):
    return str(root / "text_encoders" / name)


# ---------------------------------------------------------------------------
# GET /models/minimax-h3/text-encoders
# ---------------------------------------------------------------------------

def test_listing_reports_the_default_selection_and_both_kinds_of_file(tmp_path):
    root = _tree(tmp_path)
    choices = h3_loader.describe_minimax_h3_text_encoder_choices(_dit(root))

    assert choices["selected"] == _te(root, RELEASED_NAME)
    assert choices["selected_reason"]
    by_name = {os.path.basename(entry["path"]): entry for entry in choices["text_encoders"]}
    assert set(by_name) == {RELEASED_NAME, CONVERTED_NAME}

    released = by_name[RELEASED_NAME]
    assert released["compatible"] is True and released["variant"] == "bf16"
    # The released files declare no geometry of their own; it comes from
    # official/text_encoder/config.json.
    assert released["hidden_size"] is None and released["num_hidden_layers"] is None
    assert released["requires_projection"] is False
    assert released["agreement"] is None

    converted = by_name[CONVERTED_NAME]
    assert converted["compatible"] is True and converted["variant"] == "converted_small"
    assert converted["hidden_size"] == CONVERTED_HIDDEN
    assert converted["num_hidden_layers"] == CONVERTED_TAP
    assert converted["requires_projection"] is True
    assert converted["agreement"]["cosine"] == 0.826
    assert converted["agreement"]["source"] == "published"
    assert converted["agreement"]["projection"] == PROJECTION_NAME

    projections = choices["clip_projections"]
    assert [os.path.basename(spec["path"]) for spec in projections] == [PROJECTION_NAME]
    assert (projections[0]["d_in"], projections[0]["d_out"], projections[0]["tap"]) == (
        CONVERTED_HIDDEN, TEXT_DIM, CONVERTED_TAP)


def test_listing_reports_no_agreement_for_an_unmeasured_pairing(tmp_path):
    """A pairing with no local measurement and no fallback must borrow nothing."""
    root = _tree(tmp_path, projection=False)
    (root / "clip_projections").mkdir()
    _projection(root / "clip_projections" / "some-other-projection.safetensors")

    choices = h3_loader.describe_minimax_h3_text_encoder_choices(str(root))
    converted = next(entry for entry in choices["text_encoders"]
                     if entry["path"].endswith(CONVERTED_NAME))
    assert converted["requires_projection"] is True
    assert converted["agreement"] is None


def test_listing_refuses_a_path_that_is_not_an_h3_tree(tmp_path):
    with pytest.raises(ValueError, match=r"does not resolve to a MiniMax-H3 model tree"):
        h3_loader.describe_minimax_h3_text_encoder_choices(str(tmp_path))


# ---------------------------------------------------------------------------
# Hazard 3: an explicit named parameter at every hop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("func", [
    ModelLoader.load_model,
    ModelLoader.load_from_safetensors,
    ModelLoader.load_from_diffusers,
    ModelLoader.load_minimax_h3_from_path,
])
def test_every_hop_names_both_fields_explicitly(func):
    """A `**kwargs`-only hop would drop them and load the default encoder."""
    parameters = inspect.signature(func).parameters
    for name in ("text_encoder_file", "clip_projection_file"):
        assert name in parameters, f"{func.__name__} takes {name} via **kwargs"


def test_both_fields_reach_the_h3_loader_from_a_diffusers_tree(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    seen = {}

    def stub(model_path, torch_dtype=None, *, te_override=None, te_projection_override=None):
        seen.update(model_path=model_path, te_override=te_override,
                    te_projection_override=te_projection_override)
        return {"type": "minimax_h3"}

    monkeypatch.setattr(h3_loader, "load_minimax_h3_from_path", stub)
    ModelLoader.load_from_diffusers(
        str(root),
        text_encoder_file=_te(root, CONVERTED_NAME),
        clip_projection_file=str(root / "clip_projections" / PROJECTION_NAME))

    assert seen["te_override"] == _te(root, CONVERTED_NAME)
    assert seen["te_projection_override"] == str(root / "clip_projections" / PROJECTION_NAME)


def test_both_fields_reach_the_h3_loader_from_a_dit_file(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    seen = {}
    monkeypatch.setattr(
        h3_loader, "load_minimax_h3_from_path",
        lambda model_path, torch_dtype=None, *, te_override=None, te_projection_override=None:
            seen.update(te_override=te_override, te_projection_override=te_projection_override))

    ModelLoader.load_from_safetensors(_dit(root), text_encoder_file=_te(root, CONVERTED_NAME))
    assert seen == {"te_override": _te(root, CONVERTED_NAME), "te_projection_override": None}


@pytest.mark.parametrize("field", ["text_encoder_file", "clip_projection_file"])
def test_a_non_h3_architecture_refuses_both_fields(tmp_path, monkeypatch, field):
    """Dropping them would load SDXL's own components and report success."""
    path = tmp_path / "some-sdxl.safetensors"
    _write_header(path, {"x": ("F32", [1])})
    monkeypatch.setattr(ModelLoader, "detect_model_type", staticmethod(lambda _p: "sdxl"))

    with pytest.raises(ValueError) as excinfo:
        ModelLoader.load_from_safetensors(str(path), **{field: "M:/whatever.safetensors"})
    message = str(excinfo.value)
    assert field in message and "minimax_h3" in message and "'sdxl'" in message


def test_a_huggingface_source_refuses_both_fields():
    with pytest.raises(ValueError, match=r"cannot be used with a huggingface source"):
        ModelLoader.load_model("huggingface", "some/repo",
                               text_encoder_file="M:/whatever.safetensors")


# ---------------------------------------------------------------------------
# A projection named for an encoder that takes none
# ---------------------------------------------------------------------------

def test_a_projection_for_the_released_encoder_is_refused_not_ignored(tmp_path, monkeypatch):
    root = _tree(tmp_path, converted=False)
    built = []
    monkeypatch.setattr(h3_loader, "_build_text_encoder",
                        lambda *a: (built.append(a) or (object(), object())))

    with pytest.raises(ValueError) as excinfo:
        h3_loader.load_minimax_h3_from_path(
            str(root),
            te_projection_override=str(root / "clip_projections" / PROJECTION_NAME))
    message = str(excinfo.value)
    assert PROJECTION_NAME in message and RELEASED_NAME in message
    assert "is already the DiT's 5120-wide conditioning" in message
    assert not built, "the refusal must land before the encoder is mapped"


# ---------------------------------------------------------------------------
# Hazards 1 and 2: the two paths that would swallow the request
# ---------------------------------------------------------------------------

class _LoadAttempted(Exception):
    """Raised in place of a real full load."""


@pytest.fixture
def h3_manager(tmp_path, monkeypatch):
    """A PipelineManager with an H3 tree "loaded", and no real load path.

    `__init__` only assigns attributes, so this is a real manager; the full
    load is replaced by a raise, which is how the tests below tell "reloaded"
    from "silently kept the loaded encoder".
    """
    from core import pipeline as pipeline_module

    root = _tree(tmp_path)
    manager = pipeline_module.DiffusionPipelineManager()
    manager.is_minimax_h3_model = True
    manager.current_model = f"diffusers:{root}"
    manager.current_model_info = {"source": str(root), "type": "minimax_h3"}
    manager.minimax_h3_components = {
        "type": "minimax_h3",
        "text_encoder_path": _te(root, RELEASED_NAME),
        "te_projection": None,
    }
    monkeypatch.setattr(pipeline_module, "LAST_MODEL_CONFIG_FILE", tmp_path / "last_model.json")
    monkeypatch.setattr(
        pipeline_module.ModelLoader, "load_model",
        staticmethod(lambda **kwargs: (_ for _ in ()).throw(_LoadAttempted())))
    manager._root = root
    return manager


def _fast_path_calls(manager, monkeypatch):
    calls = []
    monkeypatch.setattr(manager, "_reload_minimax_h3_dit_only",
                        lambda *args: (calls.append(args), True)[1])
    return calls


def test_the_dit_only_fast_path_is_skipped_when_the_encoder_changes(h3_manager, monkeypatch):
    """Hazard 1: that path recomputes both layouts without the override."""
    calls = _fast_path_calls(h3_manager, monkeypatch)

    with pytest.raises(RuntimeError, match=r"Failed to load model"):
        h3_manager._load_model_locked(
            "safetensors", _dit(h3_manager._root, "ref2va"),
            text_encoder_file=_te(h3_manager._root, CONVERTED_NAME))
    assert not calls, "a text-encoder change must not be served by the DiT-only reload"


def test_the_dit_only_fast_path_still_runs_for_a_pure_dit_switch(h3_manager, monkeypatch):
    """The complement: naming the loaded encoder keeps the cheap path."""
    calls = _fast_path_calls(h3_manager, monkeypatch)

    h3_manager._load_model_locked(
        "safetensors", _dit(h3_manager._root, "ref2va"),
        text_encoder_file=_te(h3_manager._root, RELEASED_NAME))
    assert len(calls) == 1


def test_the_same_model_early_return_does_not_swallow_an_encoder_change(h3_manager):
    """Hazard 2: the model id is identical when only the encoder differs."""
    with pytest.raises(RuntimeError, match=r"Failed to load model"):
        h3_manager._load_model_locked(
            "diffusers", str(h3_manager._root), force_reload=False,
            text_encoder_file=_te(h3_manager._root, CONVERTED_NAME))


def test_the_same_model_early_return_still_fires_for_the_loaded_encoder(h3_manager):
    assert h3_manager._load_model_locked(
        "diffusers", str(h3_manager._root), force_reload=False,
        text_encoder_file=_te(h3_manager._root, RELEASED_NAME)) is None


def test_a_projection_change_alone_also_reloads(h3_manager):
    with pytest.raises(RuntimeError, match=r"Failed to load model"):
        h3_manager._load_model_locked(
            "diffusers", str(h3_manager._root), force_reload=False,
            clip_projection_file=str(h3_manager._root / "clip_projections" / PROJECTION_NAME))


def test_the_load_reports_the_pairing_it_actually_built(h3_manager, monkeypatch):
    """`GET /models/current` must name the substituted encoder and projection."""
    from core import pipeline as pipeline_module

    encoder = _te(h3_manager._root, CONVERTED_NAME)
    projection = str(h3_manager._root / "clip_projections" / PROJECTION_NAME)
    monkeypatch.setattr(pipeline_module.ModelLoader, "load_model", staticmethod(
        lambda **kwargs: {
            "type": "minimax_h3", "variant": "fl2va",
            "text_encoder_path": kwargs["text_encoder_file"],
            "te_projection": {"path": kwargs["clip_projection_file"]},
            "te_text_only": True,
        }))

    h3_manager._load_model_locked("diffusers", str(h3_manager._root),
                                  text_encoder_file=encoder,
                                  clip_projection_file=projection)
    info = h3_manager.current_model_info
    assert info["text_encoder_file"] == CONVERTED_NAME
    assert info["clip_projection_file"] == PROJECTION_NAME
    assert info["te_text_only"] is True


# ---------------------------------------------------------------------------
# Persistence across a restart
# ---------------------------------------------------------------------------

def test_the_chosen_pairing_is_persisted_and_replayed(h3_manager, monkeypatch):
    from core import pipeline as pipeline_module

    encoder = _te(h3_manager._root, CONVERTED_NAME)
    projection = str(h3_manager._root / "clip_projections" / PROJECTION_NAME)
    h3_manager._minimax_h3_te_request = (encoder, projection)
    h3_manager._save_last_model("diffusers", str(h3_manager._root), "txt2img",
                                *h3_manager._minimax_h3_te_request)

    with open(pipeline_module.LAST_MODEL_CONFIG_FILE, encoding="utf-8") as fh:
        assert json.load(fh)["text_encoder_file"] == encoder

    replayed = {}
    monkeypatch.setattr(h3_manager, "load_model",
                        lambda **kwargs: replayed.update(kwargs))
    h3_manager._auto_load_last_model()
    assert replayed["text_encoder_file"] == encoder
    assert replayed["clip_projection_file"] == projection


def test_a_last_model_file_without_the_two_keys_still_auto_loads(h3_manager, monkeypatch):
    from core import pipeline as pipeline_module

    with open(pipeline_module.LAST_MODEL_CONFIG_FILE, "w", encoding="utf-8") as fh:
        json.dump({"source_type": "diffusers", "source": str(h3_manager._root),
                   "pipeline_type": "txt2img"}, fh)

    replayed = {}
    monkeypatch.setattr(h3_manager, "load_model", lambda **kwargs: replayed.update(kwargs))
    h3_manager._auto_load_last_model()
    assert replayed["source"] == str(h3_manager._root)
    assert replayed["text_encoder_file"] is None
    assert replayed["clip_projection_file"] is None


# ---------------------------------------------------------------------------
# The route itself
# ---------------------------------------------------------------------------

def test_post_models_load_forwards_both_fields(monkeypatch):
    from api import routes

    seen = {}
    monkeypatch.setattr(routes.pipeline_manager, "load_model",
                        lambda **kwargs: seen.update(kwargs))
    response = asyncio.run(routes.load_model(
        source_type="diffusers", source="M:/model/minimax_h3", revision=None, force=False,
        text_encoder_file="M:/model/minimax_h3/text_encoders/" + CONVERTED_NAME,
        clip_projection_file="M:/model/minimax_h3/clip_projections/" + PROJECTION_NAME))

    assert response["success"] is True
    assert seen["text_encoder_file"].endswith(CONVERTED_NAME)
    assert seen["clip_projection_file"].endswith(PROJECTION_NAME)
    assert seen["force_reload"] is False


def test_post_models_load_sends_none_when_the_fields_are_absent(monkeypatch):
    """An omitted multipart field arrives as "" from some clients, not as None."""
    from api import routes

    seen = {}
    monkeypatch.setattr(routes.pipeline_manager, "load_model",
                        lambda **kwargs: seen.update(kwargs))
    asyncio.run(routes.load_model(source_type="safetensors", source="M:/model/sdxl/x.safetensors",
                                  revision=None, force=False,
                                  text_encoder_file="", clip_projection_file=None))
    assert seen["text_encoder_file"] is None and seen["clip_projection_file"] is None


@pytest.mark.parametrize("field", ["text_encoder_file", "clip_projection_file"])
def test_post_models_load_defaults_both_fields_to_absent(field):
    """Optional: an existing client that sends neither keeps working."""
    from api import routes

    assert inspect.signature(routes.load_model).parameters[field].default.default is None
