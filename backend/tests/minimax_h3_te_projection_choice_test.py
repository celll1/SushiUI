"""MiniMax-H3: choosing between several projections that fit one encoder.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_te_projection_choice_test.py -v

When two files in ``clip_projections/`` declare the same ``d_in``, which one was
trained for a given encoder is not derivable from the files, so auto-resolution
refuses. Refusing is right; being unable to say WHICH is not. These tests pin
that the listing reports the whole candidate set with each file's own gate
verdict, and that naming one explicitly gets through on both surfaces -- the
load path and the live component switch.

Everything is header-sized: the encoders and projections here are toy-dimension
files and only ``_build_text_encoder`` is stubbed, so the real discovery and the
real pairing gates run.
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
from core.models.components import component_catalog, component_switcher  # noqa: E402
from core.models.components.component_switcher import (  # noqa: E402
    ComponentSwitchError, ComponentSwitchFailed, switch_component,
)

TEXT_DIM = 128
HIDDEN = 64
TAP = 2
CONVERTED = "qwen3vl_4b_heretic_tap24_bf16.safetensors"
RELEASED = "qwen3vl_32b_minimax_h3_bf16.safetensors"
FIRST = "mmh3-4b-clipproj-celeb-mlp.safetensors"
SECOND = "mmh3-4b-clipproj-rival-mlp.safetensors"


def _header_only(path, keys, metadata=None):
    header = dict(keys)
    header["__metadata__"] = metadata or {"format": "pt"}
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
    return str(path)


def _converted_te(path, *, hidden=HIDDEN, tap=TAP, vocab=32):
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


_DENSE_SHAPES = {
    "self_attn.q_proj": [8192, 5120], "self_attn.k_proj": [1024, 5120],
    "self_attn.v_proj": [1024, 5120], "self_attn.o_proj": [5120, 8192],
    "mlp.gate_proj": [25600, 5120], "mlp.up_proj": [25600, 5120],
    "mlp.down_proj": [5120, 25600],
}


def _released_te(path):
    keys = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [151936, 5120]}}
    for layer in range(50):
        for suffix, shape in _DENSE_SHAPES.items():
            keys[f"model.layers.{layer}.{suffix}.weight"] = {"dtype": "BF16", "shape": shape}
    return _header_only(path, keys)


def _projection(path, *, d_in=HIDDEN, d_out=TEXT_DIM, tap=TAP, hidden=4):
    g = torch.Generator().manual_seed(5)
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
    for name in ("diffusion_models", "text_encoders", "clip_projections"):
        (tmp_path / name).mkdir()
    dit = _header_only(tmp_path / "diffusion_models" / "dit.safetensors",
                       {"condition_proj.weight": {"dtype": "BF16", "shape": [256, TEXT_DIM]}})
    return str(tmp_path), dit


class _Manager:
    def __init__(self, dit, components):
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
    def build(te_path, official_dir):
        return SimpleNamespace(name=os.path.basename(te_path)), {"official": official_dir}

    monkeypatch.setattr(loader, "_build_text_encoder", build)
    monkeypatch.setattr(component_switcher, "_release_device_cache", lambda: None)
    import core.keep_hot as keep_hot
    monkeypatch.setattr(keep_hot, "clear_resident", lambda _manager: None)


def _components(root, dit, te_path, *, projection=None, text_only=False):
    return {
        "transformer": object(), "vae": object(), "audio_vae": object(),
        "text_encoder": object(), "text_encoder_config": {},
        "text_encoder_path": te_path, "text_encoder_origin": "architecture_default",
        "te_projection": projection, "te_text_only": text_only,
        "dit_path": dit, "official_dir": os.path.join(root, "official"),
    }


def _switch(manager, path, projection_path=None):
    return switch_component(manager, "text_encoder", {
        "compatibility": "compatible", "switchable": True, "_path": path,
    }, manager.model_revision, manager.component_revision, projection_path=projection_path)


def _entry(dit, basename=CONVERTED):
    choices = loader.describe_minimax_h3_text_encoder_choices(dit)
    return next(entry for entry in choices["text_encoders"]
                if os.path.basename(entry["path"]) == basename)


def _two_projection_tree(tmp_path):
    root, dit = _tree(tmp_path)
    _released_te(tmp_path / "text_encoders" / RELEASED)
    converted = _converted_te(tmp_path / "text_encoders" / CONVERTED)
    first = _projection(tmp_path / "clip_projections" / FIRST)
    second = _projection(tmp_path / "clip_projections" / SECOND)
    return root, dit, converted, first, second


# ---------------------------------------------------------------------------
# The listing reports the set, not just the winner
# ---------------------------------------------------------------------------

def test_two_matching_projections_are_both_listed_with_no_auto_winner(tmp_path):
    _root, dit, _converted, first, second = _two_projection_tree(tmp_path)

    entry = _entry(dit)

    assert entry["requires_projection"] is True
    assert entry["projection"] is None
    assert "Name one explicitly" in entry["projection_reason"]
    assert [c["path"] for c in entry["projection_candidates"]] == [first, second]
    for candidate in entry["projection_candidates"]:
        assert candidate["usable"] is True and candidate["reason"] is None
        assert (candidate["d_in"], candidate["d_out"], candidate["tap"]) == (HIDDEN, TEXT_DIM, TAP)


def test_a_candidate_that_fails_a_later_gate_is_listed_as_unusable(tmp_path):
    """Matching d_in but the wrong tap: listed with its reason, not omitted."""
    root, dit = _tree(tmp_path)
    _converted_te(tmp_path / "text_encoders" / CONVERTED)
    good = _projection(tmp_path / "clip_projections" / FIRST)
    wrong_tap = _projection(tmp_path / "clip_projections" / "tap-24.safetensors", tap=24)

    entry = _entry(dit)

    by_path = {c["path"]: c for c in entry["projection_candidates"]}
    assert set(by_path) == {good, wrong_tap}
    assert by_path[good]["usable"] is True
    assert by_path[wrong_tap]["usable"] is False
    assert "tap=24" in by_path[wrong_tap]["reason"]
    assert "num_hidden_layers=2" in by_path[wrong_tap]["reason"]
    # Only one candidate PASSES, but two declare d_in, so discovery still refuses.
    assert entry["projection"] is None
    assert "Name one explicitly" in entry["projection_reason"]


def test_a_candidate_whose_d_out_is_another_dits_width_is_listed_as_unusable(tmp_path):
    root, dit = _tree(tmp_path)
    _converted_te(tmp_path / "text_encoders" / CONVERTED)
    _projection(tmp_path / "clip_projections" / FIRST)
    narrow = _projection(tmp_path / "clip_projections" / "for-another-dit.safetensors", d_out=64)

    entry = _entry(dit)

    unusable = next(c for c in entry["projection_candidates"] if c["path"] == narrow)
    assert unusable["usable"] is False
    assert "d_out=64" in unusable["reason"] and f"text_dim={TEXT_DIM}" in unusable["reason"]


def test_one_matching_projection_still_auto_resolves_and_is_the_only_candidate(tmp_path):
    root, dit = _tree(tmp_path)
    _converted_te(tmp_path / "text_encoders" / CONVERTED)
    only = _projection(tmp_path / "clip_projections" / FIRST)

    entry = _entry(dit)

    assert entry["projection"] == only
    assert entry["projection_reason"] is None
    assert [c["path"] for c in entry["projection_candidates"]] == [only]
    assert entry["projection_candidates"][0]["usable"] is True


def test_an_encoder_that_needs_no_projection_lists_no_candidates(tmp_path):
    root, dit = _tree(tmp_path)
    _released_te(tmp_path / "text_encoders" / RELEASED)
    _projection(tmp_path / "clip_projections" / FIRST)

    entry = _entry(dit, RELEASED)

    assert entry["requires_projection"] is False
    assert entry["projection_candidates"] == []


# ---------------------------------------------------------------------------
# Auto-resolution still refuses; an explicit pick gets through (load path)
# ---------------------------------------------------------------------------

def test_auto_resolution_still_refuses_when_two_declare_the_width(tmp_path):
    _root, dit, converted, _first, _second = _two_projection_tree(tmp_path)

    with pytest.raises(ValueError, match=r"Name one explicitly"):
        loader.build_minimax_h3_text_encoder_bundle(
            converted, None, root=str(tmp_path), dit_path=dit)


@pytest.mark.parametrize("which", [0, 1])
def test_an_explicitly_named_projection_is_adopted_on_the_load_path(tmp_path, which):
    _root, dit, converted, first, second = _two_projection_tree(tmp_path)
    chosen = (first, second)[which]

    bundle = loader.build_minimax_h3_text_encoder_bundle(
        converted, None, root=str(tmp_path), dit_path=dit, projection_override=chosen)

    assert bundle["te_projection"]["path"] == chosen
    assert bundle["te_projection"]["spec"]["d_in"] == HIDDEN
    assert bundle["te_text_only"] is True


# ---------------------------------------------------------------------------
# The switch path
# ---------------------------------------------------------------------------

def test_the_switch_installs_the_named_projection_of_an_ambiguous_pair(tmp_path):
    root, dit, converted, _first, second = _two_projection_tree(tmp_path)
    released = os.path.join(root, "text_encoders", RELEASED)
    manager = _Manager(dit, _components(root, dit, released))

    _switch(manager, converted, projection_path=second)

    components = manager.minimax_h3_components
    assert components["text_encoder_path"] == converted
    assert components["te_projection"]["path"] == second
    assert manager.current_model_info["clip_projection_file"] == SECOND
    assert manager.component_health == "ready"


def test_the_switch_without_a_named_projection_still_refuses_and_restores(tmp_path):
    root, dit, converted, _first, _second = _two_projection_tree(tmp_path)
    released = os.path.join(root, "text_encoders", RELEASED)
    manager = _Manager(dit, _components(root, dit, released))

    with pytest.raises(ComponentSwitchFailed, match=r"Name one explicitly"):
        _switch(manager, converted)

    components = manager.minimax_h3_components
    assert components["text_encoder_path"] == released
    assert components["te_projection"] is None
    assert manager.component_health == "ready"


def test_a_named_projection_still_passes_every_gate_at_switch_time(tmp_path):
    root, dit = _tree(tmp_path)
    released = _released_te(tmp_path / "text_encoders" / RELEASED)
    converted = _converted_te(tmp_path / "text_encoders" / CONVERTED)
    _projection(tmp_path / "clip_projections" / FIRST)
    wrong_tap = _projection(tmp_path / "clip_projections" / "tap-24.safetensors", tap=24)
    manager = _Manager(dit, _components(root, dit, released))

    with pytest.raises(ComponentSwitchFailed, match=r"tap=24"):
        _switch(manager, converted, projection_path=wrong_tap)

    assert manager.minimax_h3_components["text_encoder_path"] == released
    assert manager.component_health == "ready"


def test_a_failed_switch_restores_the_explicitly_chosen_projection_too(tmp_path):
    """The running pair was reached by naming SECOND; the restore must name it."""
    root, dit, converted, _first, second = _two_projection_tree(tmp_path)
    released = os.path.join(root, "text_encoders", RELEASED)
    unpairable = _converted_te(tmp_path / "text_encoders" / "qwen3vl_8b_instruct_tap24_bf16.safetensors",
                               hidden=96)
    manager = _Manager(dit, _components(root, dit, released))
    _switch(manager, converted, projection_path=second)
    assert manager.minimax_h3_components["te_projection"]["path"] == second

    with pytest.raises(ComponentSwitchFailed, match=r"d_in=96"):
        _switch(manager, unpairable)

    components = manager.minimax_h3_components
    assert components["text_encoder_path"] == converted
    # Discovery would refuse here (two files declare d_in=64); only the restore's
    # explicit override can bring the running pair back.
    assert components["te_projection"]["path"] == second
    assert components["te_text_only"] is True
    assert manager.current_model_info["clip_projection_file"] == SECOND
    assert manager.component_health == "ready"
    assert manager.component_revision == 2


def test_one_matching_projection_still_auto_resolves_on_the_switch_path(tmp_path):
    root, dit = _tree(tmp_path)
    released = _released_te(tmp_path / "text_encoders" / RELEASED)
    converted = _converted_te(tmp_path / "text_encoders" / CONVERTED)
    only = _projection(tmp_path / "clip_projections" / FIRST)
    manager = _Manager(dit, _components(root, dit, released))

    _switch(manager, converted)

    assert manager.minimax_h3_components["te_projection"]["path"] == only


def test_a_projection_path_is_refused_for_a_slot_that_cannot_use_one(tmp_path):
    root, dit = _tree(tmp_path)
    released = _released_te(tmp_path / "text_encoders" / RELEASED)
    projection = _projection(tmp_path / "clip_projections" / FIRST)
    manager = _Manager(dit, _components(root, dit, released))

    with pytest.raises(ComponentSwitchError, match=r"only to a MiniMax-H3 text-encoder switch"):
        switch_component(manager, "vision_encoder", {
            "compatibility": "compatible", "switchable": True, "_path": released,
        }, manager.model_revision, manager.component_revision, projection_path=projection)


# ---------------------------------------------------------------------------
# The route that carries the choice
# ---------------------------------------------------------------------------

def test_the_switch_route_forwards_projection_path(monkeypatch):
    """A dropped field would switch the encoder onto a guessed projection."""
    import asyncio

    from api import routes
    from api.routes import ComponentSwitchRequest

    seen = {}
    monkeypatch.setattr(routes.pipeline_manager, "current_model_info",
                        {"type": "minimax_h3", "source": "M:/model/minimax_h3"}, raising=False)

    async def catalog(_db):
        return {"text_encoder": [{"candidate_id": "abc", "_path": "M:/te.safetensors"}]}

    monkeypatch.setattr(routes, "_current_component_catalog", catalog)
    monkeypatch.setattr(component_catalog, "find_candidate",
                        lambda catalog, slot, candidate_id: catalog[slot][0])
    monkeypatch.setattr(component_catalog, "build_response",
                        lambda manager, catalog, operation=None: {"slots": []})
    monkeypatch.setattr(component_switcher, "switch_component",
                        lambda *args, **kwargs: seen.update(args=args, kwargs=kwargs) or {"state": "succeeded"})

    asyncio.run(routes.switch_current_model_component(
        ComponentSwitchRequest(
            slot="text_encoder", candidate_id="abc",
            expected_model_revision=1, expected_component_revision=1,
            projection_path="M:/model/minimax_h3/clip_projections/" + SECOND),
        db=None))

    assert seen["kwargs"]["projection_path"].endswith(SECOND)


def test_the_switch_request_defaults_projection_path_to_absent():
    from api.routes import ComponentSwitchRequest

    request = ComponentSwitchRequest(slot="vae", candidate_id="abc",
                                     expected_model_revision=1, expected_component_revision=1)
    assert request.projection_path is None


# ---------------------------------------------------------------------------
# The catalog the live-switch UI offers from
# ---------------------------------------------------------------------------

def _catalog_entry(tmp_path, manager, dit, basename=CONVERTED):
    choices = loader.describe_minimax_h3_text_encoder_choices(dit)["text_encoders"]
    for entry in choices:
        entry["name"] = os.path.splitext(os.path.basename(entry["path"]))[0]
    catalog = component_catalog.build_catalog(manager, h3_text_encoders=choices)
    return next(candidate for candidate in catalog["text_encoder"]
                if str(candidate.get("_path") or "").endswith(basename))


def test_the_catalog_offers_an_ambiguous_encoder_with_its_candidate_set(tmp_path):
    root, dit, _converted, first, second = _two_projection_tree(tmp_path)
    released = os.path.join(root, "text_encoders", RELEASED)
    manager = _Manager(dit, _components(root, dit, released))

    candidate = _catalog_entry(tmp_path, manager, dit)

    assert candidate["switchable"] is True and candidate["compatibility"] == "compatible"
    assert candidate["projection"] is None
    assert "send projection_path naming one" in candidate["switch_reason"]
    assert [c["path"] for c in candidate["projection_candidates"]] == [first, second]


def test_the_catalog_still_refuses_an_encoder_with_no_usable_projection(tmp_path):
    root, dit = _tree(tmp_path)
    released = _released_te(tmp_path / "text_encoders" / RELEASED)
    _converted_te(tmp_path / "text_encoders" / CONVERTED)
    manager = _Manager(dit, _components(root, dit, released))

    candidate = _catalog_entry(tmp_path, manager, dit)

    assert candidate["switchable"] is False
    assert candidate["compatibility"] == "incompatible"
    assert candidate["projection_candidates"] == []
