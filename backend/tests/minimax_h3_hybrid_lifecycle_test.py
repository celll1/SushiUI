"""C4: a MiniMax-H3 hybrid is its own loaded model.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_hybrid_lifecycle_test.py -v

Design doc: `docs/guides/MINIMAX_H3_HYBRID_LOADER_DESIGN.md` (rev2) sections 5.1,
5.4, 7 and 9.2. Four properties, each of which has a plausible implementation
that gets it wrong:

* `variant` is set to `"hybrid"` EXPLICITLY. `loader._layout_from_root` derives
  it by SUBSTRING MATCH on the filename, so a hybrid whose base is
  `minimax_h3_fl2va_*.safetensors` inherits `fl2va` and walks through all five
  gates C1 closed. The control test right below it loads the same base file
  base-only and gets `fl2va`, so "hybrid" is a property of the merge, not of
  this fixture.
* the model identity carries the recipe, so "same base, different overlay" and
  "same pair, different range" rebuild instead of hitting the same-model early
  return.
* the DiT-only reload keeps its atomicity: a refused preflight or a failed merge
  leaves the live components untouched.
* `last_model.json` can reconstruct the pair.

EVERYTHING HERE IS HEADER-ONLY. The fixtures are the fake trees from
`minimax_h3_hybrid_preflight_test` (struct-packed JSON header, zero-length data
section) and `_build_transformer` is always stubbed; no checkpoint under
`M:/model/minimax_h3` is opened.
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TESTS_DIR)
sys.path.insert(0, os.path.dirname(_TESTS_DIR))

from minimax_h3_hybrid_preflight_test import _h3_header, _tree  # noqa: E402
from minimax_h3_model_listing_test import _write_fake_h3_dit  # noqa: E402

from core.models.minimax_h3 import reload as h3_reload  # noqa: E402
from core.models.minimax_h3.hybrid_spec import (  # noqa: E402
    DEFAULT_BLOCK_RANGE_END,
    DEFAULT_BLOCK_RANGE_START,
    HYBRID_COMPONENT_KEYS,
    PRESET_BLOCK_RANGE_ADALN,
    MiniMaxH3HybridRefusal,
    MiniMaxH3HybridSpec,
    hybrid_component_fields,
    hybrid_model_identity,
    hybrid_request_from_spec,
    normalize_hybrid_request,
    preflight_hybrid_request,
)

# The fake trees hold 6 blocks, so the doc's 25..49 default names blocks that do
# not exist there. Every preflight below therefore passes an explicit range.
_RANGE = {"block_range_start": 2, "block_range_end": 3}


def _request(overlay, **overrides):
    request = {"overlay_file": overlay, **_RANGE}
    request.update(overrides)
    return request


def _preflight(base, overlay, **overrides):
    return preflight_hybrid_request(base, _request(overlay, **overrides))


def _components():
    """A loaded H3 bundle: the shared objects must survive a DiT-only reload."""
    shared = {name: object() for name in (
        "text_encoder", "text_encoder_config", "tokenizer", "processor",
        "vae", "vae_config", "audio_vae", "audio_vae_config",
        "scheduler", "audio_scheduler",
    )}
    return {
        "type": "minimax_h3",
        "variant": "fl2va",
        "transformer": object(),
        "transformer_config": object(),
        **shared,
    }


def _stub_build(monkeypatch, seen=None):
    """Replace the real transformer build; record whether a hybrid reached it."""
    def build(path, dtype, official, hybrid=None):
        if seen is not None:
            seen.append(hybrid)
        return object(), object()

    monkeypatch.setattr(h3_reload, "_build_transformer", build)


# ---------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------

def test_no_request_is_a_base_only_load():
    assert normalize_hybrid_request(None) is None


def test_a_bare_overlay_gets_the_documented_defaults():
    normalized = normalize_hybrid_request({"overlay_file": "o.safetensors"})
    assert normalized == {
        "overlay_file": "o.safetensors",
        "preset": PRESET_BLOCK_RANGE_ADALN,
        "block_range_start": DEFAULT_BLOCK_RANGE_START,
        "block_range_end": DEFAULT_BLOCK_RANGE_END,
        "final_adaln_from_overlay": False,
    }


def _request_refusal(request):
    with pytest.raises(MiniMaxH3HybridRefusal) as excinfo:
        normalize_hybrid_request(request)
    return excinfo.value.code


def test_a_misspelt_field_refuses_instead_of_loading_the_default_range():
    assert _request_refusal(
        {"overlay_file": "o", "block_range_stat": 30}) == "hybrid_request_unknown_field"


def test_a_request_with_no_overlay_refuses_rather_than_degrading_to_base_only():
    assert _request_refusal({"preset": PRESET_BLOCK_RANGE_ADALN}) == "hybrid_request_no_overlay"
    assert _request_refusal({"overlay_file": ""}) == "hybrid_request_no_overlay"


def test_a_string_flag_is_not_coerced():
    """`bool("false")` is True, which would silently turn the toggle on."""
    assert _request_refusal(
        {"overlay_file": "o", "final_adaln_from_overlay": "false"}) == "hybrid_request_invalid"


def test_more_than_one_overlay_refuses():
    assert _request_refusal({"overlay_file": ["a", "b"]}) == "multiple_overlays"


# ---------------------------------------------------------------------------
# section 7 -- model identity
# ---------------------------------------------------------------------------

def test_an_unvalidated_spec_has_no_identity():
    spec = MiniMaxH3HybridSpec(base_dit_path="b", overlay_dit_path="o")
    with pytest.raises(MiniMaxH3HybridRefusal) as excinfo:
        hybrid_model_identity(spec)
    assert excinfo.value.code == "hybrid_identity_unvalidated"


def test_identity_states_the_recipe_and_the_digest(tmp_path):
    base, overlay = _tree(tmp_path)
    identity = hybrid_model_identity(_preflight(base, overlay).spec)
    assert identity.startswith("hybrid1:block_range_adaln:2-3:final=0:")
    assert identity.endswith(_preflight(base, overlay).spec.compatibility_digest)


def test_the_same_pair_and_recipe_is_the_same_identity(tmp_path):
    base, overlay = _tree(tmp_path)
    assert (hybrid_model_identity(_preflight(base, overlay).spec)
            == hybrid_model_identity(_preflight(base, overlay).spec))


def test_a_different_range_a_different_toggle_and_a_different_overlay_all_differ(tmp_path):
    base, overlay = _tree(tmp_path)
    other_overlay = os.path.join(
        os.path.dirname(overlay), "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors")
    _write_fake_h3_dit(other_overlay, header=_h3_header())

    identities = {
        "default": hybrid_model_identity(_preflight(base, overlay).spec),
        "range": hybrid_model_identity(
            _preflight(base, overlay, block_range_start=3).spec),
        "final_adaln": hybrid_model_identity(
            _preflight(base, overlay, final_adaln_from_overlay=True).spec),
        "overlay": hybrid_model_identity(_preflight(base, other_overlay).spec),
    }
    assert len(set(identities.values())) == len(identities), identities


# ---------------------------------------------------------------------------
# section 7 -- the same-model early return, through the real _load_model_locked
# ---------------------------------------------------------------------------

class _ReachedFullReload(Exception):
    """Raised from the first step past both same-model shortcuts."""


def _locked_load(monkeypatch, *, source, hybrid=None, current_model=None,
                 components=None):
    """Run the real `_load_model_locked` and report where it got to.

    Returns `(calls, model_id)`: `calls` records which shortcut fired, `model_id`
    is the identity the DiT-only path was asked to install.
    """
    from core import pipeline as pipeline_module
    import core.keep_hot as keep_hot

    calls = []
    seen = {}

    def dit_only(_source_type, _source, _current_source, _pipeline_type, model_id,
                 *, hybrid=None):
        calls.append("dit_only")
        seen["model_id"] = model_id
        seen["hybrid"] = hybrid
        return True

    manager = SimpleNamespace(
        current_model=current_model,
        current_model_info={"source": source},
        component_health="ready",
        is_minimax_h3_model=True,
        minimax_h3_components=components if components is not None else {},
        _minimax_h3_te_selection_differs=lambda *_a: False,
        _reload_minimax_h3_dit_only=dit_only,
    )

    def stop(_manager):
        calls.append("full_reload")
        raise _ReachedFullReload()

    monkeypatch.setattr(keep_hot, "clear_resident", stop)
    try:
        pipeline_module.DiffusionPipelineManager._load_model_locked(
            manager, "safetensors", source, "txt2img", hybrid=hybrid)
    except _ReachedFullReload:
        pass
    return calls, seen


def test_a_base_only_load_keeps_the_historical_model_id(monkeypatch, tmp_path):
    """The identity of a non-hybrid load is byte-identical to what it was."""
    base, _overlay = _tree(tmp_path)
    calls, seen = _locked_load(monkeypatch, source=base)
    assert calls == ["dit_only"]
    assert seen["model_id"] == f"safetensors:{base}"
    assert seen["hybrid"] is None


def test_a_hybrid_load_is_a_different_model_id_than_its_base(monkeypatch, tmp_path):
    base, overlay = _tree(tmp_path)
    _calls, seen = _locked_load(monkeypatch, source=base, hybrid=_request(overlay))
    assert seen["model_id"].startswith(f"safetensors:{base}#hybrid1:")
    assert seen["hybrid"] is not None and seen["hybrid"].spec.overlay_dit_path == overlay


def test_the_same_hybrid_twice_returns_early(monkeypatch, tmp_path):
    base, overlay = _tree(tmp_path)
    _calls, seen = _locked_load(monkeypatch, source=base, hybrid=_request(overlay))
    calls, _ = _locked_load(monkeypatch, source=base, hybrid=_request(overlay),
                            current_model=seen["model_id"])
    assert calls == []


@pytest.mark.parametrize("change", [
    {"block_range_end": 4},
    {"final_adaln_from_overlay": True},
])
def test_a_changed_recipe_rebuilds_instead_of_returning_early(monkeypatch, tmp_path, change):
    """The failure this pins: `f"{source_type}:{source}"` is the same string for
    both requests, so the early return would answer the second one with the
    model already loaded."""
    base, overlay = _tree(tmp_path)
    _calls, first = _locked_load(monkeypatch, source=base, hybrid=_request(overlay))
    calls, second = _locked_load(monkeypatch, source=base,
                                 hybrid=_request(overlay, **change),
                                 current_model=first["model_id"])
    assert calls == ["dit_only"]
    assert second["model_id"] != first["model_id"]


def test_a_changed_overlay_rebuilds_instead_of_returning_early(monkeypatch, tmp_path):
    base, overlay = _tree(tmp_path)
    other_overlay = os.path.join(
        os.path.dirname(overlay), "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors")
    _write_fake_h3_dit(other_overlay, header=_h3_header())

    _calls, first = _locked_load(monkeypatch, source=base, hybrid=_request(overlay))
    calls, second = _locked_load(monkeypatch, source=base, hybrid=_request(other_overlay),
                                 current_model=first["model_id"])
    assert calls == ["dit_only"]
    assert second["model_id"] != first["model_id"]


def test_dropping_the_overlay_rebuilds_the_base_alone(monkeypatch, tmp_path):
    base, overlay = _tree(tmp_path)
    _calls, first = _locked_load(monkeypatch, source=base, hybrid=_request(overlay))
    calls, second = _locked_load(monkeypatch, source=base, current_model=first["model_id"])
    assert calls == ["dit_only"]
    assert second["model_id"] == f"safetensors:{base}"
    assert second["hybrid"] is None


def test_a_refused_preflight_never_reaches_the_loader(monkeypatch, tmp_path):
    """A hybrid refusal happens BEFORE the current model is touched."""
    base, overlay = _tree(tmp_path)
    live = _components()
    before = dict(live)
    with pytest.raises(MiniMaxH3HybridRefusal) as excinfo:
        _locked_load(monkeypatch, source=base, components=live,
                     hybrid={"overlay_file": overlay,
                             "block_range_start": 2, "block_range_end": 99})
    assert excinfo.value.code == "block_range_out_of_range"
    assert live == before


def test_a_hybrid_on_a_source_that_is_not_h3_refuses(monkeypatch, tmp_path):
    not_h3 = tmp_path / "sdxl.safetensors"
    not_h3.write_bytes(b"not a model")
    with pytest.raises(MiniMaxH3HybridRefusal) as excinfo:
        preflight_hybrid_request(str(not_h3), {"overlay_file": "o", **_RANGE})
    assert excinfo.value.code == "not_an_h3_tree"


# ---------------------------------------------------------------------------
# section 5.1 -- THE FILENAME TRAP
# ---------------------------------------------------------------------------

def test_a_hybrid_whose_base_is_named_fl2va_reports_variant_hybrid(monkeypatch, tmp_path):
    base, overlay = _tree(tmp_path)
    assert "fl2va" in os.path.basename(base)  # the substring the loader matches on
    _stub_build(monkeypatch)
    hybrid = _preflight(base, overlay)

    replacement = h3_reload.build_dit_only_reload(
        _components(), base, base, hybrid=hybrid)

    assert replacement["variant"] == "hybrid"
    assert replacement["base_variant"] == "fl2va"
    assert replacement["overlay_variant"] == "ref2va"
    assert replacement["hybrid_recipe"]["block_range_start"] == 2
    assert replacement["hybrid_provenance"]["base_file"] == os.path.basename(base)
    assert os.sep not in replacement["hybrid_provenance"]["overlay_file"]


def test_the_strip_list_covers_every_field_a_hybrid_writes(tmp_path):
    """`HYBRID_COMPONENT_KEYS` and `hybrid_component_fields` are two hand-kept
    lists of the same thing; the strip in `build_dit_only_reload` reads the
    tuple, so a field added only to the writer would survive a base-only reload.
    Every test that iterates the tuple passes vacuously in that case -- this one
    compares the two sources against each other.

    `variant` is excluded because the reload reassigns it unconditionally rather
    than stripping it.
    """
    base, overlay = _tree(tmp_path)
    fields = hybrid_component_fields(_preflight(base, overlay))
    assert set(fields) - {"variant"} == set(HYBRID_COMPONENT_KEYS)


def test_the_same_base_loaded_base_only_still_reports_fl2va(monkeypatch, tmp_path):
    """The control for the test above: `hybrid` is what changes the label, not
    the fixture. This is also the trap -- without an explicit assignment the
    hybrid would carry exactly this string."""
    base, _overlay = _tree(tmp_path)
    _stub_build(monkeypatch)
    replacement = h3_reload.build_dit_only_reload(_components(), base, base)
    assert replacement["variant"] == "fl2va"
    assert not any(key in replacement for key in HYBRID_COMPONENT_KEYS)


def test_the_c1_gates_refuse_that_hybrid(monkeypatch, tmp_path):
    """End to end: the variant a merged DiT reports reaches the route gates as
    `hybrid` and every one of them refuses it."""
    from api.arch_capabilities import chain_context_for
    from api.error_handlers import ValidationError
    from api.generation_utils import resolve_minimax_h3_outpaint_reference_gate
    from minimax_h3_hybrid_variant_gate_test import _app, _post, _StubPipelineManager

    base, overlay = _tree(tmp_path)
    _stub_build(monkeypatch)
    variant = h3_reload.build_dit_only_reload(
        _components(), base, base, hybrid=_preflight(base, overlay))["variant"]

    app = _app(monkeypatch, _StubPipelineManager(variant=variant),
               "/generate/txt2vid", "generate_txt2vid")
    status, payload = _post(app, "/generate/txt2vid", json={"prompt": "a cat"})
    assert status == 400, payload
    assert "hybrid" in payload["error"]

    with pytest.raises(ValidationError):
        resolve_minimax_h3_outpaint_reference_gate(
            variant, has_reference_images=True, placement="extend_forward")
    assert (chain_context_for("minimax_h3", variant)
            != chain_context_for("minimax_h3", "fl2va"))


# ---------------------------------------------------------------------------
# section 7 -- DiT-only reload atomicity, with a hybrid
# ---------------------------------------------------------------------------

def test_a_hybrid_reload_keeps_every_shared_component(monkeypatch, tmp_path):
    base, overlay = _tree(tmp_path)
    _stub_build(monkeypatch)
    current = _components()

    replacement = h3_reload.build_dit_only_reload(
        current, base, base, hybrid=_preflight(base, overlay))

    assert replacement is not current
    assert replacement["transformer"] is not current["transformer"]
    for name in ("text_encoder", "text_encoder_config", "tokenizer", "processor",
                 "vae", "vae_config", "audio_vae", "audio_vae_config",
                 "scheduler", "audio_scheduler"):
        assert replacement[name] is current[name]


def test_the_hybrid_reaches_the_transformer_build(monkeypatch, tmp_path):
    base, overlay = _tree(tmp_path)
    seen = []
    _stub_build(monkeypatch, seen)
    hybrid = _preflight(base, overlay)
    h3_reload.build_dit_only_reload(_components(), base, base, hybrid=hybrid)
    assert seen == [hybrid]


def test_a_failed_hybrid_build_leaves_the_live_components_untouched(monkeypatch, tmp_path):
    base, overlay = _tree(tmp_path)
    hybrid = _preflight(base, overlay)

    def fail(*_args, **_kwargs):
        raise RuntimeError("bad merge")

    monkeypatch.setattr(h3_reload, "_build_transformer", fail)
    current = _components()
    before = dict(current)

    with pytest.raises(RuntimeError, match="bad merge"):
        h3_reload.build_dit_only_reload(current, base, base, hybrid=hybrid)
    assert current == before
    assert current["variant"] == "fl2va"


def test_a_base_only_reload_strips_a_previous_hybrids_provenance(monkeypatch, tmp_path):
    """The copy in `build_dit_only_reload` carries the OLD dict's keys; a plain
    fl2va DiT must not keep reporting the recipe of the DiT it replaced."""
    base, overlay = _tree(tmp_path)
    _stub_build(monkeypatch)
    hybrid_components = h3_reload.build_dit_only_reload(
        _components(), base, base, hybrid=_preflight(base, overlay))

    replacement = h3_reload.build_dit_only_reload(hybrid_components, base, base)

    assert replacement["variant"] == "fl2va"
    assert not any(key in replacement for key in HYBRID_COMPONENT_KEYS)


def test_a_base_that_is_not_the_validated_one_refuses(monkeypatch, tmp_path):
    """Returning `None` here would serve a hybrid request as a base-only load."""
    base, overlay = _tree(tmp_path)
    _stub_build(monkeypatch)
    hybrid = _preflight(base, overlay)
    with pytest.raises(ValueError, match="validated"):
        h3_reload.build_dit_only_reload(_components(), base, overlay, hybrid=hybrid)


# ---------------------------------------------------------------------------
# the pipeline swap: model info and persistence
# ---------------------------------------------------------------------------

def _swap(monkeypatch, source, hybrid):
    import utils.hash_cache as hash_cache
    from core import pipeline as pipeline_module

    _stub_build(monkeypatch)
    monkeypatch.setattr(pipeline_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hash_cache, "get_cached_file_hash", lambda _path: "deadbeef")
    saved = {}
    manager = SimpleNamespace(
        minimax_h3_components=_components(),
        _save_last_model=lambda *args, **kwargs: saved.update(
            {"args": args, "kwargs": kwargs}),
        _minimax_h3_te_request=(None, None),
    )
    reloaded = pipeline_module.DiffusionPipelineManager._reload_minimax_h3_dit_only(
        manager, "safetensors", source, source, "txt2img",
        f"safetensors:{source}", hybrid=hybrid)
    return reloaded, manager, saved


def test_a_base_only_swap_records_no_hybrid_fields(monkeypatch, tmp_path):
    """Also pins the base-only CALL SHAPE: this stub takes three positional
    arguments and no keyword, exactly as the caller passed before C4."""
    base, _overlay = _tree(tmp_path)
    monkeypatch.setattr(
        h3_reload, "build_dit_only_reload",
        lambda components, current_source, source: dict(
            components, transformer=object(), variant="fl2va"))
    reloaded, manager, saved = _swap(monkeypatch, base, None)

    assert reloaded is True
    assert manager.current_model_info["variant"] == "fl2va"
    assert "hybrid" not in manager.current_model_info
    assert saved["kwargs"] == {}


def test_a_hybrid_swap_records_sanitised_provenance_and_persists_the_request(
        monkeypatch, tmp_path):
    base, overlay = _tree(tmp_path)
    hybrid = _preflight(base, overlay)
    _reloaded, manager, saved = _swap(monkeypatch, base, hybrid)

    info = manager.current_model_info
    assert info["variant"] == "hybrid"
    assert info["base_variant"] == "fl2va" and info["overlay_variant"] == "ref2va"
    assert info["hybrid"]["base_file"] == os.path.basename(base)
    assert info["hybrid"]["compatibility_digest"] == hybrid.spec.compatibility_digest
    # Section 5.4: basenames, not absolute paths.
    assert not any(isinstance(v, str) and os.sep in v for v in info["hybrid"].values())
    assert saved["kwargs"]["hybrid"] == hybrid_request_from_spec(hybrid.spec)


# ---------------------------------------------------------------------------
# section 5.4 -- generation metadata
# ---------------------------------------------------------------------------

def test_generation_metadata_records_the_pair_and_the_recipe(monkeypatch, tmp_path):
    from api.generation_utils import record_model_variant

    base, overlay = _tree(tmp_path)
    hybrid = _preflight(base, overlay)
    _reloaded, manager, _saved = _swap(monkeypatch, base, hybrid)
    manager.is_minimax_h3_model = True

    params = {}
    assert record_model_variant(params, manager) == "hybrid"
    assert params["model_variant"] == "hybrid"
    assert params["model_hybrid_base"] == os.path.basename(base)
    assert params["model_hybrid_overlay"] == os.path.basename(overlay)
    assert params["model_hybrid_preset"] == PRESET_BLOCK_RANGE_ADALN
    assert params["model_hybrid_block_range"] == "2..3"
    assert params["model_hybrid_final_adaln_from_overlay"] is False
    assert params["model_hybrid_digest"] == hybrid.spec.compatibility_digest
    assert params["model_hybrid_quantization"] == hybrid.quant_format
    # Every recorded value is a scalar the DB/metadata writers already handle,
    # and none of them is an absolute path.
    assert not any(isinstance(v, str) and os.sep in v for v in params.values())


def test_a_base_only_generation_records_no_hybrid_keys(monkeypatch, tmp_path):
    from api.generation_utils import record_model_variant

    manager = SimpleNamespace(
        is_minimax_h3_model=True,
        current_model_info={"type": "minimax_h3", "variant": "fl2va"})
    params = {}
    assert record_model_variant(params, manager) == "fl2va"
    assert list(params) == ["model_variant"]


# ---------------------------------------------------------------------------
# last_model.json
# ---------------------------------------------------------------------------

def _last_model_file(monkeypatch, tmp_path):
    from core import pipeline as pipeline_module

    path = tmp_path / "last_model.json"
    monkeypatch.setattr(pipeline_module, "LAST_MODEL_CONFIG_FILE", path)
    return pipeline_module, path


def test_a_base_only_save_writes_the_same_file_it_always_did(monkeypatch, tmp_path):
    pipeline_module, path = _last_model_file(monkeypatch, tmp_path)
    pipeline_module.DiffusionPipelineManager._save_last_model(
        SimpleNamespace(), "safetensors", "m.safetensors", "txt2img")
    assert json.loads(path.read_text()) == {
        "source_type": "safetensors", "source": "m.safetensors",
        "pipeline_type": "txt2img"}


def test_a_hybrid_save_round_trips_into_a_restored_load(monkeypatch, tmp_path):
    pipeline_module, path = _last_model_file(monkeypatch, tmp_path)
    base, overlay = _tree(tmp_path)
    request = hybrid_request_from_spec(_preflight(base, overlay).spec)

    pipeline_module.DiffusionPipelineManager._save_last_model(
        SimpleNamespace(), "safetensors", base, "txt2img", hybrid=request)
    assert json.loads(path.read_text())["hybrid"] == request

    calls = {}
    manager = SimpleNamespace(load_model=lambda **kwargs: calls.update(kwargs))
    pipeline_module.DiffusionPipelineManager._auto_load_last_model(manager)

    assert calls["source"] == base
    assert calls["hybrid"] == request
    # The digest is deliberately NOT persisted: the restore re-derives it, so a
    # replaced overlay is refused rather than merged silently.
    assert "compatibility_digest" not in calls["hybrid"]


@pytest.mark.parametrize("stored", [{}, [], "overlay.safetensors"])
def test_a_malformed_stored_hybrid_is_handed_on_and_refused_by_name(
        monkeypatch, tmp_path, stored):
    """A hand-edited or truncated entry must reach the refusal.

    The restore must NOT fold a falsy value to `None`: that would turn "this
    file asks for a merge I cannot read" into a base-only load reporting
    success, which is the degradation `normalize_hybrid_request` refuses for
    every other shape.
    """
    pipeline_module, path = _last_model_file(monkeypatch, tmp_path)
    path.write_text(json.dumps({
        "source_type": "safetensors", "source": "m.safetensors",
        "pipeline_type": "txt2img", "hybrid": stored}))

    seen = {}
    manager = SimpleNamespace(load_model=lambda **kwargs: seen.update(kwargs))
    pipeline_module.DiffusionPipelineManager._auto_load_last_model(manager)
    assert seen["hybrid"] == stored

    with pytest.raises(MiniMaxH3HybridRefusal):
        normalize_hybrid_request(seen["hybrid"])


def test_a_restored_base_only_load_passes_no_hybrid(monkeypatch, tmp_path):
    pipeline_module, _path = _last_model_file(monkeypatch, tmp_path)
    pipeline_module.DiffusionPipelineManager._save_last_model(
        SimpleNamespace(), "safetensors", "m.safetensors", "txt2img")

    calls = {}
    manager = SimpleNamespace(load_model=lambda **kwargs: calls.update(kwargs))
    pipeline_module.DiffusionPipelineManager._auto_load_last_model(manager)
    assert calls["hybrid"] is None
