"""Focused tests for the MiniMax-H3 shared-component reload path."""

import os
import sys
from types import MethodType, SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_h3 import reload as h3_reload  # noqa: E402


def _layout(root, dit, variant):
    return {
        "root": root,
        "official": os.path.join(root, "official"),
        "dit": dit,
        "vae": os.path.join(root, "vae", "video.safetensors"),
        "audio_vae": os.path.join(root, "vae", "audio.safetensors"),
        "text_encoder": os.path.join(root, "text_encoders", "qwen.safetensors"),
        "variant": variant,
    }


def _components():
    shared = {name: object() for name in (
        "text_encoder", "text_encoder_config", "tokenizer", "processor",
        "vae", "vae_config", "audio_vae", "audio_vae_config",
        "scheduler", "audio_scheduler",
    )}
    return {
        "type": "minimax_h3",
        "variant": "ref2va",
        "transformer": object(),
        "transformer_config": object(),
        **shared,
    }


def test_same_tree_reload_replaces_only_the_transformer(monkeypatch, tmp_path):
    root = str(tmp_path / "h3")
    old_source = os.path.join(root, "diffusion_models", "ref_fp8.safetensors")
    new_source = os.path.join(root, "diffusion_models", "ref_w4a8.safetensors")
    layouts = {
        old_source: _layout(root, old_source, "ref2va"),
        new_source: _layout(root, new_source, "ref2va"),
    }
    new_transformer = object()
    new_config = object()
    monkeypatch.setattr(h3_reload, "detect_minimax_h3_layout", layouts.get)
    monkeypatch.setattr(
        h3_reload, "_build_transformer",
        lambda path, dtype, official: (new_transformer, new_config),
    )

    current = _components()
    replacement = h3_reload.build_dit_only_reload(current, old_source, new_source)

    assert replacement is not current
    assert current["transformer"] is not new_transformer
    assert replacement["transformer"] is new_transformer
    assert replacement["transformer_config"] is new_config
    for name in (
        "text_encoder", "text_encoder_config", "tokenizer", "processor",
        "vae", "vae_config", "audio_vae", "audio_vae_config",
        "scheduler", "audio_scheduler",
    ):
        assert replacement[name] is current[name]


def test_different_tree_falls_back_without_building(monkeypatch, tmp_path):
    old_root = str(tmp_path / "old")
    new_root = str(tmp_path / "new")
    old_source = os.path.join(old_root, "diffusion_models", "ref.safetensors")
    new_source = os.path.join(new_root, "diffusion_models", "ref.safetensors")
    layouts = {
        old_source: _layout(old_root, old_source, "ref2va"),
        new_source: _layout(new_root, new_source, "ref2va"),
    }
    monkeypatch.setattr(h3_reload, "detect_minimax_h3_layout", layouts.get)
    monkeypatch.setattr(
        h3_reload, "_build_transformer",
        lambda *args: pytest.fail("a different tree must use the full loader"),
    )

    assert h3_reload.build_dit_only_reload(
        _components(), old_source, new_source) is None


def test_failed_transformer_build_leaves_current_components_untouched(monkeypatch, tmp_path):
    root = str(tmp_path / "h3")
    old_source = os.path.join(root, "diffusion_models", "ref_fp8.safetensors")
    new_source = os.path.join(root, "diffusion_models", "ref_w4a8.safetensors")
    layouts = {
        old_source: _layout(root, old_source, "ref2va"),
        new_source: _layout(root, new_source, "ref2va"),
    }
    monkeypatch.setattr(h3_reload, "detect_minimax_h3_layout", layouts.get)

    def fail(*_args):
        raise RuntimeError("bad replacement")

    monkeypatch.setattr(h3_reload, "_build_transformer", fail)
    current = _components()
    old_transformer = current["transformer"]

    with pytest.raises(RuntimeError, match="bad replacement"):
        h3_reload.build_dit_only_reload(current, old_source, new_source)
    assert current["transformer"] is old_transformer


def test_a_failed_hybrid_build_leaves_current_components_untouched(monkeypatch, tmp_path):
    """The same guarantee as the test above, for a merged DiT.

    A hybrid changes only WHICH tensors the replacement is built from, so the
    build-then-swap order that protects a base-only reload has to keep
    protecting this one. The recipe/identity side is in
    `minimax_h3_hybrid_lifecycle_test.py`; this file owns the atomicity.
    """
    root = str(tmp_path / "h3")
    base = os.path.join(root, "diffusion_models", "minimax_h3_fl2va.safetensors")
    layouts = {base: _layout(root, base, "fl2va")}
    hybrid = SimpleNamespace(spec=SimpleNamespace(base_dit_path=base))
    monkeypatch.setattr(h3_reload, "detect_minimax_h3_layout", layouts.get)

    def fail(*_args, **_kwargs):
        raise RuntimeError("bad merge")

    monkeypatch.setattr(h3_reload, "_build_transformer", fail)
    current = _components()
    before = dict(current)

    with pytest.raises(RuntimeError, match="bad merge"):
        h3_reload.build_dit_only_reload(current, base, base, hybrid=hybrid)
    assert current == before


def test_a_hybrid_whose_base_is_not_the_validated_one_refuses(monkeypatch, tmp_path):
    """Falling back to the full loader here would serve a hybrid request as a
    base-only load of the same file and report success."""
    root = str(tmp_path / "h3")
    base = os.path.join(root, "diffusion_models", "minimax_h3_fl2va.safetensors")
    other = os.path.join(root, "diffusion_models", "minimax_h3_ref2va.safetensors")
    layouts = {base: _layout(root, base, "fl2va"), other: _layout(root, other, "ref2va")}
    monkeypatch.setattr(h3_reload, "detect_minimax_h3_layout", layouts.get)
    monkeypatch.setattr(
        h3_reload, "_build_transformer",
        lambda *args, **kwargs: pytest.fail("nothing may be built for a mismatched base"))

    hybrid = SimpleNamespace(spec=SimpleNamespace(base_dit_path=base))
    with pytest.raises(ValueError, match="validated"):
        h3_reload.build_dit_only_reload(_components(), base, other, hybrid=hybrid)


def _health_probe(existing_info, failure_leaves):
    """Run load_model with a failing _load_model_locked and report the health.

    failure_leaves is what current_model_info holds once the load has failed:
    the same object (the live model survived), a different one (a partial
    state), or None (nothing loaded).
    """
    import threading
    from core import pipeline as pipeline_module

    manager = SimpleNamespace(
        current_model_info=existing_info,
        current_model="safetensors:old.safetensors",
        component_health="ready",
        model_revision=3,
        component_revision=9,
        _load_model_lock=threading.Lock(),
    )

    def failing_load(*args, **kwargs):
        manager.current_model_info = failure_leaves
        raise RuntimeError("build failed")

    manager._load_model_locked = failing_load
    with pytest.raises(RuntimeError, match="build failed"):
        pipeline_module.DiffusionPipelineManager.load_model(
            manager, "safetensors", "new.safetensors", "txt2img")
    return manager.component_health


def test_failed_load_that_keeps_the_live_model_stays_ready():
    """The DiT-only path retains the current model when its build fails.

    Marking that degraded would 503 every generation against a model that is
    still fully loaded, and re-selecting it could not clear the flag.
    """
    live = {"source": "old.safetensors"}
    assert _health_probe(live, failure_leaves=live) == "ready"


def test_failed_load_that_replaced_the_model_is_degraded():
    live = {"source": "old.safetensors"}
    assert _health_probe(live, failure_leaves={"source": "half.safetensors"}) == "degraded"


def test_failed_load_that_unloaded_everything_reports_unloaded():
    live = {"source": "old.safetensors"}
    assert _health_probe(live, failure_leaves=None) == "unloaded"


class _ReachedFullReload(Exception):
    """Raised from the first step past both same-model shortcuts."""


def _same_model_reload(monkeypatch, health):
    """Re-select the already-loaded checkpoint and report where it got to."""
    from core import pipeline as pipeline_module
    import core.keep_hot as keep_hot

    calls = []

    def dit_only(*_args, **_kwargs):
        calls.append("dit_only")
        return True

    manager = SimpleNamespace(
        current_model="safetensors:same.safetensors",
        current_model_info={"source": "same.safetensors"},
        component_health=health,
        is_minimax_h3_model=True,
        minimax_h3_components={},
        _minimax_h3_te_selection_differs=lambda *_a: False,
        _reload_minimax_h3_dit_only=dit_only,
        # See minimax_h3_hybrid_lifecycle_test.py's matching fixture comment:
        # `_load_model_locked` also gates on the MiniMax Music 3 selection,
        # and this bare SimpleNamespace has no real class to inherit it from.
        is_minimax_music3_model=False,
        minimax_music3_components={},
    )
    # The REAL implementation, bound via `types.MethodType`, not a stub
    # (audit F5) -- see minimax_h3_hybrid_lifecycle_test.py's matching
    # comment for why a stub here would hide the `is_minimax_music3_model`
    # guard from every test using this fixture.
    manager._minimax_music3_te_selection_differs = MethodType(
        pipeline_module.DiffusionPipelineManager._minimax_music3_te_selection_differs, manager)

    def stop(_manager):
        calls.append("full_reload")
        raise _ReachedFullReload()

    monkeypatch.setattr(keep_hot, "clear_resident", stop)
    try:
        pipeline_module.DiffusionPipelineManager._load_model_locked(
            manager, "safetensors", "same.safetensors", "txt2img")
    except _ReachedFullReload:
        pass
    return calls


def test_same_model_reload_is_a_no_op_when_healthy(monkeypatch):
    assert _same_model_reload(monkeypatch, "ready") == []


def test_degraded_model_reloads_fully_instead_of_returning_early(monkeypatch):
    """Re-selecting the same checkpoint while degraded is a repair request.

    Neither shortcut may serve it: the early return does nothing at all, and
    the DiT-only path carries the existing text encoder over untouched -- which
    is the very slot a failed switch leaves empty.
    """
    assert _same_model_reload(monkeypatch, "degraded") == ["full_reload"]


def test_pipeline_swap_updates_model_state_after_success(monkeypatch):
    from core import pipeline as pipeline_module

    current = _components()
    replacement = dict(current, transformer=object(), transformer_config=object())
    monkeypatch.setattr(
        h3_reload, "build_dit_only_reload",
        lambda components, current_source, source: replacement,
    )
    monkeypatch.setattr(pipeline_module.torch.cuda, "is_available", lambda: False)
    manager = SimpleNamespace(
        minimax_h3_components=current,
        _save_last_model=lambda *args: None,
        # The load-time TE/projection request this bundle came from; a DiT-only
        # reload replays it so `last_model.json` keeps naming the same pairing.
        _minimax_h3_te_request=(None, None),
    )

    reloaded = pipeline_module.DiffusionPipelineManager._reload_minimax_h3_dit_only(
        manager,
        "safetensors",
        "new.safetensors",
        "old.safetensors",
        "txt2img",
        "safetensors:new.safetensors",
    )

    assert reloaded is True
    assert manager.minimax_h3_components is replacement
    assert manager.current_model == "safetensors:new.safetensors"
    assert manager.current_model_info["source"] == "new.safetensors"
    assert manager._runtime_int8_converted is False
