"""Focused tests for the MiniMax-H3 shared-component reload path."""

import os
import sys
from types import SimpleNamespace

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
