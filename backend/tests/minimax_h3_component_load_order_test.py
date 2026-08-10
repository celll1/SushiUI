"""MiniMax-H3 maps its largest component before the other checkpoints."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_h3 import loader


def test_text_encoder_is_mapped_before_transformer_and_vaes(tmp_path, monkeypatch):
    official = tmp_path / "official"
    for component in ("text_encoder", "vae", "audio_vae"):
        directory = official / component
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text("{}", encoding="utf-8")

    layout = {
        "root": str(tmp_path),
        "dit": str(tmp_path / "dit.safetensors"),
        "vae": str(tmp_path / "video_vae.safetensors"),
        "audio_vae": str(tmp_path / "audio_vae.safetensors"),
        "text_encoder": str(tmp_path / "text_encoder.safetensors"),
        "official": str(official),
        "variant": "ref2va",
    }
    calls = []

    monkeypatch.setattr(loader, "detect_minimax_h3_layout", lambda _path: layout)
    monkeypatch.setattr(loader, "_build_text_encoder",
                        lambda *_args: (calls.append("text_encoder") or object(), object()))
    monkeypatch.setattr(loader, "_build_transformer",
                        lambda *_args: (calls.append("transformer") or object(), object()))
    monkeypatch.setattr(loader, "_build_video_vae",
                        lambda *_args: (calls.append("video_vae") or object(), {}))
    monkeypatch.setattr(loader, "_build_audio_vae",
                        lambda *_args: (calls.append("audio_vae") or object(), {}))
    monkeypatch.setattr(loader, "_load_tokenizer_and_processor", lambda *_args: (None, None))
    monkeypatch.setattr(loader, "_load_schedulers", lambda *_args: (object(), object()))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = loader.load_minimax_h3_from_path(layout["dit"])

    assert calls == ["text_encoder", "transformer", "video_vae", "audio_vae"]
    assert result["text_encoder"] is not None


def test_geometry_only_load_still_skips_text_encoder(tmp_path, monkeypatch):
    official = tmp_path / "official"
    for component in ("vae", "audio_vae"):
        directory = official / component
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text("{}", encoding="utf-8")

    layout = {
        "root": str(tmp_path),
        "dit": str(tmp_path / "dit.safetensors"),
        "vae": str(tmp_path / "video_vae.safetensors"),
        "audio_vae": str(tmp_path / "audio_vae.safetensors"),
        "text_encoder": None,
        "official": str(official),
        "variant": "fl2va",
    }
    calls = []

    monkeypatch.setattr(loader, "detect_minimax_h3_layout", lambda _path: layout)
    monkeypatch.setattr(loader, "_build_text_encoder",
                        lambda *_args: calls.append("text_encoder"))
    monkeypatch.setattr(loader, "_build_transformer", lambda *_args: (object(), object()))
    monkeypatch.setattr(loader, "_build_video_vae", lambda *_args: (object(), {}))
    monkeypatch.setattr(loader, "_build_audio_vae", lambda *_args: (object(), {}))
    monkeypatch.setattr(loader, "_load_tokenizer_and_processor", lambda *_args: (None, None))
    monkeypatch.setattr(loader, "_load_schedulers", lambda *_args: (object(), object()))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = loader.load_minimax_h3_from_path(layout["dit"], load_text_encoder=False)

    assert calls == []
    assert result["text_encoder"] is None
