"""Saved ACE checkpoints must reach the audio loader through the public entry."""
import pytest
import torch
from safetensors.torch import save_file


@pytest.mark.parametrize("prefix", ["", "net."])
def test_public_single_file_load_dispatches_declared_ace(tmp_path, monkeypatch, prefix):
    from core.model_loader import ModelLoader

    path = tmp_path / "saved_full.safetensors"
    save_file({prefix + "null_condition_emb": torch.zeros(1, 2),
               prefix + "decoder.layers.0.weight": torch.ones(2, 2)}, str(path),
              metadata={"model_type": "acestep"})
    owner = {"type": "acestep"}
    calls = []

    def load(source, dtype):
        calls.append((source, dtype))
        return owner

    monkeypatch.setattr(ModelLoader, "load_acestep_from_path", load)
    result = ModelLoader.load_model("safetensors", str(path), device="cpu")
    assert result is owner
    assert calls == [(str(path), torch.bfloat16)]
