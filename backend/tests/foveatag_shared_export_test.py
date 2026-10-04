"""Inference exports cannot change the resumable LoRA parameterization."""

from dataclasses import asdict
import json

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.pretraining_checkpoint import save_pretrained_encoder
from core.tagger.foveatag.processor import FoveaProcessor
from core.tagger.foveatag.trainer_model import FoveaTrainerModel, load_inference_export
from core.tagger.foveatag.views import ViewConfig


def model(method="lora"):
    return FoveaTrainerModel(5,
        encoder_config=FoveaEncoderConfig(widths=(8, 16, 32, 32), depths=(1, 1, 1, 1),
            attention_heads=(2, 2), scale_dropout=0),
        feature_dim=16, evidence_slots=2, pool_heads=2, temperature=.1,
        training_method=method, lora_rank=2, lora_alpha=2)


def batch():
    return FoveaProcessor(ViewConfig(overview_size=32, local_size=32, local_views=2),
        seed=42, training=False)(images=[Image.new("RGB", (96, 48), (32, 64, 128))],
                                return_tensors="pt")["fovea_views"]


@pytest.mark.parametrize("mode", ["lora", "merged"])
def test_export_preserves_live_model_optimizer_and_exact_resume(mode, tmp_path):
    torch.manual_seed(29)
    instance = model()
    inputs = batch()
    optimizer = torch.optim.AdamW(instance.trainable_parameters(), lr=.01)
    instance(inputs).square().mean().backward()
    optimizer.step()
    instance.eval()
    expected = instance(inputs).detach()
    before = {key: value.clone() for key, value in instance.state_dict().items()}
    parameter_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    path = instance.save_checkpoint(tmp_path, "latest", {"checkpoint_save_mode": mode})
    assert parameter_ids == [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    for key, value in before.items():
        torch.testing.assert_close(value, instance.state_dict()[key], rtol=0, atol=0)
    resumed = model()
    resumed.load_weights_inplace(path)
    resumed.eval()
    torch.testing.assert_close(resumed(inputs), expected, rtol=0, atol=0)
    metadata = json.loads((tmp_path / "latest_metadata.json").read_text())
    assert metadata["resume_weight_mode"] == "full_unmerged"
    assert metadata["inference_export"] == "latest_export.safetensors"
    base_path = None
    if mode == "lora":
        # Export the actual frozen scratch backbone, not merged adapter weights.
        base = model("full").vision_encoder
        base.load_state_dict(instance.frozen_encoder_state(), strict=True)
        base_path = save_pretrained_encoder(tmp_path / "scratch.safetensors", base,
            run_contract={"encoder": asdict(base.config)}, epoch=1, global_step=1, validation={})
    exported, export_metadata = load_inference_export(tmp_path / metadata["inference_export"],
                                                       base_encoder_path=base_path)
    torch.testing.assert_close(exported(inputs), expected, rtol=2e-5, atol=2e-6)
    assert export_metadata["contains_frozen_base"] == (mode == "merged")
    if mode == "lora":
        assert (tmp_path / "latest_export.safetensors").stat().st_size < (tmp_path / "latest.safetensors").stat().st_size
    else:
        assert not any("lora_A" in key or "lora_B" in key or ".base." in key
                       for key in load_file(str(tmp_path / "latest_export.safetensors")))


def test_compact_export_refuses_missing_or_wrong_base(tmp_path):
    original = model()
    original.save_checkpoint(tmp_path, "latest", {"checkpoint_save_mode": "lora"})
    export_path = tmp_path / "latest_export.safetensors"
    with pytest.raises(ValueError, match="requires its scratch backbone"):
        load_inference_export(export_path)
    different = model("full").vision_encoder
    base_path = save_pretrained_encoder(tmp_path / "different.safetensors", different,
        run_contract={"encoder": asdict(different.config)}, epoch=1, global_step=1, validation={})
    with pytest.raises(ValueError, match="fingerprint differs"):
        load_inference_export(export_path, base_encoder_path=base_path)


def test_invalid_export_mode_does_not_write_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="save mode"):
        model().save_checkpoint(tmp_path, "invalid", {"checkpoint_save_mode": "unknown"})
    assert not list(tmp_path.iterdir())


def test_shared_dispatch_and_retention_include_native_exports(tmp_path, monkeypatch):
    initially_initialized = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **k: None)
    monkeypatch.setattr(torch._C, "_cuda_init", lambda *a, **k: None)
    from core.tagger.tagger_trainer import _save_model_checkpoint, _prune_step_checkpoints
    instance = model()
    for name in ("latest", "best_f1", "step_000001", "step_000002"):
        path = _save_model_checkpoint(instance, str(tmp_path), name, {}, "merged")
        assert path == str(tmp_path / f"{name}.safetensors")
        assert (tmp_path / f"{name}_export.safetensors").is_file()
    _prune_step_checkpoints(str(tmp_path), 1)
    assert not any(path.name.startswith("step_000001") for path in tmp_path.iterdir())
    for name in ("latest", "best_f1", "step_000002"):
        for suffix in (".safetensors", "_metadata.json", "_export.safetensors", "_export_metadata.json"):
            assert (tmp_path / f"{name}{suffix}").is_file()
    assert torch.cuda.is_initialized() == initially_initialized
