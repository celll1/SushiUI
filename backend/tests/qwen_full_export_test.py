import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from api.training_export_job import latest_complete_checkpoint
from core.models.qwen_image_21.artifact import artifact_metadata, load_manifest
from core.training import qwen_full_export


def test_latest_complete_checkpoint_requires_optimizer_and_all_shards(tmp_path):
    name = "example"
    for step in (1, 2):
        stem = f"{name}_step_{step:06d}"
        (tmp_path / f"{stem}_state.json").write_text("{}", encoding="utf-8")
        (tmp_path / f"{stem}_optimizer.pt").write_bytes(b"optimizer")
        (tmp_path / f"{stem}.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": f"{stem}-00001-of-00001.safetensors"}}),
            encoding="utf-8",
        )
    (tmp_path / f"{name}_step_000001-00001-of-00001.safetensors").write_bytes(b"weights")
    assert latest_complete_checkpoint(tmp_path, name).name == f"{name}_step_000001.safetensors.index.json"


def test_qwen_full_export_keeps_float_checkpoint_and_publishes_quantized_artifact(tmp_path, monkeypatch):
    companion = tmp_path / "source"
    companion.mkdir()
    (companion / "processor").mkdir()
    (companion / "text_encoder_config.json").write_text("{}", encoding="utf-8")
    config = {"width": 256}
    for component in ("transformer", "text_encoder", "vae"):
        tensors = {"linear.weight": torch.randn(256, 256, dtype=torch.bfloat16)} if component != "vae" else {
            "dummy": torch.ones(1, dtype=torch.bfloat16)
        }
        save_file(tensors, companion / f"{component}.safetensors",
                  metadata=artifact_metadata(component, config))
    (companion / "manifest.json").write_text(json.dumps({
        "model_type": "qwen_image_21", "format_version": "1", "variant": "bf16",
        "components": {
            "transformer": "transformer.safetensors", "text_encoder": "text_encoder.safetensors",
            "vae": "vae.safetensors", "processor": "processor",
            "text_encoder_config": "text_encoder_config.json",
        },
    }), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "example_step_000010.safetensors"
    metadata = artifact_metadata("training_bundle", config)
    metadata.update(step="10", companion_path=str(companion))
    save_file({
        "transformer.linear.weight": torch.randn(256, 256, dtype=torch.bfloat16),
        "text_encoder.linear.weight": torch.randn(256, 256, dtype=torch.bfloat16),
    }, checkpoint, metadata=metadata)
    monkeypatch.setattr(qwen_full_export, "_linear_names", lambda *_: {"linear"})
    exported = qwen_full_export.export_qwen_full_checkpoint(checkpoint, run_dir)
    manifest = load_manifest(str(exported))
    assert checkpoint.exists()
    for member in (manifest.transformer, manifest.text_encoder):
        with safe_open(member, framework="pt", device="cpu") as handle:
            assert handle.metadata()["variant"] == "int8_convrot"
            assert handle.get_tensor("linear.weight").dtype == torch.int8
            assert "linear.weight_scale" in handle.keys()
    with pytest.raises(FileExistsError):
        qwen_full_export.export_qwen_full_checkpoint(checkpoint, run_dir)
