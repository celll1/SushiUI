"""Load native training artifacts through the existing inference API manager."""

from dataclasses import asdict
import io
import json

import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

from core.tagger.foveatag.pretraining_checkpoint import save_pretrained_encoder
from core.tagger.foveatag.views import ViewConfig
from core.tagger.siglip2_inference_manager import SigLIP2InferenceManager
from core.tagger.tag_metrics_accumulator import TagMetricsAccumulator
from foveatag_shared_export_test import model, batch


def artifact(tmp_path, method, mode):
    instance = model(method).eval()
    meta = {"checkpoint_save_mode": mode, "fovea_view_config": asdict(ViewConfig(
        overview_size=32, local_size=32, local_views=2)), "max_source_pixels": 10000,
        "fovea_seed": 42, "mixed_precision": "fp32"}
    primary = instance.save_checkpoint(tmp_path, "best_f1", meta)
    vocab = {"idx_to_tag": {str(i): f"tag_{i}" for i in range(5)}, "tag_to_category": {}}
    (tmp_path / "best_f1_vocabulary.json").write_text(json.dumps(vocab))
    metrics = TagMetricsAccumulator(5)
    metrics.update(torch.full((10, 5), .6), torch.ones(10, 5))
    metrics.save(str(tmp_path / "best_f1_tag_metrics.npz"))
    base_path = None
    if mode == "lora":
        base = model("full").vision_encoder
        base.load_state_dict(instance.frozen_encoder_state(), strict=True)
        base_path = save_pretrained_encoder(tmp_path / "scratch.safetensors", base,
            run_contract={"encoder": asdict(base.config)}, epoch=1, global_step=1, validation={})
    path = primary if mode == "full" else str(tmp_path / "best_f1_export.safetensors")
    return instance, path, base_path


@pytest.mark.parametrize("method,mode", [("full", "full"), ("lora", "full"),
                                        ("lora", "lora"), ("lora", "merged")])
def test_native_manager_load_predict_without_pretrained_processor(tmp_path, monkeypatch, method, mode):
    instance, path, base = artifact(tmp_path, method, mode)
    def forbidden(*args, **kwargs):
        raise AssertionError("native load must not fetch a pretrained processor")
    monkeypatch.setattr(AutoProcessor, "from_pretrained", forbidden)
    manager = SigLIP2InferenceManager()
    manager.device = "cpu"
    manager.load_model(path, "ignored.json", vision_encoder_path=str(base or ""))
    assert manager.status["architecture"] == "foveatag"
    assert manager.status["has_tag_metrics"]
    assert manager.build_training_assist(want_per_tag=True, want_calibration=True,
        want_ood=False)["architecture"] == "foveatag"
    assert manager.get_tag_metrics_path() == str(tmp_path / "best_f1_tag_metrics.npz")
    payload = io.BytesIO()
    Image.new("RGB", (96, 48), (32, 64, 128)).save(payload, format="PNG")
    result = manager.predict(payload.getvalue(), threshold=0)
    expected = torch.sigmoid(instance(batch())[0]).detach().numpy()
    actual = {entry["tag"]: entry["prob"] for entry in result["tags"]}
    for index, probability in enumerate(expected):
        assert actual[f"tag_{index}"] == pytest.approx(float(probability), rel=3e-5, abs=3e-6)
    assert result == manager.predict(payload.getvalue(), threshold=0, max_num_patches=1)
    assert manager.predict(payload.getvalue(), use_calibration=True)["has_calibration"]
    manager.unload()
    assert not manager.status["loaded"] and manager.native_sidecar_base is None


def test_native_manager_ood_uses_native_feature_tap(tmp_path):
    import numpy as np
    instance, path, _ = artifact(tmp_path, "full", "full")
    captured = []
    hook = instance.register_ood_feature_hook(lambda module, args: captured.append(args[0].detach()))
    instance(batch())
    hook.remove()
    assert captured[0].shape == (1, 32)
    np.savez(tmp_path / "best_f1_ood_ref.npz", mu=np.zeros(32), cov_inv=np.eye(32),
             p50=np.float64(1), p95=np.float64(2))
    manager = SigLIP2InferenceManager()
    manager.device = "cpu"
    manager.load_model(path, "")
    payload = io.BytesIO()
    Image.new("RGB", (96, 48), (32, 64, 128)).save(payload, format="PNG")
    result = manager.predict(payload.getvalue(), use_ood_detection=True)
    assert result["ood_distance"] is not None
    assert manager._last_cls_emb.shape == (32,)


def test_native_manager_refuses_wrong_vocab_and_missing_view_contract(tmp_path):
    _, path, _ = artifact(tmp_path, "full", "full")
    manager = SigLIP2InferenceManager()
    manager.device = "cpu"
    vocab_path = tmp_path / "best_f1_vocabulary.json"
    vocab = json.loads(vocab_path.read_text())
    vocab["idx_to_tag"].pop("4")
    vocab_path.write_text(json.dumps(vocab))
    with pytest.raises(ValueError, match="vocabulary differs"):
        manager.load_model(path, "")
    assert not manager.status["loaded"]
    metadata_path = tmp_path / "best_f1_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("fovea_view_config")
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(KeyError, match="fovea_view_config"):
        manager.load_model(path, "")
