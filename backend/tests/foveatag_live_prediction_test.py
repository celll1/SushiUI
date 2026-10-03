"""Native training-model inference without source-image or GPU reads."""

import io
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.processor import FoveaProcessor
from core.tagger.foveatag.trainer_model import FoveaTrainerModel
from core.tagger.foveatag.views import ViewConfig
from core.tagger.tagger_offload import TaggerTrainerHandle


def image_bytes():
    values = np.arange(96 * 48 * 3, dtype=np.uint8).reshape(48, 96, 3)
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format="PNG")
    return buffer.getvalue()


def handle(model, processor, *, cap=10000):
    result = TaggerTrainerHandle("native-test", "", *[threading.Event() for _ in range(3)])
    vocab = SimpleNamespace(idx_to_tag={i: f"tag_{i}" for i in range(5)}, tag_to_category={})
    result.attach(model, None, None, processor=processor, vocabulary=vocab, max_source_pixels=cap)
    return result


@pytest.mark.parametrize("method", ["full", "lora"])
@pytest.mark.parametrize("training", [False, True])
def test_live_native_predict_deterministic_without_foreign_assist(monkeypatch, method, training):
    model = FoveaTrainerModel(5, encoder_config=FoveaEncoderConfig(
        widths=(8, 16, 32, 32), depths=(1, 1, 1, 1), attention_heads=(2, 2), scale_dropout=0),
        feature_dim=16, evidence_slots=2, pool_heads=2, temperature=.1,
        training_method=method, lora_rank=2, lora_alpha=2)
    processor = FoveaProcessor(ViewConfig(overview_size=32, local_size=32, local_views=2),
                              seed=13, training=True)
    result = handle(model, processor)
    monkeypatch.setattr(result, "can_predict", lambda: True)
    model.train(training)
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    state = torch.get_rng_state().clone()
    assist = {"ood_ref": {"mu": np.zeros(1)}, "name_calibration": {"tag_0": [0.]},
              "name_metrics": {"tag_0": (.9, .9, 100)}}
    first = result.predict(image_bytes(), 0., assist=assist, use_ood_detection=True,
                           use_calibration=True, use_per_tag_threshold=True)
    second = result.predict(image_bytes(), 0.)
    assert first == second
    assert first["source"] == "training_model"
    assert first["ood_distance"] is None and not first["has_calibration"]
    assert model.training == training and processor.training
    assert torch.equal(state, torch.get_rng_state())
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights[key])


def test_cap_refused_before_decode(monkeypatch):
    model = torch.nn.Linear(3, 5)
    result = handle(model, None, cap=1)
    monkeypatch.setattr(result, "can_predict", lambda: True)
    def forbidden(*args, **kwargs):
        raise AssertionError("decode must not run")
    payload = image_bytes()
    monkeypatch.setattr(Image.Image, "convert", forbidden)
    with pytest.raises(ValueError, match="source-pixel cap"):
        result.predict(payload, .5)


def test_inference_failure_restores_training_mode(monkeypatch):
    class BrokenModel(torch.nn.Linear):
        architecture = "foveatag"
        def forward(self, views):
            assert not self.training
            raise ValueError("synthetic forward failure")
    model = BrokenModel(3, 5)
    processor = FoveaProcessor(ViewConfig(overview_size=32, local_size=32, local_views=0),
                              seed=13, training=False)
    result = handle(model, processor)
    monkeypatch.setattr(result, "can_predict", lambda: True)
    with pytest.raises(ValueError, match="synthetic forward failure"):
        result.predict(image_bytes(), .5)
    assert model.training
