"""Actual-forward diagnostics, threshold replay and common learned routing."""
import json
import os
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from api.param_defaults import TAGGER_TRAINING_DEFAULTS
from core.tagger.diagnostic_capture import DiagnosticBatch, capture_forward, training_forward
from core.tagger.diagnostic_thresholds import decision_thresholds
from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.joint_metrics import JointMetricsRuntime
from core.tagger.foveatag.processor import FoveaProcessor
from core.tagger.foveatag.trainer_model import FoveaTrainerModel, load_native_checkpoint
from core.tagger.foveatag.views import ViewConfig
from core.tagger.tagger_loss import AsymmetricLossOptimized
from core.tagger.tag_vocabulary import TagVocabulary


def vocabulary():
    return TagVocabulary.from_dict({"tag_to_idx": {"synthetic_a": 0, "synthetic_b": 1},
        "idx_to_tag": {"0": "synthetic_a", "1": "synthetic_b"}, "tag_to_category": {}, "num_tags": 2})


def routed_model(method="full"):
    return FoveaTrainerModel(2, encoder_config=FoveaEncoderConfig(widths=(8,16,32,32),
        depths=(1,1,2,1), attention_heads=(2,2), scale_dropout=0), feature_dim=16,
        evidence_slots=2, pool_heads=2, temperature=.1, training_method=method, lora_rank=2, lora_alpha=2,
        routing={"candidate_count":4,"selection_budget":2,"random_count":1,"oracle_interval":1,
            "oracle_chunk_size":2,"router_weight":1.,"routing_dim":16,"routing_slots":2})


def processor():
    return FoveaProcessor(ViewConfig(overview_size=32,local_size=32,local_views=4,
        overview_resize_mode="stretch",local_border_mode="shift_inside"),seed=42,training=False)


@pytest.mark.parametrize("method", ["full", "lora"])
def test_common_routed_training_and_staged_inference_match(method, tmp_path):
    torch.manual_seed(42)
    model, proc = routed_model(method), processor()
    image = Image.new("RGB", (96,48), (20,60,100))
    inputs = proc(images=[image],return_tensors="pt")["fovea_views"]
    labels,masks = torch.tensor([[1.,0.]]), torch.ones(1,2)
    logits,classification,total,focus = training_forward(model, inputs, None,None,labels,masks,AsymmetricLossOptimized(),0)
    total.backward()
    assert focus[0]["boxes"].shape == (2,4)
    assert focus[0]["random_slots"].tolist() == [False,True]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.router.parameters())
    assert model.head.weight.grad is not None
    assert torch.isfinite(classification)
    model.eval()
    expected = model(inputs)
    torch.testing.assert_close(model.predict_native(image,proc), expected)
    path = model.save_checkpoint(tmp_path,"routed", {"fovea_view_config":asdict(proc.config)})
    restored, _ = load_native_checkpoint(path)
    restored.eval()
    assert restored.routing == model.routing
    torch.testing.assert_close(restored.predict_native(image,proc), expected)


def test_per_tag_thresholds_use_captured_support_and_floor():
    snapshot={"probabilities":torch.zeros(4),"threshold":.35,
        "tag_metrics":{"best_thr":torch.tensor([.01,.7,float("nan"),.8]),"n_pos":torch.tensor([5,5,10,4])}}
    settings={"threshold_mode":"per_tag","threshold":.4,"min_samples_for_per_tag":5,"min_best_thr":.05}
    vector,fallback,count=decision_thresholds(snapshot,settings)
    np.testing.assert_allclose(vector,[.05,.7,.4,.4])
    assert fallback == .4 and count == 2
    settings["threshold_mode"]="training"
    np.testing.assert_allclose(decision_thresholds(snapshot,settings)[0],[.35]*4)


@pytest.mark.parametrize("loss_name", ["asl", "cs_asl", "h_cs_asl", "la_s_asl", "fw_bbce"])
def test_routed_oracle_keeps_configured_weighted_loss(loss_name, monkeypatch):
    from core.tagger.tagger_loss import CSASL, HCSASL, LASASL, FWBBCE
    import core.tagger.foveatag.router_training as routing
    pi, positives, negatives = torch.tensor([.2,.8]), torch.tensor([2.,8.]), torch.tensor([8.,2.])
    losses = {"asl": AsymmetricLossOptimized(), "cs_asl": CSASL(pi=pi),
        "h_cs_asl": HCSASL(pi=pi,N_pos=positives,N_neg=negatives),
        "la_s_asl": LASASL(pi=pi), "fw_bbce": FWBBCE(pi=pi,N_pos=positives,N_neg=negatives)}
    criterion = losses[loss_name]
    original = routing.router_objective
    observed = []
    def spy(*args, **kwargs):
        observed.append(kwargs["criterion"])
        return original(*args, **kwargs)
    monkeypatch.setattr(routing, "router_objective", spy)
    model, proc = routed_model(), processor()
    inputs = proc(images=[Image.new("RGB", (96,48))],return_tensors="pt")["fovea_views"]
    logits, classification, total, _ = training_forward(model,inputs,None,None,
        torch.tensor([[1.,0.]]),torch.ones(1,2),criterion,0)
    total.backward()
    assert observed == [criterion]
    assert criterion.reduction == "mean" and torch.isfinite(total)


@pytest.mark.skipif(os.environ.get("FOVEATAG_FLASH_GPU_TEST") != "1", reason="explicit bounded GPU probe")
@pytest.mark.parametrize("method", ["full", "lora"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_routed_selected_only_gpu_forward_backward(method, dtype):
    torch.manual_seed(42)
    model = FoveaTrainerModel(2,encoder_config=FoveaEncoderConfig(widths=(32,64,192,192),
        depths=(1,1,2,1),attention_heads=(4,4),scale_dropout=0),feature_dim=32,
        evidence_slots=2,pool_heads=4,temperature=.1,training_method=method,lora_rank=2,lora_alpha=2,
        routing={"candidate_count":4,"selection_budget":2,"random_count":1,"oracle_interval":1,
            "oracle_chunk_size":2,"router_weight":1.,"routing_dim":32,"routing_slots":2}).cuda()
    model.vision_encoder.set_sdpa(True,allow_flash=True,external_flash=True)
    proc = processor()
    image = Image.new("RGB",(96,48),(20,60,100))
    inputs = proc(images=[image],return_tensors="pt")["fovea_views"]
    assert inputs.local_images.device.type == "cpu"
    with torch.autocast("cuda",dtype=dtype):
        _, _, total, _ = training_forward(model,inputs,None,None,torch.tensor([[1.,0.]],device="cuda"),
            torch.ones(1,2,device="cuda"),AsymmetricLossOptimized(),0)
    total.backward()
    assert torch.isfinite(total)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.router.parameters())
    model.eval()
    with torch.no_grad(),torch.autocast("cuda",dtype=dtype):
        torch.testing.assert_close(model(inputs),model.predict_native(image,proc),rtol=.02,atol=.02)


def test_actual_forward_capture_accepts_online_bytes():
    batch=DiagnosticBatch((None,)*5, ({"source_bytes":b"captured","source_index":12},))
    snapshot=capture_forward(batch,torch.tensor([[0.,1.]]),torch.tensor([[1.,0.]]),torch.tensor([[1.,0.]]),
        step=8,epoch=0,threshold=.35,vocabulary=vocabulary())
    assert snapshot["source"]["source_bytes"] == b"captured"
    assert snapshot["known"].tolist() == [True,False]
    assert snapshot["focus_kind"] == "none"
    assert snapshot["step"] == 8


def test_save_random_scratch_base_is_optional_and_preserves_initialization(tmp_path):
    from core.tagger.foveatag.shared_factory import save_initial_base
    from core.tagger.foveatag.pretraining_checkpoint import load_pretrained_encoder
    from core.tagger.foveatag.encoder import FoveaEncoder
    model = routed_model()
    config = {"save_base_model": False}
    save_initial_base(model,config,tmp_path)
    assert not (tmp_path / "base_model.safetensors").exists()
    config["save_base_model"] = True
    save_initial_base(model,config,tmp_path)
    expected = {key: value.clone() for key,value in model.vision_encoder.state_dict().items()}
    restored = FoveaEncoder(model.vision_encoder.config)
    load_pretrained_encoder(tmp_path / config["base_model_path"],restored)
    for key,value in restored.state_dict().items():
        torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    save_initial_base(model,config,tmp_path)
    load_pretrained_encoder(tmp_path / config["base_model_path"],restored)
    for key,value in restored.state_dict().items():
        torch.testing.assert_close(value,expected[key],rtol=0,atol=0)


def test_joint_metrics_checkpoint_is_portable_and_resizes_rolling_buffer(tmp_path):
    keys=("save_tag_metrics","train_f1_eval_every_n_steps","train_f1_threshold_search_every_n_steps",
        "train_f1_initial_threshold","train_f1_buffer_batches")
    config={key:TAGGER_TRAINING_DEFAULTS[key] for key in keys}
    config.update(train_f1_eval_every_n_steps=1,train_f1_threshold_search_every_n_steps=1)
    runtime=JointMetricsRuntime(vocabulary(),config)
    event=runtime.update(torch.tensor([[1.,5.]]),torch.tensor([[1.,0.]]),torch.tensor([[1.,0.]]),1)
    assert event["train_precision"] == 1
    record=runtime.save(tmp_path,1)
    target=tmp_path / "another_root"
    target.mkdir()
    restored=JointMetricsRuntime(vocabulary(),{**config,"train_f1_buffer_batches":2})
    restored.restore(target,record,1)
    assert restored.threshold == runtime.threshold and restored.start_step == 1
    np.testing.assert_array_equal(restored.accumulator.total_hist_cur,runtime.accumulator.total_hist_cur)
    assert len(restored.buffer) == 1
    damaged={**record,"files":{**record["files"],"rolling_f1":{**record["files"]["rolling_f1"],"data":b"bad"}}}
    with pytest.raises(ValueError,match="hash"):
        restored.restore(target,damaged,1)


def test_direct_joint_artifact_uses_common_loaded_inference_and_thresholds(tmp_path):
    import io
    from core.tagger.foveatag.artifact import FoveaArtifactConfig, InferenceMode, save_inference_artifact
    from core.tagger.siglip2_inference_manager import SigLIP2InferenceManager
    model=routed_model().eval()
    cfg=FoveaArtifactConfig(num_tags=2,feature_dim=16,evidence_slots_per_view=2,pool_heads=2,
        encoder=model.core.encoder.config,views=processor().config,routing_dim=16,routing_slots=2,
        inference_modes={"balanced":InferenceMode(32,2)})
    path=save_inference_artifact(tmp_path / "review.safetensors",model.core,cfg,router=model.router,
        vocabulary=vocabulary().to_dict(),provenance={"stage":"direct_joint_owner_review","global_step":10})
    runtime=JointMetricsRuntime(vocabulary(),{key:TAGGER_TRAINING_DEFAULTS[key] for key in (
        "save_tag_metrics","train_f1_eval_every_n_steps","train_f1_threshold_search_every_n_steps",
        "train_f1_initial_threshold","train_f1_buffer_batches")})
    runtime.accumulator.update(torch.full((10,2),.6),torch.ones(10,2))
    runtime.accumulator.save(str(tmp_path / "review_tag_metrics.npz"))
    manager=SigLIP2InferenceManager()
    manager.device="cpu"
    manager.load_model(str(path),"")
    assert manager.status["architecture"] == "foveatag" and manager.status["has_tag_metrics"]
    image=io.BytesIO()
    Image.new("RGB",(96,48),(20,60,100)).save(image,format="PNG")
    result=manager.predict(image.getvalue(),threshold=0,use_per_tag_threshold=True,min_best_thr=0)
    assert result["has_calibration"]
    assert manager.build_training_assist(want_per_tag=True,want_calibration=True,want_ood=False)["architecture"] == "foveatag"
    manager.unload()
