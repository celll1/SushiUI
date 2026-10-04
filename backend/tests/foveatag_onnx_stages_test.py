"""Native ONNX reproduces staged crop selection and common response probabilities."""
import io
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from PIL import Image

from core.tagger.siglip2_inference_manager import SigLIP2InferenceManager
from tagger_shared_diagnostics_routing_test import routed_model, processor, vocabulary


@pytest.mark.parametrize("method", ["full", "lora"])
@pytest.mark.parametrize("direct", [False, True])
def test_native_onnx_export_load_and_staged_prediction(tmp_path,method,direct):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    instance=routed_model(method).eval()
    proc=processor()
    path=instance.save_checkpoint(tmp_path,"native",{"fovea_view_config":asdict(proc.config),
        "fovea_seed":42,"max_source_pixels":10000,"mixed_precision":"fp32"})
    (tmp_path / "native_vocabulary.json").write_text(json.dumps(vocabulary().to_dict()))
    if direct:
        from core.tagger.siglip2_tagger_model import LoRALinear
        for module in instance.core.modules():
            for name, child in list(module.named_children()):
                if isinstance(child, LoRALinear):
                    child.merge_into_base()
                    setattr(module, name, child.base)
        from core.tagger.foveatag.artifact import FoveaArtifactConfig,InferenceMode,save_inference_artifact
        config=FoveaArtifactConfig(num_tags=2,feature_dim=16,evidence_slots_per_view=2,pool_heads=2,
            encoder=instance.core.encoder.config,views=proc.config,routing_dim=16,routing_slots=2,
            inference_modes={"balanced":InferenceMode(32,2)})
        path=str(save_inference_artifact(tmp_path/"direct.safetensors",instance.core,config,router=instance.router,
            vocabulary=vocabulary().to_dict(),provenance={"stage":"direct_joint_owner_review","global_step":10}))
    manager=SigLIP2InferenceManager()
    manager.device="cpu"
    manager.load_model(path,"")
    image=io.BytesIO()
    Image.new("RGB",(96,48),(20,60,100)).save(image,format="PNG")
    expected=manager.predict(image.getvalue(),threshold=0)
    exported,vocab=manager.export_onnx(str(tmp_path / "export"),use_model_stem=True)
    manager.unload()
    manager.load_model(exported,vocab)
    actual=manager.predict(image.getvalue(),threshold=0)
    assert manager.status["architecture"] == "foveatag"
    expected={entry["tag"]:entry["prob"] for entry in expected["tags"]}
    actual={entry["tag"]:entry["prob"] for entry in actual["tags"]}
    assert actual.keys() == expected.keys()
    for tag in expected:
        assert actual[tag] == pytest.approx(expected[tag],rel=3e-4,abs=3e-5)
    manager.unload()


def test_native_manager_merge_lora_preserves_router(tmp_path):
    instance=routed_model("lora").eval()
    proc=processor()
    path=instance.save_checkpoint(tmp_path,"native",{"fovea_view_config":asdict(proc.config),
        "fovea_seed":42,"max_source_pixels":10000,"mixed_precision":"fp32"})
    (tmp_path / "native_vocabulary.json").write_text(json.dumps(vocabulary().to_dict()))
    manager=SigLIP2InferenceManager()
    manager.device="cpu"
    manager.load_model(path,"")
    image=io.BytesIO()
    Image.new("RGB",(96,48),(20,60,100)).save(image,format="PNG")
    expected=manager.predict(image.getvalue(),threshold=0)
    merged=manager.merge_lora_and_save(str(tmp_path/"merged"))
    manager.unload()
    manager.load_model(merged,"")
    assert manager.model_type == "full" and manager.model.router is not None
    assert manager.predict(image.getvalue(),threshold=0) == expected
    manager.unload()


def test_native_encoder_extraction_merges_lora(tmp_path):
    from core.tagger.siglip2_extractor import extract_vision_encoder
    from core.tagger.foveatag.encoder import FoveaEncoder
    from core.tagger.foveatag.pretraining_checkpoint import load_pretrained_encoder
    instance=routed_model("lora").eval()
    for adapter in instance._lora_modules.values():
        with torch.no_grad():
            adapter.lora_B.fill_(.01)
    path=instance.save_checkpoint(tmp_path,"native",{})
    result=extract_vision_encoder(path,str(tmp_path/"encoder.safetensors"))
    restored=FoveaEncoder(instance.core.encoder.config)
    load_pretrained_encoder(result["output_path"],restored)
    image=torch.randn(1,3,32,32)
    geometry=torch.zeros(1,10)
    torch.testing.assert_close(restored(image,geometry).global_summary,
        instance.vision_encoder(image,geometry).global_summary,rtol=1e-4,atol=1e-5)
