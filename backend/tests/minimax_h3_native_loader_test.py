"""Public native H3 loading restores real target weights and exact companions."""

from unittest.mock import patch

import pytest
import torch

from minimax_h3_component_sources_test import sources
from minimax_h3_video_artifact_test import fixture, suppress_import_cuda_context


def companions(monkeypatch, calls):
    from core.models.minimax_h3 import loader
    monkeypatch.setattr(loader,"detect_minimax_h3_layout",lambda *a,**k:pytest.fail("rediscovery"))
    monkeypatch.setattr(loader,"_build_transformer",lambda *a,**k:pytest.fail("ordinary weights"))
    monkeypatch.setattr(loader,"_build_video_vae",lambda *a,**k:pytest.fail("source codec"))
    monkeypatch.setattr(loader,"_te_file_declaration",lambda path:{"modalities":"text"})
    monkeypatch.setattr(loader,"resolve_minimax_h3_te_projection",
        lambda **kw:calls.append(("projection",kw)) or {"path":kw["override"]})
    monkeypatch.setattr(loader,"_build_text_encoder_for",
        lambda *args:calls.append(("encoder",args)) or (torch.nn.Linear(1,1),{"synthetic":True}))
    monkeypatch.setattr(loader,"_build_audio_vae",
        lambda *args:calls.append(("audio",args)) or (torch.nn.Linear(1,1),{"latents_mean":[0.]}))
    monkeypatch.setattr(loader,"_load_tokenizer_and_processor",lambda path:("tokenizer","processor"))
    monkeypatch.setattr(loader,"_load_schedulers",lambda path:("scheduler","audio_scheduler"))
    monkeypatch.setattr("core.models.minimax_h3.te_agreement.maybe_measure_substitution",
        lambda value:calls.append(("agreement",value["text_encoder_path"])))


@pytest.mark.parametrize("sharded",[False,True])
@pytest.mark.parametrize("channels",[1,4])
def test_public_detection_and_native_load(tmp_path,monkeypatch,sharded,channels):
    from core.model_loader import ModelLoader
    from core.models.common.vae_source import _WeightFile
    from core.models.minimax_h3.video_artifact import save_checkpoint,native_metadata
    trainer = fixture(tmp_path,channels)
    refs = sources(tmp_path)
    trainer.transformer.transformer._h3_component_sources = refs
    path = save_checkpoint(trainer,tmp_path/"full.safetensors",step=3,epoch=0,
                           max_shard_bytes=24000 if sharded else 10000000)
    assert native_metadata(path)["model_type"] == "minimax_h3"
    assert ModelLoader.detect_model_type(str(path)) == "minimax_h3"
    assert ModelLoader._looks_like_minimax_h3(str(path))
    calls = []
    companions(monkeypatch,calls)
    original = _WeightFile.tensors
    def tensors(self,*args,**kwargs):
        assert any(name == "encoder" for name,_ in calls)
        calls.append(("weights",str(self)))
        return original(self,*args,**kwargs)
    with patch.object(_WeightFile,"tensors",tensors):
        result = ModelLoader.load_minimax_h3_from_path(str(path),torch_dtype=torch.float32,
            text_encoder_file=refs["text_encoder"],clip_projection_file=refs["te_projection"])
    assert [name for name,_ in calls[:2]] == ["projection","encoder"]
    assert calls[-2][0] == "audio" and calls[-1][0] == "agreement"
    assert result["text_encoder_path"] == refs["text_encoder"]
    assert result["te_projection"]["path"] == refs["te_projection"]
    assert result["text_encoder_origin"] == "selected_external"
    assert result["source_vae_path"] == refs["source_vae"]
    assert result["vae_path"] == str(path)
    assert result["vae_origin"] == "embedded_checkpoint"
    assert result["image_vae"] is None and result["te_text_only"]
    assert result["latent_channels"] == channels
    assert result["vae_scale_factor_spatial"] == 2
    assert result["vae_scale_factor_temporal"] == 1
    assert result["transformer"].config.in_channels == channels
    assert result["transformer_config"]["in_channels"] == max(2,channels)
    assert result["transformer"].transformer._h3_component_sources == refs
    for key,value in trainer.vae.inner.state_dict().items():
        torch.testing.assert_close(result["vae"].inner.state_dict()[key],value,rtol=0,atol=0)
    for key,value in trainer.transformer.transformer.state_dict().items():
        torch.testing.assert_close(result["transformer"].transformer.state_dict()[key],value,rtol=0,atol=0)
    if sharded:
        assert native_metadata(tmp_path/"full.safetensors") == native_metadata(path)


def test_geometry_load_skips_encoder_and_diagnostics(tmp_path,monkeypatch):
    from core.models.minimax_h3.loader import load_minimax_h3_from_path
    from core.models.minimax_h3.video_artifact import save_checkpoint
    trainer = fixture(tmp_path,4)
    trainer.transformer.transformer._h3_component_sources = sources(tmp_path)
    path = save_checkpoint(trainer,tmp_path/"full.safetensors",step=0,epoch=0)
    calls = []
    companions(monkeypatch,calls)
    result = load_minimax_h3_from_path(str(path),torch_dtype=torch.float32,load_text_encoder=False)
    assert [name for name,_ in calls] == ["audio"]
    assert result["text_encoder"] is None
    assert result["text_encoder_path"] == trainer.transformer.transformer._h3_component_sources["text_encoder"]


@pytest.mark.parametrize("kwargs",[{"te_override":"other.gguf"},
    {"te_projection_override":"other.safetensors"},{"hybrid":object()},
    {"video_vae_dtype":torch.float32}])
def test_native_options_refuse_before_tensor_reads(tmp_path,kwargs):
    from core.models.minimax_h3.loader import load_minimax_h3_from_path
    from core.models.minimax_h3.video_artifact import save_checkpoint
    from core.models.common.vae_source import _WeightFile
    trainer = fixture(tmp_path,4)
    trainer.transformer.transformer._h3_component_sources = sources(tmp_path)
    path = save_checkpoint(trainer,tmp_path/"full.safetensors",step=0,epoch=0)
    with patch.object(_WeightFile,"tensors",side_effect=AssertionError("tensor read")):
        with pytest.raises(ValueError):
            load_minimax_h3_from_path(str(path),**kwargs)


def test_missing_references_refuse_public_load(tmp_path):
    from core.models.minimax_h3.loader import load_minimax_h3_from_path
    from core.models.minimax_h3.video_artifact import save_checkpoint
    trainer = fixture(tmp_path,4)
    path = save_checkpoint(trainer,tmp_path/"full.safetensors",step=0,epoch=0)
    with pytest.raises(ValueError,match="companion references"):
        load_minimax_h3_from_path(str(path))


@pytest.mark.parametrize("content",["invalid","[]",'{"metadata":[]}','{"metadata":"bad"}'])
def test_marker_probe_is_soft(tmp_path,content):
    from core.models.minimax_h3.video_artifact import native_metadata
    for name in ("invalid.safetensors","invalid.safetensors.index.json"):
        path = tmp_path/name
        path.write_text(content)
        assert native_metadata(path) is None


def test_recognized_corrupt_bundle_stays_on_strict_native_path(tmp_path):
    from safetensors.torch import save_file
    from core.models.minimax_h3.loader import load_minimax_h3_from_path
    from core.models.minimax_h3.video_artifact import FORMAT
    path = tmp_path/"bad.safetensors"
    save_file({"dummy":torch.zeros(1)},str(path),metadata={"format":FORMAT,"model_type":"minimax_h3"})
    with patch("core.models.minimax_h3.loader.detect_minimax_h3_layout",side_effect=AssertionError("fallback")):
        with pytest.raises(ValueError):
            load_minimax_h3_from_path(str(path))
