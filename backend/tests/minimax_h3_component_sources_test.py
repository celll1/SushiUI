"""H3 companion selections survive native bundles without directory discovery."""

import json
from unittest.mock import patch

import pytest

from core.models.minimax_h3.component_sources import (
    KEY, capture_sources, companion_layout, read_sources, validate_sources,
)
from minimax_h3_video_artifact_test import fixture, suppress_import_cuda_context


def sources(tmp_path):
    root = tmp_path/"installation"
    official = root/"official"
    official.mkdir(parents=True)
    (official/"model_index.json").write_text(json.dumps({"_class_name":"MiniMaxH3ModularPipeline"}))
    for name in ("vae", "audio_vae", "text_encoder"):
        directory = official/name
        directory.mkdir()
        (directory/"config.json").write_text("{}")
    for name in ("tokenizer", "processor", "scheduler", "audio_scheduler"):
        (official/name).mkdir()
    layout = dict(root=str(root), official=str(official), variant="ref2va")
    for key in ("vae", "audio_vae", "text_encoder", "image_vae"):
        path = root/(key+".safetensors")
        path.write_bytes(b"synthetic companion reference")
        layout[key] = str(path)
    projection = tmp_path/"external_projection.safetensors"
    projection.write_bytes(b"synthetic projection reference")
    return capture_sources(layout,{"path":str(projection)})


def test_exact_external_selection_and_no_auto_discovery(tmp_path):
    value = sources(tmp_path)
    # The encoder/projection can live outside the original model tree.
    external = tmp_path/"external_encoder.gguf"
    external.write_bytes(b"synthetic external encoder")
    value["text_encoder"] = str(external)
    md = {KEY:json.dumps(value)}
    with patch("core.models.minimax_h3.loader.detect_minimax_h3_layout",side_effect=AssertionError("discovery")):
        layout = companion_layout(md)
    assert layout["text_encoder"] == str(external.resolve())
    assert layout["te_projection"] == value["te_projection"]
    assert layout["vae"] == value["source_vae"]
    assert layout["variant"] == "ref2va"
    assert read_sources({}) is None
    with pytest.raises(ValueError,match="requires companion"):
        companion_layout({})


def test_invalid_or_missing_companions_refuse(tmp_path):
    value = sources(tmp_path)
    for bad in (dict(value,version=True),dict(value,variant="hybrid"),dict(value,unknown="x"),
                dict(value,source_vae="relative.safetensors"),dict(value,text_encoder=None),
                dict(value,source_vae=value["audio_vae"])):
        with pytest.raises(ValueError):
            validate_sources(bad)
    with pytest.raises(FileNotFoundError):
        validate_sources(dict(value,text_encoder=str(tmp_path/"missing.gguf")))
    optional = dict(value,image_vae=None,te_projection=None)
    assert validate_sources(optional) == optional
    (tmp_path/"installation"/"official"/"model_index.json").write_text('{"_class_name":"OtherPipeline"}')
    with pytest.raises(ValueError,match="MiniMaxH3ModularPipeline"):
        validate_sources(value)


@pytest.mark.parametrize("sharded",[False,True])
def test_sources_follow_actual_native_bundle_restore_and_resave(tmp_path,sharded):
    from core.models.minimax_h3.video_artifact import save_checkpoint,load_checkpoint,preflight_checkpoint
    trainer = fixture(tmp_path,4)
    value = sources(tmp_path)
    trainer.transformer.transformer._h3_component_sources = value
    path = save_checkpoint(trainer,tmp_path/"full.safetensors",step=0,epoch=0,
                           max_shard_bytes=24000 if sharded else 10000000)
    md = preflight_checkpoint(path)[0]
    assert read_sources(md) == value
    restored,vae,identity = load_checkpoint(path)
    assert restored.transformer._h3_component_sources == value
    trainer.transformer,trainer.vae,trainer.vae_identity = restored,vae,identity
    again = save_checkpoint(trainer,tmp_path/"again.safetensors",step=1,epoch=0)
    assert read_sources(preflight_checkpoint(again)[0]) == value
    # Missing references fail before allocating a transformer or mapping a codec.
    from core.models.common.vae_source import _WeightFile
    from pathlib import Path
    Path(value["text_encoder"]).unlink()
    with patch("accelerate.init_empty_weights",side_effect=AssertionError("allocation")), \
            patch.object(_WeightFile,"tensors",side_effect=AssertionError("tensor read")):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(path)
