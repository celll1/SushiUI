"""Krea loader and shared architecture detection accept the same directory forms."""
import json

import pytest

from core.model_loader import ModelLoader


@pytest.mark.parametrize("nested", [False,True])
@pytest.mark.parametrize("config", [
    {"_class_name":"Krea2Transformer2DModel"},
    {"architectures":["Krea2Transformer2DModel"]},
    {"num_layerwise_text_blocks":2,"axes_dims_rope":[32,48,48]},
])
def test_transformer_root_and_pipeline_directory_are_detected(tmp_path,nested,config):
    folder=tmp_path/"transformer" if nested else tmp_path
    folder.mkdir(exist_ok=True)
    (folder/"config.json").write_text(json.dumps(config),encoding="utf-8")
    assert ModelLoader.detect_model_type(str(tmp_path))=="krea2"


@pytest.mark.parametrize("config", [
    {"_class_name":"UnrelatedTransformer","axes_dims_rope":[32,48,48]},
    {"num_layerwise_text_blocks":2},
])
def test_partial_config_does_not_claim_krea(tmp_path,config):
    (tmp_path/"config.json").write_text(json.dumps(config),encoding="utf-8")
    assert ModelLoader.detect_model_type(str(tmp_path))!="krea2"


def test_corrupt_root_does_not_hide_nested_krea(tmp_path):
    (tmp_path/"config.json").write_text("{",encoding="utf-8")
    nested=tmp_path/"transformer";nested.mkdir()
    (nested/"config.json").write_text(json.dumps({"_class_name":"Krea2Transformer2DModel"}),encoding="utf-8")
    assert ModelLoader.detect_model_type(str(tmp_path))=="krea2"
