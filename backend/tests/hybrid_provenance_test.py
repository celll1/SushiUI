"""Initialization is lineage, not an attestation of current fine-tuned weights."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from safetensors.torch import save_file

from core.rewiring import provenance
from core.rewiring.artifacts import read_artifact_topology
from core.rewiring.spec import checkpoint_metadata, digest, read_checkpoint_metadata
from hybrid_initialization_test import TinyModel
from hybrid_rewiring_test import declaration


@pytest.fixture(autouse=True)
def cpu_only_imports(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **k: None)
    monkeypatch.setattr(torch._C, "_cuda_init", lambda *a, **k: None)
    yield
    assert not torch.cuda.is_initialized()


def record():
    return dict(schema_version=1, method="native_teacher_velocity", source_sha256="1"*64,
                calibration_sha256="2"*64, initialized_weights_sha256="3"*64,
                origin_spec=declaration(), sample_count=32, seed=718,
                phases=[dict(kind="data_noising", updates=300, learning_rate=0.001),
                        dict(kind="native_trajectory", updates=1000, learning_rate=0.0003)])


def test_roundtrip_retains_origin_after_codec_rebase_and_training():
    model = TinyModel()
    original = record()
    model._residual_unet_spec["base_config_sha256"] = "4"*64
    metadata = provenance.checkpoint_metadata(original, model._residual_unet_spec)
    provenance.restore(model, metadata)
    with torch.no_grad():
        for parameter in model.rewiring_modules.parameters():
            parameter.add_(0.01)
    assert provenance.model_metadata(model) == metadata
    assert model._hybrid_initialization["origin_spec"] == original["origin_spec"]
    from core.rewiring.initialization import frozen_native_teacher
    with pytest.raises(ValueError, match="must not be fitted again"):
        with frozen_native_teacher(model):
            pass


def test_anima_checkpoint_writer_emits_record():
    from core.models.anima.residual_unet import model_topology_metadata
    from core.models.anima.config import ANIMA_DIT_CONFIG
    from core.rewiring.hybrid_spec import default_hybrid_spec
    # The metadata writer needs only config attributes, not a native model allocation.
    from types import SimpleNamespace
    model = SimpleNamespace(**ANIMA_DIT_CONFIG)
    from core.models.anima.residual_unet import default_topology
    model._residual_unet_spec = default_hybrid_spec(default_topology(model),
                                                   {"main": model.model_channels})
    value = record()
    value["origin_spec"] = deepcopy(model._residual_unet_spec)
    model._hybrid_initialization = value
    metadata = model_topology_metadata(model)
    assert read_checkpoint_metadata(metadata) == model._residual_unet_spec
    assert provenance.read_checkpoint_metadata(metadata, model._residual_unet_spec) == value


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(schema_version=True),
    lambda value: value.update(seed=True),
    lambda value: value.update(sample_count=0),
    lambda value: value.update(source_sha256="wrong"),
    lambda value: value.update(extra=1),
    lambda value: value["phases"][0].update(updates=0),
    lambda value: value["phases"][0].update(learning_rate=float("nan")),
    lambda value: value["phases"][0].update(learning_rate=True),
    lambda value: value["phases"][0].update(kind="unknown"),
])
def test_invalid_record_refused(mutate):
    value = record()
    mutate(value)
    with pytest.raises(ValueError):
        provenance.validate_record(value, declaration())


def test_no_provenance_remains_unknown_and_restoration_clears_stale_record():
    model = TinyModel()
    model._hybrid_initialization = record()
    provenance.restore(model, {})
    assert not hasattr(model, "_hybrid_initialization")
    assert provenance.model_metadata(model) == {}
    assert read_checkpoint_metadata({}) is None
    assert read_checkpoint_metadata(checkpoint_metadata(declaration())) == declaration()


def test_incomplete_or_corrupt_metadata_refused_before_loading():
    spec = declaration()
    value = provenance.checkpoint_metadata(record(), spec)
    topology = checkpoint_metadata(spec)
    for metadata in ({provenance.META_RECORD: value[provenance.META_RECORD]},
                     dict(value, **{provenance.META_HASH: "0"*64}), value):
        with pytest.raises(ValueError):
            read_checkpoint_metadata(metadata)
    metadata = dict(topology, **value)
    metadata[provenance.META_RECORD] = metadata[provenance.META_RECORD].replace(
        '"seed":718', '"seed":718,"seed":718')
    with pytest.raises(ValueError, match="Duplicate"):
        read_checkpoint_metadata(metadata)


def test_different_module_layout_refused():
    current = declaration()
    current["stream_widths"] = {"main": 16}
    from core.rewiring.hybrid_spec import module_layout
    current["module_layout_sha256"] = digest(module_layout(current))
    with pytest.raises(ValueError, match="different native blocks or modules"):
        provenance.validate_record(record(), current)


def test_index_and_shard_initialization_must_agree(tmp_path):
    import json
    spec = declaration()
    metadata = dict(checkpoint_metadata(spec), **provenance.checkpoint_metadata(record(), spec))
    save_file({"weight": torch.zeros(1)}, str(tmp_path/"shard.safetensors"), metadata=metadata)
    index = tmp_path/"model.safetensors.index.json"
    payload = dict(metadata=metadata, weight_map={"weight": "shard.safetensors"})
    index.write_text(json.dumps(payload), encoding="utf-8")
    assert read_artifact_topology(index) == spec
    payload["metadata"] = checkpoint_metadata(spec)
    index.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="initialization declarations disagree"):
        read_artifact_topology(index)


def test_directory_shards_must_have_one_initialization_lineage(tmp_path):
    from core.rewiring.artifacts import read_directory_declarations
    spec = declaration()
    value = record()
    metadata = dict(checkpoint_metadata(spec), **provenance.checkpoint_metadata(value, spec))
    save_file({"first": torch.zeros(1)}, str(tmp_path/"one.safetensors"), metadata=metadata)
    save_file({"second": torch.zeros(1)}, str(tmp_path/"two.safetensors"), metadata=metadata)
    assert read_directory_declarations(tmp_path) == (spec, value)
    value["seed"] += 1
    metadata.update(provenance.checkpoint_metadata(value, spec))
    save_file({"second": torch.zeros(1)}, str(tmp_path/"two.safetensors"), metadata=metadata)
    with pytest.raises(ValueError, match="initialization declarations disagree"):
        read_directory_declarations(tmp_path)
