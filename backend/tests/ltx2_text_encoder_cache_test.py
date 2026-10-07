import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from safetensors.torch import load_file, save_file

from core.models.ltx2.text_encoder_cache import prepare_bf16_backing


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "gemma3"}))
    state = {"weight": torch.randn(7, 9), "scalar": torch.tensor(1.25),
             "integer": torch.arange(3), "empty": torch.empty(0)}
    save_file(state, str(source / "model.safetensors"))
    return source, state


def test_exact_cast_and_reuse_without_overwriting_source(tmp_path, monkeypatch):
    source, state = _source(tmp_path)
    raw_bytes = (source / "model.safetensors").read_bytes()
    target = prepare_bf16_backing(source, tmp_path / "cache")
    converted = load_file(str(target / "model.safetensors"))
    for key, tensor in state.items():
        expected = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
        assert torch.equal(converted[key], expected)
    assert (source / "model.safetensors").read_bytes() == raw_bytes
    def fail(*args, **kwargs):
        raise AssertionError("A reusable cache must not be converted again")
    monkeypatch.setattr("core.models.ltx2.text_encoder_cache.write_stream", fail)
    assert prepare_bf16_backing(source, tmp_path / "cache") == target


def test_changed_cache_content_is_rejected(tmp_path):
    source, _ = _source(tmp_path)
    target = prepare_bf16_backing(source, tmp_path / "cache")
    path = target / "model.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="backing content changed"):
        prepare_bf16_backing(source, tmp_path / "cache")


def test_changed_source_gets_distinct_backing(tmp_path):
    source, state = _source(tmp_path)
    first = prepare_bf16_backing(source, tmp_path / "cache")
    state["weight"].add_(1)
    save_file(state, str(source / "model.safetensors"))
    second = prepare_bf16_backing(source, tmp_path / "cache")
    assert first != second
    assert torch.equal(load_file(str(second / "model.safetensors"))["weight"], state["weight"].bfloat16())


def test_failed_conversion_cannot_publish_partial_cache(tmp_path, monkeypatch):
    source, _ = _source(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic writer failure")
    monkeypatch.setattr("core.models.ltx2.text_encoder_cache.write_stream", fail)
    with pytest.raises(RuntimeError, match="writer failure"):
        prepare_bf16_backing(source, tmp_path / "cache")
    assert not [path for path in (tmp_path / "cache").iterdir() if path.is_dir()]


def test_concurrent_preparation_reuses_one_complete_cache(tmp_path):
    source, _ = _source(tmp_path)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: prepare_bf16_backing(source, tmp_path / "cache"), range(2)))
    assert results[0] == results[1]
    assert len([path for path in (tmp_path / "cache").iterdir() if path.is_dir()]) == 1


def test_shard_index_missing_tensor_is_rejected(tmp_path):
    source, state = _source(tmp_path)
    mapping = dict.fromkeys(state, "model.safetensors")
    mapping["absent"] = "model.safetensors"
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError, match="index has missing tensors"):
        prepare_bf16_backing(source, tmp_path / "cache")


def test_already_bf16_source_needs_no_cache(tmp_path):
    source, state = _source(tmp_path)
    state = {key: tensor.bfloat16() if tensor.is_floating_point() else tensor for key, tensor in state.items()}
    save_file(state, str(source / "model.safetensors"))
    assert prepare_bf16_backing(source, tmp_path / "cache") == source
    assert not (tmp_path / "cache").exists()


def test_sharded_index_and_integer_buffers_are_preserved(tmp_path):
    source, state = _source(tmp_path)
    first = {"weight": state["weight"], "empty": state["empty"]}
    second = {"integer": state["integer"], "scalar": state["scalar"]}
    save_file(first, str(source / "first.safetensors"))
    save_file(second, str(source / "second.safetensors"))
    mapping = {**dict.fromkeys(first, "first.safetensors"), **dict.fromkeys(second, "second.safetensors")}
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    target = prepare_bf16_backing(source, tmp_path / "cache")
    index = json.loads((target / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == mapping
    assert index["metadata"]["total_size"] == state["weight"].numel() * 2 + 2 + 3 * 8
    assert torch.equal(load_file(str(target / "second.safetensors"))["integer"], state["integer"])


def test_insufficient_space_does_not_create_partial_cache(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source, _ = _source(tmp_path)
    monkeypatch.setattr("core.models.ltx2.text_encoder_cache.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="free cache space"):
        prepare_bf16_backing(source, tmp_path / "cache")
    assert not [path for path in (tmp_path / "cache").iterdir() if path.is_dir()]


def test_converted_real_gemma_weights_and_logits_match(tmp_path):
    from ltx2_mapped_text_encoder_test import _package
    from core.models.ltx2.mapped_text_encoder import load_matching_mapped_encoder
    source = tmp_path / "source"
    original = _package(source).float()
    original.save_pretrained(source)
    target = prepare_bf16_backing(source, tmp_path / "cache")
    loaded = load_matching_mapped_encoder(target, torch.bfloat16)
    original.bfloat16()
    for key, value in original.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key
    tokens = torch.tensor([[1, 5, 7]])
    with torch.no_grad():
        assert torch.equal(original(input_ids=tokens).logits, loaded(input_ids=tokens).logits)


@pytest.mark.parametrize("filename", ["../model.safetensors", "/outside.safetensors"])
def test_parent_path_in_index_is_rejected(tmp_path, filename):
    source, _ = _source(tmp_path)
    (source / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"weight": filename}}))
    with pytest.raises(ValueError, match="escapes"):
        prepare_bf16_backing(source, tmp_path / "cache")


def test_hub_snapshot_file_symlink_is_readable(tmp_path):
    source, _ = _source(tmp_path)
    blob = tmp_path / "blob.safetensors"
    (source / "model.safetensors").rename(blob)
    try:
        (source / "model.safetensors").symlink_to(blob)
    except OSError:
        pytest.skip("File symlinks are unavailable on this host")
    target = prepare_bf16_backing(source, tmp_path / "cache")
    assert (target / "backing.json").is_file()
