"""CPU native adapter target and canonical factor-group contracts."""
import pytest
import torch
from core.models.ming_image_01.model_definition import create_transformer
from core.models.ming_image_01.lora import (iter_lora_slots, flatten_to_key,
    declared_branch_count, normalise_lora_state_dict, build_lora_branch)
from ming_image_01_native_test import tiny_config


@pytest.mark.parametrize("padding", ["zero_masked", "learned"])
def test_native_slots_and_reversible_keys(padding):
    model = create_transformer(tiny_config(padding), device="cpu", dtype=torch.float32)
    slots = list(iter_lora_slots(model))
    assert len(slots) == 17
    paths = [path for _, _, path in slots]
    assert len(set(paths)) == len(paths)
    assert len(set(map(flatten_to_key, paths))) == len(paths)
    assert len(list(iter_lora_slots(model, projection=False))) == 16
    assert [path for _, _, path in iter_lora_slots(model, attention=False)] == ["cap_embedder.1"]
    assert list(iter_lora_slots(model, attention=False, projection=False)) == []
    raw = {}
    for parent, attr, path in slots:
        base = parent[attr] if isinstance(attr, int) else getattr(parent, attr)
        assert model.get_submodule(path) is base
        stem = flatten_to_key(path)
        raw[stem + ".lora_down.weight"] = torch.randn(2, base.in_features)
        raw[stem + ".lora_up.weight"] = torch.randn(base.out_features, 2)
        raw[stem + ".alpha"] = torch.tensor(2.)
    assert declared_branch_count(raw) == 17
    groups = normalise_lora_state_dict(raw)
    assert set(groups) == set(paths)
    original_ids = {path: id(model.get_submodule(path)) for path in paths}
    for parent, attr, path in slots:
        base = model.get_submodule(path)
        branch = build_lora_branch(base, groups[path], path)
        data = torch.randn(2, base.in_features)
        expected = base(data) + torch.nn.functional.linear(
            torch.nn.functional.linear(data, raw[flatten_to_key(path) + ".lora_down.weight"]),
            raw[flatten_to_key(path) + ".lora_up.weight"])
        torch.testing.assert_close(branch(data), expected)
        branch(data).square().sum().backward()
        if isinstance(attr, int): parent[attr] = branch
        else: setattr(parent, attr, branch)
    assert [path for _, _, path in iter_lora_slots(model)] == paths
    assert all(id(model.get_submodule(path)) != original_ids[path] for path in paths)
    assert not torch.cuda.is_initialized()
