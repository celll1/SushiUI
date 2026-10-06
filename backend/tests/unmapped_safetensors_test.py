"""Owned safetensors reads retain exact values without a writable file view."""

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from core.models.common.unmapped_safetensors import UnmappedTensorReader


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int8, torch.float8_e4m3fn])
def test_owned_reads_survive_close_without_changing_source(tmp_path, dtype):
    path = tmp_path / "weights.safetensors"
    value = torch.arange(24).reshape(4, 6).to(dtype)
    save_file({"weight": value, "empty": torch.empty(0, dtype=dtype)}, str(path))
    before = path.read_bytes()
    with UnmappedTensorReader(path) as reader, safe_open(str(path), framework="pt") as mapped:
        owned = reader.get_tensor("weight")
        assert owned.data_ptr() != mapped.get_tensor("weight").data_ptr()
        assert torch.equal(owned.view(torch.uint8), value.view(torch.uint8))
        assert reader.get_tensor("empty").numel() == 0
    owned.view(torch.uint8).zero_()
    assert path.read_bytes() == before


def test_truncated_owned_read_refuses_partial_tensor(tmp_path):
    path = tmp_path / "weights.safetensors"
    save_file({"weight": torch.arange(24).reshape(4, 6)}, str(path))
    path.write_bytes(path.read_bytes()[:-1])
    with UnmappedTensorReader(path) as reader, pytest.raises(OSError):
        reader.get_tensor("weight")
