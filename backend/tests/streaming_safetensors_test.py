"""Atomic streaming writer read-back, cancellation and shape validation."""

import pytest
import torch
from safetensors.torch import load_file
from core.models.common.streaming_safetensors import write_stream


def test_scalar_empty_and_chunked_bf16_roundtrip(tmp_path):
    tensors = {"a":torch.empty(0),"b":torch.arange(24).reshape(6,4).bfloat16(),"c":torch.tensor(7,dtype=torch.int64)}
    specs = {"a":dict(dtype="F32",shape=[0]),"b":dict(dtype="BF16",shape=[6,4]),"c":dict(dtype="I64",shape=[])}
    def chunks():
        yield "b",tensors["b"][:2]
        yield "b",tensors["b"][2:]
        yield "c",tensors["c"]
    output = write_stream(tmp_path/"stream.safetensors",specs,{"test":"yes"},chunks())
    loaded = load_file(str(output))
    for key,tensor in tensors.items():
        torch.testing.assert_close(loaded[key],tensor,rtol=0,atol=0)


@pytest.mark.parametrize("failure",["missing","oversize","dtype","cancel"])
def test_failed_write_cleans_its_temporary_only(tmp_path,failure):
    output = tmp_path/"stream.safetensors"
    def chunks():
        if failure == "cancel":
            yield "a",torch.ones(1)
            raise RuntimeError("cancel")
        value = torch.ones(3 if failure == "oversize" else 1)
        yield "a",value.double() if failure == "dtype" else value
    with pytest.raises((RuntimeError,ValueError)):
        write_stream(output,{"a":dict(dtype="F32",shape=[2])},{},chunks())
    assert not output.exists() and not list(tmp_path.glob("*.partial"))
