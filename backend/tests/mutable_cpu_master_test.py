"""Pageable CPU masters borrow tensors instead of duplicating full checkpoints."""

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from core.memory_management.layer_offload_conductor import LayerOffloadConductor
from core.memory_management.layer_offload_strategy import LayerOffloadStrategy
from core.memory_management.offload_transfer_engine import MutableLruTransferEngine


def initialize(layers, pinned=False):
    conductor = LayerOffloadConductor.__new__(LayerOffloadConductor)
    conductor.layers = layers
    conductor.swappable = (1,)
    conductor.device = torch.device("cpu")
    conductor.use_pinned_memory = pinned
    conductor.strategy = LayerOffloadStrategy(len(layers),1,conductor.device)
    conductor._masters,conductor._layouts,conductor._trainable,conductor.layer_cpu_copies = {},{},{},{}
    conductor.ring_size = 1
    conductor._pinned_empty = lambda numel,dtype:torch.empty(numel,dtype=dtype)
    conductor._initialize_layers()
    return conductor


def test_borrowed_safetensors_masters_preserve_source_and_write_back(tmp_path):
    path = tmp_path/"source.safetensors"
    save_file(dict(weight=torch.arange(12).reshape(3,4).float(),bias=torch.arange(3).float()),str(path))
    before = path.read_bytes()
    layers = torch.nn.ModuleList([torch.nn.Linear(4,3),torch.nn.Linear(4,3)])
    with safe_open(str(path),framework="pt",device="cpu") as source:
        layers[1].load_state_dict({key:source.get_tensor(key) for key in source.keys()},assign=True)
    pointers = {id(p):p.data_ptr() for p in layers[1].parameters()}
    conductor = initialize(layers)
    assert all(pointers[id(p)] == p.data_ptr() for p in layers[1].parameters())
    assert len(conductor._masters[1]) == 2
    engine = MutableLruTransferEngine(keys=[1],masters=conductor._masters,ring_size=1,
        device="cpu",point_bundle=conductor._point_bundle)
    for _ in range(2):
        engine.acquire(1)
        for parameter in layers[1].parameters():
            parameter.data.add_(1)
        engine.release(1,dirty=True)
        engine.synchronize()
    assert all(pointers[id(p)] == p.data_ptr() for p in layers[1].parameters())
    torch.testing.assert_close(layers[1].weight,torch.arange(12).reshape(3,4).float()+2,rtol=0,atol=0)
    assert path.read_bytes() == before
    engine.close()


def test_pinned_request_retains_packed_planes():
    layers = torch.nn.ModuleList([torch.nn.Linear(4,3),torch.nn.Linear(4,3)])
    pointers = [p.data_ptr() for p in layers[1].parameters()]
    conductor = initialize(layers,pinned=True)
    assert list(conductor._masters[1]) == [torch.float32]
    assert [p.data_ptr() for p in layers[1].parameters()] != pointers
