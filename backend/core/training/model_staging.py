"""Place dense resident weights without materializing swappable blocks on CUDA."""

import torch
from torch import nn


def stage_dense_block_model(model, layers, blocks_to_swap, device):
    from core.memory_management.layer_offload_strategy import LayerOffloadStrategy
    from core.training.adapters.base_adapter import reject_quantized_base
    reject_quantized_base(model, model_label="Dense block staging")
    strategy = LayerOffloadStrategy(len(layers), blocks_to_swap, torch.device(device))
    modules = list(model.modules())
    block_owners = {}
    for index, layer in enumerate(layers):
        for tensor in list(layer.parameters())+list(layer.buffers()):
            if id(tensor) in block_owners and block_owners[id(tensor)] != index:
                raise ValueError("Block staging cannot preserve tensors shared across blocks")
            block_owners[id(tensor)] = index
    selected = {id(module) for i, layer in enumerate(layers) if strategy.is_offloadable(i)
                for module in layer.modules()}
    if not selected.issubset({id(module) for module in modules}):
        raise ValueError("Staged blocks must belong to the training model")
    ownership = {}
    for module in modules:
        tensors = list(module.parameters(recurse=False))+list(module.buffers(recurse=False))
        for tensor in tensors:
            current = id(module) in selected
            if id(tensor) in ownership and ownership[id(tensor)] != current:
                raise ValueError("A tensor cannot be shared between staged and resident blocks")
            ownership[id(tensor)] = current
            if current and tensor.device.type != "cpu":
                raise ValueError("Deferred training blocks must start on CPU")
    # Adapter/optimizer construction follows this initial placement.
    for module in modules:
        if id(module) not in selected:
            nn.Module._apply(module, lambda tensor: tensor.to(device), recurse=False)
    return model
