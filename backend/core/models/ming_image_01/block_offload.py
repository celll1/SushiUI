"""Native block swap and residency that preserves mutable CPU master views."""
import torch


def setup_training_block_swap(trainer):
    if trainer.blocks_to_swap <= 0 or getattr(trainer, "layer_offload_conductor", None) is not None:
        return
    if getattr(trainer, "block_swap_h2d_only", False):
        raise ValueError("Ming mutable training block swap requires CPU writeback")
    transformer = trainer.transformer
    if not trainer.gradient_checkpointing:
        raise ValueError("Ming block swap requires gradient checkpointing")
    limit = getattr(transformer, "_training_gradient_checkpointing_blocks", len(transformer.layers))
    if limit < len(transformer.layers):
        raise ValueError("Ming block swap requires checkpointing every main DiT layer")
    transformer.enable_gradient_checkpointing()
    from core.memory_management import LayerOffloadConductor
    conductor = LayerOffloadConductor(layers=transformer.layers,
        blocks_to_swap=trainer.blocks_to_swap, device=trainer.device,
        use_pinned_memory=trainer.use_pinned_memory, enable_prefetch=True,
        enable_activation_offload=False, ring_size=trainer.block_swap_ring_size)
    conductor.register_hooks()
    trainer.layer_offload_conductor = conductor
    transformer._layer_offload_conductor = conductor


def _managed_ids(module, *, include_resident=False):
    conductor = getattr(module, "_layer_offload_conductor", None)
    if conductor is None:
        return set()
    indices = range(len(conductor.layers)) if include_resident else conductor.swappable
    return {id(value) for index in indices
            for value in (*conductor.layers[index].parameters(), *conductor.layers[index].buffers())}


def execution_device(module):
    # The conductor initially places resident blocks on CUDA before the outer IO moves.
    managed = _managed_ids(module, include_resident=True)
    locations = {value.device for value in (*module.parameters(), *module.buffers()) if id(value) not in managed}
    if len(locations) != 1:
        raise ValueError("Ming component requires one unmanaged execution device")
    return next(iter(locations))


def move_component(module, device):
    conductor = getattr(module, "_layer_offload_conductor", None)
    if conductor is None:
        return module.to(device=device)
    conductor.flush()
    managed = _managed_ids(module)
    target = torch.device(device)
    # A whole-module .to() would detach swapped tensors from their CPU masters.
    with torch.no_grad():
        for value in (*module.parameters(), *module.buffers()):
            if id(value) in managed or value.device == target:
                continue
            value.data = value.data.to(device=target)
            if isinstance(value, torch.nn.Parameter) and value.grad is not None:
                value.grad = value.grad.to(device=target)
    return module
