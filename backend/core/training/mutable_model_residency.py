"""Move resident weights without detaching mutable block-swap CPU masters."""

import torch


def move_with_mutable_conductors(model, device, conductors):
    if model is None:
        return False
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Mutable training residency requires CPU or CUDA")
    tensors = tuple(model.parameters())+tuple(model.buffers())
    identities = {id(tensor) for tensor in tensors}
    selected, managed = [], set()
    for conductor in conductors:
        owned = conductor.managed_tensor_ids()
        if not identities.intersection(owned):
            continue
        if not owned.issubset(identities):
            raise ValueError("A mutable conductor spans multiple model residency owners")
        selected.append(conductor)
        managed.update(owned)
    if not selected:
        return False
    if device.type == "cpu":
        for conductor in selected:
            conductor.park_device_slots()
    with torch.no_grad():
        for tensor in tensors:
            if id(tensor) not in managed:
                tensor.data = tensor.data.to(device)
                if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None:
                    tensor.grad = tensor.grad.to(device)
    if device.type == "cuda":
        for conductor in selected:
            conductor.resume_device_slots()
    return True
