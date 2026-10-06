"""Zero-copy CPU master layouts shared by mutable and frozen offload."""

def borrow_contiguous_cpu_tensors(tensors):
    if not tensors or any(t.device.type != "cpu" or not t.is_contiguous() for _, _, t in tensors):
        return None
    planes, layout = {}, []
    for number, (owner, name, tensor) in enumerate(tensors):
        plane = (tensor.dtype, number)
        planes[plane] = tensor.detach().reshape(-1)
        layout.append((owner, name, plane, 0, tensor.numel(), tuple(tensor.shape)))
    return planes, layout
