"""Stable content identity before modules change device residency."""
import hashlib
import json

import torch


def module_fingerprint(module):
    digest = hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        if value.device.type != "cpu" or value.layout != torch.strided or value.is_quantized:
            raise ValueError("Module content hashes require plain CPU tensors")
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        tensor_hash = hashlib.sha256(memoryview(raw)).hexdigest()
        digest.update(json.dumps([name,list(value.shape),str(value.dtype),tensor_hash]).encode())
    return digest.hexdigest()
