"""Owned tensor reads for transforms that would otherwise retain mmap pages."""

import json
import struct

import torch

from .streaming_safetensors import DTYPES, tensor_bytes


class UnmappedTensorReader:
    def __init__(self, path):
        self.file = open(path, "rb")
        try:
            size = struct.unpack("<Q", self.file.read(8))[0]
            self.header = json.loads(self.file.read(size))
            self.base = 8 + size
        except BaseException:
            self.file.close()
            raise

    def get_tensor(self, key):
        spec = self.header[key]
        count = tensor_bytes(spec)
        start, end = spec["data_offsets"]
        if end - start != count or start < 0:
            raise ValueError(f"Invalid tensor offsets: {key}")
        if not count:
            return torch.empty(spec["shape"], dtype=DTYPES[spec["dtype"]])
        self.file.seek(self.base + start)
        buffer = bytearray(count)
        if self.file.readinto(buffer) != count:
            raise OSError(f"Incomplete tensor read: {key}")
        return torch.frombuffer(buffer, dtype=DTYPES[spec["dtype"]]).reshape(spec["shape"])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()
