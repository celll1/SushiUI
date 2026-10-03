"""Atomic local manifests and content-addressed feature cache."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from .features import CaptionFeatures


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".caption-", dir=path.parent)
    os.close(fd)
    try:
        writer(temp)
        with open(temp, "r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path, value):
    def writer(temp):
        with open(temp, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
    atomic_write(path, writer)


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def move_features(features, device):
    return CaptionFeatures(**{key: value.to(device) for key, value in vars(features).items()})


class FeatureCache:
    def __init__(self, root, *, identity, max_bytes):
        self.root = Path(root) / canonical_hash(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.used_bytes = sum(p.stat().st_size for p in self.root.glob("*.pt"))

    def path(self, image_hash):
        if len(image_hash) != 64 or any(c not in "0123456789abcdef" for c in image_hash):
            raise ValueError("Expected SHA256 image content identity")
        return self.root / f"{image_hash}.pt"

    def get(self, image_hash):
        path = self.path(image_hash)
        if not path.exists():
            return None
        values = torch.load(path, map_location="cpu", weights_only=True)
        if set(values) != set(CaptionFeatures.__dataclass_fields__):
            raise ValueError("Invalid feature cache record")
        return CaptionFeatures(**values)

    def put(self, image_hash, features):
        path = self.path(image_hash)
        if path.exists():
            return
        values = {key: value.detach().cpu() for key, value in vars(features).items()}
        estimate = sum(v.numel() * v.element_size() for v in values.values()) + 16384
        if self.used_bytes + estimate > self.max_bytes:
            return  # Online extraction remains available when the cache is full.
        def writer(temp):
            with open(temp, "wb") as stream:
                torch.save(values, stream)
        atomic_write(path, writer)
        self.used_bytes += path.stat().st_size
