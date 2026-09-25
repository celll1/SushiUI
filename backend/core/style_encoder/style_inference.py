"""CPU inference from a distributable style encoder artifact and its metadata."""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import threading
from typing import Iterable

import torch
from PIL import Image, UnidentifiedImageError
from transformers import Siglip2ImageProcessor

from .style_artifacts import ENCODER_SUFFIX, METADATA_SUFFIX, load_style_encoder


def decode_style_image(encoded: str) -> Image.Image:
    payload = encoded.split(",", 1)[1] if encoded.startswith("data:") and "," in encoded else encoded
    try:
        raw = base64.b64decode(payload, validate=True)
        if not raw or len(raw) > 32 * 1024 * 1024:
            raise ValueError("image must be between 1 byte and 32 MiB")
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError("image exceeds 40 million pixels")
            return image.convert("RGB")
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"invalid style image: {exc}") from exc


class StyleInferenceService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._key = None
        self._model = None
        self._processor = None
        self._max_num_patches = None

    def _load(self, path: str) -> None:
        path = os.path.abspath(path)
        if not path.endswith(ENCODER_SUFFIX) or not os.path.isfile(path):
            raise FileNotFoundError(f"style encoder artifact not found: {path}")
        meta_path = path[:-len(ENCODER_SUFFIX)] + METADATA_SUFFIX
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"style encoder metadata not found: {meta_path}")
        key = (path, *(os.stat(p).st_mtime_ns for p in (path, meta_path)),
               *(os.stat(p).st_size for p in (path, meta_path)))
        if key == self._key:
            return
        with open(meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        processor = Siglip2ImageProcessor.from_dict(metadata["preprocessing"])
        max_num_patches = int(metadata["style_max_num_patches"])
        if max_num_patches < 1:
            raise ValueError("artifact metadata has an invalid patch budget")
        model = load_style_encoder(path, "sdpa").to("cpu").eval()
        self._model, self._processor = model, processor
        self._max_num_patches = max_num_patches
        self._key = key

    def _embed(self, image: Image.Image) -> list[float]:
        inputs = self._processor(images=[image], max_num_patches=self._max_num_patches, return_tensors="pt")
        with torch.inference_mode():
            embedding = self._model.embed(inputs["pixel_values"], inputs["pixel_attention_mask"],
                                          inputs["spatial_shapes"])[0]
        return embedding.float().tolist()

    def embed(self, path: str, image: Image.Image) -> list[float]:
        with self._lock:
            self._load(path)
            return self._embed(image)

    def similarity(self, path: str, first: Image.Image, second: Image.Image) -> float:
        with self._lock:
            self._load(path)
            a = torch.tensor(self._embed(first))
            b = torch.tensor(self._embed(second))
            return float((a * b).sum().clamp(-1.0, 1.0))

    def select_reference(self, path: str, query: Image.Image,
                         candidates: Iterable[Image.Image]) -> tuple[int, list[float]]:
        with self._lock:
            self._load(path)
            query_embedding = torch.tensor(self._embed(query))
            scores = [float((query_embedding * torch.tensor(self._embed(image))).sum().clamp(-1, 1))
                      for image in candidates]
        return max(range(len(scores)), key=scores.__getitem__), scores


style_inference_service = StyleInferenceService()
