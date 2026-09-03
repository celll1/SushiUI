"""Checkpoint codec registry and format normalization for adapter checkpoints.

Detects, normalizes, and validates adapter checkpoints across:
1. SushiUI canonical format (lora_down.weight, lora_up.weight, hada_*, lokr_*)
2. Kohya / LyCORIS format (lora_unet_*, lora_te_*, ss_network_module)
3. Diffusers / Hugging Face PEFT format (*.lora_A.weight, *.lora_B.weight)

Supports algorithms: 'lora', 'loha', 'lokr', and weight decomposition (DoRA).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

import torch

from .layers import TUCKER_TENSOR_NAMES
from .spec import (ALGORITHM_UNKNOWN, ALGORITHMS, FORMAT_LYCORIS, FORMAT_PEFT,
                   FORMAT_SUSHIUI, FORMAT_UNKNOWN, FORMATS,
                   METADATA_ALGORITHM, METADATA_ALPHA, METADATA_FORMAT,
                   METADATA_RANK, METADATA_WEIGHT_DECOMPOSE,
                   parse_metadata_bool)


@dataclass(frozen=True)
class CodecSpec:
    """Declared or detected adapter format and algorithm properties."""

    algorithm: str  # "lora", "loha", "lokr", "unknown"
    weight_decompose: bool  # True if DoRA (magnitude vector scaling present)
    format: str  # "sushiui_canonical", "lycoris_kohya", "diffusers_peft", "unknown"
    rank: Optional[int] = None
    alpha: Optional[float] = None
    #: Tucker form: hada_t1/hada_t2/lokr_t2. Detected here so
    #: ``AdapterSpec.validate`` can refuse it -- a Linear cannot honour it.
    use_tucker: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)


#: Suffix -> the axis of that factor that IS the rank, mirroring
#: ``groups.TensorGroup.rank``. PEFT's ``lora_A`` is ``[rank, in]`` exactly like
#: ``lora_down`` -- only the LyCORIS ``*_a`` factors are ``[out, rank]``, and
#: reading them alike reported in_features as the rank for ten real spellings.
_RANK_AXIS_BY_SUFFIX: Dict[str, int] = {
    ".lora_down.weight": 0,
    ".lora_A.weight": 0,
    ".lora_A.default.weight": 0,
    ".hada_w1_a": 1,
    ".lokr_w1_a": 1,
    ".lokr_w2_a": 1,
}


def _sniff_rank(tensors: Mapping[str, torch.Tensor]) -> Optional[int]:
    """The rank a checkpoint's factor shapes imply, or ``None``.

    ``None`` is a real answer for the full/full LoKr, which has no rank.
    A tensor too small for its axis is SKIPPED rather than indexed: a
    ``lora_bias=True`` PEFT export carries a 1-D ``.lora_A.bias``.
    """
    for key, tensor in tensors.items():
        axis = None
        for suffix, candidate in _RANK_AXIS_BY_SUFFIX.items():
            if key.endswith(suffix):
                axis = candidate
                break
        else:
            # Any other PEFT adapter name (``.lora_A.<name>.weight``).
            if ".lora_A." in key and key.endswith(".weight"):
                axis = 0
        if axis is not None and len(getattr(tensor, "shape", ())) > axis:
            return int(tensor.shape[axis])
    return None


class CodecRegistry:
    """Registry for detecting, normalizing, and registering adapter checkpoint codecs."""

    @staticmethod
    def detect(
        tensors: Mapping[str, torch.Tensor],
        metadata: Optional[Mapping[str, str]] = None,
    ) -> CodecSpec:
        """Detect the algorithm, format, and decomposition of an adapter checkpoint."""
        meta = dict(metadata or {})
        keys = list(tensors.keys())

        # 1. Weight decomposition check (DoRA)
        has_dora_scale = any(k.endswith(".dora_scale") or k.endswith(".dora_scale.weight") or "dora_scale" in k for k in keys)
        meta_dora = (parse_metadata_bool(meta.get(METADATA_WEIGHT_DECOMPOSE))
                     or parse_metadata_bool(meta.get("dora")))
        weight_decompose = has_dora_scale or meta_dora

        use_tucker = any(k.rpartition(".")[2] in TUCKER_TENSOR_NAMES for k in keys)

        # 2. Algorithm detection: Priority 1 Metadata
        meta_algo = meta.get(METADATA_ALGORITHM, "").lower()
        ss_module = meta.get("ss_network_module", "").lower()

        if meta_algo in ALGORITHMS:
            algorithm = meta_algo
        elif "loha" in ss_module:
            algorithm = "loha"
        elif "lokr" in ss_module:
            algorithm = "lokr"
        elif "locon" in ss_module or "lora" in ss_module:
            algorithm = "lora"
        else:
            # Priority 2: Tensor keys
            if any("hada_w1_a" in k or "hada_w1_b" in k for k in keys):
                algorithm = "loha"
            elif any("lokr_w1" in k or "lokr_w2" in k for k in keys):
                algorithm = "lokr"
            elif any(k.endswith(".lora_down.weight") or k.endswith(".lora_up.weight")
                     or ".lora_A." in k or ".lora_B." in k for k in keys):
                algorithm = "lora"
            else:
                algorithm = ALGORITHM_UNKNOWN

        # 3. Format detection
        meta_format = meta.get(METADATA_FORMAT, "").lower()
        if meta_format in FORMATS:
            format_name = meta_format
        elif any(".lora_A." in k or ".lora_B." in k or k.startswith("base_model.model.") for k in keys):
            format_name = FORMAT_PEFT
        elif ss_module or any(k.startswith("lora_unet_") or k.startswith("lora_te_") for k in keys):
            format_name = FORMAT_LYCORIS
        elif algorithm != ALGORITHM_UNKNOWN:
            format_name = FORMAT_SUSHIUI
        else:
            format_name = FORMAT_UNKNOWN

        # 4. Rank and Alpha extraction
        rank = None
        alpha = None

        if METADATA_RANK in meta:
            try:
                rank = int(meta[METADATA_RANK])
            except ValueError:
                pass
        if METADATA_ALPHA in meta:
            try:
                alpha = float(meta[METADATA_ALPHA])
            except ValueError:
                pass
        elif "alpha" in meta:
            try:
                alpha = float(meta["alpha"])
            except ValueError:
                pass

        if rank is None:
            rank = _sniff_rank(tensors)

        if alpha is None:
            # Check for scalar alpha tensor in checkpoint
            for k, tensor in tensors.items():
                if k.endswith(".alpha") and tensor.numel() == 1:
                    alpha = float(tensor.item())
                    break

        return CodecSpec(
            algorithm=algorithm,
            weight_decompose=weight_decompose,
            format=format_name,
            rank=rank,
            alpha=alpha,
            use_tucker=use_tucker,
            metadata=meta,
        )

    @classmethod
    def normalize_keys(
        cls,
        tensors: Mapping[str, torch.Tensor],
        spec: Optional[CodecSpec] = None,
    ) -> Dict[str, torch.Tensor]:
        """Normalize foreign tensor keys (PEFT, Kohya) to canonical format."""
        if spec is None:
            spec = cls.detect(tensors)

        normalized: Dict[str, torch.Tensor] = {}
        for key, tensor in tensors.items():
            new_key = key

            # Strip Hugging Face PEFT prefix
            if new_key.startswith("base_model.model."):
                new_key = new_key[len("base_model.model."):]

            # Normalize Diffusers / PEFT A/B names to down/up
            if ".lora_A.weight" in new_key:
                new_key = new_key.replace(".lora_A.weight", ".lora_down.weight")
            elif ".lora_B.weight" in new_key:
                new_key = new_key.replace(".lora_B.weight", ".lora_up.weight")
            elif ".lora_A.default.weight" in new_key:
                new_key = new_key.replace(".lora_A.default.weight", ".lora_down.weight")
            elif ".lora_B.default.weight" in new_key:
                new_key = new_key.replace(".lora_B.default.weight", ".lora_up.weight")

            # Normalize Kohya / LyCORIS prefixes if needed
            # (preserves stem structure while standardizing suffix)
            normalized[new_key] = tensor

        return normalized


# Default module-level convenience functions
detect_adapter_codec = CodecRegistry.detect
normalize_adapter_keys = CodecRegistry.normalize_keys
