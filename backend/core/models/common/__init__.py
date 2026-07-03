"""Shared single-file format infrastructure for sushiUI model checkpoints.

Owns the sushiUI single-file v2 conventions (key prefixes, tied-weight dedup,
threshold-switched diffusers-convention sharding, and the shard/index reader)
so per-architecture modules only supply config/variant detection and key remaps.
"""

from .single_file_format import (
    DEFAULT_MAX_SHARD_BYTES,
    TRANSFORMER_PREFIX,
    TEXT_ENCODER_PREFIX,
    VAE_PREFIX,
    dedup_tensors,
    save_single_file_state,
    read_state_dict,
    load_component_state_dict,
    split_prefixed_state_dict,
    strip_prefix,
    build_component_metadata,
    parse_component_metadata,
    is_index_path,
    reattach_embedded_weights,
)

__all__ = [
    "DEFAULT_MAX_SHARD_BYTES",
    "TRANSFORMER_PREFIX",
    "TEXT_ENCODER_PREFIX",
    "VAE_PREFIX",
    "dedup_tensors",
    "save_single_file_state",
    "read_state_dict",
    "load_component_state_dict",
    "split_prefixed_state_dict",
    "strip_prefix",
    "build_component_metadata",
    "parse_component_metadata",
    "is_index_path",
    "reattach_embedded_weights",
]
