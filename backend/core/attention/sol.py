"""Strict adapter for the optional official Sol-Attn forward kernels."""

from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache
from typing import Callable, Optional

import torch

from .contracts import AttentionMode
from .dispatch import dispatch_attention
from .observed import note_backend


@lru_cache(maxsize=1)
def sol_attention_available() -> bool:
    return importlib.util.find_spec("sol_attn") is not None


@lru_cache(maxsize=1)
def _load_sol_attention() -> Callable:
    if not sol_attention_available():
        raise RuntimeError(
            "h3_sol_attn requires the optional official sol-attn package; "
            "install backend/requirements-attention-experimental.txt"
        )
    return importlib.import_module("sol_attn").sol_attn


def _validate_sol_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Sol-Attn requires equal query/key/value shapes")
    if query.device.type != "cuda":
        raise RuntimeError("h3_sol_attn requires a CUDA device")
    if query.dtype != torch.bfloat16:
        raise RuntimeError(f"h3_sol_attn requires bfloat16 Q/K/V; got {query.dtype}")
    if query.shape[-1] != 128:
        raise RuntimeError(f"h3_sol_attn requires head dimension 128; got {query.shape[-1]}")
    capability = tuple(torch.cuda.get_device_capability(query.device))
    if capability[0] < 8:
        raise RuntimeError(
            "h3_sol_attn requires CUDA compute capability 8.0 or newer; "
            f"got SM{capability[0]}{capability[1]}"
        )


def _dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: Optional[float],
    backend: str,
) -> torch.Tensor:
    return dispatch_attention(
        query,
        key,
        value,
        scale=scale,
        dropout_p=0.0,
        is_causal=False,
        backend=backend,
        mode=AttentionMode.INFERENCE,
        layout="BSHD",
    )


def dispatch_sol_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan,
    *,
    scale: Optional[float],
    dense_backend: str,
    layer_index: Optional[int],
) -> torch.Tensor:
    """Run the official kernel and restore exact H3 prefix-query rows."""
    if plan.use_dense(layer_index):
        return _dense(query, key, value, scale=scale, backend=dense_backend)

    _validate_sol_qkv(query, key, value)
    kernel = _load_sol_attention()
    output = kernel(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        scale=scale,
        tau=plan.tau,
        thresh_type=plan.threshold_type,
        kv_splits=plan.kv_splits,
        sink_start=plan.prefix_start,
        sink_tokens=plan.prefix_tokens,
    )
    if output.shape != query.shape:
        raise RuntimeError(
            f"sol-attn returned shape {tuple(output.shape)}, expected {tuple(query.shape)}"
        )

    if plan.prefix_tokens:
        start = plan.prefix_start
        stop = start + plan.prefix_tokens
        prefix = _dense(
            query[:, start:stop], key, value, scale=scale, backend=dense_backend
        )
        output[:, start:stop].copy_(prefix)

    note_backend("sol_attn")
    return output


__all__ = ["dispatch_sol_attention", "sol_attention_available"]
