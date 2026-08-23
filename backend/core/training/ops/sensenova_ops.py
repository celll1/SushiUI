"""Training-only SenseNova decoder operations.

Registration waits until the full SenseNova train step is implemented.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _assert_immutable_prefix_cache(prefix_cache: Any, expected_layers: int) -> None:
    if prefix_cache is None:
        raise ValueError("SenseNova generation training requires a prefix KV cache")
    layers = getattr(prefix_cache, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("SenseNova training requires non-empty prefix KV cache layers")
    if len(layers) != expected_layers:
        raise ValueError(
            f"SenseNova prefix KV cache has {len(layers)} layer(s), expected {expected_layers}"
        )
    if any(
        getattr(prefix_cache, name, None) is not None
        for name in ("_kv_cache_streamer", "_kv_cache_streamer_branch")
    ):
        raise ValueError("SenseNova training cannot use the inference KV cache streamer")

    for layer in layers:
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                raise ValueError(f"SenseNova prefix KV cache layer is missing non-empty {name}")
            if tensor.requires_grad:
                raise ValueError(f"SenseNova prefix KV cache {name} tensors must be detached")
        if any(
            getattr(layer, name, None) is not None
            for name in ("flash_k_cache", "flash_v_cache")
        ):
            raise ValueError("SenseNova training cannot use prepared inference flash KV buffers")


def forward_gen_decoder_layers(
    model: Any,
    hidden_states: torch.Tensor,
    *,
    indexes: torch.Tensor,
    prefix_cache: Any,
    attention_mask: Optional[torch.Tensor] = None,
    checkpoint_layers: bool = False,
) -> torch.Tensor:
    """Run the all-generation-token Qwen3 decoder against immutable prefix K/V.

    Calling PyTorch's base ``Module.__call__`` bypasses Transformers'
    cache-dropping checkpoint wrapper while preserving module hooks. The cache
    is read through the differentiable ``update_cache=False`` concat path.
    """

    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("SenseNova generation training model has no decoder layers")
    _assert_immutable_prefix_cache(prefix_cache, len(layers))
    image_gen_indicators = torch.ones(
        hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
    )

    for layer in layers:
        def layer_forward(states: torch.Tensor, _layer=layer) -> torch.Tensor:
            # Skip only Transformers' cache-dropping wrapper; keep Module hooks.
            return nn.Module.__call__(
                _layer,
                states,
                image_gen_indicators=image_gen_indicators,
                exist_non_image_gen_tokens=False,
                exist_image_gen_tokens=True,
                indexes=indexes,
                attention_mask=attention_mask,
                past_key_values=prefix_cache,
                use_cache=False,
                update_cache=False,
            )

        if checkpoint_layers:
            hidden_states = checkpoint(layer_forward, hidden_states, use_reentrant=False)
        else:
            hidden_states = layer_forward(hidden_states)

    return model.norm_mot_gen(hidden_states)
