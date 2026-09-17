"""SenseNova prefix construction and immediate Chimera bridge reduction."""

from __future__ import annotations

import torch

from .conditioning_bridge import ChimeraBridgeOutput, ConditioningBridge
from .understanding import UnderstandingPrefix, capture_understanding_prefix


def capture_chimera_prompt_prefix(
    transformer,
    tokenizer,
    prompt: str,
    selected_layers: tuple[int, ...],
) -> UnderstandingPrefix:
    """Capture the frozen SenseNova prefix without applying the trainable bridge."""
    query = transformer._build_t2i_query(prompt, append_text="<img>")
    encoded = tokenizer(query, return_tensors="pt")
    device = next(transformer.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    return capture_understanding_prefix(
        transformer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        selected_layers=selected_layers,
    )


def move_understanding_prefix(
    prefix: UnderstandingPrefix,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> UnderstandingPrefix:
    device = torch.device(device)
    return UnderstandingPrefix(
        hidden_states=prefix.hidden_states.to(device, non_blocking=non_blocking),
        layer_kv={
            layer: (
                key.to(device, non_blocking=non_blocking),
                value.to(device, non_blocking=non_blocking),
            )
            for layer, (key, value) in prefix.layer_kv.items()
        },
        attention_mask=prefix.attention_mask.to(device, non_blocking=non_blocking),
        positions=prefix.positions.to(device, non_blocking=non_blocking),
    )


def encode_chimera_conditioning(
    transformer,
    tokenizer,
    bridge: ConditioningBridge,
    prompt: str,
    *,
    prefix: UnderstandingPrefix | None = None,
) -> ChimeraBridgeOutput:
    """Return canonical SDXL conditioning without retaining native prefix KV."""
    if prefix is None:
        prefix = capture_chimera_prompt_prefix(
            transformer, tokenizer, prompt, tuple(bridge.config.selected_layers)
        )
    bridge_device = next(bridge.parameters()).device
    if prefix.hidden_states.device != bridge_device:
        prefix = move_understanding_prefix(prefix, bridge_device, non_blocking=True)
    output = bridge(
        prefix.hidden_states,
        prefix.layer_kv,
        prefix.attention_mask,
        prefix.positions,
    )
    del prefix
    return output
