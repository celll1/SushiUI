"""SenseNova prefix construction and immediate Chimera bridge reduction."""

from __future__ import annotations

import torch

from .conditioning_bridge import ChimeraBridgeOutput, ConditioningBridge
from .understanding import capture_understanding_prefix


def encode_chimera_conditioning(
    transformer,
    tokenizer,
    bridge: ConditioningBridge,
    prompt: str,
) -> ChimeraBridgeOutput:
    """Return canonical SDXL conditioning without retaining native prefix KV."""
    query = transformer._build_t2i_query(prompt, append_text="<img>")
    encoded = tokenizer(query, return_tensors="pt")
    device = next(transformer.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    selected_layers = tuple(bridge.config.selected_layers)
    prefix = capture_understanding_prefix(
        transformer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        selected_layers=selected_layers,
    )
    output = bridge(
        prefix.hidden_states,
        prefix.layer_kv,
        prefix.attention_mask,
        prefix.positions,
    )
    del prefix
    return output
