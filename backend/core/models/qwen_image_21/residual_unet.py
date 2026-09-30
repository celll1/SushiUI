"""Qwen image-tail residual rewiring; native block execution remains authoritative."""
import torch

from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import validate_spec


def run_rewired_qwen(model, hidden, modulation, rotary, mask, target_mask,
                     segments, key_valid, image_shape, spec):
    from .vendor.transformer import build_qwenimage21_block_causal_mask

    spec = validate_spec(spec)
    if not model.config.causal_condition:
        raise ValueError("Qwen rewiring currently requires native causal modulation")
    frame, height, width = image_shape
    if frame != 1:
        raise ValueError("Qwen rewiring requires one 2D target grid")
    prefix = hidden.shape[1] - height * width
    if (prefix < 0 or target_mask is None or target_mask.ndim != 1
            or target_mask.shape[0] != hidden.shape[1]
            or not target_mask[prefix:].all() or target_mask[:prefix].any()):
        raise ValueError("Qwen rewiring requires a contiguous target-token tail")
    if spec["architecture"] != "qwen_image_21" or spec["streams"] != [
            {"name": "main", "block_count": len(model.transformer_blocks)}]:
        raise ValueError("Qwen rewiring architecture or block layout mismatch")

    def coarse_context(context, value, h, w, text_count):
        old_rotary, old_mask, old_target, old_valid = context
        image_mask = torch.zeros(value.shape[1], dtype=torch.bool, device=value.device)
        image_mask[text_count:] = True
        new_rotary = model.pos_embed([(1, h, w)], image_mask, value.device)
        new_valid = None if old_valid is None else torch.cat((
            old_valid[:, :text_count], old_valid.new_ones(value.shape[0], h*w)), dim=1)
        new_mask = None
        if old_mask is not None:
            image_ids = torch.full_like(image_mask, -1, dtype=torch.long)
            image_ids[text_count:] = 0
            new_mask = build_qwenimage21_block_causal_mask(
                image_ids, new_valid, value.shape[0], value.device)
        return new_rotary, new_mask, image_mask, new_valid

    def execute(index, value, context):
        block = model.transformer_blocks[index]
        current_rotary, current_mask, current_target, current_valid = context
        checkpoint_blocks = int(getattr(model, "_training_gradient_checkpointing_blocks",
                                        len(model.transformer_blocks)))
        if torch.is_grad_enabled() and model.gradient_checkpointing and index < checkpoint_blocks:
            return model._gradient_checkpointing_func(
                block, value, modulation, current_rotary, current_mask,
                current_target, None, None, None, segments, current_valid)
        return block(hidden_states=value, modulation=modulation, rotary_emb=current_rotary,
                     attention_mask=current_mask, target_token_mask=current_target,
                     segments=segments, key_valid=current_valid)

    return run_rewired_stream(hidden, spec, "main", height, width, prefix,
                              (rotary, mask, target_mask, key_valid), execute, coarse_context)
