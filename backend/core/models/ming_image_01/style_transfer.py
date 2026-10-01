"""Main-stream style K/V capture with native target-only geometry."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


def style_attention(attn, query, key, value, mask):
    info = getattr(attn, "_ming_style_info", None)
    if info is None:
        return query, key, value, mask
    ctx, index, count, frames = info["context"], info["index"], info["count"], info["frames"]
    if not ctx.active_for_block(index):
        return query, key, value, mask
    start = count // frames if info["exclude_auxiliary"] else 0
    if not 0 <= start < count <= query.shape[1] or count > key.shape[1]:
        raise ValueError("Ming style target slice disagrees with native attention geometry")
    if ctx.mode == "capture":
        ctx.store[index] = tuple(tensor[:, start:count].detach().clone() for tensor in (query, key, value))
        return query, key, value, mask
    from core.inference.reference_style import inject_kv_multi, inject_kv, make_ref_value
    original_length = key.shape[1]
    if ctx.refs is not None:
        refs = ctx.collect_block_refs(index, value[:, start:count], key.device, key.dtype)
        if refs:
            key, value, query = inject_kv_multi(key, value, query, start, count, refs, ctx.combine_mode)
    elif index in ctx.store:
        ref_q, ref_k, ref_v = ctx.store[index]
        config = ctx.config
        ref_v = make_ref_value(value[:, start:count], ref_v, config.value_mode,
                              config.value_adain_strength, config.ref_value_mix)
        freq = config.get_freq_scale_vector(key.shape[-1], ctx.progress, key.device, key.dtype)
        key, value, query = inject_kv(key, value, ref_k, ref_v, start, count,
            config.ref_k_strength, freq, config.adain_strength, q=query, ref_q=ref_q)
    added = key.shape[1] - original_length
    if added and mask is not None:
        if mask.shape[-1] != original_length:
            raise ValueError("Ming style key mask disagrees with native sequence length")
        mask = F.pad(mask, (0, added), value=True if mask.dtype == torch.bool else 0.)
    return query, key, value, mask


@contextmanager
def style_scope(model, context, latents):
    """Restore attention processors and per-call geometry on every exit."""
    from .attention import MingAttentionProcessor
    if torch.is_grad_enabled():
        raise ValueError("Ming style transfer is inference-only")
    shapes = [tuple(value.shape) for value in latents]
    if not shapes or any(shape != shapes[0] for shape in shapes) or len(shapes[0]) != 4:
        raise ValueError("Ming style requires uniform native target latent grids")
    _, frames, height, width = shapes[0]
    if min(frames, height, width) < 1 or height % 2 or width % 2:
        raise ValueError("Ming style requires nonempty patch-aligned target grids")
    auxiliary = bool(model.config.multi_frame_output)
    if auxiliary and frames < 2:
        raise ValueError("Ming Layer style requires auxiliary canvas plus target layers")
    configurations = [context.config] if context.refs is None else [cfg for _, cfg in context.refs]
    for config in configurations:
        axes = tuple(model.config.axes_dims)
        if config.axes_dims != axes or config.rope_layout != "interleaved":
            config._freq_cache.clear()
        config.axes_dims = axes
        config.rope_layout = "interleaved"
        config.resolve_default_block_range(len(model.layers))
    missing = object()
    saved = []
    try:
        for index, block in enumerate(model.layers):
            attn = block.attention
            processor = attn.processor
            backend = getattr(processor, "_attention_backend", None) or "native"
            if backend not in {"native", "normal"} or getattr(processor, "_parallel_config", None) is not None:
                raise ValueError("Ming style transfer requires native, non-distributed attention")
            saved.append((attn, processor, getattr(attn, "_ming_style_info", missing)))
            attn._ming_style_info = {"context":context, "index":index,
                "count":frames * (height // 2) * (width // 2), "frames":frames,
                "exclude_auxiliary":auxiliary}
            attn.set_processor(MingAttentionProcessor("native",getattr(processor,"_query_chunk_tokens",None)))
        yield
    finally:
        for attn, processor, previous in reversed(saved):
            attn.set_processor(processor)
            if previous is missing:
                del attn._ming_style_info
            else:
                attn._ming_style_info = previous
