"""Native MiniMax-H3 packed AV block execution with spatial residual transport."""
from copy import deepcopy
import inspect

import torch

from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import preview_plan, validate_model_identity
from .packed_residual_layout import prepare_layout, coarse_context
from .vendor.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM

GRID_POLICY_KEY = 'sushi_residual_video_grid_policy'
GRID_POLICY = 'packed_target_spatial_frames_v1'


def native_config(config):
    from .vendor import MiniMaxH3Transformer3DModel
    parameters = inspect.signature(MiniMaxH3Transformer3DModel.__init__).parameters
    resolved = {name: deepcopy(config.get(name, parameter.default)) for name, parameter in parameters.items()
                if name != 'self' and parameter.default is not inspect.Parameter.empty}
    for name in ('num_attention_heads', 'attention_head_dim', 'hidden_size', 'num_layers',
                 'num_refiner_layers', 'ffn_dim', 'in_channels', 'audio_in_channels', 'text_dim',
                 'freq_dim', 'time_embed_hidden_dim', 'time_embed_dim', 'rope_freq_dim'):
        if type(resolved[name]) is not int or resolved[name] < 1:
            raise ValueError(f'MiniMax-H3 native {name} must be a positive integer')
    patch = resolved['patch_size']
    if not isinstance(patch, (list, tuple)) or len(patch) != 3 or any(type(n) is not int or n < 1 for n in patch):
        raise ValueError('MiniMax-H3 native patch size must contain three positive integers')
    if resolved['attention_head_dim'] < 6 * resolved['rope_freq_dim']:
        raise ValueError('MiniMax-H3 native head channels cannot hold its rotary axes')
    curve = resolved['adaln_curve_grid']
    if curve is not None and (type(curve) is not int or curve < 2):
        raise ValueError('MiniMax-H3 AdaLN curve requires at least two samples')
    if config.get(GRID_POLICY_KEY, GRID_POLICY) != GRID_POLICY:
        raise ValueError('Unsupported MiniMax-H3 residual video grid policy')
    resolved[GRID_POLICY_KEY] = GRID_POLICY
    return resolved


def validate_topology_identity(spec, config):
    config = native_config(config)
    return validate_model_identity(spec, 'minimax_h3', [{'name':'main', 'block_count':config['num_layers']}], config)


def restore_topology(model, spec):
    inner = getattr(model, 'transformer', model)
    if spec is None:
        inner.__dict__.pop('_residual_unet_spec', None)
        return
    graph = validate_topology_identity(spec, inner.config)
    if len(inner.transformer_blocks) != native_config(inner.config)['num_layers']:
        raise ValueError('MiniMax-H3 actual block count disagrees with native config')
    inner._residual_unet_spec = deepcopy(graph)


def wrap_transformer(model, spec):
    from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper
    restore_topology(model, spec)
    return model if isinstance(model, MiniMaxH3BlockLoopWrapper) else MiniMaxH3BlockLoopWrapper(model)


def preflight_forward(wrapper, hidden, audio, text, timestep, timestep_indices, tags,
                      positions, video_indices, audio_indices, text_indices, grid, attention_kwargs):
    inner = wrapper.transformer
    graph = validate_topology_identity(inner._residual_unet_spec, inner.config)
    if len(inner.transformer_blocks) != native_config(inner.config)['num_layers']:
        raise ValueError('MiniMax-H3 actual block count disagrees with native config')
    if (attention_kwargs or wrapper._fbcache is not None or wrapper._skip_blocks is not None
            or wrapper._residual_probe is not None or getattr(inner, '_attention_plan', None) is not None):
        raise ValueError('MiniMax-H3 residual execution refuses custom attention, caches, skips and probes')
    if not isinstance(grid, dict) or set(grid) != {'frames', 'height', 'width', 'target_indices'}:
        raise ValueError('MiniMax-H3 residual execution requires an explicit target-video grid')
    layout = prepare_layout(positions, tags, timestep_indices, **grid)
    total = positions.shape[0]
    indices = (video_indices, audio_indices, text_indices)
    if any(value.ndim != 1 or value.dtype != torch.long or value.device != positions.device for value in indices):
        raise ValueError('MiniMax-H3 modality indices must share the native position device')
    rows = torch.cat(indices)
    if (rows.numel() != total or not torch.equal(rows.sort().values, torch.arange(total, device=rows.device))
            or not torch.isin(grid['target_indices'], video_indices).all()):
        raise ValueError('MiniMax-H3 modality indices must partition every native packed row')
    patch = inner.config.patch_size
    channels = (inner.config.in_channels * patch[0]*patch[1]*patch[2],
                inner.config.audio_in_channels, inner.config.text_dim)
    if hidden.ndim != 3 or hidden.shape[0] < 1:
        raise ValueError('MiniMax-H3 residual input requires a nonempty native batch')
    batch = hidden.shape[0]
    if any(value.ndim != 3 or value.shape != (batch, index.numel(), channel)
           or value.device != positions.device
           for value, index, channel in zip((hidden, audio, text), indices, channels)):
        raise ValueError('MiniMax-H3 projected inputs disagree with native packed rows/channels')
    if (timestep.ndim != 1 or timestep.numel() < 1 or not torch.isfinite(timestep).all()
            or (timestep_indices >= timestep.numel()).any()):
        raise ValueError('MiniMax-H3 packed timestep labels exceed the native timestep table')
    plan = preview_plan(graph, layout.height, layout.width)
    for level in plan['levels']:
        if level['depth']:
            h, w = level['height'], level['width']
            shape_only = torch.empty((batch, layout.prefix+layout.frames*h*w, 0), device='meta')
            coarse_context(layout.context, shape_only, layout.frames*h, w, layout.prefix)
    return graph, layout


def run_native_blocks(wrapper, hidden, temb, state):
    graph, layout = state
    inner = wrapper.transformer
    offloader = wrapper._block_offloader
    swap = offloader is not None and getattr(offloader, 'blocks_to_swap', 0) > 0
    checkpoint = torch.is_grad_enabled() and inner.gradient_checkpointing

    def with_rotary(context):
        return {**context, 'rotary':inner.rope(context['position_ids'])}

    def coarsen(context, hidden, height, width, prefix):
        return with_rotary(coarse_context(context, hidden, height, width, prefix))

    def execute(index, hidden, context):
        if swap:
            offloader.wait_for_block(index)
        labels = context['timestep_indices'] * MINIMAX_H3_MODALITY_NUM + context['token_tags']
        block = inner.transformer_blocks[index]
        if checkpoint:
            result = inner._gradient_checkpointing_func(block, hidden, temb, labels, context['rotary'])
        else:
            result = block(hidden, temb, labels, context['rotary'])
        if swap:
            offloader.submit_move_blocks_forward(index)
        return result

    result = run_rewired_stream(layout.pack(hidden), graph, 'main', layout.frames*layout.height,
        layout.width, layout.prefix, with_rotary(layout.context), execute, coarsen, grid_groups=layout.frames)
    return layout.unpack(result)
