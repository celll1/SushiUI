"""Native MiniMax-H3 packed AV block execution with spatial residual transport."""
from copy import deepcopy
import inspect

import torch

from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import preview_plan, validate_model_identity
from .packed_residual_layout import prepare_layout, coarse_context
from core.rewiring.training import NO_RESUME_SPEC
from .vendor.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM

GRID_POLICY_KEY = 'sushi_residual_video_grid_policy'
GRID_POLICY = 'packed_target_spatial_frames_v1'
VIDEO_CODEC_KEY = 'sushi_video_codec_contract'


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
    codec = config.get(VIDEO_CODEC_KEY)
    if codec is not None:
        from .video_alignment import VIDEO_POSITION_POLICY_KEY,VIDEO_POSITION_POLICY
        if config.get(VIDEO_POSITION_POLICY_KEY) != VIDEO_POSITION_POLICY:
            raise ValueError('H3 video position policy is missing or unsupported')
        resolved[VIDEO_POSITION_POLICY_KEY] = VIDEO_POSITION_POLICY
        from dataclasses import asdict
        from core.models.components.latent_adapter import ImageCodecContract
        from core.models.components.video_latent_adapter import VideoCodecContract
        if not isinstance(codec, dict) or set(codec) != {'image', 'temporal_scale', 'temporal_policy'}:
            raise ValueError('H3 video codec identity is missing')
        try:
            contract = VideoCodecContract(ImageCodecContract(**codec['image']), codec['temporal_scale'], codec['temporal_policy'])
        except (TypeError, AttributeError) as exc:
            raise ValueError('H3 video codec identity is malformed') from exc
        if contract.image.latent_channels > resolved['in_channels']:
            raise ValueError('H3 target codec width exceeds the internal backbone')
        resolved[VIDEO_CODEC_KEY] = asdict(contract)
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


def forward_kwargs(transformer, layout):
    """Use target geometry recorded by the native layout builder, never modality guesses."""
    inner = getattr(transformer, 'transformer', transformer)
    codec_kwargs = {}
    if getattr(inner, '_video_latent_alignment', None) is not None:
        from .video_alignment import forward_kwargs as codec_forward_kwargs
        codec_kwargs = codec_forward_kwargs(transformer,layout)
    if getattr(inner, '_residual_unet_spec', None) is None:
        return codec_kwargs
    from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper
    if not isinstance(transformer, MiniMaxH3BlockLoopWrapper):
        raise ValueError('MiniMax-H3 residual generation/training requires its native execution wrapper')
    grid = layout.get('target_video_grid')
    if not isinstance(grid, dict) or set(grid) != {'frames', 'height', 'width', 'start_row'}:
        raise ValueError('MiniMax-H3 residual execution requires native layout target geometry')
    if any(type(value) is not int for value in grid.values()) or any(grid[key] < 1 for key in ('frames','height','width')):
        raise ValueError('MiniMax-H3 native layout target geometry is invalid')
    count = grid['frames'] * grid['height'] * grid['width']
    start = grid['start_row']
    positions = layout['position_ids']
    if start < 0 or start + count != positions.shape[0] or positions.shape[0] != layout['sequence_length']:
        raise ValueError('MiniMax-H3 native layout target geometry disagrees with packed sequence')
    target = torch.arange(start, start+count, device=positions.device)
    return {**codec_kwargs,'residual_video_grid':dict(frames=grid['frames'], height=grid['height'],
                                     width=grid['width'], target_indices=target)}


def model_topology_metadata(model):
    from core.rewiring.spec import checkpoint_metadata
    inner = getattr(model, 'transformer', model)
    graph = getattr(inner, '_residual_unet_spec', None)
    if graph is None:
        return {}
    graph = validate_topology_identity(graph, inner.config)
    if len(inner.transformer_blocks) != native_config(inner.config)['num_layers']:
        raise ValueError('MiniMax-H3 export block count disagrees with native config')
    return checkpoint_metadata(graph)


def configure_training_topology(model, config, *, resuming=False, pending_adapter_resume=False,
                                resume_spec=NO_RESUME_SPEC, install=True, trainer=None):
    from types import SimpleNamespace
    from api.param_defaults import TRAINING_DEFAULTS
    from core.rewiring.training import resolve_training_topology
    from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper
    inner = getattr(model, 'transformer', model)
    model_topology_metadata(model)
    existing = getattr(inner, '_residual_unet_spec', None)
    declared = config.get('dit_rewiring_spec', TRAINING_DEFAULTS['dit_rewiring_spec'])
    needs_identity = config.get('dit_rewiring_mode') == 'configure' or existing is not None or declared is not None or (
        resume_spec is not NO_RESUME_SPEC and resume_spec is not None)
    native = native_config(inner.config) if needs_identity else {}
    from core.rewiring.spec import digest
    source_config = getattr(inner,"_video_source_config",None)
    view = SimpleNamespace(config=native, _residual_unet_spec=existing,
        _rewiring_source_config_sha256=digest(source_config) if source_config is not None else None)
    resolved = resolve_training_topology(view, dict(config), 'minimax_h3', validate_topology_identity,
        lambda owner, graph: None, default_topology, resuming=resuming, pending_adapter_resume=pending_adapter_resume,
        resume_spec=resume_spec, install=False)
    if resolved is not None:
        if len(inner.transformer_blocks) != native['num_layers']:
            raise ValueError('MiniMax-H3 actual block count disagrees with native config')
        for key in ('_tread_router', '_block_dropout', '_blockskip_config', '_attention_plan'):
            if getattr(model, key, None) is not None:
                raise ValueError(f'MiniMax-H3 residual training is incompatible with {key}')
        if install and trainer is None and not isinstance(model, MiniMaxH3BlockLoopWrapper):
            raise ValueError('MiniMax-H3 residual training requires its wrapper or trainer owner')
    if install:
        restore_topology(model, resolved)
        if trainer is not None and resolved is not None:
            wrapper = wrap_transformer(model, resolved)
            trainer.transformer = wrapper
            trainer.transformer_original = inner
            components = getattr(trainer, 'minimax_h3_components', None)
            if components is not None:
                components['transformer'] = wrapper
        config['dit_rewiring_mode'] = config.get('dit_rewiring_mode', TRAINING_DEFAULTS['dit_rewiring_mode'])
        config['dit_rewiring_spec'] = resolved
    return resolved


def resolve_generation_topology(model, files):
    from types import SimpleNamespace
    from core.rewiring.spec import read_checkpoint_metadata
    from core.rewiring.adapters import resolve_adapter_topology
    inner = getattr(model, 'transformer', model)
    model_topology_metadata(model)
    existing = getattr(inner, '_residual_unet_spec', None)
    needs_identity = existing is not None or any(read_checkpoint_metadata(file.metadata) is not None for file in files)
    native = native_config(inner.config) if needs_identity else {}
    view = SimpleNamespace(config=native, _residual_unet_spec=existing)
    return resolve_adapter_topology(view, files, validate_topology_identity)


def default_topology(config):
    from api.param_defaults import DIT_REWIRING_PRESETS
    from core.rewiring.spec import OPERATORS, build_spec
    config = native_config(config)
    preset = DIT_REWIRING_PRESETS['minimax_h3']
    if config['num_layers'] != preset['block_count']:
        return None
    return build_spec('minimax_h3', [{'name':'main','block_count':config['num_layers']}],
                      [{**region, **OPERATORS} for region in preset['regions']], config)


def video_geometry(config, spatial_scale=None):
    from .loader import MINIMAX_H3_VAE_SPATIAL_COMPRESSION
    config = native_config(config)
    scale = MINIMAX_H3_VAE_SPATIAL_COMPRESSION if spatial_scale is None else spatial_scale
    if type(scale) is not int or scale < 1:
        raise ValueError('MiniMax-H3 native spatial scale must be a positive integer')
    return dict(temporal_scale_factor=None, height_scale_factor=scale*config['patch_size'][1],
                width_scale_factor=scale*config['patch_size'][2], grid_policy=GRID_POLICY)


def inspect_artifact(path):
    import re
    from pathlib import Path
    from core.rewiring.artifacts import read_artifact_topology
    from .loader import _guard_component_file, _synthesize_transformer_config
    source = Path(path)
    header, metadata = _guard_component_file(str(source), label='transformer', allow_h3_int8_convrot=True)
    if not {'video_patch_proj.weight','audio_patch_proj.weight','condition_proj.weight'}.issubset(header):
        raise ValueError('MiniMax-H3 inspection requires native model headers, not adapter weights')
    official = next((parent/'official' for parent in source.parents
                     if (parent/'official'/'transformer'/'config.json').is_file()),None)
    config = _synthesize_transformer_config(header, None if official is None else str(official))
    blocks = {int(match.group(1)) for key in header if (match := re.match(r'blocks\.(\d+)\.',key))}
    if blocks != set(range(config['num_layers'])):
        raise ValueError('MiniMax-H3 native checkpoint block indices are not contiguous')
    graph = read_artifact_topology(source)
    if graph is not None:
        graph = validate_topology_identity(graph,config)
    return config,graph


def begin_generation_observation(model, params):
    params.pop('dit_rewiring_actual', None)
    inner = getattr(model, 'transformer', model)
    inner.__dict__.pop('_residual_unet_last_grid', None)


def record_execution(model, state, video_tokens, audio_tokens, text_tokens):
    from core.rewiring.spec import digest
    graph, layout = state
    inner = getattr(model, 'transformer', model)
    inner._residual_unet_last_grid = dict(spec_sha256=digest(graph),
        video_grid=[layout.frames, layout.height, layout.width], prefix_tokens=layout.prefix,
        condition_video_tokens=video_tokens-layout.frames*layout.height*layout.width,
        audio_tokens=audio_tokens, text_tokens=text_tokens)


def generation_metadata(model):
    from core.rewiring.spec import digest
    inner = getattr(model, 'transformer', model)
    graph = resolve_generation_topology(model, [])
    if graph is None:
        return None
    observed = getattr(inner, '_residual_unet_last_grid', None)
    if observed is None or observed['spec_sha256'] != digest(graph):
        raise ValueError('MiniMax-H3 generation did not observe the current residual graph')
    frames, height, width = observed['video_grid']
    preview_plan(graph, height, width)
    return dict(spec=deepcopy(graph), spec_sha256=digest(graph), grid_policy=GRID_POLICY,
        video_geometry=video_geometry(inner.config), native_video_grid=[frames,height,width],
        native_video_tokens=frames*height*width, native_audio_tokens=observed['audio_tokens'],
        native_text_tokens=observed['text_tokens'], native_prefix_tokens=observed['prefix_tokens'],
        native_condition_video_tokens=observed['condition_video_tokens'],
        audio_grid_policy='native_unscaled', kv_cache=False)
