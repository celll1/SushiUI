"""Temporal residual transport for MiniMax Music 3's native flow DiT."""
from copy import deepcopy
import inspect
import math

import torch
import torch.nn.functional as F

from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import checkpoint_metadata, digest, preview_plan, validate_model_identity

GRID_POLICY_KEY = 'sushi_residual_grid_policy'
GRID_POLICY = 'zero_token_tail_v1'


def native_config(config):
    from .vendor.transformer_minimax_music3 import MiniMaxMusic3Transformer1DModel
    parameters = inspect.signature(MiniMaxMusic3Transformer1DModel.__init__).parameters
    resolved = {name:deepcopy(config.get(name, parameter.default)) for name,parameter in parameters.items()
                if name != 'self' and parameter.default is not inspect.Parameter.empty}
    if any(type(value) is not int or value < 1 for value in resolved.values()):
        raise ValueError('MiniMax Music 3 native flow dimensions must be positive integers')
    if (resolved['rotary_dim'] % 2 or resolved['rotary_dim'] > resolved['attention_head_dim']
            or resolved['fourier_embedding_dim'] % 2):
        raise ValueError('MiniMax Music 3 native Fourier/rotary dimensions are invalid')
    if config.get(GRID_POLICY_KEY, GRID_POLICY) != GRID_POLICY:
        raise ValueError('Unsupported MiniMax Music 3 residual grid policy')
    resolved[GRID_POLICY_KEY] = GRID_POLICY
    return resolved


def validate_topology_identity(spec, config):
    resolved = native_config(config)
    graph = validate_model_identity(spec, 'minimax_music3',
        [{'name':'main', 'block_count':resolved['num_layers']}], resolved)
    if any(region['down_factor'][1] != 1 for region in graph['regions']):
        raise ValueError('MiniMax Music 3 residual regions shrink the temporal axis only: [factor,1]')
    return graph


def default_topology(config):
    from api.param_defaults import DIT_REWIRING_PRESETS
    from core.rewiring.spec import OPERATORS, build_spec
    resolved = native_config(config)
    preset = DIT_REWIRING_PRESETS['minimax_music3']
    if resolved['num_layers'] != preset['block_count']:
        return None
    return build_spec('minimax_music3', [{'name':'main','block_count':resolved['num_layers']}],
        [{**region, **OPERATORS} for region in preset['regions']], resolved)


def temporal_grid(spec, tokens):
    if type(tokens) is not int or tokens < 1:
        raise ValueError('MiniMax Music 3 residual flow requires a nonempty latent timeline')
    factors = {}
    alignment = 1
    for region in spec['regions']:
        factor = factors.get(region['parent'], 1) * region['down_factor'][0]
        factors[region['id']] = factor
        alignment = math.lcm(alignment, factor)
    padded = ((tokens+alignment-1)//alignment)*alignment
    preview_plan(spec, padded, 1)
    return padded


def restore_topology(model, spec):
    if spec is None:
        model.__dict__.pop('_residual_unet_spec', None)
        model.__dict__.pop('_residual_unet_last_grid', None)
        return
    graph = validate_topology_identity(spec, model.config)
    if len(model.transformer_blocks) != native_config(model.config)['num_layers']:
        raise ValueError('MiniMax Music 3 actual flow depth disagrees with native config')
    model._residual_unet_spec = deepcopy(graph)


def model_topology_metadata(model):
    graph = getattr(model, '_residual_unet_spec', None)
    if graph is None:
        return {}
    if len(model.transformer_blocks) != native_config(model.config)['num_layers']:
        raise ValueError('MiniMax Music 3 actual flow depth disagrees with native config')
    return checkpoint_metadata(validate_topology_identity(graph, model.config))


def preflight_forward(model, hidden, timestep, condition):
    model.__dict__.pop('_residual_unet_last_grid', None)
    graph = validate_topology_identity(model._residual_unet_spec, model.config)
    model_topology_metadata(model)
    if (hidden.ndim != 3 or hidden.shape[0] < 1 or hidden.shape[1] != model.config.in_channels
            or condition.shape != (hidden.shape[0], hidden.shape[2], model.config.condition_dim)
            or condition.device != hidden.device or timestep.shape != (hidden.shape[0],)
            or not torch.isfinite(timestep).all()):
        raise ValueError('MiniMax Music 3 residual flow requires native batch/channel/conditioning shapes')
    for name in ('_fbcache', '_skip_blocks', '_tread_router', '_block_offloader', '_attention_plan'):
        if getattr(model, name, None) is not None:
            raise ValueError(f'MiniMax Music 3 residual flow is incompatible with {name}')
    return graph, temporal_grid(graph, hidden.shape[2])


def run_native_blocks(model, hidden, state):
    graph, padded = state
    native = hidden.shape[1]-1
    hidden = F.pad(hidden, (0,0,0,padded-native))
    def rotary(states):
        return model.rotary_emb(states.shape[1], states.device)
    def coarse_context(context, states, height, width, prefix):
        if width != 1 or prefix != 1:
            raise ValueError('MiniMax Music 3 residual flow requires one preserved time token')
        return rotary(states)
    def execute(index, states, context):
        block = model.transformer_blocks[index]
        if torch.is_grad_enabled() and model.gradient_checkpointing:
            return model._gradient_checkpointing_func(block, states, context)
        return block(states, context)
    restored = run_rewired_stream(hidden, graph, 'main', padded, 1, 1,
        rotary(hidden), execute, coarse_context)
    return restored[:, :native+1]


def record_execution(model, state, tokens):
    graph, padded = state
    model._residual_unet_last_grid = dict(spec_sha256=digest(graph),
        native_tokens=tokens, padded_tokens=padded)


def generation_metadata(model):
    graph = getattr(model, '_residual_unet_spec', None)
    if graph is None:
        return None
    model_topology_metadata(model)
    observed = getattr(model, '_residual_unet_last_grid', None)
    if observed is None or observed['spec_sha256'] != digest(graph):
        raise ValueError('MiniMax Music 3 generation did not observe the current residual graph')
    return dict(spec=deepcopy(graph), spec_sha256=digest(graph), grid_policy=GRID_POLICY,
        native_audio_tokens=observed['native_tokens'], padded_audio_tokens=observed['padded_tokens'],
        prefix_tokens=1, rotary_policy='coarse_integer', kv_cache=False)


def export_metadata(model):
    from core.rewiring.spec import canonical_json
    metadata = model_topology_metadata(model)
    metadata.update(model_type='minimax_music3',
                    transformer_config=canonical_json(native_config(model.config)))
    return metadata


def preflight_directory(path, config):
    from core.rewiring.artifacts import read_directory_topology
    graph = read_directory_topology(path)
    return None if graph is None else validate_topology_identity(graph, config)
