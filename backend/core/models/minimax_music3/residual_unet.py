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


def preflight_flat_headers(graph, keys, config):
    if graph is None:
        return None
    graph = validate_topology_identity(graph, config)
    _validate_flat_headers(keys, config)
    return graph


def _validate_flat_headers(keys, config):
    import re
    blocks = {int(match.group(1)) for key in keys
              if (match := re.match(r'diffusion_transformer\.transformer\.layers\.(\d+)\.', key))}
    required = {'diffusion_transformer.transformer.project_in.weight',
                'diffusion_transformer.transformer.project_out.weight',
                'diffusion_transformer.timestep_features.weight', 'cond_layer_logits',
                'cond_layer_scale', 'latent_conditioners.0.weight', 'latent_conditioners.0.bias'}
    if blocks != set(range(native_config(config)['num_layers'])) or not required.issubset(keys):
        raise ValueError('MiniMax Music 3 flat artifacts require complete native flow/condition headers')


def begin_flow_observation(pipeline):
    pipeline.__dict__.pop('_dit_rewiring_flow_actual', None)
    model = pipeline.transformer
    if getattr(model, '_residual_unet_spec', None) is None:
        return None
    model.__dict__.pop('_residual_unet_last_grid', None)
    model_topology_metadata(model)
    return dict(spec_sha256=digest(model._residual_unet_spec), chunks=[])


def observe_flow_evaluation(pipeline, observation):
    if observation is None:
        return None
    actual = generation_metadata(pipeline.transformer)
    if actual is None or actual['spec_sha256'] != observation['spec_sha256']:
        raise ValueError('MiniMax Music 3 residual graph changed during flow generation')
    return actual


def finish_flow_observation(pipeline, observation):
    if observation is None:
        return
    actual = observe_flow_evaluation(pipeline, observation)
    if not observation['chunks']:
        raise ValueError('MiniMax Music 3 residual flow did not observe any chunks')
    pipeline._dit_rewiring_flow_actual = dict(
        **{key:actual[key] for key in ('spec','spec_sha256','grid_policy','prefix_tokens','rotary_policy','kv_cache')},
        scope='flow_input_chunks', chunks=deepcopy(observation['chunks']))


def capture_generation_observation(pipeline, params, model):
    params.pop('dit_rewiring_actual', None)
    if getattr(model, '_residual_unet_spec', None) is None:
        return
    model_topology_metadata(model)
    actual = getattr(pipeline, '_dit_rewiring_flow_actual', None)
    if actual is None or actual['spec_sha256'] != digest(model._residual_unet_spec):
        raise ValueError('MiniMax Music 3 generation did not observe its completed flow chunks')
    params['dit_rewiring_actual'] = deepcopy(actual)


def temporal_geometry(condition_config):
    from .vendor.condition_embedder_minimax_music3 import MiniMaxMusic3ConditionEncoder
    from .defaults import CHUNK_FRAMES
    parameters = inspect.signature(MiniMaxMusic3ConditionEncoder.__init__).parameters
    values = {name:condition_config.get(name, parameters[name].default) for name in
              ('input_sampling_rate','input_hop_length','output_sampling_rate','output_hop_length')}
    if any(type(value) is not int or value < 1 for value in values.values()):
        raise ValueError('MiniMax Music 3 native conditioning rates/hops must be positive integers')
    frames = max(1,int(CHUNK_FRAMES * values['output_sampling_rate'] / values['input_sampling_rate']
                       * values['input_hop_length'] / values['output_hop_length']))
    return dict(frame_rate=values['output_sampling_rate']/values['output_hop_length'],
                patch_size=1, preview_frames=frames, grid_policy=GRID_POLICY)


def model_info(components):
    model = components.get('transformer')
    graph = getattr(model, '_residual_unet_spec', None)
    info = dict(dit_rewiring_spec=None)
    if graph is None:
        return info
    model_topology_metadata(model)
    condition_config = components.get('condition_encoder_config')
    if condition_config is None:
        condition_config = getattr(components.get('condition_encoder'), 'config', None)
    if not isinstance(condition_config, dict):
        raise ValueError('MiniMax Music 3 restored topology requires native conditioning geometry')
    config = native_config(model.config)
    from .vendor.condition_embedder_minimax_music3 import MiniMaxMusic3ConditionEncoder
    parameters = inspect.signature(MiniMaxMusic3ConditionEncoder.__init__).parameters
    out_dim = condition_config.get('out_dim', parameters['out_dim'].default)
    if type(out_dim) is not int or out_dim != config['condition_dim']:
        raise ValueError('MiniMax Music 3 flow and condition encoder channels disagree')
    info['dit_rewiring_spec'] = deepcopy(graph)
    info['dit_rewiring_temporal_geometry'] = temporal_geometry(condition_config)
    return info


def find_native_directory(path):
    import json
    from pathlib import Path
    path = Path(path)
    base = path if path.is_dir() else path.parent
    for directory in (base/'official'/'transformer', base/'transformer', base):
        source = directory/'config.json'
        if source.is_file():
            config = json.loads(source.read_text(encoding='utf-8'))
            if isinstance(config, dict) and config.get('_class_name') == 'MiniMaxMusic3Transformer1DModel':
                return directory
    return None


def inspect_artifact(path):
    import re
    from pathlib import Path
    from safetensors import safe_open
    from core.rewiring.artifacts import read_artifact_topology, read_directory_topology
    from core.rewiring.spec import read_checkpoint_metadata
    from .loader import _read_component_config
    source = Path(path)
    directory = find_native_directory(source)
    native = directory is not None and (source.is_dir() or str(source).endswith('.safetensors.index.json'))
    if directory is not None and source.is_file() and source.suffix == '.safetensors':
        with safe_open(str(source), framework='pt', device='cpu') as handle:
            native = {'proj_in.weight','proj_out.weight','time_proj.weight'}.issubset(handle.keys())
    official = directory.parent if directory is not None else next(
        (parent/'official' for parent in source.parents
         if (parent/'official'/'transformer'/'config.json').is_file()), None)
    if official is None:
        raise ValueError('MiniMax Music 3 inspection requires companion flow/condition configs')
    config = _read_component_config(str(official), 'transformer', 'MiniMaxMusic3Transformer1DModel')
    condition = _read_component_config(str(official), 'condition_encoder', 'MiniMaxMusic3ConditionEncoder')
    config = native_config(config)
    from .vendor.condition_embedder_minimax_music3 import MiniMaxMusic3ConditionEncoder
    expected_out = inspect.signature(MiniMaxMusic3ConditionEncoder.__init__).parameters['out_dim'].default
    if type(condition.get('out_dim', expected_out)) is not int or condition.get('out_dim', expected_out) != config['condition_dim']:
        raise ValueError('MiniMax Music 3 flow and condition encoder channels disagree')
    geometry = temporal_geometry(condition)
    shapes = {}
    if native:
        graph = read_directory_topology(directory)
        files = sorted(directory.glob('*.safetensors'))
        if not files:
            raise ValueError('MiniMax Music 3 native inspection requires flow checkpoint headers')
        for file in files:
            with safe_open(str(file), framework='pt', device='cpu') as handle:
                for key in handle.keys():
                    if key in shapes:
                        raise ValueError('MiniMax Music 3 native flow headers contain duplicate tensors')
                    view = handle.get_slice(key)
                    if view.get_dtype() not in {'F16','BF16','F32','F64'}:
                        raise ValueError('MiniMax Music 3 native component inspection refuses quantized flow headers')
                    shapes[key] = view.get_shape()
        blocks = {int(match.group(1)) for key in shapes
                  if (match := re.match(r'transformer_blocks\.(\d+)\.', key))}
        if blocks != set(range(config['num_layers'])):
            raise ValueError('MiniMax Music 3 native flow headers disagree with block count')
        hidden = config['num_attention_heads']*config['attention_head_dim']
        channels = 2*config['in_channels']+config['condition_dim']
        required = {'proj_in.weight':[hidden,channels], 'proj_out.weight':[config['in_channels'],hidden],
                    'time_proj.weight':[config['fourier_embedding_dim']//2,1]}
        for index in range(config['num_layers']):
            required[f'transformer_blocks.{index}.attn.to_q.weight'] = [hidden,hidden]
    elif source.suffix.lower() == '.gguf':
        from core.models.common import gguf_container
        header = gguf_container.parse_gguf_header(str(source))
        if header.metadata.get('general.architecture') != 'minimax_music3':
            raise ValueError('Residual GGUF inspection requires a MiniMax Music 3 flow artifact')
        gguf_container.refuse_unsupported_tensor_types(header, arch='MiniMax Music 3', label='inspection')
        graph = read_checkpoint_metadata(header.metadata)
        shapes = {item.name:list(item.torch_shape) for item in header.tensors}
    else:
        graph = read_artifact_topology(source)
        with safe_open(str(source), framework='pt', device='cpu') as handle:
            shapes = {key:handle.get_slice(key).get_shape() for key in handle.keys()}
    if not native:
        _validate_flat_headers(set(shapes), config)
        hidden = config['num_attention_heads']*config['attention_head_dim']
        channels = 2*config['in_channels']+config['condition_dim']
        required = {'diffusion_transformer.transformer.project_in.weight':[hidden,channels],
                    'diffusion_transformer.transformer.project_out.weight':[config['in_channels'],hidden],
                    'diffusion_transformer.timestep_features.weight':[config['fourier_embedding_dim']//2,1]}
        for index in range(config['num_layers']):
            required[f'diffusion_transformer.transformer.layers.{index}.self_attn.to_qkv.weight'] = [3*hidden,hidden]
    if any(shapes.get(key) != shape for key,shape in required.items()):
        raise ValueError('MiniMax Music 3 flow checkpoint headers disagree with native geometry')
    if graph is not None:
        graph = validate_topology_identity(graph, config)
    return config,graph,geometry
