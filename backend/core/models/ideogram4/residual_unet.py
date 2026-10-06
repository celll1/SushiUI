"""Ideogram 4 image-only residual transport with native packed segment context."""
import inspect
import math

import torch
import torch.nn.functional as F

from core.rewiring.training import NO_RESUME_SPEC
from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import validate_model_identity, checkpoint_metadata


def native_config(config):
    from .vendor.transformer import Ideogram4Transformer2DModel
    source = config if isinstance(config, dict) else dict(config.config)
    signature = inspect.signature(Ideogram4Transformer2DModel.__init__)
    resolved = {key: source.get(key, value.default) for key, value in signature.parameters.items()
                if key != "self" and value.default is not inspect.Parameter.empty}
    for key in ("in_channels", "num_layers", "attention_head_dim", "num_attention_heads",
                "intermediate_size", "adaln_dim", "llm_features_dim", "rope_theta"):
        if type(resolved[key]) is not int or resolved[key] < 1:
            raise ValueError(f"Ideogram 4 native {key} must be a positive integer")
    head = resolved["attention_head_dim"]
    if head % 2:
        raise ValueError("Ideogram 4 rotary head width must be even")
    sections = resolved["mrope_section"]
    if not isinstance(sections, (list, tuple)) or len(sections) != 3 or any(
            type(value) is not int or value < 0 for value in sections):
        raise ValueError("Ideogram 4 MRoPE requires three nonnegative section sizes")
    if any(size and 3*(size-1)+axis >= head//2 for axis,size in enumerate(sections) if axis):
        raise ValueError("Ideogram 4 MRoPE section exceeds native frequency width")
    resolved["mrope_section"] = list(sections)
    epsilon = resolved["norm_eps"]
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Ideogram 4 norm epsilon must be finite and positive")
    resolved["norm_eps"] = float(epsilon)
    return resolved


def validate_topology_identity(spec, config):
    resolved = native_config(config)
    return validate_model_identity(spec, "ideogram4",
        [{"name": "main", "block_count": resolved["num_layers"]}], resolved)


def restore_topology(model, spec):
    if spec is None:
        model.__dict__.pop("_residual_unet_spec", None)
    else:
        model._residual_unet_spec = validate_topology_identity(spec, dict(model.config))


def model_topology_metadata(model):
    spec = getattr(model, "_residual_unet_spec", None)
    return {} if spec is None else checkpoint_metadata(validate_topology_identity(spec, dict(model.config)))


def configure_training_topology(model, config, *, unconditional_transformer=None, resuming=False,
                                pending_adapter_resume=False, resume_spec=NO_RESUME_SPEC, install=True):
    from core.rewiring.training import resolve_training_topology
    pair_topology_metadata(model,unconditional_transformer)
    staged = dict(config)
    resolved = resolve_training_topology(model,staged,"ideogram4",validate_topology_identity,
        restore_topology,default_topology,resuming=resuming,pending_adapter_resume=pending_adapter_resume,
        resume_spec=resume_spec,install=False)
    if unconditional_transformer is not None:
        other = resolve_training_topology(unconditional_transformer,dict(config),"ideogram4",
            validate_topology_identity,restore_topology,default_topology,resuming=resuming,
            pending_adapter_resume=pending_adapter_resume,resume_spec=resume_spec,install=False)
        if other != resolved:
            raise ValueError("Ideogram 4 conditional and unconditional training topologies disagree")
    if install:
        restore_topology(model,resolved)
        if unconditional_transformer is not None:restore_topology(unconditional_transformer,resolved)
        from api.param_defaults import TRAINING_DEFAULTS
        config["dit_rewiring_mode"] = config.get("dit_rewiring_mode",TRAINING_DEFAULTS["dit_rewiring_mode"])
        config["dit_rewiring_spec"] = resolved
    return resolved


def default_topology(config):
    from api.param_defaults import DIT_REWIRING_PRESETS
    from core.rewiring.spec import build_spec, OPERATORS
    resolved = native_config(config)
    preset = DIT_REWIRING_PRESETS["ideogram4"]
    if resolved["num_layers"] != preset["block_count"]:
        return None
    return validate_topology_identity(build_spec("ideogram4",
        [{"name":"main","block_count":resolved["num_layers"]}],
        [dict(region,**OPERATORS) for region in preset["regions"]],resolved),resolved)


def inspect_artifact(path):
    import json
    import re
    from pathlib import Path
    from safetensors import safe_open
    source = Path(path)
    if source.is_dir():
        if source.name == "transformer" and (source.parent/"unconditional_transformer").is_dir():
            source = source.parent
        topology,_ = preflight_directory(source)
        configs = {};keys = {}
        for component in ("transformer","unconditional_transformer"):
            directory = source/component
            with (directory/"config.json").open(encoding="utf-8") as handle:configs[component] = json.load(handle)
            keys[component] = []
            for shard in directory.glob("*.safetensors"):
                with safe_open(str(shard),framework="pt",device="cpu") as handle:keys[component].extend(handle.keys())
    else:
        if not source.exists() and Path(str(source)+".index.json").is_file():source = Path(str(source)+".index.json")
        topology,_ = preflight_checkpoint(source)
        if str(source).endswith(".safetensors.index.json"):
            with source.open(encoding="utf-8") as handle:index = json.load(handle)
            metadata = index.get("metadata") or {};all_keys = list(index["weight_map"])
        else:
            with safe_open(str(source),framework="pt",device="cpu") as handle:
                metadata = handle.metadata() or {};all_keys = list(handle.keys())
        configs = {};keys = {}
        for component in ("transformer","unconditional_transformer"):
            keys[component] = [key.removeprefix(component+".") for key in all_keys if key.startswith(component+".")]
            declaration = metadata.get(component+"_config")
            if not isinstance(declaration,str):
                raise ValueError("Ideogram 4 inspection requires native configs for both backbones")
            configs[component] = json.loads(declaration)
    for component,config in configs.items():
        if not isinstance(config,dict):raise ValueError("Ideogram 4 native inspection config must be an object")
        if "num_layers" not in config:raise ValueError("Ideogram 4 inspection requires explicit native block counts")
        configs[component] = native_config(config)
        native_keys = keys[component]
        if "input_proj.weight" not in native_keys:
            raise ValueError("Ideogram 4 inspection requires native model weights, not adapters")
        blocks = {int(match.group(1)) for key in native_keys if (match := re.match(r"^layers\.(\d+)\.",key))}
        if len(blocks) != configs[component]["num_layers"] or blocks != set(range(len(blocks))):
            raise ValueError("Ideogram 4 native config and checkpoint block counts disagree")
    if configs["transformer"] != configs["unconditional_transformer"]:
        raise ValueError("Ideogram 4 editor requires matching native backbone configs")
    return configs["transformer"],topology


def pair_topology_metadata(transformer, unconditional_transformer=None):
    metadata = model_topology_metadata(transformer)
    if unconditional_transformer is not None and model_topology_metadata(unconditional_transformer) != metadata:
        raise ValueError("Ideogram 4 conditional and unconditional topologies disagree")
    return metadata


def preflight_checkpoint(path, *, load_unconditional=True):
    import json
    from pathlib import Path
    from safetensors import safe_open
    from core.rewiring.artifacts import read_artifact_topology
    source = Path(path)
    if not source.exists() and Path(str(source)+".index.json").is_file():
        source = Path(str(source)+".index.json")
    spec = read_artifact_topology(source)
    if spec is None:
        return None, {}
    if str(source).endswith(".safetensors.index.json"):
        with source.open(encoding="utf-8") as handle:index = json.load(handle)
        metadata = index.get("metadata") or {};keys = list(index["weight_map"])
    else:
        with safe_open(str(source),framework="pt",device="cpu") as handle:
            metadata = handle.metadata() or {};keys = list(handle.keys())
    configs = {}
    for component in ("transformer","unconditional_transformer"):
        present = any(key.startswith(component+".") for key in keys)
        if not present:
            if component == "transformer" or load_unconditional:
                raise ValueError(f"Rewired Ideogram 4 checkpoint requires {component} weights")
            continue
        declaration = metadata.get(component+"_config")
        if not isinstance(declaration,str):
            raise ValueError(f"Rewired Ideogram 4 requires native {component}_config JSON")
        config = json.loads(declaration)
        if not isinstance(config,dict):
            raise ValueError("Ideogram 4 native transformer config must be an object")
        configs[component] = native_config(config)
        validate_topology_identity(spec,configs[component])
    return spec, configs


def preflight_directory(path, *, load_unconditional=True):
    import json
    from pathlib import Path
    from core.rewiring.artifacts import read_directory_topology
    root = Path(path)
    cond = read_directory_topology(root/"transformer")
    uncond_dir = root/"unconditional_transformer"
    uncond = read_directory_topology(uncond_dir) if uncond_dir.is_dir() else None
    if uncond_dir.is_dir() and cond != uncond:
        raise ValueError("Ideogram 4 conditional and unconditional topologies disagree")
    if cond is None:
        return None, {}
    if load_unconditional and not uncond_dir.is_dir():
        raise ValueError("Rewired Ideogram 4 directory requires unconditional_transformer")
    configs = {}
    for component in ("transformer","unconditional_transformer"):
        directory = root/component
        if not directory.is_dir():continue
        with (directory/"config.json").open(encoding="utf-8") as handle:config = json.load(handle)
        if not isinstance(config,dict):
            raise ValueError("Ideogram 4 native directory config must be an object")
        configs[component] = native_config(config)
        validate_topology_identity(cond,configs[component])
    return cond, configs


def save_checkpoint(transformer, unconditional_transformer, path, *, metadata=None, max_shard_bytes=None, vae=None):
    import json
    from core.models.common.single_file_format import dedup_tensors, save_single_file_state, DEFAULT_MAX_SHARD_BYTES
    from core.rewiring.spec import META_VERSION, META_SPEC, META_HASH
    topology_metadata = pair_topology_metadata(transformer,unconditional_transformer)
    from .vae_alignment import pair_metadata
    alignment_metadata = pair_metadata(transformer,unconditional_transformer)
    if topology_metadata and unconditional_transformer is None:
        raise ValueError("Rewired Ideogram 4 inference export requires both transformers")
    declarations = {"transformer":native_config(transformer)}
    if unconditional_transformer is not None:
        declarations["unconditional_transformer"] = native_config(unconditional_transformer)
    saved = {str(key):str(value) for key,value in (metadata or {}).items()
             if key not in (META_VERSION,META_SPEC,META_HASH)}
    saved.update(model_type="ideogram4",format="pt")
    saved.update({key+"_config":json.dumps(config) for key,config in declarations.items()})
    saved.update(topology_metadata)
    saved.update(alignment_metadata)
    named = (("transformer."+key,value) for key,value in transformer.state_dict().items())
    if unconditional_transformer is not None:
        from itertools import chain
        named = chain(named,(("unconditional_transformer."+key,value)
                            for key,value in unconditional_transformer.state_dict().items()))
    named = list(named)
    if vae is not None:
        from core.models.components.latent_adapter import NormalizedImageVAE
        if isinstance(vae,NormalizedImageVAE):
            vae = vae.inner
        named.extend(("vae."+key,value) for key,value in vae.state_dict().items())
    state,dropped = dedup_tensors(named)
    # Both backbones need complete keys even when a caller shares storage.
    for key,value in named:
        if key in dropped:state[key] = value.detach().cpu().contiguous().clone()
    return save_single_file_state(state,saved,str(path),
        max_shard_bytes=DEFAULT_MAX_SHARD_BYTES if max_shard_bytes is None else max_shard_bytes)


def run_rewired_ideogram4(model, hidden, position_ids, segment_ids, indicator, adaln_input):
    from .vendor.transformer import Ideogram4AttnProcessor, OUTPUT_IMAGE_INDICATOR, LLM_TOKEN_INDICATOR
    spec = validate_topology_identity(model._residual_unet_spec, dict(model.config))
    if len(model.layers) != spec["streams"][0]["block_count"]:
        raise ValueError("Ideogram 4 native block count disagrees with topology")
    for key in ("_fbcache", "_style_ctx", "_nag_active", "_negpip_active", "_tread_router",
                "_blockskip_config", "_repa_tap_depth"):
        if getattr(model, key, None) is not None and getattr(model, key) is not False:
            raise ValueError(f"Ideogram 4 residual U-Net is incompatible with {key}")
    if any(type(block.attention.processor) is not Ideogram4AttnProcessor for block in model.layers):
        raise ValueError("Ideogram 4 residual U-Net requires native attention processors")
    batch,length,_ = hidden.shape
    if indicator.shape != (batch,length) or segment_ids.shape != indicator.shape or position_ids.shape != (batch,length,3):
        raise ValueError("Ideogram 4 residual transport requires aligned packed context")
    if adaln_input.shape[:2] != (batch,1):
        raise ValueError("Ideogram 4 residual transport requires per-sample timestep conditioning")
    image = indicator == OUTPUT_IMAGE_INDICATOR
    count = int(image[0].sum().item())
    prefix = length-count
    if count < 1 or not bool(image[:,prefix:].all()) or bool(image[:,:prefix].any()):
        raise ValueError("Ideogram 4 residual transport requires one contiguous image tail per sample")
    if not bool(((indicator[:,:prefix] == LLM_TOKEN_INDICATOR) | (indicator[:,:prefix] == 0)
                 | (indicator[:,:prefix] == -1)).all()):
        raise ValueError("Ideogram 4 residual transport encountered unsupported packed roles")
    image_segments = segment_ids[:,prefix:]
    if not bool((image_segments == image_segments[:,:1]).all()):
        raise ValueError("Ideogram 4 residual transport requires one image segment per sample")
    positions = position_ids[:,prefix:].float()
    height = int(torch.unique(positions[0,:,1]).numel())
    width = int(torch.unique(positions[0,:,2]).numel())
    rows = torch.arange(height,device=hidden.device).repeat_interleave(width)
    cols = torch.arange(width,device=hidden.device).repeat(height)
    if height*width != count:
        raise ValueError("Ideogram 4 residual transport requires a complete image grid")
    expected = positions[:,:1].expand(-1,count,-1).clone()
    expected[:,:,1] += rows;expected[:,:,2] += cols
    if not torch.equal(positions,expected):
        raise ValueError("Ideogram 4 residual transport requires native row-major image positions")

    def make_context(pos, segments):
        cos,sin = model.rotary_emb(pos)
        mask = (segments.unsqueeze(2) == segments.unsqueeze(1)).unsqueeze(1)
        return pos,segments,mask,(cos.to(hidden.dtype),sin.to(hidden.dtype))

    def coarse_context(context, value, h, w, text_prefix):
        pos,segments,_,_ = context
        old_h = int(torch.unique(pos[0,text_prefix:,1]).numel())
        old_w = int(torch.unique(pos[0,text_prefix:,2]).numel())
        grid = pos[:,text_prefix:].transpose(1,2).reshape(batch,3,old_h,old_w)
        pooled = F.avg_pool2d(grid.float(),(old_h//h,old_w//w)).flatten(2).transpose(1,2)
        pos = torch.cat((pos[:,:text_prefix].float(),pooled),dim=1)
        segments = torch.cat((segments[:,:text_prefix],segments[:,text_prefix:text_prefix+1].expand(-1,h*w)),dim=1)
        return make_context(pos,segments)

    offloader = getattr(model,"_block_offloader",None)
    def execute(index, value, context):
        _,segments,mask,rope = context
        block = model.layers[index]
        if offloader is not None:offloader.wait_for_block(index)
        if torch.is_grad_enabled() and model.gradient_checkpointing:
            output = model._gradient_checkpointing_func(block,value,mask,rope,adaln_input,segments)
        else:
            output = block(value,mask,rope,adaln_input,segments)
        if offloader is not None:offloader.submit_move_blocks_forward(index)
        return output
    return run_rewired_stream(hidden,spec,"main",height,width,prefix,
                              make_context(position_ids,segment_ids),execute,coarse_context)
