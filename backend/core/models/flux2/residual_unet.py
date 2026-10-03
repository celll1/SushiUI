"""Native Flux2 double/single residual execution without patching diffusers files."""
from types import MethodType
import json
from pathlib import Path

import torch

from .single_file import FLUX2_DEFAULT_CONFIG
from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import validate_model_identity, preview_plan, checkpoint_metadata
from core.rewiring.training import NO_RESUME_SPEC


def native_model(model):
    return getattr(model, "transformer", model)


def native_config(config):
    source = config if isinstance(config, dict) else dict(native_model(config).config)
    values = {key: source.get(key, value) for key, value in FLUX2_DEFAULT_CONFIG.items()}
    axes = values["axes_dims_rope"]
    if not isinstance(axes, (list, tuple)) or len(axes) != 4 or any(type(value) is not int or value < 2 or value % 2 for value in axes):
        raise ValueError("Flux2 rewiring requires four even native RoPE axes")
    values["axes_dims_rope"] = list(axes)
    if type(values["patch_size"]) is not int or values["patch_size"] != 1 or sum(axes) != values["attention_head_dim"]:
        raise ValueError("Flux2 rewiring requires native packed patches and matching RoPE head geometry")
    return values


def validate_topology_identity(spec, config):
    resolved = native_config(config)
    return validate_model_identity(spec, "flux2", [
        {"name":"double", "block_count":resolved["num_layers"]},
        {"name":"single", "block_count":resolved["num_single_layers"]}], resolved)


def restore_topology(model, spec):
    model = native_model(model)
    if spec is None:
        model.__dict__.pop("_residual_unet_spec", None)
        saved = model.__dict__.pop("_residual_unet_forward_restore", None)
        if saved is not None:
            if saved[0]: model.forward = saved[1]
            else: model.__dict__.pop("forward", None)
        return
    spec = validate_topology_identity(spec, model)
    if not hasattr(model, "_residual_unet_forward_restore"):
        model._residual_unet_forward_restore = ("forward" in model.__dict__, model.__dict__.get("forward"))
    model._residual_unet_spec = spec
    model.forward = MethodType(run_rewired_flux2, model)


def model_topology_metadata(model):
    model = native_model(model)
    spec = getattr(model, "_residual_unet_spec", None)
    return {} if spec is None else checkpoint_metadata(validate_topology_identity(spec, model))


def configure_training_topology(model, config, *, resuming=False, pending_adapter_resume=False,
                                resume_spec=NO_RESUME_SPEC, install=True):
    from core.rewiring.training import resolve_training_topology
    return resolve_training_topology(native_model(model), config, "flux2",
        validate_topology_identity, restore_topology, default_topology, resuming=resuming,
        pending_adapter_resume=pending_adapter_resume, resume_spec=resume_spec, install=install)


def default_topology(config):
    from copy import deepcopy
    from api.param_defaults import DIT_REWIRING_PRESETS
    from core.rewiring.spec import build_spec, OPERATORS
    resolved = native_config(config)
    preset = deepcopy(DIT_REWIRING_PRESETS["flux2"])
    streams = [{"name": "double", "block_count": resolved["num_layers"]},
               {"name": "single", "block_count": resolved["num_single_layers"]}]
    if streams != preset["streams"]:
        return None
    return validate_topology_identity(build_spec("flux2", streams,
        [dict(region, **OPERATORS) for region in preset["regions"]], resolved), resolved)


def generation_token_stride(vae):
    blocks = getattr(getattr(vae, "config", None), "block_out_channels", None)
    return (2 ** (len(blocks) - 1) if blocks else 8) * 2


def preflight_checkpoint(path):
    from safetensors import safe_open
    from core.rewiring.artifacts import read_artifact_topology
    path = Path(path)
    if not path.exists() and str(path).endswith(".safetensors"):
        path = Path(str(path) + ".index.json")
    spec = read_artifact_topology(path)
    if str(path).endswith(".index.json"):
        with path.open(encoding="utf-8") as handle:
            metadata = json.load(handle).get("metadata") or {}
    else:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    serialized = metadata.get("transformer_config")
    if serialized is None:
        if spec is not None:
            raise ValueError("Rewired Flux2 checkpoint requires transformer_config")
        return spec, None
    if not isinstance(serialized, str):
        raise ValueError("Flux2 transformer_config must be JSON text")
    config = json.loads(serialized)
    if not isinstance(config, dict):
        raise ValueError("Flux2 transformer_config must be an object")
    if spec is not None:
        config = native_config(config)
        validate_topology_identity(spec, config)
    else:
        config = {key:value for key,value in config.items() if key in FLUX2_DEFAULT_CONFIG}
    return spec, config


def _shared_ids(ids, batch, count, label):
    if ids is None or ids.ndim not in (2,3):
        raise ValueError(f"Flux2 {label} positions require native 2D/3D ids")
    if ids.ndim == 3:
        if ids.shape[0] != batch or not torch.equal(ids, ids[:1].expand_as(ids)):
            raise ValueError(f"Flux2 {label} positions must share one grid across the batch")
        ids = ids[0]
    if ids.shape != (count,4):
        raise ValueError(f"Flux2 {label} position length mismatch")
    return ids


def run_rewired_flux2(model, hidden_states, encoder_hidden_states=None, timestep=None,
                     img_ids=None, txt_ids=None, guidance=None, joint_attention_kwargs=None,
                     return_dict=True, kv_cache=None, kv_cache_mode=None, num_ref_tokens=0,
                     ref_fixed_timestep=0.0, *, block_offloader=None):
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModelOutput
    spec = validate_topology_identity(model._residual_unet_spec, model)
    if kv_cache is not None or kv_cache_mode is not None or num_ref_tokens:
        raise ValueError("Flux2 residual execution does not support reference KV caching")
    for name in ("_style_ctx", "_fbcache", "_tread_config", "_blockskip_config", "_block_skip_config", "_repa_tap_depth"):
        if getattr(model, name, None) is not None:
            raise ValueError(f"Flux2 residual execution is incompatible with {name}")
    attention = dict(joint_attention_kwargs or {})
    if attention.pop("scale", 1.0) != 1.0 or attention:
        raise ValueError("Flux2 residual execution requires native attention without extra conditioning")
    blocks = [*model.transformer_blocks, *model.single_transformer_blocks]
    for block in blocks:
        if any(name in type(block.attn.processor).__name__ for name in ("NAG", "NegPip", "Style")):
            raise ValueError("Flux2 residual execution requires inactive attention guidance/style processors")
    if hidden_states.ndim != 3 or encoder_hidden_states is None or encoder_hidden_states.ndim != 3 or hidden_states.shape[0] != encoder_hidden_states.shape[0]:
        raise ValueError("Flux2 residual execution requires matching image/text batches")
    batch, count, _ = hidden_states.shape
    prefix = encoder_hidden_states.shape[1]
    image_ids = _shared_ids(img_ids, batch, count, "image")
    text_ids = _shared_ids(txt_ids, batch, prefix, "text")
    if count < 1 or not torch.isfinite(image_ids).all():
        raise ValueError("Flux2 residual execution requires a finite native image grid")
    height, width = int(image_ids[:,1].max().item())+1, int(image_ids[:,2].max().item())+1
    linear = torch.equal(image_ids[:,3], torch.arange(count,device=image_ids.device,dtype=image_ids.dtype))
    constant = torch.equal(image_ids[:,3], image_ids[:1,3].expand(count))
    if not linear and not constant:
        raise ValueError("Flux2 residual execution requires constant or row-major L coordinates")

    def positions(h,w):
        ids = image_ids.new_zeros(h*w,4)
        ids[:,0] = image_ids[0,0]
        ids[:,1] = torch.arange(h,device=ids.device).repeat_interleave(w)
        ids[:,2] = torch.arange(w,device=ids.device).repeat(h)
        ids[:,3] = torch.arange(h*w,device=ids.device) if linear else image_ids[0,3]
        return ids

    if height*width != count or not torch.equal(image_ids, positions(height,width)):
        raise ValueError("Flux2 residual execution requires one row-major 2D image grid without references")
    preview_plan(spec,height,width)
    if len(model.transformer_blocks) != model.config.num_layers or len(model.single_transformer_blocks) != model.config.num_single_layers:
        raise ValueError("Flux2 residual execution block layout mismatch")
    dual_count = len(model.transformer_blocks)
    if block_offloader is not None and getattr(block_offloader,"num_dual_blocks",dual_count) != dual_count:
        raise ValueError("Flux2 residual execution offloader layout mismatch")
    temb = model.time_guidance_embed(timestep.to(hidden_states.dtype)*1000,
        None if guidance is None else guidance.to(hidden_states.dtype)*1000)
    image_mod = model.double_stream_modulation_img(temb)
    text_mod = model.double_stream_modulation_txt(temb)
    single_mod = model.single_stream_modulation(temb)
    image = model.x_embedder(hidden_states)
    text = model.context_embedder(encoder_hidden_states)
    text_rotary = model.pos_embed(text_ids)

    def rotary(h,w):
        image_rotary = model.pos_embed(positions(h,w))
        return tuple(torch.cat((txt,img),dim=0) for txt,img in zip(text_rotary,image_rotary))

    def coarse_context(context,value,h,w,extras): return rotary(h,w)

    def execute(index,value,context,stream):
        unified = index if stream == "double" else dual_count+index
        if block_offloader is not None: block_offloader.wait_for_block(unified)
        if stream == "double":
            block = model.transformer_blocks[index]
            args = (value[:,prefix:],value[:,:prefix],image_mod,text_mod,context,None)
        else:
            block = model.single_transformer_blocks[index]
            args = (value,None,single_mod,context,None)
        if torch.is_grad_enabled() and model.gradient_checkpointing:
            output = model._gradient_checkpointing_func(block,*args)
        else: output = block(*args)
        if stream == "double": output = torch.cat((output[0],output[1]),dim=1)
        if block_offloader is not None: block_offloader.submit_move_blocks_forward(unified)
        return output

    hidden = torch.cat((text,image),dim=1)
    for stream in ("double","single"):
        hidden = run_rewired_stream(hidden,spec,stream,height,width,prefix,rotary(height,width),
            lambda index,value,context: execute(index,value,context,stream),coarse_context)
    output = model.proj_out(model.norm_out(hidden[:,prefix:],temb))
    return Flux2Transformer2DModelOutput(sample=output) if return_dict else (output,)
