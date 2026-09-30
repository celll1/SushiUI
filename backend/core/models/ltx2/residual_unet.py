"""Frame-local spatial residual transport for LTX-2's joint video/audio blocks."""
from copy import deepcopy
import inspect

import torch
import torch.nn.functional as F

from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import validate_model_identity, preview_plan
from core.rewiring.training import NO_RESUME_SPEC

GRID_POLICY_KEY = "sushi_residual_video_grid_policy"
GRID_POLICY = "independent_spatial_frames_v1"


def native_config(config):
    from diffusers import LTX2VideoTransformer3DModel
    if not isinstance(config, dict):
        config = dict(config)
    parameters = inspect.signature(LTX2VideoTransformer3DModel.__init__).parameters
    resolved = {name: deepcopy(config.get(name, parameter.default)) for name, parameter in parameters.items()
                if name != "self" and parameter.default is not inspect.Parameter.empty}
    for name, source in (("out_channels", "in_channels"), ("audio_out_channels", "audio_in_channels")):
        if resolved[name] is None:
            resolved[name] = resolved[source]
    for name in ("in_channels", "out_channels", "num_attention_heads", "attention_head_dim", "cross_attention_dim",
                 "audio_in_channels", "audio_out_channels", "audio_num_attention_heads", "audio_attention_head_dim",
                 "audio_cross_attention_dim", "num_layers", "audio_scale_factor", "audio_sampling_rate", "audio_hop_length"):
        if type(resolved[name]) is not int or resolved[name] < 1:
            raise ValueError(f"LTX-2 native {name} must be a positive integer")
    if any(type(resolved[key]) is not int or resolved[key] != 1 for key in
           ("patch_size", "patch_size_t", "audio_patch_size", "audio_patch_size_t")):
        raise ValueError("LTX-2 residual execution currently requires native unit patches")
    factors = resolved["vae_scale_factors"]
    if not isinstance(factors,(list,tuple)) or len(factors) != 3 or any(type(value) is not int or value < 1 for value in factors):
        raise ValueError("LTX-2 residual execution requires native positive temporal/spatial VAE scale factors")
    if resolved["rope_type"] not in ("interleaved", "split"):
        raise ValueError("LTX-2 residual execution requires native interleaved or split RoPE")
    policy = config.get(GRID_POLICY_KEY, GRID_POLICY)
    if policy != GRID_POLICY:
        raise ValueError("Unsupported LTX-2 residual video grid policy")
    resolved[GRID_POLICY_KEY] = policy
    return resolved


def default_topology(config):
    from api.param_defaults import DIT_REWIRING_PRESETS
    from core.rewiring.spec import OPERATORS, build_spec
    config = native_config(config)
    preset = DIT_REWIRING_PRESETS["ltx2"]
    if config["num_layers"] != preset["block_count"]:
        return None
    return build_spec("ltx2", [{"name":"main", "block_count":config["num_layers"]}],
                      [{**region, **OPERATORS} for region in preset["regions"]], config)


def video_geometry(config):
    config = native_config(config)
    time, height, width = config["vae_scale_factors"]
    return dict(temporal_scale_factor=time, height_scale_factor=height,
                width_scale_factor=width, grid_policy=GRID_POLICY)


def validate_topology_identity(spec, config):
    resolved = native_config(config)
    return validate_model_identity(spec, "ltx2", [{"name":"main", "block_count":resolved["num_layers"]}], resolved)


def restore_topology(model, spec):
    inner = getattr(model, "transformer", model)
    if spec is None:
        inner.__dict__.pop("_residual_unet_spec", None)
        return
    graph = validate_topology_identity(spec, inner.config)
    if len(inner.transformer_blocks) != native_config(inner.config)["num_layers"]:
        raise ValueError("LTX-2 actual block count disagrees with native config")
    inner._residual_unet_spec = deepcopy(graph)


def wrap_transformer(model, spec):
    from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper
    restore_topology(model, spec)
    return model if isinstance(model, Ltx2BlockLoopWrapper) else Ltx2BlockLoopWrapper(model)


def configure_training_topology(model, config, *, resuming=False, pending_adapter_resume=False,
                                resume_spec=NO_RESUME_SPEC, install=True, trainer=None):
    from types import SimpleNamespace
    from api.param_defaults import TRAINING_DEFAULTS
    from core.rewiring.training import resolve_training_topology
    from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper
    inner = getattr(model, "transformer", model)
    model_topology_metadata(model)
    existing = getattr(inner, "_residual_unet_spec", None)
    declared = config.get("dit_rewiring_spec", TRAINING_DEFAULTS["dit_rewiring_spec"])
    needs_identity = config.get("dit_rewiring_mode") == "configure" or existing is not None or declared is not None or (
        resume_spec is not NO_RESUME_SPEC and resume_spec is not None)
    native = native_config(inner.config) if needs_identity else {}
    view = SimpleNamespace(config=native, _residual_unet_spec=existing)
    resolved = resolve_training_topology(view, dict(config), "ltx2", validate_topology_identity,
        lambda owner, graph: None, default_topology, resuming=resuming, pending_adapter_resume=pending_adapter_resume,
        resume_spec=resume_spec, install=False)
    if resolved is not None:
        if len(inner.transformer_blocks) != native["num_layers"]:
            raise ValueError("LTX-2 actual block count disagrees with native config")
        for key in ("_tread_router", "_block_dropout", "_blockskip_config"):
            if getattr(model, key, None) is not None:
                raise ValueError(f"LTX-2 residual training is incompatible with {key}")
        if install and trainer is None and not isinstance(model, Ltx2BlockLoopWrapper):
            raise ValueError("LTX-2 residual training requires an execution wrapper or trainer owner")
    if install:
        restore_topology(model, resolved)
        if trainer is not None and resolved is not None:
            wrapper = wrap_transformer(model, resolved)
            trainer.transformer = wrapper
            trainer.transformer_original = inner
            trainer.ltx2_block_loop_wrapper = wrapper
            pipeline = getattr(trainer, "ltx2_pipeline", None)
            if pipeline is not None:
                pipeline.transformer = wrapper
        config["dit_rewiring_mode"] = config.get("dit_rewiring_mode", TRAINING_DEFAULTS["dit_rewiring_mode"])
        config["dit_rewiring_spec"] = resolved
    return resolved


def preflight_forward(wrapper, hidden, audio, frames, height, width, coords, *,
                      timestep, audio_timestep, sigma, audio_sigma, attention_kwargs,
                      stg_blocks, perturbation_mask):
    inner = wrapper.transformer
    graph = validate_topology_identity(inner._residual_unet_spec, inner.config)
    config = native_config(inner.config)
    if len(inner.transformer_blocks) != config["num_layers"]:
        raise ValueError("LTX-2 actual block count disagrees with native config")
    for value in (frames, height, width):
        if type(value) is not int or value < 1:
            raise ValueError("LTX-2 residual execution requires explicit positive native frame/height/width dimensions")
    preview_plan(graph, height, width)
    if hidden.ndim != 3 or hidden.shape[1:] != (frames * height * width, config["in_channels"]):
        raise ValueError("LTX-2 residual video tokens must match the declared frame grid")
    if audio.ndim != 3 or audio.shape[0] != hidden.shape[0] or audio.shape[2] != config["audio_in_channels"]:
        raise ValueError("LTX-2 residual audio must keep its native packed token sequence")
    if coords is not None:
        if (coords.shape != (hidden.shape[0], 3, hidden.shape[1], 2) or not torch.isfinite(coords).all()):
            raise ValueError("LTX-2 residual video coordinates must match the native token grid")
        grid = coords.reshape(hidden.shape[0], 3, frames, height, width, 2)
        if (not torch.equal(grid[:,0],grid[:,0,:,0:1,0:1].expand_as(grid[:,0])) or
                not torch.equal(grid[:,1],grid[:,1,:,:,0:1].expand_as(grid[:,1])) or
                not torch.equal(grid[:,2],grid[:,2,:,0:1,:].expand_as(grid[:,2]))):
            raise ValueError("LTX-2 residual coordinates require a rectangular frame-local spatial lattice")
    for name, value in (("timestep",timestep),("audio_timestep",audio_timestep)):
        count = hidden.shape[1] if name == "timestep" else audio.shape[1]
        if value is not None and value.numel() not in (hidden.shape[0], hidden.shape[0] * count):
            raise ValueError(f"LTX-2 residual {name} must be global or native per-token")
    if audio_timestep is None and timestep.numel() not in (hidden.shape[0], hidden.shape[0] * audio.shape[1]):
        raise ValueError("LTX-2 tokenwise video timesteps require an explicit native audio timestep")
    for value in (sigma, audio_sigma):
        if value is not None and value.numel() != hidden.shape[0]:
            raise ValueError("LTX-2 residual prompt sigmas must be global per sample")
    if attention_kwargs or stg_blocks or perturbation_mask is not None:
        raise ValueError("LTX-2 residual execution currently refuses custom attention kwargs and STG perturbations")
    for key in ("_fbcache", "_spectrum_video", "_spectrum_audio", "_tread_router", "_block_dropout",
                "_blockskip_config", "_style_cfg", "_style_refs", "_style_processors"):
        if getattr(wrapper, key, None) is not None:
            raise ValueError(f"LTX-2 residual execution is incompatible with {key}")
    return graph


def run_rewired_blocks(wrapper, hidden, audio, frames, height, width, coords, *,
                       encoder_hidden, audio_encoder_hidden, temb, temb_audio,
                       video_scale_shift, audio_scale_shift, video_gate, audio_gate,
                       temb_prompt, temb_prompt_audio, video_rope, audio_rope,
                       video_cross_rope, audio_cross_rope, encoder_mask, audio_encoder_mask,
                       isolate_modalities, grad_ckpt):
    inner = wrapper.transformer
    offloader = wrapper._block_offloader
    swap_on = offloader is not None and getattr(offloader,"blocks_to_swap",0) > 0
    audio_shape = audio.shape
    context = dict(coords=coords, height=height, width=width, temb=temb,
                   scale_shift=video_scale_shift, gate=video_gate,
                   rope=video_rope, cross_rope=video_cross_rope)

    def coarse(old, states, flat_height, new_width, prefix):
        new_height = flat_height // frames
        factor = (old["height"] // new_height, old["width"] // new_width)
        def pool(value):
            if value is None or value.shape[1] == 1:
                return value
            if value.ndim != 3 or value.shape[1] != frames * old["height"] * old["width"]:
                raise ValueError("LTX-2 video modulation must be global or match the current spatial grid")
            batch, _, channels = value.shape
            grid = value.reshape(batch,frames,old["height"],old["width"],channels)
            grid = grid.permute(0,1,4,2,3).reshape(batch*frames,channels,old["height"],old["width"])
            pooled = F.avg_pool2d(grid,factor)
            return pooled.reshape(batch,frames,channels,new_height,new_width).permute(0,1,3,4,2).flatten(1,3)
        raw = old["coords"]
        positions = raw.permute(0,2,1,3).flatten(2)
        positions = pool(positions).reshape(raw.shape[0],-1,3,2).permute(0,2,1,3)
        return dict(coords=positions,height=new_height,width=new_width,
                    temb=pool(old["temb"]),scale_shift=pool(old["scale_shift"]),gate=pool(old["gate"]),
                    rope=inner.rope(positions,device=states.device),
                    cross_rope=inner.cross_attn_rope(positions[:,0:1],device=states.device))

    def execute(index, states, current):
        nonlocal audio
        if swap_on:
            offloader.wait_for_block(index)
        block = inner.transformer_blocks[index]
        def native(video, sound):
            return block(hidden_states=video,audio_hidden_states=sound,
                encoder_hidden_states=encoder_hidden,audio_encoder_hidden_states=audio_encoder_hidden,
                temb=current["temb"],temb_audio=temb_audio,
                temb_ca_scale_shift=current["scale_shift"],temb_ca_audio_scale_shift=audio_scale_shift,
                temb_ca_gate=current["gate"],temb_ca_audio_gate=audio_gate,
                temb_prompt=temb_prompt,temb_prompt_audio=temb_prompt_audio,
                video_rotary_emb=current["rope"],audio_rotary_emb=audio_rope,
                ca_video_rotary_emb=current["cross_rope"],ca_audio_rotary_emb=audio_cross_rope,
                encoder_attention_mask=encoder_mask,audio_encoder_attention_mask=audio_encoder_mask,
                self_attention_mask=None,audio_self_attention_mask=None,
                a2v_cross_attention_mask=None,v2a_cross_attention_mask=None,
                use_a2v_cross_attention=not isolate_modalities,use_v2a_cross_attention=not isolate_modalities,
                perturbation_mask=None,all_perturbed=False)
        video, sound = inner._gradient_checkpointing_func(native,states,audio) if grad_ckpt else native(states,audio)
        if sound.shape != audio_shape:
            raise ValueError("LTX-2 block changed the native audio token layout")
        audio = sound
        if swap_on:
            offloader.submit_move_blocks_forward(index)
        return video

    output = run_rewired_stream(hidden,inner._residual_unet_spec,"main",frames*height,width,0,
                                context,execute,coarse,grid_groups=frames)
    return output, audio


def model_topology_metadata(model):
    from core.rewiring.spec import checkpoint_metadata
    inner = getattr(model, "transformer", model)
    graph = getattr(inner, "_residual_unet_spec", None)
    if graph is None:
        return {}
    if len(inner.transformer_blocks) != native_config(inner.config)["num_layers"]:
        raise ValueError("LTX-2 export block count disagrees with native config")
    return checkpoint_metadata(validate_topology_identity(graph, inner.config))


def resolve_generation_topology(model, files):
    from types import SimpleNamespace
    from core.rewiring.spec import read_checkpoint_metadata
    from core.rewiring.adapters import resolve_adapter_topology
    inner = getattr(model, "transformer", model)
    model_topology_metadata(model)
    existing = getattr(inner, "_residual_unet_spec", None)
    needs_identity = existing is not None or any(read_checkpoint_metadata(file.metadata) is not None for file in files)
    native = native_config(inner.config) if needs_identity else {}
    view = SimpleNamespace(config=native, _residual_unet_spec=existing)
    required = resolve_adapter_topology(view, files, validate_topology_identity)
    if required is not None and len(inner.transformer_blocks) != native["num_layers"]:
        raise ValueError("LTX-2 actual block count disagrees with native config")
    return required


def export_metadata(model):
    import json
    inner = getattr(model, "transformer", model)
    metadata = model_topology_metadata(model)
    if hasattr(inner, "config"):
        raw = dict(inner.config)
        if metadata:
            raw[GRID_POLICY_KEY] = GRID_POLICY
        metadata["transformer_config"] = json.dumps(raw, allow_nan=False)
    return metadata


def preflight_checkpoint(path, *, directory_config=None):
    import json
    import re
    from pathlib import Path
    from safetensors import safe_open
    from core.rewiring.artifacts import read_artifact_topology
    source = Path(path)
    if not source.exists() and Path(str(source)+".index.json").is_file():
        source = Path(str(source)+".index.json")
    graph = read_artifact_topology(source)
    if str(source).endswith(".safetensors.index.json"):
        index = json.loads(source.read_text(encoding="utf-8"))
        metadata = index.get("metadata") or {}
        keys = list(index["weight_map"])
        sources = {source.parent / name for name in index["weight_map"].values()}
    else:
        with safe_open(str(source),framework="pt",device="cpu") as handle:
            metadata = handle.metadata() or {}
            keys = list(handle.keys())
        sources = {source}
    declaration = metadata.get("transformer_config")
    if declaration is None:
        if graph is not None:
            raise ValueError("Rewired LTX-2 checkpoint requires native transformer_config")
        if directory_config is None:
            return None, None
    raw = json.loads(declaration) if declaration is not None else dict(directory_config)
    if not isinstance(raw,dict) or type(raw.get("num_layers")) is not int:
        raise ValueError("LTX-2 checkpoint requires an explicit native transformer depth")
    config = native_config(raw)
    if graph is not None:
        graph = validate_topology_identity(graph,config)
    if any(key.startswith("net.") for key in keys) and not all(key.startswith("net.") for key in keys):
        raise ValueError("LTX-2 checkpoint mixes native and net-prefixed tensors")
    native_keys = [key.removeprefix("net.") for key in keys]
    blocks = {int(match.group(1)) for key in native_keys
              if (match := re.match(r"transformer_blocks\.(\d+)\.",key))}
    if blocks != set(range(config["num_layers"])) or not {
            "proj_in.weight","audio_proj_in.weight","proj_out.weight","audio_proj_out.weight"}.issubset(native_keys):
        raise ValueError("LTX-2 checkpoint requires complete native AV transformer headers, not adapter weights")
    video_dim = config["num_attention_heads"] * config["attention_head_dim"]
    audio_dim = config["audio_num_attention_heads"] * config["audio_attention_head_dim"]
    shapes = {"proj_in.weight":[video_dim,config["in_channels"]],
              "audio_proj_in.weight":[audio_dim,config["audio_in_channels"]],
              "proj_out.weight":[config["out_channels"],video_dim],
              "audio_proj_out.weight":[config["audio_out_channels"],audio_dim]}
    for layer in range(config["num_layers"]):
        shapes[f"transformer_blocks.{layer}.attn1.to_q.weight"] = [video_dim,video_dim]
        shapes[f"transformer_blocks.{layer}.audio_attn1.to_q.weight"] = [audio_dim,audio_dim]
    seen = set()
    checked_shapes = set()
    for shard in sources:
        with safe_open(str(shard),framework="pt",device="cpu") as handle:
            declared = (handle.metadata() or {}).get("transformer_config")
            if declared is None and declaration is None and directory_config is not None:
                shard_config = dict(directory_config)
            else:
                shard_config = json.loads(declared) if declared is not None else None
            if shard_config != raw:
                raise ValueError("LTX-2 checkpoint index and shard native configs disagree")
            for key in handle.keys():
                if key in seen or key not in keys or (str(source).endswith(".index.json") and index["weight_map"][key] != shard.name):
                    raise ValueError("LTX-2 checkpoint index and shard tensor ownership disagree")
                seen.add(key)
                native_key = key.removeprefix("net.")
                if native_key in shapes:
                    if list(handle.get_slice(key).get_shape()) != shapes[native_key]:
                        raise ValueError(f"LTX-2 native header geometry disagrees: {native_key}")
                    checked_shapes.add(native_key)
    if seen != set(keys) or checked_shapes != set(shapes):
        raise ValueError("LTX-2 checkpoint requires complete native AV transformer headers")
    return graph, raw
