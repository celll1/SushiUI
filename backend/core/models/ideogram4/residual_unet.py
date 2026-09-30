"""Ideogram 4 image-only residual transport with native packed segment context."""
import inspect
import math

import torch
import torch.nn.functional as F

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
