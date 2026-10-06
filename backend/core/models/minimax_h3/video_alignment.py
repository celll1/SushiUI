"""Video-only paired boundary on H3's target-native packed AV sequence."""

from dataclasses import asdict
from math import prod

import torch

from core.models.components.latent_adapter import _Config
from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper
from .h3_pipeline_ops import patchify_video_latents, unpatchify_video_rows

VIDEO_POSITION_POLICY_KEY = "sushi_video_position_policy"
VIDEO_POSITION_POLICY = "pixel_frame_v1"


def forward_kwargs(transformer, layout):
    inner = getattr(transformer,"transformer",transformer)
    if getattr(inner,"_video_latent_alignment",None) is None:
        return {}
    if not isinstance(transformer,AlignedMiniMaxH3BlockLoopWrapper):
        raise ValueError("Aligned H3 requires its video execution wrapper")
    if (layout.get("video_codec_contract") != asdict(inner._video_codec_contract)
            or layout.get("video_position_policy") != VIDEO_POSITION_POLICY
            or inner.config.get(VIDEO_POSITION_POLICY_KEY) != VIDEO_POSITION_POLICY):
        raise ValueError("H3 layout disagrees with the codec/position contract")
    validate_codec_grids(layout.get("video_codec_grids"),layout["video_indices"].numel(),
                         layout["position_ids"].device)
    return dict(video_codec_grids=layout["video_codec_grids"])


def validate_codec_grids(grids, count, device):
    if not isinstance(grids,(list,tuple)) or not grids:
        raise ValueError("Aligned H3 requires explicit grids for every target/reference video segment")
    indices = []
    for group in grids:
        if not isinstance(group,dict) or set(group) != {"rows","frames","height","width"}:
            raise ValueError("H3 codec grid segment fields differ")
        frames,height,width = (group[key] for key in ("frames","height","width"))
        if any(type(n) is not int or n < 1 for n in (frames,height,width)) or height % 2 or width % 2:
            raise ValueError("H3 codec segment requires an even positive spatial grid")
        rows = group["rows"]
        if (not torch.is_tensor(rows) or rows.dtype != torch.long or rows.ndim != 1 or rows.device != device
                or rows.numel() != frames*(height//2)*(width//2)):
            raise ValueError("H3 codec segment rows disagree with their device/grid")
        indices.append(rows)
    covered = torch.cat(indices)
    if covered.numel() != count or not torch.equal(covered.sort().values,torch.arange(count,device=device)):
        raise ValueError("H3 codec grids must partition every video row exactly once")
    return indices


def map_video_rows(bridge, rows, *, grid, patch, clean_fraction=None, velocity=False):
    if (len(grid) != 3 or any(type(n) is not int or n < 1 for n in grid)
            or tuple(patch) != (1,2,2) or grid[1] % 2 or grid[2] % 2):
        raise ValueError("H3 video boundary requires declared frame/height/width and 1x2x2 patches")
    frames, height, width = grid
    count = frames*(height//2)*(width//2)
    if rows.ndim != 3 or rows.shape[0] < 1 or rows.shape[1:] != (count,4*bridge.source_channels):
        raise ValueError("H3 video rows disagree with their codec grid")
    latent = unpatchify_video_rows(rows.to(bridge.projection.weight.dtype),frames,height,width,bridge.source_channels,patch)
    if velocity:
        mapped = bridge.flow_velocity(latent,convention="x0_minus_eps")
    else:
        if clean_fraction is None or clean_fraction.shape != (count,):
            raise ValueError("H3 video boundary requires a clean fraction per patch row")
        coefficient = clean_fraction.to(device=latent.device,dtype=latent.dtype)
        if not torch.isfinite(coefficient).all() or (coefficient < 0).any() or (coefficient > 1).any():
            raise ValueError("H3 clean fractions must be finite values from zero to one")
        coefficient = coefficient.reshape(1,1,frames,height//2,width//2).repeat_interleave(2,-2).repeat_interleave(2,-1)
        mapped = bridge.epsilon(latent)+coefficient*bridge._bias_like(latent)
    return patchify_video_latents(mapped,patch)


class AlignedMiniMaxH3BlockLoopWrapper(MiniMaxH3BlockLoopWrapper):
    def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states,
                timestep, timestep_indices, token_tags, position_ids, video_indices, audio_indices, text_indices,
                attention_kwargs=None, return_dict=True, *, residual_video_grid=None, video_codec_grids=None):
        pair = self.transformer._video_latent_alignment
        total = token_tags.numel()
        indices = (video_indices,audio_indices,text_indices)
        if (token_tags.ndim != 1 or token_tags.dtype != torch.long
                or token_tags.device != hidden_states.device or (token_tags < 0).any() or (token_tags >= 3).any()
                or position_ids.shape != (total,3)
                or position_ids.device != hidden_states.device or not torch.isfinite(position_ids).all()
                or any(index.ndim != 1 or index.dtype != torch.long or index.device != hidden_states.device
                       for index in indices)):
            raise ValueError("H3 boundary requires device-local packed modality indices and positions")
        packed = torch.cat(indices)
        if packed.numel() != total or not torch.equal(packed.sort().values,torch.arange(total,device=packed.device)):
            raise ValueError("H3 boundary modality indices must partition the packed sequence")
        if (hidden_states.ndim != 3 or hidden_states.shape[0] < 1
                or hidden_states.shape[1:] != (video_indices.numel(),4*pair.input.source_channels)):
            raise ValueError("H3 video rows disagree with the external codec width")
        count = video_indices.numel()
        groups = validate_codec_grids(video_codec_grids,count,hidden_states.device)
        if (timestep.ndim != 1 or not torch.isfinite(timestep).all() or (timestep < 0).any()
                or (timestep > 1).any() or timestep.device != hidden_states.device
                or timestep_indices.device != hidden_states.device or timestep_indices.dtype != torch.long
                or timestep_indices.shape != token_tags.shape or (timestep_indices < 0).any()
                or (timestep_indices >= timestep.numel()).any()):
            raise ValueError("H3 boundary requires valid packed clean-fraction timesteps")
        fractions = timestep[timestep_indices[video_indices]]
        patch = self.transformer.config.patch_size
        internal = pair.input.projection.weight.new_empty(hidden_states.shape[0],count,4*pair.input.target_channels)
        for group,indices in zip(video_codec_grids,groups):
            mapped = map_video_rows(pair.input,hidden_states.index_select(1,indices),
                grid=(group["frames"],group["height"],group["width"]),patch=patch,
                clean_fraction=fractions.index_select(0,indices))
            internal = internal.index_copy(1,indices,mapped)
        result = super().forward(internal,audio_hidden_states,encoder_hidden_states,timestep,timestep_indices,
            token_tags,position_ids,video_indices,audio_indices,text_indices,attention_kwargs,return_dict,
            residual_video_grid=residual_video_grid)
        video = result[0] if isinstance(result,tuple) else result.sample
        external = pair.output.projection.weight.new_empty(video.shape[0],count,4*pair.output.target_channels)
        for group,indices in zip(video_codec_grids,groups):
            mapped = map_video_rows(pair.output,video.index_select(1,indices),
                grid=(group["frames"],group["height"],group["width"]),patch=patch,velocity=True)
            external = external.index_copy(1,indices,mapped)
        if isinstance(result,tuple):
            return external,result[1]
        result.sample = external
        return result


def install_video_alignment(model, alignment, contract):
    from core.models.components.latent_io import resize_latent_io
    from core.models.components.wiring import LatentIOSpec
    from core.rewiring.spec import build_spec
    from .residual_unet import native_config,restore_topology,validate_topology_identity,VIDEO_CODEC_KEY
    inner = getattr(model,"transformer",model)
    if getattr(inner,"_video_latent_alignment",None) is not None:
        raise ValueError("H3 video alignment is already installed")
    if not alignment.ready:
        raise ValueError("H3 video alignment requires fitted calibration directions")
    for name in ("_fbcache","_skip_blocks","_residual_probe","_block_offloader"):
        if getattr(model,name,None) is not None:
            raise ValueError("Install H3 video alignment before runtime controllers")
    if getattr(inner,"_attention_plan",None) is not None:
        raise ValueError("Install H3 video alignment before attention controllers")
    source = native_config(inner.config)
    d = alignment.declaration
    if source["in_channels"] != d["old_channels"] or d["old_scale"] != 16 or tuple(source["patch_size"]) != (1,2,2):
        raise ValueError("H3 source geometry disagrees with the video alignment")
    if (contract.image.latent_channels,contract.image.spatial_scale,contract.image.latent_rank,
            contract.image.normalization,contract.image.normalization_pack,contract.image.pixel_channels) != (
            d["new_channels"],d["new_scale"],d["ndim"],d["norm"],d["norm_pack"],d["input_channels"]):
        raise ValueError("H3 target codec disagrees with the video alignment")
    graph = getattr(inner,"_residual_unet_spec",None)
    if graph is not None:
        validate_topology_identity(graph,inner.config)
    spec = LatentIOSpec("proj_in","proj_out",extra_in_channels=0,in_repeat=1,out_bias=True,
                        in_kind="packed_linear",out_kind="packed_linear",pack_elems=prod(source["patch_size"]),
                        in_channel_order="outer",out_channel_order="outer")
    inner._video_resize_report = resize_latent_io(inner,spec,max(d["old_channels"],d["new_channels"]))
    inner.register_to_config(**{VIDEO_CODEC_KEY:asdict(contract),VIDEO_POSITION_POLICY_KEY:VIDEO_POSITION_POLICY})
    inner._video_latent_alignment = alignment.to(device=inner.proj_in.weight.device,dtype=inner.proj_in.weight.dtype)
    inner._video_codec_contract,inner._video_source_config = contract,source
    if graph is not None:
        restore_topology(inner,build_spec("minimax_h3",[{"name":"main","block_count":source["num_layers"]}],
                                        graph["regions"],native_config(inner.config)))
    wrapper = AlignedMiniMaxH3BlockLoopWrapper(inner)
    wrapper.config = _Config(dict(inner.config),in_channels=d["new_channels"])
    return wrapper
