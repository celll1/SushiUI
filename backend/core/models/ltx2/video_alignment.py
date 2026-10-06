"""LTX video-only target-grid boundary; audio retains its native geometry."""

from core.models.components.latent_adapter import _Config
from core.models.components.video_latent_adapter import map_video_tokens
from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper


class AlignedLtx2BlockLoopWrapper(Ltx2BlockLoopWrapper):
    def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states,
                audio_encoder_hidden_states, timestep, audio_timestep=None, sigma=None,
                audio_sigma=None, encoder_attention_mask=None, audio_encoder_attention_mask=None,
                num_frames=None, height=None, width=None, fps=24.0, audio_num_frames=None,
                video_coords=None, audio_coords=None, isolate_modalities=False,
                spatio_temporal_guidance_blocks=None, perturbation_mask=None,
                use_cross_timestep=False, attention_kwargs=None, return_dict=True):
        arguments = dict(locals())
        arguments.pop("self")
        arguments.pop("__class__", None)
        pair = self.transformer._video_latent_alignment
        grid = (num_frames, height, width)
        # Sigma conditions prompt modulation; timestep defines the actual video state.
        state_sigma = timestep / 1000
        arguments["hidden_states"] = map_video_tokens(pair.input, hidden_states, grid=grid, sigma=state_sigma)
        result = super().forward(**arguments)
        if isinstance(result, tuple):
            return (map_video_tokens(pair.output, result[0], grid=grid, velocity=True), result[1])
        result.sample = map_video_tokens(pair.output, result.sample, grid=grid, velocity=True)
        return result


def install_video_alignment(model, alignment, contract):
    """Install before optimizer/controllers; serialization is a separate acceptance gate."""
    from core.models.components.latent_io import resize_latent_io
    from core.models.components.wiring import LTX2_WIRING
    from core.models.ltx2.residual_unet import native_config, restore_topology
    from core.rewiring.spec import build_spec

    inner = getattr(model, "transformer", model)
    if getattr(inner, "_video_latent_alignment", None) is not None:
        raise ValueError("LTX video alignment is already installed")
    for name in ("_fbcache", "_spectrum_video", "_spectrum_audio", "_tread_router",
                 "_blockskip_config", "_block_dropout", "_style_cfg", "_style_refs", "_block_offloader"):
        if getattr(model, name, None) is not None:
            raise ValueError("Install LTX video alignment before runtime controllers")
    config = native_config(inner.config)
    declaration = alignment.declaration
    if (config["in_channels"] != declaration["old_channels"] or
            config["out_channels"] != declaration["old_channels"] or
            tuple(config["vae_scale_factors"][1:]) != (declaration["old_scale"],)*2):
        raise ValueError("LTX source geometry disagrees with the paired alignment")
    if (contract.image.latent_channels != declaration["new_channels"] or
            contract.image.spatial_scale != declaration["new_scale"] or
            contract.image.latent_rank != declaration["ndim"] or
            contract.image.normalization != declaration["norm"] or
            contract.image.normalization_pack != declaration["norm_pack"] or
            contract.image.pixel_channels != declaration["input_channels"]):
        raise ValueError("LTX target codec disagrees with the paired alignment")
    graph = getattr(inner, "_residual_unet_spec", None)
    inner._video_resize_report = resize_latent_io(inner, LTX2_WIRING.latent_io,
        max(declaration["old_channels"], declaration["new_channels"]))
    scales = [contract.temporal_scale, contract.image.spatial_scale, contract.image.spatial_scale]
    inner.register_to_config(vae_scale_factors=scales)
    inner.rope.scale_factors = tuple(scales)
    inner.rope.audio_latents_per_second = float(config["audio_sampling_rate"]) / config["audio_hop_length"] / scales[0]
    inner._video_latent_alignment = alignment.to(device=inner.proj_in.weight.device, dtype=inner.proj_in.weight.dtype)
    inner._video_codec_contract = contract
    inner._video_source_config = config
    if graph is not None:
        graph = build_spec("ltx2", [{"name": "main", "block_count": config["num_layers"]}],
                           graph["regions"], native_config(inner.config))
        restore_topology(inner, graph)
    return aligned_wrapper(inner)


def aligned_wrapper(inner):
    declaration = inner._video_latent_alignment.declaration
    wrapper = AlignedLtx2BlockLoopWrapper(inner)
    wrapper.config = _Config(dict(inner.config), in_channels=declaration["new_channels"],
                            out_channels=declaration["new_channels"])
    return wrapper
