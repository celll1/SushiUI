"""ACE decoder boundaries; the frozen conditioning codec remains independent."""

from copy import deepcopy

import torch
from torch import nn

from core.models.components.audio_latent_adapter import calibration_audio_grid


def bind_encoding_contract(codec, alignment):
    from core.models.components.audio_latent_adapter import AudioCodecContract
    from dataclasses import asdict
    if AudioCodecContract.from_oobleck(codec) != alignment.target:
        raise ValueError("ACE target codec differs from the audio encoding contract")
    codec._sushi_audio_encoding_contract = dict(target=asdict(alignment.target),
        sample_alignment=alignment.target.hop_length, policy="zero_tail_target_hop_v1")


def _uniform_mask(mask, channels):
    if mask.ndim != 3 or mask.shape[-1] < 1 or not torch.equal(mask, mask[..., :1].expand_as(mask)):
        raise ValueError("ACE audio swap requires channel-uniform chunk masks")
    return mask[..., :1].expand(-1, -1, channels)


def native_context_to_target(alignment, context, *, target_frames=None):
    """Convert conditioning only; RVQ and timbre still consume native latents."""
    old, new = alignment.source.latent_channels, alignment.target.latent_channels
    if context.ndim != 3 or context.shape[-1] != 2 * old or min(context.shape) < 1:
        raise ValueError("ACE native context must contain source latents and chunk masks")
    samples = alignment.source.waveform_samples(context.shape[1])
    if target_frames is None:
        frames = (samples + alignment.target.hop_length - 1) // alignment.target.hop_length
    else:
        if type(target_frames) is not int or target_frames < 1:
            raise ValueError("ACE target conditioning frame count must be positive")
        frames = target_frames
        required = (alignment.target.waveform_samples(frames) + alignment.source.hop_length - 1) // alignment.source.hop_length
        if required != context.shape[1]:
            raise ValueError("ACE native conditioning duration differs from the target grid")
    source, mask = context.split(old, dim=-1)
    source = calibration_audio_grid(source.transpose(1, 2), frames)
    source = torch.nn.functional.pad(source, (0, 0, 0, alignment.backbone_channels-old))
    target = alignment.output(source).transpose(1, 2)
    mask = _uniform_mask(mask, new).transpose(1, 2)
    # Masks describe intervals, not an affine latent variable.
    mask = torch.nn.functional.interpolate(mask.float(), size=frames, mode="nearest").to(context.dtype)
    return torch.cat((target, mask.transpose(1, 2)), dim=-1)


def decoder_inputs(alignment, hidden, context, sigma):
    new = alignment.target.latent_channels
    if hidden.ndim != 3 or hidden.shape[-1] != new or context.shape != (*hidden.shape[:2], 2*new):
        raise ValueError("ACE target hidden/context geometry differs")
    source, mask = context.split(new, dim=-1)
    hidden = alignment.input.flow_state(hidden.transpose(1, 2), sigma).transpose(1, 2)
    source = alignment.input(source.transpose(1, 2)).transpose(1, 2)
    mask = _uniform_mask(mask, alignment.backbone_channels)
    return hidden, torch.cat((source, mask), dim=-1)


def decoder_output(alignment, velocity):
    return alignment.output.flow_velocity(velocity.transpose(1, 2)).transpose(1, 2)


def sampler_target_inputs(alignment, source, attention, clean, mask, crossfade):
    """Map sampler state and intervals while RVQ inputs stay on the source grid."""
    if source.ndim != 3 or source.shape[-1] != alignment.source.latent_channels or min(source.shape) < 1:
        raise ValueError("ACE sampler source geometry differs from its conditioning codec")
    frames = (source.shape[1] * alignment.source.hop_length + alignment.target.hop_length-1) // alignment.target.hop_length
    def latent(value):
        if value.shape != source.shape:
            raise ValueError("ACE sampler clean source geometry differs")
        teacher = calibration_audio_grid(value.transpose(1, 2), frames)
        teacher = torch.nn.functional.pad(teacher, (0, 0, 0, alignment.backbone_channels-alignment.source.latent_channels))
        return alignment.output(teacher).transpose(1, 2)
    def interval(value):
        if value.shape != source.shape[:2]:
            raise ValueError("ACE sampler interval geometry differs from the source grid")
        return torch.nn.functional.interpolate(value[:, None].float(), size=frames, mode="nearest")[:, 0].to(value.dtype)
    if (clean is None) != (mask is None):
        raise ValueError("ACE repaint requires both clean source and interval mask")
    if type(crossfade) is not int or crossfade < 0:
        raise ValueError("ACE repaint crossfade must contain nonnegative source frames")
    return dict(source=latent(source), attention=None if attention is None else interval(attention),
                clean=None if clean is None else latent(clean), mask=None if mask is None else interval(mask),
                crossfade=(crossfade*alignment.source.hop_length+alignment.target.hop_length-1)//alignment.target.hop_length)


def install_decoder_alignment(model, alignment, *, position_policy="source_hop_v1"):
    """Install before optimizer creation; outer RVQ config is not rewritten."""
    from .residual_unet import native_config, restore_topology, model_topology_metadata, decoder_identity_config, AUDIO_CODEC_KEY, AUDIO_TIME_KEY
    from dataclasses import asdict
    from core.rewiring.spec import build_spec

    decoder = getattr(model, "decoder", model)
    if position_policy not in ("source_hop_v1", "target_index_v1"):
        raise ValueError("Unsupported ACE audio position policy")
    if getattr(decoder, "_audio_latent_alignment", None) is not None:
        raise ValueError("ACE audio alignment is already installed")
    model_topology_metadata(model)
    decoder_identity_config(model)
    source = alignment.source.latent_channels
    config = native_config(decoder.config)
    if config["audio_acoustic_hidden_dim"] != source or config["in_channels"] != 3*source:
        raise ValueError("ACE native decoder channels differ from the source codec")
    inp, out = decoder.proj_in[1], decoder.proj_out[1]
    if not isinstance(inp, nn.Conv1d) or not isinstance(out, nn.ConvTranspose1d) or inp.groups != 1 or out.groups != 1:
        raise ValueError("ACE audio swap requires unwrapped floating-point decoder IO")
    if inp.in_channels != 3*source or out.out_channels != source:
        raise ValueError("ACE decoder IO and config disagree")
    patch = (config["patch_size"],)
    if (inp.kernel_size != patch or out.kernel_size != patch or inp.stride != patch or
            out.stride != patch or inp.padding != (0,) or out.padding != (0,) or
            inp.dilation != (1,) or out.dilation != (1,) or out.output_padding != (0,) or
            inp.out_channels != config["hidden_size"] or out.in_channels != inp.out_channels):
        raise ValueError("ACE decoder patch geometry differs")
    if not inp.weight.is_floating_point() or not out.weight.is_floating_point():
        raise ValueError("ACE audio swap requires floating-point decoder IO")
    width = alignment.backbone_channels
    target_config = deepcopy(decoder.config)
    target_config.in_channels, target_config.audio_acoustic_hidden_dim = 3*width, width
    setattr(target_config, AUDIO_CODEC_KEY, asdict(alignment.target))
    if position_policy == "source_hop_v1":
        setattr(target_config, AUDIO_TIME_KEY, dict(policy=position_policy, source_hop=alignment.source.hop_length))
    graph = getattr(decoder, "_residual_unet_spec", None)
    if graph is not None:
        graph = build_spec("acestep", [{"name": "main", "block_count": config["num_hidden_layers"]}],
                           deepcopy(graph["regions"]), native_config(target_config))
    if width != source:
        with torch.random.fork_rng(devices=[]):
            new_in = nn.Conv1d(3*width, inp.out_channels, inp.kernel_size, stride=inp.stride,
                padding=inp.padding, dilation=inp.dilation, bias=inp.bias is not None)
            new_out = nn.ConvTranspose1d(out.in_channels, width, out.kernel_size, stride=out.stride,
                padding=out.padding, output_padding=out.output_padding, dilation=out.dilation,
                bias=out.bias is not None)
        new_in, new_out = new_in.to(inp.weight), new_out.to(out.weight)
        with torch.no_grad():
            new_in.weight.zero_()
            for group in range(3):
                new_in.weight[:, group*width:group*width+source].copy_(inp.weight[:, group*source:(group+1)*source])
            if inp.bias is not None:
                new_in.bias.copy_(inp.bias)
            new_out.weight.zero_()
            new_out.weight[:, :source].copy_(out.weight)
            if out.bias is not None:
                new_out.bias.zero_()
                new_out.bias[:source].copy_(out.bias)
        decoder.proj_in[1], decoder.proj_out[1] = new_in, new_out
    decoder.config = target_config
    alignment.position_policy = position_policy
    decoder._audio_latent_alignment = alignment.to(inp.weight)
    if position_policy == "source_hop_v1":
        for layer in decoder.layers:
            if layer.self_attn.sliding_window is not None:
                layer.self_attn.sliding_window = config["sliding_window"] * alignment.source.hop_length // alignment.target.hop_length
    if graph is not None:
        restore_topology(model, graph)
