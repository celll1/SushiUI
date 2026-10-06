"""ACE decoder boundaries; the frozen conditioning codec remains independent."""

from copy import deepcopy

import torch
from torch import nn

from core.models.components.audio_latent_adapter import calibration_audio_grid


def _uniform_mask(mask, channels):
    if mask.ndim != 3 or mask.shape[-1] < 1 or not torch.equal(mask, mask[..., :1].expand_as(mask)):
        raise ValueError("ACE audio swap requires channel-uniform chunk masks")
    return mask[..., :1].expand(-1, -1, channels)


def native_context_to_target(alignment, context):
    """Convert conditioning only; RVQ and timbre still consume native latents."""
    old, new = alignment.source.latent_channels, alignment.target.latent_channels
    if context.ndim != 3 or context.shape[-1] != 2 * old or min(context.shape) < 1:
        raise ValueError("ACE native context must contain source latents and chunk masks")
    samples = alignment.source.waveform_samples(context.shape[1])
    frames = alignment.target.latent_frames(samples)
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


def install_decoder_alignment(decoder, alignment):
    """Install before optimizer creation; outer RVQ config is not rewritten."""
    from .residual_unet import native_config, restore_topology
    from core.rewiring.spec import build_spec

    if getattr(decoder, "_audio_latent_alignment", None) is not None:
        raise ValueError("ACE audio alignment is already installed")
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
    decoder._audio_latent_alignment = alignment.to(inp.weight)
    if graph is not None:
        restore_topology(decoder, graph)
