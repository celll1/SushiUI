"""Construction and exact donor-parity checks for the Chimera U-Net."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


def parameter_census(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(parameter.shape) for name, parameter in module.named_parameters()}


def trainable_parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _all_parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _config_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    if hasattr(config, "to_dict"):
        return copy.deepcopy(dict(config.to_dict()))
    return copy.deepcopy(dict(config))


def _attention_head_dims(unet: torch.nn.Module) -> tuple[int, ...]:
    widths = set()
    for module in unet.modules():
        heads = getattr(module, "heads", None)
        to_q = getattr(module, "to_q", None)
        if heads and to_q is not None and hasattr(to_q, "out_features"):
            inner = int(to_q.out_features)
            if inner % int(heads):
                raise ValueError(f"attention inner width {inner} is not divisible by {heads} heads")
            widths.add(inner // int(heads))
    if not widths:
        raise ValueError("donor U-Net exposes no attention modules")
    return tuple(sorted(widths))


@dataclass(frozen=True)
class ChimeraUNetBuildReport:
    initialization: str
    parameter_count: int
    parameter_shapes: dict[str, tuple[int, ...]]
    attention_head_dims: tuple[int, ...]


def build_donor_equal_unet(
    donor_unet: torch.nn.Module,
    *,
    initialization: str = "scratch",
    seed: int = 0,
    output_initialization: str = "zero",
) -> tuple[torch.nn.Module, ChimeraUNetBuildReport]:
    """Build an exact-shape U-Net from a donor and initialize it explicitly.

    ``scratch`` retains normal diffusers/PyTorch initialization.
    ``sdxl_transplant`` strict-loads every donor tensor. The output convolution
    is either zeroed or reset to its default initialization explicitly.
    """
    if initialization not in {"scratch", "sdxl_transplant"}:
        raise ValueError(
            f"unknown Chimera U-Net initialization {initialization!r}; "
            "expected 'scratch' or 'sdxl_transplant'"
        )
    if output_initialization not in {"zero", "default"}:
        raise ValueError("output_initialization must be 'zero' or 'default'")
    from diffusers import UNet2DConditionModel

    config = _config_dict(donor_unet.config)
    if int(config.get("in_channels", 0)) != 4 or int(config.get("out_channels", 0)) != 4:
        raise ValueError(
            "Chimera format v2 requires a four-channel SDXL donor U-Net, got "
            f"in={config.get('in_channels')} out={config.get('out_channels')}"
        )
    cross_dim = config.get("cross_attention_dim")
    cross_values = cross_dim if isinstance(cross_dim, (list, tuple)) else (cross_dim,)
    if any(int(value) != 2048 for value in cross_values if value is not None):
        raise ValueError(
            f"Chimera format v2 requires SDXL cross_attention_dim=2048, got {cross_dim!r}"
        )

    devices = [] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        built = UNet2DConditionModel.from_config(config)
    if initialization == "sdxl_transplant":
        built.load_state_dict(donor_unet.state_dict(), strict=True)
    conv_out = getattr(built, "conv_out", None)
    if conv_out is None:
        raise ValueError("SDXL donor U-Net has no conv_out")
    if output_initialization == "zero":
        with torch.no_grad():
            conv_out.weight.zero_()
            if conv_out.bias is not None:
                conv_out.bias.zero_()
    elif initialization == "sdxl_transplant":
        conv_out.reset_parameters()

    donor_census = parameter_census(donor_unet)
    built_census = parameter_census(built)
    if built_census != donor_census:
        missing = sorted(set(donor_census) - set(built_census))
        extra = sorted(set(built_census) - set(donor_census))
        mismatched = sorted(
            name for name in set(donor_census) & set(built_census)
            if donor_census[name] != built_census[name]
        )
        raise ValueError(
            "Chimera U-Net parameter census differs from donor: "
            f"missing={missing[:5]}, extra={extra[:5]}, shape_mismatch={mismatched[:5]}"
        )
    donor_count = _all_parameter_count(donor_unet)
    built_count = _all_parameter_count(built)
    if built_count != donor_count:
        raise ValueError(
            f"Chimera U-Net parameter count {built_count} differs from donor {donor_count}"
        )
    head_dims = _attention_head_dims(built)
    illegal = [width for width in head_dims if width % 4]
    if illegal:
        raise ValueError(
            f"SenseNova three-axis RoPE requires head widths divisible by 4, got {illegal}"
        )
    return built, ChimeraUNetBuildReport(
        initialization=initialization,
        parameter_count=built_count,
        parameter_shapes=built_census,
        attention_head_dims=head_dims,
    )


class ChimeraPolarRadialHead(nn.Module):
    """Predict one radial speed from the conditioned U-Net mid-block."""

    VERSION = 1

    def __init__(self, mid_channels: int):
        super().__init__()
        width = int(mid_channels)
        if width <= 0:
            raise ValueError("mid_channels must be positive")
        self.mid_channels = width
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width + 2, 1)

    def forward(
        self,
        mid_block: torch.Tensor,
        timestep: torch.Tensor,
        log_radius: torch.Tensor,
    ) -> torch.Tensor:
        if mid_block.ndim < 3 or mid_block.shape[1] != self.mid_channels:
            raise ValueError(
                f"expected mid-block [B,{self.mid_channels},...], got {tuple(mid_block.shape)}"
            )
        batch = mid_block.shape[0]
        if timestep.ndim == 0:
            timestep = timestep.expand(batch)
        if log_radius.ndim == 0:
            log_radius = log_radius.expand(batch)
        if timestep.shape != (batch,) or log_radius.shape != (batch,):
            raise ValueError("timestep and log_radius must be scalar or [B]")
        parameter = self.projection.weight
        pooled = mid_block.float().flatten(2).mean(dim=2).to(parameter)
        pooled = self.norm(pooled)
        scalars = torch.stack((timestep, log_radius), dim=1).to(parameter)
        return self.projection(torch.cat((pooled, scalars), dim=1)).flatten().float()


def install_polar_radial_head(
    unet: torch.nn.Module,
    *,
    mid_channels: int | None = None,
) -> ChimeraPolarRadialHead:
    existing = getattr(unet, "chimera_polar_radial_head", None)
    if existing is not None:
        if not isinstance(existing, ChimeraPolarRadialHead):
            raise ValueError("U-Net carries an incompatible chimera_polar_radial_head")
        if mid_channels is not None and existing.mid_channels != int(mid_channels):
            raise ValueError("existing Chimera radial head width differs from config")
        return existing
    if mid_channels is None:
        block_channels = tuple(getattr(unet.config, "block_out_channels", ()))
        if not block_channels:
            raise ValueError("U-Net config has no block_out_channels for radial head")
        mid_channels = int(block_channels[-1])
    head = ChimeraPolarRadialHead(int(mid_channels))
    unet.add_module("chimera_polar_radial_head", head)
    return head


def polar_unet_forward(
    unet: torch.nn.Module,
    spatial_input: torch.Tensor,
    timestep: torch.Tensor,
    *,
    log_radius: torch.Tensor,
    **unet_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw tangent and radial predictions from one U-Net evaluation."""
    head = getattr(unet, "chimera_polar_radial_head", None)
    if not isinstance(head, ChimeraPolarRadialHead):
        raise ValueError("polar U-Net forward requires an installed radial head")
    captured: dict[str, torch.Tensor] = {}

    def capture_mid_block(_module, _inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(value, torch.Tensor):
            raise TypeError("U-Net mid-block did not return a tensor")
        captured["value"] = value

    handle = unet.mid_block.register_forward_hook(capture_mid_block)
    try:
        output = unet(spatial_input, timestep, **unet_kwargs)
    finally:
        handle.remove()
    if "value" not in captured:
        raise RuntimeError("U-Net mid-block hook produced no activation")
    tangent = output[0] if isinstance(output, (tuple, list)) else output.sample
    radial = head(captured["value"], timestep, log_radius)
    return tangent, radial
