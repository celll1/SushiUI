"""Construction and exact donor-parity checks for the Chimera U-Net."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import torch


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
) -> tuple[torch.nn.Module, ChimeraUNetBuildReport]:
    """Build an exact-shape U-Net from a donor and initialize it explicitly.

    ``scratch`` retains normal diffusers/PyTorch initialization and zeros only
    ``conv_out``. ``sdxl_transplant`` strict-loads every donor tensor.
    """
    if initialization not in {"scratch", "sdxl_transplant"}:
        raise ValueError(
            f"unknown Chimera U-Net initialization {initialization!r}; "
            "expected 'scratch' or 'sdxl_transplant'"
        )
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
    else:
        conv_out = getattr(built, "conv_out", None)
        if conv_out is None:
            raise ValueError("SDXL donor U-Net has no conv_out to zero-initialize")
        with torch.no_grad():
            conv_out.weight.zero_()
            if conv_out.bias is not None:
                conv_out.bias.zero_()

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
