"""Optional local latent-grid correction head for SenseNova generation."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


REFINER_VERSION = 1
REFINER_INPUTS = ["x0_head", "z_cin"]
REFINER_NORM = "channel_rms_v1"
REFINER_EPS = 1e-6
REFINER_PREFIX = "fm_modules.fm_refiner."


class ChannelRMSNorm2d(nn.Module):
    """RMS-normalize channels independently at each NCHW spatial position."""

    def __init__(self, channels: int, *, eps: float = REFINER_EPS) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        inv_rms = torch.rsqrt(x_float.square().mean(dim=1, keepdim=True) + self.eps)
        normalized = x_float * inv_rms
        return (normalized * self.weight.float()[None, :, None, None]).to(dtype)


def _sinusoidal_embedding(t: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    exponent = -math.log(10_000.0) * torch.arange(
        half, device=t.device, dtype=torch.float32
    ) / max(half, 1)
    args = t.float()[:, None] * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class LatentRefinerBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1 = ChannelRMSNorm2d(width)
        self.norm2 = ChannelRMSNorm2d(width)
        self.film = nn.Linear(width, 2 * width)
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(emb).chunk(2, dim=1)
        u = self.norm1(h)
        u = u * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        u = self.conv1(torch.nn.functional.silu(u))
        u = self.conv2(torch.nn.functional.silu(self.norm2(u)))
        return h + u


class LatentRefiner(nn.Module):
    """Version-1 SenseNova residual refiner."""

    def __init__(self, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.depth = int(depth)
        if not 16 <= self.width <= 1024 or self.width % 16:
            raise ValueError("latent refiner width must be a multiple of 16 in [16, 1024]")
        if not 1 <= self.depth <= 8:
            raise ValueError("latent refiner depth must be in [1, 8]")

        self.stem = nn.Conv2d(2 * self.in_channels, self.width, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.width, 4 * self.width),
            nn.SiLU(),
            nn.Linear(4 * self.width, self.width),
        )
        self.blocks = nn.ModuleList([LatentRefinerBlock(self.width) for _ in range(self.depth)])
        self.out = nn.Conv2d(self.width, self.in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("gate", torch.tensor(1.0, dtype=torch.float32), persistent=True)

    @property
    def receptive_field_radius(self) -> int:
        return 2 * self.depth + 2

    def forward(
        self,
        x0_head: torch.Tensor,
        z_grid: torch.Tensor,
        t: torch.Tensor,
        noise_scale: float,
        *,
        checkpoint_blocks: bool = False,
    ) -> torch.Tensor:
        if x0_head.ndim != 4 or z_grid.shape != x0_head.shape:
            raise ValueError(
                "latent refiner requires equal BCHW x0_head and z_grid tensors, got "
                f"{tuple(x0_head.shape)} and {tuple(z_grid.shape)}"
            )
        batch = x0_head.shape[0]
        t_flat = torch.as_tensor(t, device=x0_head.device, dtype=torch.float32).reshape(-1)
        if t_flat.numel() == 1:
            t_flat = t_flat.expand(batch)
        if t_flat.numel() != batch:
            raise ValueError(f"latent refiner got {t_flat.numel()} timestep(s) for batch {batch}")
        scale = float(noise_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"latent refiner requires a finite positive noise_scale, got {noise_scale!r}")

        device_type = x0_head.device.type
        param_dtype = self.stem.weight.dtype
        with torch.autocast(device_type=device_type, enabled=False):
            c_in = torch.rsqrt(t_flat.square() + (1 - t_flat).square() * (scale * scale))
            z_cin = z_grid.to(torch.float32) * c_in[:, None, None, None]
            inputs = torch.cat([x0_head.to(torch.float32), z_cin], dim=1).to(param_dtype)
            emb = self.time_mlp(_sinusoidal_embedding(t_flat, self.width).to(param_dtype))
            h = self.stem(inputs)
            for block in self.blocks:
                if checkpoint_blocks and torch.is_grad_enabled():
                    h = checkpoint(block, h, emb, use_reentrant=False)
                else:
                    h = block(h, emb)
            delta = self.out(h)
            return x0_head + self.gate.to(device=x0_head.device, dtype=x0_head.dtype) * delta.to(x0_head.dtype)


def apply_latent_refiner(
    refiner: Optional[LatentRefiner],
    x0_head: torch.Tensor,
    z_grid: torch.Tensor,
    t: torch.Tensor,
    noise_scale: float,
    *,
    checkpoint_blocks: bool = False,
) -> torch.Tensor:
    if refiner is None:
        return x0_head
    return refiner(
        x0_head, z_grid, t, noise_scale, checkpoint_blocks=checkpoint_blocks
    )


def expected_refiner_state_shapes(
    in_channels: int, width: int, depth: int
) -> Dict[str, tuple[int, ...]]:
    c, w, d = int(in_channels), int(width), int(depth)
    shapes: Dict[str, tuple[int, ...]] = {
        "gate": (),
        "stem.weight": (w, 2 * c, 3, 3),
        "stem.bias": (w,),
        "time_mlp.0.weight": (4 * w, w),
        "time_mlp.0.bias": (4 * w,),
        "time_mlp.2.weight": (w, 4 * w),
        "time_mlp.2.bias": (w,),
        "out.weight": (c, w, 3, 3),
        "out.bias": (c,),
    }
    for index in range(d):
        base = f"blocks.{index}."
        shapes.update({
            base + "norm1.weight": (w,),
            base + "norm2.weight": (w,),
            base + "film.weight": (2 * w, w),
            base + "film.bias": (2 * w,),
            base + "conv1.weight": (w, w, 3, 3),
            base + "conv1.bias": (w,),
            base + "conv2.weight": (w, w, 3, 3),
            base + "conv2.bias": (w,),
        })
    return shapes


def validate_gen_refiner_declaration(
    config_dict: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    *,
    checkpoint_step: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Validate declaration and tensor payload before model construction."""
    declaration = config_dict.get("gen_refiner")
    present = {
        key[len(REFINER_PREFIX):]: tensor
        for key, tensor in state_dict.items()
        if key.startswith(REFINER_PREFIX)
    }
    if declaration is None:
        if present:
            raise ValueError("SenseNova refiner tensors are present without gen_refiner declaration")
        return None
    if not isinstance(declaration, Mapping):
        raise ValueError("SenseNova gen_refiner declaration must be an object")
    required = {
        "version", "width", "depth", "inputs", "norm",
        "detach_anchor_step", "detach_steps", "detach_accum",
    }
    if set(declaration) != required:
        raise ValueError(
            "SenseNova gen_refiner declaration fields differ: "
            f"missing={sorted(required - set(declaration))}, "
            f"extra={sorted(set(declaration) - required)}"
        )
    if type(declaration["version"]) is not int or declaration["version"] != REFINER_VERSION:
        raise ValueError(f"Unsupported SenseNova gen_refiner version {declaration['version']!r}")
    if declaration["inputs"] != REFINER_INPUTS:
        raise ValueError(f"SenseNova gen_refiner inputs must be exactly {REFINER_INPUTS!r}")
    if declaration["norm"] != REFINER_NORM:
        raise ValueError(f"SenseNova gen_refiner norm must be {REFINER_NORM!r}")
    width, depth = declaration["width"], declaration["depth"]
    if type(width) is not int or not 16 <= width <= 1024 or width % 16:
        raise ValueError("SenseNova gen_refiner width must be a multiple of 16 in [16, 1024]")
    if type(depth) is not int or not 1 <= depth <= 8:
        raise ValueError("SenseNova gen_refiner depth must be an integer in [1, 8]")

    detach_values = [declaration[name] for name in (
        "detach_anchor_step", "detach_steps", "detach_accum"
    )]
    if any(value is None for value in detach_values) and not all(value is None for value in detach_values):
        raise ValueError("SenseNova gen_refiner detach fields must be all null or all present")
    if all(value is not None for value in detach_values):
        anchor, steps, accum = detach_values
        if any(type(value) is not int for value in detach_values) or anchor < 0 or steps < 1 or accum < 1:
            raise ValueError("SenseNova gen_refiner detach fields are invalid")
        if checkpoint_step is None:
            raise ValueError("Annealing SenseNova gen_refiner requires checkpoint step metadata")

    channels = config_dict.get("gen_in_channels")
    if type(channels) is not int or channels <= 0:
        raise ValueError("SenseNova gen_refiner requires positive integer gen_in_channels")
    expected = expected_refiner_state_shapes(channels, width, depth)
    if set(present) != set(expected):
        raise ValueError(
            "SenseNova gen_refiner tensor set differs: "
            f"missing={sorted(set(expected) - set(present))[:5]}, "
            f"extra={sorted(set(present) - set(expected))[:5]}"
        )
    for name, shape in expected.items():
        tensor = present[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"SenseNova gen_refiner tensor {name} has shape {tuple(tensor.shape)}, expected {shape}"
            )
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"SenseNova gen_refiner tensor {name} contains non-finite values")
    gate = present["gate"]
    if gate.dtype != torch.float32 or not 0.0 <= float(gate.item()) <= 1.0:
        raise ValueError("SenseNova gen_refiner gate must be scalar float32 in [0, 1]")

    if all(value is not None for value in detach_values):
        anchor, steps, accum = detach_values
        step_sched = int(checkpoint_step) // int(accum)
        expected_gate = min(1.0, max(0.0, 1.0 - (step_sched - anchor) / steps))
        if abs(float(gate.item()) - expected_gate) > 1e-6:
            raise ValueError(
                f"SenseNova gen_refiner gate {float(gate.item())} disagrees with detach clock {expected_gate}"
            )
    return dict(declaration)
