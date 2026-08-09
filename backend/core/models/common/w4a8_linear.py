"""Packed ConvRot W4A8 Linear backed by Comfy-Kitchen."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def require_w4a8_runtime() -> None:
    """Fail before checkpoint payloads are read when the runtime is unavailable."""
    try:
        from comfy_kitchen.tensor import w4a8_int8_linear  # noqa: F401
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "This checkpoint uses asym_w4a8_int8 weights. Install the backend "
            "requirements (comfy-kitchen==0.2.28 is required) and restart the backend."
        ) from exc


class W4A8Linear(nn.Module):
    """Linear holding packed INT4 codes plus group/channel scale sidecars."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        compute_dtype: torch.dtype,
        *,
        group_size: int = 16,
        convrot_groupsize: int = 256,
        has_codebook: bool = True,
        has_correction: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if in_features % 2 or in_features % group_size:
            raise ValueError(
                f"W4A8 in_features={in_features} must be divisible by 2 and group_size={group_size}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.group_size = group_size
        self.convrot_groupsize = convrot_groupsize
        self.register_buffer(
            "weight", torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            "weight_s_rel",
            torch.empty(
                out_features,
                in_features // group_size,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
        )
        self.register_buffer(
            "weight_s_channel", torch.empty(out_features, dtype=torch.float32, device=device)
        )
        if has_codebook:
            self.register_buffer("weight_codebook", torch.empty(16, dtype=torch.float32, device=device))
        else:
            self.weight_codebook = None
        if has_correction:
            self.register_buffer(
                "weight_correction",
                torch.empty(
                    in_features // group_size,
                    out_features,
                    dtype=compute_dtype,
                    device=device,
                ),
            )
        else:
            self.weight_correction = None
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype, device=device))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            from comfy_kitchen.tensor import (
                dequantize_w4a8_int8_weight,
                w4a8_int8_linear,
            )
        except (ImportError, AttributeError) as exc:  # pragma: no cover - guarded at load
            raise RuntimeError("Comfy-Kitchen W4A8 runtime is unavailable") from exc
        if torch.is_grad_enabled() and x.requires_grad:
            weight = dequantize_w4a8_int8_weight(
                self.weight,
                self.weight_s_rel,
                self.weight_s_channel,
                codebook=self.weight_codebook,
                correction=self.weight_correction,
                group_size=self.group_size,
                convrot_groupsize=self.convrot_groupsize,
                output_dtype=x.dtype,
            )
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, weight, bias)
        return w4a8_int8_linear(
            x,
            self.weight,
            self.weight_s_rel,
            self.weight_s_channel,
            codebook=self.weight_codebook,
            correction=self.weight_correction,
            bias=self.bias,
            group_size=self.group_size,
            convrot_groupsize=self.convrot_groupsize,
            out_dtype=x.dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, packed_w4a8=True"
        )


def swap_linears_to_w4a8(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    layer_configs: dict[str, dict[str, Any]],
    compute_dtype: torch.dtype,
    *,
    prefix: str = "",
) -> int:
    """Replace exactly the Linears declared by mapped W4A8 metadata."""
    swapped = 0
    for name, child in list(module.named_children()):
        child_path = f"{prefix}{name}"
        config = layer_configs.get(child_path)
        if config is not None:
            if not isinstance(child, nn.Linear):
                raise TypeError(
                    f"W4A8 metadata targets '{child_path}', but it is {type(child).__name__}, not nn.Linear"
                )
            weight = state_dict.get(f"{child_path}.weight")
            if weight is None:
                raise ValueError(f"W4A8 layer '{child_path}' has no mapped weight")
            logical_shape = (child.out_features, child.in_features)
            if tuple(weight.shape) != (logical_shape[0], logical_shape[1] // 2):
                raise ValueError(
                    f"W4A8 layer '{child_path}' packed weight has shape {tuple(weight.shape)}, "
                    f"expected {(logical_shape[0], logical_shape[1] // 2)}"
                )
            setattr(
                module,
                name,
                W4A8Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                    group_size=int(config["group_size"]),
                    convrot_groupsize=int(config["convrot_groupsize"]),
                    has_codebook=f"{child_path}.weight_codebook" in state_dict,
                    has_correction=f"{child_path}.weight_correction" in state_dict,
                    device=child.weight.device,
                ),
            )
            swapped += 1
        else:
            swapped += swap_linears_to_w4a8(
                child,
                state_dict,
                layer_configs,
                compute_dtype,
                prefix=f"{child_path}.",
            )
    return swapped


def describe_gemm_path(module: nn.Module) -> str:
    """Opaque generation-metadata label for a loaded packed W4A8 module."""
    return "w4a8_int8(comfy-kitchen)" if any(
        isinstance(child, W4A8Linear) for child in module.modules()
    ) else ""
