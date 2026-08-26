"""Hadamard ConvRot INT8 Linear backed by Comfy-Kitchen."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from core.models.ideogram4.vendor.int8_linear import Int8Linear


_DTYPE_CODES = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


def require_convrot_int8_runtime() -> None:
    """Fail before checkpoint payloads are installed when the runtime is absent."""
    try:
        from comfy_kitchen import int8_linear  # noqa: F401

        op = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype
        if op is None:  # pragma: no cover
            raise AttributeError("dequantize_int8_convrot_weight_dtype")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "This checkpoint uses int8_tensorwise ConvRot weights. Install the backend "
            "requirements (comfy-kitchen==0.2.28 is required) and restart the backend."
        ) from exc


class ConvRotInt8Linear(Int8Linear):
    """INT8 Linear whose stored rows quantize ``W @ H.T`` in 256-wide groups."""

    _fixed_quantized_gemm_path = "convrot_int8(comfy-kitchen)"

    # Debug-only ablation override (set per-instance, e.g. by sensenova/loader.py's
    # SUSHI_SENSENOVA_CONVROT_DEQUANT). Class default False, so ordinary loads
    # -- H3 included, which shares this class -- are unaffected.
    _force_dequant: bool = False

    # CANDIDATE frozen-base training path (opt-in, default off). Class defaults
    # so ordinary loads cost nothing per instance and never touch state_dict;
    # set per instance only by
    # ``quantized_frozen_training.enable_frozen_training_fused``. Deliberately
    # NOT `_force_dequant`/`_allow_int8_mm`/grad mode reused.
    _frozen_training_fused: bool = False
    _frozen_training_path: str = ""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        compute_dtype: torch.dtype,
        *,
        convrot_groupsize: int,
        marker_numel: int,
        device: torch.device | str | None = None,
    ) -> None:
        if convrot_groupsize != 256 or in_features % convrot_groupsize:
            raise ValueError(
                f"ConvRot INT8 requires K divisible by groupsize 256, got K={in_features}, "
                f"groupsize={convrot_groupsize}"
            )
        super().__init__(
            in_features,
            out_features,
            bias,
            compute_dtype,
            device=device,
        )
        self.convrot_groupsize = int(convrot_groupsize)
        self.register_buffer(
            "comfy_quant",
            torch.empty(marker_numel, dtype=torch.uint8, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._force_dequant or (torch.is_grad_enabled() and x.requires_grad):
            if self._frozen_training_fused and not self._force_dequant:
                from core.models.common.quantized_frozen_training import (
                    maybe_frozen_fused_forward,
                )

                out = maybe_frozen_fused_forward(self, x)
                if out is not None:
                    return out
            return self._dequant_forward(x)
        from comfy_kitchen import int8_linear

        return int8_linear(
            x,
            self.weight,
            self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
            convrot=True,
            convrot_groupsize=self.convrot_groupsize,
        )

    def _dequant_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recover the quantized weight's original basis for an autograd-visible matmul."""
        import comfy_kitchen  # noqa: F401 - registers the torch custom op

        dtype_code = _DTYPE_CODES.get(x.dtype)
        if dtype_code is None:
            raise ValueError(f"ConvRot INT8 does not support activation dtype {x.dtype}")
        weight = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
            self.weight,
            self.weight_scale.reshape(-1, 1),
            self.convrot_groupsize,
            dtype_code,
        )
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, int8=convrot, "
            f"groupsize={self.convrot_groupsize}"
        )


def swap_linears_to_convrot_int8(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    layer_configs: dict[str, dict[str, int]],
    compute_dtype: torch.dtype,
    *,
    prefix: str = "",
) -> int:
    """Replace exactly the Linears declared by validated ConvRot markers."""
    swapped = 0
    for name, child in list(module.named_children()):
        child_path = f"{prefix}{name}"
        config = layer_configs.get(child_path)
        if config is not None:
            if not isinstance(child, torch.nn.Linear):
                raise TypeError(
                    f"ConvRot metadata targets '{child_path}', but it is "
                    f"{type(child).__name__}, not nn.Linear"
                )
            weight = state_dict.get(f"{child_path}.weight")
            scale = state_dict.get(f"{child_path}.weight_scale")
            marker = state_dict.get(f"{child_path}.comfy_quant")
            if weight is None or scale is None or marker is None:
                raise ValueError(
                    f"ConvRot INT8 layer '{child_path}' is missing weight, scale or marker"
                )
            expected = (child.out_features, child.in_features)
            if tuple(weight.shape) != expected or weight.dtype is not torch.int8:
                raise ValueError(
                    f"ConvRot INT8 layer '{child_path}' weight is {tuple(weight.shape)} "
                    f"{weight.dtype}, expected {expected} torch.int8"
                )
            if scale.numel() != child.out_features or scale.dtype is not torch.float32:
                raise ValueError(
                    f"ConvRot INT8 layer '{child_path}' scale has {scale.numel()} "
                    f"{scale.dtype} value(s), expected {child.out_features} float32"
                )
            setattr(
                module,
                name,
                ConvRotInt8Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                    convrot_groupsize=int(config["convrot_groupsize"]),
                    marker_numel=marker.numel(),
                    device=child.weight.device,
                ),
            )
            swapped += 1
        else:
            swapped += swap_linears_to_convrot_int8(
                child,
                state_dict,
                layer_configs,
                compute_dtype,
                prefix=f"{child_path}.",
            )
    return swapped


def describe_gemm_path(module: torch.nn.Module) -> str:
    """Opaque generation-metadata label for a loaded ConvRot INT8 module.

    Suffixed ``,dequant`` when any matching child has ``_force_dequant`` set
    (the SUSHI_SENSENOVA_CONVROT_DEQUANT ablation): the fused and dequant
    arms are numerically different, so the label must distinguish them, same
    as the rest of ``extract_fp8_gemm_info``'s vocabulary. Unaffected when
    nothing sets the flag (H3, which shares this class, included).
    """
    children = [child for child in module.modules() if isinstance(child, ConvRotInt8Linear)]
    if not children:
        return ""
    if any(getattr(child, "_force_dequant", False) for child in children):
        return f"{ConvRotInt8Linear._fixed_quantized_gemm_path},dequant"
    return ConvRotInt8Linear._fixed_quantized_gemm_path
