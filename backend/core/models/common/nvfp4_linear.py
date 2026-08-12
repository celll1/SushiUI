"""NVFP4/AWQ Linear backed by Comfy-Kitchen -- weight-only, dequant-on-device.

Verified against the co-distributed MiniMax-H3 text encoder
(``qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors``); see
``scratchpad/minimax_h3_te_nvfp4_verification.md`` for the derivation and the
numeric gate this module's output is held to.

THE CONTRACT, exactly (per quantized Linear ``<name>``, logical in_features K):

    <name>.weight             uint8    (out, K/2)   packed E2M1 codes, two per
                                                      byte, ``hi_first=True``
    <name>.weight_scale       float8_e4m3fn (out, K/16)  block scale, in
                                                      comfy-kitchen's own
                                                      swizzled layout (opaque
                                                      here -- ``dequantize_nvfp4``
                                                      un-swizzles internally;
                                                      see the verification note,
                                                      section 3-4, for why a
                                                      Python re-swizzle is not
                                                      attempted in this repo)
    <name>.weight_scale_2     float32  scalar        per-tensor scale
    <name>.pre_quant_scale    bf16     (K,)          ONLY on the two Linears
                                                      per decoder layer whose
                                                      input is not a layernorm
                                                      output (``o_proj``,
                                                      ``down_proj``); AWQ input
                                                      smoothing that must
                                                      multiply the ACTIVATION,
                                                      not the weight

Every other quantized Linear in this file has its AWQ smoothing folded into
the PRECEDING layernorm instead (loaded as-is, unmodified, by the ordinary
``load_state_dict`` path) -- ``pre_quant_scale`` is the only piece that has
nowhere to fold into and is therefore stored, and applied, explicitly.

Dequantization: ``comfy_kitchen.dequantize_nvfp4(weight, weight_scale_2,
weight_scale, output_type, hi_first=True)`` -- run ON THE GPU inside
``forward``, never materialised on the host: the three buffers this module
holds stay exactly what ``safe_open(..., framework="pt")`` mapped from disk,
so a 14.6 GB file never expands into a dense bf16 copy in host RAM.

NOT A SPEED PATH. sm_89 (Ada) has no native FP4 tensor-core GEMM
(``TensorCoreNVFP4Layout.MIN_SM_VERSION = (10, 0)``, i.e. Blackwell); this
module's only claim is footprint on disk/host RAM, never latency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Marker dtype/shape contract this module's buffers are built to. Kept as
# module-level names (not magic literals in the swap function) because the
# loader's own marker validator must agree with these exactly.
NVFP4_WEIGHT_DTYPE = torch.uint8
NVFP4_BLOCK_SCALE_DTYPE = torch.float8_e4m3fn
NVFP4_GLOBAL_SCALE_DTYPE = torch.float32
NVFP4_PRE_QUANT_SCALE_DTYPE = torch.bfloat16
NVFP4_BLOCK_SIZE = 16

# Output dtypes ``comfy_kitchen.DTYPE_TO_CODE`` recognizes. Any activation
# dtype outside this set (e.g. float8 from a future path) falls back to
# bfloat16 -- the file's own storage dtype -- rather than raising a KeyError
# out of a third-party dict deep in the dequant call.
_SUPPORTED_OUTPUT_DTYPES = frozenset({torch.float32, torch.float16, torch.bfloat16})


def require_nvfp4_runtime() -> None:
    """Fail before checkpoint payloads are installed when the runtime is absent."""
    try:
        from comfy_kitchen import dequantize_nvfp4  # noqa: F401

        op = torch.ops.comfy_kitchen.dequantize_nvfp4
        if op is None:  # pragma: no cover
            raise AttributeError("dequantize_nvfp4")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "This checkpoint uses nvfp4/AWQ weights. Install the backend "
            "requirements (comfy-kitchen==0.2.28 is required) and restart the backend."
        ) from exc


class Nvfp4Linear(nn.Module):
    """Linear layer holding packed NVFP4 codes + block/global scale, weight-only.

    ``in_features`` is the LOGICAL (unpacked) reduction dimension; the stored
    ``weight`` buffer is ``(out_features, in_features // 2)``. ``has_pre_quant_scale``
    registers the optional AWQ input-smoothing buffer -- present on exactly
    ``self_attn.o_proj`` and ``mlp.down_proj`` in the verified checkpoint, never
    elsewhere (the loader's marker validator enforces that placement, not this
    class, which only knows whether ITS OWN layer carries one).
    """

    _fixed_quantized_gemm_path = "nvfp4(comfy-kitchen, dequant-only)"

    weight: torch.Tensor
    weight_scale: torch.Tensor
    weight_scale_2: torch.Tensor
    pre_quant_scale: torch.Tensor | None
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        compute_dtype: torch.dtype,
        *,
        has_pre_quant_scale: bool,
        marker_numel: int,
        device: torch.device | str | None = None,
    ) -> None:
        if in_features % NVFP4_BLOCK_SIZE:
            raise ValueError(
                f"NVFP4 requires in_features divisible by the block size "
                f"{NVFP4_BLOCK_SIZE}, got K={in_features}"
            )
        if in_features % 2:
            raise ValueError(f"NVFP4 requires an even in_features, got K={in_features}")
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.compute_dtype = compute_dtype
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features // 2, dtype=NVFP4_WEIGHT_DTYPE, device=device),
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(
                out_features, in_features // NVFP4_BLOCK_SIZE,
                dtype=NVFP4_BLOCK_SCALE_DTYPE, device=device,
            ),
        )
        self.register_buffer(
            "weight_scale_2", torch.empty((), dtype=NVFP4_GLOBAL_SCALE_DTYPE, device=device)
        )
        if has_pre_quant_scale:
            self.register_buffer(
                "pre_quant_scale",
                torch.empty(in_features, dtype=NVFP4_PRE_QUANT_SCALE_DTYPE, device=device),
            )
        else:
            self.pre_quant_scale = None
        if bias:
            self.register_buffer(
                "bias", torch.empty(out_features, dtype=compute_dtype, device=device)
            )
        else:
            self.bias = None
        # Kept as a live module buffer purely as checkpoint provenance (mirrors
        # ``ConvRotInt8Linear.comfy_quant``); never read by ``forward``.
        self.register_buffer(
            "comfy_quant",
            torch.empty(marker_numel, dtype=torch.uint8, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import comfy_kitchen  # noqa: F401 - registers the torch custom op

        if self.pre_quant_scale is not None:
            # AWQ smoothing: the stored weight is W_true / pre_quant_scale, so
            # the activation must be pre-multiplied for the composed function
            # to equal the true (unsmoothed) matmul. See the module docstring
            # and the verified direction in the scratchpad note (section E) --
            # the OPPOSITE direction (dividing the activation) was tested there
            # as a control and is decisively worse.
            x = x * self.pre_quant_scale.to(x.dtype)

        out_dtype = x.dtype if x.dtype in _SUPPORTED_OUTPUT_DTYPES else torch.bfloat16
        weight = comfy_kitchen.dequantize_nvfp4(
            self.weight, self.weight_scale_2, self.weight_scale,
            output_type=out_dtype, hi_first=True,
        )
        weight = weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, nvfp4=weight-only, "
            f"pre_quant_scale={self.pre_quant_scale is not None}"
        )


def swap_linears_to_nvfp4(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    layer_configs: dict[str, dict],
    compute_dtype: torch.dtype,
    *,
    prefix: str = "",
) -> int:
    """Replace exactly the Linears declared by validated NVFP4 markers."""
    swapped = 0
    for name, child in list(module.named_children()):
        child_path = f"{prefix}{name}"
        config = layer_configs.get(child_path)
        if config is not None:
            if not isinstance(child, nn.Linear):
                raise TypeError(
                    f"NVFP4 metadata targets '{child_path}', but it is "
                    f"{type(child).__name__}, not nn.Linear"
                )
            weight = state_dict.get(f"{child_path}.weight")
            scale = state_dict.get(f"{child_path}.weight_scale")
            scale_2 = state_dict.get(f"{child_path}.weight_scale_2")
            marker = state_dict.get(f"{child_path}.comfy_quant")
            if weight is None or scale is None or scale_2 is None or marker is None:
                raise ValueError(
                    f"NVFP4 layer '{child_path}' is missing weight, weight_scale, "
                    f"weight_scale_2 or marker"
                )
            expected_weight = (child.out_features, child.in_features // 2)
            if tuple(weight.shape) != expected_weight or weight.dtype is not NVFP4_WEIGHT_DTYPE:
                raise ValueError(
                    f"NVFP4 layer '{child_path}' weight is {tuple(weight.shape)} "
                    f"{weight.dtype}, expected {expected_weight} {NVFP4_WEIGHT_DTYPE}"
                )
            expected_scale = (child.out_features, child.in_features // NVFP4_BLOCK_SIZE)
            if tuple(scale.shape) != expected_scale or scale.dtype is not NVFP4_BLOCK_SCALE_DTYPE:
                raise ValueError(
                    f"NVFP4 layer '{child_path}' weight_scale is {tuple(scale.shape)} "
                    f"{scale.dtype}, expected {expected_scale} {NVFP4_BLOCK_SCALE_DTYPE}"
                )
            if scale_2.numel() != 1 or scale_2.dtype is not NVFP4_GLOBAL_SCALE_DTYPE:
                raise ValueError(
                    f"NVFP4 layer '{child_path}' weight_scale_2 must be a single "
                    f"{NVFP4_GLOBAL_SCALE_DTYPE} value, got {scale_2.numel()} {scale_2.dtype}"
                )
            has_pre_quant_scale = bool(config.get("has_pre_quant_scale"))
            pqs_key = f"{child_path}.pre_quant_scale"
            pqs = state_dict.get(pqs_key)
            if has_pre_quant_scale:
                if pqs is None or tuple(pqs.shape) != (child.in_features,) \
                        or pqs.dtype is not NVFP4_PRE_QUANT_SCALE_DTYPE:
                    raise ValueError(
                        f"NVFP4 layer '{child_path}' declares pre_quant_scale but it is "
                        f"missing or malformed (expected ({child.in_features},) "
                        f"{NVFP4_PRE_QUANT_SCALE_DTYPE})"
                    )
            elif pqs is not None:
                raise ValueError(
                    f"NVFP4 layer '{child_path}' carries an unexpected '{pqs_key}' -- "
                    f"only self_attn.o_proj/mlp.down_proj are validated to have one"
                )
            setattr(
                module,
                name,
                Nvfp4Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                    has_pre_quant_scale=has_pre_quant_scale,
                    marker_numel=marker.numel(),
                    device=child.weight.device,
                ),
            )
            swapped += 1
        else:
            swapped += swap_linears_to_nvfp4(
                child, state_dict, layer_configs, compute_dtype, prefix=f"{child_path}.",
            )
    return swapped


def describe_gemm_path(module: nn.Module) -> str:
    """Opaque generation-metadata label for a loaded NVFP4 module."""
    return Nvfp4Linear._fixed_quantized_gemm_path if any(
        isinstance(child, Nvfp4Linear) for child in module.modules()
    ) else ""
