"""Weight-only INT8 ``nn.Embedding``, gather-then-scale.

WHY THIS EXISTS. ``core.models.common.int8_linear.swap_linears_to_int8`` walks
``named_children()`` and swaps only ``nn.Linear``; an ``nn.Embedding`` with a
per-row-scaled int8 weight and no rotation (the exact layout the co-distributed
MiniMax-H3 ``qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`` stores at
``model.embed_tokens``, marker ``{"format": "int8_tensorwise"}``, NO
``"convrot"`` key) passes ``quantized_checkpoint_guard``'s declared-semantics
check (a known format with no rotation is treated as an ordinary scaled
tensor) and then falls straight through to ``load_state_dict``, which would
DTYPE-CAST the int8 codes into the module's bf16 ``Parameter`` -- the same
silent corruption ``int8_linear``/``fp8_linear`` exist to prevent for Linears,
just with no Linear-shaped swap to catch it for an Embedding.

DEQUANTIZATION: ``weight.to(dtype) * weight_scale[:, None]``, per-output-row,
identical convention to ``int8_linear.Int8Linear``. The whole point of doing
this at the EMBEDDING (gather) rather than dequantizing the full table once at
load time is that only the rows a forward pass actually indexes are ever
widened to a floating dtype -- the full ``[vocab, hidden]`` int8 table (740 MB
at MiniMax-H3's 151936x5120) stays memory-mapped/int8-resident, and a single
forward touches at most a few hundred rows.
"""

from __future__ import annotations

import torch
import torch.nn as nn


INT8_EMBEDDING_WEIGHT_DTYPE = torch.int8
INT8_EMBEDDING_SCALE_DTYPE = torch.float32
INT8_EMBEDDING_SCALE_SUFFIX = ".weight_scale"


class Int8Embedding(nn.Module):
    """Embedding lookup table with a per-output-row int8 weight + float32 scale.

    Both buffers (not parameters), same reasoning as ``Int8Linear``: they load
    via ``load_state_dict`` and are excluded from optimizer/grad machinery.
    """

    weight: torch.Tensor
    weight_scale: torch.Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        compute_dtype: torch.dtype,
        *,
        marker_numel: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.compute_dtype = compute_dtype
        self.register_buffer(
            "weight",
            torch.empty(
                num_embeddings, embedding_dim,
                dtype=INT8_EMBEDDING_WEIGHT_DTYPE, device=device,
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(num_embeddings, dtype=INT8_EMBEDDING_SCALE_DTYPE, device=device),
        )
        # Checkpoint provenance only (mirrors ConvRotInt8Linear.comfy_quant /
        # Nvfp4Linear.comfy_quant); never read by ``forward``.
        self.register_buffer(
            "comfy_quant",
            torch.empty(marker_numel, dtype=torch.uint8, device=device),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Gather-then-scale: index_select touches only the rows this call
        # needs, not the whole [num_embeddings, embedding_dim] table.
        codes = self.weight[input_ids]
        scale = self.weight_scale[input_ids].unsqueeze(-1)
        return (codes.to(torch.float32) * scale).to(self.compute_dtype)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"int8=per-row"
        )


def swap_embedding_to_int8(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    layer_configs: dict[str, dict],
    compute_dtype: torch.dtype,
    *,
    prefix: str = "",
) -> int:
    """Replace exactly the ``nn.Embedding``\\ s declared by ``layer_configs``.

    Same shape as ``convrot_int8_linear.swap_linears_to_convrot_int8`` /
    ``nvfp4_linear.swap_linears_to_nvfp4``, targeting ``nn.Embedding`` instead
    of ``nn.Linear``. Returns the count.
    """
    swapped = 0
    for name, child in list(module.named_children()):
        child_path = f"{prefix}{name}"
        config = layer_configs.get(child_path)
        if config is not None:
            if not isinstance(child, nn.Embedding):
                raise TypeError(
                    f"INT8 embedding metadata targets '{child_path}', but it is "
                    f"{type(child).__name__}, not nn.Embedding"
                )
            weight = state_dict.get(f"{child_path}.weight")
            scale = state_dict.get(f"{child_path}{INT8_EMBEDDING_SCALE_SUFFIX}")
            marker = state_dict.get(f"{child_path}.comfy_quant")
            if weight is None or scale is None or marker is None:
                raise ValueError(
                    f"INT8 embedding '{child_path}' is missing weight, weight_scale or marker"
                )
            expected = (child.num_embeddings, child.embedding_dim)
            if tuple(weight.shape) != expected or weight.dtype is not INT8_EMBEDDING_WEIGHT_DTYPE:
                raise ValueError(
                    f"INT8 embedding '{child_path}' weight is {tuple(weight.shape)} "
                    f"{weight.dtype}, expected {expected} {INT8_EMBEDDING_WEIGHT_DTYPE}"
                )
            if scale.numel() != child.num_embeddings or scale.dtype is not INT8_EMBEDDING_SCALE_DTYPE:
                raise ValueError(
                    f"INT8 embedding '{child_path}' weight_scale has {scale.numel()} "
                    f"{scale.dtype} value(s), expected {child.num_embeddings} "
                    f"{INT8_EMBEDDING_SCALE_DTYPE}"
                )
            setattr(
                module,
                name,
                Int8Embedding(
                    child.num_embeddings,
                    child.embedding_dim,
                    compute_dtype,
                    marker_numel=marker.numel(),
                    device=child.weight.device,
                ),
            )
            swapped += 1
        else:
            swapped += swap_embedding_to_int8(
                child, state_dict, layer_configs, compute_dtype, prefix=f"{child_path}.",
            )
    return swapped
