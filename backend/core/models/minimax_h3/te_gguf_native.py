"""Map a Qwen3-VL Q8_0 GGUF straight in as MiniMax-H3's text encoder.

``te_gguf_convert`` writes the same weights out as a bf16 safetensors file
first; this module skips that step. ``gguf.GGUFReader`` memory-maps the file
exactly as ``safe_open`` maps a safetensors one, so the Q8_0 codes stay packed
on the CPU (4.28 GB on disk against 5.24 GB converted) and are dequantized per
layer on the GPU inside ``functional_call`` -- the same streaming shape the 32B
int8 encoder already uses.

The name map, the dropped tensors and the output convention are
``te_gguf_convert``'s: one map, used by both paths. The one thing that differs
is the depth. A converted file was truncated at write time and DECLARES its
depth; a raw GGUF carries every block, so the trained projection's ``tap``
chooses how many to map and blocks at or beyond it are never touched.

Only ``Q8_0`` and ``F32`` are implemented. Any other GGML type is refused by
name -- ``te_gguf_convert`` reads those through upstream ``gguf.dequantize``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.minimax_h3.te_gguf_convert import _gguf_shape, map_name, read_text_config

GGUF_SUFFIX = ".gguf"

# ``{ggml_fp16_t d; int8_t qs[32]}``: 34 bytes carry 32 values, ``x = d * qs``.
Q8_0_BLOCK = 32
Q8_0_BLOCK_BYTES = 34

SUPPORTED_GGML_TYPES = ("F32", "Q8_0")

# The GGUF equivalent of a converted file's ``minimax_h3_te`` block. ``tap`` is
# deliberately absent: it is the projection's, not the file's.
_OUTPUT_CONVENTION = "unnormalized_hidden_after_the_projection's_tap"


class GgufTextEncoderError(RuntimeError):
    pass


def is_gguf_path(path: Optional[str]) -> bool:
    return bool(path) and str(path).lower().endswith(GGUF_SUFFIX)


# ---------------------------------------------------------------------------
# Q8_0
# ---------------------------------------------------------------------------

def dequantize_q8_0(packed: torch.Tensor, in_features: int,
                    *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``[..., blocks*34]`` uint8 -> ``[..., in_features]``.

    A row's packed width is not ``in_features``: each 32 values carry their own
    fp16 scale ahead of them, so a 2560-wide row occupies 2720 bytes.
    """
    if packed.dtype is not torch.uint8:
        raise GgufTextEncoderError(f"Q8_0 blocks must be uint8, got {packed.dtype}")
    blocks, remainder = divmod(int(packed.shape[-1]), Q8_0_BLOCK_BYTES)
    if remainder or blocks * Q8_0_BLOCK != int(in_features):
        raise GgufTextEncoderError(
            f"{packed.shape[-1]} packed byte(s) per row is not {in_features} Q8_0 values "
            f"({in_features // Q8_0_BLOCK} block(s) of {Q8_0_BLOCK_BYTES} bytes)")
    view = packed.reshape(*packed.shape[:-1], blocks, Q8_0_BLOCK_BYTES)
    scales = view[..., :2].contiguous().view(torch.float16)
    codes = view[..., 2:].contiguous().view(torch.int8)
    values = codes.to(dtype) * scales.to(dtype)
    return values.reshape(*packed.shape[:-1], in_features)


def q8_0_row_bytes(in_features: int) -> int:
    blocks, remainder = divmod(int(in_features), Q8_0_BLOCK)
    if remainder:
        raise GgufTextEncoderError(
            f"Q8_0 stores {Q8_0_BLOCK} values per block; {in_features} is not a multiple")
    return blocks * Q8_0_BLOCK_BYTES


class GgufQ8Linear(nn.Module):
    """Linear holding its weight as packed Q8_0 blocks, dequantized in forward.

    Same shape as ``ConvRotInt8Linear``/``Nvfp4Linear``: the packed bytes are a
    buffer, so ``_gpu_module_params`` streams them to the GPU untouched (uint8
    is outside its widen set) and the CPU side stays attached to the mmap.
    """

    q_packed: torch.Tensor

    def __init__(self, in_features: int, out_features: int,
                 *, device: torch.device | str | None = None) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer(
            "q_packed",
            torch.empty(self.out_features, q8_0_row_bytes(self.in_features),
                        dtype=torch.uint8, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_q8_0(self.q_packed, self.in_features, dtype=torch.float32)
        return F.linear(x, weight.to(x.dtype))

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias=False, gguf=Q8_0")


class GgufQ8Embedding(nn.Module):
    """Embedding whose table stays packed; only the indexed rows are dequantized."""

    q_packed: torch.Tensor

    def __init__(self, num_embeddings: int, embedding_dim: int, compute_dtype: torch.dtype,
                 *, device: torch.device | str | None = None) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.compute_dtype = compute_dtype
        self.register_buffer(
            "q_packed",
            torch.empty(self.num_embeddings, q8_0_row_bytes(self.embedding_dim),
                        dtype=torch.uint8, device=device),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows = self.q_packed[input_ids]
        return dequantize_q8_0(rows, self.embedding_dim).to(self.compute_dtype)

    def extra_repr(self) -> str:
        return (f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
                f"gguf=Q8_0")


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------

def _as_torch(array) -> torch.Tensor:
    # The mmap is opened read-only, which torch only warns about; nothing here
    # writes, and copying is the whole cost this path exists to avoid.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(array)


def open_gguf(path: str):
    from gguf import GGUFReader

    return GGUFReader(str(path))


def read_gguf_te_declaration(path: str) -> Dict[str, Any]:
    """The ``minimax_h3_te``-shaped declaration a raw GGUF supports.

    Carries every dim ``_te_declared_dims`` requires plus ``block_count``, and
    deliberately no ``num_hidden_layers``: the depth is the projection's.
    """
    reader = open_gguf(path)
    try:
        cfg = read_text_config(reader)
        name = reader.fields.get("general.name")
        size = reader.fields.get("general.size_label")
        types = sorted({tensor.tensor_type.name for tensor in reader.tensors})
    finally:
        del reader
    return {
        **cfg.as_metadata(),
        "block_count": cfg.block_count,
        "modalities": "text",
        "output": _OUTPUT_CONVENTION,
        "source_gguf": str(path).replace("\\", "/").rsplit("/", 1)[-1],
        "source_name": str(name.contents()) if name is not None else None,
        "source_size_label": str(size.contents()) if size is not None else None,
        "ggml_types": types,
    }


def _refuse_unsupported_types(unsupported: Dict[str, List[str]], path: str) -> None:
    if not unsupported:
        return
    detail = "; ".join(
        f"{ggml_type} ({len(names)} tensor(s), e.g. {', '.join(sorted(names)[:3])})"
        for ggml_type, names in sorted(unsupported.items())
    )
    raise GgufTextEncoderError(
        f"{path} carries GGML type(s) this loader does not implement: {detail}. Native GGUF "
        f"loading implements {', '.join(SUPPORTED_GGML_TYPES)} only. Convert the file first "
        f"with core.models.minimax_h3.te_gguf_convert, which dequantizes every type through "
        f"upstream gguf.dequantize.")


def plan_gguf_text_encoder(reader, tap: int, path: str, rewrite) -> Dict[str, Any]:
    """``{state_dict, linear_configs, embedding_configs, config, mapped, skipped}``.

    ``rewrite`` is the loader's ``_rewrite_te_key``, so the module paths here
    and the ones the converted file loads into are produced by one function.
    Tensors of blocks at or beyond ``tap`` are never mapped, so their bytes are
    never touched.
    """
    cfg = read_text_config(reader)
    if tap < 1 or tap > cfg.block_count:
        raise GgufTextEncoderError(
            f"the trained projection asks for tap={tap} but {path} carries {cfg.block_count} "
            f"block(s); the projection is fitted to one specific layer's hidden state and this "
            f"file cannot produce it.")

    state_dict: Dict[str, torch.Tensor] = {}
    linear_configs: Dict[str, Dict[str, int]] = {}
    embedding_configs: Dict[str, Dict[str, int]] = {}
    unsupported: Dict[str, List[str]] = {}
    skipped = 0

    for tensor in reader.tensors:
        target = map_name(tensor.name, tap)
        if target is None:
            skipped += 1
            continue
        ggml_type = tensor.tensor_type.name
        if ggml_type not in SUPPORTED_GGML_TYPES:
            unsupported.setdefault(ggml_type, []).append(tensor.name)
            continue
        shape = _gguf_shape(tensor)
        module_path = rewrite(target)
        if ggml_type == "F32":
            value = _as_torch(tensor.data)
            if tuple(value.shape) != shape:
                value = value.reshape(shape)
            state_dict[module_path] = value
            continue
        if len(shape) != 2:
            raise GgufTextEncoderError(
                f"{tensor.name} is Q8_0 with shape {shape}; only 2-D Q8_0 weights are mapped")
        out_features, in_features = shape
        packed = _as_torch(tensor.data)
        expected = (out_features, q8_0_row_bytes(in_features))
        if tuple(packed.shape) != expected:
            raise GgufTextEncoderError(
                f"{tensor.name} packs as {tuple(packed.shape)}, expected {expected} for a "
                f"{out_features}x{in_features} Q8_0 weight")
        stem = module_path[: -len(".weight")]
        state_dict[stem + ".q_packed"] = packed
        if target == "model.embed_tokens.weight":
            embedding_configs[stem] = {"num_embeddings": out_features,
                                       "embedding_dim": in_features}
        else:
            linear_configs[stem] = {"in_features": in_features, "out_features": out_features}

    _refuse_unsupported_types(unsupported, path)
    return {
        "state_dict": state_dict,
        "linear_configs": linear_configs,
        "embedding_configs": embedding_configs,
        "config": cfg,
        "mapped": len(state_dict),
        "skipped": skipped,
    }


def swap_modules_to_gguf_q8(
    module: nn.Module,
    linear_configs: Dict[str, Dict[str, int]],
    embedding_configs: Dict[str, Dict[str, int]],
    compute_dtype: torch.dtype,
    *,
    prefix: str = "",
) -> Tuple[int, int]:
    """Replace exactly the declared ``nn.Linear``/``nn.Embedding`` modules.

    Same shape as ``swap_linears_to_convrot_int8``; returns
    ``(linears, embeddings)``.
    """
    linears = embeddings = 0
    for name, child in list(module.named_children()):
        child_path = f"{prefix}{name}"
        linear = linear_configs.get(child_path)
        embedding = embedding_configs.get(child_path)
        if linear is not None:
            if not isinstance(child, nn.Linear):
                raise GgufTextEncoderError(
                    f"GGUF Q8_0 weight targets '{child_path}', but it is "
                    f"{type(child).__name__}, not nn.Linear")
            if child.bias is not None:
                raise GgufTextEncoderError(
                    f"'{child_path}' has a bias; the GGUF carries none for it")
            if (child.in_features, child.out_features) != (linear["in_features"],
                                                           linear["out_features"]):
                raise GgufTextEncoderError(
                    f"'{child_path}' is {child.in_features}->{child.out_features} but the GGUF "
                    f"tensor is {linear['in_features']}->{linear['out_features']}")
            setattr(module, name, GgufQ8Linear(
                linear["in_features"], linear["out_features"], device=child.weight.device))
            linears += 1
        elif embedding is not None:
            if not isinstance(child, nn.Embedding):
                raise GgufTextEncoderError(
                    f"GGUF Q8_0 embedding targets '{child_path}', but it is "
                    f"{type(child).__name__}, not nn.Embedding")
            if (child.num_embeddings, child.embedding_dim) != (embedding["num_embeddings"],
                                                               embedding["embedding_dim"]):
                raise GgufTextEncoderError(
                    f"'{child_path}' is {child.num_embeddings}x{child.embedding_dim} but the "
                    f"GGUF tensor is {embedding['num_embeddings']}x{embedding['embedding_dim']}")
            setattr(module, name, GgufQ8Embedding(
                embedding["num_embeddings"], embedding["embedding_dim"], compute_dtype,
                device=child.weight.device))
            embeddings += 1
        else:
            child_linears, child_embeddings = swap_modules_to_gguf_q8(
                child, linear_configs, embedding_configs, compute_dtype,
                prefix=f"{child_path}.")
            linears += child_linears
            embeddings += child_embeddings
    return linears, embeddings
