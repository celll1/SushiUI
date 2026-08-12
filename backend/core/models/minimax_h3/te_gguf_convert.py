"""Convert a Qwen3-VL GGUF into a truncated bf16 text encoder for MiniMax-H3.

MiniMax-H3 ships a Qwen3-VL-32B text encoder truncated to its first 50 decoder
layers; ``loader.py::_build_text_encoder`` reads exactly that shape. This
converter produces the same file shape from a much smaller Qwen3-VL GGUF, so a
4B/8B encoder can stand in for the 32B one:

* flat HF naming (``model.layers.N....``), which ``_rewrite_te_key`` maps onto
  the live ``Qwen3VLForConditionalGeneration`` module paths unchanged;
* blocks ``0..tap-1`` only;
* **no** ``lm_head.weight`` and **no** final ``model.norm.weight`` -- the two
  keys in ``loader._TE_EXPECTED_MISSING``. The declared output is the
  unnormalised hidden state after the last kept layer, so the GGUF's
  ``output_norm.weight``/``output.weight`` are dropped rather than mapped.

Every dimension comes from the GGUF's own KV metadata, so the 4B and the 8B go
through the same code. Nothing is hardcoded to one of them.

The source GGUFs have no vision tower at all (no ``v.*``/``mmproj``/deepstack
tensors), so the result is text-only.

Run:
    venv/Scripts/python.exe -m core.models.minimax_h3.te_gguf_convert \
        <input.gguf> <output.safetensors> --tap 24            # from backend/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

CONVERTER = "minimax_h3_te_gguf_convert"
CONVERTER_VERSION = "1"

# Same wording as the shipped 32B files' own `minimax_h3_te` metadata, which
# reads `unnormalized_hidden_after_layer_50`.
_OUTPUT_CONVENTION = "unnormalized_hidden_after_layer_{tap}"

_DTYPES: Dict[str, Tuple[torch.dtype, str]] = {
    "bf16": (torch.bfloat16, "BF16"),
    "fp32": (torch.float32, "F32"),
}

# Per-block GGUF suffix -> HF suffix. Validated at 277/277 keys with zero
# unmapped names in either direction and zero transposes needed.
_BLOCK_SUFFIXES: Dict[str, str] = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

_TOP_LEVEL_MAP: Dict[str, str] = {"token_embd.weight": "model.embed_tokens.weight"}

# Dropped by the output convention, not unmapped: `output_norm` is the final
# norm the file deliberately omits and `output` is the lm_head (present in the
# 8B, absent from the tied-embedding 4B).
_DROPPED: Dict[str, str] = {
    "output_norm.weight": "final norm (output is the unnormalised hidden state)",
    "output.weight": "lm_head (never run by the text encoder)",
}

# fp32 bytes per dequantized chunk. Q8_0 expands ~3.8x, so this bounds the
# transient at ~256 MB regardless of row width; the whole point of streaming
# is that peak RSS never scales with the model.
_CHUNK_BYTES = 256 << 20


class ConversionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# GGUF metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    mrope_section: Tuple[int, ...]
    vocab_size: int
    block_count: int

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "mrope_section": list(self.mrope_section),
            "vocab_size": self.vocab_size,
        }


def _kv(fields: Dict[str, Any], key: str) -> Any:
    field = fields.get(key)
    if field is None:
        raise ConversionError(f"GGUF metadata has no {key!r}")
    return field.contents()


def _gguf_shape(tensor) -> Tuple[int, ...]:
    """``[out, in]``: GGUF stores dimensions fastest-varying first."""
    return tuple(int(v) for v in reversed(tensor.shape))


def read_text_config(reader) -> TextConfig:
    """Every dimension from the GGUF's own KV, nothing assumed."""
    arch = str(_kv(reader.fields, "general.architecture"))
    if arch != "qwen3vl":
        raise ConversionError(f"expected a qwen3vl GGUF, got architecture {arch!r}")

    head_dim = int(_kv(reader.fields, f"{arch}.attention.key_length"))
    value_length = int(_kv(reader.fields, f"{arch}.attention.value_length"))
    if value_length != head_dim:
        raise ConversionError(
            f"key_length {head_dim} != value_length {value_length}; this converter "
            f"emits one head_dim")

    # llama.cpp pads `rope.dimension_sections` to 4 entries with a trailing 0;
    # transformers' `mrope_section` is the unpadded prefix.
    sections = [int(v) for v in _kv(reader.fields, f"{arch}.rope.dimension_sections")]
    while sections and sections[-1] == 0:
        sections.pop()
    if sum(sections) * 2 != head_dim:
        raise ConversionError(
            f"mrope_section {sections} does not sum to head_dim/2 ({head_dim // 2})")

    embed = next((t for t in reader.tensors if t.name == "token_embd.weight"), None)
    if embed is None:
        raise ConversionError("GGUF has no token_embd.weight")
    vocab_size, hidden_from_embed = _gguf_shape(embed)

    hidden_size = int(_kv(reader.fields, f"{arch}.embedding_length"))
    if hidden_from_embed != hidden_size:
        raise ConversionError(
            f"token_embd is {hidden_from_embed}-wide but embedding_length is {hidden_size}")

    return TextConfig(
        hidden_size=hidden_size,
        num_attention_heads=int(_kv(reader.fields, f"{arch}.attention.head_count")),
        num_key_value_heads=int(_kv(reader.fields, f"{arch}.attention.head_count_kv")),
        head_dim=head_dim,
        intermediate_size=int(_kv(reader.fields, f"{arch}.feed_forward_length")),
        rms_norm_eps=float(_kv(reader.fields, f"{arch}.attention.layer_norm_rms_epsilon")),
        rope_theta=float(_kv(reader.fields, f"{arch}.rope.freq_base")),
        mrope_section=tuple(sections),
        vocab_size=vocab_size,
        block_count=int(_kv(reader.fields, f"{arch}.block_count")),
    )


def _provenance(reader, cfg: TextConfig, gguf_path: Path, tap: int) -> Dict[str, Any]:
    def optional(key: str) -> Optional[str]:
        field = reader.fields.get(key)
        return str(field.contents()) if field is not None else None

    return {
        "converter": CONVERTER,
        "converter_version": CONVERTER_VERSION,
        "tap": tap,
        "source_gguf": gguf_path.name,
        "source_gguf_sha256": sha256_file(gguf_path),
        "source_arch": optional("general.architecture"),
        "source_name": optional("general.name"),
        "source_basename": optional("general.basename"),
        "source_finetune": optional("general.finetune"),
        "source_size_label": optional("general.size_label"),
        "source_block_count": cfg.block_count,
    }


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Name map / expected geometry
# ---------------------------------------------------------------------------

def expected_shapes(cfg: TextConfig, tap: int) -> Dict[str, Tuple[int, ...]]:
    """The output's full key -> shape contract, derived from the dims alone.

    Independent of what the GGUF tensors actually look like, so comparing the
    two is a real cross-check rather than a tautology.
    """
    q_out = cfg.num_attention_heads * cfg.head_dim
    kv_out = cfg.num_key_value_heads * cfg.head_dim
    shapes: Dict[str, Tuple[int, ...]] = {
        "model.embed_tokens.weight": (cfg.vocab_size, cfg.hidden_size),
    }
    per_layer = {
        "input_layernorm.weight": (cfg.hidden_size,),
        "post_attention_layernorm.weight": (cfg.hidden_size,),
        "self_attn.q_proj.weight": (q_out, cfg.hidden_size),
        "self_attn.k_proj.weight": (kv_out, cfg.hidden_size),
        "self_attn.v_proj.weight": (kv_out, cfg.hidden_size),
        "self_attn.o_proj.weight": (cfg.hidden_size, q_out),
        "self_attn.q_norm.weight": (cfg.head_dim,),
        "self_attn.k_norm.weight": (cfg.head_dim,),
        "mlp.gate_proj.weight": (cfg.intermediate_size, cfg.hidden_size),
        "mlp.up_proj.weight": (cfg.intermediate_size, cfg.hidden_size),
        "mlp.down_proj.weight": (cfg.hidden_size, cfg.intermediate_size),
    }
    for layer in range(tap):
        for suffix, shape in per_layer.items():
            shapes[f"model.layers.{layer}.{suffix}"] = shape
    return shapes


def map_name(gguf_name: str, tap: int) -> Optional[str]:
    """HF name, or ``None`` when the tensor is deliberately not emitted.

    Raises for a name with no rule at all -- a silently skipped tensor is a
    silently broken encoder.
    """
    if gguf_name in _DROPPED:
        return None
    if gguf_name in _TOP_LEVEL_MAP:
        return _TOP_LEVEL_MAP[gguf_name]
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".", 2)
        if len(parts) == 3 and parts[1].isdigit():
            suffix = _BLOCK_SUFFIXES.get(parts[2])
            if suffix is not None:
                layer = int(parts[1])
                return None if layer >= tap else f"model.layers.{layer}.{suffix}"
    raise ConversionError(
        f"no mapping rule for GGUF tensor {gguf_name!r}; refusing to write a text "
        f"encoder that silently drops it")


# ---------------------------------------------------------------------------
# Streaming safetensors writer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Planned:
    name: str
    source: str
    shape: Tuple[int, ...]
    nbytes: int


def _stream_tensor(fh, tensor, torch_dtype: torch.dtype) -> int:
    """Dequantize ``tensor`` row-chunked and append it to ``fh``.

    GGUFReader hands Q8_0 back as block-padded bytes (``token_embd`` is
    151936x2720 for a 2560-wide model), one row per row, so a row slice is a
    contiguous byte range and dequantizing it chunk-wise is exact -- see the
    bit-identity check in ``_chunked_dequant_selfcheck``.
    """
    from gguf import dequantize

    data = tensor.data
    written = 0

    def emit(block: np.ndarray) -> int:
        arr = np.ascontiguousarray(block)
        if not arr.flags.writeable:  # memmap-backed F32 tensors reach here uncopied
            arr = arr.copy()
        out = torch.from_numpy(arr).to(torch_dtype)
        raw = out.contiguous().view(torch.uint8).numpy().tobytes()
        fh.write(raw)
        return len(raw)

    if data.ndim < 2:
        return emit(dequantize(data, tensor.tensor_type))

    rows_per_chunk = max(1, _CHUNK_BYTES // (int(data.shape[1]) * 4))
    for start in range(0, int(data.shape[0]), rows_per_chunk):
        chunk = data[start:start + rows_per_chunk]
        written += emit(dequantize(chunk, tensor.tensor_type))
    return written


def _write_safetensors(
    out_path: Path,
    plan: Sequence[_Planned],
    tensors: Dict[str, Any],
    torch_dtype: torch.dtype,
    dtype_name: str,
    metadata: Dict[str, str],
) -> int:
    """Two passes: a header from shapes alone, then one tensor at a time."""
    header: Dict[str, Any] = {}
    offset = 0
    for item in plan:
        header[item.name] = {
            "dtype": dtype_name,
            "shape": list(item.shape),
            "data_offsets": [offset, offset + item.nbytes],
        }
        offset += item.nbytes
    header["__metadata__"] = metadata

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)

    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)))
            fh.write(blob)
            for item in plan:
                written = _stream_tensor(fh, tensors[item.source], torch_dtype)
                if written != item.nbytes:
                    raise ConversionError(
                        f"{item.name}: wrote {written} bytes, header declared {item.nbytes}")
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return 8 + len(blob) + offset


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_output(path: Path, expected: Dict[str, Tuple[int, ...]], dtype_name: str,
                  metadata_expected: Dict[str, Any]) -> None:
    """Re-open the written file through the real reader; no tensor bytes read."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        declared = handle.metadata() or {}
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}
        dtypes = {key: handle.get_slice(key).get_dtype() for key in keys}

    missing = sorted(set(expected) - keys)
    unexpected = sorted(keys - set(expected))
    if missing or unexpected:
        raise ConversionError(
            f"output key set is wrong: missing {missing[:5]} unexpected {unexpected[:5]}")
    bad = {k: (shapes[k], expected[k]) for k in expected if shapes[k] != expected[k]}
    if bad:
        raise ConversionError(f"output shapes disagree with the declared dims: {list(bad.items())[:5]}")
    bad_dtype = sorted(k for k in keys if dtypes[k] != dtype_name)
    if bad_dtype:
        raise ConversionError(f"{len(bad_dtype)} tensors are not {dtype_name}: {bad_dtype[:5]}")
    if json.loads(declared.get("minimax_h3_te", "{}")) != metadata_expected:
        raise ConversionError("the written minimax_h3_te metadata does not round-trip")


def _chunked_dequant_selfcheck(tensors: Dict[str, Any], plan: Sequence[_Planned]) -> Optional[float]:
    """Row-chunked dequantize vs. one-shot ``gguf.dequantize`` on the smallest 2D tensor."""
    from gguf import dequantize

    candidates = [item for item in plan if tensors[item.source].data.ndim == 2]
    if not candidates:
        return None
    source = tensors[min(candidates, key=lambda i: i.nbytes).source]
    data = source.data
    whole = dequantize(data, source.tensor_type)
    rows = max(1, int(data.shape[0]) // 4)
    chunked = np.concatenate(
        [dequantize(data[s:s + rows], source.tensor_type) for s in range(0, int(data.shape[0]), rows)]
    )
    return float(np.max(np.abs(whole.astype(np.float64) - chunked.astype(np.float64))))


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(gguf_path: str | os.PathLike, out_path: str | os.PathLike, tap: int = 24,
            dtype: str = "bf16", quiet: bool = False) -> Dict[str, Any]:
    from gguf import GGUFReader

    gguf_path = Path(gguf_path)
    out_path = Path(out_path)
    if dtype not in _DTYPES:
        raise ConversionError(f"unsupported dtype {dtype!r}; choose from {sorted(_DTYPES)}")
    if tap < 1:
        raise ConversionError(f"--tap must be >= 1, got {tap}")
    _refuse_shipped_filename(out_path)
    torch_dtype, dtype_name = _DTYPES[dtype]
    itemsize = torch.empty(0, dtype=torch_dtype).element_size()

    reader = GGUFReader(str(gguf_path))
    cfg = read_text_config(reader)
    if tap > cfg.block_count:
        raise ConversionError(f"--tap {tap} exceeds the GGUF's {cfg.block_count} blocks")

    tensors = {t.name: t for t in reader.tensors}
    expected = expected_shapes(cfg, tap)

    plan: List[_Planned] = []
    dead = 0
    for name, tensor in tensors.items():
        target = map_name(name, tap)
        if target is None:
            dead += 1
            continue
        shape = _gguf_shape(tensor)
        if shape != expected[target]:
            raise ConversionError(
                f"{name} -> {target}: GGUF shape {shape} != expected {expected[target]}")
        plan.append(_Planned(target, name, shape, int(np.prod(shape)) * itemsize))

    plan.sort(key=lambda item: _order_key(item.name))
    missing = sorted(set(expected) - {item.name for item in plan})
    if missing:
        raise ConversionError(f"GGUF is missing {len(missing)} required tensors: {missing[:5]}")

    declared = {
        "num_hidden_layers": tap,
        **cfg.as_metadata(),
        "output": _OUTPUT_CONVENTION.format(tap=tap),
        "modalities": "text",
        **_provenance(reader, cfg, gguf_path, tap),
    }

    started = time.time()
    self_check = _chunked_dequant_selfcheck(tensors, plan)
    total = _write_safetensors(
        out_path, plan, tensors, torch_dtype, dtype_name,
        {"minimax_h3_te": json.dumps(declared)},
    )
    del tensors, reader
    verify_output(out_path, expected, dtype_name, declared)

    summary = {
        "output": str(out_path),
        "tensors_written": len(plan),
        "bytes": total,
        "dropped_dead": dead,
        "tap": tap,
        "dtype": dtype,
        "seconds": round(time.time() - started, 1),
        "chunked_dequant_max_abs_diff": self_check,
        "peak_rss_bytes": _peak_rss(),
    }
    if not quiet:
        print(json.dumps({"minimax_h3_te": declared}, indent=2))
        print(f"[te_gguf_convert] wrote {summary['tensors_written']} tensors, "
              f"{total / 2**30:.2f} GiB -> {out_path}")
        print(f"[te_gguf_convert] dropped {dead} tensor(s) as dead weight "
              f"(blocks >= {tap}, final norm, lm_head)")
        print(f"[te_gguf_convert] chunked-vs-oneshot dequant max|diff| = {self_check}")
        print(f"[te_gguf_convert] verified: keys, shapes, {dtype_name} dtype, metadata round-trip")
        peak = summary["peak_rss_bytes"]
        if peak:
            print(f"[te_gguf_convert] peak RSS {peak / 2**30:.2f} GiB, {summary['seconds']}s")
    return summary


def _order_key(name: str) -> Tuple[int, int, str]:
    if name.startswith("model.layers."):
        parts = name.split(".", 3)
        return (1, int(parts[2]), parts[3])
    return (0, 0, name)


def _refuse_shipped_filename(out_path: Path) -> None:
    """A truncated small encoder must never land on a name the loader auto-selects."""
    try:
        from core.models.minimax_h3.loader import MINIMAX_H3_TE_PATTERNS
    except Exception:  # standalone use, outside the backend package
        return
    if out_path.name in MINIMAX_H3_TE_PATTERNS:
        raise ConversionError(
            f"{out_path.name} is one of the shipped 32B text-encoder filenames "
            f"({MINIMAX_H3_TE_PATTERNS}); the loader would auto-select this degraded "
            f"encoder. Choose another name.")


def _peak_rss() -> Optional[int]:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", info.rss))
    except Exception:
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gguf", help="input Qwen3-VL GGUF")
    parser.add_argument("output", help="output .safetensors path")
    parser.add_argument("--tap", type=int, default=24,
                        help="keep blocks 0..tap-1 (default: 24)")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    args = parser.parse_args(argv)

    try:
        import psutil

        print(f"[te_gguf_convert] host RAM available: "
              f"{psutil.virtual_memory().available / 2**30:.1f} GiB")
    except Exception:
        pass

    try:
        convert(args.gguf, args.output, tap=args.tap, dtype=args.dtype)
    except ConversionError as exc:
        print(f"[te_gguf_convert] FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
