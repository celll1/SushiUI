"""Native GGUF v3 container reader -- no ``gguf`` pip dependency.

Reads the flat ComfyUI-layout GGUF tensor names MiniMax Music 3 ships
(``docs/guides/MINIMAX_MUSIC3_DESIGN.md``, "GGUF weights"), which
``core.models.minimax_music3.flat_remap`` / ``pruned_text_encoder_remap``
already turn into the vendored modules' state dicts -- the dependency
decision and full rationale are recorded in that design doc, not repeated
here.

SCOPE: header/metadata/tensor-descriptor parsing, and materialization of
F32/F16/BF16 tensors only. Any other GGML type (Q8_0 above all) is
RECOGNIZED but UNSUPPORTED -- ``GGUFStateDict.__getitem__`` refuses it via
``refuse_unsupported_tensor_types``, HEADER-ONLY (no tensor byte read).

DIMENSION ORDER: GGUF's ``ne[]`` has ``ne[0]`` FASTEST-varying (the opposite
of a torch/numpy shape, whose LAST entry is fastest), so
``torch_shape = reversed(gguf_dims)`` plus a plain (non-transposing)
``reshape`` reconstructs the tensor. Verified against the real staged DiT's
non-square fused ``to_qkv.weight`` and cross-checked against the installed
``gguf`` package's own reader (dev-time only; never imported at runtime) --
see the design doc's "GGUF weights" section.

MEMORY: ``parse_gguf_header`` touches no tensor byte, so it is safe on a
multi-GB file. ``GGUFStateDict`` memory-maps the file and materializes one
tensor per ``__getitem__`` call as an OWNED (copied) CPU tensor, never a
view into the mmap -- a caller may hold that tensor (eventually as a real
``nn.Parameter``) long after ``close()`` unmaps the file, and a torch tensor
that is still a view into an unmapped region is a dangling pointer, the same
hazard ``flat_remap.apply_flat_dit_state_dict`` already guards against by
cloning every ``torch.chunk`` result. The read stays lazy either way: only
the requested tensor's bytes are ever touched.

Validation refuses (never silently tolerates) a foreign or truncated file --
wrong magic, unsupported version, a header field or tensor data range that
runs past end of file, a duplicate tensor name -- and every declared length
is bounds-checked against the file's actual size BEFORE it is used to size a
read or an allocation. Only version 3 is implemented; this module does not
attempt the byte-order-swap detection an older/foreign-endian GGUF file
would need, since none of this repo's targets need it.
"""

from __future__ import annotations

import mmap
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

__all__ = [
    "GGUF_MAGIC",
    "GGUF_SUPPORTED_VERSION",
    "GGUF_DEFAULT_ALIGNMENT",
    "GGML_TYPE_NAMES",
    "GGML_TORCH_DTYPE",
    "GGML_QUANT_LAYOUT",
    "GGUFFormatError",
    "GGUFUnsupportedTensorTypeError",
    "GGUFTensorInfo",
    "GGUFHeader",
    "parse_gguf_header",
    "unsupported_tensor_types",
    "refuse_unsupported_tensor_types",
    "GGUFStateDict",
]

GGUF_MAGIC = b"GGUF"
GGUF_SUPPORTED_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32  # GGUF spec default when no `general.alignment` key is present.

# ---------------------------------------------------------------------------
# GGUF metadata value type ids (13 total: 11 scalar kinds + STRING + ARRAY).
# ---------------------------------------------------------------------------
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16 = 0, 1, 2, 3
_T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = 4, 5, 6, 7
_T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = 8, 9, 10, 11, 12

# GGUF has no legitimate use for metadata arrays nested more than one level
# deep; see `_Cursor.value`'s ARRAY branch.
_MAX_METADATA_ARRAY_DEPTH = 8

# struct format for every scalar type except BOOL and STRING, which need their
# own decoding (BOOL -> python bool; STRING -> length-prefixed UTF-8, handled
# by `_Cursor.string`). `_Cursor._struct` derives the byte count itself via
# `struct.calcsize`, so no size is stored here alongside the format.
_SCALAR_STRUCT: Dict[int, str] = {
    _T_UINT8: "<B",
    _T_INT8: "<b",
    _T_UINT16: "<H",
    _T_INT16: "<h",
    _T_UINT32: "<I",
    _T_INT32: "<i",
    _T_FLOAT32: "<f",
    _T_UINT64: "<Q",
    _T_INT64: "<q",
    _T_FLOAT64: "<d",
}

# ---------------------------------------------------------------------------
# GGML tensor type ids -> names, and -> (block_size, block_bytes). This is the
# public GGML quantization spec table (llama.cpp's `ggml_type_size` /
# `ggml_blck_size`), hardcoded here -- not imported from the `gguf` package at
# runtime, per this module's docstring -- so `unsupported_tensor_types` and the
# file-range validation below can compute an exact byte count and refuse an
# out-of-range tensor for ANY declared type, not only the ones this reader
# materializes.
# ---------------------------------------------------------------------------
GGML_TYPE_NAMES: Dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0",
}

# {ggml_type_id: (block_size elements, block_bytes)}.
GGML_QUANT_LAYOUT: Dict[int, Tuple[int, int]] = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 40), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66),
    17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4),
    27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54),
    35: (256, 66), 39: (32, 17), 40: (64, 36), 41: (128, 18),
}

_GGML_BF16 = 30

# The three GGML types this phase materializes into a torch tensor, and the
# numpy dtype used to view their raw bytes (BF16 has no native numpy dtype --
# it is read as uint16 and bit-reinterpreted with `torch.Tensor.view`).
GGML_TORCH_DTYPE: Dict[int, torch.dtype] = {0: torch.float32, 1: torch.float16, 30: torch.bfloat16}
_NUMPY_VIEW_DTYPE: Dict[int, Any] = {0: np.float32, 1: np.float16, 30: np.uint16}


class GGUFFormatError(RuntimeError):
    """Not readable as a v3 GGUF container: bad magic, unsupported version, a
    truncated header, a duplicate tensor name, or a tensor data range that
    falls outside the file."""


class GGUFUnsupportedTensorTypeError(GGUFFormatError):
    """The file parses as a valid GGUF container, but a requested tensor's
    GGML type is not one this reader materializes (F32/F16/BF16 only)."""


# ---------------------------------------------------------------------------
# Header parsing: sequential reads from a plain buffered file handle. No
# tensor byte is ever touched here -- only the header (magic + version +
# counts + metadata KV pairs + tensor info records).
# ---------------------------------------------------------------------------

class _Cursor:
    """Sequential little-endian reader over a buffered file handle, tracking
    its own byte offset so every truncation is reported at the offset it
    happened at.

    ``file_size`` is known up front (``os.path.getsize``, before the file is
    even opened) specifically so ``read`` can refuse an attacker-declared
    length BEFORE handing it to ``BufferedReader.read``, which otherwise
    pre-allocates the requested buffer before it can discover the file is
    short -- a 35-byte file with a declared string length of 200,000,000
    would commit +200 MB, and 2**40 raises a bare (uncaught-by-this-module)
    ``MemoryError`` rather than a clean refusal. Bounding every ``read`` this
    way also covers ``tensor_count`` / ``metadata_kv_count`` / ``n_dims``
    being declared absurdly large: each of those only drives a Python
    ``range()`` loop (cheap to construct at any size), and the loop's first
    genuine read past the true end of file now raises immediately instead of
    after however many small reads happened to still be satisfiable."""

    __slots__ = ("fh", "path", "pos", "file_size")

    def __init__(self, fh, path: str, file_size: int) -> None:
        self.fh = fh
        self.path = path
        self.pos = 0
        self.file_size = file_size

    def read(self, n: int) -> bytes:
        if n < 0:
            raise GGUFFormatError(f"{self.path}: negative read length {n} at offset {self.pos}")
        remaining = self.file_size - self.pos
        if n > remaining:
            raise GGUFFormatError(
                f"{self.path}: a declared length of {n} byte(s) at offset {self.pos} exceeds "
                f"the {remaining} byte(s) remaining in the file (size {self.file_size}) -- "
                f"the file is truncated or its header is corrupt. Refused BEFORE allocating a "
                f"buffer for the declared length."
            )
        data = self.fh.read(n)
        if len(data) != n:
            raise GGUFFormatError(
                f"{self.path}: unexpected end of file at offset {self.pos} (wanted {n} "
                f"byte(s), got {len(data)}) -- the file is truncated or not a GGUF container."
            )
        self.pos += n
        return data

    def _struct(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))[0]

    def uint32(self) -> int:
        return self._struct("<I")

    def uint64(self) -> int:
        return self._struct("<Q")

    def string(self) -> str:
        length = self.uint64()
        raw = self.read(length)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFFormatError(
                f"{self.path}: a GGUF string at offset {self.pos - length} is not valid UTF-8"
            ) from exc

    def value(self, type_id: int, *, depth: int = 0) -> Any:
        if type_id == _T_STRING:
            return self.string()
        if type_id == _T_BOOL:
            return bool(self._struct("<B"))
        if type_id == _T_ARRAY:
            # GGUF has no legitimate use for array nesting beyond one level;
            # refuse deeper recursion outright rather than risk a
            # `RecursionError` (a ~12 KB file with 5,000 nested ARRAY levels
            # would otherwise raise one, which is not a `GGUFFormatError` and
            # would escape both music3 loader builders uncaught).
            if depth >= _MAX_METADATA_ARRAY_DEPTH:
                raise GGUFFormatError(
                    f"{self.path}: metadata array nesting exceeds "
                    f"{_MAX_METADATA_ARRAY_DEPTH} level(s) at offset {self.pos} -- refusing "
                    f"rather than recursing further."
                )
            element_type_id = self.uint32()
            count = self.uint64()
            # Every GGUF value type is at least 1 byte; an element count
            # larger than the bytes actually remaining can never be genuine.
            # Bounding it here stops a declared 2**62-element array from
            # building a multi-billion-entry Python list before some later
            # read finally hits EOF.
            remaining = self.file_size - self.pos
            if count > remaining:
                raise GGUFFormatError(
                    f"{self.path}: a metadata array at offset {self.pos} declares {count} "
                    f"element(s), more than the {remaining} byte(s) remaining in the file -- "
                    f"the file is truncated or its header is corrupt."
                )
            return [self.value(element_type_id, depth=depth + 1) for _ in range(count)]
        fmt = _SCALAR_STRUCT.get(type_id)
        if fmt is None:
            raise GGUFFormatError(
                f"{self.path}: unknown GGUF metadata value type id {type_id} at offset {self.pos}"
            )
        return self._struct(fmt)


@dataclass(frozen=True)
class GGUFTensorInfo:
    """One tensor's descriptor from the header -- no data bytes attached."""

    name: str
    ggml_type_id: int
    ggml_type_name: str
    gguf_dims: Tuple[int, ...]     # as stored on disk: ne[0] is FASTEST-varying.
    torch_shape: Tuple[int, ...]   # reversed(gguf_dims) -- see module docstring.
    n_elements: int
    data_offset: int               # ABSOLUTE file offset (already past the aligned data-section base).
    n_bytes: int
    torch_dtype: Optional[torch.dtype]  # None if this reader does not materialize this GGML type.


@dataclass(frozen=True)
class GGUFHeader:
    """The full header census of one GGUF file: metadata + every tensor's
    descriptor. Producing this touches no tensor byte."""

    path: str
    version: int
    tensor_count: int
    metadata: Dict[str, Any]
    tensors: Tuple[GGUFTensorInfo, ...]
    data_start: int
    alignment: int
    file_size: int

    def tensor_names(self) -> List[str]:
        return [t.name for t in self.tensors]

    def dtype_histogram(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for t in self.tensors:
            hist[t.ggml_type_name] = hist.get(t.ggml_type_name, 0) + 1
        return hist


def _tensor_nbytes(ggml_type_id: int, n_elements: int, type_name: str, *, path: str, tensor_name: str) -> int:
    layout = GGML_QUANT_LAYOUT.get(ggml_type_id)
    if layout is None:
        raise GGUFFormatError(
            f"{path}: tensor {tensor_name!r} declares GGML type id {ggml_type_id} "
            f"({type_name}), which has no known block layout in this reader -- cannot even "
            f"validate its data range without one."
        )
    block_elements, block_bytes = layout
    if n_elements % block_elements != 0:
        raise GGUFFormatError(
            f"{path}: tensor {tensor_name!r} has {n_elements} element(s), not a multiple of "
            f"{type_name}'s block size ({block_elements})."
        )
    return (n_elements // block_elements) * block_bytes


def parse_gguf_header(path: str) -> GGUFHeader:
    """Parse a GGUF v3 header: magic, version, metadata, tensor descriptors.

    Reads nothing beyond the header itself -- safe to call on a multi-GB file.
    Raises ``GGUFFormatError`` for a wrong magic, an unsupported version, a
    truncated read, a duplicate tensor name, or any tensor whose computed
    data range falls outside the file (the file is corrupt or truncated after
    a header that otherwise parsed).
    """
    file_size = os.path.getsize(path)
    with open(path, "rb", buffering=64 * 1024) as fh:
        cur = _Cursor(fh, path, file_size)

        magic = cur.read(4)
        if magic != GGUF_MAGIC:
            raise GGUFFormatError(
                f"{path}: not a GGUF file (magic {magic!r}, expected {GGUF_MAGIC!r}) -- "
                f"refusing a foreign file rather than guessing at its layout."
            )
        version = cur.uint32()
        if version != GGUF_SUPPORTED_VERSION:
            raise GGUFFormatError(
                f"{path}: GGUF version {version} is not supported (this reader implements "
                f"v{GGUF_SUPPORTED_VERSION} only, the version both staged MiniMax Music 3 "
                f"GGUF files declare)."
            )
        tensor_count = cur.uint64()
        metadata_kv_count = cur.uint64()

        metadata: Dict[str, Any] = {}
        for i in range(metadata_kv_count):
            key = cur.string()
            value_type_id = cur.uint32()
            value = cur.value(value_type_id)
            if key in metadata:
                raise GGUFFormatError(f"{path}: duplicate metadata key {key!r} (entry {i})")
            metadata[key] = value

        tensor_records: List[Tuple[str, Tuple[int, ...], int, int]] = []
        for i in range(tensor_count):
            name = cur.string()
            n_dims = cur.uint32()
            dims = tuple(cur.uint64() for _ in range(n_dims))
            ggml_type_id = cur.uint32()
            rel_offset = cur.uint64()
            tensor_records.append((name, dims, ggml_type_id, rel_offset))

        raw_alignment = metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
        # Type-guard BEFORE `int()`: a string-typed value raises `ValueError`
        # from `int()`, an array-typed one raises `TypeError` -- neither
        # names the file or the field, and both would escape as the wrong
        # exception type for a caller catching `GGUFFormatError`. `bool` is
        # excluded even though it is technically an `int` subclass in
        # Python -- a BOOL-typed `general.alignment` is not a real alignment
        # value either.
        if isinstance(raw_alignment, bool) or not isinstance(raw_alignment, int):
            raise GGUFFormatError(
                f"{path}: general.alignment has type {type(raw_alignment).__name__} "
                f"({raw_alignment!r}), expected an integer."
            )
        alignment = raw_alignment
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise GGUFFormatError(
                f"{path}: general.alignment={raw_alignment!r} is not a positive power of two."
            )
        padding = (-cur.pos) % alignment
        data_start = cur.pos + padding

    names_seen = set()
    tensors: List[GGUFTensorInfo] = []
    for name, dims, ggml_type_id, rel_offset in tensor_records:
        if name in names_seen:
            raise GGUFFormatError(f"{path}: duplicate tensor name {name!r} in the tensor info list")
        names_seen.add(name)

        type_name = GGML_TYPE_NAMES.get(ggml_type_id, f"UNKNOWN({ggml_type_id})")
        n_elements = 1
        for d in dims:
            n_elements *= int(d)
        n_bytes = _tensor_nbytes(ggml_type_id, n_elements, type_name, path=path, tensor_name=name)

        # `rel_offset` is relative to `data_start` (the aligned data-section
        # base computed above) -- NOT an absolute file offset. See module
        # docstring. `rel_offset` was read as an unsigned uint64
        # (`_Cursor.uint64`), so `data_offset` can never fall below
        # `data_start` -- only the upper bound needs checking.
        data_offset = data_start + rel_offset
        if data_offset + n_bytes > file_size:
            raise GGUFFormatError(
                f"{path}: tensor {name!r} ({type_name}) data range "
                f"[{data_offset}, {data_offset + n_bytes}) falls outside the file "
                f"(size {file_size} byte(s)) -- the file is truncated or its header is corrupt."
            )

        tensors.append(GGUFTensorInfo(
            name=name,
            ggml_type_id=ggml_type_id,
            ggml_type_name=type_name,
            gguf_dims=tuple(int(d) for d in dims),
            torch_shape=tuple(reversed([int(d) for d in dims])),
            n_elements=n_elements,
            data_offset=data_offset,
            n_bytes=n_bytes,
            torch_dtype=GGML_TORCH_DTYPE.get(ggml_type_id),
        ))

    # No separate "parsed count == declared count" check here: the loop above
    # ran exactly `tensor_count` times and appended exactly one entry per
    # iteration (or raised), so `len(tensors) == tensor_count` always holds
    # by construction.
    return GGUFHeader(
        path=path,
        version=version,
        tensor_count=tensor_count,
        metadata=metadata,
        tensors=tuple(tensors),
        data_start=data_start,
        alignment=alignment,
        file_size=file_size,
    )


# ---------------------------------------------------------------------------
# Header-only "can this reader materialize every tensor" gate.
# ---------------------------------------------------------------------------

def unsupported_tensor_types(header: GGUFHeader) -> Dict[str, List[str]]:
    """``{ggml_type_name: [tensor names]}`` for every tensor this reader
    cannot materialize (anything outside F32/F16/BF16). Header-only -- reads
    ``GGUFTensorInfo.torch_dtype``, already resolved by ``parse_gguf_header``,
    and touches no tensor byte."""
    out: Dict[str, List[str]] = {}
    for t in header.tensors:
        if t.torch_dtype is None:
            out.setdefault(t.ggml_type_name, []).append(t.name)
    return out


def refuse_unsupported_tensor_types(header: GGUFHeader, *, arch: str, label: str) -> None:
    """Raise ``GGUFUnsupportedTensorTypeError`` if ``header`` declares any
    tensor this reader cannot materialize. HEADER-ONLY: never reads a tensor
    byte, so it is safe to call before opening a multi-GB file's data section
    at all.

    ``Q8_0`` is called out by name (design doc phase 12, packed residency
    dequantized at use) because it is what the staged MiniMax Music 3 GGUF
    text encoder carries; any other unsupported type is reported generically.
    """
    unsupported = unsupported_tensor_types(header)
    if not unsupported:
        return
    q8_0_names = unsupported.pop("Q8_0", None)
    reasons: List[str] = []
    if q8_0_names:
        reasons.append(
            f"{len(q8_0_names)} Q8_0 tensor(s) (block-quantized, 32 values / 34-byte block; "
            f"e.g. {', '.join(q8_0_names[:3])}). This reader loads F32/F16/BF16 only; Q8_0 "
            f"residency -- packed weights with dequantization at use, following "
            f"core.models.common.convrot_int8_linear's runtime shape -- is design doc phase "
            f"12, not yet implemented. Refusing rather than dequantizing to bf16 at load "
            f"(the design doc rejected that as a hollow feature: the resulting resident text "
            f"encoder would be no smaller than the bf16 file already staged) or silently "
            f"skipping the tensors (which would drop them from the state dict entirely)"
        )
    for type_name, names in sorted(unsupported.items()):
        reasons.append(
            f"{len(names)} {type_name} tensor(s) (e.g. {', '.join(names[:3])}), which this "
            f"reader does not implement at all"
        )
    raise GGUFUnsupportedTensorTypeError(
        f"the {arch} {label} GGUF checkpoint ({header.path}) declares tensor type(s) this "
        f"reader cannot materialize: " + "; ".join(reasons) + ". "
        f"Header-only refusal -- no tensor byte of this {header.file_size}-byte file was read."
    )


# ---------------------------------------------------------------------------
# Lazy, mmap-backed tensor access.
# ---------------------------------------------------------------------------

class GGUFStateDict(Mapping):
    """``Mapping[str, torch.Tensor]`` over one GGUF file's tensors, backed by
    a memory-mapped file. Exactly the contract
    ``core.models.minimax_music3.flat_remap.apply_flat_dit_state_dict`` /
    ``apply_flat_text_encoder_state_dict`` and ``pruned_text_encoder_remap.
    apply_pruned_text_encoder_state_dict`` already expect from a safetensors
    read, so a caller can hand this object to either remap unchanged.

    ``__getitem__`` materializes exactly ONE tensor per call, as an OWNED
    (copied) CPU tensor -- see the module docstring's "MEMORY DISCIPLINE"
    section for why a copy rather than a zero-copy mmap view. ``.keys()`` /
    iteration never touch tensor bytes at all.

    ``arch``/``label`` name this file for ``GGUFUnsupportedTensorTypeError``
    messages raised from an unsupported ``__getitem__`` -- this class lives in
    ``core/models/common/`` and is not MiniMax Music 3-specific (design doc
    phase 12 will exercise it too), so a caller wanting an architecture-named
    refusal passes its own; the defaults are deliberately neutral.

    Must be closed (``close()``, or used as a context manager) once every
    tensor it will hand out has been extracted; the mmap stays open and valid
    until then. Also closable via garbage collection (``__del__``) as a
    last-resort safety net -- ``close()`` remains the documented contract.
    """

    def __init__(self, header: GGUFHeader, *, arch: str = "model", label: str = "checkpoint") -> None:
        self._header = header
        self._arch = arch
        self._label = label
        self._by_name: Dict[str, GGUFTensorInfo] = {t.name: t for t in header.tensors}
        self._closed = True  # until the open sequence below completes
        self._fh = open(header.path, "rb")
        try:
            # Re-validate the file's size against what the header was parsed
            # from: a truncation between `parse_gguf_header` and this
            # `mmap.mmap` call would otherwise mmap a now-short file, and a
            # later `np.frombuffer` read past its true end is a SIGBUS-class
            # process kill on POSIX, not a catchable Python exception -- one
            # cheap `os.fstat` closes that window.
            actual_size = os.fstat(self._fh.fileno()).st_size
            if actual_size != header.file_size:
                raise GGUFFormatError(
                    f"{header.path}: file size changed from {header.file_size} to "
                    f"{actual_size} byte(s) between header parse and open -- refusing to "
                    f"memory-map a file that no longer matches its own parsed header."
                )
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            # `self._fh` must not leak if anything above raises (the size
            # re-check, or `mmap.mmap` itself) -- there would be no `close()`
            # call to release it, holding a Windows file lock on a
            # multi-gigabyte file for the life of the exception's traceback.
            self._fh.close()
            raise
        self._closed = False

    @property
    def header(self) -> GGUFHeader:
        return self._header

    def __len__(self) -> int:
        return len(self._by_name)

    def __iter__(self):
        return iter(self._by_name)

    def __contains__(self, key: object) -> bool:
        return key in self._by_name

    def keys(self):
        return self._by_name.keys()

    def __getitem__(self, name: str) -> torch.Tensor:
        if self._closed:
            raise RuntimeError(f"{self._header.path}: GGUFStateDict is closed; cannot read {name!r}")
        info = self._by_name[name]
        if info.torch_dtype is None:
            refuse_unsupported_tensor_types(
                # A single-tensor header slice, so the shared refusal message
                # (and its Q8_0-specific reason) fires for this ONE lookup
                # without re-scanning every other tensor in the file.
                GGUFHeader(
                    path=self._header.path, version=self._header.version,
                    tensor_count=1, metadata={}, tensors=(info,),
                    data_start=self._header.data_start, alignment=self._header.alignment,
                    file_size=self._header.file_size,
                ),
                arch=self._arch, label=self._label,
            )
        np_view_dtype = _NUMPY_VIEW_DTYPE[info.ggml_type_id]
        raw = np.frombuffer(self._mm, dtype=np.uint8, count=info.n_bytes, offset=info.data_offset)
        # `.copy()` on the numpy array here (not `torch.from_numpy` on the
        # read-only mmap view) does two things at once: it is the detach from
        # the mmap that "MEMORY DISCIPLINE" requires (a fresh, owned buffer,
        # not aliasing this class's mmap -- the same reason
        # `flat_remap.apply_flat_dit_state_dict` clones every `torch.chunk`
        # result), and it sidesteps `torch.from_numpy`'s non-writable-array
        # warning without touching the GLOBAL warnings filter
        # (`warnings.catch_warnings()` is process-wide state, and this repo
        # loads model components on background threads -- mutating it here
        # would race any other thread's own warning filtering). One copy, not
        # a view-then-clone.
        array = raw.view(np_view_dtype).reshape(info.torch_shape).copy()
        tensor = torch.from_numpy(array)
        if info.ggml_type_id == _GGML_BF16:
            tensor = tensor.view(torch.bfloat16)
        return tensor

    def close(self) -> None:
        if self._closed:
            return
        self._mm.close()
        self._fh.close()
        self._closed = True

    def __enter__(self) -> "GGUFStateDict":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort safety net for a caller that forgets `close()` -- the
        # documented contract is still explicit closing (or the context
        # manager); this only guards against holding a Windows file lock on a
        # multi-gigabyte file indefinitely if that contract is missed.
        try:
            self.close()
        except Exception:
            pass
