"""``core.models.common.gguf_container`` -- design doc phase 11.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_gguf_container_test.py -v

Every test here uses a tiny, hand-built, real GGUF file
(``tests.minimax_music3_gguf_fixture``) -- no multi-GB checkpoint is opened.
A real load against the staged snapshot
(``<MODEL_ROOT>/minimax-music3/diffusion_models/minimax_music3_dit_BF16.gguf`` /
``text_encoders/minimax_music3_text_encoder_pruned_Q8_0.gguf``) was verified
manually while writing this reader (header census matching the design doc's
own numbers, a dim-order proof against the installed ``gguf`` package, and a
bit-exactness comparison against ``official/``); that is not repeated here
because it requires the model snapshot, which is not part of this repo --
same convention ``minimax_music3_loader_test.py`` follows for its own
weight-bearing claims.
"""

import os
import struct
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.common import gguf_container as g  # noqa: E402
from tests.minimax_music3_gguf_fixture import write_gguf  # noqa: E402


# ---------------------------------------------------------------------------
# Round trip: F32/F16/BF16, non-square shapes, dim-order.
# ---------------------------------------------------------------------------

def test_round_trips_f32_f16_bf16_tensors_with_correct_values(tmp_path):
    path = os.path.join(str(tmp_path), "tiny.gguf")
    t32 = torch.randn(3, 5, generator=torch.Generator().manual_seed(1))
    t16 = torch.randn(2, 4, generator=torch.Generator().manual_seed(2)).to(torch.float16)
    tbf16 = torch.randn(4, 2, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    write_gguf(path, {"a.weight": t32, "b.weight": t16, "c.weight": tbf16},
               {"general.architecture": "minimax_music3"})

    header = g.parse_gguf_header(path)
    assert header.version == 3
    assert header.tensor_count == 3
    assert header.metadata["general.architecture"] == "minimax_music3"
    assert header.dtype_histogram() == {"F32": 1, "F16": 1, "BF16": 1}

    sd = g.GGUFStateDict(header)
    try:
        assert torch.equal(sd["a.weight"], t32)
        assert torch.equal(sd["b.weight"], t16)
        assert torch.equal(sd["c.weight"], tbf16)
        assert sd["a.weight"].dtype == torch.float32
        assert sd["b.weight"].dtype == torch.float16
        assert sd["c.weight"].dtype == torch.bfloat16
    finally:
        sd.close()


def test_dim_order_is_reversed_and_not_a_transpose(tmp_path):
    """A deliberately NON-SQUARE tensor: if the reader silently transposed
    instead of reversing dims-then-reshaping, this would still "round trip"
    for a square tensor but fail here."""
    path = os.path.join(str(tmp_path), "tiny.gguf")
    t = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3)  # NOT symmetric under transpose
    write_gguf(path, {"fused.weight": t}, {"general.architecture": "minimax_music3"})

    header = g.parse_gguf_header(path)
    info = {ti.name: ti for ti in header.tensors}["fused.weight"]
    # On-disk ne[] is the REVERSE of the torch shape.
    assert info.gguf_dims == (3, 6)
    assert info.torch_shape == (6, 3)

    sd = g.GGUFStateDict(header)
    try:
        got = sd["fused.weight"]
        assert got.shape == (6, 3)
        assert torch.equal(got, t)  # exact values, not just shape
    finally:
        sd.close()


def test_mapping_contract_matches_flat_remap_expectations(tmp_path):
    """``GGUFStateDict`` must satisfy ``Mapping[str, torch.Tensor]`` the way
    ``flat_remap.apply_flat_dit_state_dict`` expects: ``.keys()`` and
    ``__getitem__`` with no data read for the former."""
    path = os.path.join(str(tmp_path), "tiny.gguf")
    t = torch.randn(2, 2)
    write_gguf(path, {"x.weight": t}, {})
    header = g.parse_gguf_header(path)
    sd = g.GGUFStateDict(header)
    try:
        assert set(sd.keys()) == {"x.weight"}
        assert len(sd) == 1
        assert "x.weight" in sd
        assert list(iter(sd)) == ["x.weight"]
    finally:
        sd.close()


# ---------------------------------------------------------------------------
# Q8_0 refusal: header-only, no tensor byte read.
# ---------------------------------------------------------------------------

def test_q8_0_tensor_is_refused_header_only_naming_phase_12(tmp_path):
    path = os.path.join(str(tmp_path), "tiny.gguf")
    write_gguf(
        path, {"ok.weight": torch.randn(2, 2)}, {"general.architecture": "minimax_music3"},
        extra_raw_tensors={"quantized.weight": b"\x00" * 34}, extra_raw_ggml_type_id=8,
    )
    header = g.parse_gguf_header(path)
    unsupported = g.unsupported_tensor_types(header)
    assert unsupported == {"Q8_0": ["quantized.weight"]}

    with pytest.raises(g.GGUFUnsupportedTensorTypeError, match="phase 12"):
        g.refuse_unsupported_tensor_types(header, arch="MiniMax Music 3", label="test")


def test_q8_0_refusal_never_opens_the_data_section(tmp_path):
    """Corrupt every tensor's actual bytes after the header -- if the refusal
    read them, this would raise a DIFFERENT error (or succeed on garbage),
    not the same clean refusal."""
    path = os.path.join(str(tmp_path), "tiny.gguf")
    write_gguf(
        path, {}, {"general.architecture": "minimax_music3"},
        extra_raw_tensors={"quantized.weight": b"\x00" * 34}, extra_raw_ggml_type_id=8,
    )
    header = g.parse_gguf_header(path)  # header-only by construction -- see module docstring
    with open(path, "r+b") as fh:
        fh.seek(header.data_start)
        fh.write(b"\xff" * (header.file_size - header.data_start))
    # Refusal reads only the already-parsed header's type census -- the
    # corrupted bytes are never touched.
    with pytest.raises(g.GGUFUnsupportedTensorTypeError, match="phase 12"):
        g.refuse_unsupported_tensor_types(header, arch="MiniMax Music 3", label="test")


def test_gguf_statedict_refuses_open_if_file_shrank_since_header_parse(tmp_path):
    """A truncation between `parse_gguf_header` and `GGUFStateDict(header)`
    must be caught by the `os.fstat` re-check at open time -- a stale header
    mmap'd against a now-shorter file would otherwise let a later read past
    the true end of file take down the whole process (SIGBUS-class) instead
    of raising a catchable exception."""
    path = os.path.join(str(tmp_path), "shrinks.gguf")
    write_gguf(path, {"x.weight": torch.randn(4, 4)}, {})
    header = g.parse_gguf_header(path)
    with open(path, "r+b") as fh:
        fh.truncate(header.data_start)  # drop the tensor's data bytes entirely
    with pytest.raises(g.GGUFFormatError, match="file size changed"):
        g.GGUFStateDict(header)


def test_gguf_statedict_open_failure_does_not_leak_the_file_handle(tmp_path, monkeypatch):
    """If `mmap.mmap` itself raises, `GGUFStateDict.__init__` must still close
    the file handle it already opened -- otherwise a caller who catches the
    exception and retries holds a leaked Windows lock on the file forever."""
    import mmap as mmap_module

    path = os.path.join(str(tmp_path), "mmap_fails.gguf")
    write_gguf(path, {"x.weight": torch.randn(2, 2)}, {})
    header = g.parse_gguf_header(path)

    original_mmap = mmap_module.mmap

    def _boom(*_args, **_kwargs):
        raise OSError("simulated mmap failure")

    monkeypatch.setattr(mmap_module, "mmap", _boom)
    try:
        with pytest.raises(OSError, match="simulated mmap failure"):
            g.GGUFStateDict(header)
    finally:
        monkeypatch.setattr(mmap_module, "mmap", original_mmap)
    # The file must be freely re-openable/rewritable -- a leaked handle would
    # make this fail on Windows (a different-process-open file cannot be
    # deleted or exclusively reopened for write while a handle is held).
    with open(path, "r+b"):
        pass
    os.remove(path)


def test_getitem_on_a_q8_0_tensor_refuses_without_reading_other_tensors(tmp_path):
    """A file with BOTH a supported and an unsupported tensor: fetching the
    supported one works; fetching the unsupported one refuses by name."""
    path = os.path.join(str(tmp_path), "tiny.gguf")
    t = torch.randn(2, 2)
    write_gguf(path, {"ok.weight": t}, {}, extra_raw_tensors={"bad.weight": b"\x00" * 34}, extra_raw_ggml_type_id=8)
    header = g.parse_gguf_header(path)
    sd = g.GGUFStateDict(header)
    try:
        assert torch.equal(sd["ok.weight"], t)
        with pytest.raises(g.GGUFUnsupportedTensorTypeError, match="bad.weight"):
            sd["bad.weight"]
    finally:
        sd.close()


def test_non_q8_0_unsupported_type_is_refused_generically(tmp_path):
    """An unsupported type OTHER than Q8_0 is refused too, without the
    phase-12-specific wording (that is Q8_0's own reason)."""
    path = os.path.join(str(tmp_path), "tiny.gguf")
    header_bytes = bytearray()
    header_bytes += b"GGUF"
    header_bytes += struct.pack("<I", 3)
    header_bytes += struct.pack("<Q", 1)  # tensor_count
    header_bytes += struct.pack("<Q", 0)  # metadata_kv_count
    name = "q4.weight".encode("utf-8")
    header_bytes += struct.pack("<Q", len(name)) + name
    header_bytes += struct.pack("<I", 1)  # n_dims
    header_bytes += struct.pack("<Q", 32)  # ne[0] -- one Q4_0 block (32 elements)
    header_bytes += struct.pack("<I", 2)  # ggml type id 2 = Q4_0
    header_bytes += struct.pack("<Q", 0)  # rel_offset
    pad = (-len(header_bytes)) % 32
    header_bytes += b"\x00" * pad
    with open(path, "wb") as fh:
        fh.write(bytes(header_bytes))
        fh.write(b"\x00" * 18)  # Q4_0: 32 elements / block, 18 bytes / block
    header = g.parse_gguf_header(path)
    assert header.dtype_histogram() == {"Q4_0": 1}
    with pytest.raises(g.GGUFUnsupportedTensorTypeError, match="Q4_0") as excinfo:
        g.refuse_unsupported_tensor_types(header, arch="MiniMax Music 3", label="test")
    assert "phase 12" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Format validation: magic, version, truncation, out-of-range, duplicates.
# ---------------------------------------------------------------------------

def test_refuses_wrong_magic(tmp_path):
    path = os.path.join(str(tmp_path), "not_gguf.bin")
    with open(path, "wb") as fh:
        fh.write(b"NOPE" + b"\x00" * 32)
    with pytest.raises(g.GGUFFormatError, match="not a GGUF file"):
        g.parse_gguf_header(path)


def test_refuses_unsupported_version(tmp_path):
    path = os.path.join(str(tmp_path), "v2.gguf")
    with open(path, "wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<I", 2))
        fh.write(struct.pack("<QQ", 0, 0))
    with pytest.raises(g.GGUFFormatError, match="version 2"):
        g.parse_gguf_header(path)


def test_refuses_truncated_header(tmp_path):
    """5 declared tensors but the file ends right after the counts -- the
    first tensor-info read's declared length (its name string) exceeds the
    file's remaining bytes, so this is refused by the length-bound check
    (F2's fix) before ``BufferedReader.read`` is ever called, rather than by
    hitting a real EOF partway through a read."""
    path = os.path.join(str(tmp_path), "truncated.gguf")
    with open(path, "wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<I", 3))
        fh.write(struct.pack("<Q", 5))  # claims 5 tensors
        fh.write(struct.pack("<Q", 0))  # 0 metadata kv
        # ... but the file ends here, no tensor info records at all.
    with pytest.raises(g.GGUFFormatError, match="exceeds the .* byte"):
        g.parse_gguf_header(path)


def test_refuses_a_tensor_data_range_outside_the_file(tmp_path):
    """A header that declares an offset/size combination the file is too
    short to actually hold -- truncated or corrupt after an otherwise valid
    header."""
    path = os.path.join(str(tmp_path), "short.gguf")
    write_gguf(path, {"x.weight": torch.randn(4, 4)}, {})
    # Chop the file short, after the header but before the tensor's declared
    # data range ends.
    with open(path, "r+b") as fh:
        header = g.parse_gguf_header(path)
        fh.truncate(header.data_start + 4)  # far short of a 4x4 float32 tensor
    with pytest.raises(g.GGUFFormatError, match="falls outside the file"):
        g.parse_gguf_header(path)


def test_refuses_duplicate_tensor_names(tmp_path):
    path = os.path.join(str(tmp_path), "dup.gguf")
    header_bytes = bytearray()
    header_bytes += b"GGUF"
    header_bytes += struct.pack("<I", 3)
    header_bytes += struct.pack("<Q", 2)
    header_bytes += struct.pack("<Q", 0)
    for _ in range(2):
        name = "dup.weight".encode("utf-8")
        header_bytes += struct.pack("<Q", len(name)) + name
        header_bytes += struct.pack("<I", 1)
        header_bytes += struct.pack("<Q", 1)
        header_bytes += struct.pack("<I", 0)  # F32
        header_bytes += struct.pack("<Q", 0)
    pad = (-len(header_bytes)) % 32
    header_bytes += b"\x00" * pad
    with open(path, "wb") as fh:
        fh.write(bytes(header_bytes))
        fh.write(b"\x00" * 4)
    with pytest.raises(g.GGUFFormatError, match="duplicate tensor name"):
        g.parse_gguf_header(path)


def test_default_alignment_is_32_and_data_start_is_padded(tmp_path):
    path = os.path.join(str(tmp_path), "aligned.gguf")
    write_gguf(path, {"x.weight": torch.randn(1)}, {})
    header = g.parse_gguf_header(path)
    assert header.alignment == g.GGUF_DEFAULT_ALIGNMENT == 32
    assert header.data_start % 32 == 0


def test_custom_alignment_metadata_is_honored(tmp_path):
    path = os.path.join(str(tmp_path), "aligned64.gguf")
    write_gguf(path, {"x.weight": torch.randn(1)}, {"general.alignment": 64}, alignment=64)
    header = g.parse_gguf_header(path)
    assert header.alignment == 64
    assert header.data_start % 64 == 0
    sd = g.GGUFStateDict(header)
    try:
        assert torch.allclose(sd["x.weight"], torch.zeros(1) + sd["x.weight"])  # just prove it reads
    finally:
        sd.close()


def test_relative_offsets_are_not_absolute_file_offsets(tmp_path):
    """The tensor `rel_offset` field is relative to the (aligned) data
    section base, not to byte 0 of the file -- a reader that treated it as
    absolute would compute the wrong `data_offset` for every file whose
    header is non-trivial (i.e. every real one)."""
    path = os.path.join(str(tmp_path), "offsets.gguf")
    write_gguf(path, {"a.weight": torch.randn(4), "b.weight": torch.randn(4)}, {"k": "some metadata to grow the header"})
    header = g.parse_gguf_header(path)
    by_name = {t.name: t for t in header.tensors}
    assert by_name["a.weight"].data_offset >= header.data_start
    assert by_name["b.weight"].data_offset > by_name["a.weight"].data_offset
    # And well past byte 0 -- the header (magic+version+counts+metadata+tensor
    # info) is non-trivial for this file.
    assert header.data_start > 24


# ---------------------------------------------------------------------------
# F2: an attacker-declared length must be refused BEFORE it is used to size
# an allocation, not merely eventually caught by hitting a real EOF.
# ---------------------------------------------------------------------------

def _write_string_length_only_file(path: str, declared_length: int) -> None:
    """The smallest possible file that gets as far as decoding ONE metadata
    key's declared STRING length before running out of bytes: magic +
    version + counts (1 metadata kv, 0 tensors) + a length prefix -- and then
    nothing else. Whatever `declared_length` claims, there is no key data
    and no value at all behind it."""
    with open(path, "wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<I", 3))
        fh.write(struct.pack("<Q", 0))  # tensor_count
        fh.write(struct.pack("<Q", 1))  # metadata_kv_count
        fh.write(struct.pack("<Q", declared_length))  # the metadata KEY's string length


def test_refuses_oversized_declared_string_length_without_large_allocation(tmp_path):
    """A 28-byte file (magic+version+counts+one length prefix) declaring a
    200,000,000-byte string: the old code would hand that straight to
    `BufferedReader.read`, which pre-allocates the buffer before discovering
    the file is short. The fix must refuse via the length-bound check, fast,
    with a `GGUFFormatError` naming the file -- not by actually attempting
    the allocation."""
    path = os.path.join(str(tmp_path), "oversized_string.gguf")
    _write_string_length_only_file(path, 200_000_000)
    import time
    t0 = time.time()
    with pytest.raises(g.GGUFFormatError, match="exceeds the .* byte"):
        g.parse_gguf_header(path)
    elapsed = time.time() - t0
    # Refused by a length comparison, not by touching 200 MB -- generous
    # bound (this must never approach the time an actual 200 MB read/alloc
    # would take on any real machine).
    assert elapsed < 2.0


def test_refuses_absurdly_large_declared_string_length_as_gguf_format_error(tmp_path):
    """2**40 (1 TB): the old code's `BufferedReader.read(2**40)` raises a
    bare `MemoryError`, not a `GGUFFormatError` -- it would propagate out of
    both music3 loader builders uncaught. The fix must raise the same clean,
    named `GGUFFormatError` a merely-oversized-but-plausible length gets."""
    path = os.path.join(str(tmp_path), "absurd_string.gguf")
    _write_string_length_only_file(path, 2 ** 40)
    with pytest.raises(g.GGUFFormatError, match="exceeds the .* byte"):
        g.parse_gguf_header(path)


def test_refuses_huge_tensor_count_on_a_short_file(tmp_path):
    """A declared tensor_count of 2**62 with nothing behind it: the `range()`
    loop itself is cheap to construct at any size, but the FIRST tensor-info
    read (its name's length prefix) must refuse immediately once bytes run
    out, rather than iterating any meaningful number of times."""
    path = os.path.join(str(tmp_path), "huge_tensor_count.gguf")
    with open(path, "wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<I", 3))
        fh.write(struct.pack("<Q", 2 ** 62))  # tensor_count
        fh.write(struct.pack("<Q", 0))  # metadata_kv_count
    import time
    t0 = time.time()
    with pytest.raises(g.GGUFFormatError):
        g.parse_gguf_header(path)
    assert time.time() - t0 < 2.0


# ---------------------------------------------------------------------------
# F3: `general.alignment` of an unexpected TYPE.
# ---------------------------------------------------------------------------

def test_refuses_string_typed_alignment(tmp_path):
    path = os.path.join(str(tmp_path), "bad_alignment_str.gguf")
    write_gguf(path, {"x.weight": torch.randn(1)}, {"general.alignment": "not a number"})
    with pytest.raises(g.GGUFFormatError, match="general.alignment"):
        g.parse_gguf_header(path)


def test_refuses_array_typed_alignment(tmp_path):
    from tests.minimax_music3_gguf_fixture import GGUFArrayValue, T_UINT32

    path = os.path.join(str(tmp_path), "bad_alignment_array.gguf")
    write_gguf(
        path, {"x.weight": torch.randn(1)},
        {"general.alignment": GGUFArrayValue(T_UINT32, [32, 64])},
    )
    with pytest.raises(g.GGUFFormatError, match="general.alignment"):
        g.parse_gguf_header(path)


def test_refuses_bool_typed_alignment(tmp_path):
    """`bool` is technically an `int` subclass in Python -- must not sneak
    past the type guard as if it were a real alignment value."""
    from tests.minimax_music3_gguf_fixture import GGUFValue, T_BOOL

    path = os.path.join(str(tmp_path), "bad_alignment_bool.gguf")
    write_gguf(path, {"x.weight": torch.randn(1)}, {"general.alignment": GGUFValue(T_BOOL, True)})
    with pytest.raises(g.GGUFFormatError, match="general.alignment"):
        g.parse_gguf_header(path)


# ---------------------------------------------------------------------------
# F4: nested metadata arrays -- depth limit and element-count bound.
# ---------------------------------------------------------------------------

def _nested_array(depth: int, elem_type_id: int = 4, leaf_value: int = 7) -> "object":
    from tests.minimax_music3_gguf_fixture import GGUFArrayValue, T_ARRAY

    value = leaf_value
    for _ in range(depth):
        value = GGUFArrayValue(T_ARRAY if isinstance(value, GGUFArrayValue) else elem_type_id, [value])
    return value


def test_nested_array_within_depth_limit_round_trips(tmp_path):
    """Two levels of nesting is well within the limit and must decode to the
    expected nested Python list, not merely "not raise"."""
    nested = _nested_array(2)
    path = os.path.join(str(tmp_path), "nested_ok.gguf")
    write_gguf(path, {"x.weight": torch.randn(1)}, {"nested": nested})
    header = g.parse_gguf_header(path)
    assert header.metadata["nested"] == [[7]]


def test_nested_array_exceeding_depth_limit_is_refused_not_a_recursion_error(tmp_path):
    """20 nested ARRAY levels (the depth limit is 8): must raise
    `GGUFFormatError`, never `RecursionError` -- a `RecursionError` is not a
    `GGUFFormatError` and would escape both music3 loader builders."""
    nested = _nested_array(20)
    path = os.path.join(str(tmp_path), "nested_too_deep.gguf")
    write_gguf(path, {"x.weight": torch.randn(1)}, {"nested": nested})
    with pytest.raises(g.GGUFFormatError, match="nesting exceeds"):
        g.parse_gguf_header(path)


def test_array_element_count_bounded_by_remaining_file_bytes(tmp_path):
    """A metadata array declaring far more elements than the file has bytes
    left: must refuse via the bound check, not attempt to build the list."""
    path = os.path.join(str(tmp_path), "huge_array.gguf")
    with open(path, "wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<I", 3))
        fh.write(struct.pack("<Q", 0))  # tensor_count
        fh.write(struct.pack("<Q", 1))  # metadata_kv_count
        key = b"k"
        fh.write(struct.pack("<Q", len(key)) + key)
        fh.write(struct.pack("<I", 9))  # ARRAY
        fh.write(struct.pack("<I", 0))  # element type UINT8 (1 byte/element)
        fh.write(struct.pack("<Q", 2 ** 40))  # declared element count
    import time
    t0 = time.time()
    with pytest.raises(g.GGUFFormatError, match="more than the"):
        g.parse_gguf_header(path)
    assert time.time() - t0 < 2.0


# ---------------------------------------------------------------------------
# F5/F6: all 13 metadata value types, duplicate keys, non-UTF-8 strings, and
# more structural-boundary truncations.
# ---------------------------------------------------------------------------

def test_all_13_metadata_value_types_round_trip(tmp_path):
    from tests.minimax_music3_gguf_fixture import (
        GGUFArrayValue, GGUFValue, T_ARRAY, T_BOOL, T_FLOAT32, T_FLOAT64,
        T_INT8, T_INT16, T_INT32, T_INT64, T_STRING, T_UINT8, T_UINT16,
        T_UINT32, T_UINT64,
    )

    metadata = {
        "v_uint8": GGUFValue(T_UINT8, 200),
        "v_int8": GGUFValue(T_INT8, -100),
        "v_uint16": GGUFValue(T_UINT16, 60000),
        "v_int16": GGUFValue(T_INT16, -30000),
        "v_uint32": GGUFValue(T_UINT32, 4_000_000_000),
        "v_int32": GGUFValue(T_INT32, -2_000_000_000),
        "v_float32": GGUFValue(T_FLOAT32, 3.5),
        "v_bool": GGUFValue(T_BOOL, True),
        "v_string": GGUFValue(T_STRING, "hello gguf"),
        "v_array": GGUFArrayValue(T_UINT32, [1, 2, 3]),
        "v_uint64": GGUFValue(T_UINT64, 18_000_000_000_000_000_000),
        "v_int64": GGUFValue(T_INT64, -9_000_000_000_000_000_000),
        "v_float64": GGUFValue(T_FLOAT64, 2.718281828),
    }
    assert len(metadata) == 13, "one entry per GGUF metadata value type"

    path = os.path.join(str(tmp_path), "all_types.gguf")
    write_gguf(path, {"x.weight": torch.randn(1)}, metadata)
    header = g.parse_gguf_header(path)

    assert header.metadata["v_uint8"] == 200
    assert header.metadata["v_int8"] == -100
    assert header.metadata["v_uint16"] == 60000
    assert header.metadata["v_int16"] == -30000
    assert header.metadata["v_uint32"] == 4_000_000_000
    assert header.metadata["v_int32"] == -2_000_000_000
    assert header.metadata["v_float32"] == pytest.approx(3.5)
    assert header.metadata["v_bool"] is True
    assert header.metadata["v_string"] == "hello gguf"
    assert header.metadata["v_array"] == [1, 2, 3]
    assert header.metadata["v_uint64"] == 18_000_000_000_000_000_000
    assert header.metadata["v_int64"] == -9_000_000_000_000_000_000
    assert header.metadata["v_float64"] == pytest.approx(2.718281828)


def test_refuses_duplicate_metadata_keys(tmp_path):
    path = os.path.join(str(tmp_path), "dup_meta.gguf")
    header_bytes = bytearray()
    header_bytes += b"GGUF"
    header_bytes += struct.pack("<I", 3)
    header_bytes += struct.pack("<Q", 0)  # tensor_count
    header_bytes += struct.pack("<Q", 2)  # metadata_kv_count
    for _ in range(2):
        key = b"dup.key"
        header_bytes += struct.pack("<Q", len(key)) + key
        header_bytes += struct.pack("<I", 4)  # UINT32
        header_bytes += struct.pack("<I", 1)
    with open(path, "wb") as fh:
        fh.write(bytes(header_bytes))
    with pytest.raises(g.GGUFFormatError, match="duplicate metadata key"):
        g.parse_gguf_header(path)


def test_refuses_non_utf8_metadata_key(tmp_path):
    path = os.path.join(str(tmp_path), "bad_utf8_key.gguf")
    with open(path, "wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<I", 3))
        fh.write(struct.pack("<Q", 0))  # tensor_count
        fh.write(struct.pack("<Q", 1))  # metadata_kv_count
        bad = b"\xff\xfe\xfd"  # not valid UTF-8
        fh.write(struct.pack("<Q", len(bad)) + bad)
        fh.write(struct.pack("<I", 4))  # UINT32
        fh.write(struct.pack("<I", 1))
    with pytest.raises(g.GGUFFormatError, match="not valid UTF-8"):
        g.parse_gguf_header(path)


def test_refuses_non_utf8_tensor_name(tmp_path):
    path = os.path.join(str(tmp_path), "bad_utf8_tensor.gguf")
    header_bytes = bytearray()
    header_bytes += b"GGUF"
    header_bytes += struct.pack("<I", 3)
    header_bytes += struct.pack("<Q", 1)  # tensor_count
    header_bytes += struct.pack("<Q", 0)  # metadata_kv_count
    bad = b"\xff\xfe\xfd"
    header_bytes += struct.pack("<Q", len(bad)) + bad
    header_bytes += struct.pack("<I", 1)  # n_dims
    header_bytes += struct.pack("<Q", 1)  # dims[0]
    header_bytes += struct.pack("<I", 0)  # F32
    header_bytes += struct.pack("<Q", 0)  # rel_offset
    with open(path, "wb") as fh:
        fh.write(bytes(header_bytes))
        fh.write(b"\x00" * 64)
    with pytest.raises(g.GGUFFormatError, match="not valid UTF-8"):
        g.parse_gguf_header(path)


def test_truncation_at_every_structural_boundary_of_a_real_header_is_refused(tmp_path):
    """Build one real, valid header (metadata + two tensors, so every field
    kind appears at least once), then truncate the FILE at every possible
    byte length from 0 up to (but not including) the full header. Every one
    of those truncations must raise `GGUFFormatError` -- never succeed,
    never raise anything else (a `struct.error`, an `IndexError`, ...) --
    covering every structural boundary (magic, version, counts, each
    metadata field, each tensor-info field) by construction rather than by
    naming each one."""
    from tests.minimax_music3_gguf_fixture import GGUFArrayValue, T_UINT32

    full_path = os.path.join(str(tmp_path), "full.gguf")
    write_gguf(
        full_path,
        {"a.weight": torch.randn(2, 3), "b.weight": torch.randn(4).to(torch.float16)},
        {"general.architecture": "minimax_music3", "some.array": GGUFArrayValue(T_UINT32, [1, 2])},
    )
    header = g.parse_gguf_header(full_path)
    with open(full_path, "rb") as fh:
        full_bytes = fh.read()

    truncated_path = os.path.join(str(tmp_path), "trunc.gguf")
    # Every length up to (not including) the start of the tensor DATA section
    # -- beyond that point a truncation only shortens tensor bytes, which
    # `parse_gguf_header` (header-only) cannot see at all; that class of
    # truncation is `GGUFStateDict`'s concern (covered by the file-size
    # re-validation test in the loader test file), not this function's.
    for cut in range(0, header.data_start):
        with open(truncated_path, "wb") as fh:
            fh.write(full_bytes[:cut])
        try:
            g.parse_gguf_header(truncated_path)
        except g.GGUFFormatError:
            continue
        except Exception as exc:  # pragma: no cover - failure path
            raise AssertionError(
                f"truncation at {cut} byte(s) raised {type(exc).__name__}, not GGUFFormatError"
            ) from exc
        else:
            raise AssertionError(f"truncation at {cut} byte(s) did not raise at all")
