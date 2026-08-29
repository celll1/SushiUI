"""The hybrid tensor reader: base-only parity, and per-key source selection.

Run with (explicit path -- a broad `-k minimax_h3` selector also picks up tests
that map real 12-21 GB checkpoints):

    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_hybrid_reader_test.py -v

Contract: `docs/guides/MINIMAX_H3_HYBRID_LOADER_DESIGN.md`. No test here opens
a distributed checkpoint: the "real file" fixtures are a few hundred KB of tiny tensors written
with `safetensors.torch.save_file`, which is enough for a genuine `safe_open`
mmap and therefore enough to prove parity against one.
"""

import dataclasses
import json
import os
import sys

import pytest
import torch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(_TESTS_DIR)
for _p in (_TESTS_DIR, BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from minimax_h3_model_listing_test import _build_h3_tree  # noqa: E402

from core.models.minimax_h3.hybrid_reader import (  # noqa: E402
    HybridTensorReader,
    SingleTensorReader,
    open_dit_reader,
)
from core.models.minimax_h3.hybrid_spec import (  # noqa: E402
    BASE,
    OVERLAY,
    BlockRangeAdalnSelector,
    preflight_minimax_h3_hybrid,
)
from core.models.minimax_h3 import loader  # noqa: E402
from core.models.minimax_h3.loader import (  # noqa: E402
    _build_transformer,
    _int8_convrot_layers_from_markers,
    _map_dit_state_dict,
    _rename_dit_key,
    _verify_hybrid_overlay_reads,
    read_safetensors_header,
)


HEADS, HEAD_DIM = 2, 4
INNER = HEADS * HEAD_DIM
CONFIG = {"num_attention_heads": HEADS, "attention_head_dim": HEAD_DIM, "adaln_curve_grid": 1}

CONVROT_MARKER = b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
PLAIN_MARKER = b'{"format": "float8_e4m3fn_scaled"}'


def _marker(payload: bytes) -> torch.Tensor:
    return torch.tensor(list(payload), dtype=torch.uint8)


# ---------------------------------------------------------------------------
# fixtures: a small but structurally complete DiT
# ---------------------------------------------------------------------------

def _dit_tensors(num_blocks: int, *, offset: float, adaln_bias: bool = True,
                 quant_marker: bool = True):
    """Every value transform `_map_dit_state_dict` performs, in miniature.

    `offset` shifts every value so a base file and an overlay file with the same
    header have different contents everywhere -- which is what makes "this tensor
    came from the overlay" checkable.
    """
    def bf16(shape, seed):
        g = torch.Generator().manual_seed(seed)
        return (torch.rand(shape, generator=g) + offset).to(torch.bfloat16)

    tensors = {
        # `token_refiner.` is half of the H3 single-file signature.
        "token_refiner.blocks.0.attn.qkv_proj.weight": bf16((3 * INNER, INNER), 1),
        "adaln_t_table": (torch.arange(16 * 8, dtype=torch.float32).view(16, 8) + offset),
        "rope.inv_freq": torch.arange(4, dtype=torch.float32),  # dropped by policy
        "final_layer.adaln_proj.linear.weight": bf16((12, 8), 2),
        # The three projections `_synthesize_transformer_config` derives the
        # geometry from; 12 video patch channels over a 1x2x2 patch.
        "video_patch_proj.weight": bf16((INNER, 12), 3),
        "audio_patch_proj.weight": bf16((INNER, 4), 5),
        "condition_proj.weight": bf16((INNER, 6), 6),
    }
    if adaln_bias:
        tensors["final_layer.adaln_proj.linear.bias"] = bf16((12,), 4)
    for n in range(num_blocks):
        tensors[f"blocks.{n}.attn.qkv_proj.weight"] = bf16((3 * INNER, INNER), 100 + n)
        tensors[f"blocks.{n}.mlp.fc1.weight"] = bf16((2 * INNER, INNER), 200 + n)
        tensors[f"blocks.{n}.mlp.fc2.weight"] = bf16((INNER, INNER), 300 + n)
        tensors[f"blocks.{n}.adaln_proj.linear.weight"] = bf16((12, 8), 400 + n)
        if adaln_bias:
            tensors[f"blocks.{n}.adaln_proj.linear.bias"] = bf16((12,), 500 + n)
    # A scaled layer plus its dropped `.input_scale`, and a provenance marker.
    tensors["blocks.0.mlp.fc2.weight_scale"] = torch.tensor([0.5 + offset], dtype=torch.float32)
    tensors["blocks.0.mlp.fc2.input_scale"] = torch.tensor([0.25], dtype=torch.float32)
    if quant_marker:
        # An fp8 declaration over a bf16 weight, which is exactly what
        # `_guard_component_file` refuses -- so the files that reach the real
        # guard (`_real_pair`) leave it out.
        tensors["blocks.0.mlp.fc2.comfy_quant"] = _marker(PLAIN_MARKER)
    return tensors


def _write_dit(path, tensors, metadata=None):
    save_file(tensors, str(path), metadata=metadata or {"format": "pt"})
    return str(path)


def _header_of(path):
    header = read_safetensors_header(path)
    header.pop("__metadata__", None)
    return header


class _RecordingHandle:
    """A `safe_open`-shaped stand-in that remembers what it was asked for."""

    def __init__(self, tensors, name):
        self.tensors = tensors
        self.name = name
        self.requested = []

    def get_tensor(self, key):
        self.requested.append(key)
        return self.tensors[key]


def _fake_pair(num_blocks=50, **kwargs):
    base = _RecordingHandle(_dit_tensors(num_blocks, offset=0.0, **kwargs), "base")
    overlay = _RecordingHandle(_dit_tensors(num_blocks, offset=10.0, **kwargs), "overlay")
    return base, overlay


# ---------------------------------------------------------------------------
# base-only parity: a reader is substitutable for the handle it wraps
# ---------------------------------------------------------------------------

def test_single_reader_returns_the_handles_own_tensor_without_copying(tmp_path):
    path = _write_dit(tmp_path / "dit.safetensors", _dit_tensors(3, offset=0.0))
    with safe_open(path, framework="pt", device="cpu") as handle:
        reader = SingleTensorReader(handle)
        for key in _header_of(path):
            direct, through = handle.get_tensor(key), reader.get_tensor(key)
            assert through.dtype == direct.dtype and through.shape == direct.shape
            # Same mmap page: the reader adds no mapping and copies nothing.
            assert through.data_ptr() == direct.data_ptr()
            assert torch.equal(through, direct)


def test_base_only_mapping_through_the_reader_is_identical_to_the_direct_handle(tmp_path):
    """The C3 parity claim, measured rather than asserted.

    Same file, same header, same config: the state dict `_map_dit_state_dict`
    produces through `SingleTensorReader` is compared to the one it produces
    through the raw handle, key for key, dtype for dtype, bit for bit -- and the
    transform statistics with it.
    """
    path = _write_dit(tmp_path / "dit.safetensors", _dit_tensors(6, offset=0.0))
    header = _header_of(path)

    with safe_open(path, framework="pt", device="cpu") as handle:
        direct, direct_stats = _map_dit_state_dict(handle, header, CONFIG, torch.bfloat16)
        through, through_stats = _map_dit_state_dict(
            SingleTensorReader(handle), header, CONFIG, torch.bfloat16)

    assert direct_stats == through_stats
    assert direct_stats["qkv_split"] and direct_stats["swiglu_swapped"]
    assert direct_stats["input_scale_dropped"] == 1 and direct_stats["dropped"] == 1
    assert direct_stats["markers"] == 1 and direct_stats["scales_broadcast"] == 1
    assert set(direct) == set(through)
    for key, value in direct.items():
        other = through[key]
        assert other.dtype == value.dtype and other.shape == value.shape, key
        assert torch.equal(other, value), key


def test_convrot_markers_validate_identically_through_the_reader(tmp_path):
    tensors = _dit_tensors(2, offset=0.0)
    tensors["blocks.1.mlp.fc2.weight"] = torch.zeros(8, 256, dtype=torch.int8)
    tensors["blocks.1.mlp.fc2.weight_scale"] = torch.ones(8, 1, dtype=torch.float32)
    tensors["blocks.1.mlp.fc2.comfy_quant"] = _marker(CONVROT_MARKER)
    path = _write_dit(tmp_path / "dit.safetensors", tensors)
    header = _header_of(path)

    with safe_open(path, framework="pt", device="cpu") as handle:
        direct = _int8_convrot_layers_from_markers(handle, header, path=path)
        through = _int8_convrot_layers_from_markers(
            SingleTensorReader(handle), header, path=path)

    assert direct == through
    assert set(direct) == {"blocks.1.mlp.fc2"}


def test_open_dit_reader_without_an_overlay_yields_a_single_reader(tmp_path):
    path = _write_dit(tmp_path / "dit.safetensors", _dit_tensors(1, offset=0.0))
    with open_dit_reader(path) as reader:
        assert isinstance(reader, SingleTensorReader) and not reader.is_hybrid
        assert torch.equal(reader.get_tensor("rope.inv_freq"),
                           torch.arange(4, dtype=torch.float32))


# ---------------------------------------------------------------------------
# source selection: exactly the AdaLN of blocks 25..49 comes from the overlay
# ---------------------------------------------------------------------------

def _selector(start=25, end=49, **kwargs):
    return BlockRangeAdalnSelector(block_range_start=start, block_range_end=end, **kwargs)


def _expected_overlay_keys(start, end, *, bias, final=False):
    keys = {f"blocks.{n}.adaln_proj.linear.weight" for n in range(start, end + 1)}
    if bias:
        keys |= {f"blocks.{n}.adaln_proj.linear.bias" for n in range(start, end + 1)}
    if final:
        keys |= {"final_layer.adaln_proj.linear.weight"}
        if bias:
            keys |= {"final_layer.adaln_proj.linear.bias"}
    return keys


@pytest.mark.parametrize("overlay_bias", [False, True])
def test_mapping_reads_only_the_selected_adaln_from_the_overlay(overlay_bias):
    base, overlay = _fake_pair()
    header = {k: {"dtype": "BF16", "shape": list(v.shape)} for k, v in base.tensors.items()}
    reader = HybridTensorReader(base, overlay, _selector(overlay_bias=overlay_bias))

    mapped, _stats = _map_dit_state_dict(reader, header, CONFIG, torch.bfloat16)

    expected = _expected_overlay_keys(25, 49, bias=overlay_bias)
    assert set(overlay.requested) == expected
    assert reader.overlay_keys_read == expected
    # Nothing was read twice from the wrong file, and the base saw everything else.
    assert expected.isdisjoint(base.requested)
    assert set(base.requested) | expected == {
        k for k in header if k != "rope.inv_freq" and not k.endswith(".input_scale")}

    # ...and the overlay's VALUES are the ones that landed in the state dict.
    for n in (25, 37, 49):
        key = f"blocks.{n}.adaln_proj.linear.weight"
        assert torch.equal(mapped[_rename_dit_key(key)],
                           overlay.tensors[key].to(torch.float32))
    for n in (0, 24):
        key = f"blocks.{n}.adaln_proj.linear.weight"
        assert torch.equal(mapped[_rename_dit_key(key)],
                           base.tensors[key].to(torch.float32))


def test_the_final_layer_adaln_stays_base_unless_its_own_toggle_is_set():
    base, overlay = _fake_pair()
    header = {k: {"dtype": "BF16", "shape": list(v.shape)} for k, v in base.tensors.items()}

    reader = HybridTensorReader(base, overlay, _selector(overlay_bias=True))
    _map_dit_state_dict(reader, header, CONFIG, torch.bfloat16)
    assert "final_layer.adaln_proj.linear.weight" not in reader.overlay_keys_read

    base2, overlay2 = _fake_pair()
    reader2 = HybridTensorReader(
        base2, overlay2, _selector(overlay_bias=True, final_adaln_from_overlay=True))
    _map_dit_state_dict(reader2, header, CONFIG, torch.bfloat16)
    assert reader2.overlay_keys_read == _expected_overlay_keys(25, 49, bias=True, final=True)


def test_marker_reads_follow_their_weight_through_the_same_reader():
    """A `.comfy_quant` on a SELECTED AdaLN Linear is an overlay read.

    The second handle consumer: `_int8_convrot_layers_from_markers` reads marker
    CONTENTS, and reading them from the base while the weight comes from the
    overlay pins provenance to the wrong file.
    """
    base, overlay = _fake_pair(num_blocks=50)
    for handle in (base, overlay):
        handle.tensors["blocks.30.adaln_proj.linear.comfy_quant"] = _marker(PLAIN_MARKER)
    header = {k: {"dtype": "BF16", "shape": list(v.shape)} for k, v in base.tensors.items()}
    reader = HybridTensorReader(base, overlay, _selector(overlay_bias=True))

    _int8_convrot_layers_from_markers(reader, header, path="fixture.safetensors")

    assert "blocks.30.adaln_proj.linear.comfy_quant" in reader.overlay_keys_read
    assert "blocks.0.mlp.fc2.comfy_quant" in base.requested
    assert "blocks.30.adaln_proj.linear.comfy_quant" not in base.requested


def test_a_selector_that_answers_neither_base_nor_overlay_is_refused():
    class _Broken:
        def source_for(self, key):
            return "either"

    base, overlay = _fake_pair(num_blocks=1)
    reader = HybridTensorReader(base, overlay, _Broken())
    with pytest.raises(ValueError, match="must be 'base' or 'overlay'"):
        reader.get_tensor("blocks.0.mlp.fc2.weight")
    assert BASE == "base" and OVERLAY == "overlay"


def test_a_hybrid_reader_without_a_selector_is_refused():
    base, overlay = _fake_pair(num_blocks=1)
    with pytest.raises(ValueError, match="needs a selector"):
        HybridTensorReader(base, overlay, None)


# ---------------------------------------------------------------------------
# the realised selection is checked against the preflight's
# ---------------------------------------------------------------------------

class _Preflight:
    def __init__(self, overlay_keys):
        self.overlay_keys = tuple(overlay_keys)


def test_a_realised_selection_that_differs_from_the_preflight_is_refused():
    base, overlay = _fake_pair(num_blocks=50)
    header = {k: {"dtype": "BF16", "shape": list(v.shape)} for k, v in base.tensors.items()}
    reader = HybridTensorReader(base, overlay, _selector(overlay_bias=False))
    _map_dit_state_dict(reader, header, CONFIG, torch.bfloat16)

    expected = _expected_overlay_keys(25, 49, bias=False)
    _verify_hybrid_overlay_reads(reader, _Preflight(expected), path="fixture.safetensors")

    with pytest.raises(RuntimeError, match="not read"):
        _verify_hybrid_overlay_reads(
            reader, _Preflight(expected | {"blocks.3.adaln_proj.linear.weight"}),
            path="fixture.safetensors")
    with pytest.raises(RuntimeError, match="read but not selected"):
        _verify_hybrid_overlay_reads(
            reader, _Preflight(expected - {"blocks.25.adaln_proj.linear.weight"}),
            path="fixture.safetensors")


# ---------------------------------------------------------------------------
# end to end over real (tiny) files: preflight -> reader -> mapped state dict
# ---------------------------------------------------------------------------

def _real_pair(tmp_path, *, num_blocks=50):
    root = str(tmp_path / "h3")
    _build_h3_tree(root)
    base = os.path.join(root, "diffusion_models",
                        "minimax_h3_fl2va_pruned_fp8_scaled.safetensors")
    overlay = os.path.join(root, "diffusion_models",
                           "minimax_h3_ref2va_pruned_fp8_scaled.safetensors")
    _write_dit(base, _dit_tensors(num_blocks, offset=0.0, quant_marker=False))
    _write_dit(overlay, _dit_tensors(num_blocks, offset=10.0, quant_marker=False))
    # The head split cannot be read off the header, so the config tree supplies
    # the fixture's tiny one.
    transformer_dir = os.path.join(root, "official", "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    with open(os.path.join(transformer_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"num_attention_heads": HEADS, "attention_head_dim": HEAD_DIM,
                   "patch_size": [1, 2, 2]}, fh)
    return base, overlay, os.path.join(root, "official")


def test_preflight_selection_and_reader_agree_on_real_files(tmp_path):
    base, overlay, _official = _real_pair(tmp_path)
    preflight = preflight_minimax_h3_hybrid(base, overlay,
                                            block_range_start=25, block_range_end=49)
    assert preflight.overlay_bias_eligible
    assert set(preflight.overlay_keys) == _expected_overlay_keys(25, 49, bias=True)

    header = _header_of(base)
    with open_dit_reader(base, overlay_path=overlay, selector=preflight.selector) as reader:
        assert isinstance(reader, HybridTensorReader)
        mapped, _stats = _map_dit_state_dict(reader, header, CONFIG, torch.bfloat16)
        _verify_hybrid_overlay_reads(reader, preflight, path=base)

    with safe_open(base, framework="pt", device="cpu") as bh, \
            safe_open(overlay, framework="pt", device="cpu") as oh:
        for n in (24, 25, 49):
            key = f"blocks.{n}.adaln_proj.linear.weight"
            source = oh if 25 <= n <= 49 else bh
            assert torch.equal(mapped[_rename_dit_key(key)],
                               source.get_tensor(key).to(torch.float32)), key
        # A non-AdaLN tensor of an in-range block still comes from the base.
        key = "blocks.30.mlp.fc2.weight"
        assert torch.equal(mapped[_rename_dit_key(key)], bh.get_tensor(key))


def test_building_from_a_file_the_preflight_did_not_validate_is_refused(tmp_path):
    base, overlay, _official = _real_pair(tmp_path, num_blocks=6)
    preflight = preflight_minimax_h3_hybrid(base, overlay,
                                            block_range_start=2, block_range_end=3)
    other = _write_dit(tmp_path / "elsewhere.safetensors", _dit_tensors(6, offset=0.0))
    with pytest.raises(ValueError, match="validated"):
        _build_transformer(other, torch.bfloat16, None, hybrid=preflight)


class _StopAfterReader(Exception):
    pass


_PROBE_KEYS = ("blocks.2.adaln_proj.linear.weight", "blocks.0.adaln_proj.linear.weight")


def _capture_reader(monkeypatch, *, dit_path, official, hybrid=None):
    """Run `_build_transformer` up to the mapping; report the reader it built.

    The probe values are read INSIDE the reader's context -- the handles are
    closed when it exits, which is the point of opening them there.
    """
    seen = {}

    def fake_markers(handle, header, *, path):
        seen["markers"] = handle
        return {}

    def fake_map(handle, header, config, dtype, **kwargs):
        seen["map"] = handle
        seen["probe"] = {k: handle.get_tensor(k).float().mean().item() for k in _PROBE_KEYS}
        raise _StopAfterReader()

    monkeypatch.setattr(loader, "_int8_convrot_layers_from_markers", fake_markers)
    monkeypatch.setattr(loader, "_map_dit_state_dict", fake_map)
    with pytest.raises(_StopAfterReader):
        loader._build_transformer(dit_path, torch.bfloat16, official, hybrid=hybrid)
    assert seen["markers"] is seen["map"], "the two handle consumers got different readers"
    return seen["map"], seen["probe"]


def test_build_transformer_gives_both_consumers_one_reader(tmp_path, monkeypatch):
    """The wiring itself: one reader object, both consumers, no bare handle."""
    base, overlay, official = _real_pair(tmp_path, num_blocks=6)

    single, base_probe = _capture_reader(monkeypatch, dit_path=base, official=official)
    assert isinstance(single, SingleTensorReader)
    assert all(value < 5 for value in base_probe.values())  # base offset 0.0

    preflight = preflight_minimax_h3_hybrid(base, overlay,
                                            block_range_start=2, block_range_end=3)
    reader, probe = _capture_reader(monkeypatch, dit_path=base, official=official,
                                    hybrid=preflight)
    assert isinstance(reader, HybridTensorReader)
    assert probe["blocks.2.adaln_proj.linear.weight"] > 5  # in range -> overlay (offset 10.0)
    assert probe["blocks.0.adaln_proj.linear.weight"] < 5  # out of range -> base


def test_both_files_reach_the_declared_semantics_guard(tmp_path, monkeypatch):
    """The preflight compares headers; only this guard reads marker CONTENTS."""
    base, overlay, official = _real_pair(tmp_path, num_blocks=6)
    preflight = preflight_minimax_h3_hybrid(base, overlay,
                                            block_range_start=2, block_range_end=3)
    real_guard = loader._guard_component_file
    guarded = []

    def spy(path, **kwargs):
        guarded.append(path)
        return real_guard(path, **kwargs)

    monkeypatch.setattr(loader, "_guard_component_file", spy)
    _capture_reader(monkeypatch, dit_path=base, official=official, hybrid=preflight)
    assert guarded == [base, overlay]


# ---------------------------------------------------------------------------
# the spec reaching the loader must be the one the preflight produced
# ---------------------------------------------------------------------------

def test_an_overlay_replaced_after_preflight_is_refused(tmp_path, monkeypatch):
    """Doc section 7's second identity check, at the point the file is opened.

    The realised-read check cannot see this -- it derives its expectations from
    the BASE header -- and neither can the strict load.
    """
    base, overlay, official = _real_pair(tmp_path, num_blocks=6)
    preflight = preflight_minimax_h3_hybrid(base, overlay,
                                            block_range_start=2, block_range_end=3)

    # An added tensor: same shapes everywhere else, so only the census sees it.
    replaced = _dit_tensors(6, offset=10.0, quant_marker=False)
    replaced["blocks.0.extra.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)
    _write_dit(overlay, replaced)
    with pytest.raises(RuntimeError, match="changed between preflight and load"):
        _capture_reader(monkeypatch, dit_path=base, official=official, hybrid=preflight)

    # A shape change on one tensor, and a swap of the BASE file, likewise.
    replaced = _dit_tensors(6, offset=10.0, quant_marker=False)
    replaced["blocks.0.adaln_proj.linear.weight"] = torch.zeros(12, 8, dtype=torch.float32)
    _write_dit(overlay, replaced)
    with pytest.raises(RuntimeError, match="changed between preflight and load"):
        _capture_reader(monkeypatch, dit_path=base, official=official, hybrid=preflight)


def test_a_same_shape_value_only_rewrite_is_NOT_detected(tmp_path, monkeypatch):
    """The measured limit of the header-level identity check.

    Rewriting the overlay with the same keys, shapes and dtypes produces a
    byte-identical header and an identical file size, so the digest is unchanged
    and the load proceeds. Catching it would mean hashing 12-21 GB of tensor data
    on every load. Asserted rather than left implied, so nobody reads the check
    above as "the file cannot have changed".
    """
    base, overlay, official = _real_pair(tmp_path, num_blocks=6)
    preflight = preflight_minimax_h3_hybrid(base, overlay,
                                            block_range_start=2, block_range_end=3)
    _write_dit(overlay, _dit_tensors(6, offset=99.0, quant_marker=False))

    _reader, probe = _capture_reader(monkeypatch, dit_path=base, official=official,
                                     hybrid=preflight)
    assert probe["blocks.2.adaln_proj.linear.weight"] > 50  # the REPLACED values


def test_an_unvalidated_or_overlay_less_spec_is_refused(tmp_path):
    base, overlay, official = _real_pair(tmp_path, num_blocks=6)
    preflight = preflight_minimax_h3_hybrid(base, overlay,
                                            block_range_start=2, block_range_end=3)

    unvalidated = dataclasses.replace(preflight, spec=dataclasses.replace(
        preflight.spec, compatibility_digest=None))
    with pytest.raises(ValueError, match="never went through"):
        _build_transformer(base, torch.bfloat16, official, hybrid=unvalidated)

    overlay_less = dataclasses.replace(preflight, spec=dataclasses.replace(
        preflight.spec, overlay_dit_path=""))
    with pytest.raises(ValueError, match="names no overlay"):
        _build_transformer(base, torch.bfloat16, official, hybrid=overlay_less)


def test_a_hybrid_load_that_ran_on_a_single_reader_is_refused():
    base, _overlay = _fake_pair(num_blocks=1)
    with pytest.raises(RuntimeError, match="reads one file"):
        _verify_hybrid_overlay_reads(SingleTensorReader(base), _Preflight(()),
                                     path="fixture.safetensors")
