"""Block swap must never pair two Linear weights of different dtypes.

THE DEFECT. Both offloaders in ``core/memory_management`` swap a pair of blocks
by pairing their Linear modules BY NAME AND SHAPE and exchanging the two weight
tensors through staging buffers allocated with ``empty_like`` from one side.
``Tensor.copy_`` converts between dtypes silently, so a pair whose shapes match
and whose dtypes do not writes int8 codes into bf16 storage (and bf16 values into
int8 storage) with no error and no warning. The quantized module then keeps
running -- ``Int8Linear._dequant_forward`` accepts a bf16 weight -- on numbers
that mean nothing. ``test_a_name_and_shape_only_pairing_casts_silently`` pins
that mechanism directly, so the guard cannot be removed on the belief that the
copy would have raised.

WHEN BLOCKS BECOME HETEROGENEOUS (both are cross-arch, and neither is exotic):

* a PARTIAL runtime INT8 conversion. ``apply_runtime_int8_quantization``
  explicitly designs for a failure part-way (CUDA OOM at layer N), sets
  ``manager._runtime_int8_partial`` and returns ``converted=False`` with the
  layers converted so far left as ``Int8Linear``. The blocks after the failure
  are untouched ``nn.Linear``, so the same module path is int8 in one block and
  bf16 in another.
* a COMPLETE conversion whose per-layer choice differed. int8-vs-e4m3 is decided
  from each layer's own weights, so the same path can be int8 in one block and
  float8_e4m3fn in another. Both are quantized, both have the same shape.

Every arch in ``RUNTIME_INT8_ARCHS`` that also supports block swap shares these
two offloaders, so the guard lives in the shared code and is inherited rather
than re-implemented per arch. ``quantized_capability_parity_test`` holds that
line (every offloader class must inherit the mixin).

WHAT THIS FILE DOES NOT COVER. No checkpoint is loaded and no CUDA memory is
allocated: the offloaders are built with ``device=cpu`` and the pairing is
exercised through ``_build_weight_swap_jobs``, which is the whole decision. The
CUDA stream/event plumbing around it is untested here and unchanged by the fix.
"""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.memory_management.block_offloading import (  # noqa: E402
    DtypeSplitGuardMixin, TransformerBlockOffloader, dtype_split_linear_paths,
    pairable_block_indices,
)
from core.memory_management.flux_block_offloading import FluxBlockOffloader  # noqa: E402
from core.models.ideogram4.vendor.int8_linear import Int8Linear  # noqa: E402

CPU = torch.device("cpu")


def _linear(kind: str) -> nn.Module:
    """One Linear-like module in the state the named conversion leaves it in."""
    if kind == "int8":
        module = Int8Linear(8, 8, bias=False, compute_dtype=torch.bfloat16)
        module.weight.data = torch.full((8, 8), 7, dtype=torch.int8)
        module.weight_scale.data = torch.full((8,), 0.01)
        return module
    if kind == "e4m3":
        module = Int8Linear(8, 8, bias=False, compute_dtype=torch.bfloat16)
        module.weight.data = torch.full((8, 8), 0.5).to(torch.float8_e4m3fn)
        module.weight_scale.data = torch.full((8,), 0.01)
        return module
    module = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    module.weight.data = torch.full((8, 8), 0.25, dtype=torch.bfloat16)
    # The transformer is frozen on every generate path (and for LoRA training).
    module.requires_grad_(False)
    return module


class _Block(nn.Module):
    """attn.to_q / attn.to_k / ff.net -- three paths, independently converted."""

    def __init__(self, to_q="bf16", to_k="bf16", net="bf16"):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = _linear(to_q)
        self.attn.to_k = _linear(to_k)
        self.ff = nn.Module()
        self.ff.net = _linear(net)


class _SingleBlock(_Block):
    """A second block class, standing in for FLUX.2's single-stream blocks."""


def _offloader(blocks, blocks_to_swap=None, supports_backward=True):
    """An offloader whose rotation can pair EVERY block, by default.

    ``PairingTest`` is about the pairing predicate, not about which blocks the
    rotation reaches, so the default (backward-enabled, maximum swap count) makes
    the resolution set cover all blocks -- see ``pairable_block_indices``.
    ``ResidentBlocksAreNotResolvedTest`` covers the restriction itself.
    """
    if blocks_to_swap is None:
        blocks_to_swap = len(blocks) - 1
    return TransformerBlockOffloader(
        blocks=nn.ModuleList(blocks), blocks_to_swap=blocks_to_swap, device=CPU,
        target_dtype=torch.bfloat16, supports_backward=supports_backward)


class _MetaTargetGuard(DtypeSplitGuardMixin):
    """``_move_deferred_pairs`` with a target device that is not the CPU.

    The offloaders under test are built with ``device=cpu`` so no CUDA memory is
    needed, but that also makes both halves of a deferred move no-ops, which would
    hide a self-pair being moved. ``meta`` is a device the move is observable on
    without allocating anything: a real->meta->cpu round trip raises
    ``NotImplementedError`` ("Cannot copy out of meta tensor"), which is exactly
    the "moved to the device and then straight back to CPU" sequence.
    """

    device = torch.device("meta")


def _paths(block, pairs, index):
    lookup = {id(m): n for n, m in block.named_modules()}
    return sorted(lookup[id(p[index])] for p in pairs)


class NameAndShapePairingIsNotEnoughTest(unittest.TestCase):
    """The mechanism, pinned independently of the code under test."""

    def test_a_name_and_shape_only_pairing_casts_silently(self):
        quantized = _linear("int8")     # the block on GPU, going to CPU
        plain = _linear("bf16")         # the block on CPU, coming to GPU
        self.assertEqual(quantized.weight.shape, plain.weight.shape)

        # Exactly the staging sequence swap_weight_devices runs, minus the CUDA
        # stream/event plumbing (which does not participate in dtype).
        cuda_data_view = quantized.weight.data
        cpu_data_view = plain.weight.data
        staging_a = torch.empty_like(cuda_data_view, device="cpu")
        staging_b = torch.empty_like(cuda_data_view, device="cpu")
        staging_a.copy_(cuda_data_view.data)
        staging_b.copy_(plain.weight.data)
        cuda_data_view.copy_(staging_b)
        cpu_data_view.copy_(staging_a)
        quantized.weight.data = cpu_data_view
        plain.weight.data = cuda_data_view

        self.assertIs(quantized.weight.dtype, torch.bfloat16,
                      "the int8 module kept its dtype -- the mechanism this guard "
                      "exists for has changed and the guard needs re-deriving")
        self.assertIs(plain.weight.dtype, torch.int8)
        # And it is silent: the quantized module still computes.
        out = quantized(torch.ones(1, 8, dtype=torch.bfloat16))
        self.assertTrue(torch.isfinite(out).all())


class DtypeSplitDetectionTest(unittest.TestCase):

    def test_partial_conversion_is_detected(self):
        converted = _Block("int8", "int8", "int8")
        untouched = _Block()
        split = dtype_split_linear_paths([converted, untouched])
        self.assertEqual(
            split, {"_Block": {"attn.to_q": ("torch.bfloat16", "torch.int8"),
                               "attn.to_k": ("torch.bfloat16", "torch.int8"),
                               "ff.net": ("torch.bfloat16", "torch.int8")}})

    def test_a_completed_conversion_that_diverged_on_one_layer_is_detected(self):
        split = dtype_split_linear_paths([
            _Block("int8", "int8", "bf16"), _Block("e4m3", "int8", "bf16")])
        self.assertEqual(list(split["_Block"]), ["attn.to_q"])

    def test_mixed_dtypes_within_a_block_are_not_a_split(self):
        """int8 + e4m3 + bf16 in one block is what a converted block LOOKS like."""
        kinds = ("int8", "e4m3", "bf16")
        self.assertEqual(dtype_split_linear_paths([_Block(*kinds), _Block(*kinds)]), {})

    def test_bf16_only_blocks_are_not_a_split(self):
        self.assertEqual(dtype_split_linear_paths([_Block(), _Block(), _Block()]), {})

    def test_block_classes_are_grouped_separately(self):
        """A dual block and a single block are never paired, so never compared."""
        self.assertEqual(
            dtype_split_linear_paths([_Block("int8", "int8", "int8"), _SingleBlock()]), {})


class PairingTest(unittest.TestCase):

    def test_heterogeneous_paths_are_excluded_and_moved_individually(self):
        converted = _Block("int8", "int8", "int8")
        untouched = _Block()
        offloader = _offloader([converted, untouched])

        jobs, deferred = offloader._build_weight_swap_jobs(converted, untouched)
        self.assertEqual(jobs, [], "a mismatched pair was still handed to the staging swap")
        self.assertEqual(_paths(untouched, deferred, 1),
                         ["attn.to_k", "attn.to_q", "ff.net"])

        offloader._move_deferred_pairs(deferred)
        self.assertIs(converted.attn.to_q.weight.dtype, torch.int8)
        self.assertIs(untouched.attn.to_q.weight.dtype, torch.bfloat16)
        self.assertEqual(converted.attn.to_q.weight.flatten()[0].item(), 7)
        self.assertEqual(untouched.attn.to_q.weight.flatten()[0].item(), 0.25)

    def test_only_the_diverging_path_is_excluded(self):
        a = _Block("int8", "int8", "bf16")
        b = _Block("e4m3", "int8", "bf16")
        offloader = _offloader([a, b])
        jobs, deferred = offloader._build_weight_swap_jobs(a, b)
        self.assertEqual(_paths(b, deferred, 1), ["attn.to_q"])
        self.assertEqual(_paths(b, jobs, 1), ["attn.to_k", "ff.net"])

    def test_a_complete_homogeneous_conversion_still_pairs_everything(self):
        kinds = ("int8", "e4m3", "bf16")
        a, b = _Block(*kinds), _Block(*kinds)
        offloader = _offloader([a, b])
        jobs, deferred = offloader._build_weight_swap_jobs(a, b)
        self.assertEqual(deferred, [])
        self.assertEqual(_paths(b, jobs, 1), ["attn.to_k", "attn.to_q", "ff.net"])

    def test_bf16_only_block_swap_is_unaffected(self):
        a, b = _Block(), _Block()
        offloader = _offloader([a, b])
        jobs, deferred = offloader._build_weight_swap_jobs(a, b)
        self.assertEqual(deferred, [])
        self.assertEqual(_paths(b, jobs, 1), ["attn.to_k", "attn.to_q", "ff.net"])
        self.assertEqual(offloader._dtype_split_paths, {})

    def test_the_job_list_is_identical_for_every_pair(self):
        """The cached staging buffers are allocated once from the FIRST swap.

        Deciding per pair (rather than from one set of paths resolved over all
        blocks) would make the job list length depend on which two blocks are
        swapping, and the cached buffers would then be zipped against a shifted
        job list.
        """
        blocks = [_Block("int8", "int8", "bf16"),
                  _Block("e4m3", "int8", "bf16"),
                  _Block("e4m3", "int8", "bf16")]
        offloader = _offloader(blocks)
        lengths = set()
        for i in range(len(blocks)):
            for j in range(len(blocks)):
                if i == j:
                    continue
                jobs, _deferred = offloader._build_weight_swap_jobs(blocks[i], blocks[j])
                lengths.add(tuple(_paths(blocks[j], jobs, 1)))
        self.assertEqual(lengths, {("attn.to_k", "ff.net")})

    def test_a_mismatch_appearing_after_resolution_raises(self):
        a, b = _Block(), _Block()
        offloader = _offloader([a, b])
        offloader._build_weight_swap_jobs(a, b)      # resolves an empty split map
        a.attn.to_q = _linear("int8")                # module tree changed underneath
        with self.assertRaises(RuntimeError) as caught:
            offloader._build_weight_swap_jobs(a, b)
        self.assertIn("attn.to_q", str(caught.exception))
        self.assertIn("different dtypes", str(caught.exception))

    def test_a_self_pair_is_left_alone(self):
        """``blocks_to_swap == 1`` collapses the forward-only rotation.

        ``block_idx_to_gpu`` then equals ``block_idx_to_cpu``, so the block is
        swapped with itself. The paired staging path is a self-to-self no-op that
        leaves the block resident; the deferred path must be one too. Moving the
        weight to the device and then straight back to CPU would leave the block
        SPLIT ACROSS DEVICES and the next forward would raise "Expected all
        tensors to be on the same device".
        """
        block = _Block("int8", "bf16", "bf16")
        offloader = _offloader([block, _Block()])
        jobs, deferred = offloader._build_weight_swap_jobs(block, block)
        self.assertEqual(_paths(block, deferred, 1), ["attn.to_q"],
                         "the self-pair no longer reaches the deferred path; this test "
                         "is not exercising what it claims")

        before = block.attn.to_q.weight.data_ptr()
        _MetaTargetGuard()._move_deferred_pairs(deferred)
        self.assertEqual(block.attn.to_q.weight.data_ptr(), before,
                         "a self-swap moved the weight; the block is now split across "
                         "devices (to_q evicted, every paired path still resident)")
        self.assertEqual(block.attn.to_q.weight.device.type,
                         block.attn.to_k.weight.device.type)
        # And the offloader's own bound method behaves the same.
        offloader._move_deferred_pairs(deferred)
        self.assertEqual(block.attn.to_q.weight.data_ptr(), before)

    def test_shape_mismatches_keep_their_existing_handling(self):
        a, b = _Block(), _Block()
        b.attn.to_q = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
        b.attn.to_q.requires_grad_(False)
        offloader = _offloader([a, b])
        jobs, deferred = offloader._build_weight_swap_jobs(a, b)
        self.assertEqual(deferred, [])
        self.assertEqual(_paths(b, jobs, 1), ["attn.to_k", "ff.net"])


class ResidentBlocksAreNotResolvedTest(unittest.TestCase):
    """Only blocks the rotation can actually pair take part in the resolution.

    A block that never leaves the GPU is never one half of a pair, so a divergence
    confined to it is not a hazard -- excluding its paths would cost every swap the
    slower individual moves (and emit a warning) for nothing. Anima is exactly that
    case in inference: its only divergence is ``blocks.0.mlp.layer2``, and block 0
    is permanently resident (``transformer_registry`` clamps ``blocks_to_swap`` to
    ``num_blocks - 1``).
    """

    def test_the_pairable_range_matches_the_rotation(self):
        # forward-only: submit_move_blocks_forward returns early below
        # num_blocks - blocks_to_swap and wraps back to it.
        self.assertEqual(pairable_block_indices(6, 2, True), [4, 5])
        self.assertEqual(pairable_block_indices(6, 5, True), [1, 2, 3, 4, 5])
        # backward-enabled: the forward branch pairs 0..blocks_to_swap and the
        # backward hooks pair the tail (to CPU) with the head (to GPU).
        self.assertEqual(pairable_block_indices(6, 2, False), [0, 1, 2, 4, 5])
        self.assertEqual(pairable_block_indices(6, 0, True), [])

    def test_anima_shaped_divergence_in_block_0_is_not_excluded(self):
        blocks = [_Block("bf16", "bf16", "e4m3")] + [_Block() for _ in range(5)]
        offloader = _offloader(blocks, blocks_to_swap=5, supports_backward=False)
        jobs, deferred = offloader._build_weight_swap_jobs(blocks[4], blocks[5])
        self.assertEqual(offloader._dtype_split_paths, {})
        self.assertEqual(deferred, [])
        self.assertEqual(_paths(blocks[5], jobs, 1), ["attn.to_k", "attn.to_q", "ff.net"])

    def test_a_divergence_among_the_swappable_blocks_is_still_excluded(self):
        blocks = [_Block() for _ in range(5)] + [_Block("bf16", "bf16", "e4m3")]
        offloader = _offloader(blocks, blocks_to_swap=5, supports_backward=False)
        jobs, deferred = offloader._build_weight_swap_jobs(blocks[4], blocks[5])
        self.assertEqual(list(offloader._dtype_split_paths["_Block"]), ["ff.net"])
        self.assertEqual(_paths(blocks[5], deferred, 1), ["ff.net"])

    def test_a_resident_only_divergence_is_still_excluded_when_training(self):
        """Backward-enabled swaps blocks 0..blocks_to_swap, so block 0 IS pairable."""
        blocks = [_Block("bf16", "bf16", "e4m3")] + [_Block() for _ in range(5)]
        offloader = _offloader(blocks, blocks_to_swap=5, supports_backward=True)
        offloader._build_weight_swap_jobs(blocks[0], blocks[1])
        self.assertEqual(list(offloader._dtype_split_paths["_Block"]), ["ff.net"])


class CleanupTest(unittest.TestCase):

    def test_cleanup_drops_the_resolved_map(self):
        """It is derived from the blocks' weights and has the buffers' lifetime."""
        blocks = [_Block("int8", "int8", "int8"), _Block()]
        offloader = _offloader(blocks)
        offloader._build_weight_swap_jobs(blocks[0], blocks[1])
        self.assertNotEqual(offloader._dtype_split_paths, {})
        offloader.cleanup()
        self.assertIsNone(offloader._dtype_split_paths)


class FluxOffloaderTest(unittest.TestCase):
    """The FLUX.2 offloader is a separate class with the same pairing."""

    def _offloader(self, dual, single, blocks_to_swap=None, supports_backward=True):
        if blocks_to_swap is None:
            blocks_to_swap = len(dual) + len(single) - 1
        return FluxBlockOffloader(
            transformer_blocks=nn.ModuleList(dual),
            single_transformer_blocks=nn.ModuleList(single),
            blocks_to_swap=blocks_to_swap, device=CPU, target_dtype=torch.bfloat16,
            supports_backward=supports_backward)

    def test_it_inherits_the_guard(self):
        self.assertTrue(issubclass(FluxBlockOffloader, DtypeSplitGuardMixin))

    def test_heterogeneous_dual_blocks_are_excluded(self):
        a, b = _Block("int8", "int8", "int8"), _Block()
        offloader = self._offloader([a, b], [_SingleBlock(), _SingleBlock()])
        jobs, deferred = offloader._build_weight_swap_jobs(a, b)
        self.assertEqual(jobs, [])
        self.assertEqual(_paths(b, deferred, 1), ["attn.to_k", "attn.to_q", "ff.net"])

    def test_a_dual_split_does_not_exclude_the_single_blocks(self):
        dual = [_Block("int8", "int8", "int8"), _Block()]
        single = [_SingleBlock(), _SingleBlock()]
        offloader = self._offloader(dual, single)
        jobs, deferred = offloader._build_weight_swap_jobs(single[0], single[1])
        self.assertEqual(deferred, [])
        self.assertEqual(_paths(single[1], jobs, 1), ["attn.to_k", "attn.to_q", "ff.net"])

    def test_bf16_only_block_swap_is_unaffected(self):
        dual = [_Block(), _Block()]
        single = [_SingleBlock(), _SingleBlock()]
        offloader = self._offloader(dual, single)
        jobs, deferred = offloader._build_weight_swap_jobs(dual[0], dual[1])
        self.assertEqual(deferred, [])
        self.assertEqual(len(jobs), 3)
        self.assertEqual(offloader._dtype_split_paths, {})

    def test_a_self_pair_is_left_alone(self):
        """Same collapsed rotation as TransformerBlockOffloader (unified index)."""
        dual = [_Block("int8", "bf16", "bf16"), _Block()]
        offloader = self._offloader(dual, [_SingleBlock()])
        jobs, deferred = offloader._build_weight_swap_jobs(dual[0], dual[0])
        self.assertEqual(_paths(dual[0], deferred, 1), ["attn.to_q"])
        before = dual[0].attn.to_q.weight.data_ptr()
        _MetaTargetGuard()._move_deferred_pairs(deferred)
        offloader._move_deferred_pairs(deferred)
        self.assertEqual(dual[0].attn.to_q.weight.data_ptr(), before)

    def test_resident_blocks_are_not_resolved(self):
        """Unified index: only the last blocks_to_swap blocks pair in inference."""
        dual = [_Block("bf16", "bf16", "e4m3"), _Block(), _Block()]
        single = [_SingleBlock(), _SingleBlock()]
        offloader = self._offloader(dual, single, blocks_to_swap=3,
                                    supports_backward=False)
        offloader._build_weight_swap_jobs(single[0], single[1])
        self.assertEqual(offloader._dtype_split_paths, {})

    def test_cleanup_drops_the_resolved_map(self):
        dual = [_Block("int8", "int8", "int8"), _Block()]
        offloader = self._offloader(dual, [_SingleBlock()])
        offloader._build_weight_swap_jobs(dual[0], dual[1])
        self.assertNotEqual(offloader._dtype_split_paths, {})
        offloader.cleanup()
        self.assertIsNone(offloader._dtype_split_paths)


if __name__ == "__main__":
    unittest.main()
