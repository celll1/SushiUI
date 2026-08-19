"""``MiniMaxH3BlockLoopWrapper.attach_block_skip`` -- Phase 1c ablation tool.

Identity-skips named ``transformer_blocks`` indices at inference (pure
residual passthrough, no compute), for a later phase to drive over the API
and measure which of MiniMax-H3's 50 blocks matter for a given clip shape
(still-image, ``num_frames=1``, in particular). This file only exercises the
MECHANISM -- no ablation experiment runs here.

Fixture pattern (tiny real vendored model, CPU) copied from
``minimax_h3_block_loop_wrapper_test.py`` rather than importing it: each
MiniMax-H3 wrapper test file is self-contained in this repo.
"""

import sys
import unittest
from pathlib import Path

import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel  # noqa: E402
from core.models.minimax_h3_block_loop_wrapper import (  # noqa: E402
    MiniMaxH3BlockLoopWrapper,
)

_TINY = dict(
    num_attention_heads=2, attention_head_dim=16, hidden_size=16, num_layers=4,
    num_refiner_layers=2, ffn_dim=32, in_channels=4, audio_in_channels=6,
    patch_size=(1, 2, 2), text_dim=10, freq_dim=8, time_embed_hidden_dim=16,
    time_embed_dim=8, rope_freq_dim=2, adaln_curve_grid=33,
)


class _StubOffloader:
    """Satisfies the wrapper's offloader contract and records the call order."""

    def __init__(self, blocks_to_swap):
        self.blocks_to_swap = blocks_to_swap
        self.calls = []

    def wait_for_block(self, idx):
        self.calls.append(("wait", idx))

    def submit_move_blocks_forward(self, idx):
        self.calls.append(("submit", idx))


def _model(seed=0):
    torch.manual_seed(seed)
    model = MiniMaxH3Transformer3DModel(**_TINY).to(torch.float32).eval()
    with torch.no_grad():
        model.adaln_t_table.copy_(torch.randn_like(model.adaln_t_table))
    return model


def _inputs(model, num_video=6, num_audio=4, num_text=5, batch=1, seed=1):
    torch.manual_seed(seed)
    cfg = model.config
    patch = cfg.patch_size[0] * cfg.patch_size[1] * cfg.patch_size[2]
    total = num_text + num_audio + num_video
    text_indices = torch.arange(0, num_text)
    audio_indices = torch.arange(num_text, num_text + num_audio)
    video_indices = torch.arange(num_text + num_audio, total)
    token_tags = torch.empty(total, dtype=torch.long)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0
    timestep_indices = torch.zeros(total, dtype=torch.long)
    timestep_indices[audio_indices] = 1
    return dict(
        hidden_states=torch.randn(batch, num_video, cfg.in_channels * patch),
        audio_hidden_states=torch.randn(batch, num_audio, cfg.audio_in_channels),
        encoder_hidden_states=torch.randn(batch, num_text, cfg.text_dim),
        timestep=torch.tensor([0.3333, 1.0]),
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=torch.randint(0, 5, (total, 3)),
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        return_dict=False,
    )


class AttachBlockSkipValidationTest(unittest.TestCase):
    def setUp(self):
        self.model = _model()

    def test_rejects_an_out_of_range_index(self):
        wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        n = len(self.model.transformer_blocks)
        with self.assertRaisesRegex(ValueError, "out of range"):
            wrapper.attach_block_skip({n})
        self.assertIsNone(wrapper._skip_blocks, "a rejected attach must not partially apply")

    def test_rejects_a_negative_index(self):
        wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        with self.assertRaisesRegex(ValueError, "out of range"):
            wrapper.attach_block_skip({-1})

    def test_clearing_with_none_or_empty_is_a_no_op(self):
        wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        wrapper.attach_block_skip(None)
        self.assertIsNone(wrapper._skip_blocks)
        wrapper.attach_block_skip([])
        self.assertIsNone(wrapper._skip_blocks)

    def test_rejects_block_0_when_fbcache_is_attached(self):
        from core.inference.fbcache import FirstBlockCache

        wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        wrapper.attach_fbcache(
            FirstBlockCache(threshold=0.08), rows_per_frame=2, condition_video_rows=0)
        with self.assertRaisesRegex(ValueError, "FBCache reads"):
            wrapper.attach_block_skip({0})

    def test_a_non_zero_index_is_fine_with_fbcache_attached(self):
        from core.inference.fbcache import FirstBlockCache

        wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        wrapper.attach_fbcache(
            FirstBlockCache(threshold=0.08), rows_per_frame=2, condition_video_rows=0)
        wrapper.attach_block_skip({1})
        self.assertEqual(wrapper._skip_blocks, frozenset({1}))

    def test_attaching_fbcache_after_block_0_is_already_skipped_is_also_refused(self):
        """The reverse attach order (skip attached first, as production does it).

        `_ensure_minimax_h3_swap_and_offload` always attaches block skip before
        any per-step `attach_fbcache` call, so THIS is the check that actually
        fires for that call order -- the mirrored guard inside `attach_fbcache`.
        """
        from core.inference.fbcache import FirstBlockCache

        wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        wrapper.attach_block_skip({0})
        with self.assertRaisesRegex(ValueError, "FBCache reads"):
            wrapper.attach_fbcache(
                FirstBlockCache(threshold=0.08), rows_per_frame=2, condition_video_rows=0)


class AnyFeatureActiveTest(unittest.TestCase):
    def test_becomes_true_with_only_skip_blocks_attached(self):
        model = _model()
        wrapper = MiniMaxH3BlockLoopWrapper(model)
        self.assertFalse(wrapper._any_feature_active())
        wrapper.attach_block_skip({1})
        self.assertTrue(wrapper._any_feature_active())
        wrapper.attach_block_skip(None)
        self.assertFalse(wrapper._any_feature_active())


class GradEnabledGuardTest(unittest.TestCase):
    def test_raises_when_grad_is_enabled(self):
        model = _model()
        wrapper = MiniMaxH3BlockLoopWrapper(model)
        wrapper.attach_block_skip({1})
        inputs = _inputs(model)
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            wrapper(**inputs)  # grad enabled by default (no torch.no_grad())


class SkippedBlockIsNeverCalledTest(unittest.TestCase):
    """The skipped block's module must never run, and hidden_states must pass
    through the iteration bit-identical to what it entered with."""

    def setUp(self):
        self.model = _model()
        self.inputs = _inputs(self.model)

    def test_skipped_blocks_module_is_never_invoked(self):
        wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        wrapper.attach_block_skip({1, 2})

        call_counts = [0] * len(self.model.transformer_blocks)
        handles = [
            block.register_forward_hook(
                lambda _m, _a, _o, index=index: call_counts.__setitem__(
                    index, call_counts[index] + 1)
            )
            for index, block in enumerate(self.model.transformer_blocks)
        ]
        try:
            with torch.no_grad():
                wrapper(**self.inputs)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(call_counts, [1, 0, 0, 1], "blocks 1 and 2 must never be called")

    def test_hidden_states_entering_a_skipped_iteration_is_unchanged_by_it(self):
        """Compare, across two SEPARATE forward passes on the same (deterministic,
        eval-mode) model and the same inputs: (a) what block 1 receives as input
        with nothing skipped, against (b) what block 2 receives as input with
        block 1 skipped. If block 1's identity skip is truly a no-op passthrough,
        these must be bit-identical -- block 2 in run (b) sees exactly what block 1
        saw in run (a), because nothing touched hidden_states in between."""
        baseline_captured = {}

        def baseline_pre_hook(_module, args):
            baseline_captured["into_block_1"] = args[0].clone()

        baseline_wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        baseline_handle = self.model.transformer_blocks[1].register_forward_pre_hook(
            baseline_pre_hook)
        try:
            with torch.no_grad():
                baseline_wrapper(**self.inputs)
        finally:
            baseline_handle.remove()

        skip_captured = {}

        def skip_pre_hook(_module, args):
            skip_captured["into_block_2"] = args[0].clone()

        skip_wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        skip_wrapper.attach_block_skip({1})
        skip_handle = self.model.transformer_blocks[2].register_forward_pre_hook(skip_pre_hook)
        try:
            with torch.no_grad():
                skip_wrapper(**self.inputs)
        finally:
            skip_handle.remove()

        self.assertTrue(
            torch.equal(skip_captured["into_block_2"], baseline_captured["into_block_1"]),
            "block 1's identity skip must leave hidden_states bit-identical to what "
            "block 1 itself would have received as input")


class OffloaderStillDrivenForASkippedIndexTest(unittest.TestCase):
    def test_wait_and_submit_still_run_for_a_skipped_index_with_block_swap(self):
        model = _model()
        offloader = _StubOffloader(blocks_to_swap=2)
        wrapper = MiniMaxH3BlockLoopWrapper(model, block_offloader=offloader)
        wrapper.attach_block_skip({1})
        inputs = _inputs(model)

        call_counts = [0] * len(model.transformer_blocks)
        handles = [
            block.register_forward_hook(
                lambda _m, _a, _o, index=index: call_counts.__setitem__(
                    index, call_counts[index] + 1)
            )
            for index, block in enumerate(model.transformer_blocks)
        ]
        try:
            with torch.no_grad():
                wrapper(**inputs)
        finally:
            for handle in handles:
                handle.remove()

        n = len(model.transformer_blocks)
        self.assertEqual(
            offloader.calls,
            [c for i in range(n) for c in (("wait", i), ("submit", i))],
            "the offloader's wait/submit schedule must stay in lock-step across "
            "every block index, including a skipped one")
        self.assertEqual(call_counts, [1, 0, 1, 1], "block 1 stays skipped even with swap on")


class EnsureSwapAndOffloadWiringTest(unittest.TestCase):
    """`_ensure_minimax_h3_swap_and_offload` is the one place a request-level
    `skip_blocks` value turns into an attached wrapper -- exercised directly
    (not just the wrapper class in isolation) so a future edit that stops
    wrapping the raw transformer, or validates too late, fails here."""

    def setUp(self):
        from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

        class _Backend(MiniMaxH3Mixin):
            pass

        self.model = _model()
        self.backend = _Backend()
        self.backend.minimax_h3_components = {"transformer": self.model}

    def test_skip_blocks_alone_forces_a_wrapper_not_the_raw_transformer(self):
        transformer, offloader, probe_records = self.backend._ensure_minimax_h3_swap_and_offload(
            {"_minimax_h3_debug_skip_blocks": {1}}, torch.device("cpu"))

        self.assertIsInstance(transformer, MiniMaxH3BlockLoopWrapper)
        self.assertIsNone(offloader)
        self.assertIsNone(probe_records, "skip_blocks alone must not arm the residual probe")
        self.assertEqual(transformer._skip_blocks, frozenset({1}))

    def test_block_swap_and_skip_blocks_together_attach_to_the_same_wrapper(self):
        transformer, offloader, probe_records = self.backend._ensure_minimax_h3_swap_and_offload(
            {"_minimax_h3_debug_skip_blocks": {1}, "blocks_to_swap": 2}, torch.device("cpu"))

        self.assertIsInstance(transformer, MiniMaxH3BlockLoopWrapper)
        self.assertIsNotNone(offloader, "block swap must still build its own offloader")
        self.assertIs(transformer._block_offloader, offloader)
        self.assertIsNone(probe_records)
        self.assertEqual(transformer._skip_blocks, frozenset({1}))

    def test_an_out_of_range_index_raises_before_the_transformer_is_moved(self):
        moved = []
        self.backend._minimax_h3_move = lambda name, device: moved.append((name, device))

        n = len(self.model.transformer_blocks)
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.backend._ensure_minimax_h3_swap_and_offload(
                {"_minimax_h3_debug_skip_blocks": {n}}, torch.device("cpu"))

        self.assertEqual(moved, [], "an invalid skip set must be refused before staging the DiT")

    def test_no_skip_blocks_is_unaffected(self):
        transformer, offloader, probe_records = self.backend._ensure_minimax_h3_swap_and_offload(
            {}, torch.device("cpu"))

        self.assertIs(transformer, self.model, "the byte-identical default path returns the raw transformer")
        self.assertIsNone(offloader)
        self.assertIsNone(probe_records)


class RouteRefusesSkipBlocksOffMiniMaxH3Test(unittest.TestCase):
    """`generate_txt2vid`'s own gate: the field is meaningless off MiniMax-H3
    and must be refused, not silently dropped -- see `openapi.yaml`'s
    `minimax_h3_debug_skip_blocks` description for the same claim."""

    def test_source_refuses_the_field_for_a_non_minimax_h3_arch(self):
        path = str(Path(__file__).resolve().parents[1] / "api" / "routes.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        start = source.index('params.pop("minimax_h3_debug_skip_blocks"')
        block = source[start:start + 700]
        self.assertIn('_vid_arch != "minimax_h3"', block)
        self.assertIn("CustomValidationError", block)
        self.assertIn('params["_minimax_h3_debug_skip_blocks"] = _h3_skip_blocks', block)


if __name__ == "__main__":
    unittest.main()
