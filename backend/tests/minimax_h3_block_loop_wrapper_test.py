"""``MiniMaxH3BlockLoopWrapper`` must reproduce the stock forward exactly.

The wrapper re-owns the 50-block loop so block swap and gradient checkpointing
have somewhere to live, and it executes every OTHER stage by
calling the inner model's own submodules. That replication is the failure mode:
a stage copied slightly wrong (the AdaLN-curve lookup's max-clamp, the
``index_copy`` scatter order, the output head's dtype alignment) produces a model
that runs, stays finite, and is silently a different model -- which is precisely
what happened often enough elsewhere in this repo that LTX-2.3's wrapper carries
a submodule-name assert.

Everything here runs on a ~1.7 M-parameter build of the real vendored class on
the CPU: the property under test is "the same arithmetic in the same order", and
that does not depend on the model being 33 B parameters.

WHAT IS NOT COVERED: the block offloader itself. The stub below satisfies the
wrapper's contract (``blocks_to_swap`` / ``wait_for_block`` /
``submit_move_blocks_forward``) and records the call order, which is what the
wrapper owns; whether a REAL ``TransformerBlockOffloader`` moves the right
weights is `block_swap_dtype_split_test`'s and the offloader's own concern.
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

# A tiny build of the REAL class, in the AdaLN-CURVE ("pruned") variant -- the
# one every released checkpoint uses, and the one whose timestep path the
# wrapper has to replicate by hand.
_TINY = dict(
    # attention_head_dim must leave room for the rotary block: the 3-axis RoPE
    # rotates `2 * 3 * rope_freq_dim` of the head's channels and passes the rest
    # through, so head_dim >= 6 * rope_freq_dim. (Real checkpoints: 128 and 16.)
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
    # The curve table ships as data, not as an initialisation; zeros would make
    # every modulation identical and hide an indexing error in the lookup.
    with torch.no_grad():
        model.adaln_t_table.copy_(torch.randn_like(model.adaln_t_table))
    return model


def _inputs(model, num_video=6, num_audio=4, num_text=5, batch=1, seed=1):
    """One packed layout: [text | audio | video], two distinct timesteps."""
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
    # Two distinct noise levels, so the (timestep, modality) AdaLN addressing is
    # actually exercised rather than collapsing to one table row per modality.
    timestep_indices = torch.zeros(total, dtype=torch.long)
    timestep_indices[audio_indices] = 1
    return dict(
        hidden_states=torch.randn(batch, num_video, cfg.in_channels * patch),
        audio_hidden_states=torch.randn(batch, num_audio, cfg.audio_in_channels),
        encoder_hidden_states=torch.randn(batch, num_text, cfg.text_dim),
        # Deliberately not on a grid point: t = 1.0 exercises the lookup's
        # max-clamp and 0.3333 lands mid-interval.
        timestep=torch.tensor([0.3333, 1.0]),
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=torch.randint(0, 5, (total, 3)),
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        return_dict=False,
    )


class WrapperReproducesTheStockForwardTest(unittest.TestCase):
    def setUp(self):
        self.model = _model()
        self.inputs = _inputs(self.model)
        with torch.no_grad():
            self.expected = self.model(**self.inputs)

    def test_no_feature_attached_is_the_stock_forward_object_for_object(self):
        """The fast path must not merely agree -- it must BE the stock call."""
        wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        self.assertFalse(wrapper._any_feature_active())
        with torch.no_grad():
            got = wrapper(**self.inputs)
        for a, b in zip(got, self.expected):
            self.assertTrue(torch.equal(a, b))

    def test_the_custom_block_loop_is_bitwise_identical(self):
        """Block swap attached -> the re-owned loop runs. It must not drift.

        Bitwise, not `allclose`: the wrapper calls the SAME submodules with the
        SAME tensors in the SAME order, so any difference at all means a stage
        was replicated wrong, not that floating point happened.
        """
        offloader = _StubOffloader(blocks_to_swap=2)
        wrapper = MiniMaxH3BlockLoopWrapper(self.model, block_offloader=offloader)
        self.assertTrue(wrapper._any_feature_active())
        with torch.no_grad():
            got = wrapper(**self.inputs)
        for name, a, b in zip(("video", "audio"), got, self.expected):
            self.assertTrue(
                torch.equal(a, b),
                f"{name} output differs from the stock forward "
                f"(max |delta| {(a - b).abs().max().item():.3e}); the custom path's "
                f"replication of the RoPE / projections / token refiner / AdaLN "
                f"lookup / output heads has drifted from the vendored model.")

    def test_the_offloader_is_driven_once_per_block_in_order(self):
        offloader = _StubOffloader(blocks_to_swap=2)
        wrapper = MiniMaxH3BlockLoopWrapper(self.model, block_offloader=offloader)
        with torch.no_grad():
            wrapper(**self.inputs)
        n = len(self.model.transformer_blocks)
        self.assertEqual(
            offloader.calls,
            [c for i in range(n) for c in (("wait", i), ("submit", i))],
            "the wrapper must wait for each block before running it and submit "
            "the next prefetch after; a missing or reordered call desyncs the "
            "offloader's rotation and runs a block against another block's weights")

    def test_a_zero_blocks_to_swap_offloader_does_not_arm_the_custom_path(self):
        """An attached-but-inactive offloader must leave the fast path in place."""
        offloader = _StubOffloader(blocks_to_swap=0)
        wrapper = MiniMaxH3BlockLoopWrapper(self.model, block_offloader=offloader)
        self.assertFalse(wrapper._any_feature_active())
        with torch.no_grad():
            wrapper(**self.inputs)
        self.assertEqual(offloader.calls, [])

    def test_guarded_fbcache_reuses_one_packed_residual_for_both_outputs(self):
        from core.inference.fbcache import FirstBlockCache

        wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        cache = FirstBlockCache(
            threshold=1.0,
            warmup_steps=0,
            max_consecutive_hits=2,
            total_steps=4,
            tail_steps=1,
        )
        wrapper.attach_fbcache(cache, rows_per_frame=2, condition_video_rows=0)
        block_calls = [0] * len(self.model.transformer_blocks)
        handles = [
            block.register_forward_hook(
                lambda _module, _args, _out, index=index: block_calls.__setitem__(
                    index, block_calls[index] + 1
                )
            )
            for index, block in enumerate(self.model.transformer_blocks)
        ]
        try:
            with torch.no_grad():
                first = wrapper(**self.inputs)
                wrapper._fbcache_step = 1
                second = wrapper(**self.inputs)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(cache.n_hits, 1)
        self.assertEqual(block_calls, [2, 1, 1, 1])
        for a, b in zip(first, second):
            self.assertTrue(torch.allclose(a, b, rtol=1e-5, atol=1e-6))

    def test_fbcache_guard_tolerates_a_different_audio_row_count_between_calls(self):
        """The guard indicator is built from VIDEO rows alone.

        ``video_residual = first_residual.index_select(1, video_indices)``, then
        sliced by ``condition_video_rows`` and reshaped by ``rows_per_frame`` --
        neither reads ``audio_indices`` or the audio row count anywhere. This is
        the invariant a PARTIAL audio pin (``h3_pipeline_ops``'s
        ``pinned_audio_latents``) depends on: unlike the spatial-mask video pin
        (refused ahead of generation, `minimax_h3_spatial_mask_fbcache_test.py`,
        because ITS row-level pin can break the video tiling this guard
        assumes), an audio-only change -- a different row COUNT, a permuted
        order, different content -- must never touch this arithmetic. Two calls
        with the SAME video geometry but a DIFFERENT audio geometry must both
        run FBCache to completion without error.
        """
        from core.inference.fbcache import FirstBlockCache

        wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        cache = FirstBlockCache(
            threshold=1.0, warmup_steps=0, max_consecutive_hits=2,
            total_steps=4, tail_steps=1,
        )
        wrapper.attach_fbcache(cache, rows_per_frame=2, condition_video_rows=0)

        with torch.no_grad():
            wrapper(**self.inputs)

        # A DIFFERENT audio layout -- more rows, its own indices and content --
        # built as an entirely separate packed request. The video part (6 rows,
        # `rows_per_frame=2`) is the same SHAPE as `self.inputs`, which is what
        # the guard actually reads; nothing here asks the two audio blocks to
        # agree with each other.
        other = _inputs(self.model, num_video=6, num_audio=9, num_text=5, seed=99)
        with torch.no_grad():
            wrapper(**other)

        self.assertEqual(cache.n_hits + cache.n_miss, 2)

    def test_fbcache_refuses_block_swap(self):
        from core.inference.fbcache import FirstBlockCache

        wrapper = MiniMaxH3BlockLoopWrapper(
            self.model, block_offloader=_StubOffloader(blocks_to_swap=1)
        )
        with self.assertRaisesRegex(ValueError, "cannot run with Block Swap"):
            wrapper.attach_fbcache(
                FirstBlockCache(threshold=0.08), rows_per_frame=2
            )

    def test_the_test_can_fail(self):
        """Premise: this comparison really does detect a mis-replicated stage.

        Without it, "bitwise identical" could be true because both sides run the
        same code -- which is exactly what the fast path does, and exactly what
        the custom path must NOT do.
        """
        offloader = _StubOffloader(blocks_to_swap=2)
        wrapper = MiniMaxH3BlockLoopWrapper(self.model, block_offloader=offloader)
        broken = dict(self.inputs)
        # The one-line mistake this class of wrapper actually makes: reading the
        # AdaLN curve without the max-clamp, i.e. at a t the table cannot index.
        broken["timestep"] = torch.tensor([0.3333, 0.9999])
        with torch.no_grad():
            got = wrapper(**broken)
        self.assertFalse(torch.equal(got[0], self.expected[0]))


class PassthroughTest(unittest.TestCase):
    """LoRA save/load and the quantized export must see through the wrapper."""

    def setUp(self):
        self.model = _model()
        self.wrapper = MiniMaxH3BlockLoopWrapper(self.model)

    def test_state_dict_keys_are_the_inner_models_module_paths(self):
        self.assertEqual(list(self.wrapper.state_dict().keys()),
                         list(self.model.state_dict().keys()))

    def test_load_state_dict_reaches_the_inner_model(self):
        sd = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        self.wrapper.load_state_dict(sd)
        self.assertTrue(torch.equal(self.model.proj_out.weight,
                                    torch.zeros_like(self.model.proj_out.weight)))

    def test_attribute_access_falls_through(self):
        self.assertIs(self.wrapper.transformer_blocks, self.model.transformer_blocks)
        self.assertIs(self.wrapper.config, self.model.config)
        self.assertFalse(self.wrapper.use_adaln_curves is None)

    def test_the_export_unwrap_hook_finds_the_inner_model(self):
        from core.models.common.quantized_export import layout_unwrap

        self.assertIs(layout_unwrap("minimax_h3", self.wrapper), self.model)
        self.assertIs(layout_unwrap("minimax_h3", self.model), self.model)


if __name__ == "__main__":
    unittest.main()
