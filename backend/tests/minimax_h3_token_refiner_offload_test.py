"""``token_refiner`` must not be permanently GPU-resident under block swap.

``_ensure_minimax_h3_swap_and_offload`` leaves ``token_refiner`` off the
device when ``blocks_to_swap > 0`` (``pipeline_backends/minimax_h3.py``,
the non-block-modules loop), and ``MiniMaxH3BlockLoopWrapper._custom_forward``
stages it onto the device for the length of its own single call and back off
immediately after, before the block loop runs (``minimax_h3_block_loop_wrapper.py``).
That staging is gated on the SAME ``blocks_to_swap > 0`` condition as block
swap itself (the wrapper's local ``swap_on``) -- there is no separate opt-in.

Measured motivation (not re-derived here): on the real
``minimax_h3_fl2va_pruned_w4a8_mixed.safetensors`` checkpoint, ``token_refiner``
is 1.4356 GiB and was, before this change, the only non-block module left
permanently resident at every ``blocks_to_swap`` setting even though the block
stack it sits next to (9.12 of the DiT's 11.82 GiB of non-block-loop weight)
IS swappable.

WHAT THIS FILE COVERS: device residency during the two states block swap can
be in (on / off), not VRAM totals -- those are a probe against the real
checkpoint (``scratchpad/minimax_h3_token_refiner_probe.py``), not a
unit-testable property. WHAT THIS FILE DOES NOT COVER: numerical
equivalence -- see ``TokenRefinerStagingIsNumericallyTransparentTest`` below,
run at bf16 (the production dtype) rather than only fp32.
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

# A tiny build of the REAL class (same geometry convention as
# minimax_h3_block_loop_wrapper_test.py's `_TINY`), in the AdaLN-curve
# ("pruned") variant every released checkpoint ships.
_TINY = dict(
    num_attention_heads=2, attention_head_dim=16, hidden_size=16, num_layers=4,
    num_refiner_layers=2, ffn_dim=32, in_channels=4, audio_in_channels=6,
    patch_size=(1, 2, 2), text_dim=10, freq_dim=8, time_embed_hidden_dim=16,
    time_embed_dim=8, rope_freq_dim=2, adaln_curve_grid=33,
)


class _RealOffloader:
    """The block-swap contract the wrapper actually depends on.

    Deliberately not a Mock: `wait_for_block` / `submit_move_blocks_forward`
    are the ONLY methods `_custom_forward` calls on it (see
    `minimax_h3_block_loop_wrapper_test.py`'s `_StubOffloader` for the same
    contract) -- a `.device` attribute is NOT part of that contract, which is
    exactly why the staging code below reads the device off the input tensor
    instead.
    """

    def __init__(self, blocks_to_swap):
        self.blocks_to_swap = blocks_to_swap

    def wait_for_block(self, idx):
        pass

    def submit_move_blocks_forward(self, idx):
        pass


def _model(dtype=torch.float32, seed=0):
    """Build the tiny model at ``dtype``, honoring the AdaLN-curve mixed
    precision the real loader pins (``loader.py``'s ``_dit_target_dtype``):
    ``adaln_t_table`` and every ``adaln_proj.linear`` / ``norm_out.linear``
    stay float32 even when the rest of the model (this test's bf16 case) is
    lower precision -- ``torch.lerp`` on the curve table requires its
    interpolation weight and both endpoints to share a dtype, so at bf16 a
    blanket ``model.to(dtype)`` breaks the stock forward before this
    change's code is ever reached. Fp32-only (the default) needs no split.
    """
    torch.manual_seed(seed)
    model = MiniMaxH3Transformer3DModel(**_TINY).to(dtype).eval()
    if dtype != torch.float32:
        model.adaln_t_table.data = model.adaln_t_table.data.to(torch.float32)
        for block in model.transformer_blocks:
            block.adaln_proj.linear.to(torch.float32)
        model.norm_out.linear.to(torch.float32)
    with torch.no_grad():
        model.adaln_t_table.copy_(torch.randn_like(model.adaln_t_table))
    return model


def _inputs(model, device, dtype, num_video=6, num_audio=4, num_text=5, seed=1):
    torch.manual_seed(seed)
    cfg = model.config
    patch = cfg.patch_size[0] * cfg.patch_size[1] * cfg.patch_size[2]
    total = num_text + num_audio + num_video
    text_indices = torch.arange(0, num_text, device=device)
    audio_indices = torch.arange(num_text, num_text + num_audio, device=device)
    video_indices = torch.arange(num_text + num_audio, total, device=device)
    token_tags = torch.empty(total, dtype=torch.long, device=device)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0
    timestep_indices = torch.zeros(total, dtype=torch.long, device=device)
    timestep_indices[audio_indices] = 1
    return dict(
        hidden_states=torch.randn(1, num_video, cfg.in_channels * patch, device=device, dtype=dtype),
        audio_hidden_states=torch.randn(1, num_audio, cfg.audio_in_channels, device=device, dtype=dtype),
        encoder_hidden_states=torch.randn(1, num_text, cfg.text_dim, device=device, dtype=dtype),
        timestep=torch.tensor([0.3333, 1.0], device=device, dtype=torch.float32),
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=torch.randint(0, 5, (total, 3), device=device),
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        return_dict=False,
    )


@unittest.skipUnless(torch.cuda.is_available(), "residency across devices needs a real CUDA device")
class TokenRefinerResidencyDuringTheBlockLoopTest(unittest.TestCase):
    """The property under test: WHERE `token_refiner` sits while blocks run."""

    def setUp(self):
        self.device = torch.device("cuda")
        self.model = _model()
        self.inputs = _inputs(self.model, "cpu", torch.float32)
        # Everything the fast path needs resident, as the caller
        # (`_ensure_minimax_h3_swap_and_offload`'s `blocks_to_swap <= 0`
        # branch) would leave it: the WHOLE model on the device.
        self.model.to(self.device)
        self.inputs = _inputs(self.model, self.device, torch.float32)

        self.device_during_block_loop = []

        def _record_refiner_device(module, args):
            self.device_during_block_loop.append(
                next(self.model.token_refiner.parameters()).device.type)

        self._hook_handle = self.model.transformer_blocks[0].register_forward_pre_hook(
            _record_refiner_device)

    def tearDown(self):
        self._hook_handle.remove()

    def test_block_swap_on_leaves_token_refiner_off_device_for_the_block_loop(self):
        """`blocks_to_swap > 0`: staged for its own call, off the device before block 0 runs."""
        # Simulate the caller's `blocks_to_swap > 0` branch: leave
        # `token_refiner` off the device (that loop's own skip), everything
        # else resident.
        self.model.token_refiner.to("cpu")
        wrapper = MiniMaxH3BlockLoopWrapper(
            self.model, block_offloader=_RealOffloader(blocks_to_swap=2))
        self.assertTrue(wrapper._any_feature_active())

        with torch.no_grad():
            wrapper(**self.inputs)

        self.assertEqual(
            self.device_during_block_loop, ["cpu"],
            "token_refiner must be OFF the device by the time the block loop starts "
            "when block swap is on -- it is staged for its own call only.")
        self.assertEqual(
            next(self.model.token_refiner.parameters()).device.type, "cpu",
            "token_refiner must be back on the CPU after the forward call returns.")

    def test_block_swap_off_leaves_token_refiner_on_device(self):
        """`blocks_to_swap <= 0` (or an inactive offloader): unaffected, stays resident."""
        # This is the state the `blocks_to_swap <= 0` caller branch leaves
        # things in: the whole model, `token_refiner` included, resident.
        wrapper = MiniMaxH3BlockLoopWrapper(
            self.model, block_offloader=_RealOffloader(blocks_to_swap=0))
        self.assertFalse(wrapper._any_feature_active())

        with torch.no_grad():
            wrapper(**self.inputs)

        self.assertEqual(
            self.device_during_block_loop, ["cuda"],
            "with block swap off, token_refiner must stay resident on the device "
            "through the block loop -- this path must be unchanged by the fix.")
        self.assertEqual(
            next(self.model.token_refiner.parameters()).device.type, "cuda",
            "token_refiner must remain on the device after the call when block swap is off.")


@unittest.skipUnless(torch.cuda.is_available(), "the production dtype path needs a real CUDA device")
class TokenRefinerStagingIsNumericallyTransparentTest(unittest.TestCase):
    """Staging on/off the device must not change a single output value, in bf16."""

    def test_staged_and_always_resident_token_refiner_agree_in_bf16(self):
        device = torch.device("cuda")
        dtype = torch.bfloat16

        model_a = _model(dtype=dtype).to(device)  # stays fully resident
        model_b = _model(dtype=dtype).to(device)
        with torch.no_grad():
            for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
                p_b.data.copy_(p_a.data)
            for b_a, b_b in zip(model_a.buffers(), model_b.buffers()):
                b_b.data.copy_(b_a.data)

        inputs = _inputs(model_a, device, dtype)

        # Reference: the stock forward, everything resident, no wrapper.
        with torch.no_grad():
            expected = model_a(**inputs)

        # Under test: token_refiner starts off the device (as the caller
        # leaves it when block swap is on) and is staged on/off by the
        # wrapper for its own call.
        model_b.token_refiner.to("cpu")
        wrapper = MiniMaxH3BlockLoopWrapper(
            model_b, block_offloader=_RealOffloader(blocks_to_swap=2))
        with torch.no_grad():
            got = wrapper(**inputs)

        for name, a, b in zip(("video", "audio"), got, expected):
            self.assertTrue(
                torch.equal(a, b),
                f"{name} output differs between an always-resident token_refiner and a "
                f"staged one (bf16); max |delta| "
                f"{(a.float() - b.float()).abs().max().item():.3e}. Staging is a placement "
                f"change and must not perturb a single bit of the result.")
        self.assertEqual(next(model_b.token_refiner.parameters()).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
