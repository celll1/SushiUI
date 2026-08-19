"""``MiniMaxH3BlockLoopWrapper.attach_residual_probe`` -- Experiment B
instrumentation tool (the follow-up to Phase 1c's ablation knob and the
static AdaLN-gate audit).

Records each block's ACTUAL residual contribution
(``‖Δh_video‖ / ‖h_video‖``, generated-video rows only) during a real forward
pass -- measurement, not ablation. This file only exercises the MECHANISM --
no still-image-vs-multi-frame comparison runs here.

Fixture pattern (tiny real vendored model, CPU) copied from
``minimax_h3_block_skip_test.py`` rather than importing it: each MiniMax-H3
wrapper test file is self-contained in this repo.
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


class AttachResidualProbeTest(unittest.TestCase):
    def test_attaching_and_clearing(self):
        wrapper = MiniMaxH3BlockLoopWrapper(_model())
        self.assertIsNone(wrapper._residual_probe)

        recorded = []
        recorder = recorded.append
        wrapper.attach_residual_probe(recorder)
        self.assertIs(wrapper._residual_probe, recorder)

        wrapper.attach_residual_probe(None)
        self.assertIsNone(wrapper._residual_probe)


class AnyFeatureActiveTest(unittest.TestCase):
    def test_becomes_true_with_only_the_probe_attached(self):
        wrapper = MiniMaxH3BlockLoopWrapper(_model())
        self.assertFalse(wrapper._any_feature_active())
        wrapper.attach_residual_probe(lambda *args: None)
        self.assertTrue(wrapper._any_feature_active())
        wrapper.attach_residual_probe(None)
        self.assertFalse(wrapper._any_feature_active())


class GradEnabledGuardTest(unittest.TestCase):
    def test_raises_when_grad_is_enabled(self):
        model = _model()
        wrapper = MiniMaxH3BlockLoopWrapper(model)
        wrapper.attach_residual_probe(lambda *args: None)
        inputs = _inputs(model)
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            wrapper(**inputs)  # grad enabled by default (no torch.no_grad())


class RecorderCalledPerBlockTest(unittest.TestCase):
    """The recorder must fire once per block that actually ran, with a
    plausible (positive) ratio -- a random-init block on random input never
    produces an exact zero delta."""

    def setUp(self):
        self.model = _model()
        self.inputs = _inputs(self.model)

    def test_called_once_per_block_with_a_positive_ratio(self):
        wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        records = []
        wrapper.attach_residual_probe(
            lambda block_idx, step_idx, rel: records.append((block_idx, step_idx, rel)))

        with torch.no_grad():
            wrapper(**self.inputs)

        n = len(self.model.transformer_blocks)
        self.assertEqual([r[0] for r in records], list(range(n)))
        for _block_idx, _step_idx, rel in records:
            self.assertIsInstance(rel, float)
            self.assertGreater(rel, 0.0)

    def test_step_idx_defaults_to_minus_one_when_nothing_sets_it(self):
        wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        records = []
        wrapper.attach_residual_probe(
            lambda block_idx, step_idx, rel: records.append((block_idx, step_idx, rel)))

        with torch.no_grad():
            wrapper(**self.inputs)

        self.assertTrue(all(step_idx == -1 for _b, step_idx, _r in records))

    def test_step_idx_is_read_from_the_transformer_attribute_when_set(self):
        """Mirrors how `h3_pipeline_ops.call_transformer` threads the step index:
        set on the object the sampler calls, read back defensively inside the
        block loop (`getattr(t, "_probe_step_idx", -1)`)."""
        wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        records = []
        wrapper.attach_residual_probe(
            lambda block_idx, step_idx, rel: records.append((block_idx, step_idx, rel)))
        wrapper._probe_step_idx = 7

        with torch.no_grad():
            wrapper(**self.inputs)

        self.assertTrue(all(step_idx == 7 for _b, step_idx, _r in records))

    def test_not_called_for_a_block_that_is_simultaneously_skipped(self):
        wrapper = MiniMaxH3BlockLoopWrapper(self.model)
        wrapper.attach_block_skip({1})
        records = []
        wrapper.attach_residual_probe(
            lambda block_idx, step_idx, rel: records.append((block_idx, step_idx, rel)))

        with torch.no_grad():
            wrapper(**self.inputs)

        n = len(self.model.transformer_blocks)
        self.assertEqual(
            [r[0] for r in records], [i for i in range(n) if i != 1],
            "a skipped block's delta is zero by definition -- not signal, so it "
            "must not be recorded")


class OffloaderStillDrivenWithProbeAttachedTest(unittest.TestCase):
    def test_wait_and_submit_still_run_for_every_index_with_block_swap(self):
        model = _model()
        offloader = _StubOffloader(blocks_to_swap=2)
        wrapper = MiniMaxH3BlockLoopWrapper(model, block_offloader=offloader)
        wrapper.attach_residual_probe(lambda *args: None)
        inputs = _inputs(model)

        with torch.no_grad():
            wrapper(**inputs)

        n = len(model.transformer_blocks)
        self.assertEqual(
            offloader.calls,
            [c for i in range(n) for c in (("wait", i), ("submit", i))])


class EnsureSwapAndOffloadWiringTest(unittest.TestCase):
    """`_ensure_minimax_h3_swap_and_offload` is the one place a request-level
    `_minimax_h3_debug_probe_residuals` value turns into an attached wrapper
    plus its recording list -- exercised directly (not just the wrapper class
    in isolation) so a future edit that stops wrapping the raw transformer
    fails here."""

    def setUp(self):
        from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

        class _Backend(MiniMaxH3Mixin):
            pass

        self.model = _model()
        self.backend = _Backend()
        self.backend.minimax_h3_components = {"transformer": self.model}

    def test_probe_residuals_alone_forces_a_wrapper_and_returns_a_records_list(self):
        transformer, offloader, probe_records = self.backend._ensure_minimax_h3_swap_and_offload(
            {"_minimax_h3_debug_probe_residuals": True}, torch.device("cpu"))

        self.assertIsInstance(transformer, MiniMaxH3BlockLoopWrapper)
        self.assertIsNone(offloader)
        self.assertIsNotNone(transformer._residual_probe)
        self.assertEqual(probe_records, [])

    def test_block_swap_and_probe_together_attach_to_the_same_wrapper(self):
        transformer, offloader, probe_records = self.backend._ensure_minimax_h3_swap_and_offload(
            {"_minimax_h3_debug_probe_residuals": True, "blocks_to_swap": 2}, torch.device("cpu"))

        self.assertIsInstance(transformer, MiniMaxH3BlockLoopWrapper)
        self.assertIsNotNone(offloader, "block swap must still build its own offloader")
        self.assertIsNotNone(transformer._residual_probe)
        self.assertEqual(probe_records, [])

    def test_no_probe_flag_is_unaffected(self):
        transformer, offloader, probe_records = self.backend._ensure_minimax_h3_swap_and_offload(
            {}, torch.device("cpu"))

        self.assertIs(transformer, self.model, "the byte-identical default path returns the raw transformer")
        self.assertIsNone(offloader)
        self.assertIsNone(probe_records)

    def test_attached_recorder_actually_populates_the_returned_list(self):
        transformer, _offloader, probe_records = self.backend._ensure_minimax_h3_swap_and_offload(
            {"_minimax_h3_debug_probe_residuals": True}, torch.device("cpu"))
        inputs = _inputs(self.model)

        with torch.no_grad():
            transformer(**inputs)

        n = len(self.model.transformer_blocks)
        self.assertEqual(len(probe_records), n)
        self.assertEqual({r["block_idx"] for r in probe_records}, set(range(n)))


class RouteRefusesProbeResidualsOffMiniMaxH3Test(unittest.TestCase):
    """`generate_txt2vid`'s own gate: the field is meaningless off MiniMax-H3
    and must be refused, not silently dropped -- see `openapi.yaml`'s
    `minimax_h3_debug_probe_residuals` description for the same claim."""

    def test_source_refuses_the_field_for_a_non_minimax_h3_arch(self):
        path = str(Path(__file__).resolve().parents[1] / "api" / "routes.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        start = source.index('params.pop("minimax_h3_debug_probe_residuals"')
        block = source[start:start + 700]
        self.assertIn('_vid_arch != "minimax_h3"', block)
        self.assertIn("CustomValidationError", block)
        self.assertIn(
            'params["_minimax_h3_debug_probe_residuals"] = True', block)


class StepIdxPlumbingLinkTest(unittest.TestCase):
    """Pins the ONE fact `RecorderCalledPerBlockTest` cannot: that
    `call_transformer` writes `_probe_step_idx` on the SAME object
    (`transformer`, the wrapper whenever a probe is attached) the block loop
    reads it back from (`self`, inside `MiniMaxH3BlockLoopWrapper`) -- a
    mismatch here would silently degrade every recorded step index to the
    `-1` default with no error anywhere."""

    def test_call_transformer_sets_the_attribute_the_wrapper_reads(self):
        path = str(
            Path(__file__).resolve().parents[1] / "core" / "models" / "minimax_h3"
            / "h3_pipeline_ops.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("transformer._probe_step_idx = step_idx", source)

        wrapper_path = str(
            Path(__file__).resolve().parents[1] / "core" / "models"
            / "minimax_h3_block_loop_wrapper.py")
        with open(wrapper_path, encoding="utf-8") as handle:
            wrapper_source = handle.read()
        self.assertIn('getattr(self, "_probe_step_idx", -1)', wrapper_source)


if __name__ == "__main__":
    unittest.main()
