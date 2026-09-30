"""CPU contracts for packed MiniMax-H3 spatial residual transport."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from core.models.minimax_h3.h3_pipeline_ops import build_packed_layout
from core.models.minimax_h3.packed_residual_layout import prepare_layout, coarse_context
from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import OPERATORS, build_spec


class PackedResidualLayoutContracts(unittest.TestCase):
    def setUp(self):
        self.frames, self.height, self.width = 3, 8, 12
        self.native = build_packed_layout(num_text_tokens=4, num_audio_latents=3,
            num_latent_frames=self.frames, latent_height=16, latent_width=24,
            patch_size=(1, 2, 2), keyframe_anchors=('first',))
        self.target = torch.arange(self.native['sequence_length'] - 288, self.native['sequence_length'])
        self.times = torch.zeros(self.native['sequence_length'], dtype=torch.long)
        self.times[self.native['video_indices'][:96]] = 1
        self.layout = self.prepare()
        self.spec = build_spec('minimax_h3', [{'name':'main', 'block_count':7}], [
            dict(id='outer', stream='main', parent=None, start_block=1, end_block=6,
                 down_factor=[2, 2], **OPERATORS),
            dict(id='inner', stream='main', parent='outer', start_block=2, end_block=5,
                 down_factor=[2, 2], **OPERATORS)], {'fixture': 'packed_native'})

    def prepare(self, **changes):
        args = dict(position_ids=self.native['position_ids'], token_tags=self.native['token_tags'],
                    timestep_indices=self.times, target_indices=self.target,
                    frames=self.frames, height=self.height, width=self.width)
        args.update(changes)
        return prepare_layout(**args)

    def test_arbitrary_native_row_order_and_gradients_restore(self):
        total = self.native['sequence_length']
        permutation = torch.randperm(total)
        inverse = permutation.argsort()
        layout = self.prepare(position_ids=self.native['position_ids'][permutation],
            token_tags=self.native['token_tags'][permutation], timestep_indices=self.times[permutation],
            target_indices=inverse[self.target])
        hidden = torch.randn(2, total, 16, requires_grad=True)
        restored = layout.unpack(layout.pack(hidden))
        self.assertTrue(torch.equal(restored, hidden))
        restored.sum().backward()
        self.assertTrue(torch.equal(hidden.grad, torch.ones_like(hidden)))

    def test_reference_audio_text_rows_and_media_clock_survive_both_levels(self):
        context = self.layout.context
        for h, w in ((4, 6), (2, 3)):
            context = coarse_context(context, torch.zeros(2, self.layout.prefix + 3*h*w, 16),
                                     3*h, w, self.layout.prefix)
            for name in ('position_ids', 'token_tags', 'timestep_indices'):
                self.assertTrue(torch.equal(context[name][:self.layout.prefix],
                                            self.layout.context[name][:self.layout.prefix]))
            clock = context['position_ids'][self.layout.prefix:, 0].reshape(3, h*w)
            native_clock = self.native['position_ids'][self.target, 0].reshape(3, 96)[:, 0]
            self.assertTrue(torch.equal(clock, native_clock[:, None].expand_as(clock)))

    def test_mixed_spatial_pins_refuse_but_whole_frame_pins_survive(self):
        times = self.times.clone(); times[self.target[0]] = 1
        layout = self.prepare(timestep_indices=times)
        with self.assertRaisesRegex(ValueError, 'mix pinned'):
            coarse_context(layout.context, torch.zeros(1, layout.prefix+72, 16), 12, 6, layout.prefix)
        times[self.target[:96]] = 1
        layout = self.prepare(timestep_indices=times)
        ctx = coarse_context(layout.context, torch.zeros(1, layout.prefix+72, 16), 12, 6, layout.prefix)
        self.assertTrue((ctx['timestep_indices'][layout.prefix:layout.prefix+24] == 1).all())
        self.assertTrue((ctx['timestep_indices'][layout.prefix+24:] == 0).all())

    def test_invalid_target_indices_and_nonrectangular_coordinates_refuse(self):
        for target in (self.target[:-1], self.target.flip(0), torch.zeros_like(self.target),
                       self.target + self.native['sequence_length']):
            with self.subTest(target=target[:3].tolist()), self.assertRaises(ValueError):
                self.prepare(target_indices=target)
        positions = self.native['position_ids'].clone(); positions[self.target[1], 0] += 1
        with self.assertRaisesRegex(ValueError, 'rectangular'):
            self.prepare(position_ids=positions)

    def test_native_blocks_restore_rows_and_backpropagate_without_cuda(self):
        from core.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel
        torch.manual_seed(208)
        model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=16,
            hidden_size=16, num_layers=7, num_refiner_layers=2, ffn_dim=32,
            in_channels=4, audio_in_channels=6, patch_size=(1,2,2), text_dim=10,
            freq_dim=8, time_embed_hidden_dim=16, time_embed_dim=8, rope_freq_dim=2,
            adaln_curve_grid=33).float().eval()
        with torch.no_grad(): model.adaln_t_table.copy_(torch.randn_like(model.adaln_t_table))
        model._stamp_attention_backend()
        temb = model.adaln_t_table[[5, 24]]
        original = torch.randn(1, self.native['sequence_length'], 16, requires_grad=True)
        observed = []
        def execute(index, hidden, context):
            observed.append(hidden.shape[1])
            labels = context['timestep_indices'] * 3 + context['token_tags']
            return model.transformer_blocks[index](hidden, temb, labels, model.rope(context['position_ids']))
        result = run_rewired_stream(self.layout.pack(original), self.spec, 'main', 24, 12,
            self.layout.prefix, deepcopy(self.layout.context), execute, coarse_context, grid_groups=3)
        restored = self.layout.unpack(result)
        self.assertEqual(observed, [self.layout.prefix+n for n in (288,72,18,18,18,72,288)])
        self.assertEqual(restored.shape, original.shape)
        expected = restored.detach().clone()
        restored.square().mean().backward()
        self.assertTrue(torch.isfinite(original.grad).all())
        self.assertGreater(torch.count_nonzero(original.grad).item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for b in model.transformer_blocks for p in b.parameters()))
        with torch.no_grad():
            inference = run_rewired_stream(self.layout.pack(original.detach()), self.spec, 'main', 24, 12,
                self.layout.prefix, self.layout.context, execute, coarse_context, grid_groups=3)
        torch.testing.assert_close(self.layout.unpack(inference), expected, rtol=2e-5, atol=2e-6)
        self.assertFalse(torch.cuda.is_initialized())
