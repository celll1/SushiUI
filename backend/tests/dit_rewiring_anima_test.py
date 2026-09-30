"""Anima native grid/RoPE, cross-context and checkpointed backward contracts."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from core.rewiring.spec import OPERATORS, build_spec


class AnimaRuntimeContracts(unittest.TestCase):
    def setUp(self):
        for target, name, value in (
                (torch.cuda, "get_device_capability", lambda *a, **k: (8, 9)),
                (torch.cuda, "_lazy_init", lambda *a, **k: None),
                (torch._C, "_cuda_init", lambda *a, **k: None)):
            guard = patch.object(target, name, value)
            guard.start(); self.addCleanup(guard.stop)
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)), \
             patch.object(torch.cuda, "_lazy_init"), patch.object(torch._C, "_cuda_init"):
            from core.models.anima.anima_models import Anima
            from core.models.anima.residual_unet import native_config, restore_topology
        torch.manual_seed(31)
        self.model = Anima(max_img_h=32, max_img_w=32, max_frames=8,
            in_channels=4, out_channels=4, model_channels=24, num_blocks=7,
            num_heads=2, mlp_ratio=1.0, crossattn_emb_channels=12,
            use_adaln_lora=True, adaln_lora_dim=8, use_llm_adapter=False)
        with torch.no_grad():
            for parameter in self.model.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(0, 0.02)
        self.spec = build_spec("anima", [{"name": "main", "block_count": 7}], [
            dict(id="outer", stream="main", parent=None, start_block=1,
                 end_block=6, down_factor=[2, 2], **OPERATORS),
            dict(id="inner", stream="main", parent="outer", start_block=2,
                 end_block=5, down_factor=[2, 2], **OPERATORS)], native_config(self.model))
        restore_topology(self.model, self.spec)
        self.image = torch.randn(1, 4, 1, 16, 24, requires_grad=True)
        self.text = torch.randn(1, 3, 12, requires_grad=True)
        self.padding = torch.zeros(1, 1, 16, 24)

    def forward(self, image=None):
        return self.model(self.image if image is None else image,
                          torch.tensor([0.5]), self.text, padding_mask=self.padding)

    def test_native_grid_rope_context_and_offloader_order(self):
        seen = []; events = []
        class Offloader:
            def wait_for_block(self, index): events.append(("wait", index))
            def submit_move_blocks_forward(self, index): events.append(("submit", index))
        self.model._block_offloader = Offloader()
        def observe(module, args, kwargs):
            image, embedding, text, attention, fp32 = args
            self.assertIs(text, self.text)
            seen.append(tuple(image.shape[2:4]))
            self.assertEqual(kwargs["rope_emb_L_1_1_D"].shape[0], image.shape[2]*image.shape[3])
        handles = [block.register_forward_pre_hook(observe, with_kwargs=True) for block in self.model.blocks]
        count = sum(p.numel() for p in self.model.parameters())
        try:
            output = self.forward()
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(seen, [(8, 12), (4, 6), (2, 3), (2, 3), (2, 3), (4, 6), (8, 12)])
        self.assertEqual(events, [(name, index) for index in range(7) for name in ("wait", "submit")])
        self.assertEqual(output.shape, self.image.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(count, sum(p.numel() for p in self.model.parameters()))

    def test_native_checkpoint_output_and_gradients(self):
        parameters = [self.image, self.text, *self.model.parameters()]
        expected = self.forward()
        before = torch.autograd.grad(expected.square().mean(), parameters, allow_unused=True)
        self.model.enable_gradient_checkpointing()
        actual = self.forward()
        after = torch.autograd.grad(actual.square().mean(), parameters, allow_unused=True)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(before, after):
            if a is None:
                self.assertIsNone(b)
            else:
                self.assertTrue(torch.isfinite(b).all())
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        self.assertGreater(float(after[1].abs().sum()), 0)

    def test_incompatible_features_and_identity_refusal(self):
        from core.models.anima.residual_unet import restore_topology, validate_topology_identity, native_config
        for name in ("_fbcache", "_tread_config", "_blockskip_config", "_block_skip_config", "_repa_tap_depth"):
            setattr(self.model, name, {} if name != "_repa_tap_depth" else 0)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                self.forward()
            setattr(self.model, name, None)
        with self.assertRaisesRegex(ValueError, "single 2D"):
            self.forward(self.image.detach().repeat(1, 1, 2, 1, 1))
        with self.assertRaisesRegex(ValueError, "absolute positions"):
            validate_topology_identity(self.spec, dict(native_config(self.model), extra_per_block_abs_pos_emb=True))
        with self.assertRaisesRegex(ValueError, "native config mismatch"):
            restore_topology(self.model, dict(self.spec, base_config_sha256="0"*64))
        self.assertEqual(self.model._residual_unet_spec, self.spec)
        restore_topology(self.model, None)
        self.assertEqual(self.forward().shape, self.image.shape)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
