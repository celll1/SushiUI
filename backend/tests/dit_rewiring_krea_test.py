"""Native Krea coordinate, mask and checkpointed backward contracts."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from core.rewiring.spec import OPERATORS, build_spec


class KreaRuntimeContracts(unittest.TestCase):
    def setUp(self):
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)), \
             patch.object(torch.cuda, "_lazy_init"), patch.object(torch._C, "_cuda_init"):
            from core.models.krea2.vendor.transformer import Krea2Transformer2DModel
            from core.models.krea2.residual_unet import native_config, restore_topology
        torch.manual_seed(21)
        self.model = Krea2Transformer2DModel(in_channels=4, num_layers=7,
            attention_head_dim=12, num_attention_heads=2, num_key_value_heads=1,
            intermediate_size=32, timestep_embed_dim=8, text_hidden_dim=12,
            num_text_layers=2, text_num_attention_heads=2, text_num_key_value_heads=1,
            text_intermediate_size=16, num_layerwise_text_blocks=1,
            num_refiner_text_blocks=1, axes_dims_rope=(4, 4, 4))
        self.spec = build_spec("krea2", [{"name": "main", "block_count": 7}], [
            dict(id="outer", stream="main", parent=None, start_block=1,
                 end_block=6, down_factor=[2, 2], **OPERATORS),
            dict(id="inner", stream="main", parent="outer", start_block=2,
                 end_block=5, down_factor=[2, 2], **OPERATORS)],
            native_config(dict(self.model.config)))
        restore_topology(self.model, self.spec)
        self.image = torch.randn(1, 8*12, 4, requires_grad=True)
        self.text = torch.randn(1, 3, 2, 12, requires_grad=True)
        self.positions = torch.zeros(3+8*12, 3)
        self.positions[3:, 1] = torch.arange(8).repeat_interleave(12)
        self.positions[3:, 2] = torch.arange(12).repeat(8)
        self.mask = torch.tensor([[True, True, False]])

    def forward(self, positions=None):
        return self.model(self.image, self.text, torch.tensor([0.5]),
                          self.positions if positions is None else positions,
                          self.mask, return_dict=False)[0]

    def test_native_shapes_masks_and_parameter_count(self):
        count = sum(p.numel() for p in self.model.parameters())
        seen = []
        handles = []
        def observe(module, args):
            hidden, modulation, rotary, mask = args
            seen.append(hidden.shape[1])
            self.assertTrue(torch.equal(mask[..., :3], self.mask[:, None, None, :]))
            self.assertTrue(mask[..., 3:].all())
            self.assertEqual(mask.shape[-1], hidden.shape[1])
        try:
            for block in self.model.transformer_blocks:
                handles.append(block.register_forward_pre_hook(observe))
            output = self.forward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(seen, [99, 27, 9, 9, 9, 27, 99])
        self.assertEqual(output.shape, self.image.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(count, sum(p.numel() for p in self.model.parameters()))

    def test_checkpoint_output_and_gradient_equivalence(self):
        parameters = [self.image, self.text, *self.model.parameters()]
        expected = self.forward()
        gradients = torch.autograd.grad(expected.square().mean(), parameters, allow_unused=True)
        self.model.enable_gradient_checkpointing()
        actual = self.forward()
        restored = torch.autograd.grad(actual.square().mean(), parameters, allow_unused=True)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(gradients, restored):
            if a is None:
                self.assertIsNone(b)
            else:
                self.assertTrue(torch.isfinite(b).all())
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)

    def test_refuse_non_native_positions_and_features(self):
        malformed = self.positions.clone()
        malformed[3:, 0] = 1
        with self.assertRaisesRegex(ValueError, "positions"):
            self.forward(malformed)
        self.model._repa_tap_depth = 2
        with self.assertRaisesRegex(ValueError, "REPA"):
            self.forward()
        self.model._repa_tap_depth = None
        from core.models.krea2.residual_unet import restore_topology
        wrong = dict(self.spec, base_config_sha256="0"*64)
        with self.assertRaises(ValueError):
            restore_topology(self.model, wrong)
        self.assertEqual(self.model._residual_unet_spec, self.spec)
        restore_topology(self.model, None)
        self.assertEqual(self.forward(malformed).shape, self.image.shape)


    def test_full_and_lora_artifact_round_trip(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from core.models.krea2.vendor import single_file
        from core.models.krea2.residual_unet import restore_topology
        from core.rewiring.artifacts import read_artifact_topology
        from core.rewiring.spec import checkpoint_metadata, read_checkpoint_metadata
        from core.training.adapters.krea2_adapter import Krea2LoRAAdapter
        trainer = SimpleNamespace(transformer=self.model, config={}, krea2_is_distilled=False)
        adapter = Krea2LoRAAdapter(trainer, 4, 4)
        self.assertEqual(read_checkpoint_metadata(adapter.checkpoint_metadata({}, 1, 0)), self.spec)
        with torch.no_grad():
            expected = self.forward()
        for limit in (10**9, 12000):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"krea.safetensors"
                single_file.save_single_file(str(path), self.model, False, max_shard_bytes=limit)
                actual_path = path if path.exists() else Path(str(path)+".index.json")
                self.assertEqual(read_artifact_topology(actual_path), self.spec)
                restored = single_file.load_single_file(str(actual_path), torch.float32)["transformer"]
                self.assertEqual(restored._residual_unet_spec, self.spec)
                for key, value in self.model.state_dict().items():
                    torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
                with torch.no_grad():
                    actual = restored(self.image, self.text, torch.tensor([0.5]),
                                      self.positions, self.mask, return_dict=False)[0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "override"):
                    single_file.save_single_file(str(Path(directory)/"bad.safetensors"),
                        self.model, False, extra_metadata=checkpoint_metadata(
                            dict(self.spec, base_config_sha256="0"*64)))
                if str(actual_path).endswith(".index.json"):
                    index = json.loads(actual_path.read_text(encoding="utf-8"))
                    index["metadata"]["krea2_config"] = json.dumps({
                        **dict(self.model.config), "norm_eps": 0.01})
                    actual_path.write_text(json.dumps(index), encoding="utf-8")
                    with patch.object(single_file, "_read_safetensors") as reader:
                        with self.assertRaisesRegex(ValueError, "native config mismatch"):
                            single_file.load_single_file(str(actual_path), torch.float32)
                        reader.assert_not_called()
        restore_topology(self.model, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"dense.safetensors"
            single_file.save_single_file(str(path), self.model, False)
            self.assertIsNone(read_artifact_topology(path))
            restored = single_file.load_single_file(str(path), torch.float32)["transformer"]
            self.assertFalse(hasattr(restored, "_residual_unet_spec"))

    def test_directory_preflight_and_restoration(self):
        import json
        import tempfile
        from safetensors.torch import save_file
        from core.models.krea2 import krea2_loader
        from core.rewiring.spec import checkpoint_metadata
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.model.save_config(root)
            save_file(self.model.state_dict(), str(root/"diffusion_pytorch_model.safetensors"),
                      metadata={"format": "pt", **checkpoint_metadata(self.spec)})
            restored = krea2_loader._build_transformer_from_dir(directory, torch.float32)
            self.assertEqual(restored._residual_unet_spec, self.spec)
            for key, value in self.model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            config = json.loads((root/"config.json").read_text(encoding="utf-8"))
            config["norm_eps"] = 0.01
            (root/"config.json").write_text(json.dumps(config), encoding="utf-8")
            with patch.object(krea2_loader, "_load_component_state_dict") as reader:
                with self.assertRaisesRegex(ValueError, "native config mismatch"):
                    krea2_loader._build_transformer_from_dir(directory, torch.float32)
                reader.assert_not_called()

if __name__ == "__main__":
    unittest.main()
