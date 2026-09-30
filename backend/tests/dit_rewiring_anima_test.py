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
        self.assertEqual(native_config(self.model)["crossattn_emb_channels"], 12)
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


    def test_full_writer_loader_and_lora_metadata(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        from core.models.anima import anima_loader
        from core.models.anima.residual_unet import native_config
        from core.training.adapters.anima_adapter import AnimaFullParameterAdapter, AnimaLoRAAdapter
        from core.rewiring.spec import read_checkpoint_metadata
        trainer = SimpleNamespace(transformer=self.model, config={}, vae=None)
        adapter = AnimaFullParameterAdapter(trainer)
        lora = AnimaLoRAAdapter(trainer, 2, 2)
        self.assertEqual(read_checkpoint_metadata(lora.checkpoint_metadata({}, 1, 0)), self.spec)
        with torch.no_grad():
            expected = self.forward()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"anima.safetensors"
            with patch("api.param_defaults.resolve_bundle_vae", return_value=False), \
                 patch("core.training.vae_swap.swap_metadata", return_value=(None, False, {})):
                written = adapter.write_checkpoint(1, 0, path)
            self.assertEqual(written, path)
            restored = anima_loader.load_anima_dit(str(path), dtype=torch.float32)
            self.assertEqual(restored._residual_unet_spec, self.spec)
            self.assertEqual(native_config(restored), native_config(self.model))
            for key, value in self.model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            with torch.no_grad():
                actual = restored(self.image, torch.tensor([0.5]), self.text, padding_mask=self.padding)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
            from core.models.common.single_file_format import save_single_file_state
            sharded = Path(directory)/"sharded.safetensors"
            save_single_file_state(load_file(str(path)), metadata, str(sharded), max_shard_bytes=12000)
            from core.rewiring.artifacts import read_artifact_topology
            self.assertEqual(read_artifact_topology(sharded), self.spec)
            restored_shards = anima_loader.load_anima_dit(str(sharded), dtype=torch.float32)
            self.assertEqual(restored_shards._residual_unet_spec, self.spec)
            for key, value in self.model.state_dict().items():
                torch.testing.assert_close(value, restored_shards.state_dict()[key], rtol=0, atol=0)
            altered = json.loads(metadata["transformer_config"]); altered["mlp_ratio"] = 2.0
            metadata["transformer_config"] = json.dumps(altered)
            save_file(load_file(str(path)), str(path), metadata=metadata)
            with patch.object(anima_loader, "Anima") as constructor:
                with self.assertRaisesRegex(ValueError, "native config mismatch"):
                    anima_loader.load_anima_dit(str(path), dtype=torch.float32)
                constructor.assert_not_called()
        self.assertFalse(torch.cuda.is_initialized())

    @patch("core.rewiring.training.RUNTIME_ARCHITECTURES", ("anima",))
    def test_lora_resume_atomicity_and_topology_agreement(self):
        import tempfile
        from types import SimpleNamespace
        from safetensors.torch import save_file
        from core.models.anima.residual_unet import configure_training_topology, restore_topology
        from core.training.adapters.anima_adapter import AnimaLoRAAdapter
        from core.training.lora_trainer import LoRATrainer
        from core.rewiring.spec import checkpoint_metadata
        restore_topology(self.model, None)
        trainer = SimpleNamespace(transformer=self.model, config={}, is_anima=True,
            log_prefix="[AnimaResumeTest]", lora_rank=2, lora_alpha=2,
            adapter_algorithm="lora", weight_decompose=False, adapter_config={})
        adapter = AnimaLoRAAdapter(trainer, 2, 2)
        layers = {}; adapter.apply_lora_to_unet(layers); trainer.lora_layers = layers
        tensors = {f"{name}.{key}": value.detach().clone()+0.01
                   for name, layer in layers.items() for key, value in layer.branch_tensors().items()}
        before = {f"{name}.{key}": value.detach().clone()
                  for name, layer in layers.items() for key, value in layer.branch_tensors().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"resume.safetensors"
            metadata = {"model_type": "anima", "step": "17", **checkpoint_metadata(self.spec)}
            save_file({"unrelated": torch.ones(1)}, str(path), metadata=metadata)
            with self.assertRaisesRegex(ValueError, "missing required tensor"):
                LoRATrainer.load_checkpoint(trainer, str(path))
            self.assertFalse(hasattr(self.model, "_residual_unet_spec"))
            self.assertEqual(trainer.config, {})
            for name, layer in layers.items():
                for key, value in layer.branch_tensors().items():
                    torch.testing.assert_close(value, before[f"{name}.{key}"], rtol=0, atol=0)
            save_file(tensors, str(path), metadata=metadata)
            self.assertEqual(LoRATrainer.load_checkpoint(trainer, str(path)), 17)
            self.assertEqual(self.model._residual_unet_spec, self.spec)
            for name, layer in layers.items():
                for key, value in layer.branch_tensors().items():
                    torch.testing.assert_close(value, tensors[f"{name}.{key}"], rtol=0, atol=0)
        self.assertEqual(configure_training_topology(self.model, trainer.config, resuming=True), self.spec)
        changed = dict(self.spec, regions=[self.spec["regions"][0],
            {**self.spec["regions"][1], "end_block": 4}])
        with self.assertRaisesRegex(ValueError, "without edits"):
            configure_training_topology(self.model,
                {"dit_rewiring_mode": "configure", "dit_rewiring_spec": changed}, resuming=True)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            configure_training_topology(self.model, {**trainer.config, "repa_enable": True})

    def test_generation_lora_scope_and_effective_metadata(self):
        import tempfile
        from safetensors.torch import save_file
        from core.pipeline_backends.anima import AnimaMixin
        from core.models.anima.residual_unet import restore_topology
        from core.adapters import AdapterIncompatible, CompositeAdapterLayer
        from core.rewiring.spec import checkpoint_metadata, digest
        class Manager(AnimaMixin):
            def __init__(manager):
                manager.anima_components = {"transformer": self.model}
            @staticmethod
            def _anima_resolve_lora_path(path):
                return str(path) if Path(path).is_file() else None
        manager = Manager()
        restore_topology(self.model, None)
        original = self.model.blocks[0].self_attn.q_proj
        with tempfile.TemporaryDirectory() as directory:
            def write(name, topology, wrong=False):
                path = Path(directory)/f"{name}.safetensors"
                save_file({
                    "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.randn(2, 23 if wrong else 24),
                    "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": torch.randn(24, 2)}, str(path),
                    metadata={"model_type": "anima", **({} if topology is None else checkpoint_metadata(topology))})
                return {"path": str(path), "strength": 1.0}
            good = write("good", self.spec)
            dense = write("dense", None)
            malformed = write("bad_shape", self.spec, True)
            self.assertEqual(manager._load_lora_anima([good]), 1)
            self.assertIsInstance(self.model.blocks[0].self_attn.q_proj, CompositeAdapterLayer)
            self.assertEqual(self.model._residual_unet_spec, self.spec)
            params = {}
            manager._anima_rewiring_metadata(params, self.model, 128, 192)
            self.assertEqual(params["dit_rewiring_actual"]["spec_sha256"], digest(self.spec))
            with self.assertRaisesRegex(Exception, "style"):
                manager._anima_rewiring_metadata({"style_transfer": {"enabled": True}},
                    self.model, 128, 192)
            with self.assertRaisesRegex(Exception, "FBCache"):
                manager._anima_rewiring_metadata({"fbcache_enable": True, "fbcache_threshold": 0.1},
                    self.model, 128, 192)
            with self.assertRaisesRegex(Exception, "divisible"):
                manager._anima_rewiring_metadata({}, self.model, 144, 192)
            manager._unload_lora_anima()
            self.assertIs(self.model.blocks[0].self_attn.q_proj, original)
            self.assertFalse(hasattr(self.model, "_residual_unet_spec"))
            for files in ([good, dense], [good, malformed]):
                with self.assertRaises(RuntimeError):
                    manager._load_lora_anima(files)
                self.assertIs(self.model.blocks[0].self_attn.q_proj, original)
                self.assertFalse(hasattr(self.model, "_residual_unet_spec"))
            restore_topology(self.model, self.spec)
            with self.assertRaises(AdapterIncompatible):
                manager._load_lora_anima([dense])
            manager._load_lora_anima([good]); manager._unload_lora_anima()
            self.assertEqual(self.model._residual_unet_spec, self.spec)

    def test_internal_default_preset(self):
        from collections import Counter
        from types import SimpleNamespace
        from core.models.anima.anima_models import ANIMA_DIT_CONFIG
        from core.models.anima.residual_unet import default_topology, configure_training_topology
        from core.rewiring.spec import preview_plan
        model = SimpleNamespace(config=dict(ANIMA_DIT_CONFIG))
        with patch("core.rewiring.training.RUNTIME_ARCHITECTURES", ("anima",)):
            spec = configure_training_topology(model, {"dit_rewiring_mode": "configure"})
        self.assertEqual(spec, default_topology(model.config))
        self.assertEqual(Counter(b["tokens"] for b in preview_plan(spec, 64, 64)["blocks"]),
                         {4096: 8, 1024: 16, 256: 4})
        self.assertIsNone(default_topology(dict(model.config, num_blocks=7)))


if __name__ == "__main__":
    unittest.main()
