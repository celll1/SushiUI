"""CPU contracts for the acoustic-only YuE2 residual block loop."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

from core.models.yue2.vendor.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from core.models.yue2.vendor.nar import CachedNAR, Chunk
from core.models.yue2.residual_unet import (native_config, restore_topology,
    default_topology, model_topology_metadata)
from core.rewiring.spec import OPERATORS, build_spec, preview_plan, read_checkpoint_metadata


def fixture(tokens=13):
    torch.manual_seed(67)
    config = YuE2Config(hidden_size=16, num_hidden_layers=7, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, intermediate_size=24, vocab_size=32,
        max_position_embeddings=64, max_latent_frames=64)
    model = YuE2ForCausalLM(config).eval()
    graph = build_spec("yue2", [{"name": "nar", "block_count": 7}], [
        dict(id="r0", stream="nar", parent=None, start_block=1, end_block=6,
             down_factor=[2, 1], **OPERATORS),
        dict(id="r1", stream="nar", parent="r0", start_block=2, end_block=5,
             down_factor=[2, 1], **OPERATORS)], native_config(config))
    return model, graph, Chunk([1, 2, 3], torch.randn(tokens, 64), nar_cond_end=2)


class YuE2Residual(unittest.TestCase):
    def test_complete_file_restores_native_execution(self):
        from core.models.yue2 import loader
        from core.models.yue2.single_file import save_yue2_single_file
        model, graph, chunk = fixture()
        model = model.bfloat16()
        restore_topology(model, graph)
        vae = torch.nn.Module()
        vae.register_buffer("scale", torch.ones(1))
        tokenizer = SimpleNamespace(payload='{"version":"1.0"}')
        engine = CachedNAR(model, chunk)
        expected = engine.velocity(chunk.noise.bfloat16(), .3)
        engine.close()
        def empty():
            with torch.device("meta"):
                copied = YuE2ForCausalLM(deepcopy(model.config))
                copied_vae = torch.nn.Module()
                copied_vae.register_buffer("scale", torch.ones(1))
            return copied, copied_vae
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete.safetensors"
            save_yue2_single_file(path, model, vae, tokenizer)
            with patch.object(loader, "build_empty_models", side_effect=empty), \
                    patch.object(loader, "EmbeddedTokenizer", side_effect=lambda payload: SimpleNamespace(payload=payload)):
                preflight = loader.preflight_yue2(path)
                self.assertEqual(preflight["residual_topology"], graph)
                loaded = loader.load_yue2_from_path(path)
            self.assertEqual(loaded["transformer"]._residual_unet_spec, graph)
            self.assertEqual(loaded["tokenizer"].payload, tokenizer.payload)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(loaded["transformer"].state_dict()[key], value, rtol=0, atol=0)
            restored = CachedNAR(loaded["transformer"], chunk)
            torch.testing.assert_close(restored.velocity(chunk.noise.bfloat16(), .3), expected, rtol=0, atol=0)
            restored.close()

    def test_invalid_metadata_refuses_before_payload_read(self):
        from core.models.yue2 import loader
        from core.rewiring.spec import checkpoint_metadata, META_SPEC, META_HASH
        model, graph, chunk = fixture()
        metadata = checkpoint_metadata(graph)
        base = {"model.diffusion_model.vae2llm.weight": {},
                "model.diffusion_model.latent_pos_embed.pe": {}, loader.TOKENIZER_KEY: {}}
        variants = [{META_SPEC: metadata[META_SPEC]}, {**metadata, META_HASH: "0" * 64}]
        for declaration in variants:
            with self.subTest(declaration=declaration), \
                    patch.object(loader, "read_header", return_value={**base, "__metadata__": {"yue2_format": "1", **declaration}}), \
                    patch.object(loader, "build_empty_models", side_effect=AssertionError("model construction reached")), \
                    patch.object(loader, "safe_open", side_effect=AssertionError("payload reader reached")):
                with self.assertRaises(ValueError):
                    loader.preflight_yue2("invalid.safetensors")
        model.config.timestep_shift = 2.
        with patch.object(loader, "read_header", return_value={**base, "__metadata__": {"yue2_format": "1", **metadata}}), \
                patch.object(loader, "build_empty_models", return_value=(model, torch.nn.Module())), \
                patch.object(loader, "safe_open", side_effect=AssertionError("payload reader reached")):
            with self.assertRaisesRegex(ValueError, "config"):
                loader.preflight_yue2("wrong_config.safetensors")

    def test_writer_refuses_injected_topology_and_dense_reload_is_dense(self):
        from core.models.yue2 import loader
        from core.models.yue2.single_file import save_yue2_single_file
        from core.rewiring.spec import checkpoint_metadata
        model, graph, chunk = fixture()
        vae = torch.nn.Module()
        vae.register_buffer("scale", torch.ones(1))
        tokenizer = SimpleNamespace(payload='{"version":"1.0"}')
        def empty():
            with torch.device("meta"):
                copied = YuE2ForCausalLM(deepcopy(model.config))
                copied_vae = torch.nn.Module()
                copied_vae.register_buffer("scale", torch.ones(1))
            return copied, copied_vae
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.safetensors"
            with self.assertRaisesRegex(ValueError, "disagrees"):
                save_yue2_single_file(path, model, vae, tokenizer, extra_metadata=checkpoint_metadata(graph))
            self.assertFalse(path.exists())
            save_yue2_single_file(path, model, vae, tokenizer)
            with patch.object(loader, "build_empty_models", side_effect=empty), \
                    patch.object(loader, "EmbeddedTokenizer", side_effect=lambda payload: SimpleNamespace(payload=payload)):
                preflight = loader.preflight_yue2(path)
                self.assertIsNone(preflight["residual_topology"])
                loaded = loader.load_yue2_from_path(path)
                self.assertFalse(hasattr(loaded["transformer"], "_residual_unet_spec"))
                # The file was replaced with a dense artifact after a residual preflight.
                preflight["residual_topology"] = graph
                with patch.object(loader, "preflight_yue2", return_value=preflight):
                    with self.assertRaisesRegex(ValueError, "changed after preflight"):
                        loader.load_yue2_from_path(path)

    def test_native_loop_preserves_ar_cache_and_weights(self):
        model, graph, chunk = fixture()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        dense = CachedNAR(model, chunk)
        cached = [(k.clone(), v.clone()) for k, v in dense.cache]
        dense.close()
        restore_topology(model, graph)
        calls, handles = [], []
        for index, layer in enumerate(model.model.layers):
            def hook(module, inputs, index=index):
                calls.append((index, inputs[0].shape[1]))
            handles.append(layer.nar_input_layernorm.register_forward_pre_hook(hook))
        engine = CachedNAR(model, chunk)
        self.assertEqual(engine.residual_state["padded_tokens"], 16)
        for (k, v), (expected_k, expected_v) in zip(engine.cache, cached):
            torch.testing.assert_close(k, expected_k, rtol=0, atol=0)
            torch.testing.assert_close(v, expected_v, rtol=0, atol=0)
        cache_ptrs = [(k.data_ptr(), v.data_ptr()) for k, v in engine.cache]
        with patch.object(engine, "_prefill", side_effect=AssertionError("AR replay")):
            out = engine.solve(steps=2)
        self.assertEqual(out.shape, chunk.noise.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(calls[:7], list(enumerate([18, 10, 6, 6, 6, 10, 18])))
        self.assertEqual(len(calls), 28)
        self.assertEqual(cache_ptrs, [(k.data_ptr(), v.data_ptr()) for k, v in engine.cache])
        for (k, v), (expected_k, expected_v) in zip(engine.cache, cached):
            torch.testing.assert_close(k, expected_k, rtol=0, atol=0)
            torch.testing.assert_close(v, expected_v, rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        for handle in handles:
            handle.remove()
        self.assertEqual(read_checkpoint_metadata(model_topology_metadata(model)), graph)
        engine.close()

    def test_zero_updates_preserve_native_boundaries_and_head(self):
        model, graph, chunk = fixture()
        with torch.no_grad():
            for layer in model.model.layers:
                layer.nar_self_attn.o_proj.weight.zero_()
                layer.nar_mlp.down_proj.weight.zero_()
        dense = CachedNAR(model, chunk)
        expected = dense.velocity(chunk.noise, .3)
        dense.close()
        restore_topology(model, graph)
        residual = CachedNAR(model, chunk)
        torch.testing.assert_close(residual.velocity(chunk.noise, .3), expected, rtol=0, atol=0)
        residual.close()

    def test_boundary_rotary_positions_and_ar_visibility(self):
        model, graph, chunk = fixture(12)
        restore_topology(model, graph)
        engine = CachedNAR(model, chunk)
        seen = []
        original = model.model.rotary_emb.forward
        def rotary(positions):
            seen.append(positions.clone())
            return original(positions)
        with patch.object(model.model.rotary_emb, "forward", side_effect=rotary):
            engine.velocity(chunk.noise, .2)
        self.assertEqual([len(p[0]) for p in seen], [14, 8, 5])
        for positions in seen:
            self.assertEqual(positions[0, :2].tolist(), [3, 16])
            self.assertEqual(positions[0, 2:].tolist(), list(range(4, 4 + len(positions[0]) - 2)))
        self.assertTrue(all(len(k) == 2 and len(v) == 2 for k, v in engine.cache))
        engine.close()

    def test_padded_rotary_does_not_duplicate_boundary(self):
        model, graph, chunk = fixture(13)
        restore_topology(model, graph)
        engine = CachedNAR(model, chunk)
        seen = []
        original = model.model.rotary_emb.forward
        def rotary(positions):
            seen.append(positions[0].tolist())
            return original(positions)
        with patch.object(model.model.rotary_emb, "forward", side_effect=rotary):
            engine.velocity(chunk.noise, .2)
        self.assertEqual(seen[0], [3, 17] + list(range(4, 17)) + [18, 19, 20])
        self.assertTrue(all(len(set(positions)) == len(positions) for positions in seen))
        engine.close()

    def test_refuses_graph_changes_before_velocity(self):
        model, graph, chunk = fixture()
        engine = CachedNAR(model, chunk)
        restore_topology(model, graph)
        with self.assertRaisesRegex(ValueError, "changed after AR prefill"):
            engine.velocity(chunk.noise, 0.)
        engine.close()
        engine = CachedNAR(model, chunk)
        restore_topology(model, None)
        with self.assertRaisesRegex(ValueError, "changed after AR prefill"):
            engine.velocity(chunk.noise, 0.)
        engine.close()

    def test_context_and_config_refuse_before_ar_prefill(self):
        model, graph, chunk = fixture(13)
        model.config.max_position_embeddings = 19
        graph = build_spec("yue2", graph["streams"], graph["regions"], native_config(model.config))
        restore_topology(model, graph)
        with patch.object(CachedNAR, "_prefill", side_effect=AssertionError("prefill reached")):
            with self.assertRaisesRegex(ValueError, "padded acoustic chunk"):
                CachedNAR(model, chunk)
            model.config.timestep_shift = 2.
            with self.assertRaisesRegex(ValueError, "config"):
                CachedNAR(model, chunk)

    def test_default_is_nar_only_with_four_deepest_blocks(self):
        config = YuE2Config()
        graph = default_topology(config)
        self.assertEqual(graph["streams"], [{"name": "nar", "block_count": 28}])
        plan = preview_plan(graph, 100, 1)
        self.assertEqual([sum(block["depth"] == depth for block in plan["blocks"])
                          for depth in range(3)], [8, 16, 4])
        self.assertEqual(graph["regions"][1]["end_block"] - graph["regions"][1]["start_block"], 4)
        self.assertEqual(default_topology(fixture()[0].config), None)
        from core.rewiring.training import RUNTIME_ARCHITECTURES
        self.assertNotIn("yue2", RUNTIME_ARCHITECTURES)


if __name__ == "__main__":
    unittest.main()
