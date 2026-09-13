"""CPU parity and protocol gates for YuE2; never loads production weight payloads."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

from core.models.yue2.loader import key_mapping, keys_look_yue2
from core.models.yue2.vendor.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from core.models.yue2.vendor.nar import CachedNAR, song_chunks
from core.models.yue2.vendor.protocol import LATENT_START, LATENT_END, LATENT_PAD, chunk_ranges, SongRequest, negative_prefix, token_prefixes
from core.models.yue2.vendor.modeling_vae import YuE2VAE, YuE2VAEConfig


class YuE2CPU(unittest.TestCase):
    def test_generated_abc_reserves_semantic_context_before_staging(self):
        from core.models.yue2.pipeline import validate_context_budget
        from core.models.yue2.vendor.protocol import GenerationConfig, Sampling
        class Tokenizer:
            def encode(self, text):
                return [1] * len(text)
        request = SongRequest("style", "lyrics", cot="full", seed=1)
        config = GenerationConfig(semantic=Sampling(max_tokens=21000))
        with self.assertRaisesRegex(ValueError, "Prefix .* budget exceeds"):
            validate_context_budget(request, Tokenizer(), config)
        off = SongRequest("style", "lyrics", cot="off", seed=1)
        self.assertTrue(validate_context_budget(off, Tokenizer(), config))

    def test_supplied_abc_uses_exact_semantic_context(self):
        from dataclasses import replace
        from core.models.yue2.pipeline import validate_context_budget
        from core.models.yue2.vendor.protocol import GenerationConfig
        class Tokenizer:
            def encode(self, text):
                return [1] * len(text)
        request = SongRequest("style", "lyrics", cot="full", abc="X:1\nK:C\nCDEF", seed=1)
        config = GenerationConfig()
        expected = token_prefixes(request, Tokenizer())
        config = replace(config, semantic=replace(config.semantic, max_tokens=config.context - len(expected)))
        self.assertEqual(validate_context_budget(request, Tokenizer(), config), expected)
        overflow = replace(config, semantic=replace(config.semantic, max_tokens=config.semantic.max_tokens + 1))
        with self.assertRaisesRegex(ValueError, "Prefix .* budget exceeds"):
            validate_context_budget(request, Tokenizer(), overflow)

    def test_zero_top_k_leaves_all_protocol_tokens(self):
        from core.models.yue2.vendor.protocol import Sampling, VOCAB_SIZE, CODEC_SIZE
        from core.models.yue2.vendor.sampling import distribution
        settings = Sampling(top_k=0, top_p=1, min_tokens=0, max_tokens=1)
        scores = distribution(torch.arange(VOCAB_SIZE)[None].float(), settings, [], 0, "semantic")
        self.assertEqual(torch.isfinite(scores).sum().item(), CODEC_SIZE + 1)

    def test_noise_is_one_cpu_song_draw(self):
        chunks = song_chunks([1, 2, 3], list(range(25)), 731, context=20)
        expected = torch.randn((25, 64), generator=torch.Generator().manual_seed(731))
        self.assertTrue(torch.equal(torch.cat([chunk.noise for chunk in chunks]), expected))
        self.assertEqual([len(chunk.noise) for chunk in chunks], [7, 7, 7, 4])

    def test_fused_row_splits_cover_source(self):
        for key, rows in (("text_encoders.model.layers.0.self_attn.qkv_proj.weight", 4096),
                          ("model.diffusion_model.model.layers.27.mlp.gate_up_proj.weight_scale", 12288)):
            source = torch.arange(rows)
            plan = key_mapping(key)
            self.assertTrue(torch.equal(torch.cat([source[p.start:p.end] for p in plan]), source))
        marker = key_mapping("text_encoders.model.layers.0.self_attn.qkv_proj.comfy_quant")
        self.assertEqual(len(marker), 3)
        self.assertTrue(all(p.start is None for p in marker))

    def test_cfg_reuses_exact_abc(self):
        class Tokenizer:
            def encode(self, text):
                return [42]
        request = SongRequest("style", "lyrics", cot="full", seed=1)
        ids = [11, 22, 33]
        self.assertEqual(token_prefixes(request, Tokenizer(), ids)[-5:-2], ids)
        self.assertEqual(negative_prefix(request, Tokenizer(), ids)[-5:-2], ids)
        with self.assertRaises(ValueError):
            SongRequest("style", "lyrics", cot="off", seed=1, abc="X:1")

    def test_convrot_embedding_gathers_before_dequantizing(self):
        from core.models.yue2.loader import ConvRotEmbedding
        from core.models.common.convrot_int8_linear import require_convrot_int8_runtime
        require_convrot_int8_runtime()
        embedding = ConvRotEmbedding(9, 256, 72, torch.float32)
        embedding.load_state_dict({"weight": torch.randint(-50, 50, (9, 256), dtype=torch.int8),
                                   "weight_scale": torch.linspace(.01, .09, 9),
                                   "comfy_quant": torch.zeros(72, dtype=torch.uint8)}, assign=True)
        ids = torch.tensor([[5, 2, 5]])
        from comfy_kitchen.tensor.int8_utils import _build_hadamard
        reference = (embedding.weight.float() * embedding.weight_scale[:, None]) @ _build_hadamard(256)
        torch.testing.assert_close(embedding(ids), reference[ids], rtol=1e-5, atol=1e-5)

    def test_quantized_timestep_embedder_has_no_parameters(self):
        from core.models.yue2.vendor.modeling_yue2 import TimestepEmbedder
        from core.models.common.convrot_int8_linear import ConvRotInt8Linear
        embedder = TimestepEmbedder(256)
        for index in (0, 2):
            linear = ConvRotInt8Linear(256, 256, True, torch.float32,
                                      convrot_groupsize=256, marker_numel=72, device="cpu")
            linear.weight.zero_()
            linear.weight_scale.fill_(1)
            linear.bias.zero_()
            embedder.mlp[index] = linear
        self.assertEqual(list(embedder.parameters()), [])
        with torch.inference_mode():
            self.assertTrue(torch.equal(embedder(torch.tensor([.5])), torch.zeros(1, 256)))

    def test_cached_nar_matches_dense_hybrid(self):
        torch.manual_seed(81)
        cfg = YuE2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, intermediate_size=48,
                         vocab_size=184704, max_position_embeddings=32, max_latent_frames=32)
        model = YuE2ForCausalLM(cfg).eval()
        chunk = song_chunks([1, 2], [0, 1, 2], seed=91)[0]
        engine = CachedNAR(model, chunk)
        tokens = torch.tensor([chunk.ar_tokens + [LATENT_START] + [LATENT_PAD] * 3 + [LATENT_END]])
        ar = torch.arange(tokens.shape[1])[None] < len(chunk.ar_tokens)
        nar = ~ar
        content = nar.clone()
        content[:, len(chunk.ar_tokens)] = False
        content[:, -1] = False
        dense = model.nar_velocity(tokens, ar, nar, content, chunk.noise, .3)
        cached = engine.velocity(chunk.noise, .3)
        torch.testing.assert_close(cached, dense, atol=1e-6, rtol=1e-5)
        with patch.object(engine, "velocity", wraps=engine.velocity) as velocity:
            result = engine.solve(steps=32)
            self.assertEqual(velocity.call_count, 64)
            self.assertEqual(result.shape, (3, 64))
        engine.close()

    def test_decoder_length_and_tiles(self):
        config = YuE2VAEConfig(
            encoder_config=dict(in_channels=2, channels=2, c_mults=[1, 2, 4, 8, 16, 32],
                                strides=[2, 2, 4, 4, 5, 6], latent_dim=128, use_snake=True),
            decoder_config=dict(out_channels=2, channels=2, c_mults=[1, 2, 4, 8, 16, 32],
                                strides=[2, 2, 4, 4, 5, 6], latent_dim=64, use_snake=True,
                                final_tanh=False))
        vae = YuE2VAE(config, decoder_only=True).eval()
        self.assertEqual(vae.natural_output_length(25), 1920 * 25 - 64)
        latent = torch.randn(1, 64, 20)
        full = vae.decode(latent)
        tiled = vae.decode_tiled(latent, core_frames=7, halo_frames=16)
        self.assertEqual(full.shape, (1, 2, 1920 * 20 - 64))
        torch.testing.assert_close(full, tiled, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
