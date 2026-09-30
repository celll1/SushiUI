"""Compare native logical parameter counts with independently constructed models."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from core.rewiring.parameters import parameter_counts
with patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)), \
     patch.object(torch.cuda, "_lazy_init"), patch.object(torch._C, "_cuda_init"):
    from core.models.qwen_image_21.vendor.transformer import QwenImage21Transformer2DModel
    from core.models.qwen_image_21.artifact import DEFAULT_TRANSFORMER_CONFIG


class ParameterCensusContracts(unittest.TestCase):
    def test_native_meta_models_match_per_block_and_shared_census(self):
        configs = [dict(DEFAULT_TRANSFORMER_CONFIG),
                   dict(DEFAULT_TRANSFORMER_CONFIG, num_layers=7, num_attention_heads=1,
                        attention_head_dim=12, axes_dims_rope=(4, 4, 4), context_in_dim=17,
                        mlp_ratio=2, patch_size=2, in_channels=5, out_channels=9),
                   dict(DEFAULT_TRANSFORMER_CONFIG, num_layers=3, num_attention_heads=2,
                        attention_head_dim=12, axes_dims_rope=(4, 4, 4), context_in_dim=7,
                        mlp_ratio=1, in_channels=3, out_channels=None)]
        for config in configs:
            with self.subTest(config=config), torch.device("meta"):
                native = QwenImage21Transformer2DModel(**config)
                actual = parameter_counts("qwen_image_21", config)
                self.assertEqual(actual["total"], sum(p.numel() for p in native.parameters()))
                for block, count in zip(native.transformer_blocks, actual["blocks"]):
                    self.assertEqual(count["count"], sum(p.numel() for p in block.parameters()))
                self.assertEqual(actual["shared"], actual["total"] - sum(b["count"] for b in actual["blocks"]))
                self.assertEqual(actual["added"], 0)
        self.assertEqual(parameter_counts("qwen_image_21", DEFAULT_TRANSFORMER_CONFIG)["total"], 7115124736)
        self.assertFalse(torch.cuda.is_initialized())

    def test_unknown_census_is_not_a_guessed_parameter_count(self):
        self.assertIsNone(parameter_counts("ltx2", {}))
        with self.assertRaises(ValueError):
            parameter_counts("qwen_image_21", dict(DEFAULT_TRANSFORMER_CONFIG, num_layers=True))
