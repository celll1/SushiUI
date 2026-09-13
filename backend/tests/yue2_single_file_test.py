"""Dense YuE2 single-file mapping and writer tests."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models.yue2.loader import TOKENIZER_KEY, build_empty_models, key_mapping
from core.models.yue2.single_file import model_key_to_single_file, save_yue2_single_file


class YuE2SingleFile(unittest.TestCase):
    def test_native_and_complete_names_round_trip(self):
        model, _ = build_empty_models()
        for key in model.state_dict():
            complete = model_key_to_single_file(key)
            self.assertEqual([item.target for item in key_mapping(complete)], [key])

    def test_writer_emits_dense_complete_metadata(self):
        class TinyTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = torch.nn.Linear(2, 3, bias=False)

        class TinyVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.ones(1))

        tokenizer = SimpleNamespace(payload='{"version":"1.0"}')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.safetensors"
            save_yue2_single_file(path, TinyTransformer(), TinyVAE(), tokenizer)
            with safe_open(path, framework="pt", device="cpu") as handle:
                self.assertEqual(handle.metadata()["model_type"], "yue2")
                self.assertEqual(handle.metadata()["yue2_weight_storage"], "dense_bf16")
                self.assertEqual(handle.get_tensor("text_encoders.model.lm_head.weight").dtype,
                                 torch.bfloat16)
                self.assertEqual(bytes(handle.get_tensor(TOKENIZER_KEY).tolist()).decode("utf-8"),
                                 tokenizer.payload)
                self.assertIn("vae.scale", handle.keys())


if __name__ == "__main__":
    unittest.main()
