"""Actual full-checkpoint writer/loader topology round trips and refusal cases."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from core.rewiring.spec import OPERATORS, build_spec, META_HASH
from core.rewiring.artifacts import read_artifact_topology
with patch.object(torch.cuda,"get_device_capability",return_value=(8,9)), \
     patch.object(torch.cuda,"_lazy_init"),patch.object(torch._C,"_cuda_init"):
    from core.models.qwen_image_21.vendor.transformer import QwenImage21Transformer2DModel
    from core.models.qwen_image_21.artifact import load_transformer
    from core.models.qwen_image_21.residual_unet import native_config, restore_topology
    import core.training.adapters.qwen_image_21_adapter as adapters


class ArtifactContracts(unittest.TestCase):
    def test_full_writer_loader_and_lora_metadata(self):
        torch.manual_seed(9)
        model = QwenImage21Transformer2DModel(num_layers=7,num_attention_heads=1,
            attention_head_dim=12,axes_dims_rope=(4,4,4),context_in_dim=12,mlp_ratio=1)
        config = dict(model.config)
        spec = build_spec("qwen_image_21",[{"name":"main","block_count":7}], [
            dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,
                 down_factor=[2,2],**OPERATORS),
            dict(id="inner",stream="main",parent="outer",start_block=2,end_block=5,
                 down_factor=[2,2],**OPERATORS)], native_config(config))
        restore_topology(model,spec)
        model.eval()
        trainer = SimpleNamespace(transformer=model,config={},train_text_encoder=False,
                                  model_path="unused_companion")
        adapter = adapters.QwenImage21FullParameterAdapter(trainer)
        lora = adapters.QwenImage21LoRAAdapter(trainer,4,4)
        from core.rewiring.spec import read_checkpoint_metadata
        self.assertEqual(read_checkpoint_metadata(lora.checkpoint_metadata({},1,0)),spec)
        x = torch.randn(1,96,64)
        text = torch.randn(1,5,12)
        slots = torch.zeros(1,29,dtype=torch.bool); slots[:,5:] = True
        with torch.no_grad():
            original = model(x,text,torch.tensor([0.5]),[[(1,8,12)]],slots).sample
        with tempfile.TemporaryDirectory() as directory:
            for limit in (10**9,12000):
                with patch.object(adapters,"DEFAULT_EXPORT_SHARD_BYTES",limit):
                    path = adapter.write_checkpoint(1,0,Path(directory)/f"test_{limit}.safetensors")
                self.assertEqual(read_artifact_topology(path),spec)
                restored,variant = load_transformer(str(path),config,torch.float32)
                self.assertEqual(restored._residual_unet_spec,spec)
                self.assertEqual(variant,"bf16")
                with torch.no_grad():
                    result = restored(x,text,torch.tensor([0.5]),[[(1,8,12)]],slots).sample
                for key,value in model.state_dict().items():
                    torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
                torch.testing.assert_close(original,result,rtol=1e-5,atol=5e-7)
                with self.assertRaisesRegex(ValueError,"native config mismatch"):
                    load_transformer(str(path),{**config,"eps":1e-4},torch.float32)
                if str(path).endswith(".index.json"):
                    index = json.loads(Path(path).read_text(encoding="utf-8"))
                    self.assertEqual(read_artifact_topology(str(path)[:-len(".index.json")]),spec)
                    from safetensors import safe_open
                    from safetensors.torch import load_file,save_file
                    shard = Path(path).parent / next(iter(index["weight_map"].values()))
                    with safe_open(str(shard),framework="pt") as handle:
                        metadata = dict(handle.metadata())
                    metadata[META_HASH] = "0"*64
                    tensors = load_file(str(shard))
                    save_file(tensors,str(shard),metadata=metadata)
                    with self.assertRaisesRegex(ValueError,"hash mismatch"):
                        load_transformer(str(path),config,torch.float32)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
