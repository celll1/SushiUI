"""Actual adapter-session topology installation, atomic refusal and restoration."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import save_file
with patch.object(torch.cuda,"get_device_capability",return_value=(8,9)), \
     patch.object(torch.cuda,"_lazy_init"),patch.object(torch._C,"_cuda_init"):
    from core.models.qwen_image_21.vendor.transformer import QwenImage21Transformer2DModel
    from core.models.qwen_image_21.residual_unet import native_config, restore_topology
    from core.pipeline_backends.qwen_image_21 import QwenImage21Mixin
    from core.adapters import AdapterIncompatible, CompositeAdapterLayer
from core.rewiring.spec import OPERATORS,build_spec,checkpoint_metadata


class Manager(QwenImage21Mixin):
    def __init__(self,model):
        self.qwen_image_21_components={"transformer":model,"transformer_variant":"bf16"}
        self.cancel_requested=False
        self.device="cpu"
    @staticmethod
    def _qwen21_resolve_lora_path(path):
        return str(path) if Path(path).is_file() else None


class LoRATopologyContracts(unittest.TestCase):
    def test_real_session_atomicity_and_generation(self):
        model=QwenImage21Transformer2DModel(num_layers=7,num_attention_heads=1,
            attention_head_dim=12,axes_dims_rope=(4,4,4),context_in_dim=12,mlp_ratio=1)
        regions=[dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,
            down_factor=[2,2],**OPERATORS),dict(id="inner",stream="main",parent="outer",
            start_block=2,end_block=5,down_factor=[2,2],**OPERATORS)]
        topology=build_spec("qwen_image_21",[{"name":"main","block_count":7}],regions,native_config(dict(model.config)))
        other=build_spec("qwen_image_21",topology["streams"],
            [regions[0],{**regions[1],"end_block":4}],native_config(dict(model.config)))
        manager=Manager(model)
        original=model.transformer_blocks[0].attn.to_q
        with tempfile.TemporaryDirectory() as folder:
            def write(name,spec,wrong_shape=False):
                path=Path(folder)/f"{name}.safetensors"
                tensors={"lora_unet_transformer_blocks__0__attn__to_q.lora_down.weight":torch.randn(2,11 if wrong_shape else 12),
                         "lora_unet_transformer_blocks__0__attn__to_q.lora_up.weight":torch.randn(12,2)}
                save_file(tensors,str(path),metadata={"model_type":"qwen_image_21",**({} if spec is None else checkpoint_metadata(spec))})
                return {"path":str(path),"strength":1.0}
            good=write("good",topology); conflict=write("conflict",other); dense=write("dense",None)
            bad_shape=write("bad_shape",topology,True)
            self.assertEqual(manager._load_lora_qwen21([good]),1)
            self.assertEqual(model._residual_unet_spec,topology)
            self.assertIsInstance(model.transformer_blocks[0].attn.to_q,CompositeAdapterLayer)
            manager._unload_lora_qwen21()
            self.assertIs(model.transformer_blocks[0].attn.to_q,original)
            self.assertFalse(hasattr(model,"_residual_unet_spec"))
            for files in ([good,conflict],[good,dense],[good,bad_shape]):
                with self.assertRaises(RuntimeError): manager._load_lora_qwen21(files)
                self.assertIs(model.transformer_blocks[0].attn.to_q,original)
                self.assertFalse(hasattr(model,"_residual_unet_spec"))
            manager._load_lora_qwen21([dense])
            self.assertFalse(hasattr(model,"_residual_unet_spec"))
            manager._unload_lora_qwen21()
            restore_topology(model,topology)
            with self.assertRaises(AdapterIncompatible): manager._load_lora_qwen21([dense])
            self.assertEqual(model._residual_unet_spec,topology)
            manager._load_lora_qwen21([good]); manager._unload_lora_qwen21()
            self.assertEqual(model._residual_unet_spec,topology)
            restore_topology(model,None)
            calls=[]
            class Pipe:
                def __call__(self,**kwargs):
                    calls.append(kwargs)
                    return SimpleNamespace(images=["image"])
            manager.qwen_image_21_components["pipeline"]=Pipe()
            params=dict(seed=1,width=192,height=128,steps=2,loras=[good],
                        qwen_image_21_partition={"mode":"off"},qwen_image_21_kv_cache=True)
            image,_,_=manager._qwen_image_21_run(params)
            self.assertEqual(image,"image")
            self.assertFalse(calls[0]["use_kv_cache"])
            self.assertEqual(params["dit_rewiring_actual"]["spec"],topology)
            self.assertFalse(hasattr(model,"_residual_unet_spec"))
            self.assertIs(model.transformer_blocks[0].attn.to_q,original)
            class BrokenPipe:
                def __call__(self,**kwargs):
                    raise RuntimeError("denoising failed")
            manager.qwen_image_21_components["pipeline"]=BrokenPipe()
            with self.assertRaisesRegex(RuntimeError,"denoising failed"):
                manager._qwen_image_21_run(params)
            self.assertFalse(hasattr(model,"_residual_unet_spec"))
            self.assertIs(model.transformer_blocks[0].attn.to_q,original)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
