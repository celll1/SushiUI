"""CPU contracts for native Flux2 double/single residual execution."""
from pathlib import Path
import sys
import unittest
import json
import tempfile
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from core.rewiring.spec import build_spec, OPERATORS


class Flux2RuntimeContracts(unittest.TestCase):
    def setUp(self):
        for target,name,value in ((torch.cuda,"get_device_capability",lambda *a,**k:(8,9)),
                (torch.cuda,"_lazy_init",lambda *a,**k:None),(torch._C,"_cuda_init",lambda *a,**k:None)):
            guard=patch.object(target,name,value);guard.start();self.addCleanup(guard.stop)
        from diffusers import Flux2Transformer2DModel
        from core.models.flux2.residual_unet import native_config
        torch.manual_seed(57)
        self.model=Flux2Transformer2DModel(num_layers=3,num_single_layers=7,in_channels=4,
            num_attention_heads=2,attention_head_dim=8,axes_dims_rope=(2,2,2,2),
            joint_attention_dim=12,timestep_guidance_channels=8,mlp_ratio=1,guidance_embeds=True)
        self.spec=build_spec("flux2",[{"name":"double","block_count":3},{"name":"single","block_count":7}],[
            dict(id="double_outer",stream="double",parent=None,start_block=1,end_block=2,down_factor=[2,2],**OPERATORS),
            dict(id="single_outer",stream="single",parent=None,start_block=1,end_block=6,down_factor=[2,2],**OPERATORS),
            dict(id="single_inner",stream="single",parent="single_outer",start_block=2,end_block=5,down_factor=[2,2],**OPERATORS)],native_config(self.model))
        self.images=torch.randn(2,96,4,requires_grad=True)
        self.text=torch.randn(2,5,12,requires_grad=True)
        self.times=torch.tensor([0.3,0.7]);self.guidance=torch.tensor([1.5,2.5])
        self.ids=torch.zeros(96,4);self.ids[:,0]=1
        self.ids[:,1]=torch.arange(8).repeat_interleave(12);self.ids[:,2]=torch.arange(12).repeat(8)
        self.ids[:,3]=torch.arange(96)
        self.text_ids=torch.zeros(5,4);self.text_ids[:,3]=torch.arange(5)

    def forward(self,**options):
        args=dict(hidden_states=self.images,encoder_hidden_states=self.text,timestep=self.times,
                  guidance=self.guidance,img_ids=self.ids,txt_ids=self.text_ids,return_dict=False)
        args.update(options)
        return self.model(**args)[0]

    def test_full_checkpoint_native_loader_roundtrip_and_preflight(self):
        from core.models.flux2.residual_unet import restore_topology, native_config, preflight_checkpoint
        from core.training.adapters.flux2_adapter import FLUX2FullParameterAdapter, FLUX2LoRAAdapter
        from core.models.common.single_file_format import read_state_dict, save_single_file_state
        from core.model_loader import ModelLoader
        from safetensors.torch import save_file
        restore_topology(self.model,self.spec)
        expected=self.forward().detach()
        trainer=SimpleNamespace(transformer=self.model,train_unet=True,train_text_encoder=False,
            vae=None,text_encoder=None,bundle_vae=False,config={},is_distilled=False,base_model_repo="test/tiny-flux2")
        lora_metadata=FLUX2LoRAAdapter(trainer,4,4).checkpoint_metadata({},1,0)
        from core.rewiring.spec import read_checkpoint_metadata
        self.assertEqual(read_checkpoint_metadata(lora_metadata),self.spec)
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);base=folder/"base"; (base/"transformer").mkdir(parents=True)
            (base/"vae").mkdir()
            companion=dict(native_config(self.model),num_layers=4,num_single_layers=6,joint_attention_dim=16)
            (base/"transformer"/"config.json").write_text(json.dumps(companion),encoding="utf-8")
            single=FLUX2FullParameterAdapter(trainer).write_checkpoint(1,0,folder/"full.safetensors")
            self.assertEqual(preflight_checkpoint(single),(self.spec,native_config(self.model)))
            state,metadata=read_state_dict(str(single))
            sharded=save_single_file_state(state,metadata,str(folder/"sharded.safetensors"),max_shard_bytes=12000)
            vae=torch.nn.Linear(1,1);vae.config=SimpleNamespace(latent_channels=1)
            with ExitStack() as stack:
                for target,value in (
                    ("huggingface_hub.snapshot_download",str(base)),
                    ("core.models.common.vae_store.resolve_vae_dir",str(base/"vae")),
                    ("diffusers.AutoencoderKLFlux2.from_pretrained",vae),
                    ("transformers.Qwen3ForCausalLM.from_pretrained",torch.nn.Linear(1,1)),
                    ("transformers.Qwen2TokenizerFast.from_pretrained",object()),
                    ("diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained",object())):
                    stack.enter_context(patch(target,return_value=value))
                for candidate in (single,Path(sharded),folder/"sharded.safetensors"):
                    restored=ModelLoader.load_flux2_from_safetensors(str(candidate),device="cpu",torch_dtype=torch.float32)["transformer"]
                    self.assertEqual(restored._residual_unet_spec,self.spec)
                    self.assertEqual(native_config(restored),native_config(self.model))
                    for key,value in self.model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
                    with patch.object(self,"model",restored):
                        torch.testing.assert_close(self.forward(),expected,rtol=1e-5,atol=2e-7)
            for name,broken in (("missing-config",{k:v for k,v in metadata.items() if k!="transformer_config"}),
                    ("wrong-config",dict(metadata,transformer_config=json.dumps(companion)))):
                path=folder/f"{name}.safetensors";save_file(state,str(path),metadata=broken)
                with patch("core.models.common.single_file_format.read_state_dict",side_effect=AssertionError("tensor read")), \
                     patch("huggingface_hub.snapshot_download",side_effect=AssertionError("download")):
                    with self.assertRaises(ValueError):
                        ModelLoader.load_flux2_from_safetensors(str(path),device="cpu",torch_dtype=torch.float32)
            dense=folder/"dense.safetensors";save_file({"weight":torch.ones(1)},str(dense))
            self.assertEqual(preflight_checkpoint(dense),(None,None))
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_stream_lengths_rope_offloader_and_parameter_identity(self):
        from core.models.flux2.residual_unet import restore_topology
        from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper
        restore_topology(self.model,self.spec)
        parameters={name:id(value) for name,value in self.model.named_parameters()}
        text_rotary=self.model.pos_embed(self.text_ids)
        seen=[]
        def observe(stream):
            def hook(block,args):
                value=args[0];rotary=args[4 if stream=="double" else 3]
                image_count=value.shape[1] if stream=="double" else value.shape[1]-5
                if stream=="double":self.assertEqual(args[1].shape[:2],(2,5))
                for actual,expected in zip(rotary,text_rotary):torch.testing.assert_close(actual[:5],expected,rtol=0,atol=0)
                self.assertEqual(rotary[0].shape[0],5+image_count)
                seen.append((stream,image_count))
            return hook
        handles=[block.register_forward_pre_hook(observe(stream)) for stream,blocks in
                 (("double",self.model.transformer_blocks),("single",self.model.single_transformer_blocks)) for block in blocks]
        events=[]
        class Offloader:
            blocks_to_swap=1;num_dual_blocks=3
            def wait_for_block(owner,index):events.append(("wait",index))
            def submit_move_blocks_forward(owner,index):events.append(("submit",index))
        wrapper=Flux2BlockSwapWrapper(self.model,Offloader())
        try:
            actual=wrapper(self.images,self.text,timestep=self.times,img_ids=self.ids,
                           txt_ids=self.text_ids,guidance=self.guidance,return_dict=False)[0]
        finally:
            for handle in handles:handle.remove()
        self.assertEqual(seen,[("double",n) for n in (96,24,96)]+[("single",n) for n in (96,24,6,6,6,24,96)])
        self.assertEqual(events,[(name,index) for index in range(10) for name in ("wait","submit")])
        self.assertEqual(actual.shape,self.images.shape);self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(parameters,{name:id(value) for name,value in self.model.named_parameters()})
        torch.testing.assert_close(actual,self.forward(),rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_checkpointed_native_gradients_and_output(self):
        from core.models.flux2.residual_unet import restore_topology
        restore_topology(self.model,self.spec)
        parameters=[self.images,self.text,*self.model.parameters()]
        expected=self.forward();before=torch.autograd.grad(expected.square().mean(),parameters,allow_unused=True)
        self.model.enable_gradient_checkpointing()
        actual=self.forward();after=torch.autograd.grad(actual.square().mean(),parameters,allow_unused=True)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for a,b in zip(before,after):
            if a is None:self.assertIsNone(b)
            else:
                self.assertTrue(torch.isfinite(b).all());torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
        self.assertGreater(float(after[1].abs().sum()),0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_dense_restore_ids_and_feature_refusals(self):
        from core.models.flux2.residual_unet import restore_topology,model_topology_metadata,native_config
        from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper
        from core.rewiring.spec import read_checkpoint_metadata
        dense=self.forward();original=self.model.forward
        restore_topology(self.model,self.spec)
        self.assertEqual(read_checkpoint_metadata(model_topology_metadata(self.model)),self.spec)
        from copy import deepcopy
        clone=deepcopy(self.model)
        output=clone(self.images,self.text,timestep=self.times,img_ids=self.ids,
                     txt_ids=self.text_ids,guidance=self.guidance,return_dict=False)[0]
        torch.testing.assert_close(output,self.forward(),rtol=0,atol=0)
        restore_topology(clone,None)
        self.assertFalse(hasattr(clone,"_residual_unet_spec"))
        self.assertTrue(hasattr(self.model,"_residual_unet_spec"))
        batched_ids=self.ids.unsqueeze(0).repeat(2,1,1)
        torch.testing.assert_close(self.forward(img_ids=batched_ids),self.forward(),rtol=0,atol=0)
        constant_ids=self.ids.clone();constant_ids[:,0]=0;constant_ids[:,3]=0
        self.assertTrue(torch.isfinite(self.forward(img_ids=constant_ids)).all())
        for options in ({"kv_cache_mode":"extract"},{"num_ref_tokens":1},{"joint_attention_kwargs":{"scale":0.5}},
                        {"img_ids":self.ids.flip(0)}, {"img_ids":batched_ids+torch.tensor([0,1])[:,None,None]},
                        {"encoder_hidden_states":self.text[:1]}):
            with self.assertRaises(ValueError):self.forward(**options)
        for name in ("_style_ctx","_fbcache","_tread_config","_block_skip_config","_repa_tap_depth"):
            setattr(self.model,name,object())
            try:
                with self.assertRaises(ValueError):self.forward()
            finally:delattr(self.model,name)
        from core.inference.nag_flux2 import NAGFlux2AttnProcessor
        attention=self.model.transformer_blocks[0].attn
        processor=attention.processor
        attention.set_processor(object.__new__(NAGFlux2AttnProcessor))
        try:
            with self.assertRaisesRegex(ValueError,"processors"):self.forward()
        finally:attention.set_processor(processor)
        wrapper=Flux2BlockSwapWrapper(self.model,single_procs=[])
        with self.assertRaisesRegex(ValueError,"wrappers"):
            wrapper(self.images,self.text,timestep=self.times,img_ids=self.ids,txt_ids=self.text_ids)
        restore_topology(self.model,None)
        self.assertEqual(self.model.forward,original);self.assertNotIn("forward",self.model.__dict__)
        self.assertEqual(model_topology_metadata(self.model),{})
        torch.testing.assert_close(self.forward(),dense,rtol=0,atol=0)
        with self.assertRaises(ValueError):native_config(dict(self.model.config,axes_dims_rope=[2,2,2]))
        with self.assertRaises(ValueError):native_config(dict(self.model.config,patch_size=True))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__":unittest.main()
