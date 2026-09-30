"""CPU contracts for Ideogram 4 native packed residual transport."""
from pathlib import Path
import sys
import unittest
import tempfile
import json
from copy import deepcopy
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from core.rewiring.spec import build_spec, OPERATORS, read_checkpoint_metadata


class Ideogram4RuntimeContracts(unittest.TestCase):
    def setUp(self):
        for target,name,value in ((torch.cuda,"get_device_capability",lambda *a,**k:(8,9)),
                (torch.cuda,"_lazy_init",lambda *a,**k:None),(torch._C,"_cuda_init",lambda *a,**k:None)):
            guard=patch.object(target,name,value);guard.start();self.addCleanup(guard.stop)
        from core.models.ideogram4.vendor.transformer import Ideogram4Transformer2DModel
        from core.models.ideogram4.ideogram4_pipeline_ops import _prepare_ids
        from core.models.ideogram4.residual_unet import native_config
        torch.manual_seed(104)
        self.model=Ideogram4Transformer2DModel(in_channels=8,num_layers=7,attention_head_dim=12,
            num_attention_heads=2,intermediate_size=32,adaln_dim=8,llm_features_dim=12,mrope_section=(2,2,2))
        self.spec=build_spec("ideogram4",[{"name":"main","block_count":7}],[
            dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,2],**OPERATORS),
            dict(id="inner",stream="main",parent="outer",start_block=2,end_block=5,down_factor=[2,2],**OPERATORS)],native_config(self.model))
        self.positions,self.segments,self.indicator=_prepare_ids([3,5],8,12,5,torch.device("cpu"))
        self.images=torch.randn(2,101,8,requires_grad=True)
        self.features=torch.randn(2,101,12,requires_grad=True)
        self.times=torch.tensor([0.3,0.7])

    def forward(self,**options):
        arguments=dict(hidden_states=self.images,timestep=self.times,encoder_hidden_states=self.features,
            position_ids=self.positions,segment_ids=self.segments,indicator=self.indicator,return_dict=False)
        arguments.update(options);return self.model(**arguments)[0]

    def test_native_masks_positions_evolving_text_and_offloader(self):
        from core.models.ideogram4.residual_unet import restore_topology
        restore_topology(self.model,self.spec)
        identities={name:id(value) for name,value in self.model.named_parameters()}
        counts=[];previous=[];events=[];rope_values=[]
        def observe(block,args):
            hidden,mask,rope,adaln,segments=args
            count=hidden.shape[1]-5;counts.append(count)
            self.assertEqual(segments[:,:5].tolist(),self.segments[:,:5].tolist())
            self.assertTrue(torch.equal(mask,(segments[:,:,None]==segments[:,None,:])[:,None]))
            self.assertFalse(mask[0,0,0,-1]);self.assertTrue(mask[0,0,2,-1])
            self.assertEqual(rope[0].shape,(2,count+5,12))
            if previous:torch.testing.assert_close(hidden[:,:5],previous[-1],rtol=0,atol=0)
        handles=[block.register_forward_pre_hook(observe) for block in self.model.layers]
        handles += [block.register_forward_hook(lambda block,args,out:previous.append(out[:,:5])) for block in self.model.layers]
        handle=self.model.rotary_emb.register_forward_pre_hook(lambda module,args:rope_values.append(args[0].detach().clone()))
        class Offloader:
            def wait_for_block(owner,index):events.append(("wait",index))
            def submit_move_blocks_forward(owner,index):events.append(("submit",index))
        self.model._block_offloader=Offloader()
        try:output=self.forward()
        finally:
            for item in handles:item.remove()
            handle.remove();del self.model._block_offloader
        self.assertEqual(counts,[96,24,6,6,6,24,96])
        self.assertEqual(events,[event for i in range(7) for event in (("wait",i),("submit",i))])
        self.assertEqual(identities,{name:id(value) for name,value in self.model.named_parameters()})
        self.assertEqual(output.shape,self.images.shape);self.assertTrue(torch.isfinite(output).all())
        coarse=next(pos for pos in rope_values if pos.shape[1]==29)
        deep=next(pos for pos in rope_values if pos.shape[1]==11)
        torch.testing.assert_close(coarse[0,5],torch.tensor([65536.,65536.5,65536.5]))
        torch.testing.assert_close(deep[0,5],torch.tensor([65536.,65537.5,65537.5]))
        torch.testing.assert_close(coarse[:,:5],self.positions[:,:5].float(),rtol=0,atol=0)
        # The unconditional backbone uses the same native image-only layout.
        value=self.forward(hidden_states=self.images[:,5:],encoder_hidden_states=self.features[:,5:],
            position_ids=self.positions[:,5:],segment_ids=self.segments[:,5:],indicator=self.indicator[:,5:])
        self.assertEqual(value.shape,(2,96,8));self.assertTrue(torch.isfinite(value).all())
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_checkpoint_output_gradients_and_dense_restoration(self):
        from core.models.ideogram4.residual_unet import restore_topology
        dense=self.forward().detach();restore_topology(self.model,self.spec)
        parameters=[self.images,self.features,*self.model.parameters()]
        direct=self.forward();grads=torch.autograd.grad(direct.square().mean(),parameters,allow_unused=True)
        self.model.enable_gradient_checkpointing()
        checkpointed=self.forward();checkpoint_grads=torch.autograd.grad(checkpointed.square().mean(),parameters,allow_unused=True)
        torch.testing.assert_close(direct,checkpointed,rtol=0,atol=0)
        for before,after in zip(grads,checkpoint_grads):
            self.assertEqual(before is None,after is None)
            if before is not None:
                self.assertTrue(torch.isfinite(after).all());torch.testing.assert_close(before,after)
        self.assertTrue(torch.count_nonzero(checkpoint_grads[0]));self.assertTrue(torch.count_nonzero(checkpoint_grads[1]))
        restore_topology(self.model,None);self.model.disable_gradient_checkpointing()
        torch.testing.assert_close(self.forward(),dense,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_combined_directory_artifacts_and_header_refusals(self):
        from core.models.ideogram4.residual_unet import (
            restore_topology,native_config,save_checkpoint,preflight_checkpoint,pair_topology_metadata)
        from core.models.ideogram4.ideogram4_loader import load_ideogram4_single_file
        from core.training.adapters.ideogram4_adapter import Ideogram4LoRAAdapter
        from core.model_loader import ModelLoader
        from core.models.common.single_file_format import read_state_dict,save_single_file_state
        from safetensors.torch import save_file
        restore_topology(self.model,self.spec);uncond=deepcopy(self.model)
        expected=self.forward().detach();config=native_config(self.model)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/"base";(base/"transformer").mkdir(parents=True)
            (base/"text_encoder").mkdir();(base/"unconditional_transformer").mkdir()
            for name in ("transformer","unconditional_transformer"):
                (base/name/"config.json").write_text(json.dumps(dict(config,num_layers=3)),encoding="utf-8")
            single=Path(save_checkpoint(self.model,uncond,root/"full.safetensors",
                metadata={"component.base_dir":str(base)}))
            state,metadata=read_state_dict(str(single))
            self.assertEqual(read_checkpoint_metadata(metadata),self.spec)
            self.assertEqual(preflight_checkpoint(single),(self.spec,{"transformer":config,"unconditional_transformer":config}))
            adapter=Ideogram4LoRAAdapter(SimpleNamespace(transformer=self.model,transformer_uncond=uncond,config={}),2,2)
            self.assertEqual(read_checkpoint_metadata(adapter.checkpoint_metadata({},1,0)),self.spec)
            sharded=Path(save_checkpoint(self.model,uncond,root/"sharded.safetensors",
                metadata={"component.base_dir":str(base)},max_shard_bytes=20000))
            shared=Path(save_checkpoint(self.model,self.model,root/"shared.safetensors",
                metadata={"component.base_dir":str(base)}))
            directory=root/"directory";directory.mkdir()
            for name,model in (("transformer",self.model),("unconditional_transformer",uncond)):
                target=directory/name;target.mkdir()
                (target/"config.json").write_text(json.dumps(dict(config,_class_name="Ideogram4Transformer2DModel")),encoding="utf-8")
                save_single_file_state(model.state_dict(),metadata,str(target/"diffusion_pytorch_model.safetensors"),max_shard_bytes=20000)
            tokenizer=SimpleNamespace(encode=lambda *a,**k:[1])
            with ExitStack() as stack:
                for target,value in (
                    ("core.models.ideogram4.ideogram4_loader.load_ideogram4_text_encoder",torch.nn.Linear(1,1)),
                    ("transformers.AutoTokenizer.from_pretrained",tokenizer),
                    ("diffusers.AutoencoderKLFlux2.from_pretrained",torch.nn.Linear(1,1)),
                    ("diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained",object())):
                    stack.enter_context(patch(target,return_value=value))
                for path in (single,sharded,root/"sharded.safetensors",directory,shared):
                    components=ModelLoader.load_ideogram4_from_path(str(path),torch_dtype=torch.float32)
                    for name,source in (("transformer",self.model),("unconditional_transformer",uncond)):
                        loaded=components[name];self.assertEqual(loaded._residual_unet_spec,self.spec)
                        self.assertEqual(native_config(loaded),config)
                        for key,value in source.state_dict().items():torch.testing.assert_close(loaded.state_dict()[key],value,rtol=0,atol=0)
                        with patch.object(self,"model",loaded):torch.testing.assert_close(self.forward(),expected,rtol=0,atol=0)
                only=load_ideogram4_single_file(str(single),torch_dtype=torch.float32,load_unconditional=False)
                self.assertIsNone(only["unconditional_transformer"])
                broken=root/"broken.safetensors"
                for md in (dict(metadata,transformer_config="{}"),dict(metadata,unconditional_transformer_config="[]")):
                    save_file(state,str(broken),metadata=md)
                    with patch("core.models.common.single_file_format.read_state_dict",side_effect=AssertionError("weight allocation")):
                        with self.assertRaises(ValueError):ModelLoader.load_ideogram4_from_path(str(broken),torch_dtype=torch.float32)
                partial=root/"partial.safetensors"
                save_file({k:v for k,v in state.items() if k!="transformer.input_proj.weight"},str(partial),metadata=metadata)
                with patch("core.models.ideogram4.ideogram4_loader.load_ideogram4_text_encoder",side_effect=AssertionError("companion loading")):
                    with self.assertRaises(RuntimeError):ModelLoader.load_ideogram4_from_path(str(partial),torch_dtype=torch.float32)
                # A dense unconditional declaration cannot silently accompany a residual conditional graph.
                uncond_dir=directory/"unconditional_transformer"
                index=next(uncond_dir.glob("*.index.json"));data=json.loads(index.read_text(encoding="utf-8"))
                data["metadata"]={};index.write_text(json.dumps(data),encoding="utf-8")
                with patch("core.models.ideogram4.ideogram4_loader._load_component_state_dict",side_effect=AssertionError("weight allocation")):
                    with self.assertRaises(ValueError):ModelLoader.load_ideogram4_from_path(str(directory),torch_dtype=torch.float32)
            restore_topology(uncond,None)
            with self.assertRaisesRegex(ValueError,"topologies disagree"):pair_topology_metadata(self.model,uncond)
            with self.assertRaisesRegex(ValueError,"both transformers"):save_checkpoint(self.model,None,root/"invalid.safetensors")
            dense=root/"dense.safetensors";save_file({"placeholder":torch.ones(1)},str(dense))
            self.assertEqual(preflight_checkpoint(dense),(None,{}))
        self.assertFalse(torch.cuda.is_initialized())

    def test_training_pair_resolution_atomic_lora_resume_and_dual_loss(self):
        from core.models.ideogram4.residual_unet import configure_training_topology,restore_topology,native_config
        from core.training.adapters.ideogram4_adapter import Ideogram4LoRAAdapter
        from core.training.lora_trainer import LoRATrainer
        from core.training.ops import ideogram4_ops
        from core.rewiring.spec import checkpoint_metadata
        from safetensors.torch import save_file
        uncond=deepcopy(self.model);config={}
        restore_topology(self.model,self.spec)
        with self.assertRaisesRegex(ValueError,"topologies disagree"):
            configure_training_topology(self.model,config,unconditional_transformer=uncond)
        self.assertEqual(config,{})
        restore_topology(uncond,self.spec)
        self.assertEqual(configure_training_topology(self.model,config,unconditional_transformer=uncond),self.spec)
        for key in ("repa_enable","tread_enable","blockskip_enable","dit_partition_training_enabled"):
            with self.assertRaisesRegex(ValueError,key):
                configure_training_topology(self.model,dict(config,**{key:True}),unconditional_transformer=uncond)
        restore_topology(self.model,None);restore_topology(uncond,None)
        config={}
        self.assertEqual(configure_training_topology(self.model,config,unconditional_transformer=uncond,
            resume_spec=self.spec,install=False),self.spec)
        self.assertEqual(config,{})
        incompatible=type(uncond)(**dict(native_config(uncond),num_layers=8))
        with self.assertRaises(ValueError):
            configure_training_topology(self.model,config,unconditional_transformer=incompatible,resume_spec=self.spec)
        self.assertEqual(config,{});self.assertFalse(hasattr(self.model,"_residual_unet_spec"))
        trainer=SimpleNamespace(transformer=self.model,transformer_uncond=uncond,config={},is_ideogram4=True,
            ideogram4_train_uncond=True,log_prefix="[IdeogramResumeTest]",lora_rank=2,lora_alpha=2,
            adapter_algorithm="lora",weight_decompose=False,adapter_config={})
        self.model.requires_grad_(False);uncond.requires_grad_(False)
        adapter=Ideogram4LoRAAdapter(trainer,2,2,scope={"attn":True,"mlp":True,"mod":False})
        layers={};adapter.apply_lora_to_unet(layers);trainer.lora_layers=layers
        before={f"{name}.{key}":value.detach().clone() for name,layer in layers.items()
                for key,value in layer.branch_tensors().items()}
        tensors={key:value+0.01 for key,value in before.items()}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"resume.safetensors"
            metadata={"model_type":"ideogram4","step":"17","train_uncond":"True",**checkpoint_metadata(self.spec)}
            save_file({"unrelated":torch.ones(1)},str(path),metadata=metadata)
            with self.assertRaisesRegex(ValueError,"missing required tensor"):LoRATrainer.load_checkpoint(trainer,str(path))
            self.assertEqual(trainer.config,{})
            for model in (self.model,uncond):self.assertFalse(hasattr(model,"_residual_unet_spec"))
            for name,layer in layers.items():
                for key,value in layer.branch_tensors().items():torch.testing.assert_close(value,before[f"{name}.{key}"],rtol=0,atol=0)
            save_file(tensors,str(path),metadata=metadata)
            self.assertEqual(LoRATrainer.load_checkpoint(trainer,str(path)),17)
            for model in (self.model,uncond):self.assertEqual(model._residual_unet_spec,self.spec)
            for name,layer in layers.items():
                for key,value in layer.branch_tensors().items():torch.testing.assert_close(value,tensors[f"{name}.{key}"],rtol=0,atol=0)
            trainer.transformer_uncond=None
            with self.assertRaisesRegex(ValueError,"requires a loaded unconditional"):
                LoRATrainer.load_checkpoint(trainer,str(path))
            trainer.transformer_uncond=uncond
            trainer.ideogram4_train_uncond=False
            with self.assertRaisesRegex(ValueError,"train_uncond"):LoRATrainer.load_checkpoint(trainer,str(path))
            trainer.ideogram4_train_uncond=True
            save_file(tensors,str(path),metadata={"model_type":"ideogram4"})
            with self.assertRaisesRegex(ValueError,"disagree"):LoRATrainer.load_checkpoint(trainer,str(path))
        self.model.enable_gradient_checkpointing();uncond.enable_gradient_checkpointing()
        trainer.device=torch.device("cpu");trainer.training_dtype=torch.float32;trainer.mixed_precision=False
        trainer.reconstruction_loss_weight=0.1;trainer.ideogram4_uncond_loss_weight=0.5
        features=torch.randn(2,3,5,4);mask=torch.tensor([[1,1,1,0,0],[1]*5])
        loss,pred,recon=ideogram4_ops.train_step(trainer,self.images[:,5:].detach(),features,mask,
            timesteps=self.times,latent_h=8,latent_w=12)
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(pred));self.assertTrue(torch.isfinite(recon))
        loss.backward()
        for prefix in ("lora_unet_","lora_uncond_"):
            grads=[value.grad for name,layer in layers.items() if name.startswith(prefix)
                   for value in layer.parameters() if value.requires_grad]
            self.assertTrue(grads);self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertTrue(any(torch.count_nonzero(g) for g in grads))
        self.assertFalse(torch.cuda.is_initialized())

    def test_training_loaders_resolve_before_checkpointing(self):
        from core.models.ideogram4.residual_unet import restore_topology
        from core.training.ops import ideogram4_ops
        restore_topology(self.model,self.spec);uncond=deepcopy(self.model)
        components={"transformer":self.model,"unconditional_transformer":uncond,"vae":torch.nn.Linear(1,1),
                    "text_encoder":torch.nn.Linear(1,1),"tokenizer":object(),"scheduler":object()}
        checkpoints=[]
        def enable(model):
            def apply():
                self.assertEqual(trainer.config["dit_rewiring_spec"],self.spec)
                checkpoints.append(model);model.enable_gradient_checkpointing_orig()
            return apply
        for model in (self.model,uncond):model.enable_gradient_checkpointing_orig=model.enable_gradient_checkpointing
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/"model.safetensors";file.touch()
            for source in (str(Path(folder)/"directory"),str(file)):
                trainer=SimpleNamespace(model_path=source,config={"ideogram4_train_uncond":True},log_prefix="[IdeogramLoadTest]",
                    weight_dtype=torch.float32,vae_dtype=torch.float32,device=torch.device("cpu"),
                    gradient_checkpointing=True,blocks_to_swap=0,use_flash_attention=False)
                with patch("core.models.ideogram4.ideogram4_loader.load_ideogram4_components",return_value=components) as directory, \
                     patch("core.models.ideogram4.ideogram4_loader.load_ideogram4_single_file",return_value=components) as single, \
                     patch.object(self.model,"enable_gradient_checkpointing",side_effect=enable(self.model)), \
                     patch.object(uncond,"enable_gradient_checkpointing",side_effect=enable(uncond)):
                    ideogram4_ops.load_components(trainer)
                    self.assertEqual(single.call_count,int(source==str(file)))
                    self.assertEqual(directory.call_count,int(source!=str(file)))
                self.assertEqual(trainer.config["dit_rewiring_spec"],self.spec)
                self.assertTrue(all(not value.requires_grad for model in (self.model,uncond) for value in model.parameters()))
        self.assertEqual(len(checkpoints),4)
        self.assertFalse(torch.cuda.is_initialized())

    def test_metadata_identity_native_grid_and_feature_refusals(self):
        from core.models.ideogram4.residual_unet import restore_topology,model_topology_metadata,native_config
        restore_topology(self.model,self.spec)
        self.assertEqual(read_checkpoint_metadata(model_topology_metadata(self.model)),self.spec)
        for key,value in (("num_layers",True),("norm_eps",float("nan")),("mrope_section",[2,3,3])):
            with self.assertRaises(ValueError):native_config(dict(self.model.config,**{key:value}))
        for key,value in (("_fbcache",object()),("_repa_tap_depth",0),("_nag_active",True)):
            setattr(self.model,key,value)
            try:
                with self.assertRaisesRegex(ValueError,key):self.forward()
            finally:self.model.__dict__.pop(key,None)
        positions=self.positions.clone();positions[0,8,2]+=1
        with self.assertRaisesRegex(ValueError,"row-major"):self.forward(position_ids=positions)
        indicator=self.indicator.clone();indicator[0,7]=3
        with self.assertRaisesRegex(ValueError,"contiguous image tail"):self.forward(indicator=indicator)
        segments=self.segments.clone();segments[0,7]=8
        with self.assertRaisesRegex(ValueError,"one image segment"):self.forward(segment_ids=segments)
        with self.assertRaisesRegex(ValueError,"per-sample timestep"):
            self.forward(timestep=self.times[:,None].expand(-1,101))
        with self.assertRaises(ValueError):restore_topology(self.model,dict(self.spec,base_config_sha256="0"*64))
        restore_topology(self.model,None);self.assertEqual(model_topology_metadata(self.model),{})
        self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__":unittest.main()
