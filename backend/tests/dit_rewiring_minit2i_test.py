"""CPU contracts for native MiniT2I residual image/text execution."""
from pathlib import Path
import sys
import unittest
import tempfile
import json
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from core.rewiring.spec import build_spec, OPERATORS, read_checkpoint_metadata


class MiniT2IRuntimeContracts(unittest.TestCase):
    def setUp(self):
        for target,name,value in ((torch.cuda,"get_device_capability",lambda *a,**k:(8,9)),
                (torch.cuda,"_lazy_init",lambda *a,**k:None),(torch._C,"_cuda_init",lambda *a,**k:None)):
            guard=patch.object(target,name,value);guard.start();self.addCleanup(guard.stop)
        from core.models.minit2i.vendor.transformer import MiniT2IMMJiTModel
        from core.models.minit2i.residual_unet import native_config
        torch.manual_seed(75)
        self.model=MiniT2IMMJiTModel(image_size=16,patch_size=2,in_channels=4,txt_input_size=12,
            hidden_size=16,txt_hidden_size=16,cond_vec_size=16,depth_double=7,txt_preamble_depth=2,
            num_heads=2,head_dim=12,mlp_ratio=2,pca_channels=8,prompt_length=5,vae_type="sdxl",noise_scale=1)
        self.spec=build_spec("minit2i",[{"name":"main","block_count":7}],[
            dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,2],**OPERATORS),
            dict(id="inner",stream="main",parent="outer",start_block=2,end_block=5,down_factor=[2,2],**OPERATORS)],native_config(self.model))
        self.images=torch.randn(2,4,16,24,requires_grad=True)
        self.text=torch.randn(2,5,12,requires_grad=True)
        self.mask=torch.tensor([[1,1,1,0,0],[1]*5])
        self.times=torch.tensor([0.3,0.7])

    def forward(self,**options):
        arguments=dict(img=self.images,t=self.times,context=self.text,attn_mask=self.mask)
        arguments.update(options);return self.model(**arguments)

    def test_native_preamble_positions_evolving_text_and_offloader(self):
        from core.models.minit2i.residual_unet import restore_topology
        from core.models.minit2i.vendor.mmjit import get_2d_sincos_pos_embed,AttentionMode
        restore_topology(self.model,self.spec);net=self.model.model.net
        parameters={name:id(value) for name,value in self.model.named_parameters()}
        image,h,w=net.img_embedder(self.images)
        expected=image+get_2d_sincos_pos_embed(16,h,w,image.device,image.dtype)[None]
        lengths=[];grids=[];events=[];previous=[];preamble=[]
        def observe(block,args):
            image,text,vec,gh,gw=args
            if not lengths:torch.testing.assert_close(image,expected,rtol=0,atol=0)
            lengths.append(image.shape[1]);grids.append((gh,gw))
            self.assertEqual(text.shape,(2,5,16))
            if previous:torch.testing.assert_close(text,previous[-1],rtol=0,atol=0)
        handles=[block.register_forward_pre_hook(observe) for block in net.double_blocks]
        handles += [block.register_forward_hook(lambda block,args,output:previous.append(output[1])) for block in net.double_blocks]
        handles += [block.register_forward_hook(lambda *args:preamble.append(True)) for block in net.txt_preamble_blocks]
        class Offloader:
            def wait_for_block(owner,index):events.append(("wait",index))
            def submit_move_blocks_forward(owner,index):events.append(("submit",index))
        net._block_offloader=Offloader()
        try:output=self.forward()
        finally:
            for handle in handles:handle.remove()
            del net._block_offloader
        self.assertEqual(lengths,[96,24,6,6,6,24,96])
        self.assertEqual(grids,[(8,12),(4,6),(2,3),(2,3),(2,3),(4,6),(8,12)])
        self.assertEqual(len(preamble),2)
        self.assertEqual(events,[event for index in range(7) for event in (("wait",index),("submit",index))])
        self.assertEqual(parameters,{name:id(value) for name,value in self.model.named_parameters()})
        self.assertEqual(output.shape,self.images.shape);self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(block._attn_mode==AttentionMode.TRAINING for block in net.double_blocks))
        velocity=self.model.pred_velocity(self.images,self.times,self.text,self.mask)
        torch.testing.assert_close(velocity,(output-self.images)/torch.clamp(1-self.times[:,None,None,None],min=0.05))
        torch.testing.assert_close(self.forward(img=self.images.permute(0,2,3,1)),output)
        self.assertFalse(torch.cuda.is_initialized())

    def test_training_inheritance_lora_resume_and_production_train_step(self):
        from core.models.minit2i.residual_unet import configure_training_topology,restore_topology
        from core.training.adapters.minit2i_adapter import MiniT2ILoRAAdapter
        from core.training.lora_trainer import LoRATrainer
        from core.training.ops import minit2i_ops
        from core.rewiring.spec import checkpoint_metadata
        from safetensors.torch import save_file
        restore_topology(self.model,self.spec)
        config={};self.assertEqual(configure_training_topology(self.model,config,resuming=True),self.spec)
        for key in ("repa_enable","tread_enable","blockskip_enable","dit_partition_training_enabled"):
            with self.assertRaisesRegex(ValueError,key):
                configure_training_topology(self.model,dict(config,**{key:True}))
        with self.assertRaisesRegex(ValueError,"block_skip_rate"):
            configure_training_topology(self.model,dict(config,block_skip_rate=0.1))
        restore_topology(self.model,None)
        untouched={}
        self.assertEqual(configure_training_topology(self.model,untouched,resume_spec=self.spec,install=False),self.spec)
        self.assertEqual(untouched,{})
        self.assertFalse(hasattr(self.model,"_residual_unet_spec"))
        trainer=SimpleNamespace(transformer=self.model,config={},is_minit2i=True,
            log_prefix="[MiniT2IResumeTest]",lora_rank=2,lora_alpha=2,
            adapter_algorithm="lora",weight_decompose=False,adapter_config={})
        self.model.requires_grad_(False)
        adapter=MiniT2ILoRAAdapter(trainer,2,2,scope={"attn":True,"mlp":True,"txt_embed":False})
        layers={};adapter.apply_lora_to_unet(layers);trainer.lora_layers=layers
        self.assertGreater(len(layers),0)
        before={f"{name}.{key}":value.detach().clone()
                for name,layer in layers.items() for key,value in layer.branch_tensors().items()}
        tensors={key:value+0.01 for key,value in before.items()}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"resume.safetensors"
            metadata={"model_type":"minit2i","step":"17",**checkpoint_metadata(self.spec)}
            save_file({"unrelated":torch.ones(1)},str(path),metadata=metadata)
            with self.assertRaisesRegex(ValueError,"missing required tensor"):
                LoRATrainer.load_checkpoint(trainer,str(path))
            self.assertFalse(hasattr(self.model,"_residual_unet_spec"));self.assertEqual(trainer.config,{})
            for name,layer in layers.items():
                for key,value in layer.branch_tensors().items():
                    torch.testing.assert_close(value,before[f"{name}.{key}"],rtol=0,atol=0)
            save_file(tensors,str(path),metadata=metadata)
            self.assertEqual(LoRATrainer.load_checkpoint(trainer,str(path)),17)
            self.assertEqual(self.model._residual_unet_spec,self.spec)
            self.assertEqual(self.model.model.net._residual_unet_spec,self.spec)
            for name,layer in layers.items():
                for key,value in layer.branch_tensors().items():
                    torch.testing.assert_close(value,tensors[f"{name}.{key}"],rtol=0,atol=0)
            save_file(tensors,str(path),metadata={"model_type":"minit2i"})
            with self.assertRaisesRegex(ValueError,"disagree"):
                LoRATrainer.load_checkpoint(trainer,str(path))
            self.assertEqual(self.model._residual_unet_spec,self.spec)
        self.model.enable_gradient_checkpointing()
        trainer.device=torch.device("cpu");trainer.training_dtype=torch.float32;trainer.mixed_precision=False
        trainer.minit2i_noise_scale=1;trainer.reconstruction_loss_weight=0
        trainer.stash_cfg_null_per_sample_loss=lambda *a:None
        loss,prediction_loss,reconstruction_loss=minit2i_ops.train_step(trainer,self.images.detach(),
            self.text.detach(),self.mask,timesteps=self.times)
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(prediction_loss))
        self.assertTrue(torch.isfinite(reconstruction_loss));loss.backward()
        grads=[value.grad for layer in layers.values() for value in layer.parameters() if value.requires_grad]
        self.assertTrue(grads);self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(torch.count_nonzero(g) for g in grads))
        self.assertFalse(torch.cuda.is_initialized())

    def test_training_loaders_inherit_before_checkpointing_and_freeze(self):
        from core.models.minit2i.residual_unet import configure_training_topology,restore_topology
        from core.training.ops import minit2i_ops
        from core.training.base_trainer import BaseTrainer
        from core.model_loader import ModelLoader
        restore_topology(self.model,self.spec)
        components=dict(transformer=self.model,text_encoder=torch.nn.Linear(1,1),tokenizer=object(),
                        scheduler=object(),variant="b16",vae_type="sdxl",vae=torch.nn.Linear(1,1))
        for resume in (False,True):
            trainer=SimpleNamespace(config={},model_path="full.safetensors",weight_dtype=torch.float32,
                vae_dtype=torch.float32,log_prefix="[MiniT2ILoadTest]",resume_from_checkpoint="full.safetensors" if resume else None)
            def resolve_then_stop(model,config,**options):
                self.assertEqual(options["resuming"],resume)
                self.assertEqual(configure_training_topology(model,config,**options),self.spec)
                raise RuntimeError("stop before checkpointing")
            with patch.object(ModelLoader,"detect_model_type",return_value="minit2i"), \
                 patch("core.models.minit2i.minit2i_loader.load_minit2i_components",return_value=components), \
                 patch("core.training.vae_swap.apply_latent_space"), \
                 patch("core.models.minit2i.residual_unet.configure_training_topology",side_effect=resolve_then_stop):
                with self.assertRaisesRegex(RuntimeError,"stop before checkpointing"):
                    if resume:BaseTrainer._load_checkpoint_as_base(trainer,"full.safetensors")
                    else:minit2i_ops.load_components(trainer)
            self.assertEqual(trainer.config["dit_rewiring_spec"],self.spec)
        trainer=SimpleNamespace(config={},model_path="full.safetensors",weight_dtype=torch.float32,
            vae_dtype=torch.float32,log_prefix="[MiniT2ILoadTest]",gradient_checkpointing=True,
            blocks_to_swap=0,device=torch.device("cpu"),use_flash_attention=False)
        checkpoint_calls=[];native_checkpoint=self.model.enable_gradient_checkpointing
        def observe_checkpoint(**options):
            self.assertEqual(trainer.config["dit_rewiring_spec"],self.spec)
            checkpoint_calls.append(options);native_checkpoint(**options)
        with patch("core.models.minit2i.minit2i_loader.load_minit2i_components",return_value=components), \
             patch("core.training.vae_swap.apply_latent_space"), \
             patch.object(self.model,"enable_gradient_checkpointing",side_effect=observe_checkpoint):
            minit2i_ops.load_components(trainer)
        self.assertEqual(len(checkpoint_calls),1)
        self.assertTrue(self.model.gradient_checkpointing);self.assertTrue(self.model.model.net.gradient_checkpointing)
        self.assertTrue(all(not value.requires_grad for value in self.model.parameters()))
        self.assertTrue(self.model.training);self.assertEqual(self.model._residual_unet_spec,self.spec)
        self.assertFalse(torch.cuda.is_initialized())

    def test_single_shard_and_directory_native_artifact_roundtrip(self):
        from core.models.minit2i.residual_unet import restore_topology,native_config,preflight_checkpoint
        from core.models.minit2i.vendor.single_file import save_single_file,load_single_file
        from core.models.common.single_file_format import read_state_dict,save_single_file_state
        from core.training.adapters.minit2i_adapter import MiniT2IFullParameterAdapter,MiniT2ILoRAAdapter
        from core.model_loader import ModelLoader
        from safetensors.torch import save_file
        restore_topology(self.model,self.spec);config=native_config(self.model);expected=self.forward().detach()
        trainer=SimpleNamespace(transformer=self.model,text_encoder=None,vae=None,config={},
                                minit2i_variant="b16",train_text_encoder=False,bundle_vae=False)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            single=MiniT2IFullParameterAdapter(trainer).write_checkpoint(1,0,root/"full.safetensors")
            index=save_single_file(str(root/"sharded.safetensors"),self.model,"b16",max_shard_bytes=20000)
            self.assertTrue(str(index).endswith(".index.json"))
            self.assertEqual(preflight_checkpoint(single),(self.spec,config))
            self.assertEqual(read_checkpoint_metadata(MiniT2ILoRAAdapter(trainer,2,2).checkpoint_metadata({},1,0)),self.spec)
            state,metadata=read_state_dict(str(single))
            directory=root/"directory";tdir=directory/"transformer";tdir.mkdir(parents=True)
            (tdir/"config.json").write_text(json.dumps(dict(config,_class_name="MiniT2IMMJiTModel")))
            save_file(self.model.state_dict(),str(tdir/"diffusion_pytorch_model.safetensors"),metadata={
                key:value for key,value in metadata.items() if key.startswith("sushi.rewire.")})
            directory_shards=root/"directory-shards";sdir=directory_shards/"transformer";sdir.mkdir(parents=True)
            (sdir/"config.json").write_text((tdir/"config.json").read_text())
            save_single_file_state(self.model.state_dict(),{k:v for k,v in metadata.items() if k.startswith("sushi.rewire.")},
                str(sdir/"diffusion_pytorch_model.safetensors"),max_shard_bytes=20000)
            with patch("core.models.minit2i.minit2i_loader._resolve_flan_t5",return_value="stub"), \
                 patch("core.models.minit2i.minit2i_loader._load_flan_t5",return_value=(object(),torch.nn.Linear(1,1))), \
                 patch("core.models.minit2i.minit2i_loader.load_minit2i_vae",return_value=torch.nn.Linear(1,1)), \
                 patch.object(type(self.model),"from_pretrained",side_effect=AssertionError("native pretrained allocation")):
                for path in (single,Path(index),root/"sharded.safetensors",directory,directory_shards):
                    restored=ModelLoader.load_minit2i_from_path(str(path),torch_dtype=torch.float32)["transformer"]
                    self.assertEqual(restored._residual_unet_spec,self.spec)
                    self.assertEqual(restored.model.net._residual_unet_spec,self.spec)
                    self.assertEqual(native_config(restored),config)
                    for key,value in self.model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
                    with patch.object(self,"model",restored):torch.testing.assert_close(self.forward(),expected)
            for name,broken in (("missing",{k:v for k,v in metadata.items() if k!="mmjit_config"}),
                    ("wrong",dict(metadata,mmjit_config=json.dumps(dict(config,depth_double=8))))):
                path=root/f"{name}.safetensors";save_file(state,str(path),metadata=broken)
                with patch("core.models.minit2i.vendor.single_file.read_state_dict",side_effect=AssertionError("tensor loading")), \
                     patch.object(type(self.model),"__init__",autospec=True,side_effect=AssertionError("model allocation")):
                    with self.assertRaises(ValueError):load_single_file(str(path))
            partial=root/"partial.safetensors"
            save_file({k:v for k,v in state.items() if k!="transformer.model.net.img_embedder.proj1.weight"},str(partial),metadata=metadata)
            with patch("core.models.minit2i.minit2i_loader._load_flan_t5",side_effect=AssertionError("companion loading")):
                with self.assertRaisesRegex(RuntimeError,"missing transformer keys"):
                    ModelLoader.load_minit2i_from_path(str(partial),torch_dtype=torch.float32)
            (tdir/"config.json").write_text(json.dumps(dict(config,depth_double=8)))
            with patch("core.models.common.single_file_format.read_state_dict",side_effect=AssertionError("tensor loading")), \
                 patch.object(type(self.model),"__init__",autospec=True,side_effect=AssertionError("model allocation")):
                with self.assertRaises(ValueError):ModelLoader.load_minit2i_from_path(str(directory))
            (tdir/"config.json").write_text(json.dumps(config))
            save_file({k:v for k,v in self.model.state_dict().items() if k!="model.net.img_embedder.proj1.weight"},
                str(tdir/"diffusion_pytorch_model.safetensors"),metadata={k:v for k,v in metadata.items() if k.startswith("sushi.rewire.")})
            with patch("core.models.minit2i.minit2i_loader._load_flan_t5",side_effect=AssertionError("companion loading")):
                with self.assertRaisesRegex(ValueError,"weights do not match"):
                    ModelLoader.load_minit2i_from_path(str(directory))
            dense=root/"dense.safetensors";save_file({"placeholder":torch.ones(1)},str(dense))
            self.assertEqual(preflight_checkpoint(dense),(None,None))
        self.assertFalse(torch.cuda.is_initialized())

    def test_checkpoint_outputs_and_gradients_agree(self):
        from core.models.minit2i.residual_unet import restore_topology
        restore_topology(self.model,self.spec)
        parameters=[self.images,self.text,*self.model.parameters()]
        outputs=[];gradients=[]
        for enabled in (False,True):
            self.model.enable_gradient_checkpointing() if enabled else self.model.disable_gradient_checkpointing()
            self.model.zero_grad(set_to_none=True);self.images.grad=None;self.text.grad=None
            output=self.forward();output.square().mean().backward()
            outputs.append(output.detach());gradients.append([None if p.grad is None else p.grad.detach().clone() for p in parameters])
        torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
        for first,second in zip(*gradients):
            if first is None:self.assertIsNone(second)
            else:
                self.assertTrue(torch.isfinite(first).all());torch.testing.assert_close(first,second,rtol=2e-5,atol=2e-7)
        self.assertGreater(torch.count_nonzero(gradients[0][0]),0)
        self.assertGreater(torch.count_nonzero(gradients[0][1]),0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_dense_restore_pixel_variant_and_refusals(self):
        from core.models.minit2i.residual_unet import restore_topology,native_config,model_topology_metadata
        dense=self.forward().detach();restore_topology(self.model,self.spec)
        self.assertEqual(read_checkpoint_metadata(model_topology_metadata(self.model)),self.spec)
        net=self.model.model.net
        for name in ("_style_ctx","_fbcache","_repa_tap_depth","_tread_config","_blockskip_config","_block_skip_config"):
            setattr(net,name,object())
            try:
                with self.assertRaises(ValueError):self.forward()
            finally:setattr(net,name,None)
        for name,value in (("_nag_active",True),("_negpip_active",True),("_style_ctx",object())):
            setattr(net.double_blocks[0],name,value)
            try:
                with self.assertRaises(ValueError):self.forward()
            finally:delattr(net.double_blocks[0],name)
        for options in (dict(img=self.images[:,:,:15]),dict(img=self.images[:,:,:14]),
                dict(context=self.text[:,:0],attn_mask=self.mask[:,:0]),dict(t=self.times[:1]),dict(attn_mask=self.mask[:1])):
            with self.assertRaises(ValueError):self.forward(**options)
        for override in (dict(head_dim=10),dict(hidden_size=18),dict(txt_hidden_size=24),dict(patch_size=True),dict(noise_scale=float("nan"))):
            with self.assertRaises(ValueError):native_config(dict(native_config(self.model),**override))
        restore_topology(self.model,None)
        self.assertFalse(hasattr(net,"_residual_unet_spec"));self.assertEqual(model_topology_metadata(self.model),{})
        torch.testing.assert_close(self.forward(),dense,rtol=0,atol=0)
        pixel=type(self.model)(**dict(native_config(self.model),patch_size=16,in_channels=3,vae_type="none",noise_scale=2))
        pixel_spec=build_spec("minit2i",self.spec["streams"],self.spec["regions"],native_config(pixel))
        restore_topology(pixel,pixel_spec)
        output=pixel(torch.randn(2,3,128,192),self.times,self.text,self.mask)
        self.assertEqual(output.shape,(2,3,128,192));self.assertTrue(torch.isfinite(output).all())
        padded=type(self.model)(**dict(native_config(self.model),head_dim=52))
        padded_spec=build_spec("minit2i",self.spec["streams"],self.spec["regions"],native_config(padded))
        restore_topology(padded,padded_spec)
        with torch.no_grad():
            output=padded(self.images,self.times,self.text,self.mask)
        self.assertEqual(output.shape,self.images.shape);self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(torch.cuda.is_initialized())
