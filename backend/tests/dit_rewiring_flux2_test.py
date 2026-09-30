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

    def test_training_inheritance_and_lora_resume_atomicity(self):
        from core.models.flux2.residual_unet import restore_topology, configure_training_topology
        from core.training.adapters.flux2_adapter import FLUX2LoRAAdapter
        from core.training.lora_trainer import LoRATrainer
        from core.rewiring.spec import checkpoint_metadata
        from safetensors.torch import save_file
        restore_topology(self.model,self.spec)
        config={}
        self.assertEqual(configure_training_topology(self.model,config,resuming=True),self.spec)
        self.assertEqual(config["dit_rewiring_spec"],self.spec)
        before=dict(config)
        with self.assertRaisesRegex(ValueError,"block_skip_rate"):
            configure_training_topology(self.model,dict(config,block_skip_rate=0.1))
        self.assertEqual(config,before)
        restore_topology(self.model,None)
        untouched={}
        self.assertEqual(configure_training_topology(self.model,untouched,resume_spec=self.spec,install=False),self.spec)
        self.assertEqual(untouched,{})
        self.assertFalse(hasattr(self.model,"_residual_unet_spec"))
        trainer=SimpleNamespace(transformer=self.model,config={},is_flux2=True,
            log_prefix="[Flux2ResumeTest]",lora_rank=2,lora_alpha=2,
            adapter_algorithm="lora",weight_decompose=False,adapter_config={})
        adapter=FLUX2LoRAAdapter(trainer,2,2)
        self.model.requires_grad_(False)
        layers={};adapter.apply_lora_to_unet(layers);trainer.lora_layers=layers
        self.assertGreater(len(layers),0)
        tensors={f"{name}.{key}":value.detach().clone()+0.01
                 for name,layer in layers.items() for key,value in layer.branch_tensors().items()}
        before={f"{name}.{key}":value.detach().clone()
                for name,layer in layers.items() for key,value in layer.branch_tensors().items()}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"resume.safetensors"
            metadata={"model_type":"flux2","step":"17",**checkpoint_metadata(self.spec)}
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
            for name,layer in layers.items():
                for key,value in layer.branch_tensors().items():
                    torch.testing.assert_close(value,tensors[f"{name}.{key}"],rtol=0,atol=0)
            save_file(tensors,str(path),metadata={"model_type":"flux2"})
            with self.assertRaisesRegex(ValueError,"disagree"):
                LoRATrainer.load_checkpoint(trainer,str(path))
            self.assertEqual(self.model._residual_unet_spec,self.spec)
        self.model.enable_gradient_checkpointing()
        from core.training.ops import flux2_ops
        trainer.device=torch.device("cpu");trainer.training_dtype=torch.float32;trainer.mixed_precision=False
        trainer.noise_scheduler=None;trainer.snr_regularization_loss=None
        trainer.energy_regularization_loss=None;trainer.reconstruction_loss_weight=0
        loss,prediction_loss,reconstruction_loss=flux2_ops.train_step(trainer,self.images.detach(),self.text.detach(),
            self.ids.unsqueeze(0).expand(2,-1,-1),self.text_ids.unsqueeze(0).expand(2,-1,-1),
            timesteps=self.times,guidance=self.guidance)
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(prediction_loss))
        self.assertTrue(torch.isfinite(reconstruction_loss));loss.backward()
        grads=[value.grad for layer in layers.values() for value in layer.parameters() if value.requires_grad]
        self.assertTrue(grads);self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(torch.count_nonzero(g) for g in grads))
        with self.assertRaisesRegex(ValueError,"reference image grids"):
            flux2_ops.train_step(trainer,self.images,self.text,self.ids,self.text_ids,
                                 reference_latents_nested=[[torch.ones(1)]])
        with self.assertRaisesRegex(ValueError,"reference image grids"):
            flux2_ops.generate_sample(trainer,"test",reference_image_path="reference.png")
        self.assertFalse(torch.cuda.is_initialized())

    def test_public_default_and_custom_multistream_training_configuration(self):
        from collections import Counter
        from core.models.flux2.residual_unet import native_config,default_topology,configure_training_topology
        from core.rewiring.spec import preview_plan
        from api.routes import TrainingRunCreateRequest
        from core.training.training_config import _build_train_section
        model=type(self.model)(**native_config(dict(self.model.config,num_layers=5,num_single_layers=20)))
        request=TrainingRunCreateRequest(training_method="lora",base_model_path="flux2",total_steps=1,
                                        dit_rewiring_mode="configure")
        config=_build_train_section(request.model_dump(),total_steps=1,epochs=None,
                                   train_unet=True,train_text_encoder=False,arch="flux2")
        spec=configure_training_topology(model,config)
        self.assertEqual(spec,default_topology(native_config(model)))
        self.assertEqual(spec["streams"],[{"name":"double","block_count":5},{"name":"single","block_count":20}])
        self.assertEqual([(r["stream"],r["start_block"],r["end_block"]) for r in spec["regions"]],
                         [("single",4,16),("single",8,12)])
        plan=preview_plan(spec,64,64)
        self.assertEqual(Counter(b["tokens"] for b in plan["blocks"]),{4096:13,1024:8,256:4})
        seen=[]
        handles=[block.register_forward_pre_hook(lambda block,args:seen.append(args[0].shape[1]-5))
                 for block in model.single_transformer_blocks]
        try:
            with patch.object(self,"model",model):output=self.forward()
        finally:
            for handle in handles:handle.remove()
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(seen,[96]*4+[24]*4+[6]*4+[24]*4+[96]*4)
        self.assertIsNone(default_topology(native_config(self.model)))
        with self.assertRaisesRegex(ValueError,"No rewiring preset"):
            configure_training_topology(self.model,{"dit_rewiring_mode":"configure"})
        custom={"dit_rewiring_mode":"configure","dit_rewiring_spec":self.spec}
        self.assertEqual(configure_training_topology(self.model,custom),self.spec)
        with self.assertRaisesRegex(ValueError,"without edits"):
            configure_training_topology(self.model,{"dit_rewiring_mode":"configure",
                "dit_rewiring_spec":dict(self.spec,regions=self.spec["regions"][:2])},resuming=True)
        self.assertFalse(torch.cuda.is_initialized())

    def test_public_header_inspection_and_multistream_preview_without_tensor_load(self):
        import asyncio
        import httpx
        from fastapi import FastAPI
        from safetensors.torch import save_file
        from core.models.flux2.residual_unet import native_config
        from core.rewiring.spec import checkpoint_metadata
        from core.rewiring.inspect import inspect_model
        from core.models.common.single_file_format import save_single_file_state
        from api.rewiring_routes import router
        config=native_config(self.model)
        metadata={"model_type":"flux2","transformer_config":json.dumps(config),**checkpoint_metadata(self.spec)}
        tensors={"model.diffusion_model."+key:value.detach().contiguous() for key,value in self.model.state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);single=root/"model.safetensors"
            save_file(tensors,str(single),metadata=metadata)
            sharded=save_single_file_state(tensors,metadata,str(root/"sharded.safetensors"),max_shard_bytes=12000)
            directory=root/"directory";directory.mkdir()
            (directory/"config.json").write_text(json.dumps(dict(config,_class_name="Flux2Transformer2DModel")),encoding="utf-8")
            save_file(self.model.state_dict(),str(directory/"diffusion_pytorch_model.safetensors"),metadata=metadata)
            app=FastAPI();app.include_router(router,prefix="/api/v1")
            with patch("core.models.common.single_file_format.read_state_dict",side_effect=AssertionError("tensor loading")), \
                 patch.object(type(self.model),"__init__",side_effect=AssertionError("model allocation")):
                info=inspect_model(str(single))
                self.assertEqual(info["streams"],self.spec["streams"])
                self.assertEqual(info["topology"],self.spec);self.assertIsNone(info["default_spec"])
                self.assertEqual(info["latent_scale_factor"],16)
                for path in (Path(sharded),root/"sharded.safetensors",directory):
                    self.assertEqual(inspect_model(str(path)),info)
                async def check():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
                        schema=(await client.get("/api/v1/schema/dit-rewiring")).json()
                        self.assertIn("flux2",schema["runtime_architectures"])
                        self.assertEqual(schema["presets"]["flux2"]["streams"],[{"name":"double","block_count":5},{"name":"single","block_count":20}])
                        for path in (single,Path(sharded),directory):
                            response=await client.post("/api/v1/models/rewiring/inspect",json={"model_path":str(path)})
                            self.assertEqual(response.status_code,200,response.text);self.assertEqual(response.json(),info)
                        response=await client.post("/api/v1/models/rewiring/validate",json={"spec":self.spec,"token_height":8,"token_width":12})
                        self.assertEqual(response.status_code,200,response.text)
                        blocks=response.json()["plan"]["blocks"]
                        self.assertEqual([b["stream"] for b in blocks],["double"]*3+["single"]*7)
                        adapter=root/"adapter.safetensors"
                        save_file({"lora_transformer_transformer_blocks_0_attn_to_q.lora_down.weight":torch.ones(2,16)},str(adapter),metadata=metadata)
                        response=await client.post("/api/v1/models/rewiring/inspect",json={"model_path":str(adapter)})
                        self.assertEqual(response.status_code,422)
                        broken=root/"mismatch.safetensors"
                        save_file(tensors,str(broken),metadata={"model_type":"flux2","transformer_config":json.dumps(dict(config,num_layers=4))})
                        response=await client.post("/api/v1/models/rewiring/inspect",json={"model_path":str(broken)})
                        self.assertEqual(response.status_code,422)
                asyncio.run(check())
        self.assertFalse(torch.cuda.is_initialized())

    def test_both_pipeline_load_branches_report_checkpoint_topology(self):
        from core.pipeline import DiffusionPipelineManager
        from core.model_loader import ModelLoader
        from core.models.flux2.residual_unet import restore_topology
        restore_topology(self.model,self.spec)
        vae=torch.nn.Linear(1,1);vae.config=SimpleNamespace(block_out_channels=[1,2,3,4],latent_channels=1)
        for tagged in (True,False):
            manager=DiffusionPipelineManager();manager.device="cpu"
            components={"transformer":self.model,"vae":vae,"text_encoder":torch.nn.Linear(1,1),
                        "tokenizer":object(),"scheduler":object(),"config":{}}
            if tagged:components["model_type"]="flux2"
            with patch.object(ModelLoader,"load_model",return_value=components), \
                 patch.object(manager,"_save_last_model"),patch.object(torch.cuda,"is_available",return_value=False):
                manager._load_model_locked("safetensors","cpu-flux2-test.safetensors")
            self.assertEqual(manager.current_model_info["type"],"flux2")
            self.assertEqual(manager.current_model_info["dit_rewiring_spec"],self.spec)
            self.assertEqual(manager.current_model_info["dit_rewiring_token_stride"],16)
        self.assertFalse(torch.cuda.is_initialized())

    def test_generation_adapter_topology_scope_and_refusals(self):
        from copy import deepcopy
        from core.models.flux2.residual_unet import restore_topology
        from core.pipeline_backends.flux2 import Flux2Mixin
        from core.rewiring.spec import checkpoint_metadata,digest
        from core.adapters import CompositeAdapterLayer,AdapterIncompatible
        from safetensors.torch import save_file
        class Manager(Flux2Mixin):
            def __init__(manager):
                manager.flux2_components={"transformer":self.model,"text_encoder":None,
                    "vae":SimpleNamespace(config=SimpleNamespace(block_out_channels=[1,2,3,4]))}
            def _flux2_resolve_lora_path(manager,path):return str(path)
        manager=Manager();restore_topology(self.model,None)
        original=self.model.transformer_blocks[0].attn.to_q
        with tempfile.TemporaryDirectory() as folder:
            def write(name,topology,wrong=False):
                path=Path(folder)/f"{name}.safetensors"
                save_file({"lora_transformer_transformer_blocks_0_attn_to_q.lora_down.weight":torch.randn(2,15 if wrong else 16),
                           "lora_transformer_transformer_blocks_0_attn_to_q.lora_up.weight":torch.randn(16,2)},str(path),
                    metadata={"model_type":"flux2",**({} if topology is None else checkpoint_metadata(topology))})
                return {"path":str(path),"strength":1.0}
            good,dense,bad=write("good",self.spec),write("dense",None),write("bad",self.spec,True)
            changed=write("changed",dict(self.spec,regions=self.spec["regions"][:2]))
            self.assertEqual(manager._load_lora_flux2([good]),1)
            self.assertIsInstance(self.model.transformer_blocks[0].attn.to_q,CompositeAdapterLayer)
            self.assertEqual(self.model._residual_unet_spec,self.spec)
            params={};manager._flux2_rewiring_metadata(params,self.model,128,192)
            self.assertEqual(params["dit_rewiring_actual"]["spec_sha256"],digest(self.spec))
            self.assertTrue(torch.isfinite(self.forward()).all())
            for options,message in (({"ref_images":["ref.png"]},"reference"),({"style_transfers":[{}]},"style"),
                    ({"fbcache_enable":True,"fbcache_threshold":0.1},"FBCache"),({"nag_enable":True},"NAG"),
                    ({"prompt":"(blurry:-1)"},"NegPip")):
                with self.assertRaisesRegex(Exception,message):manager._flux2_rewiring_metadata(options,self.model,128,192)
            with self.assertRaisesRegex(Exception,"divisible"):
                manager._flux2_rewiring_metadata({},self.model,144,192)
            manager._unload_lora_flux2()
            self.assertIs(self.model.transformer_blocks[0].attn.to_q,original)
            self.assertFalse(hasattr(self.model,"_residual_unet_spec"))
            manager._flux2_rewiring_metadata(params,self.model,128,192)
            self.assertNotIn("dit_rewiring_actual",params)
            for files in ([good,dense],[good,changed],[good,bad]):
                with self.assertRaises(RuntimeError):manager._load_lora_flux2(files)
                self.assertIs(self.model.transformer_blocks[0].attn.to_q,original)
                self.assertFalse(hasattr(self.model,"_residual_unet_spec"))
            restore_topology(self.model,self.spec)
            with self.assertRaises(AdapterIncompatible):manager._load_lora_flux2([dense])
            manager._load_lora_flux2([good]);manager._unload_lora_flux2()
            self.assertEqual(self.model._residual_unet_spec,self.spec)
            restore_topology(self.model,None)
            manager._load_lora_flux2([good])
            replacement=deepcopy(self.model);restore_topology(replacement,None)
            replacement.transformer_blocks[0].attn.to_q=deepcopy(original)
            manager.flux2_components["transformer"]=replacement
            manager._load_lora_flux2([good]);manager._unload_lora_flux2()
            self.assertFalse(hasattr(replacement,"_residual_unet_spec"))
            self.assertFalse(hasattr(self.model,"_residual_unet_spec"))
        self.assertFalse(torch.cuda.is_initialized())

    def test_cleanup_restores_topology_despite_offload_failure(self):
        from core.models.flux2.residual_unet import restore_topology
        from core.pipeline_backends.flux2 import Flux2Mixin
        import weakref
        restore_topology(self.model,self.spec)
        events=[]
        class FailingOffload:
            def to(module,device):events.append(device);raise RuntimeError("offload failed")
        manager=Flux2Mixin()
        manager.flux2_components={"transformer":self.model,"vae":FailingOffload()}
        manager._flux2_topology_restore=(weakref.ref(self.model),None)
        manager._flux2_cleanup(gen_succeeded=False)
        self.assertFalse(hasattr(self.model,"_residual_unet_spec"));self.assertEqual(events,["cpu"])
        with patch.object(manager,"_unload_lora_flux2",side_effect=RuntimeError("restore failed")):
            manager._flux2_topology_restore=(weakref.ref(self.model),None)
            manager._flux2_cleanup(gen_succeeded=False)
        self.assertEqual(events,["cpu","cpu"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_all_generation_entries_select_and_restore_before_staging(self):
        from PIL import Image
        from core.pipeline_backends.flux2 import Flux2Mixin
        from core.models.flux2.residual_unet import restore_topology
        from core.rewiring.spec import checkpoint_metadata
        from safetensors.torch import save_file
        restore_topology(self.model,None)
        class Manager(Flux2Mixin):
            device="cpu"
            def __init__(manager):
                vae=torch.nn.Linear(1,1);vae.config=SimpleNamespace(block_out_channels=[1,2,3,4])
                manager.flux2_components={"transformer":self.model,"vae":vae,"text_encoder":torch.nn.Linear(1,1),
                                          "tokenizer":object(),"scheduler":object(),"config":{}}
                manager.seen=[];manager.cleaned=[]
            def _flux2_resolve_lora_path(manager,path):return str(path)
            def _flux2_rewiring_metadata(manager,params,transformer,height,width):
                super()._flux2_rewiring_metadata(params,transformer,height,width)
                manager.seen.append(params.get("dit_rewiring_actual",{}).get("spec"))
                raise RuntimeError("CPU stop before staging")
            def _flux2_cleanup(manager,**options):
                super()._flux2_cleanup(**options)
                manager.cleaned.append(getattr(self.model,"_residual_unet_spec",None))
        manager=Manager()
        original=self.model.transformer_blocks[0].attn.to_q
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"good.safetensors"
            save_file({"lora_transformer_transformer_blocks_0_attn_to_q.lora_down.weight":torch.randn(2,16),
                       "lora_transformer_transformer_blocks_0_attn_to_q.lora_up.weight":torch.randn(16,2)},str(path),
                metadata={"model_type":"flux2",**checkpoint_metadata(self.spec)})
            image=Image.new("RGB",(192,128));mask=Image.new("L",image.size)
            for entry,args in ((manager._generate_txt2img_flux2,()),(manager._generate_img2img_flux2,(image,)),
                               (manager._generate_inpaint_flux2,(image,mask))):
                params={"loras":[{"path":str(path),"strength":1.0}],"height":128,"width":192,"seed":42}
                with self.assertRaisesRegex(RuntimeError,"CPU stop before staging"):entry(params,*args)
                self.assertIs(self.model.transformer_blocks[0].attn.to_q,original)
                self.assertFalse(hasattr(self.model,"_residual_unet_spec"))
                self.assertEqual(params["dit_rewiring_actual"]["spec"],self.spec)
                manager._load_lora_flux2(params["loras"])
                empty={"height":128,"width":192,"seed":42}
                with self.assertRaisesRegex(RuntimeError,"CPU stop before staging"):entry(empty,*args)
                self.assertNotIn("dit_rewiring_actual",empty)
                self.assertIs(self.model.transformer_blocks[0].attn.to_q,original)
                self.assertFalse(hasattr(self.model,"_residual_unet_spec"))
            self.assertEqual(manager.seen,[self.spec,None]*3);self.assertEqual(manager.cleaned,[None]*6)
        self.assertFalse(torch.cuda.is_initialized())

    def test_both_training_loaders_inherit_before_checkpointing_or_freeze(self):
        from core.models.flux2.residual_unet import restore_topology
        from core.training.ops import flux2_ops
        from core.training.base_trainer import BaseTrainer
        from core.model_loader import ModelLoader
        restore_topology(self.model,self.spec)
        components=dict(transformer=self.model,vae=torch.nn.Linear(1,1),text_encoder=torch.nn.Linear(1,1),
                        tokenizer=object(),scheduler=object(),config={})
        for resume in (False,True):
            trainer=SimpleNamespace(config={},model_path="full.safetensors",weight_dtype=torch.float32,
                vae_dtype=torch.float32,log_prefix="[Flux2LoadTest]",gradient_checkpointing=True)
            sentinel="stop before checkpointing" if resume else "stop before quantization setup"
            target="core.models.flux2.residual_unet.configure_training_topology"
            from core.models.flux2.residual_unet import configure_training_topology
            def resolve_then_stop(model,config,**options):
                resolved=configure_training_topology(model,config,**options)
                self.assertEqual(resolved,self.spec)
                raise RuntimeError(sentinel)
            with patch.object(ModelLoader,"detect_model_type",return_value="flux2"), \
                 patch.object(ModelLoader,"load_flux2_from_safetensors",return_value=components), \
                 patch("core.training.vae_swap.apply_latent_space"),patch(target,side_effect=resolve_then_stop):
                with self.assertRaisesRegex(RuntimeError,sentinel):
                    if resume:BaseTrainer._load_checkpoint_as_base(trainer,"full.safetensors")
                    else:flux2_ops.load_components(trainer)
            self.assertEqual(trainer.config["dit_rewiring_spec"],self.spec)
        self.assertFalse(torch.cuda.is_initialized())

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
