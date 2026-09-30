"""CPU contracts for Lens residual execution and native block checkpoint modes."""
from pathlib import Path
import sys
import unittest
import tempfile
import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from core.rewiring.spec import build_spec, OPERATORS, read_checkpoint_metadata


class LensRuntimeContracts(unittest.TestCase):
    def setUp(self):
        for target, name, value in ((torch.cuda, "get_device_capability", lambda *a, **k: (8, 9)),
                (torch.cuda, "_lazy_init", lambda *a, **k: None), (torch._C, "_cuda_init", lambda *a, **k: None)):
            guard = patch.object(target, name, value); guard.start(); self.addCleanup(guard.stop)
        from core.models.lens.vendor.transformer import LensTransformer2DModel
        from core.models.lens.residual_unet import native_config
        torch.manual_seed(64)
        self.model = LensTransformer2DModel(patch_size=2, in_channels=4, out_channels=1, num_layers=7,
            attention_head_dim=12, num_attention_heads=2, inner_dim=24, enc_hidden_dim=12,
            axes_dims_rope=(4,4,4), selected_layer_index=(5,11))
        self.spec = build_spec("lens", [{"name":"main", "block_count":7}], [
            dict(id="outer", stream="main", parent=None, start_block=1, end_block=6, down_factor=[2,2], **OPERATORS),
            dict(id="inner", stream="main", parent="outer", start_block=2, end_block=5, down_factor=[2,2], **OPERATORS)],
            native_config(self.model))
        self.images = torch.randn(2,96,4,requires_grad=True)
        self.features = [torch.randn(2,5,12,requires_grad=True) for _ in range(2)]
        self.mask = torch.tensor([[True,True,True,False,False], [True]*5])
        self.times = torch.tensor([300.0,700.0])

    def forward(self, **options):
        arguments = dict(hidden_states=self.images, encoder_hidden_states=self.features,
            encoder_hidden_states_mask=self.mask, timestep=self.times, img_shapes=[(1,8,12)])
        arguments.update(options)
        return self.model(**arguments)

    def test_native_lengths_text_rotary_masks_and_offloader(self):
        from core.models.lens.residual_unet import restore_topology
        restore_topology(self.model,self.spec)
        original = {name:id(value) for name,value in self.model.named_parameters()}
        full_rotary = self.model.pos_embed([(1,8,12)],[5],device=torch.device("cpu"))[1]
        text_mask = self.model._build_joint_attention_mask(self.mask,96)[...,-5:]
        lengths = []; events = []; previous_text = []
        def observe(block,args):
            image,text,modulation,rotary,mask = args
            lengths.append(image.shape[1]); self.assertEqual(text.shape[:2],(2,5))
            if previous_text:torch.testing.assert_close(text,previous_text[-1],rtol=0,atol=0)
            self.assertEqual(rotary[0].shape[0],image.shape[1])
            h,w = {96:(8,12),24:(4,6),6:(2,3)}[image.shape[1]]
            expected_rotary = self.model.pos_embed([(1,h,w)],[5],device=torch.device("cpu"))[0]
            torch.testing.assert_close(rotary[0],expected_rotary,rtol=0,atol=0)
            torch.testing.assert_close(rotary[1],full_rotary,rtol=0,atol=0)
            torch.testing.assert_close(mask[...,-5:],text_mask,rtol=0,atol=0)
            self.assertTrue(torch.equal(mask[...,:image.shape[1]],torch.zeros_like(mask[...,:image.shape[1]])))
        handles = [block.register_forward_pre_hook(observe) for block in self.model.transformer_blocks]
        handles.extend(block.register_forward_hook(lambda block,args,output:previous_text.append(output[0]))
                       for block in self.model.transformer_blocks)
        class Offloader:
            def wait_for_block(owner,index):events.append(("wait",index))
            def submit_move_blocks_forward(owner,index):events.append(("submit",index))
        self.model._block_offloader = Offloader()
        try: output = self.forward()
        finally:
            for handle in handles:handle.remove()
        self.assertEqual(lengths,[96,24,6,6,6,24,96])
        self.assertEqual(events,[(name,i) for i in range(7) for name in ("wait","submit")])
        self.assertEqual(output.shape,self.images.shape);self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(original,{name:id(value) for name,value in self.model.named_parameters()})
        self.assertFalse(torch.cuda.is_initialized())

    def test_public_preset_and_custom_training_configuration(self):
        from collections import Counter
        from core.models.lens.residual_unet import native_config, default_topology, configure_training_topology
        from core.rewiring.spec import preview_plan
        from api.routes import TrainingRunCreateRequest
        from core.training.training_config import _build_train_section
        model=type(self.model)(**native_config(dict(self.model.config,num_layers=48)))
        request=TrainingRunCreateRequest(training_method="lora",base_model_path="lens",total_steps=1,
                                        dit_rewiring_mode="configure")
        config=_build_train_section(request.model_dump(),total_steps=1,epochs=None,
                                    train_unet=True,train_text_encoder=False,arch="lens")
        spec=configure_training_topology(model,config)
        self.assertEqual(spec,default_topology(native_config(model)))
        self.assertEqual([(r["start_block"],r["end_block"]) for r in spec["regions"]],[(4,44),(22,26)])
        plan=preview_plan(spec,64,64)
        self.assertEqual(Counter(b["tokens"] for b in plan["blocks"]),{4096:8,1024:36,256:4})
        lengths=[]
        handles=[block.register_forward_pre_hook(lambda block,args:lengths.append(args[0].shape[1]))
                 for block in model.transformer_blocks]
        try:
            with patch.object(self,"model",model):self.assertTrue(torch.isfinite(self.forward()).all())
        finally:
            for handle in handles:handle.remove()
        self.assertEqual(lengths,[96]*4+[24]*18+[6]*4+[24]*18+[96]*4)
        self.assertIsNone(default_topology(native_config(self.model)))
        with self.assertRaisesRegex(ValueError,"No rewiring preset"):
            configure_training_topology(self.model,{"dit_rewiring_mode":"configure"})
        self.assertEqual(configure_training_topology(self.model,
            {"dit_rewiring_mode":"configure","dit_rewiring_spec":self.spec}),self.spec)
        other=build_spec("lens",self.spec["streams"],self.spec["regions"][:1],native_config(self.model))
        with self.assertRaisesRegex(ValueError,"without edits"):
            configure_training_topology(self.model,{"dit_rewiring_mode":"configure","dit_rewiring_spec":other},resuming=True)
        self.assertFalse(torch.cuda.is_initialized())

    def test_public_header_inspection_and_api_without_tensor_or_model_load(self):
        import asyncio
        import httpx
        from fastapi import FastAPI
        from safetensors.torch import save_file
        from core.models.lens.residual_unet import native_config
        from core.rewiring.spec import checkpoint_metadata
        from core.rewiring.inspect import inspect_model
        from core.models.common.single_file_format import save_single_file_state
        from api.rewiring_routes import router
        config=native_config(self.model)
        metadata={"model_type":"lens","transformer_config":json.dumps(config),**checkpoint_metadata(self.spec)}
        tensors={"net."+key:value.detach().contiguous() for key,value in self.model.state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);single=root/"model.safetensors"
            save_file(tensors,str(single),metadata=metadata)
            sharded=save_single_file_state(tensors,metadata,str(root/"sharded.safetensors"),max_shard_bytes=20000)
            directory=root/"directory";directory.mkdir()
            (directory/"config.json").write_text(json.dumps(dict(config,_class_name="LensTransformer2DModel")),encoding="utf-8")
            save_file(self.model.state_dict(),str(directory/"diffusion_pytorch_model.safetensors"),metadata=metadata)
            app=FastAPI();app.include_router(router,prefix="/api/v1")
            with patch("core.models.common.single_file_format.read_state_dict",side_effect=AssertionError("tensor loading")), \
                 patch.object(type(self.model),"__init__",autospec=True,side_effect=AssertionError("model allocation")):
                info=inspect_model(str(single))
                self.assertEqual(info["streams"],self.spec["streams"])
                self.assertEqual(info["topology"],self.spec);self.assertIsNone(info["default_spec"])
                self.assertEqual(info["latent_scale_factor"],16)
                for path in (Path(sharded),root/"sharded.safetensors",directory):
                    self.assertEqual(inspect_model(str(path)),info)
                async def check():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
                        response=await client.get("/api/v1/schema/dit-rewiring")
                        schema=response.json();self.assertIn("lens",schema["runtime_architectures"])
                        self.assertEqual(schema["presets"]["lens"]["block_count"],48)
                        for path in (single,Path(sharded),directory):
                            response=await client.post("/api/v1/models/rewiring/inspect",json={"model_path":str(path)})
                            self.assertEqual(response.status_code,200,response.text);self.assertEqual(response.json(),info)
                        response=await client.post("/api/v1/models/rewiring/validate",json={"spec":self.spec,"token_height":8,"token_width":12})
                        self.assertEqual(response.status_code,200,response.text)
                        self.assertEqual([b["tokens"] for b in response.json()["plan"]["blocks"]],[96,24,6,6,6,24,96])
                        adapter=root/"adapter.safetensors"
                        save_file({"lora_unet_transformer_blocks_0_attn_img_qkv.lora_down.weight":torch.ones(2,24)},str(adapter),metadata=metadata)
                        response=await client.post("/api/v1/models/rewiring/inspect",json={"model_path":str(adapter)})
                        self.assertEqual(response.status_code,422)
                        broken=root/"mismatch.safetensors"
                        save_file(tensors,str(broken),metadata={"model_type":"lens","transformer_config":json.dumps(dict(config,num_layers=8))})
                        response=await client.post("/api/v1/models/rewiring/inspect",json={"model_path":str(broken)})
                        self.assertEqual(response.status_code,422)
                asyncio.run(check())
        self.assertFalse(torch.cuda.is_initialized())

    def test_pipeline_model_info_reports_checkpoint_topology(self):
        from core.pipeline import DiffusionPipelineManager
        from core.model_loader import ModelLoader
        from core.models.lens.residual_unet import restore_topology
        restore_topology(self.model,self.spec)
        vae=torch.nn.Linear(1,1);vae.config=SimpleNamespace(block_out_channels=[1,2,3,4],latent_channels=1)
        manager=DiffusionPipelineManager();manager.device="cpu"
        components={"type":"lens","transformer":self.model,"vae":vae,"text_encoder":torch.nn.Linear(1,1),
                    "tokenizer":object(),"scheduler":object()}
        with patch.object(ModelLoader,"load_model",return_value=components), \
             patch.object(manager,"_save_last_model"),patch.object(torch.cuda,"is_available",return_value=False):
            manager._load_model_locked("safetensors","cpu-lens-test.safetensors")
        self.assertEqual(manager.current_model_info["type"],"lens")
        self.assertEqual(manager.current_model_info["dit_rewiring_spec"],self.spec)
        self.assertEqual(manager.current_model_info["dit_rewiring_token_stride"],16)
        self.assertFalse(torch.cuda.is_initialized())

    def test_generation_lora_topology_scope_and_model_replacement(self):
        from copy import deepcopy
        from core.models.lens.residual_unet import restore_topology, native_config
        from core.pipeline_backends.lens import LensMixin
        from core.training.adapters.lens_adapter import LensLoRAAdapter
        from core.rewiring.spec import checkpoint_metadata
        from core.adapters import CompositeAdapterLayer, AdapterIncompatible
        from safetensors.torch import save_file
        class Manager(LensMixin):
            def __init__(manager,model):manager.lens_components={"transformer":model}
            def _lens_resolve_lora_path(manager,path):return str(path)
        base=deepcopy(self.model);manager=Manager(base)
        original=base.transformer_blocks[0].attn.img_qkv
        adapter=LensLoRAAdapter(SimpleNamespace(transformer=self.model,config={}),2,2)
        self.model.requires_grad_(False);layers={};adapter.apply_lora_to_unet(layers)
        with torch.no_grad():
            for layer in layers.values():
                for value in layer.branch_tensors().values():value.add_(0.01)
        tensors={f"{name}.{key}":value.detach().contiguous()
                 for name,layer in layers.items() for key,value in layer.branch_tensors().items()}
        restore_topology(self.model,self.spec)
        expected=self.forward().detach()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);good=root/"good.safetensors";dense=root/"dense.safetensors"
            changed=root/"changed.safetensors";bad=root/"bad.safetensors"
            md={"model_type":"lens",**checkpoint_metadata(self.spec)}
            save_file(tensors,str(good),metadata=md)
            save_file(tensors,str(dense),metadata={"model_type":"lens"})
            other=build_spec("lens",self.spec["streams"],self.spec["regions"][:1],native_config(base))
            save_file(tensors,str(changed),metadata={"model_type":"lens",**checkpoint_metadata(other)})
            save_file({"lora_unet_ghost.lora_down.weight":torch.ones(2,24),
                       "lora_unet_ghost.lora_up.weight":torch.ones(24,2)},str(bad),metadata=md)
            request=lambda paths:[{"path":str(path),"strength":1.0} for path in paths]
            for paths in ([good,dense],[good,changed],[good,bad]):
                with self.assertRaises(RuntimeError):manager._load_lora_lens(request(paths))
                self.assertIs(base.transformer_blocks[0].attn.img_qkv,original)
                self.assertFalse(hasattr(base,"_residual_unet_spec"))
            self.assertGreater(manager._load_lora_lens(request([good])),0)
            self.assertEqual(base._residual_unet_spec,self.spec)
            with patch.object(self,"model",base):torch.testing.assert_close(self.forward(),expected)
            manager._load_lora_lens([])
            self.assertIs(base.transformer_blocks[0].attn.img_qkv,original)
            self.assertFalse(hasattr(base,"_residual_unet_spec"))
            restore_topology(base,self.spec)
            with self.assertRaises(AdapterIncompatible):manager._load_lora_lens(request([dense]))
            manager._load_lora_lens(request([good]));manager._unload_lora_lens()
            self.assertEqual(base._residual_unet_spec,self.spec)
            restore_topology(base,None)
            manager._load_lora_lens(request([good]))
            replacement=type(base)(**native_config(base))
            manager.lens_components["transformer"]=replacement
            replacement_original=replacement.transformer_blocks[0].attn.img_qkv
            manager._unload_lora_lens()
            self.assertIs(replacement.transformer_blocks[0].attn.img_qkv,replacement_original)
            self.assertFalse(hasattr(replacement,"_residual_unet_spec"))
            self.assertFalse(hasattr(base,"_residual_unet_spec"))
            self.assertFalse(any(isinstance(m,CompositeAdapterLayer) for m in replacement.modules()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_all_generation_entries_preflight_before_staging(self):
        from core.models.lens.residual_unet import restore_topology
        from core.pipeline_backends.lens import LensMixin
        from core.rewiring.spec import digest
        from PIL import Image
        class Manager(LensMixin):
            def __init__(manager):
                manager.device=torch.device("cpu")
                manager.lens_components={"transformer":self.model,"tokenizer":object(),"vae":object(),"scheduler":object()}
        manager=Manager();restore_topology(self.model,self.spec)
        image=Image.new("RGB",(192,128));mask=Image.new("L",(192,128))
        calls=[(manager._generate_txt2img_lens,()),(manager._generate_img2img_lens,(image,)),
               (manager._generate_inpaint_lens,(image,mask))]
        with patch.object(manager,"_lens_kh_setup",side_effect=RuntimeError("stop before staging")) as setup:
            for generate,args in calls:
                params={"width":192,"height":128,"seed":1}
                with self.assertRaisesRegex(RuntimeError,"stop before staging"):generate(params,*args)
                self.assertEqual(params["dit_rewiring_actual"],
                    {"spec":self.spec,"spec_sha256":digest(self.spec),"kv_cache":False})
                for options,message in (({"style_transfer":{"image":True}},"style injection"),
                        ({"ref_images":["reference.png"]},"reference image grids"),
                        ({"fbcache_enable":True,"fbcache_threshold":0.1},"FBCache"),
                        ({"nag_enable":True},"NAG"),({"prompt":"(blurry:-1)"},"NegPip"),
                        ({"height":144},"divisible")):
                    before=setup.call_count
                    with self.assertRaisesRegex(Exception,message):generate(dict(params,**options),*args)
                    self.assertEqual(setup.call_count,before)
            restore_topology(self.model,None)
            params={"dit_rewiring_actual":{"stale":True}}
            self.assertEqual(manager._lens_rewiring_preflight(params,128,192),[])
            self.assertNotIn("dit_rewiring_actual",params)
        self.assertFalse(torch.cuda.is_initialized())

    def test_training_inheritance_lora_resume_and_real_train_step(self):
        from core.models.lens.residual_unet import configure_training_topology, restore_topology
        from core.training.adapters.lens_adapter import LensLoRAAdapter
        from core.training.lora_trainer import LoRATrainer
        from core.training.ops import lens_ops
        from core.rewiring.spec import checkpoint_metadata
        from safetensors.torch import save_file
        restore_topology(self.model,self.spec)
        config={}
        self.assertEqual(configure_training_topology(self.model,config,resuming=True),self.spec)
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
        trainer=SimpleNamespace(transformer=self.model,config={},is_lens=True,
            log_prefix="[LensResumeTest]",lora_rank=2,lora_alpha=2,
            adapter_algorithm="lora",weight_decompose=False,adapter_config={})
        self.model.requires_grad_(False)
        adapter=LensLoRAAdapter(trainer,2,2)
        layers={};adapter.apply_lora_to_unet(layers);trainer.lora_layers=layers
        self.assertGreater(len(layers),0)
        before={f"{name}.{key}":value.detach().clone()
                for name,layer in layers.items() for key,value in layer.branch_tensors().items()}
        tensors={key:value+0.01 for key,value in before.items()}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"resume.safetensors"
            metadata={"model_type":"lens","step":"17",**checkpoint_metadata(self.spec)}
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
            save_file(tensors,str(path),metadata={"model_type":"lens"})
            with self.assertRaisesRegex(ValueError,"disagree"):
                LoRATrainer.load_checkpoint(trainer,str(path))
            self.assertEqual(self.model._residual_unet_spec,self.spec)
        self.model.enable_gradient_checkpointing()
        trainer.device=torch.device("cpu");trainer.training_dtype=torch.float32;trainer.mixed_precision=False
        trainer.noise_scheduler=None;trainer.reconstruction_loss_weight=0
        trainer.stash_cfg_null_per_sample_loss=lambda *a:None
        loss,prediction_loss,reconstruction_loss=lens_ops.train_step(trainer,self.images.detach(),
            torch.stack([feature.detach() for feature in self.features],dim=1),self.mask,
            timesteps=torch.tensor([0.3,0.7]),latent_h=8,latent_w=12)
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(prediction_loss))
        loss.backward()
        grads=[value.grad for layer in layers.values() for value in layer.parameters() if value.requires_grad]
        self.assertTrue(grads);self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(torch.count_nonzero(g) for g in grads))
        self.assertFalse(torch.cuda.is_initialized())

    def test_training_loaders_inherit_before_checkpointing_and_freeze(self):
        from core.models.lens.residual_unet import restore_topology, configure_training_topology
        from core.training.ops import lens_ops
        from core.training.base_trainer import BaseTrainer
        from core.model_loader import ModelLoader
        restore_topology(self.model,self.spec)
        components=dict(transformer=self.model,vae=torch.nn.Linear(1,1),text_encoder=torch.nn.Linear(1,1),
                        tokenizer=object(),scheduler=object(),base_dir="base")
        for resume in (False,True):
            trainer=SimpleNamespace(config={},model_path="full.safetensors",weight_dtype=torch.float32,
                vae_dtype=torch.float32,log_prefix="[LensLoadTest]",resume_from_checkpoint="full.safetensors" if resume else None)
            def resolve_then_stop(model,config,**options):
                self.assertEqual(options["resuming"],resume)
                self.assertEqual(configure_training_topology(model,config,**options),self.spec)
                raise RuntimeError("stop before checkpointing")
            with patch.object(ModelLoader,"detect_model_type",return_value="lens"), \
                 patch.object(ModelLoader,"load_lens_from_path",return_value=components), \
                 patch("core.training.vae_swap.apply_latent_space"), \
                 patch("core.models.lens.residual_unet.configure_training_topology",side_effect=resolve_then_stop):
                with self.assertRaisesRegex(RuntimeError,"stop before checkpointing"):
                    if resume:BaseTrainer._load_checkpoint_as_base(trainer,"full.safetensors")
                    else:lens_ops.load_components(trainer)
            self.assertEqual(trainer.config["dit_rewiring_spec"],self.spec)
        trainer=SimpleNamespace(config={},model_path="full.safetensors",weight_dtype=torch.float32,
            vae_dtype=torch.float32,log_prefix="[LensLoadTest]",gradient_checkpointing=True,
            blocks_to_swap=0,device=torch.device("cpu"),use_flash_attention=False)
        checkpoint_calls=[]
        native_checkpoint=self.model.enable_gradient_checkpointing
        def observe_checkpoint(**options):
            self.assertEqual(trainer.config["dit_rewiring_spec"],self.spec)
            checkpoint_calls.append(options);native_checkpoint(**options)
        with patch.object(ModelLoader,"load_lens_from_path",return_value=components), \
             patch("core.training.vae_swap.apply_latent_space"), \
             patch.object(self.model,"enable_gradient_checkpointing",side_effect=observe_checkpoint):
            lens_ops.load_components(trainer)
        self.assertEqual(len(checkpoint_calls),1)
        self.assertTrue(all(not value.requires_grad for value in self.model.parameters()))
        self.assertEqual(self.model._residual_unet_spec,self.spec)
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_writer_native_loader_single_shard_and_preflight(self):
        from core.models.lens.residual_unet import restore_topology,native_config,preflight_checkpoint
        from core.training.adapters.lens_adapter import LensFullParameterAdapter,LensLoRAAdapter
        from core.models.common.single_file_format import read_state_dict,save_single_file_state
        from core.model_loader import ModelLoader
        from safetensors.torch import save_file
        restore_topology(self.model,self.spec)
        expected=self.forward().detach()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/"base";(base/"transformer").mkdir(parents=True)
            (base/"vae").mkdir();(base/"vae"/"config.json").write_text("{}")
            config=native_config(self.model)
            companion=dict(config,num_layers=3,selected_layer_index=[1])
            (base/"transformer"/"config.json").write_text(json.dumps(dict(companion,_class_name="LensTransformer2DModel")))
            trainer=SimpleNamespace(transformer=self.model,vae=None,text_encoder=None,config={},bundle_vae=False,lens_base_dir=str(base))
            single=LensFullParameterAdapter(trainer).write_checkpoint(1,0,root/"full.safetensors")
            lora_metadata=LensLoRAAdapter(trainer,2,2).checkpoint_metadata({},1,0)
            self.assertEqual(read_checkpoint_metadata(lora_metadata),self.spec)
            self.assertEqual(preflight_checkpoint(single),(self.spec,config))
            state,metadata=read_state_dict(str(single))
            index=save_single_file_state(state,metadata,str(root/"sharded.safetensors"),max_shard_bytes=20000)
            text_encoder=torch.nn.Linear(1,1)
            text_encoder.set_selected_layers=lambda layers:setattr(text_encoder,"selected",layers)
            tokenizer=SimpleNamespace(pad_token_id=0,padding_side="left",encode=lambda *a,**k:[1])
            vae=torch.nn.Linear(1,1)
            with ExitStack() as stack:
                for target,value in (
                    ("core.models.lens.vendor.LensGptOssEncoder.from_pretrained",text_encoder),
                    ("diffusers.AutoencoderKLFlux2.from_pretrained",vae),
                    ("transformers.AutoTokenizer.from_pretrained",tokenizer),
                    ("diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained",object())):
                    stack.enter_context(patch(target,return_value=value))
                stack.enter_context(patch.object(type(self.model),"from_pretrained",side_effect=AssertionError("base DiT allocation")))
                for path in (single,Path(index),root/"sharded.safetensors"):
                    result=ModelLoader.load_lens_from_path(str(path),torch_dtype=torch.float32)
                    restored=result["transformer"]
                    self.assertEqual(restored._residual_unet_spec,self.spec)
                    self.assertEqual(native_config(restored),config)
                    self.assertEqual(text_encoder.selected,(5,11))
                    for key,value in self.model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
                    with patch.object(self,"model",restored):
                        torch.testing.assert_close(self.forward(),expected,rtol=1e-5,atol=2e-7)
            for name,broken in (("missing-config",{k:v for k,v in metadata.items() if k!="transformer_config"}),
                    ("wrong-config",dict(metadata,transformer_config=json.dumps(companion)))):
                path=root/f"{name}.safetensors";save_file(state,str(path),metadata=broken)
                with patch("core.models.common.single_file_format.read_state_dict",side_effect=AssertionError("tensor loading")), \
                     patch("core.models.lens.lens_loader._resolve_lens_base_dir",side_effect=AssertionError("base resolution")):
                    with self.assertRaises(ValueError):ModelLoader.load_lens_from_path(str(path),torch_dtype=torch.float32)
            partial=root/"partial.safetensors"
            save_file({key:value for key,value in state.items() if key!="net.img_in.weight"},str(partial),metadata=metadata)
            with patch("core.models.lens.lens_loader.load_lens_components",side_effect=AssertionError("companion loading")):
                with self.assertRaisesRegex(ValueError,"weights do not match"):
                    ModelLoader.load_lens_from_path(str(partial),torch_dtype=torch.float32)
            dense=root/"dense.safetensors";save_file({"placeholder":torch.ones(1)},str(dense))
            self.assertEqual(preflight_checkpoint(dense),(None,None))
        self.assertFalse(torch.cuda.is_initialized())

    def test_directory_native_restore_and_header_refusals(self):
        from core.models.lens.residual_unet import restore_topology, native_config, preflight_directory
        from core.models.common.single_file_format import save_single_file_state
        from core.rewiring.spec import checkpoint_metadata
        from core.model_loader import ModelLoader
        from safetensors.torch import save_file, load_file
        restore_topology(self.model,self.spec)
        config=native_config(self.model);expected=self.forward().detach()
        metadata=checkpoint_metadata(self.spec)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            directories=[]
            for name,sharded in (("single",False),("sharded",True)):
                directory=root/name;transformer_dir=directory/"transformer"
                transformer_dir.mkdir(parents=True);(directory/"vae").mkdir()
                (directory/"vae"/"config.json").write_text("{}")
                (transformer_dir/"config.json").write_text(json.dumps(dict(config,_class_name="LensTransformer2DModel")))
                save_single_file_state(self.model.state_dict(),metadata,
                    str(transformer_dir/"diffusion_pytorch_model.safetensors"),
                    max_shard_bytes=20000 if sharded else 10**9)
                directories.append(directory)
                spec,declared,source=preflight_directory(directory)
                self.assertEqual(spec,self.spec);self.assertEqual(declared,config)
                self.assertTrue(source.is_file())
            text_encoder=torch.nn.Linear(1,1)
            text_encoder.set_selected_layers=lambda layers:setattr(text_encoder,"selected",layers)
            tokenizer=SimpleNamespace(pad_token_id=0,encode=lambda *a,**k:[1])
            with ExitStack() as stack:
                for target,value in (("core.models.lens.vendor.LensGptOssEncoder.from_pretrained",text_encoder),
                    ("diffusers.AutoencoderKLFlux2.from_pretrained",torch.nn.Linear(1,1)),
                    ("transformers.AutoTokenizer.from_pretrained",tokenizer),
                    ("diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained",object())):
                    stack.enter_context(patch(target,return_value=value))
                stack.enter_context(patch.object(type(self.model),"from_pretrained",side_effect=AssertionError("native pretrained allocation")))
                for directory in directories:
                    restored=ModelLoader.load_lens_from_path(str(directory),torch_dtype=torch.float32)["transformer"]
                    self.assertEqual(restored._residual_unet_spec,self.spec)
                    self.assertEqual(text_encoder.selected,(5,11))
                    for key,value in self.model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
                    with patch.object(self,"model",restored):torch.testing.assert_close(self.forward(),expected)
            config_path=directories[0]/"transformer"/"config.json"
            config_path.write_text(json.dumps(dict(config,num_layers=3)))
            with patch("core.models.common.single_file_format.read_state_dict",side_effect=AssertionError("tensor loading")), \
                 patch.object(type(self.model),"__init__",autospec=True,side_effect=AssertionError("native allocation")):
                with self.assertRaises(ValueError):ModelLoader.load_lens_from_path(str(directories[0]))
            config_path.write_text(json.dumps(config))
            weights=directories[0]/"transformer"/"diffusion_pytorch_model.safetensors"
            save_file({key:value for key,value in self.model.state_dict().items() if key!="img_in.weight"},str(weights),metadata=metadata)
            with patch("core.models.lens.vendor.LensGptOssEncoder.from_pretrained",side_effect=AssertionError("companion loading")):
                with self.assertRaisesRegex(ValueError,"weights do not match"):
                    ModelLoader.load_lens_from_path(str(directories[0]))
            index_path=directories[1]/"transformer"/"diffusion_pytorch_model.safetensors.index.json"
            index=json.loads(index_path.read_text())
            shard=index_path.parent/next(iter(index["weight_map"].values()))
            save_file(load_file(str(shard)),str(shard),metadata={})
            with patch("core.models.common.single_file_format.read_state_dict",side_effect=AssertionError("tensor loading")), \
                 patch.object(type(self.model),"__init__",autospec=True,side_effect=AssertionError("native allocation")):
                with self.assertRaisesRegex(ValueError,"declarations disagree"):
                    ModelLoader.load_lens_from_path(str(directories[1]))
            dense=root/"dense";(dense/"transformer").mkdir(parents=True)
            save_file({"placeholder":torch.ones(1)},str(dense/"transformer"/"diffusion_pytorch_model.safetensors"))
            self.assertEqual(preflight_directory(dense),(None,None,None))
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_checkpoint_modes_preserve_output_and_gradients(self):
        from core.models.lens.residual_unet import restore_topology
        restore_topology(self.model,self.spec)
        parameters = [self.images,*self.features,*self.model.parameters()]
        outputs = []; gradients = []
        for mode in ("none","standard","cpu_offload","async_cpu_offload"):
            self.model.disable_gradient_checkpointing()
            if mode != "none":
                self.model.enable_gradient_checkpointing(cpu_offload=mode=="cpu_offload",async_offload=mode=="async_cpu_offload")
            output = self.forward(); outputs.append(output.detach())
            gradients.append(torch.autograd.grad(output.square().mean(),parameters))
        for output,grads in zip(outputs,gradients):
            torch.testing.assert_close(output,outputs[0],rtol=0,atol=0)
            for actual,expected in zip(grads,gradients[0]):
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-7)
        self.assertTrue(any(torch.count_nonzero(value) for value in gradients[0][1:3]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_single_text_features_dense_restore_metadata_and_refusals(self):
        from core.models.lens.residual_unet import restore_topology,model_topology_metadata,native_config
        dense = self.forward().detach()
        restore_topology(self.model,self.spec)
        self.assertEqual(read_checkpoint_metadata(model_topology_metadata(self.model)),self.spec)
        for shapes in ([(2,8,6)],[(1,8,12),(1,12,8)],[(1,8,11)],[(True,8,12)]):
            with self.assertRaises(ValueError):self.forward(img_shapes=shapes)
        for options in (dict(encoder_hidden_states=self.features[:1]),dict(encoder_hidden_states_mask=self.mask[:,:4]),
                        dict(attention_kwargs={"scale":0.5})):
            with self.assertRaises(ValueError):self.forward(**options)
        for name in ("_style_ctx","_fbcache","_repa_tap_depth","_tread_config","_blockskip_config"):
            setattr(self.model,name,object())
            try:
                with self.assertRaises(ValueError):self.forward()
            finally:delattr(self.model,name)
        for name,value in (("_nag_enabled",True),("_negpip_enabled",True),("_style_ctx",object())):
            setattr(self.model.transformer_blocks[0].attn,name,value)
            try:
                with self.assertRaises(ValueError):self.forward()
            finally:delattr(self.model.transformer_blocks[0].attn,name)
        with patch.object(self.model.pos_embed,"scale_rope",False):
            with self.assertRaisesRegex(ValueError,"centered RoPE"):self.forward()
        for shapes in ([(1,8,12)],(1,8,12),[[(1,8,12)]],[(1,8,12)]*2):
            self.assertTrue(torch.isfinite(self.forward(img_shapes=shapes)).all())
        restore_topology(self.model,None)
        torch.testing.assert_close(self.forward(),dense,rtol=0,atol=0)
        self.assertEqual(model_topology_metadata(self.model),{})
        with self.assertRaises(ValueError):native_config(dict(self.model.config,axes_dims_rope=[2,2,2]))
        with self.assertRaises(ValueError):native_config(dict(self.model.config,patch_size=True))
        from core.models.lens.vendor.transformer import LensTransformer2DModel
        model = LensTransformer2DModel(**dict(native_config(self.model),multi_layer_encoder_feature=False))
        restore_topology(model,build_spec("lens",self.spec["streams"],self.spec["regions"],native_config(model)))
        with patch.object(self,"model",model):
            self.assertTrue(torch.isfinite(self.forward(encoder_hidden_states=self.features[0])).all())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":unittest.main()
