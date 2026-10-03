"""CPU contracts for standalone ACE-Step full artifacts and frozen sources."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file


class AceStepSourceContracts(unittest.TestCase):
    def setUp(self):
        for obj,name,value in ((torch.cuda,"get_device_capability",lambda *a,**k:(8,9)),
                (torch.cuda,"_lazy_init",lambda *a,**k:None),(torch._C,"_cuda_init",lambda *a,**k:None)):
            guard=patch.object(obj,name,value);guard.start();self.addCleanup(guard.stop)
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.root=Path(folder.name)/"original"
        for name in ("diffusion_models","vae","text_encoders","tokenizer"):
            (self.root/name).mkdir(parents=True)
        self.vae=self.root/"vae"/"ace_1.5_vae.safetensors"
        self.te=self.root/"text_encoders"/"qwen_0.6b_ace15.safetensors"
        for path in (self.vae,self.te): save_file({"marker":torch.ones(1)},str(path))
        self.tokenizer=self.root/"tokenizer"
        self.sources=dict(schema_version=1,root=str(self.root),vae=str(self.vae),
                          text_encoder=str(self.te),tokenizer=str(self.tokenizer))
        from core.models.acestep.vendor import AceStepConfig, AceStepConditionGenerationModel
        from core.models.acestep.residual_unet import native_config, restore_topology
        from core.rewiring.spec import build_spec, OPERATORS
        config=AceStepConfig(hidden_size=16,intermediate_size=24,num_hidden_layers=7,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,patch_size=2,
            in_channels=12,audio_acoustic_hidden_dim=4,encoder_hidden_size=16,
            sliding_window=3,num_lyric_encoder_hidden_layers=1,num_timbre_encoder_hidden_layers=1,
            num_attention_pooler_hidden_layers=1,num_audio_decoder_hidden_layers=1,
            text_hidden_dim=8,timbre_hidden_dim=4,fsq_dim=16,fsq_input_levels=[2,2],pool_window_size=2)
        config._attn_implementation="sdpa"
        self.owner=AceStepConditionGenerationModel(config).eval()
        self.spec=build_spec("acestep",[{"name":"main","block_count":7}],[
            dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,1],**OPERATORS),
            dict(id="inner",stream="main",parent="outer",start_block=2,end_block=5,down_factor=[2,1],**OPERATORS)],
            native_config(config))
        restore_topology(self.owner,self.spec)
        self.owner._acestep_component_sources=self.sources
        self.path=Path(folder.name)/"outputs"/"full.safetensors"
        from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter
        AceStepFullParameterAdapter(SimpleNamespace(transformer=self.owner)).write_checkpoint(3,1,self.path)

    def test_dense_full_export_with_external_sources_roundtrips(self):
        from core.models.acestep.residual_unet import restore_topology
        from core.models.acestep.loader import _build_dit, detect_acestep_layout
        from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter
        from core.rewiring.spec import read_checkpoint_metadata
        restore_topology(self.owner,None)
        path=self.path.parent/"dense.safetensors"
        AceStepFullParameterAdapter(SimpleNamespace(transformer=self.owner)).write_checkpoint(1,1,path)
        with safe_open(str(path),framework="pt",device="cpu") as handle:
            metadata=handle.metadata()
        self.assertIsNone(read_checkpoint_metadata(metadata))
        self.assertEqual(json.loads(metadata["acestep_components"]),self.sources)
        restored, _=_build_dit(str(path),torch.float32)
        self.assertIsNone(getattr(restored,"_residual_unet_spec",None))
        for key,value in self.owner.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
        self.assertEqual(detect_acestep_layout(str(path))["vae"],str(self.vae))
        self.assertFalse(torch.cuda.is_initialized())

    def test_public_temporal_inspection_and_configure_api(self):
        import asyncio
        import httpx
        from fastapi import FastAPI
        from api.rewiring_routes import router
        from core.rewiring.inspect import inspect_model
        from core.models.acestep.residual_unet import configure_training_topology, restore_topology, temporal_grid
        with patch("safetensors.torch.load_file", side_effect=AssertionError("No tensor loading")), \
                patch("core.models.acestep.vendor.AceStepConditionGenerationModel", side_effect=AssertionError("No allocation")):
            info = inspect_model(str(self.path))
        self.assertEqual(info["topology"], self.spec)
        self.assertEqual(info["streams"], [{"name":"main", "block_count":7}])
        self.assertEqual(info["temporal_geometry"], dict(frame_rate=25, patch_size=2,
            preview_frames=750, grid_policy="zero_token_tail_v1"))
        self.assertIsNone(info["default_spec"])
        self.assertEqual(info["initial_spec"]["regions"][0]["down_factor"], [2,1])
        restore_topology(self.owner, None)
        custom = configure_training_topology(self.owner, dict(dit_rewiring_mode="configure",
            dit_rewiring_spec=info["initial_spec"]))
        self.assertEqual(custom, info["initial_spec"])
        padded, _ = temporal_grid(custom, self.owner.config, 375)
        app=FastAPI(); app.include_router(router,prefix="/api/v1")
        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
                response=await client.post("/api/v1/models/rewiring/inspect",json={"model_path":str(self.path)})
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(response.json(),info)
                schema=(await client.get("/api/v1/schema/dit-rewiring")).json()
                self.assertIn("acestep",schema["runtime_architectures"])
                reply=await client.post("/api/v1/models/rewiring/validate",json=dict(
                    spec=custom,token_height=padded,token_width=1))
                self.assertEqual(reply.status_code,200,reply.text)
                self.assertEqual([b["tokens"] for b in reply.json()["plan"]["blocks"]],
                                 [376,188,188,188,188,188,376])
                self.assertEqual((await client.post("/api/v1/models/rewiring/validate",json=dict(
                    spec=custom,token_height=375,token_width=1))).status_code,422)
        asyncio.run(check())
        self.assertFalse(torch.cuda.is_initialized())

    def test_temporal_directory_and_legacy_identity_inspection(self):
        import shutil
        from core.rewiring.inspect import inspect_model
        from core.models.acestep.residual_unet import native_config, temporal_grid, GRID_POLICY_KEY
        from core.rewiring.spec import build_spec, checkpoint_metadata
        direct=inspect_model(str(self.path))
        selected=self.root/"diffusion_models"/"acestep_v1.5_turbo.safetensors"
        shutil.copyfile(self.path,selected)
        self.assertEqual(inspect_model(str(self.root)),direct)
        old_native=native_config(self.owner.config);old_native.pop(GRID_POLICY_KEY)
        legacy=build_spec("acestep",self.spec["streams"],self.spec["regions"],old_native)
        raw=self.owner.config.to_dict();raw.pop(GRID_POLICY_KEY,None)
        state={"net."+key:value for key,value in self.owner.state_dict().items()}
        saved=self.path.parent/"legacy.safetensors"
        save_file(state,str(saved),metadata={"model_type":"acestep","acestep_config":json.dumps(raw),
            **checkpoint_metadata(legacy)})
        info=inspect_model(str(saved))
        self.assertEqual(info["topology"],legacy)
        self.assertNotEqual(info["base_config_sha256"],legacy["base_config_sha256"])
        with self.assertRaises(ValueError): temporal_grid(legacy,raw,375)
        self.assertEqual(temporal_grid(legacy,raw,376),(376,"require_divisible"))
        self.assertFalse(torch.cuda.is_initialized())

    def test_default_four_deep_blocks_and_native_cpu_backward(self):
        from core.models.acestep.vendor import AceStepConfig, AceStepDiTModel
        from core.models.acestep.residual_unet import configure_training_topology, temporal_grid, generation_metadata
        from core.rewiring.spec import preview_plan
        raw=self.owner.config.to_dict(); raw.update(num_hidden_layers=24,layer_types=None)
        config=AceStepConfig(**raw); config._attn_implementation="sdpa"
        model=AceStepDiTModel(config)
        params=dict(dit_rewiring_mode="configure")
        spec=configure_training_topology(model,params)
        self.assertEqual(params["dit_rewiring_spec"],spec)
        self.assertEqual([(r["start_block"],r["end_block"]) for r in spec["regions"]],[(4,20),(10,14)])
        padded, _=temporal_grid(spec,config,375)
        self.assertEqual(padded,376)
        plan=preview_plan(spec,padded,1)
        self.assertEqual([sum(b["depth"]==d for b in plan["blocks"]) for d in range(3)],[8,12,4])
        self.assertEqual(generation_metadata(model,750)["transport_padding_tokens"],1)
        audio=torch.randn(1,45,4,requires_grad=True)
        output=model(hidden_states=audio,timestep=torch.tensor([.5]),timestep_r=torch.tensor([.1]),
            attention_mask=torch.ones(1,45),encoder_hidden_states=torch.randn(1,3,16),
            encoder_attention_mask=torch.ones(1,3),context_latents=torch.randn(1,45,8),use_cache=False)[0]
        self.assertEqual(output.shape,audio.shape);self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward();self.assertTrue(torch.isfinite(audio.grad).all())
        edited=json.loads(json.dumps(spec));edited["regions"][1]["end_block"]=15
        with self.assertRaisesRegex(ValueError,"without edits"):
            configure_training_topology(model,dict(dit_rewiring_mode="configure",dit_rewiring_spec=edited),resuming=True)
        self.assertEqual(model._residual_unet_spec,spec)
        self.assertFalse(torch.cuda.is_initialized())

    def test_inspection_refuses_custom_missing_config_and_wrong_native_shapes(self):
        from core.rewiring.inspect import inspect_model
        state={"net."+key:value for key,value in self.owner.state_dict().items()}
        path=self.path.parent/"invalid.safetensors"
        save_file(state,str(path),metadata={"model_type":"acestep"})
        with self.assertRaisesRegex(ValueError,"block counts"):
            inspect_model(str(path))
        raw=self.owner.config.to_dict()
        state["net.decoder.proj_in.1.weight"]=torch.ones(16,11,2)
        save_file(state,str(path),metadata={"model_type":"acestep","acestep_config":json.dumps(raw)})
        with self.assertRaisesRegex(ValueError,"header geometry"):
            inspect_model(str(path))
        self.assertFalse(torch.cuda.is_initialized())

    def test_standalone_loader_uses_exact_sources_and_reexports_them(self):
        from core.models.acestep.loader import detect_acestep_layout, load_acestep_from_path
        from core.model_loader import ModelLoader
        from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter
        self.assertEqual(ModelLoader.detect_model_type(str(self.path)),"acestep")
        layout=detect_acestep_layout(str(self.path))
        self.assertEqual(layout["vae"],str(self.vae));self.assertEqual(layout["text_encoder"],str(self.te))
        self.assertEqual(layout["dit"],str(self.path));self.assertEqual(layout["tokenizer_source"],str(self.tokenizer))
        vae=torch.nn.Linear(1,1);te=torch.nn.Linear(1,1);tokenizer=object()
        with patch("core.models.acestep.loader._build_vae",return_value=vae) as build_vae, \
                patch("core.models.acestep.loader._build_text_encoder",return_value=(te,{})) as build_te, \
                patch("transformers.AutoTokenizer.from_pretrained",return_value=tokenizer) as build_tokenizer, \
                patch.object(torch.cuda,"is_available",return_value=False):
            loaded=load_acestep_from_path(str(self.path),torch.float32)
        build_vae.assert_called_once_with(str(self.vae),torch.float32)
        build_te.assert_called_once_with(str(self.te),torch.float32)
        build_tokenizer.assert_called_once_with(str(self.tokenizer))
        self.assertIs(loaded["tokenizer"],tokenizer)
        self.assertEqual(loaded["dit"]._acestep_component_sources,self.sources)
        self.assertEqual(loaded["dit"]._residual_unet_spec,self.spec)
        for name,value in self.owner.state_dict().items():
            torch.testing.assert_close(value,loaded["dit"].state_dict()[name],rtol=0,atol=0)
        second=self.path.with_name("next.safetensors")
        AceStepFullParameterAdapter(SimpleNamespace(transformer=loaded["dit"])).write_checkpoint(4,1,second)
        with safe_open(str(second),framework="pt") as handle:
            self.assertEqual(json.loads(handle.metadata()["acestep_components"]),self.sources)
        self.assertFalse(torch.cuda.is_initialized())

    def test_declared_sources_take_priority_and_legacy_tree_still_detects(self):
        from core.models.acestep.loader import detect_acestep_layout
        from core.model_loader import ModelLoader
        legacy=self.root/"diffusion_models"/"acestep_v1.5_turbo.safetensors"
        save_file(self.owner.state_dict(),str(legacy))
        layout=detect_acestep_layout(str(legacy))
        self.assertEqual(layout["vae"],str(self.vae))
        self.assertEqual(layout["text_encoder"],str(self.te))
        self.assertEqual(ModelLoader.detect_model_type(str(self.root)),"acestep")
        self.assertEqual(detect_acestep_layout(str(self.root)),layout)
        with safe_open(str(self.path),framework="pt") as handle: metadata=handle.metadata()
        decoy=self.path.parent/"diffusion_models"
        decoy.mkdir()
        copied=decoy/"full.safetensors"
        save_file(load_file(str(self.path)),str(copied),metadata=metadata)
        layout=detect_acestep_layout(str(copied))
        self.assertEqual(layout["root"],str(self.root))
        self.assertEqual(layout["vae"],str(self.vae))
        self.assertEqual(detect_acestep_layout(str(self.path.parent)),layout)

    def test_declared_sources_refuse_before_model_allocation(self):
        from core.models.acestep.loader import load_acestep_from_path
        with safe_open(str(self.path),framework="pt") as handle: metadata=handle.metadata()
        weights=load_file(str(self.path))
        invalid=[{**self.sources,"schema_version":2},{**self.sources,"vae":str(self.path.parent/"missing")},
                 {**self.sources,"text_encoder":str(self.root)},{**self.sources,"root":"relative"},
                 {**self.sources,"tokenizer":str(self.root/"missing")},{**self.sources,"extra":1}]
        for sources in invalid:
            with self.subTest(sources=sources):
                save_file(weights,str(self.path),metadata={**metadata,"acestep_components":json.dumps(sources)})
                with patch("core.models.acestep.loader._build_dit",side_effect=AssertionError("No allocation")), \
                        self.assertRaises(ValueError):
                    load_acestep_from_path(str(self.path),torch.float32)
        self.assertFalse(torch.cuda.is_initialized())

    def test_legacy_and_new_grid_policies_roundtrip_in_full_config(self):
        from core.models.acestep.residual_unet import native_config, restore_topology, generation_metadata, GRID_POLICY_KEY, GRID_POLICY
        from core.models.acestep.loader import _build_dit
        from core.rewiring.spec import build_spec
        from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter
        legacy_config=native_config(self.owner.config);legacy_config.pop(GRID_POLICY_KEY)
        legacy=build_spec("acestep",self.spec["streams"],self.spec["regions"],legacy_config)
        restore_topology(self.owner,legacy)
        AceStepFullParameterAdapter(SimpleNamespace(transformer=self.owner)).write_checkpoint(4,1,self.path)
        with safe_open(str(self.path),framework="pt") as handle: metadata=handle.metadata()
        raw=json.loads(metadata["acestep_config"])
        self.assertEqual(raw[GRID_POLICY_KEY],"require_divisible")
        loaded,config=_build_dit(str(self.path),torch.float32)
        self.assertEqual(loaded._residual_unet_spec,legacy)
        self.assertEqual(native_config(native_config(config)),native_config(config))
        with self.assertRaisesRegex(ValueError,"not divisible"): generation_metadata(loaded,45)
        raw[GRID_POLICY_KEY]=GRID_POLICY
        save_file(load_file(str(self.path)),str(self.path),metadata={**metadata,"acestep_config":json.dumps(raw)})
        with patch("core.models.acestep.vendor.AceStepConditionGenerationModel",side_effect=AssertionError("No allocation")), \
                self.assertRaisesRegex(ValueError,"explicit grid policy"):
            _build_dit(str(self.path),torch.float32)
        restore_topology(self.owner,self.spec)
        AceStepFullParameterAdapter(SimpleNamespace(transformer=self.owner)).write_checkpoint(5,1,self.path)
        loaded,config=_build_dit(str(self.path),torch.float32)
        self.assertEqual(generation_metadata(loaded,250)["transport_padding_tokens"],3)
        self.assertEqual(getattr(config,GRID_POLICY_KEY),GRID_POLICY)
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_resume_dispatches_to_native_component_loader(self):
        from core.training.base_trainer import BaseTrainer
        trainer=SimpleNamespace(model_path="unused",log_prefix="[CPU]")
        paths=[]
        with patch("core.training.ops.acestep_ops.load_components",side_effect=lambda owner:paths.append(owner.model_path)):
            BaseTrainer._load_checkpoint_as_base(trainer,str(self.path))
        self.assertEqual(paths,[str(self.path)])
        self.assertTrue(trainer.is_acestep)
        self.assertEqual(trainer.model_path,str(self.path))
        self.assertFalse(torch.cuda.is_initialized())
