"""Read-only model inspection and actual FastAPI inspection contracts."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import save_file
with patch.object(torch.cuda,"get_device_capability",return_value=(8,9)), \
     patch.object(torch.cuda,"_lazy_init"),patch.object(torch._C,"_cuda_init"):
    from api.rewiring_routes import router
    from core.models.qwen_image_21.artifact import artifact_metadata, DEFAULT_TRANSFORMER_CONFIG
    from core.models.qwen_image_21.residual_unet import default_topology
    from core.rewiring.inspect import inspect_model
from core.rewiring.spec import checkpoint_metadata


class InspectionContracts(unittest.TestCase):
    def test_file_directory_and_api_without_model_load(self):
        import httpx
        from fastapi import FastAPI
        config = dict(DEFAULT_TRANSFORMER_CONFIG)
        topology = default_topology(config)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "transformer.safetensors"
            save_file({"placeholder":torch.ones(1)},str(model),
                      metadata={**artifact_metadata("transformer",config),**checkpoint_metadata(topology)})
            (root/"manifest.json").write_text(json.dumps(dict(
                model_type="qwen_image_21",format_version="1",transformer_config=config,
                components=dict(transformer=model.name,text_encoder="te",vae="vae",processor="processor",text_encoder_config="te.json"))),encoding="utf-8")
            with patch("core.models.qwen_image_21.artifact.read_state_dict",side_effect=AssertionError("Tensor loading forbidden")), \
                 patch("core.models.qwen_image_21.artifact.QwenImage21Transformer2DModel",side_effect=AssertionError("Model allocation forbidden")):
                file_info = inspect_model(str(model))
                self.assertEqual(file_info,inspect_model(str(root)))
                self.assertEqual(file_info["topology"],topology)
                self.assertEqual(file_info["parameter_counts"]["total"], 7115124736)
                self.assertEqual(file_info["parameter_counts"]["added"], 0)
                self.assertEqual(file_info["streams"][0]["block_count"],32)
                app=FastAPI(); app.include_router(router,prefix="/api/v1")
                async def check():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
                        reply=await client.post("/api/v1/models/rewiring/inspect",json={"model_path":str(model)})
                        self.assertEqual(reply.status_code,200,reply.text)
                        self.assertEqual(reply.json(),file_info)
                        missing=await client.post("/api/v1/models/rewiring/inspect",json={"model_path":str(root/"missing")})
                        self.assertEqual(missing.status_code,422)
                        invalid=root/"invalid.safetensors"; invalid.write_text("invalid",encoding="utf-8")
                        self.assertEqual((await client.post("/api/v1/models/rewiring/inspect",json={"model_path":str(invalid)})).status_code,422)
                asyncio.run(check())
        self.assertFalse(torch.cuda.is_initialized())


    def test_krea_file_directory_and_api_without_model_load(self):
        import httpx
        from fastapi import FastAPI
        from core.models.krea2.vendor import single_file
        from core.models.krea2.residual_unet import default_topology
        config = dict(single_file.KREA2_DEFAULT_CONFIG)
        topology = default_topology(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/"diffusion_pytorch_model.safetensors"
            save_file({"placeholder": torch.ones(1)}, str(model), metadata={
                "model_type": "krea2", "krea2_config": json.dumps(config),
                **checkpoint_metadata(topology)})
            (root/"config.json").write_text(json.dumps({
                **config, "_class_name": "Krea2Transformer2DModel"}), encoding="utf-8")
            with patch.object(single_file, "_read_safetensors", side_effect=AssertionError("Tensor loading forbidden")), \
                 patch.object(single_file, "Krea2Transformer2DModel", side_effect=AssertionError("Model allocation forbidden")):
                info = inspect_model(str(model))
                self.assertEqual(info, inspect_model(str(root)))
                self.assertEqual(info["architecture"], "krea2")
                self.assertEqual(info["topology"], topology)
                self.assertEqual(info["pixel_alignment"], 16)
                app = FastAPI(); app.include_router(router, prefix="/api/v1")
                async def check():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                        response = await client.post("/api/v1/models/rewiring/inspect", json={"model_path": str(root)})
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertEqual(response.json(), info)
                        (root/"config.json").write_text(json.dumps({"_class_name": []}), encoding="utf-8")
                        invalid = await client.post("/api/v1/models/rewiring/inspect", json={"model_path": str(root)})
                        self.assertEqual(invalid.status_code, 422)
                asyncio.run(check())
        self.assertFalse(torch.cuda.is_initialized())

    def test_custom_layout_initial_spec_is_editable_through_api(self):
        import httpx
        from fastapi import FastAPI
        from types import SimpleNamespace
        from core.models.krea2.vendor.single_file import KREA2_DEFAULT_CONFIG
        from core.models.krea2.residual_unet import configure_training_topology
        from core.rewiring.spec import validate_spec
        config = dict(KREA2_DEFAULT_CONFIG, num_layers=7)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/"custom.safetensors"
            save_file({"placeholder": torch.ones(1)}, str(model), metadata={
                "model_type": "krea2", "krea2_config": json.dumps(config)})
            info = inspect_model(str(model))
            self.assertIsNone(info["default_spec"])
            initial = validate_spec(info["initial_spec"])
            self.assertEqual(initial["regions"][0]["start_block"], 1)
            self.assertEqual(initial["regions"][0]["end_block"], 6)
            native = SimpleNamespace(config=config)
            resolved = configure_training_topology(native,
                {"dit_rewiring_mode": "configure", "dit_rewiring_spec": initial})
            self.assertEqual(resolved, initial)
            app = FastAPI(); app.include_router(router, prefix="/api/v1")
            async def check():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.post("/api/v1/models/rewiring/validate", json={
                        "spec": initial, "token_height": 8, "token_width": 12})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual([b["tokens"] for b in response.json()["plan"]["blocks"]],
                                     [96, 24, 24, 24, 24, 24, 96])
            asyncio.run(check())
        self.assertFalse(torch.cuda.is_initialized())

    def test_anima_native_file_split_shards_and_api(self):
        import httpx
        from fastapi import FastAPI
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)), \
             patch.object(torch.cuda, "_lazy_init"), patch.object(torch._C, "_cuda_init"):
            from core.models.anima.anima_models import Anima
        from core.models.anima.residual_unet import native_config, configure_training_topology
        from core.models.common.single_file_format import save_single_file_state
        from core.rewiring.spec import OPERATORS, build_spec
        model = Anima(model_channels=24, num_blocks=7, num_heads=2,
            mlp_ratio=1.0, crossattn_emb_channels=12, adaln_lora_dim=8, use_llm_adapter=False)
        config = native_config(model)
        topology = build_spec("anima", [{"name": "main", "block_count": 7}], [dict(
            id="r0", stream="main", parent=None, start_block=1, end_block=6,
            down_factor=[2, 2], **OPERATORS)], config)
        tensors = {"net."+name: value for name, value in model.state_dict().items()}
        metadata = {"model_type": "anima", "transformer_config": json.dumps(config),
                    **checkpoint_metadata(topology)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/"split_files"/"diffusion_models"; folder.mkdir(parents=True)
            source = folder/"anima.safetensors"
            save_file(tensors, str(source), metadata=metadata)
            with patch("safetensors.torch.load_file", side_effect=AssertionError("Tensor loading forbidden")), \
                 patch.object(Anima, "__init__", side_effect=AssertionError("Model allocation forbidden")):
                info = inspect_model(str(source))
                self.assertEqual(info, inspect_model(str(root)))
                self.assertEqual(info["architecture"], "anima")
                self.assertEqual(info["parameter_counts"]["total"], sum(p.numel() for p in model.parameters()))
                self.assertEqual(info["topology"], topology)
                self.assertEqual(info["streams"], [{"name": "main", "block_count": 7}])
                self.assertIsNone(info["default_spec"])
                resolved = configure_training_topology(model, {
                    "dit_rewiring_mode": "configure", "dit_rewiring_spec": info["initial_spec"]})
                self.assertEqual(resolved, info["initial_spec"])
                sharded = root/"sharded.safetensors"
                save_single_file_state(tensors, metadata, str(sharded), max_shard_bytes=12000)
                self.assertEqual(info, inspect_model(str(sharded)))
                app = FastAPI(); app.include_router(router, prefix="/api/v1")
                async def check():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                        for path in (source, root, Path(str(sharded)+".index.json")):
                            response = await client.post("/api/v1/models/rewiring/inspect", json={"model_path": str(path)})
                            self.assertEqual(response.status_code, 200, response.text)
                            self.assertEqual(response.json(), info)
                        schema = (await client.get("/api/v1/schema/dit-rewiring")).json()
                        self.assertIn("anima", schema["runtime_architectures"])
                        self.assertEqual(schema["presets"]["anima"]["regions"][1]["end_block"], 16)
                        invalid = root/"adapter.safetensors"
                        save_file({"lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.ones(2, 24)},
                            str(invalid), metadata={"model_type": "anima"})
                        reply = await client.post("/api/v1/models/rewiring/inspect", json={"model_path": str(invalid)})
                        self.assertEqual(reply.status_code, 422)
                asyncio.run(check())
        self.assertFalse(torch.cuda.is_initialized())


    def test_zimage_file_shard_directory_and_public_inspection_without_weights(self):
        import httpx
        from fastapi import FastAPI
        import safetensors
        from core.models.zimage_residual_unet import native_config, default_topology, configure_training_topology
        from types import SimpleNamespace
        from core.models.common.single_file_format import save_single_file_state
        config = native_config({})
        topology = default_topology(config)
        original = safetensors.safe_open
        class HeaderOnly:
            def __init__(self,*args,**kwargs): self.handle=original(*args,**kwargs)
            def __enter__(self): self.handle.__enter__(); return self
            def __exit__(self,*args): return self.handle.__exit__(*args)
            def metadata(self): return self.handle.metadata()
            def keys(self): return self.handle.keys()
            def get_tensor(self,*args): raise AssertionError("Tensor loading forbidden")
        metadata = {"model_type":"zimage", "transformer_config":json.dumps(config), **checkpoint_metadata(topology)}
        weights = {"all_x_embedder.2-1.weight":torch.ones(2,2), "layers.0.attention.to_q.weight":torch.ones(2,2)}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"model.safetensors"
            save_file(weights,str(source),metadata=metadata)
            (root/"config.json").write_text(json.dumps({**config,"_class_name":"ZImageTransformer2DModel"}),encoding="utf-8")
            shard_dir=root/"sharded"; shard_dir.mkdir()
            index=save_single_file_state(weights,metadata,str(shard_dir/"model.safetensors"),max_shard_bytes=16)
            with patch("core.rewiring.inspect.safe_open",HeaderOnly), patch("core.rewiring.artifacts.safe_open",HeaderOnly), \
                 patch.object(safetensors,"safe_open",HeaderOnly):
                info=inspect_model(str(source))
                self.assertEqual(info,inspect_model(str(root)))
                self.assertEqual(info,inspect_model(str(index)))
                self.assertEqual(info,inspect_model(str(shard_dir/"model.safetensors")))
                self.assertEqual(info["architecture"],"zimage")
                self.assertEqual(info["streams"],[{"name":"main","block_count":30}])
                self.assertEqual(info["default_spec"],topology)
                app=FastAPI(); app.include_router(router,prefix="/api/v1")
                async def check():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
                        schema=(await client.get("/api/v1/schema/dit-rewiring")).json()
                        self.assertIn("zimage",schema["runtime_architectures"])
                        self.assertEqual(schema["presets"]["zimage"]["regions"][1]["end_block"],17)
                        reply=await client.post("/api/v1/models/rewiring/inspect",json={"model_path":str(source)})
                        self.assertEqual(reply.status_code,200,reply.text)
                        self.assertEqual(reply.json(),info)
                        for value in (None, 2, [True], [0]):
                            invalid=root/"invalid_config.safetensors"
                            save_file(weights,str(invalid),metadata={"model_type":"zimage",
                                "transformer_config":json.dumps(dict(config,all_patch_size=value))})
                            refused=await client.post("/api/v1/models/rewiring/inspect",json={"model_path":str(invalid)})
                            self.assertEqual(refused.status_code,422,refused.text)
                asyncio.run(check())
                adapter=root/"adapter.safetensors"
                save_file({"lora_transformer_layers_0_attention_to_q.lora_down.weight":torch.ones(2,2)},str(adapter),metadata=metadata)
                with self.assertRaisesRegex(ValueError,"not adapter weights"): inspect_model(str(adapter))
                legacy=root/"legacy.safetensors"; save_file(weights,str(legacy),metadata={"model_type":"zimage"})
                with self.assertRaisesRegex(ValueError,"native config metadata"): inspect_model(str(legacy))
                custom=root/"custom.safetensors"; custom_config=dict(config,n_layers=7)
                save_file(weights,str(custom),metadata={"model_type":"zimage","transformer_config":json.dumps(custom_config)})
                custom_info=inspect_model(str(custom))
                self.assertIsNone(custom_info["default_spec"])
                model=SimpleNamespace(**custom_config,config=custom_config)
                self.assertEqual(configure_training_topology(model,{"dit_rewiring_mode":"configure",
                    "dit_rewiring_spec":custom_info["initial_spec"]}),custom_info["initial_spec"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_anima_inspection_keeps_native_stack_and_cuda_unloaded(self):
        import subprocess
        from core.models.anima.config import ANIMA_DIT_CONFIG
        keys = {name: torch.ones(1) for name in (
            "blocks.0.self_attn.q_proj.weight", "blocks.0.cross_attn.q_proj.weight",
            "blocks.0.adaln_modulation_self_attn.1.weight", "final_layer.linear.weight")}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"native.safetensors"
            save_file(keys, str(source))
            script = r"""
import sys
sys.path.insert(0, sys.argv[1])
import torch, safetensors
from unittest.mock import patch
with patch.object(torch.cuda, 'get_device_capability', return_value=(8, 9)), \
     patch.object(torch.cuda, '_lazy_init'), patch.object(torch._C, '_cuda_init'):
    import api.param_defaults
assert not torch.cuda.is_initialized()
original = safetensors.safe_open
class HeaderOnly:
    def __init__(self, *args, **kwargs): self.handle = original(*args, **kwargs)
    def __enter__(self): self.handle.__enter__(); return self
    def __exit__(self, *args): return self.handle.__exit__(*args)
    def metadata(self): return self.handle.metadata()
    def keys(self): return self.handle.keys()
with patch.object(safetensors, 'safe_open', HeaderOnly), \
     patch.object(torch.cuda, 'get_device_capability', side_effect=AssertionError('GPU discovery forbidden')), \
     patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')), \
     patch.object(torch._C, '_cuda_init', side_effect=AssertionError('CUDA forbidden')):
    from core.rewiring.inspect import inspect_model
    info = inspect_model(sys.argv[2])
assert info['architecture'] == 'anima'
assert info['parameter_counts']['total'] == 2091068928
assert info['streams'][0]['block_count'] == 28
assert info['default_spec']['regions'][1]['end_block'] == 16
assert len(info['hybrid_candidates']) == 1
assert info['hybrid_candidates'][0]['spec']['skips'][0]['rank'] == 256
assert 'core.models.anima.anima_models' not in sys.modules
assert 'core.models.anima.anima_loader' not in sys.modules
assert not torch.cuda.is_initialized()
"""
            completed = subprocess.run([sys.executable, "-c", script,
                str(Path(__file__).resolve().parents[1]), str(source)], capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            config = dict(ANIMA_DIT_CONFIG, patch_spatial=4)
            save_file(keys, str(source), metadata={"model_type": "anima", "transformer_config": json.dumps(config)})
            info = inspect_model(str(source))
            self.assertEqual(info["latent_scale_factor"], 32)
            self.assertEqual(info["pixel_alignment"], 32)
            self.assertEqual(info["hybrid_candidates"], [])
            for value in (True, 0):
                save_file(keys, str(source), metadata={"model_type": "anima", "transformer_config": json.dumps(dict(config, patch_spatial=value))})
                with self.assertRaisesRegex(ValueError, "patch size"):
                    inspect_model(str(source))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
