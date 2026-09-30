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

if __name__ == "__main__":
    unittest.main()
