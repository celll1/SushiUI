"""Streaming acoustic checkpoint export through the real API, CPU only."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
torch.cuda.get_device_capability=lambda *a,**k:(8,9)
torch.cuda._lazy_init=lambda *a,**k:None
torch._C._cuda_init=lambda *a,**k:None

from tests.dit_rewiring_yue2_test import fixture
from core.models.yue2 import loader
from core.models.yue2.single_file import save_yue2_single_file
from core.models.yue2.vendor.modeling_yue2 import YuE2ForCausalLM
from core.models.yue2.vendor.nar import CachedNAR
from core.models.yue2.residual_unet import restore_topology
from core.rewiring.export import export_acoustic_topology
from core.rewiring.inspect import inspect_model


def payload(path):
    with Path(path).open("rb") as stream:
        size=struct.unpack("<Q",stream.read(8))[0]
        stream.seek(size,1)
        return stream.read()


class AcousticExport(unittest.TestCase):
    def test_public_export_preserves_bytes_and_restores_yue2_execution(self):
        import httpx
        from fastapi import FastAPI
        from api.rewiring_routes import router
        model,graph,chunk=fixture()
        model.bfloat16()
        vae=torch.nn.Module()
        vae.register_buffer("scale",torch.ones(1))
        tokenizer=SimpleNamespace(payload='{"version":"1.0"}')
        def empty():
            with torch.device("meta"):
                copied=YuE2ForCausalLM(deepcopy(model.config))
                copied_vae=torch.nn.Module()
                copied_vae.register_buffer("scale",torch.ones(1))
            return copied,copied_vae
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"dense.safetensors"
            save_yue2_single_file(source,model,vae,tokenizer,extra_metadata={"user.note":"keep"})
            original=source.read_bytes()
            app=FastAPI();app.include_router(router,prefix="/api/v1")
            async def check():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
                    response=await client.post("/api/v1/models/rewiring/export",json={
                        "model_path":str(source),"output_name":"unet.safetensors","spec":graph})
                    self.assertEqual(response.status_code,200,response.text)
                    return Path(response.json()["model_path"])
            with patch.object(loader,"build_empty_models",side_effect=empty), \
                    patch.object(loader,"EmbeddedTokenizer",side_effect=AssertionError("Tokenizer payload read")):
                exported=asyncio.run(check())
                self.assertEqual(inspect_model(exported)["topology"],graph)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(payload(exported),payload(source))
            with patch.object(loader,"build_empty_models",side_effect=empty), \
                    patch.object(loader,"EmbeddedTokenizer",side_effect=lambda text: SimpleNamespace(payload=text)):
                restored=loader.load_yue2_from_path(exported)["transformer"]
            self.assertEqual(restored._residual_unet_spec,graph)
            restore_topology(model,graph)
            expected=CachedNAR(model,chunk)
            actual=CachedNAR(restored,chunk)
            torch.testing.assert_close(actual.velocity(chunk.noise.bfloat16(),.3),
                                       expected.velocity(chunk.noise.bfloat16(),.3),rtol=0,atol=0)
            expected.close();actual.close()
            with loader.safe_open(str(exported),framework="pt",device="cpu") as handle:
                self.assertEqual(handle.metadata()["user.note"],"keep")
            with patch.object(loader,"build_empty_models",side_effect=empty):
                for name in ("../escape.safetensors","sub/file.safetensors","sub\\file.safetensors",
                             "C:bad.safetensors","CON.safetensors","unet.safetensors","dense.safetensors"):
                    with self.subTest(name=name),self.assertRaises(ValueError):
                        export_acoustic_topology(source,name,graph)
                wrong=deepcopy(graph);wrong["base_config_sha256"]="0"*64
                with self.assertRaisesRegex(ValueError,"different native"):
                    export_acoustic_topology(source,"bad.safetensors",wrong)
                with patch("core.rewiring.export.shutil.copyfileobj",side_effect=OSError("disk failure")):
                    with self.assertRaisesRegex(OSError,"disk failure"):
                        export_acoustic_topology(source,"failed.safetensors",graph)
                self.assertFalse((Path(directory)/"failed.safetensors").exists())
                def claimed(temporary, output):
                    Path(output).write_bytes(b"claimed by another writer")
                    raise FileExistsError("claimed")
                with patch("core.rewiring.export.os.link",side_effect=claimed):
                    with self.assertRaisesRegex(FileExistsError,"claimed"):
                        export_acoustic_topology(source,"racing.safetensors",graph)
                self.assertEqual((Path(directory)/"racing.safetensors").read_bytes(),b"claimed by another writer")
                self.assertFalse(list(Path(directory).glob(".sushi-unet-*")))
            self.assertEqual(source.read_bytes(),original)
        self.assertFalse(torch.cuda.is_initialized())

    def test_music3_flat_export_reuses_companions_and_preserves_payload(self):
        from safetensors.torch import load_file,save_file
        from tests.minimax_music3_flat_dit_fixture import write_tiny_flat_dit_and_official_tree
        with tempfile.TemporaryDirectory() as directory:
            data=write_tiny_flat_dit_and_official_tree(directory)
            config_path=Path(data["official"])/"transformer"/"config.json"
            config=json.loads(config_path.read_text(encoding="utf-8"));config["num_layers"]=7
            config_path.write_text(json.dumps(config),encoding="utf-8")
            source=Path(data["dit_path"])
            state=load_file(str(source))
            first={key:value for key,value in state.items() if ".layers.0." in key}
            for index in range(1,7):
                state.update({key.replace(".layers.0.",f".layers.{index}."):value.clone() for key,value in first.items()})
            save_file(state,str(source),metadata={"model_type":"minimax_music3","user.note":"keep"})
            info=inspect_model(source)
            exported=Path(export_acoustic_topology(source,"unet.safetensors",info["initial_spec"])["model_path"])
            self.assertEqual(payload(exported),payload(source))
            self.assertEqual(inspect_model(exported)["topology"],info["initial_spec"])
            self.assertEqual(inspect_model(exported)["base_config_sha256"],info["base_config_sha256"])
            self.assertIsNone(inspect_model(source)["topology"])
        self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__":
    unittest.main()
