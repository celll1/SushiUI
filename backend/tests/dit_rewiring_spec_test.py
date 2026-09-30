"""Topology artifact, interval and editor-plan contracts (CPU only)."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.rewiring.spec import (
    OPERATORS, META_HASH, META_SPEC, build_spec, checkpoint_metadata,
    preview_plan, read_checkpoint_metadata, validate_spec, validate_model_identity,
)


def spec():
    regions = [dict(id="r0", stream="main", parent=None, start_block=4,
                    end_block=28, down_factor=[2, 2], **OPERATORS),
               dict(id="r1", stream="main", parent="r0", start_block=14,
                    end_block=18, down_factor=[2, 2], **OPERATORS)]
    return build_spec("qwen_image_21", [{"name": "main", "block_count": 32}],
                      regions, {"num_layers": 32, "hidden_size": 4096})


class TopologyContracts(unittest.TestCase):
    def test_artifact_and_grid_roundtrip(self):
        value = spec()
        metadata = checkpoint_metadata(value)
        self.assertEqual(read_checkpoint_metadata(metadata), value)
        plan = preview_plan(value, 64, 64)
        counts = {n: sum(block["tokens"] == n for block in plan["blocks"])
                  for n in (4096, 1024, 256)}
        self.assertEqual(counts, {4096: 8, 1024: 20, 256: 4})
        self.assertEqual([(e["source"], e["target"]) for e in plan["skip_edges"]],
                         [(4, 28), (14, 18)])
        rectangular = preview_plan(value, 64, 96)
        self.assertEqual(rectangular["levels"][1]["tokens"], 384)

    def test_parent_crossing_and_boolean_guards(self):
        mutations = [lambda s: s["regions"][1].update(parent=None),
                     lambda s: s["regions"][1].update(end_block=29),
                     lambda s: s["regions"][1].update(start_block=True),
                     lambda s: s["regions"][1].update(stream=[]),
                     lambda s: s.update(schema_version=True),
                     lambda s: s["regions"][1].update(align_corners=True)]
        for mutate in mutations:
            value = spec(); mutate(value)
            with self.assertRaises(ValueError): validate_spec(value)
        with self.assertRaises(ValueError): preview_plan(spec(), 66, 64)

    def test_tampered_incomplete_and_duplicate_metadata(self):
        metadata = checkpoint_metadata(spec())
        self.assertIsNone(read_checkpoint_metadata({}))
        for broken in ({META_SPEC: metadata[META_SPEC]},
                       {**metadata, META_HASH: "0" * 64},
                       {**metadata, META_SPEC: spec()}):
            with self.assertRaises(ValueError): read_checkpoint_metadata(broken)
        with self.assertRaises(ValueError):
            validate_spec('{"schema_version":1,"schema_version":1}')
        with self.assertRaises(ValueError): validate_spec(" " * 65537)

    def test_native_identity(self):
        value = spec()
        self.assertEqual(validate_model_identity(value, "qwen_image_21", value["streams"],
                                                {"num_layers":32,"hidden_size":4096}), value)
        with self.assertRaises(ValueError):
            validate_model_identity(value, "qwen_image_21", value["streams"], {"num_layers":32})
        with self.assertRaises(ValueError):
            validate_model_identity(value, "krea2", value["streams"], {})


    def test_multiple_stream_preview_and_cross_stream_parent_refusal(self):
        from collections import Counter
        streams = [{"name":"dual", "block_count":7}, {"name":"single", "block_count":11},
                   {"name":"native", "block_count":5}]
        regions = [dict(id="dual_outer", stream="dual", parent=None, start_block=1,
                        end_block=6, down_factor=[2,2], **OPERATORS),
                   dict(id="single_outer", stream="single", parent=None, start_block=2,
                        end_block=9, down_factor=[4,2], **OPERATORS),
                   dict(id="single_inner", stream="single", parent="single_outer", start_block=4,
                        end_block=6, down_factor=[2,2], **OPERATORS)]
        value = build_spec("multistream_fixture", streams, regions, {"streams":streams})
        plan = preview_plan(value,64,96)
        self.assertEqual(read_checkpoint_metadata(checkpoint_metadata(value)),value)
        self.assertEqual(Counter(b["stream"] for b in plan["blocks"]),{"dual":7,"single":11,"native":5})
        self.assertEqual({name:Counter(b["tokens"] for b in plan["blocks"] if b["stream"]==name)
                          for name in ("dual","single","native")},
                         {"dual":{6144:2,1536:5},"single":{6144:4,768:5,192:2},"native":{6144:5}})
        self.assertEqual({(e["stream"],e["source"],e["target"]) for e in plan["skip_edges"]},
                         {("dual",1,6),("single",2,9),("single",4,6)})
        broken = deepcopy(value)
        next(r for r in broken["regions"] if r["id"]=="single_inner")["parent"]="dual_outer"
        with self.assertRaises(ValueError): validate_spec(broken)
        import torch
        from unittest.mock import patch
        with patch.object(torch.cuda,"get_device_capability",return_value=(8,9)), \
             patch.object(torch.cuda,"_lazy_init"), patch.object(torch._C,"_cuda_init"):
            from api.rewiring_routes import router
        import httpx
        from fastapi import FastAPI
        app=FastAPI(); app.include_router(router,prefix="/api/v1")
        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
                reply=await client.post("/api/v1/models/rewiring/validate",json={"spec":value,"token_height":64,"token_width":96})
                self.assertEqual(reply.status_code,200,reply.text)
                self.assertEqual(reply.json()["plan"],plan)
                refusal=await client.post("/api/v1/models/rewiring/validate",json={"spec":broken,"token_height":64,"token_width":96})
                self.assertEqual(refusal.status_code,422,refusal.text)
        asyncio.run(check())
        self.assertFalse(torch.cuda.is_initialized())

    def test_api_roundtrip(self):
        import torch
        from unittest.mock import patch
        with patch.object(torch.cuda, "get_device_capability", return_value=(8,9)), \
             patch.object(torch.cuda, "_lazy_init"), patch.object(torch._C, "_cuda_init"):
            from api.rewiring_routes import router
        import httpx
        from fastapi import FastAPI
        app = FastAPI(); app.include_router(router, prefix="/api/v1")
        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                schema = await client.get("/api/v1/schema/dit-rewiring")
                self.assertEqual(schema.status_code, 200)
                self.assertEqual(schema.json()["runtime_architectures"], ["qwen_image_21", "krea2", "anima", "zimage", "flux2", "lens"])
                self.assertEqual(schema.json()["presets"]["qwen_image_21"]["regions"][1]["end_block"], 18)
                body = dict(spec=spec(), token_height=64, token_width=64)
                valid = await client.post("/api/v1/models/rewiring/validate", json=body)
                self.assertEqual(valid.status_code, 200, valid.text)
                self.assertEqual(valid.json()["spec"], body["spec"])
                body["token_height"] = 66
                self.assertEqual((await client.post("/api/v1/models/rewiring/validate", json=body)).status_code, 422)
                body["unknown"] = True
                self.assertEqual((await client.post("/api/v1/models/rewiring/validate", json=body)).status_code, 422)
        asyncio.run(check())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
