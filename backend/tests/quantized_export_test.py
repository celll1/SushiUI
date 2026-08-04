"""Single-file export of a live weight-only quantized transformer.

Covers the parts that are cheap to state and expensive to get wrong: the prefix
invariant that ties the runtime export to the offline tool, the destination
guards, and a real safetensors round trip of an int8 module.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.models.common import quantized_export as qe
from core.models.ideogram4.vendor.int8_linear import Int8Linear, quantize_weight_to_int8


def _int8_module() -> nn.Module:
    """A two-Linear module with the first layer converted to int8."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 32, bias=True), nn.Linear(32, 16, bias=False))
    src = model[0]
    q, scale = quantize_weight_to_int8(src.weight.detach().float())
    replacement = Int8Linear(64, 32, bias=True, compute_dtype=torch.bfloat16)
    replacement.weight = q.contiguous()
    replacement.weight_scale = scale.to(torch.float32).contiguous()
    replacement.bias = src.bias.detach().to(torch.bfloat16)
    model[0] = replacement
    model[1].to(torch.bfloat16)
    return model


class LayoutInvariantTest(unittest.TestCase):
    def test_offline_and_live_prefixes_compose(self):
        # offline_prefix + source_prefix == the primary module's live prefix for
        # every arch: the offline tool and the runtime export must produce the
        # same on-disk key for the same module path.
        self.assertEqual(qe.check_layout_prefixes(), [])

    def test_anima_writes_the_net_prefix_a_live_module_lacks(self):
        self.assertEqual(qe.primary_live_prefix("anima"), "net.")
        self.assertEqual(qe.primary_live_prefix("krea2"), "transformer.")

    def test_shipped_archs_declare_exactly_one_component(self):
        # krea2 and anima are single-module; the multi-module machinery must not
        # change their shape.
        self.assertEqual(qe.layout_module_specs("krea2"), (("transformer", "transformer."),))
        self.assertEqual(qe.layout_module_specs("anima"), (("transformer", "net."),))

    def test_ideogram4_declares_both_transformers(self):
        # THE multi-component arch: asymmetric CFG runs both branches every step,
        # so an artifact holding one of them is not a smaller model but a broken
        # one. Both must be declared, conditional FIRST (the primary, whose
        # config the metadata builder gets).
        self.assertEqual(
            qe.layout_module_specs("ideogram4"),
            (("transformer", "transformer."),
             ("unconditional_transformer", "unconditional_transformer.")))
        self.assertEqual(qe.primary_live_prefix("ideogram4"), "transformer.")

    def test_source_transform_defaults_to_identity(self):
        # An arch that declares no transform gets the identity one; only an arch
        # whose SOURCE keys are not module paths declares one (flux2: the BFL key
        # remap + fused-qkv split; ideogram4: the fused-qkv split its loader runs
        # BEFORE the quantized swap).
        for arch, layout in qe.EXPORT_LAYOUTS.items():
            if "source_transform" in layout:
                self.assertIsNot(qe.layout_source_transform(arch),
                                 qe.identity_source_transform, arch)
                continue
            self.assertIs(qe.layout_source_transform(arch), qe.identity_source_transform, arch)
        self.assertEqual(qe.identity_source_transform("a.weight", None), (("a.weight", None),))
        self.assertEqual({a for a, l in qe.EXPORT_LAYOUTS.items() if "source_transform" in l},
                         {"flux2", "ideogram4"})

    def test_ideogram4_source_transform_splits_fused_qkv(self):
        transform = qe.layout_source_transform("ideogram4")
        # Key-enumeration pass and per-tensor pass must agree on the key set.
        fused = "layers.7.attention.qkv.weight"
        keys_only = [k for k, _t in transform(fused, None)]
        weight = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
        pairs = transform(fused, weight)
        self.assertEqual(keys_only, [k for k, _t in pairs])
        self.assertEqual(keys_only, ["layers.7.attention.to_q.weight",
                                     "layers.7.attention.to_k.weight",
                                     "layers.7.attention.to_v.weight"])
        # The split is along the OUTPUT rows, in q/k/v order, and loses nothing.
        self.assertTrue(torch.equal(torch.cat([t for _k, t in pairs], dim=0), weight))
        # ``o`` renames; everything else is identity.
        self.assertEqual(transform("layers.7.attention.o.bias", None),
                         (("layers.7.attention.to_out.0.bias", None),))
        self.assertEqual(transform("final_layer.linear.weight", None),
                         (("final_layer.linear.weight", None),))

    def test_ideogram4_source_transform_survives_a_wrapper_prefix(self):
        # A COMBINED single file carries every key under the component prefix,
        # and the transform runs BEFORE prefix stripping by contract. An anchored
        # pattern matched none of those keys: the fused qkv passed through
        # unsplit and 135 of 241 selectable Linears per transformer were silently
        # left unquantized (measured: 105 selected instead of 241).
        transform = qe.layout_source_transform("ideogram4")
        for prefix in ("transformer.", "unconditional_transformer."):
            self.assertEqual(
                [k for k, _t in transform(f"{prefix}layers.7.attention.qkv.weight", None)],
                [f"{prefix}layers.7.attention.to_{p}.weight" for p in "qkv"], prefix)
            self.assertEqual(
                transform(f"{prefix}layers.7.attention.o.weight", None),
                ((f"{prefix}layers.7.attention.to_out.0.weight", None),), prefix)
            self.assertEqual(transform(f"{prefix}input_proj.weight", None),
                             ((f"{prefix}input_proj.weight", None),), prefix)

    def test_multi_component_export_writes_one_file(self):
        # Both prefixes into ONE artifact, read back with both present: two files
        # would not be a single-file export, which is the whole feature.
        modules = {"transformer": _int8_module(),
                   "unconditional_transformer": _int8_module()}
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "ide4.safetensors")
            summary = qe.export_quantized_transformer(
                modules, "ideogram4", out, config={"num_layers": 34})
            self.assertEqual(summary["output_path"], out)
            self.assertEqual([c["component"] for c in summary["components"]],
                             ["transformer", "unconditional_transformer"])
            self.assertEqual(summary["inventory"]["int8"], 2)
            with safe_open(out, framework="pt") as fh:
                keys = set(fh.keys())
                metadata = fh.metadata()
            self.assertIn("transformer.0.weight_scale", keys)
            self.assertIn("unconditional_transformer.0.weight_scale", keys)
            # Same layer count under each prefix: neither branch was dropped.
            self.assertEqual(
                {k[len("transformer."):] for k in keys if k.startswith("transformer.")},
                {k[len("unconditional_transformer."):] for k in keys
                 if k.startswith("unconditional_transformer.")})
            # The loader reads both configs out of the metadata.
            self.assertEqual(json.loads(metadata["transformer_config"]),
                             json.loads(metadata["unconditional_transformer_config"]))

    def test_multi_component_export_refuses_a_bare_module(self):
        # Half a model in a file that claims to be whole is worse than no file.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                qe.export_quantized_transformer(
                    _int8_module(), "ideogram4",
                    os.path.join(tmp, "ide4.safetensors"))
            with self.assertRaises(ValueError):
                qe.export_quantized_transformer(
                    {"transformer": _int8_module()}, "ideogram4",
                    os.path.join(tmp, "ide4.safetensors"))


class Flux2SourceTransformTest(unittest.TestCase):
    """FLUX.2's source transform, against the ``source_transform`` contract."""

    def setUp(self):
        self.transform = qe.layout_source_transform("flux2")

    def test_diffusers_keys_pass_through_untouched(self):
        # The rename table maps ``...modulation_img.lin`` -> ``...linear`` with a
        # plain str.replace, so running it over an ALREADY-diffusers key would
        # produce ``linearear``. It must not run at all for those.
        for key in ("transformer_blocks.0.attn.to_q.weight",
                    "single_transformer_blocks.3.attn.to_out.weight",
                    "double_stream_modulation_img.linear.weight",
                    "norm_out.linear.weight", "x_embedder.weight", "proj_out.weight"):
            self.assertEqual(self.transform(key, None), ((key, None),), key)

    def test_fused_qkv_fans_out_to_three(self):
        weight = torch.arange(36, dtype=torch.float32).reshape(9, 4)
        out = dict(self.transform("double_blocks.0.img_attn.qkv.weight", weight))
        self.assertEqual(sorted(out), [
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_v.weight",
        ])
        torch.testing.assert_close(out["transformer_blocks.0.attn.to_q.weight"], weight[:3])
        torch.testing.assert_close(out["transformer_blocks.0.attn.to_v.weight"], weight[6:])

    def test_key_only_pass_agrees_with_the_tensor_pass(self):
        # THE contract: called with tensor=None it must return the same KEY set
        # it would for a real tensor. The offline tool makes its selection from
        # the key-only pass and writes from the tensor pass.
        keys = [
            "img_in.weight", "txt_in.weight", "time_in.in_layer.weight",
            "final_layer.linear.weight", "final_layer.adaLN_modulation.1.weight",
            "double_stream_modulation_img.lin.weight",
            "single_stream_modulation.lin.weight",
            "double_blocks.0.img_attn.qkv.weight",
            "double_blocks.0.txt_attn.qkv.weight",
            "double_blocks.0.img_attn.norm.query_norm.scale",
            "double_blocks.0.img_mlp.0.weight",
            "single_blocks.0.linear1.weight", "single_blocks.0.linear2.weight",
            "single_blocks.0.norm.key_norm.scale",
        ]
        for key in keys:
            probe = torch.zeros(12, 4)
            key_only = [k for k, _ in self.transform(key, None)]
            with_tensor = [k for k, _ in self.transform(key, probe)]
            self.assertEqual(key_only, with_tensor, key)
            self.assertTrue(all(t is None for _k, t in self.transform(key, None)), key)

    def test_config_pin_counts_blocks_in_both_spellings(self):
        from core.models.flux2 import single_file as f2

        bfl = ["double_blocks.0.img_attn.qkv.weight", "double_blocks.1.img_mlp.0.weight",
               "single_blocks.0.linear1.weight", "single_blocks.7.linear2.weight",
               "img_in.weight"]
        self.assertEqual(f2.count_flux2_blocks(bfl), (2, 2))
        diffusers = ["transformer_blocks.0.attn.to_q.weight",
                     "single_transformer_blocks.0.attn.to_out.weight",
                     "single_transformer_blocks.1.attn.to_out.weight",
                     "x_embedder.weight"]
        # ``single_transformer_blocks.`` must not also be counted as a double
        # block: it contains ``transformer_blocks.`` as a substring.
        self.assertEqual(f2.count_flux2_blocks(diffusers), (1, 2))

    def test_config_pin_refuses_an_unknown_geometry(self):
        from core.models.flux2 import single_file as f2

        keys = [f"double_blocks.{i}.img_attn.qkv.weight" for i in range(8)]
        keys += [f"single_blocks.{i}.linear1.weight" for i in range(48)]
        with self.assertRaises(ValueError) as ctx:
            f2.flux2_config_for_state_dict(keys)
        self.assertIn("8 double block(s) + 48 single block(s)", str(ctx.exception))
        # ...but an explicitly supplied config is honoured verbatim.
        self.assertEqual(f2.flux2_config_for_state_dict(keys, {"num_layers": 8}),
                         {"num_layers": 8})

    def test_adaln_halves_are_swapped(self):
        # BFL stores (shift, scale); diffusers stores (scale, shift).
        weight = torch.cat([torch.zeros(2, 3), torch.ones(2, 3)])
        out = dict(self.transform("final_layer.adaLN_modulation.1.weight", weight))
        self.assertEqual(list(out), ["norm_out.linear.weight"])
        torch.testing.assert_close(out["norm_out.linear.weight"],
                                   torch.cat([torch.ones(2, 3), torch.zeros(2, 3)]))


class MultiModuleLayoutTest(unittest.TestCase):
    """An arch whose layout declares two components writes ONE file with both."""

    ARCH = "test_two_transformers"

    def setUp(self):
        qe.EXPORT_LAYOUTS[self.ARCH] = {
            "modules": (("transformer", "transformer."),
                        ("unconditional_transformer", "unconditional_transformer.")),
            "offline_prefix": "transformer.",
            "source_prefix": "",
            "metadata": lambda cfg: {"model_type": self.ARCH, "format": "pt"},
            "siblings": (),
            "sibling_root": ".",
            "output_subdir": "",
        }

    def tearDown(self):
        qe.EXPORT_LAYOUTS.pop(self.ARCH, None)

    def test_both_prefixes_land_in_one_file(self):
        cond, uncond = _int8_module(), _int8_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "two.safetensors")
            result = qe.export_quantized_transformer(
                {"transformer": cond, "unconditional_transformer": uncond},
                self.ARCH, path)
            # ONE file, not two.
            self.assertTrue(os.path.isfile(result["output_path"]))
            self.assertEqual(
                [f for f in os.listdir(tmp) if f.endswith(".safetensors")],
                ["two.safetensors"])
            # The inventory covers both modules.
            self.assertEqual(result["inventory"]["int8"], 2)
            self.assertEqual(result["inventory"]["plain"], 2)

            with safe_open(result["output_path"], framework="pt", device="cpu") as fh:
                keys = set(fh.keys())
                cond_w = fh.get_tensor("transformer.0.weight")
                uncond_w = fh.get_tensor("unconditional_transformer.0.weight")
            self.assertEqual(
                keys,
                {f"transformer.{k}" for k in cond.state_dict()}
                | {f"unconditional_transformer.{k}" for k in uncond.state_dict()})
            self.assertEqual(cond_w.dtype, torch.int8)
            self.assertEqual(uncond_w.dtype, torch.int8)
            self.assertTrue(torch.equal(cond_w, cond.state_dict()["0.weight"].cpu()))

    def test_a_missing_component_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                qe.export_quantized_transformer(
                    {"transformer": _int8_module()}, self.ARCH,
                    os.path.join(tmp, "x.safetensors"))

    def test_a_bare_module_is_refused_for_a_multi_component_arch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                qe.export_quantized_transformer(
                    _int8_module(), self.ARCH, os.path.join(tmp, "x.safetensors"))


class DestinationGuardTest(unittest.TestCase):
    def test_path_carrying_a_rejected_quant_token_is_refused(self):
        # krea2's loader matches these tokens against the PATH, so an otherwise
        # valid export written there would be unreadable.
        with self.assertRaises(ValueError):
            qe.reject_quant_tokens_in_path(r"D:\exports\nvfp4_test\model.safetensors")
        qe.reject_quant_tokens_in_path(r"D:\exports\anima_int8\model.safetensors")

    def test_unquantized_module_is_refused(self):
        model = nn.Sequential(nn.Linear(8, 8))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                qe.export_quantized_transformer(
                    model, "krea2", os.path.join(tmp, "x.safetensors"))

    def test_unknown_arch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                qe.export_quantized_transformer(
                    _int8_module(), "sdxl", os.path.join(tmp, "x.safetensors"))

    def test_existing_file_is_not_overwritten_by_default(self):
        model = _int8_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.safetensors")
            qe.export_quantized_transformer(model, "krea2", path)
            with self.assertRaises(FileExistsError):
                qe.export_quantized_transformer(model, "krea2", path)
            qe.export_quantized_transformer(model, "krea2", path, overwrite=True)


class RoundTripTest(unittest.TestCase):
    def test_int8_buffers_survive_the_write(self):
        model = _int8_module()
        inventory = qe.quantized_linear_inventory(model)
        self.assertEqual(inventory["int8"], 1)
        self.assertEqual(inventory["plain"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny.safetensors")
            result = qe.export_quantized_transformer(
                model, "krea2", path, audit=None, audit_note="test")
            self.assertTrue(os.path.isfile(result["output_path"]))
            # No audit document supplied -> nothing is fabricated.
            self.assertIsNone(result["audit_path"])
            self.assertEqual(result["metadata"]["quant_audit"], "unavailable")
            self.assertEqual(result["metadata"]["quant_format"], "int8_perrow")

            state = model.state_dict()
            with safe_open(result["output_path"], framework="pt", device="cpu") as fh:
                keys = set(fh.keys())
                tensors = {k: fh.get_tensor(k) for k in keys}
            self.assertEqual(keys, {f"transformer.{k}" for k in state})
            for k, v in state.items():
                out = tensors[f"transformer.{k}"]
                self.assertEqual(out.dtype, v.dtype, k)
                self.assertTrue(torch.equal(out, v.cpu()), k)
            self.assertEqual(tensors["transformer.0.weight"].dtype, torch.int8)
            self.assertEqual(tensors["transformer.0.weight_scale"].dtype, torch.float32)

    def test_tied_tensors_are_cloned_not_dropped(self):
        model = _int8_module()
        # Tie the second Linear's weight to a new buffer under another name.
        model.register_buffer("alias", model[1].weight)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tied.safetensors")
            result = qe.export_quantized_transformer(model, "krea2", path)
            self.assertEqual(len(result["tied_weights_cloned"]), 1)
            with safe_open(result["output_path"], framework="pt", device="cpu") as fh:
                keys = set(fh.keys())
            # Both keys are present: the loaders do not re-tie, so dropping the
            # alias would leave a missing key.
            self.assertIn("transformer.alias", keys)
            self.assertIn("transformer.1.weight", keys)


class JobGuardTest(unittest.TestCase):
    def test_export_inside_the_loaded_model_tree_is_refused(self):
        from api import quantized_export_job as job

        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "model", "krea2.safetensors")
            os.makedirs(os.path.dirname(source), exist_ok=True)
            open(source, "wb").close()

            class FakeManager:
                current_model_info = {"type": "krea2", "source": source}
                krea2_components = {"transformer": _int8_module()}
                _runtime_int8_audit = None

            manager = FakeManager()
            inside = os.path.join(os.path.dirname(source), "export.safetensors")
            with self.assertRaises(job.ExportUnavailableError):
                job.start_export(manager, inside)

            with self.assertRaises(job.ExportUnavailableError):
                job.start_export(manager, os.path.join(tmp, "elsewhere", "export.bin"))

    def test_status_reports_a_non_exportable_model(self):
        from api import quantized_export_job as job

        class FakeManager:
            current_model_info = {"type": "sdxl", "source": "x"}

        status = job.export_status(FakeManager())
        self.assertFalse(status["exportable"])
        self.assertIn("sdxl", status["reason"])
        self.assertEqual(status["job"]["state"], "idle")


if __name__ == "__main__":
    unittest.main()
