"""The strict=False silent-drop guard.

A weight-only quantized checkpoint handed to an architecture without a
quantized-Linear swap must fail loudly, not load "successfully" into a silently
wrong model. Loader-level coverage of the guarded sites (flux2 / lens / minit2i)
needs a real load path and lives outside this suite; here the decision rule
itself is pinned, plus the fact that the call sites exist.
"""

import ast
import os
import sys
import unittest
from pathlib import Path

import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.models.common import quantized_checkpoint_guard as guard


def _plain_sd():
    return {
        "blocks.0.attn.to_q.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
        "blocks.0.attn.to_q.bias": torch.zeros(8, dtype=torch.bfloat16),
        "norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }


class DetectionTest(unittest.TestCase):
    def test_plain_checkpoint_is_not_flagged(self):
        self.assertIsNone(guard.quantized_state_dict_report(_plain_sd()))
        self.assertIsNone(guard.quantized_state_dict_report({}))
        guard.refuse_quantized_state_dict(_plain_sd(), arch="flux2", path="x.safetensors")

    def test_scale_key_is_enough(self):
        sd = _plain_sd()
        sd["blocks.0.attn.to_q.weight_scale"] = torch.ones(8, dtype=torch.float32)
        report = guard.quantized_state_dict_report(sd)
        self.assertIsNotNone(report)
        self.assertEqual(report["scale_keys"], 1)

    def test_quantized_weight_dtype_alone_is_enough(self):
        # A file whose scales were stripped is still unreadable, and casting int8
        # codes into a bf16 parameter is exactly the silent failure.
        sd = _plain_sd()
        sd["blocks.0.attn.to_q.weight"] = torch.zeros(8, 8, dtype=torch.int8)
        report = guard.quantized_state_dict_report(sd)
        self.assertIsNotNone(report)
        self.assertEqual(report["quantized_weight_keys"], 1)

    def test_the_report_splits_float8_from_integer_weights(self):
        sd = _plain_sd()
        sd["blocks.0.attn.to_q.weight"] = torch.zeros(8, 8, dtype=torch.float8_e4m3fn)
        sd["blocks.0.attn.to_k.weight"] = torch.zeros(8, 8, dtype=torch.int8)
        report = guard.quantized_state_dict_report(sd)
        self.assertEqual(report["quantized_weight_keys"], 2)
        self.assertEqual(report["float8_weight_keys"], 1)
        self.assertEqual(report["int_weight_keys"], 1)

    def test_a_pure_float8_cast_is_not_scaled_quantization(self):
        # The dominant ComfyUI "fp8" release shape: every Linear weight cast to
        # e4m3, no scales anywhere. Not a quantization format -- the loaders read
        # it by casting back, exactly -- so it must not be refused.
        sd = _plain_sd()
        sd["blocks.0.attn.to_q.weight"] = torch.zeros(8, 8, dtype=torch.float8_e4m3fn)
        report = guard.quantized_state_dict_report(sd)
        self.assertIsNotNone(report)
        self.assertIsNone(guard.scaled_quantization_report(report, arch="Krea 2"))

    def test_scaleless_integer_weights_stay_refused(self):
        # int8 codes without their per-row scale are not an approximation of the
        # weight; casting them into bf16 measured 103020% error.
        sd = _plain_sd()
        sd["blocks.0.attn.to_q.weight"] = torch.zeros(8, 8, dtype=torch.int8)
        report = guard.quantized_state_dict_report(sd)
        self.assertIsNotNone(guard.scaled_quantization_report(report, arch="Anima"))

    def test_a_float8_file_with_scales_stays_scaled(self):
        sd = _plain_sd()
        sd["blocks.0.attn.to_q.weight"] = torch.zeros(8, 8, dtype=torch.float8_e4m3fn)
        sd["blocks.0.attn.to_q.weight_scale"] = torch.ones(8, 1)
        report = guard.quantized_state_dict_report(sd)
        self.assertIsNotNone(guard.scaled_quantization_report(report, arch="FLUX.2"))

    def test_an_integer_buffer_does_not_trip_it(self):
        # Only ``.weight`` keys are dtype-tested, so an index/mask buffer is safe.
        sd = _plain_sd()
        sd["blocks.0.attn.mask"] = torch.zeros(8, dtype=torch.int8)
        self.assertIsNone(guard.quantized_state_dict_report(sd))

    def test_refusal_names_the_file_and_the_architecture(self):
        sd = _plain_sd()
        sd["blocks.0.attn.to_q.weight_scale"] = torch.ones(8, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as ctx:
            guard.refuse_quantized_state_dict(sd, arch="flux2", path=r"D:\x\model.safetensors")
        message = str(ctx.exception)
        self.assertIn("model.safetensors", message)
        self.assertIn("flux2", message)
        self.assertIn("does not support quantized checkpoints", message)
        # ...and points at the architectures that DO read these files, derived
        # from int8_runtime_quantize rather than written out.
        self.assertIn("Krea 2", message)


class CallSiteTest(unittest.TestCase):
    """The guard has to be CALLED, and before the tolerant load."""

    # flux2 is deliberately NOT here any more: its loader gained the
    # Int8Linear/Fp8Linear swap when FLUX.2 joined RUNTIME_INT8_ARCHS, so it now
    # READS these files. ``SupportedLoaderTest`` below covers it instead, and the
    # two lists are kept complementary by
    # ``test_no_arch_both_refuses_and_supports``.
    SITES = {
        "core/models/lens/lens_loader.py": "lens",
        "core/models/minit2i/vendor/single_file.py": "minit2i",
    }

    def test_each_tolerant_dit_loader_calls_the_guard_first(self):
        for relative, arch in self.SITES.items():
            source = (Path(_BACKEND) / relative).read_text(encoding="utf-8")
            self.assertIn("refuse_quantized_state_dict(", source, relative)
            self.assertIn(f'arch="{arch}"', source, relative)
            tree = ast.parse(source)
            guard_lines, load_lines = [], []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name == "refuse_quantized_state_dict":
                    guard_lines.append(node.lineno)
                elif name == "load_state_dict" and any(
                        kw.arg == "strict" and getattr(kw.value, "value", None) is False
                        for kw in node.keywords):
                    load_lines.append(node.lineno)
            self.assertTrue(guard_lines, relative)
            if load_lines:
                self.assertLess(min(guard_lines), max(load_lines), relative)


class SupportedLoaderTest(unittest.TestCase):
    """An arch that CLAIMS to read quantized checkpoints must actually swap."""

    # arch -> (loader source file, the swap entry point it must call)
    SUPPORTED = {
        "flux2": ("core/model_loader.py", "_swap_flux2_quantized_linears"),
        "anima": ("core/models/anima/anima_loader.py", "_swap_quantized_linears"),
        "krea2": ("core/models/krea2/vendor/single_file.py", "swap_linears_to_int8"),
        "ideogram4": ("core/models/ideogram4/ideogram4_loader.py", "swap_linears_to_fp8"),
        "ltx2": ("core/models/ltx2/loader.py", "_swap_ltx2_quantized_linears"),
        "acestep": ("core/models/acestep/loader.py", "_swap_quantized_linears"),
        "zimage": ("core/model_loader.py", "_swap_zimage_quantized_linears"),
    }

    def test_every_supported_arch_swaps_before_the_tolerant_load(self):
        for arch, (relative, entry) in self.SUPPORTED.items():
            source = (Path(_BACKEND) / relative).read_text(encoding="utf-8")
            self.assertIn(entry, source, f"{arch}: {relative}")

    def test_every_supported_arch_verifies_the_swap_it_made(self):
        # Owning a swap is not enough: the swap helpers require BOTH the scale
        # sibling and the weight dtype while the census fires on either, so a
        # scale-less or path-mismatched quantized file swaps FEWER (or none) and
        # the remainder reaches the tolerant load as plain nn.Linear.
        for arch, (relative, _entry) in self.SUPPORTED.items():
            source = (Path(_BACKEND) / relative).read_text(encoding="utf-8")
            self.assertIn("verify_quantized_swap(", source, f"{arch}: {relative}")

    def test_every_supported_arch_narrows_the_census_first(self):
        # ``verify_quantized_swap`` alone would refuse a PURE FLOAT8 CAST (float8
        # weights, no scales), which every one of these loaders reads correctly
        # by casting back. The narrowing has to happen at the census, before any
        # branch keys off it.
        for arch, (relative, _entry) in self.SUPPORTED.items():
            source = (Path(_BACKEND) / relative).read_text(encoding="utf-8")
            self.assertIn("scaled_quantization_report(", source, f"{arch}: {relative}")

    def test_supported_set_matches_the_shared_tuple(self):
        from core.models.common.int8_runtime_quantize import QUANTIZED_LINEAR_ARCHS

        self.assertEqual(set(self.SUPPORTED), set(QUANTIZED_LINEAR_ARCHS))

    def test_no_arch_both_refuses_and_supports(self):
        # The guard's message is built from QUANTIZED_LINEAR_ARCHS, so an arch
        # that both calls the refusal and owns a swap would advertise itself as
        # supported in the very error that rejects its own file.
        self.assertFalse(set(self.SUPPORTED) & set(CallSiteTest.SITES.values()))


class ScalelessLoadMatrixTest(unittest.TestCase):
    """The four inputs a supporting loader must tell apart, end to end.

    Run against REAL loader entry points on deliberately tiny geometries (~18 k
    parameters for Ideogram 4, ~90 k for Krea 2) -- the decision under test is the
    census-versus-swap comparison, which does not depend on how big the model is.

    (a) an ordinary bf16 checkpoint loads;
    (b) a correctly quantized one loads AND swaps every layer;
    (c) one whose scales were stripped RAISES -- it used to satisfy the census's
        dtype half, swap nothing, and load the int8 codes into bf16 parameters;
    (d) one whose module paths do not match the model the config built RAISES,
        with a different diagnosis (the counts agree; the paths do not).
    (e) a PURE FLOAT8 CAST -- float8 weights, no scales anywhere -- LOADS, with
        no quantized module and no float8 parameter left behind. It is the
        ComfyUI fp8 release shape, it is not a quantization format, and refusing
        it would misdiagnose a legitimate file as (c).
    """

    IDEOGRAM4_CONFIG = dict(
        adaln_dim=32, attention_head_dim=16, in_channels=8, intermediate_size=32,
        llm_features_dim=64, mrope_section=[8, 4, 4], norm_eps=1e-5,
        num_attention_heads=2, num_layers=1, rope_theta=5000000)
    KREA2_CONFIG = dict(
        in_channels=8, num_layers=1, attention_head_dim=8, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=32, timestep_embed_dim=16,
        text_hidden_dim=16, num_text_layers=2, text_num_attention_heads=2,
        text_num_key_value_heads=2, text_intermediate_size=32,
        num_layerwise_text_blocks=1, num_refiner_text_blocks=1,
        axes_dims_rope=[2, 3, 3], rope_theta=1000.0, norm_eps=1e-5)

    @staticmethod
    def _quantize(state_dict, model):
        from core.models.ideogram4.vendor.int8_linear import quantize_weight_to_int8

        linear_weights = {f"{n}.weight" for n, m in model.named_modules()
                          if isinstance(m, torch.nn.Linear)}
        out = {}
        for key, value in state_dict.items():
            if key in linear_weights and value.dim() == 2:
                q, scale = quantize_weight_to_int8(value.float())
                out[key] = q
                out[f"{key[: -len('.weight')]}.weight_scale"] = scale
            else:
                out[key] = value
        return out

    def _matrix(self, reference, build):
        plain = {k: v.clone() for k, v in reference.state_dict().items()}
        quantized = self._quantize(plain, reference)
        n_scales = sum(1 for k in quantized if k.endswith(".weight_scale"))
        self.assertGreater(n_scales, 1)

        # (a) + (b)
        self.assertEqual(self._quantized_module_count(build(dict(plain))), 0)
        self.assertEqual(self._quantized_module_count(build(dict(quantized))), n_scales)

        # (c) scales stripped: nothing can be swapped, and nothing may load.
        scaleless = {k: v for k, v in quantized.items() if not k.endswith(".weight_scale")}
        with self.assertRaises(RuntimeError) as ctx:
            build(scaleless)
        self.assertIn("weight-only QUANTIZED", str(ctx.exception))
        self.assertIn("per-row scale", str(ctx.exception))

        # (d) scales present, one module path renamed: counts agree, paths do not.
        mismatched = dict(quantized)
        victim = next(k for k in quantized if k.endswith(".weight_scale"))
        stem = victim[: -len(".weight_scale")]
        mismatched[f"{stem}_bogus.weight"] = mismatched.pop(f"{stem}.weight")
        mismatched[f"{stem}_bogus.weight_scale"] = mismatched.pop(victim)
        with self.assertRaises(RuntimeError) as ctx:
            build(mismatched)
        self.assertIn("module paths", str(ctx.exception))

        # (e) pure float8 cast: loads, casts back exactly, keeps no float8 param.
        linear_weights = {f"{n}.weight" for n, m in reference.named_modules()
                          if isinstance(m, torch.nn.Linear)}
        cast = {k: (v.to(torch.float8_e4m3fn)
                    if k in linear_weights and v.dim() == 2 else v)
                for k, v in plain.items()}
        model = build(dict(cast))
        self.assertEqual(self._quantized_module_count(model), 0)
        self.assertEqual(self._float8_param_count(model), 0)
        loaded = dict(model.state_dict())
        for key, value in cast.items():
            if value.dtype is torch.float8_e4m3fn and key in loaded:
                self.assertTrue(torch.equal(loaded[key].float(), value.float()), key)

    @staticmethod
    def _quantized_module_count(model):
        return sum(1 for m in model.modules()
                   if type(m).__name__ in ("Int8Linear", "Fp8Linear"))

    @staticmethod
    def _float8_param_count(model):
        return sum(1 for p in model.parameters()
                   if p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2))

    def test_ideogram4(self):
        from core.models.ideogram4.ideogram4_loader import (
            _build_ideogram4_transformer_from_state,
        )
        from core.models.ideogram4.vendor import Ideogram4Transformer2DModel

        reference = Ideogram4Transformer2DModel.from_config(
            self.IDEOGRAM4_CONFIG).to(torch.bfloat16)
        self._matrix(reference, lambda sd: _build_ideogram4_transformer_from_state(
            self.IDEOGRAM4_CONFIG, sd, torch.bfloat16, "test"))

    def test_krea2(self):
        from core.models.krea2.vendor.single_file import build_krea2_transformer
        from core.models.krea2.vendor.transformer import Krea2Transformer2DModel

        reference = Krea2Transformer2DModel.from_config(
            self.KREA2_CONFIG).to(torch.bfloat16)
        self._matrix(reference, lambda sd: build_krea2_transformer(
            sd, self.KREA2_CONFIG, torch.bfloat16))

    def test_anima(self):
        # Anima is the loader whose load is ``assign=True`` (its module is built
        # on the meta device), so case (e) there needs the state dict's float8
        # tensors cast BEFORE the load -- assignment would otherwise install
        # float8 parameters. That is what ``_float8_param_count`` pins.
        from core.models.anima import anima_loader as al

        config = dict(al.ANIMA_DIT_CONFIG)
        config.update(model_channels=64, num_blocks=1, num_heads=2, adaln_lora_dim=16,
                      crossattn_emb_channels=32, max_img_h=32, max_img_w=32, max_frames=4)
        reference = al.Anima(**config).to(torch.bfloat16)

        def build(state_dict):
            original = al.ANIMA_DIT_CONFIG
            al.ANIMA_DIT_CONFIG = config
            try:
                return al.load_anima_dit("<synthetic>", device="cpu",
                                         dtype=torch.bfloat16, state_dict=state_dict)
            finally:
                al.ANIMA_DIT_CONFIG = original

        self._matrix(reference, build)

    # A ~180 k-parameter Z-Image DiT: dim 48, 1 layer, 1 refiner layer, 3 heads,
    # 14 Linears. The real (full-size) geometry is 276 Linears / 6.15 G
    # parameters -- CONFIG-DERIVED, not measured against a checkpoint (see
    # ``ARCH_QUANT_POLICY["zimage"]`` in ``int8_runtime_quantize.py``, which
    # spells out the same caveat: there is NO Z-Image weights file on the
    # machine this was written on, so those numbers come from building the
    # transformer on the META device from the published config, not from a real
    # safetensors header). This fixture is, uniquely among the archs in this
    # file, also the only end-to-end exercise the arch's quantized branch gets.
    # What it does cover is the whole decision under test (census versus swap,
    # and the assign=True float8 cast), through the real code path; what it
    # cannot cover is anything that depends on a real file's key set.
    ZIMAGE_CONFIG = dict(
        all_patch_size=(2,), all_f_patch_size=(1,), in_channels=4, dim=48,
        n_layers=1, n_refiner_layers=1, n_heads=3, n_kv_heads=3, norm_eps=1e-5,
        qk_norm=True, cap_feat_dim=32, rope_theta=256.0, t_scale=1000.0,
        axes_dims=[4, 6, 6], axes_lens=[64, 32, 32])

    def test_zimage(self):
        # Driven through ``_build_zimage_transformer_from_state``, the real
        # build-and-install step of ``load_zimage_from_comfy_safetensors``. The
        # public entry point cannot be called here: it snapshot-downloads a base
        # repo and constructs a VAE, a Qwen3 text encoder, a tokenizer and a
        # scheduler around the transformer, none of which the quantized decision
        # touches.
        #
        # ``layout="official"`` because that is what every writer of a quantized
        # Z-Image artifact emits (the runtime export reads the live module, which
        # is always split-qkv; the offline tool refuses a comfy-layout source).
        # The comfy-layout refusal is asserted separately below.
        from core.model_loader import ModelLoader
        from core.models.zimage_transformer import ZImageTransformer2DModel

        reference = ZImageTransformer2DModel(**self.ZIMAGE_CONFIG).to(torch.bfloat16)
        self._matrix(reference, lambda sd: ModelLoader._build_zimage_transformer_from_state(
            self.ZIMAGE_CONFIG, self.ZIMAGE_CONFIG["n_layers"],
            self.ZIMAGE_CONFIG["in_channels"], sd, "official", torch.bfloat16,
            path="<synthetic>"))

    def test_zimage_refuses_a_quantized_comfy_layout_file(self):
        """A quantized file in the fused-qkv layout must not be rewritten.

        ``_convert_comfy_to_official_state_dict`` row-splits ``attention.qkv``,
        and a per-output-row ``.weight_scale`` sitting next to it would be split
        by the same rule -- producing three plausible-looking projections whose
        scales are wrong for two of them. Nothing legitimate produces this pair,
        so it is refused rather than handled.
        """
        from core.model_loader import ModelLoader
        from core.models.zimage_transformer import ZImageTransformer2DModel

        reference = ZImageTransformer2DModel(**self.ZIMAGE_CONFIG).to(torch.bfloat16)
        quantized = self._quantize(
            {k: v.clone() for k, v in reference.state_dict().items()}, reference)
        with self.assertRaises(RuntimeError) as ctx:
            ModelLoader._build_zimage_transformer_from_state(
                self.ZIMAGE_CONFIG, self.ZIMAGE_CONFIG["n_layers"],
                self.ZIMAGE_CONFIG["in_channels"], quantized, "comfy",
                torch.bfloat16, path="<synthetic>")
        self.assertIn("ComfyUI (fused-qkv) layout", str(ctx.exception))

    # A ~105 k-parameter LTX2VideoTransformer3DModel: 1 layer, 2 heads, 62
    # Linears. The decision under test is the census-versus-swap comparison,
    # which does not depend on geometry -- and the real thing is 18.98 G of
    # Linear parameters in a 37 GB component directory, so it is not a test
    # fixture.
    LTX2_CONFIG = {
        "activation_fn": "gelu-approximate", "attention_bias": True,
        "attention_head_dim": 16, "attention_out_bias": True,
        "audio_attention_head_dim": 8, "audio_cross_attention_dim": 32,
        "audio_cross_attn_mod": True, "audio_gated_attn": True,
        "audio_hop_length": 160, "audio_in_channels": 8,
        "audio_num_attention_heads": 2, "audio_out_channels": 8,
        "audio_patch_size": 1, "audio_patch_size_t": 1,
        "audio_pos_embed_max_pos": 20, "audio_sampling_rate": 16000,
        "audio_scale_factor": 4, "base_height": 64, "base_width": 64,
        "caption_channels": 48, "causal_offset": 1, "cross_attention_dim": 64,
        "cross_attn_mod": True, "cross_attn_timestep_scale_multiplier": 1000,
        "gated_attn": True, "in_channels": 8, "norm_elementwise_affine": False,
        "norm_eps": 1e-06, "num_attention_heads": 2, "num_layers": 1,
        "out_channels": 8, "patch_size": 1, "patch_size_t": 1,
        "perturbed_attn": True, "pos_embed_max_pos": 20,
        "qk_norm": "rms_norm_across_heads", "rope_double_precision": True,
        "rope_theta": 10000.0, "rope_type": "split",
        "timestep_scale_multiplier": 1000, "use_prompt_embeddings": False,
        "vae_scale_factors": [8, 32, 32],
    }

    def test_ltx2(self):
        # Driven through the REAL entry point,
        # ``ltx2.loader._load_quantized_ltx2_transformer``, which is
        # DIRECTORY-shaped rather than state-dict-shaped: LTX-2.3 has no
        # single-file variant, so its quantized branch reads a component
        # directory's headers, decides, and only then materialises anything.
        import json
        import tempfile

        from safetensors.torch import save_file

        from core.models.ltx2.loader import _load_quantized_ltx2_transformer
        from diffusers import LTX2VideoTransformer3DModel

        reference = LTX2VideoTransformer3DModel.from_config(
            self.LTX2_CONFIG).to(torch.bfloat16)

        def build(state_dict):
            with tempfile.TemporaryDirectory() as root:
                component = os.path.join(root, "transformer")
                os.makedirs(component)
                with open(os.path.join(component, "config.json"), "w", encoding="utf-8") as fh:
                    json.dump(self.LTX2_CONFIG, fh)
                save_file({k: v.contiguous() for k, v in state_dict.items()},
                          os.path.join(component, "diffusion_pytorch_model.safetensors"))
                model = _load_quantized_ltx2_transformer(root, torch.bfloat16)
                if model is not None:
                    return model
                # ``None`` means "this is not a scaled quantized file; let
                # from_pretrained load it" -- cases (a) and (e). Reproduce
                # exactly what diffusers then does, so the assertions still
                # describe the production path: build from the config and
                # ``load_state_dict`` WITHOUT assign, which casts into the
                # existing bf16 parameters (that cast is what makes a pure
                # float8 file readable, and it is exact).
                built = LTX2VideoTransformer3DModel.from_config(
                    self.LTX2_CONFIG).to(torch.bfloat16)
                built.load_state_dict(state_dict, strict=False)
                return built

        self._matrix(reference, build)


class CastAfterSwapTest(unittest.TestCase):
    """What a dtype cast AFTER the quantized swap really does, per class.

    The ACE-Step loader casts BEFORE the load for this reason, and its comment
    used to state a mechanism that is false ("Module.to converts the weight
    buffer without complaint", said of both classes). `Module.to(dtype=)` SKIPS
    integral tensors, so the two classes fail differently, and the difference
    decides whether an int8-only file is safe under the old ordering -- it is
    not, only quieter. Pinned here because a future maintainer reading either
    comment will act on it.
    """

    def test_a_later_cast_destroys_an_fp8_weight_and_degrades_both_scales(self):
        from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
        from core.models.ideogram4.vendor.int8_linear import Int8Linear

        int8 = Int8Linear(8, 8, bias=True, compute_dtype=torch.bfloat16)
        fp8 = Fp8Linear(8, 8, bias=True, compute_dtype=torch.bfloat16)
        self.assertIs(int8.weight.dtype, torch.int8)
        self.assertIs(fp8.weight.dtype, torch.float8_e4m3fn)
        self.assertIs(int8.weight_scale.dtype, torch.float32)
        self.assertIs(fp8.weight_scale.dtype, torch.float32)

        int8.to(dtype=torch.bfloat16)
        fp8.to(dtype=torch.bfloat16)

        self.assertIs(
            int8.weight.dtype, torch.int8,
            "an int8 weight survived the cast until now; if torch starts casting "
            "integral buffers, the ACE-Step loader's ordering comment (which says "
            "the int8 weight SURVIVES and only its scale is degraded) is wrong")
        self.assertIs(
            fp8.weight.dtype, torch.bfloat16,
            "the fp8 weight is no longer destroyed by a later cast; the ordering "
            "comment's first bullet needs restating")
        for name, module in (("Int8Linear", int8), ("Fp8Linear", fp8)):
            self.assertIs(
                module.weight_scale.dtype, torch.bfloat16,
                f"{name}: weight_scale was expected to be silently downcast by the "
                f"cast (that is the loss the ordering avoids on BOTH classes)")


class DeclaredSemanticsGuardTest(unittest.TestCase):
    """A LAYOUT-compatible file whose declared MEANING differs must be refused.

    Comfy-Org's ``int8_tensorwise`` distribution stores ``W @ H^T`` -- a
    Hadamard-rotated weight -- as int8 codes, and says so only in a
    ``.comfy_quant`` sidecar. Everything else about the file is the layout this
    repo reads: int8 ``.weight``, float32 ``.weight_scale``. So the census
    fires, the swap takes every layer, and ``verify_quantized_swap``'s three
    counts agree. Reconstructing with the scales alone then yields a rotated
    weight and the forward mixes the input through an orthogonal matrix -- a
    wrong model, not a degraded one.

    THE ANCHOR. The marker bytes below are the literal JSON read out of
    ``Comfy-Org/MiniMax-H3``'s
    ``minimax_h3_fl2va_pruned_int8_convrot.safetensors`` header (see
    ``scratchpad/minimax_h3_weight_formats.md``), spelled as a raw byte string
    rather than built from this repo's own constants: neither the guard nor this
    test owns that spelling, so a guard that drifts to some other key name or
    encoding fails here.

    THE NEGATIVE CONTROL. ``test_an_unmarked_int8_checkpoint_is_unaffected`` and
    ``test_a_convrot_false_marker_is_not_refused`` load and SWAP through the
    real entry points. A guard replaced by a no-op passes those two and fails
    every other test in this class; a guard widened to "refuse anything
    quantized" passes the refusals and fails those two.
    """

    # Read literally from the real file's header. Do not regenerate with
    # json.dumps: the point is that the bytes come from outside this repo.
    CONVROT_MARKER = (
        b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
    )
    PLAIN_MARKER = (
        b'{"format": "int8_tensorwise", "convrot": false}'
    )

    @staticmethod
    def _marker(payload: bytes) -> torch.Tensor:
        return torch.tensor(list(payload), dtype=torch.uint8)

    @staticmethod
    def _int8_sd(scale_shape=(8,)):
        """A file the supported path reads: int8 codes + per-output-row scales."""
        return {
            "fc.weight": torch.zeros(8, 8, dtype=torch.int8),
            "fc.weight_scale": torch.ones(*scale_shape, dtype=torch.float32),
        }

    @staticmethod
    def _module():
        model = torch.nn.Module()
        model.fc = torch.nn.Linear(8, 8, bias=False)
        return model

    # -- negative controls ---------------------------------------------------

    def test_an_unmarked_int8_checkpoint_is_unaffected(self):
        from core.models.ideogram4.vendor.int8_linear import (
            is_int8_state_dict, swap_linears_to_int8,
        )

        sd = self._int8_sd()
        self.assertIsNone(guard.unsupported_quant_semantics_report(sd))
        self.assertIsNotNone(guard.quantized_state_dict_report(sd))
        self.assertTrue(is_int8_state_dict(sd))
        model = self._module()
        self.assertEqual(swap_linears_to_int8(model, sd, torch.bfloat16), 1)
        self.assertEqual(type(model.fc).__name__, "Int8Linear")

    def test_a_convrot_false_marker_is_not_refused(self):
        # A marker is provenance, not by itself a different contract: with
        # ``convrot`` false and a format whose layout this build expresses, the
        # file must take exactly the path it would with no marker at all.
        from core.models.ideogram4.vendor.int8_linear import (
            is_int8_state_dict, swap_linears_to_int8,
        )

        sd = self._int8_sd()
        sd["fc.comfy_quant"] = self._marker(self.PLAIN_MARKER)
        self.assertIsNone(guard.unsupported_quant_semantics_report(sd))
        self.assertTrue(is_int8_state_dict(sd))
        model = self._module()
        self.assertEqual(swap_linears_to_int8(model, sd, torch.bfloat16), 1)

    def test_a_plain_checkpoint_has_no_markers(self):
        self.assertEqual(guard.comfy_quant_markers(_plain_sd()), {})
        self.assertIsNone(guard.unsupported_quant_semantics_report(_plain_sd()))
        guard.refuse_unsupported_quant_semantics(_plain_sd(), arch="flux2")

    # -- the refusal ---------------------------------------------------------

    def test_the_real_convrot_marker_decodes(self):
        parsed = guard.decode_comfy_quant_marker(self._marker(self.CONVROT_MARKER))
        self.assertEqual(parsed, {"format": "int8_tensorwise", "convrot": True,
                                  "convrot_groupsize": 256})

    def test_a_convrot_file_is_refused_by_the_census(self):
        sd = self._int8_sd()
        sd["fc.comfy_quant"] = self._marker(self.CONVROT_MARKER)
        with self.assertRaises(guard.UnsupportedQuantSemanticsError) as ctx:
            guard.quantized_state_dict_report(
                sd, arch="MiniMax-H3", path=r"M:\x\int8_convrot.safetensors")
        message = str(ctx.exception)
        self.assertIn("int8_convrot.safetensors", message)
        self.assertIn("convrot", message)
        self.assertIn("256", message)
        self.assertIn("fc", message)
        self.assertIn("W @ H^T", message)
        self.assertIn(".weight_scale", message)
        # A RuntimeError subclass, so every existing ``except RuntimeError``
        # around a load still catches it.
        self.assertIsInstance(ctx.exception, RuntimeError)

    def test_a_convrot_file_is_refused_by_the_int8_swap_path(self):
        from core.models.ideogram4.vendor.int8_linear import (
            is_int8_state_dict, swap_linears_to_int8,
        )

        sd = self._int8_sd()
        sd["fc.comfy_quant"] = self._marker(self.CONVROT_MARKER)
        with self.assertRaises(guard.UnsupportedQuantSemanticsError):
            is_int8_state_dict(sd)
        model = self._module()
        with self.assertRaises(guard.UnsupportedQuantSemanticsError):
            swap_linears_to_int8(model, sd, torch.bfloat16)
        # Nothing was converted: the refusal is ahead of the first swap.
        self.assertIsInstance(model.fc, torch.nn.Linear)

    def test_the_fp8_swap_path_is_guarded_too(self):
        from core.models.ideogram4.vendor.fp8_linear import (
            is_fp8_state_dict, swap_linears_to_fp8,
        )

        sd = {
            "fc.weight": torch.zeros(8, 8, dtype=torch.float8_e4m3fn),
            "fc.weight_scale": torch.ones(8, dtype=torch.float32),
            "fc.comfy_quant": self._marker(
                b'{"format": "float8_e4m3fn", "convrot": true, "convrot_groupsize": 256}'),
        }
        with self.assertRaises(guard.UnsupportedQuantSemanticsError):
            is_fp8_state_dict(sd)
        with self.assertRaises(guard.UnsupportedQuantSemanticsError):
            swap_linears_to_fp8(self._module(), sd, torch.bfloat16)

    def test_squeezing_the_scale_shape_cannot_defeat_the_guard(self):
        """The obvious "fix" must not turn the guards green.

        Today a Comfy convrot file is stopped only INCIDENTALLY: its scale is
        ``[out, 1]`` and ``Int8Linear`` registers ``(out,)``, so
        ``load_state_dict`` raises a size mismatch. This state dict is the
        post-squeeze one -- ``(out,)``, fully load-compatible, every layout
        guard satisfied -- and it must still be refused, on the marker, before
        any shape is looked at.
        """
        from core.models.ideogram4.vendor.int8_linear import swap_linears_to_int8

        squeezed = self._int8_sd(scale_shape=(8,))
        squeezed["fc.comfy_quant"] = self._marker(self.CONVROT_MARKER)
        # Layout-wise indistinguishable from a supported file...
        without_marker = {k: v for k, v in squeezed.items() if k != "fc.comfy_quant"}
        report = guard.quantized_state_dict_report(without_marker)
        self.assertEqual(report["scale_keys"], 1)
        self.assertEqual(report["quantized_weight_keys"], 1)
        guard.verify_quantized_swap(report, 1, arch="MiniMax-H3")  # passes!
        # ...and refused anyway once the marker is there.
        with self.assertRaises(guard.UnsupportedQuantSemanticsError):
            swap_linears_to_int8(self._module(), squeezed, torch.bfloat16)

    def test_an_unknown_format_is_refused(self):
        sd = self._int8_sd()
        sd["fc.comfy_quant"] = self._marker(b'{"format": "nvfp4_awq"}')
        with self.assertRaises(guard.UnsupportedQuantSemanticsError) as ctx:
            guard.quantized_state_dict_report(sd)
        self.assertIn("nvfp4_awq", str(ctx.exception))

    def test_an_unread_marker_field_is_refused_not_ignored(self):
        sd = self._int8_sd()
        sd["fc.comfy_quant"] = self._marker(
            b'{"format": "int8_tensorwise", "weight_permutation": "interleaved"}')
        with self.assertRaises(guard.UnsupportedQuantSemanticsError) as ctx:
            guard.quantized_state_dict_report(sd)
        self.assertIn("weight_permutation", str(ctx.exception))

    def test_a_garbled_marker_is_unknown_not_absent(self):
        for payload in (b"\xff\xfe not json", b"[1, 2, 3]", b""):
            sd = self._int8_sd()
            sd["fc.comfy_quant"] = self._marker(payload)
            self.assertIsNone(
                guard.decode_comfy_quant_marker(sd["fc.comfy_quant"]), payload)
            with self.assertRaises(guard.UnsupportedQuantSemanticsError, msg=payload):
                guard.quantized_state_dict_report(sd)

    def test_awq_pre_quant_scale_vectors_are_refused(self):
        # ComfyUI multiplies these into the layer's INPUT at runtime; installing
        # the weights without them is as wrong as ignoring a rotation. No marker
        # is needed -- the tensor's presence is the declaration.
        sd = self._int8_sd()
        sd["fc.pre_quant_scale"] = torch.ones(8, dtype=torch.bfloat16)
        with self.assertRaises(guard.UnsupportedQuantSemanticsError) as ctx:
            guard.quantized_state_dict_report(sd)
        self.assertIn("pre_quant_scale", str(ctx.exception))
        self.assertIn("INPUT", str(ctx.exception))

    def test_the_unsupported_archs_refusal_also_runs_it(self):
        # lens / minit2i call ``refuse_quantized_state_dict``; it must not report
        # a convrot file as a merely-quantized one.
        sd = self._int8_sd()
        sd["fc.comfy_quant"] = self._marker(self.CONVROT_MARKER)
        with self.assertRaises(guard.UnsupportedQuantSemanticsError):
            guard.refuse_quantized_state_dict(sd, arch="lens", path="m.safetensors")

    def test_convrot_support_is_not_claimed_anywhere(self):
        # Point 3 of the scope: the guard ships, the un-rotation does not. If a
        # future commit implements it, this test is the reminder to widen
        # KNOWN_COMFY_QUANT_FIELDS' handling deliberately rather than by accident.
        self.assertNotIn("convrot", guard.KNOWN_COMFY_QUANT_FORMATS)
        sd = self._int8_sd()
        sd["fc.comfy_quant"] = self._marker(self.CONVROT_MARKER)
        self.assertEqual(
            guard.unsupported_quant_semantics_report(sd)["convrot_layers"], ["fc"])


if __name__ == "__main__":
    unittest.main()
