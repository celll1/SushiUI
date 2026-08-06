"""Per-layer INT8 / e4m3 selection, shared by the offline tool and the runtime
converter, plus the in-place runtime converter itself.

WHY THIS MODULE EXISTS
----------------------
The decision rule for "which Linear becomes int8, which falls back to e4m3, and
which is not quantized at all" used to live only in
``subapps/fp8_quantize/quantize_transformer_fp8.py``. That was fine while the
only consumer was the offline tool, but SushiUI now also converts an ordinary
bf16 checkpoint IN PLACE at generation time (``quantize_linears_in_place``
below), and two copies of a selection rule drift. Both callers import from here;
the shared import IS the pin. A pinning test on synthetic weights lives at
``tmp/int8_runtime_selection_pin.py``.

THE RULE, in the order it is applied
------------------------------------
1. **Shape filters** (``select_targets``)
   * no weight in the checkpoint (offline only; a live module always has one)
   * ``in_features`` or ``out_features`` not a multiple of the format's GEMM
     alignment (``FORMAT_MIN_ALIGN``: 8 for int8, 16 for fp8). Such a layer can
     never reach the fast path, so quantizing it buys error for no speed.
   * (optional, int8 only) ``in_features < _MIN_WORK_K`` or
     ``out_features < _MIN_WORK_N``: the runtime min-work gate can never admit
     the layer at any ``m``, so it would always run
     ``Int8Linear._dequant_forward`` -- slower than the ``F.linear`` an
     unquantized checkpoint runs. Whether this pays is per-architecture, which
     is what ``ARCH_QUANT_POLICY`` records.
   * user-supplied exclude regexes.
2. **Per-layer format choice** (``audit_and_quantize_int8``)
   * crest pre-filter: mean per-row crest above ``crest_threshold`` -> fallback.
   * MEASURED backstop: both candidate quantizations are always performed and
     both relative RMS weight errors always measured; unless int8 is STRICTLY
     better than e4m3 the layer falls back. This, not the crest, is the actual
     decision -- the crest is the predictive explanation for it.

Both callers emit the same audit document shape (``audit_document``), so a
runtime conversion can be diffed against the committed offline artifact.

NOT HERE: DECLARED-SEMANTICS REFUSAL. Everything in this module operates on a
LIVE ``nn.Module`` whose weights are already ordinary tensors, so a foreign
checkpoint's declared quantization contract (Comfy-Org's ``.comfy_quant``
markers, AWQ ``.pre_quant_scale`` vectors) cannot reach it -- such a file is
refused at LOAD time by
``core.models.common.quantized_checkpoint_guard.refuse_unsupported_quant_semantics``,
which runs inside ``quantized_state_dict_report`` and inside the int8/fp8
detectors and swap entry points. That module's docstring carries the ConvRot
mechanism and why this repo implements the guard and not the rotation.
"""

from __future__ import annotations

import re
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from core.models.ideogram4.vendor.fp8_linear import (
    Fp8Linear,
    quantize_weight_to_fp8,
)
from core.models.ideogram4.vendor.int8_linear import (
    Int8Linear,
    quantize_weight_to_int8,
    weight_crest_factors,
)
# The runtime min-work gate's SHAPE conditions, imported so the offline
# --skip-below-work-gate filter and the runtime converter cannot drift from what
# Int8Linear._int_mm_forward actually enforces. The third condition
# (_MIN_WORK_MKN) depends on m and therefore on the call, not the layer, so it
# has no offline equivalent.
from core.models.ideogram4.vendor.int8_linear import (
    _MIN_WORK_K as INT8_MIN_WORK_K,
    _MIN_WORK_N as INT8_MIN_WORK_N,
)

__all__ = [
    "FORMAT_MIN_ALIGN",
    "DEFAULT_CREST_THRESHOLD",
    "INT8_MIN_WORK_K",
    "INT8_MIN_WORK_N",
    "ARCH_QUANT_POLICY",
    "RUNTIME_INT8_ARCHS",
    "QUANTIZED_LINEAR_ARCHS",
    "ARCH_DISPLAY_NAMES",
    "arch_names",
    "LoraWrappedError",
    "arch_policy",
    "linear_paths",
    "select_targets",
    "audit_and_quantize_int8",
    "audit_document",
    "already_weight_only_quantized",
    "float8_weight_linear_count",
    "lora_wrapped_count",
    "quantize_linears_in_place",
]


class LoraWrappedError(RuntimeError):
    """Raised when the module carries LoRA wrappers; NOTHING was converted.

    Distinct from every other failure of ``quantize_linears_in_place`` because
    its consequence is different: the refusal happens before the first layer is
    touched, so the model is byte-identical afterwards. A caller must not report
    it with the partial-conversion message.
    """


# Per-format GEMM alignment. A layer that cannot satisfy its format's fast-path
# alignment can never reach that path, so quantizing it buys error for no speed.
FORMAT_MIN_ALIGN = {"fp8": 16, "int8": 8}

# Default crest-factor threshold above which int8 loses to e4m3. Derived, not
# tuned: int8's relative error is crest/(127*sqrt(12)) = crest/440 and e4m3's is
# flat at ~2.63e-02, so they cross at crest ~= 11.6.
#
# It is NOT true that the real checkpoint leaves a wide empty gap around 12.0 --
# the full 263-layer Krea 2 run has layers at crest 9.22, 9.43, 12.14, 12.44 and
# 32.56, i.e. two of them sit just above the threshold. What makes the placement
# safe is stronger than a gap: on that run the MEASURED backstop alone
# (``err_int8 < err_e4m3``) reproduces exactly the same 4-layer selection, with
# every chosen int8 layer at an int8-over-e4m3 error advantage >= 1.199 and every
# selected-out layer <= 0.928. The two rules agree, and the measurement -- not the
# crest -- is what actually decides.
DEFAULT_CREST_THRESHOLD = 12.0


# ---------------------------------------------------------------------------
# Per-architecture policy
# ---------------------------------------------------------------------------
#
# The knobs that differ per architecture, in ONE place, so the offline
# invocation and the runtime converter cannot disagree about an arch. Anything
# absent falls back to the format defaults.
#
#   skip_below_work_gate  int8 only. True where the arch has enough Linears
#                         under the runtime min-work gate that quantizing them
#                         is a measured loss.
#   excludes              extra module-path regexes (none today).
#   note                  why the entry reads the way it does.
ARCH_QUANT_POLICY: Dict[str, Dict[str, object]] = {
    "krea2": {
        "skip_below_work_gate": False,
        "excludes": (),
        "note": (
            "Krea 2 has few Linears below the runtime min-work gate, so they are "
            "quantized for the VRAM (time_mod_proj alone is 36864x6144). The "
            "shipped krea2_int8 artifact was produced with the flag off; turning "
            "it on here would stop reproducing it."
        ),
    },
    "flux2": {
        "skip_below_work_gate": False,
        "excludes": (),
        "note": (
            "FLUX.2 Klein 4B has 109 Linears holding 3.8755 G 2-D parameters, and only "
            "THREE of them sit below the runtime min-work gate (k>=2048, n>=1024): "
            "x_embedder (3072x128), proj_out (128x3072) and the timestep embedder's "
            "linear_1 (3072x256), 0.0016 G parameters between them -- 0.04% of the "
            "total. Every attention and MLP projection is 3072-wide or wider, the two "
            "largest being the fused img_attn.qkv (9216x3072) and single_blocks.linear1 "
            "(27648x3072). The filter therefore has almost nothing to filter, so it "
            "stays off (the Krea 2 setting for the Krea 2 reason; Anima's 283 sub-gate "
            "Linears out of 515 are what make it pay there)."
        ),
    },
    "ideogram4": {
        "skip_below_work_gate": True,
        "excludes": (),
        "note": (
            "Ideogram 4 is the largest image architecture here: 279 Linears holding "
            "9.2779 G 2-D parameters PER TRANSFORMER, and it ships two of them (asymmetric "
            "CFG), so 558 Linears and 18.5559 G parameters in one model. Every shape is "
            "8-aligned, so nothing is lost to the GEMM-alignment filter. 38 of the 279 "
            "sit below the runtime min-work gate (k>=2048, n>=1024), and 34 of those are "
            "the SAME shape class Anima's roll-up measured as a net loss: AdaLN modulation "
            "Linears, 512x18432, whose k is 512 and whose m is the batch size, so the gate "
            "can never admit them at any m and they would always run the dequant path -- "
            "slower than the F.linear an unquantized checkpoint runs. They are 13.6% of the "
            "layers but only 3.52% of the parameters (0.3268 G), so filtering them removes "
            "68 per-step dequant calls per model for ~0.33 GB of a 9.3 GB saving. NOTE the "
            "provenance: this is Anima's MEASUREMENT applied to a matching shape class plus "
            "Ideogram 4's own layer census, not a timing run on Ideogram 4 -- the only "
            "local checkpoint is the FP8 one, which cannot be an int8 baseline."
        ),
    },
    "ltx2": {
        "skip_below_work_gate": True,
        "excludes": (),
        "note": (
            "LTX-2.3's DiT is the largest module in this repo: 1660 nn.Linear modules "
            "holding 18.9777 G 2-D parameters, enumerated from LTX2VideoTransformer3DModel "
            "on the meta device (NOT from 'every 2-D tensor in the checkpoint directory', "
            "which totals 34.3396 G and counts the Gemma-3 text encoder's 12.1855 G and the "
            "text connectors' 3.1717 G alongside it -- see the census note in "
            "EXPORT_LAYOUTS['ltx2']). Every shape is 8-aligned, so nothing is lost to the "
            "GEMM-alignment filter. 300 of the 1660 sit below the runtime min-work gate "
            "(k>=2048, n>=1024); 288 of those are the per-attention 'to_gate_logits' "
            "projections, whose out_features is 32, so the gate can never admit them AT ANY "
            "m and they would always run Int8Linear._dequant_forward -- slower than the "
            "F.linear an unquantized checkpoint runs. Unlike Ideogram 4 (3.52% of "
            "parameters) and Anima (~9% of the saving), the filter here costs almost "
            "nothing: those 300 layers hold 0.0362 G, i.e. 0.19% of the DiT's Linear "
            "parameters, so filtering them removes 300 per-step dequant calls for 0.19% of "
            "an 18.94 G saving. NOTE THE PROVENANCE, same caveat Ideogram 4's entry "
            "records: this is LTX-2.3's own SHAPE census plus Anima's MEASUREMENT applied "
            "to a matching shape class (Linears the gate cannot admit at any m). It is NOT "
            "a timing run on LTX-2.3 -- the only local checkpoint is 37 GB of bf16 and a "
            "timing arm would need an int8 one built first."
        ),
    },
    "acestep": {
        "skip_below_work_gate": True,
        "excludes": (),
        "note": (
            "ACE-Step 1.5's DiT is the tidiest architecture here for this work: 392 "
            "nn.Linear modules holding 2.3922 G 2-D parameters, enumerated from "
            "AceStepConditionGenerationModel with accelerate's init_empty_weights (NOT "
            "torch.device('meta'): the vendored ResidualFSQ calls Tensor.item() in its "
            "__init__, which meta tensors refuse). The census is the DiT ALONE and the "
            "checkpoint file agrees exactly -- acestep_v1.5_turbo.safetensors holds 677 "
            "tensors / 2.3939 G, of which 392 / 2.3922 G are 2-D, i.e. every 2-D tensor in "
            "the file is one of those Linear weights and the model has NO non-Linear 2-D "
            "parameter at all (the lyric/text embed_tokens are Linears here, not "
            "nn.Embeddings). The co-shipped Oobleck VAE (0.1687 G, zero 2-D tensors) and "
            "the Qwen3-Embedding text encoders (0.5958 G for the 0.6B tier this DiT "
            "requires, plus unused 1.7B/4B tiers) are separate FILES and separate "
            "components; neither the offline enumeration nor the runtime walk can reach "
            "them, so this arch has none of the whole-directory inflation LTX-2.3's census "
            "note records. Attention is SPLIT q/k/v (2048x2048 and 2048x1024 for the 8 KV "
            "heads), so there is no fused-qkv source transform to write.\n"
            "Two Linears are lost to the GEMM-alignment filter whatever this flag says: "
            "the FSQ audio tokenizer's quantizer.project_in (2048x6) and project_out "
            "(6x2048), whose 6 is the FSQ level count. NINE of the 392 sit below the "
            "runtime min-work gate (k>=2048, n>=1024) -- those two plus the timestep "
            "embedders' two 256x2048 linear_1, text_projector and lyric embed_tokens "
            "(1024x2048), the timbre embed_tokens and audio_acoustic_proj (64x2048) and "
            "proj_out (2048x64) -- and they hold 0.005661 G, 0.237% of the DiT's Linear "
            "parameters. So 99.76% of the parameters are gate-reachable and the filter "
            "costs almost nothing, the LTX-2.3 situation rather than Anima's. NOTE THE "
            "PROVENANCE, the same caveat Ideogram 4's and LTX-2.3's entries record: this is "
            "ACE-Step's own SHAPE census plus Anima's MEASUREMENT applied to a matching "
            "shape class (Linears the gate cannot admit at any m, which therefore always "
            "run Int8Linear._dequant_forward -- slower than the F.linear an unquantized "
            "checkpoint runs). It is NOT a timing run on ACE-Step. One honest difference "
            "from the video/image archs: most of these nine are ONE-SHOT projections "
            "(embedders, text/timbre projections) that run once per generation rather than "
            "once per denoise step, so the calls removed are ~9 per generation, not 9 per "
            "step. The direction of the trade is unchanged -- a layer that can only ever "
            "run the dequant path is a strict loss to quantize -- only its size is."
        ),
    },
    "zimage": {
        "skip_below_work_gate": True,
        "excludes": (),
        "note": (
            "THE CENSUS BEHIND THIS ENTRY IS DERIVED FROM CODE AND CONFIG, NOT MEASURED "
            "FROM A CHECKPOINT. There is no Z-Image weights file on the machine this was "
            "written on, so unlike every other entry in this table (each of which was "
            "cross-checked against a real safetensors header) the numbers below come from "
            "building ``core.models.zimage_transformer.ZImageTransformer2DModel`` on the "
            "META device from the published transformer/config.json (Tongyi-MAI/Z-Image-Turbo, "
            "the file ``load_zimage_from_comfy_safetensors`` itself downloads and reads; the "
            "locally cached copy is byte-equal to the class's own signature defaults: dim 3840, "
            "n_layers 30, n_refiner_layers 2, n_heads = n_kv_heads = 30, cap_feat_dim 2560, "
            "in_channels 16, all_patch_size (2,), all_f_patch_size (1,)).\n"
            "DERIVED census: 276 nn.Linear modules holding 6.1539 G 2-D parameters (521 "
            "state-dict keys). Every shape is 8-aligned, so NOTHING is lost to the "
            "GEMM-alignment filter -- the same situation as Ideogram 4, LTX-2.3 and (except "
            "for its two FSQ projections) ACE-Step. 37 of the 276 sit below the runtime "
            "min-work gate (k>=2048, n>=1024) and hold 0.1278 G, i.e. 13.4% of the layers "
            "for 2.08% of the parameters. THIRTY-TWO of those 37 are the exact shape class "
            "Anima's roll-up measured as a net loss: AdaLN modulation Linears, 256x15360 "
            "(one per transformer block plus the two noise_refiner blocks), whose k is 256 "
            "and whose m is the batch size, so the gate can never admit them at any m and "
            "they would always run Int8Linear._dequant_forward -- slower than the F.linear "
            "an unquantized checkpoint runs, once per block per denoise step. The other five "
            "are one-shot or terminal projections: the FinalLayer modulation (256x3840), the "
            "two timestep-embedder MLP Linears (256x1024 and 1024x256), the patch embedder "
            "all_x_embedder['2-1'] (64x3840) and the final projection "
            "all_final_layer['2-1'].linear (3840x64).\n"
            "So the trade is Ideogram 4's, one notch cheaper: 13.4% of the layers removed "
            "for 2.08% of a 6.15 G saving. PROVENANCE, same caveat Ideogram 4's, LTX-2.3's "
            "and ACE-Step's entries record and one degree weaker: this is Z-Image's own "
            "CONFIG-DERIVED shape census plus Anima's MEASUREMENT applied to a matching "
            "shape class. It is NOT a timing run on Z-Image and it is NOT a header census "
            "of a Z-Image file. WHAT WOULD CHANGE IT: a checkpoint with a different "
            "n_layers (the loader auto-detects it from the ``layers.N`` keys, so a pruned "
            "model is a supported input) moves the 32 AdaLN Linears one-for-one with the "
            "block count and leaves the percentages within a fraction of a point; a 4-channel "
            "(SDXL-VAE) variant, which the loader also supports, changes only the two "
            "embedder shapes, both already below the gate. A GQA variant (n_kv_heads < "
            "n_heads) would shrink to_k/to_v but they stay far above the gate at any "
            "plausible head count. None of these flips the direction of the trade."
        ),
    },
    "anima": {
        "skip_below_work_gate": True,
        "excludes": (),
        "note": (
            "283 of Anima's 515 Linears sit below the runtime min-work gate (168 "
            "AdaLN modulation Linears alone). A Linear-only roll-up put the naive "
            "all-int8 artifact below break-even at 384x384 and behind the filtered "
            "one at every resolution measured, so Anima ships filtered."
        ),
    },
    "minimax_h3": {
        # DEQUANT-ONLY, AND NOT WIRED FOR THE RUNTIME CONVERTER. This entry is
        # here to RECORD that (an arch silently absent from this table would read
        # as an oversight), not to configure a conversion this arch never runs --
        # `minimax_h3` is deliberately NOT in RUNTIME_INT8_ARCHS below.
        #
        # Why there is no runtime conversion. The released generation checkpoint
        # (`minimax_h3_fl2va_pruned_fp8_scaled.safetensors`, 21 GB) is ALREADY
        # weight-only FP8: its loader swaps 300 `nn.Linear` modules for
        # `Fp8Linear` before the first forward, so there is no unquantized
        # transformer for an in-place converter to convert. The `*_pruned_bf16`
        # variant (40 GB) exists upstream and is not downloaded; if it ever
        # becomes the generation file, THAT is the condition under which this
        # entry gains a runtime path, and the note is here so the decision is
        # re-read rather than re-derived.
        #
        # COUNTS, because two different ones are correct and they get confused:
        # the FILE carries 200 quantized tensors (what carries a `.comfy_quant`
        # marker) and the LIVE model holds 300 `Fp8Linear` modules, because the
        # loader splits each fused qkv tensor into three. Both numbers are
        # printed by the loader on every load; quote it rather than deriving
        # either from the other.
        #
        # Why W8A8 is off for the whole architecture. Two measured facts about
        # the file, both in `models/minimax_h3/loader.py::_dit_quantization_policy`:
        # 50 of the 200 quantized tensors -- exactly the 50 `mlp.fc2` -- carry
        # `{"format": "float8_e4m3fn", "full_precision_matrix_mult": true}`, i.e.
        # the writer declares their product must NOT be computed in fp8; and the
        # other 150 carry an `input_scale` that this repo's `Fp8Linear` does not
        # read (it quantizes activations dynamically), so running them through
        # the scaled GEMM would apply a different activation-scaling contract
        # than the file declares. The loader therefore calls `disable_scaled_mm`
        # over the WHOLE DiT at load time, which is the authoritative per-module
        # gate (it outranks the `SUSHI_FP8_SCALED_MM` env flag, the
        # `quantized_gemm_mode` request and grad mode alike). K0.1 verified all
        # four gates structurally, including a negative control proving the fast
        # path IS reachable when it is allowed to be.
        #
        # CONSEQUENCE FOR `quantized_gemm_mode`, stated so it is not a surprise:
        # this architecture accepts the parameter (it is in
        # QUANTIZED_LINEAR_ARCHS, because it really does own quantized Linears),
        # and `"w8a8"` resolves to the dequantized matmul anyway. That is
        # reported, not silent: `report_quantized_gemm_outcome` reads the
        # RESOLVED path out of `extract_fp8_gemm_info` and files a
        # `quantization_fallback` warning on the generation.
        "skip_below_work_gate": False,
        "excludes": (),
        "note": (
            "MiniMax-H3's released DiT ships weight-only FP8 (200 quantized tensors in the "
            "file; 300 Fp8Linear modules once the loader has split the fused qkv), so no "
            "runtime int8 conversion is registered for it -- there is no "
            "unquantized transformer to convert. Its quantization policy is DEQUANT-ONLY and "
            "is enforced at load time by disable_scaled_mm over the whole DiT, because 50 of "
            "the 200 tensors are marked full_precision_matrix_mult and the other 150 carry an "
            "input_scale this repo's Fp8Linear does not read. A runtime path becomes "
            "relevant only if the *_pruned_bf16 variant ever becomes the generation file."
        ),
    },
}

# Architectures the RUNTIME converter is wired for. A superset entry in
# ARCH_QUANT_POLICY (offline-only arch) would simply not be listed here.
#
# THIS TUPLE IS THE SINGLE SOURCE OF TRUTH for "which archs accept
# unet_quantization='int8'". Every other place that used to spell the set out
# now reads it from here:
#   * backend/core/vram_optimization.py  -- the refusal warning's prose
#     (``arch_names`` below renders it), and the arch gate itself;
#   * backend/api/arch_capabilities.py   -- re-exported and served by
#     GET /schema/arch-capabilities as ``runtime_int8_archs``;
#   * frontend/src/utils/api.ts          -- reads that field instead of its own
#     hardcoded list.
# Adding an arch here is therefore the whole rollout switch on the UI side.
#
# `minimax_h3` IS DELIBERATELY ABSENT and that absence is a decision, not a gap:
# its released generation checkpoint is already weight-only FP8, so there is no
# unquantized transformer for the in-place converter to act on. See its
# ARCH_QUANT_POLICY entry above for the full reason and for the one condition
# that would change it (the *_pruned_bf16 variant becoming the generation file).
# It IS in QUANTIZED_LINEAR_ARCHS below, because its loader really does swap in
# the quantized Linear classes.
RUNTIME_INT8_ARCHS = ("anima", "krea2", "flux2", "ideogram4", "ltx2", "acestep",
                      "zimage")

# Architectures whose LOADERS swap in the weight-only quantized Linear classes
# (``Fp8Linear`` / ``Int8Linear``), i.e. the archs where a quantized-GEMM path
# exists to select at all. Kept as a SUPERSET expression rather than collapsed
# into the tuple above: it was a strict superset while Ideogram 4 had no in-place
# runtime conversion (its checkpoints ship FP8/nf4 quantized, so it owned
# quantized Linears all the same), the two sets coincide today, and the next arch
# whose loader reads a quantized file before its runtime path is wired will make
# them differ again. Consumed by ``backend/api/quantized_gemm.py``.
#
# ``minimax_h3`` is that next arch: its loader swaps 200 ``nn.Linear`` for
# ``Fp8Linear`` from the released `*_pruned_fp8_scaled` file, and it has no
# runtime int8 path (see RUNTIME_INT8_ARCHS above). So the two sets are a strict
# superset relation again, which is why this stays an expression.
QUANTIZED_LINEAR_ARCHS = tuple(sorted({"ideogram4", "minimax_h3", *RUNTIME_INT8_ARCHS}))

# Display spelling of an arch id, for user-facing prose only. Every arch that
# can appear in either tuple above needs an entry; ``arch_names`` falls back to
# the raw id so a missing one degrades to "anima" rather than raising.
ARCH_DISPLAY_NAMES: Dict[str, str] = {
    "anima": "Anima",
    "krea2": "Krea 2",
    "ideogram4": "Ideogram 4",
    "flux2": "FLUX.2",
    "zimage": "Z-Image",
    "lens": "Lens",
    "minit2i": "MiniT2I",
    "sd15": "SD1.5",
    "sdxl": "SDXL",
    "ltx2": "LTX-2.3",
    "acestep": "ACE-Step",
    "minimax_h3": "MiniMax H3",
}


def arch_names(archs: Sequence[str]) -> str:
    """Render an arch tuple as English prose: ``"Anima and Krea 2"``.

    Exists so a warning message cannot name a different set than the code that
    enforces it; the whole point of the tuples above is that adding an arch
    updates the message too.
    """
    names = [ARCH_DISPLAY_NAMES.get(a, a) for a in archs]
    if not names:
        return "no architecture"
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def arch_policy(arch: Optional[str], fmt: str = "int8") -> Dict[str, object]:
    """Resolved selection knobs for ``arch``/``fmt``.

    Returns ``{"min_align", "skip_below_work_gate", "excludes", "note"}``.
    ``skip_below_work_gate`` is forced False for any format other than int8:
    its two constants are ``int8_linear``'s runtime gate and ``fp8_linear`` has
    no ``_MIN_WORK_*`` at all, so applying them to an e4m3 conversion would
    filter it against a rule that governs nothing it will ever run.
    """
    entry = ARCH_QUANT_POLICY.get(arch or "", {})
    return {
        "min_align": FORMAT_MIN_ALIGN.get(fmt, 0),
        "skip_below_work_gate": bool(entry.get("skip_below_work_gate", False)) if fmt == "int8" else False,
        "excludes": tuple(entry.get("excludes", ()) or ()),
        "note": entry.get("note", ""),
    }


# ---------------------------------------------------------------------------
# Linear enumeration + shape selection
# ---------------------------------------------------------------------------

def linear_paths(model: nn.Module) -> Dict[str, Tuple[int, int]]:
    """{module path: (in_features, out_features)} for every ``nn.Linear``."""
    return {
        name: (m.in_features, m.out_features)
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
    }


def select_targets(
    linears: Dict[str, Tuple[int, int]],
    present_keys: set,
    min_align: int,
    excludes: Sequence[re.Pattern],
    skip_below_work_gate: bool = False,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Split the Linears into (quantize, [(skipped, reason)]).

    ``present_keys`` holds module paths ALREADY stripped of an arch's
    ``source_prefix``, so it is directly comparable with the model's paths. The
    runtime converter passes the live module's own paths, for which the check is
    trivially true; it is kept in the shared body so both callers run the same
    function rather than two similar ones.

    ``skip_below_work_gate`` is applied verbatim if set; the INT8-only scoping
    lives in the callers (``arch_policy`` for the runtime path, ``main`` for the
    CLI), the same place the other int8-only selectors are scoped.
    """
    targets: List[str] = []
    skipped: List[Tuple[str, str]] = []
    for name, (in_f, out_f) in sorted(linears.items()):
        if f"{name}.weight" not in present_keys:
            skipped.append((name, "no weight in checkpoint"))
            continue
        if min_align and (in_f % min_align or out_f % min_align):
            skipped.append((name, f"unaligned {in_f}x{out_f} (cannot reach the fast GEMM path)"))
            continue
        if skip_below_work_gate and (in_f < INT8_MIN_WORK_K or out_f < INT8_MIN_WORK_N):
            skipped.append((
                name,
                f"{in_f}x{out_f} below the runtime min-work gate "
                f"(k>={INT8_MIN_WORK_K}, n>={INT8_MIN_WORK_N}) at any m: it would always "
                f"run the dequant path, which is slower than the unquantized F.linear"))
            continue
        pattern = next((p for p in excludes if p.search(name)), None)
        if pattern is not None:
            skipped.append((name, f"excluded by /{pattern.pattern}/"))
            continue
        targets.append(name)
    return targets, skipped


# ---------------------------------------------------------------------------
# Per-layer format selection + audit (int8 only)
# ---------------------------------------------------------------------------

def _rel_rms(reference: torch.Tensor, approx: torch.Tensor) -> float:
    """Relative RMS error of ``approx`` against ``reference``, in float32."""
    ref = reference.to(torch.float32)
    err = approx.to(torch.float32) - ref
    denom = ref.pow(2).mean().sqrt()
    if not torch.isfinite(denom) or denom == 0:
        return float("nan")
    return float(err.pow(2).mean().sqrt() / denom)


def audit_and_quantize_int8(
    name: str,
    tensor: torch.Tensor,
    crest_threshold: float,
    fallback: str,
) -> Tuple[str, torch.Tensor, Optional[torch.Tensor], Dict]:
    """Choose int8 / e4m3 / bf16 for one Linear weight and return the audit row.

    BOTH candidate quantizations are always performed and both errors always
    measured, whatever the crest says. That costs one extra pass over a weight
    that is already resident and makes the audit table a record of what was
    actually true rather than of what the heuristic predicted.

    Returns ``(chosen_format, weight, scale_or_None, audit_row)``.
    """
    crest = weight_crest_factors(tensor)
    crest_mean = float(crest.mean())
    crest_p99 = float(crest.quantile(0.99)) if crest.numel() > 1 else crest_mean
    crest_max = float(crest.amax())

    q_i8, s_i8 = quantize_weight_to_int8(tensor)
    q_f8, s_f8 = quantize_weight_to_fp8(tensor)
    err_i8 = _rel_rms(tensor, q_i8.to(torch.float32) * s_i8.unsqueeze(1))
    err_f8 = _rel_rms(tensor, q_f8.to(torch.float32) * s_f8.unsqueeze(1))

    # Two independent reasons to select a layer out. The crest rule is the
    # documented, predictive one; the measured comparison is the backstop for a
    # weight whose distribution the crest model does not describe (it cannot,
    # for instance, see a bimodal row). Either one is sufficient.
    if crest_mean > crest_threshold:
        reason = f"crest {crest_mean:.2f} > {crest_threshold:.2f}"
        chosen = fallback
    elif not (err_i8 < err_f8):
        # Also catches NaN errors (a degenerate all-zero or non-finite weight):
        # `not (a < b)` is False only when int8 is strictly better.
        reason = f"measured int8 {err_i8:.5f} not better than e4m3 {err_f8:.5f}"
        chosen = fallback
    else:
        reason = f"int8 {err_i8:.5f} < e4m3 {err_f8:.5f} at crest {crest_mean:.2f}"
        chosen = "int8"

    row = {
        "name": name,
        "shape": list(tensor.shape),
        "int8_rel_rms": err_i8,
        "e4m3_rel_rms": err_f8,
        "advantage_int8_over_e4m3": (err_f8 / err_i8) if err_i8 else float("inf"),
        "crest_mean": crest_mean,
        "crest_p99": crest_p99,
        "crest_max": crest_max,
        "chosen": chosen,
        "reason": reason,
    }
    if chosen == "int8":
        return chosen, q_i8, s_i8, row
    if chosen == "e4m3":
        return chosen, q_f8, s_f8, row
    return "bf16", tensor, None, row


def audit_document(rows: List[Dict], settings: Dict) -> Dict:
    """The audit JSON body, identical in shape for the offline and runtime paths.

    ``{"settings", "format_counts", "geomean_advantage", "layers"}``. The offline
    tool writes this to ``<stem>.int8_audit.json``; the runtime converter returns
    it so a conversion can be diffed against a committed artifact.
    """
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["chosen"]] = counts.get(r["chosen"], 0) + 1
    finite = [r["advantage_int8_over_e4m3"] for r in rows
              if r["advantage_int8_over_e4m3"] not in (float("inf"),)
              and r["advantage_int8_over_e4m3"] == r["advantage_int8_over_e4m3"]]
    geomean = None
    if finite:
        geomean = float(torch.tensor(finite, dtype=torch.float64).log().mean().exp())
    return {
        "settings": settings,
        "format_counts": counts,
        "geomean_advantage": geomean,
        "layers": rows,
    }


# ---------------------------------------------------------------------------
# In-place runtime conversion
# ---------------------------------------------------------------------------

def already_weight_only_quantized(model: nn.Module) -> int:
    """Count ``Int8Linear`` / ``Fp8Linear`` modules under ``model``.

    Non-zero means the module already owns weight-only quantized Linears --
    either from an offline-quantized checkpoint or from a previous in-place
    runtime conversion. Detection is by module type, not by weight dtype, so it
    cannot be confused by a module that merely happens to store float8.
    """
    return sum(1 for m in model.modules() if isinstance(m, (Int8Linear, Fp8Linear)))


_FLOAT8_DTYPES = tuple(
    getattr(torch, _n) for _n in
    ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if hasattr(torch, _n)
)


def float8_weight_linear_count(model: nn.Module) -> int:
    """Count plain ``nn.Linear`` modules whose weight ALREADY holds float8.

    The type-based ``already_weight_only_quantized`` above cannot see these: the
    legacy runtime FP8 path (``vram_optimization._anima_patch_linear_fp8``) casts
    ``linear.weight.data`` to e4m3/e5m2 and monkeypatches ``forward``, leaving the
    module an ``nn.Linear``. Quantizing such a weight to int8 would quantize an
    ALREADY-rounded weight -- measured at 0.04400 relative RMS on Anima against
    0.00394 for a direct int8 conversion of the same checkpoint (11.2x), i.e.
    worse than either format alone. The dtype check is the missing half of the
    type check, and the converter's callers refuse on it.
    """
    if not _FLOAT8_DTYPES:
        return 0
    return sum(
        1 for m in model.modules()
        if isinstance(m, nn.Linear) and m.weight is not None
        and m.weight.dtype in _FLOAT8_DTYPES
    )


def lora_wrapped_count(model: nn.Module) -> int:
    """Count LoRA wrappers under ``model`` (by class name, no import needed).

    A wrapped Linear is no longer an ``nn.Linear``, so converting a LoRA'd module
    would silently skip every wrapped layer and select a DIFFERENT set than the
    offline audit. The converter refuses instead.

    Public because a multi-component caller must be able to ask BEFORE it starts:
    ``quantize_linears_in_place`` refuses per module, and discovering the refusal
    on the second transformer would already have converted the first.
    """
    return sum(1 for m in model.modules() if type(m).__name__ == "LoRALinearLayer")


# Historical private spelling, kept because it is the name the refusal below
# reads by.
_lora_wrapped_count = lora_wrapped_count


def _resolve_parent(root: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    """``(parent module, attribute name)`` for a dotted module path under root."""
    if "." not in dotted:
        return root, dotted
    parent_path, attr = dotted.rsplit(".", 1)
    return root.get_submodule(parent_path), attr


def _filled_quantized_linear(
    src: nn.Linear,
    chosen: str,
    q: torch.Tensor,
    scale: torch.Tensor,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """Build an ``Int8Linear``/``Fp8Linear`` already holding ``q``/``scale``.

    The sibling ``swap_linears_to_*`` helpers are LOADER-shaped: they gate on a
    state dict and construct an EMPTY module for ``load_state_dict`` to fill.
    This is the live-module constructor -- it takes the source ``nn.Linear`` and
    the quantized tensors and assigns the buffers directly, so no second copy of
    the model ever exists.
    """
    cls = Int8Linear if chosen == "int8" else Fp8Linear
    mod = cls(
        src.in_features,
        src.out_features,
        bias=src.bias is not None,
        compute_dtype=compute_dtype,
    )
    mod.weight = q.contiguous().to(device)
    mod.weight_scale = scale.to(torch.float32).contiguous().to(device)
    if src.bias is not None:
        mod.bias = src.bias.detach().to(compute_dtype).to(device)
    return mod


def quantize_linears_in_place(
    model: nn.Module,
    *,
    arch: Optional[str] = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    work_device: Optional[torch.device] = None,
    crest_threshold: float = DEFAULT_CREST_THRESHOLD,
    fallback: str = "e4m3",
    min_align: Optional[int] = None,
    excludes: Iterable[str] = (),
    skip_below_work_gate: Optional[bool] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    label: str = "transformer",
) -> Dict:
    """Convert every selected ``nn.Linear`` under ``model`` to int8/e4m3 IN PLACE.

    MIXED by construction, exactly like the offline tool: a layer the measured
    backstop rejects becomes an ``Fp8Linear`` rather than being left alone (1
    layer on Krea 2, 4 on Anima).

    MEMORY, measured, not asserted. NO SECOND COPY OF THE MODULE IS BUILT: each
    source weight is dropped as its replacement is installed, and the module's own
    parameter bytes fall (Anima 3.895 -> 2.327 GB, Krea 2 23.879 -> 11.948 GB).
    PROCESS RSS is a different quantity and does NOT fall: on real Anima, RSS goes
    0.958 GB after load -> 6.159 GB peak -> 6.159 GB steady after gc, against
    2.327 GB of resulting module bytes. The safetensors mapping of the SOURCE
    checkpoint stays resident because the layers this selection skips (283 of
    Anima's 515 Linears) and every non-Linear parameter still reference it, so
    steady-state host memory is roughly source + quantized module ~= 1.6x the
    source, held until the model is reloaded. Budget host RAM accordingly: ~6 GB
    for Anima, ~36 GB for a 24 GB bf16 Krea 2 transformer. It is still far below
    the SD1.5/SDXL-style ``copy.deepcopy`` + retained second CPU copy, which is
    what makes it viable at Krea 2 scale at all.

    DEVICE. The quantization math runs on ``work_device`` (CUDA when available,
    else wherever the weight lives) and each result is placed back on the
    weight's ORIGINAL device, so this is safe to call with the module on CPU
    (before staging, which is what both callers do) or on GPU. The math itself
    materialises the weight in float32 (``quantize_weight_to_int8``), which is
    why it is done per layer and freed immediately. A layer whose float32 working
    set does not fit on ``work_device`` is retried on the weight's own device
    rather than aborting the conversion; the fallbacks are listed in the returned
    document under ``oom_fallback_layers``.

    ONE-WAY. There is no inverse: the source bf16 weights are dropped. The model
    stays quantized until it is reloaded.

    RESUMABLE. Selection walks ``nn.Linear`` only and a converted layer is no
    longer one, so re-running after a failure converts exactly the layers that
    are still unconverted. On failure the exception carries the partial audit
    document as ``_int8_partial_document`` so the caller can report and later
    merge it.

    Returns the audit document (``audit_document``) with the extra keys
    ``elapsed_s``, ``converted`` and ``oom_fallback_layers``.
    """
    if _lora_wrapped_count(model):
        raise LoraWrappedError(
            f"refusing to quantize a LoRA-wrapped {label}: the wrappers hide the "
            f"underlying Linears, so the selection would silently differ from the "
            f"offline audit. Convert before applying LoRAs."
        )

    policy = arch_policy(arch, "int8")
    if min_align is None:
        min_align = int(policy["min_align"])
    if skip_below_work_gate is None:
        skip_below_work_gate = bool(policy["skip_below_work_gate"])
    patterns = [re.compile(p) for p in (tuple(excludes) + tuple(policy["excludes"]))]

    linears = linear_paths(model)
    present = {f"{name}.weight" for name in linears}
    targets, skipped = select_targets(
        linears, present, min_align, patterns, skip_below_work_gate=skip_below_work_gate)

    print(f"[RuntimeInt8] {label}: {len(linears)} nn.Linear module(s); "
          f"converting {len(targets)}, skipping {len(skipped)} "
          f"(arch={arch}, min_align={min_align}, skip_below_work_gate={skip_below_work_gate})")

    rows: List[Dict] = []
    counts = {"int8": 0, "e4m3": 0, "bf16": 0}
    oom_fallbacks: List[str] = []
    total = len(targets)
    t0 = time.perf_counter()

    def _document() -> Dict:
        doc = audit_document(rows, {
            "arch": arch,
            "format": "int8",
            "mode": "runtime_in_place",
            "min_align": min_align,
            "skip_below_work_gate": skip_below_work_gate,
            "min_work_k": INT8_MIN_WORK_K,
            "min_work_n": INT8_MIN_WORK_N,
            "crest_threshold": crest_threshold,
            "fallback": fallback,
            "compute_dtype": str(compute_dtype),
            "skipped": [{"name": n, "reason": r} for n, r in skipped],
        })
        doc["elapsed_s"] = time.perf_counter() - t0
        doc["converted"] = dict(counts)
        doc["oom_fallback_layers"] = list(oom_fallbacks)
        return doc

    for i, name in enumerate(targets):
        try:
            parent, attr = _resolve_parent(model, name)
            src = getattr(parent, attr)
            if not isinstance(src, nn.Linear):
                # Only reachable if the module tree changed under us.
                continue
            weight = src.weight.detach()
            orig_device = weight.device
            compute_on = work_device if work_device is not None else orig_device
            staged = weight.to(compute_on) if compute_on != orig_device else weight
            try:
                chosen, q, scale, row = audit_and_quantize_int8(
                    name, staged, crest_threshold, fallback)
            except torch.cuda.OutOfMemoryError:
                # This layer's float32 working set did not fit on the work
                # device. Retrying it on the weight's OWN device costs time but
                # keeps the conversion whole -- an abort here is precisely what
                # leaves a half-quantized module behind.
                if compute_on == orig_device:
                    raise
                del staged
                torch.cuda.empty_cache()
                oom_fallbacks.append(name)
                print(f"[RuntimeInt8] {label}: CUDA OOM quantizing {name} "
                      f"({tuple(weight.shape)}); retrying on {orig_device}")
                staged = weight
                chosen, q, scale, row = audit_and_quantize_int8(
                    name, staged, crest_threshold, fallback)
            if chosen == "bf16":
                # fallback="bf16": leave the source Linear untouched.
                rows.append(row)
                counts[chosen] = counts.get(chosen, 0) + 1
                del staged, q
                continue
            setattr(parent, attr, _filled_quantized_linear(
                src, chosen, q, scale, compute_dtype, orig_device))
            rows.append(row)
            counts[chosen] = counts.get(chosen, 0) + 1
            del src, weight, staged, q, scale
        except Exception as err:
            # The module is now PARTIALLY converted. Hand the caller everything
            # measured so far, plus where it stopped, so the failure can be
            # reported accurately and the remaining layers resumed later (the
            # selection walks nn.Linear, and the layers already replaced are no
            # longer nn.Linear, so a re-run picks up exactly the remainder).
            doc = _document()
            doc["partial"] = True
            doc["failed_layer"] = name
            doc["converted_before_failure"] = counts["int8"] + counts["e4m3"]
            doc["remaining"] = total - (counts["int8"] + counts["e4m3"] + counts["bf16"])
            try:
                setattr(err, "_int8_partial_document", doc)
            except Exception:
                pass
            print(f"[RuntimeInt8] {label}: FAILED at {name} after converting "
                  f"{doc['converted_before_failure']} of {total} layer(s): {err}")
            raise
        if progress_cb is not None:
            try:
                progress_cb(i + 1, total, name)
            except Exception:
                pass

    doc = _document()
    elapsed = doc["elapsed_s"]
    print(f"[RuntimeInt8] {label}: {counts.get('int8', 0)} int8 + "
          f"{counts.get('e4m3', 0)} e4m3 Linear(s) in {elapsed:.1f}s "
          f"(skipped {len(skipped)}"
          + (f", {len(oom_fallbacks)} OOM fallback(s)" if oom_fallbacks else "") + ")")
    return doc
