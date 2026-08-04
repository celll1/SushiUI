#!/usr/bin/env python3
"""Quantize a transformer checkpoint to the repo's weight-only FP8 or INT8 layout.

Produces a checkpoint that the NORMAL production loader path accepts: both Linear
layouts are exactly the ones ``backend/core/models/ideogram4/vendor/fp8_linear.py``
and ``int8_linear.py`` define and their ``swap_linears_to_*`` helpers gate on, so
no loader change is needed.

    <name>.weight        float8_e4m3fn  (out, in)   [--format fp8]
    <name>.weight        int8           (out, in)   [--format int8]
    <name>.weight_scale  float32        (out,)      <- presence gates the swap
    <name>.bias          original dtype (out,)      [untouched]

The two formats share the ``.weight_scale`` suffix; the WEIGHT DTYPE is what
tells them apart, and both loaders key on it. That is deliberate -- ``--format
int8`` produces a MIXED checkpoint (see below) in which some layers are int8 and
some are e4m3, and a single suffix convention lets one load pass serve both.

PER-LAYER FORMAT SELECTION (``--format int8``)
----------------------------------------------
int8 spends 254 uniform levels across each output row's range, so its relative
error scales with the row's CREST FACTOR (row amax / row RMS): a uniform rounding
error of ``amax/127`` has RMS ``amax/(127*sqrt(12))``, i.e. ``crest/440`` relative
to the row. e4m3 instead spends a floating exponent per element and sits flat at
~2.63e-02 whatever the distribution. Setting the two equal gives a break-even
crest of ~11.6, which is where ``--crest-threshold`` defaults (12.0).

Measured on the full 263-layer Krea 2 transformer conversion
(``krea2_int8.int8_audit.json``): mean per-row crest is 4.5-6 for typical layers,
7-9 for the marginal ones, and 12.14 / 12.44 / 32.56 for three -- so the
threshold does NOT sit in an empty gap; two layers land just above it.

What makes the placement safe is that the two rules AGREE on that checkpoint. The
crest rule is the documented, predictive one, but the measured per-layer error of
both formats is computed anyway for the audit table and any layer whose int8
error exceeds its e4m3 error is selected out regardless of crest -- and on the
real run that measured backstop ALONE reproduces the identical 4-layer selection:
every layer kept in int8 has an int8-over-e4m3 error advantage of at least 1.199,
every layer selected out at most 0.928, with nothing in between. The crest is the
explanation; the measurement is the decision.

Selected-out layers fall back to e4m3 (``--fallback e4m3``, the default: keeps the
VRAM saving, and with the FP8 W8A8 toggle off -- which is the default -- they run
the dequantized matmul, i.e. the highest-quality path available) or to the source
dtype (``--fallback bf16``: no quantization error at all, at full weight size).

AUDIT TABLE
-----------
``--format int8`` ALWAYS writes ``<output stem>.int8_audit.json`` next to the
output and prints a summary: per layer, the measured int8 and e4m3 relative RMS
weight error, the mean/p99/max per-row crest, the chosen format, and the reason.
Unconditional on purpose -- the outlier branch is the part of this design most
likely to be wrong on a checkpoint nobody has looked at, and diagnosing it from a
committed artifact beats re-running a 26 GB conversion to find out.

Everything that is not a quantized ``nn.Linear`` weight (norms, embeddings,
biases, modulation tables, non-Linear parameters) is copied through in its
original dtype.

The quantization itself is ``fp8_linear.quantize_weight_to_fp8`` -- the repo's
own function, not a reimplementation -- so a checkpoint made here differs from a
natively-FP8 checkpoint only in provenance.

WHY THIS EXISTS
---------------
The FP8 W8A8 ``torch._scaled_mm`` fast path (opt-in, ``SUSHI_FP8_SCALED_MM=1``)
has to be measured against a bf16 baseline of the SAME architecture on the SAME
hardware. Krea 2 ships bf16 locally and is a single transformer that fits VRAM,
so it is the speed vehicle; this tool produces its matched FP8 arm. See
``examples/api/bench_fp8_scaled_mm.py`` for the measurement protocol and the
pre-registered decision rule.

WHICH LINEARS ARE QUANTIZED
---------------------------
Every ``nn.Linear`` in the model, EXCEPT those whose ``in_features`` or
``out_features`` is not a multiple of ``--min-align``. The default follows the
format: 16 for fp8 (``_scaled_mm``'s alignment) and 8 for int8
(``torch._int_mm``'s). Rationale: the fast path rejects unaligned shapes
outright, so an unaligned layer can never reach it -- quantizing it would add
quantization error for exactly zero speed. For Krea 2 this excludes one layer
under either setting, ``text_fusion.projector``, which is 12x1.

By default it does NOT exclude layers that are merely too small or too thin for
the RUNTIME min-work gate (``int8_linear._MIN_WORK_*``), nor the timestep MLPs
whose ``m`` is the batch size and can never clear ``torch._int_mm``'s ``m > 16``
floor. Those layers still get quantized, for VRAM: ``time_mod_proj`` alone is
36864x6144 = 226M parameters, the single largest weight in the model, and it
costs 226 MB as int8 against 452 MB as bf16 while running the dequant path
either way.

``--skip-below-work-gate`` reverses that trade for architectures where it does
not pay. A layer whose ``in_features < _MIN_WORK_K`` or
``out_features < _MIN_WORK_N`` can never be admitted by the runtime gate AT ANY
``m``, so it always runs ``Int8Linear._dequant_forward`` -- which is SLOWER than
the ``F.linear`` the unquantized checkpoint would have run, because it pays a
full ``(n, k)`` weight expansion first. Whether that matters depends entirely on
how many such layers the architecture has:

* Krea 2 has few, so the default (quantize them, take the VRAM) is right and the
  shipped ``krea2_int8`` artifact is unaffected by this flag existing.
* Anima has 283 of them out of 515 Linears -- 168 AdaLN modulation Linears alone,
  whose ``m`` is the batch size -- and a Linear-only per-pass roll-up over the
  real layer census (RTX 6000 Ada, bf16, batch 1; harness preserved at
  ``tmp/anima_int8_rollup_probe.py``) puts the naive all-int8 artifact BELOW
  break-even at 384x384 (~0.9x vs the bf16 checkpoint) and behind the filtered
  artifact at every resolution measured, while the filtered artifact is ~1.3x at
  384x384 rising to ~2x at 1024x1024 and above. Read those to ONE significant
  digit and treat <=512x512 as "break-even to modestly positive": the low-``m``
  rows are dispatch-bound, not arithmetic-bound, and independent harnesses
  disagree there (0.9x-1.3x at 384x384). None of it is end-to-end -- attention,
  norms, the TE and the VAE are excluded and unchanged, so the whole-generation
  effect is strictly closer to 1.0.
  The flag costs ~369 MB of the saving: 2.4987 GB as shipped vs ~2.13 GB fully
  quantized, against a 4.1822 GB bf16 source (-40% instead of -49%).

The flag is a pure SHAPE test using the same constants the runtime gate uses
(imported, not retyped), exactly like ``--min-align``, and it is INT8-ONLY --
``fp8_linear`` has no ``_MIN_WORK_*`` at all, so ``--format fp8`` ignores it with
a printed notice rather than filtering an e4m3 conversion against int8's rule.
It cannot express the ``m``-dependent third condition (``_MIN_WORK_MKN``), which
is a property of the call, not of the layer.

The reference FP8 checkpoint this format comes from -- ideogram-4-fp8 --
quantizes every Linear including the input/output projections and the timestep
MLP, so "all Linears" is the matching convention, not a narrowed subset.

Use ``--exclude`` (repeatable regex, matched against the module path) to carve
out more.

RELATED: THE RUNTIME EXPORT
---------------------------
``POST /api/v1/models/export-quantized`` writes the LOADED transformer out in
this same layout, which is how an in-place runtime conversion
(``unet_quantization: "int8"`` on anima / krea2 / flux2) is made to survive a
restart.
It shares this tool's writer, sibling-junction helper, key prefixes and metadata
builders through ``core.models.common.quantized_export`` -- the shared import is
the pin against the two emitting different files. This tool remains the way to
convert a checkpoint WITHOUT loading it into the backend.

STREAMING
---------
Source and destination are read/written shard-by-shard. The Krea 2 bf16
transformer is ~26 GB; materialising it whole (source) plus the whole output
would need far more RAM than shard-at-a-time does.

USAGE
-----
    venv/Scripts/python.exe subapps/fp8_quantize/quantize_transformer_fp8.py \
        --arch krea2 --format fp8 \
        --source "<bf16 model dir>/diffusion_pytorch_model.safetensors.index.json" \
        --output "<scratch dir>/krea2_fp8/krea2_fp8.safetensors" \
        --link-siblings "<bf16 model dir>"

    venv/Scripts/python.exe subapps/fp8_quantize/quantize_transformer_fp8.py \
        --arch krea2 --format int8 \
        --source "<bf16 model dir>/diffusion_pytorch_model.safetensors.index.json" \
        --output "<scratch dir>/krea2_int8/krea2_int8.safetensors" \
        --link-siblings "<bf16 model dir>"

    venv/Scripts/python.exe subapps/fp8_quantize/quantize_transformer_fp8.py \
        --arch anima --format int8 --skip-below-work-gate \
        --source "<anima root>/split_files/diffusion_models/<dit>.safetensors" \
        --output "<scratch dir>/anima_int8/split_files/diffusion_models/anima_int8.safetensors" \
        --link-siblings "<anima root>"

    venv/Scripts/python.exe subapps/fp8_quantize/quantize_transformer_fp8.py \
        --arch flux2 --format int8 \
        --source "<flux2 root>/flux-2-klein-base-4b.safetensors" \
        --output "<scratch dir>/flux2_int8/flux2_int8.safetensors"

FLUX.2 sources may be in EITHER on-disk layout. A BFL/Comfy single file
(``double_blocks.*`` / ``single_blocks.*``) is remapped to diffusers keys --
including the fused ``img_attn.qkv`` / ``txt_attn.qkv`` split -- by the arch's
``source_transform``, which calls the same diffusers converter the production
loader calls; a source that is already in diffusers layout passes through
untouched. Either way the OUTPUT is diffusers-keyed, so the loader reads it
through its "already in diffusers format" branch and learns nothing new. FLUX.2
gets no ``--link-siblings`` treatment: its loader resolves the VAE from the
Apache-2.0 VAE store and the text encoder / tokenizer / scheduler from the
detected base repo, and probes nothing next to the checkpoint.

Write the output to a scratch location, NOT under a ``M:/model/<arch>/`` root:
those roots hold the vanilla checkpoints and their sibling directories are
completion sources for the loaders.

``--link-siblings SRC`` creates directory junctions (``mklink /J``, no admin
rights needed) for ``text_encoder`` / ``vae`` / ``tokenizer`` / ``scheduler``
from SRC next to the output, so the loader's sibling probe resolves the same
text encoder and VAE the source checkpoint uses. Without them the loader would
fall back to a hub download and the arms would no longer be matched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND = os.path.join(REPO_ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from safetensors import safe_open  # noqa: E402

from core.models.ideogram4.vendor.fp8_linear import (  # noqa: E402
    FP8_SCALE_SUFFIX,
    quantize_weight_to_fp8,
)
from core.models.ideogram4.vendor.int8_linear import (  # noqa: E402
    INT8_SCALE_SUFFIX,
)
# The scale suffix as the GUARD spells it (both formats share it; only the weight
# dtype tells them apart). Imported rather than retyped so the refusal below and
# the loader-side guard cannot disagree about what a quantized file looks like.
from core.models.common.quantized_checkpoint_guard import (  # noqa: E402
    QUANT_SCALE_SUFFIX,
    QUANT_WEIGHT_DTYPES,
)
# THE selection logic lives in the shared module, which the RUNTIME in-place
# converter imports too. One module, two callers -- the shared import is the pin
# against the two rules drifting apart. Everything below is a thin caller:
# ``select_targets`` (shape filters, incl. the min-work gate whose constants are
# int8_linear's own), ``audit_and_quantize_int8`` (crest pre-filter + the
# measured int8-vs-e4m3 backstop that actually decides), the per-arch knobs, and
# the audit document's shape.
from core.models.common.int8_runtime_quantize import (  # noqa: E402
    DEFAULT_CREST_THRESHOLD,
    FORMAT_MIN_ALIGN,
    INT8_MIN_WORK_K,
    INT8_MIN_WORK_N,
    arch_policy,
    audit_and_quantize_int8,
    audit_document,
    linear_paths,
    select_targets,
)
# Private in the writer module on purpose (they are format constants, not API);
# imported rather than re-typed so a change to the on-disk convention cannot
# leave this tool emitting the old one.
from core.models.common.single_file_format import _INDEX_SUFFIX, _SHARD_SUFFIX  # noqa: E402
# The streaming writer, the sibling-junction helper, the per-arch key prefixes
# and the per-arch metadata builders live in the backend module so the RUNTIME
# export (POST /models/export-quantized, which writes a live in-place-quantized
# transformer) and this offline tool emit byte-compatible files. Same reasoning
# as the int8_runtime_quantize import above: the shared import IS the pin.
from core.models.common.quantized_export import (  # noqa: E402
    DEFAULT_EXPORT_SHARD_BYTES as DEFAULT_OUT_SHARD_BYTES,
    EXPORT_LAYOUTS,
    SIBLING_DIRS,
    ShardWriter,
    identity_source_transform,
    layout_source_transform,
    link_siblings,
)

# FORMAT_MIN_ALIGN and DEFAULT_CREST_THRESHOLD are imported from
# core.models.common.int8_runtime_quantize (see the import block above): they are
# part of the selection rule, and the runtime converter applies the same values.


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------
#
# Each entry knows how to (a) build the module on the META device so its
# ``nn.Linear`` paths can be enumerated without allocating 13 B parameters, and
# (b) declare the key prefixes and metadata the arch's own single-file loader
# expects.
#
# Keys:
#   prefix         (required) prepended to every OUTPUT key -- the layout the
#                  arch's loader reads.
#   source_prefix  (optional, default "") stripped from every SOURCE key before
#                  it is matched against a module path. Needed whenever the
#                  checkpoint wraps the module (Anima ships ``net.*``, which its
#                  loader strips); without it every Linear silently fails to
#                  match and the tool quantizes nothing.
#   source_transform (optional, default identity) ``(key, tensor) ->
#                  [(key', tensor'), ...]``, applied to every streamed source
#                  tensor BEFORE source_prefix stripping. "Source key == module
#                  path modulo one prefix" holds for krea2 and anima and fails
#                  for FLUX.2 (BFL key remap + fused-qkv split) and Ideogram 4
#                  (its loader splits fused qkv before the quantized swap, so the
#                  checkpoint key is not the module path). Declared in
#                  EXPORT_LAYOUTS, next to source_prefix, because it is the same
#                  kind of fact; see that table for the full contract.
#   config / build_meta / metadata  (required) as for krea2.
#   siblings       (optional, default SIBLING_DIRS) component directories
#                  --link-siblings junctions next to the output; may be
#                  relative paths.
#   sibling_root   (optional, default ".") where the sibling names are rooted,
#                  relative to the OUTPUT's directory. Krea 2 writes its output
#                  at the layout root so "." is right; Anima's output sits at
#                  ``<root>/split_files/diffusion_models/``, so its layout root
#                  is two levels up.
#
# "Add an arch: add one entry and nothing else" holds for an arch whose
# checkpoint keys are already module paths and whose loader probes the default
# sibling names. Anima satisfied neither, which is what ``source_prefix`` and
# ``siblings`` are for; both are generic and default to today's behaviour, so no
# existing entry changes.


def _krea2_build_meta(config: dict) -> nn.Module:
    from accelerate import init_empty_weights

    from core.models.krea2.vendor.transformer import Krea2Transformer2DModel

    with init_empty_weights():
        return Krea2Transformer2DModel.from_config(config)


def _krea2_config(source: str) -> dict:
    """Resolve the transformer config for a Krea 2 source (dir or file)."""
    from core.models.krea2.vendor.single_file import KREA2_DEFAULT_CONFIG

    config = dict(KREA2_DEFAULT_CONFIG)
    base = source if os.path.isdir(source) else os.path.dirname(source)
    for cand in (os.path.join(base, "config.json"), os.path.join(base, "transformer", "config.json")):
        if os.path.isfile(cand):
            with open(cand, encoding="utf-8") as fh:
                file_cfg = json.load(fh)
            for k in KREA2_DEFAULT_CONFIG:
                if k in file_cfg:
                    config[k] = file_cfg[k]
            print(f"[fp8] transformer config from {cand}")
            break
    else:
        print("[fp8] no config.json next to the source; using KREA2_DEFAULT_CONFIG")
    return config


def _anima_build_meta(config: dict) -> nn.Module:
    from accelerate import init_empty_weights

    from core.models.anima.anima_models import Anima

    with init_empty_weights():
        return Anima(**config)


def _anima_config(source: str) -> dict:
    """Anima's DiT geometry is a fixed constant, not a per-checkpoint config.

    ``anima_loader.load_anima_dit`` instantiates ``Anima(**ANIMA_DIT_CONFIG)``
    unconditionally and reads no config.json, so the enumeration model here must
    use exactly that dict or the module paths would not correspond to what the
    loader will build.
    """
    from core.models.anima.anima_models import ANIMA_DIT_CONFIG

    print("[fp8] Anima DiT geometry from ANIMA_DIT_CONFIG (the loader uses no config.json)")
    return dict(ANIMA_DIT_CONFIG)


def _flux2_build_meta(config: dict) -> nn.Module:
    from accelerate import init_empty_weights

    from diffusers import Flux2Transformer2DModel

    with init_empty_weights():
        return Flux2Transformer2DModel.from_config(config)


def _flux2_config(source: str) -> dict:
    """Resolve the FLUX.2 transformer geometry from the SOURCE checkpoint's keys.

    Unlike Krea 2 (a config.json next to the weights) and Anima (a compiled-in
    constant), a FLUX.2 single file ships no config at all: the production loader
    downloads one from the base repo it detects. That is right for a loader --
    it needs the text encoder, tokenizer and scheduler from that repo anyway --
    and wrong here, where the only thing wanted is a module tree to enumerate
    ``nn.Linear`` paths from. So this reads the block counts out of the
    checkpoint's own key names and looks them up in the pinned table
    (``core.models.flux2.single_file``), which REFUSES an unrecognised geometry
    instead of defaulting to 4B. ``--config`` overrides it with a real
    ``transformer/config.json`` when one is at hand.

    Only the safetensors HEADER is read (``_source_shards`` already has it), so
    resolving the config costs no tensor bytes.
    """
    from core.models.flux2.single_file import (
        count_flux2_blocks, flux2_config_for_state_dict,
    )

    _shards, key_to_shard = _source_shards(source)
    keys = list(key_to_shard)
    config = flux2_config_for_state_dict(keys)
    n_double, n_single = count_flux2_blocks(keys)
    print(f"[fp8] FLUX.2 geometry from the checkpoint's keys: {n_double} double + "
          f"{n_single} single block(s) -> pinned config "
          f"(num_layers={config['num_layers']}, num_single_layers={config['num_single_layers']}, "
          f"num_attention_heads={config['num_attention_heads']})")
    return config


def _ideogram4_build_meta(config: dict) -> nn.Module:
    from accelerate import init_empty_weights

    from core.models.ideogram4.vendor import Ideogram4Transformer2DModel

    with init_empty_weights():
        return Ideogram4Transformer2DModel.from_config(config)


def _ideogram4_config(source: str) -> dict:
    """Read an Ideogram 4 transformer config from a component dir (or a model root).

    Unlike Krea 2 (a default dict topped up from a file) and FLUX.2 (a pinned
    table), Ideogram 4's geometry is ALWAYS read from the checkpoint's own
    ``config.json``: the published directories carry one per component and the
    loader (``_build_ideogram4_transformer``) reads exactly that file, so nothing
    here may fall back to a compiled-in default -- a wrong ``num_layers`` would
    enumerate module paths that match no weight.
    """
    for cand in (os.path.join(source, "config.json"),
                 os.path.join(source, "transformer", "config.json")):
        if os.path.isfile(cand):
            with open(cand, encoding="utf-8") as fh:
                config = json.load(fh)
            print(f"[fp8] transformer config from {cand}")
            return config
    raise FileNotFoundError(
        f"no Ideogram 4 transformer config.json under {source}; the geometry is read from "
        f"the checkpoint's own config (there is no pinned default), so pass --config or a "
        f"path that has one")


def _ideogram4_sources(source: str) -> List[Dict[str, object]]:
    """The per-component passes for an Ideogram 4 source. TWO of them, always.

    Ideogram 4 is the only architecture here that is TWO transformers -- a
    conditional and an unconditional branch of identical geometry, both required
    by its asymmetric CFG -- and an artifact holding one of them would load
    (``load_ideogram4_single_file`` skips a missing unconditional branch with a
    print) and then generate with a bf16 branch against a quantized one. So both
    are located up front, and a source that can only supply one is REFUSED
    rather than half-converted.

    Two source shapes are accepted, and they differ in where the component
    prefix comes from:

    * a published diffusers MODEL ROOT (``<root>/transformer/`` +
      ``<root>/unconditional_transformer/``): each component is its own
      checkpoint, whose keys are bare module paths, so ``source_prefix`` is empty
      and the component's live prefix is added on OUTPUT;
    * a COMBINED single file or shard index (sushiUI's own save, whose keys
      already carry ``transformer.`` / ``unconditional_transformer.``): one file
      read twice, each pass taking only its own prefix and writing it back
      unchanged.

    Either way ``out_prefix + source_prefix`` is the component's live prefix,
    which is the invariant ``quantized_export.check_layout_prefixes`` states for
    the single-component architectures.
    """
    from core.models.common.quantized_export import layout_module_specs

    specs = layout_module_specs("ideogram4")

    if os.path.isdir(source):
        missing = [name for name, _p in specs if not os.path.isdir(os.path.join(source, name))]
        if not missing:
            return [{
                "component": name,
                "out_prefix": prefix,
                "source_prefix": "",
                "require_source_prefix": False,
                "source": os.path.join(source, name),
                "config_source": os.path.join(source, name),
            } for name, prefix in specs]
        if os.path.isfile(os.path.join(source, "config.json")):
            raise RuntimeError(
                f"{source} looks like a single Ideogram 4 component directory. Point --source "
                f"at the MODEL ROOT (the directory holding "
                f"{' + '.join(n for n, _p in specs)}) or at a combined single file: Ideogram 4 "
                f"needs both transformers in one artifact, and quantizing one of them would "
                f"produce a file that loads with one branch quantized and the other absent.")
        raise FileNotFoundError(
            f"{source} is missing the Ideogram 4 component director(y/ies) "
            f"{', '.join(missing)}")

    _shards, key_to_shard = _source_shards(source)
    absent = [prefix for _name, prefix in specs
              if not any(k.startswith(prefix) for k in key_to_shard)]
    if absent:
        raise RuntimeError(
            f"{source} carries no key under {', '.join(repr(p) for p in absent)}; an Ideogram 4 "
            f"artifact must hold BOTH transformers (asymmetric CFG runs both every step). "
            f"Use a combined single file, or the published model root directory.")

    from core.models.ideogram4.ideogram4_loader import _resolve_ideogram4_base_dir

    base_dir = _resolve_ideogram4_base_dir(source)
    print(f"[fp8] combined single-file source; configs from base directory {base_dir}")
    return [{
        "component": name,
        "out_prefix": "",
        "source_prefix": prefix,
        "require_source_prefix": True,
        "source": source,
        "config_source": os.path.join(base_dir, name),
    } for name, prefix in specs]


def _from_layout(arch: str, **extra) -> Dict[str, object]:
    """An ARCHS entry: the shared on-disk layout plus this tool's source knobs.

    ``prefix`` / ``source_prefix`` / ``source_transform`` / ``metadata`` /
    ``siblings`` / ``sibling_root`` come from ``EXPORT_LAYOUTS`` -- the same
    table the runtime exporter reads -- so the offline artifact and a runtime
    export of the same architecture land on identical keys and identical
    metadata. Only ``config`` and ``build_meta`` (how to enumerate Linears from a
    SOURCE checkpoint, which the runtime path does not need because it has the
    live module) and ``sources`` (how one ``--source`` maps onto the layout's
    components; absent means "one component, this source") are local to this
    tool.

    EVERY arch goes through here, including one that needs a key remap: the
    remap is a ``source_transform`` entry in the shared layout, not a bespoke
    ``ARCHS`` entry that bypasses this function. The offline artifact and the
    runtime export being the same file is the invariant that keeps nine
    architectures on one artifact format, and it only survives while both routes
    read one table.
    """
    layout = EXPORT_LAYOUTS[arch]
    entry: Dict[str, object] = {
        "prefix": layout["offline_prefix"],
        "source_prefix": layout["source_prefix"],
        "source_transform": layout_source_transform(arch),
        "metadata": layout["metadata"],
        "siblings": layout.get("siblings", SIBLING_DIRS),
        "sibling_root": layout.get("sibling_root", "."),
    }
    entry.update(extra)
    return entry


ARCHS = {
    "krea2": _from_layout("krea2", config=_krea2_config, build_meta=_krea2_build_meta),
    "anima": _from_layout("anima", config=_anima_config, build_meta=_anima_build_meta),
    # FLUX.2 is the first arch here whose SOURCE keys are not module paths: a BFL
    # single file needs the diffusers key remap plus a fused-qkv split first. That
    # is declared as the layout's ``source_transform`` (see EXPORT_LAYOUTS), not
    # as a special case in this table, so the offline artifact and a runtime
    # export stay the same file.
    "flux2": _from_layout("flux2", config=_flux2_config, build_meta=_flux2_build_meta),
    # Ideogram 4 is the first arch here that is more than ONE module: two
    # transformers of identical geometry into one artifact (``sources`` below),
    # and its checkpoint keys are not module paths either -- the loader splits a
    # fused qkv before it swaps the quantized Linears in, which is the layout's
    # ``source_transform``.
    "ideogram4": _from_layout("ideogram4", config=_ideogram4_config,
                              build_meta=_ideogram4_build_meta,
                              sources=_ideogram4_sources),
}


# ---------------------------------------------------------------------------
# Source reading (streaming)
# ---------------------------------------------------------------------------

def _source_shards(source: str) -> Tuple[List[str], Dict[str, str]]:
    """Return (shard file paths, {key: shard file}) for a checkpoint source.

    Accepts a ``<stem>.safetensors.index.json``, a single ``.safetensors``, or a
    directory holding either under the diffusers component name.
    """
    if os.path.isdir(source):
        for basename in ("diffusion_pytorch_model", "model"):
            idx = os.path.join(source, f"{basename}{_INDEX_SUFFIX}")
            single = os.path.join(source, f"{basename}{_SHARD_SUFFIX}")
            if os.path.isfile(idx):
                source = idx
                break
            if os.path.isfile(single):
                source = single
                break
        else:
            raise FileNotFoundError(f"no safetensors / shard index found in {source}")

    if source.endswith(_INDEX_SUFFIX):
        with open(source, encoding="utf-8") as fh:
            index = json.load(fh)
        directory = os.path.dirname(source)
        weight_map = index.get("weight_map", {}) or {}
        key_to_shard = {k: os.path.join(directory, v) for k, v in weight_map.items()}
        shards = sorted(set(key_to_shard.values()))
        return shards, key_to_shard

    with safe_open(source, framework="pt", device="cpu") as fh:
        keys = list(fh.keys())
    return [source], {k: source for k in keys}


# ---------------------------------------------------------------------------
# Linear enumeration
# ---------------------------------------------------------------------------

def _strip_prefix(key: str, prefix: str) -> str:
    """``key`` with the arch's source prefix removed, if it carries it."""
    return key[len(prefix):] if prefix and key.startswith(prefix) else key


# ``linear_paths`` / ``select_targets`` / ``audit_and_quantize_int8`` are
# imported from ``core.models.common.int8_runtime_quantize``. They used to live
# here; they moved when the runtime in-place converter needed the identical rule.


# ---------------------------------------------------------------------------
# Component passes
# ---------------------------------------------------------------------------
#
# A "pass" is one component of the output artifact: which source it reads, which
# prefix its source keys carry, and which prefix its output keys get. Almost
# every architecture here has exactly ONE, which is the historical behaviour
# spelled as a single-element plan; Ideogram 4 has two transformers that must
# land in one file, so its ARCHS entry declares a ``sources`` resolver.
#
# INVARIANT, per pass: ``out_prefix + source_prefix`` is that component's live
# state-dict prefix (``EXPORT_LAYOUTS[...]["modules"]``). It is what makes an
# offline artifact and a runtime export of the same model the same file.

def _plan_passes(arch_name: str, arch: Dict[str, object], source: str) -> List[Dict[str, object]]:
    """The component passes for ``source``; one, unless the arch says otherwise."""
    resolver = arch.get("sources")
    if resolver is None:
        return [{
            "component": None,
            "out_prefix": arch["prefix"],
            "source_prefix": arch.get("source_prefix", ""),
            "require_source_prefix": False,
            "source": source,
            "config_source": source,
        }]
    passes = list(resolver(source))
    if not passes:
        raise RuntimeError(f"arch {arch_name!r} resolved no component passes for {source}")
    return passes


def _select_pass(tag: str, args, arch: Dict[str, object], spec: Dict[str, object],
                 source_transform, min_align: int, excludes, skip_gate: bool) -> Dict[str, object]:
    """Enumerate one component's Linears and choose which of them to quantize.

    Returns ``spec`` extended with ``shards`` / ``canonical_keys`` / ``config`` /
    ``targets`` / ``skipped``. Selection for EVERY pass happens before the first
    byte is written, so a dry run reports the whole artifact and a real run
    cannot fail half way through for a reason that was knowable up front.
    """
    label = f"{tag} [{spec['component']}]" if spec.get("component") else tag
    source = str(spec["source"])
    source_prefix = str(spec["source_prefix"])
    shards, key_to_shard = _source_shards(source)
    print(f"{label} source has {len(key_to_shard)} tensors in {len(shards)} shard(s)")

    # The transform maps raw checkpoint keys into the arch's canonical source
    # layout (identity for an arch whose keys ALREADY are module paths). Run it
    # key-only here -- the contract is that ``tensor=None`` yields the same key
    # set -- so the selection below is made against the keys the write loop will
    # actually produce, not against the raw ones.
    raw_keys = [k for k in key_to_shard
                if not spec.get("require_source_prefix") or k.startswith(source_prefix)]
    canonical_keys = set()
    for _k in raw_keys:
        for _ck, _ in source_transform(_k, None):
            canonical_keys.add(_ck)
    if source_transform is not identity_source_transform:
        print(f"{label} source_transform: {len(raw_keys)} source key(s) -> "
              f"{len(canonical_keys)} canonical key(s)")
        if not canonical_keys:
            raise RuntimeError(
                f"arch {args.arch!r} declares a source_transform that produced no keys at "
                f"all for this checkpoint; refusing to run (the output would be empty)")

    # Match module paths, not raw keys: a source that wraps the module (Anima's
    # ``net.``, Ideogram 4's per-component prefix in a combined file) must have
    # that prefix removed before the comparison, or nothing matches and the tool
    # silently quantizes zero layers.
    if source_prefix:
        n_pref = sum(1 for k in canonical_keys if k.startswith(source_prefix))
        print(f"{label} source_prefix={source_prefix!r}: {n_pref}/{len(canonical_keys)} keys carry it")
        if n_pref == 0:
            raise RuntimeError(
                f"arch {args.arch!r} declares source_prefix={source_prefix!r} but no source key "
                f"starts with it; refusing to run (every Linear would silently be skipped)")
    module_keys = {_strip_prefix(k, source_prefix) for k in canonical_keys}

    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            config = json.load(fh)
        print(f"{label} transformer config from --config {args.config}")
    else:
        config = arch["config"](spec.get("config_source", source))
    meta_model = arch["build_meta"](config)
    linears = linear_paths(meta_model)
    targets, skipped = select_targets(linears, module_keys, min_align, excludes,
                                      skip_below_work_gate=skip_gate)
    if not targets:
        raise RuntimeError(
            f"no Linear weight matched between the {len(linears)} module path(s) and the "
            f"{len(canonical_keys)} source key(s); nothing would be quantized")

    print(f"{label} {len(linears)} nn.Linear module(s); quantizing {len(targets)}, "
          f"skipping {len(skipped)}")
    for name, reason in skipped:
        print(f"{label}   skip {name}: {reason}")

    return {**spec, "shards": shards, "canonical_keys": canonical_keys,
            # RAW key -> shard, for the already-quantized refusal below: the
            # source's weight DTYPES live in the shard headers and are read from
            # there (no tensor bytes), and the raw key is what indexes them.
            "key_to_shard": {k: key_to_shard[k] for k in raw_keys},
            "config": config, "targets": targets, "skipped": skipped}


def _quantized_dtype_names() -> set:
    """The safetensors header spellings of the guard's quantized weight dtypes.

    Derived from safetensors' own name->torch.dtype table where it is available,
    so the set follows the guard's ``QUANT_WEIGHT_DTYPES`` rather than a second
    hand-written list; the literal fallback covers a future safetensors that
    stops exposing the table (the names are part of the on-disk format, not of
    the library's API, so they are stable).
    """
    try:
        import safetensors.torch as _st
        table = getattr(_st, "_TYPES", None) or {}
        names = {name for name, dtype in table.items() if dtype in QUANT_WEIGHT_DTYPES}
        if names:
            return names
    except Exception:
        pass
    return {"I8", "U8", "F8_E4M3", "F8_E4M3FNUZ", "F8_E5M2", "F8_E5M2FNUZ"}


def _quantized_source_weights(
    selections: List[Dict[str, object]],
) -> Tuple[int, List[str], set]:
    """(count, up to 3 example keys, dtype names) of quantized ``.weight`` tensors.

    Header-only: one ``safe_open`` per shard, ``get_slice(key).get_dtype()`` per
    key. No tensor bytes are read and nothing is materialised.
    """
    quantized_names = _quantized_dtype_names()
    total = 0
    examples: List[str] = []
    seen_dtypes: set = set()
    for selection in selections:
        key_to_shard = selection.get("key_to_shard") or {}
        by_shard: Dict[str, List[str]] = {}
        for key, shard in key_to_shard.items():
            if key.endswith(".weight"):
                by_shard.setdefault(shard, []).append(key)
        for shard, keys in by_shard.items():
            with safe_open(shard, framework="pt", device="cpu") as fh:
                for key in keys:
                    dtype = fh.get_slice(key).get_dtype()
                    if dtype in quantized_names:
                        total += 1
                        seen_dtypes.add(dtype)
                        if len(examples) < 3:
                            examples.append(key)
    return total, examples, seen_dtypes


def _refuse_quantized_source(tag: str, args, selections: List[Dict[str, object]]) -> None:
    """Refuse a source that is ALREADY weight-only quantized.

    Re-quantizing an e4m3 (or int8) checkpoint is not a smaller version of
    quantizing a bf16 one: the weights have already been rounded once, so the
    second pass measures and encodes the ROUNDING, and the source's own
    ``.weight_scale`` keys would additionally collide with the ones this tool
    writes for the same layer -- one silently overwriting the other in the shard
    buffer. The runtime converter refuses the same input for the same reason
    (``quantization_superseded``, measured at 11.2x the weight error of a direct
    conversion on Anima). Detected on the KEY SET, which the shard index already
    gives us, so it costs no tensor bytes.

    It matters most for Ideogram 4, whose published checkpoints are FP8 or nf4:
    the only correct int8 source for it is a bf16 one (a sushiUI full fine-tune
    save, or a bf16 release), and without this check the tool would happily
    produce a plausible-looking artifact from the FP8 one.

    TWO PIECES OF EVIDENCE, either sufficient, exactly as the loader-side
    ``quantized_state_dict_report`` uses:

    * a ``.weight_scale`` key -- a scaled artifact from this tool or from
      ``POST /models/export-quantized``;
    * a ``.weight`` whose stored DTYPE is int8 / uint8 / float8. The scale test
      alone misses the commonest already-rounded source there is: the scale-less
      ComfyUI fp8 CAST (``flux-2-klein-fp8_e4m3fn.safetensors`` and its
      equivalents for the other archs), whose keys remap onto module paths like
      any other source, so nothing else here refuses it. A loader reads that file
      correctly -- casting e4m3 back to bf16 is exact -- but quantizing it is
      still measuring and encoding a rounding, which is the harm above.

    The dtypes come from the safetensors HEADER (``get_slice(...).get_dtype()``),
    so this reads zero tensor bytes, and the dtype set is imported from the
    loader-side guard so the two cannot disagree about what "already quantized"
    means.

    A DRY RUN reports it and continues -- it writes nothing, and the selection it
    prints is still the honest answer to "which layers would be chosen".
    """
    scales = 0
    examples: List[str] = []
    for selection in selections:
        for key in selection["canonical_keys"]:
            if key.endswith(QUANT_SCALE_SUFFIX):
                scales += 1
                if len(examples) < 3:
                    examples.append(key)

    quant_weights, quant_examples, quant_dtypes = _quantized_source_weights(selections)

    if not scales and not quant_weights:
        return
    if scales:
        evidence = (f"{scales} '{QUANT_SCALE_SUFFIX}' key(s) (e.g. "
                    f"{', '.join(examples)})")
        if quant_weights:
            evidence += (f" and {quant_weights} quantized '.weight' tensor(s) "
                         f"({'/'.join(sorted(quant_dtypes))})")
    else:
        evidence = (f"{quant_weights} '.weight' tensor(s) stored as "
                    f"{'/'.join(sorted(quant_dtypes))} with no '{QUANT_SCALE_SUFFIX}' "
                    f"sibling -- a plain dtype cast, or a foreign scale convention such "
                    f"as ComfyUI's '.scale_weight' (e.g. {', '.join(quant_examples)})")
    collision = (" and its existing scales would collide with the ones written here"
                 if scales else "")
    message = (
        f"the source checkpoint is ALREADY weight-only quantized: {evidence}. "
        f"Quantizing it again "
        f"would encode weights that have already been rounded once -- worse than either "
        f"format alone{collision}. "
        f"Use an unquantized (bf16) checkpoint of this architecture as the source.")
    if args.dry_run:
        print(f"{tag} WARNING: {message}")
        print(f"{tag} (a real run would refuse; the selection below is still what it would pick)")
        return
    raise RuntimeError(message)


def write_audit(path: str, rows: List[Dict], args_summary: Dict) -> str:
    """Write the per-layer audit JSON and print a summary table.

    The JSON BODY is built by ``int8_runtime_quantize.audit_document`` -- the
    same function the runtime in-place converter returns -- so an offline
    artifact and a runtime conversion are directly diffable. Only the printed
    table is local to this tool.
    """
    document = audit_document(rows, args_summary)
    counts = document["format_counts"]
    geomean = document["geomean_advantage"]
    selected_out = [r for r in rows if r["chosen"] != "int8"]

    print("\n[audit] per-layer weight-error audit "
          f"({len(rows)} quantizable Linear weights)")
    print(f"[audit] {'layer':<44} {'int8':>9} {'e4m3':>9} {'adv':>6} "
          f"{'crest':>7} {'p99':>7}  format")
    for r in rows:
        print(f"[audit] {r['name'][:44]:<44} {r['int8_rel_rms']:9.5f} "
              f"{r['e4m3_rel_rms']:9.5f} {r['advantage_int8_over_e4m3']:6.3f} "
              f"{r['crest_mean']:7.2f} {r['crest_p99']:7.2f}  {r['chosen']}")
    print(f"[audit] format counts: {counts}")
    if selected_out:
        print(f"[audit] selected out of int8 ({len(selected_out)}):")
        for r in selected_out:
            print(f"[audit]   {r['name']} -> {r['chosen']} ({r['reason']})")
    else:
        print("[audit] no layer was selected out of int8")
    if geomean is not None:
        print(f"[audit] geomean int8-over-e4m3 weight-error advantage: {geomean:.3f}x")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=1)
    print(f"[audit] wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=sorted(ARCHS))
    ap.add_argument("--format", choices=sorted(FORMAT_MIN_ALIGN), default="fp8",
                    help="weight format: fp8 (e4m3, every eligible Linear) or int8 "
                         "(per-layer selection between int8 and the fallback)")
    ap.add_argument("--source", required=True,
                    help="bf16 checkpoint: shard index, single safetensors, or a directory")
    ap.add_argument("--output", required=True,
                    help="destination <stem>.safetensors (shards + index written beside it "
                         "when the result exceeds --max-shard-bytes)")
    ap.add_argument("--min-align", type=int, default=None,
                    help="skip Linears whose in/out features are not a multiple of this "
                         "(they can never take the fast path). Defaults to the format's "
                         "GEMM alignment (fp8: 16, int8: 8). 0 disables the check.")
    ap.add_argument("--crest-threshold", type=float, default=DEFAULT_CREST_THRESHOLD,
                    help="[--format int8] mean per-row crest factor above which a layer "
                         "falls back instead of going int8")
    ap.add_argument("--fallback", choices=("e4m3", "bf16"), default="e4m3",
                    help="[--format int8] what a selected-out layer becomes")
    ap.add_argument("--exclude", action="append", default=[],
                    help="regex matched against the module path; repeatable")
    ap.add_argument("--skip-below-work-gate", dest="skip_below_work_gate",
                    action="store_true", default=None,
                    help=f"[--format int8 ONLY; ignored with a notice for other formats] "
                         f"also skip Linears whose in_features < {INT8_MIN_WORK_K} or "
                         f"out_features < {INT8_MIN_WORK_N}: the runtime min-work gate can "
                         f"never admit them at any m, so they would always run the dequant "
                         f"path, which is SLOWER than the unquantized F.linear. Costs VRAM, "
                         f"buys speed. Measured necessary for Anima (283/515 Linears; the "
                         f"naive artifact falls below break-even at 384x384 and is behind the "
                         f"filtered one at every resolution measured -- see "
                         f"tmp/anima_int8_rollup_probe.py); not for Krea 2. DEFAULTS TO THE "
                         f"ARCH TABLE (int8_runtime_quantize.ARCH_QUANT_POLICY: on for anima, "
                         f"off for krea2), which is the same table the runtime in-place "
                         f"converter reads; pass this flag or --no-skip-below-work-gate to "
                         f"override it.")
    ap.add_argument("--no-skip-below-work-gate", dest="skip_below_work_gate",
                    action="store_false",
                    help="force the min-work-gate filter OFF regardless of the arch table")
    ap.add_argument("--config", metavar="CONFIG_JSON",
                    help="transformer config.json to build the enumeration model from, "
                         "instead of the arch's own resolution (a config.json next to the "
                         "source for krea2, a compiled-in constant for anima, the pinned "
                         "geometry table for flux2). Use it for a variant this repo does "
                         "not pin; it is used VERBATIM, so it must be that checkpoint's "
                         "real config -- a mismatched geometry produces module paths that "
                         "match no weight and the run then fails with 'no Linear weight "
                         "matched'.")
    ap.add_argument("--max-shard-bytes", type=int, default=DEFAULT_OUT_SHARD_BYTES)
    ap.add_argument("--link-siblings", metavar="SRC_DIR",
                    help="create text_encoder/vae/tokenizer/scheduler junctions from SRC_DIR "
                         "next to the output so the loader's sibling probe resolves them")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be quantized and exit without writing")
    args = ap.parse_args()

    arch = ARCHS[args.arch]
    fmt = args.format
    tag = f"[{fmt}]"
    # The per-arch knobs come from the SHARED table, not from CLI defaults, so
    # the offline artifact and a runtime in-place conversion of the same arch
    # select the same layers. An explicit flag still wins.
    policy = arch_policy(args.arch, fmt)
    excludes = [re.compile(p) for p in (list(args.exclude) + list(policy["excludes"]))]
    min_align = int(policy["min_align"]) if args.min_align is None else args.min_align

    # --skip-below-work-gate is an INT8-ONLY selector, scoped here the way
    # --fallback and --crest-threshold are scoped by the writer's `fmt == "int8"`
    # branch. Its two constants are int8_linear's runtime gate; fp8_linear has no
    # _MIN_WORK_* at all (the e4m3 path has a different profile and no such
    # shape gate), so applying them to an e4m3 conversion would filter it against
    # a rule that governs nothing it will ever run. Ignored rather than silently
    # honoured, and said out loud rather than ignored silently. ``arch_policy``
    # has already forced the arch default to False for a non-int8 format; the
    # notice below covers an EXPLICIT flag on such a format.
    skip_gate = bool(policy["skip_below_work_gate"]) if args.skip_below_work_gate is None \
        else bool(args.skip_below_work_gate)
    if skip_gate and fmt != "int8":
        print(f"{tag} --skip-below-work-gate IGNORED: it is an int8-only selector "
              f"(its k>={INT8_MIN_WORK_K} / n>={INT8_MIN_WORK_N} constants are "
              f"int8_linear's runtime gate; the {fmt} path has no equivalent).")
        skip_gate = False

    source_transform = arch.get("source_transform") or identity_source_transform
    print(f"{tag} arch={args.arch} format={fmt} min_align={min_align} "
          f"skip_below_work_gate={skip_gate} source={args.source}")

    plan = _plan_passes(args.arch, arch, args.source)
    multi = len(plan) > 1
    if multi:
        print(f"{tag} {len(plan)} component pass(es) into ONE artifact: " + ", ".join(
            f"{p['component']} (out={p['out_prefix']!r}, src={p['source_prefix']!r})"
            for p in plan))

    selections = [
        _select_pass(tag, args, arch, p, source_transform, min_align, excludes, skip_gate)
        for p in plan
    ]
    _refuse_quantized_source(tag, args, selections)

    total_targets = sum(len(s["targets"]) for s in selections)
    if multi:
        print(f"{tag} TOTAL across {len(plan)} component(s): quantizing {total_targets}, "
              f"skipping {sum(len(s['skipped']) for s in selections)}")

    if args.dry_run:
        print(f"{tag} dry run: nothing written")
        return 0

    # The metadata builder gets the PRIMARY component's config, exactly as the
    # runtime exporter's does (``quantized_export_job`` reads ``modules[0]``).
    metadata = arch["metadata"](selections[0]["config"])
    metadata["quantized_linears"] = str(total_targets)
    metadata["quant_source"] = os.path.abspath(args.source)
    # NOTE: deliberately NOT written into a key the Krea 2 loader scans for
    # rejected quant layouts. `single_file._REJECTED_QUANT_TOKENS` matches
    # ("int8_convrot", "mxfp8", "nvfp4") against the PATH plus
    # metadata["quantization"], so this format must neither be called
    # "int8_convrot" nor be written to a path containing that token. The label
    # below ("int8_perrow") and the "quant_format" key avoid both.
    metadata["quant_format"] = "int8_perrow" if fmt == "int8" else "fp8_e4m3_perrow"
    if fmt == "fp8":
        # Preserved for checkpoints produced before --format existed.
        metadata["fp8_quantized_linears"] = str(total_targets)
        metadata["fp8_source"] = os.path.abspath(args.source)

    writer = ShardWriter(args.output, metadata, args.max_shard_bytes)
    t0 = time.perf_counter()
    quantized = 0
    passthrough = 0
    audit: List[Dict] = []
    # EVERY component into the SAME writer. Two files would not be a single-file
    # artifact, and the Ideogram 4 loader reads both branches out of one.
    for selection in selections:
        source_prefix = selection["source_prefix"]
        prefix = selection["out_prefix"]
        component = selection["component"]
        target_set = set(selection["targets"])
        if multi:
            print(f"{tag} component '{component}': {len(target_set)} target(s), "
                  f"out_prefix={prefix!r}, source_prefix={source_prefix!r}")
        for shard in selection["shards"]:
            print(f"{tag} reading {os.path.basename(shard)}")
            with safe_open(shard, framework="pt", device="cpu") as fh:
                for raw_key in fh.keys():
                    if selection["require_source_prefix"] and not raw_key.startswith(source_prefix):
                        # A component pass over a COMBINED source: the other
                        # component's keys belong to the other pass, and writing
                        # them here would duplicate them under the wrong prefix.
                        continue
                    raw_tensor = fh.get_tensor(raw_key)
                    # ONE source tensor may become several canonical ones (a fused
                    # qkv weight -> q/k/v) or none. Per-row scales make "quantize the
                    # fused weight then split" and "split then quantize" identical
                    # (the rows are independent), so splitting FIRST costs nothing
                    # and buys the property that matters: the runtime export sees the
                    # live module after its loader has split, so the offline artifact
                    # must carry the split keys to stay comparable with it.
                    for key, tensor in source_transform(raw_key, raw_tensor):
                        # ``base`` is a MODULE PATH (source_prefix stripped) so it can be
                        # compared with target_set; ``key`` keeps the source layout so the
                        # output is key-for-key identical apart from dtype + the new scales.
                        base = (_strip_prefix(key[: -len(".weight")], source_prefix)
                                if key.endswith(".weight") else None)
                        if base is not None and base in target_set:
                            if tensor.dim() != 2:
                                raise RuntimeError(f"{key}: expected a 2-D Linear weight, got {tuple(tensor.shape)}")
                            # The scale is a SIBLING of the weight key, so it must be built
                            # from ``key`` (source layout), not from the stripped ``base``:
                            # both swap helpers look for ``<weight key minus .weight>.weight_scale``.
                            scale_stem = f"{prefix}{key[: -len('.weight')]}"
                            if fmt == "int8":
                                # The audit row is named by MODULE PATH, namespaced by
                                # component when there is more than one: the two
                                # Ideogram 4 transformers have identical geometry and
                                # therefore identical paths, so an un-namespaced table
                                # would look like one transformer audited twice. Same
                                # rule as the runtime converter's merged document.
                                row_name = f"{component}.{base}" if multi else base
                                chosen, q, scale, row = audit_and_quantize_int8(
                                    row_name, tensor, args.crest_threshold, args.fallback)
                                audit.append(row)
                                writer.add(f"{prefix}{key}", q.contiguous())
                                if scale is not None:
                                    writer.add(f"{scale_stem}{INT8_SCALE_SUFFIX}", scale.contiguous())
                                quantized += chosen != "bf16"
                                passthrough += chosen == "bf16"
                            else:
                                q, scale = quantize_weight_to_fp8(tensor)
                                writer.add(f"{prefix}{key}", q.contiguous())
                                writer.add(f"{scale_stem}{FP8_SCALE_SUFFIX}", scale.contiguous())
                                quantized += 1
                        else:
                            writer.add(f"{prefix}{key}", tensor.contiguous())
                            passthrough += 1
                        del tensor
                    del raw_tensor
    written = writer.close()
    elapsed = time.perf_counter() - t0

    print(f"{tag} quantized {quantized} Linear weight(s), passed through {passthrough} tensor(s)")
    print(f"{tag} wrote {written} ({writer.total_bytes / 2**30:.2f} GB) in {elapsed:.1f}s")
    if fmt == "fp8" and quantized != total_targets:
        print(f"{tag} WARNING: expected {total_targets} quantized weights, wrote {quantized}")

    if fmt == "int8":
        stem = os.path.basename(args.output)
        if stem.endswith(_SHARD_SUFFIX):
            stem = stem[: -len(_SHARD_SUFFIX)]
        audit_path = os.path.join(os.path.dirname(os.path.abspath(args.output)),
                                  f"{stem}.int8_audit.json")
        write_audit(audit_path, audit, {
            "arch": args.arch, "format": fmt, "min_align": min_align,
            "skip_below_work_gate": skip_gate,
            "min_work_k": INT8_MIN_WORK_K, "min_work_n": INT8_MIN_WORK_N,
            "crest_threshold": args.crest_threshold, "fallback": args.fallback,
            "source": os.path.abspath(args.source), "output": written,
            "components": [s["component"] for s in selections] if multi else None,
            "skipped": [
                {"name": (f"{s['component']}.{n}" if multi else n), "reason": r}
                for s in selections for n, r in s["skipped"]
            ],
        })

    if args.link_siblings:
        print(f"{tag} linking companion component dirs")
        sibling_dest = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(args.output)), arch.get("sibling_root", ".")))
        link_siblings(args.link_siblings, sibling_dest,
                      names=arch.get("siblings", SIBLING_DIRS))

    print(f"{tag} load it with: source_type=safetensors source={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
