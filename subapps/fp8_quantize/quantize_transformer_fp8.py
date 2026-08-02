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
import subprocess
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
from safetensors.torch import save_file  # noqa: E402

from core.models.ideogram4.vendor.fp8_linear import (  # noqa: E402
    FP8_SCALE_SUFFIX,
    quantize_weight_to_fp8,
)
from core.models.ideogram4.vendor.int8_linear import (  # noqa: E402
    INT8_SCALE_SUFFIX,
    quantize_weight_to_int8,
    weight_crest_factors,
)
# The runtime min-work gate's SHAPE conditions, imported so --skip-below-work-gate
# cannot drift from what Int8Linear._int_mm_forward actually enforces. The third
# condition (_MIN_WORK_MKN) depends on m and therefore on the call, not the layer,
# so it has no offline equivalent.
from core.models.ideogram4.vendor.int8_linear import (  # noqa: E402
    _MIN_WORK_K as INT8_MIN_WORK_K,
    _MIN_WORK_N as INT8_MIN_WORK_N,
)
# Private in the writer module on purpose (they are format constants, not API);
# imported rather than re-typed so a change to the on-disk convention cannot
# leave this tool emitting the old one.
from core.models.common.single_file_format import _INDEX_SUFFIX, _SHARD_SUFFIX  # noqa: E402

# Output shard threshold. Smaller than the repo default (10 GB) because the
# writer buffers a whole shard in RAM while the source shard is also resident.
DEFAULT_OUT_SHARD_BYTES = 4 * 1024 ** 3

# Default companion component directories to junction next to the output. An
# arch whose loader probes different names overrides this with its own
# ``siblings`` entry (see ARCHS); entries may be RELATIVE PATHS, not just names.
SIBLING_DIRS = ("text_encoder", "vae", "tokenizer", "scheduler")

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


def _krea2_metadata(config: dict) -> Dict[str, str]:
    from core.models.krea2.vendor.single_file import KREA2_DEFAULT_CONFIG

    return {
        "model_type": "krea2",
        "variant": "raw",
        "is_distilled": "0",
        "krea2_config": json.dumps({k: config[k] for k in KREA2_DEFAULT_CONFIG if k in config}),
        "has_text_encoder": "0",
        "format": "pt",
    }


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


def _anima_metadata(config: dict) -> Dict[str, str]:
    # ``modelspec.architecture`` is the fast path in ``is_anima_safetensors``;
    # the key-signature check behind it also still passes, because quantization
    # renames nothing (it only changes weight dtypes and adds ``.weight_scale``).
    return {
        "modelspec.architecture": "anima",
        "model_type": "anima",
        "format": "pt",
    }


ARCHS = {
    "krea2": {
        # sushiUI single-file layout: transformer weights live under this prefix.
        "prefix": "transformer.",
        "config": _krea2_config,
        "build_meta": _krea2_build_meta,
        "metadata": _krea2_metadata,
    },
    "anima": {
        # Anima DiT single-files carry the module tree verbatim under ``net.``;
        # the loader strips that prefix, so the output keeps it (an identical
        # layout to the source) and the SOURCE prefix is stripped for matching.
        "prefix": "",
        "source_prefix": "net.",
        "config": _anima_config,
        "build_meta": _anima_build_meta,
        "metadata": _anima_metadata,
        # anima_loader.detect_anima_split_layout walks up from the DiT file to a
        # ``split_files/diffusion_models`` parent and probes these two siblings
        # for the Qwen3 text encoder and the Qwen-Image VAE.
        "siblings": ("split_files/text_encoders", "split_files/vae"),
        "sibling_root": os.path.join("..", ".."),
    },
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
    excludes: List[re.Pattern],
    skip_below_work_gate: bool = False,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Split the Linears into (quantize, [(skipped, reason)]).

    ``present_keys`` holds module paths ALREADY stripped of the arch's
    ``source_prefix``, so it is directly comparable with the meta model's paths.

    ``skip_below_work_gate`` is applied verbatim if set; the INT8-only scoping
    lives in ``main`` (which clears it for other formats), the same place the
    other int8-only selectors are scoped.
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
) -> Tuple[str, torch.Tensor, torch.Tensor, Dict]:
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


def write_audit(path: str, rows: List[Dict], args_summary: Dict) -> str:
    """Write the per-layer audit JSON and print a summary table."""
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["chosen"]] = counts.get(r["chosen"], 0) + 1
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
    finite = [r["advantage_int8_over_e4m3"] for r in rows
              if r["advantage_int8_over_e4m3"] not in (float("inf"),)
              and r["advantage_int8_over_e4m3"] == r["advantage_int8_over_e4m3"]]
    geomean = None
    if finite:
        geomean = float(torch.tensor(finite, dtype=torch.float64).log().mean().exp())
        print(f"[audit] geomean int8-over-e4m3 weight-error advantage: {geomean:.3f}x")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"settings": args_summary, "format_counts": counts,
                   "geomean_advantage": geomean, "layers": rows}, fh, indent=1)
    print(f"[audit] wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """Buffer tensors and flush diffusers-convention shards + an index.

    Shard naming and the index schema match
    ``core.models.common.single_file_format.save_single_file_state`` exactly, so
    the produced checkpoint is read by ``read_state_dict`` like any other.
    """

    def __init__(self, out_path: str, metadata: Dict[str, str], max_shard_bytes: int):
        self.directory = os.path.dirname(os.path.abspath(out_path))
        stem = os.path.basename(out_path)
        if stem.endswith(_SHARD_SUFFIX):
            stem = stem[: -len(_SHARD_SUFFIX)]
        self.stem = stem
        self.metadata = {k: str(v) for k, v in metadata.items()}
        self.max_shard_bytes = max_shard_bytes
        self.buffer: Dict[str, torch.Tensor] = {}
        self.buffer_bytes = 0
        self.total_bytes = 0
        self.shards: List[Tuple[str, List[str]]] = []  # (temp name, keys)
        os.makedirs(self.directory, exist_ok=True)

    def add(self, key: str, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if self.buffer and self.buffer_bytes + nbytes > self.max_shard_bytes:
            self._flush()
        self.buffer[key] = tensor
        self.buffer_bytes += nbytes
        self.total_bytes += nbytes

    def _flush(self) -> None:
        if not self.buffer:
            return
        # Written under a provisional name; renamed once the shard COUNT is known
        # (the diffusers convention encodes the total in every filename).
        tmp = os.path.join(self.directory, f"{self.stem}-part{len(self.shards):05d}.tmp.safetensors")
        save_file(self.buffer, tmp, metadata=self.metadata)
        self.shards.append((tmp, list(self.buffer)))
        print(f"[fp8]   wrote shard {len(self.shards)} ({self.buffer_bytes / 2**30:.2f} GB, "
              f"{len(self.buffer)} tensors)")
        self.buffer = {}
        self.buffer_bytes = 0

    def close(self) -> str:
        self._flush()
        n = len(self.shards)
        if n == 1:
            final = os.path.join(self.directory, f"{self.stem}{_SHARD_SUFFIX}")
            os.replace(self.shards[0][0], final)
            return final
        weight_map: Dict[str, str] = {}
        for i, (tmp, keys) in enumerate(self.shards, start=1):
            name = f"{self.stem}-{i:05d}-of-{n:05d}.safetensors"
            os.replace(tmp, os.path.join(self.directory, name))
            for k in keys:
                weight_map[k] = name
        index_path = os.path.join(self.directory, f"{self.stem}{_INDEX_SUFFIX}")
        with open(index_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"metadata": {**self.metadata, "total_size": self.total_bytes}, "weight_map": weight_map},
                fh,
                indent=2,
            )
        return index_path


# ---------------------------------------------------------------------------
# Sibling junctions
# ---------------------------------------------------------------------------

def link_siblings(src_dir: str, dest_dir: str, names=SIBLING_DIRS) -> List[str]:
    """Create directory junctions dest_dir/<name> -> src_dir/<name>.

    ``names`` may contain RELATIVE PATHS (Anima's components live under
    ``split_files/``), so the link's parent directory is created as needed.

    Junctions (``mklink /J``) need no administrator rights and work across local
    volumes; a symlink would need developer mode. POSIX falls back to symlinks.
    """
    made = []
    os.makedirs(dest_dir, exist_ok=True)
    for name in names:
        target = os.path.join(src_dir, name)
        link = os.path.join(dest_dir, name)
        if not os.path.isdir(target):
            continue
        if os.path.exists(link):
            print(f"[fp8]   sibling '{name}' already present, leaving as is")
            continue
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.name == "nt":
            # cmd parses a leading "/" as a switch, so forward-slash paths must be
            # normalised to backslashes before they reach mklink.
            link, target = os.path.normpath(link), os.path.normpath(target)
            res = subprocess.run(["cmd", "/c", "mklink", "/J", link, target],
                                 capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[fp8]   WARNING: could not link '{name}': {res.stdout.strip()} {res.stderr.strip()}")
                continue
        else:
            os.symlink(target, link, target_is_directory=True)
        made.append(name)
        print(f"[fp8]   linked {link} -> {target}")
    return made


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
    ap.add_argument("--skip-below-work-gate", action="store_true",
                    help=f"[--format int8 ONLY; ignored with a notice for other formats] "
                         f"also skip Linears whose in_features < {INT8_MIN_WORK_K} or "
                         f"out_features < {INT8_MIN_WORK_N}: the runtime min-work gate can "
                         f"never admit them at any m, so they would always run the dequant "
                         f"path, which is SLOWER than the unquantized F.linear. Costs VRAM, "
                         f"buys speed. Measured necessary for Anima (283/515 Linears; the "
                         f"naive artifact falls below break-even at 384x384 and is behind the "
                         f"filtered one at every resolution measured -- see "
                         f"tmp/anima_int8_rollup_probe.py); not for Krea 2. Off by default so "
                         f"existing artifacts reproduce.")
    ap.add_argument("--max-shard-bytes", type=int, default=DEFAULT_OUT_SHARD_BYTES)
    ap.add_argument("--link-siblings", metavar="SRC_DIR",
                    help="create text_encoder/vae/tokenizer/scheduler junctions from SRC_DIR "
                         "next to the output so the loader's sibling probe resolves them")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be quantized and exit without writing")
    args = ap.parse_args()

    arch = ARCHS[args.arch]
    excludes = [re.compile(p) for p in args.exclude]
    fmt = args.format
    tag = f"[{fmt}]"
    min_align = FORMAT_MIN_ALIGN[fmt] if args.min_align is None else args.min_align

    # --skip-below-work-gate is an INT8-ONLY selector, scoped here the way
    # --fallback and --crest-threshold are scoped by the writer's `fmt == "int8"`
    # branch. Its two constants are int8_linear's runtime gate; fp8_linear has no
    # _MIN_WORK_* at all (the e4m3 path has a different profile and no such
    # shape gate), so applying them to an e4m3 conversion would filter it against
    # a rule that governs nothing it will ever run. Ignored rather than silently
    # honoured, and said out loud rather than ignored silently.
    skip_gate = bool(args.skip_below_work_gate)
    if skip_gate and fmt != "int8":
        print(f"{tag} --skip-below-work-gate IGNORED: it is an int8-only selector "
              f"(its k>={INT8_MIN_WORK_K} / n>={INT8_MIN_WORK_N} constants are "
              f"int8_linear's runtime gate; the {fmt} path has no equivalent).")
        skip_gate = False

    source_prefix = arch.get("source_prefix", "")
    print(f"{tag} arch={args.arch} format={fmt} min_align={min_align} "
          f"skip_below_work_gate={skip_gate} source={args.source}")
    shards, key_to_shard = _source_shards(args.source)
    print(f"{tag} source has {len(key_to_shard)} tensors in {len(shards)} shard(s)")

    # Match module paths, not raw keys: a source that wraps the module (Anima's
    # ``net.``) must have that prefix removed before the comparison, or nothing
    # matches and the tool silently quantizes zero layers.
    if source_prefix:
        n_pref = sum(1 for k in key_to_shard if k.startswith(source_prefix))
        print(f"{tag} source_prefix={source_prefix!r}: {n_pref}/{len(key_to_shard)} keys carry it")
        if n_pref == 0:
            raise RuntimeError(
                f"arch {args.arch!r} declares source_prefix={source_prefix!r} but no source key "
                f"starts with it; refusing to run (every Linear would silently be skipped)")
    module_keys = {_strip_prefix(k, source_prefix) for k in key_to_shard}

    config = arch["config"](args.source)
    meta_model = arch["build_meta"](config)
    linears = linear_paths(meta_model)
    targets, skipped = select_targets(linears, module_keys, min_align, excludes,
                                      skip_below_work_gate=skip_gate)
    if not targets:
        raise RuntimeError(
            f"no Linear weight matched between the {len(linears)} module path(s) and the "
            f"{len(key_to_shard)} source key(s); nothing would be quantized")

    print(f"{tag} {len(linears)} nn.Linear module(s); quantizing {len(targets)}, skipping {len(skipped)}")
    for name, reason in skipped:
        print(f"{tag}   skip {name}: {reason}")

    if args.dry_run:
        print(f"{tag} dry run: nothing written")
        return 0

    target_set = set(targets)
    prefix = arch["prefix"]
    metadata = arch["metadata"](config)
    metadata["quantized_linears"] = str(len(targets))
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
        metadata["fp8_quantized_linears"] = str(len(targets))
        metadata["fp8_source"] = os.path.abspath(args.source)

    writer = ShardWriter(args.output, metadata, args.max_shard_bytes)
    t0 = time.perf_counter()
    quantized = 0
    passthrough = 0
    audit: List[Dict] = []
    for shard in shards:
        print(f"{tag} reading {os.path.basename(shard)}")
        with safe_open(shard, framework="pt", device="cpu") as fh:
            for key in fh.keys():
                tensor = fh.get_tensor(key)
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
                        chosen, q, scale, row = audit_and_quantize_int8(
                            base, tensor, args.crest_threshold, args.fallback)
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
    written = writer.close()
    elapsed = time.perf_counter() - t0

    print(f"{tag} quantized {quantized} Linear weight(s), passed through {passthrough} tensor(s)")
    print(f"{tag} wrote {written} ({writer.total_bytes / 2**30:.2f} GB) in {elapsed:.1f}s")
    if fmt == "fp8" and quantized != len(targets):
        print(f"{tag} WARNING: expected {len(targets)} quantized weights, wrote {quantized}")

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
            "skipped": [{"name": n, "reason": r} for n, r in skipped],
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
