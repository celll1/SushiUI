#!/usr/bin/env python3
"""Convert the staged SenseNova-U1.5-8B-MoT bf16 checkpoint to sushiUI's int8
single-file layout.

UNIT-1 SCOPE: acquisition/conversion only, no loader/pipeline backend. Targets
are selected by REGEX over the checkpoint's own keys (``QUANTIZE_KEY_RE``),
alignment-gated only -- no crest/e4m3 fallback, no min-work-gate filtering.

WHAT IS QUANTIZED: both MoT branches' attention and MLP Linears --
``language_model.model.layers.{N}.self_attn.{q,k,v,o}_proj[_mot_gen]`` and
``language_model.model.layers.{N}.mlp[_mot_gen].{gate,up,down}_proj`` (588 of
them across 42 layers: 14 per layer). Everything else stays bf16.

STREAMING: one tensor at a time via ``safetensors.safe_open`` (mmap-backed,
no whole-file read) into ``quantized_export.ShardWriter``. Verification is
header-only (dtype/shape round-trip, no tensor data re-read).

USAGE
-----
    venv/Scripts/python.exe subapps/sensenova_convert/convert_sensenova.py \\
        --source D:/sensenova_staging/base \\
        --output M:/model/sensenova/sensenova_int8.safetensors \\
        --wait-for-download
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND = os.path.join(REPO_ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import psutil  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from core.models.common.int8_runtime_quantize import (  # noqa: E402
    FORMAT_MIN_ALIGN,
    INT8_MIN_WORK_K,
    INT8_MIN_WORK_N,
    audit_document,
)
from core.models.ideogram4.vendor.fp8_linear import (  # noqa: E402
    quantize_weight_to_fp8,
)
from core.models.ideogram4.vendor.int8_linear import (  # noqa: E402
    INT8_SCALE_SUFFIX,
    quantize_weight_to_int8,
    weight_crest_factors,
)
from core.models.common.quantized_export import (  # noqa: E402
    DEFAULT_EXPORT_SHARD_BYTES,
    EXPORT_LAYOUTS,
    ShardWriter,
    link_siblings,
    sensenova_export_metadata,
)
from core.models.common.single_file_format import (  # noqa: E402
    _INDEX_SUFFIX,
    _SHARD_SUFFIX,
    is_index_path,
)

MIN_ALIGN = FORMAT_MIN_ALIGN["int8"]  # 8, torch._int_mm's k/n divisibility floor

# This arch's registered on-disk layout (backend/core/models/common/quantized_export.py).
# ``offline_prefix`` is what every written key is prepended with; using the
# registered value rather than a local constant keeps this script from being
# able to drift from what ``ModelLoader``/``generation_utils`` actually read.
SENSENOVA_LAYOUT = EXPORT_LAYOUTS["sensenova"]
OUT_PREFIX = str(SENSENOVA_LAYOUT["offline_prefix"])

# Both MoT branches, both Linear groups. ``_mot_gen`` sits on the LINEAR name
# for self_attn (q_proj_mot_gen) but on the PARENT MODULE for mlp
# (mlp_mot_gen.gate_proj) -- one regex, two alternations, matching that.
# Confirmed against the vendored source now in this repo:
# backend/core/models/sensenova/vendor/modeling_qwen3.py
QUANTIZE_KEY_RE = re.compile(
    r"^language_model\.model\.layers\.\d+\."
    r"(?:self_attn\.(?:q|k|v|o)_proj(?:_mot_gen)?"
    r"|mlp(?:_mot_gen)?\.(?:gate|up|down)_proj)"
    r"\.weight$"
)

# 14 quantize-eligible Linears per decoder layer: self_attn {q,k,v,o}_proj (4)
# doubled for the MoT gen branch (8), plus mlp {gate,up,down}_proj (3) doubled
# likewise (6). Used both to gate a live download (wait_for_complete_download)
# and, hard, right before conversion starts (main): a differently-keyed
# revision would match zero (or some other wrong count) of QUANTIZE_KEY_RE and
# must not silently produce an all-bf16 file that claims to be the int8
# conversion.
QUANTIZE_TARGETS_PER_LAYER = 14

# Small companion files, copied (not junctioned) next to the written index.
TOKENIZER_AND_CONFIG_FILES = (
    "config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def _fmt_gb(nbytes: int) -> str:
    return f"{nbytes / (1024 ** 3):.3f} GB"


def _rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def _rel_rms(reference: torch.Tensor, approx: torch.Tensor) -> float:
    """Relative RMS error of ``approx`` against ``reference``, in float32.

    Local copy of ``int8_runtime_quantize._rel_rms`` (private there, so not
    imported): both callers need the identical formula for
    ``int8_audit.json`` to be diffable against a runtime conversion's
    ``audit_document``, but the formula itself is eight lines and not worth a
    cross-module private import.
    """
    ref = reference.to(torch.float32)
    err = approx.to(torch.float32) - ref
    denom = ref.pow(2).mean().sqrt()
    if not torch.isfinite(denom) or denom == 0:
        return float("nan")
    return float(err.pow(2).mean().sqrt() / denom)


class PeakRss:
    """Tracks the process's peak RSS across explicit ``sample()`` calls.

    Not a background sampler: RSS is read only at call sites bracketing each
    tensor's processing, which is enough to catch a leak (peak stays flat, not
    proportional to bytes processed) without paying a monitoring thread.
    """

    def __init__(self) -> None:
        self.peak_gb = _rss_gb()

    def sample(self) -> float:
        cur = _rss_gb()
        if cur > self.peak_gb:
            self.peak_gb = cur
        return cur


# ---------------------------------------------------------------------------
# Source resolution + download-completeness gate
# ---------------------------------------------------------------------------

def _index_path(staging_dir: str) -> str:
    return os.path.join(staging_dir, "model.safetensors.index.json")


def _load_index(index_path: str) -> Tuple[Dict[str, str], dict]:
    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)
    return dict(index.get("weight_map", {}) or {}), dict(index.get("metadata", {}) or {})


def _load_hf_config(staging_dir: str) -> Optional[dict]:
    path = os.path.join(staging_dir, "config.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def expected_quantize_target_count(hf_config: dict) -> int:
    """Number of ``QUANTIZE_KEY_RE`` matches this checkpoint's own geometry implies.

    Derived from ``hf_config["llm_config"]["num_hidden_layers"]``, NOT from a
    hardcoded constant re-derived from a specific index.json (that would be
    circular: it could only ever validate the exact file it was copied from,
    and would reject a legitimately different revision with a different layer
    count).
    """
    n_layers = int(hf_config["llm_config"]["num_hidden_layers"])
    return n_layers * QUANTIZE_TARGETS_PER_LAYER


def _quantize_key_match_count(weight_map: Dict[str, str]) -> int:
    return sum(1 for key in weight_map if QUANTIZE_KEY_RE.match(key))


def _shard_is_complete(shard_path: str) -> bool:
    """True iff ``shard_path``'s header parses and its declared extents fit the file.

    ``safe_open`` parses the header WITHOUT reading tensor data (mmap-backed),
    but it validates the header's declared byte extents against the actual
    file size while opening: a truncated (still-downloading) file whose header
    finished writing before the data did raises ``SafetensorError: ... file
    not fully covered`` right there. There is no separate tensor-read pass
    here -- an earlier version of this function called ``fh.get_tensor(key)``
    on every key hoping to "catch truncation", but ``get_tensor`` is a
    zero-copy mmap view (measured: RSS grew 0 for a 192 MB tensor) and the
    truncation it appeared to catch was always caught one line earlier, by
    ``safe_open`` itself.
    """
    try:
        with safe_open(shard_path, framework="pt", device="cpu") as fh:
            if not list(fh.keys()):
                return False
    except Exception:
        return False
    return True


def wait_for_complete_download(
    staging_dir: str,
    *,
    poll_s: int = 30,
    timeout_s: Optional[int] = None,
) -> Tuple[str, Dict[str, str], dict]:
    """Block until the staged checkpoint is complete, or ``timeout_s`` elapses.

    "Complete" means: the index exists, its quantize-target key count matches
    ``expected_quantize_target_count`` (derived from ``config.json``, once
    that file is on disk), and every shard it names exists on disk and passes
    ``_shard_is_complete``. Returns ``(index_path, weight_map, index_metadata)``.
    Raises ``TimeoutError`` if ``timeout_s`` is given and exceeded;
    ``timeout_s=None`` waits forever (the caller is expected to run this in
    the background and be notified). Raises ``RuntimeError`` immediately
    (not a retryable condition) if the geometry check fails once every shard
    is present and complete -- more polling cannot fix a structural mismatch.
    """
    index_path = _index_path(staging_dir)
    t0 = time.time()
    attempt = 0
    while True:
        attempt += 1
        if os.path.isfile(index_path):
            try:
                weight_map, metadata = _load_index(index_path)
            except Exception as exc:
                weight_map, metadata = {}, {}
                print(f"[SenseNova] attempt {attempt}: index.json present but unreadable "
                      f"({type(exc).__name__}: {exc}); retrying")
            else:
                shards = sorted(set(weight_map.values()))
                missing = [s for s in shards if not os.path.isfile(os.path.join(staging_dir, s))]
                if missing:
                    print(f"[SenseNova] attempt {attempt}: {len(missing)}/{len(shards)} "
                          f"shard file(s) not yet present (e.g. {missing[0]})")
                else:
                    incomplete = [s for s in shards
                                  if not _shard_is_complete(os.path.join(staging_dir, s))]
                    if incomplete:
                        print(f"[SenseNova] attempt {attempt}: {len(incomplete)}/{len(shards)} "
                              f"shard file(s) present but still writing (e.g. {incomplete[0]})")
                    else:
                        hf_config = _load_hf_config(staging_dir)
                        if hf_config is None:
                            print(f"[SenseNova] attempt {attempt}: all {len(shards)} shard(s) "
                                  f"complete but config.json is not yet on disk; waiting for it "
                                  f"to derive the expected quantize-target tensor count")
                        else:
                            expected = expected_quantize_target_count(hf_config)
                            matched = _quantize_key_match_count(weight_map)
                            if matched != expected:
                                raise RuntimeError(
                                    f"{index_path} names {len(weight_map)} tensor(s), of which "
                                    f"{matched} match {QUANTIZE_KEY_RE.pattern!r}, but config.json's "
                                    f"llm_config.num_hidden_layers implies exactly {expected}. "
                                    f"Every shard is present and complete, so this is not a "
                                    f"download-in-progress condition -- refusing rather than "
                                    f"polling forever against a structural mismatch.")
                            print(f"[SenseNova] download complete: {len(weight_map)} tensors "
                                  f"({matched} quantize-target matches, as expected) across "
                                  f"{len(shards)} shard(s), confirmed after {attempt} poll(s) / "
                                  f"{time.time() - t0:.0f}s")
                            return index_path, weight_map, metadata
        else:
            print(f"[SenseNova] attempt {attempt}: {index_path} not yet present")
        if timeout_s is not None and time.time() - t0 > timeout_s:
            raise TimeoutError(
                f"SenseNova checkpoint at {staging_dir} was not complete after "
                f"{timeout_s}s ({attempt} poll(s)); staging is left untouched, re-run "
                f"once the download finishes")
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# Key census
# ---------------------------------------------------------------------------

def classify_key(key: str) -> str:
    """One of: quantize_candidate, fm_modules_bf16, vision_bf16, lm_head_bf16,
    embed_bf16, norm_bf16, other_bf16. Declarative, matching the brief's
    never-quantize list; not itself a filter -- ``select_quant_target`` is.
    """
    if QUANTIZE_KEY_RE.match(key):
        return "quantize_candidate"
    if "fm_modules" in key:
        # Catches fm_head, timestep_embedder, noise_scale_embedder AND the
        # gen-branch vision tower, which lives at fm_modules.vision_model_mot_gen.*
        # (an nn.ModuleDict entry), not as a vision_model_mot_gen.* top-level sibling.
        return "fm_modules_bf16"
    if key.startswith("vision_model."):
        return "vision_bf16"
    if "lm_head" in key:
        return "lm_head_bf16"
    if "embed_tokens" in key:
        return "embed_bf16"
    if "norm" in key:
        return "norm_bf16"
    return "other_bf16"


def run_census(weight_map: Dict[str, str], staging_dir: str) -> Dict[str, Dict]:
    """Header-only per-key dtype/shape + category, grouped by shard for one open each.

    Returns ``{key: {"category", "dtype", "shape"}}``.
    """
    by_shard: Dict[str, List[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)

    info: Dict[str, Dict] = {}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(os.path.join(staging_dir, shard), framework="pt", device="cpu") as fh:
            for key in keys:
                sl = fh.get_slice(key)
                info[key] = {
                    "category": classify_key(key),
                    "dtype": sl.get_dtype(),
                    "shape": tuple(sl.get_shape()),
                    "shard": shard,
                }
    return info


def select_quant_target(key: str, shape: Tuple[int, ...]) -> Tuple[bool, str]:
    """``(will_quantize, reason)`` for a ``quantize_candidate`` key.

    Alignment-only, per the brief -- no crest/e4m3 fallback, no min-work-gate
    filter. ``in_f`` is ``shape[1]``, ``out_f`` is ``shape[0]``
    (safetensors/PyTorch Linear weight convention, ``(out, in)``).
    """
    if len(shape) != 2:
        return False, f"expected a 2-D Linear weight, got {shape}"
    out_f, in_f = shape
    if in_f % MIN_ALIGN or out_f % MIN_ALIGN:
        # The real reason (int8_runtime_quantize.py's shape-filter rule, not
        # a loader limitation): Int8Linear loads ANY shape -- an unaligned one
        # merely declines the torch._int_mm fast path and always runs
        # _dequant_forward, i.e. it buys quantization error for zero speedup.
        return False, (f"unaligned {in_f}x{out_f} (not a multiple of {MIN_ALIGN}): "
                        f"can never reach the int8 fast GEMM path, so quantizing it "
                        f"would buy error for no speed; left bf16")
    below_gate = in_f < INT8_MIN_WORK_K or out_f < INT8_MIN_WORK_N
    note = (f"; below the runtime min-work gate (k>={INT8_MIN_WORK_K}, n>={INT8_MIN_WORK_N}) "
            f"but quantized anyway -- this script applies alignment only" if below_gate else "")
    return True, f"aligned {in_f}x{out_f}{note}"


def print_census_report(info: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Print the matched/unmatched key groups; return them grouped by category
    (quantize_candidate further split into quantize/align_failed).
    """
    groups: Dict[str, List[str]] = {}
    for key, row in info.items():
        groups.setdefault(row["category"], []).append(key)

    print(f"\n[SenseNova] key census: {len(info)} tensor(s) total")
    quant_will, quant_align_failed = [], []
    for key in sorted(groups.get("quantize_candidate", [])):
        will, reason = select_quant_target(key, info[key]["shape"])
        (quant_will if will else quant_align_failed).append(key)
        info[key]["will_quantize"] = will
        info[key]["reason"] = reason

    for cat in sorted(groups):
        keys = sorted(groups[cat])
        print(f"[SenseNova]   {cat}: {len(keys)} tensor(s), e.g. {keys[0]}")
    print(f"[SenseNova]   -> of quantize_candidate: {len(quant_will)} will be int8, "
          f"{len(quant_align_failed)} fail alignment and stay bf16")
    for key in quant_align_failed:
        print(f"[SenseNova]     align-failed: {key}: {info[key]['reason']}")

    groups["quantize_int8"] = quant_will
    groups["quantize_align_failed"] = quant_align_failed
    return groups


# ---------------------------------------------------------------------------
# Byte accounting / disk space
# ---------------------------------------------------------------------------

def predicted_output_bytes(
    info: Dict[str, Dict],
    groups: Dict[str, List[str]],
    *,
    keep_fp32_buffers: bool = False,
) -> int:
    """Exact predicted output size from the header census, no data read."""
    total = 0
    quant_set = set(groups["quantize_int8"])
    for key, row in info.items():
        numel = 1
        for d in row["shape"]:
            numel *= d
        if key in quant_set:
            out_f = row["shape"][0]
            total += numel * 1 + out_f * 4  # int8 weight + fp32 per-row scale
        elif keep_fp32_buffers and row["dtype"] == "F32":
            total += numel * 4  # kept fp32, --keep-fp32-buffers
        else:
            total += numel * 2  # bf16, whatever the source dtype was
    return total


def check_disk_space(output_dir: str, predicted_bytes: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    free = shutil.disk_usage(output_dir).free
    # 10% headroom: shard temp-file renames and the audit/report files are small,
    # but the last shard's tmp and final copies can coexist briefly during close().
    needed = int(predicted_bytes * 1.10)
    print(f"[SenseNova] predicted output size {_fmt_gb(predicted_bytes)}, "
          f"{_fmt_gb(free)} free at {output_dir}")
    if free < needed:
        raise RuntimeError(
            f"not enough free space at {output_dir}: need ~{_fmt_gb(needed)} "
            f"(predicted {_fmt_gb(predicted_bytes)} + 10% headroom), have {_fmt_gb(free)}")


# ---------------------------------------------------------------------------
# Overwrite handling
# ---------------------------------------------------------------------------

def _existing_artifacts(output_path: str) -> List[str]:
    """Every file a previous run at this destination stem could have left:
    the single-file form, the index, every numbered shard (whatever count it
    was written with), leftover ``-part*.tmp.safetensors`` from a run killed
    mid-flush, and a previous audit sidecar.
    """
    directory = os.path.dirname(output_path)
    stem = os.path.basename(output_path)
    if stem.endswith(_SHARD_SUFFIX):
        stem = stem[: -len(_SHARD_SUFFIX)]
    found: List[str] = []
    for candidate in (
        os.path.join(directory, f"{stem}{_SHARD_SUFFIX}"),
        os.path.join(directory, f"{stem}{_INDEX_SUFFIX}"),
        os.path.join(directory, f"{stem}.int8_audit.json"),
    ):
        if os.path.isfile(candidate):
            found.append(candidate)
    if os.path.isdir(directory):
        prefix = f"{stem}-"
        for name in os.listdir(directory):
            if name.startswith(prefix) and (
                name.endswith(_SHARD_SUFFIX) or name.endswith(".tmp.safetensors")
            ):
                found.append(os.path.join(directory, name))
    return sorted(set(found))


def _clear_existing_artifacts(output_path: str, *, overwrite: bool) -> None:
    """Refuse (default) or delete (``--overwrite``) a previous run's output.

    Deletion happens UP FRONT, before the new run writes anything, rather than
    writing a full new copy at a temp name and swapping: the destination's
    free space is not assumed to hold old + new simultaneously (a re-run with
    a different ``--max-shard-bytes`` produces differently-NAMED shards, e.g.
    ``-of-00004`` instead of ``-of-00005``, so ``os.replace`` would land on new
    paths and never touch the stale ones -- they would otherwise sit beside
    the new output forever, consuming up to 2x the disk).
    """
    existing = _existing_artifacts(output_path)
    if not existing:
        return
    if not overwrite:
        raise FileExistsError(
            f"{len(existing)} existing artifact(s) at this destination "
            f"(e.g. {existing[0]}); pass --overwrite to replace them, or choose "
            f"another --output")
    print(f"[SenseNova] --overwrite: removing {len(existing)} existing artifact(s) "
          f"before writing: {existing}")
    for path in existing:
        os.remove(path)


# ---------------------------------------------------------------------------
# Streaming conversion
# ---------------------------------------------------------------------------

def convert(
    staging_dir: str,
    weight_map: Dict[str, str],
    info: Dict[str, Dict],
    groups: Dict[str, List[str]],
    output_path: str,
    hf_config: dict,
    *,
    max_shard_bytes: int,
    keep_fp32_buffers: bool = False,
) -> Dict[str, object]:
    """Stream-convert and write every key under ``OUT_PREFIX`` (the prefix
    ``EXPORT_LAYOUTS["sensenova"]`` declares).
    """
    quant_set = set(groups["quantize_int8"])
    align_failed_set = set(groups["quantize_align_failed"])

    metadata = dict(sensenova_export_metadata(hf_config))
    metadata["quant_format"] = "int8_perrow"
    metadata["quant_origin"] = "sensenova_offline_conversion_script"
    # NOT the absolute staging path: that is a path on someone else's machine
    # (see quantized_export.ideogram4_export_metadata's docstring for the same
    # rule applied to `_name_or_path`). A role descriptor is enough provenance
    # without being machine-identifying.
    metadata["quant_source"] = "offline_hf_checkpoint_staging_dir"
    metadata["quant_converted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    writer = ShardWriter(output_path, metadata, max_shard_bytes)
    rss = PeakRss()
    # Measured, not asserted: a real run peaked at 20.3 GB RSS (well inside the
    # ~30 GB budget) despite no single tensor or buffer exceeding a few GB
    # (largest tensor 1.16 GB bf16, ShardWriter buffer up to max_shard_bytes).
    # WHY is not verified here -- see the module's git history / audit notes
    # for the candidate mechanisms (CPU allocator retention vs. mmap page
    # residency across the ~47 GB sequential read); this function reports the
    # measured number only.
    print(f"[SenseNova] host RAM: no single tensor or buffer should individually "
          f"exceed a few GB (largest tensor 1.16 GB bf16, ShardWriter buffer "
          f"{_fmt_gb(max_shard_bytes)}); measured peak RSS runs above that -- see "
          f"the final report for the actual number. RSS at start: {rss.peak_gb:.3f} GB")

    by_shard: Dict[str, List[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)

    counts = {
        "int8": 0,
        "bf16_kept": 0,
        "bf16_cast_from_fp32": 0,
        "fp32_kept": 0,
        "align_failed_bf16": 0,
    }
    narrowed_by_category: Dict[str, int] = {}
    narrowed_examples: Dict[str, str] = {}
    audit_rows: List[Dict] = []
    written_manifest: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
    total_in_bytes = 0
    t0 = time.perf_counter()
    n_done = 0
    n_total = len(info)

    try:
        for shard in sorted(by_shard):
            keys = by_shard[shard]
            print(f"[SenseNova] reading {shard} ({len(keys)} tensor(s))")
            with safe_open(os.path.join(staging_dir, shard), framework="pt", device="cpu") as fh:
                for key in keys:
                    tensor = fh.get_tensor(key)
                    total_in_bytes += tensor.numel() * tensor.element_size()
                    row = info[key]

                    out_key = f"{OUT_PREFIX}{key}"
                    if key in quant_set:
                        # Un-narrowed: quantize_weight_to_int8 casts to float32
                        # internally regardless of input dtype, so pre-casting a
                        # fp32 source tensor to bf16 first only adds a second
                        # rounding for free error (measured 0.00947 int8-rel-rms
                        # un-narrowed vs 0.00953 narrowed-through-bf16-first).
                        q, scale = quantize_weight_to_int8(tensor)
                        writer.add(out_key, q.contiguous())
                        scale_key = f"{out_key[: -len('.weight')]}{INT8_SCALE_SUFFIX}"
                        writer.add(scale_key, scale.contiguous())
                        written_manifest[out_key] = ("I8", tuple(q.shape))
                        written_manifest[scale_key] = ("F32", tuple(scale.shape))
                        counts["int8"] += 1

                        # Audit sidecar: both candidate quantizations' errors are
                        # ~free to compute while the tensor is already resident.
                        # "chosen" records what THIS script actually did
                        # (alignment-only), not a re-derived crest/measured
                        # decision -- e4m3_rel_rms is recorded for provenance /
                        # diffability against a runtime conversion, it is not a
                        # second opinion this script acts on.
                        crest = weight_crest_factors(tensor)
                        q_f8, s_f8 = quantize_weight_to_fp8(tensor)
                        err_i8 = _rel_rms(tensor, q.to(torch.float32) * scale.unsqueeze(1))
                        err_f8 = _rel_rms(tensor, q_f8.to(torch.float32) * s_f8.unsqueeze(1))
                        audit_rows.append({
                            "name": key,
                            "shape": list(tensor.shape),
                            "int8_rel_rms": err_i8,
                            "e4m3_rel_rms": err_f8,
                            "advantage_int8_over_e4m3": (err_f8 / err_i8) if err_i8 else float("inf"),
                            "crest_mean": float(crest.mean()),
                            "crest_p99": float(crest.quantile(0.99)) if crest.numel() > 1 else float(crest.mean()),
                            "crest_max": float(crest.amax()),
                            "chosen": "int8",
                            "reason": row.get("reason", "aligned"),
                        })
                        del q_f8, s_f8
                    else:
                        was_fp32 = tensor.dtype == torch.float32
                        if was_fp32 and keep_fp32_buffers:
                            writer.add(out_key, tensor.contiguous())
                            written_manifest[out_key] = ("F32", tuple(tensor.shape))
                            counts["fp32_kept"] += 1
                        else:
                            if tensor.dtype != torch.bfloat16:
                                tensor = tensor.to(torch.bfloat16)
                            writer.add(out_key, tensor.contiguous())
                            written_manifest[out_key] = ("BF16", tuple(tensor.shape))
                            if key in align_failed_set:
                                counts["align_failed_bf16"] += 1
                            elif was_fp32:
                                counts["bf16_cast_from_fp32"] += 1
                                cat = row["category"]
                                narrowed_by_category[cat] = narrowed_by_category.get(cat, 0) + 1
                                narrowed_examples.setdefault(cat, key)
                            else:
                                counts["bf16_kept"] += 1
                    del tensor
                    n_done += 1
                    rss.sample()
            if n_done % 200 < len(keys):
                print(f"[SenseNova]   {n_done}/{n_total} tensors, RSS now {rss.peak_gb:.3f} GB peak")

        written_path = writer.close()
    except BaseException:
        writer.abort()
        raise
    elapsed = time.perf_counter() - t0

    audit_document_body = audit_document(audit_rows, {
        "arch": "sensenova",
        "format": "int8",
        "mode": "offline_alignment_only",
        "min_align": MIN_ALIGN,
        "min_work_k": INT8_MIN_WORK_K,
        "min_work_n": INT8_MIN_WORK_N,
        "selection_rule": (
            "alignment-only (no crest/e4m3 fallback, no min-work-gate filter); "
            "int8_rel_rms/e4m3_rel_rms are recorded for provenance and to allow "
            "a diff against a future runtime conversion, they do not select the "
            "format here -- 'chosen' is always 'int8' for every row in this file"
        ),
        "exported_to": written_path,
    })
    stem = os.path.basename(output_path)
    if stem.endswith(_SHARD_SUFFIX):
        stem = stem[: -len(_SHARD_SUFFIX)]
    audit_path = os.path.join(os.path.dirname(written_path), f"{stem}.int8_audit.json")
    with open(audit_path, "w", encoding="utf-8") as fh:
        json.dump(audit_document_body, fh, indent=1)

    # Companions: config + tokenizer files, copied next to the written index.
    out_dir = os.path.dirname(written_path)
    linked = link_siblings(staging_dir, out_dir, names=TOKENIZER_AND_CONFIG_FILES)

    return {
        "written_path": written_path,
        "counts": counts,
        "narrowed_by_category": narrowed_by_category,
        "narrowed_examples": narrowed_examples,
        "elapsed_s": elapsed,
        "total_in_bytes": total_in_bytes,
        "total_out_bytes": writer.total_bytes,
        "peak_rss_gb": rss.peak_gb,
        "written_manifest": written_manifest,
        "linked_siblings": linked,
        "audit_path": audit_path,
        "audit_document": audit_document_body,
    }


# ---------------------------------------------------------------------------
# Header-only round-trip verification
# ---------------------------------------------------------------------------

def verify_roundtrip(written_path: str, written_manifest: Dict[str, Tuple[str, Tuple[int, ...]]]) -> None:
    """Header-only: confirm the written artifact's key set and every dtype/shape.

    Handles BOTH forms ``ShardWriter.close()`` can return: a
    ``<stem>.safetensors.index.json`` (multiple shards) or a bare
    ``<stem>.safetensors`` with no index (a single shard -- reachable via
    ``--max-shard-bytes`` large enough to hold the whole output at once).
    Deliberately NOT ``single_file_format.read_state_dict`` -- it always loads
    every tensor into RAM (no header-only mode), which would defeat the whole
    RAM budget this script exists to respect on a ~20 GiB artifact.
    """
    if is_index_path(written_path):
        with open(written_path, encoding="utf-8") as fh:
            index = json.load(fh)
        weight_map: Dict[str, str] = index.get("weight_map", {}) or {}
        model_type = (index.get("metadata", {}) or {}).get("model_type")
        directory = os.path.dirname(written_path)
        by_shard: Dict[str, List[str]] = {}
        for key, shard in weight_map.items():
            by_shard.setdefault(shard, []).append(key)
        shard_paths = {shard: os.path.join(directory, shard) for shard in by_shard}
    else:
        with safe_open(written_path, framework="pt", device="cpu") as fh:
            model_type = (fh.metadata() or {}).get("model_type")
            keys = list(fh.keys())
        shard_name = os.path.basename(written_path)
        weight_map = {k: shard_name for k in keys}
        by_shard = {shard_name: keys}
        shard_paths = {shard_name: written_path}

    if model_type != "sensenova":
        raise AssertionError(f"written artifact metadata['model_type'] != 'sensenova' (got {model_type!r})")

    expected_keys = set(written_manifest)
    actual_keys = set(weight_map)
    if expected_keys != actual_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        raise AssertionError(
            f"round-trip key mismatch: {len(missing)} missing, {len(extra)} extra "
            f"(e.g. missing={list(missing)[:3]}, extra={list(extra)[:3]})")

    mismatches: List[str] = []
    for shard, keys in by_shard.items():
        with safe_open(shard_paths[shard], framework="pt", device="cpu") as fh:
            for key in keys:
                sl = fh.get_slice(key)
                exp_dtype, exp_shape = written_manifest[key]
                got_dtype, got_shape = sl.get_dtype(), tuple(sl.get_shape())
                if got_dtype != exp_dtype or got_shape != exp_shape:
                    mismatches.append(
                        f"{key}: wrote {exp_dtype} {exp_shape}, read back {got_dtype} {got_shape}")
    if mismatches:
        raise AssertionError(f"{len(mismatches)} round-trip mismatch(es): {mismatches[:5]}")
    print(f"[SenseNova] round-trip OK: {len(expected_keys)} key(s) verified header-only "
          f"across {len(by_shard)} shard(s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="staging directory holding model.safetensors.index.json + shards "
                         "+ config.json + tokenizer files (e.g. D:/sensenova_staging/base)")
    ap.add_argument("--output", required=True,
                    help="destination <stem>.safetensors (shards + index written beside it)")
    ap.add_argument("--max-shard-bytes", type=int, default=DEFAULT_EXPORT_SHARD_BYTES)
    ap.add_argument("--wait-for-download", action="store_true",
                    help="poll --source until the checkpoint is complete instead of failing "
                         "immediately on missing/partial shards")
    ap.add_argument("--wait-poll-s", type=int, default=30)
    ap.add_argument("--wait-timeout-s", type=int, default=None,
                    help="give up waiting after this many seconds (default: wait forever)")
    ap.add_argument("--census-only", action="store_true",
                    help="print the key census and exit without writing anything")
    ap.add_argument("--overwrite", action="store_true",
                    help="delete a previous run's artifacts at --output's stem before writing "
                         "(refuses if any exist and this is not passed)")
    ap.add_argument("--keep-fp32-buffers", action="store_true",
                    help="write non-quantized conditioning tensors that are fp32 in the source "
                         "(the whole gen-branch MoT stack for layers 0-38, fm_modules embedders) "
                         "as fp32 instead of narrowing them to bf16")
    ap.add_argument("--allow-align-failed-fallback", action="store_true",
                    help="if any quantize_candidate key fails the alignment gate, fall back to "
                         "writing it bf16 instead of refusing outright. Every shape in the "
                         "current checkpoint is 8-aligned, so this never triggers on a normal "
                         "run; it exists so a genuinely misaligned layer is not silently "
                         "downgraded by default")
    args = ap.parse_args()

    staging_dir = os.path.abspath(args.source)
    if args.wait_for_download:
        index_path, weight_map, _index_metadata = wait_for_complete_download(
            staging_dir, poll_s=args.wait_poll_s, timeout_s=args.wait_timeout_s)
    else:
        index_path = _index_path(staging_dir)
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"{index_path} not found; pass --wait-for-download to poll for it, or "
                f"finish the download first")
        weight_map, _index_metadata = _load_index(index_path)
        shards = sorted(set(weight_map.values()))
        incomplete = [s for s in shards
                      if not _shard_is_complete(os.path.join(staging_dir, s))]
        if incomplete:
            raise RuntimeError(
                f"{len(incomplete)}/{len(shards)} shard(s) not fully downloaded "
                f"(e.g. {incomplete[0]}); pass --wait-for-download to poll instead of "
                f"failing here")

    hf_config = _load_hf_config(staging_dir)
    if hf_config is None:
        raise FileNotFoundError(
            f"{os.path.join(staging_dir, 'config.json')} not found; needed both to derive the "
            f"expected quantize-target tensor count and to write into the output's metadata "
            f"(the loader reads it back as its PRIMARY geometry source)")

    # H1: hard gate BEFORE any census/conversion work. A differently-keyed
    # revision could match zero (or some other wrong count) of
    # QUANTIZE_KEY_RE and otherwise sail through as an all-bf16 file that
    # still claims (via metadata) to be the int8 conversion.
    expected_targets = expected_quantize_target_count(hf_config)
    matched_targets = _quantize_key_match_count(weight_map)
    n_layers = int(hf_config["llm_config"]["num_hidden_layers"])
    if matched_targets != expected_targets:
        raise RuntimeError(
            f"quantize-target key census: {matched_targets} key(s) in {index_path} match "
            f"{QUANTIZE_KEY_RE.pattern!r}, but config.json's llm_config.num_hidden_layers="
            f"{n_layers} implies exactly {expected_targets} ({QUANTIZE_TARGETS_PER_LAYER} "
            f"quantize-eligible Linears per decoder layer). Refusing rather than producing a "
            f"file that claims to be the int8 conversion of a checkpoint whose keys do not "
            f"actually match this script's target pattern.")

    print(f"[SenseNova] source={staging_dir}")
    print(f"[SenseNova] quantize-target key census: {matched_targets} matched "
          f"(expected {expected_targets} from num_hidden_layers={n_layers} x "
          f"{QUANTIZE_TARGETS_PER_LAYER})")
    info = run_census(weight_map, staging_dir)
    groups = print_census_report(info)

    if groups["quantize_align_failed"] and not args.allow_align_failed_fallback:
        raise RuntimeError(
            f"{len(groups['quantize_align_failed'])} quantize_candidate key(s) failed the "
            f"alignment gate (listed above); refusing rather than silently falling back to "
            f"bf16 for them. Pass --allow-align-failed-fallback to keep the old permissive "
            f"behaviour.")

    if args.census_only:
        print("[SenseNova] --census-only: nothing written")
        return 0

    output_path = os.path.abspath(args.output)
    _clear_existing_artifacts(output_path, overwrite=args.overwrite)

    predicted_bytes = predicted_output_bytes(info, groups, keep_fp32_buffers=args.keep_fp32_buffers)
    check_disk_space(os.path.dirname(output_path), predicted_bytes)

    result = convert(staging_dir, weight_map, info, groups, output_path, hf_config,
                      max_shard_bytes=args.max_shard_bytes,
                      keep_fp32_buffers=args.keep_fp32_buffers)

    verify_roundtrip(result["written_path"], result["written_manifest"])

    counts = result["counts"]
    print("\n[SenseNova] conversion report")
    print(f"[SenseNova]   int8 (quantized):            {counts['int8']}")
    print(f"[SenseNova]   bf16 (kept, already bf16):    {counts['bf16_kept']}")
    print(f"[SenseNova]   bf16 (cast from fp32):        {counts['bf16_cast_from_fp32']}")
    print(f"[SenseNova]   fp32 (kept, --keep-fp32-buffers): {counts['fp32_kept']}")
    print(f"[SenseNova]   bf16 (alignment gate failed): {counts['align_failed_bf16']}")
    print(f"[SenseNova]   total tensors:                {sum(counts.values())}")
    if result["narrowed_by_category"]:
        print("[SenseNova]   fp32->bf16 narrowing, by category:")
        for cat, n in sorted(result["narrowed_by_category"].items()):
            print(f"[SenseNova]     {cat}: {n} tensor(s), e.g. {result['narrowed_examples'][cat]}")
    print(f"[SenseNova]   input bytes:  {_fmt_gb(result['total_in_bytes'])}")
    print(f"[SenseNova]   output bytes: {_fmt_gb(result['total_out_bytes'])} "
          f"(predicted {_fmt_gb(predicted_bytes)})")
    print(f"[SenseNova]   peak RSS: {result['peak_rss_gb']:.3f} GB")
    print(f"[SenseNova]   elapsed: {result['elapsed_s']:.1f}s")
    print(f"[SenseNova]   written: {result['written_path']}")
    print(f"[SenseNova]   audit sidecar: {result['audit_path']}")
    audit_rows = result["audit_document"]["layers"]
    if audit_rows:
        worst = sorted(audit_rows, key=lambda r: r["int8_rel_rms"], reverse=True)[:5]
        print("[SenseNova]   worst int8-vs-e4m3 layers (by int8_rel_rms):")
        for row in worst:
            print(f"[SenseNova]     {row['name']}: int8 {row['int8_rel_rms']:.5f} "
                  f"vs e4m3 {row['e4m3_rel_rms']:.5f}")
    print(f"[SenseNova]   companions copied: {result['linked_siblings']}")
    print(f"[SenseNova] load it with: source_type=safetensors source={result['written_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
