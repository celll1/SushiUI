#!/usr/bin/env python3
"""Quantize a transformer checkpoint to the repo's weight-only FP8 layout.

Produces a checkpoint that the NORMAL production loader path accepts: the FP8
Linear layout is exactly the one ``backend/core/models/ideogram4/vendor/fp8_linear.py``
defines and ``swap_linears_to_fp8`` gates on, so no loader change is needed.

    <name>.weight        float8_e4m3fn  (out, in)
    <name>.weight_scale  float32        (out,)      <- presence gates the swap
    <name>.bias          original dtype (out,)      [untouched]

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
``out_features`` is not a multiple of ``--min-align`` (16 by default). Rationale:
``Fp8Linear._scaled_mm_forward`` rejects unaligned shapes outright, so an
unaligned layer can never reach the fast path -- quantizing it would add
quantization error for exactly zero speed. (For Krea 2 this excludes one layer,
``text_fusion.projector``, which is 12x1.) The reference FP8 checkpoint this
format comes from -- ideogram-4-fp8 -- quantizes every Linear including the
input/output projections and the timestep MLP, so "all Linears" is the matching
convention, not a narrowed subset.

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
        --arch krea2 \
        --source "<bf16 model dir>/diffusion_pytorch_model.safetensors.index.json" \
        --output "<scratch dir>/krea2_fp8/krea2_fp8.safetensors" \
        --link-siblings "<bf16 model dir>"

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
# Private in the writer module on purpose (they are format constants, not API);
# imported rather than re-typed so a change to the on-disk convention cannot
# leave this tool emitting the old one.
from core.models.common.single_file_format import _INDEX_SUFFIX, _SHARD_SUFFIX  # noqa: E402

# Output shard threshold. Smaller than the repo default (10 GB) because the
# writer buffers a whole shard in RAM while the source shard is also resident.
DEFAULT_OUT_SHARD_BYTES = 4 * 1024 ** 3

SIBLING_DIRS = ("text_encoder", "vae", "tokenizer", "scheduler")


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------
#
# Each entry knows how to (a) build the module on the META device so its
# ``nn.Linear`` paths can be enumerated without allocating 13 B parameters, and
# (b) declare the key prefix and metadata the arch's own single-file loader
# expects. To add an arch: add one entry and nothing else.


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


ARCHS = {
    "krea2": {
        # sushiUI single-file layout: transformer weights live under this prefix.
        "prefix": "transformer.",
        "config": _krea2_config,
        "build_meta": _krea2_build_meta,
        "metadata": _krea2_metadata,
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
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Split the Linears into (quantize, [(skipped, reason)])."""
    targets: List[str] = []
    skipped: List[Tuple[str, str]] = []
    for name, (in_f, out_f) in sorted(linears.items()):
        if f"{name}.weight" not in present_keys:
            skipped.append((name, "no weight in checkpoint"))
            continue
        if min_align and (in_f % min_align or out_f % min_align):
            skipped.append((name, f"unaligned {in_f}x{out_f} (cannot reach the scaled-GEMM path)"))
            continue
        pattern = next((p for p in excludes if p.search(name)), None)
        if pattern is not None:
            skipped.append((name, f"excluded by /{pattern.pattern}/"))
            continue
        targets.append(name)
    return targets, skipped


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

def link_siblings(src_dir: str, dest_dir: str) -> List[str]:
    """Create directory junctions dest_dir/<name> -> src_dir/<name>.

    Junctions (``mklink /J``) need no administrator rights and work across local
    volumes; a symlink would need developer mode. POSIX falls back to symlinks.
    """
    made = []
    os.makedirs(dest_dir, exist_ok=True)
    for name in SIBLING_DIRS:
        target = os.path.join(src_dir, name)
        link = os.path.join(dest_dir, name)
        if not os.path.isdir(target):
            continue
        if os.path.exists(link):
            print(f"[fp8]   sibling '{name}' already present, leaving as is")
            continue
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
    ap.add_argument("--source", required=True,
                    help="bf16 checkpoint: shard index, single safetensors, or a directory")
    ap.add_argument("--output", required=True,
                    help="destination <stem>.safetensors (shards + index written beside it "
                         "when the result exceeds --max-shard-bytes)")
    ap.add_argument("--min-align", type=int, default=16,
                    help="skip Linears whose in/out features are not a multiple of this "
                         "(they can never take the scaled-GEMM path). 0 disables the check.")
    ap.add_argument("--exclude", action="append", default=[],
                    help="regex matched against the module path; repeatable")
    ap.add_argument("--max-shard-bytes", type=int, default=DEFAULT_OUT_SHARD_BYTES)
    ap.add_argument("--link-siblings", metavar="SRC_DIR",
                    help="create text_encoder/vae/tokenizer/scheduler junctions from SRC_DIR "
                         "next to the output so the loader's sibling probe resolves them")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be quantized and exit without writing")
    args = ap.parse_args()

    arch = ARCHS[args.arch]
    excludes = [re.compile(p) for p in args.exclude]

    print(f"[fp8] arch={args.arch} source={args.source}")
    shards, key_to_shard = _source_shards(args.source)
    print(f"[fp8] source has {len(key_to_shard)} tensors in {len(shards)} shard(s)")

    config = arch["config"](args.source)
    meta_model = arch["build_meta"](config)
    linears = linear_paths(meta_model)
    targets, skipped = select_targets(linears, set(key_to_shard), args.min_align, excludes)

    print(f"[fp8] {len(linears)} nn.Linear module(s); quantizing {len(targets)}, skipping {len(skipped)}")
    for name, reason in skipped:
        print(f"[fp8]   skip {name}: {reason}")

    if args.dry_run:
        print("[fp8] dry run: nothing written")
        return 0

    target_set = set(targets)
    prefix = arch["prefix"]
    metadata = arch["metadata"](config)
    metadata["fp8_quantized_linears"] = str(len(targets))
    metadata["fp8_source"] = os.path.abspath(args.source)

    writer = ShardWriter(args.output, metadata, args.max_shard_bytes)
    t0 = time.perf_counter()
    quantized = 0
    passthrough = 0
    for shard in shards:
        print(f"[fp8] reading {os.path.basename(shard)}")
        with safe_open(shard, framework="pt", device="cpu") as fh:
            for key in fh.keys():
                tensor = fh.get_tensor(key)
                base = key[: -len(".weight")] if key.endswith(".weight") else None
                if base is not None and base in target_set:
                    if tensor.dim() != 2:
                        raise RuntimeError(f"{key}: expected a 2-D Linear weight, got {tuple(tensor.shape)}")
                    q, scale = quantize_weight_to_fp8(tensor)
                    writer.add(f"{prefix}{key}", q.contiguous())
                    writer.add(f"{prefix}{base}{FP8_SCALE_SUFFIX}", scale.contiguous())
                    quantized += 1
                else:
                    writer.add(f"{prefix}{key}", tensor.contiguous())
                    passthrough += 1
                del tensor
    written = writer.close()
    elapsed = time.perf_counter() - t0

    print(f"[fp8] quantized {quantized} Linear weight(s), passed through {passthrough} tensor(s)")
    print(f"[fp8] wrote {written} ({writer.total_bytes / 2**30:.2f} GB) in {elapsed:.1f}s")
    if quantized != len(targets):
        print(f"[fp8] WARNING: expected {len(targets)} quantized weights, wrote {quantized}")

    if args.link_siblings:
        print("[fp8] linking companion component dirs")
        link_siblings(args.link_siblings, os.path.dirname(os.path.abspath(args.output)))

    print(f"[fp8] load it with: source_type=safetensors source={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
