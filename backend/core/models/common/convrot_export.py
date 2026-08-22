"""Offline INT8 ConvRot export: rotate-then-quantize a SenseNova bf16 source
into the same on-disk contract MiniMax-H3's ConvRot release files use.

Reuses ``ShardWriter``/``sensenova_export_metadata`` from
``quantized_export.py`` but has no live pipeline of its own: it reads bf16
weights straight out of source safetensors shards and quantizes them itself.
See ``core.models.common.quantized_checkpoint_guard`` for the ``.comfy_quant``
marker contract this module's output implements.

Constraints that are load-bearing (not narration):
- ROTATION DTYPE: ``_build_hadamard`` builds its Hadamard matrix in the input
  tensor's dtype, so rotating in bf16 adds a second error source on top of
  the INT8 quantization. Every weight here is upcast to fp32 first
  (``ROTATION_DTYPE``) -- see ``rotation_precision_report`` for the measured
  cost of skipping that.
- LAYER SELECTION: never dequantize the plain-int8 file to regenerate a
  layer -- that would compound two roundings. Excluded/non-quantized tensors
  are copied verbatim from the plain-int8 file instead.
- HOST RAM: both sources are opened with ``safetensors.safe_open`` (mmap,
  single-tensor reads) and processed one bf16 shard at a time, so at most one
  shard's tensors plus ``ShardWriter``'s buffer are resident at once.
"""

from __future__ import annotations

import json
import os
import struct
import time
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from core.models.common.quantized_export import (
    DEFAULT_EXPORT_SHARD_BYTES,
    ShardWriter,
    sensenova_export_metadata,
)
from core.models.common.quantized_checkpoint_guard import encode_comfy_quant_marker
from core.models.common.single_file_format import (
    TRANSFORMER_PREFIX, _INDEX_SUFFIX, _SHARD_SUFFIX,
)

__all__ = [
    "ROTATION_DTYPE",
    "GROUP_SIZE",
    "LayerSpec",
    "read_safetensors_header",
    "derive_quantized_layers",
    "filter_convrot_eligible",
    "load_bf16_weight_map",
    "group_layers_by_shard",
    "rotation_precision_report",
    "convrot_marker_fields",
    "export_sensenova_convrot",
]

# comfy_kitchen's ``_build_hadamard`` requires a power of 4; 256 = 4**4.
GROUP_SIZE = 256
# Upcast every weight before rotation -- see the module docstring.
ROTATION_DTYPE = torch.float32


def read_safetensors_header(path: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """``(header, metadata)`` of a safetensors file. ZERO tensor bytes are read."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        if header_len <= 0 or header_len > 512 * 1024 * 1024:
            raise ValueError(f"implausible safetensors header length {header_len} in {path}")
        header = json.loads(fh.read(header_len).decode("utf-8"))
    metadata = header.pop("__metadata__", {}) or {}
    return header, metadata


class LayerSpec:
    """One quantized Linear: bare module path (ends in ``.weight``) + shape."""

    __slots__ = ("name", "out_features", "in_features")

    def __init__(self, name: str, out_features: int, in_features: int) -> None:
        self.name = name
        self.out_features = int(out_features)
        self.in_features = int(in_features)

    @property
    def stem(self) -> str:
        """``name`` with the trailing ``.weight`` removed."""
        assert self.name.endswith(".weight")
        return self.name[: -len(".weight")]

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"LayerSpec({self.name!r}, [{self.out_features}, {self.in_features}])"


def derive_quantized_layers(
    plain_int8_header: Dict[str, Any],
    *,
    audit: Optional[Dict[str, Any]] = None,
) -> List[LayerSpec]:
    """The quantized-layer set the existing plain-int8 checkpoint holds.

    ``audit`` (the ``.int8_audit.json`` sidecar), when present, is
    authoritative for names/shapes but is still cross-checked against the
    header's own I8 weight / F32 ``weight_scale`` pair, so a stale audit file
    cannot steer the export onto tensors that do not exist. With no audit,
    falls back to the same header scan used for that cross-check.
    """
    def _check_header_pair(name: str, out_features: int, in_features: int) -> None:
        wkey = f"{TRANSFORMER_PREFIX}{name}"
        skey = f"{TRANSFORMER_PREFIX}{name[:-len('.weight')]}.weight_scale"
        w = plain_int8_header.get(wkey)
        s = plain_int8_header.get(skey)
        if not isinstance(w, dict) or not isinstance(s, dict):
            raise ValueError(
                f"audit names quantized layer {name!r}, but the plain-int8 header has no "
                f"{wkey!r}/{skey!r} pair -- audit and checkpoint disagree")
        if w.get("dtype") != "I8" or list(w.get("shape", [])) != [out_features, in_features]:
            raise ValueError(
                f"{wkey!r}: header says {w.get('dtype')} {w.get('shape')}, audit says "
                f"I8 [{out_features}, {in_features}]")
        if s.get("dtype") != "F32":
            raise ValueError(f"{skey!r}: header says {s.get('dtype')}, expected F32")

    if audit is not None:
        layers: List[LayerSpec] = []
        for row in audit.get("layers", []):
            name = row["name"]
            out_features, in_features = (int(x) for x in row["shape"])
            _check_header_pair(name, out_features, in_features)
            layers.append(LayerSpec(name, out_features, in_features))
        return layers

    layers = []
    for key, entry in plain_int8_header.items():
        if not key.startswith(TRANSFORMER_PREFIX) or not key.endswith(".weight"):
            continue
        if entry.get("dtype") != "I8":
            continue
        name = key[len(TRANSFORMER_PREFIX):]
        scale_key = f"{TRANSFORMER_PREFIX}{name[:-len('.weight')]}.weight_scale"
        scale = plain_int8_header.get(scale_key)
        if not isinstance(scale, dict) or scale.get("dtype") != "F32":
            continue
        shape = entry.get("shape", [])
        if len(shape) != 2:
            continue
        out_features, in_features = (int(x) for x in shape)
        layers.append(LayerSpec(name, out_features, in_features))
    return layers


def filter_convrot_eligible(
    layers: Sequence[LayerSpec], *, group_size: int = GROUP_SIZE,
) -> Tuple[List[LayerSpec], List[Tuple[LayerSpec, str]]]:
    """Split ``layers`` into (ConvRot-eligible, [(layer, exclusion reason), ...])."""
    eligible: List[LayerSpec] = []
    excluded: List[Tuple[LayerSpec, str]] = []
    for layer in layers:
        if layer.in_features % group_size:
            excluded.append(
                (layer, f"in_features={layer.in_features} not divisible by group_size={group_size}"))
        else:
            eligible.append(layer)
    return eligible, excluded


def load_bf16_weight_map(bf16_root: str) -> Dict[str, str]:
    """``{tensor name: shard basename}`` from ``<bf16_root>/model.safetensors.index.json``."""
    index_path = os.path.join(bf16_root, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"no model.safetensors.index.json under {bf16_root!r}; the bf16 source download "
            f"is not finished")
    with open(index_path, encoding="utf-8") as fh:
        data = json.load(fh)
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: empty or missing 'weight_map'")
    return {str(k): str(v) for k, v in weight_map.items()}


def group_layers_by_shard(
    layers: Sequence[LayerSpec], weight_map: Dict[str, str],
) -> Tuple[Dict[str, List[LayerSpec]], List[str]]:
    """``{shard basename: [layers in write order]}``; also the list of missing names.

    A missing name means the audit/header layer set does not line up with the
    bf16 source's own key spelling -- raised by the caller rather than here,
    so the caller can report every missing key, not just the first.
    """
    by_shard: Dict[str, List[LayerSpec]] = {}
    missing: List[str] = []
    for layer in layers:
        shard = weight_map.get(layer.name)
        if shard is None:
            missing.append(layer.name)
            continue
        by_shard.setdefault(shard, []).append(layer)
    return by_shard, missing


def convrot_marker_fields(group_size: int = GROUP_SIZE) -> Dict[str, object]:
    """The ``.comfy_quant`` JSON this export declares -- the one contract
    ``ConvRotInt8Linear``/``_supported_int8_convrot_marker`` (MiniMax-H3) implement."""
    return {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": int(group_size)}


def rotate_and_quantize(weight: torch.Tensor, *, group_size: int = GROUP_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(int8 [out, in], scale float32 [out, 1])``, rotated in ``ROTATION_DTYPE``.

    The scale keeps ``quantize_int8_rowwise``'s natural ``[out, 1]`` shape
    (keepdim) rather than being squeezed to ``[out]``: both are accepted by
    the H3 marker validator (``scale_shape not in ([out], [out, 1])``) and by
    ``ConvRotInt8Linear``'s swap helper (``scale.numel() == out_features``,
    true either way), and ``[out, 1]`` is what ``quantize_int8_convrot_weight``
    hands back with no extra reshape.
    """
    from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight

    w = weight.to(dtype=ROTATION_DTYPE)
    q, scale = quantize_int8_convrot_weight(w, group_size, stochastic_rounding=0)
    return q.to(torch.int8), scale.to(torch.float32)


def rotation_precision_report(
    weight: torch.Tensor, *, group_size: int = GROUP_SIZE,
) -> Dict[str, float]:
    """Rel-L2 reconstruction error of a fp32-rotated vs bf16-rotated quantization,
    both measured against the same fp32 reference weight (not against each
    other), so the gap is exactly the cost of rotating in bf16.
    """
    from comfy_kitchen.backends.eager.quantization import (
        dequantize_int8_convrot_weight, quantize_int8_convrot_weight,
    )

    w32 = weight.to(torch.float32)
    q32, s32 = quantize_int8_convrot_weight(w32, group_size, stochastic_rounding=0)
    recon32 = dequantize_int8_convrot_weight(q32, s32, group_size)
    err32 = (recon32 - w32).norm() / w32.norm().clamp_min(1e-12)

    w16 = w32.to(torch.bfloat16)
    q16, s16 = quantize_int8_convrot_weight(w16, group_size, stochastic_rounding=0)
    recon16 = dequantize_int8_convrot_weight(q16, s16, group_size).to(torch.float32)
    err16 = (recon16 - w32).norm() / w32.norm().clamp_min(1e-12)

    return {
        "fp32_rotation_rel_l2": float(err32.item()),
        "bf16_rotation_rel_l2": float(err16.item()),
    }


def _oracle_probe(
    weight_ref: torch.Tensor,
    q_int8: torch.Tensor,
    scale: torch.Tensor,
    *,
    device: torch.device,
    group_size: int = GROUP_SIZE,
    n_rows: int = 16,
    seed: int,
) -> Dict[str, float]:
    """Compare ``comfy_kitchen.int8_linear(convrot=True)`` against a dense
    ``F.linear`` on random bf16 activations, both against ``weight_ref``.

    A plumbing-correctness check (shape/dtype/marker wiring), not a quality
    gate: ``n_rows`` random rows, one fixed seed per layer for reproducibility.
    ``seed`` must be a stable digest of ``layer.name`` (e.g. ``zlib.crc32``) --
    NOT Python's ``hash()``, which is salted per-process and would make the
    oracle numbers non-comparable across export runs.
    """
    from comfy_kitchen import int8_linear

    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n_rows, weight_ref.shape[1], generator=gen, device=device, dtype=torch.bfloat16)
    w_ref = weight_ref.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        y_ref = F.linear(x, w_ref)
        y_q = int8_linear(
            x, q_int8.to(device), scale.to(device), bias=None, out_dtype=x.dtype,
            convrot=True, convrot_groupsize=group_size,
        )
    diff = (y_q.float() - y_ref.float())
    max_abs = diff.abs().max()
    rel_l2 = diff.norm() / y_ref.float().norm().clamp_min(1e-12)
    return {"max_abs": float(max_abs.item()), "rel_l2": float(rel_l2.item())}


def export_sensenova_convrot(
    *,
    bf16_root: str,
    plain_int8_path: str,
    output_path: str,
    audit_path: Optional[str] = None,
    device: str = "cuda",
    group_size: int = GROUP_SIZE,
    max_shard_bytes: int = DEFAULT_EXPORT_SHARD_BYTES,
    oracle_report_path: Optional[str] = None,
    run_oracle: bool = True,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Write a SenseNova INT8 ConvRot checkpoint from the bf16 source.

    Streams shard-by-shard (one bf16 shard's eligible layers at a time); every
    other tensor is copied verbatim from ``plain_int8_path`` in a second pass
    (see module docstring for why never dequantize).
    """
    from safetensors import safe_open

    if not output_path.endswith(".safetensors"):
        raise ValueError(f"destination must end in '.safetensors': {output_path}")
    if not overwrite and os.path.exists(output_path):
        raise FileExistsError(f"{output_path} already exists; pass overwrite=True")
    # Deliberately NOT calling quantized_export.reject_quant_tokens_in_path here:
    # that guard is scoped to the Krea 2 loader's own path-token deny-list
    # (which the SenseNova loader never consults), and this filename legitimately
    # names its own format.

    header, header_metadata = read_safetensors_header(plain_int8_path)

    audit = None
    if audit_path and os.path.isfile(audit_path):
        with open(audit_path, encoding="utf-8") as fh:
            audit = json.load(fh)

    all_layers = derive_quantized_layers(header, audit=audit)
    eligible, excluded = filter_convrot_eligible(all_layers, group_size=group_size)
    print(f"[ConvRotExport] {len(all_layers)} quantized layer(s) in the plain-int8 checkpoint; "
          f"{len(eligible)} ConvRot-eligible (in_features % {group_size} == 0), "
          f"{len(excluded)} excluded")
    for layer, reason in excluded:
        print(f"[ConvRotExport]   excluded: {layer.name} ({reason})")

    weight_map = load_bf16_weight_map(bf16_root)
    by_shard, missing = group_layers_by_shard(eligible, weight_map)
    if missing:
        raise ValueError(
            f"{len(missing)} eligible layer(s) have no matching key in the bf16 source's "
            f"weight_map -- e.g. {missing[:5]}. The bf16 source's key spelling does not "
            f"match the plain-int8 checkpoint's layer names.")

    sensenova_config = {}
    raw_config = header_metadata.get("sensenova_config")
    if raw_config:
        sensenova_config = json.loads(raw_config)
    metadata: Dict[str, str] = dict(sensenova_export_metadata(sensenova_config))
    metadata["quant_format"] = "int8_tensorwise_convrot"
    metadata["convrot_groupsize"] = str(group_size)
    metadata["quant_origin"] = "offline_convrot_export"
    metadata["quant_exported_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    metadata["quant_source_bf16"] = "sensenova/SenseNova-U1.5-8B-MoT"
    metadata["quant_source_bf16_shards"] = str(len(set(weight_map.values())))
    metadata["quant_source_int8"] = os.path.basename(plain_int8_path)
    metadata["quantized_convrot_linears"] = str(len(eligible))
    metadata["quantized_excluded_linears"] = str(len(excluded))

    torch_device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    if device == "cuda" and torch_device.type != "cuda":
        print("[ConvRotExport] WARNING: CUDA requested but unavailable; running on CPU")

    writer = ShardWriter(output_path, metadata, max_shard_bytes)
    written_keys = set()
    oracle_rows: List[Dict[str, object]] = []
    marker = encode_comfy_quant_marker(convrot_marker_fields(group_size))
    t0 = time.perf_counter()
    try:
        for shard_idx, (shard_name, shard_layers) in enumerate(sorted(by_shard.items())):
            shard_path = os.path.join(bf16_root, shard_name)
            print(f"[ConvRotExport] shard {shard_idx + 1}/{len(by_shard)}: {shard_name} "
                  f"({len(shard_layers)} layer(s))")
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for layer in shard_layers:
                    w_bf16 = handle.get_tensor(layer.name)
                    if tuple(w_bf16.shape) != (layer.out_features, layer.in_features):
                        raise ValueError(
                            f"{shard_path}:{layer.name}: bf16 shape {tuple(w_bf16.shape)} != "
                            f"plain-int8-derived shape [{layer.out_features}, {layer.in_features}]")
                    w_gpu = w_bf16.to(torch_device)
                    q, scale = rotate_and_quantize(w_gpu, group_size=group_size)

                    stem = layer.stem
                    writer.add(f"{TRANSFORMER_PREFIX}{stem}.weight", q.to("cpu"))
                    writer.add(f"{TRANSFORMER_PREFIX}{stem}.weight_scale", scale.to("cpu"))
                    # ``.clone()``: every layer's marker is otherwise the SAME tensor
                    # object, which safetensors refuses to save (shared storage).
                    writer.add(f"{TRANSFORMER_PREFIX}{stem}.comfy_quant", marker.clone())
                    written_keys.add(f"{TRANSFORMER_PREFIX}{stem}.weight")
                    written_keys.add(f"{TRANSFORMER_PREFIX}{stem}.weight_scale")

                    if run_oracle:
                        row = _oracle_probe(
                            w_bf16, q, scale, device=torch_device, group_size=group_size,
                            seed=zlib.crc32(layer.name.encode("utf-8")),
                        )
                        row["layer"] = layer.name
                        oracle_rows.append(row)
                    del w_gpu, q, scale

        # Second pass: everything else, verbatim from the plain-int8 file.
        copied = 0
        with safe_open(plain_int8_path, framework="pt", device="cpu") as handle:
            for key in sorted(header.keys()):
                if key in written_keys:
                    continue
                writer.add(key, handle.get_tensor(key))
                copied += 1
        print(f"[ConvRotExport] copied {copied} tensor(s) verbatim from {plain_int8_path}")

        written = writer.close()
    except BaseException:
        writer.abort()
        raise
    elapsed = time.perf_counter() - t0

    oracle_summary: Dict[str, object] = {}
    if run_oracle and oracle_rows:
        max_abs_vals = [r["max_abs"] for r in oracle_rows]
        rel_l2_vals = [r["rel_l2"] for r in oracle_rows]
        worst_max_abs = max(oracle_rows, key=lambda r: r["max_abs"])
        worst_rel_l2 = max(oracle_rows, key=lambda r: r["rel_l2"])
        oracle_summary = {
            "n_layers": len(oracle_rows),
            "max_abs_mean": sum(max_abs_vals) / len(max_abs_vals),
            "max_abs_worst": worst_max_abs,
            "rel_l2_mean": sum(rel_l2_vals) / len(rel_l2_vals),
            "rel_l2_worst": worst_rel_l2,
            "rows": oracle_rows,
        }
        if written.endswith(_INDEX_SUFFIX):
            stem = written[: -len(_INDEX_SUFFIX)]
        elif written.endswith(_SHARD_SUFFIX):
            stem = written[: -len(_SHARD_SUFFIX)]
        else:  # pragma: no cover - ShardWriter always emits one of the two
            stem = written
        report_path = oracle_report_path or f"{stem}.int8_convrot_oracle.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(oracle_summary, fh, indent=1)
        print(f"[ConvRotExport] oracle report: {report_path} "
              f"(mean rel_l2={oracle_summary['rel_l2_mean']:.3e}, "
              f"worst rel_l2={worst_rel_l2['rel_l2']:.3e} on {worst_rel_l2['layer']})")

    print(f"[ConvRotExport] wrote {written} ({writer.total_bytes / 2**30:.2f} GB) in {elapsed:.1f}s")
    return {
        "output_path": written,
        "total_bytes": writer.total_bytes,
        "elapsed_s": elapsed,
        "eligible_layers": len(eligible),
        "excluded_layers": [(l.name, reason) for l, reason in excluded],
        "oracle_summary": {k: v for k, v in oracle_summary.items() if k != "rows"},
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bf16-root", required=True)
    ap.add_argument("--plain-int8", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--audit", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--no-oracle", dest="run_oracle", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    export_sensenova_convrot(
        bf16_root=args.bf16_root,
        plain_int8_path=args.plain_int8,
        output_path=args.output,
        audit_path=args.audit,
        device=args.device,
        group_size=args.group_size,
        run_oracle=args.run_oracle,
        overwrite=args.overwrite,
    )
