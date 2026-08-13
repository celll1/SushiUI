"""Trained hidden-state projection pairing a small Qwen3-VL with MiniMax-H3.

A small Qwen3-VL encoder -- converted by ``te_gguf_convert`` or mapped straight
from its GGUF by ``te_gguf_native`` -- is 2560- (4B) or 4096-wide (8B) where the DiT's ``condition_proj`` takes 5120, so it is usable only
together with a projection trained for that exact (encoder, tap) pair. Running
the small encoder without one, or with the other size's one, is not a degraded
encode -- it is a wrong one, which is why every pairing check in
``resolve_te_projection`` refuses rather than warns.

Files live in ``<root>/clip_projections/`` and carry their own
``__metadata__`` (``d_in``/``d_out``/``tap``/``mlp_hidden``/``mlp_depth``);
discovery matches ``d_in`` to the encoder's declared ``hidden_size``.

Measured agreement with the 32B encoder, post-``token_refiner``, mean-removed
cosine: 4B 0.826, 8B 0.843, against -0.022 for a constant predictor (gate G0c).
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

MINIMAX_H3_TE_PROJECTION_DIRNAME = "clip_projections"

# ``mlp_depth`` this module implements: Linear(d_in, mlp_hidden) -> GELU(exact)
# -> Linear(mlp_hidden, d_out). Any other depth needs another forward.
_SUPPORTED_MLP_DEPTH = 1

_REQUIRED_TENSORS = (
    "W", "mean_in", "std_in", "mean_out", "std_out", "sink_out",
    "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias",
)

# The code every user-visible report of a substituted encoder carries.
TE_SUBSTITUTION_WARNING_CODE = "minimax_h3_substituted_text_encoder"

# Measured agreement per (encoder file, projection file), lowercased basenames.
# Gate G0c (`scratchpad/minimax_h3_te_small_gate_g0c.md`): 111 prompt-only
# presentations, post-`token_refiner`, row 0 excluded. `cosine` is mean-removed
# (raw cosine in 5120 dims is uninformative: a constant predictor scores 0.895
# there and -0.022 mean-removed). `rel_rms_floor` is the SAME encoder in another
# quantization, i.e. what a difference that is only rounding looks like.
#
# ONE table, so a newly measured pairing is added here and reaches the loader
# log, the generation warning and any future surface at once. A pairing absent
# from it is reported as unmeasured, never as measured-by-analogy.
MEASURED_TE_SUBSTITUTIONS: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("qwen3vl_4b_heretic_tap24_bf16.safetensors", "mmh3-4b-clipproj-celeb-mlp.safetensors"): {
        "reference": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "cosine": 0.826,
        "rel_rms": 0.214,
        "rel_rms_floor": 0.048,
        "presentations": 111,
    },
    ("qwen3vl_8b_instruct_tap24_bf16.safetensors", "mmh3-8b-clipproj-celeb-mlp.safetensors"): {
        "reference": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "cosine": 0.843,
        "rel_rms": 0.205,
        "rel_rms_floor": 0.048,
        "presentations": 111,
    },
}


def measured_te_substitution(te_path: str, projection_path: str) -> Optional[Dict[str, Any]]:
    """The recorded agreement for this exact pairing, or ``None``."""
    return MEASURED_TE_SUBSTITUTIONS.get(
        (os.path.basename(te_path).lower(), os.path.basename(projection_path).lower()))


def describe_te_substitution(te_path: str, projection_path: str) -> str:
    """One factual sentence pair naming the substitution and its measurement.

    Worded for both the loader's log line and the per-generation warning, so
    the two cannot drift.
    """
    encoder, projector = os.path.basename(te_path), os.path.basename(projection_path)
    head = (f"MiniMax-H3 text conditioning comes from {encoder} through the trained projection "
            f"{projector}, not from a released Qwen3-VL-32B text encoder.")
    measured = measured_te_substitution(te_path, projection_path)
    if measured is None:
        return head + (" No agreement with a released encoder is recorded for this "
                       "(encoder, projection) pair.")
    return head + (
        f" Measured against {measured['reference']} on {measured['presentations']} prompt-only "
        f"presentations, post-token_refiner: mean-removed cosine {measured['cosine']}, relative "
        f"RMS {measured['rel_rms']} against {measured['rel_rms_floor']} for that same encoder in "
        f"another quantization.")


def _read_header(path: str) -> Dict[str, Any]:
    """The safetensors JSON header only; zero tensor bytes read."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path} is not a safetensors file (truncated length prefix)")
        (length,) = struct.unpack("<Q", raw)
        if length <= 0 or length > (64 << 20):
            raise ValueError(f"{path} declares an implausible safetensors header of {length} bytes")
        blob = fh.read(length)
    if len(blob) != length:
        raise ValueError(f"{path} has a truncated safetensors header")
    header = json.loads(blob)
    if not isinstance(header, dict):
        raise ValueError(f"{path} has a non-object safetensors header")
    return header


def _int_field(metadata: Dict[str, Any], key: str, path: str) -> int:
    value = metadata.get(key)
    if value is None:
        raise ValueError(
            f"the MiniMax-H3 text-encoder projection {path} declares no {key!r}; it cannot be "
            f"paired with an encoder without one.")
    try:
        return int(str(value))
    except (TypeError, ValueError):
        raise ValueError(
            f"the MiniMax-H3 text-encoder projection {path} declares {key}={value!r}, "
            f"which is not an integer.") from None


def read_te_projection_spec(path: str) -> Dict[str, Any]:
    """Header-only ``{path, d_in, d_out, tap, mlp_hidden, mlp_depth}``.

    The declared dims are cross-checked against the tensors' own shapes, so a
    file whose metadata says one thing and whose weights say another is refused
    here rather than mis-paired by discovery.
    """
    header = _read_header(path)
    metadata = header.pop("__metadata__", None) or {}
    d_in = _int_field(metadata, "d_in", path)
    d_out = _int_field(metadata, "d_out", path)
    tap = _int_field(metadata, "tap", path)
    mlp_hidden = _int_field(metadata, "mlp_hidden", path)
    mlp_depth = _int_field(metadata, "mlp_depth", path)
    if mlp_depth != _SUPPORTED_MLP_DEPTH:
        raise ValueError(
            f"the MiniMax-H3 text-encoder projection {path} declares mlp_depth={mlp_depth}; "
            f"this loader implements mlp_depth={_SUPPORTED_MLP_DEPTH} "
            f"(Linear -> GELU -> Linear) only.")

    missing = [key for key in _REQUIRED_TENSORS if key not in header]
    if missing:
        raise ValueError(
            f"the MiniMax-H3 text-encoder projection {path} is missing tensor(s) {missing}; "
            f"expected {list(_REQUIRED_TENSORS)}.")
    expected = {
        "W": [d_in, d_out],
        "mean_in": [d_in],
        "std_in": [d_in],
        "mean_out": [d_out],
        "std_out": [d_out],
        "sink_out": [d_out],
        "mlp.0.weight": [mlp_hidden, d_in],
        "mlp.0.bias": [mlp_hidden],
        "mlp.2.weight": [d_out, mlp_hidden],
        "mlp.2.bias": [d_out],
    }
    wrong = {
        key: (header[key].get("shape"), shape)
        for key, shape in expected.items()
        if header[key].get("shape") != shape
    }
    if wrong:
        raise ValueError(
            f"the MiniMax-H3 text-encoder projection {path} has tensor shapes that contradict its "
            f"declared d_in={d_in}/d_out={d_out}/mlp_hidden={mlp_hidden}: "
            f"{ {k: f'{got} != {want}' for k, (got, want) in wrong.items()} }")

    return {
        "path": path, "d_in": d_in, "d_out": d_out, "tap": tap,
        "mlp_hidden": mlp_hidden, "mlp_depth": mlp_depth,
    }


def projection_dir(root: str) -> Path:
    return Path(root) / MINIMAX_H3_TE_PROJECTION_DIRNAME


def discover_te_projections(root: str, *, d_in: int) -> List[Dict[str, Any]]:
    """Specs of every ``clip_projections/`` file whose ``d_in`` matches.

    A file that fails to parse is skipped rather than fatal: it must not stop a
    correctly paired sibling from being found. The caller refuses when nothing
    matched, so a skipped file cannot turn into an unprojected encode.
    """
    directory = projection_dir(root)
    if not directory.is_dir():
        return []
    found: List[Dict[str, Any]] = []
    for candidate in sorted(directory.glob("*.safetensors"), key=lambda item: item.name.lower()):
        try:
            spec = read_te_projection_spec(str(candidate))
        except Exception as exc:
            print(f"[MiniMaxH3Loader] skipping projection {candidate.name}: {exc}")
            continue
        if spec["d_in"] == d_in:
            found.append(spec)
    return found


def resolve_te_projection(
    *,
    root: Optional[str],
    te_path: str,
    hidden_size: int,
    num_hidden_layers: int,
    text_dim: int,
    override: Optional[str] = None,
    available_blocks: Optional[int] = None,
) -> Dict[str, Any]:
    """The projection spec for this encoder, or raise naming both numbers.

    ``override`` names a file explicitly and skips discovery; it is still put
    through every pairing gate, because naming a file is a decision about WHICH
    projection, not about whether it fits.

    ``available_blocks`` switches the depth gate for an encoder file that was
    NOT truncated at write time -- a raw GGUF, which carries every block. There
    the projection's ``tap`` DEFINES how many blocks to run, so it must only fit
    (``tap <= available_blocks``); ``num_hidden_layers`` is unknown and ignored.
    A converted file declares its own depth and keeps the equality gate.
    """
    if override is not None:
        if not os.path.isfile(override):
            raise FileNotFoundError(
                f"MiniMax-H3 text-encoder projection override {override!r} is not an existing file.")
        spec = read_te_projection_spec(override)
    else:
        if not root:
            raise FileNotFoundError(
                f"the MiniMax-H3 text encoder {te_path} is {hidden_size}-wide and needs a trained "
                f"projection to the DiT's {text_dim}-wide conditioning, but no model root was "
                f"available to search for one.")
        candidates = discover_te_projections(root, d_in=hidden_size)
        if not candidates:
            raise FileNotFoundError(
                f"the MiniMax-H3 text encoder {te_path} is {hidden_size}-wide and the DiT's "
                f"conditioning is {text_dim}-wide, so it is usable only with its trained "
                f"projection -- and no file in "
                f"{projection_dir(root)} declares d_in={hidden_size}. Refusing to encode "
                f"{hidden_size}-wide conditioning.")
        if len(candidates) > 1:
            raise ValueError(
                f"{len(candidates)} projections in {projection_dir(root)} declare "
                f"d_in={hidden_size} ({[os.path.basename(c['path']) for c in candidates]}); "
                f"which one was trained for {os.path.basename(te_path)} is not derivable from "
                f"the files. Name one explicitly.")
        spec = candidates[0]

    if spec["d_in"] != hidden_size:
        raise ValueError(
            f"MiniMax-H3 text-encoder projection {spec['path']} takes d_in={spec['d_in']} but the "
            f"text encoder {te_path} is hidden_size={hidden_size} wide. These are different "
            f"encoders' projections; refusing to pair them.")
    if spec["d_out"] != text_dim:
        raise ValueError(
            f"MiniMax-H3 text-encoder projection {spec['path']} produces d_out={spec['d_out']} but "
            f"the DiT's condition_proj takes text_dim={text_dim}. The projected conditioning would "
            f"not fit the checkpoint.")
    if available_blocks is not None:
        if spec["tap"] < 1 or spec["tap"] > available_blocks:
            raise ValueError(
                f"MiniMax-H3 text-encoder projection {spec['path']} was trained on tap="
                f"{spec['tap']} but the text encoder {te_path} carries {available_blocks} "
                f"block(s). The projection is fitted to one specific layer's hidden state, and "
                f"this file cannot produce it.")
    elif spec["tap"] != num_hidden_layers:
        raise ValueError(
            f"MiniMax-H3 text-encoder projection {spec['path']} was trained on tap={spec['tap']} "
            f"but the text encoder {te_path} declares num_hidden_layers={num_hidden_layers}. The "
            f"projection is fitted to one specific layer's hidden state.")
    return spec


def load_te_projection(spec: Dict[str, Any]) -> Dict[str, Any]:
    """``{path, spec, tensors}`` with every tensor read into CPU float32.

    ~0.3 GB for the 4B pair, against a 5.24 GiB encoder that stays mapped; a
    resident float32 copy costs little and keeps the forward exact.
    """
    from safetensors.torch import load_file

    tensors = {key: value.to(torch.float32) for key, value in load_file(spec["path"]).items()}
    missing = [key for key in _REQUIRED_TENSORS if key not in tensors]
    if missing:
        raise ValueError(
            f"the MiniMax-H3 text-encoder projection {spec['path']} is missing {missing} at load "
            f"time; its header declared them.")
    return {"path": spec["path"], "spec": dict(spec), "tensors": tensors}


def apply_te_projection(hidden_states: torch.Tensor, projection: Dict[str, Any]) -> torch.Tensor:
    """Project ``[..., tokens, d_in]`` onto the 32B encoder's ``d_out`` space.

    ``y = (z @ W + mlp(z)) * std_out + mean_out`` with ``z = (x - mean_in) / std_in``:
    ``W`` is a linear SKIP around the MLP, not an alternative head. The MLP
    alone scores mean-removed cosine 0.157 against the 32B reference where the
    sum scores 0.833 (gate G0c), so dropping ``W`` looks like a simplification
    and is a broken projection.

    Token row 0 is Qwen's attention sink; it is REPLACED by the trained
    ``sink_out`` (cos 0.999929 +- 3.3e-06 to the reference's row 0,
    prompt-independent over 111 prompts). REPLACE and ADD differ by under 2.8%
    in norm; REPLACE is the convention here.
    """
    t = projection["tensors"]
    d_in = int(t["mean_in"].shape[0])
    if hidden_states.shape[-1] != d_in:
        raise ValueError(
            f"MiniMax-H3 text-encoder projection {projection['path']} takes d_in={d_in} but the "
            f"hidden state is {hidden_states.shape[-1]}-wide.")
    device = hidden_states.device
    weights = {key: value.to(device) for key, value in t.items()}

    z = (hidden_states.to(torch.float32) - weights["mean_in"]) / weights["std_in"]
    hidden = torch.nn.functional.gelu(
        z @ weights["mlp.0.weight"].T + weights["mlp.0.bias"])
    y = z @ weights["W"] + hidden @ weights["mlp.2.weight"].T + weights["mlp.2.bias"]
    y = y * weights["std_out"] + weights["mean_out"]
    y[..., 0, :] = weights["sink_out"]
    return y
