"""Per-installation measurement of a substituted MiniMax-H3 text encoder.

A small Qwen3-VL plus a trained projection (``te_projection.py``) stands in for
the released Qwen3-VL-32B. How well it stands in is a property of THREE files a
user chooses -- the small encoder, the projection, and the released encoder they
would otherwise run -- so it cannot be tabulated in the source tree. It is
measured here, on the installation that will use it.

The cost is split, because only one half is expensive:

* **the reference bank** -- the released encoder's hidden state for a fixed
  prompt suite, ``[S, 5120]`` per presentation. Built ONCE per released encoder
  file, only when explicitly asked for (``build_reference_bank``). Gate G0c
  measured ~5 min / 14-24 GiB RSS / 39 MB stored for the 111-presentation suite.
* **the substitute measurement** -- the same suite through the small encoder and
  its projection, compared against the stored bank (``measure_substitution``).
  ~4.3 s / ~1.5 GiB at 4B, which is why it can run unattended.

Comparison is at two stages. Stage A is the raw ``[S, 5120]`` conditioning; stage
B is that conditioning after the DiT's ``context_embedder -> token_refiner``,
which is what the packed sequence actually contains. The refiner is NOT
timestep-conditioned (``MiniMaxH3TokenRefinerBlock``: no AdaLN, no rotary, a
one-argument ``forward``), so one pass is the whole answer; its weights are read
header-selectively out of the DiT file rather than by building a DiT.

Read the mean-removed cosine, not the raw one: in 5120 dimensions a constant
predictor scores ~0.73 raw at stage A and ~0.90 at stage B, so every table here
carries that constant predictor beside the candidate.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# The prompt suite (tracked asset)
# ---------------------------------------------------------------------------

SUITE_ASSET_PATH = Path(__file__).resolve().parent / "assets" / "te_agreement_suite_v1.json"

# Row 0 is Qwen's attention-sink row and the projection's stored `sink_out`
# constant; it is excluded from every aggregate and reported on its own.
_SINK_ROW = 0

# The registered norm-ratio band from gate G0c, carried so a stored measurement
# reports the same clause the gate adjudicated.
_NORM_BAND = (0.8, 1.25)


def load_suite(path: Optional[str] = None) -> Dict[str, Any]:
    """The versioned prompt suite: ``{version, digest, prompts, ...}``.

    ``digest`` covers the fields that change the numbers, so an edited asset
    that kept its version string still fails to match a stored record.
    """
    asset_path = Path(path) if path else SUITE_ASSET_PATH
    with open(asset_path, encoding="utf-8") as fh:
        suite = json.load(fh)
    for key in ("version", "prompts", "composite_target_tokens"):
        if key not in suite:
            raise ValueError(f"the MiniMax-H3 TE agreement suite {asset_path} declares no {key!r}.")
    if not isinstance(suite["prompts"], list) or not suite["prompts"]:
        raise ValueError(f"the MiniMax-H3 TE agreement suite {asset_path} carries no prompts.")
    suite["digest"] = _digest({
        "version": suite["version"],
        "composite_target_tokens": int(suite["composite_target_tokens"]),
        "prompts": list(suite["prompts"]),
    })
    suite["path"] = str(asset_path)
    return suite


def build_corpus(tokenizer, suite: Optional[Dict[str, Any]] = None) -> List[Tuple[str, List[int]]]:
    """``[(name, token_ids)]``: every prompt, then the long-form composites.

    ``add_special_tokens=False`` because that is how MiniMax-H3 presents a
    ``t2va`` prompt (``h3_pipeline_ops.encode_prompt``); measuring under a
    different presentation would measure a different encoder contract.
    """
    suite = suite or load_suite()
    prompts: Sequence[str] = suite["prompts"]
    target = int(suite["composite_target_tokens"])

    def encode(text: str) -> List[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    items = [(f"p{index:03d}", encode(prompt)) for index, prompt in enumerate(prompts)]
    composites: List[List[int]] = []
    buffer: List[str] = []
    for prompt in prompts:
        buffer.append(prompt)
        ids = encode(" ".join(buffer))
        if len(ids) >= target:
            composites.append(ids)
            buffer = []
    if buffer:
        composites.append(encode(" ".join(buffer)))
    items += [(f"c{index:02d}", ids) for index, ids in enumerate(composites)]
    return items


# ---------------------------------------------------------------------------
# Content identity for a 5-48 GB weight file
# ---------------------------------------------------------------------------

# Bumped when the derivation below changes, so old keys cannot be read as new ones.
IDENTITY_ALGORITHM = "h3teid1"

# Deterministic byte sample: 8 windows of 256 KiB spread across the file.
_SAMPLE_WINDOWS = 8
_SAMPLE_WINDOW_BYTES = 256 * 1024

_IDENTITY_CACHE: Dict[Tuple[str, int, int], str] = {}


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _safetensors_header_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path} is not a safetensors file (truncated length prefix)")
        (length,) = struct.unpack("<Q", raw)
        if length <= 0 or length > (512 << 20):
            raise ValueError(f"{path} declares an implausible safetensors header of {length} bytes")
        blob = fh.read(length)
    if len(blob) != length:
        raise ValueError(f"{path} has a truncated safetensors header")
    return blob


# GGUF value type ids -> fixed byte width; 8 (string) and 9 (array) are handled
# by `_gguf_skip_value`, which is why they are absent here.
_GGUF_SCALAR_WIDTH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _gguf_skip_value(fh, value_type: int) -> None:
    if value_type == 8:
        fh.seek(struct.unpack("<Q", fh.read(8))[0], os.SEEK_CUR)
        return
    if value_type == 9:
        element_type, count = struct.unpack("<IQ", fh.read(12))
        if element_type in _GGUF_SCALAR_WIDTH:
            fh.seek(_GGUF_SCALAR_WIDTH[element_type] * count, os.SEEK_CUR)
        else:
            for _ in range(count):
                _gguf_skip_value(fh, element_type)
        return
    width = _GGUF_SCALAR_WIDTH.get(value_type)
    if width is None:
        raise ValueError(f"unknown GGUF value type {value_type}")
    fh.seek(width, os.SEEK_CUR)


def _gguf_header_bytes(path: str) -> bytes:
    """The GGUF KV metadata and tensor table, verbatim.

    Parsed by walking the two tables rather than through ``GGUFReader``, which
    mmaps the whole file and materialises every field (MEASURED: 5.4 s on a 4 GB
    encoder, against milliseconds here).
    """
    with open(path, "rb") as fh:
        magic, _version, tensor_count, kv_count = struct.unpack("<4sIQQ", fh.read(24))
        if magic != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file (magic {magic!r})")
        for _ in range(kv_count):
            fh.seek(struct.unpack("<Q", fh.read(8))[0], os.SEEK_CUR)  # key
            _gguf_skip_value(fh, struct.unpack("<I", fh.read(4))[0])
        for _ in range(tensor_count):
            fh.seek(struct.unpack("<Q", fh.read(8))[0], os.SEEK_CUR)  # name
            (dims,) = struct.unpack("<I", fh.read(4))
            fh.seek(8 * dims + 4 + 8, os.SEEK_CUR)  # shape, ggml type, data offset
        end = fh.tell()
        fh.seek(0)
        blob = fh.read(end)
    if len(blob) != end:
        raise ValueError(f"{path} has a truncated GGUF header")
    return blob


def _sampled_bytes_digest(path: str, size: int) -> str:
    """sha256 over 8 x 256 KiB read at offsets fixed by the file's size."""
    hasher = hashlib.sha256()
    window = min(_SAMPLE_WINDOW_BYTES, max(size, 1))
    span = max(size - window, 0)
    with open(path, "rb") as fh:
        for index in range(_SAMPLE_WINDOWS):
            offset = span * index // max(_SAMPLE_WINDOWS - 1, 1)
            fh.seek(offset)
            hasher.update(fh.read(window))
    return hasher.hexdigest()


def file_identity(path: str) -> str:
    """A cheap, stable content key for a weight file. Never hashes 25-48 GB.

    Combines the file's STRUCTURE (the safetensors JSON header verbatim -- every
    tensor name, dtype, shape, byte range and ``__metadata__`` entry; or a GGUF's
    KV metadata and tensor table), its total size, and 2 MiB of weight bytes read
    at offsets fixed by that size.

    The property relied on: two files collide only if they agree on every tensor
    name/dtype/shape/offset, on total length, AND on those 2 MiB. Re-quantizing,
    re-shaping, truncating or re-tagging a file changes the structure or the
    size; a same-shape weight edit (a fine-tune, a repair) changes the sampled
    bytes with overwhelming probability. It is a fingerprint, not a proof: a
    weight edit confined entirely to the un-sampled bytes would not be seen.
    """
    resolved = os.path.realpath(path)
    stat = os.stat(resolved)
    cache_key = (resolved, int(stat.st_size), int(stat.st_mtime_ns))
    cached = _IDENTITY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    size = int(stat.st_size)
    if resolved.lower().endswith(".gguf"):
        kind = "gguf"
        structure = hashlib.sha256(_gguf_header_bytes(resolved)).hexdigest()
    elif resolved.lower().endswith(".safetensors"):
        kind = "safetensors"
        structure = hashlib.sha256(_safetensors_header_bytes(resolved)).hexdigest()
    else:
        # No structural description this module understands; the sampled bytes
        # and the size are all the identity there is, and saying so in the key
        # keeps it from being confused with a parsed one.
        kind, structure = "opaque", ""
    identity = "{}:{}:{}".format(
        IDENTITY_ALGORITHM, kind,
        _digest([structure, size, _sampled_bytes_digest(resolved, size)])[:32])
    _IDENTITY_CACHE[cache_key] = identity
    return identity


def _file_ref(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    return {"basename": os.path.basename(path), "identity": file_identity(path)}


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

STORE_DIRNAME = "minimax_h3_te_agreement"
RECORD_FORMAT = 1


def store_dir(root: Optional[str] = None) -> Path:
    """``<settings.cache_dir>/minimax_h3_te_agreement``. Safe to delete whole."""
    if root:
        return Path(root)
    from config.settings import settings

    return Path(settings.cache_dir) / STORE_DIRNAME


def bank_key(suite_digest: str, reference_identity: str) -> str:
    return _digest(["bank", suite_digest, reference_identity])[:24]


def measurement_key(
    suite_digest: str,
    encoder_identity: str,
    projection_identity: str,
    reference_identity: str,
    stage_b_identity: str = "",
) -> str:
    """Keyed on all three files, the suite AND the DiT the stage-B view used.

    The DiT is in the key because stage B is that DiT's ``context_embedder`` and
    ``token_refiner``; a different DiT is a different measurement, not a stale
    one.
    """
    return _digest(["measurement", suite_digest, encoder_identity, projection_identity,
                    reference_identity, stage_b_identity])[:24]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """``None`` for anything unreadable: a broken store means no measurement."""
    try:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
    except Exception:
        return None
    if not isinstance(record, dict) or record.get("format") != RECORD_FORMAT:
        return None
    return record


def _write_json(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    os.replace(temporary, path)


def find_reference_bank(reference_path: str, *, root: Optional[str] = None,
                        suite: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The stored bank for this released encoder under this suite, or ``None``."""
    suite = suite or load_suite()
    try:
        identity = file_identity(reference_path)
    except Exception:
        return None
    directory = store_dir(root) / "banks" / bank_key(suite["digest"], identity)
    manifest = _read_json(directory / "manifest.json")
    if manifest is None or manifest.get("suite_digest") != suite["digest"]:
        return None
    if not (directory / "bank.safetensors").is_file():
        return None
    manifest["dir"] = str(directory)
    return manifest


def list_reference_banks(*, root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every readable bank in the store; an unreadable one is simply absent."""
    banks: List[Dict[str, Any]] = []
    directory = store_dir(root) / "banks"
    if not directory.is_dir():
        return banks
    for entry in sorted(directory.iterdir()):
        manifest = _read_json(entry / "manifest.json") if entry.is_dir() else None
        if manifest is not None and (entry / "bank.safetensors").is_file():
            manifest["dir"] = str(entry)
            banks.append(manifest)
    return banks


def list_measurements(*, root: Optional[str] = None) -> List[Dict[str, Any]]:
    directory = store_dir(root) / "measurements"
    if not directory.is_dir():
        return []
    records = [_read_json(path) for path in sorted(directory.glob("*.json"))]
    return [record for record in records if record is not None]


def local_te_agreement(te_path: str, projection_path: str, *,
                       root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The newest local measurement for this (encoder, projection), or ``None``.

    Every failure -- a missing store, an unreadable record, a file that no longer
    exists -- degrades to ``None``. There is no path from here to a number
    belonging to some other pairing.
    """
    try:
        encoder = file_identity(te_path)
        projection = file_identity(projection_path)
    except Exception:
        return None
    try:
        suite_digest = load_suite()["digest"]
    except Exception:
        return None
    matches = [
        record for record in list_measurements(root=root)
        if record.get("suite_digest") == suite_digest
        and (record.get("encoder") or {}).get("identity") == encoder
        and (record.get("projection") or {}).get("identity") == projection
    ]
    if not matches:
        return None
    return max(matches, key=lambda record: str(record.get("measured_at") or ""))


def summarize_measurement(record: Dict[str, Any]) -> Dict[str, Any]:
    """A stored record reduced to the fields every surface reports.

    ``stage`` says which view ``cosine``/``rel_rms`` came from: ``token_refiner``
    when the DiT's refiner was available (what the packed sequence contains),
    ``raw`` when it was not.
    """
    stage_b = record.get("stage_b") or None
    metrics = stage_b or record.get("stage_a") or {}
    return {
        "source": "local",
        "reference": (record.get("reference") or {}).get("basename"),
        "cosine": metrics.get("cos_mean_removed_median"),
        "cosine_baseline": metrics.get("baseline_cos_mean_removed_median"),
        "rel_rms": metrics.get("rel_rms"),
        "rel_rms_baseline": metrics.get("baseline_rel_rms"),
        "presentations": record.get("presentations"),
        "stage": "token_refiner" if stage_b else "raw",
        "stage_b_reason": record.get("stage_b_reason"),
        "suite_version": record.get("suite_version"),
        "measured_at": record.get("measured_at"),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _summary(values: torch.Tensor) -> Dict[str, float]:
    values = values.float()
    return {
        "mean": round(values.mean().item(), 4),
        "median": round(values.median().item(), 4),
        "p10": round(torch.quantile(values, 0.1).item(), 4),
        "min": round(values.min().item(), 4),
        "max": round(values.max().item(), 4),
        "n": int(values.numel()),
    }


def agreement_metrics(pairs: Iterable[Tuple[torch.Tensor, torch.Tensor]],
                      global_mean: torch.Tensor) -> Dict[str, Any]:
    """Per-position agreement between a reference bank and a candidate bank.

    ``pairs`` yields ``(reference [S, D], candidate [S, D])`` with S matched
    position-for-position -- the caller asserts that; aligning by index across
    different tokenizations would compare unrelated rows.

    ``global_mean`` is the REFERENCE bank's own per-dim mean over non-sink rows.
    Subtracting it from both sides before the cosine is what makes the number
    interpretable: raw cosine in 5120 dims mostly measures that shared mean.
    """
    cosines: List[torch.Tensor] = []
    mean_removed: List[torch.Tensor] = []
    norm_ratios: List[torch.Tensor] = []
    row0_cos: List[float] = []
    row0_norm: List[float] = []
    position_cos: Dict[int, List[float]] = {}
    numerator = denominator = 0.0
    presentations = 0

    for reference, candidate in pairs:
        reference = reference.float()
        candidate = candidate.float()
        if reference.shape != candidate.shape:
            raise ValueError(
                f"the reference bank holds {tuple(reference.shape)} where the candidate produced "
                f"{tuple(candidate.shape)}; these are not the same presentation.")
        presentations += 1
        cosine = torch.nn.functional.cosine_similarity(reference, candidate, dim=-1)
        row0_cos.append(cosine[_SINK_ROW].item())
        row0_norm.append((candidate[_SINK_ROW].norm()
                          / reference[_SINK_ROW].norm().clamp_min(1e-12)).item())
        body_reference = reference[_SINK_ROW + 1:]
        body_candidate = candidate[_SINK_ROW + 1:]
        if body_reference.shape[0] == 0:
            continue
        body_cosine = cosine[_SINK_ROW + 1:]
        cosines.append(body_cosine)
        norm_ratios.append(body_candidate.norm(dim=-1)
                           / body_reference.norm(dim=-1).clamp_min(1e-12))
        mean_removed.append(torch.nn.functional.cosine_similarity(
            body_reference - global_mean, body_candidate - global_mean, dim=-1))
        numerator += ((body_candidate - body_reference) ** 2).sum().item()
        denominator += (body_reference ** 2).sum().item()
        for index in range(body_reference.shape[0]):
            position_cos.setdefault(index, []).append(body_cosine[index].item())

    if not cosines:
        raise ValueError("no non-sink rows to compare; the suite produced only sink rows.")

    cosine_all = torch.cat(cosines)
    mean_removed_all = torch.cat(mean_removed)
    norm_ratio_all = torch.cat(norm_ratios)
    curve = {}
    for low, high in ((0, 4), (4, 16), (16, 64), (64, 128), (128, None)):
        values = [value for index, entries in position_cos.items()
                  if low <= index and (high is None or index < high)
                  for value in entries]
        if values:
            tensor = torch.tensor(values)
            curve[f"pos{low}-{high if high is not None else 'end'}"] = {
                "n": len(values), "median": round(tensor.median().item(), 4)}
    row0_cos_tensor = torch.tensor(row0_cos)
    row0_norm_tensor = torch.tensor(row0_norm)
    in_band = ((norm_ratio_all >= _NORM_BAND[0]) & (norm_ratio_all <= _NORM_BAND[1]))
    return {
        "presentations": presentations,
        "rows": int(cosine_all.numel()),
        "cos": _summary(cosine_all),
        "cos_median": round(cosine_all.median().item(), 4),
        "cos_mean_removed": _summary(mean_removed_all),
        "cos_mean_removed_median": round(mean_removed_all.median().item(), 4),
        "norm_ratio": _summary(norm_ratio_all),
        "norm_band": list(_NORM_BAND),
        "frac_norm_in_band": round(float(in_band.float().mean()), 4),
        "rel_rms": round((numerator / max(denominator, 1e-30)) ** 0.5, 4),
        "pos_curve": curve,
        "sink_row": {
            "cos_median": round(row0_cos_tensor.median().item(), 5),
            "cos_min": round(row0_cos_tensor.min().item(), 5),
            "norm_ratio_median": round(row0_norm_tensor.median().item(), 4),
        },
    }


def bank_global_mean(rows: Iterable[torch.Tensor]) -> torch.Tensor:
    """Per-dim mean over every non-sink row of a bank, in float32."""
    total = None
    count = 0
    for tensor in rows:
        body = tensor.float()[_SINK_ROW + 1:]
        if body.shape[0] == 0:
            continue
        total = body.sum(0) if total is None else total + body.sum(0)
        count += body.shape[0]
    if total is None or count == 0:
        raise ValueError("the reference bank has no non-sink rows.")
    return total / count


# ---------------------------------------------------------------------------
# Stage B: the DiT's context_embedder -> token_refiner, without building a DiT
# ---------------------------------------------------------------------------

# dtypes stage B's 19 tensors are allowed to be. Every released DiT stores this
# submodule unquantized in BF16; a quantized one would need its own dequant and
# must report stage B unavailable rather than load garbage.
_STAGE_B_DTYPES = frozenset({"BF16", "F16", "F32"})


def build_stage_b(dit_path: str, official_dir: Optional[str] = None, *,
                  device: Optional[torch.device] = None,
                  dtype: torch.dtype = torch.bfloat16):
    """``(context_embedder, token_refiner)`` read header-selectively from a DiT.

    19 tensors / ~1.5 GiB out of a 50-block checkpoint; no DiT is instantiated.
    """
    from safetensors import safe_open

    from core.models.minimax_h3.loader import (
        _rename_dit_key, _synthesize_transformer_config, read_safetensors_header,
    )
    from core.models.minimax_h3.vendor.transformer_minimax_h3 import MiniMaxH3TokenRefiner

    header = read_safetensors_header(dit_path)
    wanted = [key for key in header
              if key.startswith("condition_proj.") or key.startswith("token_refiner.")]
    if not wanted:
        raise ValueError(f"{dit_path} carries no condition_proj/token_refiner tensors.")
    unsupported = sorted(
        key for key in wanted if str(header[key].get("dtype")) not in _STAGE_B_DTYPES)
    if unsupported:
        raise ValueError(
            f"{os.path.basename(dit_path)} stores its token refiner quantized "
            f"({len(unsupported)} tensor(s), e.g. {unsupported[0]}: "
            f"{header[unsupported[0]].get('dtype')}); the post-refiner view needs it unquantized.")

    config = _synthesize_transformer_config(header, official_dir)
    state: Dict[str, torch.Tensor] = {}
    with safe_open(dit_path, framework="pt", device="cpu") as handle:
        for key in wanted:
            tensor = handle.get_tensor(key)
            target = _rename_dit_key(key)
            if ".attn.qkv_proj." in key:
                # Contiguous [q_all|k_all|v_all], split exactly as
                # `_map_dit_state_dict` splits it (NOT per-head interleaved).
                stem = target.split(".attn.qkv_proj.")[0] + ".attn."
                inner = tensor.shape[0] // 3
                for name, part in zip(("to_q", "to_k", "to_v"), tensor.split(inner, dim=0)):
                    state[stem + name + ".weight"] = part.contiguous()
            else:
                state[target] = tensor

    embedder = torch.nn.Linear(int(config["text_dim"]), int(config["hidden_size"]), bias=True)
    refiner = MiniMaxH3TokenRefiner(
        hidden_size=int(config["hidden_size"]),
        num_attention_heads=int(config["num_attention_heads"]),
        attention_head_dim=int(config["attention_head_dim"]),
        ffn_dim=int(config["ffn_dim"]),
        num_layers=int(config["num_refiner_layers"]),
        norm_eps=float(config["norm_eps"]),
        qk_norm_eps=float(config["qk_norm_eps"]),
        final_norm_eps=float(config["final_norm_eps"]),
    )
    prefix = "context_embedder."
    embedder.load_state_dict(
        {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)},
        strict=True, assign=True)
    prefix = "token_refiner."
    refiner.load_state_dict(
        {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)},
        strict=True, assign=True)
    target = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return embedder.to(target, dtype).eval(), refiner.to(target, dtype).eval()


@torch.no_grad()
def apply_stage_b(hidden: torch.Tensor, stage_b) -> torch.Tensor:
    """``[S, text_dim] -> [S, hidden_size]`` through embedder + refiner."""
    embedder, refiner = stage_b
    parameter = next(embedder.parameters())
    packed = refiner(embedder(hidden.to(parameter.device, parameter.dtype).unsqueeze(0)))[0]
    return packed.float().cpu()


# ---------------------------------------------------------------------------
# Building the reference bank (expensive; explicit only)
# ---------------------------------------------------------------------------

def _is_substitute(components: Dict[str, Any]) -> bool:
    return bool(components.get("te_projection"))


def build_reference_bank(
    components: Dict[str, Any],
    *,
    reference_basename: str,
    root: Optional[str] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Encode the suite with the LOADED released encoder and store the result.

    Refuses an encoder that is itself a substitute (it has a projection, so its
    hidden state is a stand-in and cannot be anybody's reference), and refuses
    when the loaded encoder is not the file ``reference_basename`` names -- the
    caller is asserting WHICH released encoder this bank speaks for, and a bank
    mislabelled that way is worse than no bank.

    Returns the manifest. Rebuilding over an existing bank is allowed and
    overwrites it; the key already pins the content, so this only happens when
    the suite or the file changed.
    """
    text_encoder = components.get("text_encoder")
    tokenizer = components.get("tokenizer")
    te_path = str(components.get("text_encoder_path") or "")
    if text_encoder is None or tokenizer is None:
        raise ValueError(
            "building a MiniMax-H3 reference bank needs the released text encoder and its "
            "tokenizer loaded; this model has no text encoder installed.")
    if _is_substitute(components):
        raise ValueError(
            f"{os.path.basename(te_path)} is a substituted text encoder (it is paired with the "
            f"projection {os.path.basename(str((components['te_projection'] or {}).get('path')))}). "
            f"A substitute is what agreement is measured AGAINST a reference; it cannot be one.")
    if not te_path or not os.path.isfile(te_path):
        raise ValueError(f"the loaded MiniMax-H3 text encoder path {te_path!r} is not a file.")
    if os.path.basename(te_path) != reference_basename:
        raise ValueError(
            f"a reference bank was requested for {reference_basename!r} but the loaded text "
            f"encoder is {os.path.basename(te_path)!r}. Load the encoder you are naming.")

    from core.models.minimax_h3 import h3_pipeline_ops as ops

    suite = load_suite()
    corpus = build_corpus(tokenizer, suite)
    identity = file_identity(te_path)
    directory = store_dir(root) / "banks" / bank_key(suite["digest"], identity)
    directory.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tensors: Dict[str, torch.Tensor] = {}
    presentations: List[Dict[str, Any]] = []
    started = time.time()
    for index, (name, token_ids) in enumerate(corpus):
        hidden = ops.encode_presentation(
            text_encoder, token_ids, device=device, dtype=torch.bfloat16)[0]
        if hidden.shape[0] != len(token_ids):
            raise RuntimeError(
                f"the reference encoder returned {hidden.shape[0]} row(s) for presentation {name}'s "
                f"{len(token_ids)} token(s); the bank is per-position and cannot store that.")
        tensors[name] = hidden.contiguous()
        presentations.append({"name": name, "tokens": len(token_ids)})
        if progress is not None:
            progress(index + 1, len(corpus), name)

    from safetensors.torch import save_file

    save_file(tensors, str(directory / "bank.safetensors"))
    manifest = {
        "format": RECORD_FORMAT,
        "suite_version": suite["version"],
        "suite_digest": suite["digest"],
        "reference": {"basename": os.path.basename(te_path), "identity": identity},
        "hidden_size": int(next(iter(tensors.values())).shape[-1]),
        "presentations": presentations,
        "token_total": sum(entry["tokens"] for entry in presentations),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build_seconds": round(time.time() - started, 1),
    }
    _write_json(directory / "manifest.json", manifest)
    manifest["dir"] = str(directory)
    return manifest


def load_reference_bank(manifest: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(os.path.join(manifest["dir"], "bank.safetensors"))


# ---------------------------------------------------------------------------
# Measuring a substitute against a stored bank (cheap)
# ---------------------------------------------------------------------------

def _constant_prediction(projection: Dict[str, Any], rows: int, width: int) -> torch.Tensor:
    """``y = mean_out`` with the sink row, i.e. the projection predicting nothing.

    Carried beside every candidate: it is the only thing that makes a cosine in
    5120 dimensions readable.
    """
    tensors = projection["tensors"]
    constant = tensors["mean_out"].float().unsqueeze(0).expand(rows, width).clone()
    constant[_SINK_ROW] = tensors["sink_out"].float()
    return constant


def measure_substitution(
    components: Dict[str, Any],
    *,
    root: Optional[str] = None,
    device: Optional[str] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    reference_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Measure the loaded substitute against a stored bank, and store the result.

    ``None`` when there is no bank to compare against -- that is "no measurement
    recorded", and the caller must say so rather than reach for another number.
    """
    projection = components.get("te_projection")
    text_encoder = components.get("text_encoder")
    tokenizer = components.get("tokenizer")
    te_path = str(components.get("text_encoder_path") or "")
    if not projection or text_encoder is None or tokenizer is None or not te_path:
        return None

    suite = load_suite()
    banks = ([find_reference_bank(reference_path, root=root, suite=suite)]
             if reference_path else list_reference_banks(root=root))
    banks = [bank for bank in banks
             if bank and bank.get("suite_digest") == suite["digest"]
             and int(bank.get("hidden_size") or 0) == int(
                 (components.get("transformer_config") or {}).get("text_dim") or 0)]
    if not banks:
        return None
    manifest = max(banks, key=lambda bank: str(bank.get("built_at") or ""))

    from core.models.minimax_h3 import h3_pipeline_ops as ops

    corpus = build_corpus(tokenizer, suite)
    bank = load_reference_bank(manifest)
    expected = {entry["name"]: int(entry["tokens"]) for entry in manifest["presentations"]}
    if {name for name, _ in corpus} != set(expected):
        raise ValueError(
            f"the reference bank in {manifest['dir']} holds {len(expected)} presentation(s) and "
            f"this suite builds {len(corpus)}; they were produced by different suites.")

    stage_b = None
    stage_b_reason = None
    dit_path = str(components.get("dit_path") or "")
    if dit_path and os.path.isfile(dit_path):
        try:
            stage_b = build_stage_b(dit_path, components.get("official_dir"))
        except Exception as exc:
            stage_b_reason = f"{os.path.basename(dit_path)}: {exc}"
    else:
        stage_b_reason = "no DiT file was recorded with the loaded components"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    text_dim = int(components["transformer_config"]["text_dim"])

    references_a: List[torch.Tensor] = []
    candidates_a: List[torch.Tensor] = []
    constants_a: List[torch.Tensor] = []
    references_b: List[torch.Tensor] = []
    candidates_b: List[torch.Tensor] = []
    constants_b: List[torch.Tensor] = []
    started = time.time()
    for index, (name, token_ids) in enumerate(corpus):
        reference = bank[name].float()
        if reference.shape[0] != len(token_ids):
            raise ValueError(
                f"presentation {name} tokenizes to {len(token_ids)} token(s) here but the "
                f"reference bank stored {reference.shape[0]}. The substitute tokenizes the suite "
                f"differently, so there is no position-for-position comparison to make.")
        hidden = ops.encode_presentation(
            text_encoder, token_ids, device=device, dtype=torch.bfloat16)
        candidate = ops.project_prompt_embeds(
            hidden, projection, text_dim=text_dim, device=device)[0].float()
        if candidate.shape[0] != reference.shape[0]:
            raise ValueError(
                f"the substitute produced {candidate.shape[0]} row(s) for presentation {name}'s "
                f"{reference.shape[0]}; the comparison is per position and is void.")
        constant = _constant_prediction(projection, reference.shape[0], text_dim)
        references_a.append(reference)
        candidates_a.append(candidate)
        constants_a.append(constant)
        if stage_b is not None:
            references_b.append(apply_stage_b(reference, stage_b))
            candidates_b.append(apply_stage_b(candidate, stage_b))
            constants_b.append(apply_stage_b(constant, stage_b))
        if progress is not None:
            progress(index + 1, len(corpus), name)

    def view(references, candidates, constants) -> Dict[str, Any]:
        global_mean = bank_global_mean(references)
        metrics = agreement_metrics(zip(references, candidates), global_mean)
        baseline = agreement_metrics(zip(references, constants), global_mean)
        metrics["baseline_cos_median"] = baseline["cos_median"]
        metrics["baseline_cos_mean_removed_median"] = baseline["cos_mean_removed_median"]
        metrics["baseline_rel_rms"] = baseline["rel_rms"]
        metrics["baseline_frac_norm_in_band"] = baseline["frac_norm_in_band"]
        return metrics

    stage_a_metrics = view(references_a, candidates_a, constants_a)
    stage_b_metrics = view(references_b, candidates_b, constants_b) if stage_b is not None else None

    record = {
        "format": RECORD_FORMAT,
        "suite_version": suite["version"],
        "suite_digest": suite["digest"],
        "identity_algorithm": IDENTITY_ALGORITHM,
        "encoder": _file_ref(te_path),
        "projection": _file_ref(str(projection.get("path") or "")),
        "reference": dict(manifest["reference"]),
        "dit": _file_ref(dit_path) if stage_b is not None else None,
        "presentations": len(corpus),
        "rows": stage_a_metrics["rows"],
        "stage_a": stage_a_metrics,
        "stage_b": stage_b_metrics,
        "stage_b_reason": stage_b_reason,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "measure_seconds": round(time.time() - started, 1),
    }
    key = measurement_key(
        suite["digest"], record["encoder"]["identity"], record["projection"]["identity"],
        record["reference"]["identity"], (record["dit"] or {}).get("identity", ""))
    _write_json(store_dir(root) / "measurements" / f"{key}.json", record)
    return record


def maybe_measure_substitution(components: Dict[str, Any], *,
                               root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The automatic hook: measure this pairing if it is cheap and not yet done.

    Returns the record, or ``None`` for every other outcome -- not a substitute,
    no bank, already measured, or the measurement failed. It NEVER raises: a
    diagnostic must not be able to fail a model load.
    """
    try:
        if not components.get("te_projection"):
            return None
        if local_te_agreement(str(components.get("text_encoder_path") or ""),
                              str((components["te_projection"] or {}).get("path") or ""),
                              root=root) is not None:
            return None
        if not list_reference_banks(root=root):
            return None
        record = measure_substitution(components, root=root)
    except Exception as exc:
        print(f"[MiniMaxH3Loader] text-encoder agreement measurement skipped: {exc}")
        return None
    if record is not None:
        summary = summarize_measurement(record)
        print(f"[MiniMaxH3Loader] measured text-encoder agreement against "
              f"{summary['reference']}: mean-removed cosine {summary['cosine']} "
              f"(constant predictor {summary['cosine_baseline']}), "
              f"relative RMS {summary['rel_rms']}, {summary['presentations']} presentations, "
              f"{summary['stage']} view.")
    return record
