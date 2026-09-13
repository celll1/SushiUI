"""Persist and validate exact YuE2 tokens, latents, settings, and hashes."""
from pathlib import Path
from dataclasses import dataclass
import hashlib
import json
import numpy as np

from .vendor.protocol import CODEC_SIZE


@dataclass(frozen=True)
class YuE2TrainingArtifacts:
    abc_ids: np.ndarray
    semantic_tokens: np.ndarray
    latents: np.ndarray
    metadata: dict


def _digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_yue2_sidecar(audio_path, result, *, media_sha256=None):
    audio_path = Path(audio_path)
    arrays = audio_path.with_suffix(".yue2.npz")
    np.savez_compressed(arrays, abc_ids=np.asarray(result.abc_ids, dtype=np.int32),
                        semantic_tokens=np.asarray(result.semantic_tokens, dtype=np.int32),
                        latents=result.latents.cpu().float().numpy())
    artifacts = {arrays.name: _digest(arrays),
                 audio_path.name: media_sha256 or _digest(audio_path)}
    if result.abc_text is not None:
        score = audio_path.with_suffix(".abc")
        score.write_text(result.abc_text, encoding="utf-8")
        artifacts[score.name] = _digest(score)
    metadata = dict(version=1, architecture="yue2", seed=result.seed, sample_rate=result.sample_rate,
                    abc_text=result.abc_text, abc_ids=result.abc_ids, truncated=result.truncated,
                    effective_config=result.effective_config, timings=result.timings,
                    model_identity=result.model_identity,
                    artifacts=artifacts)
    sidecar = audio_path.with_suffix(".yue2.json")
    sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(sidecar)


def load_yue2_training_artifacts(path, *, require_abc=False,
                                 require_semantic=False, require_latents=False):
    """Load a generated sidecar only after provenance and tensor validation."""
    path = Path(path)
    sidecar = path if path.suffixes[-2:] == [".yue2", ".json"] else path.with_suffix(".yue2.json")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid YuE2 training sidecar: {sidecar}") from exc
    if metadata.get("version") != 1 or metadata.get("architecture") != "yue2":
        raise ValueError("Unsupported YuE2 training-sidecar schema")
    arrays_path = sidecar.with_suffix(".npz")
    expected_hash = metadata.get("artifacts", {}).get(arrays_path.name)
    if not isinstance(expected_hash, str) or _digest(arrays_path) != expected_hash:
        raise ValueError("YuE2 token/latent artifact hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as stored:
        required = {"abc_ids", "semantic_tokens", "latents"}
        if not required <= set(stored.files):
            raise ValueError(f"YuE2 artifact is missing arrays: {sorted(required - set(stored.files))}")
        abc_ids = np.asarray(stored["abc_ids"])
        semantic = np.asarray(stored["semantic_tokens"])
        latents = np.asarray(stored["latents"])
    if abc_ids.ndim != 1 or abc_ids.dtype.kind not in "iu" or np.any(abc_ids < 0):
        raise ValueError("YuE2 abc_ids must be a non-negative integer vector")
    if semantic.ndim != 1 or semantic.dtype.kind not in "iu" \
            or np.any(semantic < 0) or np.any(semantic >= CODEC_SIZE):
        raise ValueError(f"YuE2 semantic_tokens must be a vector in [0,{CODEC_SIZE})")
    if latents.ndim != 2 or latents.shape[1] != 64 or latents.dtype.kind != "f" \
            or not np.isfinite(latents).all():
        raise ValueError("YuE2 latents must be a finite floating [frames,64] array")
    if semantic.size and latents.shape[0] != semantic.size:
        raise ValueError("YuE2 semantic tokens and latent frames must align 1:1")
    if require_abc and not abc_ids.size:
        raise ValueError("YuE2 objective requires ABC token IDs")
    if require_semantic and not semantic.size:
        raise ValueError("YuE2 objective requires semantic tokens")
    if require_latents and not latents.shape[0]:
        raise ValueError("YuE2 objective requires acoustic latents")
    model_identity = metadata.get("model_identity")
    if not isinstance(model_identity, dict) or not model_identity.get("checkpoint"):
        raise ValueError("YuE2 training artifacts require a model identity")
    return YuE2TrainingArtifacts(
        abc_ids=abc_ids.astype(np.int64, copy=False),
        semantic_tokens=semantic.astype(np.int64, copy=False),
        latents=latents.astype(np.float32, copy=False),
        metadata=metadata,
    )
