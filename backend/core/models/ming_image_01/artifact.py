"""CPU-only Ming package identity and safetensors preflight."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_TYPE = "ming_image_01"
FORMAT_VERSION = 1
COMPONENTS = ("mllm", "connector", "mlp", "transformer", "vae", "scheduler")
CONFIG_NAMES = {name: "config.json" for name in COMPONENTS}
CONFIG_NAMES["scheduler"] = "scheduler_config.json"
SOURCE_REVISION = "3515cd7b1e3049fc517eed7d35020251cd06519a"
MODEL_REVISIONS = {
    "generation_edit": "208087ada1486931692c1896f38d4cd16ff3df82",
    "layer_decompose": "9fabca8b62a67f1f53a957d46389c00451c11e52",
}
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "I32": 4,
    "U32": 4, "I64": 8, "U64": 8, "F16": 2, "BF16": 2, "F32": 4,
    "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1,
}


def _json(path: Path) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    with path.open(encoding="utf-8-sig") as handle:
        value = json.load(handle, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _inside(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Component/shard paths must be nonempty strings")
    relative = Path(value)
    if relative.is_absolute() or relative.drive:
        raise ValueError(f"Artifact paths must be relative: {value!r}")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes its package: {value!r}")
    return candidate


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_configs(configs: dict[str, dict]) -> str:
    dit, vae, mlp = (configs[name] for name in ("transformer", "vae", "mlp"))
    if dit.get("_class_name") != "DiffusionTransformer":
        raise ValueError("Ming requires a DiffusionTransformer config")
    padding, multi = dit.get("alignment_padding_mode"), dit.get("multi_frame_output")
    if padding == "zero_masked" and multi is False:
        profile = "generation_edit"
    elif padding == "learned" and multi is True:
        profile = "layer_decompose"
    else:
        raise ValueError("Ming padding/multi-frame profile is missing or contradictory")
    for key in ("dim", "n_layers", "n_refiner_layers", "n_heads", "n_kv_heads", "in_channels", "cap_feat_dim"):
        _positive(dit.get(key), f"transformer.{key}")
    heads, width = dit["n_heads"], dit["dim"]
    axes = dit.get("axes_dims")
    if (width % heads or not isinstance(axes, list) or len(axes) != 3
            or any(type(axis) is not int or axis <= 0 or axis % 2 for axis in axes)
            or sum(axes) != width // heads or heads % dit["n_kv_heads"]):
        raise ValueError("Ming head/RoPE geometry is inconsistent")
    if dit.get("all_patch_size") != [2] or dit.get("all_f_patch_size") != [1]:
        raise ValueError("Ming 0.1 supports spatial patch 2 and frame patch 1")
    alignment = dit.get("vae_alignment")
    if alignment is not None:
        from .vae_alignment import validate_declaration
        validate_declaration(alignment)
        if (max(alignment["old_channels"], alignment["new_channels"]) != dit["in_channels"]
                or alignment["new_channels"] != vae.get("z_dim", vae.get("latent_channels"))
                or alignment["input_channels"] != vae.get("in_channels", vae.get("input_channels", 3))):
            raise ValueError("Ming aligned VAE config disagrees with its bridge declaration")
    elif (vae.get("_class_name") != "AutoencoderKLQwenImage"
            or vae.get("input_channels") != 4 or vae.get("z_dim") != 16
            or dit["in_channels"] != vae["z_dim"]):
        raise ValueError("Ming requires its 4-channel RGBA / 16-channel latent VAE")
    scale, shift = vae.get("scaling_factor"), vae.get("shift_factor")
    if alignment is None and (type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0
            or type(shift) not in (int, float) or not math.isfinite(shift)):
        raise ValueError("Invalid Ming VAE scale/shift")
    if (mlp.get("diffusion_c_input_dim") != dit["cap_feat_dim"]
            or mlp.get("diffusion_inner_dim") != width
            or mlp.get("use_vlm_directvlm_condition") is not True
            or mlp.get("use_learnable_token_condition") is not True):
        raise ValueError("Ming dual-conditioning configuration is inconsistent")
    if (mlp.get("use_identity_mlp") is not True or mlp.get("text_encoder_norm") is not False
            or mlp.get("connector_norm") is not False):
        raise ValueError("Ming 0.1 requires the released identity-MLP and conditioning normalization policy")
    mllm = configs["mllm"]
    if mllm.get("architectures") != ["BailingMM2NativeForConditionalGeneration"]:
        raise ValueError("Ming requires its BailingMM2 native conditioner")
    depth = _positive(mllm.get("llm_config", {}).get("num_hidden_layers"), "mllm depth")
    selected = mlp.get("selected_hidden_states_layers")
    if (not isinstance(selected, list) or not selected
            or any(type(layer) is not int or not 0 <= layer <= depth for layer in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("Invalid direct-VLM hidden-state selections")
    scales = mlp.get("img_gen_scales")
    if not isinstance(scales, list) or not scales:
        raise ValueError("Ming learned-query scales are missing")
    for value in scales:
        _positive(value, "mlp.img_gen_scales")
    if configs["connector"].get("model_type") != "qwen2":
        raise ValueError("Ming connector must be the native Qwen2 component")
    if configs["scheduler"].get("_class_name") != "FlowMatchEulerDiscreteScheduler":
        raise ValueError("Ming requires a flow-matching scheduler")
    return profile


@dataclass(frozen=True)
class TensorHeader:
    shape: tuple[int, ...]
    dtype: str
    byte_count: int

    @property
    def elements(self) -> int:
        return math.prod(self.shape)


def read_tensor_headers(path: Path) -> dict[str, TensorHeader]:
    """Read only metadata, validating the data layout without mapping weights."""
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors prefix")
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size <= min(64 * 1024 * 1024, file_size - 8):
            raise ValueError(f"{path}: invalid safetensors header size")
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"{path}: duplicate header key {key!r}")
                result[key] = value
            return result
        raw = json.loads(handle.read(size), object_pairs_hook=unique)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: invalid tensor header")
    tensors, intervals = {}, []
    data_size = file_size - 8 - size
    for key, spec in raw.items():
        if key == "__metadata__":
            if not isinstance(spec, dict) or any(not isinstance(v, str) for v in spec.values()):
                raise ValueError(f"{path}: invalid metadata")
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: invalid tensor {key!r}")
        shape, dtype, offsets = (spec.get(name) for name in ("shape", "dtype", "data_offsets"))
        if (not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape)
                or not isinstance(dtype, str) or dtype not in _DTYPE_BYTES or not isinstance(offsets, list)
                or len(offsets) != 2 or any(type(o) is not int for o in offsets)):
            raise ValueError(f"{path}: malformed tensor {key!r}")
        start, end = offsets
        expected = math.prod(shape) * _DTYPE_BYTES[dtype]
        if not 0 <= start <= end <= data_size or end - start != expected:
            raise ValueError(f"{path}: invalid tensor span {key!r}")
        tensors[key] = TensorHeader(tuple(shape), dtype, expected)
        if end > start:
            intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError(f"{path}: overlapping or unused tensor payload")
        cursor = end
    if cursor != data_size:
        raise ValueError(f"{path}: trailing or missing tensor payload")
    return tensors


def component_headers(directory: Path) -> dict[str, TensorHeader]:
    indices = sorted(directory.glob("*.safetensors.index.json"))
    if len(indices) > 1:
        raise ValueError(f"{directory}: ambiguous weight indices")
    headers = {}
    if indices:
        mapping = _json(indices[0]).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"{indices[0]}: empty weight map")
        if any(not isinstance(v, str) or not v.endswith(".safetensors") for v in mapping.values()):
            raise ValueError(f"{indices[0]}: unsupported weight shard")
        for shard_name in sorted(set(mapping.values())):
            shard = _inside(directory, shard_name)
            for key, header in read_tensor_headers(shard).items():
                if key in headers or mapping.get(key) != shard_name:
                    raise ValueError(f"{indices[0]}: duplicate or misassigned tensor {key!r}")
                headers[key] = header
        if set(headers) != set(mapping):
            raise ValueError(f"{indices[0]}: index contains missing tensors")
    else:
        files = sorted(directory.glob("*.safetensors"))
        if len(files) != 1:
            raise ValueError(f"{directory}: expected one weight file or one shard index")
        headers = read_tensor_headers(files[0])
    if not headers:
        raise ValueError(f"{directory}: empty component weights")
    return headers


@dataclass(frozen=True)
class MingArtifact:
    root: Path
    profile: str
    components: dict[str, Path]
    configs: dict[str, dict]
    config_fingerprint: str
    tensor_headers: dict[str, dict[str, TensorHeader]]
    source: dict[str, str]
    global_summary: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        census = {}
        for component, tensors in self.tensor_headers.items():
            dtypes = {}
            for tensor in tensors.values():
                dtypes[tensor.dtype] = dtypes.get(tensor.dtype, 0) + tensor.byte_count
            census[component] = {
                "tensor_count": len(tensors),
                "tensor_elements": sum(t.elements for t in tensors.values()),
                "storage_bytes": sum(t.byte_count for t in tensors.values()),
                "bytes_by_dtype": dtypes,
            }
        if self.global_summary is not None:
            census['partition_global']={key:self.global_summary[key] for key in
                ('tensor_count','tensor_elements','storage_bytes','bytes_by_dtype')}
        return {"model_type": MODEL_TYPE, "format_version": FORMAT_VERSION,
                "profile": self.profile, "config_fingerprint": self.config_fingerprint,
                "main_blocks": self.configs["transformer"]["n_layers"],
                "refiner_blocks_per_stream": self.configs["transformer"]["n_refiner_layers"],
                "weights_verified": bool(self.tensor_headers), "components": census,
                "source": self.source,"partition_global":self.global_summary}


def inspect_artifact(path: str | Path, *, require_weights: bool = True) -> MingArtifact:
    candidate = Path(path).resolve()
    manifest_path = candidate / "manifest.json" if candidate.is_dir() else candidate
    root = candidate if candidate.is_dir() else candidate.parent
    raw = _json(manifest_path) if manifest_path.is_file() else None
    if not candidate.is_dir() and raw is None:
        raise FileNotFoundError(candidate)
    if raw is not None:
        if raw.get("model_type") != MODEL_TYPE or type(raw.get("format_version")) is not int or raw["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported Ming artifact manifest identity/version")
        paths = raw.get("components")
        if not isinstance(paths, dict) or set(paths) != set(COMPONENTS):
            raise ValueError("Ming manifest must identify all native components")
    else:
        paths = {name: name for name in COMPONENTS}
    components = {name: _inside(root, paths[name]) for name in COMPONENTS}
    if len(set(components.values())) != len(COMPONENTS):
        raise ValueError("Ming components must have distinct directories")
    configs = {name: _json(directory / CONFIG_NAMES[name]) for name, directory in components.items()}
    profile = validate_configs(configs)
    processor = components["mllm"]
    for filename in ("tokenizer_config.json", "preprocessor_config.json"):
        configs[filename] = _json(processor / filename)
    tokenizer = processor / "tokenizer.json"
    if require_weights and not tokenizer.is_file():
        raise FileNotFoundError(tokenizer)
    if tokenizer.is_file():
        with tokenizer.open("rb") as handle:
            configs["tokenizer_identity"] = {"sha256": hashlib.file_digest(handle, "sha256").hexdigest()}
    fingerprint = _fingerprint(configs)
    if raw is not None and (raw.get("profile") != profile or raw.get("config_fingerprint") != fingerprint):
        raise ValueError("Ming manifest profile/config fingerprint disagrees with components")
    headers = {name: component_headers(directory) for name, directory in components.items()
               if name != "scheduler"} if require_weights else {}
    if configs["transformer"].get("vae_alignment") is not None and require_weights:
        from .vae_alignment import WEIGHTS, MingVAEAlignment
        alignment_headers = read_tensor_headers(root / WEIGHTS)
        declared = MingVAEAlignment(configs["transformer"]["vae_alignment"]).state_dict()
        if set(declared) != set(alignment_headers) or any(
                tuple(value.shape) != alignment_headers[key].shape for key, value in declared.items()):
            raise ValueError("Ming alignment tensor headers disagree with its declaration")
        headers["vae_alignment"] = alignment_headers
    source = raw.get("source", {}) if raw else {}
    if not isinstance(source, dict) or any(not isinstance(v, str) for v in source.values()):
        raise ValueError("Ming source provenance must be string-valued")
    from .partition_global import inspect_global_payload
    global_summary=inspect_global_payload(root,latent_channels=configs['transformer']['in_channels'],hidden_dim=configs['transformer']['dim'])
    return MingArtifact(root, profile, components, configs, fingerprint, headers, source,global_summary)


def write_manifest(path: str | Path, *, source: dict[str, str] | None = None) -> Path:
    """Add identity to a verified native package without copying its weights."""
    artifact = inspect_artifact(path)
    manifest = artifact.root / "manifest.json"
    payload = {"model_type": MODEL_TYPE, "format_version": FORMAT_VERSION,
               "profile": artifact.profile, "config_fingerprint": artifact.config_fingerprint,
               "components": {name: directory.relative_to(artifact.root).as_posix()
                              for name, directory in artifact.components.items()},
               "source": artifact.source if source is None else source}
    if not isinstance(payload["source"], dict) or any(not isinstance(v, str) for v in payload["source"].values()):
        raise ValueError("Ming source provenance must be string-valued")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=artifact.root,
                                         prefix=".ming-manifest-", suffix=".json", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest
