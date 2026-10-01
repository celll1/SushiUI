"""Ming package validation uses tiny local files, never Torch or the API."""

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.models.ming_image_01.artifact import (
    COMPONENTS, CONFIG_NAMES, inspect_artifact, read_tensor_headers, write_manifest,
)


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def write_weights(path, tensors=None):
    tensors = tensors or {"weight": ([2, 3], "BF16", 12)}
    header, offset = {}, 0
    for name, (shape, dtype, size) in tensors.items():
        header[name] = {"shape": shape, "dtype": dtype, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


@pytest.fixture
def package(tmp_path):
    configs = {
        "transformer": {"_class_name": "DiffusionTransformer", "alignment_padding_mode": "zero_masked",
                        "multi_frame_output": False, "dim": 24, "n_heads": 3, "n_kv_heads": 3,
                        "n_layers": 30, "n_refiner_layers": 2, "in_channels": 16, "cap_feat_dim": 16,
                        "axes_dims": [2, 2, 4], "all_patch_size": [2], "all_f_patch_size": [1]},
        "vae": {"_class_name": "AutoencoderKLQwenImage", "input_channels": 4, "z_dim": 16,
                "scaling_factor": 8.0064, "shift_factor": 0.0},
        "mlp": {"diffusion_c_input_dim": 16, "diffusion_inner_dim": 24,
                "use_vlm_directvlm_condition": True, "use_learnable_token_condition": True,
                "selected_hidden_states_layers": [5, 12, 20], "img_gen_scales": [16]},
        "mllm": {"architectures": ["BailingMM2NativeForConditionalGeneration"],
                 "llm_config": {"num_hidden_layers": 20}},
        "connector": {"model_type": "qwen2"},
        "scheduler": {"_class_name": "FlowMatchEulerDiscreteScheduler"},
    }
    for name in COMPONENTS:
        directory = tmp_path / name
        directory.mkdir()
        write_json(directory / CONFIG_NAMES[name], configs[name])
        if name != "scheduler":
            write_weights(directory / "model.safetensors")
    for name in ("tokenizer_config.json", "preprocessor_config.json", "tokenizer.json"):
        write_json(tmp_path / "mllm" / name, {})
    return tmp_path


def change(package, component, **kwargs):
    path = package / component / CONFIG_NAMES[component]
    raw = json.loads(path.read_text())
    raw.update(kwargs)
    write_json(path, raw)


@pytest.mark.parametrize("profile, padding, multi", [
    ("generation_edit", "zero_masked", False), ("layer_decompose", "learned", True),
])
def test_profiles_and_manifest_round_trip(package, profile, padding, multi):
    change(package, "transformer", alignment_padding_mode=padding, multi_frame_output=multi)
    original = inspect_artifact(package)
    manifest = write_manifest(package, source={"revision": "synthetic-test-revision"})
    restored = inspect_artifact(manifest)
    assert restored.profile == profile
    assert restored.config_fingerprint == original.config_fingerprint
    assert restored.source == {"revision": "synthetic-test-revision"}
    census = restored.summary()["components"]
    assert census["mllm"]["tensor_elements"] == 6
    assert census["mllm"]["storage_bytes"] == 12
    assert census["mllm"]["bytes_by_dtype"] == {"BF16": 12}
    assert restored.summary()["weights_verified"]


@pytest.mark.parametrize("component, patch", [
    ("transformer", {"multi_frame_output": True}),
    ("transformer", {"axes_dims": [2, 2, 2]}),
    ("transformer", {"n_layers": True}),
    ("transformer", {"all_f_patch_size": [2]}),
    ("vae", {"z_dim": 64}),
    ("vae", {"input_channels": 3}),
    ("vae", {"scaling_factor": float("nan")}),
    ("mlp", {"diffusion_inner_dim": 12}),
    ("mlp", {"selected_hidden_states_layers": [21]}),
    ("connector", {"model_type": "qwen3_vl"}),
])
def test_preflight_rejects_foreign_or_inconsistent_config(package, component, patch):
    change(package, component, **patch)
    with pytest.raises(ValueError):
        inspect_artifact(package)


def test_manifest_detects_processor_or_config_drift(package):
    manifest = write_manifest(package)
    write_json(package / "mllm" / "preprocessor_config.json", {"size": 123})
    with pytest.raises(ValueError, match="fingerprint"):
        inspect_artifact(manifest)


def test_manifest_rejects_escape_and_component_alias(package):
    manifest = write_manifest(package)
    raw = json.loads(manifest.read_text())
    raw["components"]["mlp"] = "../outside"
    write_json(manifest, raw)
    with pytest.raises(ValueError, match="escapes"):
        inspect_artifact(manifest)
    raw["components"]["mlp"] = "mllm"
    write_json(manifest, raw)
    with pytest.raises(ValueError, match="distinct"):
        inspect_artifact(manifest)


def test_shard_index_detects_missing_and_wrong_tensor(package):
    directory = package / "transformer"
    write_json(directory / "model.safetensors.index.json", {"weight_map": {"weight": "missing.safetensors"}})
    with pytest.raises(FileNotFoundError):
        inspect_artifact(package)
    write_json(directory / "model.safetensors.index.json", {"weight_map": {"other": "model.safetensors"}})
    with pytest.raises(ValueError, match="misassigned"):
        inspect_artifact(package)


def test_sharded_census(package):
    directory = package / "transformer"
    write_weights(directory / "model.safetensors", {"weight": ([2, 3], "BF16", 12)})
    write_weights(directory / "part2.safetensors", {"bias": ([2], "F32", 8)})
    write_json(directory / "model.safetensors.index.json", {"weight_map": {
        "weight": "model.safetensors", "bias": "part2.safetensors"}})
    result = inspect_artifact(package).summary()["components"]["transformer"]
    assert result["tensor_elements"] == 8
    assert result["storage_bytes"] == 20
    assert result["bytes_by_dtype"] == {"BF16": 12, "F32": 8}


@pytest.mark.parametrize("shape, dtype, size", [([2, 3], "BF16", 11), ([-1], "F32", 4), ([1], "ALIEN", 4)])
def test_malformed_header_rejected(tmp_path, shape, dtype, size):
    path = tmp_path / "bad.safetensors"
    write_weights(path, {"weight": (shape, dtype, size)})
    with pytest.raises(ValueError):
        read_tensor_headers(path)


def test_configs_only_cannot_claim_weight_verification(package):
    (package / "mlp" / "model.safetensors").unlink()
    assert not inspect_artifact(package, require_weights=False).summary()["weights_verified"]
    with pytest.raises(ValueError):
        write_manifest(package)


def test_payload_truncation_and_overlap(tmp_path):
    path = tmp_path / "bad.safetensors"
    write_weights(path)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError):
        read_tensor_headers(path)
    header = {name: {"shape": [1], "dtype": "F32", "data_offsets": [0, 4]} for name in ("a", "b")}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(4))
    with pytest.raises(ValueError, match="overlapping"):
        read_tensor_headers(path)


def test_metadata_module_does_not_import_torch():
    import subprocess
    backend = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); import core.models.ming_image_01; "
        "assert 'torch' not in sys.modules", backend], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
