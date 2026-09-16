from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from safetensors.torch import save_file

from core.models.common.vae_source import ResolvedVAE, content_hash_for_state_dict
from core.models.component_registry import scan_model
from core.models.sensenova_sdxl_chimera.artifact import (
    ChimeraArtifactError,
    save_chimera_checkpoint,
)
from core.models.sensenova_sdxl_chimera.builder import (
    build_chimera_artifact_from_components,
    initialize_chimera_atomically,
)
from core.models.sensenova_sdxl_chimera.loader import load_chimera_artifact
from core.models.sensenova_sdxl_chimera.loader import preflight_chimera_artifact
from core.model_loader import ModelLoader


def test_component_registry_reads_chimera_root_artifact(tmp_path):
    artifact = tmp_path / "chimera"
    artifact.mkdir()
    (artifact / "chimera.json").write_text(
        '{"model_type":"sensenova_sdxl_chimera","format_version":1,'
        '"understanding":{"locator":"model:teacher.safetensors"},'
        '"vae":{"embedded":true,"latent_channels":4,"scale_factor":8,'
        '"scale_temporal":1}}',
        encoding="utf-8",
    )
    (artifact / "config.json").write_text(
        '{"model_type":"sensenova_sdxl_chimera","format_version":1,'
        '"unet":{"in_channels":4,"out_channels":4,"cross_attention_dim":2048},'
        '"conditioning_bridge":{"context_dim":2048,"pooled_dim":1280},'
        '"vae":{"latent_channels":4}}',
        encoding="utf-8",
    )
    save_file(
        {"unet.conv_in.weight": torch.zeros(4, 4, 1, 1)},
        artifact / "model.safetensors",
    )

    record = scan_model(str(artifact))

    assert record["arch"] == "sensenova_sdxl_chimera"
    assert record["expected"]["latent_channels"] == 4
    assert record["components"]["backbone"] == {
        "kind": "unet", "in_channels": 4, "out_channels": 4, "cond_dim": 2048,
    }
    assert record["components"]["text_encoder"] == {
        "present": True,
        "out_dim": 2048,
        "pooled_dim": 1280,
        "te_type": "sensenova_understanding",
        "embedded": False,
    }
    assert record["components"]["vae"] == {
        "present": True,
        "latent_channels": 4,
        "scale_spatial": 8,
        "scale_temporal": 1,
        "embedded": True,
    }
    assert record["mismatches"] == []


def _tiny_unet() -> UNet2DConditionModel:
    return UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=2048,
        attention_head_dim=2,
        norm_num_groups=4,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=1328,
    )


def _tiny_vae() -> tuple[AutoencoderKL, ResolvedVAE]:
    module = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=32,
        scaling_factor=0.13025,
    )
    state = {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}
    resolved = ResolvedVAE(
        source="model:tiny-sdxl",
        form="model",
        family="sdxl",
        latent_channels=4,
        scale_factor=8,
        scale_temporal=1,
        ndim=4,
        norm="shift_scale",
        norm_pack=1,
        vae_class="AutoencoderKL",
        config=dict(module.config),
        content_hash=content_hash_for_state_dict(state),
        provenance="extracted:tiny-sdxl",
        locator=None,
        struct_native=True,
        identity_native=None,
        scaling_factor=0.13025,
        shift_factor=None,
        state_dict=state,
    )
    return module, resolved


def _write_sensenova_source(path: Path) -> None:
    config = {
        "llm_config": {
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
        }
    }
    tensors = {
        "transformer.language_model.model.layers.0.self_attn.q_proj_mot_gen.weight": torch.zeros(1),
        "transformer.fm_modules.fm_head.weight": torch.zeros(1),
        "transformer.language_model.model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
    }
    save_file(tensors, str(path), metadata={"sensenova_config": json.dumps(config)})


def _write_donor_marker(path: Path) -> None:
    save_file({"model.diffusion_model.marker": torch.zeros(1)}, str(path))


def _build(tmp_path: Path, *, initialization: str = "scratch"):
    und = tmp_path / "sensenova.safetensors"
    donor_path = tmp_path / "sdxl.safetensors"
    _write_sensenova_source(und)
    _write_donor_marker(donor_path)
    donor = _tiny_unet()
    _module, vae = _tiny_vae()
    artifact = tmp_path / "chimera"
    result = build_chimera_artifact_from_components(
        str(artifact),
        understanding_source=str(und),
        sdxl_source=str(donor_path),
        donor_unet=donor,
        vae=vae,
        unet_initialization=initialization,
        initialization_seed=123,
        max_shard_bytes=100_000,
    )
    return und, donor, vae, result


def test_tiny_artifact_round_trip_and_component_identity(tmp_path):
    _und, donor, vae, result = _build(tmp_path, initialization="sdxl_transplant")
    loaded = load_chimera_artifact(result["directory"])
    for name, tensor in donor.state_dict().items():
        assert torch.equal(loaded["unet"].state_dict()[name], tensor)
    for name, tensor in vae.state_dict.items():
        assert torch.equal(loaded["vae"].state_dict()[name], tensor)
    assert loaded["manifest"]["vae"]["content_hash"] == vae.content_hash
    assert content_hash_for_state_dict(loaded["frozen_vae_state"]) == vae.content_hash
    assert not any("_mot_gen" in name for name in loaded["manifest"].keys())
    assert loaded["understanding"] is None


def test_checkpoint_preserves_pinned_vae_across_runtime_dtype_cast(tmp_path):
    _und, _donor, vae, result = _build(tmp_path)
    loaded = load_chimera_artifact(result["directory"], torch_dtype=torch.float16)
    checkpoint = save_chimera_checkpoint(
        tmp_path / "checkpoint",
        base_manifest=loaded["manifest"],
        runtime=loaded["config"],
        bridge=loaded["condition_bridge"],
        unet=loaded["unet"],
        vae=loaded["vae"],
        frozen_vae_state=loaded["frozen_vae_state"],
        stage="bridge_align",
        step=1,
        epoch=0,
        max_shard_bytes=100_000,
    )
    reloaded = load_chimera_artifact(str(checkpoint))
    assert reloaded["manifest"]["vae"]["content_hash"] == vae.content_hash
    assert content_hash_for_state_dict(reloaded["vae"].state_dict()) == vae.content_hash


def test_runtime_can_keep_bf16_backbone_and_fp16_vae(tmp_path):
    _und, _donor, _vae, result = _build(tmp_path)
    loaded = load_chimera_artifact(
        result["directory"], torch_dtype=torch.bfloat16, vae_dtype=torch.float16
    )
    assert next(loaded["unet"].parameters()).dtype == torch.bfloat16
    assert next(loaded["condition_bridge"].parameters()).dtype == torch.bfloat16
    assert next(loaded["vae"].parameters()).dtype == torch.float16


def test_understanding_relocation_same_hash_and_mutation_refusal(tmp_path):
    und, _donor, _vae, result = _build(tmp_path)
    relocated = tmp_path / "relocated.safetensors"
    shutil.copy2(und, relocated)
    loaded = load_chimera_artifact(result["directory"], understanding_override=str(relocated))
    assert loaded["understanding_path"] == str(relocated.resolve())

    with relocated.open("ab") as handle:
        handle.write(b"mutation")
    with pytest.raises(ChimeraArtifactError, match="hash mismatch"):
        load_chimera_artifact(result["directory"], understanding_override=str(relocated))


def test_atomic_builder_refuses_nonempty_target_and_publishes_last(tmp_path):
    und = tmp_path / "und.safetensors"
    donor_path = tmp_path / "donor.safetensors"
    _write_sensenova_source(und)
    _write_donor_marker(donor_path)
    _module, vae = _tiny_vae()
    kwargs = {
        "understanding_source": str(und),
        "sdxl_source": str(donor_path),
        "donor_unet": _tiny_unet(),
        "vae": vae,
        "max_shard_bytes": 100_000,
    }
    result = initialize_chimera_atomically(
        str(tmp_path / "models"), "tiny-chimera", build_kwargs=kwargs
    )
    target = Path(result["directory"])
    assert (target / "chimera.json").is_file()
    assert not list((tmp_path / "models").glob(".*.building-*"))

    with pytest.raises(FileExistsError):
        initialize_chimera_atomically(
            str(tmp_path / "models"), "tiny-chimera", build_kwargs=kwargs
        )


def test_artifact_refuses_weight_census_damage(tmp_path):
    _und, _donor, _vae, result = _build(tmp_path)
    index_path = Path(result["directory"]) / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    victim = next(key for key in index["weight_map"] if key.startswith("condition_bridge."))
    del index["weight_map"][victim]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ChimeraArtifactError, match="census mismatch"):
        load_chimera_artifact(result["directory"])


def test_directory_detection_and_header_only_preflight(tmp_path, monkeypatch):
    _und, _donor, _vae, result = _build(tmp_path)
    assert ModelLoader.detect_model_type(result["directory"]) == "sensenova_sdxl_chimera"

    def refuse_payload(*_args, **_kwargs):
        raise AssertionError("preflight must not read artifact tensor payloads")

    monkeypatch.setattr(
        "core.models.sensenova_sdxl_chimera.loader.read_state_dict", refuse_payload
    )
    checked = preflight_chimera_artifact(result["directory"])
    assert checked["manifest"]["model_type"] == "sensenova_sdxl_chimera"
