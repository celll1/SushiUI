"""A checkpoint's declared VAE identity, from its metadata to what a client reads.

Phase P0 of docs/guides/VAE_SWAP_MIGRATION_DESIGN.md: a VAE-swapped checkpoint
loads correctly today but every reader downstream of the loader reports the
architecture's 4-channel baseline. These tests pin the declaration path and the
regression bar that goes with it — a NATIVE checkpoint's record must not move.
"""

import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.model_loader import ModelLoader
from core.models.common.single_file_format import (
    build_component_metadata,
    parse_component_metadata,
)
from core.models.component_registry import scan_model


def _checkpoint(path, metadata, latent_channels=4, label_emb=True):
    """A minimal SD-family single file: an LDM U-Net's entry/exit convs."""
    state = {
        "model.diffusion_model.input_blocks.0.0.weight":
            torch.zeros(8, latent_channels, 3, 3, dtype=torch.float16),
        "model.diffusion_model.out.2.weight":
            torch.zeros(latent_channels, 8, 3, 3, dtype=torch.float16),
    }
    if label_emb:
        state["model.diffusion_model.label_emb.0.0.weight"] = torch.zeros(4, 4, dtype=torch.float16)
    save_file(state, str(path), metadata=metadata)
    return str(path)


def test_the_extended_vae_block_survives_a_write_and_a_read():
    md = build_component_metadata(
        vae_type="flux1", vae_channels=16, vae_embedded=True, vae_prefix="vae.",
        vae_class="AutoencoderKL", vae_config={"latent_channels": 16},
        vae_scale_factor=8, vae_scale_temporal=1, vae_norm="shift_scale",
        vae_norm_pack=1, vae_provenance="registry:flux1",
        vae_locator="registry:flux1", vae_hash="0123456789abcdef",
        vae_struct_native=False, vae_identity_native=False,
    )
    assert all(isinstance(v, str) for v in md.values())
    vae = parse_component_metadata(md)["vae"]
    assert vae["type"] == "flux1" and vae["class"] == "AutoencoderKL"
    assert vae["scale_factor"] == 8 and vae["norm_pack"] == 1
    assert vae["config"] == {"latent_channels": 16}
    assert vae["struct_native"] is False and vae["identity_native"] is False
    assert vae["locator"] == "registry:flux1"


def test_a_structurally_different_vae_can_never_be_the_native_latent_space():
    with pytest.raises(ValueError):
        build_component_metadata(vae_struct_native=False, vae_identity_native=True)
    # A file may still claim the impossible pair; the structural answer wins.
    forced = parse_component_metadata({
        "component.vae.struct_native": "0", "component.vae.identity_native": "1"})
    assert forced["vae"]["identity_native"] is False


def test_a_swapped_checkpoints_channels_come_from_its_own_declaration(tmp_path):
    """The legacy `sushi.*` declaration is all a pre-P0 swapped SDXL carries."""
    path = _checkpoint(tmp_path / "swapped.safetensors",
                       {"model_type": "sdxl", "modelspec.architecture": "sdxl-custom",
                        "sushi.vae_type": "flux1", "sushi.in_channels": "16"},
                       latent_channels=16)
    vae = scan_model(path)["components"]["vae"]
    assert vae["latent_channels"] == 16
    assert vae["vae_type"] == "flux1"
    assert vae["struct_native"] is False and vae["identity_native"] is False
    # The declaration replaces the baseline; it is not a defect against it.
    assert scan_model(path)["mismatches"] == []
    assert scan_model(path)["expected"]["latent_channels"] == 16


def test_a_native_checkpoint_declares_nothing_and_reports_the_baseline(tmp_path):
    path = _checkpoint(tmp_path / "native.safetensors", {"model_type": "sdxl"})
    vae = scan_model(path)["components"]["vae"]
    assert vae["latent_channels"] == 4
    assert "struct_native" not in vae and "identity_native" not in vae


@pytest.mark.parametrize("metadata,label_emb,expected", [
    ({}, False, "sd15"),
    ({"modelspec.architecture": "stable-diffusion-v1"}, False, "sd15"),
    # Both signals must beat the >6 GB size heuristic: these files are ~2 KB.
    ({"modelspec.architecture": "stable-diffusion-xl-v1-base"}, False, "sdxl"),
    ({"modelspec.architecture": "sdxl-custom"}, False, "sdxl"),
    ({}, True, "sdxl"),
])
def test_sd_family_detection_reads_the_file_before_it_reads_its_size(
        tmp_path, metadata, label_emb, expected):
    path = _checkpoint(tmp_path / f"{expected}_{len(metadata)}_{label_emb}.safetensors",
                       metadata, label_emb=label_emb)
    assert ModelLoader.detect_model_type(path) == expected
