from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch

from core.models.common.vae_source import ResolvedVAE, content_hash_for_state_dict
from core.models.sensenova_sdxl_chimera.artifact import checkpoint_content_hash
from core.models.sensenova_sdxl_chimera.source import load_sdxl_donor_components


class _Config(dict):
    __getattr__ = dict.__getitem__


class _FakeModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = _Config(config)

    def register_to_config(self, **kwargs):
        self.config.update(kwargs)


def _unet():
    return _FakeModule({
        "in_channels": 4,
        "out_channels": 4,
        "cross_attention_dim": 2048,
        "addition_embed_type": "text_time",
    })


def _resolved_vae():
    state = {"weight": torch.ones(1)}
    return ResolvedVAE(
        source="file:vae",
        form="file",
        family="sdxl",
        latent_channels=4,
        scale_factor=8,
        scale_temporal=1,
        ndim=4,
        norm="shift_scale",
        norm_pack=1,
        vae_class="AutoencoderKL",
        config={"scaling_factor": 0.13025},
        content_hash=content_hash_for_state_dict(state),
        provenance="file:vae",
        locator=None,
        struct_native=True,
        identity_native=None,
        scaling_factor=0.13025,
        shift_factor=None,
        state_dict=state,
    )


def test_directory_donor_loads_only_unet_and_vae_subdirectories(tmp_path):
    donor = tmp_path / "donor"
    (donor / "unet").mkdir(parents=True)
    (donor / "vae").mkdir()
    (donor / "unet" / "config.json").write_text("{}", encoding="utf-8")
    (donor / "vae" / "config.json").write_text("{}", encoding="utf-8")
    with patch("diffusers.UNet2DConditionModel.from_pretrained", return_value=_unet()) as load_unet, \
            patch("core.models.sensenova_sdxl_chimera.source.resolve_vae_source", return_value=_resolved_vae()) as load_vae:
        _loaded_unet, vae = load_sdxl_donor_components(str(donor))
    assert load_unet.call_args.args[0] == str((donor / "unet").resolve())
    assert load_vae.call_args.args[0] == f"file:{(donor / 'vae').resolve()}"
    assert vae.scaling_factor == 0.13025


def test_single_file_forces_sdxl_vae_normalization(tmp_path):
    donor = tmp_path / "donor.safetensors"
    donor.write_bytes(b"marker")
    wrong_default_vae = _FakeModule({"scaling_factor": 0.18215})
    with patch("diffusers.UNet2DConditionModel.from_single_file", return_value=_unet()), \
            patch("diffusers.AutoencoderKL.from_single_file", return_value=wrong_default_vae):
        _loaded_unet, vae = load_sdxl_donor_components(str(donor))
    assert vae.scaling_factor == 0.13025
    assert vae.config["scaling_factor"] == 0.13025


def test_directory_content_hash_is_path_and_content_sensitive(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "a").write_bytes(b"same")
    first = checkpoint_content_hash(root)
    (root / "a").rename(root / "b")
    assert checkpoint_content_hash(root) != first
