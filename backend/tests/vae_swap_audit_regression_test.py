"""Real small VAEs exercise config-only changes and converted checkpoint saves."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models.common.vae_source import (
    ResolvedVAE, VaeSourceError, load_declared_latent_io, resolve_vae_source,
)
from core.training.arch.sd15 import SD15ArchHandler
from core.training.vae_swap import apply_configured_vae_swap, swap_metadata


@pytest.fixture
def vae():
    from diffusers import AutoencoderKL
    return AutoencoderKL(
        block_out_channels=(32, 32, 32, 32),
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        latent_channels=4, norm_num_groups=32, scaling_factor=0.18215,
    ).eval()


def trainer(vae):
    unet = torch.nn.Module()
    unet.conv_in = torch.nn.Conv2d(4, 8, 3)
    unet.conv_out = torch.nn.Conv2d(8, 4, 3)
    return SimpleNamespace(
        vae=vae, unet=unet, config={}, vae_dtype=torch.float32,
        base_vae_identity=None, arch=SD15ArchHandler(None), bundle_vae=True,
    )


@pytest.mark.parametrize("changed", [False, True])
def test_config_only_swap_is_not_a_noop(tmp_path, vae, changed):
    source = tmp_path / "vae"
    vae.save_pretrained(source)
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    if changed:
        config["scaling_factor"] = 0.5
        config_path.write_text(json.dumps(config))
    run = trainer(vae)
    apply_configured_vae_swap(run, f"file:{source}")
    assert run.vae_identity.identity_native is (not changed)
    metadata = swap_metadata(run)[2]
    if changed:
        assert json.loads(metadata["component.vae.config"])["scaling_factor"] == 0.5
    else:
        assert metadata == {}


def test_converted_vae_save_keeps_materialized_config(tmp_path, vae):
    run = trainer(vae)
    run.vae_identity = ResolvedVAE(
        source="registry:sd15", form="registry", family="sd15",
        latent_channels=4, scale_factor=8, scale_temporal=1, ndim=4,
        norm="shift_scale", norm_pack=1, vae_class="AutoencoderKL", config={},
        content_hash="1234567890abcdef", provenance="registry:sd15",
        locator="registry:sd15", struct_native=True, identity_native=False,
        scaling_factor=0.18215, shift_factor=None,
    )
    metadata = swap_metadata(run)[2]
    assert json.loads(metadata["component.vae.config"])["block_out_channels"] == [32] * 4
    state = {f"vae.{k}": v for k, v in vae.state_dict().items()}
    state["model.diffusion_model.input_blocks.0.0.weight"] = run.unet.conv_in.weight
    path = tmp_path / "checkpoint.safetensors"
    save_file(state, str(path), metadata=metadata)
    loaded = load_declared_latent_io(str(path), arch="sd15").load_module()
    assert dict(loaded.config)["block_out_channels"] == [32] * 4
    for name, tensor in vae.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name]), name


def test_latent_identity_includes_normalization(tmp_path, vae):
    vae.save_pretrained(tmp_path)
    original = resolve_vae_source(f"file:{tmp_path}")
    changed = replace(original, scaling_factor=0.5)
    assert original.content_hash == changed.content_hash
    assert original.latent_hash != changed.latent_hash
    equivalent = replace(original, config=dict(original.config, _name_or_path="elsewhere"))
    assert original.latent_hash == equivalent.latent_hash


def test_external_vae_config_change_is_refused(tmp_path, vae):
    source = tmp_path / "vae"
    vae.save_pretrained(source)
    run = trainer(vae)
    run.bundle_vae = False
    run.vae_identity = replace(resolve_vae_source(f"file:{source}"), identity_native=False)
    checkpoint = tmp_path / "checkpoint.safetensors"
    save_file({"unet.conv_in.weight": run.unet.conv_in.weight}, str(checkpoint),
              metadata=swap_metadata(run)[2])
    assert load_declared_latent_io(str(checkpoint), arch="sd15") is not None
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config["scaling_factor"] = 0.5
    config_path.write_text(json.dumps(config))
    with pytest.raises(VaeSourceError, match="normalisation.*changed"):
        load_declared_latent_io(str(checkpoint), arch="sd15")


def test_a_flux2_sample_reference_matches_the_noise_it_is_concatenated_with():
    """The reference encode is normalise-then-patchify, like the training encode.

    FLUX.2's VAE is BatchNorm and carries no scaling_factor, and the noise this
    is concatenated with is 128ch at half resolution -- so multiplying by a
    scaling factor and skipping the patchify produced a 32ch, full-resolution
    tensor that could not concatenate. The caller swallows the exception and
    prints a warning, so only a shape assertion catches it.
    """
    from core.models.components.vae_registry import normalize
    from core.training.base_trainer import BaseTrainer
    from core.training.ops.flux2_ops import (
        _flux2_pack_latents_for_sample, _flux2_prepare_latent_ids_for_sample,
    )

    vae = SimpleNamespace(
        bn=SimpleNamespace(running_mean=torch.zeros(128), running_var=torch.ones(128)),
        config=SimpleNamespace(batch_norm_eps=1e-4, latent_channels=32,
                               scaling_factor=None))

    reference = BaseTrainer._flux2_patchify_latents_for_training(
        None, normalize(torch.randn(1, 32, 64, 64), vae))
    noise = torch.randn(1, 128, 32, 32)

    packed_reference = _flux2_pack_latents_for_sample(reference)
    packed_noise = _flux2_pack_latents_for_sample(noise)
    assert packed_reference.shape[-1] == packed_noise.shape[-1] == 128
    assert torch.cat([packed_reference, packed_noise], dim=1).shape == (1, 2048, 128)
    assert _flux2_prepare_latent_ids_for_sample(reference).shape[1] == packed_noise.shape[1]
