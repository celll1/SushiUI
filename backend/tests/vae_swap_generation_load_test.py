"""Loading a swapped-VAE checkpoint for GENERATION (design §9.1, phase P2c).

The training side already builds a base at its declared channel count; until
this phase the inference loader read only the legacy ``sushi.vae_type`` pair, so
a checkpoint that declares itself through ``component.vae.*`` alone loaded as a
plain 4-channel model and generated from the wrong latent space in silence.

These tests drive ``reconstruct_sd_sdxl_pipeline`` with a stub for diffusers'
``from_single_file`` — what is under test is which VAE and which channel count
the loader hands it, and what it refuses. The resize and the trained-conv load
are the real functions.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.models.common import vae_source as vs
from core.models.common.single_file_format import build_component_metadata


# --- fixtures ---------------------------------------------------------------

def _tiny_vae(latent_channels):
    """A real AutoencoderKL, small enough to build in milliseconds."""
    from diffusers import AutoencoderKL
    return AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,), layers_per_block=1, norm_num_groups=4,
        latent_channels=latent_channels, sample_size=32,
    )


def _backbone_state(latent_channels):
    """The two channel-dependent LDM convs a custom checkpoint must supply,
    plus the SDXL marker key the arch detector reads."""
    return {
        "model.diffusion_model.input_blocks.0.0.weight":
            torch.full((8, latent_channels, 3, 3), 0.25),
        "model.diffusion_model.input_blocks.0.0.bias": torch.full((8,), 0.5),
        "model.diffusion_model.out.2.weight":
            torch.full((latent_channels, 8, 3, 3), 0.75),
        "model.diffusion_model.out.2.bias": torch.full((latent_channels,), 1.5),
        "model.diffusion_model.label_emb.0.0.weight": torch.zeros(4, 4),
    }


def _write_checkpoint(path, latent_channels, metadata, vae=None, prefix="vae."):
    state = _backbone_state(latent_channels)
    if vae is not None:
        state.update({f"{prefix}{k}": v for k, v in vae.state_dict().items()})
    save_file(state, str(path), metadata=metadata)
    return str(path)


def _vae_config(vae):
    # A saved diffusers config.json always carries _class_name; a config built
    # in-process does not, and the resolver refuses rather than guess a class.
    return {**dict(vae.config), "_class_name": "AutoencoderKL"}


def _vae_metadata(vae, latent_channels, **overrides):
    fields = dict(
        vae_type="custom", vae_channels=latent_channels,
        vae_class="AutoencoderKL", vae_config=_vae_config(vae),
        vae_scale_factor=8, vae_scale_temporal=1, vae_norm="shift_scale",
        vae_norm_pack=1, vae_provenance="file:tiny.safetensors",
        vae_hash=vs.content_hash_for_state_dict(vae.state_dict()),
        vae_struct_native=False, vae_identity_native=False,
    )
    fields.update(overrides)
    return build_component_metadata(**fields)


class _FakeUNet(torch.nn.Module):
    """Enough of UNet2DConditionModel for resize_unet_in_out to act on."""

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.conv_in = torch.nn.Conv2d(in_channels, 8, 3, padding=1)
        self.conv_out = torch.nn.Conv2d(8, out_channels, 3, padding=1)
        self.config = SimpleNamespace(in_channels=in_channels,
                                      out_channels=out_channels)

    def register_to_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.config, key, value)


class _Recorder:
    """Stands in for StableDiffusion(XL)Pipeline.from_single_file."""

    def __init__(self):
        self.calls = []

    def from_single_file(self, path, **kwargs):
        self.calls.append(kwargs)
        unet = _FakeUNet(int(kwargs.get("num_in_channels") or 4))
        return SimpleNamespace(unet=unet, vae=kwargs.get("vae"))


@pytest.fixture
def loader(monkeypatch):
    from core import model_loader as ml
    recorder = _Recorder()
    monkeypatch.setattr(ml, "StableDiffusionXLPipeline", recorder)
    monkeypatch.setattr(ml, "StableDiffusionPipeline", recorder)
    return ml.ModelLoader, recorder


# ---------------------------------------------------------------------------
# 1. A bundled declaration drives construction, the VAE and the identity
# ---------------------------------------------------------------------------

def test_a_bundled_declaration_builds_the_backbone_at_its_channel_count(
        tmp_path, loader):
    ModelLoader, recorder = loader
    vae = _tiny_vae(16)
    path = _write_checkpoint(
        tmp_path / "swapped.safetensors", 16,
        _vae_metadata(vae, 16, vae_embedded=True, vae_prefix="vae.",
                      vae_provenance="extracted:donor"),
        vae=vae)

    pipeline = ModelLoader.reconstruct_sd_sdxl_pipeline(
        path, "sdxl", torch.float32, "cpu")

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["num_in_channels"] == 16 and call["out_channels"] == 16
    # The checkpoint's OWN VAE, not the architecture's.
    assert call["vae"].config.latent_channels == 16
    # The resize ran and the trained convs were read back over its zeros.
    assert pipeline.unet.conv_in.in_channels == 16
    assert pipeline.unet.conv_out.out_channels == 16
    assert torch.allclose(pipeline.unet.conv_in.weight,
                          torch.full((8, 16, 3, 3), 0.25))
    assert torch.allclose(pipeline.unet.conv_out.weight,
                          torch.full((16, 8, 3, 3), 0.75))
    identity = pipeline._sushi_vae_identity
    assert identity["latent_channels"] == 16
    assert identity["identity_native"] is False
    assert identity["content_hash"] == vs.content_hash_for_state_dict(vae.state_dict())
    assert pipeline._sushi_vae_source == "extracted:donor"
    assert pipeline._sushi_arch["in_channels"] == 16


def test_the_loaded_identity_reaches_current_model_info(tmp_path, loader):
    ModelLoader, _ = loader
    vae = _tiny_vae(16)
    path = _write_checkpoint(
        tmp_path / "swapped.safetensors", 16,
        _vae_metadata(vae, 16, vae_embedded=True, vae_prefix="vae.",
                      vae_provenance="file:flux1_vae.safetensors"),
        vae=vae)
    pipeline = ModelLoader.reconstruct_sd_sdxl_pipeline(
        path, "sdxl", torch.float32, "cpu")

    from core.pipeline import DiffusionPipelineManager
    manager = DiffusionPipelineManager.__new__(DiffusionPipelineManager)
    manager._sushi_wiring = None
    fields = manager._fold_sd_latent_identity(pipeline, "sdxl")

    assert fields["latent_channels"] == 16
    assert fields["vae_provenance"] == "file:flux1_vae.safetensors"
    assert fields["vae_identity_native"] is False
    assert fields["vae_struct_native"] is False
    assert fields["vae_hash"] == vs.content_hash_for_state_dict(vae.state_dict())
    # What the sampler, the override gate and /models/current all read.
    assert manager._sushi_wiring.latent_channels == 16


def test_a_same_channel_swap_replaces_the_vae_without_rebuilding(tmp_path, loader):
    """struct_native=1, identity_native=0: another 4-channel VAE. The backbone
    is untouched (§9.1) and only the VAE is replaced."""
    ModelLoader, recorder = loader
    vae = _tiny_vae(4)
    path = _write_checkpoint(
        tmp_path / "finetuned_vae.safetensors", 4,
        _vae_metadata(vae, 4, vae_embedded=True, vae_prefix="vae.",
                      vae_struct_native=True, vae_identity_native=False),
        vae=vae)

    pipeline = ModelLoader.reconstruct_sd_sdxl_pipeline(
        path, "sdxl", torch.float32, "cpu")

    call = recorder.calls[0]
    assert "num_in_channels" not in call and "out_channels" not in call
    assert call["vae"] is not None
    assert pipeline._sushi_vae_identity["latent_channels"] == 4


def test_a_native_checkpoint_takes_the_unchanged_path(tmp_path, loader):
    ModelLoader, recorder = loader
    vae = _tiny_vae(4)
    path = _write_checkpoint(tmp_path / "native.safetensors", 4,
                             {"model_type": "sdxl"}, vae=vae,
                             prefix="first_stage_model.")

    pipeline = ModelLoader.reconstruct_sd_sdxl_pipeline(
        path, "sdxl", torch.float32, "cpu")

    call = recorder.calls[0]
    assert "num_in_channels" not in call and "vae" not in call
    assert pipeline._sushi_vae_identity is None
    assert pipeline._sushi_arch["vae_type"] is None
    assert pipeline._sushi_arch["in_channels"] is None


def test_the_legacy_sushi_pair_still_names_its_registry_family(tmp_path, loader,
                                                               monkeypatch):
    """A pre-migration swapped SDXL declares only sushi.vae_type/in_channels."""
    ModelLoader, recorder = loader
    vae = _tiny_vae(16)
    monkeypatch.setattr(vs, "_registry_source", lambda key, download: "unused")
    monkeypatch.setattr(
        vs, "resolve_vae_source",
        lambda source, **kw: vs.ResolvedVAE(
            source=source, form="registry", family="flux1", latent_channels=16,
            scale_factor=8, scale_temporal=1, ndim=4, norm="shift_scale",
            norm_pack=1, vae_class="AutoencoderKL", config=dict(vae.config),
            content_hash="1111111111111111", provenance="registry:flux1",
            locator="registry:flux1", struct_native=False, identity_native=False,
            scaling_factor=0.3611, shift_factor=None,
            state_dict=dict(vae.state_dict())))
    path = _write_checkpoint(
        tmp_path / "legacy.safetensors", 16,
        {"sushi.vae_type": "flux1", "sushi.in_channels": "16"})

    pipeline = ModelLoader.reconstruct_sd_sdxl_pipeline(
        path, "sdxl", torch.float32, "cpu")

    assert recorder.calls[0]["num_in_channels"] == 16
    assert pipeline._sushi_arch["vae_type"] == "flux1"
    assert pipeline._sushi_vae_identity["provenance"] == "registry:flux1"


# ---------------------------------------------------------------------------
# 2. Refusals (§5.2, §9.1): never a silent fallback to the native VAE
# ---------------------------------------------------------------------------

def test_a_locator_whose_content_moved_is_refused_not_loaded_natively(
        tmp_path, loader):
    ModelLoader, recorder = loader
    vae = _tiny_vae(16)
    standalone = tmp_path / "standalone.safetensors"
    save_file(dict(vae.state_dict()), str(standalone),
              metadata={"component.vae.config": json.dumps(_vae_config(vae))})
    path = _write_checkpoint(
        tmp_path / "unbundled.safetensors", 16,
        _vae_metadata(vae, 16, vae_embedded=False,
                      vae_locator=f"path:{standalone}",
                      vae_hash="0000000000000000"))

    with pytest.raises(vs.VaeSourceError, match="refusing"):
        ModelLoader.reconstruct_sd_sdxl_pipeline(path, "sdxl", torch.float32, "cpu")
    assert recorder.calls == []


def test_a_matching_locator_hash_loads_the_referenced_vae(tmp_path, loader):
    ModelLoader, recorder = loader
    vae = _tiny_vae(16)
    standalone = tmp_path / "standalone.safetensors"
    save_file(dict(vae.state_dict()), str(standalone),
              metadata={"component.vae.config": json.dumps(_vae_config(vae))})
    path = _write_checkpoint(
        tmp_path / "unbundled.safetensors", 16,
        _vae_metadata(vae, 16, vae_embedded=False,
                      vae_locator=f"path:{standalone}"))

    pipeline = ModelLoader.reconstruct_sd_sdxl_pipeline(
        path, "sdxl", torch.float32, "cpu")

    assert recorder.calls[0]["num_in_channels"] == 16
    assert pipeline._sushi_vae_identity["locator"] == f"path:{standalone}"


def test_a_declared_vae_with_no_locator_is_refused(tmp_path, loader):
    ModelLoader, recorder = loader
    vae = _tiny_vae(16)
    path = _write_checkpoint(
        tmp_path / "orphan.safetensors", 16,
        _vae_metadata(vae, 16, vae_embedded=False, vae_locator=None))

    with pytest.raises(vs.VaeSourceError, match="no resolvable locator"):
        ModelLoader.reconstruct_sd_sdxl_pipeline(path, "sdxl", torch.float32, "cpu")
    assert recorder.calls == []


def test_a_checkpoint_whose_convs_disagree_with_its_declaration_is_refused(
        tmp_path, loader):
    """The backbone was saved at 4ch and the declaration says 16: loading it
    would leave the latent convs at the resize's zeros."""
    ModelLoader, _ = loader
    vae = _tiny_vae(16)
    path = _write_checkpoint(
        tmp_path / "inconsistent.safetensors", 4,
        _vae_metadata(vae, 16, vae_embedded=True, vae_prefix="vae."),
        vae=vae)

    with pytest.raises(RuntimeError, match="reconstruction failed"):
        ModelLoader.reconstruct_sd_sdxl_pipeline(path, "sdxl", torch.float32, "cpu")


# ---------------------------------------------------------------------------
# 3. The capability gate (§9.7, §13.3)
# ---------------------------------------------------------------------------

def test_the_feature_is_declared_with_its_arming_key_and_a_label():
    from api.arch_capabilities import (
        TRAINING_FEATURE_LABELS, TRAINING_FEATURE_PARAMS,
    )
    assert TRAINING_FEATURE_PARAMS["vae_swap"] == ["vae_swap_source"]
    assert TRAINING_FEATURE_LABELS["vae_swap"]


def test_only_the_landed_waves_can_swap_and_only_by_full_finetune():
    """Waves 1 (sd15/sdxl), 2 (zimage/krea2) and 3 (anima/flux2/lens/minit2i).
    ltx2 is wave 2's refusal."""
    from api.arch_capabilities import (
        TRAINING_DECLARED_ARCHS, training_feature_unsupported_reason,
    )
    landed = {"sd15", "sdxl", "zimage", "krea2",
              "anima", "flux2", "lens", "minit2i"}
    for arch in sorted(landed):
        assert training_feature_unsupported_reason(
            arch, "vae_swap", "full_finetune") is None
        for method in ("lora", "relora", "controlnet"):
            assert training_feature_unsupported_reason(arch, "vae_swap", method)
    for arch in sorted(TRAINING_DECLARED_ARCHS - landed):
        assert training_feature_unsupported_reason(
            arch, "vae_swap", "full_finetune"), arch


def test_sensenova_stays_gated_behind_its_own_wave():
    """§10: this entry does not come off until §10.6's acceptance conditions."""
    from api.arch_capabilities import TRAINING_FEATURE_UNSUPPORTED
    assert TRAINING_FEATURE_UNSUPPORTED["sensenova"]["vae_swap"]["reason"]
