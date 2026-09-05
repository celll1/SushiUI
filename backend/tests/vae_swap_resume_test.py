"""Resuming a swapped-VAE checkpoint (design §8.2, follow-up to phase P2c).

The resume path decided "SushiUI custom arch" from ``sushi.vae_type`` alone,
which only a REGISTRY swap ever writes: a ``file:``/``model:`` swap — and every
sd15 swap — took the plain ``from_single_file`` path, rebuilt the backbone at
the architecture's native channel count and continued training in the wrong
latent space.

These tests drive the real ``BaseTrainer._load_checkpoint_as_base`` with
diffusers' ``from_single_file`` stubbed, and assert the SAVE -> RESUME -> SAVE
round trip leaves the checkpoint's declaration byte-identical: a resumed run
must not degrade its own ``component.vae.*`` block.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.models.common import vae_source as vs
from core.training.base_trainer import BaseTrainer


# --- fixtures ---------------------------------------------------------------

def _tiny_vae(latent_channels=16):
    """A real AutoencoderKL, small enough to build in milliseconds."""
    from diffusers import AutoencoderKL
    return AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,), layers_per_block=1, norm_num_groups=4,
        latent_channels=latent_channels, sample_size=32,
    )


def _vae_config(vae):
    # A saved config.json always carries _class_name; a config built in-process
    # does not, and the resolver refuses rather than guess a class.
    return {**dict(vae.config), "_class_name": "AutoencoderKL"}


class _FakeUNet(nn.Module):
    """Enough of UNet2DConditionModel for the resize and the conv reload."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 8, 3, padding=1)
        self.conv_out = nn.Conv2d(8, in_channels, 3, padding=1)
        self.config = SimpleNamespace(in_channels=in_channels,
                                      out_channels=in_channels)

    def register_to_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.config, key, value)


class _TextProjection(nn.Module):
    """One key that survives the OpenCLIP converter, so the saved file carries
    the ``conditioner.embedders.1`` marker the SDXL arm detects."""

    def __init__(self):
        super().__init__()
        self.text_projection = nn.Linear(4, 4, bias=False)


@pytest.fixture
def loader(monkeypatch):
    """Stub from_single_file everywhere the two load paths reach it."""
    import diffusers

    from core import model_loader as ml

    calls = []

    class _Recorder:
        def from_single_file(self, path, **kwargs):
            calls.append(kwargs)
            unet = _FakeUNet(int(kwargs.get("num_in_channels") or 4))
            # No vae kwarg = the checkpoint's own embedded VAE, as diffusers
            # would have built it.
            return SimpleNamespace(
                unet=unet, vae=kwargs.get("vae") or _tiny_vae(4),
                text_encoder=nn.Linear(2, 2), tokenizer=object(),
                text_encoder_2=_TextProjection(), tokenizer_2=object(),
                scheduler=object())

    recorder = _Recorder()
    monkeypatch.setattr(ml, "StableDiffusionXLPipeline", recorder)
    monkeypatch.setattr(ml, "StableDiffusionPipeline", recorder)
    # The plain resume branch imports these from diffusers at call time.
    monkeypatch.setattr(diffusers, "StableDiffusionXLPipeline", recorder)
    monkeypatch.setattr(diffusers, "StableDiffusionPipeline", recorder)
    return calls


def _save_trainer(arch, vae, identity, unet_channels=16, **overrides):
    """A trainer stand-in carrying exactly what the save path reads."""
    trainer = SimpleNamespace(
        train_unet=True, train_text_encoder=(arch == "sdxl"),
        unet=_FakeUNet(unet_channels), text_encoder=None,
        text_encoder_2=_TextProjection() if arch == "sdxl" else None,
        vae=vae, bundle_vae=None, sdxl_te_type="none",
        vae_latent_channels=identity.latent_channels,
        te_adapters=None, te_custom=None, noise_process="ddpm",
        prediction_target="epsilon", vae_identity=identity,
        arch=SimpleNamespace(name=arch),
    )
    from core.training.vae_swap import legacy_vae_type_marker
    trainer.sdxl_vae_type = legacy_vae_type_marker(identity)
    for key, value in overrides.items():
        setattr(trainer, key, value)
    return trainer


def _save(trainer, arch, path):
    if arch == "sdxl":
        from core.training.adapters.sdxl_adapter import SDXLFullParameterAdapter
        SDXLFullParameterAdapter(trainer).save_checkpoint(10, 1, path)
    else:
        from core.training.adapters.sd15_adapter import SD15FullParameterAdapter
        SD15FullParameterAdapter(trainer).save_checkpoint(10, 1, path)
    return str(path)


def _declaration(path):
    """Every metadata key that says which latent space this checkpoint is in."""
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
    return {k: v for k, v in metadata.items()
            if k.startswith("component.vae.") or k.startswith("sushi.")}


def _resume(path, **overrides):
    """Run the real resume loader over ``path`` and return the trainer state."""
    trainer = SimpleNamespace(
        log_prefix="[test]", device="cpu", weight_dtype=torch.float32,
        dtype=torch.float32, vae_dtype=torch.float32, config={},
        use_flash_attention=False, attention_backend="native",
        gradient_checkpointing=False, blocks_to_swap=0,
        bundle_vae=None, train_unet=True, train_text_encoder=True,
        noise_process="ddpm", prediction_target="epsilon",
        te_adapters=None, te_custom=None,
    )
    for key, value in overrides.items():
        setattr(trainer, key, value)
    BaseTrainer._load_checkpoint_as_base(trainer, str(path))
    trainer.arch = SimpleNamespace(name="sdxl" if trainer.is_sdxl else "sd15")
    return trainer


# ---------------------------------------------------------------------------
# 1. The round trip: save -> resume -> save leaves the declaration unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["sdxl", "sd15"])
def test_a_file_declared_swap_survives_save_resume_save(tmp_path, loader, arch):
    """The assertion this whole fix exists for. A ``file:`` VAE (and every sd15
    swap) declares itself through component.vae.* ONLY, so a resume that reads
    sushi.vae_type sees nothing, rebuilds at 4ch, and the next save writes a
    native declaration over a swapped one."""
    vae = _tiny_vae(16)
    identity = vs.resolve_vae_source(f"file:{_standalone(tmp_path, vae)}",
                                     arch=arch)
    assert identity.form == "file" and identity.family == "custom"

    first = _save(_save_trainer(arch, vae, identity), arch,
                  tmp_path / "step10.safetensors")
    declaration = _declaration(first)
    assert declaration["component.vae.identity_native"] == "0"
    # The legacy key cannot express a file: VAE, so it is absent — which is
    # exactly what the old detector keyed on.
    assert "sushi.vae_type" not in declaration

    trainer = _resume(first)

    assert trainer.is_sdxl is (arch == "sdxl")
    assert trainer.vae_latent_channels == 16
    assert trainer.unet.conv_in.in_channels == 16
    assert trainer.unet.conv_out.out_channels == 16
    # The trained convs, not the resize's zeros.
    assert torch.count_nonzero(trainer.unet.conv_in.weight) > 0
    assert trainer.vae_identity is not None
    assert trainer.vae_identity.identity_native is False
    assert trainer.vae_identity.content_hash == identity.content_hash
    assert trainer.wiring.latent_channels == 16
    assert trainer.base_vae_identity is not None
    # Facts only: the session must not hold a second copy of the VAE weights.
    assert trainer.vae_identity.state_dict is None

    second = _save(_save_trainer(arch, trainer.vae, trainer.vae_identity,
                                 sdxl_vae_type=trainer.sdxl_vae_type),
                   arch, tmp_path / "step20.safetensors")

    assert _declaration(second) == declaration


def test_the_resumed_checkpoint_still_loads_as_the_same_latent_space(
        tmp_path, loader):
    """The second save is not just declared the same — it reads back the same."""
    vae = _tiny_vae(16)
    identity = vs.resolve_vae_source(f"file:{_standalone(tmp_path, vae)}",
                                     arch="sdxl")
    first = _save(_save_trainer("sdxl", vae, identity), "sdxl",
                  tmp_path / "step10.safetensors")
    trainer = _resume(first)
    second = _save(_save_trainer("sdxl", trainer.vae, trainer.vae_identity),
                   "sdxl", tmp_path / "step20.safetensors")

    a = vs.load_declared_latent_io(first, arch="sdxl")
    b = vs.load_declared_latent_io(second, arch="sdxl")
    assert b is not None
    assert (b.latent_channels, b.family, b.content_hash, b.struct_native,
            b.identity_native, b.locator, b.scaling_factor) == \
           (a.latent_channels, a.family, a.content_hash, a.struct_native,
            a.identity_native, a.locator, a.scaling_factor)
    assert set(b.state_dict) == set(a.state_dict)


def _standalone(tmp_path, vae):
    import json

    from safetensors.torch import save_file
    path = tmp_path / "standalone_vae.safetensors"
    save_file(dict(vae.state_dict()), str(path),
              metadata={"component.vae.config": json.dumps(_vae_config(vae))})
    return path


# ---------------------------------------------------------------------------
# 2. The legacy marker keeps its legacy meaning
# ---------------------------------------------------------------------------

def test_sdxl_vae_type_never_receives_a_non_registry_family(tmp_path, loader):
    """``sdxl_vae_type`` is written back out as sushi.vae_type; "custom" there
    would make the next load resolve registry:custom."""
    vae = _tiny_vae(16)
    identity = vs.resolve_vae_source(f"file:{_standalone(tmp_path, vae)}",
                                     arch="sdxl")
    first = _save(_save_trainer("sdxl", vae, identity), "sdxl",
                  tmp_path / "step10.safetensors")

    trainer = _resume(first)

    assert trainer.sdxl_vae_type == "sdxl"
    second = _save(_save_trainer("sdxl", trainer.vae, trainer.vae_identity,
                                 sdxl_vae_type=trainer.sdxl_vae_type),
                   "sdxl", tmp_path / "step20.safetensors")
    assert "sushi.vae_type" not in _declaration(second)


def test_a_legacy_sushi_checkpoint_resumes_exactly_as_before(tmp_path, loader,
                                                             monkeypatch):
    """A pre-migration swapped SDXL declares only sushi.vae_type/in_channels.
    It must keep resuming — and keep naming its registry family."""
    from safetensors.torch import save_file

    vae = _tiny_vae(16)
    unet = _FakeUNet(16)
    state = {f"model.diffusion_model.{k}": v for k, v in (
        ("input_blocks.0.0.weight", unet.conv_in.weight.detach()),
        ("input_blocks.0.0.bias", unet.conv_in.bias.detach()),
        ("out.2.weight", unet.conv_out.weight.detach()),
        ("out.2.bias", unet.conv_out.bias.detach()),
    )}
    state["conditioner.embedders.1.model.text_projection"] = torch.zeros(4, 4)
    path = tmp_path / "legacy.safetensors"
    save_file(state, str(path), metadata={
        "modelspec.architecture": "sdxl-custom",
        "sushi.vae_type": "flux1", "sushi.in_channels": "16"})

    monkeypatch.setattr(vs, "_registry_source", lambda key, download: "unused")
    monkeypatch.setattr(
        vs, "resolve_vae_source",
        lambda source, **kw: vs.ResolvedVAE(
            source=source, form="registry", family="flux1", latent_channels=16,
            scale_factor=8, scale_temporal=1, ndim=4, norm="shift_scale",
            norm_pack=1, vae_class="AutoencoderKL", config=_vae_config(vae),
            content_hash="1111111111111111", provenance="registry:flux1",
            locator="registry:flux1", struct_native=False, identity_native=False,
            scaling_factor=0.3611, shift_factor=None,
            state_dict=dict(vae.state_dict())))

    trainer = _resume(str(path))

    assert trainer.sdxl_vae_type == "flux1"
    assert trainer.vae_latent_channels == 16
    assert trainer.unet.conv_in.in_channels == 16
    assert trainer.vae_identity.family == "flux1"
    assert trainer.vae_identity.identity_native is False
    # The upgrade the resume owes the next save: a full component.vae.* block
    # beside the legacy pair it keeps writing.
    second = _save(_save_trainer("sdxl", trainer.vae, trainer.vae_identity,
                                 sdxl_vae_type=trainer.sdxl_vae_type),
                   "sdxl", tmp_path / "step20.safetensors")
    declaration = _declaration(second)
    assert declaration["sushi.vae_type"] == "flux1"
    assert declaration["sushi.in_channels"] == "16"
    assert declaration["component.vae.channels"] == "16"
    assert declaration["component.vae.type"] == "flux1"
    assert declaration["component.vae.hash"] == "1111111111111111"
    # D7: a swap run bundles by default, so the upgraded block carries the VAE
    # itself rather than the registry locator the legacy pair implied.
    assert declaration["component.vae.embedded"] == "1"
    assert declaration["component.vae.prefix"] == "vae."


# ---------------------------------------------------------------------------
# 3. A native checkpoint is untouched
# ---------------------------------------------------------------------------

def test_a_native_checkpoint_still_takes_the_plain_path(tmp_path, loader):
    from safetensors.torch import save_file

    vae = _tiny_vae(4)
    state = {"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(8, 4, 3, 3)}
    state.update({f"first_stage_model.{k}": v for k, v in vae.state_dict().items()})
    path = tmp_path / "native.safetensors"
    save_file(state, str(path), metadata={
        "modelspec.architecture": "stable-diffusion-xl-v1-base"})

    trainer = _resume(str(path))

    assert getattr(trainer, "vae_identity", None) is None
    assert trainer.sdxl_vae_type == "sdxl"
    # Plain from_single_file: no channel override was requested.
    assert loader[-1].get("num_in_channels") is None
    assert BaseTrainer._build_cache_namespace(
        SimpleNamespace(arch=SimpleNamespace(name="sdxl"), sdxl_vae_type="sdxl",
                        sdxl_te_type="none", vae_latent_channels=4,
                        vae=None, vae_dtype=torch.float16)
    ) == "sdxl__c4__dtfloat16"
