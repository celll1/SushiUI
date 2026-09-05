"""The training side of a VAE swap (design §8, phase P2b).

Covers the five things that can silently go wrong: the resize being driven by
the shared arch handler rather than an SDXL-only block, a legacy
``sdxl_vae_type`` config still resolving, the latent cache namespace staying
ADDITIVE, the preflight refusing a source that cannot be saved, and a bundled
VAE surviving the save/read round trip.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.models.common import vae_source as vs
from core.models.common.single_file_format import (
    build_component_metadata, parse_component_metadata,
)
from core.models.components.latent_io import verify_latent_io
from core.models.components.wiring import SD_UNET_LATENT_IO, SDXL_WIRING
from core.training.arch.sdxl import SDXLArchHandler
from core.training.vae_swap import (
    check_bundling, check_swap_method, legacy_source_from_config,
    preflight_vae_swap, resolve_vae_swap_source, swap_metadata,
)


# --- fixtures ---------------------------------------------------------------

def _unet(channels=4):
    unet = nn.Module()
    unet.conv_in = nn.Conv2d(channels, 8, 3, padding=1)
    unet.conv_out = nn.Conv2d(8, channels, 3, padding=1)
    unet.config = SimpleNamespace(in_channels=channels, out_channels=channels)
    return unet


def _resolved(channels=16, norm="shift_scale", norm_pack=1, form="registry",
              family="flux1", locator="registry:flux1", identity_native=False,
              struct_native=False, config=None, content_hash="abcdef0123456789"):
    return vs.ResolvedVAE(
        source=f"{form}:{family}", form=form, family=family,
        latent_channels=channels, scale_factor=8, scale_temporal=1, ndim=4,
        norm=norm, norm_pack=norm_pack, vae_class="AutoencoderKL",
        config=config or {"_class_name": "AutoencoderKL", "scaling_factor": 0.3611},
        content_hash=content_hash, provenance=f"{form}:{family}", locator=locator,
        struct_native=struct_native, identity_native=identity_native,
        scaling_factor=0.3611, shift_factor=0.1159,
    )


def _diffusers_vae_state(channels=16, downsamplers=3, seed=3):
    g = torch.Generator().manual_seed(seed)
    state = {
        "decoder.conv_in.weight": torch.rand(8, channels, 3, 3, generator=g),
        "encoder.conv_out.weight": torch.rand(2 * channels, 8, 3, 3, generator=g),
    }
    for i in range(downsamplers):
        state[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = torch.rand(
            8, 8, 3, 3, generator=g)
    return state


# ---------------------------------------------------------------------------
# 1. The resize is driven by the shared handler, not by an SDXL-only block
# ---------------------------------------------------------------------------

def test_apply_vae_swap_resizes_through_the_arch_handlers_wiring():
    trainer = SimpleNamespace(unet=_unet(4), vae=None, vae_dtype=None, config={})
    module = nn.Module()
    resolved = _resolved(channels=16, norm_pack=2)

    report = SDXLArchHandler().apply_vae_swap(trainer, resolved, module=module)

    assert report.replaced == ("unet.conv_in", "unet.conv_out")
    assert trainer.unet.conv_in.in_channels == 16
    assert trainer.unet.conv_out.out_channels == 16
    assert trainer.vae is module
    # The wiring the rest of the run reads is the grafted one, not the arch's.
    assert trainer.wiring.latent_channels == 16
    assert trainer.wiring.vae_norm_pack == 2
    assert SDXL_WIRING.latent_channels == 4 and SDXL_WIRING.vae_norm_pack == 1
    assert trainer.vae_identity is resolved
    assert trainer.vae_latent_channels == 16
    assert verify_latent_io(trainer, SD_UNET_LATENT_IO, 16) == []


def test_apply_vae_swap_copies_the_shared_channels_and_zeroes_the_rest():
    trainer = SimpleNamespace(unet=_unet(4), vae=None, vae_dtype=None, config={})
    before = trainer.unet.conv_in.weight.detach().clone()

    SDXLArchHandler().apply_vae_swap(trainer, _resolved(16), module=nn.Module())

    after = trainer.unet.conv_in.weight.detach()
    assert torch.equal(after[:, :4], before)
    assert torch.count_nonzero(after[:, 4:]) == 0


def test_apply_vae_swap_refuses_an_unsupported_new_channel_init():
    trainer = SimpleNamespace(unet=_unet(4), vae=None, vae_dtype=None,
                              config={"vae_swap_new_channel_init": "kaiming"})
    with pytest.raises(ValueError, match="new_channel_init"):
        SDXLArchHandler().apply_vae_swap(trainer, _resolved(16), module=nn.Module())


# ---------------------------------------------------------------------------
# 2. The legacy sdxl_vae_type alias
# ---------------------------------------------------------------------------

def test_legacy_sdxl_vae_type_still_resolves_to_a_registry_source():
    assert resolve_vae_swap_source({"sdxl_vae_type": "flux1"}) == "registry:flux1"
    assert legacy_source_from_config({"sdxl_vae_type": "FLUX1"}) == "registry:flux1"


def test_neutral_and_absent_legacy_values_mean_no_swap():
    for config in ({}, {"sdxl_vae_type": "none"}, {"sdxl_vae_type": "sdxl"},
                   {"sdxl_vae_type": ""}, {"vae_swap_source": ""}):
        assert resolve_vae_swap_source(config) == ""


def test_new_key_wins_over_the_legacy_alias():
    assert resolve_vae_swap_source(
        {"vae_swap_source": "file:/x/vae.safetensors",
         "sdxl_vae_type": "flux1"}) == "file:/x/vae.safetensors"


def test_a_generated_config_carries_both_keys_so_an_old_run_still_loads():
    from core.training.training_config import _build_train_section

    section = _build_train_section(
        {"sdxl_vae_type": "flux1"}, total_steps=10, epochs=None,
        train_unet=True, train_text_encoder=False)
    assert section["vae_swap_source"] == ""
    assert section["sdxl_vae_type"] == "flux1"
    assert resolve_vae_swap_source(section) == "registry:flux1"


def test_the_value_reaches_the_yaml_and_comes_back(tmp_path):
    """A Pydantic field and a vocabulary entry are not proof the VALUE travels:
    _build_train_section is an explicit whitelist."""
    import yaml

    from api.routes import _extract_request_params_from_yaml
    from core.training.training_config import TrainingConfigGenerator

    text = TrainingConfigGenerator.generate_full_finetune_config(
        {"total_steps": 1, "vae_swap_source": "registry:flux1"},
        run_name="swap", base_model_path="model.safetensors",
        output_dir=str(tmp_path), dataset_path=str(tmp_path))
    process = yaml.safe_load(text)["config"]["process"][0]
    assert process["train"]["vae_swap_source"] == "registry:flux1"
    assert process["train"]["vae_swap_new_channel_init"] == "zero"
    assert resolve_vae_swap_source(process["train"]) == "registry:flux1"

    restored = _extract_request_params_from_yaml(process, "full_finetune")
    assert restored["vae_swap_source"] == "registry:flux1"
    assert restored["vae_swap_new_channel_init"] == "zero"


# ---------------------------------------------------------------------------
# 3. The cache namespace token is additive
# ---------------------------------------------------------------------------

def _namespace(**attrs):
    from core.training.base_trainer import BaseTrainer

    stub = SimpleNamespace(
        arch=SimpleNamespace(name="sdxl"), sdxl_vae_type="sdxl",
        sdxl_te_type="none", vae_latent_channels=4, vae=None,
        vae_dtype=torch.float16, log_prefix="[test]",
    )
    for key, value in attrs.items():
        setattr(stub, key, value)
    return BaseTrainer._build_cache_namespace(stub)


def test_a_native_run_gets_no_vae_token():
    native = _namespace()
    assert native == "sdxl__c4__dtfloat16"
    # An identity-native resolution is the same "no swap" answer.
    assert _namespace(vae_identity=_resolved(4, identity_native=True,
                                             struct_native=True)) == native


def test_a_swapped_run_adds_one_token_and_changes_nothing_else():
    native = _namespace()
    swapped = _namespace(vae_identity=_resolved(16), vae_latent_channels=16)
    assert swapped == "sdxl__vae-flux1-abcdef01__c16__dtfloat16"
    # Additive: every component of the native namespace survives in order.
    assert [p for p in swapped.split("__") if not p.startswith("vae-")] == \
        [p if p != "c4" else p for p in native.replace("c4", "c16").split("__")]


def test_same_channels_different_vae_still_separates_by_hash():
    a = _namespace(vae_identity=_resolved(4, family="sdxl", content_hash="1111111122222222"))
    b = _namespace(vae_identity=_resolved(4, family="sdxl", content_hash="3333333344444444"))
    assert a != b
    assert a == "sdxl__vae-sdxl-11111111__c4__dtfloat16"


# ---------------------------------------------------------------------------
# 4. Preflight refusals
# ---------------------------------------------------------------------------

def test_an_extracted_vae_cannot_be_left_unbundled():
    with pytest.raises(ValueError, match="cannot be loaded"):
        check_bundling("model:M:/model/sdxl/other.safetensors", True)
    # The same source is fine when it is bundled (the default for a swap run).
    check_bundling("model:M:/model/sdxl/other.safetensors", False)
    check_bundling("registry:flux1", True)


def test_preflight_refuses_the_unbundleable_combination_before_touching_the_file():
    with pytest.raises(ValueError, match="cannot be loaded"):
        preflight_vae_swap(
            {"vae_swap_source": "model:M:/model/sdxl/does-not-exist.safetensors"},
            arch="sdxl", method="full_finetune", bundle_vae_explicit_false=True)


def test_preflight_refuses_a_swap_on_anything_but_a_full_finetune():
    for method in ("lora", "relora", "controlnet"):
        with pytest.raises(ValueError, match="full_finetune"):
            preflight_vae_swap({"vae_swap_source": "registry:flux1"},
                               arch="sdxl", method=method,
                               bundle_vae_explicit_false=False)
    check_swap_method("registry:flux1", "full")
    check_swap_method("registry:flux1", "full_finetune")


def test_preflight_answers_a_registry_source_without_the_file_being_present():
    assert preflight_vae_swap({"vae_swap_source": "registry:flux1"}, arch="sdxl",
                              method="full_finetune",
                              bundle_vae_explicit_false=False) == "registry:flux1"
    with pytest.raises(ValueError, match="unknown VAE registry key"):
        preflight_vae_swap({"vae_swap_source": "registry:nope"}, arch="sdxl",
                           method="full_finetune", bundle_vae_explicit_false=False)


def test_preflight_refuses_a_structurally_incompatible_family():
    # A 5-D video VAE cannot drive SDXL's 4-D latent path (§7.4).
    with pytest.raises(ValueError, match="cannot drive sdxl"):
        preflight_vae_swap({"vae_swap_source": "registry:qwen_image"}, arch="sdxl",
                           method="full_finetune", bundle_vae_explicit_false=False)
    # A BatchNorm VAE on sdxl is accepted since P7: P5's shared normalisation
    # layer handles the domain and custom_sampling now renders through it.
    assert preflight_vae_swap({"vae_swap_source": "registry:flux2"}, arch="sdxl",
                              method="full_finetune",
                              bundle_vae_explicit_false=False) == "registry:flux2"


def test_a_run_without_a_swap_is_not_checked_at_all():
    assert preflight_vae_swap({}, arch="sdxl", method="lora",
                              bundle_vae_explicit_false=True) == ""


def test_a_run_inheriting_a_bundled_swapped_base_cannot_unbundle_it(tmp_path):
    """The refusal has to reach a run that names no source of its own: the
    base's VAE exists only inside the base, so this save cannot omit it."""
    _r, _b, md = _swap_md()
    base = _write_checkpoint(tmp_path, _diffusers_vae_state(16), md,
                             name="base.safetensors")

    with pytest.raises(ValueError, match="cannot be loaded"):
        preflight_vae_swap({}, arch="sdxl", method="full_finetune",
                           bundle_vae_explicit_false=True, base_model_path=base)
    # Unset bundle_vae (the normal case) is not refused.
    assert preflight_vae_swap({}, arch="sdxl", method="full_finetune",
                              bundle_vae_explicit_false=False,
                              base_model_path=base) == ""


# ---------------------------------------------------------------------------
# 4b. strict_validation sees a latent I/O that does not match the VAE
# ---------------------------------------------------------------------------

class _FakeVAE(nn.Module):
    """Encodes to a fixed channel count at a fixed compression ratio."""

    def __init__(self, channels=16, scale=8):
        super().__init__()
        self.channels, self.scale = channels, scale
        self.probe = nn.Parameter(torch.zeros(1))

    def encode(self, sample):
        b, _c, h, w = sample.shape
        latent = torch.zeros(b, self.channels, h // self.scale, w // self.scale)
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=latent))


def _swapped_trainer(unet_channels=16, vae_channels=16, vae_scale=8):
    from core.training.vae_swap import validate_latent_io

    trainer = SimpleNamespace(
        unet=_unet(unet_channels), vae=_FakeVAE(vae_channels, vae_scale),
        vae_dtype=None, config={}, arch=SDXLArchHandler(),
        wiring=SDXL_WIRING.replace(latent_channels=vae_channels),
        vae_identity=_resolved(vae_channels),
    )
    return validate_latent_io(trainer)


def test_a_consistent_swapped_run_reports_nothing():
    assert _swapped_trainer() == []


def test_a_backbone_left_at_the_old_channel_count_is_reported():
    problems = _swapped_trainer(unet_channels=4)
    assert any("conv_in" in p and "4 channels, expected 16" in p for p in problems)
    assert any("conv_out" in p for p in problems)


def test_a_vae_with_the_wrong_compression_ratio_is_reported():
    problems = _swapped_trainer(vae_scale=16)
    assert any("compresses" in p and "expects 8x" in p for p in problems)


def test_a_native_run_is_not_probed_at_all():
    from core.training.vae_swap import validate_latent_io

    native = SimpleNamespace(unet=_unet(4), vae=None, config={},
                             arch=SDXLArchHandler(), wiring=SDXL_WIRING)
    assert validate_latent_io(native) == []


# ---------------------------------------------------------------------------
# 5. Bundled, then read back
# ---------------------------------------------------------------------------

def _write_checkpoint(tmp_path, vae_state, metadata, name="swapped.safetensors"):
    state = {"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(4, 16, 3, 3)}
    for key, value in (vae_state or {}).items():
        state[f"vae.{key}"] = value
    path = tmp_path / name
    save_file(state, str(path), metadata=metadata)
    return str(path)


def _swap_md(trainer_bundle_vae=None, resolved=None):
    trainer = SimpleNamespace(vae_identity=resolved or _resolved(16),
                              bundle_vae=trainer_bundle_vae,
                              arch=SimpleNamespace(name="sdxl"))
    return swap_metadata(trainer)


def test_swap_metadata_declares_every_key_the_reader_needs():
    resolved, bundled, md = _swap_md()
    assert bundled is True  # D7: a swap run bundles by default, for every arch
    parsed = parse_component_metadata(md)["vae"]
    assert parsed["channels"] == "16"
    assert parsed["embedded"] is True
    assert parsed["prefix"] == "vae."
    assert parsed["identity_native"] is False
    assert parsed["struct_native"] is False
    assert parsed["hash"] == resolved.content_hash
    assert parsed["norm"] == "shift_scale" and parsed["norm_pack"] == 1
    # The unobservable numbers travel in the declared config or the checkpoint
    # cannot be resolved on load.
    assert parsed["config"]["scaling_factor"] == pytest.approx(0.3611)


def test_a_bundled_swapped_checkpoint_reads_back_as_the_same_latent_space(tmp_path):
    state = _diffusers_vae_state(16)
    _resolved_vae, _bundled, md = _swap_md()
    path = _write_checkpoint(tmp_path, state, md)

    declared = vs.load_declared_latent_io(path, arch="sdxl")

    assert declared is not None
    assert declared.latent_channels == 16
    assert declared.scale_factor == 8
    assert declared.vae_class == "AutoencoderKL"
    assert declared.scaling_factor == pytest.approx(0.3611)
    assert declared.identity_native is False
    assert declared.struct_native is False
    # Trusted, not recomputed: the namespace a continuation run builds is the
    # one the swap run wrote.
    assert declared.content_hash == "abcdef0123456789"
    assert set(declared.state_dict) == set(state)


def test_the_sdxl_save_path_bundles_the_swapped_vae_and_reads_back(tmp_path):
    """The whole round trip through the shipped adapter: a swapped VAE never
    goes through the 4ch LDM converter, and the file it writes resolves."""
    from safetensors import safe_open

    from core.training.adapters.sdxl_adapter import SDXLFullParameterAdapter

    vae = nn.Module()
    for key, value in _diffusers_vae_state(16).items():
        target, _, name = key.rpartition(".")
        holder = vae
        for part in target.split("."):
            child = getattr(holder, part, None)
            if child is None:
                child = nn.Module()
                holder.add_module(part, child)
            holder = child
        holder.register_buffer(name, value)

    trainer = SimpleNamespace(
        train_unet=False, train_text_encoder=False, unet=None,
        text_encoder=None, text_encoder_2=None, vae=vae, bundle_vae=None,
        sdxl_vae_type="flux1", sdxl_te_type="none", vae_latent_channels=16,
        te_adapters=None, te_custom=None, noise_process="ddpm",
        prediction_target="epsilon", vae_identity=_resolved(16),
        arch=SimpleNamespace(name="sdxl"),
    )
    out = tmp_path / "swapped_run.safetensors"
    SDXLFullParameterAdapter(trainer).save_checkpoint(10, 1, out)

    with safe_open(str(out), framework="pt") as f:
        keys = set(f.keys())
        metadata = f.metadata()
    assert keys == {f"vae.{k}" for k in _diffusers_vae_state(16)}
    assert not any(k.startswith("first_stage_model.") for k in keys)
    assert metadata["component.vae.embedded"] == "1"
    assert metadata["component.vae.prefix"] == "vae."
    assert metadata["sushi.in_channels"] == "16"  # legacy marker still written

    declared = vs.load_declared_latent_io(str(out), arch="sdxl")
    assert declared is not None and declared.latent_channels == 16
    assert declared.identity_native is False


def test_a_native_checkpoint_declares_nothing_to_rebuild(tmp_path):
    md = build_component_metadata(vae_type="sdxl", vae_channels=4,
                                  vae_embedded=True, vae_prefix="first_stage_model.")
    path = _write_checkpoint(tmp_path, None, md, name="native.safetensors")
    assert vs.load_declared_latent_io(path, arch="sdxl") is None
    assert vs.load_declared_latent_io(path) is None


def test_a_legacy_sushi_checkpoint_is_still_recognised_as_swapped(tmp_path):
    path = _write_checkpoint(
        tmp_path, None,
        {"sushi.vae_type": "flux1", "sushi.in_channels": "16"},
        name="legacy.safetensors")
    declaration = vs.read_vae_declaration(
        {"sushi.vae_type": "flux1", "sushi.in_channels": "16"})
    assert declaration["locator"] == "registry:flux1"
    assert declaration["channels"] == 16
    assert declaration["identity_native"] is False
    assert declaration["embedded"] is False
    del path


def test_a_moved_or_replaced_locator_target_is_refused(tmp_path):
    vae_path = tmp_path / "standalone.safetensors"
    save_file(_diffusers_vae_state(16), str(vae_path),
              metadata={"component.vae.config": json.dumps(
                  {"_class_name": "AutoencoderKL", "scaling_factor": 0.3611})})
    md = build_component_metadata(
        vae_type="custom", vae_channels=16, vae_embedded=False,
        vae_locator=f"path:{vae_path}", vae_hash="0000000000000000",
        vae_struct_native=False, vae_identity_native=False)
    path = _write_checkpoint(tmp_path, None, md, name="unbundled.safetensors")

    with pytest.raises(vs.VaeSourceError, match="refusing"):
        vs.load_declared_latent_io(path, arch="sdxl")


def test_an_unbundled_checkpoint_without_a_locator_is_refused(tmp_path):
    md = build_component_metadata(vae_type="custom", vae_channels=16,
                                  vae_embedded=False, vae_struct_native=False,
                                  vae_identity_native=False)
    path = _write_checkpoint(tmp_path, None, md, name="orphan.safetensors")
    with pytest.raises(vs.VaeSourceError, match="no resolvable locator"):
        vs.load_declared_latent_io(path, arch="sdxl")
