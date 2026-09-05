"""Resolving where a swapped VAE comes from (design §7, phase P2a).

Covers the three source forms, the ordered-prefix extraction against BOTH
bundling conventions, the rule that a scaling factor is never guessed, and the
structure/identity split the rest of the migration keys off.
"""

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.models.common import vae_source as vs
from core.models.common.single_file_format import build_component_metadata
from core.models.common.vae_store import LDM_SINGLE_FILE_DEFAULT_SCALING_FACTOR


# --- fixtures ---------------------------------------------------------------

def _ldm_vae_state(channels=4, downsamplers=3, seed=0):
    """An original/LDM-keyed AutoencoderKL: one downsampler per halving."""
    g = torch.Generator().manual_seed(seed)
    state = {
        "decoder.conv_in.weight": torch.rand(8, channels, 3, 3, generator=g),
        "encoder.conv_out.weight": torch.rand(2 * channels, 8, 3, 3, generator=g),
    }
    for i in range(downsamplers):
        state[f"encoder.down.{i}.downsample.conv.weight"] = torch.rand(
            8, 8, 3, 3, generator=g)
    return state


def _diffusers_vae_state(channels=16, downsamplers=3, batchnorm=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    state = {
        "decoder.conv_in.weight": torch.rand(8, channels, 3, 3, generator=g),
        "encoder.conv_out.weight": torch.rand(2 * channels, 8, 3, 3, generator=g),
    }
    for i in range(downsamplers):
        state[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = torch.rand(
            8, 8, 3, 3, generator=g)
    if batchnorm is not None:
        state["bn.running_mean"] = torch.zeros(batchnorm)
        state["bn.running_var"] = torch.ones(batchnorm)
    return state


def _write_single_file(tmp_path, state, metadata=None, name="vae.safetensors"):
    path = tmp_path / name
    save_file(state, str(path), metadata=metadata or {})
    return str(path)


def _write_diffusers_dir(tmp_path, state, config, name="vae"):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    save_file(state, str(directory / "diffusion_pytorch_model.safetensors"))
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(directory)


def _full_checkpoint(tmp_path, vae_state, prefixes=("first_stage_model.",),
                     name="model.safetensors", metadata=None):
    state = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(8, 4, 3, 3),
        "model.diffusion_model.out.2.weight": torch.zeros(4, 8, 3, 3),
    }
    for prefix in prefixes:
        for key, value in vae_state.items():
            state[f"{prefix}{key}"] = value
    return _write_single_file(tmp_path, state, metadata=metadata, name=name)


_SDXL_CONFIG = {"_class_name": "AutoencoderKL", "latent_channels": 4,
                "scaling_factor": 0.13025, "block_out_channels": [128, 256, 512, 512]}


# --- source strings ---------------------------------------------------------

def test_the_three_source_forms_parse_and_nothing_else_does():
    assert vs.parse_vae_source("registry:flux1") == ("registry", "flux1")
    assert vs.parse_vae_source("file:M:/vae/x.safetensors") == (
        "file", "M:/vae/x.safetensors")
    assert vs.parse_vae_source(" model:M:/m/flux2.safetensors ") == (
        "model", "M:/m/flux2.safetensors")
    for bad in ("", "flux1", "registry:", "vae:flux1", "M:/vae/x.safetensors"):
        with pytest.raises(vs.VaeSourceError):
            vs.parse_vae_source(bad)


# --- file: ------------------------------------------------------------------

def test_a_standalone_diffusers_directory_resolves_from_its_own_config(tmp_path):
    directory = _write_diffusers_dir(tmp_path, _diffusers_vae_state(channels=16),
                                     {"_class_name": "AutoencoderKL",
                                      "latent_channels": 16,
                                      "scaling_factor": 0.3611,
                                      "shift_factor": 0.1159})
    resolved = vs.resolve_vae_source(f"file:{directory}")
    assert resolved.form == "file" and resolved.family == "custom"
    assert resolved.latent_channels == 16
    assert resolved.scale_factor == 8 and resolved.scale_temporal == 1
    assert resolved.ndim == 4 and resolved.norm == "shift_scale"
    assert resolved.scaling_factor == pytest.approx(0.3611)
    assert resolved.shift_factor == pytest.approx(0.1159)
    assert resolved.locator == f"path:{directory}"
    assert resolved.provenance == "file:vae"
    assert resolved.content_hash and len(resolved.content_hash) == 16


def test_a_single_file_vae_resolves_through_its_own_declaration(tmp_path):
    md = build_component_metadata(vae_type="sdxl", vae_channels=4,
                                  vae_class="AutoencoderKL", vae_config=_SDXL_CONFIG)
    path = _write_single_file(tmp_path, _ldm_vae_state(), metadata=md)
    resolved = vs.resolve_vae_source(f"file:{path}")
    assert resolved.family == "sdxl"          # a declared registry family
    assert resolved.latent_channels == 4 and resolved.scale_factor == 8
    assert resolved.vae_class == "AutoencoderKL"
    assert resolved.scaling_factor == pytest.approx(0.13025)
    assert resolved.state_dict is not None


def test_an_undeclared_single_file_is_refused_not_given_the_ldm_default(tmp_path):
    """vae_store.py:47-58: from_single_file cannot tell SDXL from SD1.5 and
    substitutes 0.18215, a 1.40x error. This resolver refuses instead."""
    path = _write_single_file(tmp_path, _ldm_vae_state())
    with pytest.raises(vs.VaeSourceError) as excinfo:
        vs.resolve_vae_source(f"file:{path}")
    assert str(LDM_SINGLE_FILE_DEFAULT_SCALING_FACTOR) not in str(excinfo.value)
    assert "normalisation" in str(excinfo.value)


def test_a_declared_norm_without_a_number_is_still_refused(tmp_path):
    md = build_component_metadata(vae_type="custom", vae_norm="shift_scale")
    path = _write_single_file(tmp_path, _ldm_vae_state(), metadata=md)
    with pytest.raises(vs.VaeSourceError) as excinfo:
        vs.resolve_vae_source(f"file:{path}")
    assert "scaling_factor" in str(excinfo.value)


def test_observation_beats_a_declaration_that_disagrees_with_the_weights(tmp_path):
    md = build_component_metadata(vae_type="custom", vae_channels=99,
                                  vae_scale_factor=64, vae_config=_SDXL_CONFIG)
    path = _write_single_file(tmp_path, _ldm_vae_state(channels=4, downsamplers=3),
                              metadata=md)
    resolved = vs.resolve_vae_source(f"file:{path}")
    assert resolved.latent_channels == 4 and resolved.scale_factor == 8


def test_a_file_that_holds_no_vae_is_refused(tmp_path):
    path = _write_single_file(tmp_path, {"transformer.x": torch.zeros(2, 2)})
    with pytest.raises(vs.VaeSourceError):
        vs.resolve_vae_source(f"file:{path}")


# --- registry: --------------------------------------------------------------

def test_a_registry_key_carries_the_family_scaling_when_the_config_omits_it(
        tmp_path, monkeypatch):
    directory = _write_diffusers_dir(
        tmp_path, _diffusers_vae_state(channels=4),
        {"_class_name": "AutoencoderKL", "latent_channels": 4})   # no scaling_factor
    monkeypatch.setattr(vs, "resolve_vae_dir", lambda key, download=True: directory)
    resolved = vs.resolve_vae_source("registry:sdxl")
    assert resolved.family == "sdxl" and resolved.locator == "registry:sdxl"
    assert resolved.provenance == "registry:sdxl"
    assert resolved.scaling_factor == pytest.approx(0.13025)   # table A, not 0.18215


def test_an_unknown_registry_key_is_refused():
    with pytest.raises(vs.VaeSourceError):
        vs.resolve_vae_source("registry:not_a_family")


# --- model: (extraction) ----------------------------------------------------

@pytest.mark.parametrize("prefix", ["first_stage_model.", "vae."])
def test_both_bundling_conventions_extract(tmp_path, prefix):
    vae_state = _ldm_vae_state()
    md = build_component_metadata(vae_type="sdxl", vae_config=_SDXL_CONFIG,
                                  vae_embedded=True, vae_prefix=prefix)
    path = _full_checkpoint(tmp_path, vae_state, prefixes=(prefix,), metadata=md)
    resolved = vs.resolve_vae_source(f"model:{path}")
    assert resolved.form == "model" and resolved.prefix == prefix
    assert set(resolved.state_dict) == set(vae_state)
    assert resolved.latent_channels == 4
    assert resolved.provenance == "extracted:model"
    # An extracted VAE has no locator: it exists only inside its source file (§8.7).
    assert resolved.locator is None


def test_the_prefix_list_is_ordered_and_vae_wins(tmp_path):
    own = _ldm_vae_state(channels=16, seed=7)
    other = _ldm_vae_state(channels=4, seed=8)
    state = {f"vae.{k}": v for k, v in own.items()}
    state.update({f"first_stage_model.{k}": v for k, v in other.items()})
    state["model.diffusion_model.input_blocks.0.0.weight"] = torch.zeros(8, 4, 3, 3)
    md = build_component_metadata(vae_type="custom", vae_config={
        "_class_name": "AutoencoderKL", "scaling_factor": 0.3611})
    path = _write_single_file(tmp_path, state, metadata=md, name="both.safetensors")
    resolved = vs.resolve_vae_source(f"model:{path}")
    assert resolved.prefix == "vae." and resolved.latent_channels == 16


def test_a_checkpoint_without_any_vae_is_refused(tmp_path):
    path = _full_checkpoint(tmp_path, {}, prefixes=())
    with pytest.raises(vs.VaeSourceError) as excinfo:
        vs.resolve_vae_source(f"model:{path}")
    assert "vae." in str(excinfo.value)


def test_a_standalone_vae_named_as_a_model_is_refused(tmp_path):
    path = _write_single_file(tmp_path, {f"vae.{k}": v
                                         for k, v in _ldm_vae_state().items()})
    with pytest.raises(vs.VaeSourceError) as excinfo:
        vs.resolve_vae_source(f"model:{path}")
    assert "file:" in str(excinfo.value)


def test_an_extracted_vae_hashes_the_same_as_the_standalone_one(tmp_path):
    vae_state = _ldm_vae_state(seed=3)
    md = build_component_metadata(vae_type="sdxl", vae_config=_SDXL_CONFIG)
    standalone = _write_single_file(tmp_path, vae_state, metadata=md,
                                    name="alone.safetensors")
    bundled = _full_checkpoint(tmp_path, vae_state, metadata=md)
    a = vs.resolve_vae_source(f"file:{standalone}")
    b = vs.resolve_vae_source(f"model:{bundled}")
    assert a.content_hash == b.content_hash

    moved = _write_single_file(tmp_path, _ldm_vae_state(seed=4), metadata=md,
                               name="other.safetensors")
    assert vs.resolve_vae_source(f"file:{moved}").content_hash != a.content_hash


# --- observation ------------------------------------------------------------

def test_a_batchnorm_vae_declares_the_domain_its_statistics_live_on():
    shapes = {k: tuple(v.shape)
              for k, v in _diffusers_vae_state(channels=32, batchnorm=128).items()}
    observed = vs.observe_vae(shapes)
    assert observed["norm"] == "batchnorm"
    assert observed["norm_pack"] == 2          # 128 / 32 = 4 -> 2x2 packed
    assert observed["vae_class"] == "AutoencoderKLFlux2"


def test_batchnorm_statistics_that_do_not_pack_squarely_are_refused():
    shapes = {k: tuple(v.shape)
              for k, v in _diffusers_vae_state(channels=32, batchnorm=96).items()}
    with pytest.raises(vs.VaeSourceError):
        vs.observe_vae(shapes)


def test_a_five_dimensional_decoder_reports_five_dimensional_latents():
    shapes = {"decoder.conv_in.weight": (8, 16, 3, 3, 3)}
    observed = vs.observe_vae(shapes)
    assert observed["ndim"] == 5
    # A 3-D stack's ratios are not read off the 2-D key convention.
    assert "scale_factor" not in observed and "scale_temporal" not in observed


# --- struct_native / identity_native ---------------------------------------

def test_the_architectures_own_shape_is_struct_native_but_identity_needs_a_hash(
        tmp_path):
    md = build_component_metadata(vae_type="sdxl", vae_config=_SDXL_CONFIG)
    path = _write_single_file(tmp_path, _ldm_vae_state(channels=4), metadata=md)
    resolved = vs.resolve_vae_source(f"file:{path}", arch="sdxl")
    assert resolved.struct_native is True
    assert resolved.identity_native is None      # unknown, never False by default

    same = vs.resolve_vae_source(f"file:{path}", arch="sdxl",
                                 native_hash=resolved.content_hash)
    assert (same.struct_native, same.identity_native) == (True, True)
    other = vs.resolve_vae_source(f"file:{path}", arch="sdxl",
                                  native_hash="0000000000000000")
    assert (other.struct_native, other.identity_native) == (True, False)


def test_a_different_channel_count_is_neither_structurally_nor_identically_native(
        tmp_path):
    directory = _write_diffusers_dir(tmp_path, _diffusers_vae_state(channels=16),
                                     {"_class_name": "AutoencoderKL",
                                      "latent_channels": 16,
                                      "scaling_factor": 0.3611})
    resolved = vs.resolve_vae_source(f"file:{directory}", arch="sdxl")
    assert resolved.struct_native is False and resolved.identity_native is False


def test_no_architecture_means_both_flags_stay_unknown(tmp_path):
    md = build_component_metadata(vae_type="sdxl", vae_config=_SDXL_CONFIG)
    path = _write_single_file(tmp_path, _ldm_vae_state(), metadata=md)
    resolved = vs.resolve_vae_source(f"file:{path}")
    assert resolved.struct_native is None and resolved.identity_native is None


# --- the family gate (§7.4) -------------------------------------------------

def test_more_channels_at_the_same_geometry_is_the_supported_swap():
    facts = {"latent_channels": 16, "scale_factor": 8, "scale_temporal": 1,
             "ndim": 4, "norm": "shift_scale"}
    assert vs.check_vae_compatibility(facts, "sdxl") == (True, None)


def test_a_different_geometry_is_refused_with_a_reason():
    for field, value in (("scale_factor", 16), ("ndim", 5), ("scale_temporal", 4)):
        facts = {"latent_channels": 16, "scale_factor": 8, "scale_temporal": 1,
                 "ndim": 4, "norm": "shift_scale", field: value}
        compatible, reason = vs.check_vae_compatibility(facts, "sdxl")
        assert compatible is False and reason


def test_crossing_the_normalisation_domain_is_refused_until_that_layer_lands():
    bn = {"latent_channels": 32, "scale_factor": 8, "scale_temporal": 1,
          "ndim": 4, "norm": "batchnorm"}
    assert vs.check_vae_compatibility(bn, "sdxl")[0] is False
    plain = dict(bn, norm="shift_scale")
    assert vs.check_vae_compatibility(plain, "flux2")[0] is False
    assert vs.check_vae_compatibility(bn, "flux2")[0] is True


def test_sensenova_accepts_any_ratio_and_says_what_it_costs():
    facts = {"latent_channels": 16, "scale_factor": 16, "scale_temporal": 1,
             "ndim": 4, "norm": "shift_scale"}
    assert vs.check_vae_compatibility(facts, "sensenova") == (True, None)
    assert vs.sensenova_token_geometry(8)["token_pixel_width"] == 32
    assert vs.sensenova_token_geometry(16)["token_pixel_width"] == 64
    band8 = vs.sensenova_token_geometry(8)["resolution_band_px"]
    band16 = vs.sensenova_token_geometry(16)["resolution_band_px"]
    assert band8 == [3_000_000, 5_000_000]
    assert band16 == [4 * band8[0], 4 * band8[1]]


def test_a_pixel_space_architecture_without_a_migration_is_refused():
    facts = {"latent_channels": 16, "scale_factor": 8, "scale_temporal": 1,
             "ndim": 4, "norm": "shift_scale"}
    compatible, reason = vs.check_vae_compatibility(facts, "minit2i")
    assert compatible is False and "pixel-space" in reason


# --- the cheap listing path -------------------------------------------------

def test_describing_a_candidate_reads_no_tensor_data(tmp_path, monkeypatch):
    directory = _write_diffusers_dir(tmp_path, _diffusers_vae_state(channels=16),
                                     {"_class_name": "AutoencoderKL",
                                      "latent_channels": 16,
                                      "scaling_factor": 0.3611})

    def _no_tensors(self, keys):
        raise AssertionError("a candidate listing must not read tensor data")

    monkeypatch.setattr(vs._WeightFile, "tensors", _no_tensors)
    described = vs.describe_vae_source(f"file:{directory}", arch="sdxl")
    assert described["latent_channels"] == 16 and described["compatible"] is True
    assert "content_hash" not in described


def test_an_unresolvable_candidate_is_described_not_raised(tmp_path):
    path = _write_single_file(tmp_path, _ldm_vae_state())     # no declaration
    described = vs.describe_vae_source(f"file:{path}", arch="sdxl")
    assert described["compatible"] is False and described["reason"]
