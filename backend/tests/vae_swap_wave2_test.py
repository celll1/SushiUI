"""Wave 2 of the VAE swap: zimage, krea2 and ltx2's refusal (design phase P6).

The three things this wave can get silently wrong, one section each: the packed
channel algebra (zimage is the only arch whose two sides are BOTH "inner" order,
krea2 the only one whose config counts packed features), which candidate each
arch's family gate admits, and whether a saved swap reads back as the same
latent space through the sharded single-file writer these three save with.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.models.common import vae_source as vs
from core.models.common.single_file_format import (
    build_component_metadata, save_single_file_state,
)
from core.models.components.latent_io import resize_latent_io, verify_latent_io
from core.models.components.wiring import (
    KREA2_WIRING, LTX2_WIRING, ZIMAGE_WIRING,
)
from core.training.vae_swap import preflight_vae_swap, swap_metadata


# --- fixtures ---------------------------------------------------------------

def _zimage(in_channels=16):
    """A real ZImageTransformer2DModel, 1 layer, built in milliseconds."""
    from core.models.zimage_transformer import ZImageTransformer2DModel
    return ZImageTransformer2DModel(
        all_patch_size=(2,), all_f_patch_size=(1,), in_channels=in_channels,
        dim=96, n_layers=1, n_refiner_layers=1, n_heads=2, n_kv_heads=2,
        norm_eps=1e-5, qk_norm=True, cap_feat_dim=32, rope_theta=10000.0,
        t_scale=1000.0, axes_dims=[16, 16, 16], axes_lens=[64, 64, 64])


def _krea2(in_channels=64):
    from core.models.krea2.vendor.transformer import Krea2Transformer2DModel
    return Krea2Transformer2DModel(
        in_channels=in_channels, num_layers=1, attention_head_dim=128,
        num_attention_heads=1, num_key_value_heads=1, intermediate_size=64,
        timestep_embed_dim=32, text_hidden_dim=32, num_text_layers=1,
        text_num_attention_heads=1, text_num_key_value_heads=1,
        text_intermediate_size=32, num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1)


def _tiny_vae(latent_channels):
    from diffusers import AutoencoderKL
    return AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(4, 4, 4, 4), layers_per_block=1, norm_num_groups=4,
        latent_channels=latent_channels, sample_size=32)


def _facts(**overrides):
    facts = {"latent_channels": 16, "ndim": 4, "scale_factor": 8,
             "scale_temporal": 1, "norm": "shift_scale",
             "vae_class": "AutoencoderKL"}
    facts.update(overrides)
    return facts


def _resolved(channels=4, **overrides):
    fields = dict(
        source="file:tiny", form="file", family="custom",
        latent_channels=channels, scale_factor=8, scale_temporal=1, ndim=4,
        norm="shift_scale", norm_pack=1, vae_class="AutoencoderKL",
        config={"_class_name": "AutoencoderKL", "scaling_factor": 0.13025},
        content_hash="00112233445566ff", provenance="file:tiny.safetensors",
        locator="path:/tmp/tiny.safetensors", struct_native=False,
        identity_native=False, scaling_factor=0.13025, shift_factor=None,
    )
    fields.update(overrides)
    return vs.ResolvedVAE(**fields)


# ---------------------------------------------------------------------------
# 1. The packed algebra of the two archs this wave adds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("old,new", [(16, 4), (4, 16)])
def test_zimage_resizes_both_latent_faces_in_both_directions(old, new):
    model = _zimage(old)
    spec = ZIMAGE_WIRING.latent_io
    report = resize_latent_io(model, spec, new)

    assert set(report.replaced) == {spec.in_module, spec.out_module}
    # pack_elems 4 = pF*pH*pW (1*2*2), the only "2-1" entry the arch builds.
    assert model.all_x_embedder["2-1"].in_features == new * 4
    assert model.all_final_layer["2-1"].linear.out_features == new * 4
    assert verify_latent_io(model, spec, new) == []
    # unpatchify reads the ROOT's out_channels; a stale one reshapes to garbage.
    assert model.in_channels == new and model.out_channels == new


def test_zimage_copies_the_shared_channels_in_inner_order():
    """Both sides are C-innermost (k = s*C + c). Slicing the flat packed axis
    instead would keep the first pack positions, not the first channels."""
    model = _zimage(16)
    before_in = model.all_x_embedder["2-1"].weight.detach().clone()
    before_out = model.all_final_layer["2-1"].linear.weight.detach().clone()

    resize_latent_io(model, ZIMAGE_WIRING.latent_io, 4)

    new_in = model.all_x_embedder["2-1"].weight.detach()
    new_out = model.all_final_layer["2-1"].linear.weight.detach()
    for s in range(4):                       # pack position
        assert torch.equal(new_in[:, s * 4:s * 4 + 4],
                           before_in[:, s * 16:s * 16 + 4])
        assert torch.equal(new_out[s * 4:s * 4 + 4],
                           before_out[s * 16:s * 16 + 4])


def test_zimage_zero_initialises_the_channels_the_old_vae_did_not_have():
    model = _zimage(4)
    resize_latent_io(model, ZIMAGE_WIRING.latent_io, 16)
    weight = model.all_x_embedder["2-1"].weight.detach()
    for s in range(4):
        fresh = weight[:, s * 16 + 4:s * 16 + 16]
        assert torch.equal(fresh, torch.zeros_like(fresh))


def test_krea2_writes_the_packed_channel_count_back_into_its_config():
    """Krea2Transformer2DModel(in_channels=...) is the PACKED width, and
    krea2_config carries it: writing the raw 4 there rebuilds img_in four times
    too narrow on the next load."""
    model = _krea2(64)
    assert KREA2_WIRING.latent_io.config_channels_packed is True

    resize_latent_io(model, KREA2_WIRING.latent_io, 4)

    assert model.img_in.in_features == 16
    assert model.final_layer.linear.out_features == 16
    assert model.config["in_channels"] == 16
    assert verify_latent_io(model, KREA2_WIRING.latent_io, 4) == []


def test_krea2s_native_config_agrees_with_its_declared_packing():
    """The default the vendor ships and the wiring's algebra are one number."""
    model = _krea2()
    spec = KREA2_WIRING.latent_io
    assert model.config["in_channels"] == (
        KREA2_WIRING.latent_channels * spec.pack_elems)


def test_a_raw_config_arch_is_untouched_by_the_packed_declaration():
    assert ZIMAGE_WIRING.latent_io.config_channels_packed is False
    model = _zimage(16)
    resize_latent_io(model, ZIMAGE_WIRING.latent_io, 32)
    assert model.in_channels == 32  # raw, not 32*4


# ---------------------------------------------------------------------------
# 2. Which candidate each arch's family gate admits (design 7.4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channels", [4, 16])
def test_zimage_accepts_an_8x_image_vae_in_both_directions(channels):
    compatible, reason = vs.check_vae_compatibility(
        _facts(latent_channels=channels), "zimage")
    assert compatible, reason


def test_zimage_refuses_a_different_compression_ratio():
    compatible, reason = vs.check_vae_compatibility(
        _facts(scale_factor=16), "zimage")
    assert not compatible and "16x" in reason


def test_krea2_accepts_a_4d_image_vae_and_refuses_a_5d_one():
    ok, _ = vs.check_vae_compatibility(_facts(latent_channels=4), "krea2")
    assert ok
    refused, reason = vs.check_vae_compatibility(_facts(ndim=5), "krea2")
    assert not refused and "5-D" in reason


def test_krea2_refuses_a_vae_that_declares_no_scaling_at_all(tmp_path):
    """7.4's krea2 row: neither latents_mean/std nor a scaling_factor means the
    latents cannot be normalised, so the resolver refuses instead of guessing."""
    state = {"decoder.conv_in.weight": torch.zeros(8, 16, 3, 3),
             "encoder.down.0.block.0.norm1.weight": torch.zeros(8)}
    for i in range(3):
        state[f"encoder.down.{i}.downsample.conv.weight"] = torch.zeros(8, 8, 3, 3)
    path = tmp_path / "no_scaling.safetensors"
    save_file(state, str(path), metadata={"format": "pt"})
    with pytest.raises(vs.VaeSourceError, match="normalisation cannot be determined"):
        vs.resolve_vae_source(f"file:{path}", arch="krea2", load_weights=False)


def test_ltx2_answers_on_its_temporal_ratio_instead_of_the_missing_field():
    """P2a left scale_temporal undeclared for every 5-D arch, so EVERY ltx2
    candidate was refused for the missing field rather than on its merits."""
    assert LTX2_WIRING.vae_scale_temporal == 8
    matched = _facts(latent_channels=128, ndim=5, scale_factor=32,
                     scale_temporal=8, norm="per_channel")
    ok, reason = vs.check_vae_compatibility(matched, "ltx2")
    assert ok, reason
    refused, why = vs.check_vae_compatibility(
        {**matched, "scale_temporal": 4}, "ltx2")
    assert not refused and "temporal compression 4x" in why


def test_an_arch_that_declares_no_temporal_ratio_still_refuses():
    """anima's wave has not read its VAE's temporal ratio; declaring 1 by
    default would have been a false structural claim."""
    from core.models.components.wiring import ANIMA_WIRING
    assert ANIMA_WIRING.vae_scale_temporal is None
    refused, why = vs.check_vae_compatibility(_facts(ndim=5), "anima")
    assert not refused and "temporal compression ratio is not declared" in why


# ---------------------------------------------------------------------------
# 3. The capability decision, and its enforcement
# ---------------------------------------------------------------------------

def test_wave_two_lifts_zimage_and_krea2_for_a_full_finetune():
    from api.arch_capabilities import training_feature_unsupported_reason
    for arch in ("zimage", "krea2"):
        assert training_feature_unsupported_reason(
            arch, "vae_swap", "full_finetune") is None
        assert training_feature_unsupported_reason(arch, "vae_swap", "lora")


def test_ltx2_stays_refused_because_its_checkpoint_has_no_reader():
    from api.arch_capabilities import training_feature_unsupported_reason
    reason = training_feature_unsupported_reason(
        "ltx2", "vae_swap", "full_finetune")
    assert reason and "no loader reads" in reason


def test_the_served_refusal_is_enforced_at_preflight():
    """A caller that posts the field anyway, or hand-writes the YAML, gets the
    same answer the frontend's greyed-out control gives."""
    with pytest.raises(ValueError, match="not supported for ltx2"):
        preflight_vae_swap({"vae_swap_source": "registry:flux1"}, arch="ltx2",
                           method="full_finetune", bundle_vae_explicit_false=False)


# ---------------------------------------------------------------------------
# 4. Save -> read back, through the writer these archs actually use
# ---------------------------------------------------------------------------

def _swapped_trainer(arch, vae, **extra):
    resolved = _resolved(
        channels=int(vae.config.latent_channels),
        content_hash=vs.content_hash_for_state_dict(vae.state_dict()),
        config={**dict(vae.config), "_class_name": "AutoencoderKL"})
    trainer = SimpleNamespace(vae=vae, bundle_vae=None, vae_identity=resolved,
                              arch=SimpleNamespace(name=arch))
    for key, value in extra.items():
        setattr(trainer, key, value)
    return trainer


def test_the_zimage_save_path_bundles_the_swapped_vae_and_reads_back(tmp_path):
    from core.training.adapters.zimage_adapter import ZImageFullParameterAdapter

    vae = _tiny_vae(4)
    trainer = _swapped_trainer("zimage", vae, train_unet=False,
                               train_text_encoder=False, transformer=None,
                               text_encoder=None)
    out = tmp_path / "zimage_swapped.safetensors"
    ZImageFullParameterAdapter(trainer).save_checkpoint(3, 0, out)

    with safe_open(str(out), framework="pt") as f:
        keys, metadata = set(f.keys()), f.metadata()
    assert keys == {f"vae.{k}" for k in vae.state_dict()}
    assert not any(k.startswith("first_stage_model.") for k in keys)
    assert metadata["component.vae.channels"] == "4"
    assert metadata["component.vae.prefix"] == "vae."

    declared = vs.load_declared_latent_io(str(out), arch="zimage")
    assert declared is not None
    assert declared.latent_channels == 4
    assert declared.struct_native is False and declared.identity_native is False


def test_the_krea2_save_path_bundles_the_swapped_vae_and_reads_back(tmp_path):
    from core.training.adapters.krea2_adapter import Krea2FullParameterAdapter

    vae = _tiny_vae(4)
    transformer = _krea2(16)  # a 4-channel latent packs to 16 features
    trainer = _swapped_trainer("krea2", vae, transformer=transformer,
                               krea2_is_distilled=False)
    out = tmp_path / "krea2_swapped.safetensors"
    Krea2FullParameterAdapter(trainer).save_checkpoint(3, 0, out)

    with safe_open(str(out), framework="pt") as f:
        keys, metadata = set(f.keys()), f.metadata()
    assert {f"vae.{k}" for k in vae.state_dict()} <= keys
    assert metadata["component.vae.channels"] == "4"
    assert '"in_channels": 16' in metadata["krea2_config"]

    declared = vs.load_declared_latent_io(str(out), arch="krea2")
    assert declared is not None and declared.latent_channels == 4


def test_a_sharded_save_keeps_the_vae_section_resolvable(tmp_path):
    """These archs save through save_single_file_state, so a full fine-tune
    above the shard threshold writes an index; the declaration has to resolve
    across the shards it lands in."""
    vae = _tiny_vae(4)
    _resolved_vae, _bundled, md = swap_metadata(_swapped_trainer("zimage", vae))
    state = {"model.diffusion_model.all_x_embedder.2-1.weight": torch.zeros(96, 16)}
    state.update({f"vae.{k}": v for k, v in vae.state_dict().items()})

    written = save_single_file_state(state, {**md, "model_type": "zimage"},
                                     str(tmp_path / "sharded.safetensors"),
                                     max_shard_bytes=4096)
    assert written.endswith(".safetensors.index.json")

    declared = vs.load_declared_latent_io(written, arch="zimage")
    assert declared is not None and declared.latent_channels == 4
    assert set(declared.state_dict) == set(vae.state_dict())


def test_a_four_channel_zimage_without_a_declaration_is_not_reported_as_swapped(
        tmp_path):
    """The 4ch SDXL-VAE Z-Image variant predates this feature and is detected
    from x_embedder's shape; it must keep loading as an ordinary checkpoint."""
    path = tmp_path / "zimage_4ch.safetensors"
    save_file({"model.diffusion_model.all_x_embedder.2-1.weight": torch.zeros(96, 16)},
              str(path), metadata={"model_type": "zimage", "format": "pt"})
    assert vs.load_declared_latent_io(str(path), arch="zimage") is None


def test_a_native_bundled_zimage_declaration_is_still_not_a_swap(tmp_path):
    """bundle_vae on a native run writes component.vae.type=flux1 and no
    channel count; that is not a declaration of a replaced latent space."""
    md = build_component_metadata(te_type="qwen3", te_embedded=False,
                                  vae_type="flux1", vae_embedded=True)
    path = tmp_path / "zimage_native_bundle.safetensors"
    save_file({"model.diffusion_model.all_x_embedder.2-1.weight": torch.zeros(96, 64)},
              str(path), metadata={**md, "model_type": "zimage"})
    assert vs.load_declared_latent_io(str(path), arch="zimage") is None


def test_the_zimage_reader_splits_the_unified_vae_prefix_out():
    """A swapped Z-Image bundles under the unified prefix; leaving those keys in
    the transformer section fails the strict load."""
    from core.model_loader import ModelLoader

    raw = {
        "model.diffusion_model.all_x_embedder.2-1.weight": torch.zeros(96, 16),
        "vae.decoder.conv_in.weight": torch.zeros(8, 4, 3, 3),
        "first_stage_model.encoder.conv_out.weight": torch.zeros(8, 8, 3, 3),
        "text_encoders.qwen3.embed.weight": torch.zeros(2, 2),
    }
    transformer_sd, vae_sd, te_sd, layout = (
        ModelLoader._normalize_zimage_state_dict(raw))
    assert set(transformer_sd) == {"all_x_embedder.2-1.weight"}
    assert set(vae_sd) == {"decoder.conv_in.weight", "encoder.conv_out.weight"}
    assert set(te_sd) == {"embed.weight"}
    assert layout == "official"


# ---------------------------------------------------------------------------
# 5. Krea 2's pixel rank: its own VAE is 3-D, a replacement is 2-D
# ---------------------------------------------------------------------------

def test_krea2_encode_and_decode_follow_the_vaes_own_pixel_rank():
    from PIL import Image

    from core.models.krea2.krea2_pipeline_ops import (
        vae_decode, vae_encode, vae_pixel_ndim,
    )

    vae = _tiny_vae(4).eval()
    assert vae_pixel_ndim(vae) == 4

    image = Image.new("RGB", (32, 32), (128, 64, 32))
    latents = vae_encode(vae, image, height=32, width=32, patch_size=2,
                         device=torch.device("cpu"), dtype=torch.float32)
    # (1, grid_h*grid_w, C*p*p) with grid = 32/8/2 = 2.
    assert latents.shape == (1, 4, 16)
    assert vae_decode(vae, latents, 2, 2, 2).size == (32, 32)


# ---------------------------------------------------------------------------
# 6. The training wiring end to end, CPU only: resolve a real standalone VAE,
#    resize the real backbone, fold the run's wiring and identity.
# ---------------------------------------------------------------------------

def _standalone_vae_dir(tmp_path, latent_channels, name):
    directory = tmp_path / name
    _tiny_vae(latent_channels).save_pretrained(str(directory))
    return str(directory)


def _swap_config(directory):
    return {"vae_swap_source": f"file:{directory}",
            "training_method": "full_finetune"}


@pytest.mark.parametrize("base,target", [(16, 4), (4, 16)])
def test_a_zimage_run_swaps_its_latent_space_through_the_arch_handler(
        tmp_path, base, target):
    """Both directions the P6 bar names: 16ch -> the 4ch SDXL-VAE variant, and
    that variant back to a 16ch VAE."""
    from core.training.ops.zimage_ops import _apply_latent_space

    directory = _standalone_vae_dir(tmp_path, target, f"vae{target}")
    trainer = SimpleNamespace(
        config=_swap_config(directory),
        transformer_original=_zimage(base), vae=_tiny_vae(base),
        vae_dtype=torch.float32, is_zimage=True, trains_base_weights=True,
        log_prefix="[test]")
    trainer.transformer = trainer.transformer_original

    _apply_latent_space(trainer, None)

    assert trainer.transformer_original.all_x_embedder["2-1"].in_features == target * 4
    assert trainer.transformer_original.out_channels == target
    assert trainer.wiring.latent_channels == target
    assert trainer.vae_latent_channels == target
    assert trainer.vae_identity.identity_native is False
    assert int(trainer.vae.config.latent_channels) == target


def test_a_krea2_run_swaps_its_latent_space_through_the_arch_handler(tmp_path):
    from core.training.ops.krea2_ops import _apply_latent_space

    trainer = SimpleNamespace(
        config=_swap_config(_standalone_vae_dir(tmp_path, 4, "vae4")),
        transformer=_krea2(64), vae=_tiny_vae(16), vae_dtype=torch.float32,
        is_krea2=True, trains_base_weights=True, log_prefix="[test]")

    _apply_latent_space(trainer, None)

    assert trainer.transformer.img_in.in_features == 16
    assert trainer.transformer.config["in_channels"] == 16
    assert trainer.wiring.latent_channels == 4
    assert trainer.vae_identity.identity_native is False


def test_a_lora_run_cannot_swap_either_arch(tmp_path):
    from core.training.ops.zimage_ops import _apply_latent_space

    config = _swap_config(_standalone_vae_dir(tmp_path, 4, "vae4"))
    config["training_method"] = "lora"
    trainer = SimpleNamespace(config=config, transformer_original=_zimage(16),
                              vae=None, is_zimage=True, log_prefix="[test]")
    with pytest.raises(ValueError, match="full_finetune"):
        _apply_latent_space(trainer, None)


def test_a_swapped_zimage_checkpoint_reloads_into_the_same_latent_space(tmp_path):
    """The reload half of the smoke bar, at shape level: save through the
    shipped adapter, then rebuild a run from that file as its base."""
    from core.training.adapters.zimage_adapter import ZImageFullParameterAdapter
    from core.training.ops.zimage_ops import _apply_latent_space

    vae = _tiny_vae(4)
    model = _zimage(16)
    resize_latent_io(model, ZIMAGE_WIRING.latent_io, 4)
    trainer = _swapped_trainer("zimage", vae, train_unet=True,
                               train_text_encoder=False, transformer=model,
                               text_encoder=None)
    out = tmp_path / "run.safetensors"
    ZImageFullParameterAdapter(trainer).save_checkpoint(3, 0, out)

    declared = vs.load_declared_latent_io(str(out), arch="zimage")
    reloaded = SimpleNamespace(config={"training_method": "full_finetune"},
                               transformer_original=_zimage(4),
                               vae=declared.load_module(), is_zimage=True,
                               trains_base_weights=True, log_prefix="[test]")
    reloaded.transformer = reloaded.transformer_original

    _apply_latent_space(reloaded, declared)

    assert reloaded.wiring.latent_channels == 4
    assert reloaded.vae_identity.identity_native is False
    # The base's declaration is what the NEXT save must inherit.
    assert reloaded.base_vae_identity.content_hash == declared.content_hash
