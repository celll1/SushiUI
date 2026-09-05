"""Wave 3 of the VAE swap: anima, flux2, lens and minit2i (design phase P7).

What this wave can get silently wrong, one section each: anima's two sides fold
C in OPPOSITE orders and its input carries a non-latent padding-mask channel;
flux2 and lens count their config channels packed (lens on one side only);
minit2i's pack size is a per-checkpoint config value that divides evenly by the
wrong constant. Then the family gate, the capability decision, and the save ->
reload round trip through each arch's own writer.

CPU only, tiny modules: no GPU and no real checkpoint is touched.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.models.common import vae_source as vs
from core.models.components.latent_io import resize_latent_io, verify_latent_io
from core.models.components.wiring import (
    ANIMA_WIRING, FLUX2_WIRING, LENS_WIRING, MINIT2I_WIRING,
)
from core.training.vae_swap import apply_latent_space, preflight_vae_swap, swap_metadata


# --- fixtures ---------------------------------------------------------------

def _anima(in_channels=16, out_channels=None):
    from core.models.anima.anima_models import Anima
    return Anima(max_img_h=64, max_img_w=64, max_frames=8,
                 in_channels=in_channels,
                 out_channels=in_channels if out_channels is None else out_channels,
                 model_channels=64, num_blocks=1, num_heads=4,
                 crossattn_emb_channels=32, adaln_lora_dim=16,
                 use_llm_adapter=False)


def _flux2(in_channels=128):
    from diffusers import Flux2Transformer2DModel
    return Flux2Transformer2DModel(
        patch_size=1, in_channels=in_channels, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
        timestep_guidance_channels=16, axes_dims_rope=(4, 4, 4, 4))


def _lens(in_channels=128, out_channels=32):
    from core.models.lens.vendor.transformer import LensTransformer2DModel
    return LensTransformer2DModel(
        patch_size=2, in_channels=in_channels, out_channels=out_channels,
        num_layers=1, attention_head_dim=8, num_attention_heads=2, inner_dim=16,
        enc_hidden_dim=16, axes_dims_rope=(2, 4, 2), selected_layer_index=(0,))


def _minit2i(in_channels=16, patch_size=2, vae_type="flux1"):
    from core.models.minit2i.vendor.transformer import MiniT2IMMJiTModel
    return MiniT2IMMJiTModel(
        image_size=64, patch_size=patch_size, in_channels=in_channels,
        txt_input_size=8, hidden_size=16, txt_hidden_size=16, cond_vec_size=16,
        depth_double=1, txt_preamble_depth=1, num_heads=2, head_dim=8,
        pca_channels=8, prompt_length=4, vae_type=vae_type,
        noise_scale=2.0 if vae_type == "none" else 1.0)


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


def _standalone_vae_dir(tmp_path, latent_channels, name):
    directory = tmp_path / name
    _tiny_vae(latent_channels).save_pretrained(str(directory))
    return str(directory)


def _swap_config(directory):
    return {"vae_swap_source": f"file:{directory}",
            "training_method": "full_finetune"}


# ---------------------------------------------------------------------------
# 1. anima: opposite orders on the two sides, plus the padding-mask channel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("old,new", [(16, 32), (32, 16)])
def test_anima_resizes_both_latent_faces_in_both_directions(old, new):
    model = _anima(old)
    spec = ANIMA_WIRING.latent_io
    report = resize_latent_io(model, spec, new)

    assert set(report.replaced) == {spec.in_module, spec.out_module}
    # The input carries C + 1: concat_padding_mask (anima_models.build_patch_embed).
    assert model.x_embedder.proj[1].in_features == (new + 1) * 4
    assert model.final_layer.linear.out_features == new * 4
    assert verify_latent_io(model, spec, new) == []
    assert model.in_channels == new and model.out_channels == new


def test_anima_folds_its_input_outer_and_its_output_inner():
    """The one architecture whose two sides disagree. Input packs
    ``(c r m n)`` -- C OUTERMOST -- and unpatchify reads ``(p1 p2 t C)`` -- C
    INNERMOST. Copying either side with the other's expression keeps the wrong
    elements, which is what the two loops below would catch."""
    model = _anima(16)
    before_in = model.x_embedder.proj[1].weight.detach().clone()
    before_out = model.final_layer.linear.weight.detach().clone()

    resize_latent_io(model, ANIMA_WIRING.latent_io, 4)

    new_in = model.x_embedder.proj[1].weight.detach()
    new_out = model.final_layer.linear.weight.detach()
    # Input, C outer: channel c occupies columns [c*4, c*4+4).
    for c in range(4):
        assert torch.equal(new_in[:, c * 4:(c + 1) * 4],
                           before_in[:, c * 4:(c + 1) * 4])
    # Output, C inner: pack position s occupies rows [s*C, s*C+C).
    for s in range(4):
        assert torch.equal(new_out[s * 4:s * 4 + 4],
                           before_out[s * 16:s * 16 + 4])


def test_anima_would_fail_if_its_two_sides_shared_one_order():
    """The regression the declaration exists for: with both sides declared
    ``outer`` the OUTPUT copy takes the first 4 pack-position rows instead of
    the first 4 channels of each position, which is a different tensor."""
    shared_order = ANIMA_WIRING.latent_io.replace(out_channel_order="outer")

    correct, wrong = _anima(16), _anima(16)
    wrong.final_layer.linear.load_state_dict(
        correct.final_layer.linear.state_dict())

    resize_latent_io(correct, ANIMA_WIRING.latent_io, 4)
    resize_latent_io(wrong, shared_order, 4)

    assert not torch.equal(correct.final_layer.linear.weight.detach(),
                           wrong.final_layer.linear.weight.detach())


def test_anima_moves_the_padding_mask_row_to_the_new_channel_count():
    """The +1 input channel is NOT a latent channel: it has to land at the new
    C, not stay at the old one, or the mask feeds weights trained for latents."""
    model = _anima(16)
    before = model.x_embedder.proj[1].weight.detach().clone()
    mask_columns = before[:, 16 * 4:17 * 4].clone()   # channel index 16 = the mask

    resize_latent_io(model, ANIMA_WIRING.latent_io, 4)

    after = model.x_embedder.proj[1].weight.detach()
    assert torch.equal(after[:, 4 * 4:5 * 4], mask_columns)


def test_anima_leaves_the_learnable_position_embedder_alone():
    """pos_embedder is sized from max_img_h // patch_spatial, and D5 pins the
    downsample ratio -- so a channel resize must not touch it."""
    model = _anima(16)
    before = {k: v.clone() for k, v in model.pos_embedder.state_dict().items()}

    resize_latent_io(model, ANIMA_WIRING.latent_io, 4)

    after = model.pos_embedder.state_dict()
    assert set(after) == set(before)
    for key, value in before.items():
        assert torch.equal(after[key], value), key


def test_anima_declares_the_temporal_ratio_of_the_qwen_image_vae():
    """4 = one halving per True in temperal_downsample=[False, True, True]
    (anima_loader.QWEN_IMAGE_VAE_CONFIG). Before it was declared, the gate
    refused every anima candidate for want of anything to compare against."""
    from core.models.anima.anima_loader import QWEN_IMAGE_VAE_CONFIG

    assert ANIMA_WIRING.vae_scale_temporal == 2 ** sum(
        1 for down in QWEN_IMAGE_VAE_CONFIG["temperal_downsample"] if down)
    matched = _facts(ndim=5, scale_temporal=4, norm="per_channel",
                     vae_class="AutoencoderKLQwenImage")
    ok, reason = vs.check_vae_compatibility(matched, "anima")
    assert ok, reason
    refused, why = vs.check_vae_compatibility(
        {**matched, "scale_temporal": 2}, "anima")
    assert not refused and "temporal compression 2x" in why


def test_anima_refuses_a_two_dimensional_image_vae():
    """anima's latent is 5-D ([B, C, T, H, W] with T=1); a 2-D VAE cannot
    produce it."""
    refused, why = vs.check_vae_compatibility(_facts(ndim=4), "anima")
    assert not refused and "4-D latents cannot drive anima" in why


# ---------------------------------------------------------------------------
# 2. flux2 / lens: the packed config numbers, and lens's asymmetry
# ---------------------------------------------------------------------------

def test_flux2_writes_the_packed_channel_count_into_both_config_numbers():
    model = _flux2(128)
    spec = FLUX2_WIRING.latent_io
    assert spec.config_in_channels_packed and spec.config_out_channels_packed

    resize_latent_io(model, spec, 16)

    assert model.x_embedder.in_features == 64
    assert model.proj_out.out_features == 64
    assert model.config.in_channels == 64 and model.config.out_channels == 64
    assert verify_latent_io(model, spec, 16) == []


def test_flux2s_native_config_agrees_with_its_declared_packing():
    model = _flux2()
    assert model.config.in_channels == (
        FLUX2_WIRING.latent_channels * FLUX2_WIRING.latent_io.pack_elems)


def test_lens_counts_one_config_number_packed_and_the_other_raw():
    """LensTransformer2DModel(in_channels=128, out_channels=32, patch_size=2):
    img_in faces the packed width, proj_out multiplies out_channels by 4 itself.
    One flag for both sides rebuilds one of them four times wrong."""
    spec = LENS_WIRING.latent_io
    assert spec.config_in_channels_packed and not spec.config_out_channels_packed

    model = _lens(128, 32)
    resize_latent_io(model, spec, 16)

    assert model.img_in.in_features == 64
    assert model.proj_out.out_features == 64
    assert model.config.in_channels == 64      # packed
    assert model.config.out_channels == 16     # raw
    assert verify_latent_io(model, spec, 16) == []

    # The proof it matters: rebuilding from the written config reproduces the
    # resized widths.
    rebuilt = _lens(model.config.in_channels, model.config.out_channels)
    assert rebuilt.img_in.in_features == model.img_in.in_features
    assert rebuilt.proj_out.out_features == model.proj_out.out_features


def test_lens_and_flux2_admit_a_batchnorm_vae_and_a_scalar_one():
    """Both normalise through their VAE's own BatchNorm on the 2x2-packed
    domain; since P5 the shared layer packs into whichever domain the VAE
    declares, so a shift_scale VAE is admissible too."""
    for arch in ("flux2", "lens"):
        ok, reason = vs.check_vae_compatibility(
            _facts(latent_channels=32, norm="batchnorm"), arch)
        assert ok, (arch, reason)
        ok, reason = vs.check_vae_compatibility(
            _facts(latent_channels=16, norm="shift_scale"), arch)
        assert ok, (arch, reason)
        refused, why = vs.check_vae_compatibility(
            _facts(scale_factor=16), arch)
        assert not refused and "16x" in why


# ---------------------------------------------------------------------------
# 3. minit2i: the pack size is a per-checkpoint config value
# ---------------------------------------------------------------------------

def test_minit2i_resolves_its_pack_size_from_the_loaded_config():
    from core.training.arch.minit2i import MiniT2IArchHandler

    handler = MiniT2IArchHandler()
    latent = SimpleNamespace(transformer=_minit2i(16, patch_size=2))
    pixel = SimpleNamespace(transformer=_minit2i(3, patch_size=16, vae_type="none"))

    assert handler.resolve_wiring(latent).latent_io.pack_elems == 4
    assert handler.resolve_wiring(latent).latent_channels == 16
    assert handler.resolve_wiring(latent).vae_scale_factor == 8
    assert handler.resolve_wiring(pixel).latent_io.pack_elems == 256
    # The wiring constant stays the pixel one; only the run's copy moves.
    assert MINIT2I_WIRING.latent_channels == 0
    assert MINIT2I_WIRING.latent_io.pack_elems == 4


def test_the_wiring_constant_would_mis_slice_a_pixel_checkpoint_in_silence():
    """final_layer.linear is 768 = 256*3 on a pixel checkpoint, which divides
    evenly by the constant's 4 -- so the wrong P reads 192 channels instead of
    3 and copies the wrong elements without an error."""
    pixel = _minit2i(3, patch_size=16, vae_type="none")
    from core.models.components.latent_io import _out_channels, _resolve

    _p, _a, out_module = _resolve(pixel, MINIT2I_WIRING.latent_io.out_module)
    assert out_module.out_features == 768
    assert _out_channels(out_module, MINIT2I_WIRING.latent_io) == 192   # wrong P

    from core.training.arch.minit2i import MiniT2IArchHandler
    resolved = MiniT2IArchHandler().resolve_wiring(
        SimpleNamespace(transformer=pixel)).latent_io
    assert _out_channels(out_module, resolved) == 3                     # right P


def test_minit2i_resizes_a_latent_checkpoint_and_syncs_every_channel_copy():
    from core.training.arch.minit2i import MiniT2IArchHandler

    model = _minit2i(16, patch_size=2)
    spec = MiniT2IArchHandler().resolve_wiring(
        SimpleNamespace(transformer=model)).latent_io

    resize_latent_io(model, spec, 4)

    net = model.model.net
    assert net.img_embedder.proj1.in_channels == 4
    assert net.final_layer.linear.out_features == 16   # 4 channels * 2*2
    # unpatchify reads cfg.in_channels; a stale one reshapes garbage.
    assert net.cfg.in_channels == 4
    assert model.config.in_channels == 4
    assert net.final_layer.out_channels == 4
    assert verify_latent_io(model, spec, 4) == []


def test_a_pixel_minit2i_checkpoint_is_refused_a_swap():
    """Design 5.1: moving pixel -> latent changes patch_size as well as the
    channel count, which is not a channel resize."""
    from core.training.arch.minit2i import MiniT2IArchHandler

    handler = MiniT2IArchHandler()
    pixel = SimpleNamespace(transformer=_minit2i(3, patch_size=16, vae_type="none"))
    refused, why = handler.check_vae_compatibility(_facts(), trainer=pixel)
    assert not refused and "pixel-space" in why

    latent = SimpleNamespace(transformer=_minit2i(16, patch_size=2))
    ok, reason = handler.check_vae_compatibility(_facts(latent_channels=4),
                                                 trainer=latent)
    assert ok, reason
    refused, why = handler.check_vae_compatibility(_facts(ndim=5), trainer=latent)
    assert not refused and "5-D" in why


def test_minit2i_peeks_its_geometry_out_of_a_saved_checkpoint(tmp_path):
    """Preflight runs before the model loads, so the pixel/latent question is
    answered from the base file's own metadata."""
    from core.models.minit2i.minit2i_loader import peek_io_config
    from core.models.minit2i.vendor.single_file import save_single_file

    for vae_type, channels, patch in (("none", 3, 16), ("flux1", 16, 2)):
        model = _minit2i(channels, patch_size=patch, vae_type=vae_type)
        out = tmp_path / f"minit2i_{vae_type}.safetensors"
        save_single_file(str(out), model, variant="b16")
        assert peek_io_config(str(out)) == {
            "in_channels": channels, "patch_size": patch, "vae_type": vae_type}

    assert peek_io_config("scratch:minit2i:b16:sdxl")["vae_type"] == "sdxl"
    assert peek_io_config(str(tmp_path / "absent.safetensors")) == {}


# ---------------------------------------------------------------------------
# 4. The capability decision, and its enforcement
# ---------------------------------------------------------------------------

def test_wave_three_lifts_all_four_for_a_full_finetune():
    from api.arch_capabilities import training_feature_unsupported_reason
    for arch in ("anima", "flux2", "lens", "minit2i"):
        assert training_feature_unsupported_reason(
            arch, "vae_swap", "full_finetune") is None
        assert training_feature_unsupported_reason(arch, "vae_swap", "lora")


def test_the_waves_that_have_not_landed_are_still_refused():
    from api.arch_capabilities import training_feature_unsupported_reason
    for arch in ("ltx2", "sensenova", "ideogram4", "minimax_h3", "acestep"):
        assert training_feature_unsupported_reason(
            arch, "vae_swap", "full_finetune"), arch


def test_a_lora_run_cannot_swap_any_of_the_four(tmp_path):
    directory = _standalone_vae_dir(tmp_path, 4, "vae4")
    for arch in ("anima", "flux2", "lens", "minit2i"):
        with pytest.raises(ValueError, match="full_finetune"):
            preflight_vae_swap({"vae_swap_source": f"file:{directory}"},
                               arch=arch, method="lora",
                               bundle_vae_explicit_false=False)


def test_preflight_refuses_a_pixel_minit2i_base(tmp_path):
    """The served capability says minit2i can swap; the checkpoint says this
    one cannot, and preflight is where that is answered."""
    from core.models.minit2i.vendor.single_file import save_single_file

    base = tmp_path / "pixel.safetensors"
    save_single_file(str(base), _minit2i(3, patch_size=16, vae_type="none"),
                     variant="b16")
    directory = _standalone_vae_dir(tmp_path, 16, "vae16")
    with pytest.raises(ValueError, match="pixel-space"):
        preflight_vae_swap({"vae_swap_source": f"file:{directory}"},
                           arch="minit2i", method="full_finetune",
                           bundle_vae_explicit_false=False,
                           base_model_path=str(base))


def test_preflight_admits_a_latent_minit2i_base(tmp_path):
    from core.models.minit2i.vendor.single_file import save_single_file

    base = tmp_path / "latent.safetensors"
    save_single_file(str(base), _minit2i(16, patch_size=2, vae_type="flux1"),
                     variant="b16")
    directory = _standalone_vae_dir(tmp_path, 4, "vae4")
    assert preflight_vae_swap(
        {"vae_swap_source": f"file:{directory}"}, arch="minit2i",
        method="full_finetune", bundle_vae_explicit_false=False,
        base_model_path=str(base)) == f"file:{directory}"


# ---------------------------------------------------------------------------
# 5. The training fold: resolve a real standalone VAE, resize the real backbone
# ---------------------------------------------------------------------------

def _trainer(arch, transformer, vae, config, **extra):
    trainer = SimpleNamespace(
        config=config, transformer=transformer, transformer_original=transformer,
        vae=vae, vae_dtype=torch.float32, trains_base_weights=True,
        log_prefix="[test]", bundle_vae=None)
    setattr(trainer, f"is_{arch}", True)
    for key, value in extra.items():
        setattr(trainer, key, value)
    return trainer


def test_a_flux2_run_swaps_its_latent_space_through_the_arch_handler(tmp_path):
    directory = _standalone_vae_dir(tmp_path, 16, "vae16")
    trainer = _trainer("flux2", _flux2(128), _tiny_vae(32), _swap_config(directory))

    apply_latent_space(trainer, None)

    assert trainer.transformer.x_embedder.in_features == 64
    assert trainer.transformer.config.in_channels == 64
    assert trainer.wiring.latent_channels == 16
    assert trainer.vae_identity.identity_native is False


def test_a_lens_run_swaps_its_latent_space_through_the_arch_handler(tmp_path):
    directory = _standalone_vae_dir(tmp_path, 16, "vae16")
    trainer = _trainer("lens", _lens(), _tiny_vae(32), _swap_config(directory))

    apply_latent_space(trainer, None)

    assert trainer.transformer.img_in.in_features == 64
    assert trainer.transformer.config.out_channels == 16
    assert trainer.wiring.latent_channels == 16
    # The VAE's normalisation travels with it: a shift_scale VAE on an arch
    # whose own is a packed BatchNorm.
    assert trainer.wiring.vae_norm == "shift_scale"
    assert trainer.wiring.vae_norm_pack == 1


def test_a_minit2i_run_swaps_its_latent_space_through_the_arch_handler(tmp_path):
    directory = _standalone_vae_dir(tmp_path, 4, "vae4")
    trainer = _trainer("minit2i", _minit2i(16, patch_size=2),
                       _tiny_vae(16), _swap_config(directory))

    apply_latent_space(trainer, None)

    net = trainer.transformer.model.net
    assert net.img_embedder.proj1.in_channels == 4
    assert net.cfg.in_channels == 4
    # vae_type is a config field the loader reads to decide whether the model is
    # latent at all; a swap that moved only weights would reload as pixel-space.
    assert net.cfg.vae_type == "custom"
    assert trainer.wiring.latent_channels == 4


def test_an_anima_run_keeps_its_five_dimensional_latent(tmp_path):
    """A 2-D image VAE is refused for anima at the same gate the frontend
    reads, so the fold never reaches the resize."""
    directory = _standalone_vae_dir(tmp_path, 16, "vae16")
    trainer = _trainer("anima", _anima(16), _tiny_vae(16), _swap_config(directory))

    with pytest.raises(ValueError, match="cannot drive anima"):
        apply_latent_space(trainer, None)


def test_an_anima_run_swaps_to_a_five_dimensional_vae(tmp_path):
    """The reachable anima swap: same rank and ratios, different weights --
    a fine-tuned Qwen-Image VAE (struct_native, not identity_native)."""
    from core.training.arch import get_arch_handler

    model = _anima(16)
    trainer = _trainer("anima", model, _tiny_vae(16),
                       {"training_method": "full_finetune"})
    resolved = _resolved(channels=32, ndim=5, scale_temporal=4,
                         norm="per_channel", vae_class="AutoencoderKLQwenImage",
                         struct_native=False, identity_native=False)

    get_arch_handler(trainer).apply_vae_swap(trainer, resolved,
                                             module=_tiny_vae(16))

    assert model.x_embedder.proj[1].in_features == (32 + 1) * 4
    assert model.final_layer.linear.out_features == 32 * 4
    assert trainer.wiring.latent_channels == 32
    assert trainer.wiring.vae_norm == "per_channel"


# ---------------------------------------------------------------------------
# 6. Save -> read back, through the writer each arch actually uses
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


def test_the_anima_save_path_bundles_the_swapped_vae_and_reads_back(tmp_path):
    from core.training.adapters.anima_adapter import AnimaFullParameterAdapter

    vae = _tiny_vae(4)
    trainer = _swapped_trainer("anima", vae, transformer=_anima(4))
    out = tmp_path / "anima_swapped.safetensors"
    AnimaFullParameterAdapter(trainer).save_checkpoint(3, 0, out)

    with safe_open(str(out), framework="pt") as f:
        keys, metadata = set(f.keys()), f.metadata()
    assert {f"vae.{k}" for k in vae.state_dict()} <= keys
    assert not any(k.startswith("first_stage_model.") for k in keys)
    assert metadata["component.vae.channels"] == "4"
    assert metadata["component.vae.prefix"] == "vae."
    assert '"in_channels": 4' in metadata["transformer_config"]

    declared = vs.load_declared_latent_io(str(out), arch="anima")
    assert declared is not None and declared.latent_channels == 4
    assert declared.identity_native is False


def test_the_flux2_save_path_bundles_the_swapped_vae_and_reads_back(tmp_path):
    from core.training.adapters.flux2_adapter import FLUX2FullParameterAdapter

    vae = _tiny_vae(4)
    trainer = _swapped_trainer("flux2", vae, transformer=_flux2(16),
                               train_unet=True, train_text_encoder=False,
                               text_encoder=None)
    out = tmp_path / "flux2_swapped.safetensors"
    FLUX2FullParameterAdapter(trainer).save_checkpoint(3, 0, out)

    with safe_open(str(out), framework="pt") as f:
        keys, metadata = set(f.keys()), f.metadata()
    assert {f"vae.{k}" for k in vae.state_dict()} <= keys
    assert not any(k.startswith("first_stage_model.") for k in keys)
    assert metadata["component.vae.channels"] == "4"

    declared = vs.load_declared_latent_io(str(out), arch="flux2")
    assert declared is not None and declared.latent_channels == 4


def test_the_flux2_reader_splits_the_unified_vae_prefix_out():
    """A swapped FLUX.2 bundles under the unified prefix; leaving those keys in
    the transformer section casts VAE weights into transformer parameters."""
    from core.model_loader import ModelLoader

    raw = {
        "model.diffusion_model.x_embedder.weight": torch.zeros(32, 16),
        "vae.decoder.conv_in.weight": torch.zeros(8, 4, 3, 3),
        "first_stage_model.encoder.conv_out.weight": torch.zeros(8, 8, 3, 3),
        "text_encoders.qwen3.embed.weight": torch.zeros(2, 2),
    }
    transformer_sd, vae_sd, te_sd = ModelLoader._split_flux2_sushiui_state_dict(raw)
    assert set(transformer_sd) == {"x_embedder.weight"}
    assert set(vae_sd) == {"decoder.conv_in.weight", "encoder.conv_out.weight"}
    assert set(te_sd) == {"embed.weight"}


def test_the_lens_save_path_bundles_the_swapped_vae_and_reads_back(tmp_path):
    from core.training.adapters.lens_adapter import LensFullParameterAdapter

    vae = _tiny_vae(4)
    trainer = _swapped_trainer("lens", vae, transformer=_lens(16, 4),
                               lens_base_dir=str(tmp_path / "base"))
    out = tmp_path / "lens_swapped.safetensors"
    LensFullParameterAdapter(trainer).save_checkpoint(3, 0, out)

    with safe_open(str(out), framework="pt") as f:
        keys, metadata = set(f.keys()), f.metadata()
    assert {f"vae.{k}" for k in vae.state_dict()} <= keys
    assert not any(k.startswith("first_stage_model.") for k in keys)
    assert metadata["component.vae.channels"] == "4"
    assert '"in_channels": 16' in metadata["transformer_config"]

    declared = vs.load_declared_latent_io(str(out), arch="lens")
    assert declared is not None and declared.latent_channels == 4


def test_the_minit2i_save_path_bundles_the_swapped_vae_and_reads_back(tmp_path):
    from core.training.adapters.minit2i_adapter import MiniT2IFullParameterAdapter

    vae = _tiny_vae(4)
    model = _minit2i(4, patch_size=2, vae_type="sdxl")
    trainer = _swapped_trainer("minit2i", vae, transformer=model,
                               minit2i_variant="b16", text_encoder=None,
                               train_text_encoder=False, repa_enable=False)
    out = tmp_path / "minit2i_swapped.safetensors"
    MiniT2IFullParameterAdapter(trainer).save_checkpoint(3, 0, out)

    with safe_open(str(out), framework="pt") as f:
        keys, metadata = set(f.keys()), f.metadata()
    assert {f"vae.{k}" for k in vae.state_dict()} <= keys
    assert metadata["component.vae.channels"] == "4"
    assert '"in_channels": 4' in metadata["mmjit_config"]

    declared = vs.load_declared_latent_io(str(out), arch="minit2i")
    assert declared is not None and declared.latent_channels == 4


def test_a_swapped_lens_checkpoint_reloads_into_the_same_latent_space(tmp_path):
    """The reload half, at shape level: the base directory builds a 32-channel
    transformer, and the declaration resizes it before the trained weights land
    on it. Without the resize the load raises on img_in's width."""
    from core.models.components.latent_io import resize_latent_io

    trained = _lens(128, 32)
    resize_latent_io(trained, LENS_WIRING.latent_io, 4)
    dit_sd = trained.state_dict()

    base = _lens(128, 32)
    with pytest.raises(RuntimeError, match="size mismatch"):
        base.load_state_dict(dit_sd, strict=False)

    resize_latent_io(base, LENS_WIRING.latent_io, 4)
    info = base.load_state_dict(dit_sd, strict=False)
    assert not getattr(info, "unexpected_keys", [])
    assert torch.equal(base.img_in.weight, trained.img_in.weight)
    assert torch.equal(base.proj_out.weight, trained.proj_out.weight)


def test_a_swapped_anima_checkpoint_rebuilds_at_its_declared_width(tmp_path):
    """anima builds from a module-level config; the declared channel count is
    what its DiT must be constructed at, mask channel included."""
    from core.models.anima.anima_models import ANIMA_DIT_CONFIG

    declared_channels = 32
    config = dict(ANIMA_DIT_CONFIG)
    expected = (declared_channels + 1) * config["patch_spatial"] ** 2 \
        * config["patch_temporal"]
    model = _anima(declared_channels)
    assert model.x_embedder.proj[1].in_features == expected
    assert model.final_layer.linear.out_features == declared_channels * 4


def test_a_native_checkpoint_of_each_arch_declares_no_swap(tmp_path):
    """A bundle_vae save on a native run is not a declaration of a replaced
    latent space, and must keep loading as an ordinary checkpoint."""
    from core.models.common.single_file_format import build_component_metadata
    from safetensors.torch import save_file

    for arch, vae_type in (("anima", "qwen_image"), ("flux2", "flux2"),
                           ("lens", "flux2"), ("minit2i", "sdxl")):
        md = build_component_metadata(vae_type=vae_type, vae_embedded=True)
        path = tmp_path / f"{arch}_native.safetensors"
        save_file({"net.x.weight": torch.zeros(2, 2)}, str(path),
                  metadata={**md, "model_type": arch, "format": "pt"})
        assert vs.load_declared_latent_io(str(path), arch=arch) is None, arch
