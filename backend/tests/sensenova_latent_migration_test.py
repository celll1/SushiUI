"""SenseNova pixel -> latent (VAE_SWAP_MIGRATION_DESIGN.md §10).

What these cover, and what they deliberately do not:

* the geometry (one token = P * vae_scale_factor pixels, exactly two tensors
  change shape, identically at 8x and 16x) and the fact that P is a per-run
  parameter -- `sensenova_gen_patch`, whose default 0 INHERITS -- not the constant
  §10.2 wrote it as;
* §10.3's initialisation and its consequences -- including the one the design
  is explicit is NOT avoided: with a zero head ``v = -z/(1-t)`` still grows as
  ``t -> 1`` and is bounded only by ``(1-t).clamp_min(t_eps)``;
* §10.6's endpoint velocity and step-0/step-1 gradient measurements;
* the refusals: the shut capability gate, the fm_modules requirement, the
  full-fine-tune-only rebuild, and a checkpoint whose config and component
  blocks disagree.

They say NOTHING about whether a swapped model trains or generates well. §10.6-5
forbids a quality claim, and no run has been made.
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sensenova_training_core_test import _Cache, _Layer  # noqa: E402

from api.param_defaults import TRAINING_DEFAULTS  # noqa: E402
from core.models.sensenova.latent_space import (  # noqa: E402
    INHERIT_GEN_PATCH,
    INHERIT_NOISE_SCALE_GAIN,
    MIN_GEN_LATENT_PATCH,
    NATIVE_GEN_LATENT_PATCH,
    PIXEL_RMS,
    apply_latent_geometry,
    apply_noise_scale_gain,
    noise_scale_gain_for_rms,
    recalibrated_noise_scale_gain,
    gen_geometry,
    latent_config_dict,
    resolution_band_mp,
    resolve_gen_patch,
    token_pixel_width,
    validate_gen_patch,
)
from core.models.sensenova.sensenova_pipeline_ops import (  # noqa: E402
    align_to_grid, normalize_resolution,
)
from core.models.sensenova.vendor.configuration_neo_vit import (  # noqa: E402
    NEOVisionConfig,
)
from core.models.sensenova.vendor.modeling_fm_modules import (  # noqa: E402
    ConvDecoder, TimestepEmbedder,
)
from core.models.sensenova.vendor.modeling_neo_chat import NEOChatModel  # noqa: E402
from core.models.sensenova.vendor.modeling_neo_vit import NEOVisionModel  # noqa: E402
from core.training.ops.sensenova_ops import (  # noqa: E402
    SenseNovaTrainingPrefix, train_step,
)

HIDDEN = 64          # llm hidden; ConvDecoder needs it divisible by 4 twice
VIT_HIDDEN = 32
CHANNELS = 16
T_EPS = 0.02
#: The patch a pixel base is migrated at when the run inherits. The SERVED
#: default is the 0 sentinel; a change to either fails here, not in a run.
PATCH = NATIVE_GEN_LATENT_PATCH


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------

def _vision(channels: int, patch: int) -> NEOVisionModel:
    """The REAL gen ViT at a small width; its patch embed is one of the two
    tensors a swap rebuilds, so a double would test the double."""
    return NEOVisionModel(NEOVisionConfig(
        num_channels=channels, patch_size=patch, hidden_size=VIT_HIDDEN,
        llm_hidden_size=HIDDEN, downsample_ratio=0.5))


class _PixelTree(nn.Module):
    """The shipped pixel geometry: patch 32, 3 channels, ps3(8) head."""

    patch_size = 16
    downsample_ratio = 0.5

    def __init__(self):
        super().__init__()
        self.use_pixel_head = True
        self.use_deep_fm_head = False
        self.config = SimpleNamespace(t_eps=T_EPS)
        self.gen_in_channels = 3
        self.gen_patch_size = 32
        self.gen_vit_patch_size = 16
        self.gen_vae_scale_factor = 1
        # The real checkpoint's own noise-scale block (M:/model/sensenova/
        # config.json), so compute_noise_scale returns production numbers here.
        self.noise_scale = 1.0
        self.noise_scale_mode = "resolution"
        self.noise_scale_base_image_seq_len = 64
        self.noise_scale_max_value = 16.0
        self.fm_modules = nn.ModuleDict({
            "vision_model_mot_gen": _vision(3, 16),
            "timestep_embedder": TimestepEmbedder(HIDDEN),
            "noise_scale_embedder": TimestepEmbedder(HIDDEN),
            "fm_head": ConvDecoder(input_dim=HIDDEN, hidden_dim=HIDDEN),
        })


def _params(module) -> dict:
    return {name: p.detach().clone()
            for name, p in module.named_parameters()}


# ---------------------------------------------------------------------------
# §10.2 -- the geometry
# ---------------------------------------------------------------------------

def test_the_pixel_head_is_what_it_always_was():
    """The generalisation's defaults reproduce the shipped model exactly:
    3 * 8^2 = 192 conv2 outputs and a final 8x shuffle."""
    head = ConvDecoder(input_dim=4096, hidden_dim=1024)
    assert tuple(head.conv2.weight.shape) == (192, 256, 3, 3)
    assert head.ps3.upscale_factor == 8

    geometry = gen_geometry(_PixelTree())
    assert (geometry.channels, geometry.patch, geometry.vit_patch) == (3, 32, 16)
    assert geometry.token_pixel_width == 32 and not geometry.is_latent


@pytest.mark.parametrize("scale", [8, 16])
def test_exactly_two_tensors_change_and_they_are_the_same_at_8x_and_16x(scale):
    tree = _PixelTree()
    before = _params(tree)
    report = apply_latent_geometry(tree, channels=CHANNELS,
                                   vae_scale_factor=scale, patch=PATCH)
    after = _params(tree)

    changed = {name for name in before
               if name not in after or before[name].shape != after[name].shape
               or not torch.equal(before[name], after[name])}
    assert changed == {
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.bias",
        "fm_modules.fm_head.conv2.weight",
        "fm_modules.fm_head.conv2.bias",
    }
    # Named because the design names them: conv1, dense_embedding, both
    # embedders and the ps layers are untouched at EVERY compression ratio.
    for name in ("fm_modules.fm_head.conv1.weight",
                 "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight",
                 "fm_modules.timestep_embedder.mlp.0.weight",
                 "fm_modules.noise_scale_embedder.mlp.0.weight"):
        assert torch.equal(before[name], after[name])

    embed = tree.fm_modules.vision_model_mot_gen.embeddings.patch_embedding
    assert tuple(embed.weight.shape) == (VIT_HIDDEN, CHANNELS, 2, 2)
    assert tuple(tree.fm_modules.fm_head.conv2.weight.shape) == (CHANNELS, HIDDEN // 4, 3, 3)
    assert tree.fm_modules.fm_head.ps3.upscale_factor == 1
    assert report.copied_elements == 0 and report.new_channels == CHANNELS
    assert token_pixel_width(tree) == PATCH * scale


def test_head_is_zero_and_patch_embed_is_a_bounded_small_normal():
    tree = _PixelTree()
    apply_latent_geometry(tree, channels=CHANNELS, vae_scale_factor=8, patch=PATCH)
    head = tree.fm_modules.fm_head
    assert torch.count_nonzero(head.conv2.weight) == 0
    assert torch.count_nonzero(head.conv2.bias) == 0

    weight = tree.fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight
    std = 1.0 / (CHANNELS * 2 * 2) ** 0.5
    assert weight.abs().max().item() <= 3 * std + 1e-6
    assert 0.3 * std <= weight.std().item() <= 1.7 * std
    assert torch.count_nonzero(weight) > 0  # NOT the rejected zero init (§10.3)


@pytest.mark.parametrize("scale", [8, 16])
def test_a_128_cell_latent_grid_is_1024_tokens_and_the_head_returns_that_grid(scale):
    """§10.6-2, at ``128 * vae_scale_factor`` px (1024px at 8x, 2048px at 16x)."""
    tree = _PixelTree()
    apply_latent_geometry(tree, channels=CHANNELS, vae_scale_factor=scale,
                          patch=PATCH)
    geometry = gen_geometry(tree)
    latent = torch.randn(1, CHANNELS, 128, 128)
    pixels = 128 * scale
    assert pixels % token_pixel_width(tree) == 0

    tokens = NEOChatModel.patchify(tree, latent, geometry.patch)
    assert tokens.shape == (1, 1024, geometry.patch ** 2 * CHANNELS)

    vit_patches = NEOChatModel.patchify(tree, latent, geometry.vit_patch,
                                        channel_first=True)
    merged = tree.fm_modules.vision_model_mot_gen(
        pixel_values=vit_patches.view(-1, vit_patches.shape[-1]),
        grid_hw=torch.tensor([[64, 64]]), return_dict=True).last_hidden_state
    assert merged.shape == (1024, HIDDEN)

    decoded = tree.fm_modules.fm_head(torch.randn(1, HIDDEN, 32, 32))
    assert decoded.shape == (1, CHANNELS, 128, 128)


def test_patchify_round_trips_at_any_channel_count():
    tree = _PixelTree()
    latent = torch.randn(1, CHANNELS, 16, 16)
    tokens = NEOChatModel.patchify(tree, latent, PATCH)
    assert tokens.shape == (1, 16, PATCH ** 2 * CHANNELS)
    back = NEOChatModel.unpatchify(tree, tokens, PATCH, 16, 16)
    assert torch.equal(back, latent)


def test_the_token_grid_and_the_resolution_band_move_together():
    assert normalize_resolution(1000, 1000, 64) == (1024, 1024)
    assert align_to_grid(4090, 64) == 4096
    low8, high8 = resolution_band_mp(32)
    low16, high16 = resolution_band_mp(64)
    assert (low8, high8) == (3.0, 5.0)
    assert (low16, high16) == (12.0, 20.0)
    # §10.6-4's checkable half: 4096^2 on a 16x VAE is INSIDE the band, so the
    # `sensenova_resolution` warning does not fire on it.
    assert low16 <= (4096 * 4096) / 1e6 <= high16


# ---------------------------------------------------------------------------
# §10.6-3 -- the endpoints, measured
# ---------------------------------------------------------------------------

class _GenVision(nn.Module):
    """The two trainable tensors of the gen ViT, at one token."""

    def __init__(self, patch_dim, merged):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.patch_embedding = nn.Linear(patch_dim, HIDDEN)
        self.embeddings.dense_embedding = nn.Linear(HIDDEN * merged, HIDDEN)

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True,
                grid_hw=None):
        patches = self.embeddings.patch_embedding(pixel_values)
        return SimpleNamespace(
            last_hidden_state=self.embeddings.dense_embedding(patches.reshape(1, -1)))


class _LatentTree(nn.Module):
    """One latent token wide: patch 4, C channels, ps3(1) head, zero conv2."""

    patch_size = 16          # the UNDERSTANDING tower's, unchanged by a swap
    downsample_ratio = 0.5

    def __init__(self, channels=CHANNELS, scale=8):
        super().__init__()
        self.use_pixel_head = True
        self.use_deep_fm_head = False
        self.config = SimpleNamespace(t_eps=T_EPS)
        self.add_noise_scale_embedding = True
        self.noise_scale = 1.0
        self.noise_scale_mode = "resolution"
        self.noise_scale_base_image_seq_len = 1.0
        self.noise_scale_max_value = 3.0
        self.gen_in_channels = channels
        self.gen_patch_size = PATCH
        self.gen_vit_patch_size = PATCH // 2
        self.gen_vae_scale_factor = scale
        model = nn.Module()
        model.layers = nn.ModuleList([_Layer()])
        model.norm_mot_gen = nn.Identity()
        self.language_model = SimpleNamespace(model=model)
        self.decoder_layer = model.layers[0]
        self.fm_modules = nn.ModuleDict({
            "vision_model_mot_gen": _GenVision(channels * 2 * 2, 4),
            "timestep_embedder": TimestepEmbedder(HIDDEN),
            "noise_scale_embedder": TimestepEmbedder(HIDDEN),
            "fm_head": ConvDecoder(input_dim=HIDDEN, hidden_dim=HIDDEN,
                                   out_channels=channels, shuffle=1),
        })
        with torch.no_grad():
            self.fm_modules["fm_head"].conv2.weight.zero_()
            self.fm_modules["fm_head"].conv2.bias.zero_()

    patchify = NEOChatModel.patchify
    unpatchify = NEOChatModel.unpatchify

    def _build_t2i_image_indexes(self, token_h, token_w, text_length, device):
        return torch.zeros(3, token_h * token_w, dtype=torch.long, device=device)

    def extract_feature(self, pixel_values, gen_model=False, grid_hw=None):
        assert gen_model
        return self.fm_modules["vision_model_mot_gen"](
            pixel_values=pixel_values, grid_hw=grid_hw).last_hidden_state


@contextmanager
def _captured_z():
    """The real ``z`` train_step built, not a re-derivation of it.

    ``_build_step_context`` is imported inside ``train_step``, so patching the
    module attribute intercepts the actual call.
    """
    import core.models.sensenova.sensenova_pipeline_ops as pipeline_ops

    seen = []
    original = pipeline_ops._build_step_context

    def recording(*args, **kwargs):
        z, embeds, timesteps = original(*args, **kwargs)
        seen.append(z.detach().clone())
        return z, embeds, timesteps

    pipeline_ops._build_step_context = recording
    try:
        yield seen
    finally:
        pipeline_ops._build_step_context = original


def _run(tree, t, latent=None):
    trainer = SimpleNamespace(transformer=tree, device=torch.device("cpu"),
                              training_dtype=torch.float32,
                              gradient_checkpointing=False)
    latents = latent if latent is not None else torch.ones(1, CHANNELS, 4, 4)
    return train_step(trainer, images=latents,
                      prefix=SenseNovaTrainingPrefix(_Cache(), text_length=3),
                      timesteps=torch.tensor([t]))


def test_velocity_stays_finite_at_both_t_endpoints():
    """§10.6-3: the clamp is what bounds it, and the zero head does not help.

    With ``x_pred = 0`` the model's velocity IS ``-z/(1-t).clamp_min(t_eps)``,
    so the number this records is the real one -- and it is 1/t_eps = 50x the
    latent norm at the clean end, not a small number.
    """
    tree = _LatentTree()
    tree.requires_grad_(False)
    x0 = torch.ones(1, CHANNELS, 4, 4)
    measured = {}
    for label, t in (("t_eps", T_EPS), ("1-t_eps", 1.0 - T_EPS)):
        with _captured_z() as seen:
            loss, value, recon = _run(tree, t, latent=x0)
        assert torch.isfinite(loss).all()
        # The zero head, observed rather than assumed: x_pred is 0, so the
        # reconstruction loss is exactly the target's own mean square, and the
        # velocity is exactly -z / (1-t).clamp_min(t_eps).
        tokens = NEOChatModel.patchify(tree, x0, PATCH)
        assert recon == pytest.approx(float((tokens ** 2).mean()), rel=1e-5)
        z = seen[0]
        velocity = -z / max(1.0 - t, T_EPS)
        norm = float(torch.linalg.vector_norm(velocity))
        assert torch.isfinite(velocity).all()
        measured[label] = (norm, float(torch.linalg.vector_norm(z)))
        print(f"[velocity] t={t:.2f}  ||z||={measured[label][1]:.4f}  ||v||={norm:.4f}")
    # The clean end is bounded by the clamp alone: same z scale, 1/t_eps = 50x
    # the divisor. Contained, not removed.
    assert measured["1-t_eps"][0] > 10 * measured["t_eps"][0]


def test_zero_head_gives_no_upstream_gradient_at_step_0_and_a_finite_one_after():
    """§10.6-3's second and third bullets, in one run.

    Step 0: ``conv2`` has a gradient (its input is non-zero), everything
    upstream of it has exactly none. Step >= 1 (the head having moved off zero):
    the upstream gradients are finite and non-zero.
    """
    tree = _LatentTree()
    tree.requires_grad_(False)
    tree.fm_modules.requires_grad_(True)
    tree.decoder_layer.requires_grad_(True)

    loss, _, _ = _run(tree, 0.25)
    loss.backward()
    head = tree.fm_modules["fm_head"]
    upstream = {
        "fm_head.conv1": head.conv1.weight,
        "gen_vit.patch_embedding": tree.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding.weight,
        "decoder_linear": tree.decoder_layer.scale,
    }
    assert float(head.conv2.weight.grad.abs().sum()) > 0
    step0 = {name: float(p.grad.abs().sum()) for name, p in upstream.items()}
    assert step0 == {name: 0.0 for name in upstream}

    # What an optimizer step does to the head; nothing else is touched.
    with torch.no_grad():
        head.conv2.weight.add_(torch.full_like(head.conv2.weight, 1e-3))
    for parameter in tree.parameters():
        parameter.grad = None

    loss, _, _ = _run(tree, 0.25)
    loss.backward()
    step1 = {name: float(p.grad.abs().sum()) for name, p in upstream.items()}
    for name, value in step1.items():
        assert value > 0 and torch.isfinite(torch.tensor(value)), name


def test_a_latent_tree_still_trains_a_finite_loss_over_three_steps():
    tree = _LatentTree()
    tree.requires_grad_(False)
    tree.fm_modules.requires_grad_(True)
    optimizer = torch.optim.SGD(tree.fm_modules.parameters(), lr=1e-3)
    losses = []
    for step in range(3):
        loss, value, _ = _run(tree, 0.1 + 0.3 * step)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(value)
    assert all(torch.isfinite(torch.tensor(v)) for v in losses)


# ---------------------------------------------------------------------------
# §10.6-1 -- the shape-invariant tensors survive a save and a reload
# ---------------------------------------------------------------------------

def _decoder_with_fm_modules():
    """The 588-Linear MoT tree the save path walks, plus a latent fm_modules."""
    from sensenova_full_finetune_save_test import _Decoder, _trained_tree

    tree = _trained_tree("gen")
    tree.fm_modules = _LatentTree().fm_modules
    tree.use_pixel_head = True
    tree.use_deep_fm_head = False
    return tree, _Decoder


def test_shape_invariant_tensors_and_the_bundled_vae_survive_a_round_trip(tmp_path):
    """§10.6-1: the property test the other architectures get does not apply
    here (the transform is a rebuild, not a partial copy), so what is pinned is
    that every tensor a swap does NOT touch is written and read back bit for
    bit -- together with the newly shaped ones and the bundled VAE."""
    from core.models.common.single_file_format import (
        TRANSFORMER_PREFIX, read_state_dict, strip_prefix,
    )
    from core.models.sensenova.loader import (
        install_sensenova_state_dict, save_sensenova_full_finetune_checkpoint,
    )

    tree, decoder_cls = _decoder_with_fm_modules()
    vae = nn.Sequential(nn.Conv2d(3, CHANNELS, 3), nn.Conv2d(CHANNELS, 3, 3))
    raw_config = latent_config_dict({"downsample_ratio": 0.5}, channels=CHANNELS,
                                    patch=PATCH)
    written, _census = save_sensenova_full_finetune_checkpoint(
        tree, str(tmp_path / "swapped_step_000100"), branch="gen",
        save_format="mixed", config=None, raw_config=raw_config, vae=vae)

    raw, metadata = read_state_dict(written)
    assert '"gen_in_channels": 16' in metadata["sensenova_config"]
    reloaded = strip_prefix(raw, TRANSFORMER_PREFIX)
    bundled = strip_prefix(raw, "vae.")
    assert set(bundled) == set(vae.state_dict())
    for name, tensor in vae.state_dict().items():
        assert torch.equal(bundled[name], tensor)

    from sensenova_full_finetune_save_test import _plain

    # The production read sequence: a freshly built tree (plain Linears, a
    # differently initialised fm_modules) with the file loaded into it.
    fresh = decoder_cls(factory=_plain)
    fresh.fm_modules = _LatentTree().fm_modules
    install_sensenova_state_dict(fresh, reloaded, {}, torch.bfloat16, path=written)
    saved = dict(tree.state_dict())
    for name, tensor in fresh.state_dict().items():
        if not name.startswith("fm_modules."):
            continue
        assert torch.equal(tensor, saved[name]), name


# ---------------------------------------------------------------------------
# Refusals and declarations
# ---------------------------------------------------------------------------

def test_the_capability_gate_is_still_shut():
    """§10.6: the entry comes off only when the acceptance conditions pass on
    real weights, which no test here can do."""
    from api.arch_capabilities import TRAINING_FEATURE_UNSUPPORTED

    assert "vae_swap" in TRAINING_FEATURE_UNSUPPORTED["sensenova"]


def test_the_swap_requires_fm_modules_training():
    from core.training.arch.sensenova import SenseNovaArchHandler

    tree = _PixelTree()
    trainer = SimpleNamespace(
        transformer=tree, config={"training_method": "full_finetune"},
        network_type="full_finetune", sensenova_train_fm_modules=False,
        train_unet=True, train_text_encoder=False, vae=None)
    resolved = SimpleNamespace(latent_channels=CHANNELS, scale_factor=8,
                               norm="shift_scale", norm_pack=1)
    with pytest.raises(ValueError, match="sensenova_train_fm_modules"):
        SenseNovaArchHandler(trainer).apply_vae_swap(trainer, resolved, module=object())


def test_the_config_block_carries_the_generation_grid():
    """The export re-embeds the block the load accepted, so a swap has to write
    its two keys into it or the file rebuilds as a pixel model."""
    out = latent_config_dict({"downsample_ratio": 0.5}, channels=CHANNELS,
                             patch=PATCH)
    assert out["gen_in_channels"] == CHANNELS
    assert out["gen_patch_size"] == PATCH
    assert out["downsample_ratio"] == 0.5


def test_a_config_and_component_block_that_disagree_are_refused():
    from core.models.sensenova.loader import _assert_declared_latent_geometry

    declared = SimpleNamespace(latent_channels=16, scale_factor=8,
                               provenance="registry:flux1")
    pixel_config = SimpleNamespace(gen_in_channels=None, gen_patch_size=None)
    with pytest.raises(ValueError, match="no gen_in_channels"):
        _assert_declared_latent_geometry(pixel_config, declared, path="x.safetensors")

    latent_config = SimpleNamespace(gen_in_channels=32, gen_patch_size=4)
    with pytest.raises(ValueError, match="32-channel generation grid"):
        _assert_declared_latent_geometry(latent_config, declared, path="x.safetensors")

    with pytest.raises(ValueError, match="no component.vae"):
        _assert_declared_latent_geometry(latent_config, None, path="x.safetensors")

    # The agreeing pair, and the native one, both pass.
    _assert_declared_latent_geometry(
        SimpleNamespace(gen_in_channels=16, gen_patch_size=4), declared, path="x")
    _assert_declared_latent_geometry(pixel_config, None, path="x")


# ---------------------------------------------------------------------------
# The patch is a parameter (`sensenova_gen_patch`), not a structure
# ---------------------------------------------------------------------------

def _swap_trainer(tree, config):
    return SimpleNamespace(
        transformer=tree, config=config, network_type="full_finetune",
        sensenova_train_fm_modules=True, train_unet=True,
        train_text_encoder=False, vae=None)


_RESOLVED_16CH_8X = SimpleNamespace(latent_channels=CHANNELS, scale_factor=8,
                                    norm="shift_scale", norm_pack=1)


def test_only_a_positive_multiple_of_four_is_a_legal_patch():
    """ps1(2)*ps2(2) leaves ps3 = P/4, and a PixelShuffle factor is an integer."""
    for legal in (4, 8, 12, 64):
        assert validate_gen_patch(legal) == legal
    for illegal in (0, -4, 2, 6, 10):
        with pytest.raises(ValueError, match="positive multiple of 4"):
            validate_gen_patch(illegal)
    assert MIN_GEN_LATENT_PATCH == 4


def test_a_coarser_patch_rebuilds_a_coarser_grid():
    """P=8 on an 8x VAE: 64px per token, ps3(2), a 4x4 patch-embed kernel."""
    tree = _PixelTree()
    apply_latent_geometry(tree, channels=CHANNELS, vae_scale_factor=8, patch=8)

    geometry = gen_geometry(tree)
    assert (geometry.patch, geometry.vit_patch, geometry.head_shuffle) == (8, 4, 2)
    assert geometry.token_pixel_width == 64
    embed = tree.fm_modules.vision_model_mot_gen.embeddings.patch_embedding
    assert tuple(embed.weight.shape) == (VIT_HIDDEN, CHANNELS, 4, 4)
    # conv2 fans out C*k^2 and ps3(k) folds it back to C on a k-times finer grid.
    assert tuple(tree.fm_modules.fm_head.conv2.weight.shape) == (
        CHANNELS * 4, HIDDEN // 4, 3, 3)
    assert tree.fm_modules.fm_head.ps3.upscale_factor == 2
    assert tree.fm_modules.fm_head(torch.randn(1, HIDDEN, 16, 16)).shape == (
        1, CHANNELS, 128, 128)


@pytest.mark.parametrize("patch,tokens,noise_scale", [(4, 2304, 6.0), (8, 576, 3.0)])
def test_token_count_and_noise_scale_at_1536px(patch, tokens, noise_scale):
    """What a coarser patch actually changes, at the checkpoint's own constants.

    ``compute_noise_scale`` is ``sqrt(tokens / 64) * 1.0``, so it follows the
    token count and falls by the factor the patch grows. NOTHING recalibrates
    it -- that is the warning's subject, not a bug.
    """
    from core.models.sensenova.sensenova_pipeline_ops import compute_noise_scale

    tree = _PixelTree()
    apply_latent_geometry(tree, channels=CHANNELS, vae_scale_factor=8, patch=patch)
    side = 1536 // token_pixel_width(tree)
    assert side * side == tokens

    merge = int(1 / tree.downsample_ratio)
    assert compute_noise_scale(tree, side * merge, side * merge,
                               merge) == pytest.approx(noise_scale)


def test_the_native_patch_is_the_pixel_model_geometry_and_its_init_is_unchanged():
    """P=4 at 8x is 32px per token -- the pixel model's own -- and the truncated
    normal is drawn exactly as before this parameter existed."""
    assert PATCH == MIN_GEN_LATENT_PATCH == 4

    tree = _PixelTree()
    apply_latent_geometry(tree, channels=CHANNELS, vae_scale_factor=8, patch=PATCH,
                          generator=torch.Generator().manual_seed(0))
    assert token_pixel_width(tree) == 32

    # The draw itself, spelled out: fp32 on CPU, std = 1/sqrt(C * (P/merge)^2),
    # truncated at 3 std. Only the trunc_normal reads the passed generator.
    std = 1.0 / (CHANNELS * 2 * 2) ** 0.5
    expected = torch.empty(VIT_HIDDEN, CHANNELS, 2, 2, dtype=torch.float32)
    nn.init.trunc_normal_(expected, std=std, a=-3 * std, b=3 * std,
                          generator=torch.Generator().manual_seed(0))
    weight = tree.fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight
    assert torch.equal(weight, expected)


def test_the_run_patch_reaches_the_rebuild_and_the_config_block(capsys):
    """The blocker: a P=8 run must not write ``gen_patch_size: 4`` into its own
    config block, which would rebuild as a different geometry on the next load."""
    from core.training.arch.sensenova import SenseNovaArchHandler

    tree = _PixelTree()
    trainer = _swap_trainer(tree, {"training_method": "full_finetune",
                                   "sensenova_gen_patch": 8,
                                   "base_resolutions": [1536]})
    SenseNovaArchHandler(trainer).apply_vae_swap(
        trainer, _RESOLVED_16CH_8X, module=object())

    assert gen_geometry(tree).patch == 8
    assert trainer.sensenova_config_dict["gen_patch_size"] == 8
    assert trainer.sensenova_config_dict["gen_in_channels"] == CHANNELS
    # The un-recalibrated schedule is announced, with the numbers.
    out = capsys.readouterr().out
    assert "sensenova_gen_patch_off_calibration" in out
    assert "576 tokens against 2304" in out
    assert "3.0000" in out


@pytest.mark.parametrize("config", [
    {"training_method": "full_finetune"},                             # key absent
    {"training_method": "full_finetune",
     "sensenova_gen_patch": TRAINING_DEFAULTS["sensenova_gen_patch"]},  # as served
])
def test_a_pixel_base_at_the_default_builds_at_the_native_patch(config, capsys):
    """Nothing to inherit: the sentinel takes the architecture's own 4, which is
    the pixel model's geometry, so the swap is the one it always was."""
    from core.training.arch.sensenova import SenseNovaArchHandler

    tree = _PixelTree()
    trainer = _swap_trainer(tree, dict(config))
    SenseNovaArchHandler(trainer).apply_vae_swap(
        trainer, _RESOLVED_16CH_8X, module=object())

    assert gen_geometry(tree).patch == PATCH
    assert trainer.sensenova_config_dict["gen_patch_size"] == PATCH
    assert "sensenova_gen_patch_off_calibration" not in capsys.readouterr().out


def test_a_latent_base_at_its_own_patch_is_not_rebuilt():
    """Re-training a P=8 checkpoint keeps its trained layers, and its own patch
    is what gets written back."""
    from core.training.arch.sensenova import SenseNovaArchHandler

    tree = _PixelTree()
    apply_latent_geometry(tree, channels=CHANNELS, vae_scale_factor=8, patch=8)
    before = _params(tree)
    trainer = _swap_trainer(tree, {"training_method": "full_finetune",
                                   "sensenova_gen_patch": 8})
    report = SenseNovaArchHandler(trainer).apply_vae_swap(
        trainer, _RESOLVED_16CH_8X, module=object())

    assert report.replaced == ()
    after = _params(tree)
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), name
    assert trainer.sensenova_config_dict["gen_patch_size"] == 8


def _latent_tree(patch: int):
    """A base already migrated to 16ch / 8x at ``patch``, as a resumed run sees it."""
    tree = _PixelTree()
    apply_latent_geometry(tree, channels=CHANNELS, vae_scale_factor=8, patch=patch)
    return tree


def _lora_trainer(tree, config):
    trainer = _swap_trainer(tree, config)
    trainer.network_type = "lora"
    return trainer


def test_the_inherit_sentinel_is_resolved_before_it_is_validated():
    """0 is not a legal patch -- it is the ABSENCE of a request, and only
    ``resolve_gen_patch`` understands it."""
    assert TRAINING_DEFAULTS["sensenova_gen_patch"] == INHERIT_GEN_PATCH == 0
    with pytest.raises(ValueError, match="positive multiple of 4"):
        validate_gen_patch(INHERIT_GEN_PATCH)

    assert resolve_gen_patch(0) == NATIVE_GEN_LATENT_PATCH
    assert resolve_gen_patch(None) == NATIVE_GEN_LATENT_PATCH
    assert resolve_gen_patch(0, base_patch=8) == 8
    assert resolve_gen_patch(None, base_patch=12) == 12
    # A positive value is an explicit request; the base's patch does not soften it.
    assert resolve_gen_patch(4, base_patch=8) == 4
    with pytest.raises(ValueError, match="positive multiple of 4"):
        resolve_gen_patch(6, base_patch=8)


def test_a_lora_run_may_not_rebuild_the_latent_io():
    """P1: the rebuild is full-fine-tune-only whatever asked for it.

    ``vae_swap_source`` is empty here, so the capability table's method gate --
    which keys on that field -- never sees this run; the base's own declaration
    is what brings it to ``apply_vae_swap``.
    """
    from core.training.arch.sensenova import SenseNovaArchHandler

    tree = _latent_tree(4)
    before = _params(tree)
    trainer = _lora_trainer(tree, {"training_method": "lora",
                                   "vae_swap_source": "",
                                   "sensenova_gen_patch": 8})
    with pytest.raises(ValueError, match="only under a full fine-tune"):
        SenseNovaArchHandler(trainer).apply_vae_swap(
            trainer, _RESOLVED_16CH_8X, module=object())

    # Refused BEFORE anything was replaced.
    assert gen_geometry(tree).patch == 4
    after = _params(tree)
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), name


def test_a_lora_run_on_a_matching_latent_base_still_trains():
    """The other half of P1: restoring the base's own geometry is not a rebuild,
    so an ordinary LoRA on a swapped checkpoint keeps working."""
    from core.training.arch.sensenova import SenseNovaArchHandler

    tree = _latent_tree(8)
    before = _params(tree)
    trainer = _lora_trainer(tree, {"training_method": "lora",
                                   "vae_swap_source": ""})
    report = SenseNovaArchHandler(trainer).apply_vae_swap(
        trainer, _RESOLVED_16CH_8X, module=object())

    assert report.replaced == ()
    after = _params(tree)
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), name
    assert gen_geometry(tree).patch == 8
    assert trainer.sensenova_config_dict["gen_patch_size"] == 8
    assert trainer.wiring.latent_channels == CHANNELS


def test_an_explicit_patch_of_four_rebuilds_a_p8_base_under_full_finetune():
    """P2: 4 is a value a caller can ask for again, not "the field is absent"."""
    from core.training.arch.sensenova import SenseNovaArchHandler

    tree = _latent_tree(8)
    trainer = _swap_trainer(tree, {"training_method": "full_finetune",
                                   "sensenova_gen_patch": 4})
    report = SenseNovaArchHandler(trainer).apply_vae_swap(
        trainer, _RESOLVED_16CH_8X, module=object())

    assert report.replaced == (
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding",
        "fm_modules.fm_head.conv2")
    assert gen_geometry(tree).patch == 4
    assert trainer.sensenova_config_dict["gen_patch_size"] == 4


@pytest.mark.parametrize("config", [
    {"training_method": "full_finetune"},                             # key absent
    {"training_method": "full_finetune",
     "sensenova_gen_patch": TRAINING_DEFAULTS["sensenova_gen_patch"]},  # as served
])
def test_the_default_against_a_p8_base_inherits_8(config):
    """The hazard the sentinel exists for: update_training_run resends every
    Pydantic default, and that must not replace the two trained layers."""
    from core.training.arch.sensenova import SenseNovaArchHandler

    tree = _latent_tree(8)
    before = _params(tree)
    trainer = _swap_trainer(tree, dict(config))
    report = SenseNovaArchHandler(trainer).apply_vae_swap(
        trainer, _RESOLVED_16CH_8X, module=object())

    assert report.replaced == ()
    after = _params(tree)
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), name
    assert gen_geometry(tree).patch == 8
    assert trainer.sensenova_config_dict["gen_patch_size"] == 8


def test_the_preflight_reads_the_sentinel_as_no_request(tmp_path):
    """The same reading before the load: 0 asks for nothing, a positive value
    needs something that actually rebuilds the grid."""
    from core.training.train_runner import (
        _apply_sensenova_full_finetune_contract,
    )

    base = str(tmp_path / "sensenova.safetensors")  # no component.vae.* block

    config = {"sensenova_gen_patch": TRAINING_DEFAULTS["sensenova_gen_patch"]}
    _apply_sensenova_full_finetune_contract(config, base_model_path=base)

    with pytest.raises(ValueError, match="requires a vae_swap_source"):
        _apply_sensenova_full_finetune_contract(
            {"sensenova_gen_patch": 8}, base_model_path=base)
    with pytest.raises(ValueError, match="positive multiple of 4"):
        _apply_sensenova_full_finetune_contract(
            {"sensenova_gen_patch": 6, "vae_swap_source": "registry:flux1"},
            base_model_path=base)
    # An explicit patch WITH something that rebuilds is accepted.
    _apply_sensenova_full_finetune_contract(
        {"sensenova_gen_patch": 8, "vae_swap_source": "registry:flux1",
         "sensenova_train_fm_modules": True},
        base_model_path=base)


def test_the_loader_accepts_the_patch_the_checkpoint_declares():
    """A P=8 checkpoint has to be reloadable, or the parameter is useless."""
    from core.models.sensenova.loader import (
        _assert_built_latent_geometry, _assert_declared_latent_geometry,
    )

    declared = SimpleNamespace(latent_channels=CHANNELS, scale_factor=8,
                               provenance="registry:flux1")
    for patch in (4, 8, 64):
        _assert_declared_latent_geometry(
            SimpleNamespace(gen_in_channels=CHANNELS, gen_patch_size=patch),
            declared, path="x.safetensors")
    with pytest.raises(ValueError, match="positive multiple of 4"):
        _assert_declared_latent_geometry(
            SimpleNamespace(gen_in_channels=CHANNELS, gen_patch_size=6),
            declared, path="x.safetensors")

    # And the tree that was built has to face the grid the file declared.
    config = SimpleNamespace(gen_in_channels=CHANNELS, gen_patch_size=8)
    _assert_built_latent_geometry(
        SimpleNamespace(gen_in_channels=CHANNELS, gen_patch_size=8),
        config, path="x.safetensors")
    with pytest.raises(ValueError, match="patch 4"):
        _assert_built_latent_geometry(
            SimpleNamespace(gen_in_channels=CHANNELS, gen_patch_size=4),
            config, path="x.safetensors")


def test_a_p8_config_block_survives_the_metadata_codec(tmp_path):
    """Write -> read -> same geometry, through the save path's own embedder."""
    import json

    from core.models.common.single_file_format import read_state_dict
    from core.models.sensenova.loader import (
        _assert_declared_latent_geometry, save_sensenova_full_finetune_checkpoint,
    )

    tree, _decoder_cls = _decoder_with_fm_modules()
    raw_config = latent_config_dict({"downsample_ratio": 0.5},
                                    channels=CHANNELS, patch=8)
    written, _census = save_sensenova_full_finetune_checkpoint(
        tree, str(tmp_path / "p8_step_000100"), branch="gen",
        save_format="mixed", config=None, raw_config=raw_config, vae=None)

    _raw, metadata = read_state_dict(written)
    reread = json.loads(metadata["sensenova_config"])
    assert reread["gen_patch_size"] == 8 and reread["gen_in_channels"] == CHANNELS
    _assert_declared_latent_geometry(
        SimpleNamespace(gen_in_channels=reread["gen_in_channels"],
                        gen_patch_size=reread["gen_patch_size"]),
        SimpleNamespace(latent_channels=CHANNELS, scale_factor=8,
                        provenance="registry:flux1"),
        path=written)


# --- The noise-scale recalibration a swap needs (§10.4) -----------------------


def test_the_gain_sentinel_and_its_one_contradiction():
    """0 is the ABSENCE of a request, and auto/explicit cannot both decide."""
    from core.training.arch.sensenova import SenseNovaArchHandler as _H

    assert TRAINING_DEFAULTS["sensenova_noise_scale_gain"] == INHERIT_NOISE_SCALE_GAIN == 0
    assert TRAINING_DEFAULTS["sensenova_noise_scale_auto"] is False
    assert _H.resolve_noise_scale_config({}) == (0.0, False)
    assert _H.resolve_noise_scale_config({"sensenova_noise_scale_gain": 1.462}) == (1.462, False)
    assert _H.resolve_noise_scale_config({"sensenova_noise_scale_auto": True}) == (0.0, True)
    with pytest.raises(ValueError, match="exactly one of them"):
        _H.resolve_noise_scale_config({"sensenova_noise_scale_gain": 1.462,
                                       "sensenova_noise_scale_auto": True})
    with pytest.raises(ValueError, match="must be >= 0"):
        _H.resolve_noise_scale_config({"sensenova_noise_scale_gain": -1})


def test_the_gain_restores_the_pixel_era_signal_to_noise_ratio():
    """The gain is the RATIO of data scales, so `t*RMS / ((1-t)*noise_scale)`
    comes out the same in both spaces."""
    gain = noise_scale_gain_for_rms(1.0)
    assert gain == pytest.approx(1.0 / PIXEL_RMS)
    for t in (0.2, 0.5, 0.8):
        pixel = t * PIXEL_RMS / ((1 - t) * 3.0)
        latent = t * 1.0 / ((1 - t) * 3.0 * gain)
        assert latent == pytest.approx(pixel)


def test_a_resume_cannot_compound_the_gain():
    """apply_vae_swap re-runs on every resume, so the gain must land on the
    value the checkpoint had BEFORE any earlier recalibration."""
    tree = SimpleNamespace(noise_scale=1.0)
    config = apply_noise_scale_gain(tree, {"noise_scale": 1.0}, 1.462,
                                    provenance="config")
    assert config["noise_scale"] == pytest.approx(1.462)
    assert config["gen_noise_scale_base"] == 1.0
    assert tree.noise_scale == pytest.approx(1.462)

    again = apply_noise_scale_gain(tree, config, 1.462, provenance="config")
    assert again["noise_scale"] == pytest.approx(1.462)
    assert tree.noise_scale == pytest.approx(1.462)

    # A CHANGED gain still lands on base * gain, not on the recalibrated value.
    changed = apply_noise_scale_gain(tree, again, 2.0, provenance="config")
    assert changed["noise_scale"] == pytest.approx(2.0)
    assert changed["gen_noise_scale_base"] == 1.0


def test_the_recalibrated_scale_survives_the_metadata_codec(tmp_path):
    """Write -> read -> the value the model is rebuilt from, and a resume of the
    RE-READ block still cannot compound the gain."""
    import json

    from core.models.common.single_file_format import read_state_dict
    from core.models.sensenova.loader import save_sensenova_full_finetune_checkpoint

    tree, _decoder_cls = _decoder_with_fm_modules()
    tree.noise_scale = 1.0
    raw_config = latent_config_dict({"downsample_ratio": 0.5, "noise_scale": 1.0},
                                    channels=CHANNELS, patch=8)
    raw_config = apply_noise_scale_gain(tree, raw_config, 1.462,
                                        provenance="measured")
    written, _census = save_sensenova_full_finetune_checkpoint(
        tree, str(tmp_path / "gain_step_000100"), branch="gen",
        save_format="mixed", config=None, raw_config=raw_config, vae=None)

    reread = json.loads(read_state_dict(written)[1]["sensenova_config"])
    assert reread["noise_scale"] == pytest.approx(1.462)
    assert reread["gen_noise_scale_base"] == 1.0
    assert reread["gen_patch_size"] == 8
    assert recalibrated_noise_scale_gain(reread) == pytest.approx(1.462)
    assert recalibrated_noise_scale_gain({"noise_scale": 1.0}) is None

    # What a resume does: rebuild from the re-read block, swap again, same gain.
    resumed = SimpleNamespace(noise_scale=reread["noise_scale"])
    again = apply_noise_scale_gain(resumed, reread, 1.462, provenance="config")
    assert again["noise_scale"] == pytest.approx(1.462)
    assert resumed.noise_scale == pytest.approx(1.462)


def test_a_lora_may_not_recalibrate_the_noise_scale():
    """A LoRA reaches apply_vae_swap through a latent base's OWN declaration and
    saves no config block, so the scale would exist for the run and for nothing
    that loads its output."""
    from core.training.arch.sensenova import SenseNovaArchHandler
    from core.training.train_runner import _check_sensenova_noise_scale

    trainer = _lora_trainer(_latent_tree(8), {
        "training_method": "lora", "vae_swap_source": "",
        "sensenova_noise_scale_gain": 1.462})
    with pytest.raises(ValueError, match="requires a full fine-tune"):
        SenseNovaArchHandler(trainer).apply_vae_swap(
            trainer, _RESOLVED_16CH_8X, module=object())

    # And refused before the load, for both settings and with no swap source.
    for asked in ({"sensenova_noise_scale_gain": 1.462},
                  {"sensenova_noise_scale_auto": True}):
        with pytest.raises(ValueError, match="requires training_method"):
            _check_sensenova_noise_scale(dict(asked), "", is_full_finetune=False)


def test_the_config_reads_refuse_python_truthiness():
    """Every other SenseNova bool goes through the strict normaliser; a
    hand-written "false" must not enable auto."""
    from core.training.train_runner import _check_sensenova_noise_scale

    with pytest.raises(ValueError, match="must be a boolean"):
        _check_sensenova_noise_scale({"sensenova_noise_scale_auto": "yes"}, "",
                                     is_full_finetune=True)
    with pytest.raises(ValueError, match="must be a number"):
        _check_sensenova_noise_scale({"sensenova_noise_scale_gain": True}, "",
                                     is_full_finetune=True)
    # "false" is a boolean here, not a non-empty truthy string.
    _check_sensenova_noise_scale({"sensenova_noise_scale_auto": "false"}, "",
                                 is_full_finetune=True)


def test_an_explicit_gain_is_applied_at_swap_time():
    """The config path stamps at apply_vae_swap, before the freeze/optimizer."""
    from core.training.arch.sensenova import SenseNovaArchHandler

    tree = _PixelTree()
    trainer = _swap_trainer(tree, {"training_method": "full_finetune",
                                   "vae_swap_source": "registry:sdxl",
                                   "sensenova_gen_patch": 8,
                                   "base_resolutions": [1536],
                                   "sensenova_noise_scale_gain": 1.462})
    trainer.sensenova_config_dict = {"noise_scale": 1.0}
    SenseNovaArchHandler(trainer).apply_vae_swap(
        trainer, _RESOLVED_16CH_8X, module=object())

    assert tree.noise_scale == pytest.approx(1.462)
    assert trainer.sensenova_config_dict["noise_scale"] == pytest.approx(1.462)
    assert trainer.sensenova_config_dict["gen_noise_scale_base"] == 1.0


def test_a_pixel_space_run_may_not_recalibrate():
    """Nothing to correct: the sample IS the data the schedule was calibrated on."""
    from core.training.train_runner import _check_sensenova_noise_scale

    _check_sensenova_noise_scale({"vae_swap_source": "registry:sdxl",
                                  "sensenova_noise_scale_gain": 1.462}, "",
                                 is_full_finetune=True)
    for asked in ({"sensenova_noise_scale_gain": 1.462},
                  {"sensenova_noise_scale_auto": True}):
        with pytest.raises(ValueError, match="requires a latent space"):
            _check_sensenova_noise_scale(dict(asked, vae_swap_source=""), "",
                                         is_full_finetune=True)
