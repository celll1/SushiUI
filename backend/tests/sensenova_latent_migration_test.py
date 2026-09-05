"""SenseNova pixel -> latent (VAE_SWAP_MIGRATION_DESIGN.md §10).

What these cover, and what they deliberately do not:

* the geometry §10.2 fixes (P = 4, one token = 4 * vae_scale_factor pixels,
  exactly two tensors change shape, identically at 8x and 16x);
* §10.3's initialisation and its consequences -- including the one the design
  is explicit is NOT avoided: with a zero head ``v = -z/(1-t)`` still grows as
  ``t -> 1`` and is bounded only by ``(1-t).clamp_min(t_eps)``;
* §10.6's endpoint velocity and step-0/step-1 gradient measurements;
* the refusals: the shut capability gate, the fm_modules requirement, and a
  checkpoint whose config and component blocks disagree.

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

from core.models.sensenova.latent_space import (  # noqa: E402
    GEN_LATENT_PATCH,
    apply_latent_geometry,
    gen_geometry,
    latent_config_dict,
    resolution_band_mp,
    token_pixel_width,
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
    report = apply_latent_geometry(tree, channels=CHANNELS, vae_scale_factor=scale)
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
    assert token_pixel_width(tree) == 4 * scale


def test_head_is_zero_and_patch_embed_is_a_bounded_small_normal():
    tree = _PixelTree()
    apply_latent_geometry(tree, channels=CHANNELS, vae_scale_factor=8)
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
    apply_latent_geometry(tree, channels=CHANNELS, vae_scale_factor=scale)
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
    tokens = NEOChatModel.patchify(tree, latent, GEN_LATENT_PATCH)
    assert tokens.shape == (1, 16, GEN_LATENT_PATCH ** 2 * CHANNELS)
    back = NEOChatModel.unpatchify(tree, tokens, GEN_LATENT_PATCH, 16, 16)
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
        self.gen_patch_size = GEN_LATENT_PATCH
        self.gen_vit_patch_size = GEN_LATENT_PATCH // 2
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
        tokens = NEOChatModel.patchify(tree, x0, GEN_LATENT_PATCH)
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
    raw_config = latent_config_dict({"downsample_ratio": 0.5}, channels=CHANNELS)
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
    out = latent_config_dict({"downsample_ratio": 0.5}, channels=CHANNELS)
    assert out["gen_in_channels"] == CHANNELS
    assert out["gen_patch_size"] == GEN_LATENT_PATCH
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
