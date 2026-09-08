"""SenseNova's REPA tap: a MoT decoder loop that lives in this repo, not a vendor.

``sensenova_ops.forward_gen_decoder_layers`` runs the generation half against the
understanding half's prefix K/V, so the sequence it carries is image tokens only
-- no text prefix to slice off. The tap is an explicit assignment inside that
loop rather than a forward hook, so the tap IS the tensor the loss
differentiates and there is no hook ordering or checkpoint recompute behaviour
to reason about.

What is pinned here:
  (a) unarmed, the loop neither reads nor writes anything and the forward is
      bit-identical;
  (b) armed, the gradient reaches the layers at and below the tap and stops
      above it, including under gradient checkpointing;
  (c) the tap's tokens are row-major over the TOKEN grid (one row per
      ``patch * vae_scale_factor`` px, not per latent cell), in the order
      ``patchify`` builds the sequence in, ``_build_t2i_image_indexes`` assigns
      (h, w) coordinates in, and ``encode_repa_targets`` builds the teacher grid
      in;
  (d) a full fine-tune pairs the projector sidecar with the file it wrote --
      which is the RESOLVED path, since the save appends the suffix;
  (e) the packed batch form ``[1, B*N, D]`` regroups item-major, so item i's
      rows meet item i's teacher features.

The transformer is a toy: the real vendored generation ViT, patchify and index
builder at a 2x3 token grid, with stub decoder layers. No checkpoint is read and
no GPU is used.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/repa_sensenova_tap_test.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from api.arch_capabilities import TRAINING_FEATURE_UNSUPPORTED  # noqa: E402
from core.models.sensenova.vendor.configuration_neo_vit import NEOVisionConfig  # noqa: E402
from core.models.sensenova.vendor.modeling_neo_chat import NEOChatModel  # noqa: E402
from core.models.sensenova.vendor.modeling_neo_vit import NEOVisionModel  # noqa: E402
from core.models.sensenova.vendor.modeling_qwen3 import (  # noqa: E402
    PackedSegments, Qwen3RMSNorm,
)
from core.training import repa as repa_module  # noqa: E402
from core.training.arch import ARCH_REGISTRY  # noqa: E402
from core.training.arch.base_arch import TrainStepContext  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.ops import sensenova_ops  # noqa: E402
from core.training.ops.sensenova_ops import SenseNovaTrainingPrefix  # noqa: E402

HIDDEN = 16          # decoder width; the shipped model is 42 layers of 3584
VIT_HIDDEN = 8
VIT_PATCH = 16       # the shipped value
MERGE = 2            # 1 / downsample_ratio, the shipped value
PATCH = VIT_PATCH * MERGE   # 32px per token, the pixel model's own grid
LAYERS = 4
TH, TW = 2, 3
HEIGHT, WIDTH = TH * PATCH, TW * PATCH
TOKENS = TH * TW
ENC_DIM = 8
ENC_GRID = 4
REPA_SIZE = 32


# ---------------------------------------------------------------------------
# Toy transformer
# ---------------------------------------------------------------------------

class _Layer(nn.Module):
    """One decoder layer. ``mix`` averages over the sequence, so a perturbation
    of one token reaches every other one -- what makes the row-major measurement
    a ratio against a non-zero median rather than against exact zeros."""

    def __init__(self, mix: bool):
        super().__init__()
        self.mix = mix
        self.proj = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, hidden_states, **kwargs):
        assert kwargs["update_cache"] is False and kwargs["use_cache"] is False
        if self.mix:
            hidden_states = hidden_states + 0.25 * hidden_states.mean(dim=1, keepdim=True)
        return self.proj(hidden_states)


class _Decoder(nn.Module):
    def __init__(self, mix: bool):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(mix) for _ in range(LAYERS)])
        self.norm_mot_gen = Qwen3RMSNorm(HIDDEN)


class _TimestepEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, HIDDEN)

    def forward(self, t):
        return self.proj(t.reshape(-1, 1).to(self.proj.weight.dtype))


class _Toy(nn.Module):
    """The surface ``sensenova_ops.train_step`` reads, with the real generation
    ViT, ``patchify`` and ``_build_t2i_image_indexes``."""

    patch_size = VIT_PATCH
    downsample_ratio = 1.0 / MERGE
    noise_scale = 1.0
    noise_scale_mode = "resolution"
    noise_scale_base_image_seq_len = 64
    noise_scale_max_value = 16.0
    add_noise_scale_embedding = False
    use_pixel_head = True
    use_deep_fm_head = False
    config = SimpleNamespace(t_eps=0.05)
    patchify = NEOChatModel.patchify
    extract_feature = NEOChatModel.extract_feature
    _build_t2i_image_indexes = NEOChatModel._build_t2i_image_indexes

    def __init__(self, mix: bool = True):
        super().__init__()
        vision_config = NEOVisionConfig(
            num_channels=3, patch_size=VIT_PATCH, hidden_size=VIT_HIDDEN,
            llm_hidden_size=HIDDEN, downsample_ratio=1.0 / MERGE,
        )
        self.fm_modules = nn.ModuleDict({
            "vision_model_mot_gen": NEOVisionModel(vision_config),
            "timestep_embedder": _TimestepEmbedder(),
            "fm_head": nn.Sequential(nn.Conv2d(HIDDEN, 3 * PATCH * PATCH, 1),
                                     nn.PixelShuffle(PATCH)),
        })
        self.decoder = _Decoder(mix)
        self.language_model = SimpleNamespace(model=self.decoder)


class _CacheLayer:
    def __init__(self):
        self.keys = torch.ones(1, 1, 2, 1)
        self.values = torch.ones(1, 1, 2, 1)
        self.flash_k_cache = None
        self.flash_v_cache = None


class _Cache:
    def __init__(self, packed=None):
        self.layers = [_CacheLayer() for _ in range(LAYERS)]
        self.packed = packed
        self._kv_cache_streamer = None
        self._kv_cache_streamer_branch = None


def _toy(mix: bool = True, seed: int = 0) -> _Toy:
    torch.manual_seed(seed)
    return _Toy(mix)


def _prefix(batch: int = 1):
    if batch == 1:
        return SenseNovaTrainingPrefix(_Cache(), text_length=3)
    lengths = [3, 2][:batch]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    segments = PackedSegments(torch.tensor(offsets, dtype=torch.int32))
    return SenseNovaTrainingPrefix(_Cache(segments), text_length=lengths[0],
                                   packed=segments, text_lengths=lengths)


class _StubEncoder(nn.Module):
    """A frozen teacher shaped like SigLIP2: a square token grid, no CLS."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        feat = torch.linspace(0, 1, ENC_GRID * ENC_GRID * ENC_DIM)
        feat = feat.reshape(1, ENC_GRID * ENC_GRID, ENC_DIM).expand(b, -1, -1)
        return SimpleNamespace(last_hidden_state=feat.contiguous())


def _sensenova_trainer(model=None, **config):
    cfg = {"repa_enable": True, "repa_tagger_model_dir": "unused-because-stubbed"}
    cfg.update(config)
    return SimpleNamespace(
        config=cfg,
        arch=ARCH_REGISTRY["sensenova"](),
        transformer=model if model is not None else _toy(),
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        gradient_checkpointing=False,
        model_path="",
        log_prefix="[test]",
    )


def _stub_encoder_loader(monkeypatch):
    monkeypatch.setattr(repa_module, "load_repa_encoder",
                        lambda source, **kwargs: (_StubEncoder(), ENC_DIM, REPA_SIZE))


def _train_step_trainer(model, align_depth, logged, **overrides):
    trainer = _sensenova_trainer(model)
    trainer.repa_enable = True  # _setup_repa's own flag, not the config key
    trainer.crop_decode_loss_enable = False
    trainer.repa_encoder = _StubEncoder()
    trainer.repa_projector = repa_module.RepaProjector(HIDDEN, ENC_DIM, hidden=16)
    trainer.repa_size = REPA_SIZE
    trainer.repa_weight = 1.0
    trainer.repa_align_depth = align_depth
    trainer._repa_tap_module = model.decoder
    trainer._ensure_repa_on_device = lambda: None
    trainer.log_extra_metric = lambda name, value: logged.append((name, value))
    for key, value in overrides.items():
        setattr(trainer, key, value)
    model.decoder._repa_tap_depth = align_depth
    return trainer


def _images(batch: int = 1, seed: int = 3):
    torch.manual_seed(seed)
    return torch.rand(batch, 3, HEIGHT, WIDTH) * 2 - 1


def _train_step(trainer, *, images=None, repa_pixels, batch=1, seed=7, t=0.25):
    torch.manual_seed(seed)
    return sensenova_ops.train_step(
        trainer,
        images=_images(batch) if images is None else images,
        prefix=_prefix(batch),
        timesteps=torch.full((batch,), t),
        repa_pixels=repa_pixels,
    )


def _pixels(batch: int = 1):
    torch.manual_seed(21)
    return torch.rand(batch, 3, REPA_SIZE, REPA_SIZE) * 2 - 1


# ---------------------------------------------------------------------------
# (a) the disabled path
# ---------------------------------------------------------------------------

def _decoder_forward(model, hidden, batch=1):
    return sensenova_ops.forward_gen_decoder_layers(
        model.decoder, hidden,
        indexes=model._build_t2i_image_indexes(TH, TW, 3, torch.device("cpu")),
        prefix_cache=_Cache(),
    )


def test_an_unarmed_decoder_carries_no_tap_state_at_all():
    """The loop is ours, not a vendored forward, so nothing is declared on the
    module: unarmed it neither reads nor writes."""
    model = _toy()
    torch.manual_seed(1)
    hidden = torch.randn(1, TOKENS, HIDDEN)

    with torch.no_grad():
        _decoder_forward(model, hidden)

    assert not hasattr(model.decoder, "_repa_tap_depth")
    assert not hasattr(model.decoder, "_repa_tap_out")


def test_arming_the_tap_does_not_change_the_decoder_output():
    model = _toy().eval()
    torch.manual_seed(1)
    hidden = torch.randn(1, TOKENS, HIDDEN)

    with torch.no_grad():
        disabled = _decoder_forward(model, hidden)
        model.decoder._repa_tap_depth = 1
        armed = _decoder_forward(model, hidden)

    assert torch.equal(disabled, armed)
    assert model.decoder._repa_tap_out is not None


def test_the_train_step_is_bit_identical_with_the_tap_armed_but_repa_off():
    """The two halves of invariant 1: arming writes an attribute and nothing
    else, and ``repa_enable=false`` never enters the alignment block."""
    plain = _toy(seed=5)
    armed = _toy(seed=5)
    armed.decoder._repa_tap_depth = 1
    trainer_plain = SimpleNamespace(transformer=plain, device=torch.device("cpu"),
                                    training_dtype=torch.float32,
                                    gradient_checkpointing=False)
    trainer_armed = SimpleNamespace(transformer=armed, device=torch.device("cpu"),
                                    training_dtype=torch.float32,
                                    gradient_checkpointing=False,
                                    repa_enable=False)

    images = _images()
    pixels = _pixels()
    torch.manual_seed(7)
    loss_a, value_a, recon_a = sensenova_ops.train_step(
        trainer_plain, images=images, prefix=_prefix(), timesteps=torch.tensor([0.25]))
    torch.manual_seed(7)
    loss_b, value_b, recon_b = sensenova_ops.train_step(
        trainer_armed, images=images, prefix=_prefix(), timesteps=torch.tensor([0.25]),
        repa_pixels=pixels)

    assert torch.equal(loss_a, loss_b)
    assert (value_a, recon_a) == (value_b, recon_b)


def test_a_forward_clears_what_a_previous_one_stashed():
    """Otherwise a step whose tap does not fire would read stale tokens."""
    model = _toy().eval()
    torch.manual_seed(1)
    hidden = torch.randn(1, TOKENS, HIDDEN)
    model.decoder._repa_tap_depth = 1

    with torch.no_grad():
        _decoder_forward(model, hidden)
        assert model.decoder._repa_tap_out is not None

        # A depth no layer carries: the tap cannot fire, so only the clear can
        # decide what is left behind.
        model.decoder._repa_tap_depth = LAYERS + 5
        _decoder_forward(model, hidden)

    assert model.decoder._repa_tap_out is None


# ---------------------------------------------------------------------------
# the handler's answer
# ---------------------------------------------------------------------------

def test_the_handler_answers_with_the_decoder_its_width_and_its_layer_count():
    model = _toy()
    trainer = _sensenova_trainer(model)

    tap = trainer.arch.repa_tap(trainer)

    assert tap.module is model.decoder
    assert tap.hidden_size == HIDDEN
    assert tap.depth == LAYERS == len(model.decoder.layers)
    # The same layer list the depth axis for both MoT halves is measured on.
    assert trainer.arch.depth_blocks(trainer) is model.decoder.layers


def test_the_width_comes_from_the_live_norm_not_from_a_config():
    """A config that no longer describes the loaded tree must not size the
    projector (the Lens precedent)."""
    model = _toy()
    model.decoder.config = SimpleNamespace(hidden_size=999)
    trainer = _sensenova_trainer(model)

    assert trainer.arch.repa_tap(trainer).hidden_size == HIDDEN


def test_a_tree_without_the_generation_norm_is_refused():
    model = _toy()
    del model.decoder.norm_mot_gen
    trainer = _sensenova_trainer(model)

    with pytest.raises(ValueError, match="norm_mot_gen"):
        trainer.arch.repa_tap(trainer)


def test_setup_repa_arms_the_sensenova_decoder(monkeypatch):
    _stub_encoder_loader(monkeypatch)
    model = _toy()
    trainer = _sensenova_trainer(model)

    BaseTrainer._setup_repa(trainer)

    assert trainer.repa_align_depth == LAYERS // 3
    assert model.decoder._repa_tap_depth == LAYERS // 3
    assert trainer._repa_tap_module is model.decoder
    assert trainer.repa_projector.net[0].in_features == HIDDEN


def test_setup_repa_is_inert_for_sensenova_when_disabled():
    model = _toy()
    trainer = _sensenova_trainer(model, repa_enable=False)

    BaseTrainer._setup_repa(trainer)

    assert trainer.repa_enable is False
    assert trainer._repa_tap_module is None
    assert not hasattr(model.decoder, "_repa_tap_depth")
    assert not hasattr(trainer, "repa_projector")


def test_sensenova_does_not_consume_the_block_loop_features():
    """TREAD / stochastic depth / DiT-BlockSkip have no path in the generation
    decoder loop, so the depth-conflict check stays switched off for it."""
    assert ARCH_REGISTRY["sensenova"].consumes_block_loop_features is False


# ---------------------------------------------------------------------------
# (b) the gradient reaches the model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("checkpoint_layers", [False, True])
def test_the_tap_carries_gradient_to_the_layers_at_and_below_it(checkpoint_layers):
    model = _toy()
    model.train()
    tap_depth = 1
    model.decoder._repa_tap_depth = tap_depth
    projector = repa_module.RepaProjector(HIDDEN, ENC_DIM, hidden=16)

    torch.manual_seed(1)
    hidden = torch.randn(1, TOKENS, HIDDEN, requires_grad=True)
    sensenova_ops.forward_gen_decoder_layers(
        model.decoder, hidden,
        indexes=model._build_t2i_image_indexes(TH, TW, 3, torch.device("cpu")),
        prefix_cache=_Cache(),
        checkpoint_layers=checkpoint_layers,
    )
    tokens = model.decoder._repa_tap_out
    targets = torch.zeros(1, TOKENS, ENC_DIM)
    targets[..., 0] = 1.0
    loss = repa_module.repa_loss(tokens, targets, projector)
    loss.backward()

    def _grad_norm(module):
        return sum(float(p.grad.abs().sum()) for p in module.parameters()
                   if p.grad is not None)

    assert torch.isfinite(loss)
    assert tokens.requires_grad and tokens.grad_fn is not None
    for depth in range(tap_depth + 1):
        assert _grad_norm(model.decoder.layers[depth]) > 0, depth
    for depth in range(tap_depth + 1, LAYERS):
        assert all(p.grad is None
                   for p in model.decoder.layers[depth].parameters()), depth
    assert hidden.grad is not None and float(hidden.grad.abs().sum()) > 0


# ---------------------------------------------------------------------------
# (c) spatial correspondence
# ---------------------------------------------------------------------------

def test_the_tap_is_the_image_tokens_at_the_token_grid():
    """One row per 32px token, not per pixel or per latent cell: the sequence
    the generation half runs on carries no text, so nothing is sliced off."""
    model = _toy()
    trainer = _train_step_trainer(model, 1, [])

    with torch.no_grad():
        _train_step(trainer, repa_pixels=None)

    assert model.decoder._repa_tap_out.shape == (1, TOKENS, HIDDEN)


@pytest.mark.parametrize("cell", [(0, 0), (0, 2), (1, 1), (1, 2)])
def test_the_tokens_are_row_major_over_the_token_grid(cell):
    """Perturb one token's worth of pixels; the token that moves most must be
    the one at h*TW + w -- the index encode_repa_targets puts that cell's
    teacher feature at. Measured through the real patchify and generation ViT."""
    model = _toy()
    trainer = _train_step_trainer(model, 1, [])
    h, w = cell
    base = _images()
    perturbed = base.clone()
    perturbed[:, :, h * PATCH:(h + 1) * PATCH, w * PATCH:(w + 1) * PATCH] += 1.0

    def _tap(images):
        with torch.no_grad():
            _train_step(trainer, images=images, repa_pixels=None)
        return model.decoder._repa_tap_out.clone()

    moved = (_tap(perturbed) - _tap(base)).norm(dim=-1)[0]

    index = h * TW + w
    assert int(moved.argmax()) == index
    assert float(moved[index]) > 2.0 * float(moved.median())


def test_the_index_builder_lays_the_same_grid_out_row_major():
    """The second statement of the order, in the coordinates the decoder's RoPE
    reads: token i is at (i // token_w, i % token_w)."""
    model = _toy()
    indexes = model._build_t2i_image_indexes(TH, TW, 3, torch.device("cpu"))

    assert indexes[1].tolist() == [i // TW for i in range(TOKENS)]
    assert indexes[2].tolist() == [i % TW for i in range(TOKENS)]


# ---------------------------------------------------------------------------
# the per-step path
# ---------------------------------------------------------------------------

def test_train_step_adds_a_finite_alignment_term_that_reaches_the_decoder():
    model = _toy()
    model.train()
    logged = []
    trainer = _train_step_trainer(model, 1, logged)

    loss, value, recon = _train_step(trainer, repa_pixels=_pixels())
    loss.backward()

    assert torch.isfinite(loss)
    assert [name for name, _ in logged] == ["repa_loss"]
    rloss = logged[0][1]
    assert 0.0 <= rloss <= 2.0
    # The alignment term is ON TOP of the diffusion loss the trainer reports.
    assert float(loss) == pytest.approx(value + trainer.repa_weight * rloss, rel=1e-5)
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in trainer.repa_projector.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in model.decoder.layers[0].parameters())
    assert recon >= 0.0


def test_train_step_without_repa_pixels_is_the_plain_diffusion_loss():
    model = _toy()
    model.train()
    logged = []
    trainer = _train_step_trainer(model, 1, logged)

    loss, value, _ = _train_step(trainer, repa_pixels=None)

    assert float(loss) == pytest.approx(value, rel=1e-6)
    assert logged == []


def test_three_steps_run_and_stay_finite():
    model = _toy()
    model.train()
    logged = []
    trainer = _train_step_trainer(model, 1, logged)
    pixels = _pixels()
    opt = torch.optim.SGD(list(model.parameters())
                          + list(trainer.repa_projector.parameters()), lr=1e-4)

    for step in range(3):
        opt.zero_grad(set_to_none=True)
        loss, _, _ = _train_step(trainer, repa_pixels=pixels, seed=step)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), step

    assert [name for name, _ in logged] == ["repa_loss"] * 3
    assert all(0.0 <= value <= 2.0 for _, value in logged)


def test_a_step_with_no_stashed_tap_refuses_instead_of_dropping_the_term():
    model = _toy()
    model.train()
    trainer = _train_step_trainer(model, 1, [])
    del model.decoder._repa_tap_depth  # forward writes nothing

    with pytest.raises(RuntimeError, match="stashed nothing"):
        _train_step(trainer, repa_pixels=_pixels())


def test_a_tap_that_is_not_the_token_grid_refuses_rather_than_misaligning():
    model = _toy()
    model.train()
    trainer = _train_step_trainer(model, 1, [])
    # Stand in for the module the trainer reads, holding a sequence of the wrong
    # length -- what a tap taken on some other stream would give.
    trainer._repa_tap_module = SimpleNamespace(
        _repa_tap_out=torch.randn(1, TOKENS + 3, HIDDEN))

    with pytest.raises(RuntimeError, match="not correspond row for row"):
        _train_step(trainer, repa_pixels=_pixels())


def test_the_arch_handler_passes_the_batch_pixels_through():
    model = _toy()
    trainer = _train_step_trainer(model, 1, [])
    seen = {}

    def _spy(_trainer, **kwargs):
        seen.update(kwargs)
        return torch.zeros(()), 0.0, 0.0

    pixels = _pixels()
    original = sensenova_ops.train_step
    sensenova_ops.train_step = _spy
    try:
        trainer.arch.train_step(trainer, TrainStepContext(
            latents=_images(), sensenova_prefix=_prefix(),
            timesteps=torch.tensor([0.25]), repa_pixels=pixels))
    finally:
        sensenova_ops.train_step = original

    assert seen["repa_pixels"] is pixels


# ---------------------------------------------------------------------------
# (e) the packed batch form
# ---------------------------------------------------------------------------

def test_the_packed_sequence_regroups_item_major():
    """batch > 1 packs every item's tokens along ONE sequence axis, so the tap
    arrives as [1, B*N, D]. Regrouped, item i's rows must be the rows a
    single-item step gives for the same image -- otherwise the teacher features
    of item i would be compared against item j's tokens."""
    model = _toy(mix=False)   # no cross-token mixing: the two forms then agree exactly
    trainer = _train_step_trainer(model, 1, [])
    images = _images(batch=2)
    # t=1 is clean, so z_image is the image itself and the two runs do not have
    # to draw the same noise for a batch of 2 and a batch of 1.
    clean = dict(repa_pixels=None, t=1.0)

    with torch.no_grad():
        _train_step(trainer, images=images, batch=2, **clean)
        packed = model.decoder._repa_tap_out.clone()
        singles = []
        for index in range(2):
            _train_step(trainer, images=images[index:index + 1], **clean)
            singles.append(model.decoder._repa_tap_out.clone())

    assert packed.shape == (1, 2 * TOKENS, HIDDEN)
    regrouped = packed.reshape(2, TOKENS, HIDDEN)
    for index, single in enumerate(singles):
        torch.testing.assert_close(regrouped[index], single[0])


def test_a_packed_step_takes_the_alignment_term_per_item():
    model = _toy(mix=False)
    model.train()
    logged = []
    trainer = _train_step_trainer(model, 1, logged)

    loss, value, _ = _train_step(trainer, repa_pixels=_pixels(batch=2), batch=2)
    loss.backward()

    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(value + trainer.repa_weight * logged[0][1],
                                        rel=1e-5)
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in model.decoder.layers[0].parameters())


def test_a_packed_tap_with_the_wrong_item_count_refuses():
    model = _toy(mix=False)
    model.train()
    trainer = _train_step_trainer(model, 1, [])
    trainer._repa_tap_module = SimpleNamespace(
        _repa_tap_out=torch.randn(1, 3 * TOKENS, HIDDEN))

    with pytest.raises(RuntimeError, match="packed"):
        _train_step(trainer, repa_pixels=_pixels(batch=2), batch=2)


# ---------------------------------------------------------------------------
# (d) the full-FT save
# ---------------------------------------------------------------------------

_FULL_FT_LAYERS = 42
_FULL_FT_IN, _FULL_FT_OUT = 8, 4


def _full_ft_tree():
    """The 42-layer MoT attribute layout ``iter_sensenova_lora_targets`` walks,
    with the generation half materialized as a full fine-tune leaves it."""
    from core.models.ideogram4.vendor.int8_linear import (
        Int8Linear, quantize_weight_to_int8,
    )
    from core.models.sensenova.loader import materialize_int8_decoder_linears

    def _quant(seed: int) -> Int8Linear:
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(_FULL_FT_OUT, _FULL_FT_IN, generator=generator,
                             dtype=torch.float32)
        codes, scale = quantize_weight_to_int8(weight)
        module = Int8Linear(_FULL_FT_IN, _FULL_FT_OUT, False, torch.bfloat16)
        module.weight.copy_(codes)
        module.weight_scale.copy_(scale)
        return module

    transformer = nn.Module()
    blocks = []
    seed = 0
    for _ in range(_FULL_FT_LAYERS):
        block = nn.Module()
        attn = nn.Module()
        mlp, mlp_gen = nn.Module(), nn.Module()
        for stem in ("q_proj", "k_proj", "v_proj", "o_proj"):
            for name in (stem, f"{stem}_mot_gen"):
                setattr(attn, name, _quant(seed))
                seed += 1
        for stem in ("gate_proj", "up_proj", "down_proj"):
            for parent in (mlp, mlp_gen):
                setattr(parent, stem, _quant(seed))
                seed += 1
        block.self_attn = attn
        block.mlp = mlp
        block.mlp_mot_gen = mlp_gen
        blocks.append(block)
    core = nn.Module()
    core.layers = nn.ModuleList(blocks)
    language_model = nn.Module()
    language_model.model = core
    transformer.language_model = language_model
    materialize_int8_decoder_linears(transformer, branch="gen")
    return transformer


def test_a_full_finetune_save_pairs_the_projector_with_the_file_it_wrote(tmp_path):
    """The base adapter needs the RESOLVED path back; handed a suffixless stem,
    the sidecar has to land on the file the save actually wrote."""
    from core.training.adapters.sensenova_adapter import SenseNovaFullParameterAdapter

    trainer = SimpleNamespace(
        transformer=_full_ft_tree(),
        train_unet=True,
        train_text_encoder=False,
        config={},
        sensenova_full_finetune_save_format="mixed",
        sensenova_model_config=None,
        model_path=None,
        log_prefix="[test]",
        repa_enable=True,
        repa_projector=repa_module.RepaProjector(HIDDEN, ENC_DIM, hidden=16),
    )

    written = SenseNovaFullParameterAdapter(trainer).save_checkpoint(
        100, 1, tmp_path / "run_step_000100")

    assert written == str(tmp_path / "run_step_000100.safetensors")
    assert (tmp_path / "run_step_000100.repa.safetensors").is_file()


def test_a_sharded_save_puts_the_projector_where_the_resume_will_look(tmp_path, monkeypatch):
    """The real save is always multi-shard at 16B parameters, so it returns the
    index path. The sidecar has to sit beside THAT, and _setup_repa's gate has
    to accept it -- otherwise a resume drops a trained projector in silence.
    """
    from core.training.adapters.sensenova_adapter import SenseNovaFullParameterAdapter
    from core.models.common import quantized_export

    monkeypatch.setattr(quantized_export, "DEFAULT_EXPORT_SHARD_BYTES", 4096)

    trainer = SimpleNamespace(
        transformer=_full_ft_tree(),
        train_unet=True,
        train_text_encoder=False,
        config={},
        sensenova_full_finetune_save_format="mixed",
        sensenova_model_config=None,
        model_path=None,
        log_prefix="[test]",
        repa_enable=True,
        repa_projector=repa_module.RepaProjector(HIDDEN, ENC_DIM, hidden=16),
    )

    written = SenseNovaFullParameterAdapter(trainer).save_checkpoint(
        100, 1, tmp_path / "run_step_000100")

    assert written.endswith(".safetensors.index.json"), written
    sidecar = repa_module.repa_sidecar_path(written)
    assert sidecar == str(tmp_path / "run_step_000100.repa.safetensors")
    assert Path(sidecar).is_file()
    assert written.endswith(repa_module.CHECKPOINT_SUFFIXES)


# ---------------------------------------------------------------------------
# the capability table
# ---------------------------------------------------------------------------

def test_sensenova_is_offered_the_repa_control():
    assert "repa" not in TRAINING_FEATURE_UNSUPPORTED.get("sensenova", {})
