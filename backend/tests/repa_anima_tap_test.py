"""Anima's REPA tap: the second architecture wired onto the shared foundation.

The tap is an explicit assignment inside Anima's own block loop, not a hook: the
blocks run under ``torch.utils.checkpoint``, and a forward hook there would hand
REPA a detached tensor and a gradient of exactly zero, with a loss value that
still looks alive.

What is pinned here:
  (a) with the tap unarmed the forward is bit-identical and stashes nothing;
  (b) armed, the gradient reaches the blocks at and below the tap;
  (c) the tap's tokens correspond row-major to the image grid the frozen teacher
      is resampled onto;
  (d) each block-loop feature that would void the tap is refused at setup, and
      the one path that CAN still write it (DiT-BlockSkip's middle span) does.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/repa_anima_tap_test.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.arch_capabilities import TRAINING_FEATURE_UNSUPPORTED
from core.models.anima.anima_models import Anima
from core.training import repa as repa_module
from core.training.arch import ARCH_REGISTRY
from core.training.arch.base_arch import ArchHandler, TrainStepContext
from core.training.base_trainer import BaseTrainer
from core.training.ops import anima_ops

BLOCKS = 6
CHANNELS = 64
LATENT_C = 4
LATENT_H = 8
LATENT_W = 12
ENC_DIM = 8
ENC_GRID = 4
TEXT_LEN = 5
TEXT_DIM = 16


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubLLMAdapter(nn.Module):
    """Anima._preprocess_text_embeds calls this; nothing here reads it."""

    def forward(self, source_hidden_states, target_input_ids,
                target_attention_mask=None, source_attention_mask=None):
        return source_hidden_states


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


def _tiny_anima(num_blocks: int = BLOCKS) -> Anima:
    torch.manual_seed(0)
    model = Anima(
        max_img_h=64, max_img_w=64, max_frames=8,
        in_channels=LATENT_C, out_channels=LATENT_C,
        patch_spatial=2, patch_temporal=1, model_channels=CHANNELS,
        num_blocks=num_blocks, num_heads=4, crossattn_emb_channels=TEXT_DIM,
        use_llm_adapter=False, concat_padding_mask=True,
    )
    model.llm_adapter = _StubLLMAdapter()
    return model


def _inputs(batch: int = 1):
    torch.manual_seed(1)
    return dict(
        x=torch.randn(batch, LATENT_C, 1, LATENT_H, LATENT_W),
        t=torch.rand(batch),
        ctx=torch.randn(batch, TEXT_LEN, TEXT_DIM),
        pad=torch.zeros(batch, 1, LATENT_H, LATENT_W),
    )


def _forward(model, inp):
    return model.forward_mini_train_dit(
        inp["x"], inp["t"], inp["ctx"], padding_mask=inp["pad"])


def _anima_trainer(model=None, **config):
    cfg = {"repa_enable": True, "repa_tagger_model_dir": "unused-because-stubbed"}
    cfg.update(config)
    return SimpleNamespace(
        config=cfg,
        arch=ARCH_REGISTRY["anima"](),
        transformer=model if model is not None else _tiny_anima(),
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        model_path="",
        log_prefix="[test]",
        tread_config=None,
        block_skip_config=None,
        blockskip_config=None,
    )


def _stub_encoder_loader(monkeypatch):
    monkeypatch.setattr(repa_module, "load_repa_encoder",
                        lambda source, **kwargs: (_StubEncoder(), ENC_DIM, 32))


def _train_step_trainer(model, align_depth, logged):
    trainer = _anima_trainer(model)
    trainer.repa_enable = True  # _setup_repa's own flag, not the config key
    trainer.mixed_precision = False
    trainer.timestep_sampler = None
    trainer.reconstruction_loss_weight = 0.0
    trainer.repa_encoder = _StubEncoder()
    trainer.repa_projector = repa_module.RepaProjector(CHANNELS, ENC_DIM, hidden=16)
    trainer.repa_size = 32
    trainer.repa_weight = 1.0
    trainer.repa_align_depth = align_depth
    trainer._repa_tap_module = model
    trainer._ensure_repa_on_device = lambda: None
    trainer.log_extra_metric = lambda name, value: logged.append((name, value))
    model._repa_tap_depth = align_depth
    return trainer


def _train_step(trainer, model, *, repa_pixels, seed=7):
    torch.manual_seed(seed)
    inp = _inputs()
    aux = {
        "source_mask": torch.ones(1, TEXT_LEN, dtype=torch.bool),
        "t5_input_ids": torch.zeros(1, TEXT_LEN, dtype=torch.long),
        "t5_attn_mask": torch.ones(1, TEXT_LEN, dtype=torch.bool),
    }
    return anima_ops.train_step(
        trainer,
        latents=inp["x"].squeeze(2),
        prompt_embeds=inp["ctx"],
        anima_aux=aux,
        timesteps=inp["t"],
        repa_pixels=repa_pixels,
    )


# ---------------------------------------------------------------------------
# (a) the disabled path
# ---------------------------------------------------------------------------

def test_a_fresh_anima_carries_the_tap_attributes_unarmed():
    """Inference loads the same class; the attributes must exist and be inert."""
    model = _tiny_anima()
    assert model._repa_tap_depth is None
    assert model._repa_tap_out is None


def test_arming_the_tap_does_not_change_the_forward():
    model = _tiny_anima()
    model.eval()
    inp = _inputs()

    disabled = _forward(model, inp)
    assert model._repa_tap_out is None

    model._repa_tap_depth = 2
    armed = _forward(model, inp)

    assert torch.equal(disabled, armed)
    assert model._repa_tap_out is not None


def test_a_forward_clears_what_a_previous_one_stashed():
    """Otherwise a step whose tap does not fire would read stale tokens."""
    model = _tiny_anima()
    model.eval()
    inp = _inputs()
    model._repa_tap_depth = 2
    _forward(model, inp)
    stale = model._repa_tap_out

    assert stale is not None

    # A depth no block carries: the tap cannot fire, so only the clear can
    # decide what is left behind. Asserting a *different* tensor instead would
    # pass with the clear deleted, because a firing tap overwrites anyway.
    model._repa_tap_depth = len(model.blocks) + 5
    _forward(model, inp)

    assert model._repa_tap_out is None


# ---------------------------------------------------------------------------
# the handler's answer
# ---------------------------------------------------------------------------

def test_the_handler_answers_with_the_dit_its_width_and_its_block_count():
    model = _tiny_anima()
    trainer = _anima_trainer(model)

    tap = trainer.arch.repa_tap(trainer)

    assert tap.module is model
    assert tap.hidden_size == CHANNELS == model.model_channels
    assert tap.depth == BLOCKS == len(model.blocks)


def test_the_handler_unwraps_the_training_wrapper():
    """Same `inner` the TREAD/BlockSkip arming uses, so both write one module."""
    model = _tiny_anima()
    trainer = _anima_trainer(SimpleNamespace(module=model))

    assert trainer.arch.repa_tap(trainer).module is model


def test_setup_repa_arms_the_anima_dit(monkeypatch):
    _stub_encoder_loader(monkeypatch)
    model = _tiny_anima()
    trainer = _anima_trainer(model)

    BaseTrainer._setup_repa(trainer)

    assert trainer.repa_align_depth == BLOCKS // 3
    assert model._repa_tap_depth == BLOCKS // 3
    assert trainer._repa_tap_module is model
    assert trainer.repa_projector.net[0].in_features == CHANNELS


# ---------------------------------------------------------------------------
# (b) the gradient reaches the model
# ---------------------------------------------------------------------------

def test_the_tap_carries_gradient_to_the_blocks_at_and_below_it():
    model = _tiny_anima()
    model.train()
    model.enable_gradient_checkpointing()  # the case a forward hook would break
    tap_depth = 2
    model._repa_tap_depth = tap_depth
    projector = repa_module.RepaProjector(CHANNELS, ENC_DIM, hidden=16)

    _forward(model, _inputs())
    tap = model._repa_tap_out
    tokens = tap.flatten(1, 3)
    targets = torch.zeros(tokens.shape[0], tokens.shape[1], ENC_DIM)
    targets[..., 0] = 1.0
    loss = repa_module.repa_loss(tokens, targets, projector)
    loss.backward()

    def _grad_norm(module):
        return sum(float(p.grad.abs().sum()) for p in module.parameters()
                   if p.grad is not None)

    assert torch.isfinite(loss)
    for depth in range(tap_depth + 1):
        assert _grad_norm(model.blocks[depth]) > 0, depth
    for depth in range(tap_depth + 1, BLOCKS):
        assert all(p.grad is None for p in model.blocks[depth].parameters()), depth
    assert _grad_norm(model.x_embedder) > 0


# ---------------------------------------------------------------------------
# (c) spatial correspondence
# ---------------------------------------------------------------------------

def test_the_tap_grid_is_the_latent_grid_over_the_patch_size():
    model = _tiny_anima()
    model.eval()
    model._repa_tap_depth = 1

    _forward(model, _inputs())

    _, frames, gh, gw, dim = model._repa_tap_out.shape
    assert frames == 1                       # image training: one temporal step
    assert (gh, gw) == (LATENT_H // 2, LATENT_W // 2)
    assert dim == CHANNELS


@pytest.mark.parametrize("cell", [(0, 0), (1, 3), (3, 5), (2, 2)])
def test_the_tokens_are_row_major_over_that_grid(cell):
    """Perturb one patch of the input; the token that moves most must be the one
    at h*gw + w -- the index encode_repa_targets puts that patch's teacher
    feature at."""
    model = _tiny_anima()
    model.eval()
    model._repa_tap_depth = 1
    h, w = cell
    gw = LATENT_W // 2
    inp = _inputs()

    _forward(model, inp)
    before = model._repa_tap_out.clone()
    perturbed = dict(inp)
    perturbed["x"] = inp["x"].clone()
    perturbed["x"][:, :, :, h * 2:(h + 1) * 2, w * 2:(w + 1) * 2] += 5.0
    _forward(model, perturbed)
    after = model._repa_tap_out

    moved = (after - before).flatten(1, 3).norm(dim=-1)[0]
    assert int(moved.argmax()) == h * gw + w


# ---------------------------------------------------------------------------
# the per-step path
# ---------------------------------------------------------------------------

def test_train_step_adds_a_finite_alignment_term_that_reaches_the_dit():
    model = _tiny_anima()
    model.train()
    logged = []
    trainer = _train_step_trainer(model, 2, logged)
    pixels = torch.rand(1, 3, 32, 32) * 2 - 1

    loss, pred, _ = _train_step(trainer, model, repa_pixels=pixels)
    loss.backward()

    assert torch.isfinite(loss)
    assert [name for name, _ in logged] == ["repa_loss"]
    rloss = logged[0][1]
    assert 0.0 <= rloss <= 2.0
    # The alignment term is ON TOP of the diffusion loss the trainer reports.
    assert float(loss) == pytest.approx(pred + trainer.repa_weight * rloss, rel=1e-5)
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in trainer.repa_projector.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in model.blocks[0].parameters())


def test_three_steps_run_and_stay_finite():
    model = _tiny_anima()
    model.train()
    logged = []
    trainer = _train_step_trainer(model, 2, logged)
    pixels = torch.rand(1, 3, 32, 32) * 2 - 1
    opt = torch.optim.SGD(list(model.parameters())
                          + list(trainer.repa_projector.parameters()), lr=1e-4)

    for step in range(3):
        opt.zero_grad(set_to_none=True)
        loss, _, _ = _train_step(trainer, model, repa_pixels=pixels, seed=step)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), step

    assert [name for name, _ in logged] == ["repa_loss"] * 3
    assert all(0.0 <= value <= 2.0 for _, value in logged)


def test_a_step_with_no_stashed_tap_refuses_instead_of_dropping_the_term():
    model = _tiny_anima()
    model.train()
    trainer = _train_step_trainer(model, 2, [])
    model._repa_tap_depth = None  # forward writes nothing

    with pytest.raises(RuntimeError, match="stashed nothing"):
        _train_step(trainer, model, repa_pixels=torch.rand(1, 3, 32, 32))


def test_the_arch_handler_passes_the_batch_pixels_through():
    model = _tiny_anima()
    model.train()
    trainer = _train_step_trainer(model, 2, [])
    seen = {}

    def _spy(_trainer, **kwargs):
        seen.update(kwargs)
        return torch.zeros(()), 0.0, 0.0

    pixels = torch.rand(1, 3, 32, 32)
    original = anima_ops.train_step
    anima_ops.train_step = _spy
    try:
        trainer.arch.train_step(trainer, TrainStepContext(
            latents=torch.zeros(1, LATENT_C, LATENT_H, LATENT_W),
            repa_pixels=pixels))
    finally:
        anima_ops.train_step = original

    assert seen["repa_pixels"] is pixels


# ---------------------------------------------------------------------------
# (d) the block-loop features that would void the tap
# ---------------------------------------------------------------------------

def test_tread_refuses_the_default_tap(monkeypatch):
    """Shipped defaults on both sides: the auto tap (depth//3) of Anima's 28
    blocks is 9, inside the routed span [2, 26)."""
    _stub_encoder_loader(monkeypatch)
    trainer = _anima_trainer(_tiny_anima(28))
    trainer.tread_config = {"start_block": 2, "end_block": 26, "drop_ratio": 0.5}

    with pytest.raises(ValueError, match="TREAD routed span"):
        BaseTrainer._setup_repa(trainer)


def test_dit_blockskip_refuses_a_tap_in_a_skipped_span(monkeypatch):
    _stub_encoder_loader(monkeypatch)
    trainer = _anima_trainer(_tiny_anima(28), repa_align_depth=1)
    trainer.blockskip_config = {"front": 4, "back": 4}

    with pytest.raises(ValueError, match="BlockSkip"):
        BaseTrainer._setup_repa(trainer)


def test_dit_blockskip_accepts_a_tap_in_the_span_that_trains(monkeypatch):
    _stub_encoder_loader(monkeypatch)
    model = _tiny_anima(28)
    trainer = _anima_trainer(model)
    trainer.blockskip_config = {"front": 4, "back": 4}

    BaseTrainer._setup_repa(trainer)

    assert model._repa_tap_depth == 28 // 3


def test_stochastic_depth_refuses_a_tap_it_may_drop(monkeypatch):
    """A dropped block is identity and writes no tap, so REPA would contribute
    nothing on those steps -- silently, with the run still training."""
    _stub_encoder_loader(monkeypatch)
    trainer = _anima_trainer(_tiny_anima(28), repa_align_depth=2)
    trainer.block_skip_config = {"skip_rate": 0.1, "protect_start": 6,
                                 "protect_end": 22}

    with pytest.raises(ValueError, match="stochastic depth may drop"):
        BaseTrainer._setup_repa(trainer)


def test_stochastic_depth_accepts_a_tap_inside_the_protected_span(monkeypatch):
    _stub_encoder_loader(monkeypatch)
    model = _tiny_anima(28)
    trainer = _anima_trainer(model)
    trainer.block_skip_config = {"skip_rate": 0.1, "protect_start": 6,
                                 "protect_end": 22}

    BaseTrainer._setup_repa(trainer)

    assert model._repa_tap_depth == 28 // 3  # 9, inside [6, 22)


def test_a_dropped_block_really_would_not_write_the_tap(monkeypatch):
    """What the refusal above is protecting against, exercised directly. The
    drop is forced rather than sampled: the point is the consequence, not the
    Bernoulli draw."""
    from core.training import block_dropout

    tap_depth = 2
    monkeypatch.setattr(
        block_dropout, "compute_skip_mask",
        lambda n, *a, **k: ([i == tap_depth for i in range(n)], list(range(n))))
    model = _tiny_anima()
    model.train()
    model._repa_tap_depth = tap_depth
    model._block_skip_config = {"skip_rate": 0.1, "protect_start": 0,
                                "protect_end": 0}

    _forward(model, _inputs())

    assert model._repa_tap_out is None


def test_the_blockskip_middle_span_still_writes_the_tap():
    """DiT-BlockSkip runs its own two-pass forward. The tap has to fire on the
    GRADIENT pass; firing on pass 1 (no_grad) would zero the REPA gradient, and
    not firing at all would drop the term."""
    model = _tiny_anima()
    model.train()
    model._repa_tap_depth = 2
    model._blockskip_config = {"front": 1, "back": 1,
                               "on_residual": lambda a, b: (a.detach(), b.detach())}

    _forward(model, _inputs())

    tap = model._repa_tap_out
    assert tap is not None
    assert tap.requires_grad and tap.grad_fn is not None


def test_a_full_finetune_save_pairs_the_projector_with_the_file_it_wrote(tmp_path):
    """The base adapter needs the RESOLVED path back; handed a directory, the
    sidecar has to land on the checkpoint written inside it."""
    from core.training.adapters.anima_adapter import AnimaFullParameterAdapter

    model = _tiny_anima(1)
    trainer = _anima_trainer(model)
    trainer.repa_enable = True
    trainer.repa_projector = repa_module.RepaProjector(CHANNELS, ENC_DIM, hidden=16)
    trainer.vae = None
    trainer.bundle_vae = False

    written = AnimaFullParameterAdapter(trainer).save_checkpoint(5, 0, tmp_path)

    assert written == tmp_path / "anima_step_5.safetensors"
    assert (tmp_path / "anima_step_5.repa.safetensors").is_file()


# ---------------------------------------------------------------------------
# the capability table and the handlers say the same thing
# ---------------------------------------------------------------------------

_WIRED = {"minit2i", "anima", "lens", "krea2", "ideogram4", "sensenova",
          "sd15", "sdxl"}


@pytest.mark.parametrize("arch", sorted(ARCH_REGISTRY))
def test_the_capability_table_matches_the_handlers(arch):
    handler = ARCH_REGISTRY[arch]
    declines = handler.repa_tap is ArchHandler.repa_tap
    refused = "repa" in TRAINING_FEATURE_UNSUPPORTED.get(arch, {})
    assert declines == refused, (
        f"{arch}: repa_tap {'declines' if declines else 'is implemented'} but "
        f"the capability table {'refuses' if refused else 'offers'} repa")
    assert declines == (arch not in _WIRED)
