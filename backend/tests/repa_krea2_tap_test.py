"""Krea 2's REPA tap: a single-stream DiT whose sequence carries a text prefix.

The tap is an explicit assignment inside the vendored block loop rather than a
forward hook, so the tap IS the tensor the loss differentiates and there is no
hook ordering or checkpoint recompute behaviour to reason about.

What is pinned here:
  (a) with the tap unarmed the forward is bit-identical -- both against the
      armed/unarmed pair and against the pre-tap revision of the vendored
      module, at a geometry that is not the shipped one;
  (b) armed, the gradient reaches the blocks at and below the tap;
  (c) the tap's tokens are row-major over the latent grid, in the order
      ``krea2_pipeline_ops.pack_latents`` packs the sequence and
      ``encode_repa_targets`` builds the teacher grid;
  (d) a full fine-tune pairs the projector sidecar with the file it wrote;
  (e) the text prefix is sliced off at the boundary the post-loop line uses, so
      the tap length is the image grid whatever the prompt length.

Everything runs on a randomly initialised toy-geometry transformer; no
checkpoint is read and no GPU is used.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/repa_krea2_tap_test.py -v
"""

from __future__ import annotations

import importlib.util
import subprocess
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
from core.models.krea2.krea2_pipeline_ops import (pack_latents,  # noqa: E402
                                                  prepare_position_ids)
from core.models.krea2.vendor.transformer import Krea2Transformer2DModel  # noqa: E402
from core.training import repa as repa_module  # noqa: E402
from core.training.arch import ARCH_REGISTRY  # noqa: E402
from core.training.arch.base_arch import TrainStepContext  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.ops import krea2_ops  # noqa: E402

VENDOR_REL = "backend/core/models/krea2/vendor/transformer.py"

BLOCKS = 4
HEADS = 3
HEAD_DIM = 8
HIDDEN = HEADS * HEAD_DIM     # 24; the shipped model is 48 * 128
IN_CHANNELS = 8               # C * patch**2, so raw latent C = 2
LATENT_C = IN_CHANNELS // 4
GH, GW = 2, 3
TEXT_LEN = 5
TEXT_DIM = 6
TEXT_LAYERS = 2
ENC_DIM = 8
ENC_GRID = 4


# ---------------------------------------------------------------------------
# Stubs and fixtures
# ---------------------------------------------------------------------------

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


def _tiny_krea2(cls=Krea2Transformer2DModel, num_layers: int = BLOCKS):
    torch.manual_seed(0)
    return cls(
        in_channels=IN_CHANNELS, num_layers=num_layers, attention_head_dim=HEAD_DIM,
        num_attention_heads=HEADS, num_key_value_heads=HEADS, intermediate_size=16,
        timestep_embed_dim=8, text_hidden_dim=TEXT_DIM, num_text_layers=TEXT_LAYERS,
        text_num_attention_heads=2, text_num_key_value_heads=2,
        text_intermediate_size=8, num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1, axes_dims_rope=(2, 2, 4), rope_theta=1000.0,
    )


def _inputs(batch: int = 1, seed: int = 1, text_len: int = TEXT_LEN):
    torch.manual_seed(seed)
    return dict(
        hidden_states=torch.randn(batch, GH * GW, IN_CHANNELS),
        encoder_hidden_states=torch.randn(batch, text_len, TEXT_LAYERS, TEXT_DIM),
        timestep=torch.full((batch,), 0.5),
        position_ids=prepare_position_ids(text_len, GH, GW, torch.device("cpu")),
        encoder_attention_mask=torch.ones(batch, text_len, dtype=torch.bool),
    )


def _forward(model, inp):
    return model(return_dict=False, **inp)[0]


def _krea2_trainer(model=None, **config):
    cfg = {"repa_enable": True, "repa_tagger_model_dir": "unused-because-stubbed"}
    cfg.update(config)
    return SimpleNamespace(
        config=cfg,
        arch=ARCH_REGISTRY["krea2"](),
        transformer=model if model is not None else _tiny_krea2(),
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        model_path="",
        log_prefix="[test]",
    )


def _stub_encoder_loader(monkeypatch):
    monkeypatch.setattr(repa_module, "load_repa_encoder",
                        lambda source, **kwargs: (_StubEncoder(), ENC_DIM, 32))


def _train_step_trainer(model, align_depth, logged):
    trainer = _krea2_trainer(model)
    trainer.repa_enable = True  # _setup_repa's own flag, not the config key
    trainer.mixed_precision = False
    trainer.timestep_sampler = None
    trainer.krea2_discrete_flow_shift = 2.5
    trainer.crop_decode_loss_enable = False
    trainer.repa_encoder = _StubEncoder()
    trainer.repa_projector = repa_module.RepaProjector(HIDDEN, ENC_DIM, hidden=16)
    trainer.repa_size = 32
    trainer.repa_weight = 1.0
    trainer.repa_align_depth = align_depth
    trainer._repa_tap_module = model
    trainer._ensure_repa_on_device = lambda: None
    trainer.log_extra_metric = lambda name, value: logged.append((name, value))
    model._repa_tap_depth = align_depth
    return trainer


def _train_step(trainer, *, repa_pixels, seed=7):
    inp = _inputs(seed=seed)
    torch.manual_seed(seed)
    return krea2_ops.train_step(
        trainer,
        latents=inp["hidden_states"],
        encoder_features=inp["encoder_hidden_states"],
        encoder_mask=inp["encoder_attention_mask"],
        timesteps=torch.rand(1),
        latent_h=GH,
        latent_w=GW,
        repa_pixels=repa_pixels,
    )


# ---------------------------------------------------------------------------
# (a) the disabled path
# ---------------------------------------------------------------------------

def test_a_fresh_krea2_carries_the_tap_attributes_unarmed():
    """Inference loads the same class; the attributes must exist and be inert."""
    model = _tiny_krea2()
    assert model._repa_tap_depth is None
    assert model._repa_tap_out is None


def test_arming_the_tap_does_not_change_the_forward():
    model = _tiny_krea2().eval()
    inp = _inputs()

    with torch.no_grad():
        disabled = _forward(model, inp)
        assert model._repa_tap_out is None
        model._repa_tap_depth = 2
        armed = _forward(model, inp)

    assert torch.equal(disabled, armed)
    assert model._repa_tap_out is not None


def _pre_tap_vendor_module():
    """The newest revision of the vendored transformer without the tap, loaded as
    its own module. Returns None when git cannot supply one."""
    repo = str(BACKEND.parent)
    try:
        revs = subprocess.run(["git", "rev-list", "-n", "200", "HEAD", "--", VENDOR_REL],
                              cwd=repo, capture_output=True, check=True).stdout.split()
    except Exception:
        return None
    for rev in revs:
        try:
            src = subprocess.run(["git", "show", f"{rev.decode()}:{VENDOR_REL}"],
                                 cwd=repo, capture_output=True, check=True).stdout
        except Exception:
            continue
        if b"_repa_tap_depth" in src:
            continue
        path = Path(repo) / "backend" / "tests" / "__pycache__" / "krea2_pre_tap_vendor.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(src)
        spec = importlib.util.spec_from_file_location("krea2_pre_tap_vendor", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["krea2_pre_tap_vendor"] = module
        spec.loader.exec_module(module)
        return module
    return None


def test_the_disabled_forward_matches_the_revision_before_the_tap():
    """Not the same as arming/unarming one module: this runs the pre-tap SOURCE.
    The geometry is deliberately not the shipped one, so a change that only holds
    for 28 blocks of 6144 would show up here."""
    module = _pre_tap_vendor_module()
    if module is None:
        pytest.skip("no pre-tap revision of the vendored transformer in git history")

    before = _tiny_krea2(module.Krea2Transformer2DModel).eval()
    after = _tiny_krea2().eval()
    inp = _inputs()

    with torch.no_grad():
        assert torch.equal(_forward(before, inp), _forward(after, inp))
    assert after._repa_tap_out is None


def test_a_forward_clears_what_a_previous_one_stashed():
    """Otherwise a step whose tap does not fire would read stale tokens."""
    model = _tiny_krea2().eval()
    inp = _inputs()
    model._repa_tap_depth = 2
    with torch.no_grad():
        _forward(model, inp)
        assert model._repa_tap_out is not None

        # A depth no block carries: the tap cannot fire, so only the clear can
        # decide what is left behind.
        model._repa_tap_depth = len(model.transformer_blocks) + 5
        _forward(model, inp)

    assert model._repa_tap_out is None


# ---------------------------------------------------------------------------
# the handler's answer
# ---------------------------------------------------------------------------

def test_the_handler_answers_with_the_dit_its_width_and_its_block_count():
    model = _tiny_krea2()
    trainer = _krea2_trainer(model)

    tap = trainer.arch.repa_tap(trainer)

    assert tap.module is model
    assert tap.hidden_size == HIDDEN == model.hidden_size
    assert tap.depth == BLOCKS == len(model.transformer_blocks)


def test_the_width_is_the_attribute_because_the_config_has_no_such_key():
    """``hidden_size`` is derived in ``__init__`` from attention_head_dim x
    num_attention_heads and never registered, so the config cannot answer."""
    model = _tiny_krea2()

    assert "hidden_size" not in dict(model.config)
    assert model.hidden_size == HIDDEN


def test_setup_repa_arms_the_krea2_dit(monkeypatch):
    _stub_encoder_loader(monkeypatch)
    model = _tiny_krea2()
    trainer = _krea2_trainer(model)

    BaseTrainer._setup_repa(trainer)

    assert trainer.repa_align_depth == BLOCKS // 3
    assert model._repa_tap_depth == BLOCKS // 3
    assert trainer._repa_tap_module is model
    assert trainer.repa_projector.net[0].in_features == HIDDEN


def test_setup_repa_is_inert_for_krea2_when_disabled():
    model = _tiny_krea2()
    trainer = _krea2_trainer(model, repa_enable=False)

    BaseTrainer._setup_repa(trainer)

    assert trainer.repa_enable is False
    assert trainer._repa_tap_module is None
    assert model._repa_tap_depth is None
    assert not hasattr(trainer, "repa_projector")


def test_krea2_does_not_consume_the_block_loop_features():
    """TREAD / stochastic depth / DiT-BlockSkip have no path in Krea 2's forward,
    so the depth-conflict check must stay switched off for it."""
    assert ARCH_REGISTRY["krea2"].consumes_block_loop_features is False


# ---------------------------------------------------------------------------
# (b) the gradient reaches the model
# ---------------------------------------------------------------------------

def test_the_tap_carries_gradient_to_the_blocks_at_and_below_it():
    model = _tiny_krea2()
    model.train()
    model.enable_gradient_checkpointing()  # the case a forward hook would break
    tap_depth = 1
    model._repa_tap_depth = tap_depth
    projector = repa_module.RepaProjector(HIDDEN, ENC_DIM, hidden=16)

    _forward(model, _inputs())
    tokens = model._repa_tap_out
    targets = torch.zeros(tokens.shape[0], tokens.shape[1], ENC_DIM)
    targets[..., 0] = 1.0
    loss = repa_module.repa_loss(tokens, targets, projector)
    loss.backward()

    def _grad_norm(module):
        return sum(float(p.grad.abs().sum()) for p in module.parameters()
                   if p.grad is not None)

    assert torch.isfinite(loss)
    assert tokens.requires_grad and tokens.grad_fn is not None
    for depth in range(tap_depth + 1):
        assert _grad_norm(model.transformer_blocks[depth]) > 0, depth
    for depth in range(tap_depth + 1, BLOCKS):
        assert all(p.grad is None
                   for p in model.transformer_blocks[depth].parameters()), depth
    assert _grad_norm(model.img_in) > 0


# ---------------------------------------------------------------------------
# (c) spatial correspondence and (e) the text prefix
# ---------------------------------------------------------------------------

def test_the_tap_is_the_image_tokens_at_the_latent_grid():
    model = _tiny_krea2().eval()
    model._repa_tap_depth = 1

    with torch.no_grad():
        _forward(model, _inputs())

    assert model._repa_tap_out.shape == (1, GH * GW, HIDDEN)


@pytest.mark.parametrize("text_len", [1, TEXT_LEN, TEXT_LEN + 7])
def test_the_text_prefix_is_sliced_off_whatever_its_length(text_len):
    """The tap slices at the same boundary the post-loop line does. A tap taken
    before the slice would grow with the prompt and misalign every token."""
    model = _tiny_krea2().eval()
    model._repa_tap_depth = 1

    with torch.no_grad():
        _forward(model, _inputs(text_len=text_len))

    assert model._repa_tap_out.shape[1] == GH * GW


@pytest.mark.parametrize("cell", [(0, 0), (0, 2), (1, 1), (1, 2)])
def test_the_tokens_are_row_major_over_the_latent_grid(cell):
    """Perturb one latent cell; the token that moves most must be the one at
    h*GW + w -- the index encode_repa_targets puts that cell's teacher feature
    at. The sequence is built with pack_latents, the packer the trainer uses."""
    model = _tiny_krea2().eval()
    model._repa_tap_depth = 1
    h, w = cell
    inp = _inputs()

    torch.manual_seed(5)
    spatial = torch.randn(1, LATENT_C, GH * 2, GW * 2)
    perturbed = spatial.clone()
    perturbed[:, :, h * 2:(h + 1) * 2, w * 2:(w + 1) * 2] += 5.0

    def _tap(x):
        with torch.no_grad():
            _forward(model, dict(inp, hidden_states=pack_latents(x, patch_size=2)))
        return model._repa_tap_out.clone()

    moved = (_tap(perturbed) - _tap(spatial)).norm(dim=-1)[0]

    index = h * GW + w
    assert int(moved.argmax()) == index
    assert float(moved[index]) > 2.0 * float(moved.median())


# ---------------------------------------------------------------------------
# the per-step path
# ---------------------------------------------------------------------------

def test_train_step_adds_a_finite_alignment_term_that_reaches_the_dit():
    model = _tiny_krea2()
    model.train()
    logged = []
    trainer = _train_step_trainer(model, 1, logged)
    pixels = torch.rand(1, 3, 32, 32) * 2 - 1

    loss, pred, _ = _train_step(trainer, repa_pixels=pixels)
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
               for p in model.transformer_blocks[0].parameters())


def test_train_step_without_repa_pixels_is_the_plain_diffusion_loss():
    model = _tiny_krea2()
    model.train()
    logged = []
    trainer = _train_step_trainer(model, 1, logged)

    loss, pred, _ = _train_step(trainer, repa_pixels=None)

    assert float(loss) == pytest.approx(pred, rel=1e-6)
    assert logged == []


def test_three_steps_run_and_stay_finite():
    model = _tiny_krea2()
    model.train()
    logged = []
    trainer = _train_step_trainer(model, 1, logged)
    pixels = torch.rand(1, 3, 32, 32) * 2 - 1
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
    model = _tiny_krea2()
    model.train()
    trainer = _train_step_trainer(model, 1, [])
    model._repa_tap_depth = None  # forward writes nothing

    with pytest.raises(RuntimeError, match="stashed nothing"):
        _train_step(trainer, repa_pixels=torch.rand(1, 3, 32, 32))


def test_a_tap_that_is_not_the_grid_refuses_rather_than_misaligning():
    """The length guard is what a wrong slice boundary would trip: a tap still
    carrying the text prefix has more rows than the grid has cells."""
    model = _tiny_krea2()
    model.train()
    trainer = _train_step_trainer(model, 1, [])
    # Stand in for the module the trainer reads, holding an unsliced sequence.
    trainer._repa_tap_module = SimpleNamespace(
        _repa_tap_out=torch.randn(1, TEXT_LEN + GH * GW, HIDDEN))

    with pytest.raises(RuntimeError, match="would not correspond row for row"):
        _train_step(trainer, repa_pixels=torch.rand(1, 3, 32, 32))


def test_the_arch_handler_passes_the_batch_pixels_through():
    model = _tiny_krea2()
    trainer = _train_step_trainer(model, 1, [])
    seen = {}

    def _spy(_trainer, **kwargs):
        seen.update(kwargs)
        return torch.zeros(()), 0.0, 0.0

    pixels = torch.rand(1, 3, 32, 32)
    original = krea2_ops.train_step
    krea2_ops.train_step = _spy
    try:
        trainer.arch.train_step(trainer, TrainStepContext(
            latents=torch.zeros(1, GH * GW, IN_CHANNELS),
            encoder_features=torch.zeros(1, TEXT_LEN, TEXT_LAYERS, TEXT_DIM),
            encoder_mask=torch.ones(1, TEXT_LEN, dtype=torch.bool),
            latent_h=GH, latent_w=GW,
            repa_pixels=pixels))
    finally:
        krea2_ops.train_step = original

    assert seen["repa_pixels"] is pixels


# ---------------------------------------------------------------------------
# (d) the full-FT save
# ---------------------------------------------------------------------------

def test_a_full_finetune_save_pairs_the_projector_with_the_file_it_wrote(tmp_path):
    """The base adapter needs the RESOLVED path back; handed a directory, the
    sidecar has to land on the checkpoint written inside it."""
    from core.training.adapters.krea2_adapter import Krea2FullParameterAdapter

    model = _tiny_krea2(num_layers=1)
    trainer = _krea2_trainer(model)
    trainer.repa_enable = True
    trainer.repa_projector = repa_module.RepaProjector(HIDDEN, ENC_DIM, hidden=16)
    trainer.vae = None
    trainer.bundle_vae = False
    trainer.krea2_is_distilled = False

    written = Krea2FullParameterAdapter(trainer).save_checkpoint(5, 0, tmp_path)

    assert written == tmp_path / "krea2_step_5.safetensors"
    assert (tmp_path / "krea2_step_5.repa.safetensors").is_file()


# ---------------------------------------------------------------------------
# the capability table
# ---------------------------------------------------------------------------

def test_krea2_is_offered_the_repa_control():
    assert "repa" not in TRAINING_FEATURE_UNSUPPORTED.get("krea2", {})
