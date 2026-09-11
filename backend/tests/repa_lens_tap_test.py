"""Lens's REPA tap: the third architecture wired onto the shared foundation.

The tap is an explicit assignment inside the vendored Lens block loop rather
than a forward hook, so the tap IS the tensor the loss differentiates and there
is no hook ordering or checkpoint recompute behaviour to reason about. (A hook
would work here -- Lens checkpoints with ``use_reentrant=False`` -- but it
would work for a reason that is a flag away from being false.)

What is pinned here:
  (a) with the tap unarmed the forward is bit-identical and stashes nothing;
  (b) armed, the gradient reaches the blocks at and below the tap;
  (c) the tap's tokens are row-major over the latent grid, in the order
      ``lens_pipeline_ops.vae_encode`` packs the sequence and
      ``encode_repa_targets`` builds the teacher grid;
  (d) a full fine-tune pairs the projector sidecar with the file it wrote, and
      the second forward path (inference-only FBCache) cannot silently feed REPA
      a stale or absent tap.

Everything runs on a randomly initialised toy-geometry transformer; no
checkpoint is read and no GPU is used.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/repa_lens_tap_test.py -v
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

from api import arch_capabilities  # noqa: E402
from api.arch_capabilities import TRAINING_FEATURE_UNSUPPORTED  # noqa: E402
from core.models.lens.vendor.transformer import LensTransformer2DModel  # noqa: E402
from core.training import repa as repa_module  # noqa: E402
from core.training.arch import ARCH_REGISTRY  # noqa: E402
from core.training.arch.base_arch import ArchHandler, TrainStepContext  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.ops import lens_ops  # noqa: E402

BLOCKS = 4
HEADS = 2
HEAD_DIM = 8
INNER_DIM = HEADS * HEAD_DIM      # what LensTransformer2DModel.__init__ recomputes
LATENT_C = 2                      # raw latent channels; the sequence carries C*4
IN_CHANNELS = LATENT_C * 4
LATENT_H = 2
LATENT_W = 3
TEXT_LEN = 5
TEXT_DIM = 12
ENC_DIM = 8
ENC_GRID = 4



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


def _tiny_lens(num_layers: int = BLOCKS) -> LensTransformer2DModel:
    torch.manual_seed(0)
    return LensTransformer2DModel(
        # patch_size 1 with out_channels == in_channels: proj_out emits
        # patch_size**2 * out_channels, which the velocity target compares
        # against the packed latents (128 = 2**2 * 32 in the shipped config).
        patch_size=1, in_channels=IN_CHANNELS, out_channels=IN_CHANNELS,
        num_layers=num_layers, attention_head_dim=HEAD_DIM,
        num_attention_heads=HEADS, enc_hidden_dim=TEXT_DIM,
        axes_dims_rope=(2, 2, 4), multi_layer_encoder_feature=True,
        selected_layer_index=(0, 1),
    )


def _inputs(batch: int = 1, seed: int = 1):
    torch.manual_seed(seed)
    return dict(
        hidden=torch.randn(batch, LATENT_H * LATENT_W, IN_CHANNELS),
        text=[torch.randn(batch, TEXT_LEN, TEXT_DIM) for _ in range(2)],
        mask=torch.ones(batch, TEXT_LEN, dtype=torch.bool),
        timestep=torch.full((batch,), 0.5),
    )


def _forward(model, inp):
    return model(hidden_states=inp["hidden"], encoder_hidden_states=inp["text"],
                 encoder_hidden_states_mask=inp["mask"], timestep=inp["timestep"],
                 img_shapes=[(1, LATENT_H, LATENT_W)])


def _lens_trainer(model=None, **config):
    cfg = {"repa_enable": True, "repa_tagger_model_dir": "unused-because-stubbed"}
    cfg.update(config)
    return SimpleNamespace(
        config=cfg,
        arch=ARCH_REGISTRY["lens"](),
        transformer=model if model is not None else _tiny_lens(),
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        model_path="",
        log_prefix="[test]",
    )


def _stub_encoder_loader(monkeypatch):
    monkeypatch.setattr(repa_module, "load_repa_encoder",
                        lambda source, **kwargs: (_StubEncoder(), ENC_DIM, 32))


def _train_step_trainer(model, align_depth, logged):
    trainer = _lens_trainer(model)
    trainer.repa_enable = True  # _setup_repa's own flag, not the config key
    trainer.mixed_precision = False
    trainer.timestep_sampler = None
    trainer.stash_cfg_null_per_sample_loss = lambda pred, target: None
    trainer.repa_encoder = _StubEncoder()
    trainer.repa_projector = repa_module.RepaProjector(INNER_DIM, ENC_DIM, hidden=16)
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
    return lens_ops.train_step(
        trainer,
        latents=inp["hidden"],
        encoder_features=torch.stack(inp["text"], dim=1),  # [B, layers, L, D]
        encoder_mask=inp["mask"],
        timesteps=torch.rand(1),
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        repa_pixels=repa_pixels,
    )



def test_a_fresh_lens_carries_the_tap_attributes_unarmed():
    """Inference loads the same class; the attributes must exist and be inert."""
    model = _tiny_lens()
    assert model._repa_tap_depth is None
    assert model._repa_tap_out is None


def test_arming_the_tap_does_not_change_the_forward():
    model = _tiny_lens().eval()
    inp = _inputs()

    with torch.no_grad():
        disabled = _forward(model, inp)
        assert model._repa_tap_out is None
        model._repa_tap_depth = 2
        armed = _forward(model, inp)

    assert torch.equal(disabled, armed)
    assert model._repa_tap_out is not None


def test_a_forward_clears_what_a_previous_one_stashed():
    """Otherwise a step whose tap does not fire would read stale tokens."""
    model = _tiny_lens().eval()
    inp = _inputs()
    model._repa_tap_depth = 2
    with torch.no_grad():
        _forward(model, inp)
        assert model._repa_tap_out is not None

        # A depth no block carries: the tap cannot fire, so only the clear can
        # decide what is left behind. Asserting a different tensor instead would
        # pass with the clear deleted, since a firing tap overwrites anyway.
        model._repa_tap_depth = len(model.transformer_blocks) + 5
        _forward(model, inp)

    assert model._repa_tap_out is None



def test_the_handler_answers_with_the_dit_its_width_and_its_block_count():
    model = _tiny_lens()
    trainer = _lens_trainer(model)

    tap = trainer.arch.repa_tap(trainer)

    assert tap.module is model
    assert tap.hidden_size == INNER_DIM == model.inner_dim
    assert tap.depth == BLOCKS == len(model.transformer_blocks)


def test_the_width_is_the_recomputed_one_not_the_config_key():
    """``__init__`` overwrites the ``inner_dim`` argument with heads x head_dim;
    reading the config key would size the projector for a 1536-wide model."""
    trainer = _lens_trainer()

    assert trainer.transformer.config.inner_dim != INNER_DIM
    assert trainer.arch.repa_tap(trainer).hidden_size == INNER_DIM


def test_setup_repa_arms_the_lens_dit(monkeypatch):
    _stub_encoder_loader(monkeypatch)
    model = _tiny_lens()
    trainer = _lens_trainer(model)

    BaseTrainer._setup_repa(trainer)

    assert trainer.repa_align_depth == BLOCKS // 3
    assert model._repa_tap_depth == BLOCKS // 3
    assert trainer._repa_tap_module is model
    assert trainer.repa_projector.net[0].in_features == INNER_DIM


def test_setup_repa_is_inert_for_lens_when_disabled():
    model = _tiny_lens()
    trainer = _lens_trainer(model, repa_enable=False)

    BaseTrainer._setup_repa(trainer)

    assert trainer.repa_enable is False
    assert trainer._repa_tap_module is None
    assert model._repa_tap_depth is None
    assert not hasattr(trainer, "repa_projector")


def test_lens_does_not_consume_the_block_loop_features():
    """TREAD / stochastic depth / DiT-BlockSkip have no path in Lens's forward,
    so the depth-conflict check must stay switched off for it."""
    assert ARCH_REGISTRY["lens"].consumes_block_loop_features is False



def test_the_tap_carries_gradient_to_the_blocks_at_and_below_it():
    model = _tiny_lens()
    model.train()
    model.enable_gradient_checkpointing()  # the case a forward hook would break
    tap_depth = 1
    model._repa_tap_depth = tap_depth
    projector = repa_module.RepaProjector(INNER_DIM, ENC_DIM, hidden=16)

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



def test_the_tap_is_the_image_stream_at_the_latent_grid():
    model = _tiny_lens().eval()
    model._repa_tap_depth = 1

    with torch.no_grad():
        _forward(model, _inputs())

    assert model._repa_tap_out.shape == (1, LATENT_H * LATENT_W, INNER_DIM)


@pytest.mark.parametrize("cell", [(0, 0), (0, 2), (1, 1), (1, 2)])
def test_the_tokens_are_row_major_over_the_latent_grid(cell):
    """Perturb one latent cell; the token that moves most must be the one at
    h*latent_w + w -- the index encode_repa_targets puts that cell's teacher
    feature at. The sequence is built with vae_encode's OWN packing expression.
    """
    from einops import rearrange

    model = _tiny_lens().eval()
    model._repa_tap_depth = 1
    h, w = cell
    inp = _inputs()

    torch.manual_seed(5)
    spatial = torch.randn(1, LATENT_C, LATENT_H * 2, LATENT_W * 2)
    perturbed = spatial.clone()
    perturbed[:, :, h * 2:(h + 1) * 2, w * 2:(w + 1) * 2] += 5.0

    def _tap(x):
        packed = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                           p1=2, p2=2, h=LATENT_H, w=LATENT_W)
        with torch.no_grad():
            _forward(model, dict(inp, hidden=packed))
        return model._repa_tap_out.clone()

    moved = (_tap(perturbed) - _tap(spatial)).norm(dim=-1)[0]

    index = h * LATENT_W + w
    assert int(moved.argmax()) == index
    assert float(moved[index]) > 2.0 * float(moved.median())



def test_train_step_adds_a_finite_alignment_term_that_reaches_the_dit():
    model = _tiny_lens()
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
    model = _tiny_lens()
    model.train()
    logged = []
    trainer = _train_step_trainer(model, 1, logged)

    loss, pred, _ = _train_step(trainer, repa_pixels=None)

    assert float(loss) == pytest.approx(pred, rel=1e-6)
    assert logged == []


def test_three_steps_run_and_stay_finite():
    model = _tiny_lens()
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
    model = _tiny_lens()
    model.train()
    trainer = _train_step_trainer(model, 1, [])
    model._repa_tap_depth = None  # forward writes nothing

    with pytest.raises(RuntimeError, match="stashed nothing"):
        _train_step(trainer, repa_pixels=torch.rand(1, 3, 32, 32))


def test_the_arch_handler_passes_the_batch_pixels_through():
    model = _tiny_lens()
    trainer = _train_step_trainer(model, 1, [])
    seen = {}

    def _spy(_trainer, **kwargs):
        seen.update(kwargs)
        return torch.zeros(()), 0.0, 0.0

    pixels = torch.rand(1, 3, 32, 32)
    original = lens_ops.train_step
    lens_ops.train_step = _spy
    try:
        trainer.arch.train_step(trainer, TrainStepContext(
            latents=torch.zeros(1, LATENT_H * LATENT_W, IN_CHANNELS),
            encoder_features=torch.zeros(1, 2, TEXT_LEN, TEXT_DIM),
            encoder_mask=torch.ones(1, TEXT_LEN, dtype=torch.bool),
            latent_h=LATENT_H, latent_w=LATENT_W,
            repa_pixels=pixels))
    finally:
        lens_ops.train_step = original

    assert seen["repa_pixels"] is pixels



class _MissingFBCache:
    """Enough of core.inference.fbcache for that branch to run (always a miss)."""

    def use_cache(self, indicator, step):
        return False

    def store(self, residuals):
        self.stored = residuals


def test_the_fbcache_branch_leaves_the_tap_empty_rather_than_stale():
    """FBCache is attached only by the inference denoise loop, so it never sees
    a training forward -- but if it ever did, REPA must find nothing and raise
    (the train_step case above) rather than align another step's tokens."""
    model = _tiny_lens().eval()
    model._repa_tap_depth = 1
    inp = _inputs()
    with torch.no_grad():
        _forward(model, inp)
        assert model._repa_tap_out is not None

        model._fbcache = _MissingFBCache()
        _forward(model, inp)

    assert model._repa_tap_out is None


def test_a_full_finetune_save_pairs_the_projector_with_the_file_it_wrote(tmp_path):
    """The base adapter needs the RESOLVED path back; handed a directory, the
    sidecar has to land on the checkpoint written inside it."""
    from core.training.adapters.lens_adapter import LensFullParameterAdapter

    model = _tiny_lens(1)
    trainer = _lens_trainer(model)
    trainer.repa_enable = True
    trainer.repa_projector = repa_module.RepaProjector(INNER_DIM, ENC_DIM, hidden=16)
    trainer.vae = None
    trainer.bundle_vae = False
    trainer.lens_base_dir = ""

    written = LensFullParameterAdapter(trainer).save_checkpoint(5, 0, tmp_path)

    assert written == tmp_path / "lens_step_5.safetensors"
    assert (tmp_path / "lens_step_5.repa.safetensors").is_file()



def test_lens_is_offered_the_repa_control():
    assert "repa" not in TRAINING_FEATURE_UNSUPPORTED.get("lens", {})


@pytest.mark.parametrize("arch", sorted(ARCH_REGISTRY))
def test_the_capability_table_matches_the_handlers(arch):
    handler = ARCH_REGISTRY[arch]
    declines = handler.repa_tap is ArchHandler.repa_tap
    refused = "repa" in TRAINING_FEATURE_UNSUPPORTED.get(arch, {})
    assert declines == refused, (
        f"{arch}: repa_tap {'declines' if declines else 'is implemented'} but "
        f"the capability table {'refuses' if refused else 'offers'} repa")


def test_the_capability_reasons_and_the_trainer_refusals_share_their_markers():
    """The API table cannot import the trainer's REPA module, so it carries its
    own copy of the marker phrases and checks its own reasons against them at
    import. This test is the only place the two copies meet."""
    assert (arch_capabilities._REPA_REFUSAL_MARKERS
            == repa_module.REPA_REFUSAL_MARKERS)
    assert set(repa_module.REPA_REFUSAL_MARKERS) == set(repa_module.REPA_REFUSALS)
    for arch, marker in repa_module.REPA_REFUSAL_MARKERS.items():
        assert marker in repa_module.REPA_REFUSALS[arch]
        assert marker in TRAINING_FEATURE_UNSUPPORTED[arch]["repa"]["reason"]
