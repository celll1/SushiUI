"""Ideogram 4's REPA tap: one packed sequence, text first, two transformers.

The tap is an explicit assignment inside the vendored block loop rather than a
forward hook, so the tap IS the tensor the loss differentiates and there is no
hook ordering or checkpoint recompute behaviour to reason about. What it stashes
is the WHOLE packed sequence; ``train_step`` drops the text prefix at the same
``max_text`` boundary it takes ``v_pred`` at.

What is pinned here:
  (a) with the tap unarmed the forward is bit-identical -- both against the
      armed/unarmed pair and against the pre-tap revision of the vendored
      module, at a geometry that is not the shipped one;
  (b) armed, the gradient reaches the blocks at and below the tap;
  (c) the image rows are row-major over the latent grid, the order
      ``build_training_conditioning`` lays the image position ids out in and
      ``encode_repa_targets`` builds the teacher grid in;
  (d) a LoRA checkpoint pairs the projector sidecar with it, and full
      fine-tuning refuses before it can return a path at all;
  (e) the slice lands at ``max_text``, so the tap length is the image grid
      whatever the prompt length.

Everything runs on a randomly initialised toy-geometry transformer; no
checkpoint is read and no GPU is used.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/repa_ideogram4_tap_test.py -v
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
from core.models.ideogram4.ideogram4_pipeline_ops import \
    build_training_conditioning  # noqa: E402
from core.models.ideogram4.vendor.transformer import \
    Ideogram4Transformer2DModel  # noqa: E402
from core.training import repa as repa_module  # noqa: E402
from core.training.arch import ARCH_REGISTRY  # noqa: E402
from core.training.arch.base_arch import TrainStepContext  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.ops import ideogram4_ops  # noqa: E402

VENDOR_REL = "backend/core/models/ideogram4/vendor/transformer.py"

BLOCKS = 4
HEADS = 2
HEAD_DIM = 8
HIDDEN = HEADS * HEAD_DIM      # 16; the shipped model is 18 * 256
IN_CHANNELS = 8
GH, GW = 2, 3
TEXT_LEN = 5
TEXT_LAYERS = 3                # the shipped model taps 13 Qwen3-VL layers
LAYER_DIM = 4
LLM_DIM = TEXT_LAYERS * LAYER_DIM
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


def _tiny_ideogram4(cls=Ideogram4Transformer2DModel, num_layers: int = BLOCKS):
    torch.manual_seed(0)
    return cls(
        in_channels=IN_CHANNELS, num_layers=num_layers, attention_head_dim=HEAD_DIM,
        num_attention_heads=HEADS, intermediate_size=16, adaln_dim=8,
        llm_features_dim=LLM_DIM, rope_theta=1000, mrope_section=(2, 1, 1),
    )


def _packed_inputs(batch: int = 1, seed: int = 1, text_len: int = TEXT_LEN):
    """The same packing ``ideogram4_ops.train_step`` builds: zeros where the text
    tokens sit, latents after them, conditioning from the shared builder."""
    torch.manual_seed(seed)
    cond = build_training_conditioning(
        torch.randn(batch, text_len, LLM_DIM), torch.ones(batch, text_len), GH, GW)
    max_text = cond["max_text_tokens"]
    latents = torch.randn(batch, GH * GW, IN_CHANNELS)
    return dict(
        hidden_states=torch.cat([torch.zeros(batch, max_text, IN_CHANNELS), latents], dim=1),
        timestep=torch.full((batch,), 0.5),
        encoder_hidden_states=cond["llm_features"],
        position_ids=cond["position_ids"],
        segment_ids=cond["segment_ids"],
        indicator=cond["indicator"],
    ), max_text, latents


def _forward(model, inp):
    return model(return_dict=False, **inp)[0]


def _ideogram4_trainer(model=None, **config):
    cfg = {"repa_enable": True, "repa_tagger_model_dir": "unused-because-stubbed"}
    cfg.update(config)
    return SimpleNamespace(
        config=cfg,
        arch=ARCH_REGISTRY["ideogram4"](),
        transformer=model if model is not None else _tiny_ideogram4(),
        transformer_uncond=None,
        ideogram4_train_uncond=False,
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        model_path="",
        log_prefix="[test]",
    )


def _stub_encoder_loader(monkeypatch):
    monkeypatch.setattr(repa_module, "load_repa_encoder",
                        lambda source, **kwargs: (_StubEncoder(), ENC_DIM, 32))


def _train_step_trainer(model, align_depth, logged):
    trainer = _ideogram4_trainer(model)
    trainer.repa_enable = True  # _setup_repa's own flag, not the config key
    trainer.mixed_precision = False
    trainer.timestep_sampler = None
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


def _train_step(trainer, *, repa_pixels, seed=7, text_len=TEXT_LEN):
    torch.manual_seed(seed)
    return ideogram4_ops.train_step(
        trainer,
        latents=torch.randn(1, GH * GW, IN_CHANNELS),
        encoder_features=torch.randn(1, TEXT_LAYERS, text_len, LAYER_DIM),
        encoder_mask=torch.ones(1, text_len),
        timesteps=torch.rand(1),
        latent_h=GH,
        latent_w=GW,
        repa_pixels=repa_pixels,
    )


# ---------------------------------------------------------------------------
# (a) the disabled path
# ---------------------------------------------------------------------------

def test_a_fresh_ideogram4_carries_the_tap_attributes_unarmed():
    """Inference loads the same class; the attributes must exist and be inert."""
    model = _tiny_ideogram4()
    assert model._repa_tap_depth is None
    assert model._repa_tap_out is None


def test_arming_the_tap_does_not_change_the_forward():
    model = _tiny_ideogram4().eval()
    inp, _, _ = _packed_inputs()

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
        path = Path(repo) / "backend" / "tests" / "__pycache__" / "ideogram4_pre_tap_vendor.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(src)
        spec = importlib.util.spec_from_file_location("ideogram4_pre_tap_vendor", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["ideogram4_pre_tap_vendor"] = module
        spec.loader.exec_module(module)
        return module
    return None


def test_the_disabled_forward_matches_the_revision_before_the_tap():
    """Not the same as arming/unarming one module: this runs the pre-tap SOURCE.
    The geometry is deliberately not the shipped one, so a change that only holds
    for 34 blocks of 4608 would show up here."""
    module = _pre_tap_vendor_module()
    if module is None:
        pytest.skip("no pre-tap revision of the vendored transformer in git history")

    before = _tiny_ideogram4(module.Ideogram4Transformer2DModel).eval()
    after = _tiny_ideogram4().eval()
    inp, _, _ = _packed_inputs()

    with torch.no_grad():
        assert torch.equal(_forward(before, inp), _forward(after, inp))
    assert after._repa_tap_out is None


def test_a_forward_clears_what_a_previous_one_stashed():
    """Otherwise a step whose tap does not fire would read stale tokens."""
    model = _tiny_ideogram4().eval()
    inp, _, _ = _packed_inputs()
    model._repa_tap_depth = 2
    with torch.no_grad():
        _forward(model, inp)
        assert model._repa_tap_out is not None

        # A depth no block carries: the tap cannot fire, so only the clear can
        # decide what is left behind.
        model._repa_tap_depth = len(model.layers) + 5
        _forward(model, inp)

    assert model._repa_tap_out is None


# ---------------------------------------------------------------------------
# the handler's answer
# ---------------------------------------------------------------------------

def test_the_handler_answers_with_the_conditional_dit_its_width_and_block_count():
    model = _tiny_ideogram4()
    trainer = _ideogram4_trainer(model)
    trainer.transformer_uncond = _tiny_ideogram4()
    trainer.ideogram4_train_uncond = True

    tap = trainer.arch.repa_tap(trainer)

    assert tap.module is model
    assert tap.module is not trainer.transformer_uncond
    assert tap.hidden_size == HIDDEN == model.hidden_size
    assert tap.depth == BLOCKS == len(model.layers)


def test_the_width_is_the_attribute_because_the_config_has_no_such_key():
    """``hidden_size`` is derived in ``__init__`` from attention_head_dim x
    num_attention_heads and never registered, so the config cannot answer."""
    model = _tiny_ideogram4()

    assert "hidden_size" not in dict(model.config)
    assert model.hidden_size == HIDDEN


def test_setup_repa_arms_only_the_conditional_dit(monkeypatch):
    _stub_encoder_loader(monkeypatch)
    model = _tiny_ideogram4()
    trainer = _ideogram4_trainer(model)
    uncond = trainer.transformer_uncond = _tiny_ideogram4()
    trainer.ideogram4_train_uncond = True

    BaseTrainer._setup_repa(trainer)

    assert trainer.repa_align_depth == BLOCKS // 3
    assert model._repa_tap_depth == BLOCKS // 3
    assert uncond._repa_tap_depth is None
    assert trainer._repa_tap_module is model
    assert trainer.repa_projector.net[0].in_features == HIDDEN


def test_setup_repa_is_inert_for_ideogram4_when_disabled():
    model = _tiny_ideogram4()
    trainer = _ideogram4_trainer(model, repa_enable=False)

    BaseTrainer._setup_repa(trainer)

    assert trainer.repa_enable is False
    assert trainer._repa_tap_module is None
    assert model._repa_tap_depth is None
    assert not hasattr(trainer, "repa_projector")


def test_ideogram4_does_not_consume_the_block_loop_features():
    """TREAD / stochastic depth / DiT-BlockSkip have no path in Ideogram 4's
    forward, so the depth-conflict check must stay switched off for it."""
    assert ARCH_REGISTRY["ideogram4"].consumes_block_loop_features is False


# ---------------------------------------------------------------------------
# (b) the gradient reaches the model
# ---------------------------------------------------------------------------

def test_the_tap_carries_gradient_to_the_blocks_at_and_below_it():
    model = _tiny_ideogram4()
    model.train()
    model.enable_gradient_checkpointing()  # the case a forward hook would break
    tap_depth = 1
    model._repa_tap_depth = tap_depth
    projector = repa_module.RepaProjector(HIDDEN, ENC_DIM, hidden=16)

    inp, max_text, _ = _packed_inputs()
    _forward(model, inp)
    tokens = model._repa_tap_out[:, max_text:]
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
        assert _grad_norm(model.layers[depth]) > 0, depth
    for depth in range(tap_depth + 1, BLOCKS):
        assert all(p.grad is None for p in model.layers[depth].parameters()), depth
    assert _grad_norm(model.input_proj) > 0


# ---------------------------------------------------------------------------
# (c) spatial correspondence and (e) the text prefix
# ---------------------------------------------------------------------------

def test_the_tap_is_the_packed_sequence_and_the_slice_is_the_grid():
    model = _tiny_ideogram4().eval()
    model._repa_tap_depth = 1
    inp, max_text, _ = _packed_inputs()

    with torch.no_grad():
        _forward(model, inp)

    packed = model._repa_tap_out
    assert packed.shape == (1, max_text + GH * GW, HIDDEN)
    assert packed[:, max_text:].shape == (1, GH * GW, HIDDEN)


@pytest.mark.parametrize("text_len", [1, TEXT_LEN, TEXT_LEN + 7])
def test_the_slice_lands_at_max_text_whatever_the_prompt_length(text_len):
    """max_text is the text token count the conditioning builder packed, so the
    boundary moves with the prompt; a fixed offset would misalign every token."""
    model = _tiny_ideogram4().eval()
    model._repa_tap_depth = 1
    inp, max_text, _ = _packed_inputs(text_len=text_len)

    with torch.no_grad():
        _forward(model, inp)

    assert max_text == text_len
    assert model._repa_tap_out[:, max_text:].shape[1] == GH * GW


@pytest.mark.parametrize("index", [0, 2, 4, 5])
def test_the_image_rows_are_row_major_over_the_latent_grid(index):
    """Perturb one image token; the tapped row that moves most must be the same
    index -- the position encode_repa_targets puts that cell's teacher feature
    at. The packing is build_training_conditioning's own."""
    model = _tiny_ideogram4().eval()
    model._repa_tap_depth = 1
    inp, max_text, latents = _packed_inputs()

    def _tap(lat):
        packed = torch.cat([torch.zeros(1, max_text, IN_CHANNELS), lat], dim=1)
        with torch.no_grad():
            _forward(model, dict(inp, hidden_states=packed))
        return model._repa_tap_out[:, max_text:].clone()

    perturbed = latents.clone()
    perturbed[:, index] += 5.0
    moved = (_tap(perturbed) - _tap(latents)).norm(dim=-1)[0]

    assert int(moved.argmax()) == index
    assert float(moved[index]) > 2.0 * float(moved.median())


# ---------------------------------------------------------------------------
# the per-step path
# ---------------------------------------------------------------------------

def test_train_step_adds_a_finite_alignment_term_that_reaches_the_dit():
    model = _tiny_ideogram4()
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
               for p in model.layers[0].parameters())


def test_train_step_without_repa_pixels_is_the_plain_diffusion_loss():
    model = _tiny_ideogram4()
    model.train()
    logged = []
    trainer = _train_step_trainer(model, 1, logged)

    loss, pred, _ = _train_step(trainer, repa_pixels=None)

    assert float(loss) == pytest.approx(pred, rel=1e-6)
    assert logged == []


def test_three_steps_run_and_stay_finite():
    model = _tiny_ideogram4()
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
    model = _tiny_ideogram4()
    model.train()
    trainer = _train_step_trainer(model, 1, [])
    model._repa_tap_depth = None  # forward writes nothing

    with pytest.raises(RuntimeError, match="stashed nothing"):
        _train_step(trainer, repa_pixels=torch.rand(1, 3, 32, 32))


def test_a_tap_that_is_not_the_grid_refuses_rather_than_misaligning():
    """The length guard is what a wrong slice boundary would trip: a sequence
    whose text prefix is longer than max_text leaves too many rows behind."""
    model = _tiny_ideogram4()
    model.train()
    trainer = _train_step_trainer(model, 1, [])
    trainer._repa_tap_module = SimpleNamespace(
        _repa_tap_out=torch.randn(1, 2 * TEXT_LEN + GH * GW, HIDDEN))

    with pytest.raises(RuntimeError, match="would not correspond row for row"):
        _train_step(trainer, repa_pixels=torch.rand(1, 3, 32, 32))


def test_the_fbcache_branch_leaves_the_tap_empty_rather_than_stale():
    """FBCache is attached only by the inference denoise loop, so it never sees a
    training forward -- but if it ever did, REPA must find nothing and raise (the
    train_step case above) rather than align another step's tokens."""

    class _MissingFBCache:
        def use_cache(self, indicator, step):
            return False

        def store(self, residuals):
            self.stored = residuals

    model = _tiny_ideogram4().eval()
    model._repa_tap_depth = 1
    inp, _, _ = _packed_inputs()
    with torch.no_grad():
        _forward(model, inp)
        assert model._repa_tap_out is not None

        model._fbcache = _MissingFBCache()
        _forward(model, inp)

    assert model._repa_tap_out is None


def test_the_arch_handler_passes_the_batch_pixels_through():
    model = _tiny_ideogram4()
    trainer = _train_step_trainer(model, 1, [])
    seen = {}

    def _spy(_trainer, **kwargs):
        seen.update(kwargs)
        return torch.zeros(()), 0.0, 0.0

    pixels = torch.rand(1, 3, 32, 32)
    original = ideogram4_ops.train_step
    ideogram4_ops.train_step = _spy
    try:
        trainer.arch.train_step(trainer, TrainStepContext(
            latents=torch.zeros(1, GH * GW, IN_CHANNELS),
            encoder_features=torch.zeros(1, TEXT_LAYERS, TEXT_LEN, LAYER_DIM),
            encoder_mask=torch.ones(1, TEXT_LEN),
            latent_h=GH, latent_w=GW,
            repa_pixels=pixels))
    finally:
        ideogram4_ops.train_step = original

    assert seen["repa_pixels"] is pixels


# ---------------------------------------------------------------------------
# (d) the checkpoint pairing
# ---------------------------------------------------------------------------

def test_a_lora_save_pairs_the_projector_with_the_checkpoint(tmp_path):
    from core.training.adapters.ideogram4_adapter import Ideogram4LoRAAdapter

    trainer = _ideogram4_trainer()
    trainer.repa_enable = True
    trainer.repa_projector = repa_module.RepaProjector(HIDDEN, ENC_DIM, hidden=16)
    path = tmp_path / "ideogram4_step_5.safetensors"

    Ideogram4LoRAAdapter(trainer, 4, 4).save_checkpoint({}, 5, 0, path)

    assert path.is_file()
    assert (tmp_path / "ideogram4_step_5.repa.safetensors").is_file()


def test_full_finetuning_refuses_before_it_could_return_no_path():
    """The base adapter raises when write_checkpoint returns None with REPA on.
    Ideogram 4 cannot reach that: full FT refuses at the top, and the refusal is
    what a bf16 base would have to lift -- with a returned path."""
    from core.training.adapters.ideogram4_adapter import Ideogram4FullParameterAdapter

    adapter = Ideogram4FullParameterAdapter(_ideogram4_trainer())

    with pytest.raises(NotImplementedError):
        adapter.write_checkpoint(5, 0, Path("unused"))


# ---------------------------------------------------------------------------
# the capability table
# ---------------------------------------------------------------------------

def test_ideogram4_is_offered_the_repa_control():
    assert "repa" not in TRAINING_FEATURE_UNSUPPORTED.get("ideogram4", {})
