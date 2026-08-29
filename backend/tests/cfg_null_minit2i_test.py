"""Delivery item 3: the shared per-batch drop mask, and MiniT2I on it.

local/strategy/cfg_null_alignment/IMPLEMENTATION_STRATEGY.md sections 5 and 6.1.
Everything here runs on synthetic tensors; no checkpoint is loaded.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/cfg_null_minit2i_test.py -v
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from api.cfg_null_resolver import CFG_KEY, LEGACY_KEY  # noqa: E402
from api.error_handlers import ValidationError  # noqa: E402
from api.param_defaults import CFG_UNCOND_DROP_DEFAULTS_BY_ARCH  # noqa: E402
from core.training.arch.minit2i import MiniT2IArchHandler  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.ops.minit2i_ops import apply_cfg_null_collated  # noqa: E402

_BASE_TRAINER_SOURCE = (BACKEND / "core" / "training"
                        / "base_trainer.py").read_text(encoding="utf-8")

# The one line of MMJiT.forward the aligned null has to agree with. Duplicated
# here rather than instantiating the model (which needs weights); the
# source-pinning test below fails if the model's line ever moves away from it.
_MMJIT_CONTEXT_LINE = (
    "context = torch.where(attn_mask[:, :, None] > 0.5, context, "
    "self.mask_token.to(dtype=context.dtype))"
)


def _mmjit_context(text, mask, mask_token):
    return torch.where(mask[:, :, None] > 0.5, text,
                       mask_token.to(dtype=text.dtype))


class _StubArch:
    name = "minit2i"
    cfg_null_stage = MiniT2IArchHandler.cfg_null_stage


class _StubTrainer:
    """Just enough of BaseTrainer for the two mask methods under test."""

    cfg_null_drop_rate = BaseTrainer.cfg_null_drop_rate
    sample_cfg_drop_mask = BaseTrainer.sample_cfg_drop_mask
    log_prefix = "[cfg-null-test]"

    def __init__(self, config, arch=None):
        self.config = dict(config)
        self.arch = arch if arch is not None else _StubArch()


def _config(**overrides):
    """A train section as the generator writes one: both keys present, null
    unless supplied, and no `_explicit_fields` (the trainer has no request)."""
    section = {CFG_KEY: None, LEGACY_KEY: None}
    section.update(overrides)
    return section


# ---------------------------------------------------------------------------
# Rate resolution on the trainer side (strategy §3 rules 2 and 4)
# ---------------------------------------------------------------------------

def test_an_omitted_key_still_label_drops_at_the_inherited_rate():
    trainer = _StubTrainer(_config())
    assert trainer.cfg_null_drop_rate() == 0.1
    assert CFG_UNCOND_DROP_DEFAULTS_BY_ARCH["minit2i"] == 0.1


def test_an_explicit_zero_disables_the_inherited_rate():
    trainer = _StubTrainer(_config(**{CFG_KEY: 0.0}))
    assert trainer.cfg_null_drop_rate() == 0.0
    assert trainer.sample_cfg_drop_mask(8) is None


def test_the_legacy_spelling_still_sets_the_rate():
    trainer = _StubTrainer(_config(**{LEGACY_KEY: 0.35}))
    assert trainer.cfg_null_drop_rate() == 0.35


def test_a_config_carrying_both_keys_is_refused():
    """The generator no longer writes one of them for the caller; a hand-edited
    YAML that sets both still has no answer that is safe to guess."""
    trainer = _StubTrainer(_config(**{CFG_KEY: 0.2, LEGACY_KEY: 0.1}))
    with pytest.raises(ValidationError):
        trainer.cfg_null_drop_rate()


def test_the_rate_is_resolved_once_per_run():
    trainer = _StubTrainer(_config(**{CFG_KEY: 0.4}))
    assert trainer.cfg_null_drop_rate() == 0.4
    trainer.config[CFG_KEY] = 0.9
    assert trainer.cfg_null_drop_rate() == 0.4


def test_a_stageless_architecture_draws_no_mask():
    class _NoStage:
        name = "sdxl"
        cfg_null_stage = None

    trainer = _StubTrainer(_config(**{LEGACY_KEY: 0.5}), arch=_NoStage())
    assert trainer.cfg_null_drop_rate() is None
    assert trainer.sample_cfg_drop_mask(4) is None


def test_the_drawn_mask_is_a_cpu_boolean_of_batch_length():
    trainer = _StubTrainer(_config(**{CFG_KEY: 0.5}))
    mask = trainer.sample_cfg_drop_mask(6)
    assert mask.shape == (6,)
    assert mask.dtype is torch.bool
    assert mask.device.type == "cpu"


def test_the_rate_is_the_selection_frequency():
    torch.manual_seed(0)
    trainer = _StubTrainer(_config(**{CFG_KEY: 0.25}))
    drawn = torch.cat([trainer.sample_cfg_drop_mask(64) for _ in range(50)])
    assert 0.20 < drawn.float().mean().item() < 0.30


# ---------------------------------------------------------------------------
# One draw per assembled batch, reused across MNT (strategy §5)
# ---------------------------------------------------------------------------

def _train_loop_function() -> ast.FunctionDef:
    tree = ast.parse(_BASE_TRAINER_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "train":
            for sub in ast.walk(node):
                if (isinstance(sub, ast.For) and isinstance(sub.target, ast.Name)
                        and sub.target.id == "mnt_idx"):
                    return node
    raise AssertionError("no train() with an MNT loop in base_trainer.py")


def _calls_named(node, name) -> int:
    return sum(
        1 for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == name
    )


def test_the_mask_is_drawn_outside_the_mnt_loop():
    """A draw inside the loop would give one item a different meaning on each
    MNT transform of the same images -- the failure §5 names."""
    train_fn = _train_loop_function()
    mnt_loops = [sub for sub in ast.walk(train_fn)
                 if isinstance(sub, ast.For) and isinstance(sub.target, ast.Name)
                 and sub.target.id == "mnt_idx"]
    assert mnt_loops
    assert _calls_named(train_fn, "sample_cfg_drop_mask") == 1
    for loop in mnt_loops:
        assert _calls_named(loop, "sample_cfg_drop_mask") == 0


def test_no_architecture_op_draws_its_own_bernoulli():
    """`torch.rand(B ...) < rate` inside a train_step is the second draw §6.1
    moves out; a returning one would silently defeat an explicit 0.0."""
    ops = sorted((BACKEND / "core" / "training" / "ops").glob("*_ops.py"))
    assert any(p.name == "minit2i_ops.py" for p in ops), ops
    for path in ops:
        source = path.read_text(encoding="utf-8")
        assert "label_drop_rate" not in source, path.name


def test_every_mnt_iteration_sees_the_same_label():
    trainer = _StubTrainer(_config(**{CFG_KEY: 0.5}))
    mask = trainer.sample_cfg_drop_mask(4)
    text = torch.randn(4, 3, 8)
    attn = torch.ones(4, 3)
    seen = []
    for _ in range(3):
        _, rewritten = apply_cfg_null_collated(text, attn, mask)
        seen.append(rewritten)
    assert all(torch.equal(seen[0], other) for other in seen[1:])
    # The label itself is untouched by being applied.
    assert mask.dtype is torch.bool and mask.shape == (4,)


# ---------------------------------------------------------------------------
# Micro-batch slicing (strategy §5: "slice it wherever the batch is sliced")
# ---------------------------------------------------------------------------

class _MicroBatchStub:
    _microbatch_two_stage = BaseTrainer._microbatch_two_stage
    _slice_aux = staticmethod(BaseTrainer._slice_aux)

    def __init__(self):
        self.seen = []

    def _execute_forward_backward(self, **kwargs):
        self.seen.append(kwargs["cfg_drop_mask"])
        return 0.0, 0.0, 0.0

    def _reset_fused_group_counters(self):
        pass

    def _flush_fused_group_partials(self):
        pass


def _micro_batch(batch_size, cfg_drop_mask):
    return dict(
        mnt_latents=torch.zeros(batch_size, 3, 4, 4),
        mnt_text_embeddings=torch.zeros(batch_size, 2, 8),
        mnt_attention_mask=torch.ones(batch_size, 2),
        mnt_pooled_embeddings=None,
        timesteps=torch.zeros(batch_size),
        debug_save_path=None,
        batch_captions=None,
        batch_reference_paths=None,
        alphas_cumprod_cached=None,
        use_condition_images=False,
        condition_images_batch=None,
        reference_latents_nested=None,
        lens_latent_shape=None,
        mnt_repa_pixels=None,
        mnt_time_ids=None,
        loss_weight_maps_batch=None,
        sensenova_prefix=None,
        cfg_drop_mask=cfg_drop_mask,
    )


def test_the_mask_survives_micro_batch_slicing():
    mask = torch.tensor([True, False, False, True])
    stub = _MicroBatchStub()
    stub._microbatch_two_stage(2, 4, _micro_batch(4, mask))
    assert len(stub.seen) == 2
    assert torch.equal(torch.cat(stub.seen), mask)
    assert torch.equal(stub.seen[0], mask[0:2])
    assert torch.equal(stub.seen[1], mask[2:4])


def test_micro_batching_without_the_mechanism_passes_none():
    stub = _MicroBatchStub()
    stub._microbatch_two_stage(2, 4, _micro_batch(4, None))
    assert stub.seen == [None, None]


def test_the_oom_retry_reuses_the_mask_rather_than_redrawing():
    """A resample on retry would change the drop pattern of the batch being
    retried, which is exactly the silent thread §5 warns about."""
    tree = ast.parse(_BASE_TRAINER_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and \
                node.name == "_forward_backward_with_oom_recovery":
            assert _calls_named(node, "sample_cfg_drop_mask") == 0
            assert "cfg_drop_mask" in {
                kw.arg for sub in ast.walk(node)
                if isinstance(sub, ast.Call) for kw in sub.keywords if kw.arg}
            return
    raise AssertionError("_forward_backward_with_oom_recovery not found")


# ---------------------------------------------------------------------------
# The MiniT2I collated rewrite (strategy §6.1)
# ---------------------------------------------------------------------------

def test_the_rewrite_is_out_of_place():
    """The conditioning belongs to the assembled batch and is handed to every
    MNT iteration; an in-place write would leak one iteration's null forward."""
    text = torch.randn(3, 4, 8)
    attn = torch.ones(3, 4)
    attn_before = attn.clone()
    mask = torch.tensor([False, True, False])

    out_text, out_attn = apply_cfg_null_collated(text, attn, mask)

    assert out_attn is not attn
    assert torch.equal(attn, attn_before)
    # The text tensor is deliberately NOT rewritten (see below), so it is the
    # same object, unchanged.
    assert out_text is text


def test_only_the_selected_rows_lose_their_mask():
    attn = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    mask = torch.tensor([False, True, False])
    _, out = apply_cfg_null_collated(torch.randn(3, 3, 4), attn, mask)
    assert torch.equal(out[0], attn[0])
    assert torch.equal(out[2], attn[2])
    assert not out[1].any()


def test_an_empty_selection_returns_the_batch_untouched():
    text = torch.randn(2, 3, 4)
    attn = torch.ones(2, 3)
    for mask in (None, torch.zeros(2, dtype=torch.bool)):
        out_text, out_attn = apply_cfg_null_collated(text, attn, mask)
        assert out_text is text
        assert out_attn is attn


def test_a_cpu_label_rewrites_conditioning_on_another_device():
    """The mask is a CPU label by contract; indexing must not depend on it
    sharing the conditioning's device."""
    attn = torch.ones(2, 3)
    mask = torch.tensor([True, False], device="cpu")
    _, out = apply_cfg_null_collated(torch.randn(2, 3, 4), attn, mask)
    assert not out[0].any()
    assert out[1].all()


def test_an_integer_mask_dtype_is_preserved():
    attn = torch.ones(2, 3, dtype=torch.long)
    _, out = apply_cfg_null_collated(torch.randn(2, 3, 4), attn,
                                     torch.tensor([True, False]))
    assert out.dtype is torch.long
    assert not out[0].any()


def test_the_dropped_row_matches_the_inference_pure_uncond_row():
    """MMJiT.forward replaces every masked text row with the learned
    mask_token, so a training row whose mask this rewrite zeroed reaches the
    blocks as the SAME context inference's `u_text=text, u_mask=zeros_like(mask)`
    branch builds -- pooled embedding included, since that is `context.mean`
    AFTER the replacement."""
    torch.manual_seed(7)
    text = torch.randn(2, 5, 6)
    attn = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]])
    mask_token = torch.randn(1, 1, 6)

    _, trained_attn = apply_cfg_null_collated(
        text, attn, torch.tensor([False, True]))
    trained_context = _mmjit_context(text, trained_attn, mask_token)

    # _predict_x0_cfg's neg_text-is-None branch, for row 1 alone.
    u_text = text[1:2]
    u_mask = torch.zeros_like(attn[1:2])
    inference_context = _mmjit_context(u_text, u_mask, mask_token)

    assert torch.equal(trained_context[1:2], inference_context)
    # Pooled path: vec = t_vec + pooled_embedder(context.mean(dim=1)).
    assert torch.equal(trained_context[1:2].mean(dim=1),
                       inference_context.mean(dim=1))
    # The kept row is still its own conditional forward.
    assert torch.equal(trained_context[0:1],
                       _mmjit_context(text[0:1], attn[0:1], mask_token))


def test_the_text_tensor_is_left_alone_and_that_is_exact():
    """Zeroing the text as well would be a different tensor with the SAME
    meaning -- the forward discards it either way. Asserted so the cheaper
    choice is a proven equivalence, not an omission."""
    torch.manual_seed(11)
    text = torch.randn(1, 4, 6)
    attn = torch.ones(1, 4)
    mask_token = torch.randn(1, 1, 6)
    _, zeroed = apply_cfg_null_collated(text, attn, torch.tensor([True]))
    assert torch.equal(_mmjit_context(text, zeroed, mask_token),
                       _mmjit_context(torch.zeros_like(text), zeroed, mask_token))


def test_the_model_line_this_parity_argument_rests_on_is_unchanged():
    source = (BACKEND / "core" / "models" / "minit2i" / "vendor"
              / "mmjit.py").read_text(encoding="utf-8")
    assert _MMJIT_CONTEXT_LINE in source


def test_the_inference_null_is_still_a_zeroed_mask_on_the_same_text():
    source = (BACKEND / "core" / "models" / "minit2i"
              / "minit2i_pipeline_ops.py").read_text(encoding="utf-8")
    assert "u_text, u_mask = text, torch.zeros_like(mask)" in source


# ---------------------------------------------------------------------------
# Handler wiring
# ---------------------------------------------------------------------------

def test_the_handler_declares_the_collated_stage_and_implements_its_hook():
    handler = MiniT2IArchHandler.__new__(MiniT2IArchHandler)
    assert handler.cfg_null_stage == "collated"

    text = torch.randn(2, 3, 4)
    attn = torch.ones(2, 3)
    _, out = handler.apply_cfg_null_collated(None, text, attn,
                                             torch.tensor([True, False]))
    assert not out[0].any()

    with pytest.raises(NotImplementedError):
        handler.encode_prompt_cfg_null(None, "a prompt")


def test_the_handler_hands_the_rewritten_mask_to_the_forward(monkeypatch):
    from core.training.arch.base_arch import TrainStepContext
    from core.training.ops import minit2i_ops

    seen = {}

    def _capture(trainer, **kwargs):
        seen.update(kwargs)
        return torch.zeros(()), 0.0, 0.0

    monkeypatch.setattr(minit2i_ops, "train_step", _capture)

    text = torch.randn(2, 3, 4)
    attn = torch.ones(2, 3)
    handler = MiniT2IArchHandler.__new__(MiniT2IArchHandler)
    handler.train_step(None, TrainStepContext(
        latents=torch.zeros(2, 3, 8, 8), text_embeddings=text,
        attention_mask=attn, cfg_drop_mask=torch.tensor([True, False])))

    assert seen["text_embeds"] is text
    assert not seen["attention_mask"][0].any()
    assert seen["attention_mask"][1].all()
    assert torch.equal(attn, torch.ones(2, 3))


def test_no_drop_mask_leaves_the_conditioning_objects_alone(monkeypatch):
    from core.training.arch.base_arch import TrainStepContext
    from core.training.ops import minit2i_ops

    seen = {}
    monkeypatch.setattr(minit2i_ops, "train_step",
                        lambda trainer, **kw: (seen.update(kw),
                                               torch.zeros(()), 0.0, 0.0)[1:])

    text, attn = torch.randn(1, 2, 3), torch.ones(1, 2)
    handler = MiniT2IArchHandler.__new__(MiniT2IArchHandler)
    handler.train_step(None, TrainStepContext(
        latents=torch.zeros(1, 3, 8, 8), text_embeddings=text,
        attention_mask=attn, cfg_drop_mask=None))
    assert seen["text_embeds"] is text
    assert seen["attention_mask"] is attn


def test_the_handler_applies_the_hook_before_the_forward():
    """train_step must consume the ALREADY-RESOLVED label; the ops body no
    longer has a rate or a draw of its own."""
    source = (BACKEND / "core" / "training" / "arch"
              / "minit2i.py").read_text(encoding="utf-8")
    assert "ctx.cfg_drop_mask" in source
    assert "apply_cfg_null_collated" in source

    ops = (BACKEND / "core" / "training" / "ops"
           / "minit2i_ops.py").read_text(encoding="utf-8")
    tree = ast.parse(ops)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "train_step":
            body = ast.dump(node)
            assert "label_drop" not in body
            assert "CFG_UNCOND_DROP_DEFAULTS_BY_ARCH" not in body
            return
    raise AssertionError("minit2i_ops.train_step not found")
