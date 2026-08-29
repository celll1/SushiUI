"""Delivery item 4: the Lens collated null rewrite.

local/strategy/cfg_null_alignment/IMPLEMENTATION_STRATEGY.md section 6.2.
Everything here runs on synthetic tensors and a randomly initialised tiny
transformer; no checkpoint is loaded.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/cfg_null_lens_test.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.arch.lens import LensArchHandler  # noqa: E402
from core.training.ops import lens_ops  # noqa: E402
from core.training.ops.lens_ops import apply_cfg_null_collated  # noqa: E402

_PIPELINE_OPS = (BACKEND / "core" / "models" / "lens"
                 / "lens_pipeline_ops.py").read_text(encoding="utf-8")
_LENS_OPS_SOURCE = (BACKEND / "core" / "training" / "ops"
                    / "lens_ops.py").read_text(encoding="utf-8")

# The two lines of `lens_pipeline_ops.encode_prompt`'s empty-negative branch the
# aligned null has to agree with. Pinned as source so the parity argument below
# fails loudly if inference ever builds its uncond row differently.
_INFERENCE_NULL_FEATURES = "neg_features = [f.new_zeros(f.shape) for f in pos_features]"
_INFERENCE_NULL_MASK = "neg_mask = torch.zeros_like(pos_mask, dtype=torch.bool)"


def _collated(pos_features, pos_mask):
    """A batch row in the shape the training path collates.

    ``lens_ops.encode_prompt`` stacks the per-layer ``[1, L, D]`` conditional
    features into ``[1, num_layers, L, D]``; the loop concatenates those on
    axis 0.
    """
    stacked = torch.stack([f.squeeze(0) for f in pos_features], dim=0)
    return stacked.unsqueeze(0), pos_mask


def _positive(num_layers=3, seq_len=5, dim=6, seed=0):
    torch.manual_seed(seed)
    features = [torch.randn(1, seq_len, dim) for _ in range(num_layers)]
    mask = torch.ones(1, seq_len, dtype=torch.bool)
    mask[:, -2:] = False  # a realistic padded tail
    return features, mask


# ---------------------------------------------------------------------------
# Representation equality with lens_pipeline_ops.encode_prompt (strategy §6.2)
# ---------------------------------------------------------------------------

def test_the_dropped_row_is_the_inference_empty_negative_row():
    """The rewritten row equals what `encode_prompt` puts on its uncond row when
    every negative is blank, built here from the SAME literal expressions."""
    pos_features, pos_mask = _positive()
    features, mask = _collated(pos_features, pos_mask)

    out_features, out_mask = apply_cfg_null_collated(
        features, mask, torch.tensor([True]))

    # encode_prompt's `all(not neg.strip() ...)` branch, verbatim.
    neg_features = [f.new_zeros(f.shape) for f in pos_features]
    neg_mask = torch.zeros_like(pos_mask, dtype=torch.bool)
    expected_features, expected_mask = _collated(neg_features, neg_mask)

    assert torch.equal(out_features, expected_features)
    assert torch.equal(out_mask, expected_mask)


def test_only_the_selected_rows_are_rewritten():
    kept_features, kept_mask = _positive(seed=1)
    dropped_features, dropped_mask = _positive(seed=2)
    features = torch.cat([_collated(kept_features, kept_mask)[0],
                          _collated(dropped_features, dropped_mask)[0]], dim=0)
    mask = torch.cat([kept_mask, dropped_mask], dim=0)

    out_features, out_mask = apply_cfg_null_collated(
        features, mask, torch.tensor([False, True]))

    assert torch.equal(out_features[0], features[0])
    assert torch.equal(out_mask[0], mask[0])
    assert not out_features[1].any()
    assert not out_mask[1].any()


def test_the_inference_expressions_this_parity_rests_on_are_unchanged():
    assert _INFERENCE_NULL_FEATURES in _PIPELINE_OPS
    assert _INFERENCE_NULL_MASK in _PIPELINE_OPS


def test_the_positive_length_is_what_inference_aligns_to():
    """`_align_text_features` pads the shorter side up to the longer one; the
    empty negative is already built at the positive's length, so the rewrite
    keeping L is the aligned behaviour, not an approximation."""
    assert "target = max(seq_pos, seq_neg)" in _PIPELINE_OPS
    assert "if cur == target:\n            return features" in _PIPELINE_OPS


class _StubTokenizer:
    """Enough of the GPT-OSS tokenizer for `_build_chat_inputs`."""

    def apply_chat_template(self, conversation, tokenize=False,
                            add_generation_prompt=False):
        user = next(m["content"] for m in conversation if m["role"] == "user")
        # _build_chat_inputs keeps everything before <|return|>.
        return f"{user}<|return|>tail"

    def __call__(self, rendered, **kwargs):
        # One row per prompt: a fixed prefix long enough to survive
        # DEFAULT_TXT_OFFSET, then one active token per prompt character.
        from core.models.lens.lens_pipeline_ops import DEFAULT_TXT_OFFSET
        lengths = [DEFAULT_TXT_OFFSET + max(1, len(r)) for r in rendered]
        width = max(lengths)
        ids = torch.zeros(len(rendered), width, dtype=torch.long)
        mask = torch.zeros(len(rendered), width, dtype=torch.long)
        for i, n in enumerate(lengths):
            ids[i, :n] = torch.arange(1, n + 1)
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _StubTextEncoder:
    def encode_layers(self, input_ids, attn_mask):
        torch.manual_seed(int(input_ids.sum()) % 9973)
        b, s = input_ids.shape
        return [torch.randn(b, s, 6) for _ in range(3)]


def test_the_rewrite_reproduces_encode_prompts_own_uncond_row():
    """End to end through the real `lens_pipeline_ops.encode_prompt`, on a stub
    tokenizer/encoder: its uncond row (blank negative) must equal what the
    training rewrite makes of its cond row, collated the training way."""
    from core.models.lens.lens_pipeline_ops import encode_prompt

    features, mask = encode_prompt(
        text_encoder=_StubTextEncoder(), tokenizer=_StubTokenizer(),
        prompt="a prompt", negative_prompt="", device="cpu",
        dtype=torch.float32, max_length=512,
    )
    # Training collation of the CONDITIONAL row (lens_ops.encode_prompt).
    cond = torch.stack([f[0] for f in features], dim=0).unsqueeze(0)
    cond_mask = mask[0:1]
    assert cond.shape[2] > 0 and cond_mask.any()

    out_features, out_mask = apply_cfg_null_collated(
        cond, cond_mask, torch.tensor([True]))

    # The same collation of encode_prompt's UNCOND row.
    expected = torch.stack([f[1] for f in features], dim=0).unsqueeze(0)
    assert torch.equal(out_features, expected)
    assert torch.equal(out_mask, mask[1:2])
    assert out_features.dtype is expected.dtype
    assert out_mask.dtype is mask.dtype


# ---------------------------------------------------------------------------
# Shape and object contracts
# ---------------------------------------------------------------------------

def test_the_rewrite_is_out_of_place():
    """Both tensors belong to the assembled batch and are handed to every MNT
    iteration; an in-place write would leak one iteration's null forward."""
    features = torch.randn(2, 3, 5, 6)
    mask = torch.ones(2, 5, dtype=torch.bool)
    features_before, mask_before = features.clone(), mask.clone()

    out_features, out_mask = apply_cfg_null_collated(
        features, mask, torch.tensor([True, False]))

    assert out_features is not features
    assert out_mask is not mask
    assert torch.equal(features, features_before)
    assert torch.equal(mask, mask_before)


def test_the_batch_geometry_is_unchanged():
    """No truncation, no re-padding: L and num_layers are the batch's own."""
    features = torch.randn(2, 4, 7, 6)
    mask = torch.ones(2, 7, dtype=torch.bool)
    out_features, out_mask = apply_cfg_null_collated(
        features, mask, torch.tensor([True, True]))
    assert out_features.shape == (2, 4, 7, 6)
    assert out_mask.shape == (2, 7)
    assert out_features.dtype is features.dtype
    assert out_mask.dtype is mask.dtype


def test_an_empty_selection_returns_the_batch_untouched():
    features = torch.randn(2, 3, 4, 6)
    mask = torch.ones(2, 4, dtype=torch.bool)
    for drop in (None, torch.zeros(2, dtype=torch.bool)):
        out_features, out_mask = apply_cfg_null_collated(features, mask, drop)
        assert out_features is features
        assert out_mask is mask


def test_a_cpu_label_rewrites_conditioning_on_another_device():
    """The mask is a CPU label by contract; indexing must not depend on it
    sharing the conditioning's device."""
    features = torch.randn(2, 3, 4, 6)
    mask = torch.ones(2, 4, dtype=torch.bool)
    drop = torch.tensor([True, False], device="cpu")
    out_features, out_mask = apply_cfg_null_collated(features, mask, drop)
    assert not out_features[0].any()
    assert out_mask[1].all()


# ---------------------------------------------------------------------------
# The structural consequence, on the vendored transformer (strategy §7)
# ---------------------------------------------------------------------------

def _tiny_transformer(seed=3):
    """A randomly initialised LensTransformer2DModel at toy geometry.

    Weightless in the sense that matters here: no checkpoint is read. The
    structural claim is about the mask, not about trained values.
    """
    from core.models.lens.vendor.transformer import LensTransformer2DModel

    torch.manual_seed(seed)
    model = LensTransformer2DModel(
        patch_size=1, in_channels=8, out_channels=4, num_layers=2,
        attention_head_dim=8, num_attention_heads=2, enc_hidden_dim=12,
        axes_dims_rope=(2, 2, 4), multi_layer_encoder_feature=True,
        selected_layer_index=(0, 1),
    )
    return model.eval()


def _forward(model, text_seed, mask):
    torch.manual_seed(101)
    hidden = torch.randn(1, 4, 8)
    timestep = torch.tensor([0.5])
    torch.manual_seed(text_seed)
    text = [torch.randn(1, 5, 12), torch.randn(1, 5, 12)]
    with torch.no_grad():
        return model(hidden_states=hidden, encoder_hidden_states=text,
                     encoder_hidden_states_mask=mask, timestep=timestep,
                     img_shapes=[(1, 2, 2)])


def test_an_all_false_text_mask_makes_the_output_independent_of_text_values():
    """Ran on the real vendored transformer at toy geometry (no weights loaded).

    With every text key masked, image queries can only consume image values and
    the output head reads only the image stream -- so the null row's text
    FEATURE values are unobservable, which is why zeroing them is exact rather
    than merely conventional."""
    model = _tiny_transformer()
    off = torch.zeros(1, 5, dtype=torch.bool)
    a = _forward(model, 11, off)
    b = _forward(model, 22, off)
    assert torch.equal(a, b)


def test_the_same_test_separates_when_the_text_mask_is_live():
    """Negative control: the equality above is the mask's doing, not the toy
    geometry collapsing every forward to the same tensor."""
    model = _tiny_transformer()
    on = torch.ones(1, 5, dtype=torch.bool)
    a = _forward(model, 11, on)
    b = _forward(model, 22, on)
    assert not torch.allclose(a, b)


def test_the_rewritten_row_and_a_zero_text_row_agree_through_the_model():
    model = _tiny_transformer()
    off = torch.zeros(1, 5, dtype=torch.bool)
    torch.manual_seed(101)
    hidden = torch.randn(1, 4, 8)
    timestep = torch.tensor([0.5])
    torch.manual_seed(31)
    features = torch.randn(1, 2, 5, 12)
    mask = torch.ones(1, 5, dtype=torch.bool)

    out_features, out_mask = apply_cfg_null_collated(
        features, mask, torch.tensor([True]))
    assert torch.equal(out_mask, off)

    with torch.no_grad():
        rewritten = model(
            hidden_states=hidden,
            encoder_hidden_states=[out_features[:, i] for i in range(2)],
            encoder_hidden_states_mask=out_mask, timestep=timestep,
            img_shapes=[(1, 2, 2)])
        original = model(
            hidden_states=hidden,
            encoder_hidden_states=[features[:, i] for i in range(2)],
            encoder_hidden_states_mask=out_mask, timestep=timestep,
            img_shapes=[(1, 2, 2)])
    assert torch.equal(rewritten, original)


# ---------------------------------------------------------------------------
# Handler and train_step wiring
# ---------------------------------------------------------------------------

def test_the_handler_declares_the_collated_stage_and_implements_its_hook():
    handler = LensArchHandler.__new__(LensArchHandler)
    assert handler.cfg_null_stage == "collated"

    features = torch.randn(2, 3, 4, 6)
    mask = torch.ones(2, 4, dtype=torch.bool)
    out_features, out_mask = handler.apply_cfg_null_collated(
        None, features, mask, torch.tensor([True, False]))
    assert not out_features[0].any()
    assert not out_mask[0].any()
    assert out_mask[1].all()

    with pytest.raises(NotImplementedError):
        handler.encode_prompt_cfg_null(None, "a prompt")


class _RecordingTransformer(torch.nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.seen = {}

    def forward(self, **kwargs):
        self.seen = kwargs
        return torch.zeros_like(kwargs["hidden_states"])


class _StubTrainer:
    device = torch.device("cpu")
    training_dtype = torch.float32
    mixed_precision = False
    timestep_sampler = None

    def __init__(self):
        self.transformer = _RecordingTransformer()
        self.arch = LensArchHandler.__new__(LensArchHandler)


def _train_step(cfg_drop_mask):
    trainer = _StubTrainer()
    torch.manual_seed(5)
    latents = torch.randn(2, 4, 128)
    features = torch.randn(2, 2, 5, 6)
    mask = torch.ones(2, 5, dtype=torch.bool)
    lens_ops.train_step(
        trainer, latents=latents, encoder_features=features,
        encoder_mask=mask, timesteps=torch.tensor([0.3, 0.7]),
        latent_h=2, latent_w=2, cfg_drop_mask=cfg_drop_mask)
    return trainer, features, mask


def test_train_step_hands_the_rewritten_conditioning_to_the_forward():
    trainer, features, mask = _train_step(torch.tensor([True, False]))
    seen = trainer.transformer.seen
    assert not seen["encoder_hidden_states_mask"][0].any()
    assert seen["encoder_hidden_states_mask"][1].all()
    for layer in seen["encoder_hidden_states"]:
        assert not layer[0].any()
        assert layer[1].any()
    # The batch's own tensors are untouched.
    assert features.any() and mask.all()


def test_train_step_without_a_label_passes_the_conditioning_through():
    trainer, features, _ = _train_step(None)
    seen = trainer.transformer.seen
    assert seen["encoder_hidden_states_mask"].all()
    assert torch.equal(seen["encoder_hidden_states"][0], features[:, 0])


def test_the_rewrite_runs_after_the_moves_and_before_the_layer_list():
    """Strategy §6.2 places it after the device/dtype moves (which can be
    identity no-ops handing back the batch's own tensors) and before the
    per-layer conditioning list is built."""
    lines = _LENS_OPS_SOURCE.splitlines()
    body = lines[next(i for i, line in enumerate(lines)
                      if line.startswith("def train_step(")):]
    move = next(i for i, line in enumerate(body)
                if "encoder_mask = encoder_mask.to(device=trainer.device" in line)
    apply_ = next(i for i, line in enumerate(body)
                  if "apply_cfg_null_collated(" in line)
    build = next(i for i, line in enumerate(body)
                 if "encoder_hidden_states_list = [" in line)
    assert move < apply_ < build


def test_the_lens_forward_is_handed_the_batch_label():
    """The one place base_trainer builds the Lens TrainStepContext must carry
    the per-batch label, or the hook is never reached in production."""
    source = (BACKEND / "core" / "training"
              / "base_trainer.py").read_text(encoding="utf-8")
    marker = "# mnt_attention_mask:  [B, L] encoder mask"
    lens_ctx = source.index(marker)
    next_arch = source.index("elif self.is_ideogram4:", lens_ctx)
    block = source[lens_ctx:next_arch]
    assert "encoder_features=mnt_text_embeddings" in block
    assert "cfg_drop_mask=cfg_drop_mask" in block
