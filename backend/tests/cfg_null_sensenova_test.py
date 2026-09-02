"""Delivery item 5: SenseNova's encode-stage null, with positional parity.

local/strategy/cfg_null_alignment/IMPLEMENTATION_STRATEGY.md section 6.3.
Everything here runs on the REAL vendor query/index builders and the real
`sensenova_ops` encode path, driven by a stub tokenizer and a stub prefix
forward; no checkpoint is read and no CUDA is touched.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/cfg_null_sensenova_test.py -v
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from api.cfg_null_resolver import (  # noqa: E402
    CFG_KEY, LEGACY_KEY, REFERENCE_IMAGES_KEY, resolve_and_check,
)
from api.error_handlers import ValidationError  # noqa: E402
from core.models.sensenova.vendor.conversation import get_conv_template  # noqa: E402
from core.models.sensenova.vendor.modeling_neo_chat import NEOChatModel  # noqa: E402
from core.models.sensenova.vendor.utils import SYSTEM_MESSAGE_FOR_GEN  # noqa: E402
from core.training.arch.sensenova import SenseNovaArchHandler  # noqa: E402
from core.training.ops import sensenova_ops  # noqa: E402

_PIPELINE_OPS = (BACKEND / "core" / "models" / "sensenova"
                 / "sensenova_pipeline_ops.py").read_text(encoding="utf-8")
_SENSENOVA_OPS = (BACKEND / "core" / "training" / "ops"
                  / "sensenova_ops.py").read_text(encoding="utf-8")
_BASE_TRAINER = (BACKEND / "core" / "training"
                 / "base_trainer.py").read_text(encoding="utf-8")

#: The text-only uncond arm of `sensenova_pipeline_ops.encode_prompt`, pinned as
#: source. Everything below rests on this being what inference builds.
_INFERENCE_UNCOND_QUERY = (
    'query_uncond = transformer._build_t2i_query(negative_prompt, '
    'append_text="<img>") if needs_cfg else None')
_INFERENCE_UNCOND_STRIP = 'negative_prompt = (negative_prompt or "").strip()'
_INFERENCE_UNCOND_INDEXES = (
    "indexes_image_uncond = transformer._build_t2i_image_indexes(\n"
    "                    token_h, token_w, indexes_uncond.shape[1], "
    "device=input_ids_uncond.device)")

_TEMPLATE = "neo1_0"
_LAYERS = 3


class _StubTokenizer:
    """One token per character, so a query's length is its own string length.

    Deterministic and monotone in the query, which is all the length claims here
    need; the real tokenizer's vocabulary is irrelevant to whether the training
    null and the inference null are the SAME string.
    """

    def __init__(self):
        self.queries = []

    def __call__(self, query, return_tensors=None):
        self.queries.append(query)
        ids = torch.arange(1, len(query) + 1, dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids}


class _StubTransformer:
    """The vendor query/index/text-input builders, bound to a stub `self`.

    `_build_t2i_query`, `_build_t2i_text_inputs` and `_build_t2i_image_indexes`
    are the REAL vendor methods (`NEOChatModel`); only the prefix FORWARD is
    stubbed, since that is the part that needs weights.
    """

    _build_t2i_query = NEOChatModel._build_t2i_query
    _build_t2i_text_inputs = NEOChatModel._build_t2i_text_inputs
    _build_t2i_image_indexes = NEOChatModel._build_t2i_image_indexes

    def __init__(self):
        self.template = _TEMPLATE
        # modeling_neo_chat.py:261-262, the model's own two lines.
        self.system_message = get_conv_template(_TEMPLATE).system_message
        self.device = torch.device("cpu")
        self.language_model = SimpleNamespace(
            model=SimpleNamespace(layers=[object()] * _LAYERS))
        self.prefix_lengths = []

    def _t2i_prefix_forward(self, input_ids, indexes, attention_mask):
        length = int(input_ids.shape[1])
        self.prefix_lengths.append(length)
        layers = [
            sensenova_ops._TrainingPrefixLayer(
                torch.zeros(1, 2, length, 4), torch.zeros(1, 2, length, 4))
            for _ in range(_LAYERS)
        ]
        return sensenova_ops._TrainingPrefixCache(layers), None

    def _it2i_prefix_forward(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("the text-only null must not take the it2i route")


class _StubTrainer:
    train_text_encoder = False
    sensenova_phase_evictor = None
    sensenova_four_phase = None

    def __init__(self):
        self.transformer = _StubTransformer()
        self.tokenizer = _StubTokenizer()
        self.arch = SenseNovaArchHandler.__new__(SenseNovaArchHandler)


def _inference_uncond_query(transformer, negative_prompt=None):
    """Rebuild inference's text-only uncond query from its own two lines."""
    negative_prompt = (negative_prompt or "").strip()
    return transformer._build_t2i_query(negative_prompt, append_text="<img>")


def _training_conditional_query(transformer, prompt):
    """`_build_prefix_inputs`'s text-only CONDITIONAL arm, for contrast."""
    return transformer._build_t2i_query(
        prompt, system_message=SYSTEM_MESSAGE_FOR_GEN,
        append_text="<think>\n\n</think>\n\n<img>")


# ---------------------------------------------------------------------------
# The query string (strategy §6.3)
# ---------------------------------------------------------------------------

def test_the_inference_expressions_this_parity_rests_on_are_unchanged():
    assert _INFERENCE_UNCOND_QUERY in _PIPELINE_OPS
    assert _INFERENCE_UNCOND_STRIP in _PIPELINE_OPS
    assert _INFERENCE_UNCOND_INDEXES in _PIPELINE_OPS


def test_the_null_query_equals_inferences_own_uncond_query():
    trainer = _StubTrainer()
    sensenova_ops._build_prefix_inputs(
        trainer, trainer.transformer, "a caption the null must ignore", [],
        cfg_null=True)
    built = trainer.tokenizer.queries[-1]
    assert built == _inference_uncond_query(trainer.transformer)


def test_the_null_query_carries_no_system_block_and_no_think_suffix():
    """The mismatch this closes, stated as the two things that differ: training
    passed SYSTEM_MESSAGE_FOR_GEN and a think suffix on BOTH arms, while the
    neo1_0 template's own system message is empty and its formatter then emits
    no system block at all."""
    trainer = _StubTrainer()
    sensenova_ops._build_prefix_inputs(
        trainer, trainer.transformer, "a cat", [], cfg_null=True)
    null_query = trainer.tokenizer.queries[-1]

    assert "system" not in null_query
    assert SYSTEM_MESSAGE_FOR_GEN not in null_query
    assert "<think>" not in null_query
    assert null_query.endswith("<img>")
    assert get_conv_template(_TEMPLATE).system_message == ""


def test_the_null_ignores_the_caption_it_is_given():
    trainer = _StubTrainer()
    for prompt in ("", "a cat", "an entirely different caption"):
        sensenova_ops._build_prefix_inputs(
            trainer, trainer.transformer, prompt, [], cfg_null=True)
    assert len(set(trainer.tokenizer.queries)) == 1


def test_an_empty_caption_is_not_the_null():
    """The reason this is an encode-stage null and not caption dropout: the
    conditional encoding of "" still carries the system block and the think
    suffix, so it is a different token sequence and a different prefix length."""
    trainer = _StubTrainer()
    conditional_empty = _training_conditional_query(trainer.transformer, "")
    null = _inference_uncond_query(trainer.transformer)
    assert conditional_empty != null
    assert len(conditional_empty) > len(null)


# ---------------------------------------------------------------------------
# Positional parity: the image indexes (strategy §6.3, audit row 7)
# ---------------------------------------------------------------------------

def test_the_prefix_length_is_the_nulls_own():
    trainer = _StubTrainer()
    null = sensenova_ops.encode_prompt(trainer, "a cat", cfg_null=True)
    conditional = sensenova_ops.encode_prompt(trainer, "a cat")

    expected_null = len(_inference_uncond_query(trainer.transformer))
    assert null.text_length == expected_null
    assert conditional.text_length > null.text_length
    # The prefix K/V forward saw the null tokens, not the conditional ones.
    assert trainer.transformer.prefix_lengths == [expected_null,
                                                  conditional.text_length]


def test_the_image_indexes_are_built_from_the_null_prefix_length():
    """`_build_t2i_image_indexes` writes the prefix length into EVERY image
    token's t coordinate, so aligning the K/V cache and leaving the indexes
    would still train a different field."""
    trainer = _StubTrainer()
    null = sensenova_ops.encode_prompt(trainer, "a cat", cfg_null=True)
    conditional = sensenova_ops.encode_prompt(trainer, "a cat")

    token_h, token_w = 2, 3
    null_indexes = trainer.transformer._build_t2i_image_indexes(
        token_h, token_w, null.text_length, device="cpu")
    cond_indexes = trainer.transformer._build_t2i_image_indexes(
        token_h, token_w, conditional.text_length, device="cpu")

    assert torch.equal(null_indexes[0],
                       torch.full((token_h * token_w,), null.text_length))
    assert not torch.equal(null_indexes[0], cond_indexes[0])
    # h/w are positional only and must not move.
    assert torch.equal(null_indexes[1], cond_indexes[1])
    assert torch.equal(null_indexes[2], cond_indexes[2])


def test_train_step_builds_the_image_indexes_from_the_prefix_it_was_given():
    """The single channel that carries the length. `train_step` needs no CFG
    knowledge at all: the null prefix arrives carrying its own text_length."""
    assert ("indexes = transformer._build_t2i_image_indexes(\n"
            "        token_h, token_w, prefix.text_length, device=device\n"
            "    )") in _SENSENOVA_OPS
    assert "cfg_null" not in _SENSENOVA_OPS[_SENSENOVA_OPS.index("def train_step("):]


# ---------------------------------------------------------------------------
# The two encode call sites see the same label (strategy §6.3)
# ---------------------------------------------------------------------------

def test_the_batch_assembly_arm_passes_the_items_own_label():
    arm = _BASE_TRAINER[_BASE_TRAINER.index("if self.is_sensenova:\n"
                                            "                            # References enter"):]
    arm = arm[:arm.index("elif text_encoding_mode")]
    assert "cfg_null=(cfg_drop_mask is not None" in arm
    assert "bool(cfg_drop_mask[item_index])" in arm


def test_the_mnt_reencode_arm_passes_this_iterations_own_label():
    """Not the batch-level `cfg_drop_mask` (mnt_index 0's draw): the MNT loop
    binds `mnt_cfg_drop_mask` from `cfg_drop_mask_for_mnt` at the top of every
    iteration, and that -- not the shared batch mask -- is what every
    downstream consumer of the label reads (cfg_null_per_mnt_test.py)."""
    call = _BASE_TRAINER[_BASE_TRAINER.index(
        ") = self._sensenova_mnt_conditioning("):]
    call = call[:call.index(")\n")]
    assert "cfg_null=(mnt_cfg_drop_mask is not None" in call
    assert "bool(mnt_cfg_drop_mask[0])" in call


def test_the_mnt_reencode_forwards_the_label_into_the_encode():
    """Behavioural, not source: the re-encode must not silently rebuild the
    conditional prefix, which would make MNT iteration 0 null and the rest
    conditional for the same image."""
    from core.training.base_trainer import BaseTrainer

    seen = []

    class _MntTrainer:
        train_text_encoder = True
        sensenova_four_phase = None
        _sensenova_mnt_conditioning = BaseTrainer._sensenova_mnt_conditioning

        def encode_caption(self, caption, requires_grad=False, cfg_null=False):
            seen.append((caption, cfg_null))
            return "rebuilt", None

    trainer = _MntTrainer()
    trainer._sensenova_mnt_conditioning(
        "assembly", captions=["a cat"], mnt_index=1, cfg_null=True)
    trainer._sensenova_mnt_conditioning(
        "assembly", captions=["a cat"], mnt_index=1, cfg_null=False)
    assert seen == [("a cat", True), ("a cat", False)]


# ---------------------------------------------------------------------------
# Frozen understanding branch + per-MNT redraw (cfg_uncond_drop_per_mnt)
# ---------------------------------------------------------------------------


def _frozen_trainer(assembly_cfg_null, rebuilt="rebuilt"):
    from core.training.base_trainer import BaseTrainer

    class _FrozenTrainer:
        train_text_encoder = False
        sensenova_four_phase = None
        _sensenova_mnt_conditioning = BaseTrainer._sensenova_mnt_conditioning
        _sensenova_prefix_cfg_null = assembly_cfg_null
        _sensenova_alt_cfg_null_prefix = None

        def __init__(self):
            self.encode_calls = []

        def encode_caption(self, caption, requires_grad=False, cfg_null=False):
            self.encode_calls.append((caption, requires_grad, cfg_null))
            return rebuilt, None

    return _FrozenTrainer()


def test_frozen_branch_reuses_the_assembly_prefix_when_the_label_matches():
    trainer = _frozen_trainer(assembly_cfg_null=False)
    assembly_prefix = object()
    result = trainer._sensenova_mnt_conditioning(
        assembly_prefix, captions=["a cat"], mnt_index=1, cfg_null=False)
    assert result[3] is assembly_prefix
    assert trainer.encode_calls == []


def test_frozen_branch_rebuilds_frozen_when_the_label_differs():
    trainer = _frozen_trainer(assembly_cfg_null=False)
    assembly_prefix = object()
    result = trainer._sensenova_mnt_conditioning(
        assembly_prefix, captions=["a cat"], mnt_index=1, cfg_null=True)
    assert result[3] == "rebuilt"
    assert trainer.encode_calls == [("a cat", False, True)]


def test_frozen_branch_memoizes_the_alternate_label_within_a_batch():
    """At most one extra build per label per batch: iterations 1 and 3 both
    want the alternate label; only iteration 1 pays for it."""
    trainer = _frozen_trainer(assembly_cfg_null=False)
    assembly_prefix = object()
    trainer._sensenova_mnt_conditioning(
        assembly_prefix, captions=["a cat"], mnt_index=1, cfg_null=True)
    trainer._sensenova_mnt_conditioning(
        assembly_prefix, captions=["a cat"], mnt_index=2, cfg_null=False)
    trainer._sensenova_mnt_conditioning(
        assembly_prefix, captions=["a cat"], mnt_index=3, cfg_null=True)
    assert trainer.encode_calls == [("a cat", False, True)]


def test_frozen_branch_mnt_index_zero_never_rebuilds_even_on_a_mismatched_label():
    """mnt_index 0 always keeps the assembly prefix -- a mismatched cfg_null
    here would mean the caller mixed up which mask belongs to iteration 0."""
    trainer = _frozen_trainer(assembly_cfg_null=False)
    assembly_prefix = object()
    result = trainer._sensenova_mnt_conditioning(
        assembly_prefix, captions=["a cat"], mnt_index=0, cfg_null=True)
    assert result[3] is assembly_prefix
    assert trainer.encode_calls == []


def test_the_label_is_drawn_before_the_prefix_encode():
    """The structural fix phase 2 left open: the shared draw happened AFTER
    batch assembly, but SenseNova's prefix is built DURING it."""
    draw = _BASE_TRAINER.index("cfg_drop_mask = self.sample_cfg_drop_mask(len(batch))")
    loop = _BASE_TRAINER.index("for item_index, (item, dataset) in enumerate(batch):")
    encode = _BASE_TRAINER.index("prefix, _ = self.encode_caption(", loop)
    assemble = _BASE_TRAINER.index("_collate_sensenova_b1_prefix(\n", loop)
    assert draw < loop < encode < assemble


def test_the_label_is_reindexed_by_the_latent_size_filter_not_redrawn():
    """The filter drops items and re-indexes every per-item list; a label that
    kept the pre-filter order would pair item k's decision with item k+1."""
    filtered = _BASE_TRAINER[_BASE_TRAINER.index("if len(valid_indices) < len(latents_list):"):]
    filtered = filtered[:filtered.index("# Skip batch if no valid latents remain")]
    assert "cfg_drop_mask = cfg_drop_mask[valid_indices]" in filtered
    assert _BASE_TRAINER.count("self.sample_cfg_drop_mask(") == 1


def test_the_batch_size_guard_is_a_check_and_not_a_second_draw():
    loop = _BASE_TRAINER.index("for item_index, (item, dataset) in enumerate(batch):")
    guard = _BASE_TRAINER[_BASE_TRAINER.index("batch_size = latents.shape[0]", loop):]
    guard = guard[:guard.index("MNT loop: Process same batch")]
    assert "cfg_drop_mask.numel() != batch_size" in guard
    assert "sample_cfg_drop_mask" not in guard


# ---------------------------------------------------------------------------
# The handler contract
# ---------------------------------------------------------------------------

def test_the_handler_declares_the_encode_stage_and_implements_its_hook():
    handler = SenseNovaArchHandler.__new__(SenseNovaArchHandler)
    assert handler.cfg_null_stage == "encode"
    with pytest.raises(NotImplementedError):
        handler.apply_cfg_null_collated(None, None, None, None)


def test_the_hook_builds_the_null_and_ignores_the_prompt():
    trainer = _StubTrainer()
    prefix = trainer.arch.encode_prompt_cfg_null(trainer, "a cat")
    assert prefix.text_length == len(_inference_uncond_query(trainer.transformer))


def test_encode_caption_routes_cfg_null_through_the_hook():
    """One call site, the counterpart of `apply_cfg_null_step`: every other
    handler's `encode_prompt_cfg_null` refuses, so a mis-routed True is an error
    rather than a silently conditional item."""
    from core.training.base_trainer import BaseTrainer
    from core.training.arch.lens import LensArchHandler

    class _CaptionTrainer(_StubTrainer):
        encode_caption = BaseTrainer.encode_caption

    trainer = _CaptionTrainer()
    prefix, auxiliary = trainer.encode_caption("a cat", cfg_null=True)
    assert auxiliary is None
    assert prefix.text_length == len(_inference_uncond_query(trainer.transformer))

    trainer.arch = LensArchHandler.__new__(LensArchHandler)
    with pytest.raises(NotImplementedError):
        trainer.encode_caption("a cat", cfg_null=True)


def test_the_null_is_not_memoized():
    """Excluded from this release (strategy §6.3): a cache is invalid when the
    understanding half is trainable, and the frozen case is unproven."""
    trainer = _StubTrainer()
    first = sensenova_ops.encode_prompt(trainer, "a cat", cfg_null=True)
    second = sensenova_ops.encode_prompt(trainer, "a dog", cfg_null=True)
    assert first.cache is not second.cache
    assert len(trainer.transformer.prefix_lengths) == 2
    assert not re.search(r"lru_cache|_null_prefix_cache", _SENSENOVA_OPS)


# ---------------------------------------------------------------------------
# Reference-conditioned items: refuse (strategy §6.3, recommended first release)
# ---------------------------------------------------------------------------

def _params(**kwargs):
    base = {CFG_KEY: None, LEGACY_KEY: None, REFERENCE_IMAGES_KEY: False,
            "danbooru_aug_enable": False,
            "danbooru_aug_caption_dropout_rate": 0.0}
    base.update(kwargs)
    base.setdefault("_explicit_fields", [k for k in kwargs
                                         if base.get(k) is not None])
    return base


def test_a_reference_conditioned_run_refuses_the_rate():
    with pytest.raises(ValidationError) as exc:
        resolve_and_check(_params(**{CFG_KEY: 0.1,
                                     REFERENCE_IMAGES_KEY: True}),
                          arch="sensenova")
    assert REFERENCE_IMAGES_KEY in exc.value.detail
    # It must say which baseline the feature aligns to.
    assert "img_cfg_scale=1" in exc.value.detail
    assert "text-only" in exc.value.detail


def test_the_refusal_does_not_offer_to_map_references_onto_the_text_null():
    with pytest.raises(ValidationError) as exc:
        resolve_and_check(_params(**{CFG_KEY: 0.1,
                                     REFERENCE_IMAGES_KEY: True}),
                          arch="sensenova")
    remedy = exc.value.detail.lower()
    assert "set use_reference_images to false" in remedy
    assert "remove cfg_uncond_drop_rate" in remedy


def test_the_reference_refusal_does_not_speak_for_other_architectures():
    """MiniT2I resolves a nonzero rate from its own default, and `references +
    MiniT2I` is invalid for reasons this refusal knows nothing about
    (`train_runner._apply_reference_training_contract`). Explaining it in terms
    of SenseNova's img_cond baseline would misattribute the refusal."""
    resolution = resolve_and_check(
        _params(**{REFERENCE_IMAGES_KEY: True}), arch="minit2i")
    assert resolution.rate == 0.1


def test_an_explicit_zero_beside_references_is_not_refused():
    """0.0 disables the mechanism; there is then no null to misalign."""
    resolution = resolve_and_check(
        _params(**{CFG_KEY: 0.0, REFERENCE_IMAGES_KEY: True}),
        arch="sensenova")
    assert resolution.rate == 0.0


def test_the_shipped_default_lets_a_reference_run_start():
    """SenseNova's per-arch default is 0.0, so reference training is unaffected
    by this feature existing."""
    resolution = resolve_and_check(
        _params(**{REFERENCE_IMAGES_KEY: True}), arch="sensenova")
    assert resolution.rate == 0.0
    assert resolution.warnings == []


def test_the_ops_layer_refuses_a_reference_null_even_if_the_gate_were_bypassed():
    trainer = _StubTrainer()
    with pytest.raises(ValueError) as exc:
        sensenova_ops.encode_prompt(
            trainer, "a cat", reference_image_paths=["/nonexistent.png"],
            cfg_null=True)
    assert "text-only" in str(exc.value)


def test_the_refusal_reaches_the_trainer_and_the_train_runner_preflight():
    """Both entry points call `resolve_and_check`, so both refuse; the
    train_runner one runs before the model loads."""
    from core.training.base_trainer import BaseTrainer

    class _Arch:
        name = "sensenova"
        cfg_null_stage = "encode"

    class _Trainer:
        cfg_null_drop_rate = BaseTrainer.cfg_null_drop_rate
        log_prefix = "[cfg-null-test]"

        def __init__(self, config):
            self.config = config
            self.arch = _Arch()

    trainer = _Trainer(_params(**{CFG_KEY: 0.1, REFERENCE_IMAGES_KEY: True}))
    with pytest.raises(ValidationError):
        trainer.cfg_null_drop_rate()

    runner = (BACKEND / "core" / "training"
              / "train_runner.py").read_text(encoding="utf-8")
    preflight = runner[runner.index("def _preflight_cfg_null_caption_conflict("):]
    preflight = preflight[:preflight.index("\ndef ")]
    assert "resolve_and_check(" in preflight


# ---------------------------------------------------------------------------
# Rate 0: nothing changes (strategy §3 rule 2)
# ---------------------------------------------------------------------------

def test_a_zero_rate_draws_no_label_at_all():
    from core.training.base_trainer import BaseTrainer

    class _Trainer:
        sample_cfg_drop_mask = BaseTrainer.sample_cfg_drop_mask

        def __init__(self, rate):
            self._cfg_null_drop_rate_resolved = rate

        cfg_null_drop_rate = BaseTrainer.cfg_null_drop_rate

    assert _Trainer(0.0).sample_cfg_drop_mask(4) is None
    assert _Trainer(None).sample_cfg_drop_mask(4) is None
    assert _Trainer(1.0).sample_cfg_drop_mask(4).all()


def test_without_a_label_the_prefix_is_the_conditional_one():
    trainer = _StubTrainer()
    prefix = sensenova_ops.encode_prompt(trainer, "a cat", cfg_null=False)
    assert trainer.tokenizer.queries[-1] == _training_conditional_query(
        trainer.transformer, "a cat")
    assert prefix.text_length == len(trainer.tokenizer.queries[-1])


def test_the_conditional_path_is_byte_identical_to_before_this_change():
    """The default argument, not merely the default behaviour: `cfg_null=False`
    reaches `_build_prefix_inputs` as the same two arms it always had."""
    inputs_src = _SENSENOVA_OPS[_SENSENOVA_OPS.index("def _build_prefix_inputs("):]
    inputs_src = inputs_src[:inputs_src.index("\ndef _build_trainable_prefix(")]
    assert "cfg_null: bool = False" in inputs_src
    assert inputs_src.index("if cfg_null:") < inputs_src.index("if not ref_images:")
    encode_src = _SENSENOVA_OPS[_SENSENOVA_OPS.index("\ndef encode_prompt("):]
    encode_src = encode_src[:encode_src.index("\ndef vae_encode(")]
    assert "cfg_null: bool = False" in encode_src
    assert encode_src.count("_build_prefix_inputs(") == 3
    assert encode_src.count("cfg_null)") == 3
