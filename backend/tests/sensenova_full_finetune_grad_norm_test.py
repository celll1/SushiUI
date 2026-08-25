"""Which component a full fine-tune's gradient norms are reported under.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/sensenova_full_finetune_grad_norm_test.py -v

THE DEFECT (SENSENOVA_TRAINING_DESIGN.md 13.4, audit item 6)
------------------------------------------------------------
``_calculate_grad_norms``'s full-FT branch buckets a parameter by the MODULE it
walked -- ``unet`` / ``text_encoder`` / ``transformer_original`` -- which is
right for every architecture that keeps one component per module, and wrong for
the one that does not. SenseNova holds both MoT halves inside
``transformer_original`` (``unet`` and ``text_encoder`` are None), so all of it
landed in the ``unet`` bucket and a ``und`` or ``both`` run reported no separate
MoT-Understanding gradient norm. ``_build_component_lr_list`` already returns the
two groups correctly, so the learning rates were never affected; the chart was.

LoRA has never had this problem: ``SenseNovaLoRAAdapter`` registers the
understanding half as ``LORA_COMPONENT_TEXT_ENCODER_1`` at injection time. The
fix gives full FT the same authority -- the adapter that built the optimizer
groups classifies its own parameters (``grad_norm_components``), driven by
``iter_sensenova_lora_targets``, not by a name test on the module path (the
shape dd0b10c7 removed).

NEGATIVE CONTROL
----------------
``MergedBucketTest`` runs the same trees through an adapter with no
``grad_norm_components`` override -- the shipped behaviour -- and records that
und and both report one merged bucket.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sensenova_int8_materialize_test import _Decoder  # noqa: E402

from core.models.sensenova.loader import materialize_int8_decoder_linears  # noqa: E402
from core.models.sensenova.sensenova_lora import (  # noqa: E402
    iter_sensenova_lora_targets,
)
from core.training.adapters.base_adapter import (  # noqa: E402
    LORA_COMPONENT_TEXT_ENCODER_1,
    LORA_COMPONENT_UNET,
)
from core.training.adapters.sensenova_adapter import (  # noqa: E402
    SenseNovaFullParameterAdapter,
)
from core.training.base_trainer import BaseTrainer  # noqa: E402

_BRANCH_FLAGS = {
    "gen": {"train_unet": True, "train_text_encoder": False},
    "und": {"train_unet": False, "train_text_encoder": True},
    "both": {"train_unet": True, "train_text_encoder": True},
}
_TARGETS_PER_BRANCH = 294


class _FullFtTrainer:
    """A trainer shaped as ``sensenova_ops.load_components`` leaves one."""

    log_prefix = "[test]"
    # The two Nones the SenseNova full-FT path exists because of.
    unet = None
    text_encoder = None
    text_encoder_2 = None
    controlnet = None
    _train_vision_encoder = False
    is_sensenova = True

    _calculate_grad_norms = BaseTrainer._calculate_grad_norms
    _full_parameter_grad_components = BaseTrainer._full_parameter_grad_components

    def __init__(self, transformer, adapter):
        self.transformer = transformer
        self.transformer_original = transformer
        self.adapter = adapter


def _trainer(branch: str, *, override: bool = True):
    transformer = _Decoder()
    materialize_int8_decoder_linears(transformer, branch=branch)
    scope = SimpleNamespace(
        transformer=transformer,
        is_sensenova=True,
        unet_lr=1e-6,
        text_encoder_1_lr=None,
        text_encoder_lr=None,
        **_BRANCH_FLAGS[branch],
    )
    adapter = SenseNovaFullParameterAdapter(scope)
    if not override:
        # The shipped behaviour: an adapter with no opinion about its own
        # parameters, which is what every other full-FT adapter is.
        adapter.grad_norm_components = lambda: {}
    trainer = _FullFtTrainer(transformer, adapter)
    for parameter in transformer.parameters():
        parameter.grad = torch.full_like(parameter, 0.5)
    return trainer, transformer


def _norms(trainer):
    total, te, te1, te2, unet, ve = trainer._calculate_grad_norms()
    return {"total": total, "te": te, "te1": te1, "te2": te2, "unet": unet, "ve": ve}


def _half_ids(transformer, half: str):
    return {
        id(parameter)
        for _, _, _, module in iter_sensenova_lora_targets(transformer, branch=half)
        for parameter in module.parameters()
    }


class ComponentMapTest(unittest.TestCase):
    """The adapter classifies exactly the parameters it optimizes."""

    def test_gen_branch_maps_294_parameters_to_the_dit_bucket(self):
        trainer, transformer = _trainer("gen")
        components = trainer.adapter.grad_norm_components()
        self.assertEqual(len(components), _TARGETS_PER_BRANCH)
        self.assertEqual(set(components.values()), {LORA_COMPONENT_UNET})
        self.assertEqual(set(components), _half_ids(transformer, "gen"))

    def test_und_branch_maps_294_parameters_to_text_encoder_1(self):
        trainer, transformer = _trainer("und")
        components = trainer.adapter.grad_norm_components()
        self.assertEqual(len(components), _TARGETS_PER_BRANCH)
        self.assertEqual(set(components.values()), {LORA_COMPONENT_TEXT_ENCODER_1})
        self.assertEqual(set(components), _half_ids(transformer, "und"))

    def test_both_branches_are_classified_separately(self):
        trainer, transformer = _trainer("both")
        components = trainer.adapter.grad_norm_components()
        self.assertEqual(len(components), 2 * _TARGETS_PER_BRANCH)
        gen = {k for k, v in components.items() if v == LORA_COMPONENT_UNET}
        und = {k for k, v in components.items() if v == LORA_COMPONENT_TEXT_ENCODER_1}
        self.assertEqual(gen, _half_ids(transformer, "gen"))
        self.assertEqual(und, _half_ids(transformer, "und"))
        self.assertEqual(gen & und, set())

    def test_the_map_matches_the_lora_adapters_registration(self):
        """The same half is the same component under both training methods."""
        trainer, _ = _trainer("und")
        self.assertEqual(
            set(trainer.adapter.grad_norm_components().values()),
            {LORA_COMPONENT_TEXT_ENCODER_1},
        )


class GradNormBucketTest(unittest.TestCase):
    def test_gen_only_is_all_dit_and_unchanged(self):
        trainer, _ = _trainer("gen")
        n = _norms(trainer)
        self.assertGreater(n["unet"], 0.0)
        self.assertAlmostEqual(n["unet"], n["total"], places=5)
        self.assertEqual((n["te"], n["te1"], n["te2"], n["ve"]), (0.0, 0.0, 0.0, 0.0))

    def test_und_only_reports_a_text_encoder_norm_and_no_dit_norm(self):
        trainer, _ = _trainer("und")
        n = _norms(trainer)
        self.assertGreater(n["te1"], 0.0)
        self.assertAlmostEqual(n["te1"], n["te"], places=5)
        self.assertAlmostEqual(n["te1"], n["total"], places=5)
        self.assertEqual(n["unet"], 0.0)
        self.assertEqual((n["te2"], n["ve"]), (0.0, 0.0))

    def test_both_reports_two_norms_that_compose_into_the_total(self):
        trainer, _ = _trainer("both")
        n = _norms(trainer)
        self.assertGreater(n["unet"], 0.0)
        self.assertGreater(n["te1"], 0.0)
        # Equal halves, equal grads: the two norms must match each other.
        self.assertAlmostEqual(n["unet"], n["te1"], places=5)
        self.assertAlmostEqual(
            n["total"], (n["unet"] ** 2 + n["te1"] ** 2) ** 0.5, places=5
        )

    def test_the_map_is_built_once_per_run(self):
        trainer, _ = _trainer("both")
        calls = []
        real = trainer.adapter.grad_norm_components
        trainer.adapter.grad_norm_components = lambda: (calls.append(1), real())[1]
        trainer._calculate_grad_norms()
        trainer._calculate_grad_norms()
        trainer._calculate_grad_norms()
        self.assertEqual(len(calls), 1)

    def test_an_adapter_that_raises_falls_back_instead_of_killing_the_step(self):
        trainer, _ = _trainer("both")

        def _boom():
            raise RuntimeError("scope changed under us")

        trainer.adapter.grad_norm_components = _boom
        n = _norms(trainer)
        self.assertAlmostEqual(n["unet"], n["total"], places=5)
        self.assertEqual(n["te1"], 0.0)


class MergedBucketTest(unittest.TestCase):
    """Negative control: the shipped behaviour, with no adapter opinion."""

    def test_und_reported_the_understanding_half_as_dit(self):
        trainer, _ = _trainer("und", override=False)
        n = _norms(trainer)
        self.assertGreater(n["unet"], 0.0)
        self.assertAlmostEqual(n["unet"], n["total"], places=5)
        # The number U-2-5 would have needed, reported as 0.0 rather than missing.
        self.assertEqual(n["te1"], 0.0)

    def test_both_reported_one_merged_bucket(self):
        trainer, _ = _trainer("both", override=False)
        n = _norms(trainer)
        self.assertAlmostEqual(n["unet"], n["total"], places=5)
        self.assertEqual((n["te"], n["te1"]), (0.0, 0.0))

    def test_the_merged_and_split_totals_agree(self):
        """Only the attribution moved: the total is the same number."""
        merged, _ = _trainer("both", override=False)
        split, _ = _trainer("both")
        self.assertAlmostEqual(
            _norms(merged)["total"], _norms(split)["total"], places=5
        )


class OtherArchitecturesUnchangedTest(unittest.TestCase):
    """Every adapter that does not override the hook keeps module bucketing."""

    class _DiTTrainer(_FullFtTrainer):
        is_sensenova = False

    def test_a_transformer_arch_with_no_override_is_all_dit(self):
        transformer = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        for parameter in transformer.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        adapter = SimpleNamespace()  # no grad_norm_components at all
        trainer = self._DiTTrainer(transformer, adapter)
        n = _norms(trainer)
        self.assertGreater(n["unet"], 0.0)
        self.assertAlmostEqual(n["unet"], n["total"], places=5)
        self.assertEqual((n["te"], n["te1"], n["te2"], n["ve"]), (0.0, 0.0, 0.0, 0.0))

    def test_the_base_adapter_default_is_empty(self):
        from core.training.adapters.base_adapter import BaseFullParameterAdapter

        self.assertEqual(
            BaseFullParameterAdapter.grad_norm_components(SimpleNamespace()), {}
        )

    def test_an_sdxl_shaped_full_ft_still_splits_by_module(self):
        """text_encoder -> TE1 and text_encoder_2 -> TE2, untouched."""

        class _SdxlTrainer(_FullFtTrainer):
            is_sensenova = False

        trainer = _SdxlTrainer(nn.Linear(4, 4), SimpleNamespace())
        trainer.text_encoder = nn.Linear(2, 2)
        trainer.text_encoder_2 = nn.Linear(3, 3)
        for module in (trainer.transformer_original, trainer.text_encoder,
                       trainer.text_encoder_2):
            for parameter in module.parameters():
                parameter.grad = torch.full_like(parameter, 0.5)
        n = _norms(trainer)
        self.assertGreater(n["te1"], 0.0)
        self.assertGreater(n["te2"], 0.0)
        self.assertGreater(n["unet"], 0.0)
        self.assertAlmostEqual(
            n["te"], (n["te1"] ** 2 + n["te2"] ** 2) ** 0.5, places=5
        )


if __name__ == "__main__":
    unittest.main()
