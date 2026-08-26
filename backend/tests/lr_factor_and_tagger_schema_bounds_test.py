"""Bounds added in this change: the nine *_lr_factor multipliers (ge=0, a
negative multiplier ascends the loss the same way a negative rate does), and
the tagger request's learning_rate (gt=0) / save_every_n_steps /
save_every_n_epochs / train_f1_eval_every_n_steps /
train_f1_threshold_search_every_n_steps / keep_last_n_checkpoints (ge=0, 0
stays legal -- it means "disabled"/"keep all"/"never save").

Run:
    venv/Scripts/python.exe -m pytest backend/tests/lr_factor_and_tagger_schema_bounds_test.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pydantic import ValidationError  # noqa: E402

from api.routes import TrainingRunCreateRequest, TaggerTrainingRunCreateRequest  # noqa: E402

LR_FACTOR_FIELDS = [
    "lens_img_lr_factor",
    "lens_txt_lr_factor",
    "ideogram4_lr_factor",
    "minit2i_lr_factor",
    "krea2_lr_factor",
    "repa_proj_lr_factor",
    "anima_attn_mlp_lr_factor",
    "anima_mod_lr_factor",
    "anima_llm_adapter_lr_factor",
]

TAGGER_GE0_FIELDS = [
    "save_every_n_steps",
    "save_every_n_epochs",
    "train_f1_eval_every_n_steps",
    "train_f1_threshold_search_every_n_steps",
    "keep_last_n_checkpoints",
]


def _base_training_kwargs():
    return dict(
        training_method="lora",
        base_model_path="model.safetensors",
        dataset_configs=[{"dataset_id": 1}],
    )


def _base_tagger_kwargs():
    return dict(
        vision_encoder_path="encoder.safetensors",
        dataset_configs=[{"dataset_id": 1, "caption_types": ["tags"]}],
    )


class LrFactorBoundsTest(unittest.TestCase):
    def test_zero_is_legal(self):
        for field in LR_FACTOR_FIELDS:
            with self.subTest(field=field):
                req = TrainingRunCreateRequest(**_base_training_kwargs(), **{field: 0.0})
                self.assertEqual(getattr(req, field), 0.0)

    def test_negative_is_refused(self):
        for field in LR_FACTOR_FIELDS:
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    TrainingRunCreateRequest(**_base_training_kwargs(), **{field: -1.0})


class TaggerSchemaBoundsTest(unittest.TestCase):
    def test_learning_rate_zero_refused(self):
        with self.assertRaises(ValidationError):
            TaggerTrainingRunCreateRequest(**_base_tagger_kwargs(), learning_rate=0.0)

    def test_learning_rate_negative_refused(self):
        with self.assertRaises(ValidationError):
            TaggerTrainingRunCreateRequest(**_base_tagger_kwargs(), learning_rate=-1e-4)

    def test_learning_rate_positive_accepted(self):
        req = TaggerTrainingRunCreateRequest(**_base_tagger_kwargs(), learning_rate=3e-4)
        self.assertEqual(req.learning_rate, 3e-4)

    def test_ge0_fields_zero_is_legal(self):
        for field in TAGGER_GE0_FIELDS:
            with self.subTest(field=field):
                req = TaggerTrainingRunCreateRequest(**_base_tagger_kwargs(), **{field: 0})
                self.assertEqual(getattr(req, field), 0)

    def test_ge0_fields_negative_refused(self):
        for field in TAGGER_GE0_FIELDS:
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    TaggerTrainingRunCreateRequest(**_base_tagger_kwargs(), **{field: -1})


if __name__ == "__main__":
    unittest.main()
