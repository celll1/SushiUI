"""``GET /params`` must return a body ``PUT`` accepts.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/params_roundtrip_defaults_test.py -v

WHAT IS PINNED
--------------
Not one field: ``_extract_request_params_from_yaml``'s contract, that
everything it returns validates as a ``TrainingRunCreateRequest``.
``EveryFieldTest`` walks the whole model against a deliberately bare YAML, so a
future field with a ``default_factory`` -- whose ``field_info.default`` is
``PydanticUndefined``, indistinguishable from a required field -- or a future
non-nullable field whose YAML key is optional, fails here rather than in the
user's edit form.

``sample_prompts`` was the reachable instance: non-nullable, factory-defaulted,
and ``generate_vae_config`` writes no ``sample`` section, so every stored
``vae_decoder`` run reported it as null and its own PUT then rejected it.
That case is pinned end to end, and pinned as NOT changing the stored config:
the regenerated YAML still has no ``sample`` section.

``NullIsOnlyCoercedWhereItIsIllegalTest`` pins the SCOPE of the null coercion,
which is what keeps "null means unset" working for the fields that mean it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, get_args

import torch
import yaml

# AGENTS.md: keep the import off the GPU the owner's training run is holding.
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from api.routes import (  # noqa: E402
    TrainingRunCreateRequest,
    _AUTO_EXTRACT_EXCLUDE,
    _extract_request_params_from_yaml,
)
from core.training.training_config import TrainingConfigGenerator  # noqa: E402


def _accepts_none(annotation: Any) -> bool:
    return (annotation is Any or annotation is None
            or type(None) in get_args(annotation))


class TheTrapIsRealTest(unittest.TestCase):
    """The Pydantic detail the extractor got wrong, asserted directly."""

    def test_a_default_factory_field_looks_required_via_default(self):
        field = TrainingRunCreateRequest.model_fields["sample_prompts"]
        self.assertIsNotNone(field.default_factory)
        self.assertEqual(field.default.__class__.__name__, "PydanticUndefinedType")
        self.assertFalse(field.is_required())
        # ...and the value the extractor must use instead.
        self.assertEqual(field.get_default(call_default_factory=True),
                         [{"positive": "", "negative": ""}])

    def test_sample_prompts_is_the_non_nullable_list(self):
        """Why this field and not its neighbours: the others are Optional."""
        self.assertFalse(_accepts_none(
            TrainingRunCreateRequest.model_fields["sample_prompts"].annotation))
        for sibling in ("dataset_configs", "base_resolutions",
                        "condition_preprocessors", "crop_smaller_scale_range"):
            self.assertTrue(
                _accepts_none(TrainingRunCreateRequest.model_fields[sibling].annotation),
                sibling)


class EveryFieldTest(unittest.TestCase):
    """Against a YAML that mentions almost nothing, the whole model survives."""

    BARE = {"train": {"steps": 100}, "network": {"type": "lora"}}

    def _extract(self, process_config: dict, job: str = "lora") -> dict:
        params = _extract_request_params_from_yaml(process_config, job)
        params.update(training_method="lora", base_model_path="model.safetensors")
        return params

    def test_no_non_nullable_field_comes_back_none(self):
        params = self._extract(self.BARE)
        offenders = [
            name for name, value in params.items()
            if value is None
            and name in TrainingRunCreateRequest.model_fields
            and not _accepts_none(TrainingRunCreateRequest.model_fields[name].annotation)
        ]
        self.assertEqual(offenders, [], f"extractor returned None for {offenders}")

    def test_the_bare_extraction_validates(self):
        request = TrainingRunCreateRequest(**self._extract(self.BARE))
        self.assertEqual(request.sample_prompts, [{"positive": "", "negative": ""}])
        # The scan above is not vacuous: the model really was populated.
        self.assertEqual(request.total_steps, 100)

    def test_an_explicit_null_is_read_as_the_absent_key_it_looks_like(self):
        """`sample:\\n  prompts:` -- a hand-edited YAML, not a generated one."""
        params = self._extract({**self.BARE, "sample": {"prompts": None},
                                "train": {"steps": 100, "optimizer": None}})
        request = TrainingRunCreateRequest(**params)
        self.assertEqual(request.sample_prompts, [{"positive": "", "negative": ""}])
        self.assertEqual(request.optimizer,
                         TrainingRunCreateRequest.model_fields["optimizer"].default)

    def test_a_section_written_with_nothing_under_it_does_not_crash(self):
        for section in ("sample", "save", "network", "dtype", "model", "train"):
            with self.subTest(section=section):
                params = self._extract({**self.BARE, section: None})
                TrainingRunCreateRequest(**params)


class VaeRunEditRoundTripTest(unittest.TestCase):
    """The shape that actually shipped broken: a real vae_decoder config."""

    def _vae_yaml(self) -> dict:
        text = TrainingConfigGenerator.generate_vae_config(
            {"total_steps": 100, "learning_rate": 1e-5},
            run_name="vae_params_roundtrip",
            base_model_path="vae.safetensors", output_dir="out", dataset_path="data",
        )
        return yaml.safe_load(text)["config"]["process"][0]

    def test_the_generator_still_writes_no_sample_section(self):
        """The premise. If this ever changes, the test below stops testing it."""
        self.assertNotIn("sample", self._vae_yaml())

    def test_params_to_put_to_regenerate(self):
        process = self._vae_yaml()
        params = _extract_request_params_from_yaml(process, "vae_decoder")
        self.assertIsNotNone(params["sample_prompts"])
        params.update(training_method="vae_decoder",
                      base_model_path=process["model"]["name_or_path"],
                      vae_config=process.get("vae"))
        params.pop("dataset_configs", None)
        request = TrainingRunCreateRequest(**params)  # this is what PUT does

        text = TrainingConfigGenerator.generate_vae_config(
            request.model_dump(), run_name="vae_params_roundtrip",
            base_model_path=request.base_model_path, output_dir="out",
            dataset_path="data",
        )
        regenerated = yaml.safe_load(text)["config"]["process"][0]
        # The materialised default does not leak into the stored config.
        self.assertNotIn("sample", regenerated)
        self.assertEqual(regenerated["train"]["steps"], process["train"]["steps"])
        self.assertEqual(regenerated["network"]["type"], "vae_decoder")


class NullIsOnlyCoercedWhereItIsIllegalTest(unittest.TestCase):
    """A deliberate null survives; only an unrepresentable one is replaced.

    The field has to be nullable AND have a NON-None default, or the assertion
    passes whether the null was kept or replaced. ``lora_rank`` is
    ``Optional[int] = 16`` and lives at ``network.linear``; dropping the
    ``_accepts_none`` guard from the extractor turns this red (it returns 16).
    """

    def _request(self, process_config: dict) -> TrainingRunCreateRequest:
        params = _extract_request_params_from_yaml(process_config, "lora")
        params.update(training_method="lora", base_model_path="model.safetensors")
        return TrainingRunCreateRequest(**params)

    def test_the_fixture_field_would_notice_a_substitution(self):
        field = TrainingRunCreateRequest.model_fields["lora_rank"]
        self.assertTrue(_accepts_none(field.annotation))
        self.assertEqual(field.get_default(call_default_factory=True), 16)

    def test_an_explicit_null_on_a_nullable_field_is_kept(self):
        request = self._request({
            "train": {"steps": 100},
            "network": {"type": "lora", "linear": None, "linear_alpha": 8},
        })
        self.assertIsNone(request.lora_rank)
        # The section really was read: the sibling key next to it came through.
        self.assertEqual(request.lora_alpha, 8)

    def test_the_same_key_absent_still_gets_the_default(self):
        request = self._request({"train": {"steps": 100}, "network": {"type": "lora"}})
        self.assertEqual(request.lora_rank, 16)

    def test_a_null_on_a_non_nullable_field_is_replaced(self):
        """The other side of the scope, at a field with a non-None default."""
        request = self._request(
            {"train": {"steps": 100, "optimizer": None}, "network": {"type": "lora"}})
        self.assertEqual(request.optimizer,
                         TrainingRunCreateRequest.model_fields["optimizer"].default)
        self.assertIsNotNone(request.optimizer)


class ExtractorContractTest(unittest.TestCase):
    """The excluded fields are excluded on purpose, not by omission."""

    def test_excluded_fields_are_not_emitted(self):
        params = _extract_request_params_from_yaml(
            {"train": {"steps": 10}, "network": {"type": "lora"}}, "lora")
        for name in _AUTO_EXTRACT_EXCLUDE:
            self.assertNotIn(name, params, name)


if __name__ == "__main__":
    unittest.main()
