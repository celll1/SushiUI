"""Where a full fine-tune's resume actually goes, and what the dead branch did.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/full_parameter_resume_path_test.py -v

THE DEFECT (SENSENOVA_TRAINING_DESIGN.md 13.4, audit item 5)
------------------------------------------------------------
``FullParameterTrainer.load_checkpoint``'s ``.safetensors`` branch imported
``core.models.checkpoint_utils.load_unified_checkpoint``. There is no
``checkpoint_utils`` module anywhere in this repository, so the branch could only
ever raise ``ModuleNotFoundError``.

WHICH ARCHITECTURES REACHED IT: none, and the audit note that says SenseNova in
particular does not reach it understates the result. ``load_checkpoint`` is
abstract on ``BaseTrainer`` and implemented here, but NOTHING calls it -- the
resume every architecture takes is ``resume_from_checkpoint``, which
``BaseTrainer.__init__`` services by loading the checkpoint as the base model
(``_load_checkpoint_as_base``, with ``_try_load_checkpoint_with_fallback``
behind it). The trainers that do call their own ``load_checkpoint`` are
``ControlNetTrainer`` and ``VaeTrainer``, which are different classes with their
own implementations.

THE DECISION: refuse loudly rather than implement or delete. Deleting is not
available (the base method is abstract, so the class would stop instantiating),
and implementing would mean inventing a reader for eleven architectures' full-FT
save formats with no caller, no consumer and no test. The second branch that was
removed with it -- a diffusers-DIRECTORY loader -- addressed a layout no
full-parameter adapter in this repo writes.

NEGATIVE CONTROLS
-----------------
``ShippedBehaviourTest`` shows the import that was there still fails, and that
the search for callers finds none.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.full_parameter_trainer import FullParameterTrainer  # noqa: E402
from core.training.lora_trainer import LoRATrainer  # noqa: E402

_FULL_PARAM_SOURCE = (
    BACKEND_ROOT / "core" / "training" / "full_parameter_trainer.py"
).read_text(encoding="utf-8")
_BASE_SOURCE = (
    BACKEND_ROOT / "core" / "training" / "base_trainer.py"
).read_text(encoding="utf-8")


class _Stub:
    """Enough of a trainer to call the unbound method on."""

    log_prefix = "[test]"


def _load_checkpoint_body() -> ast.FunctionDef:
    """``FullParameterTrainer.load_checkpoint`` as CODE, docstring dropped.

    Source-text assertions cannot be used here: the docstring deliberately names
    the dead import and the removed branches so the next reader does not have to
    dig them out of the history.
    """
    tree = ast.parse(_FULL_PARAM_SOURCE)
    for cls in ast.walk(tree):
        if not (isinstance(cls, ast.ClassDef) and cls.name == "FullParameterTrainer"):
            continue
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "load_checkpoint":
                body = list(fn.body)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    body = body[1:]
                fn.body = body
                return fn
    raise AssertionError("FullParameterTrainer.load_checkpoint is gone")


def _attribute_calls(source: str, name: str):
    """``(line, receiver.attr)`` for every real call to ``.<name>(`` in ``source``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == name):
            found.append(node.lineno)
    return found


class ShippedBehaviourTest(unittest.TestCase):
    """Negative controls: the module is missing and nothing called the method."""

    def test_the_imported_module_does_not_exist(self):
        self.assertIsNone(importlib.util.find_spec("core.models.checkpoint_utils"))
        self.assertEqual(list(BACKEND_ROOT.glob("core/models/**/checkpoint_utils.py")), [])
        with self.assertRaises(ModuleNotFoundError):
            __import__("core.models.checkpoint_utils")

    def test_no_production_code_calls_a_full_parameter_load_checkpoint(self):
        """The census of real ``.load_checkpoint(...)`` calls under backend/."""
        owners = set()
        for path in BACKEND_ROOT.rglob("*.py"):
            parts = set(path.parts)
            if "tests" in parts or "__pycache__" in parts:
                continue
            source = path.read_text(encoding="utf-8", errors="ignore")
            if "load_checkpoint" not in source:
                continue
            try:
                calls = _attribute_calls(source, "load_checkpoint")
            except SyntaxError:
                continue
            if calls:
                owners.add(path.relative_to(BACKEND_ROOT).as_posix())
        # Every remaining caller invokes ITS OWN class's implementation.
        self.assertEqual(
            owners,
            {
                "core/tagger/siglip2_inference_manager.py",  # tagger model classmethod
                "core/training/controlnet_trainer.py",
                "core/training/vae/vae_trainer.py",
            },
        )

    def test_the_resume_every_architecture_takes_is_the_base_model_reload(self):
        self.assertIn("self._load_checkpoint_as_base(checkpoint_to_load)", _BASE_SOURCE)
        self.assertIn("if self.resume_from_checkpoint and not getattr", _BASE_SOURCE)
        # FullParameterTrainer neither overrides nor calls the mechanism that is
        # used -- it inherits it, which is why the resume was never broken.
        self.assertNotIn("_load_checkpoint_as_base", FullParameterTrainer.__dict__)
        self.assertEqual(_attribute_calls(_FULL_PARAM_SOURCE, "_load_checkpoint_as_base"), [])
        self.assertTrue(hasattr(FullParameterTrainer, "_load_checkpoint_as_base"))


class RefusalTest(unittest.TestCase):
    def test_it_raises_not_implemented_and_names_the_real_resume(self):
        with self.assertRaises(NotImplementedError) as caught:
            FullParameterTrainer.load_checkpoint(_Stub(), "run_step_000100.safetensors")
        message = str(caught.exception)
        self.assertIn("resume_from_checkpoint", message)
        self.assertIn("core.models.checkpoint_utils", message)
        # The path the caller asked for is echoed, so the message is actionable
        # from a log alone.
        self.assertIn("run_step_000100.safetensors", message)

    def test_the_body_is_nothing_but_the_refusal(self):
        """Asserted on the AST: the docstring still names what was removed."""
        body = _load_checkpoint_body().body
        self.assertEqual(len(body), 1)
        self.assertIsInstance(body[0], ast.Raise)
        imports = [
            n for n in ast.walk(_load_checkpoint_body())
            if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        self.assertEqual(imports, [], "the dead import is back")

    def test_the_removed_branches_are_not_executed_anywhere_in_the_module(self):
        names = {
            n.id for n in ast.walk(ast.parse(_FULL_PARAM_SOURCE)) if isinstance(n, ast.Name)
        } | {
            n.attr for n in ast.walk(ast.parse(_FULL_PARAM_SOURCE))
            if isinstance(n, ast.Attribute)
        }
        for token in ("load_unified_checkpoint", "UNet2DConditionModel",
                      "CLIPTextModelWithProjection", "CLIPTextModel"):
            with self.subTest(token):
                self.assertNotIn(token, names)

    def test_the_abstract_contract_is_still_satisfied(self):
        # Deleting the method is not an option: the base declares it abstract,
        # so the class would stop instantiating for every architecture.
        self.assertTrue(
            getattr(BaseTrainer.load_checkpoint, "__isabstractmethod__", False)
        )
        self.assertIn("load_checkpoint", FullParameterTrainer.__dict__)
        self.assertEqual(FullParameterTrainer.__abstractmethods__, frozenset())


class UnrelatedTrainersUnchangedTest(unittest.TestCase):
    def test_lora_trainer_keeps_its_own_working_implementation(self):
        source = (BACKEND_ROOT / "core" / "training" / "lora_trainer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def load_checkpoint", source)
        self.assertNotIn("checkpoint_utils", source)
        self.assertIsNot(
            LoRATrainer.load_checkpoint, FullParameterTrainer.load_checkpoint
        )

    def test_controlnet_trainer_still_calls_its_own(self):
        source = (BACKEND_ROOT / "core" / "training" / "controlnet_trainer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("self.load_checkpoint(str(ckpt))", source)
        self.assertIn("self.adapter.load_checkpoint(self.controlnet", source)

    def test_saving_is_untouched(self):
        self.assertIn("def save_checkpoint", _FULL_PARAM_SOURCE)
        self.assertIn("self.adapter.save_checkpoint(step, epoch, checkpoint_path)",
                      _FULL_PARAM_SOURCE)


if __name__ == "__main__":
    unittest.main()
