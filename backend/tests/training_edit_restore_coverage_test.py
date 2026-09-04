"""Gate: opening a run for editing restores every field the form sends.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/training_edit_restore_coverage_test.py -v

WHY THIS EXISTS
---------------
``rescan_before_training`` was written by ``getRequestData()`` and named in
``PRESET_EXCLUDED_KEYS`` -- and in no restore list. The preset gate's
``test_every_request_key_is_accounted_for`` counts the exclusion list as
"accounted for", which is right for the SAVE question it asks and wrong for
this one: the exclusion list is a reason NOT to save, never a way to restore.
So editing an existing run silently reset the field to "off". Change a learning
rate, press save, and the pre-flight rescan the run was configured with is
gone.

The two lists answer two different questions:
  * ``PRESET_EXCLUDED_KEYS`` -- may a PRESET carry this to an unrelated run?
    ``rescan_before_training`` stays out: a preset carries no dataset_configs,
    so a stored "force" would rescan a dataset nobody chose.
  * ``PARAM_KEYS`` (+ the extra-restore and UI-state branches) -- may the run
    being EDITED get its own value back? Always yes: it is the run whose
    dataset it names.

This file asserts the second one, over the same source-shape reading style as
``training_preset_payload_test.py`` (no npm here; the repo owner runs the
type-check). Several assertions match EXACT SOURCE STRINGS, so a reformat turns
them red with no behaviour change -- update them in the same commit.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Dict, List, Set

_REPO = Path(__file__).resolve().parents[2]
_PANEL = _REPO / "frontend/src/components/training/TrainingConfig.tsx"
# The parameter surface moved to its own module; the panel keeps the code that
# reads it. These assertions are about the pair, so they read the pair.
_PARAMS = _REPO / "frontend/src/components/training/trainingParams.ts"
_ROUTES = _REPO / "backend/api/routes.py"
_CONFIG_GEN = _REPO / "backend/core/training/training_config.py"


def _source() -> str:
    return (_PARAMS.read_text(encoding="utf-8")
            + _PANEL.read_text(encoding="utf-8"))


def _slice(source: str, start: str, end: str) -> str:
    a = source.index(start)
    return source[a:source.index(end, a)]


def _string_list(source: str, declaration: str) -> List[str]:
    a = source.index(declaration)
    body = re.sub(r"//[^\n]*", "", source[a:source.index("\n];", a)])
    return re.findall(r'"([A-Za-z0-9_]+)"', body)


def _param_keys(source: str) -> List[str]:
    return _string_list(
        source, "const PARAM_KEYS: (keyof TrainingRunCreateRequest)[] = [")


def _request_data(source: str) -> str:
    return _slice(source, "const getRequestData", "const applyParamsToState")


def _apply_params(source: str) -> str:
    return _slice(source, "const applyParamsToState",
                  "// Load training run parameters for edit mode")


def _request_keys(source: str) -> Set[str]:
    """Top-level keys of the object literal getRequestData() returns."""
    return set(re.findall(r"^\s+([a-z][a-z0-9_]*): ", _request_data(source), re.M))


# Members of the two nested objects built inline (timestep_sampling,
# priority_training). They are restored through their parent, not as fields.
_NESTED_OBJECT_MEMBERS = {
    "distribution", "min_timestep", "max_timestep", "mean", "std",
    "alpha", "beta", "entries", "multiplier",
}

# Request fields restored into UI-only state rather than into `params`, each
# paired with the line that proves it. They are not `params` fields at all, so
# PARAM_KEYS cannot hold them.
UI_STATE_RESTORE: Dict[str, str] = {
    "run_name": "if (incoming.run_name) setRunName(incoming.run_name);",
    "base_model_path": "setBaseModelPath(bmp);",
    "training_method": "setTrainingMethod(incoming.training_method);",
    "dataset_configs": "if (incoming.dataset_configs) setDatasetConfigs(incoming.dataset_configs);",
}

# Fields restored by a custom-coercion or nested-object branch, each paired
# with the guard that proves the branch exists.
EXTRA_RESTORE_BRANCH: Dict[str, str] = {
    "base_resolutions": "patch.base_resolutions = incoming.base_resolutions === null",
    "regularization_type": 'patch.regularization_type = incoming.regularization_type || "none";',
    "vision_encoder_path": 'patch.vision_encoder_path = incoming.vision_encoder_path || "";',
    "controlnet_pretrained_path": "patch.controlnet_pretrained_path = incoming.controlnet_pretrained_path;",
    "condition_preprocessors": "patch.condition_preprocessors = incoming.condition_preprocessors;",
    "timestep_sampling": "if (incoming.timestep_sampling) {",
    "priority_training": "if (incoming.priority_training) {",
}

# Fields the form sends that edit mode deliberately does NOT restore. Empty on
# purpose: nothing currently qualifies. An entry here is a decision with a
# stated reason, which is the difference between this and the omission that
# lost rescan_before_training.
DELIBERATELY_NOT_RESTORED: Set[str] = set()


class ScanSanityTest(unittest.TestCase):
    """A scan that found nothing would make every assertion below vacuous."""

    def test_the_regions_parse(self):
        source = _source()
        self.assertGreater(len(source), 100_000)
        self.assertGreater(len(_param_keys(source)), 200)
        self.assertGreater(len(_request_data(source)), 10_000)
        apply_body = _apply_params(source)
        self.assertGreater(len(apply_body), 3_000)
        self.assertIn("for (const key of PARAM_KEYS) {", apply_body)
        keys = _request_keys(source)
        self.assertIn("batch_size", keys)
        self.assertIn("rescan_before_training", keys)


class EditRestoreCoverageTest(unittest.TestCase):
    """Every field the form sends comes back when the run is reopened."""

    @staticmethod
    def _unrestored(source: str) -> List[str]:
        known = (set(_param_keys(source))
                 | set(EXTRA_RESTORE_BRANCH)
                 | set(UI_STATE_RESTORE)
                 | _NESTED_OBJECT_MEMBERS
                 | DELIBERATELY_NOT_RESTORED)
        return sorted(_request_keys(source) - known)

    def test_every_request_key_is_restored_by_something(self):
        self.assertEqual(
            self._unrestored(_source()), [],
            "sent by the form but restored by nothing, so editing a run resets "
            "it to the default: add it to PARAM_KEYS (normal case), give it an "
            "applyParamsToState branch and list it in EXTRA_RESTORE_BRANCH, or "
            "name it in DELIBERATELY_NOT_RESTORED with a reason. Being in "
            "PRESET_EXCLUDED_KEYS is NOT a restore.")

    def test_the_gate_would_have_caught_the_rescan_gap(self):
        """Mutation check: put the source back the way it was and this fails."""
        source = _source()
        broken = source.replace('  "rescan_before_training",\n', "", 1)
        self.assertNotEqual(broken, source, "the PARAM_KEYS entry moved")
        self.assertNotIn("rescan_before_training", _param_keys(broken))
        self.assertIn("rescan_before_training", _request_keys(broken))
        self.assertEqual(self._unrestored(broken), ["rescan_before_training"])

    def test_rescan_before_training_is_restorable_but_not_preset_saved(self):
        """The regression proper, pinned from both sides."""
        source = _source()
        self.assertIn("rescan_before_training", _param_keys(source))
        excluded = _string_list(source, "const PRESET_EXCLUDED_KEYS: string[] = [")
        self.assertIn("rescan_before_training", excluded,
                      "a preset carries no dataset_configs, so a stored "
                      '"force" would rescan a dataset nobody chose')
        self.assertIn("rescan_before_training: params.rescan_before_training",
                      _request_data(source))

    def test_an_excluded_key_is_still_dropped_from_a_preset_after_the_fix(self):
        """PARAM_KEYS feeds PRESET_RESTORABLE_KEYS, so adding an entry there
        makes the preset loader RECOGNIZE the key. put() dropping it on the
        exclusion check is the only thing keeping the exclusion true."""
        source = _source()
        self.assertIn(
            "const PRESET_RESTORABLE_KEYS: string[] = [...PARAM_KEYS, ...PARAM_EXTRA_RESTORE_KEYS];",
            source)
        body = _slice(source, "export function presetConfigToParams",
                      "export default function TrainingConfig")
        self.assertIn("const excluded = new Set(PRESET_EXCLUDED_KEYS);", body)
        self.assertIn("if (excluded.has(key) || value === undefined) return;", body)
        save = _slice(source, "const getCurrentConfig", "const handleSavePreset")
        self.assertIn("PRESET_EXCLUDED_KEYS", save)

    def test_the_exclusion_list_is_not_a_restore_list(self):
        """Every excluded key is restored elsewhere -- which is exactly what
        was not true of rescan_before_training."""
        source = _source()
        excluded = _string_list(source, "const PRESET_EXCLUDED_KEYS: string[] = [")
        restorable = set(_param_keys(source)) | set(UI_STATE_RESTORE)
        orphans = sorted(key for key in excluded if key not in restorable)
        self.assertEqual(orphans, [],
                         f"excluded from presets AND restored by nothing: {orphans}")

    def test_the_deliberate_omission_list_is_exactly_what_was_decided(self):
        self.assertEqual(DELIBERATELY_NOT_RESTORED, set())


class RestoreBranchesExistTest(unittest.TestCase):
    """The non-PARAM_KEYS restores are real code, not just list entries."""

    def test_the_loop_over_param_keys_is_intact(self):
        body = _apply_params(_source())
        self.assertIn("for (const key of PARAM_KEYS) {", body)
        self.assertIn("(patch as any)[key] = incoming[key];", body)
        self.assertIn("setParams(prev => ({ ...prev, ...patch }));", body)

    def test_every_extra_restore_key_has_its_branch(self):
        body = _apply_params(_source())
        for key, marker in EXTRA_RESTORE_BRANCH.items():
            with self.subTest(key=key):
                self.assertIn(marker, body)

    def test_the_extra_restore_list_matches_the_branches_asserted_here(self):
        listed = set(_string_list(
            _source(), "const PARAM_EXTRA_RESTORE_KEYS: string[] = ["))
        self.assertEqual(listed, set(EXTRA_RESTORE_BRANCH))

    def test_every_ui_state_key_has_its_setter(self):
        body = _apply_params(_source())
        for key, marker in UI_STATE_RESTORE.items():
            with self.subTest(key=key):
                self.assertIn(marker, body)

    def test_the_edit_path_routes_through_apply_params_to_state(self):
        body = _slice(_source(), "const loadTrainingRunParams",
                      "// If in edit mode, load YAML parameters first")
        self.assertIn("await getTrainingRunParams(runId)", body)
        self.assertIn("applyParamsToState(params);", body)


class BackendReturnsTheValueTest(unittest.TestCase):
    """The restore can only work if /params still carries the field.

    ``_extract_request_params_from_yaml`` iterates
    ``TrainingRunCreateRequest.model_fields`` and, with no
    ``_YAML_FIELD_LOCATIONS`` entry, reads ``train[field_name]`` -- which is
    where ``training_config.py`` writes it. Three source facts, each of which
    would silently drop the value on its own.
    """

    def test_it_is_a_request_field(self):
        routes = _ROUTES.read_text(encoding="utf-8")
        self.assertIn(
            'rescan_before_training: Any = TRAINING_DEFAULTS["rescan_before_training"]',
            routes)

    def test_the_generator_writes_it_into_the_train_section(self):
        gen = _CONFIG_GEN.read_text(encoding="utf-8")
        self.assertIn(
            'train["rescan_before_training"] = p.get("rescan_before_training")',
            gen)

    def test_the_extractor_uses_the_same_name_fallback(self):
        """No relocation entry and no exclusion, so the generic
        ``train.get(field_name, default)`` path applies."""
        routes = _ROUTES.read_text(encoding="utf-8")
        locations = _slice(routes, "_YAML_FIELD_LOCATIONS: Dict[str, tuple] = {",
                           "\n}\n")
        self.assertIn('"total_steps": ("train", "steps"),', locations)  # parsed
        self.assertNotIn("rescan_before_training", locations)
        exclude = _slice(routes, "_AUTO_EXTRACT_EXCLUDE = {", "\n}\n")
        self.assertIn('"dataset_configs"', exclude)  # parsed
        self.assertNotIn("rescan_before_training", exclude)
        self.assertIn("value = train.get(field_name, default)", routes)


if __name__ == "__main__":
    unittest.main()
