"""Config-channel keys must survive a config-panel edit.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/config_edit_key_preservation_test.py -v

THE DEFECT
----------
``PUT /training/runs/{id}`` regenerates the run's YAML from
``TrainingRunCreateRequest``. Any train-section key that request model has no
field for is therefore discarded. Run 121 was created with
``optimizer_update_census`` hand-added to its YAML and lost it to a config-panel
edit -- the run kept training and its update detector was simply gone. The other
edit path, ``PATCH /training/runs/{id}/config`` (the Monitor's raw-YAML editor),
writes the submitted text verbatim and never had the problem.

This is general: the census is one instance. ``optimizer_state_host_resident``
and the video clip-length overrides sit in the same position, and any future
config-channel key would too.

THE DECISION
------------
No API surface is added. ``255a3ab5`` declined one for the two switches on
reasons this change does not disturb (the census raises, so a checkbox lets a
false positive kill a healthy run; its output is a stdout census; host-resident
state applies only to the ring-buffer optimizers). The observed harm was not
that the keys were unreachable -- run 121 proves they were reachable -- but that
a second route destroyed them. So the regenerating route preserves keys the
request model cannot express, and reports which, while keys the panel *can*
express stay panel-owned (clearing one in the form clears it in the config).

NEGATIVE CONTROL
----------------
``RegenerationDropNegativeControlTest`` reproduces the drop with the raw
generator, then shows the shipped helper stops it.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import yaml  # noqa: E402

from core.training.training_config import (  # noqa: E402
    TrainingConfigGenerator,
    preserve_unmodelled_train_keys,
    train_section_key_vocabulary,
)

_ROUTES_SOURCE = (BACKEND_ROOT / "api" / "routes.py").read_text(encoding="utf-8")
_BASE_TRAINER_SOURCE = (
    BACKEND_ROOT / "core" / "training" / "base_trainer.py"
).read_text(encoding="utf-8")
_TRAIN_RUNNER_SOURCE = (
    BACKEND_ROOT / "core" / "training" / "train_runner.py"
).read_text(encoding="utf-8")

# Every train-section key BaseTrainer/train_runner reads that no request field
# can set. Asserted exactly, so a new one cannot be introduced unnoticed.
CONFIG_CHANNEL_ONLY_KEYS = {
    "optimizer_update_census",
    "optimizer_state_host_resident",
    "allowed_clip_lengths",
    "clip_stride",
    "ltx2_clip_lengths",
    "ltx2_clip_stride",
    "seed",
    "debug_vram",
    # train_runner.py reads these off the train section, but the generator
    # emits them one level up, as siblings of "train" inside process[0] --
    # a pre-existing level mismatch, not a diagnostic switch. They read as
    # config-channel-only (an old backup with them hand-placed inside "train"
    # would work); they are also always their hardcoded default in a
    # generated run today, independent of the panel's setting.
    "prompt_chunking_mode",
    "max_prompt_chunks",
}

BASE_PARAMS = {
    "learning_rate": 1e-4,
    "batch_size": 1,
    "optimizer": "adamw8bit",
    "lr_scheduler": "constant",
    "train_unet": True,
    "train_text_encoder": False,
}

COMMON_KWARGS = {
    "run_name": "preservation_test",
    "base_model_path": "/models/does-not-exist.safetensors",
    "output_dir": "/tmp/preservation_test",
    "dataset_configs": [{"dataset_id": 1, "path": "/data/ds"}],
    "sample_prompts": [],
}


def _generate(method: str = "lora", **overrides) -> str:
    params = {**BASE_PARAMS, "total_steps": 100, **overrides}
    generator = TrainingConfigGenerator()
    builders = {
        "lora": generator.generate_lora_config,
        "relora": generator.generate_relora_config,
        "controlnet": generator.generate_controlnet_config,
        "full_finetune": generator.generate_full_finetune_config,
    }
    return builders[method](params, **COMMON_KWARGS)


def _train_section(config_yaml: str) -> dict:
    return yaml.safe_load(config_yaml)["config"]["process"][0]["train"]


def _with_train_keys(config_yaml: str, **keys) -> str:
    config = yaml.safe_load(config_yaml)
    config["config"]["process"][0]["train"].update(keys)
    return yaml.dump(config, default_flow_style=False, sort_keys=False,
                     allow_unicode=True)


class RegenerationDropNegativeControlTest(unittest.TestCase):
    """The drop, reproduced; then the fix, on the same config."""

    def test_raw_regeneration_drops_the_census(self):
        old = _with_train_keys(_generate(), optimizer_update_census=True)
        self.assertIs(_train_section(old)["optimizer_update_census"], True)

        # What PUT /training/runs/{id} did before this change: rebuild from the
        # request model and keep the result.
        regenerated = _generate()
        self.assertNotIn("optimizer_update_census", _train_section(regenerated))

    def test_preservation_stops_the_drop(self):
        old = _with_train_keys(_generate(), optimizer_update_census=True)
        fixed, preserved = preserve_unmodelled_train_keys(old, _generate())

        self.assertEqual(preserved, ["optimizer_update_census"])
        self.assertIs(_train_section(fixed)["optimizer_update_census"], True)

    def test_every_config_channel_only_key_survives(self):
        carried = {
            "optimizer_update_census": True,
            "optimizer_state_host_resident": True,
            "ltx2_clip_lengths": [17, 33],
            "ltx2_clip_stride": 4,
        }
        old = _with_train_keys(_generate(), **carried)
        fixed, preserved = preserve_unmodelled_train_keys(old, _generate())

        self.assertEqual(preserved, sorted(carried))
        for key, value in carried.items():
            self.assertEqual(_train_section(fixed)[key], value)

    def test_the_rest_of_the_config_is_untouched(self):
        old = _with_train_keys(_generate(), optimizer_update_census=True)
        new = _generate(learning_rate=5e-5)
        fixed, _ = preserve_unmodelled_train_keys(old, new)

        fixed_train = _train_section(fixed)
        # The edit lands; only the unmodelled key is added back.
        self.assertEqual(fixed_train["lr"], 5e-5)
        self.assertEqual(
            set(fixed_train) - set(_train_section(new)),
            {"optimizer_update_census"},
        )


class PanelOwnedKeysAreNotResurrectedTest(unittest.TestCase):
    """Turning a feature off in the form must turn it off in the config."""

    def test_disabling_tread_does_not_come_back(self):
        old = _generate(tread_enable=True, tread_drop_ratio=0.4)
        old_train = _train_section(old)
        self.assertIs(old_train["tread_enable"], True)

        fixed, preserved = preserve_unmodelled_train_keys(
            old, _generate(tread_enable=False))

        self.assertEqual(preserved, [])
        fixed_train = _train_section(fixed)
        self.assertIs(fixed_train["tread_enable"], False)
        # Emitted unconditionally, so the request's value wins over the old one.
        self.assertEqual(fixed_train["tread_drop_ratio"], 0.5)

    def test_disabling_ema_does_not_come_back(self):
        old = _generate(use_ema=True, ema_decay=0.999)
        self.assertIn("ema_decay", _train_section(old))

        fixed, preserved = preserve_unmodelled_train_keys(
            old, _generate(use_ema=False))

        self.assertEqual(preserved, [])
        self.assertNotIn("ema_decay", _train_section(fixed))


class UnaffectedRunsAreByteIdenticalTest(unittest.TestCase):
    """A config that uses no config-channel-only key must not move at all."""

    def test_all_training_methods_pass_through_unchanged(self):
        for method in ("lora", "relora", "controlnet", "full_finetune"):
            with self.subTest(method=method):
                old = _generate(method)
                new = _generate(method, learning_rate=2e-4)
                fixed, preserved = preserve_unmodelled_train_keys(old, new)
                self.assertEqual(preserved, [])
                self.assertIs(fixed, new)

    def test_vae_decoder_train_keys_are_all_panel_owned(self):
        # generate_vae_config builds its train section as a literal; the
        # vocabulary covers it, so a vae run with no hand-added key does not move.
        vae_train = {
            "batch_size": 1, "steps": 100, "gradient_accumulation_steps": 1,
            "lr": 1e-5, "optimizer": "adamw", "optimizer_weight_decay": 0.01,
            "max_grad_norm": 1.0, "lr_scheduler": "constant",
            "lr_warmup_steps": 0, "resume_from_checkpoint": None,
        }
        self.assertEqual(set(vae_train) - train_section_key_vocabulary(), set())
        old = yaml.dump({"config": {"process": [{"train": vae_train}]}})
        new = _generate()
        fixed, preserved = preserve_unmodelled_train_keys(old, new)
        self.assertIs(fixed, new)
        self.assertEqual(preserved, [])

    def test_no_previous_config_is_a_no_op(self):
        new = _generate()
        for old in (None, ""):
            fixed, preserved = preserve_unmodelled_train_keys(old, new)
            self.assertIs(fixed, new)
            self.assertEqual(preserved, [])

    def test_unparseable_or_foreign_config_is_a_no_op(self):
        new = _generate()
        for old in ("{{ not yaml", "just a string", "a: 1"):
            fixed, preserved = preserve_unmodelled_train_keys(old, new)
            self.assertIs(fixed, new)
            self.assertEqual(preserved, [])


class VocabularyTest(unittest.TestCase):
    """The set that separates panel-owned keys from config-channel-only ones."""

    def test_vocabulary_is_derived_and_non_empty(self):
        vocabulary = train_section_key_vocabulary()
        self.assertGreater(len(vocabulary), 50)

    def test_generated_configs_only_use_vocabulary_keys(self):
        for method in ("lora", "relora", "controlnet", "full_finetune"):
            with self.subTest(method=method):
                unknown = set(_train_section(_generate(method))) - \
                    train_section_key_vocabulary()
                self.assertEqual(unknown, set())

    def test_config_channel_only_keys_are_outside_the_vocabulary(self):
        vocabulary = train_section_key_vocabulary()
        for key in CONFIG_CHANNEL_ONLY_KEYS:
            self.assertNotIn(key, vocabulary)


class SenseNovaBlockSwapFlagRoundTripTest(unittest.TestCase):
    """A name appearing in the vocabulary is not proof the value reaches the
    YAML: ``_build_train_section`` is an explicit whitelist, and a flag with a
    Pydantic field, an openapi entry and a capability row can still be absent
    from that whitelist while both the name-based census and the vocabulary
    derivation pass. ``sensenova_mot_pageable_staging`` shipped exactly that
    way; this asserts the actual value round-trips, for it and its siblings.
    """

    SENSENOVA_BLOCK_SWAP_FLAGS = (
        "sensenova_mot_phase_eviction",
        "sensenova_four_phase_eviction",
        "sensenova_four_phase_shared_prefix",
        "sensenova_sample_kv_cache_streaming",
        "sensenova_mot_pageable_staging",
    )

    def test_flags_round_trip_through_both_generators(self):
        for method in ("lora", "full_finetune"):
            for flag in self.SENSENOVA_BLOCK_SWAP_FLAGS:
                with self.subTest(method=method, flag=flag):
                    train = _train_section(_generate(method, **{flag: True}))
                    self.assertIn(flag, train)
                    self.assertIs(train[flag], True)

    def test_grad_reduction_string_value_round_trips(self):
        for method in ("lora", "full_finetune"):
            with self.subTest(method=method):
                train = _train_section(
                    _generate(method, sensenova_four_phase_grad_reduction="mean")
                )
                self.assertEqual(train["sensenova_four_phase_grad_reduction"], "mean")


class ConfigChannelCensusTest(unittest.TestCase):
    """Which trainer-read keys no request field can set -- pinned exactly."""

    @staticmethod
    def _trainer_config_keys() -> set:
        keys = set()
        for source in (_BASE_TRAINER_SOURCE, _TRAIN_RUNNER_SOURCE):
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "get"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)):
                    continue
                target = node.func.value
                reads_config = (
                    isinstance(target, ast.Name)
                    and target.id in ("_tc", "train_config")
                ) or (
                    isinstance(target, ast.Attribute) and target.attr == "config"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                )
                if reads_config:
                    keys.add(node.args[0].value)
        return keys

    def test_census_is_exactly_the_known_set(self):
        from api.routes import TrainingRunCreateRequest

        fields = set(TrainingRunCreateRequest.model_fields)
        # train_runner.py reads several train-section keys (lr, steps,
        # optimizer_warmup_steps, ...) under names that don't match the
        # request field that sets them (learning_rate, total_steps, ...).
        # Field-name equality can't see through that translation, so also
        # clear anything the generator actually emits into "train" --
        # train_section_key_vocabulary() is the AST-derived ground truth for
        # what the panel can reach, independent of naming.
        reachable = fields | train_section_key_vocabulary()
        # The generator emits `seed` into the sample section only (as
        # sample_seed); nothing puts a "seed" key in the train section, so
        # base_trainer.py's self.config.get("seed", 0) can never be set by a
        # request and "seed" belongs in CONFIG_CHANNEL_ONLY_KEYS, not here.
        unreachable = self._trainer_config_keys() - reachable
        self.assertEqual(unreachable, CONFIG_CHANNEL_ONLY_KEYS)


class EditPathWiringTest(unittest.TestCase):
    """Which route regenerates, which preserves, and that both now behave."""

    @staticmethod
    def _handler_source(name: str) -> str:
        tree = ast.parse(_ROUTES_SOURCE)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                    node.name == name:
                return ast.get_source_segment(_ROUTES_SOURCE, node) or ""
        raise AssertionError(f"handler {name} not found in routes.py")

    def test_put_regenerates_and_preserves(self):
        source = self._handler_source("update_training_run")
        self.assertIn("generate_lora_config", source)
        self.assertIn("preserve_unmodelled_train_keys", source)
        self.assertIn("preserved_config_keys", source)

    def test_patch_writes_verbatim_and_regenerates_nothing(self):
        source = self._handler_source("update_training_config")
        self.assertIn("run.config_yaml = config_yaml", source)
        for generator in ("generate_lora_config", "generate_full_finetune_config",
                          "TrainingConfigGenerator"):
            self.assertNotIn(generator, source)

    def test_openapi_documents_the_preserved_key_list(self):
        spec = (REPO_ROOT / "openapi.yaml").read_text(encoding="utf-8")
        self.assertIn("preserved_config_keys", spec)

    def test_no_api_parameter_was_added(self):
        """255a3ab5's decision stands: neither switch became a request field."""
        from api.param_defaults import TRAINING_DEFAULTS
        from api.routes import TrainingRunCreateRequest

        for key in ("optimizer_update_census", "optimizer_state_host_resident"):
            self.assertNotIn(key, TRAINING_DEFAULTS)
            self.assertNotIn(key, TrainingRunCreateRequest.model_fields)


if __name__ == "__main__":
    unittest.main(verbosity=2)
