"""Gate: a training preset carries every parameter the form restores.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/training_preset_payload_test.py -v

WHY THIS EXISTS
---------------
``TrainingConfig.tsx`` used to hold two parallel lists of training parameters:
``PARAM_KEYS`` (what the edit form restores) and a hand-written camelCase
mapping inside ``getCurrentConfig`` / ``handleLoadPreset`` (what a preset saved
and loaded). The second list covered 115 of 247 fields, so 132 parameters --
``lr_warmup_steps``, ``use_ema``, ``gradient_checkpointing``, ``fp8_base_dtype``,
every ``danbooru_aug_*``, every ``outpaint_*`` -- silently reverted to defaults
on a preset load. ``weight_decompose`` had the same hole and was patched on its
own; the hand list was the actual defect.

The payload is now DERIVED: ``getCurrentConfig()`` is ``getRequestData()`` minus
``PRESET_EXCLUDED_KEYS``. This file fails if that stops being true, if a
``PARAM_KEYS`` entry stops being written by ``getRequestData()``, or if the
exclusion list changes without someone editing ``EXPECTED_EXCLUSIONS`` below.

These are source-shape assertions (no npm here, and the repo owner runs the
type-check), in the style of ``training_config_cache_wiring_test.py``. Several
match EXACT SOURCE STRINGS, so a Prettier pass or a line re-wrap turns them red
with no behaviour change -- reformat knowingly and update them in the same
commit. They are kept in that form deliberately: mutation-testing this gate,
that family was the only one that caught a naive camel -> snake inverse and a
dropped compatibility branch.
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
_API = _REPO / "frontend/src/utils/api.ts"


def _source() -> str:
    return (_PARAMS.read_text(encoding="utf-8")
            + _PANEL.read_text(encoding="utf-8"))


def _slice(source: str, start: str, end: str) -> str:
    a = source.index(start)
    return source[a:source.index(end, a)]


def _string_list(source: str, declaration: str) -> List[str]:
    """The quoted entries of a ``const NAME ... = [ ... ];`` literal.

    Comments are stripped first: these lists carry their rationale inline,
    and a quoted word in a comment is not an entry.
    """
    a = source.index(declaration)
    body = re.sub(r"//[^\n]*", "", source[a:source.index("\n];", a)])
    return re.findall(r'"([A-Za-z0-9_]+)"', body)


def _param_keys(source: str) -> List[str]:
    return _string_list(
        source, "const PARAM_KEYS: (keyof TrainingRunCreateRequest)[] = [")


def _request_data(source: str) -> str:
    return _slice(source, "const getRequestData", "const applyParamsToState")


def _request_keys(source: str) -> Set[str]:
    """Every key the request carries: the ones the literal names, plus the
    PARAM_KEYS ``passThroughParams`` copies (all but COMPUTED_REQUEST_KEYS)."""
    # depth 6 is the literal; depth 8 is a conditional spread inside it.
    literal = set(re.findall(r"(?:^ {6,8}|\{ )([a-z_0-9]+):",
                             _request_data(source), re.M))
    computed = set(_string_list(source, "const COMPUTED_REQUEST_KEYS = new Set<string>(["))
    return literal | (set(_param_keys(source)) - computed)


def snake_to_camel(key: str) -> str:
    """The frontend's own rule, restated so a change to it fails here."""
    return re.sub(r"_([a-z0-9])", lambda m: m.group(1).upper(), key)


# Deliberate exclusions. Each identifies a run, a machine, or a one-shot
# dataset action rather than describing how to train. Adding one means editing
# this list too, which is the point: it is a choice, not a drift.
EXPECTED_EXCLUSIONS: Set[str] = {
    "dataset_configs",          # dataset identity
    "base_model_path",          # model identity
    "run_name",                 # the new run names itself
    "training_method",          # stored as the preset's own column
    "gpu_index",                # which device this machine should use
    # Was saved before this refactor; excluding it is a deliberate narrowing.
    # A preset restoring it makes a new run continue from an unrelated run's
    # checkpoint without saying so.
    "resume_from_checkpoint",
    # Not "it does work at load" -- it scans at run start exactly as
    # configured, which is what a preference does. The reason is that the
    # preset excludes dataset_configs, so a "force" carried into an unrelated
    # dataset is a rescan the user never chose.
    "rescan_before_training",
}

# Request fields applyParamsToState() restores outside the PARAM_KEYS loop
# (custom coercion or a nested object). Saved like any other field.
EXPECTED_EXTRA_RESTORE: Set[str] = {
    "base_resolutions", "regularization_type", "vision_encoder_path",
    "controlnet_pretrained_path", "condition_preprocessors",
    "timestep_sampling", "priority_training",
}

# Every camelCase key the pre-change getCurrentConfig() wrote. Presets on disk
# carry exactly these, so each one must still resolve.
LEGACY_PRESET_KEYS: List[str] = [
    "useEpochs", "totalSteps", "epochs", "batchSize",
    "gradientAccumulationSteps", "maxGradNorm", "learningRate", "lrScheduler",
    "optimizer", "optimizerCautious", "optimizerBeta1", "optimizerBeta2",
    "optimizerEpsilon", "optimizerWeightDecay", "optimizerScheduleFree",
    "optimizerScheduleFreeR", "optimizerScheduleFreeWeightLrPower", "loraRank",
    "loraAlpha", "loraDtype", "adapterAlgorithm", "weightDecompose",
    "adapterConfig", "saveEvery", "saveEveryUnit", "maxStepSavesToKeep",
    "maxOptimizerSavesToKeep", "sampleEvery", "resumeFromCheckpoint",
    "samplePrompts", "sampleWidth", "sampleHeight", "sampleSteps",
    "sampleCfgScale", "sampleSampler", "sampleScheduleType",
    "sampleCfgScheduleType", "sampleCfgScheduleMin", "sampleCfgScheduleMax",
    "sampleCfgSchedulePower", "sampleCfgRescaleSnrAlpha",
    "sampleDynamicThresholdPercentile", "sampleDynamicThresholdMimicScale",
    "sampleNagEnable", "sampleNagScale", "sampleNagTau", "sampleNagAlpha",
    "sampleNagSigmaEnd", "sampleNagNegativePrompt", "sampleSeed",
    "sensenovaSampleTimestepShift", "sensenovaSampleImgCfgScale",
    "sensenovaSampleCfgNorm", "debugLatents", "debugLatentsEvery",
    "enableBucketing", "baseResolutions", "bucketStrategy",
    "multiResolutionMode", "cropAugmentEnable", "cropFullImageProb",
    "cropMaxBucketProb", "cropMinAreaRatio", "cropMinShortSidePx",
    "cropAspectMode", "cropPositionMode", "cropSmallerBucketMode",
    "cropSmallerScaleRange", "fullCropPositionMode", "cropMicrocondMode",
    "cropPlanSeed", "cacheLatentsToDisk", "forceRecache", "trainUnet",
    "trainTextEncoder", "unetLr", "textEncoderLr", "textEncoder1Lr",
    "textEncoder2Lr", "weightDtype", "trainingDtype", "outputDtype",
    "vaeDtype", "mixedPrecision", "attentionBackend", "attentionImpl",
    "useFlashAttention", "minSnrGamma", "reconstructionLossWeight",
    "textEncodingMode", "textEncodingSwapInterval", "latentEncodingMode",
    "latentEncodingSwapInterval", "blocksToSwap", "usePinnedMemory",
    "sensenovaMotPhaseEviction", "sensenovaFourPhaseEviction",
    "sensenovaFourPhaseSharedPrefix", "sensenovaFourPhaseGradReduction",
    "sensenovaFullFinetuneSaveFormat", "sensenovaSampleKvCacheStreaming",
    "sensenovaMotPageableStaging", "sensenovaMotOverlapTransfer",
    "sensenovaTrainFmModules", "optimizerStateHostResident",
    "numOptimizerGroups", "multiNoiseTimesteps", "timestepDistribution",
    "timestepMin", "timestepMax", "controlnetType", "controlnetPretrainedPath",
    "controlnetInitFromUnet", "llliteConditioningChannels", "llliteRank",
    "conditionPreprocessors", "conditionCacheMode", "reloraMergeEvery",
    "reloraMergeUnit", "restartWarmupSteps", "optimizerResetStrategy",
    "optimizerPruningRatio",
]

# Flat camel keys that predate the nested timestep_sampling object, and the
# component state that has no request field. Neither is a mechanical mapping.
LEGACY_TIMESTEP_KEYS = {
    "timestepDistribution", "timestepMin", "timestepMax",
    "timestepMean", "timestepStd", "timestepAlpha", "timestepBeta",
}
NON_REQUEST_PRESET_KEYS = {"useEpochs"}

# Numeric-text controls a preset must be able to CLEAR, each paired with the
# setter whose restore branch has to be null-safe.
EXPECTED_CLEARABLE = (
    ("unet_lr", "setLocalUnetLrText"),
    ("text_encoder_lr", "setLocalTextEncoderLrText"),
    ("text_encoder_1_lr", "setLocalTextEncoder1LrText"),
    ("text_encoder_2_lr", "setLocalTextEncoder2LrText"),
    ("image_encoder_lr", "setLocalImageEncoderLrText"),
    ("optimizer_beta1", "setLocalBeta1Text"),
    ("optimizer_beta2", "setLocalBeta2Text"),
    ("optimizer_epsilon", "setLocalEpsilonText"),
    ("optimizer_weight_decay", "setLocalWeightDecayText"),
)

CLEARABLE_DECL = "const PRESET_CLEARABLE_NUMERIC_KEYS: string[] = ["

# Keys of the two nested objects getRequestData() builds inline. They are
# restored through timestep_sampling / priority_training, not as top-level
# fields, so the reverse check must not demand a list entry for them.
_NESTED_OBJECT_MEMBERS = {
    "distribution", "min_timestep", "max_timestep", "mean", "std",
    "alpha", "beta", "entries", "multiplier",
}


class ScanSanityTest(unittest.TestCase):
    """A scan that found nothing would make every assertion below vacuous."""

    def test_the_panel_and_its_key_regions_are_found(self):
        source = _source()
        self.assertGreater(len(source), 100_000)
        keys = _param_keys(source)
        self.assertGreater(len(keys), 200, "PARAM_KEYS did not parse")
        self.assertIn("lora_rank", keys)
        self.assertIn("weight_decompose", keys)
        self.assertGreater(len(_request_data(source)), 10_000)


class CoverageGateTest(unittest.TestCase):
    """Every restorable parameter reaches a saved preset."""

    def test_every_param_key_is_written_by_get_request_data(self):
        source = _source()
        missing = sorted(set(_param_keys(source)) - _request_keys(source))
        self.assertEqual(
            missing, [],
            "these params are restored but never sent, so a preset cannot "
            f"carry them: {missing}")

    def test_every_param_key_survives_a_preset_round_trip(self):
        """The gate proper: written by getRequestData, or deliberately excluded."""
        source = _source()
        excluded = set(_string_list(source, "const PRESET_EXCLUDED_KEYS: string[] = ["))
        uncovered = sorted(
            (set(_param_keys(source)) - excluded) - _request_keys(source))
        self.assertEqual(
            uncovered, [],
            "add the parameter to getRequestData() (preferred) or name it in "
            f"PRESET_EXCLUDED_KEYS: {uncovered}")

    def test_every_request_key_is_accounted_for(self):
        """The converse direction, which is what catches the NEXT
        rescan_before_training: a field added to getRequestData() and to no
        list at all is sent by the form, saved into presets, and restored by
        nothing."""
        source = _source()
        known = (set(_param_keys(source))
                 | EXPECTED_EXTRA_RESTORE | EXPECTED_EXCLUSIONS
                 | _NESTED_OBJECT_MEMBERS)
        written = _request_keys(source)
        self.assertIn("batch_size", written)  # the region really parsed
        self.assertEqual(sorted(written - known), [],
                         "sent by the form but in no list: add it to PARAM_KEYS "
                         "(restored), PARAM_EXTRA_RESTORE_KEYS, or "
                         "PRESET_EXCLUDED_KEYS")

    def test_the_exclusion_list_is_exactly_what_was_decided(self):
        excluded = set(_string_list(
            _source(), "const PRESET_EXCLUDED_KEYS: string[] = ["))
        self.assertEqual(excluded, EXPECTED_EXCLUSIONS)

    def test_the_extra_restore_list_is_exactly_what_was_decided(self):
        extra = set(_string_list(
            _source(), "const PARAM_EXTRA_RESTORE_KEYS: string[] = ["))
        self.assertEqual(extra, EXPECTED_EXTRA_RESTORE)

    def test_every_named_key_is_a_real_request_field(self):
        """Catches a typo in any of the three lists."""
        api = _API.read_text(encoding="utf-8")
        start = api.index("export interface TrainingRunCreateRequest {")
        body = api[start:api.index("\n}\n", start)]
        fields = set(re.findall(r"^\s{2}([a-z][A-Za-z0-9_]*)\??:", body, re.M))
        self.assertIn("weight_decompose", fields)  # the interface really parsed
        for key in (set(_param_keys(_source()))
                    | EXPECTED_EXCLUSIONS | EXPECTED_EXTRA_RESTORE):
            self.assertIn(key, fields, key)


class PayloadIsDerivedTest(unittest.TestCase):
    """The save and load paths hold no second list of parameters."""

    def _current_config(self, source: str) -> str:
        return _slice(source, "const getCurrentConfig", "const handleSavePreset")

    def test_get_current_config_derives_from_the_request(self):
        body = self._current_config(_source())
        self.assertIn("Object.entries(getRequestData())", body)
        self.assertIn("PRESET_EXCLUDED_KEYS", body)

    def test_get_current_config_holds_no_hand_written_field_list(self):
        """What regressed before: one ``camelCase: params.x`` line per field."""
        body = self._current_config(_source())
        self.assertNotIn(": params.", body)
        # The steps/epochs pair is the ONE deliberate hand-written addition.
        params_reads = re.findall(r"params\.([a-z_]+)", body)
        self.assertEqual(sorted(set(params_reads)), ["epochs", "total_steps"])
        self.assertLess(len(body.splitlines()), 40, "a list is growing back")

    def test_the_loader_routes_through_the_shared_restore_path(self):
        body = _slice(_source(), "const handleLoadPreset", "const handleDeletePreset")
        self.assertIn("applyParamsToState(presetConfigToParams(config))", body)
        # The radio has no request field, so it stays explicit.
        self.assertIn("setUseEpochs(config.useEpochs)", body)
        self.assertIn("setTrainingMethod(preset.training_method)", body)
        self.assertNotIn("updateParam(", body)


class OldPresetsStillLoadTest(unittest.TestCase):
    """Every key a stored preset can contain resolves to somewhere."""

    def _camel_table(self) -> Dict[str, str]:
        keys = set(_param_keys(_source())) | EXPECTED_EXTRA_RESTORE
        return {snake_to_camel(key): key for key in keys}

    def test_the_camel_spelling_is_unambiguous(self):
        """Two snake keys colliding on one camel name would lose a value."""
        keys = sorted(set(_param_keys(_source())) | EXPECTED_EXTRA_RESTORE)
        camel: Dict[str, str] = {}
        for key in keys:
            other = camel.setdefault(snake_to_camel(key), key)
            self.assertEqual(other, key, f"{other} and {key} share a spelling")

    def test_every_legacy_preset_key_resolves(self):
        table = self._camel_table()
        unresolved = [
            key for key in LEGACY_PRESET_KEYS
            if key not in table
            and key not in LEGACY_TIMESTEP_KEYS
            and key not in NON_REQUEST_PRESET_KEYS
        ]
        self.assertEqual(unresolved, [],
                         f"presets on disk carry keys nothing reads: {unresolved}")

    def test_a_legacy_key_for_a_now_excluded_field_is_dropped_on_purpose(self):
        """`resumeFromCheckpoint` WAS saved before the payload was derived.
        It resolves to a known snake key, so nothing throws -- and then put()
        drops it because the key is excluded. That narrowing is the decision,
        not an accident: a preset should not make a new run continue from an
        unrelated run's checkpoint."""
        self.assertIn("resumeFromCheckpoint", LEGACY_PRESET_KEYS)
        self.assertIn("resume_from_checkpoint", self._camel_table().values())
        self.assertIn("resume_from_checkpoint", EXPECTED_EXCLUSIONS)
        body = _slice(_source(), "export function presetConfigToParams",
                      "export default function TrainingConfig")
        # put() is the only writer and it checks the exclusion set first, so an
        # excluded key is dropped whichever spelling it arrives under.
        self.assertIn("if (excluded.has(key) || value === undefined) return;", body)

    def test_the_legacy_list_is_the_real_one(self):
        """Sanity on the fixture: it is the size and shape it was measured at."""
        self.assertEqual(len(LEGACY_PRESET_KEYS), 122)
        self.assertEqual(len(set(LEGACY_PRESET_KEYS)), 122)
        for key in ("learningRate", "weightDecompose", "useFlashAttention"):
            self.assertIn(key, LEGACY_PRESET_KEYS)

    def test_camel_to_snake_is_not_mechanical_so_the_table_must_be_built(self):
        """The counterexample that forbids a runtime camel -> snake regex.

        snake -> camel is lossless; the inverse is not, because a digit gives
        no case signal. ``optimizer_beta1`` and a hypothetical
        ``optimizer_beta_1`` share one camel spelling, and the naive inverse
        picks the wrong one -- which is why PRESET_CAMEL_TO_SNAKE is built FROM
        the snake keys instead.
        """
        naive_inverse = re.sub(r"(?<!^)(?=[A-Z0-9])", "_",
                               snake_to_camel("optimizer_beta1")).lower()
        self.assertEqual(snake_to_camel("optimizer_beta1"), "optimizerBeta1")
        self.assertEqual(naive_inverse, "optimizer_beta_1")
        self.assertNotEqual(naive_inverse, "optimizer_beta1")
        source = _source()
        self.assertIn("for (const key of PRESET_RESTORABLE_KEYS) "
                      "map[snakeToCamel(key)] = key;", source)

    def test_the_frontend_rule_matches_the_one_asserted_here(self):
        self.assertIn(
            r'key.replace(/_([a-z0-9])/g, (_m, c: string) => c.toUpperCase())',
            _source())

    def test_the_normalizer_handles_the_documented_special_cases(self):
        source = _source()
        body = _slice(source, "export function presetConfigToParams",
                      "export default function TrainingConfig")
        # Legacy flat timestep keys fold into the nested object. min/max are the
        # one place the two spellings are not the same word.
        table = _slice(source, "const PRESET_LEGACY_TIMESTEP_KEYS",
                       "const PRESET_NUMERIC_TEXT_KEYS")
        for camel, nested in (("timestepDistribution", "distribution"),
                              ("timestepMin", "min_timestep"),
                              ("timestepMax", "max_timestep"),
                              ("timestepMean", "mean"), ("timestepStd", "std"),
                              ("timestepAlpha", "alpha"), ("timestepBeta", "beta")):
            self.assertIn(f'{camel}: "{nested}"', table)
        self.assertIn("incoming.timestep_sampling = { ...timestep,", body)
        # R6 attention compat, unchanged from the loader it replaced.
        self.assertIn('incoming.attention_backend = incoming.use_flash_attention '
                      '? "flash" : "native";', body)
        self.assertIn('incoming.use_flash_attention = incoming.attention_backend '
                      '=== "flash";', body)
        # Old presets stored the raw text of the scientific-notation controls.
        self.assertIn("PRESET_NUMERIC_TEXT_KEYS.has(key)", body)
        self.assertIn("const parsed = parseFloat(value);", body)

    def test_an_unknown_key_is_dropped_rather_than_thrown_on(self):
        """The removed ``optimizerIsPaged`` is still in every old preset blob."""
        source = _source()
        body = _slice(source, "export function presetConfigToParams",
                      "export default function TrainingConfig")
        # Nothing indexes a params object with an unvalidated key: every write
        # goes through put(), and put() is only reached from a known name.
        self.assertIn("const snake = PRESET_CAMEL_TO_SNAKE[key];", body)
        self.assertIn("if (snake !== undefined && incoming[snake] === undefined)",
                      body)
        # The forward-compat note survives, and only as a comment -- see
        # optimizer_is_paged_removal_test.
        for number, line in enumerate(source.splitlines(), start=1):
            if "optimizerIsPaged" in line:
                self.assertTrue(line.strip().startswith("*"), f"line {number}")
                break
        else:
            self.fail("the optimizerIsPaged forward-compat note was lost")


class EmptyControlsClearOnLoadTest(unittest.TestCase):
    """A preset saved with an empty numeric box must CLEAR that box.

    ``getRequestData()`` reads the TEXT state at submit, not ``params``, so a
    box still holding the previous preset's value is what the run trains with.
    The hand-written loader set the text unconditionally, including ``""``;
    routing through ``applyParamsToState``, whose guards were ``!== null``,
    made the stale value reachable on the preset path. Scenario: type a U-Net
    LR, then load a preset saved with that box empty -- the form shows the
    typed value and the run uses it instead of inheriting the base LR.
    """

    def test_the_clearable_list_is_exactly_what_was_decided(self):
        keys = _string_list(_source(), CLEARABLE_DECL)
        self.assertEqual(sorted(keys), sorted(key for key, _ in EXPECTED_CLEARABLE))

    def test_clearable_is_a_subset_of_the_numeric_text_keys(self):
        """put() only reaches the clearable branch for a numeric-text key, so a
        clearable key outside that set could never be cleared from an old
        preset's empty string."""
        source = _source()
        numeric = set(_string_list(
            source, "const PRESET_NUMERIC_TEXT_KEYS = new Set<string>(["))
        clearable = set(_string_list(source, CLEARABLE_DECL))
        self.assertTrue(clearable, "the clearable list did not parse")
        self.assertEqual(sorted(clearable - numeric), [])

    def test_every_clearable_key_has_a_null_safe_restore(self):
        source = _source()
        for key, setter in EXPECTED_CLEARABLE:
            with self.subTest(key=key):
                self.assertIn(
                    setter + "(incoming." + key + ' != null ? String(incoming.'
                    + key + ') : "");',
                    source)
                self.assertNotIn("incoming." + key + " !== null", source)

    def test_learning_rate_and_schedule_free_are_deliberately_not_clearable(self):
        """Negative control. Their branches call .toString() on the value, so a
        null would throw, and the main LR is never legitimately unset."""
        source = _source()
        keys = set(_string_list(source, CLEARABLE_DECL))
        for key in ("learning_rate", "optimizer_schedule_free_r",
                    "optimizer_schedule_free_weight_lr_power"):
            with self.subTest(key=key):
                self.assertNotIn(key, keys)
                self.assertIn("incoming." + key + ".toString()", source)

    def test_an_empty_box_is_saved_as_an_explicit_null(self):
        """A missing key means "leave it alone", so empty needs a value."""
        body = _slice(_source(), "const getCurrentConfig", "const handleSavePreset")
        self.assertIn("for (const key of PRESET_CLEARABLE_NUMERIC_KEYS) {", body)
        self.assertIn("if (config[key] === undefined) config[key] = null;", body)

    def test_a_legacy_empty_string_becomes_null_not_a_dropped_key(self):
        """Old presets stored "" for an empty box; dropping it would leave the
        previous value in place, which is the bug this class is about."""
        body = _slice(_source(), "export function presetConfigToParams",
                      "export default function TrainingConfig")
        self.assertIn("if (clearable.has(key)) incoming[key] = null;", body)


class OptimizerHyperparametersSurvivePresetLoadTest(unittest.TestCase):
    """A preset that also changes the optimizer keeps its own betas.

    The effect keyed on ``params.optimizer`` replaces beta1/beta2/epsilon/
    weight_decay and their text with the new optimizer's defaults, and ran
    unguarded on the preset path -- so a preset carrying a tuned beta1 and a
    different optimizer silently lost the beta1.
    """

    EFFECT_START = "// Reset optimizer hyperparameters when optimizer changes"
    UNSUPPORTED = "// Reset options that are not supported by the new optimizer"

    def test_the_loader_arms_the_guard_before_restoring_and_disarms_it_after(self):
        body = _slice(_source(), "const handleLoadPreset", "const handleDeletePreset")
        armed = body.index("skipOptimizerHyperparamResetRef.current = true;")
        restored = body.index("applyParamsToState(")
        self.assertLess(armed, restored, "the guard must be set BEFORE the restore")
        # The effect may not fire at all (identical optimizer), so it cannot own
        # the reset without poisoning the next genuine optimizer change.
        self.assertIn(
            "setTimeout(() => { skipOptimizerHyperparamResetRef.current = false; }, 0);",
            body)

    def test_only_the_hyperparameter_half_is_skipped(self):
        """Blast radius. The unsupported-option clearing below it must still
        run, or the preset parks a ticked box the chosen optimizer ignores.
        That is why the guard is its own ref and not ``restoringFromYAMLRef``,
        which would also disable the value-keyed capability-clearing effects."""
        effect = _slice(_source(), self.EFFECT_START, "const loadDatasets")
        guard = effect.index("if (!skipOptimizerHyperparamResetRef.current) {")
        boundary = effect.index(self.UNSUPPORTED)
        for inside in ('updateParam("optimizer_beta1", parseFloat(beta1));',
                       'updateParam("optimizer_epsilon", parseFloat(epsilon));',
                       'updateParam("optimizer_weight_decay", parseFloat(weight_decay));'):
            with self.subTest(line=inside):
                position = effect.index(inside)
                self.assertGreater(position, guard)
                self.assertLess(position, boundary)
        for outside in ('updateParam("optimizer_cautious", false);',
                        'updateParam("optimizer_state_host_resident", false);'):
            with self.subTest(line=outside):
                self.assertGreater(effect.index(outside), boundary)

    def test_the_preset_carries_the_hyperparameters_it_is_protecting(self):
        request = _request_data(_source())
        for key in ("optimizer_beta1", "optimizer_beta2", "optimizer_epsilon",
                    "optimizer_weight_decay"):
            with self.subTest(key=key):
                self.assertIn(key + ": ", request)
                self.assertNotIn(key, EXPECTED_EXCLUSIONS)


if __name__ == "__main__":
    unittest.main()
