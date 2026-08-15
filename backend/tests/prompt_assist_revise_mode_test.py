"""Revise mode for all three MiniMax prompt assistants (H3, Music 3 caption
rewriter, Music 3 lyrics assistant): a separate `instruction` field plus an
explicit `revise` flag, distinct from the existing "expand" behavior.

Covers, per the task's own checklist:
  - the BPM/key/exclusion allow-set now honouring `instruction` (the caption
    rewriter's `known_text` trap -- the actual reported failure class);
  - cache-key sensitivity to `instruction` and `revise` in all three modules
    (each hashes a different material dict, so each is checked separately);
  - revise output still passing the SAME structural validator as the mode
    it revises (no new validator invented for revise mode);
  - the diff summary (`_summarize_diff`), used to show what actually
    changed since "only the named parts changed" is not machine-checkable;
  - H3's no-instruction path being byte-identical to its pre-revise-mode
    behavior, proved by computing both sides rather than asserted.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.extensions.minimax_h3_prompt_assistant import (  # noqa: E402
    MiniMaxH3PromptAssistant,
    PromptAssistError,
    PromptAssistOptions,
    _summarize_diff,
    _system_prompt as h3_system_prompt,
    _user_message as h3_user_message,
    validate_prompt,
)
from core.extensions.minimax_music3_caption_rewriter import (  # noqa: E402
    MiniMaxMusic3CaptionRewriter,
    MusicCaptionAssistOptions,
    SECTION_NAMES as CAPTION_SECTION_NAMES,
    validate_caption,
)
from core.extensions.minimax_music3_lyrics_assistant import (  # noqa: E402
    MiniMaxMusic3LyricsAssistant,
    MusicLyricsAssistOptions,
    validate_structure_lyrics,
)


def _body(words: int) -> str:
    return " ".join(["tone"] * words)


def _structured_caption(global_words=90, vocal_words=90, arrangement_words=90) -> str:
    return (
        f"{CAPTION_SECTION_NAMES[0]}\n{_body(global_words)}\n\n"
        f"{CAPTION_SECTION_NAMES[1]}\n{_body(vocal_words)}\n\n"
        f"{CAPTION_SECTION_NAMES[2]}\n{_body(arrangement_words)}"
    )


def _mock_response(payload):
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


# ---------------------------------------------------------------------------
# H3: no-instruction path must be byte-identical to pre-revise-mode output.
# ---------------------------------------------------------------------------

class H3NoInstructionByteIdenticalTest(unittest.TestCase):
    """The golden strings below are copied verbatim from the module's
    `_system_prompt` template as it existed BEFORE revise mode was added
    (see the module history) -- reconstructed independently here, not by
    calling the module, so a regression in the expand path is actually
    caught rather than trivially passing against itself."""

    def _golden_system_prompt(self, mode: str, duration: float, inventory: str, mode_rule: str) -> str:
        return f"""You transform a user's intent into one MiniMax-H3 {"full-reference" if mode == "ref2va" else "base"} video prompt.
Return exactly one JSON object: {{"prompt":"...","warnings":["..."]}}.
The JSON prompt value must be the complete formatted MiniMax-H3 prompt, not a summary or a plain sentence.

Fidelity rules:
- Preserve every user-stated identity, action, style, composition, name, number, constraint, dialogue line, lyric, and visible text.
- Convert Danbooru-style tags into natural English without silently omitting or contradicting them.
- Never invent dialogue, lyrics, quotations, visible text, reference-asset details, or speaker identity.
- Dialogue, lyrics, and visible text remain verbatim and untranslated; all other output is English.
- Use only Picture, Video, and Audio labels in this inventory: {inventory}.
- In full-reference mode you may define sequential Subject labels from the user's stated intent, but never invent unsupported visual details.
- If pixels or descriptions are unavailable, do not claim visual details that were not supplied.
- Shot 1 has no timestamp. Later shots use [Shot N] At MM:SS.mmm with strictly increasing times below {duration:.3f} seconds.
- The main description must explicitly begin its timeline with [Shot 1]. Separate every required section with one blank line.
- Put dialogue, singing, and shot-synchronised sound in the main description; ambience in overall_soundscape; audience-only music in non_diegetic_music.
- If there is no audience-only music, write exactly non_diegetic_music: N/A, never None.
- Output no Markdown or alternatives.
- Never collapse the required sections into a one-sentence prompt.

Mode requirement: {mode_rule}
Base output order: integrated_multimodal_description, overall_soundscape, non_diegetic_music.
Full-reference output order: subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music.
For full-reference generation, make detailed_description explicit and useful, but do not pad it with invented facts."""

    def test_t2va_system_prompt_unchanged_with_no_instruction(self):
        golden = self._golden_system_prompt(
            "t2va", 5.0, "[]",
            "Begin directly with the three base fields; there is no alignment instruction.",
        )
        actual = h3_system_prompt("t2va", 5.0, [], revise=False, instruction="")
        self.assertEqual(actual, golden)

    def test_t2va_system_prompt_unchanged_when_revise_omitted_entirely(self):
        # Default-argument call path, matching every pre-existing call site
        # that never passes revise/instruction at all.
        golden = self._golden_system_prompt(
            "t2va", 5.0, "[]",
            "Begin directly with the three base fields; there is no alignment instruction.",
        )
        self.assertEqual(h3_system_prompt("t2va", 5.0, []), golden)

    def test_i2va_system_prompt_unchanged_with_no_instruction(self):
        from core.extensions.minimax_h3_prompt_assistant import _alignment_instruction
        golden = self._golden_system_prompt(
            "i2va", 6.0, "[]", _alignment_instruction("i2va", 6.0),
        )
        actual = h3_system_prompt("i2va", 6.0, [], revise=False, instruction="")
        self.assertEqual(actual, golden)

    def test_ref2va_system_prompt_unchanged_with_no_instruction(self):
        golden = self._golden_system_prompt(
            "ref2va", 5.0, "[]",
            "Use exactly the six full-reference sections in the documented order.",
        )
        actual = h3_system_prompt("ref2va", 5.0, [], revise=False, instruction="")
        self.assertEqual(actual, golden)

    def test_user_message_is_exactly_the_prompt_when_not_revising(self):
        options = PromptAssistOptions(
            prompt="1girl, red dress, walking through rain",
            mode="t2va", duration_seconds=5.0, references=[],
            provider="lm_studio", base_url="http://127.0.0.1:1234",
            model="local/test-model", temperature=0.2, top_p=0.9,
            max_output_tokens=512, context_length=4096, timeout_seconds=30,
        )
        self.assertEqual(h3_user_message(options), options.prompt)

    def test_revise_block_only_appears_when_revise_is_true(self):
        expand = h3_system_prompt("t2va", 5.0, [], revise=False, instruction="ignored")
        revised = h3_system_prompt("t2va", 5.0, [], revise=True, instruction="make it darker")
        self.assertNotIn("REVISE MODE", expand)
        self.assertIn("REVISE MODE", revised)
        self.assertIn("make it darker", revised)


class H3RevisePayloadTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.assistant = MiniMaxH3PromptAssistant(
            8, Path(self.temp_dir.name) / "prompt-cache.sqlite3"
        )
        self.base_prompt = (
            "integrated_multimodal_description: [Shot 1] A woman in a red dress walks through rain.\n\n"
            "overall_soundscape: Rain falls steadily on pavement.\n\n"
            "non_diegetic_music: N/A"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def options(self, **changes):
        values = {
            "prompt": self.base_prompt,
            "mode": "t2va",
            "duration_seconds": 5.0,
            "references": [],
            "provider": "lm_studio",
            "base_url": "http://127.0.0.1:1234",
            "model": "local/test-model",
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 512,
            "context_length": 4096,
            "timeout_seconds": 30,
            "force_refresh": True,
        }
        values.update(changes)
        return PromptAssistOptions(**values)

    def test_revise_without_instruction_is_rejected(self):
        with self.assertRaises(PromptAssistError):
            self.assistant.transform(self.options(revise=True, instruction="   "))

    def test_revise_cache_key_differs_from_expand_cache_key(self):
        expand_key = self.assistant._cache_key(self.options(revise=False))
        revise_key = self.assistant._cache_key(self.options(revise=True, instruction="make the drop harder"))
        self.assertNotEqual(expand_key, revise_key)

    def test_revise_cache_key_differs_by_instruction(self):
        key_a = self.assistant._cache_key(self.options(revise=True, instruction="make it darker"))
        key_b = self.assistant._cache_key(self.options(revise=True, instruction="make it brighter"))
        self.assertNotEqual(key_a, key_b)

    @patch("core.extensions.minimax_h3_prompt_assistant.requests.post")
    def test_revise_output_passes_the_same_structural_validator_and_reports_a_diff(self, post):
        revised_prompt = (
            "integrated_multimodal_description: [Shot 1] A woman in a red dress runs through pounding rain.\n\n"
            "overall_soundscape: Thunder cracks as rain hammers the pavement.\n\n"
            "non_diegetic_music: N/A"
        )
        post.side_effect = [
            _mock_response({"instance_id": "instance-1"}),
            _mock_response({"output": [{"type": "message", "content": json.dumps({"prompt": revised_prompt, "warnings": []})}]}),
            _mock_response({"status": "unloaded"}),
        ]

        result = self.assistant.transform(
            self.options(revise=True, instruction="make the rain much harder")
        )

        # Same validator as expand mode -- no new/looser validator for revise.
        self.assertEqual(validate_prompt(result["prompt"], "t2va", 5.0, []), [])
        self.assertTrue(result["valid"])
        self.assertTrue(result["revise"])
        self.assertIsNotNone(result["diff_summary"])
        self.assertIn("-overall_soundscape: Rain falls steadily on pavement.", result["diff_summary"])
        self.assertIn("+overall_soundscape: Thunder cracks as rain hammers the pavement.", result["diff_summary"])

    @patch("core.extensions.minimax_h3_prompt_assistant.requests.post")
    def test_expand_mode_diff_summary_is_none(self, post):
        rewritten = (
            "integrated_multimodal_description: [Shot 1] A woman in a red dress walks through rain.\n\n"
            "overall_soundscape: Rain falls on the pavement.\n\n"
            "non_diegetic_music: N/A"
        )
        post.side_effect = [
            _mock_response({"instance_id": "instance-2"}),
            _mock_response({"output": [{"type": "message", "content": json.dumps({"prompt": rewritten, "warnings": []})}]}),
            _mock_response({"status": "unloaded"}),
        ]
        result = self.assistant.transform(self.options(revise=False))
        self.assertIsNone(result["diff_summary"])
        self.assertFalse(result["revise"])


# ---------------------------------------------------------------------------
# Music 3 caption rewriter: the BPM/key/exclusion allow-set trap.
# ---------------------------------------------------------------------------

class CaptionInstructionAllowSetTest(unittest.TestCase):
    """The actual reported failure class: an instruction that legitimately
    states a new BPM/key/exclusion-lift must not be rejected as an
    invented fact just because it arrived via `instruction` instead of
    `caption`/`constraints`. `revise=True` is required on every one of
    these: `instruction` is meaningful to this validator only in revise
    mode (see `validate_caption`'s own docstring/section 4 comment) --
    without it a stray `instruction` argument must have no effect at all."""

    def test_bpm_from_instruction_is_not_flagged_as_invented(self):
        caption = _structured_caption(90, 90, 85) + " Tempo sits around 128 bpm throughout."
        warnings = validate_caption(
            caption, source_caption="a dreamy synth pop track",
            instruction="take it to 128 bpm", revise=True,
        )
        self.assertEqual(warnings, [])

    def test_bpm_without_the_instruction_argument_is_still_flagged(self):
        # Control: the same caption, without threading instruction through,
        # must still be rejected -- proves the allow-set is doing real work
        # rather than having been made permissive some other way.
        caption = _structured_caption(90, 90, 85) + " Tempo sits around 128 bpm throughout."
        warnings = validate_caption(caption, source_caption="a dreamy synth pop track")
        self.assertTrue(any("invent a BPM" in w for w in warnings))

    def test_bpm_from_instruction_is_ignored_when_revise_is_false(self):
        # A caller that sends `instruction` without `revise=True` must not
        # get the allow-set expansion by accident -- `instruction` is only
        # meaningful in revise mode.
        caption = _structured_caption(90, 90, 85) + " Tempo sits around 128 bpm throughout."
        warnings = validate_caption(
            caption, source_caption="a dreamy synth pop track",
            instruction="take it to 128 bpm", revise=False,
        )
        self.assertTrue(any("invent a BPM" in w for w in warnings))

    def test_key_from_instruction_is_not_flagged_as_invented(self):
        caption = _structured_caption(90, 90, 85) + " The song settles into D minor for the bridge."
        warnings = validate_caption(
            caption, source_caption="a moody piano ballad",
            instruction="take it to D minor", revise=True,
        )
        self.assertEqual(warnings, [])

    def test_exclusion_stated_only_in_the_instruction_is_still_enforced(self):
        # Before `instruction` was threaded into the exclusion check, an
        # exclusion stated ONLY in a revision instruction ("no drums this
        # time") was invisible to this check -- it only ever read
        # caption+constraints. A revised caption that ignored that
        # instruction and kept drums must still be caught.
        caption = _structured_caption(90, 90, 85) + " Punchy drums drive the chorus forward."
        warnings = validate_caption(
            caption, source_caption="a driving rock anthem", instruction="No drums.", revise=True,
        )
        self.assertTrue(any("exclusion of 'drum'" in w for w in warnings))


class ExclusionPhraseTrailingFillerTest(unittest.TestCase):
    """The LAST-WORD rule ("no heavy distortion" -> "distortion") takes an
    ordinary English intensifier as the excluded noun whenever the phrase
    ends in one, and instructions are free prose where "no X at all" is
    completely idiomatic. Pins the exact five phrasings the coordinator
    measured (three broken, two already fine) plus one already-fine
    leading-filler form ("without any X"), so both directions are covered:
    filler must strip, and ordinary un-filler-suffixed phrases must be
    completely unaffected by the stripping logic's existence."""

    def test_trailing_at_all_is_stripped(self):
        from core.extensions.minimax_music3_caption_rewriter import _extract_exclusion_stems
        self.assertEqual(_extract_exclusion_stems("no brass at all"), {"bras"})

    def test_trailing_at_all_is_stripped_for_a_different_noun(self):
        from core.extensions.minimax_music3_caption_rewriter import _extract_exclusion_stems
        self.assertEqual(_extract_exclusion_stems("no drums at all"), {"drum"})

    def test_trailing_whatsoever_is_stripped(self):
        from core.extensions.minimax_music3_caption_rewriter import _extract_exclusion_stems
        self.assertEqual(_extract_exclusion_stems("no brass whatsoever"), {"bras"})

    def test_bare_no_brass_is_unaffected_by_the_stripping_logic(self):
        from core.extensions.minimax_music3_caption_rewriter import _extract_exclusion_stems
        self.assertEqual(_extract_exclusion_stems("no brass"), {"bras"})

    def test_leading_any_form_is_unaffected_by_the_stripping_logic(self):
        # "any" is only stripped from the TRAILING end -- a leading "any"
        # ("without any brass") is part of the phrase capture, not the
        # excluded noun itself, and must be left alone.
        from core.extensions.minimax_music3_caption_rewriter import _extract_exclusion_stems
        self.assertEqual(_extract_exclusion_stems("without any brass"), {"bras"})

    def test_trailing_at_any_point_is_stripped(self):
        from core.extensions.minimax_music3_caption_rewriter import _extract_exclusion_stems
        self.assertEqual(_extract_exclusion_stems("no drums at any point"), {"drum"})

    def test_trailing_anywhere_is_stripped(self):
        from core.extensions.minimax_music3_caption_rewriter import _extract_exclusion_stems
        self.assertEqual(_extract_exclusion_stems("no guitar anywhere"), {"guitar"})

    def test_phrase_that_is_entirely_filler_yields_no_exclusion(self):
        # Degenerate: nothing left after stripping -- must not invent an
        # exclusion from a filler word instead.
        from core.extensions.minimax_music3_caption_rewriter import _extract_exclusion_stems
        self.assertEqual(_extract_exclusion_stems("no whatsoever"), set())

    def test_end_to_end_no_brass_at_all_rejects_brass_in_the_output(self):
        caption = _structured_caption(90, 90, 85) + " A bright brass section enters."
        warnings = validate_caption(
            caption, source_caption="a driving rock anthem",
            instruction="No brass at all.", revise=True,
        )
        self.assertTrue(any("exclusion of 'bras'" in w for w in warnings))

    def test_end_to_end_no_brass_at_all_allows_compliant_output(self):
        caption = _structured_caption(90, 90, 85) + " No brass anywhere in the arrangement."
        warnings = validate_caption(
            caption, source_caption="a driving rock anthem",
            instruction="No brass at all.", revise=True,
        )
        self.assertFalse(any("must preserve the exclusion" in w for w in warnings))


class ExclusionPhraseBoundaryRegressionTest(unittest.TestCase):
    """Regression for the exact defect the coordinator measured: joining
    `instruction` (or `constraints`) into a text blob with `\\n` let
    `_EXCLUSION_PHRASE_RE`'s open-ended, non-greedy capture run PAST the
    field boundary, because the old pattern used `\\s` (which matches `\\n`)
    with a terminator of `[,.;:!?]|$` under no MULTILINE -- `$` meant end of
    the WHOLE joined string, not end of line. Since only the LAST word of
    the captured phrase becomes the exclusion stem, a real "no drums"
    exclusion silently turned into a nonsense one taken from whatever
    prose followed it in the next field."""

    def test_exclusion_phrase_stops_at_a_bare_line_boundary(self):
        from core.extensions.minimax_music3_caption_rewriter import _extract_exclusion_stems
        text = "an acoustic folk tune, no drums\n\n"
        self.assertEqual(_extract_exclusion_stems(text), {"drum"})

    def test_exclusion_phrase_does_not_run_on_past_a_line_boundary_into_prose(self):
        # The coordinator's exact measured input: without the fix this
        # produced stem 'half', not 'drum'.
        from core.extensions.minimax_music3_caption_rewriter import _extract_exclusion_stems
        text = "an acoustic folk tune, no drums\n\nadd drums back in the second half"
        self.assertEqual(_extract_exclusion_stems(text), {"drum"})

    def test_caption_constraints_boundary_does_not_leak(self):
        # The pre-existing (not newly introduced) case: caption's "no
        # drums" must not run on into an unpunctuated constraints field.
        caption = _structured_caption(90, 90, 85) + " Punchy drums drive the chorus forward."
        warnings = validate_caption(
            caption,
            source_caption="an acoustic folk tune, no drums",
            constraints="make it longer and more atmospheric",
        )
        self.assertTrue(any("exclusion of 'drum'" in w for w in warnings))
        self.assertFalse(any("exclusion of 'atmospheric'" in w for w in warnings))

    def test_caption_instruction_boundary_does_not_leak(self):
        caption = _structured_caption(90, 90, 85) + " Punchy drums drive the chorus forward."
        warnings = validate_caption(
            caption,
            source_caption="an acoustic folk tune, no drums",
            instruction="make the outro longer",
            revise=True,
        )
        self.assertTrue(any("exclusion of 'drum'" in w for w in warnings))
        self.assertFalse(any("exclusion of 'longer'" in w for w in warnings))

    def test_constraints_instruction_boundary_does_not_leak(self):
        caption = _structured_caption(90, 90, 85) + " No brass anywhere in the mix."
        warnings = validate_caption(
            caption,
            source_caption="a driving rock anthem",
            constraints="no brass",
            instruction="make the outro longer",
            revise=True,
        )
        # "no brass" must still be readable as an exclusion of 'bras' (the
        # stem of 'brass' -- len>3 and ends in 's'), not corrupted into
        # 'longer' by running on into the instruction field. The caption
        # here complies (no brass present), so there must be no exclusion
        # warning of ANY kind, corrupted or not.
        self.assertFalse(any("must preserve the exclusion" in w for w in warnings))


class ExclusionReversalRuleTest(unittest.TestCase):
    """The rule: in revise mode, a BASE exclusion is not enforced if the
    instruction mentions that stem at all (mention-based, not phrase-
    based -- the validator does not try to infer add-back vs reinforce).
    An exclusion the instruction states FRESH stays enforced. Non-revise
    mode is unaffected. All four combinations below."""

    def test_base_exclusion_with_a_silent_instruction_is_enforced(self):
        caption = _structured_caption(90, 90, 85) + " Punchy drums drive the chorus forward."
        warnings = validate_caption(
            caption, source_caption="an acoustic folk tune, no drums",
            instruction="make the outro longer", revise=True,
        )
        self.assertTrue(any("exclusion of 'drum'" in w for w in warnings))

    def test_base_exclusion_mentioned_by_the_instruction_is_not_enforced(self):
        # "Add drums back in the second half" mentions "drums" without
        # itself being a "no X" phrase -- the base exclusion must lift.
        caption = _structured_caption(90, 90, 85) + " Punchy drums drive the chorus forward."
        warnings = validate_caption(
            caption, source_caption="an acoustic folk tune, no drums",
            instruction="add drums back in the second half", revise=True,
        )
        self.assertFalse(any("exclusion of 'drum'" in w for w in warnings))

    def test_instruction_stated_exclusion_is_enforced(self):
        caption = _structured_caption(90, 90, 85) + " Punchy drums drive the chorus forward."
        warnings = validate_caption(
            caption, source_caption="a driving rock anthem",
            instruction="No drums.", revise=True,
        )
        self.assertTrue(any("exclusion of 'drum'" in w for w in warnings))

    def test_instruction_stated_exclusion_with_idiomatic_filler_is_still_enforced(self):
        # Regression for the coordinator's second-round finding: "no brass
        # at all" is completely idiomatic free prose, and the last-word
        # rule used to take "all" as the excluded noun instead of "brass",
        # so this exact case silently enforced nothing. This is case 3 of
        # the four-combination matrix, re-run with the idiom that broke it.
        caption = _structured_caption(90, 90, 85) + " A bright brass section enters."
        warnings = validate_caption(
            caption, source_caption="a driving rock anthem",
            instruction="No brass at all.", revise=True,
        )
        self.assertTrue(any("exclusion of 'bras'" in w for w in warnings))

    def test_non_revise_mode_is_unaffected_by_an_instruction_mentioning_the_stem(self):
        # Same instruction as the "lifted" case above, but revise=False:
        # the base exclusion must still be enforced, exactly as it always
        # has been for a plain (non-revise) call.
        caption = _structured_caption(90, 90, 85) + " Punchy drums drive the chorus forward."
        warnings = validate_caption(
            caption, source_caption="an acoustic folk tune, no drums",
            instruction="add drums back in the second half", revise=False,
        )
        self.assertTrue(any("exclusion of 'drum'" in w for w in warnings))


class CaptionRevisePayloadTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.rewriter = MiniMaxMusic3CaptionRewriter(
            8, Path(self.temp_dir.name) / "music-prompt-cache.sqlite3"
        )
        self.base_caption = _structured_caption(90, 90, 90)

    def tearDown(self):
        self.temp_dir.cleanup()

    def options(self, **changes):
        values = {
            "caption": self.base_caption,
            "lyrics": "",
            "constraints": "",
            "provider": "lm_studio",
            "base_url": "http://127.0.0.1:1234",
            "model": "local/test-model",
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 512,
            "context_length": 4096,
            "timeout_seconds": 30,
            "force_refresh": True,
        }
        values.update(changes)
        return MusicCaptionAssistOptions(**values)

    def test_revise_without_instruction_is_rejected(self):
        with self.assertRaises(PromptAssistError):
            self.rewriter.transform(self.options(revise=True, instruction=""))

    def test_revise_cache_key_differs_from_expand_cache_key(self):
        expand_key = self.rewriter._cache_key(self.options(revise=False))
        revise_key = self.rewriter._cache_key(self.options(revise=True, instruction="make the drop harder"))
        self.assertNotEqual(expand_key, revise_key)

    def test_revise_cache_key_differs_by_instruction(self):
        key_a = self.rewriter._cache_key(self.options(revise=True, instruction="make it darker"))
        key_b = self.rewriter._cache_key(self.options(revise=True, instruction="make it brighter"))
        self.assertNotEqual(key_a, key_b)

    def test_user_message_labels_caption_as_base_text_in_revise_mode(self):
        options = self.options(revise=True, instruction="make the drop harder")
        message = MiniMaxMusic3CaptionRewriter._user_message(options)
        self.assertIn("Current Structured Caption (base text to preserve):", message)
        self.assertIn("Revision instruction (apply this as an edit; do not describe it): make the drop harder", message)

    @patch("core.extensions.minimax_music3_caption_rewriter.requests.post")
    def test_revise_output_passes_the_same_structural_validator_with_a_new_bpm_and_reports_a_diff(self, post):
        revised = _structured_caption(90, 90, 84) + " Tempo now drives at 128 bpm."
        post.side_effect = [
            _mock_response({"instance_id": "instance-1"}),
            _mock_response({"output": [{"type": "message", "content": json.dumps({"prompt": revised, "warnings": []})}]}),
            _mock_response({"status": "unloaded"}),
        ]

        result = self.rewriter.transform(
            self.options(revise=True, instruction="take it to 128 bpm")
        )

        # Same validator as expand mode, WITH instruction threaded through --
        # this is the end-to-end version of the allow-set test above.
        self.assertEqual(
            validate_caption(result["prompt"], "", "", self.base_caption, "take it to 128 bpm", revise=True),
            [],
        )
        self.assertTrue(result["valid"])
        self.assertTrue(result["revise"])
        self.assertIsNotNone(result["diff_summary"])

    @patch("core.extensions.minimax_music3_caption_rewriter.requests.post")
    def test_revise_would_fail_without_instruction_threaded_into_validation(self, post):
        # Demonstrates the trap directly against the live transform() path:
        # if `instruction` were not part of `known_text`, this exact
        # response would fail validation and burn the repair round-trip.
        revised = _structured_caption(90, 90, 84) + " Tempo now drives at 128 bpm."
        structural_warnings_without_instruction = validate_caption(
            revised, "", "", self.base_caption, instruction=""
        )
        self.assertTrue(any("invent a BPM" in w for w in structural_warnings_without_instruction))

        post.side_effect = [
            _mock_response({"instance_id": "instance-2"}),
            _mock_response({"output": [{"type": "message", "content": json.dumps({"prompt": revised, "warnings": []})}]}),
            _mock_response({"status": "unloaded"}),
        ]
        result = self.rewriter.transform(
            self.options(revise=True, instruction="take it to 128 bpm")
        )
        self.assertTrue(result["valid"])


# ---------------------------------------------------------------------------
# Music 3 lyrics assistant: the "preserve verbatim" trap in revise mode, and
# the composition with the existing structure/complete modes.
# ---------------------------------------------------------------------------

class LyricsReviseValidatorDispatchTest(unittest.TestCase):
    """`validate_complete_lyrics` is the wrong validator for revise mode:
    run against the FULL base as "supplied lines to preserve", it would
    reject the very edit the instruction asked for. Revise mode must use
    only the layout/tag contract, not the verbatim-preserve check."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.assistant = MiniMaxMusic3LyricsAssistant(
            8, Path(self.temp_dir.name) / "lyrics-assist-cache.sqlite3"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def options(self, **changes):
        values = {
            "mode": "complete",
            "theme": "",
            "lyrics": "[verse]\nHold on tonight\n[chorus]\nla la la",
            "constraints": "",
            "provider": "lm_studio",
            "base_url": "http://127.0.0.1:1234",
            "model": "local/test-model",
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 512,
            "context_length": 4096,
            "timeout_seconds": 30,
            "force_refresh": True,
        }
        values.update(changes)
        return MusicLyricsAssistOptions(**values)

    def test_revise_complete_does_not_require_the_base_line_to_survive_verbatim(self):
        options = self.options(revise=True, instruction="make the verse darker")
        # The instruction legitimately changed the verse line -- a plain
        # validate_complete_lyrics(options.lyrics, ...) call would reject
        # this exact output for dropping "Hold on tonight".
        edited = "[verse]\nShadows swallow the light\n[chorus]\nla la la"
        self.assertEqual(self.assistant._validate(options, edited), [])

    def test_non_revise_complete_still_requires_the_supplied_line_verbatim(self):
        # Control: the ordinary (non-revise) "complete" mode contract is
        # untouched -- dropping a supplied partial line is still rejected.
        options = self.options(revise=False, lyrics="Hold on tonight")
        edited = "[verse]\nShadows swallow the light\n[chorus]\nla la la"
        warnings = self.assistant._validate(options, edited)
        self.assertTrue(any("not preserved verbatim" in w for w in warnings))

    def test_revise_structure_uses_the_same_tags_only_validator(self):
        options = self.options(mode="structure", revise=True, instruction="add a bridge before the outro",
                                lyrics="[intro]\n[verse]\n[chorus]\n[outro]")
        # A documented tag, so this exercises the "still a tags-only
        # contract" path without also exercising the (separate, expected)
        # undocumented-tag warning.
        good = "[intro]\n[verse]\n[chorus]\n[bridge]\n[outro]"
        self.assertEqual(self.assistant._validate(options, good), [])
        self.assertEqual(self.assistant._validate(options, good), validate_structure_lyrics(good))

    def test_revise_still_rejects_prose_sharing_a_tag_line(self):
        options = self.options(mode="structure", revise=True, instruction="add a breakdown",
                                lyrics="[intro]\n[verse]\n[outro]")
        bad = "[intro]\n[verse] some words\n[outro]"
        warnings = self.assistant._validate(options, bad)
        self.assertTrue(warnings)


class LyricsRevisePayloadTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.assistant = MiniMaxMusic3LyricsAssistant(
            8, Path(self.temp_dir.name) / "lyrics-assist-cache.sqlite3"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def options(self, **changes):
        values = {
            "mode": "complete",
            "theme": "",
            "lyrics": "[verse]\nHold on tonight\n[chorus]\nla la la",
            "constraints": "",
            "provider": "lm_studio",
            "base_url": "http://127.0.0.1:1234",
            "model": "local/test-model",
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 512,
            "context_length": 4096,
            "timeout_seconds": 30,
            "force_refresh": True,
        }
        values.update(changes)
        return MusicLyricsAssistOptions(**values)

    def test_revise_without_instruction_is_rejected(self):
        with self.assertRaises(PromptAssistError):
            self.assistant.transform(self.options(revise=True, instruction=""))

    def test_revise_without_base_lyrics_is_rejected(self):
        with self.assertRaises(PromptAssistError):
            self.assistant.transform(self.options(revise=True, instruction="drop the bridge", lyrics=""))

    def test_revise_cache_key_differs_from_non_revise_cache_key(self):
        expand_key = self.assistant._cache_key(self.options(revise=False))
        revise_key = self.assistant._cache_key(self.options(revise=True, instruction="drop the bridge"))
        self.assertNotEqual(expand_key, revise_key)

    def test_revise_cache_key_differs_by_instruction(self):
        key_a = self.assistant._cache_key(self.options(revise=True, instruction="drop the bridge"))
        key_b = self.assistant._cache_key(self.options(revise=True, instruction="darken the verse"))
        self.assertNotEqual(key_a, key_b)

    @patch("core.extensions.minimax_music3_lyrics_assistant.requests.post")
    def test_revise_complete_mode_end_to_end_reports_a_diff(self, post):
        edited = json.dumps({
            "prompt": "[verse]\nShadows swallow the light\n[chorus]\nla la la",
            "warnings": [],
        })
        post.side_effect = [
            _mock_response({"instance_id": "instance-1"}),
            _mock_response({"output": [{"type": "message", "content": edited}]}),
            _mock_response({"status": "unloaded"}),
        ]

        result = self.assistant.transform(
            self.options(revise=True, instruction="make the verse darker")
        )

        self.assertTrue(result["valid"])
        self.assertTrue(result["revise"])
        self.assertIsNotNone(result["diff_summary"])
        self.assertIn("-Hold on tonight", result["diff_summary"])
        self.assertIn("+Shadows swallow the light", result["diff_summary"])


# ---------------------------------------------------------------------------
# Shared diff-summary helper.
# ---------------------------------------------------------------------------

class SummarizeDiffTest(unittest.TestCase):
    def test_identical_text_produces_an_empty_diff(self):
        self.assertEqual(_summarize_diff("a\nb\nc", "a\nb\nc"), "")

    def test_changed_line_appears_as_removed_and_added(self):
        diff = _summarize_diff("a\nb\nc", "a\nx\nc")
        self.assertIn("-b", diff)
        self.assertIn("+x", diff)
        self.assertNotIn("-a", diff)
        self.assertNotIn("-c", diff)


if __name__ == "__main__":
    unittest.main()
