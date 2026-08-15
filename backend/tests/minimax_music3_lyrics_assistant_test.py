import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.extensions.minimax_music3_lyrics_assistant import (  # noqa: E402
    DOCUMENTED_TAGS,
    LyricsFormatInvariantError,
    MiniMaxMusic3LyricsAssistant,
    MusicLyricsAssistOptions,
    find_lyrics_drop_warnings,
    format_lyrics,
    validate_complete_lyrics,
    validate_structure_lyrics,
)
from core.extensions.minimax_h3_prompt_assistant import PromptAssistError  # noqa: E402


class FormatLyricsInvariantTest(unittest.TestCase):
    """The one property `format_lyrics` must never violate: the ordered
    sequence of non-tag word tokens is identical before and after. Each case
    below is drawn from the design doc's own motivating examples."""

    def test_moves_text_off_a_leading_tag_line_onto_its_own_line(self):
        self.assertEqual(
            format_lyrics("[Verse] The morning light"),
            "[verse]\nThe morning light",
        )

    def test_already_separated_text_is_unchanged_in_content(self):
        # Design doc's own "must NOT change" case.
        result = format_lyrics("[Verse]\nThe morning light")
        self.assertEqual(result, "[verse]\nThe morning light")

    def test_splits_multiple_tags_sharing_a_line(self):
        self.assertEqual(format_lyrics("[Drop] [Build]"), "[drop]\n[build]")

    def test_freeform_tag_passes_through_unchanged(self):
        # Hyphens are the tag author's own formatting, not layout noise --
        # format_lyrics must not collapse them to spaces.
        self.assertEqual(
            format_lyrics("[bass-quartet-rumbles-in]"),
            "[bass-quartet-rumbles-in]",
        )

    def test_drops_blank_line_noise(self):
        result = format_lyrics("The morning light\n\n\n[Chorus] sing it loud")
        self.assertEqual(result, "The morning light\n[chorus]\nsing it loud")

    def test_tag_case_is_lowercased(self):
        self.assertIn("[verse]", format_lyrics("[VERSE]\nhello"))

    def test_word_sequence_survives_a_multi_tag_multi_text_line(self):
        original = "a [x] b [y] c"
        formatted = format_lyrics(original)
        # Word order preserved even with interleaved tags.
        words_only = [line for line in formatted.split("\n") if not line.startswith("[")]
        self.assertEqual(words_only, ["a", "b", "c"])

    def test_invariant_holds_for_representative_inputs(self):
        samples = [
            "[Verse] The morning light",
            "[Verse]\nThe morning light",
            "[Drop] [Build]",
            "[bass-quartet-rumbles-in]",
            "hello [Bridge] world",
            "[Intro]\n\n[Verse]   line one\nline two\n\n\n[Chorus]",
            "no tags at all, just prose here",
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                formatted = format_lyrics(sample)
                from core.extensions.minimax_music3_lyrics_assistant import _word_tokens
                self.assertEqual(_word_tokens(formatted), _word_tokens(sample))

    # F1 regression: `[ ]` (an empty/whitespace-only tag body) used to crash
    # format_lyrics -- `_TAG_RE` matches it on the input side, but lowercasing
    # an all-whitespace body strips it to "", which `_TAG_RE` no longer
    # matches on the output side, breaking the invariant it exists to
    # enforce. Tag emission must be total: an empty-after-lowering body keeps
    # its original (non-empty) content instead.
    def test_empty_bracket_tag_does_not_crash_and_is_preserved(self):
        result = format_lyrics("[ ] hello")
        self.assertEqual(result, "[ ]\nhello")

    def test_empty_bracket_tag_alone_does_not_crash(self):
        # Must not raise LyricsFormatInvariantError.
        format_lyrics("[ ]")


class LyricsDropWarningTest(unittest.TestCase):
    """The generation-time surfacing of the checkpoint's silent-drop defect
    (design doc, "Lyrics assistant")."""

    def test_warns_when_text_shares_a_leading_tag_line(self):
        warnings = find_lyrics_drop_warnings("[Verse] The morning light")
        self.assertEqual(len(warnings), 1)
        self.assertIn("line 1", warnings[0])
        self.assertIn("The morning light", warnings[0])

    def test_no_warning_when_text_is_already_on_its_own_line(self):
        self.assertEqual(find_lyrics_drop_warnings("[Verse]\nThe morning light"), [])

    def test_no_warning_for_tag_only_line(self):
        self.assertEqual(find_lyrics_drop_warnings("[Drop] [Build]"), [])

    def test_no_warning_for_freeform_tag_only_line(self):
        self.assertEqual(find_lyrics_drop_warnings("[bass-quartet-rumbles-in]"), [])

    # F4 regression: the checkpoint's own `_LEADING_TAGS_RE` only tolerates
    # `[ \t]*` before the first tag, so a leading non-breaking space (or any
    # other char `str.strip()` treats as whitespace but `[ \t]` does not)
    # makes the checkpoint's match fail entirely -- the WHOLE raw line
    # passes through unchanged, dropping nothing. The detector must match
    # the raw line, not a `.strip()`-ed copy, or it disagrees with the
    # checkpoint and warns about a drop that never happens.
    def test_no_warning_for_leading_non_breaking_space_before_a_tag(self):
        self.assertEqual(find_lyrics_drop_warnings("\xa0[verse] words"), [])

    def test_warns_once_per_offending_line(self):
        lyrics = "[Verse] line one\n[Chorus]\nline two\n[Bridge] line three"
        warnings = find_lyrics_drop_warnings(lyrics)
        self.assertEqual(len(warnings), 2)
        self.assertIn("line 1", warnings[0])
        self.assertIn("line 4", warnings[1])

    def test_empty_lyrics_produce_no_warning(self):
        self.assertEqual(find_lyrics_drop_warnings(""), [])


class ValidateStructureLyricsTest(unittest.TestCase):
    def test_accepts_tags_only_one_per_line(self):
        self.assertEqual(validate_structure_lyrics("[intro]\n[verse]\n[chorus]\n[outro]"), [])

    def test_rejects_prose_line(self):
        warnings = validate_structure_lyrics("[intro]\nsome narration here")
        self.assertTrue(any("only tags" in w for w in warnings))

    def test_rejects_text_sharing_a_line_with_a_tag(self):
        warnings = validate_structure_lyrics("[verse] some words")
        self.assertTrue(any("shares a line with a tag" in w for w in warnings))

    def test_rejects_empty_output(self):
        warnings = validate_structure_lyrics("")
        self.assertTrue(any("must not be empty" in w for w in warnings))

    def test_warns_on_undocumented_tag_but_does_not_reject_it(self):
        warnings = validate_structure_lyrics("[intro]\n[interlude]\n[outro]")
        self.assertTrue(any("not one of the documented" in w for w in warnings))
        # Still no "only tags" / layout complaint -- an undocumented tag is a
        # warning, never treated as a structural violation.
        self.assertFalse(any("only tags" in w for w in warnings))

    def test_every_documented_tag_is_accepted_without_warning(self):
        lyrics = "\n".join(f"[{tag}]" for tag in sorted(DOCUMENTED_TAGS))
        self.assertEqual(validate_structure_lyrics(lyrics), [])


class ValidateCompleteLyricsTest(unittest.TestCase):
    def test_accepts_output_that_preserves_supplied_lines_verbatim(self):
        supplied = "Hello world\nThis is mine"
        output = "[verse]\nHello world\nThis is mine\n[chorus]\nla la la"
        self.assertEqual(validate_complete_lyrics(supplied, output), [])

    def test_rejects_output_that_paraphrases_a_supplied_line(self):
        supplied = "Hello world"
        output = "[verse]\nHello there world\n[chorus]\nla la"
        warnings = validate_complete_lyrics(supplied, output)
        self.assertTrue(any("not preserved verbatim" in w for w in warnings))

    def test_rejects_output_that_drops_a_supplied_line_entirely(self):
        supplied = "Hello world\nSecond line here"
        output = "[verse]\nHello world\n[chorus]\nla la"
        warnings = validate_complete_lyrics(supplied, output)
        self.assertTrue(any("Second line here" in w for w in warnings))

    def test_ignores_tag_only_lines_in_the_supplied_partial_lyrics(self):
        supplied = "[verse]\nHello world"
        output = "[verse]\nHello world\n[chorus]\nla la"
        self.assertEqual(validate_complete_lyrics(supplied, output), [])

    def test_empty_supplied_lyrics_impose_no_verbatim_requirement(self):
        self.assertEqual(validate_complete_lyrics("", "[verse]\nanything at all"), [])

    def test_reflowed_but_word_identical_line_is_accepted(self):
        # A supplied line the LLM legally reflows into two output lines is
        # NOT the failure mode this check targets (that would require a
        # looser, order-agnostic search); a single joined line is what
        # "verbatim, on its own line" produces after format_lyrics.
        supplied = "Hello world this is mine"
        output = "[verse]\nHello world this is mine\n[chorus]\nla"
        self.assertEqual(validate_complete_lyrics(supplied, output), [])

    # F5 regression: joining word tokens with spaces and doing a raw
    # substring search manufactures word boundaries that were never real --
    # needle "in the rain" is a literal substring of "spin the rainbow
    # drifts" purely across token boundaries ("sp[in the rain]bow drifts"),
    # even though "in the rain" was never actually said. A contiguous
    # token-LIST comparison has no such hole.
    def test_rejects_a_substring_that_only_matches_across_token_boundaries(self):
        supplied = "in the rain"
        output = "[verse]\nspin the rainbow drifts\n[chorus]\nla la"
        warnings = validate_complete_lyrics(supplied, output)
        self.assertTrue(any("not preserved verbatim" in w for w in warnings))


class CacheKeySeparationTest(unittest.TestCase):
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
            "theme": "a bittersweet farewell",
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
            "force_refresh": False,
        }
        values.update(changes)
        return MusicLyricsAssistOptions(**values)

    def test_same_options_produce_same_key(self):
        key_a = self.assistant._cache_key(self.options())
        key_b = self.assistant._cache_key(self.options())
        self.assertEqual(key_a, key_b)

    def test_different_mode_produces_different_key(self):
        key_a = self.assistant._cache_key(self.options(mode="complete"))
        key_b = self.assistant._cache_key(self.options(mode="structure"))
        self.assertNotEqual(key_a, key_b)

    def test_different_theme_produces_different_key(self):
        key_a = self.assistant._cache_key(self.options())
        key_b = self.assistant._cache_key(self.options(theme="a triumphant homecoming"))
        self.assertNotEqual(key_a, key_b)

    def test_different_lyrics_produces_different_key(self):
        key_a = self.assistant._cache_key(self.options())
        key_b = self.assistant._cache_key(self.options(lyrics="Hello world"))
        self.assertNotEqual(key_a, key_b)

    def test_guide_version_differs_from_h3_and_from_caption_rewriter(self):
        # The whole point of this feature's own GUIDE_VERSION and cache
        # file: an identical-looking request must never collide with either
        # sibling's cache.
        from core.extensions.minimax_h3_prompt_assistant import GUIDE_VERSION as H3_GUIDE_VERSION
        from core.extensions.minimax_music3_caption_rewriter import GUIDE_VERSION as CAPTION_GUIDE_VERSION
        from core.extensions.minimax_music3_lyrics_assistant import GUIDE_VERSION as LYRICS_GUIDE_VERSION
        self.assertNotEqual(H3_GUIDE_VERSION, LYRICS_GUIDE_VERSION)
        self.assertNotEqual(CAPTION_GUIDE_VERSION, LYRICS_GUIDE_VERSION)

    def test_cache_files_are_distinct_paths_by_default(self):
        from core.extensions.minimax_music3_caption_rewriter import MiniMaxMusic3CaptionRewriter
        caption_rewriter = MiniMaxMusic3CaptionRewriter(8)
        lyrics_assistant = MiniMaxMusic3LyricsAssistant(8)
        self.assertNotEqual(str(caption_rewriter.cache.path), str(lyrics_assistant.cache.path))
        self.assertIn("lyrics", lyrics_assistant.cache.path.name)


class TransformTransportTest(unittest.TestCase):
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
            "theme": "a bittersweet farewell",
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
            "force_refresh": False,
        }
        values.update(changes)
        return MusicLyricsAssistOptions(**values)

    def response(self, payload):
        response = Mock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        return response

    @patch("core.extensions.minimax_music3_lyrics_assistant.requests.post")
    def test_cache_hit_skips_load_chat_and_unload(self, post):
        good = "[verse]\nHello there\n[chorus]\nla la la"
        post.side_effect = [
            self.response({"instance_id": "instance-1"}),
            self.response({"output": [{"type": "message", "content": json.dumps({"prompt": good, "warnings": []})}]}),
            self.response({"status": "unloaded"}),
        ]

        first = self.assistant.transform(self.options())
        second = self.assistant.transform(self.options())

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(post.call_count, 3)
        self.assertEqual(first["lyrics"], second["lyrics"])

    @patch("core.extensions.minimax_music3_lyrics_assistant.requests.post")
    def test_structure_mode_prose_drives_one_repair_retry(self, post):
        bad = json.dumps({"prompt": "[intro]\nthis is prose, not a tag", "warnings": []})
        good = json.dumps({"prompt": "[intro]\n[verse]\n[chorus]\n[outro]", "warnings": []})
        post.side_effect = [
            self.response({"instance_id": "instance-2"}),
            self.response({"output": [{"type": "message", "content": bad}]}),
            self.response({"output": [{"type": "message", "content": good}]}),
            self.response({"status": "unloaded"}),
        ]

        result = self.assistant.transform(
            self.options(mode="structure", theme="build, drop, breakdown, outro", force_refresh=True)
        )

        self.assertEqual(post.call_count, 4)
        self.assertTrue(result["valid"])

    @patch("core.extensions.minimax_music3_lyrics_assistant.requests.post")
    def test_complete_mode_dropped_line_drives_one_repair_retry(self, post):
        bad = json.dumps({"prompt": "[verse]\nsomething else entirely\n[chorus]\nla", "warnings": []})
        good = json.dumps({"prompt": "[verse]\nHold on tonight\n[chorus]\nla la la", "warnings": []})
        post.side_effect = [
            self.response({"instance_id": "instance-3"}),
            self.response({"output": [{"type": "message", "content": bad}]}),
            self.response({"output": [{"type": "message", "content": good}]}),
            self.response({"status": "unloaded"}),
        ]

        result = self.assistant.transform(
            self.options(mode="complete", lyrics="Hold on tonight", force_refresh=True)
        )

        self.assertEqual(post.call_count, 4)
        self.assertTrue(result["valid"])
        self.assertIn("Hold on tonight", result["lyrics"])

    @patch("core.extensions.minimax_music3_lyrics_assistant.requests.post")
    def test_invalid_llm_json_still_unloads_owned_instance(self, post):
        post.side_effect = [
            self.response({"instance_id": "instance-4"}),
            self.response({"output": [{"type": "message", "content": "not json"}]}),
            self.response({"output": [{"type": "message", "content": "still not json"}]}),
            self.response({"status": "unloaded"}),
        ]

        with self.assertRaises(PromptAssistError):
            self.assistant.transform(self.options(force_refresh=True))

        self.assertEqual(post.call_count, 4)
        self.assertEqual(
            post.call_args_list[-1].kwargs["json"], {"instance_id": "instance-4"}
        )

    def test_provider_url_rejects_non_loopback_hosts(self):
        with self.assertRaises(PromptAssistError):
            self.assistant.transform(
                self.options(base_url="https://example.com", force_refresh=True)
            )

    def test_structure_mode_requires_a_theme(self):
        with self.assertRaises(PromptAssistError):
            self.assistant.transform(self.options(mode="structure", theme="   "))

    def test_complete_mode_requires_a_theme_or_partial_lyrics(self):
        with self.assertRaises(PromptAssistError):
            self.assistant.transform(self.options(mode="complete", theme="   ", lyrics="   "))

    def test_complete_mode_accepts_lyrics_only_with_no_theme(self):
        good = json.dumps({"prompt": "[verse]\nHold on tonight\n[chorus]\nla la", "warnings": []})
        with patch("core.extensions.minimax_music3_lyrics_assistant.requests.post") as post:
            post.side_effect = [
                self.response({"instance_id": "instance-5"}),
                self.response({"output": [{"type": "message", "content": good}]}),
                self.response({"status": "unloaded"}),
            ]
            result = self.assistant.transform(
                self.options(mode="complete", theme="", lyrics="Hold on tonight", force_refresh=True)
            )
        self.assertTrue(result["valid"])


if __name__ == "__main__":
    unittest.main()
