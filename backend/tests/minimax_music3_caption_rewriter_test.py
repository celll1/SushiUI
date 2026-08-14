import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.extensions.minimax_music3_caption_rewriter import (  # noqa: E402
    MiniMaxMusic3CaptionRewriter,
    MusicCaptionAssistOptions,
    SECTION_NAMES,
    validate_caption,
)
from core.extensions.minimax_h3_prompt_assistant import PromptAssistError  # noqa: E402


def _body(words: int) -> str:
    return " ".join(["tone"] * words)


def _structured_caption(global_words=80, vocal_words=80, arrangement_words=80) -> str:
    return (
        f"{SECTION_NAMES[0]}\n{_body(global_words)}\n\n"
        f"{SECTION_NAMES[1]}\n{_body(vocal_words)}\n\n"
        f"{SECTION_NAMES[2]}\n{_body(arrangement_words)}"
    )


class ValidateCaptionTest(unittest.TestCase):
    def test_accepts_well_formed_caption(self):
        caption = _structured_caption(90, 90, 90)
        self.assertEqual(validate_caption(caption), [])

    def test_rejects_missing_heading(self):
        caption = (
            f"{SECTION_NAMES[0]}\n{_body(90)}\n\n"
            f"{SECTION_NAMES[2]}\n{_body(90)}\n\n"
            f"Extra Heading\n{_body(90)}"
        )
        warnings = validate_caption(caption)
        self.assertTrue(any("Vocal Details" in w for w in warnings))

    def test_rejects_wrong_order(self):
        caption = (
            f"{SECTION_NAMES[1]}\n{_body(90)}\n\n"
            f"{SECTION_NAMES[0]}\n{_body(90)}\n\n"
            f"{SECTION_NAMES[2]}\n{_body(90)}"
        )
        warnings = validate_caption(caption)
        self.assertTrue(any("required order" in w for w in warnings))

    def test_rejects_word_count_too_short(self):
        caption = _structured_caption(10, 10, 10)
        warnings = validate_caption(caption)
        self.assertTrue(any("250-450" in w for w in warnings))

    def test_rejects_word_count_too_long(self):
        caption = _structured_caption(200, 200, 200)
        warnings = validate_caption(caption)
        self.assertTrue(any("250-450" in w for w in warnings))

    def test_rejects_quoted_lyric_line(self):
        lyrics = "[verse]\nI walked alone beneath a bruised and burning sky\n[chorus]\nHold on tonight"
        caption = _structured_caption(90, 90, 85) + " I walked alone beneath a bruised and burning sky"
        warnings = validate_caption(caption, lyrics=lyrics)
        self.assertTrue(any("quote a lyric line" in w for w in warnings))

    def test_allows_lyrics_context_when_not_quoted(self):
        lyrics = "[verse]\nI walked alone beneath a bruised and burning sky"
        caption = _structured_caption(90, 90, 90)
        warnings = validate_caption(caption, lyrics=lyrics)
        self.assertEqual(warnings, [])

    def test_rejects_invented_bpm(self):
        caption = _structured_caption(90, 90, 85) + " Tempo sits around 128 bpm throughout."
        warnings = validate_caption(caption, source_caption="a dreamy synth pop track")
        self.assertTrue(any("invent a BPM" in w for w in warnings))

    def test_allows_bpm_present_in_source(self):
        caption = _structured_caption(90, 90, 85) + " Tempo sits around 128 bpm throughout."
        warnings = validate_caption(caption, source_caption="a dreamy synth pop track at 128 bpm")
        self.assertEqual(warnings, [])

    def test_rejects_invented_key(self):
        caption = _structured_caption(90, 90, 85) + " The song stays in A minor for its runtime."
        warnings = validate_caption(caption, source_caption="a moody piano ballad")
        self.assertTrue(any("invent a musical key" in w for w in warnings))

    def test_allows_key_present_in_source(self):
        caption = _structured_caption(90, 90, 85) + " The song stays in A minor for its runtime."
        warnings = validate_caption(caption, source_caption="a moody piano ballad in A minor")
        self.assertEqual(warnings, [])

    def test_rejects_contradicted_exclusion(self):
        caption = _structured_caption(90, 90, 85) + " Punchy drums drive the chorus forward."
        warnings = validate_caption(caption, source_caption="an acoustic folk tune, no drums")
        self.assertTrue(any("exclusion of 'drum'" in w for w in warnings))

    def test_allows_preserved_exclusion(self):
        caption = _structured_caption(90, 90, 85) + " There are no drums anywhere in the mix."
        warnings = validate_caption(caption, source_caption="an acoustic folk tune, no drums")
        self.assertEqual(warnings, [])


class AdversarialFidelityRuleTest(unittest.TestCase):
    """One test per probe from the phase-6 audit (F1-F6): each is a case
    where the pre-fix validator either falsely rejected compliant prose or
    falsely accepted a genuine violation. Before/after behavior is recorded
    in the assertion itself -- these all assert the CORRECTED (post-fix)
    outcome; running them against the pre-fix module would fail exactly the
    ones the audit named."""

    # F1: "a minor"/"a major" as ordinary English, not a key statement.
    def test_key_rule_does_not_flag_the_article_a_as_a_note(self):
        caption = _structured_caption(90, 90, 84) + " with a minor lift in the bridge."
        self.assertEqual(validate_caption(caption), [])

    def test_key_rule_does_not_flag_a_major_as_a_note(self):
        caption = _structured_caption(90, 90, 84) + " with a major shift at the drop."
        self.assertEqual(validate_caption(caption), [])

    # F1 follow-up: the note letter "A" collides with the English article
    # not just lowercased, but capitalised too, because it is also common
    # at the START of a sentence -- and 250-450 words of prose has plenty
    # of sentence starts. "A major"/"A minor" is only a key statement when
    # a key-context cue (in / in the key of / key of / key:) precedes it;
    # B-G have no such collision and are always counted, and so is the
    # glued shorthand form ("Am") regardless of the letter.
    def test_key_rule_does_not_flag_sentence_initial_a_major(self):
        caption = _structured_caption(90, 90, 84) + " A major lift carries it."
        self.assertEqual(validate_caption(caption, source_caption="a dreamy synth pop track"), [])

    def test_key_rule_does_not_flag_sentence_initial_a_minor(self):
        caption = _structured_caption(90, 90, 84) + " A minor swell arrives late."
        self.assertEqual(validate_caption(caption, source_caption="a dreamy synth pop track"), [])

    def test_key_rule_still_catches_in_a_minor_with_context_cue(self):
        caption = _structured_caption(90, 90, 84) + " stays in A minor throughout."
        warnings = validate_caption(caption, source_caption="a dreamy synth pop track")
        self.assertTrue(any("invent a musical key" in w for w in warnings))

    def test_key_rule_still_catches_in_the_key_of_a_major(self):
        caption = _structured_caption(90, 90, 84) + " sits in the key of A major."
        warnings = validate_caption(caption, source_caption="a dreamy synth pop track")
        self.assertTrue(any("invent a musical key" in w for w in warnings))

    def test_key_rule_still_catches_am_shorthand_without_a_cue(self):
        caption = _structured_caption(90, 90, 84) + " resolves to Am at the end."
        warnings = validate_caption(caption, source_caption="a dreamy synth pop track")
        self.assertTrue(any("invent a musical key" in w for w in warnings))

    def test_key_rule_still_catches_unambiguous_letter_at_sentence_start(self):
        # E is not an English article, sentence-initial or otherwise, so it
        # keeps no context-cue requirement.
        caption = _structured_caption(90, 90, 84) + " E minor swells here."
        warnings = validate_caption(caption, source_caption="a dreamy synth pop track")
        self.assertTrue(any("invent a musical key" in w for w in warnings))

    # F2: key rule must also catch the spellings it exists to catch.
    def test_key_rule_catches_shorthand_minor(self):
        caption = _structured_caption(90, 90, 85) + " stays in Em throughout."
        warnings = validate_caption(caption, source_caption="a dreamy synth pop track")
        self.assertTrue(any("invent a musical key" in w for w in warnings))

    def test_key_rule_catches_hyphenated_minor(self):
        caption = _structured_caption(90, 90, 85) + " stays in E-minor throughout."
        warnings = validate_caption(caption, source_caption="a dreamy synth pop track")
        self.assertTrue(any("invent a musical key" in w for w in warnings))

    def test_key_rule_catches_modal_name(self):
        caption = _structured_caption(90, 90, 85) + " sits in the key of E dorian."
        warnings = validate_caption(caption, source_caption="a dreamy synth pop track")
        self.assertTrue(any("invent a musical key" in w for w in warnings))

    # F3: exclusion rule -- misses (singular/gerund), false rejects (order
    # and window), and noun extraction (last word, not first token).
    def test_exclusion_rule_catches_singular_form(self):
        caption = _structured_caption(90, 90, 84) + " a full drum kit drives it."
        warnings = validate_caption(caption, source_caption="an acoustic folk tune, no drums")
        self.assertTrue(any("exclusion of 'drum'" in w for w in warnings))

    def test_exclusion_rule_catches_gerund_form(self):
        caption = _structured_caption(90, 90, 84) + " busy drumming throughout."
        warnings = validate_caption(caption, source_caption="an acoustic folk tune, no drums")
        self.assertTrue(any("exclusion of 'drum'" in w for w in warnings))

    def test_exclusion_rule_allows_negation_after_the_noun(self):
        caption = _structured_caption(90, 90, 84) + " drums are absent from the mix."
        warnings = validate_caption(caption, source_caption="an acoustic folk tune, no drums")
        self.assertEqual(warnings, [])

    def test_exclusion_rule_allows_omits_as_negation(self):
        caption = _structured_caption(90, 90, 84) + " the mix omits drums entirely."
        warnings = validate_caption(caption, source_caption="an acoustic folk tune, no drums")
        self.assertEqual(warnings, [])

    def test_exclusion_rule_negation_window_is_sentence_scoped_not_char_capped(self):
        caption = (
            _structured_caption(90, 90, 84)
            + " no distorted electric guitars or loud drums."
        )
        warnings = validate_caption(caption, source_caption="an acoustic folk tune, no drums")
        self.assertEqual(warnings, [])

    def test_exclusion_rule_extracts_last_noun_not_first_token(self):
        # Source excludes "distortion", not "heavy" -- "heavy reverb" must
        # not be flagged just because "heavy" was the first word extracted.
        caption = _structured_caption(90, 90, 84) + " heavy reverb tails."
        warnings = validate_caption(caption, source_caption="ballad, no heavy distortion")
        self.assertEqual(warnings, [])

    def test_exclusion_rule_recognises_avoid_as_negation(self):
        caption = _structured_caption(90, 90, 84) + " avoids heavy distortion."
        warnings = validate_caption(caption, source_caption="ballad, no heavy distortion")
        self.assertEqual(warnings, [])

    # F4: BPM detection -- prose forms it must catch, and exact-value (not
    # substring) comparison against the source.
    def test_bpm_rule_catches_beats_per_minute_phrasing(self):
        caption = _structured_caption(90, 90, 84) + " a steady 128 beats per minute."
        warnings = validate_caption(caption, source_caption="a dreamy synth pop track")
        self.assertTrue(any("invent a BPM" in w for w in warnings))

    def test_bpm_rule_catches_tempo_of_phrasing(self):
        caption = _structured_caption(90, 90, 85) + " a tempo of 92."
        warnings = validate_caption(caption, source_caption="a dreamy synth pop track")
        self.assertTrue(any("invent a BPM" in w for w in warnings))

    def test_bpm_rule_catches_spelled_out_number(self):
        caption = _structured_caption(90, 90, 84) + " ninety-two beats per minute."
        warnings = validate_caption(caption, source_caption="a dreamy synth pop track")
        self.assertTrue(any("invent a BPM" in w for w in warnings))

    def test_bpm_rule_rejects_substring_legitimisation_from_a_year(self):
        caption = _structured_caption(90, 90, 85) + " a steady 92 bpm groove."
        warnings = validate_caption(caption, source_caption="a 1992 rave revival")
        self.assertTrue(any("invent a BPM" in w for w in warnings))

    def test_bpm_rule_rejects_substring_legitimisation_from_a_longer_bpm(self):
        caption = _structured_caption(90, 90, 85) + " a steady 12 bpm groove."
        warnings = validate_caption(caption, source_caption="at 128 bpm")
        self.assertTrue(any("invent a BPM" in w for w in warnings))

    def test_bpm_rule_allows_exact_value_match(self):
        caption = _structured_caption(90, 90, 84) + " a steady 128 bpm groove."
        warnings = validate_caption(caption, source_caption="at 128 bpm")
        self.assertEqual(warnings, [])

    # F5: a CRLF response must not fail every heading check.
    def test_crlf_response_still_validates_headings(self):
        caption = (
            f"{SECTION_NAMES[0]}\r\n{_body(90)}\r\n\r\n"
            f"{SECTION_NAMES[1]}\r\n{_body(90)}\r\n\r\n"
            f"{SECTION_NAMES[2]}\r\n{_body(90)}"
        )
        self.assertEqual(validate_caption(caption), [])

    # F6: lyric-quote check survives punctuation/whitespace drift.
    def test_lyric_rule_catches_a_dropped_comma(self):
        lyric = "I walked alone beneath a bruised and burning sky"
        caption = (
            _structured_caption(90, 90, 83)
            + " I walked alone beneath a bruised, and burning sky."
        )
        warnings = validate_caption(caption, lyrics=f"[verse]\n{lyric}")
        self.assertTrue(any("quote a lyric line" in w for w in warnings))

    def test_lyric_rule_catches_a_reflowed_line(self):
        lyric = "I walked alone beneath a bruised and burning sky"
        caption = (
            _structured_caption(90, 90, 83)
            + " I walked alone beneath a bruised\nand burning sky."
        )
        warnings = validate_caption(caption, lyrics=f"[verse]\n{lyric}")
        self.assertTrue(any("quote a lyric line" in w for w in warnings))


class CacheKeyTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.rewriter = MiniMaxMusic3CaptionRewriter(
            8, Path(self.temp_dir.name) / "music-prompt-cache.sqlite3"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def options(self, **changes):
        values = {
            "caption": "a dreamy synth pop track",
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
        return MusicCaptionAssistOptions(**values)

    def test_same_options_produce_same_key(self):
        key_a = self.rewriter._cache_key(self.options())
        key_b = self.rewriter._cache_key(self.options())
        self.assertEqual(key_a, key_b)

    def test_different_caption_produces_different_key(self):
        key_a = self.rewriter._cache_key(self.options())
        key_b = self.rewriter._cache_key(self.options(caption="a driving rock anthem"))
        self.assertNotEqual(key_a, key_b)

    def test_different_lyrics_produces_different_key(self):
        key_a = self.rewriter._cache_key(self.options())
        key_b = self.rewriter._cache_key(self.options(lyrics="[verse]\nhold on"))
        self.assertNotEqual(key_a, key_b)

    def test_different_constraints_produces_different_key(self):
        key_a = self.rewriter._cache_key(self.options())
        key_b = self.rewriter._cache_key(self.options(constraints="no drums"))
        self.assertNotEqual(key_a, key_b)

    def test_key_is_independent_of_h3_guide_version(self):
        # Cache identity must not accidentally collide with H3's cache: the
        # guide version string is domain-specific.
        from core.extensions.minimax_h3_prompt_assistant import GUIDE_VERSION as H3_GUIDE_VERSION
        from core.extensions.minimax_music3_caption_rewriter import GUIDE_VERSION as MUSIC_GUIDE_VERSION
        self.assertNotEqual(H3_GUIDE_VERSION, MUSIC_GUIDE_VERSION)


class PromptAssemblyTest(unittest.TestCase):
    def test_user_message_includes_caption_lyrics_and_constraints(self):
        options = MusicCaptionAssistOptions(
            caption="a dreamy synth pop track",
            lyrics="[verse]\nhold on tonight",
            constraints="no drums",
            provider="lm_studio",
            base_url="http://127.0.0.1:1234",
            model="local/test-model",
            temperature=0.2,
            top_p=0.9,
            max_output_tokens=512,
            context_length=4096,
            timeout_seconds=30,
        )
        message = MiniMaxMusic3CaptionRewriter._user_message(options)
        self.assertIn("Caption: a dreamy synth pop track", message)
        self.assertIn("Lyrics (context only, never quote):", message)
        self.assertIn("hold on tonight", message)
        self.assertIn("Additional constraints: no drums", message)

    def test_user_message_omits_empty_optional_fields(self):
        options = MusicCaptionAssistOptions(
            caption="a dreamy synth pop track",
            lyrics="",
            constraints="",
            provider="lm_studio",
            base_url="http://127.0.0.1:1234",
            model="local/test-model",
            temperature=0.2,
            top_p=0.9,
            max_output_tokens=512,
            context_length=4096,
            timeout_seconds=30,
        )
        message = MiniMaxMusic3CaptionRewriter._user_message(options)
        self.assertNotIn("Lyrics", message)
        self.assertNotIn("Additional constraints", message)


class TransformTransportTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.rewriter = MiniMaxMusic3CaptionRewriter(
            8, Path(self.temp_dir.name) / "music-prompt-cache.sqlite3"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def options(self, **changes):
        values = {
            "caption": "a dreamy synth pop track",
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
        return MusicCaptionAssistOptions(**values)

    def response(self, payload):
        response = Mock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        return response

    @patch("core.extensions.minimax_music3_caption_rewriter.requests.post")
    def test_cache_hit_skips_load_chat_and_unload(self, post):
        rewritten = _structured_caption(90, 90, 90)
        post.side_effect = [
            self.response({"instance_id": "instance-1"}),
            self.response({"output": [{"type": "message", "content": json.dumps({"prompt": rewritten, "warnings": []})}]}),
            self.response({"status": "unloaded"}),
        ]

        first = self.rewriter.transform(self.options())
        second = self.rewriter.transform(self.options())

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(post.call_count, 3)
        self.assertEqual(first["prompt"], second["prompt"])

    @patch("core.extensions.minimax_music3_caption_rewriter.requests.post")
    def test_validation_failure_drives_one_repair_retry(self, post):
        bad = json.dumps({"prompt": "not structured at all", "warnings": []})
        good = json.dumps({"prompt": _structured_caption(90, 90, 90), "warnings": []})
        post.side_effect = [
            self.response({"instance_id": "instance-2"}),
            self.response({"output": [{"type": "message", "content": bad}]}),
            self.response({"output": [{"type": "message", "content": good}]}),
            self.response({"status": "unloaded"}),
        ]

        result = self.rewriter.transform(self.options(force_refresh=True))

        self.assertEqual(post.call_count, 4)
        self.assertTrue(result["valid"])

    @patch("core.extensions.minimax_music3_caption_rewriter.requests.post")
    def test_invalid_llm_json_still_unloads_owned_instance(self, post):
        post.side_effect = [
            self.response({"instance_id": "instance-3"}),
            self.response({"output": [{"type": "message", "content": "not json"}]}),
            self.response({"output": [{"type": "message", "content": "still not json"}]}),
            self.response({"status": "unloaded"}),
        ]

        with self.assertRaises(PromptAssistError):
            self.rewriter.transform(self.options(force_refresh=True))

        self.assertEqual(post.call_count, 4)
        self.assertEqual(
            post.call_args_list[-1].kwargs["json"], {"instance_id": "instance-3"}
        )

    def test_provider_url_rejects_non_loopback_hosts(self):
        with self.assertRaises(PromptAssistError):
            self.rewriter.transform(
                self.options(base_url="https://example.com", force_refresh=True)
            )

    def test_empty_caption_is_rejected(self):
        with self.assertRaises(PromptAssistError):
            self.rewriter.transform(self.options(caption="   "))


if __name__ == "__main__":
    unittest.main()
