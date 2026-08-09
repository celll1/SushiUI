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
    PromptAssistCache,
    PromptAssistOptions,
    PromptAssistError,
    build_template,
    normalize_prompt,
    validate_prompt,
)


class MiniMaxH3PromptAssistantTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.assistant = MiniMaxH3PromptAssistant(
            8, Path(self.temp_dir.name) / "prompt-cache.sqlite3"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def options(self, **changes):
        values = {
            "prompt": "1girl, red dress, walking through rain",
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
            "force_refresh": False,
        }
        values.update(changes)
        return PromptAssistOptions(**values)

    def response(self, payload):
        response = Mock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        return response

    def test_template_is_editable_and_not_marked_complete(self):
        prompt = build_template("a baker opens a shop", "i2va", 6.0)
        self.assertTrue(prompt.startswith("For the target video"))
        self.assertIn("integrated_multimodal_description:", prompt)
        self.assertIn("[Describe only requested", prompt)

    def test_normalize_uses_official_no_music_marker(self):
        self.assertEqual(
            normalize_prompt("non_diegetic_music: None."),
            "non_diegetic_music: N/A",
        )

    def test_validator_rejects_unknown_reference_labels(self):
        prompt = (
            "subject_definitions:\n<Subject 1> is retained from <Picture 2>.\n\n"
            "summary:\n[reference generation] A portrait.\n\n"
            "retention_analysis:\n<Subject 1> fully_preserved.\n\n"
            "detailed_description:\n[Shot 1] <Subject 1> faces the camera.\n\n"
            "overall_soundscape:\nQuiet room tone.\n\n"
            "non_diegetic_music:\nN/A"
        )
        warnings = validate_prompt(
            prompt,
            "ref2va",
            5.0,
            [{"token": "<Picture 1>", "kind": "picture", "role": "reference"}],
        )
        self.assertIn("Unknown reference labels: <Picture 2>", warnings)

    @patch("core.extensions.minimax_h3_prompt_assistant.requests.post")
    def test_cache_hit_skips_load_chat_and_unload(self, post):
        rewritten = (
            "integrated_multimodal_description: [Shot 1] Live-action, a woman in a red dress walks through rain.\n\n"
            "overall_soundscape: Rain falls on the pavement and footsteps splash through puddles.\n\n"
            "non_diegetic_music: N/A"
        )
        post.side_effect = [
            self.response({"instance_id": "instance-1"}),
            self.response({"output": [{"type": "message", "content": json.dumps({"prompt": rewritten, "warnings": []})}]}),
            self.response({"status": "unloaded"}),
        ]

        first = self.assistant.transform(self.options())
        second = self.assistant.transform(self.options())

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(post.call_count, 3)
        self.assertEqual(first["prompt"], second["prompt"])

    @patch("core.extensions.minimax_h3_prompt_assistant.requests.post")
    def test_invalid_llm_json_still_unloads_owned_instance(self, post):
        post.side_effect = [
            self.response({"instance_id": "instance-2"}),
            self.response({"output": [{"type": "message", "content": "not json"}]}),
            self.response({"output": [{"type": "message", "content": "still not json"}]}),
            self.response({"status": "unloaded"}),
        ]

        with self.assertRaises(PromptAssistError):
            self.assistant.transform(self.options(force_refresh=True))

        self.assertEqual(post.call_count, 4)
        self.assertEqual(
            post.call_args_list[-1].kwargs["json"], {"instance_id": "instance-2"}
        )

    def test_provider_url_rejects_non_loopback_hosts(self):
        with self.assertRaises(PromptAssistError):
            self.assistant.list_models("lm_studio", "https://example.com")


if __name__ == "__main__":
    unittest.main()
