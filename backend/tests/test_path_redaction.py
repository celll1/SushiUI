"""
Tests for the shareable-artifact path redactor (utils/path_redaction.py) --
the code that decides what a PNG produced by SushiUI may say about this
machine's filesystem.

Run with:
    venv\\Scripts\\python.exe -m pytest backend/tests/test_path_redaction.py -v

Two failure modes matter equally and both have occurred:

  A. UNDER-redaction -- an absolute path reaches a PNG text chunk. A PNG
     travels; its drive letters, home directory and project layout are
     personal environment information about the machine, not about the image.

  B. OVER-redaction -- the redactor rewrites text that is NOT a path. This is
     worse than it sounds: the fixtures below are REAL values taken from this
     repo's gallery.db, and earlier revisions of the regex
       * deleted four feature names from a degradation notice
         ("not compatible with NAG / NegPip / DEUS / ..." -> "NAG  FBCache"),
         i.e. made the PNG assert something false about what was disabled;
       * stripped the A1111 escapes out of "azarin \\(exs-tia\\)", turning a
         literal parenthesis into an emphasis group so the recorded prompt no
         longer reproduces the image;
       * turned the danbooru tag "\\m/" into "m/" (18 distinct real prompts).

Every prose/prompt fixture here is therefore an EQUALITY assertion against the
untouched string, not a "does not contain a path" assertion.

No network, no torch, no DB required (the optional gallery.db replay skips
itself when the database is absent).
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import unittest

# ── path setup ───────────────────────────────────────────────────────────────
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.path_redaction import (  # noqa: E402
    IDENTIFIER_KEYS,
    USER_TEXT_KEYS,
    display_name_for_path,
    redact_params_for_sharing,
    redact_paths,
)

_GALLERY_DB = os.path.join(_REPO, "gallery.db")


class TestPathsAreRedacted(unittest.TestCase):
    """Failure mode A: no absolute path may survive."""

    def test_windows_absolute_in_a_label(self):
        self.assertEqual(
            redact_paths(
                "override: Z:\\sushiUI\\training\\vae_dec_IL02_v1\\vae_dec_IL02_v1_vae "
                "(run vae_dec_IL02_v1, step 141286, EMA weights, decoder only)"
            ),
            "override: vae_dec_IL02_v1_vae "
            "(run vae_dec_IL02_v1, step 141286, EMA weights, decoder only)",
        )

    def test_forward_slash_windows_path(self):
        # Real value of outpaint_controlnet_model (80 rows).
        self.assertEqual(
            redact_paths(
                "Z:/sushiUI/training/outpaint_cn_tia_arb/"
                "outpaint_cn_tia_arb_controlnet_step_087475"
            ),
            "outpaint_cn_tia_arb_controlnet_step_087475",
        )

    def test_posix_absolute(self):
        self.assertEqual(
            redact_paths("/home/bob/models/sdxl_vae.safetensors"),
            "sdxl_vae.safetensors",
        )

    def test_unc(self):
        self.assertEqual(
            redact_paths("\\\\nas\\share\\vaes\\x.safetensors"), "x.safetensors"
        )

    def test_path_quoted_inside_prose(self):
        self.assertEqual(
            redact_paths("VAE override failed to load from Z:\\a\\b\\my_vae; using the model VAE."),
            "VAE override failed to load from my_vae; using the model VAE.",
        )

    def test_a_future_key_holding_a_personal_path(self):
        self.assertEqual(
            redact_paths("C:\\Users\\someone\\secret_project\\weights\\model.safetensors"),
            "weights/model.safetensors",
        )

    def test_relative_windows_path_collapses_to_its_name(self):
        # Real shape of loras[].path.
        self.assertEqual(
            redact_paths("20251209_221416_67bb82b9\\67bb82b9_step_0_interrupted.safetensors"),
            "67bb82b9_step_0_interrupted.safetensors",
        )

    def test_no_separator_survives_any_of_them(self):
        for value in (
            "Z:\\model\\sdxl\\VAE\\sdxl_vae_pid.safetensors",
            "/mnt/data/models/vae/config",
            "\\\\host\\share\\a\\b\\c.safetensors",
            "D:/x/y/z.pth",
        ):
            out = redact_paths(value)
            self.assertNotIn("\\", out, value)
            self.assertNotRegex(out, r"[A-Za-z]:", value)


class TestGenericNamesAreDisambiguated(unittest.TestCase):
    """A name the layout generated must resolve to one file locally."""

    def test_generic_component_dir_gets_its_model_folder(self):
        self.assertEqual(display_name_for_path("Z:\\model\\krea2\\vae"), "krea2/vae")

    def test_generic_weight_file_walks_up_twice(self):
        self.assertEqual(
            display_name_for_path("Z:\\model\\krea2\\vae\\diffusion_pytorch_model.safetensors"),
            "krea2/vae/diffusion_pytorch_model.safetensors",
        )

    def test_sharded_weight_file_is_generic_too(self):
        self.assertEqual(
            display_name_for_path(
                "Z:\\model\\lens\\microsoft-lens\\transformer\\"
                "diffusion_pytorch_model-00001-of-00002.safetensors"
            ),
            "microsoft-lens/transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        )

    def test_human_chosen_name_stays_bare(self):
        self.assertEqual(
            display_name_for_path(
                "Z:\\sushiUI\\training\\vae_dec_IL02_v1\\vae_dec_IL02_v1_vae"
            ),
            "vae_dec_IL02_v1_vae",
        )

    def test_segment_cap(self):
        deep = "Z:\\a\\b\\c\\vae\\vae\\vae\\model.safetensors"
        self.assertLessEqual(len(display_name_for_path(deep).split("/")), 3)

    def test_never_empty(self):
        for value in ("", "   ", "Z:\\", "/", None):
            self.assertTrue(display_name_for_path(value))


class TestNeverEmitsAnAccountName(unittest.TestCase):
    """The module promises the user directory is never emitted."""

    def test_windows_user_dir_is_not_prepended(self):
        self.assertEqual(
            display_name_for_path("C:\\Users\\someone\\text_encoder"), "text_encoder"
        )

    def test_posix_home_dir_is_not_prepended(self):
        self.assertEqual(
            display_name_for_path("/home/someone/vae/model.safetensors"),
            "vae/model.safetensors",
        )

    def test_a_normal_folder_under_the_user_dir_is_fine(self):
        self.assertEqual(
            display_name_for_path("C:\\Users\\someone\\models\\vae"), "models/vae"
        )

    def test_the_account_dir_itself_has_no_name(self):
        self.assertEqual(display_name_for_path("C:\\Users\\someone"), "unnamed file")

    def test_through_redact_paths(self):
        self.assertEqual(
            redact_paths("override: C:\\Users\\someone\\text_encoder"),
            "override: text_encoder",
        )
        self.assertEqual(
            redact_paths("override: /home/someone/vae/model.safetensors"),
            "override: vae/model.safetensors",
        )


class TestProseIsNotRewritten(unittest.TestCase):
    """Failure mode B, backend-authored text. Every string here is a REAL
    warning message from gallery.db that an earlier revision corrupted."""

    REAL_WARNINGS = [
        "Regional additional prompt disabled: not compatible with NAG / NegPip / DEUS / "
        "style transfer / Spectrum / FBCache in this version.",
        "Style transfer disabled: not compatible with NAG / ControlNet / Spectrum in this version.",
        "Paste-band reconciliation feather: the last 24 row(s)/column(s) of the preserved rect "
        "at its generate-adjacent edges are blended (raised-cosine) toward the decoded canvas "
        "underneath instead of pasted byte-exact; the rest of the preserved rect is unaffected.",
        "Paste-band reconciliation feather: the last 3 row(s)/column(s) of the preserved rect "
        "at its generate-adjacent edges are blended (raised-cosine) toward the decoded canvas "
        "underneath instead of pasted byte-exact; the rest of the preserved rect is unaffected.",
        "Paste-band reconciliation feather: the last 48 row(s)/column(s) of the preserved rect "
        "at its generate-adjacent edges are blended (raised-cosine) toward the decoded canvas "
        "underneath instead of pasted byte-exact; the rest of the preserved rect is unaffected.",
        "Paste-band reconciliation feather: the last 6 row(s)/column(s) of the preserved rect "
        "at its generate-adjacent edges are blended (raised-cosine) toward the decoded canvas "
        "underneath instead of pasted byte-exact; the rest of the preserved rect is unaffected.",
    ]

    NON_PATHS = [
        "24/7 and and/or and a/b",
        "stabilityai/sdxl-vae",     # HF repo id: a legitimate vae_name value
        "https://example.com/a/b",
        "krea2/vae",                # an already-redacted display name
        "embedded (checkpoint)",
        "none (pixel-space)",
        "RealESRGAN_x4plus.pth",
    ]

    def test_real_warning_messages_are_byte_identical(self):
        for msg in self.REAL_WARNINGS:
            self.assertEqual(redact_paths(msg), msg, msg[:60])

    def test_non_path_strings_are_byte_identical(self):
        for value in self.NON_PATHS:
            self.assertEqual(redact_paths(value), value, value)


class TestUserTextIsNotRewritten(unittest.TestCase):
    """Failure mode B, user-authored text. These must survive BOTH by key
    exemption and (defence in depth) by the regexes themselves."""

    PROMPT_SYNTAX = [
        "1girl, azarin \\(exs-tia\\), (bad:-1), miwano rag",        # real
        "candle, \\m/, indoors",                                     # real, 18 rows
        "3d, wrong hands, color pencil \\(medium\\\\)",              # real, doubled escape
        "ui, hud, frame, border, text, watermark",
    ]

    def test_exempt_keys_pass_through(self):
        for key in ("prompt", "negative_prompt", "region_prompt",
                    "region_negative_prompt", "lyrics", "caption"):
            self.assertIn(key, USER_TEXT_KEYS)
            for value in self.PROMPT_SYNTAX:
                self.assertEqual(redact_params_for_sharing(value, key), value, key)

    def test_prompt_syntax_survives_even_without_the_key_exemption(self):
        for value in self.PROMPT_SYNTAX:
            self.assertEqual(redact_paths(value), value, value)

    def test_identifier_keys_pass_through_but_messages_do_not(self):
        self.assertIn("code", IDENTIFIER_KEYS)
        warning = {"code": "vae_override_error",
                   "message": "failed from Z:\\a\\b\\my_vae"}
        self.assertEqual(
            redact_params_for_sharing(warning),
            {"code": "vae_override_error", "message": "failed from my_vae"},
        )

    def test_caller_object_is_not_mutated(self):
        params = {"vae_override_path": "Z:\\a\\b\\v",
                  "loras": [{"path": "Z:\\lora\\x.safetensors"}]}
        snapshot = json.dumps(params, sort_keys=True)
        redact_params_for_sharing(params)
        self.assertEqual(json.dumps(params, sort_keys=True), snapshot)


@unittest.skipUnless(os.path.isfile(_GALLERY_DB), "gallery.db not present")
class TestAgainstEveryRealValue(unittest.TestCase):
    """Replays every distinct real value in the local gallery: no user text or
    warning message may change, and no path-shaped value may survive."""

    @classmethod
    def setUpClass(cls):
        cls.messages, cls.user_text, cls.paths = set(), set(), set()
        con = sqlite3.connect("file:%s?mode=ro" % _GALLERY_DB, uri=True)
        try:
            for (raw,) in con.execute("SELECT parameters FROM generated_images"):
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except Exception:
                    continue
                for w in (row.get("effective_warnings") or []):
                    if isinstance(w, dict) and isinstance(w.get("message"), str):
                        cls.messages.add(w["message"])
                for key in ("prompt", "negative_prompt", "region_prompt",
                            "region_negative_prompt"):
                    if isinstance(row.get(key), str) and row[key]:
                        cls.user_text.add(row[key])
                for key in ("vae_path", "vae_override_path", "vae_override_source",
                            "text_encoder_path", "outpaint_controlnet_model"):
                    if isinstance(row.get(key), str) and row[key]:
                        cls.paths.add(row[key])
        finally:
            con.close()

    def test_no_warning_message_changes(self):
        altered = [m for m in self.messages if redact_paths(m) != m]
        self.assertEqual(altered, [], "redactor rewrote real warning text")

    def test_no_user_text_changes(self):
        altered = [v for v in self.user_text if redact_paths(v) != v]
        self.assertEqual(altered[:3], [], "redactor rewrote real prompt text")

    def test_every_recorded_path_is_reduced_to_a_name(self):
        for p in self.paths:
            out = redact_paths(p)
            self.assertNotIn("\\", out, p)
            self.assertNotRegex(out, r"[A-Za-z]:", p)
            self.assertTrue(out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
