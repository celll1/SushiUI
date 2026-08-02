"""Executable form of the VAE-training component toggle matrix (design.md §4).

Run from the repository root with the repo's virtualenv interpreter
(``venv/Scripts/python.exe`` on Windows, ``venv/bin/python`` on POSIX):

    venv/Scripts/python.exe -m pytest backend/tests/test_vae_refusal_matrix.py -v

or, without pytest:

    venv/Scripts/python.exe -m unittest discover -s backend/tests -p "test_vae_refusal_matrix.py"

Every row of the matrix is asserted here: the accepted combinations resolve, and
each refused combination raises ``VaeConfigError`` with a message that names the
key at fault. Phase 1 verified the same ground by an ad-hoc script in a temp
directory; this file exists so a later change cannot quietly reopen one of the
holes. If a refusal is deliberately removed, the corresponding case here must be
moved from ``_assert_refused`` to ``_assert_accepted`` in the same commit, which
makes the decision visible in review.

Scope: ``vae_config.resolve_vae_training_config`` (the gate), plus the trainer's
own second gate on encoder training and on the bare-LDM export. No model is
loaded and no training step runs, so the file is fast and GPU-free; the actual
encoder-training behaviour (parameters move, KL finite, sidecar flag) is covered
by the smoke run documented in docs/guides/VAE_TRAINING.md.

Matrix rows that are NOT implemented at all (the PiD ``pid_decoder`` network
type, the GAN loss, ``crop_consistency``) are covered by
``test_unimplemented_matrix_rows_have_no_config_surface``: they must remain
unreachable rather than becoming silently-accepted unknown keys. The
invented-HF row IS implemented, but under the ``l_invented_*`` names (see
``test_l_invented_loss.py``), so the design.md spelling ``invented_hf_weight``
stays in that absent-key list as a typo guard.
"""

from __future__ import annotations

import ast
import copy
import os
import sys
import unittest

# ── path setup ───────────────────────────────────────────────────────────────
# `backend` itself must be on sys.path: the modules under test import
# `api.param_defaults` / `core.training.*` with backend as the root package dir.
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from api.param_defaults import VAE_TRAINING_DEFAULTS
from core.training.vae.vae_config import (
    VaeConfigError,
    VALID_CROP_SCALE_POLICIES,
    VALID_DECODER_BLOCKS,
    VALID_ENCODER_BLOCKS,
    VALID_DTYPES,
    VALID_LPIPS_NETS,
    VALID_LR_SCHEDULERS,
    VALID_OPTIMIZERS,
    VALID_SOURCES,
    resolve_vae_training_config,
    strict_bool,
)

# A minimal, valid process config: decoder-only, base VAE from the run's model.
# Repo-relative on purpose -- nothing here opens the file, and a checked-in test
# must not carry a machine-specific path.
_BASE_MODEL = os.path.join(_BACKEND, "..", "models", "vae", "placeholder.safetensors")

# Every case starts from lpips_weight 0. The default is 0.1, and `_validate`
# imports `lpips` whenever it is above 0 -- so with the default in place, an
# environment WITHOUT lpips installed makes this whole file report the lpips
# refusal instead of the rule each case is about (verified: it turns several
# refusal rows green for the wrong reason). The lpips gate has its own dedicated
# case below, guarded by an availability check.
_NO_LPIPS = {"lpips_weight": 0.0}


def _lpips_available() -> bool:
    try:
        import lpips  # noqa: F401
        return True
    except Exception:
        return False


def _process(vae: dict = None, train: dict = None, save: dict = None) -> dict:
    return {
        "network": {"type": "vae_decoder"},
        "model": {"name_or_path": _BASE_MODEL},
        "train": dict(train or {}),
        "save": dict(save or {}),
        "vae": {**_NO_LPIPS, **dict(vae or {})},
    }


class VaeRefusalMatrixTest(unittest.TestCase):
    """design.md §4, row by row."""

    # ── helpers ──────────────────────────────────────────────────────────
    def _resolve(self, **kwargs):
        return resolve_vae_training_config(_process(**kwargs),
                                           base_model_path=_BASE_MODEL)

    def _assert_accepted(self, **kwargs):
        cfg = self._resolve(**kwargs)
        self.assertIsInstance(cfg, dict)
        # The resolver's contract: exactly the SSOT key set, plus resume_from.
        self.assertEqual(set(cfg), set(VAE_TRAINING_DEFAULTS) | {"resume_from"})
        return cfg

    def _assert_refused(self, expect_in_message, **kwargs):
        with self.assertRaises(VaeConfigError) as ctx:
            self._resolve(**kwargs)
        message = str(ctx.exception)
        for fragment in ([expect_in_message] if isinstance(expect_in_message, str)
                         else expect_in_message):
            self.assertIn(fragment, message,
                          f"refusal message did not mention {fragment!r}: {message}")
        return message

    # ── row 1: decoder trained, encoder frozen (the default) ─────────────
    def test_row_decoder_only_is_the_default_and_is_accepted(self):
        cfg = self._assert_accepted()
        self.assertTrue(cfg["train_decoder"])
        self.assertFalse(cfg["train_encoder"])
        self.assertFalse(cfg["acknowledge_latent_space_break"])
        self.assertEqual(cfg["decoder_blocks"], "all")
        # The base VAE defaults to the run's own model.
        self.assertEqual(cfg["vae_path"], _BASE_MODEL)

    # ── row 2: decoder block granularity ─────────────────────────────────
    def test_row_every_decoder_block_granularity_is_accepted(self):
        for blocks in VALID_DECODER_BLOCKS:
            with self.subTest(decoder_blocks=blocks):
                cfg = self._assert_accepted(vae={"decoder_blocks": blocks})
                self.assertEqual(cfg["decoder_blocks"], blocks)

    def test_out_of_enum_decoder_blocks_is_refused(self):
        self._assert_refused("decoder_blocks", vae={"decoder_blocks": "up_block"})

    # ── row 3: decoder + encoder, DOUBLE GATE satisfied ──────────────────
    def test_row_encoder_training_with_both_gate_keys_is_accepted(self):
        cfg = self._assert_accepted(vae={
            "train_encoder": True,
            "acknowledge_latent_space_break": True,
        })
        self.assertTrue(cfg["train_encoder"])
        self.assertTrue(cfg["acknowledge_latent_space_break"])
        self.assertTrue(cfg["train_decoder"])

    def test_row_every_encoder_block_granularity_is_accepted(self):
        for blocks in VALID_ENCODER_BLOCKS:
            with self.subTest(encoder_blocks=blocks):
                cfg = self._assert_accepted(vae={
                    "train_encoder": True,
                    "acknowledge_latent_space_break": True,
                    "encoder_blocks": blocks,
                })
                self.assertEqual(cfg["encoder_blocks"], blocks)

    def test_out_of_enum_encoder_blocks_is_refused(self):
        self._assert_refused("encoder_blocks", vae={
            "train_encoder": True,
            "acknowledge_latent_space_break": True,
            "encoder_blocks": "up_blocks",  # a decoder value, not an encoder one
        })

    # ── row 3, gate halves ───────────────────────────────────────────────
    # NOTE on the expected fragments in this class: they must be DISTINCTIVE to
    # the guard under test, not merely present in its message. A mutation sweep
    # found that asserting the key names alone let the encoder-only row below
    # pass against the "Nothing to train: train_decoder=false and
    # train_encoder=false" message of a different guard, so deleting the guard
    # went undetected. Each row now names a phrase only its own guard produces.
    def test_encoder_training_without_acknowledgement_is_refused(self):
        self._assert_refused("requires acknowledge_latent_space_break=true",
                             vae={"train_encoder": True})

    def test_acknowledgement_without_encoder_training_is_refused(self):
        self._assert_refused("only applies to encoder training",
                             vae={"acknowledge_latent_space_break": True})

    # ── row 4: encoder trained, decoder frozen ───────────────────────────
    def test_encoder_only_with_frozen_decoder_is_refused(self):
        self._assert_refused("deform the latent distribution", vae={
            "train_decoder": False,
            "train_encoder": True,
            "acknowledge_latent_space_break": True,
        })

    # ── row 5: nothing trainable ─────────────────────────────────────────
    def test_nothing_trainable_is_refused(self):
        self._assert_refused("Nothing to train",
                             vae={"train_decoder": False, "train_encoder": False})

    # ── row: pre-encoded latent cache ────────────────────────────────────
    def test_pre_encoded_cache_is_refused(self):
        self._assert_refused("pre_encoded_cache",
                             train={"latent_encoding_mode": "pre_encoded_cache"})

    def test_other_latent_encoding_modes_are_ignored(self):
        self._assert_accepted(train={"latent_encoding_mode": "swap_onthefly"})

    # ── row: fp16 ────────────────────────────────────────────────────────
    def test_fp16_is_refused(self):
        # "fp16" alone would also be satisfied by the out-of-enum dtype message
        # that follows this guard, so assert a phrase only the fp16 guard emits.
        # (A mutation sweep caught exactly this collision.)
        self._assert_refused("no gradient scaler", vae={"dtype": "fp16"})

    def test_allowed_dtypes_are_accepted(self):
        for dtype in VALID_DTYPES:
            with self.subTest(dtype=dtype):
                self._assert_accepted(vae={"dtype": dtype})

    def test_out_of_enum_dtype_is_refused(self):
        self._assert_refused("dtype", vae={"dtype": "bfloat16"})

    # ── row: bare-LDM export vs encoder training ─────────────────────────
    def test_bare_ldm_export_is_accepted_with_a_frozen_encoder(self):
        cfg = self._assert_accepted(vae={"export_bare_ldm": True})
        self.assertTrue(cfg["export_bare_ldm"])

    def test_bare_ldm_export_with_encoder_training_is_refused(self):
        self._assert_refused("export_bare_ldm=true is refused", vae={
            "train_encoder": True,
            "acknowledge_latent_space_break": True,
            "export_bare_ldm": True,
        })

    # ── quoted booleans: the gate must not be openable by YAML quoting ───
    def test_quoted_boolean_strings_are_parsed_by_value_not_truthiness(self):
        """``train_encoder: "false"`` must NOT enable encoder training.

        Python reads any non-empty string as True, and quoted booleans are what
        YAML editors, templating and hand-quoting routinely produce, so a bare
        ``bool()`` cast here would silently open the double gate.
        """
        for text in ("false", "False", "FALSE", "no", "off", "0", " false "):
            with self.subTest(train_encoder=text):
                cfg = self._assert_accepted(vae={
                    "train_encoder": text,
                    "acknowledge_latent_space_break": text,
                })
                self.assertIs(cfg["train_encoder"], False)
                self.assertIs(cfg["acknowledge_latent_space_break"], False)

    def test_quoted_true_still_opens_the_gate_when_both_keys_say_so(self):
        cfg = self._assert_accepted(vae={
            "train_encoder": "true",
            "acknowledge_latent_space_break": "yes",
        })
        self.assertIs(cfg["train_encoder"], True)
        self.assertIs(cfg["acknowledge_latent_space_break"], True)

    def test_quoted_true_alone_is_still_a_single_gate_refusal(self):
        self._assert_refused("requires acknowledge_latent_space_break=true",
                             vae={"train_encoder": "true"})

    def test_quoted_false_export_bare_ldm_does_not_write_the_file(self):
        cfg = self._assert_accepted(vae={"export_bare_ldm": "false"})
        self.assertIs(cfg["export_bare_ldm"], False)

    def test_uninterpretable_booleans_are_refused_rather_than_guessed(self):
        for key in ("train_decoder", "train_encoder",
                    "acknowledge_latent_space_break", "export_bare_ldm",
                    "ema_enabled"):
            for bad in ("maybe", 2, 0.5, [], "y e s"):
                with self.subTest(key=key, value=bad):
                    self._assert_refused([key, "must be a boolean"],
                                         vae={key: bad})

    def test_strict_bool_accepts_the_documented_spellings(self):
        for value in (True, 1, "true", "TRUE", "yes", "on", "1"):
            self.assertIs(strict_bool(value, "k"), True)
        for value in (False, 0, "false", "no", "off", "0"):
            self.assertIs(strict_bool(value, "k"), False)
        with self.assertRaises(VaeConfigError):
            strict_bool("truthy", "k")

    # ── losses ───────────────────────────────────────────────────────────
    def test_all_loss_weights_zero_is_refused(self):
        self._assert_refused("no training signal", vae={
            "mse_weight": 0, "l1_weight": 0, "lpips_weight": 0,
            "ycbcr_dc_weight": 0, "pattern_weight": 0,
        })

    def test_kl_weight_alone_is_not_a_training_signal(self):
        # KL regularises the posterior; it is not a reconstruction term, so it
        # deliberately does not satisfy the "at least one active loss" check.
        self._assert_refused("no training signal", vae={
            "mse_weight": 0, "l1_weight": 0, "lpips_weight": 0,
            "ycbcr_dc_weight": 0, "pattern_weight": 0, "kl_weight": 1.0,
            "train_encoder": True, "acknowledge_latent_space_break": True,
        })

    def test_negative_and_non_numeric_loss_weights_are_refused(self):
        self._assert_refused("mse_weight", vae={"mse_weight": -1.0})
        self._assert_refused("mse_weight", vae={"mse_weight": "heavy"})
        self._assert_refused("kl_weight", vae={"kl_weight": -1e-6})
        self._assert_refused("kl_weight", vae={"kl_weight": "small"})

    def test_lpips_net_enum(self):
        for net in VALID_LPIPS_NETS:
            with self.subTest(lpips_net=net):
                self._assert_accepted(vae={"lpips_net": net, "lpips_weight": 0.0})
        self._assert_refused("lpips_net", vae={"lpips_net": "resnet"})

    def test_lpips_weight_above_zero_requires_the_lpips_package(self):
        """The only case that deliberately turns LPIPS back on.

        Every other case runs at lpips_weight 0 so that this import cannot
        decide their outcome. Both environments are asserted: where `lpips` is
        installed the config resolves, and where it is not the refusal fires
        BEFORE the run rather than mid-training.
        """
        if _lpips_available():
            cfg = self._resolve(vae={"lpips_weight": 0.1})
            self.assertEqual(cfg["lpips_weight"], 0.1)
        else:
            self._assert_refused(["lpips_weight", "not importable"],
                                 vae={"lpips_weight": 0.1})

    # ── base VAE selection ───────────────────────────────────────────────
    def test_vae_source_enum(self):
        self.assertEqual(set(VALID_SOURCES), {"model", "path", "store"})
        # Assert the enum guard's OWN wording, not just "vae_source". With the
        # bare key name, deleting this guard still passed: an unknown source
        # falls through to the vae_path branch, whose message also contains
        # "vae_source=..." (mutation-tested 2026-07-29). Same failure shape as
        # the row-4 collision -- a refusal row must pin the rule that refused.
        self._assert_refused("must be one of", vae={"vae_source": "checkpoint"})

    def test_non_mapping_vae_section_is_refused(self):
        """``process.vae`` must be a mapping.

        Nothing else validates the section's TYPE, so without this the keys are
        silently ignored and the run trains with defaults it was never given.
        """
        for bad in ([], "vae_source: model", 42):
            with self.subTest(vae_section=type(bad).__name__):
                with self.assertRaises(VaeConfigError) as ctx:
                    resolve_vae_training_config(
                        {"vae": bad, "train": {"steps": 10}, "save": {}},
                        base_model_path=_BASE_MODEL,
                    )
                self.assertIn("must be a mapping", str(ctx.exception))

    def test_explicit_path_without_a_path_is_refused(self):
        self._assert_refused("vae_path", vae={"vae_source": "path", "vae_path": ""})

    def test_store_without_an_arch_is_refused(self):
        self._assert_refused("vae_arch", vae={"vae_source": "store", "vae_arch": ""})

    def test_store_with_an_arch_is_accepted(self):
        cfg = self._assert_accepted(vae={"vae_source": "store", "vae_arch": "sdxl"})
        self.assertEqual(cfg["vae_arch"], "sdxl")

    # ── shapes / cadence ─────────────────────────────────────────────────
    def test_resolution_must_be_a_multiple_of_8_and_at_least_64(self):
        self._assert_refused("resolution", vae={"resolution": 513})
        self._assert_refused("resolution", vae={"resolution": 32})
        self._assert_refused("validation_resolution", vae={"validation_resolution": 100})

    # ── crop scale policy ────────────────────────────────────────────────
    # WHICH pixels the decoder trains on. Every fragment asserted here is
    # phrasing only the guard under test emits: "must be one of" alone would
    # also be satisfied by the vae_source / decoder_blocks / dtype / lpips_net
    # enum messages, and "must be a number" / "must be >= 0" by the loss-weight
    # ones. Each row below was mutation-tested (guard deleted -> row fails).
    def test_every_crop_scale_policy_is_accepted(self):
        for policy in VALID_CROP_SCALE_POLICIES:
            with self.subTest(crop_scale_policy=policy):
                cfg = self._assert_accepted(vae={"crop_scale_policy": policy})
                self.assertEqual(cfg["crop_scale_policy"], policy)

    def test_default_crop_scale_policy_is_the_historical_downscale(self):
        """The shipped default must not change what an existing run trains on."""
        cfg = self._assert_accepted()
        self.assertEqual(cfg["crop_scale_policy"], "downscale")
        self.assertEqual(cfg["crop_scale_max_downscale"], 0.0)

    def test_out_of_enum_crop_scale_policy_is_refused(self):
        # Without this guard an unknown policy resolves cleanly (the bound check
        # below only fires for a non-zero bound), and the loader would then be
        # the first thing to notice -- after the model load, mid-startup.
        self._assert_refused("'native' crops out of the full-size pixels",
                             vae={"crop_scale_policy": "fullsize"})

    def test_max_downscale_is_refused_under_a_non_mixed_policy(self):
        """A bound only the per-sample draw reads must not be silently ignored."""
        for policy in ("downscale", "native"):
            with self.subTest(crop_scale_policy=policy):
                self._assert_refused("is only read when", vae={
                    "crop_scale_policy": policy,
                    "crop_scale_max_downscale": 2.0,
                })

    def test_max_downscale_below_one_is_refused_rather_than_clamped(self):
        # Isolated to the 'mixed' policy so that deleting THIS guard cannot be
        # covered by the non-mixed refusal above.
        self._assert_refused("is a downscale factor", vae={
            "crop_scale_policy": "mixed",
            "crop_scale_max_downscale": 0.5,
        })

    def test_negative_max_downscale_is_refused(self):
        self._assert_refused("0 = unbounded", vae={
            "crop_scale_policy": "mixed",
            "crop_scale_max_downscale": -1.0,
        })

    def test_non_numeric_max_downscale_is_refused(self):
        self._assert_refused("crop_scale_max_downscale must be a number", vae={
            "crop_scale_policy": "mixed",
            "crop_scale_max_downscale": "native-ish",
        })

    def test_mixed_with_a_bound_is_accepted_and_coerced_to_float(self):
        cfg = self._assert_accepted(vae={
            "crop_scale_policy": "mixed",
            "crop_scale_max_downscale": 2,
        })
        self.assertIsInstance(cfg["crop_scale_max_downscale"], float)
        self.assertEqual(cfg["crop_scale_max_downscale"], 2.0)

    def test_validation_resolution_default_is_1024(self):
        """1024, not 512: see docs/guides/VAE_TRAINING.md. Pinned here because a
        silent revert would move the held-out metric's regime without moving the
        chart's label."""
        self.assertEqual(VAE_TRAINING_DEFAULTS["validation_resolution"], 1024)
        self.assertEqual(self._assert_accepted()["validation_resolution"], 1024)

    def test_counts_must_be_at_least_one(self):
        for key in ("batch_size", "total_steps", "gradient_accumulation_steps"):
            with self.subTest(key=key):
                self._assert_refused("must be >= 1", vae={key: 0})

    def test_ema_decay_must_be_strictly_inside_zero_and_one(self):
        for bad in (0.0, 1.0, -0.5, 1.5):
            with self.subTest(ema_decay=bad):
                self._assert_refused("ema_decay", vae={"ema_decay": bad})
        self._assert_accepted(vae={"ema_decay": 0.9999})

    # ── typo protection ──────────────────────────────────────────────────
    def test_unknown_vae_key_is_refused(self):
        self._assert_refused("Unknown key", vae={"train_encdoer": True})

    def test_unimplemented_matrix_rows_have_no_config_surface(self):
        """design.md §4 rows that are NOT built must stay unreachable.

        PiD decoder training (Phase 3), the GAN loss and the crop-consistency
        term have no keys, so asking for them lands in the unknown-key refusal
        above rather than being silently ignored. ``invented_hf_weight`` is
        design.md's spelling of a term that shipped as ``l_invented_weight``,
        and is kept here so the old name still refuses instead of being quietly
        dropped by a config that used it.
        """
        for absent in ("gan_enabled", "disc_start", "crop_consistency_weight",
                       "invented_hf_weight", "train_lq_proj", "pid_backbone"):
            with self.subTest(key=absent):
                self.assertNotIn(absent, VAE_TRAINING_DEFAULTS)
                self._assert_refused("Unknown key", vae={absent: True})

    # ── precedence, not a refusal but part of the same contract ──────────
    def test_process_vae_overrides_the_shared_train_section(self):
        cfg = self._resolve(train={"lr": 1e-4, "batch_size": 8},
                            vae={"learning_rate": 5e-6})
        self.assertEqual(cfg["learning_rate"], 5e-6)   # process.vae wins
        self.assertEqual(cfg["batch_size"], 8)         # train section still read

    def test_resume_from_checkpoint_is_read_from_the_train_section(self):
        cfg = self._resolve(train={"resume_from_checkpoint": "latest"})
        self.assertEqual(cfg["resume_from"], "latest")


# ── the numeric gate: one verdict per VAE_TRAINING_DEFAULTS key ──────────────
# Written down so that ADDING a key without deciding what its meaningless values
# are fails a test rather than shipping an unchecked knob. The verdict text is
# the human-readable half; ``VaeNumericRefusalTest`` below asserts the rules.
#
# "enum"      - a closed value set is enforced.
# "bool"      - parsed by strict_bool, anything else refused.
# "bounded"   - numeric with an enforced range.
# "free"      - deliberately unbounded above (see the note), bounded below.
# "text"      - must be a string; emptiness rules live with vae_source.
_KEY_VERDICTS = {
    "batch_size": "bounded: >= 1",
    "total_steps": "bounded: >= 1, and > lr_warmup_steps",
    "gradient_accumulation_steps": "bounded: >= 1",
    "learning_rate": "bounded: > 0 (0 is a no-op run that reports success)",
    "optimizer": "enum: VALID_OPTIMIZERS (= everything OptimizerFactory resolves)",
    "optimizer_weight_decay": "bounded: >= 0 (negative grows the weights)",
    "max_grad_norm": "bounded: >= 0, where 0 MEANS 'no clipping' (repo-wide convention)",
    "lr_scheduler": "enum: VALID_LR_SCHEDULERS, and 'constant' forbids a warmup",
    "lr_warmup_steps": "bounded: >= 0, < total_steps, and 0 under lr_scheduler 'constant'",
    "seed": "bounded: 0 .. 2**32-1 (numpy.random.seed's domain)",
    "num_workers": "bounded: >= 0",
    "save_every": "bounded: >= 0 (0 = only the final checkpoint)",
    "max_step_saves_to_keep": "bounded: >= 0 (0 = keep all)",
    "vae_source": "enum: VALID_SOURCES",
    "vae_path": "text: required for vae_source 'path'/'model'",
    "vae_arch": "text: required for vae_source 'store'; the loader checks the key itself",
    "train_decoder": "bool",
    "decoder_blocks": "enum: VALID_DECODER_BLOCKS",
    "train_encoder": "bool (double gate)",
    "acknowledge_latent_space_break": "bool (double gate)",
    "encoder_blocks": "enum: VALID_ENCODER_BLOCKS",
    "resolution": "bounded: multiple of 8, >= 64",
    "crop_scale_policy": "enum: VALID_CROP_SCALE_POLICIES",
    "crop_scale_max_downscale": "bounded: 0 or >= 1, and only under 'mixed'",
    "dtype": "enum: VALID_DTYPES (fp16 refused with its own reason)",
    "ema_enabled": "bool",
    "ema_decay": "bounded: strictly inside (0, 1)",
    "mse_weight": "free: >= 0, unbounded above (a large weight is loud, not silent)",
    "l1_weight": "free: >= 0",
    "lpips_weight": "free: >= 0, plus the lpips-import gate above 0",
    "lpips_net": "enum: VALID_LPIPS_NETS",
    "ycbcr_dc_weight": "free: >= 0",
    "ycbcr_dc_y_weight": "bounded: >= 0, not both-zero with chroma while the term is on",
    "ycbcr_dc_chroma_weight": "bounded: >= 0, not both-zero with luma while the term is on",
    "ycbcr_dc_eps": "bounded: > 0 (0 makes the Charbonnier gradient NaN at d=0)",
    "pattern_weight": "free: >= 0",
    "pattern_size": "bounded: >= 1, and <= resolution while the term is on",
    "l_invented_weight": "bounded: 0 .. 10 (openapi's declared maximum)",
    "l_invented_y_weight": "bounded: 0 .. 4, not both-zero with chroma while the term is on",
    "l_invented_chroma_weight": "bounded: 0 .. 4, not both-zero with luma while the term is on",
    "l_invented_flat_t_y": "bounded: 0 .. 8, and > 0 while the term is on",
    "l_invented_flat_t_c": "bounded: 0 .. 8, and > 0 while the term is on",
    "kl_weight": "free: >= 0 (ignored, and logged as ignored, under a frozen encoder)",
    "export_bare_ldm": "bool (refused with train_encoder)",
    "validation_every": "bounded: >= 0 (0 = validation off)",
    "validation_num_images": "bounded: >= 1 (0 empties the TRAINING split)",
    "validation_resolution": "bounded: multiple of 8, >= 64",
}


class VaeNumericRefusalTest(unittest.TestCase):
    """Numeric validation: every value that would train something OTHER than
    what the config says, without saying so.

    The holes closed here were all of the same shape — a key that was cast to
    ``int``/``float`` and never range-checked, so an out-of-domain value
    resolved cleanly and degraded the run in silence (``learning_rate: 0`` =
    a successful run that exports the base VAE; ``validation_num_images: -1`` =
    training on one image; ``max_grad_norm: 0`` = every gradient zeroed while
    weight decay keeps shrinking the weights).

    Each row asserts a fragment only its own guard emits, for the reason
    documented in ``VaeRefusalMatrixTest``: a bare key name is satisfied by
    neighbouring messages, so deleting the guard would go undetected.
    """

    _resolve = VaeRefusalMatrixTest._resolve
    _assert_accepted = VaeRefusalMatrixTest._assert_accepted
    _assert_refused = VaeRefusalMatrixTest._assert_refused

    # ── exhaustiveness ───────────────────────────────────────────────────
    def test_every_default_key_has_a_recorded_verdict(self):
        """A new key must come with a decision about its meaningless values.

        Without this, the next key added to VAE_TRAINING_DEFAULTS would ship
        with a bare cast and no range — which is exactly how the holes above got
        in. The ledger is documentation; the tests below are the enforcement.
        """
        self.assertEqual(set(_KEY_VERDICTS), set(VAE_TRAINING_DEFAULTS))
        self.assertEqual(len(_KEY_VERDICTS), 47)

    def test_the_shipped_defaults_resolve(self):
        """Every bound above must admit the value the SSOT ships."""
        cfg = self._assert_accepted()
        for key, default in VAE_TRAINING_DEFAULTS.items():
            if key == "lpips_weight":
                continue  # forced to 0 by _NO_LPIPS
            if key == "vae_path":
                continue  # filled from base_model_path under vae_source='model'
            self.assertEqual(cfg[key], default, f"default for {key} was rejected")

    # ── learning rate ────────────────────────────────────────────────────
    def test_a_zero_learning_rate_is_refused(self):
        self._assert_refused("every optimizer step is a no-op",
                             vae={"learning_rate": 0})

    def test_a_negative_learning_rate_is_refused(self):
        self._assert_refused("ascends the loss instead of descending",
                             vae={"learning_rate": -1e-5})

    def test_a_small_positive_learning_rate_is_accepted(self):
        self.assertEqual(self._assert_accepted(
            vae={"learning_rate": 1e-9})["learning_rate"], 1e-9)

    # ── gradient clipping: 0 MEANS DISABLED, negative is refused ─────────
    def test_max_grad_norm_zero_is_accepted_and_means_no_clipping(self):
        """Refusing 0 would break the field's meaning everywhere else in the
        repo (base_trainer / fused_optimizer_groups both use `> 0` to gate the
        clip, the latter documenting "0 to disable"), and the UI has always
        offered it. The trainer is what was wrong, not the value — see
        ``VaeTrainerGateTest.test_max_grad_norm_zero_disables_clipping``."""
        self.assertEqual(self._assert_accepted(
            vae={"max_grad_norm": 0})["max_grad_norm"], 0.0)

    def test_a_negative_max_grad_norm_is_refused(self):
        self._assert_refused("flips the sign of every gradient",
                             vae={"max_grad_norm": -1.0})

    def test_an_infinite_max_grad_norm_is_routed_to_the_zero_spelling(self):
        """`inf` WAS a working way to disable clipping (clip_grad_norm_ clamps
        the scale factor at 1.0). It is refused now so that "off" has exactly
        one spelling, but the message must name that spelling rather than
        reporting a generic non-finite typo."""
        message = self._assert_refused("max_grad_norm=0",
                                       vae={"max_grad_norm": float("inf")})
        self.assertNotIn("must be a finite number", message)

    def test_a_negative_weight_decay_is_refused(self):
        self._assert_refused("grows the VAE's weights without limit",
                             vae={"optimizer_weight_decay": -0.001})

    # ── optimizer / scheduler names ──────────────────────────────────────
    def test_every_listed_optimizer_is_accepted(self):
        for name in VALID_OPTIMIZERS:
            with self.subTest(optimizer=name):
                self.assertEqual(
                    self._assert_accepted(vae={"optimizer": name})["optimizer"], name)

    def test_an_unknown_optimizer_is_refused_before_the_model_load(self):
        self._assert_refused("OptimizerFactory resolves exactly this set",
                             vae={"optimizer": "sgd"})

    def test_the_ringbuffer_optimizers_are_accepted_because_they_run(self):
        """They do NOT need the allocator their name implies.

        ``OptimizerFactory`` passes ``get_state_buffer=None`` and both
        implementations then take their "Ring Buffer disabled: GPU allocation"
        branch. Verified by construction plus a live ``step()`` on CUDA: the
        state is allocated as uint8 on cuda:0 and the parameters move. Refusing
        them would break a configuration that works today, so the misleading
        name is handled by a log line in ``build_optimizer`` instead.
        """
        for name in ("adamw8bit_ringbuffer", "lion8bit_ringbuffer"):
            with self.subTest(optimizer=name):
                self.assertIn(name, VALID_OPTIMIZERS)
                self.assertEqual(
                    self._assert_accepted(vae={"optimizer": name})["optimizer"], name)

    def test_the_optimizer_enum_is_exactly_what_the_factory_resolves(self):
        """The enum is derived from OptimizerFactory, not guessed at. Reading
        its source keeps this test free of a CUDA/bitsandbytes dependency while
        still failing if the factory grows or loses a name."""
        import inspect
        import re
        from core.training.optimizer_factory import OptimizerFactory
        source = inspect.getsource(OptimizerFactory.create_optimizer)
        names = set()
        for match in re.finditer(r'optimizer_type\s*==\s*"([a-z0-9_]+)"', source):
            names.add(match.group(1))
        for match in re.finditer(r'optimizer_type\s+in\s+\[([^\]]+)\]', source):
            names.update(re.findall(r'"([a-z0-9_]+)"', match.group(1)))
        self.assertEqual(names, set(VALID_OPTIMIZERS))

    def test_optimizer_case_is_folded_like_optimizerfactory_folds_it(self):
        self.assertEqual(
            self._assert_accepted(vae={"optimizer": "AdamW"})["optimizer"], "adamw")

    def test_every_listed_lr_scheduler_is_accepted(self):
        for name in VALID_LR_SCHEDULERS:
            with self.subTest(lr_scheduler=name):
                self.assertEqual(
                    self._assert_accepted(vae={"lr_scheduler": name})["lr_scheduler"],
                    name)

    def test_an_unrunnable_lr_scheduler_is_refused(self):
        """The trainer CATCHES a get_scheduler failure and continues at a
        constant LR, so an unknown name is not an error at run time — it is a
        silently ignored schedule. `piecewise_constant` is a real diffusers name
        that would always land there (no `step_rules` is ever passed)."""
        for name in ("piecewise_constant", "cosine_annealing", ""):
            with self.subTest(lr_scheduler=name):
                self._assert_refused("silently ignore the schedule",
                                     vae={"lr_scheduler": name})

    # ── warmup vs the schedule that is supposed to consume it ────────────
    def test_a_warmup_under_the_constant_scheduler_is_refused(self):
        """diffusers' get_scheduler returns the CONSTANT schedule *before* the
        `num_warmup_steps` argument is used at all, so the pair is recorded
        everywhere and applied nowhere. Both keys are UI-reachable and
        `constant` is the shipped default, which makes this the likeliest
        spelling of the mistake."""
        self._assert_refused("returns the constant schedule without ever",
                             vae={"lr_scheduler": "constant",
                                  "lr_warmup_steps": 500})

    def test_the_same_warmup_under_constant_with_warmup_is_accepted(self):
        cfg = self._assert_accepted(vae={"lr_scheduler": "constant_with_warmup",
                                         "lr_warmup_steps": 500})
        self.assertEqual(cfg["lr_warmup_steps"], 500)

    def test_the_constant_scheduler_with_no_warmup_is_still_the_default(self):
        cfg = self._assert_accepted()
        self.assertEqual(cfg["lr_scheduler"], "constant")
        self.assertEqual(cfg["lr_warmup_steps"], 0)

    def test_every_other_scheduler_accepts_a_warmup(self):
        for name in VALID_LR_SCHEDULERS:
            if name == "constant":
                continue
            with self.subTest(lr_scheduler=name):
                self._assert_accepted(vae={"lr_scheduler": name,
                                           "lr_warmup_steps": 100})

    # ── warmup vs run length ─────────────────────────────────────────────
    # Each row pins lr_scheduler explicitly so that it exercises the run-length
    # rule and not the 'constant ignores warmup' rule next to it.
    def test_a_warmup_at_least_as_long_as_the_run_is_refused(self):
        for warmup in (100, 500):
            with self.subTest(lr_warmup_steps=warmup):
                self._assert_refused("entire run would be warmup", vae={
                    "lr_scheduler": "constant_with_warmup",
                    "total_steps": 100, "lr_warmup_steps": warmup,
                })

    def test_a_warmup_shorter_than_the_run_is_accepted(self):
        cfg = self._assert_accepted(vae={"lr_scheduler": "constant_with_warmup",
                                         "total_steps": 100,
                                         "lr_warmup_steps": 99})
        self.assertEqual(cfg["lr_warmup_steps"], 99)

    # ── counts and cadences ──────────────────────────────────────────────
    def test_validation_num_images_below_one_is_refused(self):
        """0 makes items[:-0] == items[:0], i.e. an EMPTY training split, and -1
        resolves to items[:1] — training on a single image."""
        for bad in (0, -1):
            with self.subTest(validation_num_images=bad):
                self._assert_refused("TRAINING split empty",
                                     vae={"validation_num_images": bad})

    def test_negative_cadences_are_refused_rather_than_silently_disabling(self):
        self._assert_refused("only tests validation_every > 0",
                             vae={"validation_every": -1})
        self._assert_refused("only tests save_every > 0", vae={"save_every": -1})
        self._assert_refused("quietly fill the disk",
                             vae={"max_step_saves_to_keep": -1})
        self._assert_refused("rejected by DataLoader itself",
                             vae={"num_workers": -1})

    def test_zero_cadences_stay_accepted_because_they_mean_something(self):
        cfg = self._assert_accepted(vae={"validation_every": 0, "save_every": 0,
                                         "max_step_saves_to_keep": 0,
                                         "num_workers": 0, "lr_warmup_steps": 0})
        self.assertEqual(cfg["validation_every"], 0)
        self.assertEqual(cfg["save_every"], 0)
        self.assertEqual(cfg["max_step_saves_to_keep"], 0)
        self.assertEqual(cfg["num_workers"], 0)

    def test_a_seed_the_run_would_not_actually_use_is_refused(self):
        """Not because the value crashes anything — `random.seed(-1)` and
        `torch.manual_seed(-1)` are both legal, and the trainer already takes a
        modulus for numpy. It is refused because that modulus makes the three
        generators disagree with the seed written into train_state.json and the
        sidecar, so the recorded seed does not reproduce the run."""
        for bad in (-1, 2 ** 32, 2 ** 32 + 7):
            with self.subTest(seed=bad):
                self._assert_refused("does not use the seed it records",
                                     vae={"seed": bad})
        self.assertEqual(self._assert_accepted(
            vae={"seed": 2 ** 32 - 1})["seed"], 2 ** 32 - 1)
        self.assertEqual(self._assert_accepted(vae={"seed": 0})["seed"], 0)

    def test_the_documented_seed_aliasing_is_real(self):
        """The refusal message's claim, checked rather than asserted."""
        self.assertEqual((-1) % (2 ** 32), 4294967295)
        self.assertEqual((2 ** 32 + 7) % (2 ** 32), 7)

    # ── YCbCr sub-parameters ─────────────────────────────────────────────
    def test_a_negative_ycbcr_channel_weight_is_refused(self):
        """It does not disable the channel: the term is summed over channels, so
        a negative weight pays the run for increasing that channel's error."""
        for key in ("ycbcr_dc_y_weight", "ycbcr_dc_chroma_weight"):
            with self.subTest(key=key):
                self._assert_refused("pays the run for increasing", vae={key: -1.0})

    def test_both_ycbcr_channel_weights_zero_is_refused_while_the_term_is_on(self):
        """The hole the audit named: the 'at least one active loss' check only
        sees ycbcr_dc_weight, so this passes it while the objective is
        identically zero."""
        self._assert_refused("only looks at ycbcr_dc_weight", vae={
            "mse_weight": 0, "l1_weight": 0, "lpips_weight": 0,
            "pattern_weight": 0, "ycbcr_dc_weight": 0.1,
            "ycbcr_dc_y_weight": 0, "ycbcr_dc_chroma_weight": 0,
        })

    def test_both_ycbcr_channel_weights_zero_is_accepted_with_the_term_off(self):
        """Off by default must stay completely inert — the same rule the
        l_invented sub-parameters follow."""
        cfg = self._assert_accepted(vae={
            "ycbcr_dc_weight": 0, "ycbcr_dc_y_weight": 0,
            "ycbcr_dc_chroma_weight": 0,
        })
        self.assertEqual(cfg["ycbcr_dc_y_weight"], 0.0)

    def test_a_non_positive_charbonnier_epsilon_is_refused(self):
        for bad in (0, -1e-3):
            with self.subTest(ycbcr_dc_eps=bad):
                self._assert_refused("Charbonnier smoothing constant",
                                     vae={"ycbcr_dc_eps": bad})

    # ── pattern term ─────────────────────────────────────────────────────
    def test_pattern_size_below_one_is_refused(self):
        self._assert_refused("divides by zero at 0", vae={"pattern_size": 0})

    def test_a_pattern_cell_larger_than_the_crop_is_refused_while_active(self):
        self._assert_refused("no whole cell to group by", vae={
            "pattern_weight": 0.1, "pattern_size": 1024, "resolution": 512,
        })

    def test_a_pattern_cell_exactly_the_crop_size_is_accepted(self):
        """The boundary, pinned because ``>`` vs ``>=`` is a one-character
        mistake that no other row catches. ``PatternLoss.forward`` computes
        ``h_c = (h // p) * p``, which is 512 for p=512, so the ``h_c < p``
        early-return is NOT taken and the phase variance is a real value over a
        single cell — a legitimate (if degenerate) configuration."""
        cfg = self._assert_accepted(vae={"pattern_weight": 0.1,
                                         "pattern_size": 512,
                                         "resolution": 512})
        self.assertEqual(cfg["pattern_size"], 512)

    def test_the_single_cell_pattern_term_really_is_computable(self):
        """The reason the boundary above is accepted, asserted against the loss
        itself rather than against a comment."""
        import torch
        from core.training.vae.vae_losses import PatternLoss
        loss = PatternLoss(512)
        pred = torch.randn(1, 3, 512, 512)
        value = loss(pred, torch.zeros_like(pred))
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(float(value), 0.0)

    def test_a_pattern_cell_larger_than_the_crop_is_inert_when_the_term_is_off(self):
        cfg = self._assert_accepted(vae={"pattern_size": 1024, "resolution": 512})
        self.assertEqual(cfg["pattern_size"], 1024)

    # ── type discipline ──────────────────────────────────────────────────
    def test_non_finite_numbers_are_refused(self):
        for key in ("mse_weight", "learning_rate", "ema_decay"):
            for bad in (float("nan"), float("inf"), "nan", "-inf"):
                with self.subTest(key=key, value=bad):
                    self._assert_refused("must be a finite number", vae={key: bad})

    def test_a_fractional_count_is_refused_rather_than_truncated(self):
        for key in ("total_steps", "batch_size", "validation_num_images"):
            with self.subTest(key=key):
                self._assert_refused("Truncating it", vae={key: 2.5})

    def test_a_whole_number_written_as_a_float_is_accepted(self):
        """YAML/JSON round trips turn 2000 into 2000.0 often enough that this
        must not become a refusal."""
        cfg = self._assert_accepted(vae={"total_steps": 2000.0, "batch_size": 2.0})
        self.assertEqual(cfg["total_steps"], 2000)
        self.assertIsInstance(cfg["total_steps"], int)

    def test_booleans_are_not_numbers(self):
        """bool is an int subclass, so float(True) == 1.0 would let a stray
        `mse_weight: true` become a weight of 1.0 and `batch_size: true` a batch
        of one."""
        self._assert_refused("(a boolean)", vae={"mse_weight": True})
        self._assert_refused("(a boolean)", vae={"batch_size": True})

    def test_non_string_paths_are_refused(self):
        for key in ("vae_path", "vae_arch"):
            with self.subTest(key=key):
                self._assert_refused("must be a string", vae={key: ["a", "b"]})

    def test_a_non_numeric_count_names_the_key(self):
        self._assert_refused(["total_steps", "must be an integer"],
                             vae={"total_steps": "many"})


class VaeTrainerGateTest(unittest.TestCase):
    """The trainer's own second gate (defence in depth for non-resolver callers).

    Imports torch, so it is kept apart from the pure-config cases above.
    """

    def _cfg(self, **overrides):
        cfg = copy.deepcopy(dict(VAE_TRAINING_DEFAULTS))
        cfg["resume_from"] = None
        cfg["lpips_weight"] = 0.0
        cfg.update(overrides)
        return cfg

    def test_trainer_parses_gate_keys_strictly_too(self):
        """A cfg that bypassed the resolver must not be gated by truthiness."""
        from core.training.vae.vae_trainer import VaeTrainer
        trainer = VaeTrainer(
            self._cfg(train_encoder="false", acknowledge_latent_space_break="false"),
            output_dir=".", run_name="gate_test")
        self.assertFalse(trainer.train_encoder)
        self.assertEqual(trainer._export_suffix(), "_vae")
        with self.assertRaises(VaeConfigError):
            VaeTrainer(self._cfg(train_encoder="maybe"),
                       output_dir=".", run_name="gate_test")

    def test_trainer_refuses_a_hand_built_cfg_with_only_train_encoder(self):
        from core.training.vae.vae_trainer import VaeTrainer
        with self.assertRaises(VaeConfigError) as ctx:
            VaeTrainer(self._cfg(train_encoder=True),
                       output_dir=".", run_name="gate_test")
        self.assertIn("acknowledge_latent_space_break", str(ctx.exception))

    def test_trainer_accepts_both_gate_keys_and_flags_encoder_training(self):
        from core.training.vae.vae_trainer import VaeTrainer
        trainer = VaeTrainer(
            self._cfg(train_encoder=True, acknowledge_latent_space_break=True),
            output_dir=".", run_name="gate_test")
        self.assertTrue(trainer.train_encoder)
        self.assertEqual(trainer._export_suffix(), "_vae_encoder_trained")

    # ── gradient clipping: 0 must DISABLE it, not zero every gradient ────
    def _clip_probe(self, max_grad_norm):
        """A trainer whose only trainable tensor carries a gradient of norm 5."""
        import torch
        from core.training.vae.vae_trainer import VaeTrainer
        trainer = VaeTrainer(self._cfg(max_grad_norm=max_grad_norm),
                             output_dir=".", run_name="clip_test")
        param = torch.nn.Parameter(torch.zeros(2))
        param.grad = torch.tensor([3.0, 4.0])
        trainer.trainable_params = [param]
        return trainer, param, float(trainer._clip_gradients())

    def test_max_grad_norm_zero_disables_clipping(self):
        """The defect: ``clip_grad_norm_(params, 0)`` scales every gradient by
        ``0 / total_norm``, i.e. to exactly zero. The optimizer step is then a
        no-op except for AdamW's decoupled weight decay, so the run reports
        success while shrinking the weights it was asked to train. 0 now means
        "no clipping", as it does in base_trainer and fused_optimizer_groups.
        """
        import torch
        _, param, norm = self._clip_probe(0.0)
        self.assertTrue(torch.equal(param.grad, torch.tensor([3.0, 4.0])),
                        "max_grad_norm=0 zeroed the gradients instead of "
                        "disabling clipping")
        # The unclipped norm is still reported: it is charted as grad_norm and
        # is the only warning that a run is about to diverge.
        self.assertAlmostEqual(norm, 5.0, places=5)

    def test_a_positive_max_grad_norm_still_clips(self):
        import torch
        _, param, norm = self._clip_probe(1.0)
        self.assertAlmostEqual(norm, 5.0, places=5)   # the norm BEFORE clipping
        self.assertAlmostEqual(float(param.grad.norm()), 1.0, places=5)
        self.assertFalse(torch.equal(param.grad, torch.tensor([3.0, 4.0])))

    def test_clipping_a_parameter_set_with_no_gradients_is_not_a_crash(self):
        import torch
        from core.training.vae.vae_trainer import VaeTrainer
        trainer = VaeTrainer(self._cfg(max_grad_norm=0.0),
                             output_dir=".", run_name="clip_test")
        trainer.trainable_params = [torch.nn.Parameter(torch.zeros(2))]
        self.assertEqual(float(trainer._clip_gradients()), 0.0)

    # ── the ringbuffer names run, but not as their name says ─────────────
    def test_a_ringbuffer_optimizer_states_what_the_run_actually_gets(self):
        """The mitigation that pays for accepting the misleading names.

        They are NOT refused (they work: with no allocator both fall back to a
        GPU allocation of the 8-bit state), so the only thing standing between
        the name and a wrong expectation is this notice. Asserted on the pure
        function rather than through build_optimizer, which would compile a CUDA
        extension.
        """
        from core.training.vae.vae_trainer import VaeTrainer
        for name, plain in (("adamw8bit_ringbuffer", "adamw8bit"),
                            ("lion8bit_ringbuffer", "lion8bit"),
                            ("AdamW8bit_RingBuffer", "adamw8bit")):
            with self.subTest(optimizer=name):
                note = VaeTrainer.optimizer_placement_note(name)
                self.assertIsNotNone(note, f"no placement note for {name}")
                self.assertIn("no ring-buffer allocator", note)
                self.assertIn("allocated on the GPU", note)
                self.assertIn(plain, note)

    def test_build_optimizer_actually_prints_the_placement_note(self):
        """The wiring, not just the two ends of it.

        A note that is computed and dropped is the same as no note at all, and
        it is the ONLY thing standing between the ring-buffer name and a wrong
        expectation. OptimizerFactory is stubbed so the case stays GPU-free (the
        real one compiles a CUDA extension).
        """
        import contextlib
        import io
        import torch
        from core.training import optimizer_factory
        from core.training.vae.vae_trainer import VaeTrainer

        param = torch.nn.Parameter(torch.zeros(4))
        original = optimizer_factory.OptimizerFactory.create_optimizer
        optimizer_factory.OptimizerFactory.create_optimizer = staticmethod(
            lambda **kwargs: torch.optim.SGD([param], lr=kwargs["learning_rate"]))
        self.addCleanup(setattr, optimizer_factory.OptimizerFactory,
                        "create_optimizer", original)

        seen = {}
        for name in ("adamw8bit_ringbuffer", "adamw"):
            trainer = VaeTrainer(self._cfg(optimizer=name),
                                 output_dir=".", run_name="note_test")
            trainer.trainable_params = [param]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                trainer.build_optimizer()
            seen[name] = buf.getvalue()
        self.assertIn("no ring-buffer allocator", seen["adamw8bit_ringbuffer"])
        self.assertNotIn("ring-buffer", seen["adamw"])

    def test_ordinary_optimizers_get_no_placement_note(self):
        from core.training.vae.vae_trainer import VaeTrainer
        for name in ("adamw", "adamw8bit", "adafactor", "lion8bit"):
            with self.subTest(optimizer=name):
                self.assertIsNone(VaeTrainer.optimizer_placement_note(name))

    def test_decoder_only_export_suffix_is_unchanged(self):
        from core.training.vae.vae_trainer import VaeTrainer
        trainer = VaeTrainer(self._cfg(), output_dir=".", run_name="gate_test")
        self.assertFalse(trainer.train_encoder)
        self.assertEqual(trainer._export_suffix(), "_vae")

    # ── resume: measurement-basis changes warn, they do not refuse ────────
    def _checkpoint_with(self, saved_config: dict):
        """A throwaway checkpoint dir carrying only train_state.json."""
        import json
        import tempfile
        from pathlib import Path
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ckpt = Path(tmp.name) / "step_00001000"
        ckpt.mkdir(parents=True)
        with open(ckpt / "train_state.json", "w", encoding="utf-8") as f:
            json.dump({"config": saved_config}, f)
        return ckpt

    def _resume_output(self, saved_config: dict, **cfg_overrides):
        import contextlib
        import io
        from core.training.vae.vae_trainer import VaeTrainer
        trainer = VaeTrainer(self._cfg(**cfg_overrides),
                             output_dir=".", run_name="gate_test")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            trainer._assert_component_set_matches(
                self._checkpoint_with(saved_config))
        return buf.getvalue()

    def test_a_changed_validation_resolution_warns_on_resume(self):
        """The default moved 512 -> 1024, so a process.vae that OMITS the key now
        resolves differently than when the checkpoint was written -- and the
        resumed run appends to the SAME vae_val_psnr series with no fresh baseline
        point (global_step != 0). Nothing else detects that."""
        out = self._resume_output(
            {"train_decoder": True, "train_encoder": False,
             "decoder_blocks": "all", "validation_resolution": 512},
            validation_resolution=1024)
        self.assertIn("validation_resolution", out)
        self.assertIn("NOT comparable", out)
        self.assertIn("warning, not a refusal", out)

    def test_a_changed_crop_scale_policy_warns_on_resume(self):
        out = self._resume_output(
            {"train_decoder": True, "train_encoder": False,
             "decoder_blocks": "all", "crop_scale_policy": "downscale"},
            crop_scale_policy="native")
        self.assertIn("crop_scale_policy", out)
        self.assertIn("NOT comparable", out)

    def test_an_unchanged_measurement_basis_is_silent(self):
        out = self._resume_output(
            {"train_decoder": True, "train_encoder": False,
             "decoder_blocks": "all", "crop_scale_policy": "downscale",
             "validation_resolution": 1024})
        self.assertNotIn("WARNING", out)

    def test_a_pre_policy_checkpoint_does_not_warn_about_a_key_it_never_had(self):
        """Absent != changed. Run 113's checkpoints predate crop_scale_policy, so
        comparing against a default would invent a mismatch."""
        out = self._resume_output(
            {"train_decoder": True, "train_encoder": False,
             "decoder_blocks": "all", "validation_resolution": 1024},
            crop_scale_policy="native")
        self.assertNotIn("crop_scale_policy", out)

    def test_a_measurement_change_is_a_warning_not_an_exception(self):
        """It must never block a resume: both keys are legitimate to change."""
        self._resume_output(
            {"train_decoder": True, "train_encoder": False,
             "decoder_blocks": "all", "validation_resolution": 512,
             "crop_scale_policy": "downscale"},
            validation_resolution=1024, crop_scale_policy="mixed")

    def test_a_component_set_mismatch_still_refuses(self):
        """The fatal branch must not have been softened by the warning branch."""
        from core.training.vae.vae_trainer import VaeTrainer
        trainer = VaeTrainer(self._cfg(), output_dir=".", run_name="gate_test")
        with self.assertRaises(VaeConfigError):
            trainer._assert_component_set_matches(self._checkpoint_with(
                {"train_decoder": True, "train_encoder": False,
                 "decoder_blocks": "conv_out"}))

    def test_a_changed_optimizer_refuses_before_state_load(self):
        from core.training.vae.vae_trainer import VaeTrainer
        trainer = VaeTrainer(self._cfg(optimizer="adamw"),
                             output_dir=".", run_name="gate_test")
        with self.assertRaises(VaeConfigError) as ctx:
            trainer._assert_component_set_matches(self._checkpoint_with({
                "train_decoder": True,
                "train_encoder": False,
                "decoder_blocks": "all",
                "optimizer": "adafactor",
            }))
        self.assertIn("used optimizer", str(ctx.exception))
        self.assertIn("start a new run", str(ctx.exception))

    def test_bare_ldm_write_is_refused_for_an_encoder_trained_run(self):
        from pathlib import Path
        from core.training.vae.vae_trainer import VaeTrainer
        trainer = VaeTrainer(
            self._cfg(train_encoder=True, acknowledge_latent_space_break=True),
            output_dir=".", run_name="gate_test")
        with self.assertRaises(VaeConfigError) as ctx:
            trainer.save_bare_ldm_safetensors(Path("./gate_test_vae_encoder_trained"))
        self.assertIn("config.json", str(ctx.exception))


class VaeResumeBaseVaeIdentityTest(unittest.TestCase):
    """A resume must not splice this run's base VAE with another run's tensors.

    ``load_checkpoint`` restores ONLY the trainable tensors; everything else in
    the model comes from the base VAE this run loaded. Resuming a checkpoint that
    was trained against a different base therefore yields a hybrid model that no
    file describes, with no error and no warning — the tensor names all match.
    The guard under test compares the checkpoint's recorded base-VAE identity
    against the current one, and the authoritative axis is a fingerprint of the
    frozen (i.e. NOT restored) half of the weights.
    """

    def _cfg(self, **overrides):
        cfg = copy.deepcopy(dict(VAE_TRAINING_DEFAULTS))
        cfg["resume_from"] = None
        cfg["lpips_weight"] = 0.0
        cfg.update(overrides)
        return cfg

    def _trainer(self, identity=None, **cfg_overrides):
        from core.training.vae.vae_trainer import VaeTrainer
        trainer = VaeTrainer(self._cfg(**cfg_overrides),
                             output_dir=".", run_name="identity_test")
        if identity is not None:
            trainer._base_vae_identity = copy.deepcopy(identity)
        return trainer

    def _checkpoint(self, base_vae=None, config=None, omit_base_vae=False):
        import json
        import tempfile
        from pathlib import Path
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ckpt = Path(tmp.name) / "step_00001000"
        ckpt.mkdir(parents=True)
        state = {"config": config if config is not None else {
            "train_decoder": True, "train_encoder": False,
            "decoder_blocks": "all",
            "validation_resolution": VAE_TRAINING_DEFAULTS["validation_resolution"],
            "crop_scale_policy": VAE_TRAINING_DEFAULTS["crop_scale_policy"],
        }}
        if not omit_base_vae:
            state["base_vae"] = base_vae
        with open(ckpt / "train_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f)
        return ckpt

    @staticmethod
    def _identity(path, digest="aaaa", *, fingerprint=True, **extra):
        ident = {"format": "diffusers_dir", "path": path,
                 "class": "AutoencoderKL", "latent_channels": 4,
                 "scaling_factor": 0.13025, "shift_factor": 0.0}
        if fingerprint:
            from core.training.vae.vae_trainer import _FROZEN_FP_ALGO
            ident["frozen_fingerprint"] = {"algo": _FROZEN_FP_ALGO,
                                           "digest": digest, "tensor_count": 3}
        ident.update(extra)
        return ident

    def _run(self, trainer, ckpt):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            trainer._assert_component_set_matches(ckpt)
        return buf.getvalue()

    # ── the defect: a changed base VAE must not resume silently ──────────
    def test_a_different_base_vae_refuses(self):
        trainer = self._trainer(self._identity("M:/model/sdxl/VAE/new", "bbbb"))
        with self.assertRaises(VaeConfigError) as ctx:
            trainer._assert_component_set_matches(
                self._checkpoint(self._identity("M:/model/sdxl/VAE/old", "aaaa")))
        msg = str(ctx.exception)
        self.assertIn("DIFFERENT base VAE", msg)
        self.assertIn("frozen base weights", msg)
        self.assertIn("aaaa", msg)          # what the checkpoint was trained on
        self.assertIn("bbbb", msg)          # what this run loaded
        self.assertIn("vae_path", msg)      # how to fix it

    def test_a_different_latent_space_refuses_even_without_a_fingerprint(self):
        """Structure is recorded by every checkpoint ever written, so it is the
        fallback axis for checkpoints that predate the fingerprint."""
        trainer = self._trainer(
            self._identity("M:/model/flux2/VAE", fingerprint=False,
                           latent_channels=16))
        with self.assertRaises(VaeConfigError) as ctx:
            trainer._assert_component_set_matches(self._checkpoint(
                self._identity("M:/model/sdxl/VAE", fingerprint=False)))
        self.assertIn("latent_channels", str(ctx.exception))

    def test_a_different_vae_class_refuses(self):
        trainer = self._trainer(
            self._identity("M:/model/x", fingerprint=False,
                           **{"class": "AutoencoderKLFlux2"}))
        with self.assertRaises(VaeConfigError) as ctx:
            trainer._assert_component_set_matches(self._checkpoint(
                self._identity("M:/model/x", fingerprint=False)))
        self.assertIn("VAE class", str(ctx.exception))

    # ── the same base must still resume, however it is spelled ───────────
    def test_the_same_base_vae_resumes_silently(self):
        trainer = self._trainer(self._identity("M:/model/sdxl/VAE", "aaaa"))
        out = self._run(trainer,
                        self._checkpoint(self._identity("M:/model/sdxl/VAE", "aaaa")))
        self.assertNotIn("WARNING", out)

    def test_the_same_weights_under_a_different_path_or_format_resume_silently(self):
        """Moved drive / relative spelling / single file instead of a diffusers
        directory: the fingerprint settles it, so none of these may warn, let
        alone refuse. Over-strict identity would break routine operations.
        (A single file that ALSO resolves a different ``scaling_factor`` does
        warn about that one key — see
        ``test_a_changed_scaling_factor_warns_even_when_the_digests_match``.)"""
        trainer = self._trainer(self._identity("D:/models/vae/sdxl.safetensors",
                                               "aaaa", format="single_file"))
        out = self._run(trainer, self._checkpoint(
            self._identity("M:/model/sdxl/VAE", "aaaa")))
        self.assertNotIn("WARNING", out)

    def test_path_spelling_alone_does_not_warn(self):
        """Separator, trailing slash and (on Windows) case are spelling, not
        identity. ``os.path.abspath`` already folds the first two on nt; the
        explicit folding in ``_normalized_path`` covers the case axis and the
        POSIX side of the separator axis."""
        trainer = self._trainer(
            self._identity("M:\\model\\sdxl\\VAE\\", fingerprint=False))
        out = self._run(trainer, self._checkpoint(
            self._identity("M:/model/sdxl/VAE", fingerprint=False)))
        self.assertNotIn("WARNING", out)

        if os.name == "nt":
            trainer = self._trainer(
                self._identity("m:/MODEL/sdxl/vae", fingerprint=False))
            out = self._run(trainer, self._checkpoint(
                self._identity("M:/model/sdxl/VAE", fingerprint=False)))
            self.assertNotIn("WARNING", out)

    # ── weak evidence warns, it does not refuse ──────────────────────────
    def test_a_changed_path_without_a_fingerprint_warns_but_runs(self):
        trainer = self._trainer(self._identity("M:/model/sdxl/VAE/new",
                                               fingerprint=False))
        out = self._run(trainer, self._checkpoint(
            self._identity("M:/model/sdxl/VAE/old", fingerprint=False)))
        self.assertIn("WARNING", out)
        self.assertIn("path", out)
        self.assertIn("predates the frozen-weight fingerprint", out)

    def test_a_pre_identity_checkpoint_still_resumes(self):
        """Old format: train_state.json without a base_vae key at all."""
        trainer = self._trainer(self._identity("M:/model/sdxl/VAE", "aaaa"))
        out = self._run(trainer, self._checkpoint(omit_base_vae=True))
        self.assertIn("records no base VAE", out)
        self.assertIn("Proceeding", out)

    def test_a_trainer_with_no_identity_is_unaffected(self):
        """The measurement-warning tests drive this method on a trainer that
        never loaded a model; it must stay silent there."""
        trainer = self._trainer()
        out = self._run(trainer, self._checkpoint(
            self._identity("M:/model/sdxl/VAE", "aaaa")))
        self.assertNotIn("WARNING", out)

    def test_an_incomparable_fingerprint_algorithm_warns_rather_than_refuses(self):
        trainer = self._trainer(self._identity("M:/model/sdxl/VAE/new", "bbbb"))
        old = self._identity("M:/model/sdxl/VAE/old", "aaaa")
        old["frozen_fingerprint"]["algo"] = "some-older-algo"
        out = self._run(trainer, self._checkpoint(old))
        self.assertIn("WARNING", out)
        self.assertIn("different algorithms", out)

    # ── the fingerprint covers the right half of the model ───────────────
    def _tiny_vae(self, frozen_fill: float, trained_fill: float):
        import torch
        module = torch.nn.Module()
        module.frozen_a = torch.nn.Parameter(torch.full((2, 2), frozen_fill),
                                             requires_grad=False)
        module.frozen_b = torch.nn.Parameter(torch.full((3,), frozen_fill),
                                             requires_grad=False)
        module.trained = torch.nn.Parameter(torch.full((2, 2), trained_fill),
                                            requires_grad=True)
        return module

    def _digest(self, vae):
        trainer = self._trainer()
        trainer.vae = vae
        fp = trainer._compute_frozen_fingerprint()
        self.assertIsNotNone(fp)
        return fp

    def test_the_fingerprint_ignores_the_tensors_a_resume_overwrites(self):
        """Restarting from an EXPORT of the same run changes only the trained
        half, and the checkpoint overwrites that half anyway — so it must not be
        flagged. This is the false positive a whole-model hash would produce."""
        a = self._digest(self._tiny_vae(1.0, 1.0))
        b = self._digest(self._tiny_vae(1.0, 7.0))
        self.assertEqual(a["digest"], b["digest"])
        self.assertEqual(a["tensor_count"], 2)

    def test_the_fingerprint_changes_when_a_frozen_tensor_changes(self):
        a = self._digest(self._tiny_vae(1.0, 1.0))
        b = self._digest(self._tiny_vae(1.5, 1.0))
        self.assertNotEqual(a["digest"], b["digest"])

    def test_the_fingerprint_is_stable_across_dtypes(self):
        """The digest describes the WEIGHTS, not the dtype they are held in, so
        the exactly-representable values here survive an fp16 round trip."""
        import torch
        a = self._digest(self._tiny_vae(1.0, 1.0))
        half = self._tiny_vae(1.0, 1.0).to(dtype=torch.float16)
        self.assertEqual(a["digest"], self._digest(half)["digest"])

    def test_no_model_means_no_fingerprint_rather_than_a_crash(self):
        trainer = self._trainer()
        self.assertIsNone(trainer._compute_frozen_fingerprint())
        trainer._record_frozen_fingerprint()   # must be a no-op, not an error
        self.assertNotIn("frozen_fingerprint", trainer._base_vae_identity)

    # ── proven-identical weights outrank a structural LABEL ──────────────
    def test_a_renamed_vae_class_does_not_refuse_when_the_digests_match(self):
        """A diffusers upgrade that renames ``_class_name`` must not strand a
        long run: the digest already proves the frozen half is bit-identical, so
        no hybrid is possible and the difference is in the description only."""
        trainer = self._trainer(self._identity("M:/model/x", "aaaa",
                                               **{"class": "AutoencoderKLQwenImage"}))
        out = self._run(trainer, self._checkpoint(
            self._identity("M:/model/x", "aaaa")))
        self.assertIn("VAE class", out)
        self.assertIn("bit-identical", out)
        self.assertIn("no hybrid is possible", out)

    def test_a_newly_reported_latent_channels_does_not_refuse_when_digests_match(self):
        """Observed for real: a VAE class that reports no ``latent_channels`` is
        recorded as -1, and a later diffusers version may start reporting 16."""
        trainer = self._trainer(self._identity("M:/model/x", "aaaa",
                                               latent_channels=16))
        out = self._run(trainer, self._checkpoint(
            self._identity("M:/model/x", "aaaa", latent_channels=-1)))
        self.assertIn("latent_channels", out)
        self.assertNotIn("DIFFERENT base VAE", out)

    def test_a_structural_mismatch_is_still_fatal_when_the_digests_differ(self):
        trainer = self._trainer(self._identity("M:/model/x", "bbbb",
                                               latent_channels=16))
        with self.assertRaises(VaeConfigError) as ctx:
            trainer._assert_component_set_matches(self._checkpoint(
                self._identity("M:/model/x", "aaaa")))
        msg = str(ctx.exception)
        self.assertIn("frozen base weights", msg)
        self.assertIn("latent_channels", msg)

    # ── a changed export-baked factor is never silent ────────────────────
    def test_a_changed_scaling_factor_warns_even_when_the_digests_match(self):
        """`scaling_factor` is not spelling: `save_pretrained` bakes it into the
        exported config.json and the sidecar/inference override path reads it, so
        a resume must not change what the run finally writes in silence."""
        trainer = self._trainer(self._identity("M:/model/x", "aaaa",
                                               scaling_factor=0.18215))
        out = self._run(trainer, self._checkpoint(
            self._identity("M:/model/x", "aaaa")))
        self.assertIn("scaling_factor", out)
        self.assertIn("0.18215", out)
        self.assertIn("exported config.json", out)

    def test_a_changed_shift_factor_warns_even_when_the_digests_match(self):
        trainer = self._trainer(self._identity("M:/model/x", "aaaa",
                                               shift_factor=0.1159))
        out = self._run(trainer, self._checkpoint(
            self._identity("M:/model/x", "aaaa")))
        self.assertIn("shift_factor", out)

    def test_equal_factors_with_equal_digests_stay_silent(self):
        """The path/format axis must remain suppressed by a matching digest --
        only the export-baked factors escalate to a warning."""
        trainer = self._trainer(self._identity("D:/moved/vae.safetensors", "aaaa",
                                               format="single_file"))
        out = self._run(trainer, self._checkpoint(
            self._identity("M:/model/x", "aaaa")))
        self.assertNotIn("WARNING", out)

    # ── the recording side: select_trainable must WIRE the fingerprint in ─
    def _tiny_autoencoder(self, encoder_fill=1.0, decoder_fill=1.0):
        """A stand-in with the attributes ``select_trainable`` walks.

        ``requires_grad_(False)`` mirrors ``load_base_vae``, which freezes the
        whole model before the trainable subset is unfrozen.
        """
        import torch
        module = torch.nn.Module()
        module.encoder = torch.nn.Conv2d(3, 4, 1)
        module.decoder = torch.nn.Conv2d(4, 3, 1)
        module.post_quant_conv = torch.nn.Conv2d(4, 4, 1)
        with torch.no_grad():
            for p in module.encoder.parameters():
                p.fill_(encoder_fill)
            for p in module.decoder.parameters():
                p.fill_(decoder_fill)
        module.requires_grad_(False)
        return module

    def _select_trainable(self, trainer):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.select_trainable()

    def test_select_trainable_records_the_fingerprint_into_the_identity(self):
        """The recording wiring, not just the two ends of it.

        ``_base_vae_identity`` is the dict written verbatim into every
        ``train_state.json`` (``save_checkpoint``) and into the export sidecar, so
        if this assignment is lost the authoritative axis silently stops being
        recorded and the guard degrades to path warnings with CI still green.
        """
        trainer = self._trainer(self._identity("M:/model/x", fingerprint=False))
        trainer.vae = self._tiny_autoencoder()
        self._select_trainable(trainer)
        fp = trainer._base_vae_identity.get("frozen_fingerprint")
        self.assertIsNotNone(fp, "select_trainable did not record a fingerprint")
        self.assertEqual(fp, trainer._compute_frozen_fingerprint())
        # Only the encoder is frozen under the default decoder_blocks='all'.
        self.assertEqual(fp["tensor_count"], 2)

    def test_the_recorded_fingerprint_is_what_the_resume_guard_compares(self):
        """End to end over the real recording path: two runs whose FROZEN halves
        differ produce checkpoints that refuse each other, and a run against the
        same base resumes silently. Nothing here injects an identity by hand."""
        old = self._trainer(self._identity("M:/model/x", fingerprint=False))
        old.vae = self._tiny_autoencoder(encoder_fill=1.0, decoder_fill=1.0)
        self._select_trainable(old)
        ckpt = self._checkpoint(copy.deepcopy(old._base_vae_identity))

        same = self._trainer(self._identity("M:/model/x", fingerprint=False))
        # Same frozen weights, DIFFERENT trained weights: the checkpoint
        # overwrites those, so this must resume silently.
        same.vae = self._tiny_autoencoder(encoder_fill=1.0, decoder_fill=9.0)
        self._select_trainable(same)
        self.assertNotIn("WARNING", self._run(same, ckpt))

        other = self._trainer(self._identity("M:/model/x", fingerprint=False))
        other.vae = self._tiny_autoencoder(encoder_fill=2.0, decoder_fill=1.0)
        self._select_trainable(other)
        with self.assertRaises(VaeConfigError) as ctx:
            other._assert_component_set_matches(ckpt)
        self.assertIn("frozen base weights", str(ctx.exception))


class VaeCropScalePolicyTest(unittest.TestCase):
    """The loader side of the crop scale policy.

    Synthetic images (no dataset dependency), but the geometry asserted is the
    real thing: the reference expression for ``downscale`` is a verbatim copy of
    the pre-policy implementation, so a regression there fails here rather than
    24 hours into a fine-tune. Imports torch/PIL, hence its own class.
    """

    @classmethod
    def setUpClass(cls):
        import tempfile
        from PIL import Image

        cls._tmp = tempfile.TemporaryDirectory()
        cls.paths = {}
        # (w, h): a downscale case (short 1200 -> factor 2.34x at 512), a
        # square-ish one, and one BELOW the crop so the upscale branch is covered.
        for w, h in ((1700, 1200), (1024, 1024), (400, 300)):
            path = os.path.join(cls._tmp.name, f"{w}x{h}.png")
            # Deterministic non-uniform content: a constant image would make a
            # pixel comparison pass whatever the geometry did.
            img = Image.new("RGB", (w, h))
            img.putdata([((x * 7) % 256, (x // w * 11) % 256, (x * 3) % 256)
                         for x in range(w * h)])
            img.save(path)
            cls.paths[(w, h)] = path

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @staticmethod
    def _legacy_load(path, resolution, random_crop, rng):
        """The implementation as it stood before crop_scale_policy existed."""
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(path) as im:
            image = im.convert("RGB")
            w, h = image.size
            scale = resolution / min(w, h)
            if scale != 1.0:
                new_w = max(resolution, int(round(w * scale)))
                new_h = max(resolution, int(round(h * scale)))
                image = image.resize((new_w, new_h), Image.LANCZOS)
                w, h = new_w, new_h
            max_left, max_top = w - resolution, h - resolution
            if random_crop:
                left = rng.randint(0, max_left) if max_left > 0 else 0
                top = rng.randint(0, max_top) if max_top > 0 else 0
            else:
                left, top = max_left // 2, max_top // 2
            image = image.crop((left, top, left + resolution, top + resolution))
            arr = np.array(image).astype(np.float32) / 255.0
            arr = (arr - 0.5) * 2.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def test_downscale_policy_is_pixel_identical_to_the_legacy_loader(self):
        """Run 113 has 52k steps of history under this exact geometry."""
        import random as _random
        import torch
        from core.training.vae.vae_dataset import load_image_tensor

        for size, path in self.paths.items():
            for random_crop in (False, True):
                with self.subTest(size=size, random_crop=random_crop):
                    got = load_image_tensor(
                        path, 512, random_crop=random_crop,
                        rng=_random.Random(1234), scale_policy="downscale")
                    want = self._legacy_load(
                        path, 512, random_crop, _random.Random(1234))
                    self.assertTrue(torch.equal(got, want))

    def test_native_policy_does_not_resample_a_large_enough_image(self):
        from core.training.vae.vae_dataset import resolve_crop_scale
        self.assertEqual(resolve_crop_scale(1200, 512, scale_policy="native"), 1.0)
        self.assertEqual(resolve_crop_scale(512, 512, scale_policy="native"), 1.0)

    def test_every_policy_upscales_an_image_smaller_than_the_crop(self):
        """4.21% of the corpus: there is no resolution-sized window to crop, so
        the upscale branch must be common to all three policies."""
        from core.training.vae.vae_dataset import resolve_crop_scale
        for policy in VALID_CROP_SCALE_POLICIES:
            with self.subTest(policy=policy):
                self.assertAlmostEqual(
                    resolve_crop_scale(300, 512, scale_policy=policy),
                    512 / 300, places=12)

    def test_native_and_mixed_still_produce_the_requested_shape(self):
        import torch
        from core.training.vae.vae_dataset import load_image_tensor
        for policy in VALID_CROP_SCALE_POLICIES:
            for size, path in self.paths.items():
                with self.subTest(policy=policy, size=size):
                    t = load_image_tensor(path, 512, scale_policy=policy)
                    self.assertEqual(tuple(t.shape), (3, 512, 512))
                    self.assertEqual(t.dtype, torch.float32)
                    self.assertGreaterEqual(float(t.min()), -1.0)
                    self.assertLessEqual(float(t.max()), 1.0)

    def test_mixed_draws_cover_the_range_including_both_ends(self):
        import random as _random
        from core.training.vae.vae_dataset import resolve_crop_scale

        rng = _random.Random(7)
        f_max = 2400 / 512
        factors = [1.0 / resolve_crop_scale(2400, 512, scale_policy="mixed", rng=rng)
                   for _ in range(4000)]
        self.assertGreaterEqual(min(factors), 1.0)
        self.assertLessEqual(max(factors), f_max + 1e-9)
        # Log-uniform: the median sits at sqrt(f_max), NOT at f_max/2 (which is
        # what a linear-uniform draw would give). This is the property that keeps
        # the corpus-level distribution from being dragged towards the
        # heavily-resampled regime by the largest sources.
        factors.sort()
        median = factors[len(factors) // 2]
        self.assertAlmostEqual(median, f_max ** 0.5, delta=0.1)
        self.assertLess(median, f_max / 2)

    def test_mixed_respects_the_max_downscale_bound(self):
        import random as _random
        from core.training.vae.vae_dataset import resolve_crop_scale

        rng = _random.Random(11)
        factors = [1.0 / resolve_crop_scale(4000, 512, scale_policy="mixed",
                                            max_downscale=2.0, rng=rng)
                   for _ in range(2000)]
        self.assertGreaterEqual(min(factors), 1.0)
        self.assertLessEqual(max(factors), 2.0 + 1e-9)

    def test_a_bound_of_one_degenerates_to_native(self):
        from core.training.vae.vae_dataset import resolve_crop_scale
        self.assertEqual(
            resolve_crop_scale(4000, 512, scale_policy="mixed", max_downscale=1.0),
            1.0)

    def test_the_dataset_refuses_an_unknown_policy_at_construction(self):
        from core.training.vae.vae_dataset import VaeRawImageDataset
        items = [{"image_path": p} for p in self.paths.values()]
        with self.assertRaises(ValueError):
            VaeRawImageDataset(items, 512, scale_policy="fullsize")

    def test_the_dataset_passes_the_policy_through_per_sample(self):
        import torch
        from core.training.vae.vae_dataset import VaeRawImageDataset
        items = [{"image_path": self.paths[(1700, 1200)]}]
        native = VaeRawImageDataset(items, 512, scale_policy="native")[0]
        downscaled = VaeRawImageDataset(items, 512, scale_policy="downscale")[0]
        self.assertFalse(torch.equal(native, downscaled))

    def test_validation_batch_is_deterministic_and_policy_independent(self):
        """The held-out metric must not move because a TRAINING knob moved, and
        must be identical call to call (it is re-derived on resume)."""
        import torch
        from core.training.vae.vae_dataset import make_validation_batch
        items = [{"image_path": p} for p in self.paths.values()]
        first = make_validation_batch(items, 512, len(items))
        for _ in range(3):
            self.assertTrue(torch.equal(first, make_validation_batch(
                items, 512, len(items))))
        # It takes no policy argument at all, by construction, so there is no way
        # for a caller to make it follow crop_scale_policy.
        import inspect
        self.assertEqual(list(inspect.signature(make_validation_batch).parameters),
                         ["items", "resolution", "count"])


class VaeLossBankKlTest(unittest.TestCase):
    """The KL term exists only when the encoder is trainable."""

    def test_kl_is_not_constructed_under_a_frozen_encoder(self):
        import torch
        from core.training.vae.vae_losses import VaeLossBank
        cfg = dict(VAE_TRAINING_DEFAULTS)
        cfg["lpips_weight"] = 0.0  # keep the test free of the LPIPS weight download
        bank = VaeLossBank(cfg, torch.device("cpu"), kl_enabled=False)
        self.assertFalse(bank.kl_enabled)
        self.assertEqual(bank.kl_weight, 0.0)
        recon = torch.zeros(1, 3, 64, 64)
        target = torch.zeros(1, 3, 64, 64)
        _, parts = bank(recon, target)
        self.assertNotIn("kl", parts)

    def test_kl_is_constructed_and_finite_when_the_encoder_is_trainable(self):
        import torch
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from core.training.vae.vae_losses import VaeLossBank
        cfg = dict(VAE_TRAINING_DEFAULTS)
        cfg["lpips_weight"] = 0.0
        cfg["kl_weight"] = 1e-6
        bank = VaeLossBank(cfg, torch.device("cpu"), kl_enabled=True)
        self.assertTrue(bank.kl_enabled)
        moments = torch.randn(2, 8, 8, 8)  # [B, 2*C, H, W] -> mean/logvar split
        posterior = DiagonalGaussianDistribution(moments)
        recon = torch.zeros(2, 3, 64, 64)
        target = torch.zeros(2, 3, 64, 64)
        total, parts = bank(recon, target, posterior)
        self.assertIn("kl", parts)        # raw, literature-comparable
        self.assertIn("kl_term", parts)   # weighted contribution, charted
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(parts["kl"], 0.0)

    def test_kl_contribution_is_normalised_to_the_reconstruction_reduction(self):
        """LDM weights the KL against a recon SUMMED over C*H*W; this bank's
        recon terms are MEAN-reduced, so the KL is divided by the per-image
        element count before weighting. Without that, kl_weight=1e-6 is ~C*H*W
        times too strong and the balance moves with resolution."""
        import torch
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from core.training.vae.vae_losses import VaeLossBank
        cfg = dict(VAE_TRAINING_DEFAULTS)
        cfg["lpips_weight"] = 0.0
        cfg["mse_weight"] = 0.0
        cfg["ycbcr_dc_weight"] = 0.0
        cfg["kl_weight"] = 1e-6
        bank = VaeLossBank(cfg, torch.device("cpu"), kl_enabled=True)
        posterior = DiagonalGaussianDistribution(torch.randn(2, 8, 8, 8))
        contributions = {}
        for size in (64, 128):
            x = torch.zeros(2, 3, size, size)
            total, parts = bank(x, x, posterior)
            # The contribution is exactly weight * raw / (C*H*W).
            self.assertAlmostEqual(
                parts["kl_term"], 1e-6 * parts["kl"] / (3 * size * size), places=12)
            self.assertAlmostEqual(float(total), parts["kl_term"], places=12)
            contributions[size] = parts["kl_term"]
        # Resolution invariance: the same posterior contributes 4x less per
        # element at 2x the side length, which is what makes the knob mean the
        # same thing at any training resolution.
        self.assertAlmostEqual(contributions[64] / contributions[128], 4.0, places=6)

    def test_missing_posterior_is_an_error_rather_than_a_silent_skip(self):
        import torch
        from core.training.vae.vae_losses import VaeLossBank
        cfg = dict(VAE_TRAINING_DEFAULTS)
        cfg["lpips_weight"] = 0.0
        bank = VaeLossBank(cfg, torch.device("cpu"), kl_enabled=True)
        with self.assertRaises(ValueError):
            bank(torch.zeros(1, 3, 64, 64), torch.zeros(1, 3, 64, 64))


class VaeResumeCompletenessTest(unittest.TestCase):
    """An incomplete checkpoint must not resume as if it were complete.

    ``load_checkpoint`` used to load ``optimizer.pt`` / ``lr_scheduler.pt``
    *only if the file existed*, while still adopting the checkpoint's global
    step. A step directory that was copied or written only in part therefore
    resumed at, say, step 10,000 with freshly-initialised Adam moments or a
    schedule sitting at position 0, and reported a normal resume.

    The tiers under test (see ``VaeTrainer._assert_checkpoint_complete``):
    weights / optimizer / scheduler / train_state.json are REFUSALS, EMA and
    RNG are warn-and-repair.
    """

    # ── fixtures ─────────────────────────────────────────────────────────
    def _write_checkpoint(self, *, ema=True, scheduler=True, manifest=True,
                          step=10000, parent=None):
        """A complete, real checkpoint on disk, written the way the trainer
        writes one (including the artifact manifest).

        ``parent`` places several step directories in one ``checkpoints/``
        folder, which is what the intact-sibling listing reads.
        """
        import json
        import random as _random
        import tempfile
        import numpy as np
        import torch
        from pathlib import Path
        from safetensors.torch import save_file
        from core.training.vae.vae_trainer import _CKPT_ARTIFACTS

        if parent is None:
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            parent = Path(tmp.name)
        ckpt = Path(parent) / f"step_{step:08d}"
        ckpt.mkdir(parents=True)

        params = [torch.nn.Parameter(torch.full((4,), 3.0)) for _ in self.NAMES]
        opt = torch.optim.AdamW([{"params": [p]} for p in params], lr=1e-5)
        for p in params:                      # give Adam real moment state
            p.grad = torch.ones_like(p)
        opt.step()

        save_file({n: p.detach().clone() for n, p in zip(self.NAMES, params)},
                  str(ckpt / "vae_decoder.safetensors"))
        if ema:
            save_file({n: p.detach().clone() for n, p in zip(self.NAMES, params)},
                      str(ckpt / "ema.safetensors"))
        torch.save(opt.state_dict(), ckpt / "optimizer.pt")
        if scheduler:
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
            torch.save(sched.state_dict(), ckpt / "lr_scheduler.pt")
        torch.save({"python": _random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": None},
                   ckpt / "rng_state.pt")

        state = {
            "step": step,
            "ema_enabled": ema,
            "ema_updates": 500,
            "ema_retained_init_fraction": 0.5,
            "config": {"train_encoder": False, "train_decoder": True,
                       "decoder_blocks": "all"},
        }
        if manifest:
            state["artifacts"] = {n: (ckpt / n).stat().st_size
                                  for n in _CKPT_ARTIFACTS
                                  if (ckpt / n).is_file()}
        with open(ckpt / "train_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f)
        return ckpt

    NAMES = ["decoder.a", "decoder.b"]

    def _trainer(self, *, ema=True, scheduler=True):
        import torch
        from core.training.vae.vae_trainer import VaeTrainer
        t = VaeTrainer.__new__(VaeTrainer)
        t.log_prefix = "[VaeTrainer]"
        t.device = torch.device("cpu")
        t.train_encoder = False
        t.cfg = {"learning_rate": 1e-5, "train_decoder": True,
                 "decoder_blocks": "all", "encoder_blocks": "none",
                 "lr_scheduler": "constant", "ema_decay": 0.999,
                 "ema_enabled": ema}
        t.trainable_names = list(self.NAMES)
        t.trainable_params = [torch.nn.Parameter(torch.zeros(4))
                              for _ in self.NAMES]
        t.optimizer = torch.optim.AdamW(
            [{"params": [p]} for p in t.trainable_params], lr=1e-5)
        t.lr_scheduler = (torch.optim.lr_scheduler.LambdaLR(
            t.optimizer, lr_lambda=lambda s: 1.0) if scheduler else None)
        t.ema = ({n: p.detach().clone().float()
                  for n, p in zip(t.trainable_names, t.trainable_params)}
                 if ema else None)
        t._ema_updates = 0
        t._ema_retained_init = 1.0
        return t

    def _resume(self, trainer, ckpt):
        """Run the real load_checkpoint, capturing stdout."""
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            trainer.load_checkpoint(ckpt)
        return buf.getvalue()

    def _refusal(self, trainer, ckpt):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(VaeConfigError) as ctx:
                trainer.load_checkpoint(ckpt)
        return str(ctx.exception)

    # ── the complete checkpoint still resumes, fully ─────────────────────
    def test_a_complete_checkpoint_resumes_every_piece_of_state(self):
        import torch
        ckpt = self._write_checkpoint()
        trainer = self._trainer()
        out = self._resume(trainer, ckpt)

        self.assertEqual(trainer.global_step, 10000)
        self.assertTrue(torch.allclose(trainer.trainable_params[0].detach(),
                                       torch.full((4,), 3.0)))
        # Optimizer state actually arrived (exp_avg exists for every param).
        state = trainer.optimizer.state_dict()["state"]
        self.assertEqual(len(state), len(self.NAMES))
        self.assertTrue(all("exp_avg" in s for s in state.values()))
        # EMA was restored, so its warmup counters continue.
        self.assertEqual(trainer._ema_updates, 500)
        self.assertAlmostEqual(trainer._ema_retained_init, 0.5)
        self.assertNotIn("WARNING", out)

    def test_a_pre_manifest_checkpoint_still_resumes(self):
        """Checkpoints written before the manifest existed (every checkpoint on
        disk today) must keep resuming, on presence alone."""
        ckpt = self._write_checkpoint(manifest=False)
        out = self._resume(self._trainer(), ckpt)
        self.assertNotIn("WARNING", out)
        self.assertIn("presence only", out)

    # ── refusals ─────────────────────────────────────────────────────────
    def test_a_missing_optimizer_state_refuses(self):
        ckpt = self._write_checkpoint()
        (ckpt / "optimizer.pt").unlink()
        msg = self._refusal(self._trainer(), ckpt)
        self.assertIn("incomplete", msg)
        self.assertIn("optimizer.pt", msg)
        self.assertIn("accumulated history", msg)

    def test_a_missing_optimizer_state_refuses_without_a_manifest_too(self):
        ckpt = self._write_checkpoint(manifest=False)
        (ckpt / "optimizer.pt").unlink()
        self.assertIn("optimizer.pt", self._refusal(self._trainer(), ckpt))

    def test_a_missing_lr_scheduler_state_refuses(self):
        ckpt = self._write_checkpoint()
        (ckpt / "lr_scheduler.pt").unlink()
        msg = self._refusal(self._trainer(), ckpt)
        self.assertIn("lr_scheduler.pt", msg)
        self.assertIn("position 0", msg)

    # ── conditionally-written artifacts on a pre-manifest checkpoint ─────
    # save_checkpoint writes ema.safetensors / lr_scheduler.pt only when the run
    # HAS an EMA / a scheduler. Without a manifest, "never written" and "lost"
    # are indistinguishable, so the same absence must not be a refusal in one
    # checkpoint generation and a warning in the other.
    def test_an_absent_scheduler_state_without_a_manifest_warns_not_refuses(self):
        """The build_optimizer fallback ("LR scheduler unavailable; using
        constant LR") writes exactly this checkpoint. Refusing it would leave
        no way to resume once the cause is fixed."""
        ckpt = self._write_checkpoint(manifest=False, scheduler=False)
        trainer = self._trainer(scheduler=True)
        out = self._resume(trainer, ckpt)
        self.assertIn("WARNING", out)
        self.assertIn("lr_scheduler.pt", out)
        self.assertIn("position 0", out)
        self.assertEqual(trainer.global_step, 10000)

    def test_an_absent_ema_without_a_manifest_warns_and_reseeds(self):
        import torch
        ckpt = self._write_checkpoint(manifest=False, ema=False)
        trainer = self._trainer(ema=True)
        out = self._resume(trainer, ckpt)
        self.assertIn("WARNING", out)
        self.assertIn("ema.safetensors", out)
        self.assertEqual(trainer.global_step, 10000)
        self.assertTrue(torch.allclose(trainer.ema["decoder.a"],
                                       torch.full((4,), 3.0)))
        self.assertEqual(trainer._ema_updates, 0)

    def test_an_absent_unconditional_artifact_without_a_manifest_still_refuses(self):
        """rng_state.pt is warn-tier anyway, so the asymmetry is shown on the
        two files the writer ALWAYS writes: absent means lost, manifest or
        not."""
        for name in ("optimizer.pt", "vae_decoder.safetensors"):
            ckpt = self._write_checkpoint(manifest=False)
            (ckpt / name).unlink()
            self.assertIn(name, self._refusal(self._trainer(), ckpt))

    # ── the refusal has to say which checkpoint IS resumable ─────────────
    def test_a_refusal_lists_the_intact_sibling_checkpoints(self):
        """`resume_from: latest` is the shipped default and picks the highest
        step without inspecting it — and the damaged directory is typically the
        newest one, so the refusal must name a way out."""
        import tempfile
        from pathlib import Path
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        parent = Path(tmp.name) / "checkpoints"
        parent.mkdir()
        self._write_checkpoint(parent=parent, step=5000)
        self._write_checkpoint(parent=parent, step=7500)
        newest = self._write_checkpoint(parent=parent, step=10000)
        (newest / "optimizer.pt").unlink()

        msg = self._refusal(self._trainer(), newest)
        self.assertIn("step_00005000", msg)
        self.assertIn("step_00007500", msg)
        self.assertNotIn("step_00010000,", msg)   # not offered as a way out
        self.assertIn("resume_from_checkpoint", msg)

    def test_a_damaged_sibling_is_not_offered_as_a_way_out(self):
        import tempfile
        from pathlib import Path
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        parent = Path(tmp.name) / "checkpoints"
        parent.mkdir()
        older = self._write_checkpoint(parent=parent, step=5000)
        (older / "train_state.json").unlink()
        newest = self._write_checkpoint(parent=parent, step=10000)
        (newest / "optimizer.pt").unlink()

        msg = self._refusal(self._trainer(), newest)
        self.assertNotIn("step_00005000", msg)
        self.assertIn("No other checkpoint", msg)

    def test_a_truncated_optimizer_state_refuses(self):
        """The failure the manifest exists for: the file is THERE, and shorter
        than it was saved (an interrupted copy)."""
        ckpt = self._write_checkpoint()
        blob = (ckpt / "optimizer.pt").read_bytes()
        (ckpt / "optimizer.pt").write_bytes(blob[: len(blob) // 2])
        msg = self._refusal(self._trainer(), ckpt)
        self.assertIn("optimizer.pt", msg)
        self.assertIn("incompletely", msg)

    def test_a_zero_byte_artifact_is_refused_with_or_without_a_manifest(self):
        """An empty artifact is never valid state; the verdict must not depend
        on the checkpoint's generation."""
        for manifest in (True, False):
            ckpt = self._write_checkpoint(manifest=manifest)
            (ckpt / "optimizer.pt").write_bytes(b"")
            msg = self._refusal(self._trainer(), ckpt)
            self.assertIn("optimizer.pt", msg)
            self.assertIn("0 bytes", msg)

    def test_a_truncated_weights_file_refuses(self):
        ckpt = self._write_checkpoint()
        blob = (ckpt / "vae_decoder.safetensors").read_bytes()
        (ckpt / "vae_decoder.safetensors").write_bytes(blob[:64])
        msg = self._refusal(self._trainer(), ckpt)
        self.assertIn("vae_decoder.safetensors", msg)

    def test_several_missing_artifacts_are_all_named_at_once(self):
        ckpt = self._write_checkpoint()
        (ckpt / "optimizer.pt").unlink()
        (ckpt / "lr_scheduler.pt").unlink()
        (ckpt / "rng_state.pt").unlink()
        msg = self._refusal(self._trainer(), ckpt)
        self.assertIn("optimizer.pt", msg)
        self.assertIn("lr_scheduler.pt", msg)
        # ...and the refusal explains the supported way to reset optimisation.
        self.assertIn("start a NEW run", msg)

    def test_a_missing_train_state_refuses(self):
        ckpt = self._write_checkpoint()
        (ckpt / "train_state.json").unlink()
        msg = self._refusal(self._trainer(), ckpt)
        self.assertIn("train_state.json", msg)

    def test_an_unreadable_train_state_refuses(self):
        ckpt = self._write_checkpoint()
        (ckpt / "train_state.json").write_text("{not json", encoding="utf-8")
        self.assertIn("train_state.json", self._refusal(self._trainer(), ckpt))

    def test_a_corrupt_optimizer_file_of_the_right_size_refuses(self):
        """Sizes cannot catch same-length corruption; torch.load does, and the
        result must be a refusal rather than a silent skip."""
        ckpt = self._write_checkpoint()
        blob = bytearray((ckpt / "optimizer.pt").read_bytes())
        blob[len(blob) // 2:] = b"\x00" * (len(blob) - len(blob) // 2)
        (ckpt / "optimizer.pt").write_bytes(bytes(blob))
        msg = self._refusal(self._trainer(), ckpt)
        self.assertIn("optimizer.pt", msg)
        # A state dict written by a DIFFERENT optimizer type lands here too, so
        # the message must not diagnose "damaged file" as the only cause.
        self.assertIn("different optimizer", msg)
        self.assertIn("optimizer=", msg)

    # ── warn-and-repair tiers ────────────────────────────────────────────
    def test_a_missing_rng_state_warns_and_resumes(self):
        ckpt = self._write_checkpoint()
        (ckpt / "rng_state.pt").unlink()
        trainer = self._trainer()
        out = self._resume(trainer, ckpt)
        self.assertEqual(trainer.global_step, 10000)
        self.assertIn("WARNING", out)
        self.assertIn("rng_state.pt", out)
        self.assertIn("data_epoch", out)

    def test_a_missing_ema_warns_reseeds_and_restarts_the_ema_ramp(self):
        import torch
        ckpt = self._write_checkpoint()
        (ckpt / "ema.safetensors").unlink()
        trainer = self._trainer()
        out = self._resume(trainer, ckpt)
        self.assertIn("WARNING", out)
        self.assertIn("ema.safetensors", out)
        self.assertEqual(trainer.global_step, 10000)
        # Re-seeded from the restored weights...
        self.assertTrue(torch.allclose(trainer.ema["decoder.a"],
                                       torch.full((4,), 3.0)))
        # ...and the warmup ramp restarts: adopting the checkpoint's 500 updates
        # would average that seed in at the full decay.
        self.assertEqual(trainer._ema_updates, 0)
        self.assertAlmostEqual(trainer._ema_retained_init, 1.0)

    def test_a_truncated_ema_warns_and_reseeds(self):
        ckpt = self._write_checkpoint()
        blob = (ckpt / "ema.safetensors").read_bytes()
        (ckpt / "ema.safetensors").write_bytes(blob[:64])
        trainer = self._trainer()
        out = self._resume(trainer, ckpt)
        self.assertIn("ema.safetensors", out)
        self.assertEqual(trainer.global_step, 10000)
        self.assertEqual(trainer._ema_updates, 0)

    def test_a_checkpoint_without_ema_resumed_with_ema_on_warns(self):
        ckpt = self._write_checkpoint(ema=False)
        trainer = self._trainer(ema=True)
        out = self._resume(trainer, ckpt)
        self.assertIn("WARNING", out)
        self.assertIn("EMA disabled", out)
        self.assertEqual(trainer._ema_updates, 0)

    def test_a_checkpoint_with_ema_resumed_with_ema_off_warns(self):
        ckpt = self._write_checkpoint(ema=True)
        trainer = self._trainer(ema=False)
        out = self._resume(trainer, ckpt)
        self.assertIn("WARNING", out)
        self.assertIn("ema_enabled=false", out)
        self.assertIsNone(trainer.ema)

    def test_a_scheduler_added_after_the_checkpoint_warns_rather_than_refuses(self):
        """The manifest proves the file was never written, so its absence is a
        config change, not damage."""
        ckpt = self._write_checkpoint(scheduler=False)
        trainer = self._trainer(scheduler=True)
        out = self._resume(trainer, ckpt)
        self.assertIn("WARNING", out)
        self.assertIn("no LR scheduler", out)
        self.assertEqual(trainer.global_step, 10000)

    def test_a_scheduler_removed_after_the_checkpoint_warns(self):
        ckpt = self._write_checkpoint(scheduler=True)
        trainer = self._trainer(scheduler=False)
        out = self._resume(trainer, ckpt)
        self.assertIn("WARNING", out)
        self.assertIn("discarded", out)
        self.assertEqual(trainer.global_step, 10000)

    # ── the manifest itself, written by the REAL save_checkpoint ─────────
    def _saved_by_the_real_writer(self, trainer, step=4200):
        """Drive ``VaeTrainer.save_checkpoint`` for real, with only the DB row
        and the pruning stubbed out (neither is part of the artifact contract,
        and the DB must not be written from a test)."""
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        trainer.checkpoints_dir = Path(tmp.name) / "checkpoints"
        trainer.checkpoints_dir.mkdir(parents=True)
        trainer.run_name = "manifest_test"
        trainer.run_id = None
        trainer.resume_seq = 0
        trainer._base_vae_identity = None
        trainer._record_checkpoint_row = lambda *a, **k: None
        trainer._prune_checkpoints = lambda *a, **k: None
        with contextlib.redirect_stdout(io.StringIO()):
            return trainer.save_checkpoint(step)

    def test_the_writer_records_a_size_manifest_for_every_file_it_wrote(self):
        import json
        from core.training.vae.vae_trainer import _CKPT_ARTIFACTS
        ckpt = self._saved_by_the_real_writer(self._trainer())
        state = json.loads((ckpt / "train_state.json").read_text(encoding="utf-8"))
        manifest = state.get("artifacts")
        self.assertIsInstance(manifest, dict,
                              "save_checkpoint wrote no artifact manifest, so a "
                              "partially-copied checkpoint cannot be detected")
        on_disk = {n for n in _CKPT_ARTIFACTS if (ckpt / n).is_file()}
        self.assertEqual(set(manifest), on_disk)
        for name, size in manifest.items():
            self.assertEqual((ckpt / name).stat().st_size, size)

    def test_a_checkpoint_from_the_real_writer_round_trips_and_is_verifiable(self):
        ckpt = self._saved_by_the_real_writer(self._trainer())
        trainer = self._trainer()
        out = self._resume(trainer, ckpt)
        self.assertEqual(trainer.global_step, 4200)
        self.assertNotIn("WARNING", out)
        self.assertNotIn("presence only", out)   # i.e. the manifest was used

    def test_a_checkpoint_from_the_real_writer_detects_truncation(self):
        ckpt = self._saved_by_the_real_writer(self._trainer())
        blob = (ckpt / "optimizer.pt").read_bytes()
        (ckpt / "optimizer.pt").write_bytes(blob[:-16])
        msg = self._refusal(self._trainer(), ckpt)
        self.assertIn("optimizer.pt", msg)
        self.assertIn("incompletely", msg)

    def test_every_artifact_the_writer_produces_is_in_the_verified_list(self):
        """A new checkpoint file added to save_checkpoint without adding it to
        _CKPT_ARTIFACTS would be unverifiable on resume, silently.

        Compared against the FILES ON DISK after a real save, not against the
        source text: a future artifact written through a variable path or a
        helper would slip past a source-level check while still being
        unverified.
        """
        from core.training.vae.vae_trainer import _CKPT_ARTIFACTS
        # Both configurations, so the conditional artifacts are covered too.
        for ema, scheduler in ((True, True), (False, False)):
            ckpt = self._saved_by_the_real_writer(
                self._trainer(ema=ema, scheduler=scheduler))
            on_disk = {p.name for p in ckpt.iterdir() if p.is_file()}
            self.assertIn("train_state.json", on_disk)
            self.assertEqual(on_disk - set(_CKPT_ARTIFACTS) - {"train_state.json"},
                             set(),
                             "save_checkpoint writes a file that no resume "
                             "verifies (add it to _CKPT_ARTIFACTS)")
            self.assertIn("vae_decoder.safetensors", on_disk)


class TestTrainingOutputIsCp932Encodable(unittest.TestCase):
    """Every runtime string in the VAE training modules must survive cp932.

    The training subprocess's stdout is cp932 on this platform. A single
    non-ASCII character in a message therefore raises UnicodeEncodeError at the
    moment the message is produced -- and because that happens inside the
    subprocess, it surfaces to the UI as a generic "training failed" rather than
    as an encoding problem. On a SUCCESS path (a print, or a string a print
    consumes) that turns a working configuration into a dead one; on a refusal
    path it replaces a carefully-worded explanation with a traceback about a
    codec.

    Commit 5c257a4e swept the tree for exactly this and established the
    invariant, but left no guard, so it regressed twice (bd0d8a08, 6cc12b0e --
    ten em-dashes, one of them in the optimizer-placement note that
    build_optimizer prints unconditionally for the *_ringbuffer optimizers).
    This test is that guard.

    Scope is the VAE training package only: those modules run inside the
    training subprocess and own its stdout. It is deliberately not a repo-wide
    check. Docstrings are exempt (they are documentation, never printed);
    comments never reach a Constant node and so are exempt automatically. Every
    other string literal is checked, including ones that are returned and
    printed by a caller, because that is precisely the shape the regression
    took.
    """

    ENCODING = "cp932"

    @staticmethod
    def _module_paths():
        vae_pkg = os.path.join(_BACKEND, "core", "training", "vae")
        return sorted(
            os.path.join(vae_pkg, name)
            for name in os.listdir(vae_pkg)
            if name.endswith(".py")
        )

    @classmethod
    def _docstring_node_ids(cls, tree):
        ids = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.FunctionDef,
                                     ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                ids.add(id(body[0].value))
        return ids

    @classmethod
    def offending_literals(cls, path):
        """[(lineno, char, snippet)] for every unencodable non-docstring string."""
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source, filename=path)
        exempt = cls._docstring_node_ids(tree)
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in exempt:
                continue
            for char in node.value:
                try:
                    char.encode(cls.ENCODING)
                except UnicodeEncodeError:
                    # ascii() so this report itself cannot die on a cp932 stdout.
                    found.append((node.lineno, hex(ord(char)),
                                  ascii(node.value[:60])))
                    break
        # ast.walk is breadth-first, not source order.
        return sorted(found)

    def test_no_runtime_string_breaks_the_training_subprocess_stdout(self):
        modules = self._module_paths()
        self.assertTrue(modules, "no VAE training modules found to check")
        offences = []
        for path in modules:
            for lineno, char, snippet in self.offending_literals(path):
                offences.append(
                    f"{os.path.basename(path)}:{lineno}: U+{char[2:].upper()} "
                    f"in {snippet}")
        self.assertEqual(
            offences, [],
            "String literal(s) that cannot be encoded to "
            f"{self.ENCODING}, in modules that print to the training "
            "subprocess's stdout. Replace the character with an ASCII "
            "equivalent (' - ' for an em-dash); see this class's docstring:\n  "
            + "\n  ".join(offences))

    def test_the_check_would_catch_a_reintroduced_em_dash(self):
        """The guard above passes trivially if the detector is broken."""
        import tempfile
        source = (
            "def f():\n"
            '    """A docstring with an em-dash — which is exempt."""\n'
            '    # A comment with an em-dash — which is also exempt.\n'
            '    print(f"placement note — the same state placement")\n'
            '    raise ValueError("refused — for a reason")\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "probe.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)
            found = self.offending_literals(path)
        self.assertEqual([lineno for lineno, _, _ in found], [4, 5],
                         "the detector must flag the print and the raise, and "
                         "exempt the docstring and the comment")


if __name__ == "__main__":
    unittest.main(verbosity=2)
