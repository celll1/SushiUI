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
type, the GAN loss, ``crop_consistency`` / ``invented_hf``) are covered by
``test_unimplemented_matrix_rows_have_no_config_surface``: they must remain
unreachable rather than becoming silently-accepted unknown keys.
"""

from __future__ import annotations

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
    VALID_DECODER_BLOCKS,
    VALID_ENCODER_BLOCKS,
    VALID_DTYPES,
    VALID_LPIPS_NETS,
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

        PiD decoder training (Phase 3), the GAN loss and the crop-consistency /
        invented-HF terms have no keys, so asking for them lands in the
        unknown-key refusal above rather than being silently ignored.
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

    def test_decoder_only_export_suffix_is_unchanged(self):
        from core.training.vae.vae_trainer import VaeTrainer
        trainer = VaeTrainer(self._cfg(), output_dir=".", run_name="gate_test")
        self.assertFalse(trainer.train_encoder)
        self.assertEqual(trainer._export_suffix(), "_vae")

    def test_bare_ldm_write_is_refused_for_an_encoder_trained_run(self):
        from pathlib import Path
        from core.training.vae.vae_trainer import VaeTrainer
        trainer = VaeTrainer(
            self._cfg(train_encoder=True, acknowledge_latent_space_break=True),
            output_dir=".", run_name="gate_test")
        with self.assertRaises(VaeConfigError) as ctx:
            trainer.save_bare_ldm_safetensors(Path("./gate_test_vae_encoder_trained"))
        self.assertIn("config.json", str(ctx.exception))


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
