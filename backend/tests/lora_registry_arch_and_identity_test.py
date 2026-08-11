"""LoRA registry: seeded-dir composition, per-model sibling discovery, arch
tagging (esp. MiniMax-H3 vs FLUX.2/image LoRAs), unambiguous identifiers, and
metadata-derived recommendations.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/lora_registry_arch_and_identity_test.py -v

Two real MiniMax-H3 LoRA files live OUTSIDE the repo at
``M:/model/minimax_h3/loras``; tests that need them skip cleanly (with a
message) when absent instead of failing the suite on a machine without that
model.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.extensions.lora_manager import (  # noqa: E402
    LoRAManager,
    LoRAAmbiguousIdentifierError,
    classify_lora_keys,
)

MINIMAX_H3_LORA_NO_METADATA = (
    r"M:\model\minimax_h3\loras"
    r"\minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors"
)
MINIMAX_H3_LORA_WITH_STUDENT_STEPS = (
    r"M:\model\minimax_h3\loras\minimax_h3_fl2va_4step_lora.safetensors"
)

# A real image LoRA that ships in the repo's configured lora/ dir (SDXL,
# kohya-ss "lora_te1_/lora_te2_/lora_unet_input_blocks_*" format).
IMAGE_LORA_CANDIDATES = [
    os.path.join(_REPO, "lora", "yamatoll_style-000002.safetensors"),
]


def _require(path: str, label: str):
    if not os.path.isfile(path):
        raise unittest.SkipTest(f"{label} not present on this machine: {path}")


class ClassifyLoraKeysTest(unittest.TestCase):
    """classify_lora_keys() is the single signature table shared by scan-time
    arch tagging and get_lora_layers()."""

    def test_minimax_h3_keys_classified_as_minimax_h3_not_flux2(self):
        keys = [
            "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight",
            "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight",
            "diffusion_model.blocks.0.attn.out_proj.lora_A.weight",
            "diffusion_model.blocks.0.mlp.fc1.lora_A.weight",
            "diffusion_model.blocks.0.mlp.fc2.lora_A.weight",
            "diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight",
            "diffusion_model.token_refiner.blocks.0.attn.qkv_proj.lora_A.weight",
            "diffusion_model.final_layer.adaln_proj.linear.lora_A.weight",
        ]
        result = classify_lora_keys(keys)
        self.assertEqual(result["arch"], "minimax_h3")
        # NOT mislabeled as FLUX.2 DUAL/SING blocks (the pre-fix bug: FLUX.2's
        # "transformer_blocks_"/"single_transformer_blocks_" substrings do not
        # appear in MiniMax-H3 keys, but the classifier must not fall through
        # to any other arch's positive match either).
        self.assertFalse(any(b.startswith("DUAL") or b.startswith("SING") for b in result["blocks"]))
        self.assertIn("MMB00", result["blocks"])
        self.assertIn("TREF00", result["blocks"])
        self.assertIn("FINAL", result["blocks"])

    def test_flux2_keys_still_classified_as_flux2(self):
        keys = [
            "lora_transformer_transformer_blocks_0_attn_to_q.lora_down.weight",
            "lora_transformer_transformer_blocks_0_attn_to_q.lora_up.weight",
            "lora_transformer_single_transformer_blocks_0_attn_to_q.lora_down.weight",
        ]
        result = classify_lora_keys(keys)
        self.assertEqual(result["arch"], "flux2")

    def test_sdxl_keys_classified_as_sdxl_not_minimax_h3(self):
        keys = [
            "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
            "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_up.weight",
            "lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
            "lora_unet_input_blocks_4_1_proj_in.lora_down.weight",
            "lora_unet_input_blocks_4_1_proj_in.lora_up.weight",
        ]
        result = classify_lora_keys(keys)
        self.assertEqual(result["arch"], "sdxl")
        self.assertNotEqual(result["arch"], "minimax_h3")

    def test_unrecognized_keys_are_unknown_not_an_error(self):
        result = classify_lora_keys(["some.totally.unrecognized.key.lora_A.weight"])
        self.assertEqual(result["arch"], "unknown")
        self.assertEqual(result["blocks"], ["BASE"])


class RealMiniMaxH3FileDetectionTest(unittest.TestCase):
    """Detect the arch of the two real MiniMax-H3 LoRA files on disk (outside
    the repo). Skips cleanly if the files aren't present on this machine."""

    def test_no_metadata_file_detected_as_minimax_h3(self):
        _require(MINIMAX_H3_LORA_NO_METADATA, "MiniMax-H3 LoRA (no student_steps)")
        manager = LoRAManager(lora_dir=tempfile.mkdtemp())
        from pathlib import Path
        arch, blocks = manager._read_lora_keys_info(Path(MINIMAX_H3_LORA_NO_METADATA))
        self.assertEqual(arch, "minimax_h3")
        self.assertTrue(len(blocks) > 0)

    def test_student_steps_file_detected_as_minimax_h3(self):
        _require(MINIMAX_H3_LORA_WITH_STUDENT_STEPS, "MiniMax-H3 LoRA (student_steps)")
        manager = LoRAManager(lora_dir=tempfile.mkdtemp())
        from pathlib import Path
        arch, blocks = manager._read_lora_keys_info(Path(MINIMAX_H3_LORA_WITH_STUDENT_STEPS))
        self.assertEqual(arch, "minimax_h3")
        self.assertTrue(len(blocks) > 0)


class ImageLoraNotMinimaxH3Test(unittest.TestCase):
    """A real (or synthesized) image LoRA must never be classified minimax_h3."""

    def _find_real_image_lora(self):
        for cand in IMAGE_LORA_CANDIDATES:
            if os.path.isfile(cand):
                return cand
        return None

    def test_image_lora_not_detected_as_minimax_h3(self):
        real = self._find_real_image_lora()
        if real is not None:
            manager = LoRAManager(lora_dir=tempfile.mkdtemp())
            from pathlib import Path
            arch, _ = manager._read_lora_keys_info(Path(real))
            self.assertNotEqual(arch, "minimax_h3")
            self.assertIn(arch, ("sd15", "sdxl", "unknown"))
        else:
            # Synthesize SDXL-shaped keys (no real file under the configured
            # lora dir on this machine).
            keys = [
                "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
                "lora_unet_input_blocks_1_1_proj_in.lora_down.weight",
                "lora_unet_input_blocks_1_1_proj_in.lora_up.weight",
            ]
            result = classify_lora_keys(keys)
            self.assertNotEqual(result["arch"], "minimax_h3")


class SeededDirCompositionTest(unittest.TestCase):
    """set_additional_dirs() must COMPOSE with seeded_dirs, not replace it
    (the bug: it used to wipe the training/ dir seeded at __init__)."""

    def test_seeded_dir_survives_set_additional_dirs(self):
        manager = LoRAManager(lora_dir=tempfile.mkdtemp())
        seeded_before = list(manager.seeded_dirs)
        self.assertTrue(len(seeded_before) >= 1, "expected the training/ dir to be seeded at init")

        user_dir = tempfile.mkdtemp()
        manager.set_additional_dirs([user_dir])

        # Seeded dirs (training/) must still be present after a settings save.
        self.assertEqual([str(d) for d in manager.seeded_dirs], [str(d) for d in seeded_before])
        # And the user dir must also be present in the effective search list.
        effective = [str(d.resolve()) for d in manager._effective_extra_dirs()]
        self.assertIn(str(__import__("pathlib").Path(user_dir).resolve()), effective)

    def test_register_model_sibling_loras_finds_loras_dir(self):
        with tempfile.TemporaryDirectory() as root:
            from pathlib import Path
            root = Path(root)
            (root / "diffusion_models").mkdir()
            (root / "loras").mkdir()
            dit_file = root / "diffusion_models" / "fake_dit.safetensors"
            dit_file.write_bytes(b"")

            manager = LoRAManager(lora_dir=tempfile.mkdtemp())
            registered = manager.register_model_sibling_loras(str(dit_file))
            self.assertTrue(registered)
            effective = [str(d.resolve()) for d in manager._effective_extra_dirs()]
            self.assertIn(str((root / "loras").resolve()), effective)


class ResolveAmbiguityTest(unittest.TestCase):
    """_resolve_lora_path must ERROR on a genuine cross-directory collision
    instead of silently first-matching."""

    def test_ambiguous_bare_identifier_raises(self):
        primary = tempfile.mkdtemp()
        extra_a = tempfile.mkdtemp()
        extra_b = tempfile.mkdtemp()

        manager = LoRAManager(lora_dir=primary)
        # Force two ADDITIONAL dirs (not the primary lora_dir) to both contain
        # the same relative filename -- the exact hazard this closes: multiple
        # per-model `loras/` roots each shipping a differently-purposed file
        # with the same name.
        manager.seeded_dirs = []
        manager.set_additional_dirs([extra_a, extra_b])

        rel = "collide.safetensors"
        with open(os.path.join(extra_a, rel), "wb") as f:
            f.write(b"a")
        with open(os.path.join(extra_b, rel), "wb") as f:
            f.write(b"b")

        with self.assertRaises(LoRAAmbiguousIdentifierError):
            manager._resolve_lora_path(rel)

    def test_non_colliding_identifier_still_resolves_normally(self):
        primary = tempfile.mkdtemp()
        extra = tempfile.mkdtemp()

        manager = LoRAManager(lora_dir=primary)
        manager.seeded_dirs = []
        manager.set_additional_dirs([extra])

        rel = "only_here.safetensors"
        with open(os.path.join(extra, rel), "wb") as f:
            f.write(b"x")

        resolved = manager._resolve_lora_path(rel)
        self.assertIsNotNone(resolved)
        self.assertTrue(str(resolved).startswith(extra))

    def test_primary_dir_always_wins_without_raising(self):
        primary = tempfile.mkdtemp()
        extra = tempfile.mkdtemp()

        manager = LoRAManager(lora_dir=primary)
        manager.seeded_dirs = []
        manager.set_additional_dirs([extra])

        rel = "collide.safetensors"
        with open(os.path.join(primary, rel), "wb") as f:
            f.write(b"primary")
        with open(os.path.join(extra, rel), "wb") as f:
            f.write(b"extra")

        # The default lora_dir is checked FIRST and returned immediately --
        # this is the existing (pre-change) priority behaviour and must not
        # raise, so identifiers already stored against the default dir keep
        # working exactly as before.
        resolved = manager._resolve_lora_path(rel)
        self.assertEqual(str(resolved), str(__import__("pathlib").Path(primary) / rel))

    def test_disambiguated_tag_identifier_resolves_unambiguously(self):
        import torch
        from safetensors.torch import save_file

        primary = tempfile.mkdtemp()
        extra_a = tempfile.mkdtemp()
        extra_b = tempfile.mkdtemp()

        manager = LoRAManager(lora_dir=primary)
        manager.seeded_dirs = []
        manager.set_additional_dirs([extra_a, extra_b])

        rel = "collide.safetensors"
        # Must be a file that survives _is_valid_lora_file's validation (real
        # lora_A/lora_B tensors), otherwise the scan excludes it before the
        # disambiguation logic under test ever runs.
        tensors = {
            "lora_unet_test.lora_A.weight": torch.zeros(2, 2),
            "lora_unet_test.lora_B.weight": torch.zeros(2, 2),
        }
        save_file(tensors, os.path.join(extra_a, rel))
        save_file(tensors, os.path.join(extra_b, rel))

        loras = manager.get_available_loras(force_rescan=True)
        paths = {entry["name"]: entry["path"] for entry in loras}
        # Exactly one entry keeps the bare identifier, the other is tagged.
        tagged = [e["path"] for e in loras if "::" in e["path"]]
        bare = [e["path"] for e in loras if "::" not in e["path"]]
        self.assertEqual(len(tagged), 1)
        self.assertEqual(len(bare), 1)

        resolved = manager._resolve_lora_path(tagged[0])
        self.assertIsNotNone(resolved)


class RecommendedMetadataTest(unittest.TestCase):
    """recommended block parsed from real safetensors metadata."""

    def test_student_steps_file_yields_num_inference_steps_5(self):
        _require(MINIMAX_H3_LORA_WITH_STUDENT_STEPS, "MiniMax-H3 LoRA (student_steps)")
        manager = LoRAManager(lora_dir=tempfile.mkdtemp())
        from pathlib import Path
        recommended = manager._parse_recommended_metadata(Path(MINIMAX_H3_LORA_WITH_STUDENT_STEPS))
        self.assertIsNotNone(recommended)
        self.assertEqual(recommended["num_inference_steps"], 5)
        self.assertEqual(recommended["fbcache_enable"], False)
        self.assertEqual(recommended["spectrum_enable"], False)
        self.assertEqual(recommended["source"], "student_steps")

    def test_no_student_steps_file_yields_no_recommendation(self):
        _require(MINIMAX_H3_LORA_NO_METADATA, "MiniMax-H3 LoRA (no student_steps)")
        manager = LoRAManager(lora_dir=tempfile.mkdtemp())
        from pathlib import Path
        recommended = manager._parse_recommended_metadata(Path(MINIMAX_H3_LORA_NO_METADATA))
        self.assertIsNone(recommended)


if __name__ == "__main__":
    unittest.main()
