"""Guard: a VAE loaded from a bare single file must export the RIGHT
``scaling_factor``.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/test_vae_scaling_factor.py -v
or, without pytest:
    venv/Scripts/python.exe -m unittest discover -s backend/tests -p "test_vae_scaling_factor.py"

Why this file exists
--------------------
``AutoencoderKL.from_single_file`` cannot tell an SDXL VAE from an SD1.5 one:
the two architectures are identical, a VAE-only ``.safetensors`` carries no
``config.json``, and diffusers therefore falls back to
``LDM_VAE_DEFAULT_SCALING_FACTOR = 0.18215``. The SDXL value is 0.13025, so the
fallback is a **1.40x latent-scale error**.

Nothing in the training math reads ``scaling_factor``, which is exactly why this
was dangerous: it surfaced only at EXPORT, where ``save_pretrained`` bakes
``vae.config`` verbatim into ``config.json``, and the inference-side VAE
override trusts a directory's ``config.json`` (the single-file repair in
``pipeline.py`` does not run for a directory). Wrong images, no error, no
warning.

The invariants asserted here are about the VALUE THAT REACHES DISK, not about
any helper being called:

    * a VAE loaded from a bare single file for an ``sdxl`` run ends up at
      0.13025, and a real ``save_diffusers_vae`` export written from it carries
      0.13025 in ``config.json``;
    * an ``sd15`` run still gets 0.18215 -- the fix must not blanket-overwrite;
    * a FULL checkpoint (backbone present) is left alone: there diffusers read
      the family off the checkpoint and its answer is evidence, not a guess;
    * an architecture with no scalar scaling factor (flux2 / qwen_image) or an
      unknown ``vae_arch`` is left alone rather than being given a fabricated
      number;
    * a ``vae_arch`` whose latent-channel count contradicts the loaded file is
      REFUSED, because that config cannot be trusted to decide the number that
      gets baked into the export;
    * the registry's numbers are the ones the corresponding official configs
      carry, and the "diffusers fallback" constant is still diffusers' own.

CPU-only and hermetic: a 1-block AutoencoderKL, a few 1-element tensors written
to a temp ``.safetensors``, no download, no dataset, no GPU.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file

from api.param_defaults import VAE_TRAINING_DEFAULTS
from core.models.common import vae_store
from core.training.vae.vae_config import VaeConfigError
from core.training.vae import vae_trainer as vt

SDXL_SCALING = 0.13025
SD15_SCALING = 0.18215


def _tiny_vae(scaling_factor: float = SD15_SCALING,
              latent_channels: int = 4) -> AutoencoderKL:
    """The smallest real AutoencoderKL that still round-trips through
    ``save_pretrained`` (a mock would not exercise ``register_to_config``)."""
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,),
        layers_per_block=1,
        latent_channels=latent_channels,
        norm_num_groups=2,
        sample_size=32,
        scaling_factor=scaling_factor,
    )


def _write_bare_vae_file(path: Path) -> str:
    """A VAE-ONLY safetensors file: encoder/decoder keys and nothing else."""
    save_file({
        "encoder.conv_in.weight": torch.zeros(1),
        "decoder.conv_out.weight": torch.zeros(1),
        "quant_conv.weight": torch.zeros(1),
        "post_quant_conv.weight": torch.zeros(1),
    }, str(path))
    return str(path)


def _write_full_checkpoint_file(path: Path) -> str:
    """A full checkpoint: VAE keys PLUS a backbone, i.e. evidence of family."""
    save_file({
        "first_stage_model.encoder.conv_in.weight": torch.zeros(1),
        "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(1),
        "conditioner.embedders.0.transformer.weight": torch.zeros(1),
    }, str(path))
    return str(path)


class RegistryValuesTest(unittest.TestCase):
    """The numbers themselves, in the one place they are written down."""

    def test_registry_scaling_factors(self):
        self.assertEqual(vae_store.canonical_latent_scaling("sdxl")[0], SDXL_SCALING)
        self.assertEqual(vae_store.canonical_latent_scaling("sd15")[0], SD15_SCALING)
        # FLUX.1's VAE is the one AutoencoderKL family member with a shift.
        self.assertEqual(vae_store.canonical_latent_scaling("flux1"),
                         (0.3611, 0.1159, 16))

    def test_non_scalar_families_are_none_not_one(self):
        # AutoencoderKLFlux2 / AutoencoderKLQwenImage normalise with per-channel
        # latents_mean/latents_std; their config.json has no scaling_factor at
        # all. `None` must mean "cannot be determined", never a defaulted 1.0.
        for arch in ("flux2", "qwen_image"):
            self.assertIsNone(vae_store.canonical_latent_scaling(arch)[0], arch)

    def test_unknown_arch_is_none(self):
        self.assertIsNone(vae_store.canonical_latent_scaling("qwen"))
        self.assertIsNone(vae_store.canonical_latent_scaling(""))

    def test_fallback_constant_matches_diffusers(self):
        # If diffusers ever changes its single-file fallback, the comment in the
        # registry (and the premise of this whole guard) stops being true.
        from diffusers.loaders.single_file_utils import LDM_VAE_DEFAULT_SCALING_FACTOR
        self.assertEqual(vae_store.LDM_SINGLE_FILE_DEFAULT_SCALING_FACTOR,
                         LDM_VAE_DEFAULT_SCALING_FACTOR)
        self.assertEqual(LDM_VAE_DEFAULT_SCALING_FACTOR, SD15_SCALING)


class BareFileDetectionTest(unittest.TestCase):
    """Whether diffusers had ANY evidence is the thing that decides the repair."""

    def test_vae_only_file_is_bare(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_bare_vae_file(Path(tmp) / "vae.safetensors")
            self.assertIs(vt._is_bare_vae_single_file(p), True)

    def test_full_checkpoint_is_not_bare(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_full_checkpoint_file(Path(tmp) / "model.safetensors")
            self.assertIs(vt._is_bare_vae_single_file(p), False)

    def test_uninspectable_file_is_unknown(self):
        # .ckpt cannot be inspected without unpickling; "unknown" must not be
        # collapsed into either answer.
        self.assertIsNone(vt._is_bare_vae_single_file("model.ckpt"))
        self.assertIsNone(vt._is_bare_vae_single_file(None))


class RepairTest(unittest.TestCase):
    """The load-time repair, on a real AutoencoderKL config."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.bare = _write_bare_vae_file(self.tmp / "sdxl_vae.safetensors")
        self.full = _write_full_checkpoint_file(self.tmp / "checkpoint.safetensors")

    def test_sdxl_bare_file_is_corrected(self):
        vae = _tiny_vae(SD15_SCALING)          # what from_single_file hands back
        source = vt.repair_single_file_scaling_factor(vae, self.bare, "sdxl")
        self.assertEqual(vae.config.scaling_factor, SDXL_SCALING)
        self.assertIn("0.13025", source)

    def test_sd15_bare_file_is_left_alone(self):
        # The negative that stops the fix from being a blanket overwrite.
        vae = _tiny_vae(SD15_SCALING)
        vt.repair_single_file_scaling_factor(vae, self.bare, "sd15")
        self.assertEqual(vae.config.scaling_factor, SD15_SCALING)

    def test_full_checkpoint_is_left_alone(self):
        # from_single_file READ the family here (run 113's base checkpoint
        # loaded at 0.13025 this way). Even with a contradicting vae_arch, the
        # evidence beats the config field.
        vae = _tiny_vae(SD15_SCALING)
        source = vt.repair_single_file_scaling_factor(vae, self.full, "sdxl")
        self.assertEqual(vae.config.scaling_factor, SD15_SCALING)
        self.assertIn("full checkpoint", source)

    def test_non_scalar_arch_is_left_alone(self):
        for arch in ("flux2", "qwen_image", "typo", ""):
            vae = _tiny_vae(SD15_SCALING)
            source = vt.repair_single_file_scaling_factor(vae, self.bare, arch)
            self.assertEqual(vae.config.scaling_factor, SD15_SCALING, arch)
            self.assertIn("UNVERIFIED", source, arch)

    def test_flux1_bare_file_gets_scaling_and_shift(self):
        vae = _tiny_vae(SD15_SCALING, latent_channels=16)
        vt.repair_single_file_scaling_factor(vae, self.bare, "flux1")
        self.assertEqual(vae.config.scaling_factor, 0.3611)
        self.assertEqual(vae.config.shift_factor, 0.1159)

    def test_latent_channel_mismatch_is_refused(self):
        # vae_arch says 16-channel FLUX.1, the file is a 4-channel VAE. Stamping
        # 0.3611 onto it would be a worse export than the one being fixed.
        vae = _tiny_vae(SD15_SCALING, latent_channels=4)
        with self.assertRaises(VaeConfigError) as ctx:
            vt.repair_single_file_scaling_factor(vae, self.bare, "flux1")
        self.assertIn("vae_arch", str(ctx.exception))
        self.assertEqual(vae.config.scaling_factor, SD15_SCALING)


class _StubTrainer(vt.VaeTrainer):
    """A VaeTrainer with just enough state for load_base_vae/save_diffusers_vae.

    Built with ``__new__`` deliberately: the point is to run the REAL methods
    (the wiring is what the defect was), not a re-implementation of them.
    """

    @classmethod
    def build(cls, tmp: Path, vae_source: str, vae_path: str, vae_arch: str):
        self = cls.__new__(cls)
        cfg = dict(VAE_TRAINING_DEFAULTS)
        cfg.update({"vae_source": vae_source, "vae_path": vae_path,
                    "vae_arch": vae_arch, "train_encoder": False,
                    "acknowledge_latent_space_break": False,
                    "export_bare_ldm": False})
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.output_dir = tmp
        self.run_id = 0
        self.run_name = "unit_run"
        self.train_encoder = False
        self.ema = None
        self.vae = None
        self._base_vae_identity = {}
        return self


class LoadAndExportTest(unittest.TestCase):
    """End to end over the real methods: load -> config -> exported config.json."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.bare = _write_bare_vae_file(self.tmp / "sdxl_vae.safetensors")

        # Stand in for from_single_file WITHOUT downloading anything: return a
        # real AutoencoderKL carrying diffusers' documented fallback value, which
        # is exactly what the real call returns for a VAE-only file.
        import diffusers
        self._orig = diffusers.AutoencoderKL.from_single_file
        diffusers.AutoencoderKL.from_single_file = staticmethod(
            lambda path, **kw: _tiny_vae(SD15_SCALING))
        self.addCleanup(
            lambda: setattr(diffusers.AutoencoderKL, "from_single_file", self._orig))

    def _run(self, vae_arch: str) -> Path:
        trainer = _StubTrainer.build(self.tmp / "out", "path", self.bare, vae_arch)
        trainer.load_base_vae()
        return trainer

    def test_sdxl_load_then_export_carries_013025(self):
        trainer = self._run("sdxl")
        self.assertEqual(trainer.vae.config.scaling_factor, SDXL_SCALING)
        # The sidecar identity records the repaired value and how it was decided.
        self.assertEqual(trainer._base_vae_identity["scaling_factor"], SDXL_SCALING)
        self.assertIn("vae_arch", trainer._base_vae_identity["scaling_factor_source"])

        out_dir = trainer.save_diffusers_vae(step=1)
        written = json.load(open(out_dir / "config.json", encoding="utf-8"))
        self.assertEqual(written["scaling_factor"], SDXL_SCALING)

        sidecar = json.load(open(out_dir / "sushi_vae_training.json", encoding="utf-8"))
        self.assertEqual(sidecar["base_vae"]["scaling_factor"], SDXL_SCALING)

    def test_sd15_load_then_export_carries_018215(self):
        trainer = self._run("sd15")
        self.assertEqual(trainer.vae.config.scaling_factor, SD15_SCALING)
        out_dir = trainer.save_diffusers_vae(step=1)
        written = json.load(open(out_dir / "config.json", encoding="utf-8"))
        self.assertEqual(written["scaling_factor"], SD15_SCALING)

    def test_diffusers_directory_source_is_untouched(self):
        # The repair is single-file only: a directory has its own config.json,
        # which is authoritative and must not be second-guessed by vae_arch.
        src = self.tmp / "vae_dir"
        _tiny_vae(0.4242).save_pretrained(str(src))
        trainer = _StubTrainer.build(self.tmp / "out2", "path", str(src), "sdxl")
        trainer.load_base_vae()
        self.assertEqual(trainer.vae.config.scaling_factor, 0.4242)
        self.assertNotIn("scaling_factor_source", trainer._base_vae_identity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
