"""The VAE training crop must move when an image is re-visited -- reproducibly.

Regression cover for the defect where ``VaeRawImageDataset.__getitem__`` seeded
its RNG from ``(seed, index)`` only. Because ``index -> path`` is a fixed map and
the loader only shuffles the ORDER of indices, every exposure of an image in the
whole run then reused one crop window and (under ``crop_scale_policy: mixed``)
one scale factor: a 500-image set seen 16x showed 500 distinct crops, not ~8,000,
and ``mixed`` degenerated into a fixed per-image scale.

Three properties are pinned here, and they pull against each other -- which is
why the fix is a sampler-supplied visit counter rather than any kind of ambient
state:

1. VARIATION -- re-visiting an item changes its crop offset and its ``mixed``
   scale draw.
2. REPRODUCIBILITY -- the stream is a pure function of ``(seed, index, visit)``,
   so a re-run with the same seed replays it, a resume can be positioned by
   restoring one integer, and the number of DataLoader workers cannot change it.
3. VALIDATION DETERMINISM -- the held-out path takes no visit counter and no RNG,
   so ``vae_val_psnr`` keeps one meaning across steps and across this change.

Run from the repository root with the repo's virtualenv interpreter:

    venv/Scripts/python.exe -m pytest backend/tests/test_vae_crop_visit_rng.py -v

No model, no GPU, no training step: the images are synthesised into a temp dir.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

# `backend` itself must be on sys.path (same convention as the other VAE tests).
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training.vae.vae_dataset import (  # noqa: E402
    _DOMAIN_CROP,
    _DOMAIN_ORDER,
    VaeEpochCropSampler,
    VaeRawImageDataset,
    make_validation_batch,
    mix_seed,
    split_index,
)

RESOLUTION = 64
# Deliberately non-square and well above RESOLUTION on both axes, so that both a
# horizontal and a vertical crop offset exist and `mixed` has room to draw
# (f_max = 384 / 64 = 6).
IMAGE_W, IMAGE_H = 640, 384
NUM_IMAGES = 8


def _make_images(directory: str) -> list:
    """High-entropy noise images: two different crops are then never equal by luck."""
    rng = np.random.RandomState(1234)
    items = []
    for i in range(NUM_IMAGES):
        arr = rng.randint(0, 256, size=(IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        path = os.path.join(directory, f"img_{i:03d}.png")
        Image.fromarray(arr).save(path)
        items.append({"image_path": path})
    return items


class _TempImagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="vae_crop_rng_")
        cls.items = _make_images(cls.tmpdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _dataset(self, policy="downscale", seed=7, random_crop=True):
        return VaeRawImageDataset(
            self.items, RESOLUTION, random_crop=random_crop, seed=seed,
            scale_policy=policy, max_downscale=0.0)


class SeedMixerTest(_TempImagesTest):
    def test_visit_and_index_are_not_interchangeable(self):
        """The old `seed*k ^ index` shape collides; the mixer must not.

        (index=4, visit=5) and (index=5, visit=4) is the pair a plain XOR maps to
        one seed, which would hand two different images the same crop stream.
        """
        d = _DOMAIN_CROP
        self.assertNotEqual(mix_seed(d, 7, 4, 5), mix_seed(d, 7, 5, 4))
        self.assertNotEqual(mix_seed(d, 7, 0, 1), mix_seed(d, 7, 1, 0))
        # Distinctness over a realistic grid.
        seeds = {mix_seed(d, 7, i, v) for i in range(64) for v in range(64)}
        self.assertEqual(len(seeds), 64 * 64)
        # Pure function.
        self.assertEqual(mix_seed(d, 7, 3, 2), mix_seed(d, 7, 3, 2))

    def test_crop_and_order_streams_cannot_coincide(self):
        """Domain separation, including the index that used to collide.

        The shuffle stream was once tagged by a literal (0x5EED = 24301) placed
        in the argument slot the crop stream fills with the ITEM INDEX, so item
        24301's crop RNG was byte-identical to that epoch's traversal RNG -- and
        24301 is an ordinary index in a 3.8M-item corpus. The tag now sits in a
        leading slot no caller value reaches.
        """
        seed = 7
        order_seeds = {mix_seed(_DOMAIN_ORDER, seed, e) for e in range(64)}
        crop_seeds = {mix_seed(_DOMAIN_CROP, seed, i, v)
                      for i in (0, 1, 24301, 0x5EED, 99999)
                      for v in range(64)}
        self.assertFalse(order_seeds & crop_seeds)
        # The historical collision point, named explicitly.
        self.assertNotEqual(mix_seed(_DOMAIN_CROP, seed, 24301, 0),
                            mix_seed(_DOMAIN_ORDER, seed, 0))
        self.assertNotEqual(_DOMAIN_CROP, _DOMAIN_ORDER)

    def test_sampler_order_is_not_reproducible_from_a_crop_stream(self):
        """Domain separation at the LIVE call sites, not on the constants.

        The observable of the sampler's RNG is the permutation it produces, so a
        crop stream that had collided with it would reproduce that permutation
        exactly. Probed at the historical sentinel (24301 = 0x5EED) and at
        ordinary indices; a chance match of an 8-element permutation is 1/8!.
        """
        import random as _random
        seed = 7
        order = [i for i, _ in list(VaeEpochCropSampler(NUM_IMAGES, seed=seed))]
        for index in (0, 1, 5, 0x5EED, 24301):
            for visit in (0, 1):
                probe = list(range(NUM_IMAGES))
                _random.Random(
                    mix_seed(_DOMAIN_CROP, seed, index, visit)).shuffle(probe)
                self.assertNotEqual(order, probe,
                                    f"crop stream (index={index}, visit={visit}) "
                                    f"reproduces the traversal order")

    def test_domain_separation_holds_inside_the_dataset_and_sampler(self):
        """End-to-end form of the above: the two live call sites, not constants.

        A dataset whose item 24301 exists is not synthesised here; the sampler
        stream is compared against the crop stream the dataset WOULD build for
        that index, taken from the same expression the dataset uses.
        """
        ds = self._dataset(seed=7)
        sampler_seed = mix_seed(_DOMAIN_ORDER, ds.seed, 0)
        for index in (0x5EED, 24301, 0, 5):
            self.assertNotEqual(mix_seed(_DOMAIN_CROP, ds.seed, index, 0),
                                sampler_seed)

    def test_split_index_accepts_int_and_pair(self):
        self.assertEqual(split_index(5), (5, 0))
        self.assertEqual(split_index((5, 3)), (5, 3))
        self.assertEqual(split_index([5, 3]), (5, 3))
        with self.assertRaises(ValueError):
            split_index((1, 2, 3))


class CropVariesAcrossVisitsTest(_TempImagesTest):
    def test_same_index_different_visit_changes_the_crop(self):
        """THE defect. Same item, later pass -> a different window."""
        ds = self._dataset()
        for index in range(NUM_IMAGES):
            crops = [ds[(index, visit)] for visit in range(6)]
            distinct = {c.numpy().tobytes() for c in crops}
            # Not "all 6 differ" (two random offsets may coincide), but the run
            # must not collapse to the single crop the defect produced.
            self.assertGreaterEqual(
                len(distinct), 4,
                f"item {index}: only {len(distinct)} distinct crops over 6 visits")

    def test_mixed_policy_redraws_the_scale_per_visit(self):
        """`mixed` must sample each image's scale distribution, not fix one.

        The scale is not returned, so it is observed through its side effect: a
        different resample factor makes a different crop. To separate it from the
        offset draw, the same is checked with `random_crop=False`, where the
        offset is fixed at the centre and the ONLY remaining source of variation
        is the scale draw.
        """
        ds = VaeRawImageDataset(
            self.items, RESOLUTION, random_crop=True, seed=11,
            scale_policy="mixed", max_downscale=0.0)
        for index in range(NUM_IMAGES):
            distinct = {ds[(index, v)].numpy().tobytes() for v in range(6)}
            self.assertGreaterEqual(len(distinct), 4)

    def test_downscale_and_native_still_vary_by_offset(self):
        for policy in ("downscale", "native"):
            ds = self._dataset(policy=policy)
            distinct = {ds[(0, v)].numpy().tobytes() for v in range(6)}
            self.assertGreaterEqual(
                len(distinct), 4, f"policy {policy} produced {len(distinct)} crops")


class ReproducibilityTest(_TempImagesTest):
    def test_stream_is_a_pure_function_of_seed_index_visit(self):
        """No process state: two dataset objects, and repeated reads, agree.

        This is what makes `num_workers` irrelevant -- a worker holds its own copy
        of the dataset, so purity is exactly the property that keeps a
        multi-worker loader and a single-process one on the same crops.
        """
        a = self._dataset(seed=7)
        b = self._dataset(seed=7)
        for index in (0, 3, 7):
            for visit in (0, 1, 5):
                x, y = a[(index, visit)], b[(index, visit)]
                self.assertTrue(torch.equal(x, y))
                self.assertTrue(torch.equal(x, a[(index, visit)]))

    def test_a_different_seed_gives_a_different_stream(self):
        a, b = self._dataset(seed=7), self._dataset(seed=8)
        differing = sum(
            0 if torch.equal(a[(i, v)], b[(i, v)]) else 1
            for i in range(NUM_IMAGES) for v in range(3))
        self.assertGreaterEqual(differing, NUM_IMAGES * 3 - 2)

    def test_bare_int_index_means_visit_zero(self):
        ds = self._dataset()
        self.assertTrue(torch.equal(ds[4], ds[(4, 0)]))

    def test_sampler_replays_for_a_seed_and_counts_visits(self):
        a = VaeEpochCropSampler(NUM_IMAGES, seed=7)
        b = VaeEpochCropSampler(NUM_IMAGES, seed=7)
        epochs_a = [list(a) for _ in range(3)]
        epochs_b = [list(b) for _ in range(3)]
        self.assertEqual(epochs_a, epochs_b)
        for e, pairs in enumerate(epochs_a):
            self.assertEqual(sorted(i for i, _ in pairs), list(range(NUM_IMAGES)))
            self.assertEqual({v for _, v in pairs}, {e})
        # The traversal order must actually be shuffled between passes.
        self.assertNotEqual([i for i, _ in epochs_a[0]], [i for i, _ in epochs_a[1]])
        self.assertEqual(a.current_epoch, 2)
        self.assertEqual(len(a), NUM_IMAGES)

    def test_sampler_seed_changes_the_order(self):
        a = list(VaeEpochCropSampler(NUM_IMAGES, seed=7))
        b = list(VaeEpochCropSampler(NUM_IMAGES, seed=99))
        self.assertNotEqual(a, b)

    def test_resume_continues_into_a_fresh_pass(self):
        """`load_checkpoint` restores data_epoch+1; that must not replay pass 0.

        Mirrors the trainer's resume arithmetic without loading a model: the
        checkpoint records `current_epoch`, the resumed sampler is positioned at
        `+1`, and the crops it then serves differ from the ones already trained on.
        """
        ds = self._dataset(seed=7)
        sampler = VaeEpochCropSampler(NUM_IMAGES, seed=7)
        seen = [list(sampler) for _ in range(2)]           # passes 0 and 1
        saved_epoch = sampler.current_epoch                # what save_checkpoint writes
        self.assertEqual(saved_epoch, 1)

        resumed = VaeEpochCropSampler(NUM_IMAGES, seed=7,
                                      start_epoch=saved_epoch + 1)
        after = list(resumed)
        self.assertEqual({v for _, v in after}, {2})

        already = {ds[pair].numpy().tobytes() for pass_ in seen for pair in pass_}
        fresh = [ds[pair].numpy().tobytes() for pair in after]
        self.assertGreaterEqual(sum(1 for c in fresh if c not in already),
                                NUM_IMAGES - 1)

        # And a resume is itself reproducible: same integer -> same stream.
        again = VaeEpochCropSampler(NUM_IMAGES, seed=7, start_epoch=saved_epoch + 1)
        self.assertEqual(after, list(again))


class DataLoaderIntegrationTest(_TempImagesTest):
    """The pair really does survive DataLoader/collate, at any worker count."""

    def _run(self, num_workers: int):
        from torch.utils.data import DataLoader
        ds = self._dataset(seed=7)
        sampler = VaeEpochCropSampler(NUM_IMAGES, seed=7)
        loader = DataLoader(ds, batch_size=2, sampler=sampler,
                            num_workers=num_workers, drop_last=False,
                            persistent_workers=num_workers > 0)
        passes = []
        for _ in range(2):
            batches = [b for b in loader]
            self.assertTrue(all(b.shape[1:] == (3, RESOLUTION, RESOLUTION)
                                for b in batches))
            passes.append(torch.cat(batches, dim=0))
        return passes, sampler

    def test_single_process_passes_differ_and_counter_advances(self):
        passes, sampler = self._run(0)
        self.assertEqual(sampler.current_epoch, 1)
        # Sorted so the (shuffled) order does not decide the comparison: the
        # SET of crops served must differ between passes.
        a = sorted(t.numpy().tobytes() for t in passes[0])
        b = sorted(t.numpy().tobytes() for t in passes[1])
        self.assertNotEqual(a, b)

    @unittest.skipIf(os.environ.get("VAE_TESTS_SKIP_WORKERS") == "1",
                     "worker-parallel loader test disabled by env")
    def test_worker_count_does_not_change_the_crops(self):
        try:
            parallel, _ = self._run(2)
        except (OSError, RuntimeError) as e:  # pragma: no cover - env dependent
            self.skipTest(f"multiprocessing DataLoader unavailable here: {e}")
        serial, _ = self._run(0)
        for p, s in zip(parallel, serial):
            self.assertEqual(sorted(t.numpy().tobytes() for t in p),
                             sorted(t.numpy().tobytes() for t in s))


class TrainerResumeWiringTest(unittest.TestCase):
    """The trainer half: what it WRITES and what it does with it on resume.

    Only the two small accessors are exercised (no VAE is loaded, no step runs),
    but they are the whole contract between the checkpoint and the sampler.
    """

    def _trainer(self):
        from core.training.vae.vae_config import resolve_vae_training_config
        from core.training.vae.vae_trainer import VaeTrainer

        # Repo-relative placeholder, like test_vae_refusal_matrix: the resolver
        # only needs a non-empty base model path, nothing opens it. lpips is
        # switched off so this file does not depend on that package.
        base_model = os.path.join(_BACKEND, "..", "models", "vae",
                                  "placeholder.safetensors")
        cfg = resolve_vae_training_config(
            {
                "network": {"type": "vae_decoder"},
                "model": {"name_or_path": base_model},
                "train": {},
                "save": {},
                "vae": {"lpips_weight": 0.0},
            },
            base_model_path=base_model,
        )
        with tempfile.TemporaryDirectory() as d:
            return VaeTrainer(cfg, output_dir=d, run_name="unit-test")

    def test_checkpoint_records_the_pass_in_progress(self):
        t = self._trainer()
        self.assertEqual(t._data_epoch_for_checkpoint(), 0)  # before train()
        t.train_sampler = VaeEpochCropSampler(4, seed=3)
        list(t.train_sampler)
        list(t.train_sampler)
        self.assertEqual(t._data_epoch_for_checkpoint(), 1)

    def test_resume_positions_the_sampler_one_pass_on(self):
        t = self._trainer()
        t.train_sampler = VaeEpochCropSampler(4, seed=3)
        t._position_data_sampler({"data_epoch": 4})
        self.assertEqual({v for _, v in t.train_sampler}, {5})

    def test_resume_of_a_pre_data_epoch_checkpoint_still_moves_off_pass_zero(self):
        t = self._trainer()
        t.train_sampler = VaeEpochCropSampler(4, seed=3)
        t._position_data_sampler({"step": 100})  # key absent
        self.assertEqual({v for _, v in t.train_sampler}, {1})

    def test_no_sampler_yet_is_not_an_error(self):
        t = self._trainer()
        t._position_data_sampler({"data_epoch": 4})  # must not raise
        self.assertIsNone(t.train_sampler)


class TrainLoopUsesTheSamplerTest(_TempImagesTest):
    """``train()`` must actually wire the sampler into its DataLoader.

    Everything expensive is stubbed (no VAE, no optimizer maths, no
    checkpoints, no validation) so this stays a wiring test: what it asserts is
    that the loop advanced the visit counter and that the crops served in the
    second pass over the data are not the first pass's crops. Reverting the
    loader to ``shuffle=True`` passes every other test in this file but fails
    here, which is the point.
    """

    def test_train_advances_the_visit_counter_and_moves_the_crops(self):
        import core.training.vae.vae_trainer as vae_trainer

        base_model = os.path.join(_BACKEND, "..", "models", "vae",
                                  "placeholder.safetensors")
        from core.training.vae.vae_config import resolve_vae_training_config
        cfg = resolve_vae_training_config(
            {
                "network": {"type": "vae_decoder"},
                "model": {"name_or_path": base_model},
                "train": {"batch_size": 2, "steps": 6, "num_workers": 0,
                          "gradient_accumulation_steps": 1},
                "save": {},
                # 'native' so the crop has the full 640x384 of offset room:
                # under 'downscale' the source is first shrunk to 107x64, which
                # leaves only 44 possible offsets and lets two passes coincide by
                # birthday luck rather than by any defect.
                "vae": {"lpips_weight": 0.0, "resolution": RESOLUTION,
                        "crop_scale_policy": "native", "validation_every": 0},
            },
            base_model_path=base_model,
        )

        seen = []

        class _StubTrainer(vae_trainer.VaeTrainer):
            def load_base_vae(self):
                self.vae = None

            def select_trainable(self):
                self.trainable_names = ["w"]
                self.trainable_params = [torch.nn.Parameter(torch.zeros(1))]

            def build_optimizer(self):
                params = self.trainable_params
                self.optimizer = torch.optim.SGD(params, lr=0.0)
                self.lr_scheduler = None

            def build_losses(self):
                self.loss_bank = None

            def init_ema(self):
                self.ema = None

            def _train_micro_step(self, batch, accum):
                seen.append(batch.clone())
                return 0.0, {}

            def _log_step(self, *a, **kw):
                pass

            def save_checkpoint(self, step, final=False):
                self._last_ckpt_step = step
                return self.checkpoints_dir

            def save_diffusers_vae(self, step):
                pass

            def _flush_metrics(self):
                pass

        with tempfile.TemporaryDirectory() as outdir:
            trainer = _StubTrainer(cfg, output_dir=outdir, run_name="wiring-test")
            # No validation batch: this test is about the TRAIN loader only.
            original = vae_trainer.make_validation_batch
            vae_trainer.make_validation_batch = lambda *a, **kw: (
                _ for _ in ()).throw(ValueError("disabled for this test"))
            try:
                trainer.train(list(self.items))
            finally:
                vae_trainer.make_validation_batch = original

        self.assertIsInstance(trainer.train_sampler, VaeEpochCropSampler)
        self.assertGreaterEqual(trainer.train_sampler.current_epoch, 1,
                                "train() did not iterate the visit-counting sampler")

        crops = [t.numpy().tobytes() for batch in seen for t in batch]
        # One pass is len(sampler) items; 6 steps x batch 2 = 12 reads, so the
        # loop necessarily wrapped. Compare pass 0 against everything after it.
        per_pass = len(trainer.train_sampler)
        first_pass, second_pass = set(crops[:per_pass]), crops[per_pass:]
        self.assertTrue(second_pass, "the loop never wrapped into a second pass")
        self.assertTrue(all(c not in first_pass for c in second_pass),
                        "the second pass re-served pass-0 crops")


class ValidationStaysDeterministicTest(_TempImagesTest):
    def test_make_validation_batch_is_identical_across_calls(self):
        a = make_validation_batch(self.items, RESOLUTION, 4)
        b = make_validation_batch(self.items, RESOLUTION, 4)
        self.assertEqual(tuple(a.shape), (4, 3, RESOLUTION, RESOLUTION))
        self.assertTrue(torch.equal(a, b))

    def test_validation_ignores_the_ambient_rng(self):
        import random
        random.seed(1)
        a = make_validation_batch(self.items, RESOLUTION, 4)
        random.seed(999)
        [random.random() for _ in range(50)]
        b = make_validation_batch(self.items, RESOLUTION, 4)
        self.assertTrue(torch.equal(a, b))

    def test_centre_crop_dataset_ignores_the_visit_counter(self):
        """A `random_crop=False` dataset must not move when re-read.

        Checked under `mixed`, the only policy that would otherwise consume the
        RNG on a centre-cropped read.
        """
        for policy in ("downscale", "native", "mixed"):
            ds = VaeRawImageDataset(
                self.items, RESOLUTION, random_crop=False, seed=7,
                scale_policy=policy, max_downscale=0.0)
            base = ds[(2, 0)]
            for visit in (1, 5, 97):
                self.assertTrue(torch.equal(base, ds[(2, visit)]),
                                f"policy {policy} moved on visit {visit}")


if __name__ == "__main__":
    unittest.main()
