"""The no-bucketing dimension policy must hold for the LIFETIME of a run.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/no_bucketing_epoch_alignment_test.py -q

Static: no model, no GPU, no DB.

The defect these pin (run 121, SenseNova both-branch full FT, bucketing off,
base_resolutions [2048]): the base-area fit + arch pixel-align snap ran ONCE, at
train() setup. `reload_for_epoch` rebuilds every item dict from the dataset cache
with the DB's ORIGINAL width/height (train_runner._process_cached_items), so from
the first NATURAL epoch rollover the items carried un-snapped dims again. 2,215 of
that dataset's 4,959 images are not /32, and 1,871 of epoch 3's 3,449 batches (54%)
were dropped by the arch's divisibility check -- reported as "[CORRUPTED IMAGE]",
which is why it survived a full epoch undiagnosed. Epochs 1-2 were clean only
because epoch 2 was a mid-epoch RESUME, whose reload is skipped.

Not SenseNova-specific: any arch whose `pixel_align` exceeds the dataset's native
alignment (32 SenseNova, 16 the patchified DiTs, 8 SD/SDXL) loses the same share
of its data with bucketing disabled.

  M1  `_prepare_epoch_items` re-applies the fit per epoch. Delete that call ->
      `test_alignment_survives_two_natural_epoch_rollovers` fails at epoch 2.
  M2  the fit reads the ORIGINAL size map, so it is idempotent, and its log line
      is deduped. Re-derive dims from the item -> the resolution-curriculum
      grow-back breaks; re-log unconditionally -> the idempotence test fails.
  M3  a post-alignment violation raises `ItemDimensionError` instead of being
      absorbed. Drop `_assert_item_pixel_align` -> the violation test fails.
  M4  a non-IO failure is not labelled corruption. Return "corrupt"
      unconditionally from `_item_failure_kind` -> the labelling test fails.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.training.base_trainer import BaseTrainer, ItemDimensionError  # noqa: E402


# ===========================================================================
# helpers
# ===========================================================================

# 1224x1168 and 2150x3036 are real members of run 121's dataset; neither is /32.
ORIGINALS = [
    ("a.png", 1224, 1168),
    ("b.png", 2150, 3036),
    ("c.png", 2048, 2048),   # already conforming -- must come out untouched
    ("d.png", 1023, 999),
]


class _Trainer:
    """The BaseTrainer surface the dimension policy actually touches."""

    log_prefix = "[T]"
    is_acestep = False
    _rc_active = False
    crop_planner = None

    _arch_pixel_align = BaseTrainer._arch_pixel_align
    _seed_orig_size_from_db = BaseTrainer._seed_orig_size_from_db
    _get_original_size_for_item = BaseTrainer._get_original_size_for_item
    _fit_items_to_base_area = BaseTrainer._fit_items_to_base_area
    _rc_refit_items = BaseTrainer._rc_refit_items
    _prepare_epoch_items = BaseTrainer._prepare_epoch_items
    _restore_bucket_dims = BaseTrainer._restore_bucket_dims
    _sync_bucket_captions = BaseTrainer._sync_bucket_captions
    _assert_item_pixel_align = BaseTrainer._assert_item_pixel_align
    _item_failure_kind = staticmethod(BaseTrainer._item_failure_kind)
    _report_item_failure = BaseTrainer._report_item_failure

    def __init__(self, pixel_align=32):
        self.arch = type("_Arch", (), {"pixel_align": pixel_align})()

    def _temporal_spec(self):
        return None


class _Dataset:
    """Stands in for TrainRunnerDataset: the first epoch reuses the pre-loaded
    items, every later epoch hands back FRESH dicts built from the DB dims --
    exactly what `_process_cached_items` does."""

    def __init__(self, unique_id="ds"):
        self.unique_id = unique_id
        self.items = self._fresh()
        self._reloaded = False

    @staticmethod
    def _fresh():
        return [{"image_path": p, "width": w, "height": h, "item_type": "single"}
                for p, w, h in ORIGINALS]

    def reload_for_epoch(self, epoch_num, run_id=None):
        if not self._reloaded:
            self._reloaded = True
            return None
        return self._fresh()


def _violations(items, align):
    return [(i["image_path"], i["width"], i["height"]) for i in items
            if i["width"] % align or i["height"] % align]


# ===========================================================================
# M1 -- the alignment must survive epoch rollovers
# ===========================================================================

def test_alignment_survives_two_natural_epoch_rollovers():
    """M1. Drives the setup fit and then THREE epochs' item materialization
    (epoch 1 = pre-loaded, epochs 2 and 3 = real reloads, the transition run 121
    died on) and requires the arch invariant at every one of them.

    Two rollovers, not one: epoch 2 in run 121 was a resume and reused the
    snapped items, so a single-transition test would have passed against the
    broken code exactly as the run's own epochs 1-2 did."""
    t = _Trainer(pixel_align=32)
    datasets = [_Dataset()]

    # train()'s setup-time fit.
    t._fit_items_to_base_area(
        (item for ds in datasets for item in ds.items), [2048])
    assert _violations(datasets[0].items, 32) == []

    seen = []
    for epoch in range(3):
        all_items = t._prepare_epoch_items(datasets, epoch, run_id=1,
                                           bucket_manager=None, base_resolutions=[2048])
        items = [i for i, _ in all_items]
        assert len(items) == len(ORIGINALS)
        assert _violations(items, 32) == [], f"epoch {epoch + 1}"
        seen.append([(i["width"], i["height"]) for i in items])

    # Same dims every epoch: the reload must not shift what the run trains on.
    assert seen[0] == seen[1] == seen[2]
    # And a batch built from them is accepted by the invariant guard.
    for item in (i for i, _ in all_items):
        t._assert_item_pixel_align(item, item["width"], item["height"])


def test_bucketed_runs_are_untouched_by_the_per_epoch_fit():
    """M1's blast radius. The base-area fit must not run when a bucket manager is
    live, or it would fight the grid: under bucketing the manager is the dimension
    owner, and `_prepare_epoch_items` re-stamps ITS assignment onto the reloaded item
    dicts (which is what the priority path reads -- see
    batch_loop_alignment_paths_test)."""
    from core.training.bucketing import BucketManager
    t = _Trainer(pixel_align=32)
    datasets = [_Dataset()]
    bm = BucketManager(base_resolutions=[2048], divisibility=32,
                       strategy="resize", multi_resolution_mode="max")
    for item in datasets[0].items:
        _k, info = bm.assign_image_to_bucket(image_path=item["image_path"],
                                             width=item["width"], height=item["height"])
        item["width"], item["height"] = info["bucket_width"], info["bucket_height"]
    from_grid = [(i["width"], i["height"]) for i in datasets[0].items]
    assert from_grid != [(w, h) for _, w, h in ORIGINALS]

    datasets[0].reload_for_epoch(0)           # burn the pre-loaded epoch
    t._prepare_epoch_items(datasets, 1, run_id=1,
                           bucket_manager=bm, base_resolutions=[2048])
    assert [(i["width"], i["height"]) for i in datasets[0].items] == from_grid


# ===========================================================================
# M1/M3 -- snapped, not skipped
# ===========================================================================

def test_non_aligned_item_is_snapped_rather_than_skipped():
    """A 1224x1168 image is not corrupt and must not be dropped: it is trained at
    the largest conforming canvas inside its own size. Under the pre-fix
    behaviour this item reached vae_encode at 1224x1168 and its whole batch was
    discarded."""
    t = _Trainer(pixel_align=32)
    items = [{"image_path": "a.png", "width": 1224, "height": 1168, "item_type": "single"}]
    t._fit_items_to_base_area(items, [2048])
    assert (items[0]["width"], items[0]["height"]) == (1216, 1152)
    t._assert_item_pixel_align(items[0], items[0]["width"], items[0]["height"])

    # Over-area items are scaled AND aligned, not just aligned.
    over = [{"image_path": "b.png", "width": 2150, "height": 3036, "item_type": "single"}]
    t._fit_items_to_base_area(over, [2048])
    assert over[0]["width"] % 32 == 0 and over[0]["height"] % 32 == 0
    assert over[0]["width"] * over[0]["height"] <= 2048 * 2048


@pytest.mark.parametrize("align", [8, 16, 32])
def test_every_architectures_alignment_is_honoured(align):
    """The defect scales with `pixel_align`, so the fit is checked at all three
    shipped values: 32 (SenseNova, LTX-2.3, MiniMax-H3), 16 (the patchified DiTs
    -- anima/lens/krea2/flux2/zimage/minit2i/ideogram4), 8 (SD1.5/SDXL)."""
    t = _Trainer(pixel_align=align)
    items = _Dataset._fresh()
    t._fit_items_to_base_area(items, [1024])
    assert _violations(items, align) == []


def test_video_and_audio_items_keep_their_own_dims():
    """Video items carry VideoBucketManager dims and audio items have no canvas at
    all; the still fit must skip both, and so must the invariant guard."""
    class _VideoTrainer(_Trainer):
        def _temporal_spec(self):
            return object()

    t = _VideoTrainer(pixel_align=32)
    t.is_acestep = True
    items = [
        {"image_path": "v.webm", "width": 704, "height": 480, "item_type": "video"},
        {"image_path": "a.flac", "item_type": "audio"},
    ]
    t._fit_items_to_base_area(items, [512])
    assert (items[0]["width"], items[0]["height"]) == (704, 480)
    assert "width" not in items[1]
    for item in items:
        t._assert_item_pixel_align(item, item.get("width"), item.get("height"))


# ===========================================================================
# M2 -- idempotence and log quiet
# ===========================================================================

def test_fit_is_idempotent_and_does_not_relog(capsys):
    """M2. The fit now runs once per epoch, so re-running it must change nothing
    and must not repeat its line every epoch for the length of the run."""
    t = _Trainer(pixel_align=32)
    items = _Dataset._fresh()

    first = t._fit_items_to_base_area(items, [2048])
    assert first > 0                       # b.png is over area, so there is a line
    out1 = capsys.readouterr().out
    assert "Bucketing disabled: fitted" in out1

    before = [(i["width"], i["height"]) for i in items]
    for _ in range(3):
        assert t._fit_items_to_base_area(items, [2048]) == first
    assert [(i["width"], i["height"]) for i in items] == before
    assert "Bucketing disabled" not in capsys.readouterr().out

    # A resolution-curriculum phase change is new information and does speak up.
    t._fit_items_to_base_area(items, [1024])
    assert "Bucketing disabled: fitted" in capsys.readouterr().out


# ===========================================================================
# M3 -- a violation is fatal, not a silently skipped batch
# ===========================================================================

def test_post_alignment_violation_raises():
    """M3. Reaching the encoder with a non-conforming canvas means the dimension
    policy did not run for that item. That is a trainer bug and stops the run,
    rather than costing 54% of an epoch with nothing on the run record."""
    t = _Trainer(pixel_align=32)
    item = {"image_path": "a.png", "width": 1224, "height": 1168, "item_type": "single"}
    with pytest.raises(ItemDimensionError) as e:
        t._assert_item_pixel_align(item, 1224, 1168)
    assert "multiple of 32" in str(e.value)
    # SD/SDXL's 8 accepts what 32 refuses -- the guard is per-arch, not global.
    _Trainer(pixel_align=8)._assert_item_pixel_align(item, 1224, 1168)


# ===========================================================================
# M4 -- a dimension failure is not corruption
# ===========================================================================

def test_dimension_failure_is_not_labelled_corruption(capsys):
    """M4. The exact exception run 121 produced, 1,871 times, under the exact
    label that hid it."""
    t = _Trainer(pixel_align=32)
    exc = ValueError("SenseNova image height and width must be divisible by 32")
    assert t._item_failure_kind(exc) == "invalid"
    assert t._report_item_failure(exc, "a.png", "Batch skipped due to") == "ITEM ENCODE FAILED"
    out = capsys.readouterr().out
    assert "CORRUPTED IMAGE" not in out
    assert "ITEM ENCODE FAILED" in out and "a.png" in out
    # The non-corruption case is also raised onto the run's warnings channel --
    # once per run, not once per item (run 121 would have emitted 1,871).
    assert "item_encode_failed" in out
    t._report_item_failure(exc, "b.png", "Batch skipped due to")
    assert "item_encode_failed" not in capsys.readouterr().out


def test_genuinely_unreadable_files_are_still_called_corruption(capsys):
    """The other half of M4: PIL's decode/IO failures keep their label and keep
    being skipped."""
    from PIL import UnidentifiedImageError
    t = _Trainer(pixel_align=32)
    for exc in (OSError("image file is truncated (12 bytes not processed)"),
                FileNotFoundError("no such file"),
                UnidentifiedImageError("cannot identify image file")):
        assert t._item_failure_kind(exc) == "corrupt"
        assert t._report_item_failure(exc, "a.png", "Skipping") == "CORRUPTED IMAGE"
    assert "ITEM ENCODE FAILED" not in capsys.readouterr().out


# ===========================================================================
# M5 -- skipped batches are counted and surfaced
# ===========================================================================

def test_skipped_batches_are_a_registered_metric():
    """A skip leaves a HOLE in training_metrics -- that hole is how run 121 was
    found, and it is not a signal anyone can be expected to read. The cumulative
    count rides the next completed step through the existing extra-metrics
    channel (no DB column)."""
    from core.training.metric_registry import EXTRA_METRIC_DEFS
    d = EXTRA_METRIC_DEFS["batches_skipped"]
    assert d["axis"] == "right"            # a count, not a loss scale
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "core", "training", "base_trainer.py"),
               encoding="utf-8").read()
    assert 'self.log_extra_metric("batches_skipped"' in src
    # Every abandon-the-batch `continue` still bumps the counter the metric reads.
    assert src.count("self._batches_skipped += 1") == 4
