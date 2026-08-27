"""Every path that can hand the batch loop a canvas, driven through the batch loop.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/batch_loop_alignment_paths_test.py -q

Static: no model, no GPU, no DB, no network.

`_assert_item_pixel_align` is a HARD RAISE at the top of the per-item batch loop,
outside the per-mode try/except (commit 89cecca5). That is only safe if EVERY
producer of an item dict aligns it at the source; a producer that does not now
aborts a multi-day run instead of skipping a batch. `no_bucketing_epoch_alignment_test`
covers the no-bucketing still path. This file covers the four other producers:

  F1  Danbooru augmentation splices injected item dicts straight into `batches`.
      Their canvas comes from `_danbooru_injection_buckets` and from nothing else.
  F2  Priority training builds batches from the ITEM dicts, which `reload_for_epoch`
      resets to the DB's original dims -- under bucketing nothing used to put the
      bucket dims back.
  F3  Video items lose their VideoBucketManager annotation at every reload, which
      collapses the (spatial, clip_length) batch grouping key to (None, None, None).
      Exempt from the assert, so it fails SILENTLY -- the run-121 shape, one arch over.
  F4  The zero-dimension fallback assigned a raw, never-snapped `base_resolutions`
      value as a square canvas.

Plus the resolution-curriculum skip in `_prepare_epoch_items`, which every existing
test left unconstrained (`_rc_active` is hardcoded False in all of them).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.training.base_trainer import BaseTrainer, ItemDimensionError  # noqa: E402
from core.training.bucketing import BucketManager  # noqa: E402
from core.models.components.wiring import LTX2_TEMPORAL  # noqa: E402


BASE_TRAINER_SRC = open(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "core", "training", "base_trainer.py"),
    encoding="utf-8").read()


# ===========================================================================
# harness
# ===========================================================================

class _Trainer:
    """The BaseTrainer surface these producers touch. Real methods throughout --
    only the model/dataset plumbing is stubbed."""

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
    _danbooru_injection_buckets = BaseTrainer._danbooru_injection_buckets
    _annotate_video_items = BaseTrainer._annotate_video_items
    _assert_item_pixel_align = BaseTrainer._assert_item_pixel_align

    def __init__(self, pixel_align=32, temporal=None, arch_name="test"):
        self.arch = type("_Arch", (), {"pixel_align": pixel_align,
                                       "temporal": temporal,
                                       "name": arch_name})()
        self.config = {}

    def _temporal_spec(self):
        return getattr(self.arch, "temporal", None)


class _Dataset:
    """TrainRunnerDataset's contract: the first epoch reuses the pre-loaded items,
    every later epoch hands back FRESH dicts carrying the DB's ORIGINAL dims and
    only the media keys `_process_cached_items` copies through."""

    def __init__(self, rows, unique_id="ds", video=False):
        self._rows = rows
        self._video = video
        self.unique_id = unique_id
        self.items = self._fresh()
        self._reloaded = False

    def _fresh(self):
        out = []
        for row in self._rows:
            if self._video:
                out.append({"image_path": row[0], "video_path": row[0],
                            "width": row[1], "height": row[2],
                            "num_frames": row[3], "fps": row[4],
                            "item_type": "video"})
            else:
                out.append({"image_path": row[0], "width": row[1], "height": row[2],
                            "item_type": "single", "caption": row[3] if len(row) > 3 else ""})
        return out

    def reload_for_epoch(self, epoch_num, run_id=None):
        if not self._reloaded:
            self._reloaded = True
            return None
        return self._fresh()


def _drive_batch_loop(trainer, batches):
    """MIRROR of base_trainer's per-item batch-loop head. Pinned by
    `test_batch_loop_head_mirror_is_current`: these are the exact three lines a
    batch executes before anything is allowed to catch an exception."""
    seen = []
    for batch in batches:
        for item, _dataset in batch:
            width = item.get("width") or item.get("bucket_width")
            height = item.get("height") or item.get("bucket_height")
            trainer._assert_item_pixel_align(item, width, height)
            seen.append((item["image_path"], width, height))
    return seen


def test_batch_loop_head_mirror_is_current():
    """`_drive_batch_loop` is only evidence while it matches the real loop."""
    assert 'width = item.get("width") or item.get("bucket_width")' in BASE_TRAINER_SRC
    assert 'height = item.get("height") or item.get("bucket_height")' in BASE_TRAINER_SRC
    assert "self._assert_item_pixel_align(item, width, height)" in BASE_TRAINER_SRC


# ===========================================================================
# F1 -- injected Danbooru samples
# ===========================================================================

def _assign_bucket(bucket_res, w, h):
    """The collector's REAL bucket pick, over the trainer's bucket set."""
    from core.training.danbooru_image_augment import DanbooruImageCollector
    stub = type("_S", (), {"_bucket_resolutions": list(bucket_res)})()
    return DanbooruImageCollector._assign_bucket(stub, w, h)


def _injected_batch(trainer, base_resolutions, sizes):
    """Build the injected batch exactly as train()'s splice does: the collector
    assigns a bucket from `_danbooru_injection_buckets`, and the item dict carries
    that bucket as width/height (pinned below)."""
    assert '"width": _ri.bucket_w' in BASE_TRAINER_SRC
    assert '"height": _ri.bucket_h' in BASE_TRAINER_SRC
    bucket_res = trainer._danbooru_injection_buckets(base_resolutions)
    batch = []
    for i, (w, h) in enumerate(sizes):
        bw, bh = _assign_bucket(bucket_res, w, h)
        batch.append(({"image_path": f"danbooru://{i}", "caption": "",
                       "width": bw, "height": bh,
                       "_danbooru_image_bytes": b"", "_danbooru": True}, None))
    return [batch]


@pytest.mark.parametrize("align,base", [(32, 2048), (32, 1024), (16, 1024), (8, 512)])
def test_injected_augmentation_batch_survives_the_batch_loop(align, base):
    """F1. Injected items never pass through the bucket grid or the no-bucketing
    fit -- `_danbooru_injection_buckets` is their ONLY dimension owner. With the
    /8 snap it used to carry, SenseNova at base_resolutions [2048] aborted on the
    first injected batch (2360x1768, 2360 % 32 == 8), within ~4 batches of epoch 1."""
    t = _Trainer(pixel_align=align)
    # Real source aspect ratios, one per configured bucket and then some.
    sizes = [(1000, 1000), (1600, 1200), (1200, 1600), (3000, 2000),
             (2000, 3000), (1920, 1080), (1080, 1920), (777, 1013)]
    _drive_batch_loop(t, _injected_batch(t, [base], sizes))
    # Every bucket, not just the ones this batch happened to draw.
    for bw, bh in t._danbooru_injection_buckets([base]):
        assert bw % align == 0 and bh % align == 0, f"{bw}x{bh}"


def test_injection_buckets_stay_near_the_base_area():
    """The snap must not be paid for by an area collapse: the injected canvas is
    still centred on the base-resolution area (and never zero)."""
    t = _Trainer(pixel_align=32)
    for bw, bh in t._danbooru_injection_buckets([1024]):
        assert bw >= 32 and bh >= 32
        assert 0.75 <= (bw * bh) / (1024.0 * 1024.0) <= 1.0


# ===========================================================================
# F2 -- priority training under bucketing, after a reload
# ===========================================================================

# Sizes with no /32 factor of their own; the bucket grid is what makes them legal.
PRIORITY_ROWS = [
    ("p1.png", 1224, 1168, "1girl, solo, rare_tag"),
    ("p2.png", 2150, 3036, "1girl, rare_tag"),
    ("n1.png", 1023, 999, "1boy"),
    ("n2.png", 1500, 1000, "scenery"),
]


def _bucketed_setup(trainer, datasets, base_resolutions=(1024,)):
    """train()'s setup bucketing pass, condensed: assign every item and stamp the
    bucket dims onto the item dict (base_trainer.py's `item["width"] = image_info
    ["bucket_width"]`)."""
    bm = BucketManager(base_resolutions=list(base_resolutions),
                       divisibility=trainer._arch_pixel_align(),
                       strategy="resize", multi_resolution_mode="max")
    for dataset in datasets:
        for item in dataset.items:
            _key, info = bm.assign_image_to_bucket(
                image_path=item["image_path"], width=item["width"],
                height=item["height"], caption=item.get("caption", ""),
                dataset_unique_id=dataset.unique_id)
            item["width"], item["height"] = info["bucket_width"], info["bucket_height"]
    return bm


def _priority_batches(trainer, all_items, bucket_manager, batch_size=2):
    """The REAL priority classification + batch build."""
    from core.training.priority_training import (
        PriorityTrainingConfig, classify_items, build_priority_batches,
    )
    cfg = PriorityTrainingConfig.from_dict(
        {"entries": [{"tags": ["rare_tag"]}], "multiplier": 2})
    priority_items, _normal = classify_items(all_items, cfg)
    assert priority_items, "fixture no longer exercises the priority path"
    return build_priority_batches(priority_items, batch_size, bucket_manager)


def test_priority_batches_survive_an_epoch_rollover_under_bucketing():
    """F2. `classify_items` is fed the ITEM dicts, and `reload_for_epoch` resets
    them to the DB's original dims; `_prepare_epoch_items` deliberately skips the
    no-bucketing fit when a manager is live, so nothing used to put the bucket dims
    back. Epoch 1 passed (setup had stamped them); epoch 2 aborted."""
    t = _Trainer(pixel_align=32)
    datasets = [_Dataset(PRIORITY_ROWS)]
    bm = _bucketed_setup(t, datasets)

    for epoch in range(3):
        all_items = t._prepare_epoch_items(datasets, epoch, run_id=1,
                                           bucket_manager=bm, base_resolutions=[1024])
        batches = _priority_batches(t, all_items, bm)
        seen = _drive_batch_loop(t, batches)
        assert seen, f"epoch {epoch + 1}"
        for _path, w, h in seen:
            assert w % 32 == 0 and h % 32 == 0, f"epoch {epoch + 1}: {w}x{h}"


def test_restored_bucket_dims_are_the_managers_own():
    """The restore must reproduce the grid, not re-derive a second opinion: one
    alignment owner. Every item comes back at the dims the manager recorded."""
    t = _Trainer(pixel_align=32)
    datasets = [_Dataset(PRIORITY_ROWS)]
    bm = _bucketed_setup(t, datasets)
    from_setup = {i["image_path"]: (i["width"], i["height"]) for i in datasets[0].items}

    datasets[0].reload_for_epoch(0)      # burn the pre-loaded epoch
    t._prepare_epoch_items(datasets, 1, run_id=1, bucket_manager=bm,
                           base_resolutions=[1024])
    assert {i["image_path"]: (i["width"], i["height"])
            for i in datasets[0].items} == from_setup


def test_restore_does_not_poison_the_original_size_map():
    """SDXL micro-conditioning reads the original-size map. On the first epoch the
    items carry BUCKET dims, so the restore must not seed the map from them."""
    t = _Trainer(pixel_align=8)
    datasets = [_Dataset(PRIORITY_ROWS)]
    bm = _bucketed_setup(t, datasets)
    t._prepare_epoch_items(datasets, 0, run_id=1, bucket_manager=bm,
                           base_resolutions=[1024])
    for row in PRIORITY_ROWS:
        got = getattr(t, "_orig_size_map", {}).get(row[0])
        assert got in (None, (row[1], row[2])), f"{row[0]}: {got}"


def test_priority_normal_bucket_manager_mirrors_the_live_divisibility():
    """F2's other half: the normal-item manager the priority path builds was
    hardcoded to /8, an independent misalignment source for a /16 or /32 arch."""
    assert "divisibility=bucket_manager.divisibility," in BASE_TRAINER_SRC
    src = BASE_TRAINER_SRC[BASE_TRAINER_SRC.index("normal_bucket_manager = BucketManager("):]
    assert "divisibility=8," not in src[:400]


# ===========================================================================
# F3 -- video items keep their annotation across a reload
# ===========================================================================

VIDEO_ROWS = [
    ("a.webm", 1280, 720, 97, 24.0),
    ("b.webm", 720, 1280, 65, 24.0),
    ("c.webm", 1920, 1080, 49, 24.0),
]


def _video_group_keys(all_items):
    """train()'s video batch grouping key (base_trainer.py's `_vkey`)."""
    assert '_vkey = (_item.get("bucket_width"),' in BASE_TRAINER_SRC
    return [(i.get("bucket_width"), i.get("bucket_height"), i.get("clip_length"))
            for i, _ds in all_items if i.get("item_type") == "video"]


def test_video_items_keep_their_bucket_annotation_across_reloads():
    """F3. A reload carries only video_path/fps/num_frames/duration, so from epoch 2
    the grouping key was (None, None, None) for every clip: all videos collapsed into
    ONE 'uniform' group regardless of spatial size or frame count, at raw DB dims.
    Silent -- the assert exempts video."""
    t = _Trainer(pixel_align=32, temporal=LTX2_TEMPORAL, arch_name="ltx2")
    datasets = [_Dataset(VIDEO_ROWS, video=True)]
    t._annotate_video_items(datasets, [512])       # train()'s setup call
    from_setup = _video_group_keys([(i, None) for i in datasets[0].items])
    assert all(None not in k for k in from_setup)
    assert len(set(from_setup)) > 1, "fixture must span more than one group"

    for epoch in range(3):
        all_items = t._prepare_epoch_items(datasets, epoch, run_id=1,
                                           bucket_manager=None, base_resolutions=[512])
        keys = _video_group_keys(all_items)
        assert all(None not in k for k in keys), f"epoch {epoch + 1}: {keys}"
        # And the annotation does not DRIFT: re-bucketing an already-bucketed clip
        # would shrink it a little more every epoch.
        assert keys == from_setup, f"epoch {epoch + 1}"
        for item, _ds in all_items:
            assert (item["width"], item["height"]) == (item["bucket_width"],
                                                       item["bucket_height"])


def test_video_annotation_line_is_not_repeated_every_epoch(capsys):
    """It runs per epoch now, so it must not print per epoch for the length of a run."""
    t = _Trainer(pixel_align=32, temporal=LTX2_TEMPORAL, arch_name="ltx2")
    datasets = [_Dataset(VIDEO_ROWS, video=True)]
    t._annotate_video_items(datasets, [512])
    assert "video item(s) to" in capsys.readouterr().out
    for epoch in range(3):
        t._prepare_epoch_items(datasets, epoch, run_id=1, bucket_manager=None,
                               base_resolutions=[512])
    assert "video item(s) to" not in capsys.readouterr().out


# ===========================================================================
# F4 -- the zero-dimension fallback
# ===========================================================================

@pytest.mark.parametrize("base", [1080, 1000, 1024, 768])
def test_zero_dimension_items_get_an_aligned_square(base):
    """F4. `base_resolutions` is free-form user input and is never snapped anywhere.
    A 0/NULL-dimension item whose header read also fails took it verbatim as a square
    canvas: [1080] on SenseNova gave 1080x1080, 1080 % 32 == 24, run aborted."""
    t = _Trainer(pixel_align=32)
    datasets = [_Dataset([("zero.png", 0, 0)])]
    all_items = t._prepare_epoch_items(datasets, 0, run_id=1, bucket_manager=None,
                                       base_resolutions=[base])
    _drive_batch_loop(t, [all_items])
    item = all_items[0][0]
    assert item["width"] == item["height"] <= base
    assert item["width"] > base - 32


def test_zero_dimension_item_does_not_read_a_header_it_cannot_read():
    """The branch exists for items with no usable size at all; it must stay a pure
    fallback (a missing file must not turn into an exception out of the fit)."""
    t = _Trainer(pixel_align=16)
    items = [{"image_path": "/does/not/exist.png", "width": None, "height": 0,
              "item_type": "single"}]
    t._fit_items_to_base_area(items, [1080])
    assert items[0]["width"] % 16 == 0 and items[0]["height"] % 16 == 0


# ===========================================================================
# resolution curriculum -- the branch every other test leaves unconstrained
# ===========================================================================

def test_curriculum_skip_still_ends_the_epoch_aligned():
    """`_prepare_epoch_items` SKIPS the fit when the curriculum is active, because
    train() re-fits with the phase's resolution a few lines later. That handoff is
    the whole justification for the skip, and no test drove it: `_rc_active` is
    False in every other case. Drive both halves, at both phases."""
    t = _Trainer(pixel_align=32)
    t._rc_active = True
    t._rc_warmup_res, t._rc_normal_res = [512], [1024]
    datasets = [_Dataset(PRIORITY_ROWS)]

    for epoch, phase_res in ((0, [512]), (1, [512]), (2, [1024])):
        all_items = t._prepare_epoch_items(datasets, epoch, run_id=1,
                                           bucket_manager=None, base_resolutions=[1024])
        # The skip is real: straight out of a reload the dims are still the DB's.
        if epoch >= 1:
            assert [(i["width"], i["height"]) for i, _ in all_items] == \
                   [(r[1], r[2]) for r in PRIORITY_ROWS]
        t._rc_refit_items(all_items, phase_res)     # train()'s follow-up
        _drive_batch_loop(t, [all_items])
        area = max(phase_res) ** 2
        for item, _ds in all_items:
            assert item["width"] * item["height"] <= area


def test_curriculum_skip_does_not_apply_under_crop_augmentation():
    """The skip is conditioned on `crop_planner is None` (crop augmentation owns
    per-epoch re-bucketing instead). With a planner live the fit must still run."""
    t = _Trainer(pixel_align=32)
    t._rc_active = True
    t.crop_planner = object()
    datasets = [_Dataset(PRIORITY_ROWS)]
    datasets[0].reload_for_epoch(0)
    all_items = t._prepare_epoch_items(datasets, 1, run_id=1, bucket_manager=None,
                                       base_resolutions=[1024])
    _drive_batch_loop(t, [all_items])


# ===========================================================================
# skip accounting and the counted warning
# ===========================================================================

class _Reporter(_Trainer):
    _report_epoch_skips = BaseTrainer._report_epoch_skips
    _report_item_failure = BaseTrainer._report_item_failure
    _item_failure_kind = staticmethod(BaseTrainer._item_failure_kind)


def test_epoch_skip_summary_is_reachable_from_the_early_exits(capsys):
    """F5. The summary sat after the batch loop, so the target-steps-reached `return`
    and the KeyboardInterrupt handler both bypassed it: a stopped run printed no skip
    line at all. One reporter, three call sites, and it does not double-print."""
    for anchor in ("self._report_epoch_skips(epoch, _epoch_skips_before, len(batches))",
                   "Reached target steps"):
        assert anchor in BASE_TRAINER_SRC
    # epoch end + target-steps-reached + KeyboardInterrupt.
    assert BASE_TRAINER_SRC.count("self._report_epoch_skips(") == 3

    t = _Reporter()
    t._batches_skipped = 7
    assert t._report_epoch_skips(0, 0, 100) == 7
    assert "7 of 100 batch(es) were skipped" in capsys.readouterr().out
    assert t._report_epoch_skips(0, 0, 100) == 0        # interrupt after the epoch end
    assert capsys.readouterr().out == ""


def test_item_encode_failure_warning_carries_a_count(capsys):
    """F6. One countless say-once notice reads like one bad file; run 121 had 1,871.
    Re-announce at decade thresholds, with the count in the text (the downstream
    dedup key includes the message, so a new count is a new notice)."""
    t = _Reporter()
    exc = ValueError("SenseNova image height and width must be divisible by 32")

    t._report_item_failure(exc, "1.png", "Batch skipped due to")
    out = capsys.readouterr().out
    assert "item_encode_failed" in out and "1 item(s)" in out

    for i in range(2, 10):
        t._report_item_failure(exc, f"{i}.png", "Batch skipped due to")
    assert "item_encode_failed" not in capsys.readouterr().out

    t._report_item_failure(exc, "10.png", "Batch skipped due to")
    out = capsys.readouterr().out
    assert "item_encode_failed" in out and "10 item(s)" in out

    for i in range(11, 101):
        t._report_item_failure(exc, f"{i}.png", "Batch skipped due to")
    assert "100 item(s)" in capsys.readouterr().out

    # A genuine corruption never reaches this channel.
    t._report_item_failure(OSError("truncated"), "x.png", "Skipping")
    assert "item_encode_failed" not in capsys.readouterr().out
