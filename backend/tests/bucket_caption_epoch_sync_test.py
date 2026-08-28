"""Bucketed runs must train on the CURRENT epoch's reprocessed caption, not the
one frozen into the bucket at setup time.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/bucket_caption_epoch_sync_test.py -q

Static: no model, no GPU, no DB.

The defect: `BucketManager.assign_image_to_bucket` copies `caption` into its own
`image_info` dict at setup. `reload_for_epoch` (dropout / token shuffle) replaces
`dataset.items` with fresh dicts every epoch, but batches under bucketing are built
from `bucket_manager.buckets` (see base_trainer.py's `build_batch_indices` call
site), not from `dataset.items` -- so epoch 2+ trained on epoch 0's caption. The
fix, `BaseTrainer._sync_bucket_captions`, pushes the reloaded caption into the
existing bucket entry in place; bucket assignment (which bucket, width/height)
must not move.

  M1  captions in the bucket must equal the reloaded dataset's captions after
      `_prepare_epoch_items`, at epoch 2 and epoch 3 (not just epoch 1).
  M2  the bucket KEY and every entry's (bucket_width, bucket_height) must be
      byte-identical across epochs -- this is a caption push, not a re-bucket.
  M3  two datasets sharing an image_path must not cross-contaminate captions.
  M4  a bucket entry whose item the reload no longer has (or a dataset item never
      bucketed) must not raise and must leave the untouched side alone.
  M5  bucketing-disabled runs (bucket_manager=None) are not touched by this code
      path at all.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.bucketing import BucketManager  # noqa: E402


class _Trainer:
    """The BaseTrainer surface `_prepare_epoch_items` touches under bucketing."""

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

    def __init__(self, pixel_align=32):
        self.arch = type("_Arch", (), {"pixel_align": pixel_align})()

    def _temporal_spec(self):
        return None


class _Dataset:
    """Epoch 0 reuses the pre-loaded items; every later epoch hands back fresh
    dicts with a NEW caption -- exactly what `_process_cached_items` does for
    dropout / token shuffle."""

    def __init__(self, unique_id, items):
        self.unique_id = unique_id
        self._base_items = items
        self.items = [dict(it) for it in items]
        self._reloaded = False
        self._epoch = 0

    def reload_for_epoch(self, epoch_num, run_id=None):
        if not self._reloaded:
            self._reloaded = True
            return None
        self._epoch = epoch_num
        return [
            {**it, "caption": f"{it['caption']} epoch{epoch_num}"}
            for it in self._base_items
        ]


def _make_bucket_manager(datasets, base_resolutions=(1024,)):
    bm = BucketManager(base_resolutions=list(base_resolutions), divisibility=32,
                        strategy="resize", multi_resolution_mode="max")
    for ds in datasets:
        for item in ds.items:
            _key, info = bm.assign_image_to_bucket(
                image_path=item["image_path"],
                width=item.get("width", 1024),
                height=item.get("height", 1024),
                caption=item.get("caption", ""),
                dataset_unique_id=ds.unique_id,
            )
            item["width"] = info["bucket_width"]
            item["height"] = info["bucket_height"]
    return bm


def _flat_bucket_entries(bm):
    return [info for infos in bm.buckets.values() for info in infos]


# ===========================================================================
# M1 -- the caption in the bucket tracks the reload
# ===========================================================================

def test_bucket_caption_reflects_epoch_reload():
    ds = _Dataset("ds1", [
        {"image_path": "a.png", "width": 1024, "height": 1024, "caption": "1girl"},
        {"image_path": "b.png", "width": 1024, "height": 1024, "caption": "1boy"},
    ])
    t = _Trainer()
    bm = _make_bucket_manager([ds])

    for epoch in range(3):
        t._prepare_epoch_items([ds], epoch, run_id=1, bucket_manager=bm, base_resolutions=[1024])
        entries = {info["image_path"]: info["caption"] for info in _flat_bucket_entries(bm)}
        if epoch == 0:
            assert entries == {"a.png": "1girl", "b.png": "1boy"}
        else:
            assert entries == {
                "a.png": f"1girl epoch{epoch}",
                "b.png": f"1boy epoch{epoch}",
            }, f"epoch {epoch}"

    # And a batch actually built from the manager carries the fresh caption.
    batch = bm.build_batch_indices(batch_size=2)[0]
    captions = {it["caption"] for it in batch}
    assert captions == {"1girl epoch2", "1boy epoch2"}


# ===========================================================================
# M2 -- bucket assignment itself must not move
# ===========================================================================

def test_bucket_assignment_unchanged_across_epochs():
    ds = _Dataset("ds1", [
        {"image_path": "a.png", "width": 1024, "height": 1024, "caption": "1girl"},
        {"image_path": "b.png", "width": 512, "height": 1536, "caption": "1boy"},
    ])
    t = _Trainer()
    bm = _make_bucket_manager([ds])

    def snapshot():
        return {
            k: sorted((info["image_path"], info["bucket_width"], info["bucket_height"])
                      for info in v)
            for k, v in bm.buckets.items()
        }

    before_keys = set(bm.buckets.keys())
    before = snapshot()

    for epoch in range(3):
        t._prepare_epoch_items([ds], epoch, run_id=1, bucket_manager=bm, base_resolutions=[1024])
        assert set(bm.buckets.keys()) == before_keys
        after = snapshot()
        # Only captions may have changed; bucket membership/dims must be identical.
        assert {k: [(p, w, h) for p, w, h in v] for k, v in after.items()} == \
               {k: [(p, w, h) for p, w, h in v] for k, v in before.items()}


# ===========================================================================
# M3 -- same path across two datasets does not cross-contaminate
# ===========================================================================

def test_same_path_different_datasets_do_not_cross_contaminate():
    ds1 = _Dataset("ds1", [{"image_path": "shared.png", "width": 1024, "height": 1024, "caption": "from ds1"}])
    ds2 = _Dataset("ds2", [{"image_path": "shared.png", "width": 1024, "height": 1024, "caption": "from ds2"}])
    t = _Trainer()
    bm = _make_bucket_manager([ds1, ds2])

    t._prepare_epoch_items([ds1, ds2], 0, run_id=1, bucket_manager=bm, base_resolutions=[1024])
    t._prepare_epoch_items([ds1, ds2], 1, run_id=1, bucket_manager=bm, base_resolutions=[1024])

    by_ds = {}
    for info in _flat_bucket_entries(bm):
        by_ds[info["dataset_unique_id"]] = info["caption"]
    assert by_ds == {"ds1": "from ds1 epoch1", "ds2": "from ds2 epoch1"}


# ===========================================================================
# M4 -- missing on either side must not raise / must not touch the other
# ===========================================================================

def test_missing_item_does_not_raise_or_leak():
    base_items = [
        {"image_path": "a.png", "width": 1024, "height": 1024, "caption": "kept"},
        {"image_path": "b.png", "width": 1024, "height": 1024, "caption": "dropped-by-reload"},
    ]

    # A bucket entry the dataset no longer has after reload (simulate item removal).
    class _DroppingDataset(_Dataset):
        def reload_for_epoch(self, epoch_num, run_id=None):
            fresh = super().reload_for_epoch(epoch_num, run_id)
            if fresh is None:
                return None
            return [it for it in fresh if it["image_path"] != "b.png"]

    t = _Trainer()
    ds = _DroppingDataset("ds1", base_items)
    bm = _make_bucket_manager([ds])

    # Should not raise even though epoch 1's reload drops b.png entirely.
    t._prepare_epoch_items([ds], 0, run_id=1, bucket_manager=bm, base_resolutions=[1024])
    t._prepare_epoch_items([ds], 1, run_id=1, bucket_manager=bm, base_resolutions=[1024])

    entries = {info["image_path"]: info["caption"] for info in _flat_bucket_entries(bm)}
    # a.png synced to the new caption; b.png's stale bucket entry is left alone
    # (not deleted, not KeyError'd -- assignment stays put, per the contract).
    assert entries["a.png"] == "kept epoch1"
    assert entries["b.png"] == "dropped-by-reload"


# ===========================================================================
# M5 -- no-bucketing path is untouched
# ===========================================================================

def test_no_bucketing_path_untouched():
    ds = _Dataset("ds1", [{"image_path": "a.png", "width": 1024, "height": 1024, "caption": "1girl"}])
    t = _Trainer()
    for epoch in range(2):
        all_items = t._prepare_epoch_items([ds], epoch, run_id=1, bucket_manager=None,
                                            base_resolutions=[1024])
    captions = [it["caption"] for it, _ in all_items]
    assert captions == ["1girl epoch1"]
