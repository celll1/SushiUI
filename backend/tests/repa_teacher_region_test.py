"""REPA's teacher must encode the pixel region its item's latent encoded.

REPA aligns student tokens to teacher patches position by position, so a teacher
that squashes the WHOLE source while the latent holds a crop of it is aligning
against a different part of the picture. ``_get_repa_pixels_for_item`` used to do
exactly that: it read the file and resized straight to S x S, whatever
``bucket_strategy`` / crop augmentation had done to the latent.

What is pinned here:
  (a) bucket_strategy="resize" produces the same teacher pixels as before (the
      region IS the whole image, so no crop is taken at all);
  (b) "crop" and "random_crop" put the teacher on the latent's region, and the
      pre-change behaviour would have failed the same assertion;
  (c) the pixel LRU is keyed by region, so two epochs' crops of one path do not
      collide -- and is bounded in BYTES, because that same region key lets one
      path hold arbitrarily many entries;
  (d) repa_enable=false neither refuses nor changes encode_image's output, and
      the batch loop touches none of this.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/repa_teacher_region_test.py -v

Static: no model, no GPU, no DB, no network.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training import base_trainer  # noqa: E402
from core.training import repa as repa_module  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.crop_planner import CropSpec  # noqa: E402
from core.training.image_preprocessing import (  # noqa: E402
    flatten_to_rgb, source_region_for_strategy)

S = 16  # teacher square; small keeps the test fast, the geometry is size-agnostic

BASE_TRAINER_SRC = open(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "core", "training", "base_trainer.py"),
    encoding="utf-8").read()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _trainer(**over):
    """The BaseTrainer surface these two methods touch, with real methods bound.

    ``is_minit2i`` short-circuits encode_image's VAE staging, so the "latent" it
    returns is the [-1,1] pixel tensor of the encoded region -- which is what
    makes a region comparison possible without a VAE.
    """
    t = SimpleNamespace(
        is_minit2i=True,
        is_sensenova=False,
        arch=SimpleNamespace(vae_encode=lambda _t, tensor, **_kw: tensor),
        repa_size=S,
        repa_enable=True,
        log_prefix="[test]",
        _last_source_region=None,
    )
    t._repa_pix_verdict = lambda: BaseTrainer._repa_pix_verdict(t)
    for k, v in over.items():
        setattr(t, k, v)
    return t


def _region(trainer, item, w, h, strategy):
    trainer._repa_source_size = lambda it: BaseTrainer._repa_source_size(trainer, it)
    return BaseTrainer._repa_source_region(trainer, item, w, h, strategy)


def _teacher(trainer, item, region):
    return BaseTrainer._get_repa_pixels_for_item(trainer, item, region)


def _legacy_teacher(path):
    """``_get_repa_pixels_for_item`` before this change: whole file -> S x S."""
    img = flatten_to_rgb(Image.open(path)).resize((S, S), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous() * 2.0 - 1.0


def _striped(tmp_path, name="src.png", size=(256, 128)):
    """A source where every pixel's colour encodes its position.

    R ramps with x, G with y, B with their product -- so a coarse downsample of
    any window is a fingerprint of WHICH window it is, and two crops of the same
    file are told apart by content rather than by bookkeeping.
    """
    w, h = size
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    arr = np.stack([np.broadcast_to(xs, (h, w)), np.broadcast_to(ys, (h, w)), xs * ys],
                   axis=-1)
    path = tmp_path / name
    Image.fromarray((arr * 255.0).round().astype(np.uint8)).save(path)
    return str(path)


def _fingerprint(t, cells=8):
    """Region-identifying signature: the picture on a coarse grid, aspect removed."""
    return torch.nn.functional.interpolate(
        t.float(), size=(cells, cells), mode="area")[0].numpy()


# ---------------------------------------------------------------------------
# (a) the default preprocessing keeps its teacher pixels
# ---------------------------------------------------------------------------

def test_resize_teacher_pixels_are_bit_identical_to_the_previous_behaviour(tmp_path):
    path = _striped(tmp_path)
    item = {"image_path": path}
    t = _trainer()

    BaseTrainer.encode_image(t, Image.open(path), target_width=64, target_height=64,
                             bucket_strategy="resize")
    region = _region(t, item, 64, 64, "resize")
    assert region == (0, 0, 256, 128)

    torch.testing.assert_close(_teacher(t, item, region), _legacy_teacher(path),
                               rtol=0, atol=0)


def test_resize_teacher_is_unchanged_on_the_cache_path_too(tmp_path):
    """No encode this iteration -> the region is recomputed, and still the whole image."""
    path = _striped(tmp_path)
    t = _trainer()
    region = _region(t, {"image_path": path}, 64, 64, "resize")

    assert region == (0, 0, 256, 128)
    torch.testing.assert_close(_teacher(t, {"image_path": path}, region),
                               _legacy_teacher(path), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# (b) crop / random_crop: teacher and latent see the same region
# ---------------------------------------------------------------------------

def _assert_same_content(latent_px, teacher_px, path):
    """The two show one region, and the whole-image teacher the old code produced
    does not (so a passing assertion is evidence, not a tautology)."""
    lat, tea = _fingerprint(latent_px), _fingerprint(teacher_px)
    np.testing.assert_allclose(tea, lat, atol=0.05)
    legacy = _fingerprint(_legacy_teacher(path))
    assert np.abs(legacy - lat).max() > 0.2, (
        "the pre-change teacher would have passed this test, so it proves nothing")


def test_center_crop_teacher_follows_the_latent_region(tmp_path):
    """256x128 into a 64x64 bucket: the latent keeps the middle square only."""
    path = _striped(tmp_path)
    item = {"image_path": path}
    t = _trainer()

    latent = BaseTrainer.encode_image(t, Image.open(path), target_width=64,
                                      target_height=64, bucket_strategy="crop")
    region = _region(t, item, 64, 64, "crop")

    assert region == (64, 0, 192, 128)
    _assert_same_content(latent, _teacher(t, item, region), path)


def test_center_crop_region_is_recomputed_identically_without_a_capture(tmp_path):
    """Swap-buffer / disk-cache hit: no encode ran, so the box is recomputed."""
    path = _striped(tmp_path)
    t = _trainer()

    BaseTrainer.encode_image(t, Image.open(path), target_width=64, target_height=64,
                             bucket_strategy="crop")
    captured = _region(t, {"image_path": path}, 64, 64, "crop")
    recomputed = _region(t, {"image_path": path}, 64, 64, "crop")  # capture consumed

    assert captured == recomputed


def test_random_crop_teacher_follows_the_drawn_window(tmp_path):
    """The window is drawn inside encode_image; only the capture can report it."""
    path = _striped(tmp_path, size=(256, 256))
    item = {"image_path": path}
    seen = set()
    for _ in range(8):
        t = _trainer()
        latent = BaseTrainer.encode_image(t, Image.open(path), target_width=64,
                                          target_height=64,
                                          bucket_strategy="random_crop")
        region = _region(t, item, 64, 64, "random_crop")
        seen.add(region)
        np.testing.assert_allclose(_fingerprint(_teacher(t, item, region)),
                                   _fingerprint(latent), atol=0.05)
    assert len(seen) > 1, "windows never moved; the test is not exercising the draw"


def test_random_crop_without_a_capture_refuses_rather_than_guesses(tmp_path):
    path = _striped(tmp_path)
    t = _trainer()

    with pytest.raises(ValueError, match="no deterministic source region"):
        _region(t, {"image_path": path}, 64, 64, "random_crop")


def test_unreadable_source_skips_repa_instead_of_aborting_the_run(tmp_path):
    """A Danbooru-injected item's bytes are freed once its latent is buffered; the
    old code warned once and skipped, and a missing region must not escalate that
    into a raise."""
    t = _trainer()
    item = {"image_path": "danbooru://12345"}

    region = _region(t, item, 64, 64, "resize")

    assert region is None
    assert _teacher(t, item, region) is None


def test_crop_augment_window_reaches_the_teacher(tmp_path):
    """CropPlanner's crop_box bypasses bucket_strategy; the capture carries it."""
    path = _striped(tmp_path)
    item = {"image_path": path}
    spec = CropSpec(is_full=False, crop_box=(160, 20, 80, 80), bucket_w=64,
                    bucket_h=64, time_ids=(128, 256, 20, 160, 64, 64))
    t = _trainer()

    latent = BaseTrainer.encode_image(t, Image.open(path), target_width=spec.bucket_w,
                                      target_height=spec.bucket_h,
                                      bucket_strategy="resize",
                                      crop_box=spec.crop_box,
                                      time_ids_override=spec.time_ids)
    region = _region(t, item, spec.bucket_w, spec.bucket_h, "resize")

    assert region == (160, 20, 240, 100)
    _assert_same_content(latent, _teacher(t, item, region), path)


def test_pre_encoded_cache_reports_the_strategy_the_cache_was_written_with():
    """The disk cache is written by encode_image calls that pass no strategy."""
    assert repa_module.latent_source_strategy("pre_encoded_cache", "resize") == "crop"
    assert repa_module.latent_source_strategy("pre_encoded_cache", "random_crop") == "crop"
    assert repa_module.latent_source_strategy("swap_onthefly", "resize") == "resize"
    assert repa_module.latent_source_strategy("onthefly_gpu", "random_crop") == "random_crop"


def test_recomputed_center_crop_matches_encode_image_across_shapes(tmp_path):
    """The recomputation mirrors encode_image's arithmetic, not just one case."""
    for (w, h), (bw, bh) in [((256, 128), (64, 64)), ((100, 400), (128, 256)),
                             ((513, 97), (64, 32)), ((64, 64), (64, 64))]:
        path = _striped(tmp_path, name=f"s_{w}_{h}_{bw}_{bh}.png", size=(w, h))
        t = _trainer()
        BaseTrainer.encode_image(t, Image.open(path), target_width=bw,
                                 target_height=bh, bucket_strategy="crop")
        captured = t._last_source_region
        assert captured == source_region_for_strategy(w, h, bw, bh, "crop"), (w, h, bw, bh)


# ---------------------------------------------------------------------------
# (c) the LRU cannot serve one region's crop for another's
# ---------------------------------------------------------------------------

def test_pixel_cache_does_not_confuse_two_regions_of_one_path(tmp_path):
    path = _striped(tmp_path)
    item = {"image_path": path}
    t = _trainer()

    left = _teacher(t, item, (0, 0, 128, 128))
    right = _teacher(t, item, (128, 0, 256, 128))
    left_again = _teacher(t, item, (0, 0, 128, 128))

    assert len(t._repa_pix_cache) == 2
    torch.testing.assert_close(left_again, left, rtol=0, atol=0)
    assert np.abs(_fingerprint(left) - _fingerprint(right)).max() > 0.3


def _fill(trainer, item, n, y1=128):
    for x in range(n):
        _teacher(trainer, item, (0, 0, 1 + x, y1))


def _resident_bytes(trainer):
    return sum(v.nbytes for v in trainer._repa_pix_cache.values())


def test_pixel_cache_is_bounded_in_bytes_not_in_entries(tmp_path, monkeypatch):
    """The region key means one path can hold thousands of entries, so an entry cap
    is a budget that scales with repa_size (measured 1.72 MiB/entry at 384, 3.04 at
    512: 4096 entries was 6.9 GiB / 12.1 GiB)."""
    budget = 20 * 3 * S * S * 4
    monkeypatch.setattr(base_trainer, "_REPA_PIXEL_CACHE_BYTES", budget)
    t = _trainer()
    item = {"image_path": _striped(tmp_path)}

    _fill(t, item, 500)

    assert _resident_bytes(t) <= budget
    assert len(t._repa_pix_cache) == 20


def test_pixel_cache_budget_does_not_move_with_the_teacher_square(tmp_path,
                                                                  monkeypatch):
    """Doubling repa_size quadruples the entry, so the count must quarter -- the
    property an entry cap does not have."""
    budget = 64 * 3 * S * S * 4
    monkeypatch.setattr(base_trainer, "_REPA_PIXEL_CACHE_BYTES", budget)
    item = {"image_path": _striped(tmp_path)}

    small, big = _trainer(), _trainer(repa_size=2 * S)
    _fill(small, item, 300)
    _fill(big, item, 300)

    assert len(small._repa_pix_cache) == 64
    assert len(big._repa_pix_cache) == 16
    assert _resident_bytes(small) <= budget and _resident_bytes(big) <= budget


def test_pixel_cache_byte_count_tracks_what_is_resident(tmp_path, monkeypatch):
    """A drifting counter would under-evict forever (the leak this replaced) or
    evict everything on every insert."""
    monkeypatch.setattr(base_trainer, "_REPA_PIXEL_CACHE_BYTES", 10 * 3 * S * S * 4)
    t = _trainer()
    item = {"image_path": _striped(tmp_path)}

    _fill(t, item, 40)
    _teacher(t, item, (0, 0, 40, 128))  # a hit: must not be counted twice
    _fill(t, item, 5)

    assert t._repa_pix_cache_bytes == _resident_bytes(t)


def test_an_evicted_region_is_re_decoded_as_itself(tmp_path, monkeypatch):
    """Eviction must not resurrect the collision the region key closed: the entry
    that comes back has to be the region asked for, not the survivor beside it."""
    monkeypatch.setattr(base_trainer, "_REPA_PIXEL_CACHE_BYTES", 4 * 3 * S * S * 4)
    t = _trainer()
    item = {"image_path": _striped(tmp_path)}
    left_box, right_box = (0, 0, 128, 128), (128, 0, 256, 128)

    left = _teacher(t, item, left_box)
    _fill(t, item, 20)  # pushes both boxes out
    assert (item["image_path"], left_box) not in t._repa_pix_cache
    right = _teacher(t, item, right_box)
    left_again = _teacher(t, item, left_box)

    torch.testing.assert_close(left_again, left, rtol=0, atol=0)
    assert np.abs(_fingerprint(left_again) - _fingerprint(right)).max() > 0.3


def test_source_size_memo_keeps_working_past_its_cap(tmp_path, monkeypatch):
    """The cap used to call ``popitem(last=False)`` on a plain dict: the TypeError
    landed in _repa_source_region's except, so every path first seen past the cap
    silently lost REPA for its batch."""
    monkeypatch.setattr(base_trainer, "_REPA_SRC_SIZE_ENTRIES", 4)
    t = _trainer()
    paths = [_striped(tmp_path, name=f"m{i}.png", size=(32 + i, 16)) for i in range(12)]

    sizes = [BaseTrainer._repa_source_size(t, {"image_path": p}) for p in paths]

    assert sizes == [(32 + i, 16) for i in range(12)]
    assert len(t._repa_src_size) <= 4
    fresh = _striped(tmp_path, name="fresh.png", size=(48, 16))
    assert _region(t, {"image_path": fresh}, 8, 8, "resize") == (0, 0, 48, 16)


# ---------------------------------------------------------------------------
# (d) repa_enable=false changes nothing
# ---------------------------------------------------------------------------

def test_setup_repa_returns_before_any_check_when_disabled(monkeypatch):
    """Not even the preprocessing refusal fires: a disabled run is untouched."""
    calls = []
    monkeypatch.setattr(repa_module, "load_repa_encoder",
                        lambda *a, **k: calls.append(a) or (None, 8, 224))
    t = SimpleNamespace(config={"repa_enable": False,
                                "bucket_strategy": "random_crop",
                                "latent_encoding_mode": "swap_onthefly"})

    BaseTrainer._setup_repa(t)

    assert t.repa_enable is False
    assert calls == []
    assert not hasattr(t, "repa_projector")


def test_encode_image_output_does_not_depend_on_repa(tmp_path):
    path = _striped(tmp_path)
    off = SimpleNamespace(is_minit2i=True, is_sensenova=False,
                          arch=SimpleNamespace(vae_encode=lambda _t, x, **_kw: x))

    with_repa = BaseTrainer.encode_image(_trainer(), Image.open(path), target_width=64,
                                         target_height=64, bucket_strategy="crop")
    without = BaseTrainer.encode_image(off, Image.open(path), target_width=64,
                                       target_height=64, bucket_strategy="crop")

    torch.testing.assert_close(without, with_repa, rtol=0, atol=0)


def test_batch_loop_touches_the_region_only_under_repa_active():
    """Both new statements in the per-item loop sit under `if _repa_active:`."""
    loop = BASE_TRAINER_SRC.rindex("for item_index, (item, dataset) in enumerate(batch):")
    for stmt in ("self._last_source_region = None",
                 "repa_pixels_list.append(self._get_repa_pixels_for_item("):
        idx = BASE_TRAINER_SRC.index(stmt, loop)
        code = [ln.strip() for ln in BASE_TRAINER_SRC[loop:idx].split("\n")
                if ln.strip() and not ln.strip().startswith("#")]
        assert code[-1] == "if _repa_active:", stmt


# ---------------------------------------------------------------------------
# The setup-time refusal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy,mode", [
    ("resize", "swap_onthefly"),
    ("resize", "pre_encoded_cache"),
    ("crop", "swap_onthefly"),
    ("random_crop", "onthefly_gpu"),
    ("random_crop", "pre_encoded_cache"),  # the cache center-crops instead
])
def test_reconstructible_configurations_are_accepted(strategy, mode):
    repa_module.assert_repa_region_reconstructible(mode, strategy)


def test_random_crop_on_a_buffered_mode_is_refused():
    with pytest.raises(ValueError, match="random_crop"):
        repa_module.assert_repa_region_reconstructible("swap_onthefly", "random_crop")


def test_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="reconstructs the encoded region"):
        repa_module.assert_repa_region_reconstructible("onthefly_gpu", "smart_crop")


def test_setup_does_not_refuse_on_a_config_key_train_overrides(monkeypatch):
    from core.training.arch import ARCH_REGISTRY

    calls = []
    monkeypatch.setattr(repa_module, "load_repa_encoder",
                        lambda *a, **k: calls.append(a) or (None, 8, 224))
    t = SimpleNamespace(
        config={"repa_enable": True, "bucket_strategy": "random_crop",
                "latent_encoding_mode": "swap_onthefly",
                "repa_tagger_model_dir": "unused-because-stubbed"},
        arch=ARCH_REGISTRY["minit2i"](),
        transformer=SimpleNamespace(
            mmjit_config=SimpleNamespace(hidden_size=16, depth_double=28, patch_size=16),
            model=SimpleNamespace(net=SimpleNamespace(_repa_tap_depth=None,
                                                      _repa_tap_out=None))),
        device=torch.device("cpu"), training_dtype=torch.float32,
        model_path="", log_prefix="[test]",
        tread_config=None, block_skip_config=None, blockskip_config=None,
    )

    # train_runner hands train() bucket_strategy="resize" whatever the config
    # says, so refusing here would reject a run that aligns perfectly. The
    # refusal lives on train()'s own arguments instead.
    BaseTrainer._setup_repa(t)
    assert calls, "the encoder should have loaded: nothing here is unalignable"

    with pytest.raises(ValueError, match="random_crop"):
        repa_module.assert_repa_region_reconstructible("swap_onthefly", "random_crop")


# ---------------------------------------------------------------------------
# (e) the onthefly_gpu decode is reused rather than repeated
# ---------------------------------------------------------------------------

def _sources(tmp_path):
    """One picture written in the formats and modes the batch loop meets."""
    w, h = 96, 64
    xs = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    ys = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    rgb = Image.fromarray(np.stack(
        [np.broadcast_to(xs, (h, w)), np.broadcast_to(ys, (h, w)), xs * ys / 255.0],
        axis=-1).round().astype(np.uint8))
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(np.broadcast_to(xs, (h, w)).round().astype(np.uint8)))

    paths = []
    for name, im in (("src.png", rgb), ("src.jpg", rgb), ("src.webp", rgb),
                     ("palette.png", rgb.convert("P")), ("gray.png", rgb.convert("L")),
                     ("alpha.png", rgba), ("alpha.webp", rgba)):
        im.save(tmp_path / name)
        paths.append(str(tmp_path / name))
    return paths


@pytest.mark.parametrize("strategy", ["resize", "crop"])
def test_reusing_the_encode_decode_gives_the_same_teacher_pixels(tmp_path, strategy):
    """The decode handed over is pre-flatten_to_rgb, so the transform order holds."""
    for path in _sources(tmp_path):
        item = {"image_path": path}
        image = Image.open(path)
        image.load()
        enc = _trainer()
        BaseTrainer.encode_image(enc, image=image, target_width=32, target_height=32,
                                 bucket_strategy=strategy)
        region = enc._last_source_region

        reused = BaseTrainer._get_repa_pixels_for_item(_trainer(), item, region,
                                                       decoded_image=image)
        fresh = BaseTrainer._get_repa_pixels_for_item(_trainer(), item, region)
        assert torch.equal(reused, fresh), path


def test_encode_image_leaves_the_callers_image_untouched(tmp_path):
    """What makes the reuse legal: crop/resize rebind, they do not mutate."""
    for path in _sources(tmp_path):
        image = Image.open(path)
        image.load()
        before = (image.mode, image.size, image.tobytes())
        BaseTrainer.encode_image(_trainer(), image=image, target_width=32,
                                 target_height=32, bucket_strategy="crop")
        assert (image.mode, image.size, image.tobytes()) == before, path


# ---------------------------------------------------------------------------
# (f) the LRU stops holding host RAM it cannot serve from
# ---------------------------------------------------------------------------

def test_a_cache_that_never_hits_is_turned_off_and_freed(tmp_path, monkeypatch):
    monkeypatch.setattr(base_trainer, "_REPA_PIXEL_CACHE_BYTES", 20 * 3 * S * S * 4)
    monkeypatch.setattr(base_trainer, "_REPA_PIXEL_CACHE_PROBE_LOOKUPS", 64)
    t = _trainer()
    item = {"image_path": _striped(tmp_path)}

    _fill(t, item, 200)  # each region seen once, as an epoch shows each item once

    assert t._repa_pix_cache_off is True
    assert len(t._repa_pix_cache) == 0 and t._repa_pix_cache_bytes == 0
    box = (0, 0, 64, 128)
    torch.testing.assert_close(_teacher(t, item, box), _teacher(_trainer(), item, box),
                               rtol=0, atol=0)


def test_a_cache_that_still_hits_keeps_its_entries(tmp_path, monkeypatch):
    """Eviction alone is not the verdict: a hot set inside the budget keeps hitting."""
    monkeypatch.setattr(base_trainer, "_REPA_PIXEL_CACHE_BYTES", 20 * 3 * S * S * 4)
    monkeypatch.setattr(base_trainer, "_REPA_PIXEL_CACHE_PROBE_LOOKUPS", 64)
    t = _trainer()
    item = {"image_path": _striped(tmp_path)}

    for r in range(100):
        _fill(t, item, 15)                     # hot set: 15 of every 16 lookups
        _teacher(t, item, (0, 0, 1, 128 + r))  # a region never seen again

    assert t._repa_pix_verdict_done is True
    assert getattr(t, "_repa_pix_cache_off", False) is False
    assert len(t._repa_pix_cache) == 20


# ---------------------------------------------------------------------------
# (g) a run whose hit rate is knowable up front never holds the RAM at all
# ---------------------------------------------------------------------------

def _ds(n):
    """A dataset of n distinct paths, materialized lazily (1.5M is a real size)."""
    return SimpleNamespace(items=({"image_path": f"p{i}.png"} for i in range(n)))


def _prior(trainer, n, strategy="crop"):
    BaseTrainer._repa_pixel_cache_prior(trainer, [_ds(n)], strategy)
    return getattr(trainer, "_repa_pix_cache_off", False)


def test_the_local_dataset_sizes_land_on_the_side_they_were_measured_on():
    """The sizes fae3a13d measured, against the real 1 GiB / 384-square budget:
    404 and 654 keep the cache (~100% / ~90% of accesses hit), the rest cannot."""
    cap = base_trainer._repa_pixel_cache_entries(384)
    assert cap == 606

    assert [n for n in (2, 3, 404, 654) if _prior(_trainer(repa_size=384), n)] == []
    for n in (1416, 6779, 1548785):
        assert _prior(_trainer(repa_size=384), n) is True, n


def test_a_dataset_past_the_budget_holds_no_bytes_at_any_point(tmp_path, monkeypatch):
    """Not "is freed once measured": never allocated. The verdict path needed ~2650
    lookups (~660 steps at batch 4) of growing host RAM to reach the same answer."""
    monkeypatch.setattr(base_trainer, "_REPA_PIXEL_CACHE_BYTES", 20 * 3 * S * S * 4)
    t = _trainer()
    assert _prior(t, 41) is True  # 2x what the budget holds, plus one

    item = {"image_path": _striped(tmp_path)}
    _fill(t, item, 200)

    assert getattr(t, "_repa_pix_cache_bytes", 0) == 0
    assert len(getattr(t, "_repa_pix_cache", {})) == 0
    box = (0, 0, 64, 128)
    torch.testing.assert_close(_teacher(t, item, box), _teacher(_trainer(), item, box),
                               rtol=0, atol=0)


def test_a_box_redrawn_every_epoch_is_not_cached_at_all(tmp_path):
    """The key holds the box, so a window drawn afresh each epoch is never asked
    for twice -- 0 hits by construction, whatever the item count."""
    assert _prior(_trainer(), 8, strategy="random_crop") is True
    assert _prior(_trainer(crop_planner=object()), 8) is True


def test_a_small_dataset_still_caches_and_still_hits(tmp_path):
    """What the prior must not break: the resident case keeps serving from RAM."""
    t = _trainer()
    assert _prior(t, 404) is False

    item = {"image_path": _striped(tmp_path)}
    box = (0, 0, 64, 128)
    first = _teacher(t, item, box)
    second = _teacher(t, item, box)

    assert t._repa_pix_hits == 1 and len(t._repa_pix_cache) == 1
    assert second is first


# ---------------------------------------------------------------------------
# (h) the teacher square is resized from the bucket, not from the original again
# ---------------------------------------------------------------------------

def _encoded(trainer, path, strategy, w=64, h=64):
    image = Image.open(path)
    image.load()
    latent = BaseTrainer.encode_image(trainer, image=image, target_width=w,
                                      target_height=h, bucket_strategy=strategy)
    return latent, trainer._last_source_region, image, trainer._last_bucketed_image


@pytest.mark.parametrize("strategy", ["resize", "crop", "random_crop"])
def test_encode_image_is_bit_identical_whether_or_not_repa_reads_its_bucket(
        tmp_path, strategy):
    """The capture is a reference, taken after the last transform: the VAE's input
    cannot move. random_crop is drawn from `random`, so both arms draw the same."""
    for path in _sources(tmp_path):
        import random as _random
        state = _random.getstate()
        on, _, _, bucket = _encoded(_trainer(), path, strategy)
        _random.setstate(state)
        off = _encoded(_trainer(repa_enable=False), path, strategy)

        assert torch.equal(on, off[0]), path
        assert off[3] is None and bucket is not None


def test_the_teacher_square_comes_from_exactly_the_pixels_the_vae_encoded(tmp_path):
    """The captured image IS the array encode_image normalized -- not a re-read of
    the file, not a second downscale of it."""
    for path in _sources(tmp_path):
        for strategy in ("resize", "crop"):
            latent, _, _, bucket = _encoded(_trainer(), path, strategy)
            arr = np.array(bucket).astype(np.float32) / 255.0
            expect = torch.from_numpy((arr - 0.5) * 2.0).permute(2, 0, 1).unsqueeze(0)
            assert torch.equal(latent, expect), (path, strategy)


def test_the_bucketed_teacher_shows_the_latents_region(tmp_path):
    """256x128 into a 64x64 bucket: the middle square, as the from-file teacher
    reports it, and not the whole picture the pre-region code would have shown."""
    path = _striped(tmp_path)
    item = {"image_path": path}
    t = _trainer()
    latent, region, image, bucket = _encoded(t, path, "crop")
    assert region == (64, 0, 192, 128)

    from_bucket = BaseTrainer._get_repa_pixels_for_item(_trainer(), item, region,
                                                        bucketed_image=bucket)
    from_file = BaseTrainer._get_repa_pixels_for_item(_trainer(), item, region)

    np.testing.assert_allclose(_fingerprint(from_bucket), _fingerprint(latent),
                               atol=0.05)
    np.testing.assert_allclose(_fingerprint(from_bucket), _fingerprint(from_file),
                               atol=0.05)
    assert np.abs(_fingerprint(_legacy_teacher(path))
                  - _fingerprint(from_bucket)).max() > 0.2


def test_two_buckets_of_one_region_do_not_share_a_cache_entry(tmp_path):
    """A resolution curriculum encodes the same (path, box) at two bucket sizes, and
    the teacher square then holds different pixels -- so the bucket is in the key."""
    path = _striped(tmp_path)
    item = {"image_path": path}
    t = _trainer()
    small = _encoded(_trainer(), path, "resize", w=32, h=16)[3]
    big = _encoded(_trainer(), path, "resize", w=256, h=128)[3]
    region = (0, 0, 256, 128)

    a = BaseTrainer._get_repa_pixels_for_item(t, item, region, bucketed_image=small)
    b = BaseTrainer._get_repa_pixels_for_item(t, item, region, bucketed_image=big)

    assert len(t._repa_pix_cache) == 2 and getattr(t, "_repa_pix_hits", 0) == 0
    assert not torch.equal(a, b)
    assert BaseTrainer._get_repa_pixels_for_item(
        t, item, region, bucketed_image=small) is a
