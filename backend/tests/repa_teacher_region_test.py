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
      collide;
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


def test_pixel_cache_stays_bounded(tmp_path):
    path = _striped(tmp_path)
    item = {"image_path": path}
    t = _trainer()
    for x in range(4100):
        _teacher(t, item, (0, 0, 1 + (x % 200), 128))

    assert len(t._repa_pix_cache) <= 4096


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
    repa_module.assert_repa_region_reconstructible(
        {"bucket_strategy": strategy, "latent_encoding_mode": mode})


def test_random_crop_on_a_buffered_mode_is_refused():
    with pytest.raises(ValueError, match="random_crop"):
        repa_module.assert_repa_region_reconstructible(
            {"bucket_strategy": "random_crop", "latent_encoding_mode": "swap_onthefly"})


def test_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="reconstructs the encoded region"):
        repa_module.assert_repa_region_reconstructible(
            {"bucket_strategy": "smart_crop", "latent_encoding_mode": "onthefly_gpu"})


def test_setup_repa_refuses_before_loading_the_encoder(monkeypatch):
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

    with pytest.raises(ValueError, match="random_crop"):
        BaseTrainer._setup_repa(t)

    assert calls == [], "the encoder loaded before the refusal"
