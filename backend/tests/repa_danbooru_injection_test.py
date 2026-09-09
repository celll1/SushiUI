"""A Danbooru-injected batch must get REPA, like every other batch.

Injected items carry their pixels in ``_danbooru_image_bytes`` and a fake
``danbooru://<post_id>`` image_path. Under ``onthefly_gpu`` the batch loop
decoded those bytes, encoded the latent, freed the bytes -- and then let
``_get_repa_pixels_for_item`` try to open ``danbooru://<post_id>``, which fails.
REPA was skipped for the whole batch, once per run behind a single warning.

Handing that decode over (as ``onthefly_gpu`` already does for file-backed
items, c6292216) supplies the teacher without keeping the bytes alive. This is a
LOSS CHANGE on runs that inject: those batches now carry a REPA term.

  (a) the injected item gets teacher pixels now and did not at c6292216;
  (b) they are its own region, through the same transforms in the same order;
  (c) REPA off is bit-identical to c6292216, and a run without injection keeps
      that latent bit for bit (its teacher square now comes from the bucket);
  (d) the bytes are still freed at the same point, and the region never opens
      the un-openable path.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/repa_danbooru_injection_test.py -v

Static: no model, no GPU, no DB, no network.
"""

import os
import random
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.base_trainer import BaseTrainer  # noqa: E402

S = 16  # teacher square; the geometry is size-agnostic

REPO = Path(__file__).resolve().parents[2]
PRE_FIX_COMMIT = "c6292216"  # the batch loop as it was before this fix

BASE_TRAINER_SRC = (REPO / "backend" / "core" / "training" / "base_trainer.py").read_text(
    encoding="utf-8")
PRE_FIX_SRC = subprocess.run(
    ["git", "show", f"{PRE_FIX_COMMIT}:backend/core/training/base_trainer.py"],
    cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True).stdout


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _trainer(**over):
    """The BaseTrainer surface the onthefly_gpu block touches, real methods bound.

    ``is_minit2i`` short-circuits encode_image's VAE staging, so its "latent" is
    the [-1,1] pixel tensor of the encoded region -- which is what makes a
    region comparison possible without a VAE.
    """
    t = SimpleNamespace(
        is_minit2i=True,
        is_sensenova=False,
        arch=SimpleNamespace(vae_encode=lambda _t, tensor, **_kw: tensor),
        repa_size=S,
        repa_enable=True,
        log_prefix="[test]",
        _last_source_region=None,
        src_size_calls=[],
    )
    t._repa_pix_verdict = lambda: BaseTrainer._repa_pix_verdict(t)

    def _src_size(item):
        t.src_size_calls.append(item.get("image_path"))
        return BaseTrainer._repa_source_size(t, item)

    t._repa_source_size = _src_size
    t._temporal_spec = lambda: None
    for k, v in over.items():
        setattr(t, k, v)
    return t


def _striped(size=(256, 128)):
    """A picture whose every pixel encodes its position, so a coarse downsample
    of a window is a fingerprint of WHICH window it is."""
    w, h = size
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    arr = np.stack([np.broadcast_to(xs, (h, w)), np.broadcast_to(ys, (h, w)), xs * ys],
                   axis=-1)
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def _png_bytes(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _injected_item(post_id=12345, image=None, **over):
    """The item dict train()'s splice builds (base_trainer.py ~15630)."""
    item = {"image_path": f"danbooru://{post_id}", "caption": "",
            "width": 64, "height": 64, "_danbooru": True,
            "_danbooru_image_bytes": _png_bytes(image if image is not None else _striped())}
    item.update(over)
    return item


def _fingerprint(t, cells=8):
    return torch.nn.functional.interpolate(
        t.float(), size=(cells, cells), mode="area")[0].numpy()


# ---------------------------------------------------------------------------
# The batch-loop mirror, pinned to both sources
# ---------------------------------------------------------------------------

# The three lines that decide whether the decode reaches REPA. `current` is this
# fix; `prefix` is c6292216, where an injected item took the `elif` and left
# _repa_decoded_image None.
_HANDOVER = {
    "current": ('                                    if _danb_b is not None:\n'
                '                                        item["_danbooru_image_bytes"] = None\n'
                '                                    if _repa_active:\n'),
    "prefix": ('                                    if _danb_b is not None:\n'
               '                                        item["_danbooru_image_bytes"] = None\n'
               '                                    elif _repa_active:\n'),
}


#: The teacher call the mirror ends on. `current` also hands over the encode's
#: bucketed output, which the teacher square is resized from.
_TEACHER_CALL = {
    "current": ("                            repa_pixels_list.append(self._get_repa_pixels_for_item(\n"
                "                                item,\n"
                "                                self._repa_source_region(item, width, height,\n"
                "                                                         _repa_latent_strategy),\n"
                "                                decoded_image=_repa_decoded_image,\n"
                "                                bucketed_image=_repa_bucketed_image,\n"
                "                            ))\n"),
    "prefix": ("                            repa_pixels_list.append(self._get_repa_pixels_for_item(\n"
               "                                item,\n"
               "                                self._repa_source_region(item, width, height,\n"
               "                                                         _repa_latent_strategy),\n"
               "                                decoded_image=_repa_decoded_image,\n"
               "                            ))\n"),
}


def test_the_mirror_matches_both_sources():
    """`_drive_item` is only evidence while it matches the real loop."""
    for src, key in ((BASE_TRAINER_SRC, "current"), (PRE_FIX_SRC, "prefix")):
        assert src.count(_HANDOVER[key]) == 1
        assert src.count("                                        _repa_decoded_image = image\n") == 1
        assert "image = Image.open(BytesIO(_danb_b))" in src
        assert _TEACHER_CALL[key] in src
    # ... and while the two differ only in that guard.
    assert _HANDOVER["current"] not in PRE_FIX_SRC
    assert _HANDOVER["prefix"] not in BASE_TRAINER_SRC


def _drive_item(trainer, item, strategy="resize", *, repa_active=True,
                handover="current"):
    """MIRROR of base_trainer's onthefly_gpu per-item block. Returns
    (latent, teacher_pixels)."""
    width, height = item["width"], item["height"]
    _repa_decoded_image = None
    _repa_bucketed_image = None
    if repa_active:
        trainer._last_source_region = None
        trainer._last_bucketed_image = None

    if trainer._temporal_spec() is not None and item.get("item_type") == "video":
        latent = trainer._encode_video_clip(item)
    else:
        _danb_b = item.get("_danbooru_image_bytes")
        if _danb_b is not None:
            image = Image.open(BytesIO(_danb_b))
        else:
            image = Image.open(item["image_path"])
        image.load()
        _spec = item.get("_crop_spec")
        latent = BaseTrainer.encode_image(
            trainer, image=image, target_width=width, target_height=height,
            bucket_strategy=strategy,
            crop_box=_spec.crop_box if _spec is not None else None,
            time_ids_override=_spec.time_ids if _spec is not None else None,
        )
        if _danb_b is not None:
            item["_danbooru_image_bytes"] = None
        if handover == "current":
            if repa_active:
                _repa_decoded_image = image
                _repa_bucketed_image = trainer._last_bucketed_image
                trainer._last_bucketed_image = None
        elif _danb_b is None and repa_active:
            _repa_decoded_image = image

    teacher = None
    if repa_active:
        teacher = BaseTrainer._get_repa_pixels_for_item(
            trainer, item,
            BaseTrainer._repa_source_region(trainer, item, width, height, strategy),
            decoded_image=_repa_decoded_image,
            bucketed_image=_repa_bucketed_image,
        )
    return latent, teacher


# ---------------------------------------------------------------------------
# (a) the injected batch gets REPA now, and did not before
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["resize", "crop", "random_crop"])
def test_injected_item_gets_teacher_pixels(strategy):
    _, teacher = _drive_item(_trainer(), _injected_item(), strategy)
    assert teacher is not None and teacher.shape == (1, 3, S, S)


@pytest.mark.parametrize("strategy", ["resize", "crop", "random_crop"])
def test_injected_item_got_none_at_the_pinned_baseline(strategy, capsys):
    """The defect, executable: c6292216's guard leaves the teacher nothing to read."""
    _, teacher = _drive_item(_trainer(), _injected_item(), strategy, handover="prefix")
    assert teacher is None
    assert "clean-image load failed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# (b) same pixels, same transforms, same order as a file-backed item
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["resize", "crop", "random_crop"])
def test_teacher_pixels_are_the_items_own_region(strategy):
    """The latent is the encoded region (is_minit2i), so the two must agree on
    WHICH part of the picture they hold."""
    latent, teacher = _drive_item(_trainer(), _injected_item(), strategy)
    np.testing.assert_allclose(_fingerprint(teacher), _fingerprint(latent), atol=0.05)


@pytest.mark.parametrize("strategy", ["resize", "crop"])
def test_handover_equals_decoding_the_bytes_again(strategy, tmp_path):
    """The teacher square is the encode's own bucketed output, so the handover has
    to give what a second encode of the same bytes gives -- in every mode and
    container an injected download can arrive in."""
    for mode, fmt in (("RGB", "PNG"), ("RGB", "JPEG"), ("RGB", "WEBP"),
                      ("P", "PNG"), ("L", "PNG"), ("RGBA", "PNG"), ("RGBA", "WEBP")):
        buf = BytesIO()
        _striped().convert(mode).save(buf, format=fmt)
        raw = buf.getvalue()

        t = _trainer()
        item = _injected_item(_danbooru_image_bytes=raw)
        _, reused = _drive_item(t, item, strategy)

        # The same bytes encoded again, keeping their own bucketed output.
        t2 = _trainer()
        item2 = _injected_item(post_id=999, _danbooru_image_bytes=raw)
        BaseTrainer.encode_image(t2, image=Image.open(BytesIO(raw)), target_width=64,
                                 target_height=64, bucket_strategy=strategy)
        fresh = BaseTrainer._get_repa_pixels_for_item(
            t2, item2,
            BaseTrainer._repa_source_region(t2, item2, 64, 64, strategy),
            bucketed_image=t2._last_bucketed_image)

        assert torch.equal(reused, fresh), f"{mode}/{fmt}"


# ---------------------------------------------------------------------------
# (c) nothing else moves
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["resize", "crop", "random_crop"])
def test_repa_off_is_bit_identical_to_the_baseline(strategy):
    """A run without repa_enable never reaches the changed guard."""
    random.seed(0)
    a, teacher_a = _drive_item(_trainer(), _injected_item(), strategy,
                               repa_active=False, handover="current")
    random.seed(0)
    b, teacher_b = _drive_item(_trainer(), _injected_item(), strategy,
                               repa_active=False, handover="prefix")
    assert teacher_a is None and teacher_b is None
    assert torch.equal(a, b)


@pytest.mark.parametrize("strategy", ["resize", "crop"])
def test_a_run_without_injection_keeps_the_baselines_latent_and_region(strategy,
                                                                      tmp_path):
    """File-backed items already handed the decode over at c6292216. The latent is
    still that one bit for bit; the teacher square is now resized from the bucket
    instead of from the original, which is the same region and not the same pixels."""
    path = tmp_path / "src.png"
    _striped().save(path)
    item = {"image_path": str(path), "width": 64, "height": 64}

    a, teacher_a = _drive_item(_trainer(), dict(item), strategy, handover="current")
    b, teacher_b = _drive_item(_trainer(), dict(item), strategy, handover="prefix")
    assert torch.equal(a, b)
    np.testing.assert_allclose(_fingerprint(teacher_a), _fingerprint(teacher_b),
                               atol=0.05)


def test_a_video_item_still_takes_the_no_decode_path(tmp_path):
    """item_type=="video" never decodes a still, so the teacher falls back to the
    path -- which for a real video item is a readable file."""
    path = tmp_path / "frame.png"
    _striped().save(path)
    item = {"image_path": str(path), "width": 64, "height": 64, "item_type": "video"}
    t = _trainer(_temporal_spec=lambda: object(),
                 _encode_video_clip=lambda _it: torch.zeros(1, 3, 8, 64, 64))

    _, teacher = _drive_item(t, item, "resize")
    assert teacher is not None and teacher.shape == (1, 3, S, S)
    assert t.src_size_calls == [str(path)], "the region came from the file, as before"


# ---------------------------------------------------------------------------
# (d) region path and cache key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["resize", "crop", "random_crop"])
def test_the_region_never_opens_the_danbooru_path(strategy):
    """`danbooru://<id>` is not a file. The capture encode_image leaves is the
    only region source an injected item has, and it must be enough."""
    t = _trainer()
    _drive_item(t, _injected_item(), strategy)
    assert t.src_size_calls == []


def test_the_teacher_cache_is_keyed_per_post(tmp_path):
    """Two injected posts share a bucket and therefore a region box; only the
    path keeps their entries apart."""
    t = _trainer()
    square = _striped((256, 256))
    _, a = _drive_item(t, _injected_item(1, square), "resize")
    _, b = _drive_item(t, _injected_item(2, square.rotate(90)), "resize")

    keys = list(t._repa_pix_cache)
    assert keys == [("danbooru://1", (0, 0, 256, 256), (64, 64)),
                    ("danbooru://2", (0, 0, 256, 256), (64, 64))]
    assert not torch.equal(a, b)
    assert torch.equal(t._repa_pix_cache[keys[0]], a)


def test_the_bytes_are_still_freed_before_repa_reads_them():
    """The handover is what keeps the injected pixels reachable -- not a longer
    lifetime for the bytes."""
    item = _injected_item()
    t = _trainer()
    _, teacher = _drive_item(t, item, "resize")
    assert item["_danbooru_image_bytes"] is None
    assert teacher is not None
    # The clear precedes the handover in the real loop, too.
    i = BASE_TRAINER_SRC.index(_HANDOVER["current"])
    assert BASE_TRAINER_SRC.index("_repa_decoded_image = image", i) > i
