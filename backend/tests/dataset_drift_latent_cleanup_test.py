"""What the pre-flight orphan sweep is allowed to delete from a latent cache.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/dataset_drift_latent_cleanup_test.py -v

``cleanup_orphan_latent_cache`` runs unattended when a training run starts, so
a false positive silently destroys hours of VAE encoding. The cache filename is
an md5 of an open-ended key (any bucket resolution, any video clip window, any
audio duration), which is why liveness is decided by the source path stamped
inside each record rather than by reconstructing the hashes a live item could
own. These tests pin the cases that reconstruction gets wrong: a live item
cached at a resolution the sweep was not told about, video and audio records
that have no image-hash form at all, an unreadable record, and the text
embedding cache next door.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training import dataset_drift  # noqa: E402
from core.training import latent_cache as latent_cache_module  # noqa: E402
from core.training.latent_cache import LatentCache  # noqa: E402
from database.models import DatasetBase, DatasetItem  # noqa: E402

UID = "ds-uid"
DATASET_ID = 1


def _db(items):
    """In-memory datasets.db holding *items* (dicts of DatasetItem columns)."""
    engine = create_engine("sqlite://")
    DatasetBase.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    for index, kwargs in enumerate(items):
        fields = dict(kwargs)
        fields.setdefault("base_name", f"item{index}")
        session.add(DatasetItem(dataset_id=DATASET_ID, **fields))
    session.commit()
    return session


def _cache(tmp_path):
    return LatentCache(UID, base_cache_dir=str(tmp_path),
                       namespace="sdxl-fp16", vae_namespace="vae-abc123")


def _cleanup(db, **kwargs):
    return dataset_drift.cleanup_orphan_latent_cache(
        dataset_unique_id=UID, datasets_db=db, dataset_id=DATASET_ID, **kwargs
    )


@pytest.fixture(autouse=True)
def _cache_root(tmp_path, monkeypatch):
    """Never let the sweep near the real cache root."""
    monkeypatch.setattr(latent_cache_module, "get_cache_base_dir",
                        lambda: str(tmp_path))


def test_live_item_survives_at_a_resolution_the_caller_never_named(tmp_path):
    """A bucket outside the default list is still a live latent.

    That list holds six resolutions; a run bucketed at 900x900 matches none of
    them, and the pre-flight caller passes no list at all.
    """
    live = str(tmp_path / "ds" / "a.png")
    cache = _cache(tmp_path)
    cache.save_latent(live, 900, 900, torch.zeros(1, 4, 112, 112))
    cache.save_latent(live, 1024, 1024, torch.zeros(1, 4, 128, 128))

    removed = _cleanup(_db([{"image_path": live}]))

    assert removed == 0
    assert cache.has_latent(live, 900, 900)
    assert cache.has_latent(live, 1024, 1024)


def test_video_and_audio_latents_survive(tmp_path):
    """Clip and audio records are keyed by window, not by (width, height).

    They share ``latents/`` with the image records, so a sweep that only knows
    the image key scheme deletes every one of them on every drift check. The
    second video item stamps a ``video_path`` that differs from its
    ``image_path``, which is what ``_apply_video_metadata`` prefers.
    """
    video = str(tmp_path / "ds" / "clip.webm")
    probed = str(tmp_path / "ds" / "clip_transcoded.webm")
    audio = str(tmp_path / "ds" / "song.flac")
    cache = _cache(tmp_path)
    cache.save_clip_latent(video, 768, 512, 0, 49, 1, torch.zeros(1, 8, 3, 4, 4), fps=30.0)
    cache.save_clip_latent(probed, 768, 512, 0, 49, 1, torch.zeros(1, 8, 3, 4, 4), fps=30.0)
    cache.save_audio_latent(audio, 30.0, 44100, torch.zeros(1, 16, 64))

    db = _db([
        {"image_path": video, "item_type": "video",
         "exif_data": {"video_path": video, "fps": 30.0}},
        {"image_path": str(tmp_path / "ds" / "clip2.mp4"), "item_type": "video",
         "exif_data": {"video_path": probed, "fps": 30.0}},
        {"image_path": audio, "item_type": "audio",
         "exif_data": {"audio_path": audio, "sample_rate": 44100}},
    ])
    removed = _cleanup(db)

    assert removed == 0
    assert cache.has_clip_latent(video, 768, 512, 0, 49, 1, fps=30.0)
    assert cache.has_clip_latent(probed, 768, 512, 0, 49, 1, fps=30.0)
    assert cache.has_audio_latent(audio, 30.0, 44100)


def test_orphans_are_removed(tmp_path):
    """The point of the sweep still works: rows gone from the DB lose their
    latents, whatever modality and resolution they were cached at."""
    live = str(tmp_path / "ds" / "kept.png")
    gone_image = str(tmp_path / "ds" / "deleted.png")
    gone_video = str(tmp_path / "ds" / "deleted.webm")
    gone_audio = str(tmp_path / "ds" / "deleted.flac")
    cache = _cache(tmp_path)
    cache.save_latent(live, 900, 900, torch.zeros(1, 4, 112, 112))
    cache.save_latent(gone_image, 900, 900, torch.zeros(1, 4, 112, 112))
    cache.save_clip_latent(gone_video, 768, 512, 0, 49, 1, torch.zeros(1, 8, 3, 4, 4))
    cache.save_audio_latent(gone_audio, None, 44100, torch.zeros(1, 16, 64))

    removed = _cleanup(_db([{"image_path": live}]))

    assert removed == 3
    assert cache.has_latent(live, 900, 900)
    assert not cache.has_latent(gone_image, 900, 900)
    assert not cache.has_clip_latent(gone_video, 768, 512, 0, 49, 1)
    assert not cache.has_audio_latent(gone_audio, None, 44100)


def test_legacy_layout_is_swept_and_text_embeddings_are_not(tmp_path):
    """Only directories named ``latents`` are candidates.

    Text embeddings live one level up, are keyed by caption hash, and hold no
    source path — matching one against the item list would delete all of them.
    """
    live = str(tmp_path / "ds" / "kept.png")
    gone = str(tmp_path / "ds" / "deleted.png")
    cache = _cache(tmp_path)
    cache.save_latent(live, 900, 900, torch.zeros(1, 4, 112, 112))
    cache.save_text_embeddings("a caption", torch.zeros(1, 77, 768))
    embeddings = sorted(cache.embeddings_dir.glob("*.pt"))
    assert embeddings, "no text embedding written"

    legacy = LatentCache(UID, base_cache_dir=str(tmp_path))  # pre-namespace layout
    legacy.save_latent(gone, 640, 640, torch.zeros(1, 4, 80, 80))

    removed = _cleanup(_db([{"image_path": live}]))

    assert removed == 1
    assert cache.has_latent(live, 900, 900)
    assert all(path.exists() for path in embeddings)


def test_unreadable_record_is_kept(tmp_path):
    """No stamp, no deletion: an unrecognised file is not provably an orphan."""
    cache = _cache(tmp_path)
    stray = cache.latents_dir / "0123456789abcdef0123456789abcdef.pt"
    stray.write_bytes(b"not a torch archive")
    pathless = cache.latents_dir / "fedcba9876543210fedcba9876543210.pt"
    torch.save({"latents": torch.zeros(1, 4, 8, 8)}, pathless)

    removed = _cleanup(_db([]))

    assert removed == 0
    assert stray.exists()
    assert pathless.exists()
