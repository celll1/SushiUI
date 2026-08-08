"""`DELETE /images/{image_id}` must support two distinct, non-overlapping
destructive modes -- record-only vs record+files -- and must never leave a
row whose files are half gone reported as a success.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/gallery_delete_test.py -v
"""

import asyncio
import os
import sys

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import routes  # noqa: E402
from database.models import GalleryBase, GeneratedImage  # noqa: E402


def _make_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/gallery_delete_test.db")
    GalleryBase.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _write(path, content=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def _seed_video_row(session, outputs_dir, thumbs_dir, filename="ref2vid_20260808_212438_1514885945.mp4",
                     preview_filename=None, parameters_extra=None):
    base = os.path.splitext(filename)[0]
    _write(os.path.join(outputs_dir, filename))
    _write(os.path.join(outputs_dir, f"{base}.png"))  # poster
    _write(os.path.join(outputs_dir, f"{base}.json"))  # sidecar
    _write(os.path.join(thumbs_dir, f"{base}.png"))
    _write(os.path.join(thumbs_dir, f"{base}.webp"))
    params = {"is_video": True, **(parameters_extra or {})}
    if preview_filename:
        params["preview_filename"] = preview_filename
        _write(os.path.join(outputs_dir, preview_filename))
    row = GeneratedImage(
        filename=filename, prompt="p", generation_type="ref2vid", parameters=params,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


@pytest.fixture
def env(tmp_path, monkeypatch):
    outputs_dir = str(tmp_path / "outputs")
    thumbs_dir = str(tmp_path / "thumbnails")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(thumbs_dir, exist_ok=True)
    monkeypatch.setattr(routes.settings, "outputs_dir", outputs_dir)
    monkeypatch.setattr(routes.settings, "thumbnails_dir", thumbs_dir)
    session = _make_session(tmp_path)
    yield outputs_dir, thumbs_dir, session
    session.close()


def _files_for(outputs_dir, thumbs_dir, filename, preview_filename=None):
    base = os.path.splitext(filename)[0]
    paths = {
        "media": os.path.join(outputs_dir, filename),
        "poster": os.path.join(outputs_dir, f"{base}.png"),
        "sidecar": os.path.join(outputs_dir, f"{base}.json"),
        "thumb_png": os.path.join(thumbs_dir, f"{base}.png"),
        "thumb_webp": os.path.join(thumbs_dir, f"{base}.webp"),
    }
    if preview_filename:
        paths["preview"] = os.path.join(outputs_dir, preview_filename)
    return paths


FULL_DELETE_LABELS = {"media", "poster", "sidecar", "thumbnail_png", "thumbnail_webp", "preview_proxy"}


# --------------------------------------------------------------------------
# Record-only mode
# --------------------------------------------------------------------------

def test_record_only_deletes_row_but_leaves_every_file(env):
    """MUTANT: `delete_files=False` accidentally reaching the file-removal
    branch (e.g. an `if not delete_files: pass` typo'd to always delete).
    Verified live: temporarily forcing the file-removal branch to run
    unconditionally made this test fail (poster/thumb/media all vanished
    under delete_files=False) before the fix; reverted after confirming."""
    outputs_dir, thumbs_dir, session = env
    row = _seed_video_row(session, outputs_dir, thumbs_dir, preview_filename="ref2vid_x_preview.mp4")
    paths = _files_for(outputs_dir, thumbs_dir, row.filename, "ref2vid_x_preview.mp4")

    result = asyncio.run(routes.delete_image(image_id=row.id, delete_files=False, db=session))

    assert result["success"] is True
    assert result["delete_files"] is False
    assert session.query(GeneratedImage).filter(GeneratedImage.id == row.id).first() is None
    for label, path in paths.items():
        assert os.path.exists(path), f"{label} must survive record-only delete"


# --------------------------------------------------------------------------
# Full delete mode
# --------------------------------------------------------------------------

def test_full_delete_removes_every_artefact(env):
    """MUTANT: the original endpoint only deleted `outputs/<filename>` and
    `thumbnails/<filename>` (wrong path for video/audio thumbnails, which are
    keyed by base name + .png/.webp, not the original extension) plus the
    lossless proxy. Poster PNG, sidecar JSON, and both real thumbnail files
    were leaked. Verified live by reverting to that logic: this test then
    failed with the poster/sidecar/thumb_png/thumb_webp files still present;
    reverted after confirming."""
    outputs_dir, thumbs_dir, session = env
    row = _seed_video_row(session, outputs_dir, thumbs_dir, preview_filename="ref2vid_y_preview.mp4")
    paths = _files_for(outputs_dir, thumbs_dir, row.filename, "ref2vid_y_preview.mp4")

    result = asyncio.run(routes.delete_image(image_id=row.id, delete_files=True, db=session))

    assert result["success"] is True
    assert set(result["deleted_files"]) == FULL_DELETE_LABELS
    assert session.query(GeneratedImage).filter(GeneratedImage.id == row.id).first() is None
    for label, path in paths.items():
        assert not os.path.exists(path), f"{label} must be gone after full delete"


def test_full_delete_with_no_preview_proxy_does_not_error(env):
    """Plain (non-lossless) rows have no preview_filename; the loop must
    simply skip that artefact rather than crash on a missing key."""
    outputs_dir, thumbs_dir, session = env
    row = _seed_video_row(session, outputs_dir, thumbs_dir, preview_filename=None)

    result = asyncio.run(routes.delete_image(image_id=row.id, delete_files=True, db=session))
    assert result["success"] is True
    assert "preview_proxy" not in result["deleted_files"]


def test_media_is_deleted_last_so_a_partial_failure_stays_retryable(env):
    """MUTANT: putting `media` first in `_generated_image_file_paths` (the
    original ordering) means a later failure leaves the row pointing at a
    file that is already gone -- nothing left for a retry to recover.
    Verified live by moving `media` back to the front of the dict: this
    test then failed (media ends up first, not last); reverted after
    confirming."""
    outputs_dir, thumbs_dir, session = env
    row = _seed_video_row(session, outputs_dir, thumbs_dir, preview_filename="ref2vid_z_preview.mp4")
    paths = routes._generated_image_file_paths(row)
    assert list(paths.keys())[-1] == "media"


# --------------------------------------------------------------------------
# Plain-image rows: poster and media collapse onto the same path
# --------------------------------------------------------------------------

def test_png_row_does_not_double_count_poster_and_media(env):
    """MUTANT: unconditionally adding a `poster` entry (no `!= media_path`
    guard) would make a single locked plain-image file surface as TWO
    reported failures instead of one, and attempt `os.remove` on the same
    path twice. Verified live by dropping the guard: this test then failed
    (`poster` appeared as a separate label pointing at the same path as
    `media`); reverted after confirming."""
    outputs_dir, thumbs_dir, session = env
    filename = "txt2img_20260808_000000_1.png"
    _write(os.path.join(outputs_dir, filename))
    _write(os.path.join(thumbs_dir, f"{os.path.splitext(filename)[0]}.png"))
    _write(os.path.join(thumbs_dir, f"{os.path.splitext(filename)[0]}.webp"))
    row = GeneratedImage(filename=filename, prompt="p", generation_type="txt2img", parameters={})
    session.add(row)
    session.commit()
    session.refresh(row)

    paths = routes._generated_image_file_paths(row)
    assert "poster" not in paths
    assert paths["media"] == os.path.join(outputs_dir, filename)

    result = asyncio.run(routes.delete_image(image_id=row.id, delete_files=True, db=session))
    assert result["success"] is True
    assert "poster" not in result["deleted_files"]
    assert not os.path.exists(os.path.join(outputs_dir, filename))


# --------------------------------------------------------------------------
# Path containment: a DB-sourced name must not escape outputs_dir/thumbnails_dir
# --------------------------------------------------------------------------

def test_traversal_filename_is_refused_not_deleted(env, tmp_path):
    """MUTANT: joining `image.filename` into `outputs_dir` without a
    containment check lets a `../` filename delete a file outside
    outputs_dir. Verified live by removing the `_is_safe_output_name` guard
    on `filename`: this test then failed (the decoy above outputs_dir was
    deleted); reverted after confirming."""
    outputs_dir, thumbs_dir, session = env
    decoy = str(tmp_path / "decoy.txt")
    _write(decoy)

    row = GeneratedImage(
        filename="../decoy.txt", prompt="p", generation_type="txt2img", parameters={},
    )
    session.add(row)
    session.commit()
    session.refresh(row)

    assert routes._generated_image_file_paths(row) == {}

    result = asyncio.run(routes.delete_image(image_id=row.id, delete_files=True, db=session))
    assert result["success"] is True
    assert result["deleted_files"] == []
    assert os.path.exists(decoy), "traversal filename must not reach os.remove"


def test_traversal_preview_filename_is_refused_but_media_still_cleaned(env, tmp_path):
    """A malformed `preview_filename` in the parameters JSON must not escape
    either, but must not block cleanup of the row's own legitimate files."""
    outputs_dir, thumbs_dir, session = env
    decoy = str(tmp_path / "decoy_proxy.mp4")
    _write(decoy)

    row = _seed_video_row(
        session, outputs_dir, thumbs_dir,
        filename="ref2vid_traversal_test.mp4",
        parameters_extra={"preview_filename": "../decoy_proxy.mp4"},
    )

    paths = routes._generated_image_file_paths(row)
    assert "preview_proxy" not in paths

    result = asyncio.run(routes.delete_image(image_id=row.id, delete_files=True, db=session))
    assert result["success"] is True
    assert "preview_proxy" not in result["deleted_files"]
    assert os.path.exists(decoy), "traversal preview_filename must not reach os.remove"
    assert not os.path.exists(os.path.join(outputs_dir, row.filename)), "the row's own media must still be removed"


# --------------------------------------------------------------------------
# Malformed filename: None / empty must not crash or target a directory
# --------------------------------------------------------------------------

def test_null_filename_row_deletes_cleanly_without_crashing(env):
    """MUTANT: `os.path.splitext(None)` raises `TypeError` with no filename
    guard. Verified live by removing the `_is_safe_output_name` early return:
    this test then failed with an unhandled TypeError; reverted after
    confirming."""
    _, _, session = env
    row = GeneratedImage(filename=None, prompt="p", generation_type="txt2img", parameters={})
    session.add(row)
    session.commit()
    session.refresh(row)

    result = asyncio.run(routes.delete_image(image_id=row.id, delete_files=True, db=session))
    assert result["success"] is True
    assert result["deleted_files"] == []
    assert session.query(GeneratedImage).filter(GeneratedImage.id == row.id).first() is None


def test_empty_filename_does_not_target_the_outputs_directory(env):
    """An empty filename must not let `poster` resolve to `outputs_dir`
    itself (`os.path.splitext("")[0] + ".png"` would not, but a bare
    `outputs_dir/""` for `media` would resolve to the directory)."""
    outputs_dir, _, session = env
    row = GeneratedImage(filename="", prompt="p", generation_type="txt2img", parameters={})
    session.add(row)
    session.commit()
    session.refresh(row)

    assert routes._generated_image_file_paths(row) == {}
    result = asyncio.run(routes.delete_image(image_id=row.id, delete_files=True, db=session))
    assert result["success"] is True
    assert os.path.isdir(outputs_dir), "outputs_dir itself must never be a delete target"


# --------------------------------------------------------------------------
# Neighbouring rows
# --------------------------------------------------------------------------

def test_deleting_one_row_never_touches_a_neighbour(env):
    """MUTANT: dropping the `.filter(GeneratedImage.id == image_id)` clause
    (e.g. an accidental `.first()` on the bare query) would resolve
    "the requested row" to whichever row the table returns first, deleting
    a neighbour's files and DB row instead of the one actually requested.
    Verified live by removing the filter: this test then failed because the
    first-inserted row (the neighbour here) was the one destroyed; reverted
    after confirming."""
    outputs_dir, thumbs_dir, session = env
    # neighbor is inserted (and so returned by an unscoped `.first()`) FIRST,
    # so a filter regression grabs it instead of the actually-requested victim.
    neighbor = _seed_video_row(
        session, outputs_dir, thumbs_dir,
        filename="ref2vid_20260808_999999_0000000001.mp4",
    )
    victim = _seed_video_row(
        session, outputs_dir, thumbs_dir,
        filename="ref2vid_20260808_212438_1514885945.mp4",
    )
    neighbor_paths = _files_for(outputs_dir, thumbs_dir, neighbor.filename)

    asyncio.run(routes.delete_image(image_id=victim.id, delete_files=True, db=session))

    assert session.query(GeneratedImage).filter(GeneratedImage.id == victim.id).first() is None
    assert session.query(GeneratedImage).filter(GeneratedImage.id == neighbor.id).first() is not None
    for label, path in neighbor_paths.items():
        assert os.path.exists(path), f"neighbour's {label} must survive"


# --------------------------------------------------------------------------
# Partial-failure coherence
# --------------------------------------------------------------------------

def test_partial_file_failure_keeps_the_db_row_and_reports_the_error(env, monkeypatch):
    """MUTANT: dropping the `if errors: raise ...` guard so the DB row is
    deleted unconditionally, swallowing file errors into a silent success.
    Verified live by removing that guard: this test then failed (a 200
    response, and the row gone despite the locked file); reverted after
    confirming. A file that cannot be removed (locked, permission-denied,
    ...) must not be swallowed into success: the row must survive for a
    retry, and the response must say what failed."""
    outputs_dir, thumbs_dir, session = env
    row = _seed_video_row(session, outputs_dir, thumbs_dir)

    real_remove = os.remove

    def _flaky_remove(path):
        if path.endswith(".webp"):
            raise OSError("simulated permission denied")
        return real_remove(path)

    monkeypatch.setattr(routes.os, "remove", _flaky_remove)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes.delete_image(image_id=row.id, delete_files=True, db=session))

    assert exc_info.value.status_code == 500
    assert "thumbnail_webp" in exc_info.value.detail
    # The row was NOT deleted despite some files already being gone.
    assert session.query(GeneratedImage).filter(GeneratedImage.id == row.id).first() is not None


def test_missing_row_is_a_404_not_a_silent_noop(env):
    _, _, session = env
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes.delete_image(image_id=999999, delete_files=True, db=session))
    assert exc_info.value.status_code == 404
