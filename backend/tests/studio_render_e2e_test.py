"""End-to-end proof that a Studio render actually completes.

Background: the Studio render feature had never once succeeded. The
manifest re-validation step in `prepare_render_inputs()` re-parsed the
ALREADY-CANONICAL manifest (`start_frame`/`duration_frames`/`source_in_frame`
keys) as if it were the client's raw manifest (`start`/`duration`/`sourceIn`
keys), so every clip's timing silently collapsed to zero and the second clip
in any manifest raised `StudioRenderValidationError("clip duration must be a
number")`, which the route mapped to an HTTP 422 -- on every submission with
2 or more clips. A manifest with 0 clips passed, but the frontend refuses to
submit an empty timeline, so the bug was unreachable from the UI: 100% of
real submissions failed.

This suite drives the real code path -- `prepare_render_inputs()` through
`_render_worker()` -- with two clips and a REAL ffmpeg process (this
environment has ffmpeg installed; the render is a 64x64, 2-frame job so it
finishes in well under a second), and asserts a completed job with a
readable output file. A test that stubbed out ffmpeg or only unit-tested
`_canonical_manifest()` in isolation would have missed exactly the site of
the real defect (the SECOND call to `_canonical_manifest()`, deep inside
`prepare_render_inputs()`), so nothing here mocks the render pipeline.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/studio_render_e2e_test.py -v
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import threading

import pytest
from fastapi import UploadFile
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import studio_render_jobs as srj  # noqa: E402
from database.models import GalleryBase, GeneratedImage, StudioRenderJob  # noqa: E402
from utils.dataset_scanner import _find_ffprobe  # noqa: E402


def _png_bytes(width: int, height: int, color: tuple) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _upload(filename: str, data: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path}/studio_render_e2e.db")
    GalleryBase.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    # `_render_worker`, `_transition_job_state`, `_job_cancel_requested`,
    # `reap_stale_render_jobs`, etc. all resolve `GallerySessionLocal` from
    # this module's globals, so patching it here redirects every one of
    # them to the temp DB without needing a keyword to thread through each.
    monkeypatch.setattr(srj, "GallerySessionLocal", session_factory)

    outputs_dir = str(tmp_path / "outputs")
    thumbs_dir = str(tmp_path / "thumbnails")
    cache_dir = str(tmp_path / "cache")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(thumbs_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    monkeypatch.setattr(srj.settings, "outputs_dir", outputs_dir)
    monkeypatch.setattr(srj.settings, "thumbnails_dir", thumbs_dir)
    monkeypatch.setattr(srj.settings, "cache_dir", cache_dir)

    db = session_factory()
    yield db, outputs_dir, cache_dir
    db.close()


def _two_clip_manifest() -> dict:
    """A 64x64, 2fps, 1-second timeline: two 1-frame image clips back to
    back on one video track. Two clips is the minimum manifest shape that
    exercised the C1 defect (the first clip's re-parse defaulted its
    missing `duration` to `_as_number(None, ...)`... no -- the FIRST clip's
    fields happened to come back as a validation error on the SECOND clip,
    since `_as_number(item.get("duration"), ...)` raises for a genuinely
    missing key, and only the first clip's `duration` key could ever have
    been present by coincidence of dict iteration; either way, one clip
    alone was insufficient to prove the bug, and the frontend never submits
    zero clips, so two is both necessary and sufficient here)."""
    return {
        "project": {
            "id": "e2e-test-project",
            "revision": 1,
            "name": "E2E Test",
            "width": 64,
            "height": 64,
            "fps": 2,
            "duration": 1.0,
        },
        "assets": [
            {"id": "assetA", "kind": "image", "name": "a.png"},
            {"id": "assetB", "kind": "image", "name": "b.png"},
        ],
        "tracks": [
            {"id": "trackV1", "kind": "video", "muted": False, "visible": True},
        ],
        "clips": [
            {
                "id": "clip1", "assetId": "assetA", "trackId": "trackV1",
                "start": 0.0, "duration": 0.5, "sourceIn": 0.0,
            },
            {
                "id": "clip2", "assetId": "assetB", "trackId": "trackV1",
                "start": 0.5, "duration": 0.5, "sourceIn": 0.0,
            },
        ],
        "render": {"audio_enabled": False, "fit_mode": "cover"},
    }


@pytest.mark.skipif(_find_ffprobe() is None, reason="ffprobe/ffmpeg not available in this environment")
def test_two_clip_manifest_stages_with_nonzero_frame_positions(env):
    """MUTANT: reverting the C1 fix (re-canonicalizing `initial` instead of
    `raw_manifest`) makes this raise `StudioRenderValidationError` on the
    second clip (missing 'duration' after the first pass stripped it to
    `duration_frames`) -- i.e. this call does not even return. Verified
    live against the pre-fix code: it raised
    `StudioRenderValidationError: clip duration must be a number`."""
    db, outputs_dir, cache_dir = env
    manifest = _two_clip_manifest()
    job_id = "e2e0000000000000000000000000001"

    canonical, staging_dir = asyncio.run(
        srj.prepare_render_inputs(
            manifest,
            ["assetA", "assetB"],
            [_upload("a.png", _png_bytes(64, 64, (255, 0, 0))), _upload("b.png", _png_bytes(64, 64, (0, 255, 0)))],
            db,
            job_id,
        )
    )

    clips = {c["id"]: c for c in canonical["clips"]}
    # MUTANT re-check: the pre-fix code zeroed every clip's start/sourceIn
    # (missing keys silently defaulted to 0 by `_as_number(item.get(...,
    # 0), ...)`), which is exactly what these two assertions rule out.
    assert clips["clip1"]["start_frame"] == 0
    assert clips["clip1"]["duration_frames"] == 1
    assert clips["clip2"]["start_frame"] == 1
    assert clips["clip2"]["duration_frames"] == 1
    assert canonical["project"]["duration_frames"] == 2
    assert os.path.isdir(staging_dir)


@pytest.mark.skipif(_find_ffprobe() is None, reason="ffprobe/ffmpeg not available in this environment")
def test_full_pipeline_prepare_to_worker_produces_a_playable_file(env):
    """The full path this feature never once completed: `prepare_render_inputs`
    -> a queued `StudioRenderJob` row -> `_render_worker` -> a real `ffmpeg`
    process -> a Gallery row pointing at a file that actually decodes to the
    requested frame count."""
    db, outputs_dir, cache_dir = env
    manifest = _two_clip_manifest()
    job_id = "e2e0000000000000000000000000002"

    canonical, staging_dir = asyncio.run(
        srj.prepare_render_inputs(
            manifest,
            ["assetA", "assetB"],
            [_upload("a.png", _png_bytes(64, 64, (255, 0, 0))), _upload("b.png", _png_bytes(64, 64, (0, 255, 0)))],
            db,
            job_id,
        )
    )

    job = StudioRenderJob(
        id=job_id, state="queued", manifest=canonical, input_dir=staging_dir,
        progress=0.0, message="Queued",
    )
    db.add(job)
    db.commit()

    srj._render_worker(job_id)

    finished = db.query(StudioRenderJob).filter(StudioRenderJob.id == job_id).first()
    assert finished is not None
    assert finished.state == "completed", f"message={finished.message!r} error={finished.error!r}"
    assert finished.progress == 1.0
    assert finished.filename is not None

    output_path = os.path.join(outputs_dir, finished.filename)
    assert os.path.isfile(output_path)
    assert os.path.getsize(output_path) > 0

    image_row = db.query(GeneratedImage).filter(GeneratedImage.id == finished.gallery_image_id).first()
    assert image_row is not None
    assert image_row.generation_type == "studio_render"

    # The staging directory is removed by the worker's own `finally` block
    # once the job terminates.
    assert not os.path.isdir(staging_dir)

    # Decode the actual output with ffprobe to prove it is a real,
    # 2-frame, 64x64 video and not just a non-empty file ffmpeg happened to
    # create before failing.
    import subprocess
    ffprobe = _find_ffprobe()
    result = subprocess.run(
        [ffprobe, "-v", "error", "-count_frames", "-show_entries",
         "stream=width,height,nb_read_frames", "-of", "json", output_path],
        capture_output=True, text=True, timeout=30,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    assert int(stream["width"]) == 64
    assert int(stream["height"]) == 64
    assert int(stream["nb_read_frames"]) == 2


# ---------------------------------------------------------------------------
# M4: muting a video track's audio must not also drop its picture
# ---------------------------------------------------------------------------

def _single_clip_manifest_on_video_track(muted: bool) -> dict:
    manifest = _two_clip_manifest()
    manifest["tracks"][0]["muted"] = muted
    return manifest


def _canonical_with_fake_staged_assets(tmp_path, raw_manifest: dict) -> dict:
    """`build_render_command()` requires every asset to have a real staged
    file on disk (`_staged_path` checks `os.path.isfile`); build one
    directly rather than going through the full `prepare_render_inputs`
    staging pipeline, since this test is only exercising the filtergraph
    builder, not ingestion."""
    manifest = srj._canonical_manifest(raw_manifest)
    for asset in manifest["assets"]:
        staged_name = f"asset_{asset['id']}.png"
        Image.new("RGB", (64, 64)).save(os.path.join(tmp_path, staged_name))
        asset["staged_name"] = staged_name
    return manifest


def test_muted_video_track_still_produces_a_video_overlay(tmp_path):
    """MUTANT: `if track["kind"] != "video" or track.get("muted") or ...`
    (the pre-fix condition) treats `muted` as a reason to skip the VISUAL
    overlay filter, not just the audio graph. Verified live by reverting to
    that condition: this test then failed (no `overlay=` filter for the
    muted track's clip); reverted after confirming."""
    manifest = _canonical_with_fake_staged_assets(tmp_path, _single_clip_manifest_on_video_track(muted=True))
    command = srj.build_render_command(manifest, str(tmp_path), "ffmpeg", str(tmp_path / "out.mp4"))
    filter_arg = command[command.index("-filter_complex") + 1] if "-filter_complex" in command else None
    assert filter_arg is not None
    assert "overlay=" in filter_arg, "a muted video track must still contribute its picture to the timeline"


def test_unmuted_and_muted_video_tracks_produce_the_same_visual_graph(tmp_path):
    """The overlay filtergraph must not depend on `muted` at all -- it is
    purely an audio-graph concept. `render.audio_enabled=False` on both
    manifests here keeps the (legitimately audio-dependent) audio graph out
    of the comparison."""
    muted_manifest = _canonical_with_fake_staged_assets(tmp_path, _single_clip_manifest_on_video_track(muted=True))
    unmuted_manifest = _canonical_with_fake_staged_assets(tmp_path, _single_clip_manifest_on_video_track(muted=False))
    muted_command = srj.build_render_command(muted_manifest, str(tmp_path), "ffmpeg", str(tmp_path / "out.mp4"))
    unmuted_command = srj.build_render_command(unmuted_manifest, str(tmp_path), "ffmpeg", str(tmp_path / "out.mp4"))
    assert muted_command == unmuted_command


# ---------------------------------------------------------------------------
# H2: atomic state transitions close the queued/running and
# running/cancel_requested races
# ---------------------------------------------------------------------------

def test_transition_only_succeeds_from_the_declared_source_states(env):
    """MUTANT: a read-then-write implementation (`if job.state == "queued":
    job.state = "running"`) would let this succeed regardless of the row's
    ACTUAL current state, because the precondition check and the write are
    two separate steps an interleaved cancel could land between. Here the
    precondition is baked into the UPDATE's WHERE clause, so a job that is
    already "running" can never be re-claimed as if it were "queued"."""
    db, _, _ = env
    job = StudioRenderJob(id="h2-race-1", state="running", manifest={}, input_dir="", progress=0.0)
    db.add(job)
    db.commit()

    claimed = srj._transition_job_state("h2-race-1", ["queued"], "running", db=db)
    assert claimed is False

    row = db.query(StudioRenderJob).filter(StudioRenderJob.id == "h2-race-1").first()
    assert row.state == "running"  # untouched -- no spurious second "claim"


def test_cancel_of_a_job_the_worker_already_claimed_does_not_touch_its_staging_dir(env, tmp_path):
    """This is the exact H2(a) window: `request_cancel_render_job` asks to
    cancel a "queued" job, but the worker has ALREADY committed
    "running" by the time the cancel's UPDATE runs. Proven here without a
    real thread race by simply setting the row to "running" first --
    `_transition_job_state`'s WHERE clause must refuse the "queued"-only
    UPDATE regardless of timing, and `request_cancel_render_job` must
    therefore never call `cleanup_render_staging` for this job."""
    db, _, cache_dir = env
    staging_root = os.path.join(cache_dir, "studio_render_jobs")
    staging_dir = os.path.join(staging_root, "h2-race-2")
    os.makedirs(staging_dir, exist_ok=True)
    marker = os.path.join(staging_dir, "in_use_by_ffmpeg.txt")
    with open(marker, "w") as f:
        f.write("worker owns this now")

    job = StudioRenderJob(id="h2-race-2", state="running", manifest={}, input_dir=staging_dir, progress=0.1)
    db.add(job)
    db.commit()

    state = srj.request_cancel_render_job("h2-race-2")

    assert state == "cancel_requested"  # the running->cancel_requested branch, not queued->cancelled
    assert os.path.isfile(marker), "a job already claimed as 'running' must keep its staging directory"
    row = db.query(StudioRenderJob).filter(StudioRenderJob.id == "h2-race-2").first()
    assert row.state == "cancel_requested"


def test_cancel_after_completion_does_not_raise_and_leaves_the_row_alone(env):
    """This is H2(b): a cancel DELETE arriving after the job already reached
    a terminal state must be a harmless no-op (report the real state), not
    an unhandled exception and not a stuck `cancel_requested`."""
    db, _, _ = env
    job = StudioRenderJob(id="h2-race-3", state="completed", manifest={}, input_dir="", progress=1.0)
    db.add(job)
    db.commit()

    state = srj.request_cancel_render_job("h2-race-3")

    assert state == "completed"
    row = db.query(StudioRenderJob).filter(StudioRenderJob.id == "h2-race-3").first()
    assert row.state == "completed"
