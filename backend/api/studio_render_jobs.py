"""Server-side Studio timeline rendering.

The browser sends a frame-quantized manifest and only the files that are not
already in Gallery. Gallery media is resolved and copied here before a worker
starts, so FFmpeg never receives a client URL or filesystem path.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from queue import Empty, Queue
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from fastapi import UploadFile
from PIL import Image
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from config.settings import settings
from database import GallerySessionLocal
from database.models import GeneratedImage, StudioRenderJob
from api.param_defaults import STUDIO_RENDER_DEFAULTS
from utils.dataset_scanner import _find_ffprobe
from utils.image_utils import calculate_file_hash, create_thumbnail
from utils.video_utils import _locate_ffmpeg


class StudioRenderValidationError(ValueError):
    """A safe, client-facing manifest or media validation error."""


class StudioRenderCancelled(RuntimeError):
    """Raised when a queued or running render was cancelled."""


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_STUDIO_RENDER_VERSION = 1
_render_executor = ThreadPoolExecutor(max_workers=1)
render_submission_lock = asyncio.Lock()
_process_lock = threading.Lock()
_active_processes: Dict[str, subprocess.Popen] = {}
# In-process cancellation signal, keyed by job id. `_render_worker` creates
# one right before it starts ffmpeg and `request_cancel_render_job` sets it;
# this is the primary cancellation channel (checked every loop iteration in
# `_run_ffmpeg`) so the render loop does not have to open a fresh SQLite
# connection at 4 Hz just to notice a cancellation the same process already
# knows about (see M5 in the Studio render audit).
_cancel_events: Dict[str, threading.Event] = {}


def _transition_job_state(
    job_id: str,
    from_states: Sequence[str],
    to_state: str,
    db: Optional[Session] = None,
    **extra_values: Any,
) -> bool:
    """Atomically move a job from one of ``from_states`` to ``to_state``.

    Returns True iff the row was actually in one of ``from_states`` at the
    moment of the UPDATE and the transition happened; False if the row did
    not exist or was already in some other state.

    This is a single ``UPDATE ... WHERE id = ? AND state IN (...)``
    statement, not a read-then-write. A read-then-write (query the state,
    decide in Python, write it back) has a window between the read and the
    write in which the render worker and a cancel request can each observe
    the OTHER's pre-write state and both proceed as if they own the
    transition -- e.g. a cancel reading "queued" a moment before the worker
    commits "running", after which the cancel's write both stomps the
    worker's state AND (formerly) deleted the now-in-use staging directory
    out from under a running ffmpeg process. Collapsing the precondition
    check into the WHERE clause makes that race impossible: SQLite's own
    write lock serializes the two UPDATE statements, and whichever one runs
    second simply matches zero rows instead of clobbering the first.
    """
    own_session = db is None
    session = db or GallerySessionLocal()
    try:
        values: Dict[str, Any] = {"state": to_state, **extra_values}
        matched = (
            session.query(StudioRenderJob)
            .filter(StudioRenderJob.id == job_id, StudioRenderJob.state.in_(list(from_states)))
            .update(values, synchronize_session=False)
        )
        session.commit()
        return bool(matched)
    except OperationalError:
        # A transient SQLite lock contention (busy timeout, or a
        # SQLITE_BUSY_SNAPSHOT under WAL) must not surface to the caller as
        # an unhandled 500 -- it means "the transition did not happen this
        # time", which is the same externally-observable outcome as losing
        # the race above, and every caller here already treats `False` as
        # "someone else is responsible for this job's state instead".
        session.rollback()
        return False
    finally:
        if own_session:
            session.close()


def _now() -> datetime:
    return datetime.now()


def _as_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise StudioRenderValidationError(f"{field} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise StudioRenderValidationError(f"{field} must be a number") from None
    if not math.isfinite(result):
        raise StudioRenderValidationError(f"{field} must be finite")
    return result


def _as_positive_int(value: Any, field: str, minimum: int = 1, maximum: int = 32768) -> int:
    number = _as_number(value, field)
    result = int(number)
    if number != result or result < minimum or result > maximum:
        raise StudioRenderValidationError(f"{field} must be an integer from {minimum} to {maximum}")
    return result


def _as_bool(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise StudioRenderValidationError(f"{field} is not a valid asset or track id")
    return value


def _safe_output_name(name: Any) -> bool:
    return isinstance(name, str) and bool(name) and os.path.basename(name) == name and ".." not in name


def _inside(root: str, candidate: str) -> bool:
    root_real = os.path.realpath(root)
    candidate_real = os.path.realpath(candidate)
    return candidate_real == root_real or candidate_real.startswith(root_real + os.sep)


def _render_defaults(render: Mapping[str, Any]) -> Dict[str, Any]:
    fit_mode = render.get("fit_mode", STUDIO_RENDER_DEFAULTS["fit_mode"])
    if fit_mode not in ("cover", "contain"):
        raise StudioRenderValidationError("render.fit_mode must be 'cover' or 'contain'")
    if _as_bool(render.get("video_lossless"), STUDIO_RENDER_DEFAULTS["video_lossless"]):
        raise StudioRenderValidationError("Studio render currently supports browser-playable H.264 output only")
    return {
        "audio_enabled": _as_bool(render.get("audio_enabled"), STUDIO_RENDER_DEFAULTS["audio_enabled"]),
        "fit_mode": fit_mode,
        "video_lossless": False,
    }


def _metadata_for_file(path: str, kind: str) -> Dict[str, Any]:
    """Probe a staged asset without trusting the upload MIME or extension."""
    if kind == "image":
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                return {
                    "width": int(image.width),
                    "height": int(image.height),
                    "duration": 0.0,
                    "has_audio": False,
                }
        except Exception as exc:
            raise StudioRenderValidationError("An uploaded image could not be decoded") from exc

    ffprobe = _find_ffprobe()
    if not ffprobe:
        # A missing ffprobe is a server configuration problem, not a bad
        # request -- it is not `StudioRenderValidationError` so it surfaces
        # as a 500 (with a server-side log line) rather than a 422 blaming
        # the client for something wrong with the deployment.
        raise RuntimeError("ffprobe is required to validate video and audio assets")
    command = [
        ffprobe,
        "-v", "error",
        "-show_entries", "stream=codec_type,width,height:format=duration",
        "-of", "json",
        path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StudioRenderValidationError("Media probing failed") from exc
    if result.returncode != 0:
        raise StudioRenderValidationError("A video or audio asset could not be decoded")
    try:
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        format_duration = _as_number((payload.get("format") or {}).get("duration") or 0, "media duration")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise StudioRenderValidationError("Media probing returned invalid metadata") from exc
    stream_types = {stream.get("codec_type") for stream in streams}
    if kind not in stream_types:
        raise StudioRenderValidationError(f"The uploaded file is not a {kind} asset")
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    return {
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "duration": max(0.0, format_duration),
        "has_audio": "audio" in stream_types,
    }


def _gallery_kind(row: GeneratedImage) -> str:
    params = row.parameters or {}
    if params.get("is_video") or re.search(r"\.(mp4|webm|mkv|mov|avi)$", row.filename or "", re.I):
        return "video"
    if params.get("is_audio") or re.search(r"\.(flac|wav|mp3|ogg|m4a|aac)$", row.filename or "", re.I):
        return "audio"
    return "image"


def _gallery_source_path(row: GeneratedImage) -> str:
    if not _safe_output_name(row.filename):
        raise StudioRenderValidationError("Gallery asset has an unsafe filename")
    path = os.path.join(settings.outputs_dir, row.filename)
    if not _inside(settings.outputs_dir, path) or not os.path.isfile(path):
        raise StudioRenderValidationError("Gallery asset file is no longer available")
    return path


def _validate_limits(project: Mapping[str, Any], assets: Sequence[Mapping[str, Any]], clips: Sequence[Mapping[str, Any]]) -> Tuple[int, int, int, int, float]:
    width = _as_positive_int(project.get("width"), "width", 64, 8192)
    height = _as_positive_int(project.get("height"), "height", 64, 8192)
    if width * height > 33_177_600:
        raise StudioRenderValidationError("The render canvas is too large")
    fps_value = _as_number(project.get("fps"), "fps")
    if fps_value < 1 or fps_value > 120:
        raise StudioRenderValidationError("fps must be between 1 and 120")
    fps = fps_value
    duration = _as_number(project.get("duration"), "duration")
    if duration <= 0 or duration > float(STUDIO_RENDER_DEFAULTS["max_duration_seconds"]):
        raise StudioRenderValidationError(
            f"duration must be greater than zero and no longer than {STUDIO_RENDER_DEFAULTS['max_duration_seconds']} seconds"
        )
    timeline_frames = max(1, int(round(duration * fps)))
    canonical_duration = timeline_frames / fps
    if len(assets) > int(STUDIO_RENDER_DEFAULTS["max_assets"]):
        raise StudioRenderValidationError("Too many assets in the render manifest")
    if len(clips) > int(STUDIO_RENDER_DEFAULTS["max_clips"]):
        raise StudioRenderValidationError("Too many clips in the render manifest")
    # Bounds the OUTPUT side of the job (canvas area * full timeline length),
    # independent of `_validate_decode_budget()` below, which only bounds
    # pixel-frames actually read from source assets. A small still image
    # held for the whole timeline on a large canvas can pass the decode
    # budget by orders of magnitude while requesting an encode far larger
    # than `max_render_seconds` was sized for -- see
    # `max_output_pixel_frames`'s definition in param_defaults.py.
    output_pixel_frames = width * height * timeline_frames
    output_budget = float(STUDIO_RENDER_DEFAULTS["max_output_pixel_frames"])
    if output_pixel_frames > output_budget:
        # State both numbers and what they are made of: the caller cannot tell
        # which of the canvas, the frame rate or the length to reduce from a
        # bare refusal, and the three multiply.
        raise StudioRenderValidationError(
            f"This render would write {output_pixel_frames:,} output pixel-frames "
            f"({width}x{height} over {timeline_frames} frames), above the limit of "
            f"{int(output_budget):,}. Reduce the canvas size, the frame rate, or the timeline length."
        )
    return width, height, timeline_frames, int(round(fps * 1000)), canonical_duration


def _validate_decode_budget(manifest: Mapping[str, Any]) -> None:
    project = manifest["project"]
    fps = float(project["fps"])
    assets = {asset["id"]: asset for asset in manifest["assets"]}
    total_pixel_frames = 0.0
    for clip in manifest["clips"]:
        asset = assets[clip["asset_id"]]
        if asset["kind"] == "audio":
            continue
        width = int(asset.get("width") or project["width"])
        height = int(asset.get("height") or project["height"])
        clip_frames = int(clip["duration_frames"])
        total_pixel_frames += width * height * clip_frames
    decode_budget = float(STUDIO_RENDER_DEFAULTS["max_decode_pixel_frames"])
    if total_pixel_frames > decode_budget:
        raise StudioRenderValidationError(
            f"This render would read {int(total_pixel_frames):,} source pixel-frames, above the "
            f"limit of {int(decode_budget):,}. Use smaller sources, or shorten the clips that use them."
        )


def _canonical_manifest(raw: Mapping[str, Any], source_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise StudioRenderValidationError("manifest must be a JSON object")
    project = raw.get("project") if isinstance(raw.get("project"), Mapping) else raw
    raw_assets = raw.get("assets")
    raw_tracks = raw.get("tracks")
    raw_clips = raw.get("clips")
    if not isinstance(raw_assets, list) or not isinstance(raw_tracks, list) or not isinstance(raw_clips, list):
        raise StudioRenderValidationError("manifest must contain assets, tracks, and clips arrays")

    width, height, timeline_frames, fps_milli, duration = _validate_limits(project, raw_assets, raw_clips)
    fps = fps_milli / 1000.0
    render = _render_defaults(raw.get("render") if isinstance(raw.get("render"), Mapping) else {})

    assets: List[Dict[str, Any]] = []
    asset_ids: set[str] = set()
    metadata = source_metadata or {}
    for item in raw_assets:
        if not isinstance(item, Mapping):
            raise StudioRenderValidationError("Every asset must be an object")
        asset_id = _safe_id(item.get("id"), "asset id")
        if asset_id in asset_ids:
            raise StudioRenderValidationError(f"Duplicate asset id: {asset_id}")
        asset_ids.add(asset_id)
        kind = item.get("kind")
        if kind not in ("image", "video", "audio"):
            raise StudioRenderValidationError(f"Unsupported asset kind for {asset_id}")
        gallery_id = item.get("galleryId", item.get("gallery_id"))
        if gallery_id is not None:
            try:
                gallery_id = int(gallery_id)
            except (TypeError, ValueError):
                raise StudioRenderValidationError(f"Invalid Gallery id for {asset_id}") from None
            if gallery_id <= 0:
                raise StudioRenderValidationError(f"Invalid Gallery id for {asset_id}")
        entry: Dict[str, Any] = {
            "id": asset_id,
            "kind": kind,
            "gallery_id": gallery_id,
            "name": str(item.get("name") or asset_id)[:255],
        }
        actual = metadata.get(asset_id) or {}
        if "staged_name" in item:
            staged_name = item.get("staged_name")
            if not _safe_output_name(staged_name):
                raise StudioRenderValidationError("Invalid staged asset name")
            entry["staged_name"] = staged_name
        if actual:
            entry.update({
                "width": int(actual.get("width") or 0),
                "height": int(actual.get("height") or 0),
                "duration": float(actual.get("duration") or 0.0),
                "has_audio": bool(actual.get("has_audio")),
            })
            if actual.get("source_hash"):
                entry["source_hash"] = str(actual["source_hash"])
                entry["hash_kind"] = str(actual.get("hash_kind") or "file_bytes")
        else:
            entry["duration"] = max(0.0, _as_number(item.get("duration", 0), f"asset {asset_id} duration"))
            entry["has_audio"] = bool(item.get("has_audio", False))
        assets.append(entry)

    tracks: List[Dict[str, Any]] = []
    track_ids: set[str] = set()
    track_kind: Dict[str, str] = {}
    for item in raw_tracks:
        if not isinstance(item, Mapping):
            raise StudioRenderValidationError("Every track must be an object")
        track_id = _safe_id(item.get("id"), "track id")
        if track_id in track_ids:
            raise StudioRenderValidationError(f"Duplicate track id: {track_id}")
        kind = item.get("kind")
        if kind not in ("video", "audio"):
            raise StudioRenderValidationError(f"Unsupported track kind for {track_id}")
        track_ids.add(track_id)
        track_kind[track_id] = kind
        tracks.append({
            "id": track_id,
            "kind": kind,
            "muted": bool(item.get("muted", False)),
            "visible": bool(item.get("visible", True)),
        })

    asset_by_id = {item["id"]: item for item in assets}
    clips: List[Dict[str, Any]] = []
    clip_ids: set[str] = set()
    for item in raw_clips:
        if not isinstance(item, Mapping):
            raise StudioRenderValidationError("Every clip must be an object")
        if item.get("activeTake") is False:
            continue
        asset_id = _safe_id(item.get("assetId", item.get("asset_id")), "clip asset id")
        track_id = _safe_id(item.get("trackId", item.get("track_id")), "clip track id")
        if asset_id not in asset_by_id or track_id not in track_kind:
            raise StudioRenderValidationError("Every active clip must reference an existing asset and track")
        asset = asset_by_id[asset_id]
        expected_track_kind = "audio" if asset["kind"] == "audio" else "video"
        if track_kind[track_id] != expected_track_kind:
            raise StudioRenderValidationError(f"Asset {asset_id} is on a track with the wrong kind")
        start = _as_number(item.get("start", 0), "clip start")
        clip_duration = _as_number(item.get("duration"), "clip duration")
        source_in = _as_number(item.get("sourceIn", item.get("source_in", 0)), "clip sourceIn")
        if start < 0 or clip_duration <= 0 or source_in < 0:
            raise StudioRenderValidationError("Clip times must be non-negative and duration must be positive")
        start_frame = int(round(start * fps))
        duration_frames = int(round(clip_duration * fps))
        source_frame = int(round(source_in * fps))
        if duration_frames < 1 or start_frame + duration_frames > timeline_frames:
            raise StudioRenderValidationError("Clip lies outside the project timeline")
        presentation = item.get("presentation") or ("frame" if asset["kind"] == "image" else "clip")
        if presentation not in ("frame", "hold", "clip"):
            raise StudioRenderValidationError("Unsupported clip presentation")
        fit_mode = item.get("fitMode", item.get("fit_mode", render["fit_mode"]))
        if fit_mode not in ("cover", "contain"):
            raise StudioRenderValidationError("clip.fitMode must be 'cover' or 'contain'")
        if asset["kind"] == "image":
            if duration_frames > 1 and presentation != "hold":
                raise StudioRenderValidationError("An image longer than one frame must use presentation='hold'")
            source_frame = 0
        else:
            source_duration = float(asset.get("duration") or 0.0)
            if source_duration <= 0:
                raise StudioRenderValidationError(f"Asset {asset_id} has no usable duration")
            if source_frame / fps + duration_frames / fps > source_duration + (0.5 / fps):
                raise StudioRenderValidationError(f"Clip {asset_id} extends beyond the source duration")
        clip_id = _safe_id(item.get("id") or f"clip-{len(clips)}", "clip id")
        if clip_id in clip_ids:
            raise StudioRenderValidationError(f"Duplicate clip id: {clip_id}")
        clip_ids.add(clip_id)
        clips.append({
            "id": clip_id,
            "asset_id": asset_id,
            "track_id": track_id,
            "start_frame": start_frame,
            "duration_frames": duration_frames,
            "source_in_frame": source_frame,
            "presentation": presentation,
            "fit_mode": fit_mode,
        })

    return {
        "studio_render_version": _STUDIO_RENDER_VERSION,
        "project": {
            "id": str(project.get("id") or "")[:128],
            "revision": int(project.get("revision") or 0),
            "name": str(project.get("name") or "Untitled Studio Project")[:255],
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
            "duration_frames": timeline_frames,
        },
        "render": render,
        "assets": assets,
        "tracks": tracks,
        "clips": clips,
    }


async def _write_upload(upload: UploadFile, path: str) -> int:
    limit = int(STUDIO_RENDER_DEFAULTS["max_upload_bytes"])
    total = 0
    try:
        with open(path, "wb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise StudioRenderValidationError("An uploaded Studio asset is too large")
                output.write(chunk)
    finally:
        await upload.close()
    if total <= 0:
        raise StudioRenderValidationError("An uploaded Studio asset is empty")
    return total


async def prepare_render_inputs(
    raw_manifest: Mapping[str, Any],
    upload_ids: Optional[Sequence[str]],
    upload_files: Optional[Sequence[UploadFile]],
    db: Session,
    job_id: str,
) -> Tuple[Dict[str, Any], str]:
    """Resolve Gallery assets and stage uploads before a job is persisted."""
    initial = _canonical_manifest(raw_manifest)
    upload_ids = list(upload_ids or [])
    upload_files = list(upload_files or [])
    if len(upload_ids) != len(upload_files):
        raise StudioRenderValidationError("asset_ids and asset_files must have the same length")
    uploads: Dict[str, UploadFile] = {}
    for asset_id, upload in zip(upload_ids, upload_files):
        _safe_id(asset_id, "upload asset id")
        if asset_id in uploads:
            raise StudioRenderValidationError(f"Duplicate uploaded asset id: {asset_id}")
        uploads[asset_id] = upload

    staging_root = os.path.join(settings.cache_dir, "studio_render_jobs")
    staging_dir = os.path.join(staging_root, job_id)
    os.makedirs(staging_dir, exist_ok=False)
    metadata: Dict[str, Dict[str, Any]] = {}
    staged_info: Dict[str, Dict[str, str]] = {}
    staged_bytes = 0
    # A conservative disk-space reservation for the file `_render_worker`
    # will eventually write to this same staging directory. This is a
    # margin, not a size prediction; see `output_bytes_per_pixel_frame`'s
    # definition in param_defaults.py.
    project = initial["project"]
    output_pixel_frames = int(project["width"]) * int(project["height"]) * int(project["duration_frames"])
    estimated_output_bytes = output_pixel_frames * float(STUDIO_RENDER_DEFAULTS["output_bytes_per_pixel_frame"])
    output_margin_bytes = max(256 * 1024 * 1024, int(estimated_output_bytes))

    def _require_free_space(extra_bytes: int) -> None:
        # Checked BEFORE writing, not after: writing first and checking once
        # at the end let a submission fill the disk with up to
        # `max_total_input_bytes` (4 GiB) of staged input before the check
        # ever ran (see M2 in the Studio render audit).
        free_bytes = shutil.disk_usage(staging_root).free
        if free_bytes < extra_bytes + output_margin_bytes:
            raise StudioRenderValidationError("Not enough free disk space for the Studio render")

    try:
        # Phase 1 (event loop): read every uploaded asset onto disk. This is
        # the only part of ingestion that has to run on the event loop --
        # `UploadFile.read()` is itself a coroutine -- and it never touches
        # Gallery files, so it cannot block on a multi-GiB `shutil.copy2` or
        # an `ffprobe` subprocess the way the rest of staging can.
        upload_targets: Dict[str, Tuple[str, str]] = {}
        for asset in initial["assets"]:
            asset_id = asset["id"]
            if asset.get("gallery_id") is not None:
                continue
            upload = uploads.pop(asset_id, None)
            if upload is None:
                raise StudioRenderValidationError(
                    f"Asset {asset_id} is not a Gallery asset and has no uploaded file"
                )
            staged_name = f"asset_{asset_id}"
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix and re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
                staged_name += suffix
            target = os.path.join(staging_dir, staged_name)
            # The upload's true size is not known ahead of the read (chunked
            # `UploadFile.read()`), so the best that can be checked here is
            # remaining headroom against the worst case a single asset could
            # be (`max_upload_bytes`); the per-asset write itself is still
            # bounded by that same limit inside `_write_upload`.
            _require_free_space(int(STUDIO_RENDER_DEFAULTS["max_upload_bytes"]))
            staged_bytes += await _write_upload(upload, target)
            if staged_bytes > int(STUDIO_RENDER_DEFAULTS["max_total_input_bytes"]):
                raise StudioRenderValidationError("Total Studio render input size is too large")
            upload_targets[asset_id] = (target, staged_name)
        if uploads:
            unknown = next(iter(uploads))
            raise StudioRenderValidationError(f"Uploaded asset {unknown} is not referenced by the manifest")

        # Phase 2 (worker thread): Gallery copy, hashing, and ffprobe/PIL
        # probing are all blocking I/O or `subprocess.run()` calls. A single
        # submission can touch up to `max_total_input_bytes` (4 GiB) of
        # Gallery files and run `ffprobe` once per asset (up to `max_assets`,
        # each with its own timeout) -- running that inline on the event
        # loop stalls every other request (including this job's own cancel
        # DELETE) and every WebSocket progress update for as long as it
        # takes. `render_submission_lock` only needs to protect the
        # single-active-job check and the row insert in routes.py, not this.
        def _stage_and_probe() -> int:
            local_staged_bytes = staged_bytes
            for asset in initial["assets"]:
                asset_id = asset["id"]
                gallery_id = asset.get("gallery_id")
                if gallery_id is not None:
                    row = db.query(GeneratedImage).filter(GeneratedImage.id == gallery_id).first()
                    if not row:
                        raise StudioRenderValidationError(f"Gallery asset {gallery_id} was not found")
                    if _gallery_kind(row) != asset["kind"]:
                        raise StudioRenderValidationError(f"Gallery asset {gallery_id} has a different media kind")
                    source = _gallery_source_path(row)
                    staged_name = f"asset_{asset_id}"
                    suffix = Path(row.filename).suffix.lower()
                    if suffix and re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
                        staged_name += suffix
                    target = os.path.join(staging_dir, staged_name)
                    source_size = os.path.getsize(source)
                    if source_size > int(STUDIO_RENDER_DEFAULTS["max_upload_bytes"]):
                        raise StudioRenderValidationError("A Gallery asset is too large to render")
                    local_staged_bytes += source_size
                    if local_staged_bytes > int(STUDIO_RENDER_DEFAULTS["max_total_input_bytes"]):
                        raise StudioRenderValidationError("Total Studio render input size is too large")
                    # Checked before the copy, with the asset's actual known
                    # size (Gallery files are local, so `getsize()` is free
                    # here) -- not once at the very end after every Gallery
                    # asset has already been copied.
                    _require_free_space(source_size)
                    shutil.copy2(source, target)
                    if row.image_hash:
                        source_hash = row.image_hash
                        # Gallery still images are hashed by re-encoded pixel
                        # content (see `calculate_image_hash`); video/audio
                        # rows are hashed by raw file bytes. Record which one
                        # this value is so a later provenance check knows
                        # what it can and can't be compared against.
                        hash_kind = "image_pixels" if asset["kind"] == "image" else "file_bytes"
                    else:
                        source_hash = calculate_file_hash(target)
                        hash_kind = "file_bytes"
                else:
                    target, staged_name = upload_targets[asset_id]
                    source_hash = calculate_file_hash(target)
                    hash_kind = "file_bytes"
                probed = _metadata_for_file(target, asset["kind"])
                if asset["kind"] != "image" and probed["duration"] <= 0:
                    raise StudioRenderValidationError(f"Asset {asset_id} has no usable duration")
                probed["staged_name"] = staged_name
                probed["source_hash"] = source_hash
                probed["hash_kind"] = hash_kind
                metadata[asset_id] = probed
                staged_info[asset_id] = {
                    "staged_name": staged_name,
                    "source": "gallery" if gallery_id is not None else "upload",
                }
            return local_staged_bytes

        staged_bytes = await asyncio.to_thread(_stage_and_probe)

        # Re-run all frame-boundary checks with server-probed durations. The
        # client may not be trusted to report a video's true source length.
        #
        # This MUST re-parse `raw_manifest`, not `initial`: `initial` has
        # already been through `_canonical_manifest()` once, so its clips use
        # the canonical `start_frame`/`duration_frames`/`source_in_frame` keys
        # -- not the `start`/`duration`/`sourceIn` keys `_canonical_manifest()`
        # expects to parse. Feeding it back in a second time silently zeroed
        # every clip's start/sourceIn (missing keys defaulted to 0) and threw
        # a validation error on the second clip (missing `duration`), which
        # surfaced to the client as a 422 on every manifest with 2+ clips.
        final_manifest = _canonical_manifest(raw_manifest, source_metadata=metadata)
        for asset in final_manifest["assets"]:
            info = staged_info.get(asset["id"])
            if info is None:
                raise StudioRenderValidationError(f"Asset {asset['id']} was not staged")
            asset["staged_name"] = info["staged_name"]
            asset["source"] = info["source"]
        _validate_decode_budget(final_manifest)
        return final_manifest, staging_dir
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _staged_path(staging_dir: str, staged_name: str) -> str:
    if not _safe_output_name(staged_name):
        raise StudioRenderValidationError("Invalid staged media name")
    path = os.path.join(staging_dir, staged_name)
    if not _inside(staging_dir, path) or not os.path.isfile(path):
        raise StudioRenderValidationError("Staged media is missing")
    return path


def _scale_filter(width: int, height: int, fit_mode: str) -> str:
    if fit_mode == "contain":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
        )
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}:(iw-{width})/2:(ih-{height})/2,setsar=1"
    )


def build_render_command(manifest: Mapping[str, Any], staging_dir: str, ffmpeg: str, output_path: str) -> List[str]:
    """Build an argv-only FFmpeg filtergraph for the supported timeline model."""
    project = manifest["project"]
    render = manifest["render"]
    width = int(project["width"])
    height = int(project["height"])
    fps = float(project["fps"])
    total_frames = int(project["duration_frames"])
    duration = total_frames / fps
    assets = {asset["id"]: asset for asset in manifest["assets"]}
    tracks = {track["id"]: track for track in manifest["tracks"]}
    clips = manifest["clips"]

    command: List[str] = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    input_index = 0
    clip_inputs: Dict[str, int] = {}
    for clip in clips:
        asset = assets[clip["asset_id"]]
        path = _staged_path(staging_dir, asset["staged_name"])
        if asset["kind"] == "image":
            command += ["-loop", "1", "-framerate", str(fps), "-i", path]
        else:
            command += ["-i", path]
        clip_inputs[clip["id"]] = input_index
        input_index += 1

    filters: List[str] = []
    filters.append(f"color=c=black:s={width}x{height}:r={fps}:d={duration:.6f}[base0]")
    current_video = "base0"
    track_order = {track_id: order for order, track_id in enumerate(track["id"] for track in manifest["tracks"])}
    visual_clips = sorted(
        enumerate(clips),
        key=lambda pair: (
            track_order.get(pair[1]["track_id"], 0),
            pair[1]["start_frame"],
            pair[0],
        ),
    )
    for visual_index, (index, clip) in enumerate(visual_clips):
        track = tracks[clip["track_id"]]
        asset = assets[clip["asset_id"]]
        # `muted` is an audio-only concept (see the audio pass below, which
        # applies it to the audio graph). Excluding it here as well made
        # muting a video track's audio silently drop its picture too.
        if track["kind"] != "video" or not track.get("visible", True):
            continue
        source_in = clip["source_in_frame"] / fps
        clip_start = clip["start_frame"] / fps
        clip_duration = clip["duration_frames"] / fps
        source_label = f"clipv{index}"
        input_number = clip_inputs[clip["id"]]
        filters.append(
            f"[{input_number}:v:0]trim=start={source_in:.6f}:duration={clip_duration:.6f},"
            f"setpts=PTS-STARTPTS+{clip_start:.6f}/TB,{_scale_filter(width, height, clip.get('fit_mode', render['fit_mode']))}[{source_label}]"
        )
        next_video = f"base{visual_index + 1}"
        filters.append(
            f"[{current_video}][{source_label}]overlay=eof_action=pass:shortest=0:"
            f"format=auto:enable='between(t,{clip_start:.6f},{clip_start + clip_duration:.6f})'[{next_video}]"
        )
        current_video = next_video
    filters.append(f"[{current_video}]format=yuv420p[vout]")

    audio_sources: List[str] = []
    if render["audio_enabled"]:
        for index, clip in enumerate(clips):
            track = tracks[clip["track_id"]]
            asset = assets[clip["asset_id"]]
            if track.get("muted") or not track.get("visible", True):
                continue
            if asset["kind"] == "audio":
                has_audio = True
            elif asset["kind"] == "video" and track["kind"] == "video":
                has_audio = bool(asset.get("has_audio"))
            else:
                has_audio = False
            if not has_audio:
                continue
            source_in = clip["source_in_frame"] / fps
            clip_start = clip["start_frame"] / fps
            clip_duration = clip["duration_frames"] / fps
            input_number = clip_inputs[clip["id"]]
            label = f"clipa{index}"
            delay_ms = int(round(clip_start * 1000))
            filters.append(
                f"[{input_number}:a:0]atrim=start={source_in:.6f}:duration={clip_duration:.6f},"
                f"asetpts=PTS-STARTPTS,adelay={delay_ms}:all=1[{label}]"
            )
            audio_sources.append(label)
    if audio_sources:
        joined = "".join(f"[{label}]" for label in audio_sources)
        filters.append(
            f"{joined}amix=inputs={len(audio_sources)}:duration=longest:dropout_transition=0:normalize=0,"
            f"aresample=async=1:first_pts=0,atrim=duration={duration:.6f}[aout]"
        )

    filter_graph = ";".join(filters)
    # A large project can exceed Windows' command-line length limit when the
    # graph is passed inline. Keep small graphs readable for diagnostics and
    # spill larger graphs to the already-private staging directory.
    if len(filter_graph) > 6000:
        script_path = os.path.join(staging_dir, "filter_complex.txt")
        with open(script_path, "w", encoding="utf-8", newline="\n") as script:
            script.write(filter_graph)
        command += ["-filter_complex_script", script_path]
    else:
        command += ["-filter_complex", filter_graph]
    command += ["-map", "[vout]"]
    if audio_sources:
        command += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    command += [
        "-c:v", "libx264",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-frames:v", str(total_frames),
        "-t", f"{duration:.6f}",
        "-progress", "pipe:1",
        "-nostats",
        output_path,
    ]
    return command


def _job_cancel_requested(job_id: str) -> bool:
    db = GallerySessionLocal()
    try:
        job = db.query(StudioRenderJob).filter(StudioRenderJob.id == job_id).first()
        return bool(job and job.state == "cancel_requested")
    except OperationalError:
        # A busy/locked SQLite read here must not turn into a render
        # failure -- the in-process `_cancel_events` Event is the primary
        # cancellation channel; this DB read is only a fallback poll, so
        # "could not check this time" is treated the same as "not
        # cancelled" and the next poll a second later will retry.
        return False
    finally:
        db.close()


def _update_job(job_id: str, **values: Any) -> None:
    db = GallerySessionLocal()
    try:
        job = db.query(StudioRenderJob).filter(StudioRenderJob.id == job_id).first()
        if not job:
            return
        for key, value in values.items():
            setattr(job, key, value)
        db.commit()
    finally:
        db.close()


def _run_ffmpeg(command: List[str], job_id: str, total_frames: int, cancel_event: threading.Event) -> None:
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        raise RuntimeError("Could not start the video renderer") from exc
    with _process_lock:
        _active_processes[job_id] = process
    stderr_lines: List[str] = []
    progress_lines: Queue[Optional[str]] = Queue()

    def drain_progress() -> None:
        if process.stdout:
            for line in process.stdout:
                progress_lines.put(line)
        progress_lines.put(None)

    def drain_stderr() -> None:
        if process.stderr:
            stderr_lines.extend(process.stderr.readlines())

    progress_thread = threading.Thread(
        target=drain_progress,
        name=f"studio-render-progress-{job_id[:8]}",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=drain_stderr,
        name=f"studio-render-stderr-{job_id[:8]}",
        daemon=True,
    )
    progress_thread.start()
    stderr_thread.start()
    started = time.monotonic()
    last_db_poll = started
    try:
        while True:
            if cancel_event.is_set():
                process.kill()
                raise StudioRenderCancelled()
            now = time.monotonic()
            if now - last_db_poll >= 1.0:
                # Fallback poll for a cancellation that reached the DB
                # without going through `request_cancel_render_job`'s
                # in-process event (there is currently no such path, but a
                # multi-process deployment or an admin editing the row
                # directly would take this route). Deliberately at 1 Hz,
                # not 4 Hz: this opens a fresh SQLite connection every time,
                # and `_job_cancel_requested` already treats a locked/busy
                # read as "not cancelled yet, ask again next second" rather
                # than raising.
                last_db_poll = now
                if _job_cancel_requested(job_id):
                    cancel_event.set()
                    process.kill()
                    raise StudioRenderCancelled()
            if now - started > float(STUDIO_RENDER_DEFAULTS["max_render_seconds"]):
                process.kill()
                raise RuntimeError("Studio render exceeded the renderer time limit")
            try:
                line = progress_lines.get(timeout=0.25)
            except Empty:
                line = ""
            if line:
                if line.startswith("frame="):
                    try:
                        frame = int(line.split("=", 1)[1].strip())
                        _update_job(job_id, progress=min(0.99, max(0.0, frame / max(1, total_frames))), message="Rendering timeline")
                    except ValueError:
                        pass
                continue
            if line is None or process.poll() is not None:
                break
        return_code = process.wait()
        if return_code != 0:
            if _job_cancel_requested(job_id):
                raise StudioRenderCancelled()
            raise RuntimeError("FFmpeg could not render the Studio timeline")
    finally:
        with _process_lock:
            _active_processes.pop(job_id, None)
        if process.poll() is None:
            process.kill()
        progress_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
        stderr_text = "".join(stderr_lines)[-2000:]
        if stderr_text:
            # Keep the log useful for operators without returning paths or the
            # raw command to the browser.
            print(f"[StudioRender] ffmpeg: {stderr_text}")


def _write_poster(video_path: str, poster_path: str, duration: float) -> bool:
    try:
        ffmpeg = _locate_ffmpeg()
    except RuntimeError:
        return False
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, duration / 2):.6f}", "-i", video_path,
        "-frames:v", "1", "-vf", "scale=iw:ih:force_original_aspect_ratio=decrease",
        poster_path,
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _render_output_name(job_id: str) -> str:
    return f"studio_render_{time.strftime('%Y%m%d_%H%M%S')}_{job_id[:12]}.mp4"


def _collect_render_warnings(manifest: Mapping[str, Any]) -> List[str]:
    """Collect non-fatal notices for the completed render job."""
    warnings: List[str] = []
    render = manifest["render"]
    tracks = {track["id"]: track for track in manifest["tracks"]}
    assets = {asset["id"]: asset for asset in manifest["assets"]}
    if render["audio_enabled"]:
        has_any_audio = False
        for clip in manifest["clips"]:
            track = tracks.get(clip["track_id"])
            asset = assets.get(clip["asset_id"])
            if not track or not asset or track.get("muted") or not track.get("visible", True):
                continue
            if asset["kind"] == "audio" or (asset["kind"] == "video" and asset.get("has_audio")):
                has_any_audio = True
                break
        if not has_any_audio:
            warnings.append(
                "Audio was enabled for this render, but no unmuted clip on the timeline has an audio source; "
                "the output has no sound track."
            )
    project = manifest["project"]
    canvas_ratio = float(project["width"]) / float(project["height"])
    warned_clips: set[str] = set()
    for clip in manifest["clips"]:
        asset = assets.get(clip["asset_id"])
        if not asset or asset["kind"] == "audio":
            continue
        source_width = int(asset.get("width") or 0)
        source_height = int(asset.get("height") or 0)
        if source_width <= 0 or source_height <= 0 or clip.get("fit_mode", render["fit_mode"]) != "cover":
            continue
        if abs((source_width / source_height) - canvas_ratio) < 0.01 or clip["id"] in warned_clips:
            continue
        warned_clips.add(clip["id"])
        warnings.append(f"Clip {clip['id']} is filled to the canvas and its edges are cropped.")
    return warnings


def _persist_render_output(
    job_id: str,
    manifest: Mapping[str, Any],
    staging_dir: str,
    temp_output: str,
    filename: Optional[str] = None,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    project = manifest["project"]
    render = manifest["render"]
    os.makedirs(settings.outputs_dir, exist_ok=True)
    filename = filename or _render_output_name(job_id)
    output_path = os.path.join(settings.outputs_dir, filename)
    if not _inside(settings.outputs_dir, output_path):
        raise RuntimeError("Invalid Studio render output path")
    shutil.move(temp_output, output_path)
    num_frames = int(project["duration_frames"])
    fps = float(project["fps"])
    duration = num_frames / fps
    warnings = _collect_render_warnings(manifest)
    poster_path = os.path.join(settings.outputs_dir, f"{Path(filename).stem}.png")
    if not _write_poster(output_path, poster_path, duration):
        warnings.append("Poster frame generation failed; the Gallery entry has no preview thumbnail.")
    if os.path.isfile(poster_path):
        try:
            create_thumbnail(poster_path)
        except Exception as exc:
            print(f"[StudioRender] thumbnail creation failed: {exc}")
            warnings.append("Thumbnail generation failed for the rendered output.")

    manifest_without_staged = json.loads(json.dumps(manifest))
    for asset in manifest_without_staged.get("assets", []):
        asset.pop("staged_name", None)
    manifest_json = json.dumps(manifest_without_staged, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    media_hash = calculate_file_hash(output_path)
    params = {
        "prompt": "",
        "negative_prompt": "",
        "sampler": "ffmpeg (timeline)",
        "steps": 0,
        "cfg_scale": 1.0,
        "seed": -1,
        "width": int(project["width"]),
        "height": int(project["height"]),
        "fps": fps,
        "num_frames": num_frames,
        "duration": duration,
        "audio_enable": bool(render["audio_enabled"]),
        "is_video": True,
        "hash_kind": "file_bytes",
        "studio_render_version": _STUDIO_RENDER_VERSION,
        "studio_project_id": project.get("id"),
        "studio_project_revision": project.get("revision"),
        "studio_manifest_sha256": manifest_hash,
        "studio_manifest": manifest_without_staged,
        "source_assets": [
            {
                "asset_id": asset["id"],
                "gallery_id": asset.get("gallery_id"),
                "source_hash": asset.get("source_hash"),
                "hash_kind": asset.get("hash_kind"),
            }
            for asset in manifest["assets"]
        ],
        "fit_mode": render["fit_mode"],
        "warnings": warnings,
    }
    sidecar = {
        "generation_type": "studio_render",
        "filename": filename,
        "preview_filename": None,
        # Keys shared with `save_video_with_metadata()`'s sidecar
        # (utils/video_utils.py) so any code that reads a video sidecar
        # generically does not have to special-case Studio renders. None of
        # these have a Studio-render meaning (no prompt, no model, no
        # inference), so they are populated with the same "not applicable"
        # values a non-generative video write would use.
        "prompt": "",
        "negative_prompt": "",
        "model_name": "Studio timeline renderer",
        "model_hash": None,
        "seed": -1,
        "num_frames": num_frames,
        "fps": fps,
        "width": int(project["width"]),
        "height": int(project["height"]),
        "num_inference_steps": None,
        "guidance_scale": None,
        "audio_enable": bool(render["audio_enabled"]),
        "audio_sample_rate": None,
        "duration": duration,
        "lossless": False,
        "studio_manifest_sha256": manifest_hash,
    }
    with open(os.path.join(settings.outputs_dir, f"{Path(filename).stem}.json"), "w", encoding="utf-8") as sidecar_file:
        json.dump(sidecar, sidecar_file, ensure_ascii=False, indent=2)
    return filename, None, {"params": params, "media_hash": media_hash, "warnings": warnings}


def _render_worker(job_id: str) -> None:
    db = GallerySessionLocal()
    temp_output = ""
    staging_dir = ""
    persisted_filename = ""
    cancel_event = threading.Event()
    try:
        # Atomic queued -> running claim. This MUST be a single
        # UPDATE ... WHERE state = 'queued', not a read-then-write: a
        # read-then-write leaves a window in which `request_cancel_render_job`
        # can see "queued", cancel it, and delete its staging directory,
        # right before this commits "running" and starts reading files out
        # of that now-deleted directory (see H2 in the Studio render audit).
        claimed = _transition_job_state(
            job_id, ["queued"], "running", db=db,
            started_at=_now(), progress=0.0, message="Preparing renderer",
        )
        if not claimed:
            return
        with _process_lock:
            _cancel_events[job_id] = cancel_event
        job = db.query(StudioRenderJob).filter(StudioRenderJob.id == job_id).first()
        if not job:
            return
        manifest = job.manifest
        staging_dir = job.input_dir
        if not _inside(os.path.join(settings.cache_dir, "studio_render_jobs"), staging_dir):
            raise RuntimeError("Invalid Studio render staging directory")
        temp_output = os.path.join(staging_dir, "rendered.mp4")
        try:
            ffmpeg = _locate_ffmpeg()
        except RuntimeError as exc:
            raise RuntimeError("ffmpeg is required to render a Studio timeline") from exc
        command = build_render_command(manifest, staging_dir, ffmpeg, temp_output)
        _run_ffmpeg(command, job_id, int(manifest["project"]["duration_frames"]), cancel_event)
        if cancel_event.is_set():
            raise StudioRenderCancelled()
        persisted_filename = _render_output_name(job_id)
        filename, preview_filename, saved = _persist_render_output(
            job_id, manifest, staging_dir, temp_output, filename=persisted_filename
        )
        params = saved["params"]
        image = GeneratedImage(
            filename=filename,
            prompt="",
            negative_prompt="",
            model_name="Studio timeline renderer",
            sampler="ffmpeg (timeline)",
            steps=0,
            cfg_scale=1.0,
            seed=-1,
            ancestral_seed=-1,
            width=int(manifest["project"]["width"]),
            height=int(manifest["project"]["height"]),
            generation_type="studio_render",
            parameters=params,
            image_hash=saved["media_hash"],
        )
        db.add(image)
        db.flush()
        # The worker is the sole owner of a "running"/"cancel_requested" job
        # -- nothing else transitions those states except this same
        # atomic-UPDATE pattern -- but the transition is still expressed as
        # one so a job row that somehow vanished (never observed, but not
        # provably impossible) is a logged no-op instead of a
        # `NoneType has no attribute` crash after the Gallery row is already
        # committed.
        if not _transition_job_state(
            job_id, ["running", "cancel_requested"], "completed", db=db,
            progress=1.0, message="Render complete", gallery_image_id=image.id,
            filename=filename, preview_filename=preview_filename, finished_at=_now(),
            warnings=saved.get("warnings") or [],
        ):
            print(f"[StudioRender] job {job_id} completed but its row could not be updated to 'completed'")
    except StudioRenderCancelled:
        db.rollback()
        if not _transition_job_state(
            job_id, ["running", "cancel_requested"], "cancelled", db=db,
            message="Render cancelled", error=None, finished_at=_now(),
        ):
            print(f"[StudioRender] job {job_id} cancelled but its row could not be updated to 'cancelled'")
        if temp_output and os.path.exists(temp_output):
            os.remove(temp_output)
        if persisted_filename:
            _remove_persisted_output(persisted_filename)
    except Exception as exc:
        db.rollback()
        print(f"[StudioRender] job {job_id} failed: {exc}")
        if not _transition_job_state(
            job_id, ["running", "cancel_requested"], "failed", db=db,
            message="Render failed",
            error=(
                str(exc)[:1000]
                if isinstance(exc, StudioRenderValidationError)
                else "Studio render failed. Check the backend log for details."
            ),
            finished_at=_now(),
        ):
            print(f"[StudioRender] job {job_id} failed but its row could not be updated to 'failed'")
        if temp_output and os.path.exists(temp_output):
            os.remove(temp_output)
        if persisted_filename:
            _remove_persisted_output(persisted_filename)
    finally:
        with _process_lock:
            _cancel_events.pop(job_id, None)
        db.close()
        if staging_dir:
            shutil.rmtree(staging_dir, ignore_errors=True)


def _remove_persisted_output(filename: str) -> None:
    """Remove all artifacts created for a render whose Gallery insert failed."""
    if not _safe_output_name(filename):
        return
    base = Path(filename).stem
    paths = [
        os.path.join(settings.outputs_dir, filename),
        os.path.join(settings.outputs_dir, f"{base}.png"),
        os.path.join(settings.outputs_dir, f"{base}.json"),
        os.path.join(settings.thumbnails_dir, f"{base}.png"),
        os.path.join(settings.thumbnails_dir, f"{base}.webp"),
    ]
    for path in paths:
        if _inside(settings.outputs_dir, path) or _inside(settings.thumbnails_dir, path):
            try:
                os.remove(path)
            except OSError:
                pass


def submit_render_job(job_id: str) -> None:
    _render_executor.submit(_render_worker, job_id)


def request_cancel_render_job(job_id: str) -> Optional[str]:
    """Cancel a queued or running Studio render.

    Both branches below use `_transition_job_state`'s single
    UPDATE ... WHERE state = ? statement instead of a read-then-write, and
    the state-mutating action that follows a successful transition (staging
    cleanup, or killing the ffmpeg process) is gated on that same
    transition having actually happened:

    - queued -> cancelled: `cleanup_render_staging()` only runs if the
      UPDATE matched a row that was still "queued" in the same statement
      that wrote "cancelled". If `_render_worker` had already claimed the
      job (queued -> running) by the time this runs, the UPDATE here
      matches zero rows and staging is left untouched -- the worker's own
      files, mid-render, are never deleted out from under it.
    - running -> cancel_requested: only a job that was actually still
      "running" at the moment of the UPDATE reaches the `process.kill()`
      call, so this cannot fire against a process that already exited
      (which would otherwise leave `cancel_requested` permanently stuck --
      see the SQLITE_BUSY_SNAPSHOT scenario in the Studio render audit).
    """
    db = GallerySessionLocal()
    try:
        if _transition_job_state(
            job_id, ["queued"], "cancelled", db=db,
            message="Render cancelled", finished_at=_now(),
        ):
            cleanup_render_staging(job_id)
            return "cancelled"

        if _transition_job_state(
            job_id, ["running"], "cancel_requested", db=db,
            message="Cancelling render",
        ):
            with _process_lock:
                process = _active_processes.get(job_id)
                cancel_event = _cancel_events.get(job_id)
            if cancel_event is not None:
                cancel_event.set()
            if process and process.poll() is None:
                process.kill()
            return "cancel_requested"

        job = db.query(StudioRenderJob).filter(StudioRenderJob.id == job_id).first()
        if not job:
            return None
        return job.state
    finally:
        db.close()


def get_render_job(job_id: str, db: Session) -> Optional[Dict[str, Any]]:
    job = db.query(StudioRenderJob).filter(StudioRenderJob.id == job_id).first()
    if not job:
        return None
    result = job.to_dict()
    if job.gallery_image_id:
        image = db.query(GeneratedImage).filter(GeneratedImage.id == job.gallery_image_id).first()
        if image:
            result["image"] = image.to_dict()
    return result


def cleanup_render_staging(job_id: str) -> None:
    root = os.path.join(settings.cache_dir, "studio_render_jobs")
    target = os.path.join(root, job_id)
    if _inside(root, target):
        shutil.rmtree(target, ignore_errors=True)


def reap_stale_render_jobs(db: Session) -> None:
    """Reclaim `running`/`cancel_requested` rows this process is not
    actually backing with a subprocess.

    The single-render-slot gate in `routes.py` treats any row in one of
    these two states as "busy" and rejects every new submission with a 409.
    Under the fixed atomic transitions (see `_transition_job_state`) a row
    should never get stuck here -- but a stuck row was exactly the
    observed failure mode of the read-then-write races this replaces, and a
    permanently-wedged single slot (recoverable only by a backend restart)
    is bad enough to defend twice. `_active_processes`/`_cancel_events` are
    populated for the whole lifetime of a job this process is rendering, so
    a "running"/"cancel_requested" row this process has no record of, whose
    `started_at` is old enough to rule out "still in the setup window
    before Popen()", is reclassified as failed instead of blocking the gate
    forever.
    """
    from datetime import timedelta

    stale_before = _now() - timedelta(seconds=float(STUDIO_RENDER_DEFAULTS["max_render_seconds"]) * 2 + 60)
    try:
        candidates = (
            db.query(StudioRenderJob)
            .filter(
                StudioRenderJob.state.in_(["running", "cancel_requested"]),
                StudioRenderJob.started_at.isnot(None),
                StudioRenderJob.started_at < stale_before,
            )
            .all()
        )
    except OperationalError:
        return
    for job in candidates:
        with _process_lock:
            has_process = job.id in _active_processes or job.id in _cancel_events
        if has_process:
            continue
        _transition_job_state(
            job.id, ["running", "cancel_requested"], "failed", db=db,
            message="Render failed",
            error="Studio render job was stuck without an active renderer process and was reclaimed.",
            finished_at=_now(),
        )
