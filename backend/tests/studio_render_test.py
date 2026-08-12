"""Pure validation and filtergraph tests for Studio timeline rendering."""

from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _path in (_REPO, _BACKEND):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from api.studio_render_jobs import (  # noqa: E402
    StudioRenderValidationError,
    _canonical_manifest,
    build_render_command,
)


def _manifest(*, duration=2.0, fps=24.0, clip_duration=1.0, presentation="frame"):
    return {
        "project": {"id": "project-1", "revision": 2, "duration": duration, "fps": fps, "width": 320, "height": 240},
        "render": {"audio_enabled": False, "fit_mode": "contain"},
        "assets": [{"id": "image-1", "kind": "image", "name": "still.png"}],
        "tracks": [{"id": "video-1", "kind": "video", "visible": True, "muted": False}],
        "clips": [{
            "id": "clip-1", "assetId": "image-1", "trackId": "video-1",
            "start": 0, "duration": clip_duration, "sourceIn": 0,
            "presentation": presentation, "activeTake": True,
        }],
    }


def test_manifest_quantizes_timeline_to_output_frames():
    manifest = _canonical_manifest(_manifest(duration=2.01, fps=24, clip_duration=1.01, presentation="hold"))
    assert manifest["project"]["duration_frames"] == 48
    assert manifest["project"]["duration"] == 2.0
    assert manifest["clips"][0]["duration_frames"] == 24


def test_still_must_be_one_frame_or_explicit_hold():
    with pytest.raises(StudioRenderValidationError, match="presentation='hold'"):
        _canonical_manifest(_manifest(clip_duration=2.0, presentation="frame"))


def test_inactive_take_is_not_rendered():
    raw = _manifest()
    raw["clips"][0]["activeTake"] = False
    manifest = _canonical_manifest(raw)
    assert manifest["clips"] == []


def test_filtergraph_uses_argv_and_timeline_overlay(tmp_path):
    image = tmp_path / "still.png"
    image.write_bytes(b"not decoded by graph builder")
    manifest = _canonical_manifest(_manifest(clip_duration=1.0, presentation="hold"))
    manifest["assets"][0]["staged_name"] = "still.png"
    command = build_render_command(manifest, str(tmp_path), "ffmpeg.exe", str(tmp_path / "out.mp4"))
    assert command[0] == "ffmpeg.exe"
    assert "-filter_complex" in command
    graph = command[command.index("-filter_complex") + 1]
    assert "trim=start=0.000000:duration=1.000000" in graph
    assert "overlay=eof_action=pass" in graph
    assert "-progress" in command


def test_video_clip_cannot_extend_past_probed_source():
    raw = _manifest()
    raw["assets"][0] = {"id": "video-1", "kind": "video", "duration": 0.5}
    raw["clips"][0].update({"assetId": "video-1", "duration": 1.0, "presentation": "clip"})
    with pytest.raises(StudioRenderValidationError, match="source duration"):
        _canonical_manifest(raw)
