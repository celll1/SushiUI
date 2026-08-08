"""Regression test for the lossless-video container/proxy fix.

Before the fix, `lossless=True` wrote FFV1 into an `.mp4` container: FFV1 has
no mainstream browser decoder AND `.mp4` is the wrong container for it, so
every such file was unplayable in the gallery. The fix: the master is
FFV1-in-`.mkv` (byte-exact, download-only), and a second H.264 `.mp4` proxy
is encoded from the SAME source frames for gallery playback -- NOT
transcoded from the master (that would be a second lossy hop on top of the
proxy's own compression, for no benefit).

`subprocess.run` is monkeypatched (no real ffmpeg needed) -- this test checks
the COMMANDS ffmpeg is asked to run and the returned filenames, not actual
pixels; the byte-exact roundtrip itself is documented as empirically verified
in `video_utils.save_video_with_metadata`'s docstring.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import video_utils  # noqa: E402


class _FakeProc:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.stdout = b""
        self.stderr = b""


@pytest.fixture
def patched_env(monkeypatch, tmp_path):
    """Patch outputs_dir and ffmpeg lookup only; subprocess.run is left to
    each test so failure-branch tests can vary its behaviour per call."""
    monkeypatch.setattr(video_utils.settings, "outputs_dir", str(tmp_path))
    monkeypatch.setattr(video_utils, "_locate_ffmpeg", lambda: "ffmpeg")
    return tmp_path


def _frames():
    return np.zeros((3, 4, 4, 3), dtype=np.uint8)


def test_non_lossless_writes_single_mp4_no_proxy(monkeypatch, patched_env):
    calls = []
    monkeypatch.setattr(video_utils.subprocess, "run", lambda cmd, **kw: (calls.append(cmd), _FakeProc(0))[1])

    filename, preview_filename = video_utils.save_video_with_metadata(
        _frames(), None, None, {"frame_rate": 24.0, "seed": 1}, "test_vid",
    )
    assert filename.endswith(".mp4")
    assert preview_filename is None
    assert len(calls) == 1
    assert "libx264" in calls[0]


def test_lossless_master_is_mkv_with_h264_proxy_from_source_frames(monkeypatch, patched_env):
    calls = []
    monkeypatch.setattr(video_utils.subprocess, "run", lambda cmd, **kw: (calls.append(cmd), _FakeProc(0))[1])

    filename, preview_filename = video_utils.save_video_with_metadata(
        _frames(), None, None, {"frame_rate": 24.0, "seed": 1}, "test_vid",
        lossless=True,
    )
    # The master is FFV1-in-mkv, never .mp4 -- FFV1 belongs in mkv, not mp4,
    # and is undecodable by any browser regardless of container.
    assert filename.endswith(".mkv")
    assert not filename.endswith(".mp4")

    # A separate browser-playable proxy is produced alongside it.
    assert preview_filename is not None
    assert preview_filename.endswith(".mp4")
    assert preview_filename != filename

    assert len(calls) == 2
    master_cmd, proxy_cmd = calls
    assert "ffv1" in master_cmd
    assert "libx264" not in master_cmd
    assert "libx264" in proxy_cmd
    assert "ffv1" not in proxy_cmd
    # The proxy's OWN output path must be the mp4, not the mkv master path.
    assert proxy_cmd[-1].endswith(".mp4")
    assert master_cmd[-1].endswith(".mkv")

    # The proxy is encoded from the SAME raw frames fed on stdin ("-i -"), NOT
    # transcoded from the master file: a mutant that pointed the proxy's input
    # at the master path instead would still produce a playable .mp4 and pass
    # every assertion above, so this checks the actual ffmpeg input directly.
    assert "rawvideo" in proxy_cmd
    dash_i_index = proxy_cmd.index("-i")
    assert proxy_cmd[dash_i_index + 1] == "-"
    assert filename not in proxy_cmd  # master's own filename never appears as a proxy argv token


def test_lossless_proxy_failure_falls_back_to_master_only(monkeypatch, patched_env):
    """Master encode (1st ffmpeg call) succeeds; proxy encode (2nd call)
    fails and leaves a partial file -- master is still returned intact,
    preview_filename is None, and the partial proxy file is removed."""
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        with open(cmd[-1], "wb") as f:
            f.write(b"partial" if len(calls) == 2 else b"master")
        if len(calls) == 2:
            return _FakeProc(returncode=1)
        return _FakeProc(returncode=0)

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    filename, preview_filename = video_utils.save_video_with_metadata(
        _frames(), None, None, {"frame_rate": 24.0, "seed": 1}, "test_vid",
        lossless=True,
    )
    assert filename.endswith(".mkv")
    assert preview_filename is None
    assert len(calls) == 2

    proxy_path = calls[1][-1]
    assert not os.path.exists(proxy_path), "partial proxy file must be unlinked on encode failure"
    assert os.path.exists(os.path.join(patched_env, filename))
