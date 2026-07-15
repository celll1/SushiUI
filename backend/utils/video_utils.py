"""Video I/O helpers for LTX-2.3 (and future video architectures).

ENCODE side (unchanged): encodes generated frames to an H.264 mp4 (yuv420p)
using the system ffmpeg binary and, when present, muxes the generated audio
track. Also writes a sidecar JSON with the full parameter set and a poster PNG
(middle frame) so the existing thumbnail path can produce gallery thumbnails.

DECODE side (video temporal outpaint, Phase 2): `load_video_frames` decodes an
uploaded clip to raw RGB frames, `extract_audio_stream` pulls its original
audio track (if any) as WAV bytes, `extract_audio_window` trims+resamples a
window of that WAV to match a target sample rate/channel layout, and
`mux_audio_over_span` splices that window back into a whole-timeline generated
audio track with a crossfade confined to the generated side of each boundary.

ffmpeg (+ ffprobe) is invoked via subprocess (PyAV / imageio-ffmpeg are not
installed). Raw RGB frames are streamed to/from ffmpeg's stdin/stdout (no
per-frame temp PNGs).
"""

import os
import io
import json
import shutil
import glob
import tempfile
import time
import wave
import subprocess
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config.settings import settings


def _locate_ffmpeg() -> str:
    """Return an absolute path to an ffmpeg executable.

    Prefers PATH; falls back to known local build locations under /d/ffmpeg-*.
    Raises RuntimeError if none is found.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe

    candidates = []
    for root in ("D:/", "C:/"):
        candidates.extend(glob.glob(os.path.join(root, "ffmpeg-*", "bin", "ffmpeg.exe")))
        candidates.extend(glob.glob(os.path.join(root, "ffmpeg-*", "bin", "ffmpeg")))
    for c in candidates:
        if os.path.exists(c):
            return c

    raise RuntimeError(
        "ffmpeg executable not found. Install ffmpeg and ensure it is on PATH "
        "(or under D:/ffmpeg-*/bin)."
    )


def _write_wav(audio, audio_sample_rate: int, wav_path: str) -> int:
    """Write an audio tensor/array [channels, samples] to a 16-bit PCM wav.

    Returns the number of channels written.
    """
    # Accept torch tensors or numpy arrays.
    if hasattr(audio, "detach"):
        arr = audio.detach().float().cpu().numpy()
    else:
        arr = np.asarray(audio, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr[None, :]  # [1, samples]
    # Expect [channels, samples]; if it looks transposed ([samples, channels]) fix it.
    if arr.shape[0] > arr.shape[1] and arr.shape[1] in (1, 2):
        arr = arr.T

    num_channels = int(arr.shape[0])
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).round().astype(np.int16)  # [channels, samples]
    interleaved = pcm.T.reshape(-1)  # [samples * channels]

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(int(audio_sample_rate))
        wf.writeframes(interleaved.tobytes())

    return num_channels


def save_video_with_metadata(
    frames: np.ndarray,
    audio,
    audio_sample_rate: Optional[int],
    params: Dict[str, Any],
    generation_type: str,
    model_info: Optional[Dict[str, Any]] = None,
    lossless: bool = False,
) -> str:
    """Encode frames (+ optional audio) to an mp4 and write a metadata sidecar.

    Args:
        frames: np.uint8 array [T, H, W, 3] (RGB, 0-255).
        audio: torch.FloatTensor / np.ndarray [channels, samples], or None.
        audio_sample_rate: audio sampling rate (Hz) or None.
        params: generation parameters (seed already resolved by the caller).
        generation_type: e.g. "txt2vid".
        model_info: pipeline_manager.current_model_info (source/model_hash/type).
        lossless: when True, encode video with FFV1 (`-pix_fmt rgb24`) instead
            of libx264, and audio (when present) with FLAC instead of AAC.

            Empirically verified (see the video-outpaint Phase 2 audit): with
            this ffmpeg build, `libx264 -qp 0 -pix_fmt yuv444p/gbrp` is CLOSE
            but NOT bit-exact after an RGB->encode->decode->RGB roundtrip
            (observed max per-channel diff of 2/255, from swscale's RGB<->YUV
            /RGB<->GBR rounding) -- despite being commonly described as
            "lossless" x264 settings. FFV1 with `-pix_fmt rgb24` (no forced
            colorspace conversion) gave an EXACT roundtrip (maxdiff 0) in the
            same test, and IS accepted by ffmpeg's mp4 muxer. `lossless=True`
            therefore uses FFV1, not libx264 -qp 0.

            Trade-off: the resulting mp4 is NOT H.264 and will generally NOT
            play back in a browser's native <video> element (FFV1 has no
            mainstream browser decoder) -- this mode is for archival/
            verification of the exact preserved frames, not casual playback.
            File size is also much larger than H.264 (near-raw).

            Audio: FLAC does not introduce ANY additional lossy compression
            beyond the existing float32->int16 PCM quantization this module
            always performs in `_write_wav` -- i.e. "lossless audio" here
            means "no further loss on top of that 16-bit quantization", not
            that the audio is preserved beyond 16-bit precision.

    Returns:
        The mp4 filename (basename, relative to settings.outputs_dir).
    """
    os.makedirs(settings.outputs_dir, exist_ok=True)

    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames of shape [T, H, W, 3], got {frames.shape}")

    num_frames, height, width, _ = frames.shape
    frame_rate = float(params.get("frame_rate", 24.0)) or 24.0
    seed = int(params.get("seed", 0))
    audio_enable = bool(params.get("audio_enable", True)) and audio is not None

    ts = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"{generation_type}_{ts}_{seed}"
    mp4_name = f"{base_name}.mp4"
    mp4_path = os.path.join(settings.outputs_dir, mp4_name)

    ffmpeg = _locate_ffmpeg()

    # ffmpeg metadata tags (kept factual; values are strings)
    model_name = ""
    model_hash = ""
    if model_info:
        model_name = os.path.basename(str(model_info.get("source", ""))) or model_info.get("type", "")
        model_hash = str(model_info.get("model_hash", ""))

    duration_s = num_frames / frame_rate if frame_rate else 0.0
    meta_tags = {
        "title": base_name,
        "comment": (params.get("prompt", "") or "")[:1000],
        "generation_type": generation_type,
        "seed": str(seed),
        "num_frames": str(num_frames),
        "frame_rate": str(frame_rate),
        "model_name": model_name,
    }

    wav_path = os.path.join(settings.outputs_dir, f"{base_name}.wav")
    audio_written = False
    if audio_enable and audio_sample_rate:
        try:
            _write_wav(audio, audio_sample_rate, wav_path)
            audio_written = True
        except Exception as e:
            print(f"[VideoSave] audio wav write failed ({e}); encoding video without audio")
            audio_written = False

    # Build ffmpeg command: raw RGB frames from stdin, optional audio from wav.
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", f"{frame_rate}",
        "-i", "-",
    ]
    if audio_written:
        cmd += ["-i", wav_path]

    if lossless:
        # FFV1 -pix_fmt rgb24: no forced RGB<->YUV/GBR colorspace conversion,
        # empirically verified bit-exact after decode (see docstring). NOT
        # H.264 -- not browser-playable, much larger file size.
        cmd += ["-c:v", "ffv1", "-pix_fmt", "rgb24"]
    else:
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd += ["-movflags", "+faststart"]
    if audio_written:
        # No -shortest: video is authoritative. The vocoder audio can be
        # slightly shorter than the video (its temporal grid differs), and
        # -shortest would trim video frames down to the audio length.
        cmd += ["-c:a", "flac" if lossless else "aac"]
    for k, v in meta_tags.items():
        cmd += ["-metadata", f"{k}={v}"]
    cmd += [mp4_path]

    proc = subprocess.run(
        cmd,
        input=frames.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[-2000:]
        # Clean up partial artifacts before surfacing the error.
        if audio_written and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
        raise RuntimeError(f"ffmpeg video encode failed (code {proc.returncode}):\n{err}")

    if audio_written and os.path.exists(wav_path):
        try:
            os.remove(wav_path)
        except OSError:
            pass

    # Poster PNG (middle frame) so the existing thumbnail path can run.
    try:
        from PIL import Image
        mid = frames[num_frames // 2]
        poster_path = os.path.join(settings.outputs_dir, f"{base_name}.png")
        Image.fromarray(mid, mode="RGB").save(poster_path, format="PNG")
    except Exception as e:
        print(f"[VideoSave] poster frame write failed ({e})")

    # Sidecar JSON with the full parameter set.
    sidecar = {
        "generation_type": generation_type,
        "filename": mp4_name,
        "prompt": params.get("prompt", ""),
        "negative_prompt": params.get("negative_prompt", ""),
        "model_name": model_name,
        "model_hash": model_hash,
        "seed": seed,
        "num_frames": num_frames,
        "fps": frame_rate,
        "width": width,
        "height": height,
        "num_inference_steps": params.get("num_inference_steps"),
        "guidance_scale": params.get("guidance_scale"),
        "audio_enable": bool(audio_written),
        "audio_sample_rate": audio_sample_rate if audio_written else None,
        "duration": duration_s,
        "lossless": bool(lossless),
    }
    try:
        with open(os.path.join(settings.outputs_dir, f"{base_name}.json"), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[VideoSave] sidecar json write failed ({e})")

    return mp4_name


# ---------------------------------------------------------------------------
# DECODE side (video temporal outpaint, Phase 2)
# ---------------------------------------------------------------------------

def load_video_frames(
    video_bytes: bytes,
    max_frames: Optional[int] = None,
    trim_end_frames: int = 0,
    timeout: float = 300.0,
) -> Tuple[np.ndarray, float]:
    """Decode an uploaded video clip to raw RGB frames via ffmpeg subprocess.

    Args:
        video_bytes: The raw bytes of an uploaded video file (any container
            ffmpeg can demux -- mp4/webm/mkv/mov/...).
        max_frames: Optional cap on the number of frames NEEDED FROM THE
            START of the clip (e.g. `input_trim_start_frames + total_frames`
            for video outpaint) -- passed to ffmpeg as `-frames:v` so it
            stops decoding early instead of decoding the whole (possibly
            arbitrarily long) upload into RAM before the caller's fit check.
        trim_end_frames: If > 0, the caller intends to trim this many frames
            off the clip's OWN end (see `input_trim_end_frames`). Since a
            tail-trim needs the TRUE end of the clip, `max_frames` is widened
            (using the ffprobe-probed frame count -- no full decode required
            for that estimate) to cover the whole clip in this case, so the
            RAM-bounding above only applies when there is no tail trim.
        timeout: seconds before the ffmpeg decode subprocess is killed.

    Returns:
        (frames, fps) where frames is np.uint8 [T, H, W, 3] (RGB) and fps is
        the clip's own frame rate (probed via ffprobe; `r_frame_rate` falling
        back to `avg_frame_rate`, defaulting to 24.0 if neither is usable).

    Raises:
        RuntimeError: if ffmpeg/ffprobe are missing, the clip has no probeable
            video stream, the decode subprocess fails, or times out.
    """
    from utils.dataset_scanner import probe_video_metadata

    ffmpeg = _locate_ffmpeg()

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".input_video")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(video_bytes)

        meta = probe_video_metadata(tmp_path)
        if not meta:
            raise RuntimeError(
                "Could not probe the uploaded video (no readable video stream, "
                "or ffprobe is not installed)."
            )
        width, height = meta["width"], meta["height"]
        fps = float(meta.get("fps") or 0.0)
        if fps <= 0:
            fps = 24.0
            print("[VideoLoad] clip reported no usable frame rate; assuming 24.0 fps")

        decode_limit = max_frames
        if trim_end_frames and trim_end_frames > 0:
            probed_num_frames = int(meta.get("num_frames") or 0)
            if probed_num_frames > 0:
                decode_limit = probed_num_frames if decode_limit is None else max(decode_limit, probed_num_frames)
            else:
                # Probe couldn't estimate a frame count -- cannot safely bound
                # the decode when a tail trim is requested; decode everything.
                decode_limit = None

        cmd = [
            ffmpeg, "-y",
            "-i", tmp_path,
            "-map", "0:v:0",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
        ]
        if decode_limit is not None and decode_limit > 0:
            cmd += ["-frames:v", str(int(decode_limit))]
        cmd += ["-"]

        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffmpeg video decode timed out after {timeout}s")
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"ffmpeg video decode failed (code {proc.returncode}):\n{err}")

        frame_size = width * height * 3
        raw = proc.stdout
        num_frames = len(raw) // frame_size
        if num_frames <= 0:
            raise RuntimeError("ffmpeg decoded zero frames from the uploaded video")

        frames = np.frombuffer(raw[: num_frames * frame_size], dtype=np.uint8)
        frames = frames.reshape(num_frames, height, width, 3).copy()
        return frames, fps
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def extract_audio_stream(video_bytes: bytes, timeout: float = 60.0) -> Optional[bytes]:
    """Extract a video clip's original audio track as PCM16 WAV bytes.

    No resampling is applied (the wav muxer's default codec is pcm_s16le but
    the sample rate/channel layout are left at the source's native values) --
    resampling to a specific target only happens later, in
    `extract_audio_window`, once the target (generated audio) sample rate is
    known.

    Returns:
        WAV bytes, or None if the clip has no audio stream (or extraction
        fails for any other reason) -- callers must gracefully fall back to
        "regenerate" audio mode in that case.
    """
    ffmpeg = _locate_ffmpeg()
    from utils.dataset_scanner import _find_ffprobe

    ffprobe = _find_ffprobe()
    if not ffprobe:
        print("[VideoLoad] ffprobe not found; cannot check for an audio stream")
        return None

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".input_video")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(video_bytes)

        probe_cmd = [
            ffprobe, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=index", "-of", "csv=p=0", tmp_path,
        ]
        try:
            probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            print("[VideoLoad] audio stream probe timed out; assuming no audio")
            return None
        if probe.returncode != 0 or not probe.stdout.strip():
            print("[VideoLoad] uploaded clip has no audio stream")
            return None

        cmd = [
            ffmpeg, "-y", "-i", tmp_path,
            "-map", "0:a:0", "-vn",
            "-f", "wav", "-",
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"[VideoLoad] audio extraction timed out after {timeout}s; falling back to no-audio")
            return None
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode("utf-8", errors="replace")[-500:]
            print(f"[VideoLoad] audio extraction failed ({err}); falling back to no-audio")
            return None
        return proc.stdout
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _atempo_filter_chain(factor: float) -> str:
    """Decompose an arbitrary tempo `factor` into a chain of ffmpeg `atempo`
    filters (each individual `atempo` instance only supports [0.5, 2.0]).

    `factor` follows ffmpeg's own convention: output_duration =
    input_duration / factor (factor > 1 speeds up / shortens, factor < 1
    slows down / lengthens). Pitch-preserving (WSOLA-based), unlike a naive
    resample-rate change.
    """
    if factor <= 0:
        return "atempo=1.0"
    filters = []
    f = factor
    while f > 2.0:
        filters.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        filters.append("atempo=0.5")
        f /= 0.5
    filters.append(f"atempo={f:.6f}")
    return ",".join(filters)


def _read_wav_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode PCM16 WAV bytes to a float32 [channels, samples] array in [-1, 1]."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    pcm = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).T  # [channels, samples]
    else:
        pcm = pcm.reshape(1, -1)
    arr = (pcm.astype(np.float32) / 32767.0)
    return arr, sample_rate


def _fit_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Truncate or edge-pad `arr` (shape [channels, samples]) to exactly `target_len` samples."""
    cur_len = arr.shape[1]
    if cur_len == target_len:
        return arr
    if cur_len > target_len:
        return arr[:, :target_len]
    pad = target_len - cur_len
    if cur_len == 0:
        return np.zeros((arr.shape[0], target_len), dtype=arr.dtype)
    return np.pad(arr, ((0, 0), (0, pad)), mode="edge")


def extract_audio_window(
    wav_bytes: bytes,
    start_sec: float,
    src_dur_sec: float,
    target_dur_sec: float,
    sample_rate: int,
    channels: int,
    timeout: float = 30.0,
) -> Optional[np.ndarray]:
    """Trim `[start_sec, start_sec + src_dur_sec)` out of a WAV byte stream
    (SOURCE-clip real time), time-stretch it (pitch-preserving `atempo`) to
    `target_dur_sec` if the two durations differ non-negligibly, then
    resample/channel-match to the target layout.

    Splitting `src_dur_sec` (how much of the source to read) from
    `target_dur_sec` (how long the returned window must represent) handles
    the case where the input clip's own fps differs from the OUTPUT video's
    frame_rate: the pasted frames occupy `target_dur_sec` seconds of the
    OUTPUT timeline but were captured over `src_dur_sec` seconds of SOURCE
    real time -- splicing the untouched source audio in at its native tempo
    would drift out of sync with the frame-for-frame placed video.

    Uses `-i - ... -ss -t` (OUTPUT seeking after `-i`, since the input
    arrives on a non-seekable pipe) -- sample-accurate for PCM regardless
    (no keyframe snapping, unlike video).

    Returns:
        np.float32 array of shape [channels, round(target_dur_sec * sample_rate)],
        or **None** on ANY ffmpeg step failure/timeout. Never returns silence
        on failure -- callers MUST treat None as "skip the splice, keep the
        regenerated audio untouched" (silence would be a worse outcome).
    """
    ffmpeg = _locate_ffmpeg()
    start_sec = max(0.0, start_sec)
    src_dur_sec = max(0.0, src_dur_sec)
    target_dur_sec = max(0.0, target_dur_sec)
    if src_dur_sec <= 0 or target_dur_sec <= 0:
        return None

    # ---- Step 1: trim [start_sec, start_sec + src_dur_sec) at native rate. ----
    trim_cmd = [
        ffmpeg, "-y", "-i", "-",
        "-ss", f"{start_sec:.6f}", "-t", f"{src_dur_sec:.6f}",
        "-f", "wav", "-",
    ]
    try:
        proc = subprocess.run(trim_cmd, input=wav_bytes, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("[VideoLoad] audio window trim timed out")
        return None
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", errors="replace")[-500:]
        print(f"[VideoLoad] audio window trim failed ({err})")
        return None
    stage = proc.stdout

    # ---- Step 2: pitch-preserving time-stretch if src/target durations differ. ----
    factor = (src_dur_sec / target_dur_sec) if target_dur_sec > 0 else 1.0
    if abs(factor - 1.0) > 0.005:
        atempo_cmd = [
            ffmpeg, "-y", "-i", "-",
            "-af", _atempo_filter_chain(factor),
            "-f", "wav", "-",
        ]
        try:
            proc = subprocess.run(atempo_cmd, input=stage, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.TimeoutExpired:
            print("[VideoLoad] audio window time-stretch timed out")
            return None
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode("utf-8", errors="replace")[-500:]
            print(f"[VideoLoad] audio window time-stretch failed ({err})")
            return None
        stage = proc.stdout

    # ---- Step 3: resample + channel-match to the target layout. ----
    resample_cmd = [
        ffmpeg, "-y", "-i", "-",
        "-ar", str(int(sample_rate)), "-ac", str(int(channels)),
        "-f", "wav", "-",
    ]
    try:
        proc = subprocess.run(resample_cmd, input=stage, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("[VideoLoad] audio window resample timed out")
        return None
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", errors="replace")[-500:]
        print(f"[VideoLoad] audio window resample failed ({err})")
        return None

    arr, _sr = _read_wav_bytes(proc.stdout)
    if arr.shape[0] != channels:
        # Defensive: -ac above should already guarantee this, but re-fit if not.
        if arr.shape[0] > channels:
            arr = arr[:channels]
        else:
            arr = np.pad(arr, ((0, channels - arr.shape[0]), (0, 0)), mode="edge")

    target_len = max(0, int(round(target_dur_sec * sample_rate)))
    return _fit_length(arr, target_len)


def mux_audio_over_span(
    generated_audio: np.ndarray,
    input_audio: np.ndarray,
    offset_sec: float,
    dur_sec: float,
    sample_rate: int,
    crossfade_ms: float = 50.0,
) -> np.ndarray:
    """Overwrite [offset_sec, offset_sec + dur_sec) of `generated_audio` with
    `input_audio`, byte/sample-exact, with a short crossfade confined to the
    GENERATED side of each boundary.

    Unlike `AceStepMixin._acestep_apply_repaint_waveform_splice` (whose
    crossfade ramps bleed INTO the region it is trying to keep untouched --
    fine there because the "kept" region is the OUTSIDE), this function's
    preserved region is the INSIDE of [offset_sec, offset_sec + dur_sec), so
    the ramp zones are placed strictly OUTSIDE that window
    ([offset-crossfade, offset) and [offset+dur, offset+dur+crossfade)) and
    blend the GENERATED samples there towards a constant hold of `input_audio`'s
    boundary sample -- every sample inside the window is copied from
    `input_audio` untouched.

    Args:
        generated_audio: np.float32 [channels, samples] -- the whole-timeline
            generated audio track (e.g. the LTX-2.3 vocoder output).
        input_audio: np.float32 [channels, round(dur_sec * sample_rate)] --
            the exact segment to splice in (see `extract_audio_window`).
        offset_sec/dur_sec: placement window in `generated_audio`'s timeline.
        sample_rate: sample rate shared by both arrays.
        crossfade_ms: crossfade length in milliseconds (default 50ms).

    Returns:
        A new np.float32 [channels, samples] array (input arrays untouched).
    """
    channels = generated_audio.shape[0]
    total_samples = generated_audio.shape[1]

    start_sample = int(round(offset_sec * sample_rate))
    core_len = int(round(dur_sec * sample_rate))
    start_sample = max(0, min(start_sample, total_samples))
    end_sample = max(start_sample, min(start_sample + core_len, total_samples))
    core_len_eff = end_sample - start_sample

    result = generated_audio.copy()
    if core_len_eff <= 0:
        return result

    input_matched = _fit_length(input_audio, core_len_eff)
    if input_matched.shape[0] != channels:
        if input_matched.shape[0] > channels:
            input_matched = input_matched[:channels]
        else:
            input_matched = np.pad(input_matched, ((0, channels - input_matched.shape[0]), (0, 0)), mode="edge")

    result[:, start_sample:end_sample] = input_matched

    crossfade_samples = int(round(crossfade_ms / 1000.0 * sample_rate))
    if crossfade_samples > 0:
        # Leading ramp -- strictly BEFORE start_sample (generated-side only).
        fade_start = max(0, start_sample - crossfade_samples)
        ramp_len = start_sample - fade_start
        if ramp_len > 0:
            alpha = np.linspace(0.0, 1.0, ramp_len + 2, dtype=np.float32)[1:-1]
            hold = input_matched[:, :1]
            gen_slice = result[:, fade_start:start_sample]
            result[:, fade_start:start_sample] = (1.0 - alpha)[None, :] * gen_slice + alpha[None, :] * hold

        # Trailing ramp -- strictly AFTER end_sample (generated-side only).
        fade_end = min(total_samples, end_sample + crossfade_samples)
        ramp_len = fade_end - end_sample
        if ramp_len > 0:
            alpha = np.linspace(1.0, 0.0, ramp_len + 2, dtype=np.float32)[1:-1]
            hold = input_matched[:, -1:]
            gen_slice = result[:, end_sample:fade_end]
            result[:, end_sample:fade_end] = alpha[None, :] * hold + (1.0 - alpha)[None, :] * gen_slice

    return result
