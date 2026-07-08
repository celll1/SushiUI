"""Video output helpers for LTX-2.3 (and future video architectures).

Encodes generated frames to an H.264 mp4 (yuv420p) using the system ffmpeg
binary and, when present, muxes the generated audio track. Also writes a
sidecar JSON with the full parameter set and a poster PNG (middle frame) so the
existing thumbnail path can produce gallery thumbnails.

ffmpeg is invoked via subprocess (PyAV / imageio-ffmpeg are not installed).
Raw RGB frames are streamed to ffmpeg's stdin (no per-frame temp PNGs).
"""

import os
import json
import shutil
import glob
import time
import wave
import subprocess
from typing import Any, Dict, Optional

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
) -> str:
    """Encode frames (+ optional audio) to an mp4 and write a metadata sidecar.

    Args:
        frames: np.uint8 array [T, H, W, 3] (RGB, 0-255).
        audio: torch.FloatTensor / np.ndarray [channels, samples], or None.
        audio_sample_rate: audio sampling rate (Hz) or None.
        params: generation parameters (seed already resolved by the caller).
        generation_type: e.g. "txt2vid".
        model_info: pipeline_manager.current_model_info (source/model_hash/type).

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

    cmd += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    if audio_written:
        # No -shortest: video is authoritative. The vocoder audio can be
        # slightly shorter than the video (its temporal grid differs), and
        # -shortest would trim video frames down to the audio length.
        cmd += ["-c:a", "aac"]
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
    }
    try:
        with open(os.path.join(settings.outputs_dir, f"{base_name}.json"), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[VideoSave] sidecar json write failed ({e})")

    return mp4_name
