"""Audio output helpers for ACE-Step 1.5 (and future audio architectures).

Encodes a generated waveform to a lossless FLAC file and writes a sidecar
JSON with the full parameter set, mirroring `video_utils.py`'s
`save_video_with_metadata` (mp4 + poster PNG + sidecar JSON) for LTX-2.3.

Writer priority (first available wins):
  1. soundfile (libsndfile) -- direct FLAC write, no subprocess.
  2. torchaudio -- direct FLAC write via its ffmpeg/sox backend.
  3. ffmpeg (subprocess, located via `video_utils._locate_ffmpeg`) -- encodes
     a temporary 16-bit PCM WAV (via `video_utils._write_wav`) to FLAC.

A lightweight peak-envelope waveform PNG is also written next to the FLAC
(same base name) purely so the existing image-thumbnail path
(`utils.create_thumbnail`) can produce a gallery thumbnail for audio rows --
mirrors the video poster-frame PNG.
"""

import os
import json
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config.settings import settings


def _to_numpy_waveform(waveform) -> np.ndarray:
    """Normalize a torch tensor / numpy array to a [channels, samples] float32
    array clipped to [-1, 1]."""
    if hasattr(waveform, "detach"):
        arr = waveform.detach().float().cpu().numpy()
    else:
        arr = np.asarray(waveform, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr[None, :]  # [1, samples]
    # Expect [channels, samples]; if it looks transposed ([samples, channels]) fix it.
    if arr.shape[0] > arr.shape[1] and arr.shape[1] in (1, 2):
        arr = arr.T

    return np.clip(arr.astype(np.float32), -1.0, 1.0)


def _write_flac_soundfile(arr: np.ndarray, sample_rate: int, flac_path: str) -> bool:
    """Try writing FLAC via soundfile (libsndfile). Returns False (fall through to
    the next writer) if soundfile is missing OR the write fails for any reason."""
    try:
        import soundfile as sf
        # soundfile expects [samples, channels].
        sf.write(flac_path, arr.T, int(sample_rate), format="FLAC", subtype="PCM_16")
        return True
    except Exception as e:  # noqa: BLE001 - any failure -> fall through to next writer
        print(f"[AudioSave] soundfile writer unavailable/failed ({type(e).__name__}: {e}); trying next")
        return False


def _write_flac_torchaudio(arr: np.ndarray, sample_rate: int, flac_path: str) -> bool:
    """Try writing FLAC via torchaudio. Returns False (fall through to the next
    writer) if torchaudio is missing OR the write fails for any reason. NOTE: recent
    torchaudio (2.x) routes .save() through torchcodec and raises ImportError at
    CALL time (not import time) when torchcodec is absent -- so we must catch broad
    exceptions here, not just the import-time ImportError, or the ffmpeg fallback is
    never reached."""
    try:
        import torch
        import torchaudio
        tensor = torch.from_numpy(arr)
        torchaudio.save(flac_path, tensor, int(sample_rate), format="flac", bits_per_sample=16)
        return True
    except Exception as e:  # noqa: BLE001 - any failure -> fall through to ffmpeg
        print(f"[AudioSave] torchaudio writer unavailable/failed ({type(e).__name__}: {e}); trying next")
        return False


def _write_flac_ffmpeg(arr: np.ndarray, sample_rate: int, flac_path: str) -> None:
    """Encode a temp 16-bit PCM WAV to FLAC via ffmpeg. Raises on failure."""
    from utils.video_utils import _locate_ffmpeg, _write_wav

    ffmpeg = _locate_ffmpeg()
    tmp_wav = f"{flac_path}.tmp.wav"
    try:
        _write_wav(arr, sample_rate, tmp_wav)
        cmd = [ffmpeg, "-y", "-i", tmp_wav, "-c:a", "flac", flac_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"ffmpeg FLAC encode failed (code {proc.returncode}):\n{err}")
    finally:
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


def _write_waveform_png(arr: np.ndarray, png_path: str, size: Tuple[int, int] = (512, 256)) -> None:
    """Render a simple min/max peak-envelope waveform PNG (mono mixdown).

    Best-effort thumbnail seed for the gallery; not an audio analysis tool.
    """
    from PIL import Image, ImageDraw

    width, height = size
    mono = arr.mean(axis=0) if arr.shape[0] > 1 else arr[0]
    num_samples = mono.shape[0]
    if num_samples == 0:
        return

    samples_per_px = max(1, num_samples // width)
    img = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    mid = height // 2
    for x in range(width):
        start = x * samples_per_px
        end = min(start + samples_per_px, num_samples)
        if start >= end:
            break
        chunk = mono[start:end]
        peak_min = float(chunk.min())
        peak_max = float(chunk.max())
        y0 = mid - int(peak_max * (mid - 2))
        y1 = mid - int(peak_min * (mid - 2))
        draw.line([(x, y0), (x, y1)], fill=(90, 170, 250))
    img.save(png_path, format="PNG")


def save_audio_with_metadata(
    waveform,
    sample_rate: int,
    params: Dict[str, Any],
    generation_type: str,
    model_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Encode `waveform` to FLAC and write a metadata sidecar (+ waveform PNG).

    Args:
        waveform: torch.FloatTensor / np.ndarray [channels, samples], float32
            in [-1, 1] (e.g. ACE-Step's `generate_txt2aud` return value).
        sample_rate: sample rate in Hz (e.g. 48000).
        params: generation parameters (seed already resolved by the caller).
        generation_type: e.g. "txt2aud".
        model_info: pipeline_manager.current_model_info (source/model_hash/type).

    Returns:
        The FLAC filename (basename, relative to settings.outputs_dir).
    """
    os.makedirs(settings.outputs_dir, exist_ok=True)

    arr = _to_numpy_waveform(waveform)
    num_channels, num_samples = arr.shape
    sample_rate = int(sample_rate) or 48000
    duration_s = num_samples / sample_rate if sample_rate else 0.0
    seed = int(params.get("seed", 0))

    ts = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"{generation_type}_{ts}_{seed}"
    flac_name = f"{base_name}.flac"
    flac_path = os.path.join(settings.outputs_dir, flac_name)

    if _write_flac_soundfile(arr, sample_rate, flac_path):
        writer_used = "soundfile"
    elif _write_flac_torchaudio(arr, sample_rate, flac_path):
        writer_used = "torchaudio"
    else:
        _write_flac_ffmpeg(arr, sample_rate, flac_path)
        writer_used = "ffmpeg"
    print(
        f"[AudioSave] Wrote {flac_name} via {writer_used} "
        f"({num_channels}ch @ {sample_rate}Hz, {duration_s:.2f}s)"
    )

    # Waveform PNG (peak envelope) so the existing thumbnail path can produce
    # a gallery thumbnail. Best-effort; a missing thumbnail is not fatal.
    try:
        _write_waveform_png(arr, os.path.join(settings.outputs_dir, f"{base_name}.png"))
    except Exception as e:
        print(f"[AudioSave] waveform PNG write failed ({e})")

    model_name = ""
    model_hash = ""
    if model_info:
        model_name = os.path.basename(str(model_info.get("source", ""))) or model_info.get("type", "")
        model_hash = str(model_info.get("model_hash", ""))

    # Sidecar JSON with the full parameter set.
    sidecar = {
        "generation_type": generation_type,
        "filename": flac_name,
        "prompt": params.get("prompt") or params.get("caption") or "",
        "lyrics": params.get("lyrics", ""),
        "model_name": model_name,
        "model_hash": model_hash,
        "seed": seed,
        "sample_rate": sample_rate,
        "channels": num_channels,
        "duration": duration_s,
        "inference_steps": params.get("inference_steps"),
        "guidance_scale": params.get("guidance_scale"),
        "shift": params.get("shift"),
        "sampler_mode": params.get("sampler_mode"),
        "vocal_language": params.get("vocal_language"),
    }
    if "cover_strength" in params:
        sidecar["cover_strength"] = params.get("cover_strength")
    if "source_audio_hash" in params:
        sidecar["source_audio_hash"] = params.get("source_audio_hash")
    try:
        with open(os.path.join(settings.outputs_dir, f"{base_name}.json"), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[AudioSave] sidecar json write failed ({e})")

    return flac_name
