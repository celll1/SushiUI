"""
Latent Cache Management for Training

Caches VAE latents and optionally text embeddings to disk to reduce VRAM usage during training.
By default, only VAE latents are cached (text embeddings encoded on-the-fly).
Cache is stored per-dataset to allow reuse across multiple training runs.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

import torch
from PIL import Image

from core.training.image_preprocessing import TRANSPARENT_WEBP_PREPROCESSING_VERSION


def get_cache_base_dir() -> str:
    """
    Get the base cache directory from user settings.

    Priority:
    1. UserSettings.cache_dir (if set in database)
    2. settings.cache_dir (default from config/settings.py)

    Returns:
        Base cache directory path with /datasets suffix
    """
    try:
        from database import get_gallery_db
        from database.models import UserSettings

        db = next(get_gallery_db())
        try:
            user_settings = db.query(UserSettings).first()
            if user_settings and user_settings.cache_dir:
                # User configured cache directory (from database)
                return str(Path(user_settings.cache_dir) / "datasets")
        finally:
            db.close()
    except Exception as e:
        # Fallback to default if database query fails
        print(f"[Cache] Warning: Failed to get cache_dir from UserSettings: {e}")

    # Default cache directory (from config/settings.py)
    try:
        from config.settings import settings
        return str(Path(settings.cache_dir) / "datasets")
    except Exception as e:
        print(f"[Cache] Warning: Failed to get cache_dir from settings: {e}")
        # Ultimate fallback
        return "cache/datasets"


def _sanitize_ns_token(token: str) -> str:
    """Make an arbitrary string safe as a single path component."""
    s = str(token).strip().lower()
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("-")
    cleaned = "".join(out).strip("-_.")
    return cleaned or "x"


def build_cache_namespace(
    arch: str,
    vae_type: Optional[str] = None,
    te_type: Optional[str] = None,
    latent_channels: Optional[int] = None,
    latent_dtype: Optional[str] = None,
) -> str:
    """
    Build a cache namespace token that isolates latent / text-embedding caches
    by architecture and VAE/TE identity.

    A dataset's cache lives at ``cache/datasets/{dataset_id}/{namespace}/`` so
    that latents encoded for one model family (e.g. SDXL, 4ch) are NEVER read
    back for another (e.g. Anima, 16ch) that happens to share the dataset. The
    old scheme keyed the cache by dataset id ONLY, which allowed a silent (or
    crashing) cross-architecture / cross-VAE latent mix-up.

    Components (all deterministic for a given run config):
      - ``arch``: architecture family (sd15/sdxl/zimage/anima/lens/flux2/
        krea2/minit2i/ideogram4) — same names the trainer uses.
      - ``vae-<x>``: only when a non-standard VAE is used (SDXL custom VAE),
        since the same arch can then produce different latent channels/scale.
      - ``te-<x>``: only when a non-standard text encoder is used (SDXL custom
        TE), so text-embedding caches don't cross contaminate.
      - ``c<n>``: VAE latent channel count — directly encodes the latent shape
        that triggered the reported channel-mismatch crash.
      - ``dt<dtype>``: latent storage dtype — the "same arch, different VAE
        dtype/scaling" case called out as a silent-mismatch risk.

    Args:
        arch: Architecture family name.
        vae_type: VAE identity (e.g. SDXL ``sdxl_vae_type``); ``None``/``sdxl``
            means the standard arch VAE and adds no token.
        te_type: Text-encoder identity (e.g. SDXL ``sdxl_te_type``);
            ``None``/``clip`` means the standard TE and adds no token.
        latent_channels: VAE latent channel count (optional safety component).
        latent_dtype: Latent storage dtype string (optional).

    Returns:
        A filesystem-safe namespace token (single path component).
    """
    parts = [_sanitize_ns_token(arch or "unknown")]

    vt = str(vae_type or "").strip().lower()
    if vt and vt not in ("none", "sdxl"):
        parts.append("vae-" + _sanitize_ns_token(vt))

    tt = str(te_type or "").strip().lower()
    if tt and tt not in ("none", "clip"):
        parts.append("te-" + _sanitize_ns_token(tt))

    if latent_channels is not None:
        try:
            parts.append(f"c{int(latent_channels)}")
        except (TypeError, ValueError):
            pass

    if latent_dtype:
        dt = str(latent_dtype).replace("torch.", "").strip().lower()
        if dt and dt != "none":
            parts.append("dt" + _sanitize_ns_token(dt))

    return "__".join(parts)


class LatentCache:
    """
    Manages disk cache for VAE latents and optionally text embeddings.

    Cache directory structure:
        cache/datasets/{dataset_unique_id}/
            ├── latents/
            │   ├── {image_hash}.pt
            │   └── ...
            ├── text_embeddings/  (optional)
            │   ├── {caption_hash}_clip1.pt
            │   ├── {caption_hash}_clip2.pt  (SDXL only)
            │   ├── {caption_hash}_pooled.pt (SDXL only)
            │   └── ...
            └── cache_info.json
    """

    def __init__(self, dataset_unique_id: str, base_cache_dir: str = None,
                 namespace: str = None):
        """
        Initialize latent cache.

        Args:
            dataset_unique_id: Dataset unique ID (UUID)
            base_cache_dir: Base directory for cache (default: from user settings or "cache/datasets")
            namespace: Architecture/VAE identity component (see
                ``build_cache_namespace``). When provided, the cache lives at
                ``{base}/{dataset_id}/{namespace}/`` so caches for different
                model families / VAEs never collide. When ``None`` the legacy
                ``{base}/{dataset_id}/`` layout is used (kept for callers that
                do not know the architecture; note such entries are unlabeled
                and must not be shared across architectures).
        """
        self.dataset_unique_id = dataset_unique_id
        self.namespace = namespace
        if base_cache_dir is None:
            base_cache_dir = get_cache_base_dir()
        if namespace:
            self.cache_dir = Path(base_cache_dir) / dataset_unique_id / namespace
        else:
            self.cache_dir = Path(base_cache_dir) / dataset_unique_id
        self.latents_dir = self.cache_dir / "latents"
        self.embeddings_dir = self.cache_dir / "text_embeddings"
        self.cache_info_path = self.cache_dir / "cache_info.json"

        # Create directories
        self.latents_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_image_hash(image_path: str, width: int, height: int) -> str:
        """
        Compute hash for image cache key.

        Includes image path and target dimensions to handle bucketing. WebP keys
        also include the transparency-flattening version so caches made by the
        old alpha-dropping path are not reused.

        Args:
            image_path: Path to image
            width: Target width
            height: Target height

        Returns:
            Hash string
        """
        key = f"{image_path}_{width}_{height}"
        if Path(image_path).suffix.lower() == ".webp":
            key += f"_{TRANSPARENT_WEBP_PREPROCESSING_VERSION}"
        return hashlib.md5(key.encode()).hexdigest()

    @staticmethod
    def compute_clip_hash(
        video_path: str,
        width: int,
        height: int,
        clip_start: int,
        clip_length: int,
        stride: int,
        fps: Optional[float] = None,
        *,
        source_fps: Optional[float] = None,
        target_fps: Optional[float] = None,
        resample_policy: Optional[str] = None,
        start_time: Optional[float] = None,
        tiling_policy: Optional[str] = None,
        audio_prep_version: Optional[str] = None,
    ) -> str:
        """
        Compute hash for a VIDEO CLIP cache key (P4b, 5D temporal latents).

        Distinct clip WINDOWS of the same video (different start / length /
        stride) must cache under DISTINCT keys, so the window parameters are all
        part of the key. This is intentionally separate from
        ``compute_image_hash`` so the existing 4D image cache is untouched.

        BACKWARD COMPATIBILITY (Phase 6a). The first seven parameters produce
        BYTE-IDENTICAL keys to the pre-Phase-6a implementation: the extra
        keyword-only fields append to the key string only when they are set, and
        ``resample_policy="index"`` (the historical, implicit LTX-2.3 policy) is
        treated as unset. An LTX-2.3 user's existing cached latents therefore
        stay addressable; ``temporal_bucketing_test.py`` pins the digests.

        Args:
            video_path: Path to the source video.
            width: Target clip width.
            height: Target clip height.
            clip_start: First source frame index of the clip.
            clip_length: Number of sampled frames.
            stride: Gap between sampled frames.
            fps: Source frames-per-second (optional; folded in when provided so a
                re-encoded/resampled source does not collide).
            source_fps: Source rate, when the arch resamples (redundant with
                ``fps`` in practice, kept explicit so the pair source->target is
                readable in the key).
            target_fps: Rate the decoded clip plays at. Fixed per arch
                (MiniMax-H3: 24.0); None for archs that inherit the source rate.
            resample_policy: ``"timestamp_nearest"`` or ``"index"``. Two clips
                built from the same window under different policies are
                DIFFERENT pixel data (measured: a 30 fps source yields source
                indices [0,1,2,4,5,6,8,...] vs [0,1,2,3,...]).
            start_time: Window start in SECONDS. The addressing unit of a
                timestamp-resampled window; ``clip_start`` alone cannot express
                a start that falls between source frames.
            tiling_policy: Token identifying the VAE's spatial tiling
                configuration. Load-bearing, not cosmetic: flipping the shipped
                MiniMax-H3 tiling flags moved the latents by rel-RMS 0.355
                (384x384, K0.5) and 0.0952 (640x384, Phase 0T), so a cache built
                under one policy must never be served to a generation using the
                other.
            audio_prep_version: Token for the audio preprocessing chain, for
                window-level records that also hold an audio latent (Phase 6b).

        Returns:
            Hash string.
        """
        fps_token = "" if fps is None else f"_{float(fps):.3f}"
        key = (
            f"{video_path}_{width}_{height}"
            f"_s{int(clip_start)}_l{int(clip_length)}_st{int(stride)}{fps_token}"
        )
        # Additive suffix. Empty for every legacy (index-sampled, no-tiling-token)
        # call, which is what keeps existing keys byte-identical.
        extra = ""
        if source_fps is not None:
            extra += f"_src{float(source_fps):.3f}"
        if target_fps is not None:
            extra += f"_tgt{float(target_fps):.3f}"
        if resample_policy is not None and resample_policy != "index":
            extra += f"_rs{resample_policy}"
        if start_time is not None:
            extra += f"_t{float(start_time):.6f}"
        if tiling_policy is not None:
            extra += f"_tile{tiling_policy}"
        if audio_prep_version is not None:
            extra += f"_aud{audio_prep_version}"
        return hashlib.md5((key + extra).encode()).hexdigest()

    @staticmethod
    def compute_caption_hash(caption: str) -> str:
        """
        Compute hash for caption cache key.

        Args:
            caption: Text caption

        Returns:
            Hash string
        """
        return hashlib.md5(caption.encode()).hexdigest()

    def save_latent(
        self,
        image_path: str,
        width: int,
        height: int,
        latents: torch.Tensor,
        skip_existing: bool = True
    ):
        """
        Save VAE latents to cache.

        Args:
            image_path: Source image path
            width: Target width
            height: Target height
            latents: Latent tensor [1, 4, H/8, W/8]
            skip_existing: If True, skip if cache file already exists (default: True)

        Returns:
            True if saved (new file), False if skipped (existing file)
        """
        cache_hash = self.compute_image_hash(image_path, width, height)
        cache_path = self.latents_dir / f"{cache_hash}.pt"

        # Skip if file already exists
        if skip_existing and cache_path.exists():
            return False

        torch.save({
            'latents': latents.cpu(),
            'image_path': image_path,
            'width': width,
            'height': height,
            'created_at': datetime.utcnow().isoformat(),
        }, cache_path)
        return True

    def has_latent(
        self,
        image_path: str,
        width: int,
        height: int,
    ) -> bool:
        """
        Check if latent exists in cache WITHOUT loading it.

        Args:
            image_path: Source image path
            width: Target width
            height: Target height

        Returns:
            True if latent is cached, False otherwise
        """
        cache_hash = self.compute_image_hash(image_path, width, height)
        cache_path = self.latents_dir / f"{cache_hash}.pt"
        return cache_path.exists()

    def load_latent(
        self,
        image_path: str,
        width: int,
        height: int,
        device: str = 'cuda'
    ) -> Optional[torch.Tensor]:
        """
        Load VAE latents from cache.

        Args:
            image_path: Source image path
            width: Target width
            height: Target height
            device: Device to load tensor to

        Returns:
            Latent tensor or None if not cached
        """
        cache_hash = self.compute_image_hash(image_path, width, height)
        cache_path = self.latents_dir / f"{cache_hash}.pt"

        if not cache_path.exists():
            return None

        try:
            data = torch.load(cache_path, map_location=device)
            return data['latents']
        except Exception as e:
            print(f"[LatentCache] Warning: Failed to load cached latent {cache_path}: {e}")
            return None

    def save_clip_latent(
        self,
        video_path: str,
        width: int,
        height: int,
        clip_start: int,
        clip_length: int,
        stride: int,
        latents: torch.Tensor,
        fps: Optional[float] = None,
        skip_existing: bool = True,
        *,
        source_fps: Optional[float] = None,
        target_fps: Optional[float] = None,
        resample_policy: Optional[str] = None,
        start_time: Optional[float] = None,
        tiling_policy: Optional[str] = None,
        audio_prep_version: Optional[str] = None,
        audio_latents: Optional[torch.Tensor] = None,
        has_audio: Optional[bool] = None,
    ) -> bool:
        """
        Save a 5D temporal VAE latent for a video clip (P4b).

        ADDITIVE: shares the same ``latents/`` dir and safetensors/torch.save
        mechanism as the 4D image path, but keys by ``compute_clip_hash`` so
        image and clip entries never collide.

        Args:
            video_path: Source video path.
            width/height: Target clip dimensions.
            clip_start/clip_length/stride: Clip window parameters (part of key).
            latents: 5D latent tensor ``[1, C, T, H', W']`` (LTX: C=128,
                H'=H/32, W'=W/32, T=(clip_length-1)//8+1).
            fps: Source fps (folded into the key when provided).
            skip_existing: Skip write if the cache file already exists.
            audio_latents: OPTIONAL audio latent of the SAME window, making this
                a WINDOW-LEVEL record rather than a video-only one (Phase 6b,
                MiniMax-H3). It has to live here and not in the per-caption text
                aux: an audio latent depends on the sampled clip WINDOW, which a
                caption knows nothing about. ``None`` writes no audio field at
                all, so an LTX-2.3 record keeps exactly the shape it always had.
            has_audio: Explicit "this source HAS an audio track" flag. Recorded
                separately from ``audio_latents is not None`` so a genuinely
                SILENT window (no track, or extraction refused) is a stored fact
                the reader can act on, not an absence indistinguishable from an
                old record written before audio existed.

        Returns:
            True if written, False if skipped.
        """
        cache_hash = self.compute_clip_hash(
            video_path, width, height, clip_start, clip_length, stride, fps,
            source_fps=source_fps, target_fps=target_fps,
            resample_policy=resample_policy, start_time=start_time,
            tiling_policy=tiling_policy, audio_prep_version=audio_prep_version,
        )
        cache_path = self.latents_dir / f"{cache_hash}.pt"

        if skip_existing and cache_path.exists():
            return False

        record = {
            'latents': latents.cpu(),
            'video_path': video_path,
            'width': width,
            'height': height,
            'clip_start': int(clip_start),
            'clip_length': int(clip_length),
            'stride': int(stride),
            'fps': (None if fps is None else float(fps)),
            'is_video_clip': True,
            'created_at': datetime.utcnow().isoformat(),
        }
        # Provenance for a resampled / policy-keyed window. Written only when the
        # caller supplied it, so an LTX-2.3 record keeps exactly its old fields.
        for name, value in (
            ('source_fps', source_fps), ('target_fps', target_fps),
            ('resample_policy', resample_policy), ('start_time', start_time),
            ('tiling_policy', tiling_policy), ('audio_prep_version', audio_prep_version),
        ):
            if value is not None:
                record[name] = value
        # Window-level (video + audio) record. `is_window_record` is what tells a
        # reader that the absence of `audio_latents` means SILENT rather than
        # "written by a video-only writer".
        if audio_latents is not None or has_audio is not None:
            record['is_window_record'] = True
            record['has_audio'] = bool(has_audio if has_audio is not None
                                       else audio_latents is not None)
            record['audio_latents'] = (None if audio_latents is None
                                       else audio_latents.detach().cpu())
        torch.save(record, cache_path)
        return True

    def has_clip_latent(
        self,
        video_path: str,
        width: int,
        height: int,
        clip_start: int,
        clip_length: int,
        stride: int,
        fps: Optional[float] = None,
        *,
        source_fps: Optional[float] = None,
        target_fps: Optional[float] = None,
        resample_policy: Optional[str] = None,
        start_time: Optional[float] = None,
        tiling_policy: Optional[str] = None,
        audio_prep_version: Optional[str] = None,
    ) -> bool:
        """Check if a 5D clip latent exists in cache without loading it."""
        cache_hash = self.compute_clip_hash(
            video_path, width, height, clip_start, clip_length, stride, fps,
            source_fps=source_fps, target_fps=target_fps,
            resample_policy=resample_policy, start_time=start_time,
            tiling_policy=tiling_policy, audio_prep_version=audio_prep_version,
        )
        return (self.latents_dir / f"{cache_hash}.pt").exists()

    def load_clip_latent(
        self,
        video_path: str,
        width: int,
        height: int,
        clip_start: int,
        clip_length: int,
        stride: int,
        fps: Optional[float] = None,
        device: str = 'cuda',
        *,
        source_fps: Optional[float] = None,
        target_fps: Optional[float] = None,
        resample_policy: Optional[str] = None,
        start_time: Optional[float] = None,
        tiling_policy: Optional[str] = None,
        audio_prep_version: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """
        Load a 5D temporal VAE latent for a video clip (P4b).

        Returns the 5D latent tensor ``[1, C, T, H', W']`` or None if not cached.
        """
        cache_hash = self.compute_clip_hash(
            video_path, width, height, clip_start, clip_length, stride, fps,
            source_fps=source_fps, target_fps=target_fps,
            resample_policy=resample_policy, start_time=start_time,
            tiling_policy=tiling_policy, audio_prep_version=audio_prep_version,
        )
        cache_path = self.latents_dir / f"{cache_hash}.pt"

        if not cache_path.exists():
            return None

        try:
            data = torch.load(cache_path, map_location=device)
            latents = data['latents'] if isinstance(data, dict) else data
            return latents
        except Exception as e:
            print(f"[LatentCache] Warning: Failed to load cached clip latent {cache_path}: {e}")
            return None

    def load_clip_record(
        self,
        video_path: str,
        width: int,
        height: int,
        clip_start: int,
        clip_length: int,
        stride: int,
        fps: Optional[float] = None,
        device: str = 'cuda',
        *,
        source_fps: Optional[float] = None,
        target_fps: Optional[float] = None,
        resample_policy: Optional[str] = None,
        start_time: Optional[float] = None,
        tiling_policy: Optional[str] = None,
        audio_prep_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """The WHOLE cached clip record, not just its video latent (Phase 6b).

        ``load_clip_latent`` stays the video-only accessor every existing caller
        uses; this one exists because a MiniMax-H3 window record also carries the
        audio latent of the same window, and a caller that needs both must not
        have to read the file twice or reach into the private layout.

        Returns ``{"latents", "audio_latents", "has_audio", ...provenance}`` or
        ``None`` on a miss. ``audio_latents`` is absent for a video-only record.
        """
        cache_hash = self.compute_clip_hash(
            video_path, width, height, clip_start, clip_length, stride, fps,
            source_fps=source_fps, target_fps=target_fps,
            resample_policy=resample_policy, start_time=start_time,
            tiling_policy=tiling_policy, audio_prep_version=audio_prep_version,
        )
        cache_path = self.latents_dir / f"{cache_hash}.pt"
        if not cache_path.exists():
            return None
        try:
            data = torch.load(cache_path, map_location=device)
            if not isinstance(data, dict):
                return {'latents': data}
            return data
        except Exception as e:
            print(f"[LatentCache] Warning: Failed to load cached clip record {cache_path}: {e}")
            return None

    @staticmethod
    def compute_audio_hash(
        audio_path: str,
        clip_seconds: Optional[float],
        sample_rate: int,
    ) -> str:
        """
        Compute hash for an AUDIO CLIP cache key (ACE-Step, temporal-only 3D latents).

        Mirrors ``compute_clip_hash`` (video) but keyed by clip duration + sample
        rate instead of width/height, since audio latents have no spatial axis
        (``[1, T, 64]``). ``clip_seconds=None`` means "encode the full file"
        (no truncation), a distinct key from any explicit duration.

        Args:
            audio_path: Path to the source audio file.
            clip_seconds: Target clip duration in seconds, or None for full-length.
            sample_rate: Target sample rate (part of the key so a resampled
                source does not collide with a differently-resampled one).

        Returns:
            Hash string.
        """
        dur_token = "full" if clip_seconds is None else f"{float(clip_seconds):.3f}"
        key = f"{audio_path}_dur{dur_token}_sr{int(sample_rate)}"
        return hashlib.md5(key.encode()).hexdigest()

    def save_audio_latent(
        self,
        audio_path: str,
        clip_seconds: Optional[float],
        sample_rate: int,
        latents: torch.Tensor,
        skip_existing: bool = True,
    ) -> bool:
        """
        Save a 3D temporal VAE latent for an audio clip (ACE-Step).

        ADDITIVE: shares the ``latents/`` dir and torch.save mechanism with the
        4D image / 5D video paths, keyed by ``compute_audio_hash`` so entries
        never collide across modalities.

        Args:
            audio_path: Source audio file path.
            clip_seconds/sample_rate: Clip window parameters (part of the key).
            latents: 3D latent tensor ``[1, T, 64]`` (ACE-Step Oobleck VAE).
            skip_existing: Skip write if the cache file already exists.

        Returns:
            True if written, False if skipped.
        """
        cache_hash = self.compute_audio_hash(audio_path, clip_seconds, sample_rate)
        cache_path = self.latents_dir / f"{cache_hash}.pt"

        if skip_existing and cache_path.exists():
            return False

        torch.save({
            'latents': latents.cpu(),
            'audio_path': audio_path,
            'clip_seconds': (None if clip_seconds is None else float(clip_seconds)),
            'sample_rate': int(sample_rate),
            'is_audio_clip': True,
            'created_at': datetime.utcnow().isoformat(),
        }, cache_path)
        return True

    def has_audio_latent(
        self,
        audio_path: str,
        clip_seconds: Optional[float],
        sample_rate: int,
    ) -> bool:
        """Check if a 3D audio-clip latent exists in cache without loading it."""
        cache_hash = self.compute_audio_hash(audio_path, clip_seconds, sample_rate)
        return (self.latents_dir / f"{cache_hash}.pt").exists()

    def load_audio_latent(
        self,
        audio_path: str,
        clip_seconds: Optional[float],
        sample_rate: int,
        device: str = 'cuda',
    ) -> Optional[torch.Tensor]:
        """
        Load a 3D temporal VAE latent for an audio clip (ACE-Step).

        Returns the 3D latent tensor ``[1, T, 64]`` or None if not cached.
        """
        cache_hash = self.compute_audio_hash(audio_path, clip_seconds, sample_rate)
        cache_path = self.latents_dir / f"{cache_hash}.pt"

        if not cache_path.exists():
            return None

        try:
            data = torch.load(cache_path, map_location=device)
            latents = data['latents'] if isinstance(data, dict) else data
            return latents
        except Exception as e:
            print(f"[LatentCache] Warning: Failed to load cached audio latent {cache_path}: {e}")
            return None

    def save_text_embeddings(
        self,
        caption: str,
        text_embeddings: torch.Tensor,
        pooled_embeddings: Optional[torch.Tensor] = None,
        text_embeddings_2: Optional[torch.Tensor] = None
    ):
        """
        Save text embeddings to cache.

        Args:
            caption: Text caption
            text_embeddings: Text embeddings from first encoder [1, 77, 768]
            pooled_embeddings: Pooled embeddings (SDXL only) [1, 1280]
            text_embeddings_2: Text embeddings from second encoder (SDXL only) [1, 77, 1280]
        """
        caption_hash = self.compute_caption_hash(caption)

        # Save CLIP-L embeddings (or SD1.5 embeddings)
        clip1_path = self.embeddings_dir / f"{caption_hash}_clip1.pt"
        torch.save({
            'embeddings': text_embeddings.cpu(),
            'caption': caption,
            'created_at': datetime.utcnow().isoformat(),
        }, clip1_path)

        # Save SDXL-specific embeddings
        if pooled_embeddings is not None:
            pooled_path = self.embeddings_dir / f"{caption_hash}_pooled.pt"
            torch.save({
                'embeddings': pooled_embeddings.cpu(),
                'caption': caption,
                'created_at': datetime.utcnow().isoformat(),
            }, pooled_path)

        if text_embeddings_2 is not None:
            clip2_path = self.embeddings_dir / f"{caption_hash}_clip2.pt"
            torch.save({
                'embeddings': text_embeddings_2.cpu(),
                'caption': caption,
                'created_at': datetime.utcnow().isoformat(),
            }, clip2_path)

    def load_text_embeddings(
        self,
        caption: str,
        is_sdxl: bool = False,
        device: str = 'cuda'
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Load text embeddings from cache.

        Args:
            caption: Text caption
            is_sdxl: Whether to load SDXL embeddings (includes pooled and clip2)
            device: Device to load tensors to

        Returns:
            For SD1.5: (text_embeddings,)
            For SDXL: (text_embeddings, pooled_embeddings)
            Returns None if not cached
        """
        caption_hash = self.compute_caption_hash(caption)
        clip1_path = self.embeddings_dir / f"{caption_hash}_clip1.pt"

        if not clip1_path.exists():
            return None

        try:
            # Load CLIP-L embeddings
            data = torch.load(clip1_path, map_location=device)
            text_embeddings = data['embeddings']

            if is_sdxl:
                # Load pooled embeddings
                pooled_path = self.embeddings_dir / f"{caption_hash}_pooled.pt"
                if not pooled_path.exists():
                    return None

                pooled_data = torch.load(pooled_path, map_location=device)
                pooled_embeddings = pooled_data['embeddings']

                return (text_embeddings, pooled_embeddings)
            else:
                return (text_embeddings,)

        except Exception as e:
            print(f"[LatentCache] Warning: Failed to load cached embeddings for caption: {e}")
            return None

    def save_cache_info(self, model_path: str, model_type: str, item_count: int, training_dtype: str = 'unknown'):
        """
        Save cache metadata.

        Args:
            model_path: Path to base model
            model_type: Model type ('sdxl', 'sd15', 'zimage')
            item_count: Number of items in dataset
            training_dtype: Training dtype (e.g., 'bf16', 'fp16', 'fp32')
        """
        info = {
            'dataset_unique_id': self.dataset_unique_id,
            'model_path': model_path,
            'model_type': model_type,
            'training_dtype': training_dtype,
            'created_at': datetime.utcnow().isoformat(),
            'item_count': item_count,
        }

        with open(self.cache_info_path, 'w') as f:
            json.dump(info, f, indent=2)

    def load_cache_info(self) -> Optional[Dict]:
        """
        Load cache metadata.

        Returns:
            Cache info dict or None if not exists
        """
        if not self.cache_info_path.exists():
            return None

        try:
            with open(self.cache_info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[LatentCache] Warning: Failed to load cache info: {e}")
            return None

    def is_valid(self, model_path: str, model_type: str, training_dtype: str = 'unknown') -> bool:
        """
        Check if cache is valid for current model.

        Args:
            model_path: Current model path
            model_type: Current model type
            training_dtype: Current training dtype

        Returns:
            True if cache is valid
        """
        info = self.load_cache_info()
        if info is None:
            print(f"[LatentCache] Validation failed: No cache_info.json found")
            return False

        # Normalize paths for comparison (resolve to absolute, case-normalized)
        from pathlib import Path
        cached_model_path = info.get('model_path')
        if cached_model_path is None:
            print(f"[LatentCache] Validation failed: model_path not in cache_info.json")
            return False

        try:
            cached_path_normalized = Path(cached_model_path).resolve()
            current_path_normalized = Path(model_path).resolve()
        except Exception as e:
            print(f"[LatentCache] Warning: Path resolution failed ({e}), using string comparison")
            # If path resolution fails, fall back to string comparison
            cached_path_normalized = cached_model_path
            current_path_normalized = model_path

        # Check model compatibility (compare normalized paths)
        if cached_path_normalized != current_path_normalized:
            print(f"[LatentCache] Validation failed: Model path mismatch")
            print(f"[LatentCache]   Cached: {cached_path_normalized}")
            print(f"[LatentCache]   Current: {current_path_normalized}")
            return False

        if info.get('model_type') != model_type:
            print(f"[LatentCache] Validation failed: Model type mismatch")
            print(f"[LatentCache]   Cached: {info.get('model_type')}")
            print(f"[LatentCache]   Current: {model_type}")
            return False

        # Check training dtype (latents are stored in training dtype for memory efficiency)
        cached_dtype = info.get('training_dtype', 'unknown')
        if cached_dtype != 'unknown' and cached_dtype != training_dtype:
            print(f"[LatentCache] Validation failed: Training dtype mismatch")
            print(f"[LatentCache]   Cached: {cached_dtype}")
            print(f"[LatentCache]   Current: {training_dtype}")
            return False

        print(f"[LatentCache] Validation passed: Cache is valid for current model")
        return True

    def validate_cache_format(self, expected_channels: int = 4, sample_count: int = 5) -> bool:
        """
        Validate cache format by randomly sampling cached latents.

        Args:
            expected_channels: Expected number of latent channels (4 for SD/SDXL)
            sample_count: Number of random samples to check

        Returns:
            True if cache format is valid, False otherwise
        """
        import random

        # Get all cached latent files
        latent_files = list(self.latents_dir.glob("*.pt"))

        if len(latent_files) == 0:
            print(f"[LatentCache] No cached latents found in {self.latents_dir}")
            return False

        # Sample random files
        sample_size = min(sample_count, len(latent_files))
        sampled_files = random.sample(latent_files, sample_size)

        print(f"[LatentCache] Validating cache format (sampling {sample_size}/{len(latent_files)} cached latents)...")

        for latent_file in sampled_files:
            try:
                data = torch.load(latent_file, map_location='cpu')

                # Extract latent tensor from dict (cache format: {'latents': tensor, ...})
                if isinstance(data, dict):
                    latent = data.get('latents')
                    if latent is None:
                        print(f"[LatentCache] VALIDATION FAILED: 'latents' key not found in {latent_file.name}")
                        return False
                else:
                    # Legacy format: tensor directly saved
                    latent = data

                # Check shape.
                #   4D  [B, C, H, W]        -> existing image archs (unchanged).
                #   5D  [B, C, T, H', W']   -> video/temporal archs (P4b, LTX).
                # BOTH are accepted; the 4D contract is NOT tightened.
                is_clip = isinstance(data, dict) and data.get('is_video_clip')
                if latent.dim() == 4:
                    # Existing image-latent path.
                    if latent.shape[1] != expected_channels:
                        print(f"[LatentCache] VALIDATION FAILED: Expected {expected_channels} channels, got {latent.shape[1]}")
                        return False
                elif latent.dim() == 5:
                    # Temporal (video clip) path. Channel dim is still index 1.
                    if latent.shape[1] != expected_channels:
                        print(f"[LatentCache] VALIDATION FAILED: Expected {expected_channels} channels, got {latent.shape[1]} (5D clip)")
                        return False
                else:
                    print(f"[LatentCache] VALIDATION FAILED: Expected 4D or 5D tensor, got {latent.dim()}D")
                    return False

                # Log sample info
                kind = "5D-clip" if latent.dim() == 5 or is_clip else "4D-image"
                print(f"[LatentCache]   Sample ({kind}): shape={latent.shape}, dtype={latent.dtype}")

            except Exception as e:
                print(f"[LatentCache] VALIDATION FAILED: Error loading {latent_file.name}: {e}")
                return False

        print(f"[LatentCache] Cache format validation PASSED")
        return True

    def clear(self):
        """Clear all cached data."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.latents_dir.mkdir(parents=True, exist_ok=True)
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)
