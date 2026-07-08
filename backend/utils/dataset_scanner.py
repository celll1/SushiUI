"""Dataset directory scanner with 2-pass architecture.

Pass 1: Collect image stems and detect reference/target pairs.
Pass 2: Associate text/JSON files with image stems via prefix matching.
"""

import os
import glob
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".webm", ".mp4", ".mkv", ".mov", ".avi"}
# Both images and videos participate in stem grouping / caption sidecar matching.
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS
CAPTION_EXTS = {".txt", ".json"}

# Below this file size a frame-accurate ffprobe frame count (-count_frames,
# which decodes the whole stream) is permitted as a last resort. Larger files
# fall back to duration x fps to avoid a full decode during a scan.
_VIDEO_COUNT_FRAMES_MAX_BYTES = 32 * 1024 * 1024


def _find_ffprobe() -> Optional[str]:
    """Locate the ffprobe executable (PATH first, then common install dirs)."""
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    patterns = [
        r"D:\ffmpeg-*\bin\ffprobe.exe",
        r"C:\ffmpeg-*\bin\ffprobe.exe",
        "/d/ffmpeg-*/bin/ffprobe",
        "/c/ffmpeg-*/bin/ffprobe",
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


def _parse_fraction(value: Optional[str]) -> float:
    """Parse an ffprobe rational string ('60/1') or plain number into a float."""
    if not value:
        return 0.0
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _count_frames_ffprobe(ffprobe: str, video_path: str) -> int:
    """Frame-accurate count via ffprobe -count_frames (decodes the stream).

    Only invoked for small files (see ``_VIDEO_COUNT_FRAMES_MAX_BYTES``).
    """
    try:
        cmd = [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", video_path,
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if out.returncode == 0:
            v = out.stdout.strip()
            if v.isdigit():
                return int(v)
    except Exception as e:  # noqa: BLE001 - probe is best-effort
        print(f"[VideoProbe] frame count failed for {video_path}: {e}")
    return 0


def probe_video_metadata(video_path: str) -> Optional[Dict[str, Any]]:
    """Probe a video's metadata via ffprobe without decoding all frames.

    Returns a dict {video_path, fps, num_frames, duration, width, height, codec}
    or None when the file cannot be probed (caller should skip + log).

    num_frames resolution order:
      1. stream nb_frames (container-reported, no decode)
      2. round(fps * duration) (estimate)
      3. ffprobe -count_frames (full decode) ONLY for files below
         ``_VIDEO_COUNT_FRAMES_MAX_BYTES`` to avoid decoding large clips.
    """
    ffprobe = _find_ffprobe()
    if not ffprobe:
        print(f"[VideoProbe] ffprobe not found on PATH or common dirs; cannot probe {video_path}")
        return None
    try:
        cmd = [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,duration:format=duration",
            "-of", "json", video_path,
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            print(f"[VideoProbe] ffprobe failed for {video_path}: {out.stderr.strip()[:200]}")
            return None
        data = json.loads(out.stdout)
        streams = data.get("streams") or []
        if not streams:
            print(f"[VideoProbe] no video stream found in {video_path}")
            return None
        st = streams[0]

        width = int(st.get("width") or 0)
        height = int(st.get("height") or 0)
        if width <= 0 or height <= 0:
            print(f"[VideoProbe] invalid dimensions ({width}x{height}) for {video_path}")
            return None

        codec = st.get("codec_name") or None

        fps = _parse_fraction(st.get("r_frame_rate"))
        if fps <= 0:
            fps = _parse_fraction(st.get("avg_frame_rate"))

        duration = 0.0
        for src in (st.get("duration"), (data.get("format") or {}).get("duration")):
            try:
                if src is not None and float(src) > 0:
                    duration = float(src)
                    break
            except (ValueError, TypeError):
                continue

        num_frames = 0
        nb = st.get("nb_frames")
        try:
            if nb is not None and int(nb) > 0:
                num_frames = int(nb)
        except (ValueError, TypeError):
            num_frames = 0
        if num_frames <= 0 and fps > 0 and duration > 0:
            num_frames = int(round(fps * duration))
        if num_frames <= 0:
            try:
                if os.path.getsize(video_path) <= _VIDEO_COUNT_FRAMES_MAX_BYTES:
                    num_frames = _count_frames_ffprobe(ffprobe, video_path)
            except OSError:
                pass

        return {
            "video_path": video_path,
            "fps": round(fps, 6),
            "num_frames": int(num_frames),
            "duration": round(duration, 6),
            "width": width,
            "height": height,
            "codec": codec,
        }
    except Exception as e:  # noqa: BLE001 - probe is best-effort, never crash scan
        print(f"[VideoProbe] probe error for {video_path}: {e}")
        return None


def extract_poster_frame(video_path: str, out_path: str) -> bool:
    """Write the first frame of a video to out_path (PNG) via cv2.

    Returns True on success. Best-effort: a failure logs a warning and returns
    False so the scan can proceed without a poster thumbnail.
    """
    cap = None
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        print(f"[VideoPoster] cv2 unavailable, cannot extract poster: {e}")
        return False
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[VideoPoster] cannot open {video_path}")
            return False
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[VideoPoster] cannot read frame 0 of {video_path}")
            return False
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # imencode + manual write handles non-ASCII paths that cv2.imwrite mishandles.
        ext = os.path.splitext(out_path)[1] or ".png"
        ok2, buf = cv2.imencode(ext, frame)
        if not ok2:
            print(f"[VideoPoster] encode failed for {video_path}")
            return False
        with open(out_path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[VideoPoster] poster extraction failed for {video_path}: {e}")
        return False
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass


def scan_directory_structure(
    dir_path: str,
    recursive: bool = True,
    max_depth: Optional[int] = None,
    reference_suffixes: Optional[List[str]] = None,
    target_suffixes: Optional[List[str]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Dict]:
    """Scan directory and build structured file groups.

    Returns:
        Dict of group_name -> {
            "images": [{"path": str, "role": "main"|"reference"|"target"}],
            "captions": [{"path": str, "suffix": str, "ext": str}],
        }

    ``should_cancel`` (optional) is polled during the directory walk; if it
    returns True a ``RescanSkipped`` is raised so a training pre-flight rescan
    can abort the current dataset (see core.training.rescan_control).
    """
    reference_suffixes = reference_suffixes or []
    target_suffixes = target_suffixes or []

    # Collect all files
    all_files = _collect_files(dir_path, recursive, max_depth, should_cancel=should_cancel)

    # Pass 1: Identify image stems
    image_stems = {}  # stem -> [{"path", "role", "group_name"}]
    for fpath in all_files:
        ext = os.path.splitext(fpath)[1].lower()
        if ext not in MEDIA_EXTS:
            continue

        stem = _get_stem(fpath)
        role = "main"
        group_name = stem

        # Check reference/target suffixes
        for suffix in reference_suffixes:
            if stem.endswith(suffix):
                group_name = stem[:-len(suffix)]
                role = "reference"
                break
        if role == "main":
            for suffix in target_suffixes:
                if stem.endswith(suffix):
                    group_name = stem[:-len(suffix)]
                    role = "target"
                    break

        if group_name not in image_stems:
            image_stems[group_name] = []
        image_stems[group_name].append({
            "path": fpath,
            "role": role,
            "original_stem": stem,
        })

    # Build a sorted list of image group names for prefix matching (longest first)
    sorted_groups = sorted(image_stems.keys(), key=len, reverse=True)

    # Pass 2: Associate text/JSON files with image stems
    scan_groups = {}
    for group_name, images in image_stems.items():
        scan_groups[group_name] = {
            "images": images,
            "captions": [],
        }

    unmatched_texts = []
    for fpath in all_files:
        ext = os.path.splitext(fpath)[1].lower()
        if ext not in CAPTION_EXTS:
            continue

        stem = _get_stem(fpath)

        # Try exact match first
        if stem in image_stems:
            scan_groups[stem]["captions"].append({
                "path": fpath,
                "suffix": "",
                "ext": ext,
            })
            continue

        # Try prefix match (longest match wins)
        matched = False
        for group_name in sorted_groups:
            if stem.startswith(group_name + "_"):
                suffix = stem[len(group_name) + 1:]
                scan_groups[group_name]["captions"].append({
                    "path": fpath,
                    "suffix": suffix,
                    "ext": ext,
                })
                matched = True
                break

        if not matched:
            unmatched_texts.append(fpath)

    return scan_groups


def classify_caption_files(
    scan_groups: Dict[str, Dict],
    taglist: Set[str],
) -> Dict[str, Dict]:
    """Classify each caption file's format (tags vs natural_language).

    Adds "detected_type" and "is_tags_format" to each caption entry.
    """
    from utils.caption_detector import classify_field

    for group_name, group in scan_groups.items():
        for cap in group["captions"]:
            if cap["ext"] == ".txt":
                try:
                    with open(cap["path"], "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        field_name = cap["suffix"] if cap["suffix"] else "tags"
                        field_category, is_tags_format, match_rate = classify_field(
                            field_name, content, taglist
                        )
                        cap["detected_type"] = "tags" if is_tags_format else "natural_language"
                        cap["is_tags_format"] = is_tags_format
                        cap["tag_match_rate"] = match_rate
                        cap["field_category"] = field_category
                        cap["content_preview"] = content[:200]
                    else:
                        cap["detected_type"] = "empty"
                        cap["is_tags_format"] = False
                        cap["tag_match_rate"] = 0.0
                        cap["field_category"] = "training"
                except Exception as e:
                    cap["detected_type"] = "error"
                    cap["error"] = str(e)

            elif cap["ext"] == ".json":
                cap["detected_type"] = "json"
                cap["is_tags_format"] = False
                cap["tag_match_rate"] = 0.0
                cap["field_category"] = "training"

    return scan_groups


def build_scan_preview(
    scan_groups: Dict[str, Dict],
    max_groups: int = 100,
) -> Dict[str, Any]:
    """Build a preview summary for the frontend.

    Returns:
        {
            "total_groups": int,
            "total_images": int,
            "total_captions": int,
            "detected_suffixes": {"suffix": {"count": int, "sample_type": str}},
            "structure_type": "single" | "paired",
            "sample_groups": [first N groups with details],
        }
    """
    total_images = 0
    total_captions = 0
    suffix_stats = {}  # suffix -> {"count": int, "types": set}
    has_reference = False

    for group in scan_groups.values():
        total_images += sum(1 for img in group["images"] if img["role"] in ("main", "target"))
        total_captions += len(group["captions"])

        if any(img["role"] in ("reference", "target") for img in group["images"]):
            has_reference = True

        for cap in group["captions"]:
            suffix = cap["suffix"] or "(default)"
            if suffix not in suffix_stats:
                suffix_stats[suffix] = {"count": 0, "types": set()}
            suffix_stats[suffix]["count"] += 1
            if "detected_type" in cap:
                suffix_stats[suffix]["types"].add(cap["detected_type"])

    # Build sample groups
    sample_groups = []
    for group_name, group in list(scan_groups.items())[:max_groups]:
        sample_groups.append({
            "group_name": group_name,
            "images": [{"path": img["path"], "role": img["role"]} for img in group["images"]],
            "captions": [
                {
                    "path": cap["path"],
                    "suffix": cap["suffix"],
                    "detected_type": cap.get("detected_type", "unknown"),
                    "content_preview": cap.get("content_preview", ""),
                }
                for cap in group["captions"]
            ],
        })

    # Serialize suffix stats
    suffix_summary = {}
    for suffix, stats in sorted(suffix_stats.items()):
        suffix_summary[suffix] = {
            "count": stats["count"],
            "sample_types": sorted(stats["types"]),
        }

    return {
        "total_groups": len(scan_groups),
        "total_images": total_images,
        "total_captions": total_captions,
        "detected_suffixes": suffix_summary,
        "structure_type": "paired" if has_reference else "single",
        "sample_groups": sample_groups,
    }


def _collect_files(
    dir_path: str,
    recursive: bool = True,
    max_depth: Optional[int] = None,
    _current_depth: int = 0,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> List[str]:
    """Collect all files in directory.

    Polls ``should_cancel`` once per directory (raising ``RescanSkipped``) so a
    training pre-flight rescan can abort the walk of the current dataset within
    ~1 directory of the skip request.
    """
    if should_cancel is not None and should_cancel():
        # Local import: keeps this lightweight util free of the heavy
        # core.training package at module load; only paid on an actual skip.
        from core.training.rescan_control import RescanSkipped
        raise RescanSkipped()
    files = []
    try:
        entries = os.listdir(dir_path)
    except PermissionError:
        return files

    for entry in entries:
        full_path = os.path.join(dir_path, entry)
        if os.path.isfile(full_path):
            files.append(full_path)
        elif os.path.isdir(full_path) and recursive:
            if max_depth is None or _current_depth < max_depth:
                files.extend(_collect_files(full_path, recursive, max_depth, _current_depth + 1, should_cancel))

    return files


def _get_stem(filepath: str) -> str:
    """Get filename stem (without extension) from full path."""
    return os.path.splitext(os.path.basename(filepath))[0]
