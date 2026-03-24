"""Dataset directory scanner with 2-pass architecture.

Pass 1: Collect image stems and detect reference/target pairs.
Pass 2: Associate text/JSON files with image stems via prefix matching.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
CAPTION_EXTS = {".txt", ".json"}


def scan_directory_structure(
    dir_path: str,
    recursive: bool = True,
    max_depth: Optional[int] = None,
    reference_suffixes: Optional[List[str]] = None,
    target_suffixes: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Scan directory and build structured file groups.

    Returns:
        Dict of group_name -> {
            "images": [{"path": str, "role": "main"|"reference"|"target"}],
            "captions": [{"path": str, "suffix": str, "ext": str}],
        }
    """
    reference_suffixes = reference_suffixes or []
    target_suffixes = target_suffixes or []

    # Collect all files
    all_files = _collect_files(dir_path, recursive, max_depth)

    # Pass 1: Identify image stems
    image_stems = {}  # stem -> [{"path", "role", "group_name"}]
    for fpath in all_files:
        ext = os.path.splitext(fpath)[1].lower()
        if ext not in IMAGE_EXTS:
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
) -> List[str]:
    """Collect all files in directory."""
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
                files.extend(_collect_files(full_path, recursive, max_depth, _current_depth + 1))

    return files


def _get_stem(filepath: str) -> str:
    """Get filename stem (without extension) from full path."""
    return os.path.splitext(os.path.basename(filepath))[0]
