"""
Dataset structure auto-detection module.

Analyzes filenames in a dataset directory to detect suffix-based pairing patterns
(e.g., _source/_target/_instruction) and automatically classify them as
reference images, target images, or caption files.
"""

import os
from typing import Dict, List, Optional, Set, Tuple, TypedDict


# Keyword heuristics for suffix classification
TARGET_KEYWORDS: Set[str] = {
    "target", "teacher", "edited", "output", "result",
    "after", "gt", "ground_truth", "groundtruth",
}

REFERENCE_KEYWORDS: Set[str] = {
    "source", "reference", "ref", "input", "condition",
    "before", "original", "cond", "control",
    "depth", "normal", "edge", "pose", "sketch",
    "seg", "segmentation", "lineart", "openpose", "canny",
}

CAPTION_KEYWORDS: Set[str] = {
    "instruction", "prompt", "caption", "text",
    "desc", "description", "edit",
}

IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".webp"}
CAPTION_EXTENSIONS: Set[str] = {".txt", ".json"}
ALL_EXTENSIONS: Set[str] = IMAGE_EXTENSIONS | CAPTION_EXTENSIONS

MAX_SAMPLE_FILES = 5000
MIN_SUFFIX_OCCURRENCES = 3
MIN_OVERLAP_RATIO = 0.5
MIN_PAIRS_FOR_DETECTION = 2


class SuffixStats(TypedDict):
    image_count: int
    caption_count: int
    bases: Set[str]


class DetectionResult(TypedDict):
    structure_type: str  # "normal" or "paired"
    reference_suffixes: List[str]
    target_suffixes: List[str]
    caption_suffixes_for_reference: List[str]
    confidence: float
    unknown_suffixes: List[str]
    stats: Dict


def _collect_filenames(
    dir_path: str,
    recursive: bool,
    max_depth: Optional[int],
    valid_exts: Set[str],
    max_count: int,
) -> List[str]:
    """Collect filenames from directory with sampling limit."""
    filenames: List[str] = []

    def _collect(path: str, depth: int = 0) -> None:
        if len(filenames) >= max_count:
            return
        try:
            entries = os.listdir(path)
        except (PermissionError, OSError):
            return

        for entry in entries:
            if len(filenames) >= max_count:
                return
            entry_path = os.path.join(path, entry)
            if os.path.isfile(entry_path):
                _, ext = os.path.splitext(entry)
                if ext.lower() in valid_exts:
                    filenames.append(entry)
            elif os.path.isdir(entry_path) and recursive:
                if max_depth is None or depth < max_depth:
                    _collect(entry_path, depth + 1)

    _collect(dir_path)
    return filenames


def _classify_suffix(suffix: str) -> str:
    """
    Classify a suffix string as 'target', 'reference', 'caption', or 'unknown'.

    Args:
        suffix: The suffix including leading underscore (e.g., "_source")

    Returns:
        Classification string
    """
    keyword = suffix.lstrip("_").lower()

    if keyword in TARGET_KEYWORDS:
        return "target"
    elif keyword in REFERENCE_KEYWORDS:
        return "reference"
    elif keyword in CAPTION_KEYWORDS:
        return "caption"
    return "unknown"


def _extract_suffix(base_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (prefix, suffix) from a base_name by splitting on the last underscore.

    Returns:
        (prefix, suffix) where suffix includes the leading underscore,
        or (None, None) if no underscore found.

    Example:
        "20251005_143008_298564_source" -> ("20251005_143008_298564", "_source")
        "image001" -> (None, None)
    """
    last_underscore = base_name.rfind("_")
    if last_underscore > 0:  # Must have content before underscore
        prefix = base_name[:last_underscore]
        suffix = base_name[last_underscore:]  # e.g., "_source"
        return prefix, suffix
    return None, None


def _build_normal_result() -> DetectionResult:
    """Return a default 'normal' detection result."""
    return DetectionResult(
        structure_type="normal",
        reference_suffixes=[],
        target_suffixes=[],
        caption_suffixes_for_reference=[],
        confidence=0.0,
        unknown_suffixes=[],
        stats={
            "total_files_sampled": 0,
            "suffix_counts": {},
            "paired_groups": 0,
            "unpaired_files": 0,
        },
    )


def detect_dataset_structure(
    dir_path: str,
    recursive: bool = True,
    max_depth: Optional[int] = None,
) -> DetectionResult:
    """
    Analyze filenames in a dataset directory to detect suffix-based pairing patterns.

    The algorithm:
    1. Collects filenames (sampled up to MAX_SAMPLE_FILES)
    2. Splits each filename on last underscore to extract (base, suffix)
    3. Filters suffixes that appear at least MIN_SUFFIX_OCCURRENCES times
    4. Finds suffix pairs where base_name overlap >= MIN_OVERLAP_RATIO
    5. Classifies suffixes using keyword heuristics (target/reference/caption)
    6. Handles ambiguous cases with fallback heuristics

    Args:
        dir_path: Path to the dataset directory
        recursive: Whether to scan subdirectories
        max_depth: Maximum directory depth for recursive scanning

    Returns:
        DetectionResult with structure_type, suffixes, confidence, and stats
    """
    result = _build_normal_result()

    if not os.path.isdir(dir_path):
        return result

    # Step 1: Collect filenames
    filenames = _collect_filenames(dir_path, recursive, max_depth, ALL_EXTENSIONS, MAX_SAMPLE_FILES)
    result["stats"]["total_files_sampled"] = len(filenames)

    if len(filenames) < MIN_PAIRS_FOR_DETECTION * 2:
        return result

    # Step 2: Extract suffix candidates
    suffix_data: Dict[str, SuffixStats] = {}
    no_suffix_count = 0

    for filename in filenames:
        base_name, ext = os.path.splitext(filename)
        ext_lower = ext.lower()

        prefix, suffix = _extract_suffix(base_name)
        if prefix is not None and suffix is not None:
            if suffix not in suffix_data:
                suffix_data[suffix] = SuffixStats(image_count=0, caption_count=0, bases=set())

            suffix_data[suffix]["bases"].add(prefix)
            if ext_lower in IMAGE_EXTENSIONS:
                suffix_data[suffix]["image_count"] += 1
            elif ext_lower in CAPTION_EXTENSIONS:
                suffix_data[suffix]["caption_count"] += 1
        else:
            no_suffix_count += 1

    result["stats"]["unpaired_files"] = no_suffix_count

    # Step 3: Filter suffix candidates - keep only those appearing consistently
    viable_suffixes: Dict[str, SuffixStats] = {}
    for suffix, data in suffix_data.items():
        total_count = data["image_count"] + data["caption_count"]
        if total_count >= MIN_SUFFIX_OCCURRENCES:
            viable_suffixes[suffix] = data

    if len(viable_suffixes) < 2:
        return result

    # Step 4: Find suffix pairs by checking base name overlap
    suffix_list = list(viable_suffixes.keys())
    suffix_pairs: List[Tuple[str, str, int, float]] = []

    for i, s1 in enumerate(suffix_list):
        for s2 in suffix_list[i + 1:]:
            bases1 = viable_suffixes[s1]["bases"]
            bases2 = viable_suffixes[s2]["bases"]
            overlap = bases1 & bases2

            if len(overlap) == 0:
                continue

            smaller = min(len(bases1), len(bases2))
            overlap_ratio = len(overlap) / smaller if smaller > 0 else 0.0

            if overlap_ratio >= MIN_OVERLAP_RATIO:
                suffix_pairs.append((s1, s2, len(overlap), overlap_ratio))

    if not suffix_pairs:
        return result

    # Step 5: Classify each detected suffix
    paired_suffixes: Set[str] = set()
    for s1, s2, _overlap, _ratio in suffix_pairs:
        paired_suffixes.add(s1)
        paired_suffixes.add(s2)

    detected_reference: List[str] = []
    detected_target: List[str] = []
    detected_caption: List[str] = []
    unknown_suffixes: List[str] = []

    for suffix in paired_suffixes:
        classification = _classify_suffix(suffix)
        if classification == "target":
            detected_target.append(suffix)
        elif classification == "reference":
            detected_reference.append(suffix)
        elif classification == "caption":
            detected_caption.append(suffix)
        else:
            unknown_suffixes.append(suffix)

    # Step 5b: Check for caption-only suffixes (text files that share bases with paired image suffixes)
    for suffix, data in viable_suffixes.items():
        if suffix in paired_suffixes:
            continue
        if data["caption_count"] < MIN_SUFFIX_OCCURRENCES:
            continue
        # Must be predominantly caption files (allow some images)
        if data["image_count"] > data["caption_count"]:
            continue

        classification = _classify_suffix(suffix)
        if classification == "caption":
            # Verify it shares bases with at least one paired suffix
            for ps in paired_suffixes:
                overlap = data["bases"] & viable_suffixes[ps]["bases"]
                if len(overlap) / max(len(data["bases"]), 1) >= MIN_OVERLAP_RATIO:
                    detected_caption.append(suffix)
                    break
        elif classification == "unknown":
            # Even unknown suffixes with only caption files might be instruction files
            # Check if they share bases with paired suffixes
            for ps in paired_suffixes:
                overlap = data["bases"] & viable_suffixes[ps]["bases"]
                if len(overlap) / max(len(data["bases"]), 1) >= MIN_OVERLAP_RATIO:
                    # Only text files sharing bases with image pairs - likely captions
                    if data["image_count"] == 0:
                        detected_caption.append(suffix)
                    break

    # Step 6: Handle ambiguous cases - 2 unknown suffixes that pair with each other
    if not detected_target and not detected_reference and len(unknown_suffixes) >= 2:
        # Try to disambiguate using caption file association
        # The suffix whose bases overlap more with caption file bases is likely the target
        # (because captions typically describe what to do to produce the target)
        if detected_caption:
            caption_bases: Set[str] = set()
            for cs in detected_caption:
                if cs in viable_suffixes:
                    caption_bases |= viable_suffixes[cs]["bases"]

            best_target_suffix: Optional[str] = None
            best_target_overlap = 0

            for us in unknown_suffixes:
                overlap_count = len(viable_suffixes[us]["bases"] & caption_bases)
                if overlap_count > best_target_overlap:
                    best_target_overlap = overlap_count
                    best_target_suffix = us

            if best_target_suffix is not None and best_target_overlap > 0:
                detected_target.append(best_target_suffix)
                for us in unknown_suffixes:
                    if us != best_target_suffix:
                        # Other unknown suffixes with images are reference
                        if viable_suffixes[us]["image_count"] > 0:
                            detected_reference.append(us)
                unknown_suffixes = [
                    us for us in unknown_suffixes
                    if us not in detected_target and us not in detected_reference
                ]

    # Step 7: Final check - we need at least one target AND one reference
    if not detected_target or not detected_reference:
        result["unknown_suffixes"] = unknown_suffixes
        return result

    # Step 8: Calculate confidence
    target_bases: Set[str] = set()
    for s in detected_target:
        target_bases |= viable_suffixes[s]["bases"]
    ref_bases: Set[str] = set()
    for s in detected_reference:
        ref_bases |= viable_suffixes[s]["bases"]

    paired_count = len(target_bases & ref_bases)
    total_potential = len(target_bases | ref_bases)
    confidence = paired_count / total_potential if total_potential > 0 else 0.0

    # Build suffix counts for stats
    suffix_counts: Dict[str, int] = {}
    all_detected = set(detected_reference + detected_target + detected_caption)
    for s in all_detected:
        if s in viable_suffixes:
            suffix_counts[s] = viable_suffixes[s]["image_count"] + viable_suffixes[s]["caption_count"]

    return DetectionResult(
        structure_type="paired",
        reference_suffixes=sorted(detected_reference),
        target_suffixes=sorted(detected_target),
        caption_suffixes_for_reference=sorted(detected_caption),
        confidence=round(confidence, 3),
        unknown_suffixes=sorted(unknown_suffixes),
        stats={
            "total_files_sampled": len(filenames),
            "suffix_counts": suffix_counts,
            "paired_groups": paired_count,
            "unpaired_files": no_suffix_count,
        },
    )
