"""Phase-0 benchmark contracts and CPU-only evaluation for FoveaTag.

The benchmark deliberately separates immutable annotations from model outputs.
It does not import a model or initialize CUDA, so manifests and saved prediction
files can be validated and compared while another workload owns the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


BENCHMARK_FORMAT_VERSION = 1
PREDICTION_FORMAT_VERSION = 1

DEFAULT_AREA_BINS: Tuple[Tuple[str, float], ...] = (
    ("tiny", 0.0025),
    ("small", 0.01),
    ("medium", 0.04),
    ("large", 1.0),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_nonempty_string(value: Any, field_name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class EvidenceBox:
    """One source-pixel rectangle containing evidence for one or more tags."""

    x: float
    y: float
    width: float
    height: float
    tags: Tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, item_id: str) -> "EvidenceBox":
        tags = tuple(raw.get("tags", ()))
        _require(bool(tags), f"item {item_id}: evidence box needs at least one tag")
        _require(all(isinstance(tag, str) and tag for tag in tags), f"item {item_id}: invalid evidence tag")
        try:
            x = float(raw["x"])
            y = float(raw["y"])
            width = float(raw["width"])
            height = float(raw["height"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"item {item_id}: invalid evidence box coordinates") from error
        _require(x >= 0 and y >= 0 and width > 0 and height > 0, f"item {item_id}: evidence box must have positive area")
        return cls(x=x, y=y, width=width, height=height, tags=tags)

    def area_fraction(self, image_width: int, image_height: int) -> float:
        return (self.width * self.height) / float(image_width * image_height)


@dataclass(frozen=True)
class BenchmarkItem:
    sample_id: str
    image_path: str
    width: int
    height: int
    positive_tags: Tuple[str, ...]
    fine_detail_tags: Tuple[str, ...]
    coarse_tags: Tuple[str, ...]
    evidence: Tuple[EvidenceBox, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BenchmarkItem":
        sample_id = _require_nonempty_string(raw.get("sample_id"), "sample_id")
        image_path = _require_nonempty_string(raw.get("image_path"), f"item {sample_id}: image_path")
        try:
            width = int(raw["width"])
            height = int(raw["height"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"item {sample_id}: width and height must be integers") from error
        _require(width > 0 and height > 0, f"item {sample_id}: image dimensions must be positive")

        positive_tags = tuple(raw.get("positive_tags", ()))
        fine_tags = tuple(raw.get("fine_detail_tags", ()))
        coarse_tags = tuple(raw.get("coarse_tags", ()))
        for field_name, tags in (
            ("positive_tags", positive_tags),
            ("fine_detail_tags", fine_tags),
            ("coarse_tags", coarse_tags),
        ):
            _require(len(tags) == len(set(tags)), f"item {sample_id}: {field_name} contains duplicates")
            _require(all(isinstance(tag, str) and tag for tag in tags), f"item {sample_id}: {field_name} contains an invalid tag")

        positive_set = set(positive_tags)
        _require(set(fine_tags) <= positive_set, f"item {sample_id}: fine_detail_tags must be positive")
        _require(set(coarse_tags) <= positive_set, f"item {sample_id}: coarse_tags must be positive")
        _require(not (set(fine_tags) & set(coarse_tags)), f"item {sample_id}: fine and coarse tags overlap")

        evidence = tuple(EvidenceBox.from_dict(box, item_id=sample_id) for box in raw.get("evidence", ()))
        evidenced_tags = {tag for box in evidence for tag in box.tags}
        _require(evidenced_tags <= set(fine_tags), f"item {sample_id}: evidence tags must be fine_detail_tags")
        for box in evidence:
            _require(box.x + box.width <= width and box.y + box.height <= height, f"item {sample_id}: evidence box is outside the image")

        return cls(
            sample_id=sample_id,
            image_path=image_path,
            width=width,
            height=height,
            positive_tags=positive_tags,
            fine_detail_tags=fine_tags,
            coarse_tags=coarse_tags,
            evidence=evidence,
        )

    def evidence_area(self, tag: str) -> Optional[float]:
        areas = [box.area_fraction(self.width, self.height) for box in self.evidence if tag in box.tags]
        return min(areas) if areas else None


@dataclass(frozen=True)
class BenchmarkManifest:
    name: str
    items: Tuple[BenchmarkItem, ...]
    resolution_long_sides: Tuple[int, ...]
    area_bins: Tuple[Tuple[str, float], ...] = DEFAULT_AREA_BINS
    evidence_mode: str = "boxes"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    format_version: int = BENCHMARK_FORMAT_VERSION

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BenchmarkManifest":
        version = int(raw.get("format_version", -1))
        _require(version == BENCHMARK_FORMAT_VERSION, f"unsupported benchmark format_version={version}")
        name = _require_nonempty_string(raw.get("name"), "name")
        items = tuple(BenchmarkItem.from_dict(item) for item in raw.get("items", ()))
        _require(bool(items), "benchmark manifest must contain at least one item")
        sample_ids = [item.sample_id for item in items]
        _require(len(sample_ids) == len(set(sample_ids)), "sample_id values must be unique")

        sides = tuple(int(value) for value in raw.get("resolution_long_sides", (384, 512, 768, 1024, 1536)))
        _require(bool(sides) and all(value > 0 for value in sides), "resolution_long_sides must be positive")
        _require(tuple(sorted(set(sides))) == sides, "resolution_long_sides must be sorted and unique")

        raw_bins = raw.get("area_bins")
        bins = DEFAULT_AREA_BINS if raw_bins is None else tuple((str(row["name"]), float(row["max_fraction"])) for row in raw_bins)
        _validate_area_bins(bins)
        evidence_mode = str(raw.get("evidence_mode", "boxes"))
        _require(evidence_mode in ("boxes", "dense_oracle"), "evidence_mode must be boxes or dense_oracle")
        provenance = raw.get("provenance", {})
        _require(isinstance(provenance, Mapping), "provenance must be an object")
        if evidence_mode == "boxes":
            for item in items:
                missing_evidence = set(item.fine_detail_tags) - {
                    tag for box in item.evidence for tag in box.tags
                }
                _require(
                    not missing_evidence,
                    f"item {item.sample_id}: fine tags lack evidence boxes: {sorted(missing_evidence)}",
                )
        return cls(
            name=name,
            items=items,
            resolution_long_sides=sides,
            area_bins=bins,
            evidence_mode=evidence_mode,
            provenance=dict(provenance),
            format_version=version,
        )

    def item_by_id(self) -> Dict[str, BenchmarkItem]:
        return {item.sample_id: item for item in self.items}

    def area_bin(self, area_fraction: float) -> str:
        for name, upper in self.area_bins:
            if area_fraction <= upper:
                return name
        raise AssertionError("area bin validation must end at 1.0")

    def resolution_ladder(self, item: BenchmarkItem) -> Tuple[Tuple[str, int, int], ...]:
        """Return non-upscaled evaluation sizes plus the source-native endpoint."""
        long_side = max(item.width, item.height)
        rows: List[Tuple[str, int, int]] = []
        for target in self.resolution_long_sides:
            if target >= long_side:
                continue
            scale = target / float(long_side)
            width = max(1, round(item.width * scale))
            height = max(1, round(item.height * scale))
            rows.append((str(target), width, height))
        rows.append(("native", item.width, item.height))
        return tuple(rows)


def _validate_area_bins(bins: Sequence[Tuple[str, float]]) -> None:
    _require(bool(bins), "area_bins must not be empty")
    names = [name for name, _ in bins]
    limits = [limit for _, limit in bins]
    _require(all(name for name in names) and len(names) == len(set(names)), "area bin names must be non-empty and unique")
    _require(all(math.isfinite(limit) and 0 < limit <= 1 for limit in limits), "area bin limits must be in (0, 1]")
    _require(limits == sorted(limits) and len(limits) == len(set(limits)), "area bin limits must be sorted and unique")
    _require(limits[-1] == 1.0, "the final area bin must end at 1.0")


@dataclass(frozen=True)
class PredictionRow:
    sample_id: str
    resolution: str
    scores: Mapping[str, float]
    stage_ms: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PredictionRow":
        sample_id = _require_nonempty_string(raw.get("sample_id"), "prediction sample_id")
        resolution = _require_nonempty_string(raw.get("resolution"), f"prediction {sample_id}: resolution")
        raw_scores = raw.get("scores")
        _require(isinstance(raw_scores, Mapping) and bool(raw_scores), f"prediction {sample_id}: scores must be a non-empty object")
        scores = {str(tag): float(score) for tag, score in raw_scores.items()}
        _require(all(math.isfinite(score) and 0 <= score <= 1 for score in scores.values()), f"prediction {sample_id}: scores must be finite probabilities")
        raw_stage_ms = raw.get("stage_ms", {})
        _require(isinstance(raw_stage_ms, Mapping), f"prediction {sample_id}: stage_ms must be an object")
        stage_ms = {str(stage): float(duration) for stage, duration in raw_stage_ms.items()}
        _require(all(math.isfinite(value) and value >= 0 for value in stage_ms.values()), f"prediction {sample_id}: stage timings must be non-negative")
        return cls(sample_id=sample_id, resolution=resolution, scores=scores, stage_ms=stage_ms)


@dataclass(frozen=True)
class PredictionSet:
    model_name: str
    rows: Tuple[PredictionRow, ...]
    format_version: int = PREDICTION_FORMAT_VERSION

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PredictionSet":
        version = int(raw.get("format_version", -1))
        _require(version == PREDICTION_FORMAT_VERSION, f"unsupported prediction format_version={version}")
        model_name = _require_nonempty_string(raw.get("model_name"), "model_name")
        rows = tuple(PredictionRow.from_dict(row) for row in raw.get("predictions", ()))
        _require(bool(rows), "prediction set must contain at least one row")
        keys = [(row.sample_id, row.resolution) for row in rows]
        _require(len(keys) == len(set(keys)), "prediction sample_id/resolution pairs must be unique")
        return cls(model_name=model_name, rows=rows, format_version=version)


def load_benchmark_manifest(path: Path | str) -> BenchmarkManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        return BenchmarkManifest.from_dict(json.load(handle))


def load_prediction_set(path: Path | str) -> PredictionSet:
    with Path(path).open("r", encoding="utf-8") as handle:
        return PredictionSet.from_dict(json.load(handle))


def file_evidence(paths: Mapping[str, Path | str]) -> list[dict[str, str]]:
    rows = []
    for role, value in sorted(paths.items()):
        path = Path(value).resolve()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append({"role": role, "path": str(path), "sha256": digest.hexdigest()})
    return rows


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = sum(labels)
    if positives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    seen_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            seen_positive += 1
            precision_sum += seen_positive / rank
    return precision_sum / positives


def _binary_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, float | int]:
    tp = fp = fn = tn = 0
    for label, score in zip(labels, scores):
        predicted = score >= threshold
        if predicted and label:
            tp += 1
        elif predicted:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _stability_to_native(
    rows: Sequence[PredictionRow],
    native_by_sample: Mapping[str, PredictionRow],
    tags: Sequence[str],
    *,
    threshold: float,
) -> Dict[str, float | int | None]:
    absolute_deltas: List[float] = []
    agreements: List[float] = []
    matched_samples = 0
    for row in rows:
        native = native_by_sample.get(row.sample_id)
        if native is None:
            continue
        matched_samples += 1
        for tag in tags:
            score = float(row.scores.get(tag, 0.0))
            native_score = float(native.scores.get(tag, 0.0))
            absolute_deltas.append(abs(score - native_score))
            agreements.append(float((score >= threshold) == (native_score >= threshold)))
    return {
        "matched_samples": matched_samples,
        "score_count": len(absolute_deltas),
        "mean_absolute_score_delta": _mean(absolute_deltas),
        "threshold_agreement": _mean(agreements),
    }


def evaluate_predictions(
    manifest: BenchmarkManifest,
    predictions: PredictionSet,
    *,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate saved probabilities by tag group, evidence area, and resolution."""
    _require(0 <= threshold <= 1, "threshold must be in [0, 1]")
    items = manifest.item_by_id()
    unknown = sorted({row.sample_id for row in predictions.rows} - set(items))
    _require(not unknown, f"predictions reference unknown sample_ids: {unknown[:5]}")

    all_tags = sorted({tag for item in manifest.items for tag in item.positive_tags})
    _require(bool(all_tags), "manifest has no positive tags")
    rows_by_resolution: Dict[str, List[PredictionRow]] = {}
    for row in predictions.rows:
        rows_by_resolution.setdefault(row.resolution, []).append(row)

    fine_tags = {tag for item in manifest.items for tag in item.fine_detail_tags}
    coarse_tags = {tag for item in manifest.items for tag in item.coarse_tags}
    tag_groups = {
        "all": all_tags,
        "fine_detail": sorted(fine_tags),
        "coarse": sorted(coarse_tags),
    }
    native_by_sample = {
        row.sample_id: row for row in rows_by_resolution.get("native", ())
    }
    resolution_reports: Dict[str, Any] = {}
    for resolution, rows in sorted(rows_by_resolution.items()):
        by_sample = {row.sample_id: row for row in rows}
        per_tag: Dict[str, Dict[str, Any]] = {}
        for tag in all_tags:
            labels: List[int] = []
            scores: List[float] = []
            for item in manifest.items:
                row = by_sample.get(item.sample_id)
                if row is None:
                    continue
                labels.append(int(tag in item.positive_tags))
                scores.append(float(row.scores.get(tag, 0.0)))
            if labels:
                per_tag[tag] = {
                    "support": sum(labels),
                    "average_precision": _average_precision(labels, scores),
                    **_binary_metrics(labels, scores, threshold),
                }

        groups = {
            "all": sorted(per_tag),
            "fine_detail": sorted(fine_tags & per_tag.keys()),
            "coarse": sorted(coarse_tags & per_tag.keys()),
        }
        group_report = {
            name: {
                "tag_count": len(tags),
                "macro_average_precision": _mean(per_tag[tag]["average_precision"] for tag in tags),
                "macro_f1": _mean(float(per_tag[tag]["f1"]) for tag in tags),
            }
            for name, tags in groups.items()
        }

        area_values: Dict[str, List[float]] = {name: [] for name, _ in manifest.area_bins}
        for item in manifest.items:
            row = by_sample.get(item.sample_id)
            if row is None:
                continue
            for tag in item.fine_detail_tags:
                area = item.evidence_area(tag)
                if area is None:
                    continue
                area_values[manifest.area_bin(area)].append(float(row.scores.get(tag, 0.0)))
        area_report = {
            name: {
                "positive_count": len(values),
                "recall": sum(score >= threshold for score in values) / len(values) if values else None,
                "mean_score": sum(values) / len(values) if values else None,
            }
            for name, values in area_values.items()
        }
        resolution_reports[resolution] = {
            "sample_count": len(rows),
            "groups": group_report,
            "stability_to_native": {
                name: _stability_to_native(
                    rows,
                    native_by_sample,
                    tags,
                    threshold=threshold,
                )
                for name, tags in tag_groups.items()
            },
            "evidence_area": area_report,
            "per_tag": per_tag,
        }

    timing: Dict[str, List[float]] = {}
    for row in predictions.rows:
        for stage, value in row.stage_ms.items():
            timing.setdefault(stage, []).append(float(value))
    return {
        "benchmark": manifest.name,
        "evidence_mode": manifest.evidence_mode,
        "label_provenance": manifest.provenance.get("label_provenance", "unspecified"),
        "model_name": predictions.model_name,
        "threshold": threshold,
        "resolutions": resolution_reports,
        "stage_timing_ms": {stage: _timing_summary(values) for stage, values in sorted(timing.items())},
    }


def evaluate_dense_oracle_recovery(
    router_scores,
    oracle_utilities,
    *,
    k: int,
    candidate_mask=None,
) -> Dict[str, float | int | None]:
    """Measure how much label-derived dense-oracle utility routing recovers."""
    import torch

    if router_scores.shape != oracle_utilities.shape or router_scores.ndim != 2:
        raise ValueError("router_scores and oracle_utilities must have equal [batch, candidates] shapes")
    if k <= 0 or k > router_scores.shape[1]:
        raise ValueError("k must be within the candidate count")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(router_scores, dtype=torch.bool)
    if candidate_mask.shape != router_scores.shape:
        raise ValueError("candidate_mask must match scores")
    valid = candidate_mask.to(device=router_scores.device, dtype=torch.bool)
    if (valid.sum(dim=1) < k).any():
        raise ValueError("each sample must have at least k valid candidates")
    utilities = oracle_utilities.float()
    masked_scores = router_scores.masked_fill(~valid, float("-inf"))
    masked_utility = utilities.masked_fill(~valid, float("-inf"))
    selected = masked_scores.topk(k, dim=1).indices
    oracle = masked_utility.topk(k, dim=1).indices
    chosen_utility = torch.gather(utilities.clamp_min(0), 1, selected).sum(dim=1)
    oracle_utility = torch.gather(utilities.clamp_min(0), 1, oracle).sum(dim=1)
    recoverable = oracle_utility > 0
    recovery = torch.where(
        recoverable,
        (chosen_utility / oracle_utility.clamp_min(torch.finfo(oracle_utility.dtype).eps)).clamp(max=1),
        torch.zeros_like(oracle_utility),
    )
    overlap = (selected.unsqueeze(2) == oracle.unsqueeze(1)).any(dim=2).float().mean(dim=1)
    mean_recovery = float(recovery[recoverable].mean()) if recoverable.any() else None
    mean_overlap = float(overlap[recoverable].mean()) if recoverable.any() else None
    return {
        "batch_size": router_scores.shape[0],
        "k": k,
        "mean_utility_recovery": mean_recovery,
        "mean_oracle_topk_recall": mean_overlap,
        "recoverable_samples": int(recoverable.sum()),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _timing_summary(values: Sequence[float]) -> Dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


class StageTimer:
    """Collect wall-clock durations for staged inference or preprocessing."""

    def __init__(self) -> None:
        self.samples_ms: Dict[str, List[float]] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        _require_nonempty_string(name, "stage name")
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.samples_ms.setdefault(name, []).append(elapsed_ms)

    def summary(self) -> Dict[str, Dict[str, float | int]]:
        return {name: _timing_summary(values) for name, values in sorted(self.samples_ms.items())}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a benchmark manifest")
    validate.add_argument("manifest", type=Path)
    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved prediction set")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    evidence_paths = {"manifest": args.manifest}
    if args.command == "evaluate":
        evidence_paths["predictions"] = args.predictions
    evidence = file_evidence(evidence_paths)
    manifest = load_benchmark_manifest(args.manifest)
    if args.command == "validate":
        print(json.dumps({"name": manifest.name, "items": len(manifest.items)}, indent=2))
        return 0
    report = evaluate_predictions(manifest, load_prediction_set(args.predictions), threshold=args.threshold)
    if file_evidence(evidence_paths) != evidence:
        raise RuntimeError("benchmark inputs changed during evaluation")
    report["evidence"] = evidence
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
