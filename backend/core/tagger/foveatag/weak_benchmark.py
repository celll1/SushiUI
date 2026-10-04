"""Build a fixed box-free Phase-0 benchmark from existing image-level tags."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from ..tag_parsing import extract_caption_tags
from ..tag_vocabulary import normalize_tag
from .benchmark import BENCHMARK_FORMAT_VERSION, BenchmarkManifest


DEFAULT_FINE_TAGS = (
    "blue eyes", "red eyes", "green eyes", "brown eyes", "purple eyes",
    "yellow eyes", "pink eyes", "orange eyes", "aqua eyes", "grey eyes",
    "black eyes", "heterochromia", "colored contact lenses", "contact lenses",
    "eyeliner", "eyeshadow", "lipstick", "mascara", "earrings", "nail polish",
    "painted nails", "logo", "text",
)
DEFAULT_COARSE_TAGS = (
    "1girl", "1boy", "solo", "multiple girls", "multiple boys", "standing",
    "sitting", "indoors", "outdoors", "upper body", "full body",
)


@dataclass(frozen=True)
class WeakBenchmarkRecord:
    item_id: int
    image_path: str
    width: int
    height: int
    fine_tags: tuple[str, ...]
    coarse_tags: tuple[str, ...]


def _rank(seed: int, item_id: int, salt: str) -> bytes:
    return hashlib.blake2b(
        f"{seed}:{item_id}:{salt}".encode("utf-8"), digest_size=16
    ).digest()


def load_weak_records(
    database_path: Path | str,
    dataset_id: int,
    *,
    fine_tags: Iterable[str] = DEFAULT_FINE_TAGS,
    coarse_tags: Iterable[str] = DEFAULT_COARSE_TAGS,
    require_files: bool = True,
    limit: Optional[int] = None,
    selection: str = "first",
    seed: int = 0,
) -> list[WeakBenchmarkRecord]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when supplied")
    if selection not in ("first", "hashed"):
        raise ValueError("selection must be first or hashed")
    fine = {normalize_tag(tag) for tag in fine_tags}
    coarse = {normalize_tag(tag) for tag in coarse_tags}
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    order = "i.id, c.id" if selection == "first" else "((i.id * 1103515245 + ?) & 2147483647), i.id, c.id"
    limit_clause = " LIMIT ?" if limit is not None else ""
    query = f"""
        SELECT i.id, i.image_path, i.width, i.height, c.tag_data, c.content
        FROM dataset_items AS i
        JOIN dataset_captions AS c ON c.item_id = i.id
        WHERE i.dataset_id = ? AND c.is_tags_format = 1
        ORDER BY {order}{limit_clause}
    """
    parameters: list[int] = [dataset_id]
    if selection == "hashed":
        parameters.append(seed)
    if limit is not None:
        parameters.append(limit)
    aggregated: dict[int, dict[str, Any]] = {}
    with sqlite3.connect(uri, uri=True) as connection:
        for item_id, image_path, width, height, tag_data, content in connection.execute(
            query, parameters
        ):
            if not image_path or not width or not height:
                continue
            row = aggregated.setdefault(item_id, {
                "image_path": str(image_path),
                "width": int(width),
                "height": int(height),
                "tags": set(),
            })
            row["tags"].update(
                tag for tag in (
                    normalize_tag(value)
                    for value in extract_caption_tags(tag_data, content)
                ) if tag in fine or tag in coarse
            )
    records = []
    for item_id, row in aggregated.items():
        if require_files and not os.path.isfile(row["image_path"]):
            continue
        tags = row["tags"]
        records.append(WeakBenchmarkRecord(
            item_id=item_id,
            image_path=row["image_path"],
            width=row["width"],
            height=row["height"],
            fine_tags=tuple(sorted(tags & fine)),
            coarse_tags=tuple(sorted(tags & coarse)),
        ))
    return records


def build_weak_benchmark(
    records: Sequence[WeakBenchmarkRecord],
    *,
    dataset_id: int,
    name: str,
    positive_per_tag: int = 32,
    negative_count: int = 128,
    seed: int = 0,
    requested_fine_tags: Iterable[str] = DEFAULT_FINE_TAGS,
    label_provenance: str = "unspecified",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if positive_per_tag <= 0 or negative_count < 0:
        raise ValueError("positive_per_tag must be positive and negative_count non-negative")
    selected: dict[int, WeakBenchmarkRecord] = {}
    requested_tags = sorted({normalize_tag(tag) for tag in requested_fine_tags})
    available_tags = sorted({tag for record in records for tag in record.fine_tags})
    coverage: dict[str, int] = {}
    for tag in available_tags:
        candidates = [record for record in records if tag in record.fine_tags]
        candidates.sort(key=lambda record: _rank(seed, record.item_id, tag))
        chosen = candidates[:positive_per_tag]
        coverage[tag] = len(chosen)
        selected.update((record.item_id, record) for record in chosen)

    negatives = [record for record in records if not record.fine_tags]
    negatives.sort(key=lambda record: _rank(seed, record.item_id, "negative"))
    selected.update((record.item_id, record) for record in negatives[:negative_count])
    ordered = sorted(selected.values(), key=lambda record: _rank(seed, record.item_id, "order"))
    items = []
    for record in ordered:
        positive_tags = tuple(sorted(set(record.fine_tags) | set(record.coarse_tags)))
        items.append({
            "sample_id": f"dataset-{dataset_id}-item-{record.item_id}",
            "image_path": record.image_path,
            "width": record.width,
            "height": record.height,
            "positive_tags": list(positive_tags),
            "fine_detail_tags": list(record.fine_tags),
            "coarse_tags": list(record.coarse_tags),
            "evidence": [],
        })
    raw = {
        "format_version": BENCHMARK_FORMAT_VERSION,
        "name": name,
        "evidence_mode": "dense_oracle",
        "resolution_long_sides": [384, 512, 768, 1024, 1536],
        "items": items,
        "provenance": {
            "dataset_id": dataset_id,
            "selection": "deterministic weak-label stratification",
            "seed": seed,
            "positive_per_tag": positive_per_tag,
            "negative_count": negative_count,
            "label_provenance": label_provenance,
            "requested_fine_tags": requested_tags,
            "uncovered_fine_tags": sorted(set(requested_tags) - set(available_tags)),
        },
    }
    BenchmarkManifest.from_dict(raw)
    summary = {
        "records_scanned": len(records),
        "items_selected": len(items),
        "fine_tag_coverage": coverage,
        "uncovered_fine_tags": sorted(set(requested_tags) - set(available_tags)),
        "negative_items": sum(not record.fine_tags for record in ordered),
    }
    return raw, summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-db", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--positive-per-tag", type=int, default=32)
    parser.add_argument("--negative-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scan-limit", type=int)
    parser.add_argument("--selection", choices=("first", "hashed"), default="first")
    parser.add_argument("--skip-file-check", action="store_true")
    parser.add_argument("--label-provenance", default="unspecified")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    records = load_weak_records(
        args.datasets_db,
        args.dataset_id,
        require_files=not args.skip_file_check,
        limit=args.scan_limit,
        selection=args.selection,
        seed=args.seed,
    )
    manifest, summary = build_weak_benchmark(
        records,
        dataset_id=args.dataset_id,
        name=args.name,
        positive_per_tag=args.positive_per_tag,
        negative_count=args.negative_count,
        seed=args.seed,
        label_provenance=args.label_provenance,
    )
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
