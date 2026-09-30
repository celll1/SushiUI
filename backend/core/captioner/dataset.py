"""Immutable item-first caption snapshots and grouped holdout splits."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

from .storage import canonical_hash, file_hash
from core.tagger.tag_vocabulary import _parse_caption_tags, normalize_tag


class _HashTree:
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, index, {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance not in node[2]:
                node[2][distance] = [value, index, {}]
                return
            node = node[2][distance]

    def nearby(self, value, radius):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                yield node[1]
            pending.extend(child for edge, child in node[2].items() if abs(edge - distance) <= radius)


def image_identity(path):
    content = file_hash(path)
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        small = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        pixels = list(small.getdata())
        difference = sum(int(pixels[row * 9 + col] > pixels[row * 9 + col + 1]) << (row * 8 + col)
                         for row in range(8) for col in range(8))
        digest = hashlib.sha256(str(image.size).encode())
        digest.update(image.tobytes())
        pixel_hash = digest.hexdigest()
    return content, pixel_hash, difference


def split_records(records, *, seed, val_ratio, test_ratio, near_duplicate_distance):
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("Invalid holdout ratios")
    if not 0 <= near_duplicate_distance <= 16:
        raise ValueError("Near-duplicate distance must be 0..16")
    parent = list(range(len(records)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def unite(left, right):
        left, right = root(left), root(right)
        parent[max(left, right)] = min(left, right)

    exact, groups, tree = {}, {}, _HashTree()
    for index, record in enumerate(records):
        for key in (record["image_hash"], record["pixel_hash"]):
            if key in exact:
                unite(index, exact[key])
            exact[key] = index
        group = record.get("source_group")
        if group:
            if group in groups:
                unite(index, groups[group])
            groups[group] = index
        if near_duplicate_distance:
            value = int(record["difference_hash"])
            for match in tree.nearby(value, near_duplicate_distance):
                unite(index, match)
            tree.add(value, index)
    components = {}
    for index in range(len(records)):
        components.setdefault(root(index), []).append(index)
    identities = {group: min(records[i]["pixel_hash"] for i in members)
                  for group, members in components.items()}
    ordered = sorted(components, key=lambda group: canonical_hash([seed, identities[group]]))
    total = len(ordered)
    val_count = max(1, round(total * val_ratio)) if val_ratio else 0
    test_count = max(1, round(total * test_ratio)) if test_ratio else 0
    if total <= val_count + test_count:
        raise ValueError("Too few independent image groups for the requested holdouts")
    assignments = {}
    for position, group in enumerate(ordered):
        split = "val" if position < val_count else "test" if position < val_count + test_count else "train"
        for index in components[group]:
            assignments[index] = (split, identities[group])
    return [{**record, "split": assignments[i][0], "duplicate_group": assignments[i][1]}
            for i, record in enumerate(records)]


def build_snapshot(db, configs, *, seed, val_ratio, test_ratio, near_duplicate_distance, cancelled):
    from database.models import Dataset, DatasetItem, DatasetCaption
    records, revisions, counts = [], {}, Counter()
    seen_datasets = set()
    for config in configs:
        dataset_id = config["dataset_id"]
        if dataset_id in seen_datasets:
            raise ValueError("Dataset appears more than once")
        seen_datasets.add(dataset_id)
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
        if dataset is None:
            raise ValueError(f"Dataset not found: {dataset_id}")
        revisions[str(dataset_id)] = dataset.revision
        root = Path(dataset.path).resolve()
        targets = config["target_caption_types"]
        inputs = config["input_tag_caption_types"]
        if not targets or any(t in inputs for t in targets):
            raise ValueError("Select separate natural target and tag input caption types")
        filters = config.get("filters", {})
        if set(filters) - {"item_ids", "min_width", "min_height"}:
            raise ValueError("Captioner filters support item_ids, min_width, min_height")
        items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id,
                                             DatasetItem.item_type.notin_(["audio", "video"]))
        if "item_ids" in filters:
            items = items.filter(DatasetItem.id.in_(filters["item_ids"]))
        if "min_width" in filters:
            items = items.filter(DatasetItem.width >= filters["min_width"])
        if "min_height" in filters:
            items = items.filter(DatasetItem.height >= filters["min_height"])
        for item in items.order_by(DatasetItem.id).yield_per(1000):
            if cancelled():
                raise InterruptedError("Dataset preparation cancelled")
            captions = db.query(DatasetCaption).filter(DatasetCaption.item_id == item.id).order_by(DatasetCaption.id).all()
            natural = list(dict.fromkeys(c.content.strip() for c in captions
                           if c.caption_type in targets and not c.is_tags_format and c.content and c.content.strip()))
            tags = list(dict.fromkeys(normalize_tag(tag) for c in captions
                        if c.caption_type in inputs and c.is_tags_format for tag, _ in _parse_caption_tags(c)))
            counts["indexed_items"] += 1
            if not natural:
                counts["missing_targets"] += 1
                continue
            path = Path(item.image_path)
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if not path.is_relative_to(root):
                raise ValueError("Dataset image escapes registered root")
            try:
                content, pixels, difference = image_identity(path)
            except (OSError, ValueError, Image.DecompressionBombError):
                counts["unreadable_images"] += 1
                continue
            if not tags:
                counts["missing_input_tags"] += 1
            records.append({"dataset_id": dataset_id, "item_id": item.id, "path": str(path),
                            "image_hash": content, "pixel_hash": pixels, "difference_hash": difference,
                            "targets": natural, "tags": tags,
                            "source_group": f"{dataset_id}:{item.group_id}" if item.group_id else None})
        db.expire(dataset)
        if dataset.revision != revisions[str(dataset_id)]:
            raise ValueError("Dataset changed while preparing caption snapshot; retry")
    if not records:
        raise ValueError("No eligible image/caption records")
    records = split_records(records, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio,
                            near_duplicate_distance=near_duplicate_distance)
    payload = {"version": 1, "dataset_revisions": revisions, "records": records,
               "coverage": dict(counts), "split_counts": dict(Counter(r["split"] for r in records))}
    payload["snapshot_hash"] = canonical_hash(payload)
    return payload


def verify_snapshot(snapshot):
    payload = {key: value for key, value in snapshot.items() if key != "snapshot_hash"}
    if canonical_hash(payload) != snapshot["snapshot_hash"]:
        raise ValueError("Caption snapshot identity mismatch")
    for record in snapshot["records"]:
        if file_hash(record["path"]) != record["image_hash"]:
            raise ValueError(f"Dataset image changed for item {record['item_id']}")
