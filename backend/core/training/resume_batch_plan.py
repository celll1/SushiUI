"""Checkpoint-bound batch ledgers for dataset-selection changes during resume."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path
import tempfile
from typing import Any


VERSION = 1


def _encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str).encode("utf-8")


def digest(value: Any) -> str:
    return sha256(_encoded(value)).hexdigest()


def item_key(item: dict, dataset: Any) -> tuple[str, str]:
    return str(dataset.unique_id), str(item["image_path"])


def _batch_key(item: dict) -> str:
    return digest([item.get("bucket_width", item.get("width")),
                   item.get("bucket_height", item.get("height")),
                   item.get("reference_images"), item.get("_ve_reconstruction_mode"),
                   item.get("condition_image_path"),
                   item.get("_crop_spec")])


def dataset_manifest(datasets: list[Any], mode: str) -> dict[str, dict]:
    result = {}
    for dataset in datasets:
        dataset_id = str(dataset.unique_id)
        if dataset_id in result:
            raise ValueError(f"Duplicate training dataset ID: {dataset_id}")
        rows = []
        captions = []
        seen = set()
        for item in dataset.items:
            path = str(item["image_path"])
            if path in seen:
                raise ValueError(f"Duplicate image path in dataset {dataset_id}: {path}")
            seen.add(path)
            row = [path, item.get("width"), item.get("height"),
                   item.get("item_type"), item.get("reference_images"),
                   item.get("_ve_reconstruction_mode")]
            if mode in ("priority", "concept"):
                row.extend([item.get("raw_caption"), item.get("tag_data"),
                            item.get("is_tags_format"), item.get("_captions_by_type")])
            rows.append(row)
            captions.append([path, item.get("raw_caption"), item.get("tag_data")])
        result[dataset_id] = {
            "count": len(rows), "order_hash": digest(rows),
            "caption_hash": digest([captions, getattr(dataset, "caption_config", None)]),
        }
    return result


def make_plan(batches: list, datasets: list[Any], mode: str, signature: str,
              epoch: int, priority_keys: set[tuple[str, str]] | None = None,
              batch_size: int = 1) -> dict:
    counts: Counter[tuple[str, str]] = Counter()
    encoded_batches = []
    for batch in batches:
        encoded_batch = []
        for item, dataset in batch:
            key = item_key(item, dataset)
            occurrence = counts[key]
            counts[key] += 1
            kind = ("priority" if priority_keys and key in priority_keys else
                    "replay" if mode == "concept" and occurrence > 0 else "base")
            encoded_batch.append([*key, occurrence,
                                  kind, _batch_key(item),
                                  [*key, 0] if kind == "replay" else None])
        if encoded_batch:
            encoded_batches.append(encoded_batch)
    manifest = dataset_manifest(datasets, mode)
    return {"version": VERSION, "epoch": epoch, "revision": 0,
            "mode": mode, "signature": signature, "batch_size": batch_size,
            "selected": list(manifest), "history_manifest": manifest,
            "batches": encoded_batches}


def save_plan(output_dir: Path, run_name: str, plan: dict) -> dict:
    raw = _encoded(plan)
    checksum = sha256(raw).hexdigest()
    name = f"{run_name}_epoch_{plan['epoch']:04d}_batch_plan_{checksum[:24]}.json.gz"
    path = output_dir / name
    valid_existing = False
    if path.exists():
        try:
            valid_existing = gzip.decompress(path.read_bytes()) == raw
        except (OSError, EOFError):
            pass
    if not valid_existing:
        with tempfile.NamedTemporaryFile(mode="wb", dir=output_dir,
                                         prefix=".batch_plan_", suffix=".tmp",
                                         delete=False) as tmp:
            temp_path = Path(tmp.name)
            try:
                tmp.write(gzip.compress(raw, mtime=0))
                tmp.flush()
                os.fsync(tmp.fileno())
            except BaseException:
                temp_path.unlink(missing_ok=True)
                raise
        try:
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    return {"version": VERSION, "file": name, "sha256": checksum,
            "size": path.stat().st_size,
            "epoch": plan["epoch"], "revision": plan["revision"],
            "batch_count": len(plan["batches"])}


def load_plan(output_dir: Path, reference: dict) -> dict:
    if not isinstance(reference, dict) or reference.get("version") != VERSION:
        raise ValueError("Cannot resume batch plan: unsupported or absent ledger reference")
    name = reference.get("file")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Cannot resume batch plan: invalid ledger filename")
    path = output_dir / name
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Cannot resume batch plan: unreadable ledger {name}") from exc
    if size != reference.get("size"):
        raise ValueError("Cannot resume batch plan: ledger size mismatch")
    try:
        raw = gzip.decompress(path.read_bytes())
        plan = json.loads(raw)
    except (OSError, EOFError, ValueError) as exc:
        raise ValueError(f"Cannot resume batch plan: unreadable ledger {name}") from exc
    if sha256(raw).hexdigest() != reference.get("sha256"):
        raise ValueError("Cannot resume batch plan: ledger checksum mismatch")
    if (plan.get("version") != VERSION or plan.get("epoch") != reference.get("epoch")
            or plan.get("revision") != reference.get("revision")
            or len(plan.get("batches", [])) != reference.get("batch_count")):
        raise ValueError("Cannot resume batch plan: ledger metadata mismatch")
    return plan


def _interleave(old: list, added: list) -> list:
    if not added:
        return old
    merged = []
    for index, batch in enumerate(added, 1):
        stop = index * len(old) // len(added)
        start = (index - 1) * len(old) // len(added)
        merged.extend(old[start:stop])
        merged.append(batch)
    return merged


def rebase_plan(old: dict, cursor: int, candidate: dict) -> tuple[dict, dict]:
    if old["epoch"] != candidate["epoch"] or old["mode"] != candidate["mode"]:
        raise ValueError("Cannot rebase batch plan: epoch or batch mode changed")
    if (old["signature"] != candidate["signature"]
            or old["batch_size"] != candidate["batch_size"]):
        raise ValueError("Cannot rebase batch plan: order settings changed")
    if not 0 <= cursor <= len(old["batches"]):
        raise ValueError("Cannot rebase batch plan: invalid completed-batch cursor")
    prior = old["selected"]
    current = candidate["selected"]
    shared = set(prior) & set(current)
    if [key for key in prior if key in shared] != [key for key in current if key in shared]:
        raise ValueError("Cannot rebase batch plan: retained datasets were reordered")
    history = dict(old["history_manifest"])
    caption_changed = []
    for key in current:
        manifest = candidate["history_manifest"][key]
        if key in history and history[key]["order_hash"] != manifest["order_hash"]:
            raise ValueError(f"Cannot rebase batch plan: dataset {key} changed internally")
        if key in history and history[key]["caption_hash"] != manifest["caption_hash"]:
            caption_changed.append(key)
        history[key] = manifest
    if prior == current:
        unchanged = ({**old, "revision": old["revision"] + 1,
                      "history_manifest": history} if caption_changed else old)
        return unchanged, {"added_datasets": [], "removed_datasets": [],
                     "completed_batches": cursor,
                     "remaining_batches": len(old["batches"]) - cursor,
                     "added_occurrences": 0, "removed_occurrences": 0,
                     "caption_changed": caption_changed}
    completed = old["batches"][:cursor]
    completed_ids = {(entry[0], entry[1], entry[2])
                     for batch in completed for entry in batch}
    retained = set(current)
    removed = set(prior) - retained
    added = set(current) - set(prior)
    suffix = [[entry for entry in batch if entry[0] not in removed]
              for batch in old["batches"][cursor:]]
    suffix = [batch for batch in suffix if batch]
    incoming = []
    for batch in candidate["batches"]:
        entries = [entry for entry in batch if entry[0] in added
                   and (entry[0], entry[1], entry[2]) not in completed_ids
                   and (old["mode"] != "concept" or entry[2] == 0)]
        if entries:
            incoming.append(entries)
    if old["mode"] == "priority":
        focused = [[entry for entry in batch if entry[3] == "priority"] for batch in incoming]
        ordinary = [[entry for entry in batch if entry[3] != "priority"] for batch in incoming]
        suffix = ([batch for batch in focused if batch] + suffix
                  + [batch for batch in ordinary if batch])
    elif old["mode"] == "concept":
        suffix = (incoming + suffix if candidate.get("placement") == "front"
                  else _interleave(suffix, incoming))
    else:
        suffix = _interleave(suffix, incoming)
    revised = {**old, "revision": old["revision"] + 1,
               "selected": current, "history_manifest": history,
               "batches": completed + suffix}
    stats = {"added_datasets": sorted(added), "removed_datasets": sorted(removed),
             "completed_batches": cursor, "remaining_batches": len(suffix),
             "added_occurrences": sum(map(len, incoming)),
             "caption_changed": caption_changed,
             "partial_batches": sum(len(batch) < old["batch_size"] for batch in suffix),
             "effective_batch_size": (sum(map(len, suffix)) / len(suffix) if suffix else 0),
             "removed_occurrences": sum(
                 entry[0] in removed for batch in old["batches"][cursor:]
                 for entry in batch)}
    return revised, stats


def resolve_suffix(plan: dict, cursor: int, candidate_batches: list) -> list:
    lookup = {item_key(item, dataset): (item, dataset)
              for batch in candidate_batches for item, dataset in batch}
    result = []
    for batch in plan["batches"][cursor:]:
        resolved = []
        for dataset_id, path, _occurrence, _kind, batch_key, _source in batch:
            key = dataset_id, path
            if key not in lookup:
                raise ValueError(f"Cannot resume batch plan: missing current item {key}")
            if _batch_key(lookup[key][0]) != batch_key:
                raise ValueError(f"Cannot resume batch plan: item conditions changed for {key}")
            resolved.append(lookup[key])
        result.append(resolved)
    return result
