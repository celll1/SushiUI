"""Initialization lineage, independent of subsequent fine-tuning weights."""
from copy import deepcopy
import json
import re

from .hybrid_spec import validate_hybrid_spec
from .spec import MAX_JSON_BYTES, canonical_json, digest, _unique_object

META_RECORD = "sushi.hybrid.initialization"
META_HASH = "sushi.hybrid.initialization_sha256"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_KEYS = {"schema_version", "method", "source_sha256", "origin_spec",
         "calibration_sha256", "sample_count", "seed", "phases", "initialized_weights_sha256"}


def validate_record(value, current_spec):
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_JSON_BYTES:
            raise ValueError("Hybrid initialization record exceeds size limit")
        try:
            value = json.loads(value, object_pairs_hook=_unique_object)
        except (RecursionError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid hybrid initialization JSON") from exc
    if not isinstance(value, dict) or set(value) != _KEYS:
        raise ValueError("Hybrid initialization record has missing or unsupported fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported hybrid initialization record version")
    if value["method"] != "native_teacher_velocity":
        raise ValueError("Unsupported hybrid initialization method")
    for key in ("source_sha256", "calibration_sha256", "initialized_weights_sha256"):
        if not isinstance(value[key], str) or not _HASH.fullmatch(value[key]):
            raise ValueError(f"Invalid hybrid initialization {key}")
    origin = validate_hybrid_spec(value["origin_spec"])
    current = validate_hybrid_spec(current_spec)
    # Codec rebasing and skip residency do not replace the learned modules.
    if any(origin[key] != current[key] for key in
           ("architecture", "streams", "module_layout_sha256")):
        raise ValueError("Hybrid initialization belongs to different native blocks or modules")
    for key, low, high in (("sample_count", 1, 1000000), ("seed", 0, 2**63-1)):
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError(f"Invalid hybrid initialization {key}")
    phases = value["phases"]
    if not isinstance(phases, list) or not 1 <= len(phases) <= 32:
        raise ValueError("Hybrid initialization requires bounded fitting phases")
    for phase in phases:
        if not isinstance(phase, dict) or set(phase) != {"kind", "updates", "learning_rate"}:
            raise ValueError("Invalid hybrid initialization phase")
        if phase["kind"] not in ("data_noising", "native_trajectory"):
            raise ValueError("Unsupported hybrid initialization phase")
        if type(phase["updates"]) is not int or not 1 <= phase["updates"] <= 1000000:
            raise ValueError("Invalid hybrid initialization update count")
        if type(phase["learning_rate"]) not in (float, int) or not 0 < phase["learning_rate"] <= 1:
            raise ValueError("Invalid hybrid initialization learning rate")
    result = deepcopy(value)
    result["origin_spec"] = origin
    if len(canonical_json(result).encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("Hybrid initialization record exceeds size limit")
    return result


def checkpoint_metadata(record, spec):
    if record is None:
        return {}
    record = validate_record(record, spec)
    return {META_RECORD: canonical_json(record), META_HASH: digest(record)}


def read_checkpoint_metadata(metadata, spec):
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be an object")
    present = {META_RECORD, META_HASH}.intersection(metadata)
    if not present:
        return None
    if present != {META_RECORD, META_HASH} or any(
            not isinstance(metadata[key], str) for key in present):
        raise ValueError("Incomplete hybrid initialization metadata")
    record = validate_record(metadata[META_RECORD], spec)
    if metadata[META_HASH] != digest(record):
        raise ValueError("Hybrid initialization record hash mismatch")
    return record


def restore(model, metadata):
    record = read_checkpoint_metadata(metadata, getattr(model, "_residual_unet_spec", None))
    restore_record(model, record)


def restore_record(model, record):
    if record is None:
        model.__dict__.pop("_hybrid_initialization", None)
    else:
        model._hybrid_initialization = validate_record(record, getattr(model, "_residual_unet_spec", None))


def model_metadata(model):
    return checkpoint_metadata(getattr(model, "_hybrid_initialization", None),
                               getattr(model, "_residual_unet_spec", None))
