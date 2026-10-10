"""Initialization lineage, independent of subsequent fine-tuning weights."""
from copy import deepcopy
import json
import re

from .hybrid_spec import validate_hybrid_spec
from .spec import MAX_JSON_BYTES, canonical_json, digest, _unique_object
from .resolution_schedule import RESOLUTION_SAMPLING, initialization_resolutions

META_RECORD = "sushi.hybrid.initialization"
META_HASH = "sushi.hybrid.initialization_sha256"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_KEYS = {"schema_version", "method", "source_sha256", "origin_spec",
         "calibration_sha256", "sample_count", "seed", "phases", "initialized_weights_sha256"}
RECIPE_FIELDS = {"sample_count", "holdout_count", "resolution", "seed", "caption_type", "teacher_steps",
                 "teacher_cfg", "conditioning_dropout", "clip_norm", "optimizer", "stochastic_rounding",
                 "weight_decay", "attention_backend"}
RESOLUTION_RECIPE_FIELDS = {"additional_resolutions", "resolution_sampling"}


def validate_record(value, current_spec):
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_JSON_BYTES:
            raise ValueError("Hybrid initialization record exceeds size limit")
        try:
            value = json.loads(value, object_pairs_hook=_unique_object)
        except (RecursionError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid hybrid initialization JSON") from exc
    if not isinstance(value, dict) or not _KEYS <= set(value) or set(value)-_KEYS-{"recipe"}:
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
        required = {"kind", "updates", "learning_rate"}
        if (not isinstance(phase, dict) or not required <= set(phase)
                or set(phase) - required - {"skip_learning_rate_multiplier"}):
            raise ValueError("Invalid hybrid initialization phase")
        if phase["kind"] not in ("data_noising", "native_trajectory"):
            raise ValueError("Unsupported hybrid initialization phase")
        if type(phase["updates"]) is not int or not 1 <= phase["updates"] <= 1000000:
            raise ValueError("Invalid hybrid initialization update count")
        if type(phase["learning_rate"]) not in (float, int) or not 0 < phase["learning_rate"] <= 1:
            raise ValueError("Invalid hybrid initialization learning rate")
        if "skip_learning_rate_multiplier" in phase:
            multiplier = phase["skip_learning_rate_multiplier"]
            if (type(multiplier) not in (float, int) or not 0 < multiplier <= 1
                    or phase["learning_rate"] * multiplier <= 0):
                raise ValueError("Invalid hybrid initialization skip learning-rate multiplier")
            if multiplier < 1 and not origin["skips"]:
                raise ValueError("Skip learning-rate multiplier requires compressed skips")
    result = deepcopy(value)
    result["origin_spec"] = origin
    if "recipe" in value:
        recipe = value["recipe"]
        if (not isinstance(recipe, dict) or set(recipe) != {"schema_version", "settings"}
                or type(recipe["schema_version"]) is not int or recipe["schema_version"] != 1
                or not isinstance(recipe["settings"], dict)
                or not RECIPE_FIELDS <= set(recipe["settings"])
                or set(recipe["settings"]) - RECIPE_FIELDS - RESOLUTION_RECIPE_FIELDS):
            raise ValueError("Invalid hybrid initialization recipe")
        settings = recipe["settings"]
        resolution_fields = set(settings) & RESOLUTION_RECIPE_FIELDS
        if resolution_fields:
            if (resolution_fields != RESOLUTION_RECIPE_FIELDS
                    or settings["resolution_sampling"] != RESOLUTION_SAMPLING
                    or len(initialization_resolutions(settings)) < 2):
                raise ValueError("Invalid initialization resolution curriculum")
        for key, low, high in (("sample_count",1,64),("holdout_count",0,63),("resolution",64,1024),
                               ("seed",0,2**31-1),("teacher_steps",1,100)):
            if type(settings[key]) is not int or not low <= settings[key] <= high:
                raise ValueError(f"Invalid initialization recipe {key}")
        if (settings["resolution"]%64 or settings["holdout_count"] >= settings["sample_count"]
                or settings["sample_count"] != value["sample_count"] or settings["seed"] != value["seed"]):
            raise ValueError("Initialization recipe disagrees with calibration or seed")
        for key, low, high in (("teacher_cfg",1,20),("conditioning_dropout",0,1),
                               ("clip_norm",0,100),("weight_decay",0,1)):
            if type(settings[key]) not in (int,float) or not low <= settings[key] <= high:
                raise ValueError(f"Invalid initialization recipe {key}")
        if settings["clip_norm"] == 0 or type(settings["stochastic_rounding"]) is not bool:
            raise ValueError("Invalid initialization clipping or stochastic rounding")
        if (settings["optimizer"] not in ("adamw", "adamw8bit")
                or settings["attention_backend"] not in ("native", "flash")
                or not isinstance(settings["caption_type"],str) or not 1 <= len(settings["caption_type"]) <= 64):
            raise ValueError("Unsupported hybrid initialization recipe setting")
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
