"""Pure topology validation; no model imports or device allocation."""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy

VERSION = 1
METHOD = "residual_delta_unet"
META_VERSION = "sushi.rewire.schema_version"
META_SPEC = "sushi.rewire.spec"
META_HASH = "sushi.rewire.spec_sha256"
MAX_JSON_BYTES = 65536
OPERATORS = dict(down="avg_pool", up="bilinear", align_corners=False,
                 merge="identity_delta_add", position_policy="coarse_integer",
                 grid_policy="require_divisible", delta_dtype="float32")
_SPEC_KEYS = {"schema_version", "method", "architecture", "streams", "regions",
              "base_config_sha256", "block_layout_sha256"}
_REGION_KEYS = {"id", "stream", "parent", "start_block", "end_block", "down_factor", *OPERATORS}
_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _integer(value, label, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer between {low} and {high}")
    return value


def _name(value, label):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"Invalid {label}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate topology key: {key}")
        result[key] = value
    return result


def validate_spec(value):
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_JSON_BYTES:
            raise ValueError("Topology JSON exceeds size limit")
        try:
            value = json.loads(value, object_pairs_hook=_unique_object)
        except (RecursionError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid topology JSON") from exc
    if isinstance(value, dict) and value.get("method") == "hybrid_unet_ssdit":
        from .hybrid_spec import validate_hybrid_spec
        return validate_hybrid_spec(value)
    if not isinstance(value, dict) or set(value) != _SPEC_KEYS:
        raise ValueError("Topology has missing or unsupported fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != VERSION or value["method"] != METHOD:
        raise ValueError("Unsupported topology version or method")
    _name(value["architecture"], "architecture")
    streams = value["streams"]
    if not isinstance(streams, list) or not 1 <= len(streams) <= 16:
        raise ValueError("Topology requires 1 to 16 streams")
    counts = {}
    for stream in streams:
        if not isinstance(stream, dict) or set(stream) != {"name", "block_count"}:
            raise ValueError("Invalid stream declaration")
        _name(stream["name"], "stream name")
        if stream["name"] in counts:
            raise ValueError("Duplicate stream name")
        counts[stream["name"]] = _integer(stream["block_count"], "block_count", 3, 4096)
    ordered_streams = sorted(streams, key=lambda s: s["name"])
    if value["block_layout_sha256"] != digest(ordered_streams):
        raise ValueError("Block-layout hash mismatch")
    if not isinstance(value["base_config_sha256"], str) or not _HASH.fullmatch(value["base_config_sha256"]):
        raise ValueError("Invalid base config hash")
    regions = value["regions"]
    if not isinstance(regions, list) or not 1 <= len(regions) <= 64:
        raise ValueError("Topology requires 1 to 64 regions")
    ids = set()
    for region in regions:
        if not isinstance(region, dict) or set(region) != _REGION_KEYS:
            raise ValueError("Region has missing or unsupported fields")
        _name(region["id"], "region id")
        if region["id"] in ids:
            raise ValueError("Duplicate region id")
        ids.add(region["id"])
        _name(region["stream"], "region stream")
        if region["stream"] not in counts:
            raise ValueError("Unknown region stream")
        start = _integer(region["start_block"], "start_block", 1, counts[region["stream"]] - 2)
        end = _integer(region["end_block"], "end_block", 2, counts[region["stream"]] - 1)
        if start >= end:
            raise ValueError("Region must have start_block < end_block")
        factor = region["down_factor"]
        if (not isinstance(factor, list) or len(factor) != 2
                or any(type(v) is not int or v not in (1, 2, 4, 8) for v in factor) or factor == [1, 1]):
            raise ValueError("down_factor must contain two supported factors and reduce at least one axis")
        for key, expected in OPERATORS.items():
            if key == "down" and region[key] in ("avg_pool", "rms_pool"):
                continue
            if type(region[key]) is not type(expected) or region[key] != expected:
                raise ValueError(f"Unsupported region {key}")
        if region["parent"] is not None:
            _name(region["parent"], "parent id")
    ordered = sorted(regions, key=lambda r: (r["stream"], r["start_block"], -r["end_block"]))
    stack = []
    for region in ordered:
        while stack and (stack[-1]["stream"] != region["stream"] or stack[-1]["end_block"] <= region["start_block"]):
            stack.pop()
        if stack and not (stack[-1]["start_block"] < region["start_block"] < region["end_block"] < stack[-1]["end_block"]):
            raise ValueError("Regions must be strictly nested or disjoint")
        if region["parent"] != (stack[-1]["id"] if stack else None):
            raise ValueError("Region parent does not match its enclosing interval")
        stack.append(region)
        if len(stack) > 8:
            raise ValueError("Topology depth exceeds eight")
    result = deepcopy(value)
    result.update(streams=deepcopy(ordered_streams), regions=deepcopy(ordered))
    if len(canonical_json(result).encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("Topology JSON exceeds size limit")
    return result


def build_spec(architecture, streams, regions, base_config):
    return validate_spec(dict(schema_version=VERSION, method=METHOD,
        architecture=architecture, streams=streams, regions=regions,
        base_config_sha256=digest(base_config),
        block_layout_sha256=digest(sorted(streams, key=lambda s: s["name"]))))


def preview_plan(spec, token_height, token_width):
    spec = validate_spec(spec)
    hybrid = spec if spec["method"] == "hybrid_unet_ssdit" else None
    if spec["method"] == "hybrid_unet_ssdit":
        from .hybrid_spec import legacy_topology
        spec = legacy_topology(spec)
    h = _integer(token_height, "token_height", 1, 65536)
    w = _integer(token_width, "token_width", 1, 65536)
    levels = {}
    edges = []
    for region in spec["regions"]:
        parent = levels.get(region["parent"])
        ph, pw = (parent["height"], parent["width"]) if parent else (h, w)
        fh, fw = region["down_factor"]
        if ph % fh or pw % fw:
            raise ValueError(f"Region {region['id']} grid is not divisible by down_factor")
        levels[region["id"]] = dict(id=region["id"], depth=(parent["depth"] if parent else 0)+1,
                                   height=ph//fh, width=pw//fw, tokens=(ph//fh)*(pw//fw))
        edges.append(dict(region=region["id"], stream=region["stream"],
                          source=region["start_block"], target=region["end_block"], merge=region["merge"]))
    blocks = []
    for stream in spec["streams"]:
        for index in range(stream["block_count"]):
            active = [r for r in spec["regions"] if r["stream"] == stream["name"] and r["start_block"] <= index < r["end_block"]]
            level = levels[active[-1]["id"]] if active else dict(depth=0, height=h, width=w, tokens=h*w)
            blocks.append(dict(stream=stream["name"], index=index,
                               **{k: level[k] for k in ("depth", "height", "width", "tokens")}))
    result = dict(levels=list(levels.values()), skip_edges=edges, blocks=blocks)
    if hybrid is not None:
        from .hybrid_partition import convolution_halo
        from .hybrid_spec import hybrid_parameter_counts
        result["hybrid_parameter_counts"] = hybrid_parameter_counts(hybrid)
        result["hybrid_convolution_halo"] = {stream["name"]: list(convolution_halo(hybrid, stream["name"]))
                                              for stream in hybrid["streams"]}
    return result


def checkpoint_metadata(spec):
    spec = validate_spec(spec)
    return {META_VERSION: str(spec["schema_version"]), META_SPEC: canonical_json(spec), META_HASH: digest(spec)}


def read_checkpoint_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be an object")
    keys = {META_VERSION, META_SPEC, META_HASH}
    present = keys.intersection(metadata)
    if not present:
        return None
    if present != keys or metadata[META_VERSION] not in ("1", "2"):
        raise ValueError("Incomplete or unsupported checkpoint topology declaration")
    if any(not isinstance(metadata[key], str) for key in keys):
        raise ValueError("Checkpoint topology metadata values must be strings")
    spec = validate_spec(metadata[META_SPEC])
    if metadata[META_VERSION] != str(spec["schema_version"]):
        raise ValueError("Checkpoint topology version disagrees with spec")
    if metadata[META_HASH] != digest(spec):
        raise ValueError("Checkpoint topology hash mismatch")
    return spec


def rebase_spec(spec, base_config):
    spec = validate_spec(spec)
    return validate_spec(dict(spec, base_config_sha256=digest(base_config)))


def validate_model_identity(spec, architecture, streams, base_config):
    spec = validate_spec(spec)
    if spec["architecture"] != architecture:
        raise ValueError("Checkpoint topology architecture mismatch")
    if spec["streams"] != sorted(streams, key=lambda stream: stream["name"]):
        raise ValueError("Checkpoint topology block layout mismatch")
    if spec["base_config_sha256"] != digest(base_config):
        raise ValueError("Checkpoint topology native config mismatch")
    return spec
