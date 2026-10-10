"""Strict, experimental fixed-width hybrid graph; legacy declarations stay unchanged."""
from copy import deepcopy
import json

from .spec import canonical_json, digest, validate_spec, preview_plan, _name, _integer, _unique_object, MAX_JSON_BYTES

METHOD = "hybrid_unet_ssdit"
VERSION = 2
META_SPEC = "sushi.hybrid.spec"
META_HASH = "sushi.hybrid.spec_sha256"
EXTRA_KEYS = {"stream_widths", "boundaries", "refiners", "skips", "module_layout_sha256"}


def legacy_topology(value):
    result = {key: deepcopy(item) for key, item in value.items() if key not in EXTRA_KEYS}
    result.update(schema_version=1, method="residual_delta_unet")
    return validate_spec(result)


def module_layout(value):
    result = {key: value[key] for key in ("stream_widths", "boundaries", "refiners")}
    result["skips"] = [{key: item for key, item in skip.items() if key != "storage"} if isinstance(skip, dict) else skip
                       for skip in value["skips"]]
    return result


def skip_storage(item):
    from api.param_defaults import DIT_HYBRID_DEFAULTS
    return item.get("storage", DIT_HYBRID_DEFAULTS["skip_storage"])


def boundary_fusion(item):
    from api.param_defaults import DIT_HYBRID_DEFAULTS
    return item.get("fusion", DIT_HYBRID_DEFAULTS["boundary_fusion"])


def validate_hybrid_spec(value):
    if isinstance(value, str):
        if len(value.encode()) > MAX_JSON_BYTES:
            raise ValueError("Hybrid topology JSON exceeds size limit")
        try:
            value = json.loads(value, object_pairs_hook=_unique_object)
        except (RecursionError, json.JSONDecodeError) as error:
            raise ValueError("Invalid hybrid JSON") from error
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value.get("schema_version") != VERSION or value.get("method") != METHOD):
        raise ValueError("Unsupported hybrid topology version or method")
    base = legacy_topology(value)
    if base["architecture"] not in ("anima", "ming_image_01", "qwen_image_21"):
        raise ValueError("Hybrid native integration currently supports Anima, Ming and Qwen 2.1 only")
    if set(value) != set(base) | EXTRA_KEYS:
        raise ValueError("Hybrid topology has missing or unsupported fields")
    result = deepcopy(value)
    result.update({key: item for key, item in base.items() if key not in ("schema_version", "method")})
    widths = result["stream_widths"]
    counts = {stream["name"]: stream["block_count"] for stream in base["streams"]}
    if not isinstance(widths, dict) or set(widths) != set(counts):
        raise ValueError("Hybrid widths must cover exactly the declared streams")
    for width in widths.values():
        _integer(width, "stream width", 1, 16384)
    regions = {region["id"]: region for region in base["regions"]}
    names = set()
    def check(item, keys):
        if not isinstance(item, dict) or set(item) != set(keys):
            raise ValueError("Invalid hybrid module fields")
        _name(item["id"], "hybrid module ID")
        if item["id"] in names:
            raise ValueError("Duplicate hybrid module ID")
        names.add(item["id"])
    for key in ("boundaries", "refiners", "skips"):
        if not isinstance(result[key], list) or len(result[key]) > 128:
            raise ValueError("Hybrid module lists must contain at most 128 entries")
    boundary_regions = set()
    for item in result["boundaries"]:
        keys = ("id", "region", "rank", "kernel")
        check(item, (*keys, "fusion") if isinstance(item, dict) and "fusion" in item else keys)
        if boundary_fusion(item) not in ("none", "full_rank"):
            raise ValueError("Hybrid boundary fusion must be none or full_rank")
        if not isinstance(item["region"], str) or item["region"] not in regions or item["region"] in boundary_regions:
            raise ValueError("Hybrid boundary must own a unique existing region")
        region = regions[item["region"]]
        if region["down_factor"] != [2, 2]:
            raise ValueError("Initial hybrid boundaries require isotropic 2x downsampling")
        boundary_regions.add(item["region"])
        _integer(item["rank"], "boundary rank", 1, widths[region["stream"]])
        if type(item["kernel"]) is not int or item["kernel"] not in (3, 5):
            raise ValueError("Hybrid kernels must be 3 or 5")
    def block(item, field):
        if not isinstance(item["stream"], str) or item["stream"] not in counts:
            raise ValueError("Unknown hybrid module stream")
        _integer(item[field], field, 0, counts[item["stream"]] - 1)
        _integer(item["rank"], "module rank", 1, widths[item["stream"]])
    for item in result["refiners"]:
        check(item, ("id", "stream", "after_block", "rank", "kernel"))
        block(item, "after_block")
        if type(item["kernel"]) is not int or item["kernel"] not in (3, 5):
            raise ValueError("Hybrid kernels must be 3 or 5")
    def active(stream, index):
        return tuple(region["id"] for region in base["regions"] if region["stream"] == stream
                     and region["start_block"] <= index < region["end_block"])
    for item in result["skips"]:
        keys = ("id", "stream", "source_after", "destination_before", "rank")
        check(item, (*keys, "storage") if isinstance(item, dict) and "storage" in item else keys)
        if skip_storage(item) not in ("gpu", "cpu", "recompute"):
            raise ValueError("Hybrid skip storage must be gpu, cpu or recompute")
        block(item, "source_after")
        block(item, "destination_before")
        if item["source_after"] >= item["destination_before"]:
            raise ValueError("Hybrid skips must point forward")
        if active(item["stream"], item["source_after"]) != active(item["stream"], item["destination_before"]):
            raise ValueError("Hybrid skips must connect the same region grid")
    for item in result["skips"]:
        if skip_storage(item) == "recompute" and any(
                other["stream"] == item["stream"] and other["destination_before"] <= item["source_after"]
                for other in result["skips"]):
            raise ValueError("Hybrid encoder replay source must precede every skip destination in its stream")
    for key in ("boundaries", "refiners", "skips"):
        result[key].sort(key=lambda item: item["id"])
    if result["module_layout_sha256"] != digest(module_layout(result)):
        raise ValueError("Hybrid module-layout hash mismatch")
    if len(canonical_json(result).encode()) > MAX_JSON_BYTES:
        raise ValueError("Hybrid topology JSON exceeds size limit")
    return result


def build_hybrid_spec(base, *, stream_widths, boundaries, refiners, skips):
    result = deepcopy(validate_spec(base))
    result.update(schema_version=VERSION, method=METHOD, stream_widths=deepcopy(stream_widths),
                  boundaries=sorted(deepcopy(boundaries), key=lambda item: item["id"]),
                  refiners=sorted(deepcopy(refiners), key=lambda item: item["id"]),
                  skips=sorted(deepcopy(skips), key=lambda item: item["id"]))
    result["module_layout_sha256"] = digest(module_layout(result))
    return validate_hybrid_spec(result)


def default_hybrid_spec(base, stream_widths):
    """Opt-in fixed-width preset; mirror skips only between equal spatial grids."""
    from api.param_defaults import DIT_HYBRID_DEFAULTS
    base = legacy_topology(base) if base.get("method") == METHOD else validate_spec(base)
    settings = DIT_HYBRID_DEFAULTS
    boundaries, refiners, skips = [], [], []
    for region in base["regions"]:
        boundaries.append(dict(id=f"downup_{region['id']}", region=region["id"],
                               rank=min(settings["rank"], stream_widths[region["stream"]]), kernel=settings["kernel"]))
    for stream in base["streams"]:
        name, count = stream["name"], stream["block_count"]
        rank = min(settings["rank"], stream_widths[name])
        def active(index):
            return tuple(region["id"] for region in base["regions"] if region["stream"] == name
                         and region["start_block"] <= index < region["end_block"])
        for index in range(0, count, settings["refiner_interval"]):
            refiners.append(dict(id=f"local_{name}_{index}", stream=name, after_block=index,
                                 rank=rank, kernel=settings["kernel"]))
        for source in range(0, count // 2, settings["skip_interval"]):
            target = count - 1 - source
            if active(source) == active(target):
                skips.append(dict(id=f"skip_{name}_{source}_{target}", stream=name,
                                  source_after=source, destination_before=target, rank=rank,
                                  storage=settings["skip_storage"]))
    return build_hybrid_spec(base, stream_widths=stream_widths, boundaries=boundaries, refiners=refiners, skips=skips)


def hybrid_metadata(value):
    value = validate_hybrid_spec(value)
    return {META_SPEC: canonical_json(value), META_HASH: digest(value)}


def hybrid_parameter_counts(value):
    value = validate_hybrid_spec(value)
    widths = value["stream_widths"]
    regions = {region["id"]: region for region in value["regions"]}
    modules = []
    for item in value["boundaries"]:
        width = widths[regions[item["region"]]["stream"]]
        count = 2 * width + 2 * item["kernel"]**2 * width + 4 * width * item["rank"]
        if boundary_fusion(item) == "full_rank":
            count += 3 * width**2
        modules.append(dict(id=item["id"], kind="boundary", count=count))
    for item in value["refiners"]:
        width = widths[item["stream"]]
        count = 2 * width + item["kernel"]**2 * width + 2 * width * item["rank"]
        modules.append(dict(id=item["id"], kind="refiner", count=count))
    for item in value["skips"]:
        modules.append(dict(id=item["id"], kind="skip", count=2 * widths[item["stream"]] * item["rank"]))
    return dict(scope="learned_hybrid_modules", added=sum(item["count"] for item in modules), modules=modules)


def read_hybrid_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Hybrid metadata must be an object")
    present = {META_SPEC, META_HASH}.intersection(metadata)
    if not present:
        return None
    if present != {META_SPEC, META_HASH}:
        raise ValueError("Incomplete hybrid topology metadata")
    if any(not isinstance(metadata[key], str) for key in present):
        raise ValueError("Hybrid metadata values must be strings")
    value = validate_hybrid_spec(metadata[META_SPEC])
    if metadata[META_HASH] != digest(value):
        raise ValueError("Hybrid topology hash mismatch")
    return value


def hybrid_preview(value, height, width, *, batch_size, element_bytes):
    value = validate_hybrid_spec(value)
    _integer(batch_size, "preview batch", 1, 65536)
    _integer(element_bytes, "element bytes", 1, 16)
    preview = preview_plan(legacy_topology(value), height, width)
    blocks = {(block["stream"], block["index"]): block for block in preview["blocks"]}
    preview["learned_skip_edges"] = [dict(item, stored_bytes=(0 if skip_storage(item) == "recompute" else batch_size * item["rank"] * element_bytes *
        blocks[item["stream"], item["source_after"]]["tokens"])) for item in value["skips"]]
    preview["feature_fusion_workspace"] = feature_fusion_workspace(value, preview, height, width)
    return preview


def feature_fusion_workspace(value, preview, height, width):
    regions = {region["id"]:region for region in value["regions"]}
    levels = {level["id"]:level for level in preview["levels"]}
    return [dict(id=item["id"], region=item["region"],
        stream=regions[item["region"]]["stream"],
        elements=6 * (levels[regions[item["region"]]["parent"]]["tokens"]
            if regions[item["region"]]["parent"] else height*width) *
            value["stream_widths"][regions[item["region"]]["stream"]])
        for item in value["boundaries"] if boundary_fusion(item) == "full_rank"]


def activation_layout(value, preview, height, width):
    """Logical forward ownership, excluding autograd, aliases and temporary buffers."""
    levels = {level["id"]: level for level in preview["levels"]}
    blocks = {(block["stream"], block["index"]): block for block in preview["blocks"]}
    entries = []
    for region in value["regions"]:
        parent = levels.get(region["parent"])
        parent_tokens = parent["tokens"] if parent else height*width
        channels = value["stream_widths"][region["stream"]]
        for kind, tokens in (("bypass", parent_tokens), ("low_input", levels[region["id"]]["tokens"])):
            entries.append(dict(id=region["id"], kind=kind, stream=region["stream"],
                elements=tokens*channels, capture_block=region["start_block"], capture_phase="before",
                release_before_block=region["end_block"]))
    for item in value["skips"]:
        elements = blocks[item["stream"], item["source_after"]]["tokens"]*item["rank"]
        entries.append(dict(id=item["id"], kind="compressed_skip", stream=item["stream"],
            training_storage=skip_storage(item),
            elements=0 if skip_storage(item) == "recompute" else elements,
            recomputed_elements=elements if skip_storage(item) == "recompute" else 0,
            capture_block=item["source_after"], capture_phase="after",
            release_before_block=item["destination_before"]))
    streams = []
    for stream in value["streams"]:
        if any(item["stream"] == stream["name"] and skip_storage(item) == "recompute" for item in value["skips"]):
            entries.append(dict(id=stream["name"], kind="encoder_replay_input", stream=stream["name"],
                training_storage="cpu", elements=height*width*value["stream_widths"][stream["name"]],
                capture_block=0, capture_phase="before", release_before_block=stream["block_count"]))
        events = {}
        for entry in entries:
            if entry["stream"] != stream["name"]:
                continue
            start = 2*entry["capture_block"] + int(entry["capture_phase"] == "after")
            end = 2*entry["release_before_block"]
            events[start] = events.get(start, 0) + entry["elements"]
            events[end] = events.get(end, 0) - entry["elements"]
        live = peak = peak_phase = 0
        for phase, delta in sorted(events.items()):
            live += delta
            if live > peak:
                peak, peak_phase = live, phase
        if live:
            raise ValueError("Hybrid activation lifetime did not close")
        streams.append(dict(stream=stream["name"], peak_elements=peak,
                            peak_block=peak_phase//2, peak_phase="after" if peak_phase%2 else "before"))
    return dict(basis="per_image_grid_per_batch_item", entries=entries, streams=streams)
