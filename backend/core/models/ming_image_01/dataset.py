"""Native Ming target/reference records, explicit layer roles and paired crops."""

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch


@dataclass(frozen=True)
class MingAsset:
    path: Path
    sha256: str


@dataclass(frozen=True)
class MingDatasetRecord:
    profile: str
    root: Path
    manifest_path: Path | None
    source_size: tuple[int, int]
    targets: tuple[MingAsset, ...]
    references: tuple[MingAsset, ...]
    roles: tuple[str, ...]
    weights: tuple[float, ...]
    manifest_sha256: str | None

    def bucket_key(self, width, height):
        return self.profile, width, height, len(self.targets)


@dataclass
class MingPixelItem:
    record: MingDatasetRecord
    pixels: torch.Tensor
    references: list[Image.Image]
    weights: torch.Tensor
    caption: str


def sidecar_path(primary):
    primary = Path(primary)
    return primary.with_suffix(primary.suffix + ".ming.json")


def _inside(root, path):
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Ming dataset asset escapes its declared dataset root")
    return resolved


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Ming dataset sidecar contains duplicate keys")
        result[key] = value
    return result


def _decode(data):
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return ImageOps.exif_transpose(image).copy()


def inspect_item(primary, *, root, profile, reference_paths=()):
    """Validate every role and source asset once before bucket/model construction."""
    root = Path(root).resolve(strict=True)
    primary = _inside(root, primary)
    if profile not in {"generation_edit", "layer_decompose"}:
        raise ValueError("Unknown Ming dataset profile")
    primary_image = _decode(primary.read_bytes())
    source_size = primary_image.size
    manifest = sidecar_path(primary)
    manifest_sha = None
    manifest_path = None
    if manifest.exists():
        manifest = _inside(root, manifest)
        data = manifest.read_bytes()
        if len(data) > 1024 * 1024:
            raise ValueError("Ming dataset sidecar is too large")
        spec = json.loads(data, object_pairs_hook=_unique_object)
        if (not isinstance(spec, dict) or set(spec) != {"schema_version", "profile", "targets", "references"}
                or type(spec["schema_version"]) is not int or spec["schema_version"] != 1 or spec["profile"] != profile):
            raise ValueError("Ming dataset sidecar identity/schema mismatch")
        targets, references = spec["targets"], spec["references"]
        if not isinstance(targets, list) or not targets or not isinstance(references, list):
            raise ValueError("Ming dataset targets/references must be explicit ordered lists")
        target_paths, roles, weights = [], [], []
        for target in targets:
            if not isinstance(target, dict) or set(target) != {"path", "role", "weight"}:
                raise ValueError("Ming targets require explicit path, role and weight")
            target_paths.append(target["path"])
            roles.append(target["role"])
            weight = target["weight"]
            if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight) or weight < 0 or weight > float(np.finfo(np.float32).max):
                raise ValueError("Ming supervision weight must be finite, nonnegative and representable in float32")
            weights.append(float(np.float32(weight)))
        expected = ["image"] if profile == "generation_edit" else ["auxiliary_canvas", *[f"layer_{index}" for index in range(1, len(targets))]]
        if roles != expected or (profile == "layer_decompose" and (len(targets) < 2 or not references)):
            raise ValueError("Ming ordered target roles disagree with the loaded profile")
        total = sum(weights)
        if not total > 0 or not math.isfinite(total) or total > float(np.finfo(np.float32).max):
            raise ValueError("Ming dataset item requires positive total supervision weight")
        if any(not isinstance(path, str) or not path or Path(path).is_absolute() for path in [*target_paths, *references]):
            raise ValueError("Ming sidecar assets must be nonempty relative paths")
        target_paths = [manifest.parent / path for path in target_paths]
        reference_paths = [manifest.parent / path for path in references]
        manifest_sha = hashlib.sha256(data).hexdigest()
        manifest_path = manifest
    elif profile == "layer_decompose":
        raise ValueError("Ming Layer datasets require an explicit .ming.json supervision sidecar")
    else:
        target_paths, roles, weights = [primary], ["image"], [1.0]
    assets = []
    for index, path in enumerate([*target_paths, *reference_paths]):
        path = _inside(root, path)
        data = path.read_bytes()
        image = _decode(data)
        if image.size != source_size:
            raise ValueError("Ming paired target/reference assets require the source canvas geometry")
        if profile == "layer_decompose" and index < len(target_paths) and image.mode != "RGBA":
            raise ValueError("Ming Layer target assets require explicit RGBA alpha")
        assets.append(MingAsset(path, hashlib.sha256(data).hexdigest()))
    return MingDatasetRecord(profile, root, manifest_path, source_size, tuple(assets[:len(target_paths)]),
        tuple(assets[len(target_paths):]), tuple(roles), tuple(weights), manifest_sha)


def load_pixels(record, *, caption, width, height, crop_box=None, flip_horizontal=False):
    if not isinstance(caption, str):
        raise ValueError("Ming dataset caption must be text")
    if any(type(value) is not int or value <= 0 or value % 16 for value in (width, height)):
        raise ValueError("Ming pixel buckets require a positive 16-pixel aligned grid")
    if type(flip_horizontal) is not bool:
        raise ValueError("Ming paired flip must be boolean")
    if crop_box is None:
        crop_box = (0, 0, *record.source_size)
    if (not isinstance(crop_box, (tuple, list)) or len(crop_box) != 4
            or any(type(value) is not int for value in crop_box)):
        raise ValueError("Ming paired crop must be integer x/y/width/height")
    x, y, w, h = crop_box
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > record.source_size[0] or y + h > record.source_size[1]:
        raise ValueError("Ming paired crop escapes the source canvas")
    if record.manifest_path is not None:
        manifest = _inside(record.root, record.manifest_path)
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != record.manifest_sha256:
            raise ValueError("Ming dataset sidecar changed after preflight; rebuild its record/cache")
    images = []
    for asset in (*record.targets, *record.references):
        data = _inside(record.root, asset.path).read_bytes()
        if hashlib.sha256(data).hexdigest() != asset.sha256:
            raise ValueError("Ming dataset asset changed after preflight; rebuild its record/cache")
        image = _decode(data).convert("RGBA").crop((x, y, x + w, y + h))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        if flip_horizontal:
            image = ImageOps.mirror(image)
        images.append(image)
    targets = [torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 127.5 - 1
               for image in images[:len(record.targets)]]
    return MingPixelItem(record, torch.stack(targets, dim=1), images[len(record.targets):],
                        torch.tensor(record.weights, dtype=torch.float32), caption)


def prepare_record_batch(components, items, *, resolution):
    from .training_data import prepare_training_batch
    if not items or any(item.record.profile != components.artifact.profile for item in items):
        raise ValueError("Ming records and loaded profile disagree")
    sizes = {tuple(item.pixels.shape) for item in items}
    if len(sizes) != 1:
        raise ValueError("Ming batches require matching grids and target-frame counts")
    return prepare_training_batch(components, prompts=[item.caption for item in items],
        references=[item.references for item in items], target_pixels=torch.stack([item.pixels for item in items]),
        frame_weights=torch.stack([item.weights for item in items]), resolution=resolution)


def pixel_cache_identity(record, *, width, height, crop_box, flip_horizontal):
    payload = {"version": "ming_rgba_independent_frames_v1", "profile": record.profile,
        "targets": [asset.sha256 for asset in record.targets], "references": [asset.sha256 for asset in record.references],
        "roles": record.roles, "manifest": record.manifest_sha256, "source_size": record.source_size,
        "width": width, "height": height, "crop_box": crop_box, "flip_horizontal": flip_horizontal}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def supervision_identity(record):
    payload = {"version": "ming_supervision_v1", "profile": record.profile,
        "source_size": record.source_size, "manifest": record.manifest_sha256,
        "targets": [asset.sha256 for asset in record.targets],
        "references": [asset.sha256 for asset in record.references],
        "roles": record.roles, "weights": record.weights}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def annotate_datasets(datasets, *, profile):
    """Attach native records, excluding Layer assets that are not primary samples."""
    pending, selected = [], []
    for dataset in datasets:
        root = getattr(dataset, "dataset_root", None)
        if root is None:
            raise ValueError("Ming datasets require an explicit dataset_root")
        primaries, loose = [], []
        for item in dataset.items:
            if item.get("item_type", "image") != "image":
                raise ValueError("Ming training targets are images, not video/audio")
            if profile == "layer_decompose" and not sidecar_path(item["image_path"]).exists():
                loose.append(item)
                continue
            record = inspect_item(item["image_path"], root=root, profile=profile,
                                  reference_paths=item.get("reference_images") or ())
            primaries.append((item, record))
        assets = {asset.path for _, record in primaries for asset in (*record.targets, *record.references)}
        for item in loose:
            if _inside(Path(root).resolve(strict=True), item["image_path"]) not in assets:
                raise ValueError("Ming Layer primary images require an explicit .ming.json sidecar")
        pending.extend(primaries)
        selected.append((dataset, [item for item, _ in primaries]))
    for item, record in pending:
        item["_ming_record"] = record
        item["_ming_batch_signature"] = (record.profile, len(record.targets))
        item["_ming_supervision_identity"] = supervision_identity(record)
        item["reference_images"] = [str(asset.path) for asset in record.references]
    for dataset, items in selected:
        dataset.items = items
    return len(pending)


def homogeneous_batches(batches):
    """Preserve each planned boundary and split only incompatible native layouts."""
    result = []
    for batch in batches:
        groups = {}
        for pair in batch:
            item, _dataset = pair
            if "_ming_record" not in item or "_ming_batch_signature" not in item:
                raise ValueError("Ming batch item is missing its native preflight record")
            key = (*item["_ming_batch_signature"],
                   item.get("bucket_width") or item.get("width"),
                   item.get("bucket_height") or item.get("height"))
            groups.setdefault(key, []).append(pair)
        result.extend(groups.values())
    return result
