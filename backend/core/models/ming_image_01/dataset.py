"""Native Ming target/reference records, explicit layer roles and paired crops."""

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps
import torch


@dataclass(frozen=True)
class MingAsset:
    path: Path
    sha256: str
    file_stamp: tuple[int, ...] | None = None


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
    pixel_identity: str | None = None


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


def _file_stamp(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _header_size(data):
    with Image.open(io.BytesIO(data)) as image:
        size = image.size
        if image.format == "PNG":
            from PIL.PngImagePlugin import PngStream
            # PNG.getexif() calls load(); metadata may also follow IDAT.
            stream = PngStream(io.BytesIO(data))
            position = 8
            try:
                while position + 12 <= len(data):
                    length = int.from_bytes(data[position:position + 4], "big")
                    cid = data[position + 4:position + 8]
                    end = position + 12 + length
                    if end > len(data):
                        raise ValueError("Truncated Ming PNG metadata")
                    if cid in {b"eXIf", b"iTXt", b"zTXt", b"tEXt"}:
                        stream.fp.seek(position + 8)
                        stream.call(cid, position + 8, length)
                    if cid == b"IEND":
                        break
                    position = end
                image.info.update(stream.im_info)
            finally:
                stream.close()
            orientation = Image.Image.getexif(image).get(274)
        else:
            orientation = image.getexif().get(274)
        if orientation in (5, 6, 7, 8):
            size = size[::-1]
        return size


def inspect_item(primary, *, root, profile, reference_paths=(), validated_sizes=None, quick_plain=False):
    """Build native supervision identities before bucket/model construction."""
    root = Path(root).resolve(strict=True)
    primary = _inside(root, primary)
    if profile not in {"generation_edit", "layer_decompose"}:
        raise ValueError("Unknown Ming dataset profile")
    manifest = sidecar_path(primary)
    plain = profile == "generation_edit" and not manifest.exists() and not reference_paths
    stamp = _file_stamp(primary) if plain and quick_plain else None
    if plain and quick_plain:
        cached = (validated_sizes or {}).get(str(primary))
        if isinstance(cached, dict) and cached.get("file_stamp") == list(stamp):
            sha, size = cached.get("sha256"), cached.get("source_size")
            if (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
                    and isinstance(size, list) and len(size) == 2
                    and all(type(v) is int and v > 0 for v in size)):
                return MingDatasetRecord(profile, root, None, tuple(size),
                    (MingAsset(primary, sha, stamp),), (), ("image",), (1.0,), None)
    primary_data = primary.read_bytes()
    primary_sha = hashlib.sha256(primary_data).hexdigest()
    if plain:
        cached = (validated_sizes or {}).get(str(primary))
        if isinstance(cached, dict) and cached.get("sha256") == primary_sha:
            size = cached.get("source_size")
            if isinstance(size, list) and len(size) == 2 and all(type(v) is int and v > 0 for v in size):
                if quick_plain and _file_stamp(primary) != stamp:
                    raise ValueError("Ming dataset asset changed during preflight")
                return MingDatasetRecord(profile, root, None, tuple(size),
                    (MingAsset(primary, primary_sha, stamp),), (), ("image",), (1.0,), None)
        if quick_plain:
            size = _header_size(primary_data)
            if _file_stamp(primary) != stamp:
                raise ValueError("Ming dataset asset changed during preflight")
            return MingDatasetRecord(profile, root, None, size,
                (MingAsset(primary, primary_sha, stamp),), (), ("image",), (1.0,), None)
    primary_image = _decode(primary_data)
    source_size = primary_image.size
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
        data = primary_data if path == primary else path.read_bytes()
        image = primary_image if path == primary else _decode(data)
        if image.size != source_size:
            raise ValueError("Ming paired target/reference assets require the source canvas geometry")
        if profile == "layer_decompose" and index < len(target_paths) and image.mode != "RGBA":
            raise ValueError("Ming Layer target assets require explicit RGBA alpha")
        assets.append(MingAsset(path, primary_sha if path == primary else hashlib.sha256(data).hexdigest()))
    return MingDatasetRecord(profile, root, manifest_path, source_size, tuple(assets[:len(target_paths)]),
        tuple(assets[len(target_paths):]), tuple(roles), tuple(weights), manifest_sha)


def verify_record_content(record):
    if record.manifest_path is not None:
        manifest = _inside(record.root, record.manifest_path)
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != record.manifest_sha256:
            raise ValueError("Ming dataset sidecar changed after preflight; rebuild its record/cache")
    payloads = []
    for asset in (*record.targets, *record.references):
        data = _inside(record.root, asset.path).read_bytes()
        if hashlib.sha256(data).hexdigest() != asset.sha256:
            raise ValueError("Ming dataset asset changed after preflight; rebuild its record/cache")
        payloads.append(data)
    return payloads


def verify_record_files(record):
    """Avoid reading pixels for plain caption-only or already-prefilled batches."""
    if (record.manifest_path is None and not record.references
            and all(asset.file_stamp is not None for asset in record.targets)):
        if sidecar_path(record.targets[0].path).exists():
            raise ValueError("Ming dataset sidecar changed after preflight; rebuild its record/cache")
        for asset in record.targets:
            if _file_stamp(_inside(record.root, asset.path)) != asset.file_stamp:
                raise ValueError("Ming dataset asset changed after preflight; rebuild its record/cache")
    else:
        verify_record_content(record)


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
    images = []
    for data in verify_record_content(record):
        image = _decode(data)
        if image.size != record.source_size:
            raise ValueError("Ming dataset source geometry changed after preflight")
        image = image.convert("RGBA").crop((x, y, x + w, y + h))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        if flip_horizontal:
            image = ImageOps.mirror(image)
        images.append(image)
    targets = [torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 127.5 - 1
               for image in images[:len(record.targets)]]
    return MingPixelItem(record, torch.stack(targets, dim=1), images[len(record.targets):],
                        torch.tensor(record.weights, dtype=torch.float32), caption,
                        pixel_cache_identity(record, width=width, height=height, crop_box=crop_box, flip_horizontal=flip_horizontal))


def record_cache_options(components, items, *, resolution, cache_context=None):
    if not items or any(item.record.profile != components.artifact.profile for item in items):
        raise ValueError("Ming records and loaded profile disagree")
    sizes = {tuple(item.pixels.shape) for item in items}
    if len(sizes) != 1:
        raise ValueError("Ming batches require matching grids and target-frame counts")
    options = {}
    if cache_context is not None:
        from .cache import latent_identity, conditioning_identity, validate_cache_context
        context = cache_context
        validate_cache_context(components, context)
        if any(item.pixel_identity is None for item in items):
            raise ValueError("Ming persistent records require paired pixel identities")
        pixels = [item.pixel_identity for item in items]
        options["tensor_cache"] = context["cache"]
        if context["latent_enabled"]:
            if components.vae.dtype != context["vae_dtype"]:
                raise ValueError("Ming VAE precision changed after cache identity capture")
            options["latent_key"] = latent_identity(pixels=pixels, vae_weights=context["vae_weights"],
                vae_config=context["vae_config"], dtype=components.vae.dtype, execution=context["execution"])
        if context["conditioning_enabled"] and components.conditioner.cacheable:
            dtype = next(components.conditioner.parameters()).dtype
            if dtype != context["conditioner_dtype"]:
                raise ValueError("Ming conditioner precision changed after cache identity capture")
            options["conditioning_key"] = conditioning_identity(pixels=pixels,
                captions=[item.caption for item in items], artifact_config=context["artifact_config"],
                conditioner_weights=context["conditioner_weights"], resolution=resolution, dtype=dtype,
                topology=getattr(components.transformer, "_residual_unet_spec", None), execution=context["execution"])
    return options


def prepare_record_conditioning(components, items, *, resolution, cache_context=None, residency=None):
    from .prompting import prepare_inputs
    from .training_data import MingConditioningBatch
    options = record_cache_options(components, items, resolution=resolution, cache_context=cache_context)
    height, width = items[0].pixels.shape[-2:]
    prepared = prepare_inputs(components.processor, [item.caption for item in items],
        references=[item.references for item in items], width=width, height=height, resolution=resolution,
        vae_references=False)
    return MingConditioningBatch(prepared, components.conditioner,
        tensor_cache=options.get("tensor_cache"), conditioning_key=options.get("conditioning_key"), residency=residency)


def prepare_record_latents(components, items, *, resolution, cache_context=None, vae_device=None, residency=None):
    from .training_data import prepare_latent_batch
    options = record_cache_options(components, items, resolution=resolution, cache_context=cache_context)
    return prepare_latent_batch(components, prompts=[item.caption for item in items],
        references=[item.references for item in items], target_pixels=torch.stack([item.pixels for item in items]),
        frame_weights=torch.stack([item.weights for item in items]), resolution=resolution,
        vae_device=vae_device, residency=residency, **options)


def prepare_record_batch(components, items, *, resolution):
    return prepare_record_latents(components, items, resolution=resolution).for_step()


def pixel_cache_identity(record, *, width, height, crop_box, flip_horizontal):
    payload = {"version": "ming_rgba_independent_frames_v1", "profile": record.profile,
        "targets": [asset.sha256 for asset in record.targets], "references": [asset.sha256 for asset in record.references],
        "roles": record.roles, "manifest": record.manifest_sha256, "source_size": record.source_size,
        "width": width, "height": height, "crop_box": crop_box, "flip_horizontal": flip_horizontal}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def supervision_identity(record):
    payload = {"version": "ming_supervision_v2", "paired_crop": "ming_batch_crop_v1", "profile": record.profile,
        "source_size": record.source_size, "manifest": record.manifest_sha256,
        "targets": [asset.sha256 for asset in record.targets],
        "references": [asset.sha256 for asset in record.references],
        "roles": record.roles, "weights": record.weights}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def annotate_datasets(datasets, *, profile, progress_callback=None, stop_check=None, validation_cache_path=None):
    """Attach native records, excluding Layer assets that are not primary samples."""
    pending, selected = [], []
    validated_sizes = {}
    if validation_cache_path is not None and profile == "generation_edit":
        cache_path = Path(validation_cache_path)
        try:
            if cache_path.stat().st_size <= 32 * 1024 * 1024:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and type(cached.get("version")) is int and cached["version"] in (1, 2) and isinstance(cached.get("sizes"), dict):
                    validated_sizes = cached["sizes"]
        except (OSError, ValueError):
            pass
    total = sum(len(dataset.items) for dataset in datasets)
    processed = 0
    for dataset in datasets:
        root = getattr(dataset, "dataset_root", None)
        if root is None:
            raise ValueError("Ming datasets require an explicit dataset_root")
        primaries, loose = [], []
        for item in dataset.items:
            if stop_check is not None:
                stop_check()
            if progress_callback is not None and processed % 64 == 0:
                progress_callback(phase="initializing", step=processed, total=total,
                    detail=f"Validating Ming images: {processed}/{total}")
            processed += 1
            if item.get("item_type", "single") not in {"single", "image"}:
                raise ValueError("Ming training targets are images, not video/audio")
            if profile == "layer_decompose" and not sidecar_path(item["image_path"]).exists():
                loose.append(item)
                continue
            record = inspect_item(item["image_path"], root=root, profile=profile,
                                  reference_paths=item.get("reference_images") or (), validated_sizes=validated_sizes,
                                  quick_plain=True)
            primaries.append((item, record))
        assets = {asset.path for _, record in primaries for asset in (*record.targets, *record.references)}
        for item in loose:
            if _inside(Path(root).resolve(strict=True), item["image_path"]) not in assets:
                raise ValueError("Ming Layer primary images require an explicit .ming.json sidecar")
        pending.extend(primaries)
        selected.append((dataset, [item for item, _ in primaries]))
    if validation_cache_path is not None and profile == "generation_edit":
        sizes = {str(record.targets[0].path): {"sha256": record.targets[0].sha256,
                 "source_size": list(record.source_size),
                 "file_stamp": list(record.targets[0].file_stamp)} for _, record in pending
                 if record.manifest_path is None and not record.references}
        cache_path = Path(validation_cache_path)
        temporary = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_path.parent,
                                             prefix=cache_path.name+".", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump({"version": 2, "sizes": sizes}, handle, allow_nan=False)
            os.replace(temporary, cache_path)
        except OSError as exc:
            print(f"[Ming Dataset] Validation cache could not be saved: {exc}")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
    for item, record in pending:
        item["_ming_record"] = record
        item["_ming_batch_signature"] = (record.profile, len(record.targets))
        item["_ming_supervision_identity"] = supervision_identity(record)
        item["reference_images"] = [str(asset.path) for asset in record.references]
    for dataset, items in selected:
        dataset.items = items
    if progress_callback is not None:
        progress_callback(phase="initializing", step=total, total=total,
            detail=f"Validating Ming images: {total}/{total}")
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


def paired_crop_box(strategy, source_size, width, height, *, seed=None):
    import random
    ow, oh = source_size
    if strategy == "resize":
        return None
    if strategy == "crop":
        cw, ch = min(ow, max(1, round(oh * width / height))), min(oh, max(1, round(ow * height / width)))
        return ((ow-cw)//2, (oh-ch)//2, cw, ch)
    if strategy == "random_crop":
        if type(seed) is not int:
            raise ValueError("Ming random crop requires an explicit replay seed")
        scale = max(1, width/ow, height/oh)
        cw, ch = min(ow, max(1, round(width/scale))), min(oh, max(1, round(height/scale)))
        rng = random.Random(seed)
        return (rng.randint(0, ow-cw), rng.randint(0, oh-ch), cw, ch)
    raise ValueError("Unknown Ming paired bucket strategy")


def batch_crop_seed(trainer, record, *, item_index, width, height, epoch, batch_position):
    seed = getattr(trainer, "run_seed", None)
    if type(seed) is not int or any(type(value) is not int or value < 0 for value in (epoch, batch_position, item_index)):
        raise ValueError("Ming random crop requires a resolved run seed and nonnegative batch coordinates")
    identity = ["ming_batch_crop_v1", seed, epoch, batch_position, item_index,
                record.targets[0].sha256, record.source_size, width, height]
    return int.from_bytes(hashlib.sha256(json.dumps(identity).encode()).digest(), "big")


def load_shared_items(trainer, pairs, *, epoch=None, batch_position=None):
    """Replay paired transforms without consuming the shared augmentation RNG."""
    items = []
    epoch = getattr(trainer, "_current_epoch", 0) if epoch is None else epoch
    batch_position = getattr(trainer, "_current_batch_position", 0) if batch_position is None else batch_position
    for index, (item, _dataset) in enumerate(pairs):
        record = item["_ming_record"]
        width = item.get("bucket_width") or item["width"]
        height = item.get("bucket_height") or item["height"]
        spec = item.get("_crop_spec")
        crop = spec.crop_box if spec is not None else None
        if crop is None:
            seed = (batch_crop_seed(trainer, record, item_index=index, width=width, height=height,
                                   epoch=epoch, batch_position=batch_position)
                    if trainer.bucket_strategy == "random_crop" else None)
            crop = paired_crop_box(trainer.bucket_strategy, record.source_size, width, height, seed=seed)
        items.append(load_pixels(record, caption=item.get("caption", ""), width=width, height=height, crop_box=crop))
    return items


def shared_conditioning_context(trainer, pairs):
    from .cache import dataset_cache_context
    context = getattr(trainer, "ming_cache_context", None)
    if context is None or not context.get("dataset_owned"):
        return context
    ids = [getattr(dataset, "unique_id", None) for _item, dataset in pairs]
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Ming persistent batches require stable dataset IDs")
    # Batch padding is native; its owner is stable even for mixed-dataset batches.
    return dataset_cache_context({**context, "latent_enabled": False},
        base_cache_dir=context["dataset_cache_base"], dataset_unique_id=min(ids))


def prepare_shared_items_latents(trainer, pairs, items, *, residency=None):
    from .cache import dataset_cache_context
    from .training_data import MingLatentBatch
    context = getattr(trainer, "ming_cache_context", None)
    components = trainer.ming_image_01_components
    policy = getattr(trainer, "ming_training_residency", None) if residency is None else residency
    options = dict(resolution=trainer.ming_prompt_resolution, vae_device=trainer.device, residency=policy)
    if context is None or not context.get("dataset_owned"):
        return prepare_record_latents(components, items, cache_context=context, **options)
    if len(items) != len(pairs):
        raise ValueError("Ming native items and dataset ownership must align")
    record_cache_options(components, items, resolution=trainer.ming_prompt_resolution)
    condition_context = shared_conditioning_context(trainer, pairs)
    if not context["latent_enabled"]:
        return prepare_record_latents(components, items, cache_context=condition_context, **options)
    item_contexts = [dataset_cache_context({**context, "conditioning_enabled": False},
        base_cache_dir=context["dataset_cache_base"], dataset_unique_id=dataset.unique_id)
        for _source, dataset in pairs]
    from .training_residency import grouped_encoder_phase
    clean, references = [], []
    with grouped_encoder_phase(policy, "vae") as grouped:
        for item, item_context in zip(items, item_contexts):
            encoded = prepare_record_latents(components, [item], cache_context=item_context,
                                            **{**options, "residency": grouped})
            clean.append(encoded.clean.detach().cpu())
            reference = None if encoded.reference_latents is None else encoded.reference_latents[0]
            references.append(None if reference is None else reference.detach().cpu())
    source = prepare_record_conditioning(components, items, resolution=trainer.ming_prompt_resolution,
        cache_context=condition_context, residency=policy)
    return MingLatentBatch(torch.cat(clean), source.prepared, torch.stack([item.weights for item in items]),
        references if any(value is not None for value in references) else None, source.conditioner,
        tensor_cache=source.tensor_cache, conditioning_key=source.conditioning_key, residency=policy)


def prepare_shared_latents(trainer, pairs):
    """Encode paired native pixels once per logical batch before the MNT window."""
    return prepare_shared_items_latents(trainer, pairs, load_shared_items(trainer, pairs))


def shared_inputs_for_mnt(trainer, prepared):
    from .training_data import shared_step_inputs
    return shared_step_inputs(prepared.for_step(encoder_device=trainer.device))


def prefill_shared_batches(trainer, batches, *, epoch, batch_offset=0, progress_callback=None):
    context = getattr(trainer, "ming_cache_context", None)
    if context is None or not (context["latent_enabled"] or context["conditioning_enabled"]):
        return
    for index, pairs in enumerate(batches):
        trainer._check_stop_requested()
        items = load_shared_items(trainer, pairs, epoch=epoch, batch_position=batch_offset+index)
        options = dict(resolution=trainer.ming_prompt_resolution, cache_context=context,
                       residency=getattr(trainer, "ming_training_residency", None))
        if context["latent_enabled"]:
            prepared = prepare_shared_items_latents(trainer, pairs, items)
            if context["conditioning_enabled"]:
                prepared.for_step(encoder_device=trainer.device)
        else:
            options["cache_context"] = shared_conditioning_context(trainer, pairs)
            prepared = prepare_record_conditioning(trainer.ming_image_01_components, items, **options)
            prepared.for_step(encoder_device=trainer.device)
        del prepared, items
        if progress_callback is not None:
            progress_callback(phase="latent_cache" if context["latent_enabled"] else "text_encoder_cache",
                              step=index+1, total=len(batches))
