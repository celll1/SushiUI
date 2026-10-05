"""CPU-only concept-order contracts shared by API, handlers and native loops."""

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class ConceptOrderSupport:
    supported: bool
    reason: str = ""


_IMAGE = ConceptOrderSupport(True)
CONCEPT_ORDER_SUPPORT = {
    arch: _IMAGE for arch in (
        "sd15", "sdxl", "zimage", "flux2", "anima", "lens", "ideogram4",
        "minit2i", "krea2", "qwen_image_21", "ming_image_01", "sensenova",
        "sensenova_sdxl_chimera",
    )
}
for _arch in ("ltx2", "minimax_h3", "acestep"):
    CONCEPT_ORDER_SUPPORT[_arch] = ConceptOrderSupport(False,
        "Concept ordering requires image samples; clip/audio duration batching is not supported")
CONCEPT_ORDER_SUPPORT["yue2"] = ConceptOrderSupport(False,
    "YuE2's token-native ABC loop has no occurrence-aware concept batch plan or caption encoder")


def validate_concept_order(arch, config, *, text_encoding_mode, datasets=(), dataset_configs=(), training_method=None):
    if config is None:
        return
    if training_method == "vae_decoder":
        raise ValueError("concept_batch_order requires image generation training, not VAE-only training")
    support = CONCEPT_ORDER_SUPPORT.get(arch)
    if support is None:
        raise ValueError(f"concept_batch_order has no declared training contract for {arch}")
    if not support.supported:
        raise ValueError(f"concept_batch_order is unsupported for {arch}: {support.reason}")
    if config.multiplied and text_encoding_mode == "pre_encoded_cache":
        raise ValueError("Concept sample multiplier requires per-exposure text encoding; "
                         "use swap_onthefly, cpu_prefetch, or onthefly_gpu")
    if arch == "sensenova" and (
        any(row.get("task_views") for row in dataset_configs)
        or any(item.get("_sensenova_task_views") for dataset in datasets for item in dataset.items)
    ):
        raise ValueError("SenseNova concept_batch_order supports ordinary image generation only; "
                         "explicit task_views regroup items and replace occurrence captions")
    if any(item.get("item_type") in ("video", "audio")
           for dataset in datasets for item in dataset.items):
        raise ValueError("concept_batch_order supports image items only")


def concept_batch_key(item):
    spatial = (int(item.get("bucket_width") or item.get("width") or 0),
               int(item.get("bucket_height") or item.get("height") or 0))
    reference = bool(item.get("reference_images") or item.get("_ve_reconstruction_mode"))
    if "_concept_batch_signature" in item:
        return (*spatial, reference, json.dumps(item["_concept_batch_signature"], sort_keys=True, separators=(",", ":")))
    if "_ming_batch_signature" in item:
        return (*spatial, *item["_ming_batch_signature"])
    return (*spatial, reference)
