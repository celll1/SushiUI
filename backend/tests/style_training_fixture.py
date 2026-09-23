"""Shared CPU fixtures for the style training tests: temporary datasets.db, tiny SigLIP2, stub resolvers."""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# The trainer stack imports bitsandbytes, which creates a CUDA context; these tests never use the GPU.
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from transformers import Siglip2ImageProcessor, Siglip2VisionConfig, Siglip2VisionModel

from core.style_encoder.style_labels import StyleLabelResolver
from core.style_encoder.style_trainer import StyleEnvironment
from core.tagger.tag_alias_resolver import TagAliasResolver
from database.models import Dataset, DatasetBase, DatasetCaption, DatasetItem

PATCH = 2
MAX_PATCHES = 16


class PrefixCategories:
    """artist_* -> Artist, char_* -> Character, cp_* -> Copyright, anything else General."""

    def get_tag_group(self, tag: str) -> Optional[str]:
        for prefix, category in (("artist ", "Artist"), ("char ", "Character"), ("cp ", "Copyright")):
            if tag.startswith(prefix):
                return category
        return "General"


def label_resolver(_cfg=None) -> StyleLabelResolver:
    return StyleLabelResolver(PrefixCategories(), None, TagAliasResolver({}), ("signature",), ("original",))


def image_processor(_repo=None) -> Siglip2ImageProcessor:
    return Siglip2ImageProcessor(patch_size=PATCH, max_num_patches=MAX_PATCHES)


def tiny_vision(source=None, repo_id=None, attn_implementation="sdpa") -> Siglip2VisionModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return Siglip2VisionModel(Siglip2VisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2, patch_size=PATCH,
            num_patches=MAX_PATCHES, attn_implementation=attn_implementation))


def session_factory(db_path: Path):
    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    DatasetBase.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def add_dataset(factory, name: str, root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    with factory() as session:
        dataset = Dataset(name=name, path=str(root))
        session.add(dataset)
        session.commit()
        return dataset.id


def add_item(factory, dataset_id: int, image_path: Path, captions: list[tuple[str, Optional[list], Optional[str], bool]],
             color: Optional[tuple[int, int, int]] = None, size=(8, 12), write: bool = True) -> int:
    """captions: (caption_type, tag_data rows or None, content or None, is_tags_format)."""
    if write:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(zlib.crc32(f"{image_path.name}{color}".encode()))
        pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
        if color is not None:
            pixels[:, : size[0] // 2] = color
        Image.fromarray(pixels).save(image_path)
    with factory() as session:
        item = DatasetItem(dataset_id=dataset_id, base_name=image_path.stem, image_path=str(image_path))
        session.add(item)
        session.flush()
        for caption_type, tag_data, content, is_tags in captions:
            session.add(DatasetCaption(item_id=item.id, caption_type=caption_type, content=content or "",
                                       tag_data=json.dumps(tag_data) if tag_data is not None else None,
                                       is_tags_format=is_tags))
        session.commit()
        return item.id


def artist_names(count: int) -> list[str]:
    return [f"artist_zqxw{a:02d}" for a in range(count)]


def build_corpus(tmp_path: Path, artists: int = 10, per_artist: int = 4, corrupt: bool = False):
    """Two datasets; artist a's images split across both, each with a copyright and a character tag."""
    factory = session_factory(tmp_path / "datasets.db")
    roots = [tmp_path / "ds_a", tmp_path / "ds_b"]
    ids = [add_dataset(factory, f"ds{i}", root) for i, root in enumerate(roots)]
    names = artist_names(artists)
    for a, name in enumerate(names):
        for k in range(per_artist):
            d = k % 2
            path = roots[d] / f"a{a:02d}" / f"{k}.png"
            tags = f"{name}, cp_series{a % 3}, char_hero{(a + k) % 4}, 1girl" + (", signature" if k == 0 else "")
            add_item(factory, ids[d], path, [("tags", None, tags, True)], color=(20 * a, 255 - 20 * a, 7 * k))
            if corrupt:
                path.write_bytes(b"not an image")
    return factory, ids, names


def environment(factory) -> StyleEnvironment:
    return StyleEnvironment(open_datasets_session=factory, build_label_resolver=label_resolver,
                            load_image_processor=image_processor, load_vision_encoder=tiny_vision,
                            resolve_device=lambda _gpu: torch.device("cpu"))


def config(dataset_ids, **overrides) -> dict:
    values = {
        "dataset_configs": [{"dataset_id": d, "caption_types": []} for d in dataset_ids],
        "vision_encoder_path": "google/siglip2-base-patch16-naflex",
        "vision_encoder_repo": "google/siglip2-base-patch16-naflex",
        "training_method": "full", "lora_rank": 2, "lora_alpha": 2.0, "lora_dropout": 0.0,
        "attn_implementation": "sdpa", "mixed_precision": "bf16", "gradient_checkpointing": True,
        "gpu_index": None, "optimizer": "adamw", "learning_rate": 1e-3, "head_lr_multiplier": 10.0,
        "weight_decay": 1e-4, "warmup_steps": 1, "min_lr_ratio": 0.0, "max_grad_norm": 1.0, "epochs": 1,
        "num_workers": 0, "save_every_n_steps": 0, "keep_last_n_checkpoints": 2, "validate_every": 1,
        "log_every_n_steps": 1, "style_seed": 5, "style_split_seed": 0, "style_use_tag_aliases": True,
        "style_use_gelbooru_categories": False, "style_signature_tags": ["signature"],
        "style_non_copyright_tags": ["original"], "style_val_artist_ratio": 0.3, "style_val_image_ratio": 0.1,
        "style_min_images_per_artist": 2, "style_artists_per_batch": 2, "style_images_per_artist": 2,
        "style_same_copyright_batch_ratio": 0.5, "style_artist_sampling_power": 0.5, "style_mask_ratio_min": 0.3,
        "style_mask_ratio_max": 0.75, "style_max_num_patches": MAX_PATCHES, "style_temperature": 0.1,
        "style_view_positive_weight": 1.0, "style_positive_weight_min": 0.3, "style_feature_layers": [1, -2],
        "style_hsic_tagger_path": "",
        "style_hsic_weights": {"Character": 0.1, "Copyright": 0.1, "Rating": 0.1, "Quality": 0.0},
        "style_token_dim": 8, "style_pool_heads": 2, "style_projector_hidden_dim": 12, "style_embed_dim": 6,
        "style_val_max_items": 0, "style_val_batch_size": 5, "style_eval_chunk_size": 3,
    }
    values.update(overrides)
    return values
