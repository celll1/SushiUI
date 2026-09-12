import json
import sys
from pathlib import Path

import torch
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import database
from core.tagger.lr_matrix_builder import _collect_samples
from core.tagger.tag_parsing import resolve_caption_tags
from core.tagger.tag_vocabulary import TagVocabulary
from core.tagger.tagger_dataset import TaggerDataset
from database.models import DatasetBase, DatasetCaption, DatasetItem


class _Processor:
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros(1, 3, 8, 8)}


def _dataset_session(tmp_path):
    engine = create_engine("sqlite://")
    DatasetBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    writer = factory()
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (8, 8)).save(image_path)
    valid = DatasetItem(dataset_id=7, base_name="valid", image_path=str(image_path))
    missing = DatasetItem(
        dataset_id=7, base_name="missing", image_path=str(tmp_path / "missing.png")
    )
    writer.add_all([valid, missing])
    writer.flush()
    writer.add_all(
        [
            DatasetCaption(
                item_id=valid.id,
                caption_type="main",
                content="ignored_content",
                tag_data=json.dumps([{"tag": "Structured_Tag"}]),
                is_tags_format=True,
            ),
            DatasetCaption(
                item_id=valid.id,
                caption_type="main",
                content="fallback_tag, second_tag",
                tag_data="not-json",
                is_tags_format=True,
            ),
            DatasetCaption(
                item_id=valid.id,
                caption_type="excluded",
                content="excluded_tag",
                is_tags_format=True,
            ),
        ]
    )
    writer.commit()
    writer.close()
    return factory


def test_caption_resolution_prefers_structured_data_and_normalizes_fallback():
    assert resolve_caption_tags(
        json.dumps([{"tag": "Structured_Tag"}]), "ignored_tag"
    ) == ["structured tag"]
    assert resolve_caption_tags("invalid", "Fallback_Tag, second_tag") == [
        "fallback tag",
        "second tag",
    ]


def test_dataset_scan_preserves_filtered_samples(tmp_path):
    factory = _dataset_session(tmp_path)
    session = factory()
    vocabulary = TagVocabulary()
    vocabulary.add_tags(["structured_tag", "fallback_tag", "second_tag"])

    dataset = TaggerDataset(
        [7], vocabulary, session, _Processor(), caption_types=["main"]
    )

    assert dataset._samples == [
        (
            str(tmp_path / "valid.png"),
            ["structured tag", "fallback tag", "second tag"],
        )
    ]
    session.close()


def test_lr_scan_uses_the_same_filtered_caption_rules(tmp_path, monkeypatch):
    factory = _dataset_session(tmp_path)
    monkeypatch.setattr(database, "DatasetsSessionLocal", factory)

    assert _collect_samples([7], ["main"]) == [
        ["fallback tag", "second tag", "structured tag"]
    ]
