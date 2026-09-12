import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 9)
torch.cuda._lazy_init = lambda *args, **kwargs: None
torch._C._cuda_init = lambda *args, **kwargs: None

from core.training.train_runner import get_dataset_items_fast
from database.models import Dataset, DatasetBase, DatasetCaption, DatasetItem


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'datasets.db'}")
    DatasetBase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_snapshot_stream_preserves_caption_priority_and_media_metadata(tmp_path):
    db = _session(tmp_path)
    dataset = Dataset(name="stream", path=str(tmp_path))
    db.add(dataset)
    db.flush()

    image_path = tmp_path / "image.png"
    audio_path = tmp_path / "audio.wav"
    image_path.write_bytes(b"image")
    audio_path.write_bytes(b"audio")
    image = DatasetItem(
        dataset_id=dataset.id,
        base_name="image",
        image_path=str(image_path),
        width=640,
        height=480,
    )
    audio = DatasetItem(
        dataset_id=dataset.id,
        base_name="audio",
        image_path=str(audio_path),
        item_type="audio",
        exif_data={"sample_rate": 48000, "duration": 2.5, "channels": 2},
    )
    missing = DatasetItem(
        dataset_id=dataset.id,
        base_name="missing",
        image_path=str(tmp_path / "missing.png"),
    )
    db.add_all([image, audio, missing])
    db.flush()
    db.add_all([
        DatasetCaption(item_id=image.id, caption_type="natural_language", content="sentence"),
        DatasetCaption(
            item_id=image.id,
            caption_type="tags",
            content="tag_a, tag_b",
            tag_data='[{"tag":"tag_a"}]',
            is_tags_format=True,
        ),
        DatasetCaption(item_id=audio.id, caption_type="natural_language", content="music"),
        DatasetCaption(item_id=audio.id, caption_type="lyrics", content="la la"),
    ])
    db.commit()

    items = get_dataset_items_fast(
        db,
        dataset.id,
        caption_types=["tags", "natural_language"],
        auxiliary_caption_types=["natural_language", "lyrics"],
    )

    assert [item["image_path"] for item in items] == [str(image_path), str(audio_path)]
    assert items[0]["raw_caption"] == "tag_a, tag_b"
    assert items[0]["tag_data"] == '[{"tag":"tag_a"}]'
    assert items[0]["_captions_by_type"]["natural_language"]["content"] == "sentence"
    assert items[1]["raw_caption"] == "music"
    assert items[1]["lyrics"] == "la la"
    assert items[1]["audio_path"] == str(audio_path)
    assert items[1]["sample_rate"] == 48000
    assert items[1]["duration"] == 2.5
    assert items[1]["channels"] == 2


def test_pixels_only_snapshot_does_not_load_caption_outputs(tmp_path):
    db = _session(tmp_path)
    dataset = Dataset(name="pixels", path=str(tmp_path))
    db.add(dataset)
    db.flush()
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    audio = DatasetItem(
        dataset_id=dataset.id,
        base_name="audio",
        image_path=str(audio_path),
        item_type="audio",
        exif_data={"duration": 1.0},
    )
    db.add(audio)
    db.flush()
    db.add(DatasetCaption(item_id=audio.id, caption_type="lyrics", content="unused"))
    db.commit()

    items = get_dataset_items_fast(db, dataset.id, skip_captions=True)

    assert len(items) == 1
    assert items[0]["raw_caption"] == ""
    assert items[0]["tag_data"] is None
    assert "lyrics" not in items[0]
    assert "_captions_by_type" not in items[0]
