from pathlib import Path

from PIL import Image

from config.settings import settings
from database.models import DatasetItem
from utils.image_utils import create_thumbnail, dataset_thumbnail_key


def test_same_basename_in_different_directories_has_distinct_dataset_keys(tmp_path):
    first = tmp_path / "first" / "clip.mp4"
    second = tmp_path / "second" / "clip.mp4"

    assert dataset_thumbnail_key(str(first)) != dataset_thumbnail_key(str(second))


def test_dataset_thumbnail_uses_the_requested_collision_resistant_key(tmp_path, monkeypatch):
    thumbnails = tmp_path / "thumbnails"
    monkeypatch.setattr(settings, "thumbnails_dir", str(thumbnails))
    image = tmp_path / "poster.png"
    Image.new("RGB", (32, 16), "red").save(image)
    key = dataset_thumbnail_key(str(tmp_path / "media" / "clip.mp4"))

    result = create_thumbnail(str(image), output_key=key)

    assert result == str(thumbnails / f"{key}.png")
    assert (thumbnails / f"{key}.webp").is_file()


def test_dataset_item_prefers_hashed_webp_and_falls_back_to_legacy_png(tmp_path, monkeypatch):
    thumbnails = tmp_path / "thumbnails"
    thumbnails.mkdir()
    monkeypatch.setattr(settings, "thumbnails_dir", str(thumbnails))
    media = tmp_path / "nested" / "clip.mp4"
    item = DatasetItem(
        id=1,
        dataset_id=1,
        item_type="video",
        base_name="clip",
        image_path=str(media),
    )

    assert item.to_dict()["thumbnail_url"] == "/thumbnails/clip.png"

    key = dataset_thumbnail_key(str(media))
    Path(thumbnails / f"{key}.webp").write_bytes(b"webp")
    assert item.to_dict()["thumbnail_url"] == f"/thumbnails/{key}.webp"
