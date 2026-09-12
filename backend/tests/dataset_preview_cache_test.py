import os

from PIL import Image

from core.datasets.previews import get_or_create_preview, prune_preview_cache
from database.models import DatasetItem


def _item(path, *, item_id=1, item_type="single"):
    return DatasetItem(
        id=item_id,
        dataset_id=7,
        item_type=item_type,
        base_name=path.stem,
        image_path=str(path),
    )


def test_preview_is_bounded_and_reused_until_the_source_changes(tmp_path):
    source = tmp_path / "image.png"
    Image.new("RGB", (800, 400), "red").save(source)
    cache = tmp_path / "cache"

    first, first_etag = get_or_create_preview(
        _item(source), 256, thumbnails_dir=str(tmp_path), cache_root=str(cache)
    )
    second, second_etag = get_or_create_preview(
        _item(source), 256, thumbnails_dir=str(tmp_path), cache_root=str(cache)
    )

    assert first == second
    assert first_etag == second_etag
    with Image.open(first) as image:
        assert image.size == (256, 128)

    Image.new("RGB", (400, 800), "blue").save(source)
    os.utime(source, None)
    changed, changed_etag = get_or_create_preview(
        _item(source), 256, thumbnails_dir=str(tmp_path), cache_root=str(cache)
    )
    assert changed != first
    assert changed_etag != first_etag


def test_video_preview_prefers_path_key_and_supports_legacy_cache(tmp_path):
    media = tmp_path / "nested" / "clip.mp4"
    media.parent.mkdir()
    media.write_bytes(b"video")
    thumbnails = tmp_path / "thumbnails"
    thumbnails.mkdir()
    Image.new("RGB", (320, 180), "green").save(thumbnails / "clip.png")

    preview, _ = get_or_create_preview(
        _item(media, item_type="video"),
        128,
        thumbnails_dir=str(thumbnails),
        cache_root=str(tmp_path / "cache"),
    )
    with Image.open(preview) as image:
        assert image.size == (128, 72)


def test_cache_pruning_removes_oldest_entries(tmp_path):
    cache = tmp_path / "previews"
    cache.mkdir()
    paths = []
    for index in range(3):
        path = cache / f"{index}.webp"
        path.write_bytes(bytes(index + 1))
        os.utime(path, ns=(index + 1, index + 1))
        paths.append(path)

    prune_preview_cache(cache, max_files=2, max_bytes=100)

    assert not paths[0].exists()
    assert paths[1].exists()
    assert paths[2].exists()


def test_cache_pruning_never_evicts_the_file_being_returned(tmp_path):
    cache = tmp_path / "previews"
    cache.mkdir()
    keep = cache / "keep.webp"
    newer = cache / "newer.webp"
    keep.write_bytes(b"keep")
    newer.write_bytes(b"newer")
    os.utime(keep, ns=(1, 1))
    os.utime(newer, ns=(2, 2))

    prune_preview_cache(cache, max_files=1, max_bytes=100, keep=keep)

    assert keep.exists()
    assert not newer.exists()
