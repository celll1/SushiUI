import json

import pytest

from core.datasets.sidecars import (
    SidecarFormatError,
    read_text_sidecar,
    write_indexed_caption,
    write_text_tags,
)


def _image(tmp_path):
    path = tmp_path / "sample.webp"
    path.write_bytes(b"image-placeholder")
    return path


def test_text_writer_replaces_the_existing_sidecar_without_creating_json(tmp_path):
    image = _image(tmp_path)
    txt = image.with_suffix(".txt")
    txt.write_text("old", encoding="utf-8")

    result = write_indexed_caption(str(image), "a, b", source_field="tags")

    assert result.path == str(txt)
    assert result.format == "txt"
    assert txt.read_text(encoding="utf-8") == "a, b"
    assert not image.with_suffix(".json").exists()


def test_json_writer_preserves_unrelated_fields_and_the_existing_tags_key(tmp_path):
    image = _image(tmp_path)
    sidecar = image.with_suffix(".json")
    sidecar.write_text(
        json.dumps({"tags": "old", "caption": "prose", "score": 4}),
        encoding="utf-8",
    )

    result = write_indexed_caption(str(image), "new", source_field="tags")
    saved = json.loads(sidecar.read_text(encoding="utf-8"))

    assert result.field == "tags"
    assert saved == {"tags": "new", "caption": "prose", "score": 4}


def test_json_writer_retains_the_legacy_caption_field(tmp_path):
    image = _image(tmp_path)
    sidecar = image.with_suffix(".json")
    sidecar.write_text(json.dumps({"caption": "old", "id": 7}), encoding="utf-8")

    result = write_indexed_caption(str(image), "new")

    assert result.field == "caption"
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {
        "caption": "new",
        "id": 7,
    }


def test_json_writer_updates_a_nested_source_field(tmp_path):
    image = _image(tmp_path)
    sidecar = image.with_suffix(".json")
    sidecar.write_text(
        json.dumps({"training": {"tags": "old"}, "keep": True}),
        encoding="utf-8",
    )

    result = write_indexed_caption(
        str(image), "new", source_field="training.tags"
    )

    assert result.field == "training.tags"
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {
        "training": {"tags": "new"},
        "keep": True,
    }


def test_invalid_json_is_refused_without_replacing_the_file(tmp_path):
    image = _image(tmp_path)
    sidecar = image.with_suffix(".json")
    sidecar.write_text("{broken", encoding="utf-8")

    with pytest.raises(SidecarFormatError):
        write_indexed_caption(str(image), "new")

    assert sidecar.read_text(encoding="utf-8") == "{broken"


def test_direct_folder_helpers_keep_the_tagger_browser_contract(tmp_path):
    image = _image(tmp_path)

    result = write_text_tags(str(image), ["a", "b"])

    assert result.path == str(image.with_suffix(".txt"))
    assert read_text_sidecar(str(image)) == (["a", "b"], "a, b")


def test_writer_only_replaces_the_media_extension(tmp_path):
    image = tmp_path / "sample.variant.webp"
    image.write_bytes(b"image-placeholder")

    result = write_indexed_caption(str(image), "tag")

    assert result.path == str(tmp_path / "sample.variant.txt")
    assert not (tmp_path / "sample.txt").exists()
