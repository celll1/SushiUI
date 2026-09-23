"""Style ingestion from datasets.db: caption selection, item keys, duplicate-key report (design §5, §9)."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.style_encoder.style_dataset import SplitParams, build_style_index, stable_item_key
from core.style_encoder.style_ingest import (
    DuplicateItemKeyError, StyleIngest, dataset_relative_path, parse_dataset_configs,
)
from style_training_fixture import add_dataset, add_item, label_resolver, session_factory

SPLIT = SplitParams.create(val_artist_ratio=0.0, val_image_ratio=0.0, min_images_per_artist=1, seed=0)


def _ingest(factory, configs):
    with factory() as session:
        ingest = StyleIngest(session, parse_dataset_configs(configs), label_resolver())
        records = list(ingest.records())
    return ingest, {item_id: (key, labels) for item_id, key, labels in records}


def test_caption_type_selection_item_keys_and_skips(tmp_path):
    factory = session_factory(tmp_path / "datasets.db")
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    a, b = add_dataset(factory, "a", root_a), add_dataset(factory, "b", root_b)
    merged = add_item(factory, a, root_a / "sub" / "x.png", [
        ("tags", [{"tag": "Brand_New", "category": "Artist"}], None, True),
        ("tags", None, "char_one, cp_show", True),
        ("booru", None, "artist_other", True),        # caption type not selected for dataset a
        ("tags", None, "artist_not_tags_format", False),
    ])
    missing = add_item(factory, a, root_a / "gone.png", [("tags", None, "artist_x", True)], write=False)
    outside = add_item(factory, a, tmp_path / "elsewhere.png", [("tags", None, "artist_x", True)])
    no_caption = add_item(factory, a, root_a / "bare.png", [("tags", None, "artist_x", False)])
    every_type = add_item(factory, b, root_b / "y.png", [("booru", None, "artist_other", True),
                                                          ("tags", None, "cp_show", True)])

    ingest, records = _ingest(factory, [{"dataset_id": a, "caption_types": ["tags"]}, {"dataset_id": b}])

    assert set(records) == {merged, every_type}
    key, labels = records[merged]
    assert key == stable_item_key(a, "sub/x.png")
    assert (labels.artists, labels.characters, labels.copyrights) == ({"brand new"}, {"char one"}, {"cp show"})
    assert records[every_type][0] == stable_item_key(b, "y.png")
    assert records[every_type][1].artists == {"artist other"}
    stats = ingest.stats
    assert (stats.items_scanned, stats.items_missing_image, stats.items_outside_root,
            stats.items_without_tag_caption, stats.records) == (5, 1, 1, 1, 2)
    assert ingest.resolver.stats.items == 2
    assert missing not in records and outside not in records and no_caption not in records
    table = ingest.paths.take(np.array([every_type, merged]))
    assert [table.path(0), table.path(1)] == [str(root_b / "y.png"), str(root_a / "sub" / "x.png")]


def test_item_key_uses_the_path_relative_to_the_dataset_root(tmp_path):
    for root in (tmp_path / "old_root", tmp_path / "moved" / "root"):
        assert dataset_relative_path(str(root / "d" / "e.png"), str(root)) == "d/e.png"
    assert dataset_relative_path(str(tmp_path / "r" / ".." / "x.png"), str(tmp_path / "r")) is None
    assert dataset_relative_path(str(tmp_path / "r"), str(tmp_path / "r")) is None


def test_duplicate_item_keys_name_the_colliding_items(tmp_path):
    factory = session_factory(tmp_path / "datasets.db")
    root = tmp_path / "a"
    dataset = add_dataset(factory, "a", root)
    first = add_item(factory, dataset, root / "same.png", [("tags", None, "artist_one", True)])
    second = add_item(factory, dataset, root / "same.png", [("tags", None, "artist_two", True)], write=False)
    add_item(factory, dataset, root / "other.png", [("tags", None, "artist_one", True)])

    with factory() as session:
        ingest = StyleIngest(session, parse_dataset_configs([{"dataset_id": dataset}]), label_resolver())
        with pytest.raises(DuplicateItemKeyError) as caught:
            build_style_index(ingest.records(), SPLIT, np.zeros(0, dtype=np.int64))
    assert caught.value.collisions == [[(dataset, first, str(root / "same.png")),
                                        (dataset, second, str(root / "same.png"))]]
    assert f"item {first}" in str(caught.value) and f"item {second}" in str(caught.value)


def test_dataset_configs_are_validated():
    with pytest.raises(ValueError, match="more than once"):
        parse_dataset_configs([{"dataset_id": 1}, {"dataset_id": 1, "caption_types": ["tags"]}])
    with pytest.raises(ValueError, match="integer dataset_id"):
        parse_dataset_configs([{"dataset_id": True}])


def test_path_table_take_across_gather_chunks(monkeypatch):
    import core.style_encoder.style_ingest as ingest_module

    monkeypatch.setattr(ingest_module, "_GATHER_ROWS", 3)
    builder = ingest_module._ItemPathsBuilder()
    paths = {i: f"{'d' * (i % 4)}/画像_{i}.png" for i in (9, 2, 7, 4, 1, 8, 3)}
    for item_id, path in paths.items():
        builder.append(item_id, -item_id, 1, path)
    table = builder.finish()
    assert table.item_ids.tolist() == sorted(paths)
    order = np.array([8, 1, 9, 3, 2, 7, 4])
    taken = table.take(order)
    assert [taken.path(i) for i in range(order.size)] == [paths[i] for i in order]
    assert taken.item_keys.tolist() == (-order).tolist()
    with pytest.raises(KeyError):
        table.take(np.array([5]))


def test_a_dataset_without_usable_items_is_refused_by_name(tmp_path):
    factory = session_factory(tmp_path / "datasets.db")
    good, empty = add_dataset(factory, "good", tmp_path / "good"), add_dataset(factory, "empty", tmp_path / "empty")
    add_item(factory, good, tmp_path / "good" / "x.png", [("tags", None, "artist_one", True)])
    add_item(factory, empty, tmp_path / "empty" / "gone.png", [("tags", None, "artist_one", True)], write=False)
    with pytest.raises(ValueError, match=rf"dataset {empty} yielded no usable item.*missing image"):
        _ingest(factory, [{"dataset_id": good}, {"dataset_id": empty}])
