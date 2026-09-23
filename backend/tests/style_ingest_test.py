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


def test_a_refused_dataset_reports_only_its_own_skipped_items(tmp_path):
    factory = session_factory(tmp_path / "datasets.db")
    good, empty = add_dataset(factory, "good", tmp_path / "good"), add_dataset(factory, "empty", tmp_path / "empty")
    add_item(factory, good, tmp_path / "good" / "x.png", [("tags", None, "artist_one", True)])
    add_item(factory, good, tmp_path / "good" / "good_gone.png", [("tags", None, "artist_one", True)], write=False)
    add_item(factory, empty, tmp_path / "empty" / "empty_gone.png", [("tags", None, "artist_one", True)],
             write=False)
    with pytest.raises(ValueError, match=f"dataset {empty} yielded no usable item") as refusal:
        _ingest(factory, [{"dataset_id": good}, {"dataset_id": empty}])
    assert "empty_gone.png" in str(refusal.value) and "good_gone.png" not in str(refusal.value)


def _nested(tmp_path):
    """Dataset `outer` (lower id) has a root containing dataset `inner`'s root; one image is in both."""
    factory = session_factory(tmp_path / "datasets.db")
    outer_root = tmp_path / "library"
    inner_root = outer_root / "artist_pack"
    outer = add_dataset(factory, "outer", outer_root)
    inner = add_dataset(factory, "inner", inner_root)
    shared = inner_root / "a.png"
    kept = add_item(factory, outer, shared, [("tags", None, "artist_one, cp_show", True)])
    dropped = add_item(factory, inner, shared, [("tags", None, "artist_one", True)], write=False)
    only_inner = add_item(factory, inner, inner_root / "b.png", [("tags", None, "artist_one", True)])
    return factory, outer, inner, kept, dropped, only_inner, shared


def test_an_image_under_nested_dataset_roots_is_kept_once_from_the_lowest_dataset(tmp_path):
    factory, outer, inner, kept, dropped, only_inner, shared = _nested(tmp_path)
    ingest, records = _ingest(factory, [{"dataset_id": inner}, {"dataset_id": outer}])  # config order is irrelevant
    assert set(records) == {kept, only_inner}
    assert records[kept][0] == stable_item_key(outer, "artist_pack/a.png")
    assert ingest.stats.items_duplicate_path == 1 and ingest.stats.records == 2
    (example,) = ingest.duplicate_examples
    assert f"item {dropped} of dataset {inner}" in example and f"is item {kept} of dataset {outer}" in example
    index = build_style_index(list(records_iter(records)), SPLIT, np.zeros(0, dtype=np.int64))
    assert sorted(index.item_ids.tolist()) == sorted([kept, only_inner])


def records_iter(records):
    for item_id, (key, labels) in records.items():
        yield item_id, key, labels


def test_a_dataset_holding_only_duplicates_is_not_refused(tmp_path):
    factory, outer, inner, kept, dropped, only_inner, shared = _nested(tmp_path)
    with factory() as session:
        from database.models import DatasetItem
        session.query(DatasetItem).filter(DatasetItem.id == only_inner).delete()
        session.commit()
    ingest, records = _ingest(factory, [{"dataset_id": outer}, {"dataset_id": inner}])
    assert set(records) == {kept} and ingest.stats.items_duplicate_path == 1


@pytest.mark.skipif(sys.platform != "win32", reason="case and separator folding are Windows path semantics")
def test_duplicate_detection_folds_case_and_separators_on_windows(tmp_path):
    factory, outer, inner, kept, dropped, only_inner, shared = _nested(tmp_path)
    with factory() as session:
        from database.models import DatasetItem
        item = session.get(DatasetItem, dropped)
        item.image_path = str(shared).upper().replace("\\", "/")
        session.commit()
    ingest, records = _ingest(factory, [{"dataset_id": outer}, {"dataset_id": inner}])
    assert dropped not in records and ingest.stats.items_duplicate_path == 1


def test_a_path_hash_collision_keeps_both_items(tmp_path, monkeypatch):
    import core.style_encoder.style_ingest as ingest_module
    monkeypatch.setattr(ingest_module, "_path_hash", lambda normalized: 7)
    factory, outer, inner, kept, dropped, only_inner, shared = _nested(tmp_path)
    ingest, records = _ingest(factory, [{"dataset_id": outer}, {"dataset_id": inner}])
    assert set(records) == {kept, only_inner} and ingest.stats.items_duplicate_path == 1


def test_the_pinned_copy_is_kept_and_among_pinned_copies_the_lowest_dataset(tmp_path):
    factory, outer, inner, kept, dropped, only_inner, shared = _nested(tmp_path)
    outer_key, inner_key = stable_item_key(outer, "artist_pack/a.png"), stable_item_key(inner, "a.png")
    configs = parse_dataset_configs([{"dataset_id": outer}, {"dataset_id": inner}])
    for pins, winner in (([inner_key], dropped), ([outer_key, inner_key], kept), ([], kept)):
        with factory() as session:
            ingest = StyleIngest(session, configs, label_resolver(), pinned_keys=np.array(pins, dtype=np.int64))
            records = {item_id: key for item_id, key, _ in ingest.records()}
        assert set(records) == {winner, only_inner} and ingest.stats.items_duplicate_path == 1, pins


def test_a_pinned_copy_without_a_selected_caption_does_not_own_the_path(tmp_path):
    factory, outer, inner, kept, dropped, only_inner, shared = _nested(tmp_path)
    inner_key = stable_item_key(inner, "a.png")
    configs = parse_dataset_configs([{"dataset_id": outer, "caption_types": ["tags"]},
                                     {"dataset_id": inner, "caption_types": ["booru"]}])
    add_item(factory, inner, tmp_path / "library" / "artist_pack" / "c.png", [("booru", None, "artist_one", True)])
    with factory() as session:
        ingest = StyleIngest(session, configs, label_resolver(), pinned_keys=np.array([inner_key], dtype=np.int64))
        records = {item_id for item_id, _, _ in ingest.records()}
    assert kept in records and dropped not in records  # the outer copy is not dropped for an unusable owner
