"""CPU contracts for scalable FoveaTag Phase-2 path manifests."""

import json
import pickle
import sqlite3

from PIL import Image

from core.tagger.foveatag.pretraining_manifest import (
    DomainBalancedPretrainingDataset,
    IndexedPathManifest,
    ManifestPretrainingDataset,
    build_pretraining_manifests,
    open_domain_manifests,
)
from core.tagger.foveatag.tiled_pyramid import build_cached_tiled_pyramid
from core.tagger.foveatag.views import ViewConfig


def _database(tmp_path):
    database = tmp_path / "datasets.db"
    paths = []
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE dataset_items "
            "(id INTEGER, dataset_id INTEGER, image_path TEXT, width INTEGER, height INTEGER)"
        )
        rows = []
        for domain in (3, 7):
            for index in range(8):
                path = tmp_path / f"{domain}-{index}.png"
                Image.new("RGB", (64, 64), (domain, index, 20)).save(path)
                paths.append(str(path))
                rows.append((domain * 100 + index, domain, str(path), 64, 64))
        rows.append((999, 3, str(tmp_path / "huge.png"), 10000, 10000))
        connection.executemany("INSERT INTO dataset_items VALUES (?, ?, ?, ?, ?)", rows)
    return database, paths


def test_manifest_build_is_exact_indexed_and_reusable(tmp_path):
    database, paths = _database(tmp_path)
    output = tmp_path / "manifests"
    metadata = build_pretraining_manifests(
        database,
        output,
        dataset_ids=(3, 7),
        validation_counts={3: 2, 7: 2},
        excluded_paths=(paths[0],),
        minimum_side=32,
        max_source_pixels=100_000,
        seed=11,
    )
    repeated = build_pretraining_manifests(
        database,
        output,
        dataset_ids=(7, 3),
        validation_counts={3: 2, 7: 2},
        excluded_paths=(paths[0],),
        minimum_side=32,
        max_source_pixels=100_000,
        seed=11,
    )

    assert repeated == metadata
    assert metadata["domains"]["3"]["excluded"] == 1
    assert metadata["domains"]["3"]["source_size_filter"] == {
        "minimum_side_eligible": 9,
        "max_source_pixels": 100_000,
        "over_pixel_cap": 1,
        "eligible_after_pixel_cap": 8,
        "cached_over_pixel_cap": 0,
        "eligible_after_pixel_policy": 8,
    }
    assert metadata["domains"]["7"]["source_size_filter"]["over_pixel_cap"] == 0
    assert metadata["domains"]["3"]["train"]["count"] == 5
    assert metadata["domains"]["7"]["train"]["count"] == 6
    manifests = open_domain_manifests(output, metadata, "validation")
    assert all(len(manifest) == 2 for manifest in manifests.values())
    assert len(set(manifests[3][index] for index in range(2))) == 2
    assert "pyramid_cache" not in metadata["request"]
    restored = pickle.loads(pickle.dumps(manifests[3]))
    assert restored[0] == manifests[3][0]
    assert json.loads((output / "manifest_metadata.json").read_text())["request"]["seed"] == 11

    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE dataset_items SET width = width WHERE id = 200")
    resumed = build_pretraining_manifests(
        database,
        output,
        dataset_ids=(3, 7),
        validation_counts={3: 2, 7: 2},
        excluded_paths=(paths[0],),
        minimum_side=32,
        max_source_pixels=100_000,
        seed=11,
        reuse_existing=True,
    )
    assert resumed == metadata


def test_manifest_admits_only_validated_cached_over_cap_paths(tmp_path):
    database, _paths = _database(tmp_path)
    huge = tmp_path / "huge.png"
    Image.new("RGB", (400, 400), (1, 2, 3)).save(huge)
    cache_root = tmp_path / "cache"
    build_cached_tiled_pyramid(
        huge,
        cache_root,
        tile_size=64,
        coarsest_max_side=64,
    )
    unrelated = tmp_path / "unrelated.png"
    Image.new("RGB", (400, 400), (4, 5, 6)).save(unrelated)
    build_cached_tiled_pyramid(
        unrelated,
        cache_root,
        tile_size=64,
        coarsest_max_side=64,
    )
    Image.new("RGB", (401, 400), (7, 8, 9)).save(unrelated)
    output = tmp_path / "cached_manifests"
    metadata = build_pretraining_manifests(
        database,
        output,
        dataset_ids=(3,),
        validation_counts={3: 2},
        minimum_side=32,
        max_source_pixels=100_000,
        pyramid_cache_root=cache_root,
        seed=13,
    )

    size_filter = metadata["domains"]["3"]["source_size_filter"]
    assert size_filter["over_pixel_cap"] == 1
    assert size_filter["cached_over_pixel_cap"] == 1
    assert size_filter["eligible_after_pixel_policy"] == 9
    assert metadata["request"]["pyramid_cache"]["count"] == 1
    manifests = {
        split: open_domain_manifests(output, metadata, split)[3]
        for split in ("train", "validation")
    }
    located = [
        (manifest, index)
        for manifest in manifests.values()
        for index in range(len(manifest))
        if manifest[index] == str(huge)
    ]
    assert len(located) == 1
    manifest, index = located[0]
    dataset = ManifestPretrainingDataset(
        manifest,
        view_config=ViewConfig(overview_size=64, local_size=64),
        seed=13,
        max_source_pixels=100_000,
        pyramid_cache_root=cache_root,
    )
    assert dataset[index] is not None


def test_balanced_dataset_is_deterministic_and_category_agnostic(tmp_path):
    database, _ = _database(tmp_path)
    output = tmp_path / "manifests"
    metadata = build_pretraining_manifests(
        database,
        output,
        dataset_ids=(3, 7),
        validation_counts={3: 1, 7: 1},
        minimum_side=32,
        seed=5,
    )
    manifests = open_domain_manifests(output, metadata, "train")
    dataset = DomainBalancedPretrainingDataset(
        manifests,
        {3: 0.8, 7: 0.2},
        epoch_samples=10_000,
        view_config=ViewConfig(overview_size=64, local_size=64),
        seed=19,
        max_source_pixels=40_000_000,
    )

    selections = [dataset.selection(index) for index in range(len(dataset))]
    repeated = [dataset.selection(index) for index in range(len(dataset))]
    domain_three = sum(domain == 3 for domain, _, _ in selections)
    assert selections == repeated
    assert 7_800 <= domain_three <= 8_200
    dataset.set_epoch(1)
    assert dataset.selection(0) != selections[0]
    assert dataset[0] is not None


def test_validation_manifest_dataset_loads_fixed_pairs(tmp_path):
    database, _ = _database(tmp_path)
    output = tmp_path / "manifests"
    metadata = build_pretraining_manifests(
        database,
        output,
        dataset_ids=(3,),
        validation_counts={3: 2},
        minimum_side=32,
        seed=9,
    )
    manifest = open_domain_manifests(output, metadata, "validation")[3]
    dataset = ManifestPretrainingDataset(
        manifest,
        view_config=ViewConfig(overview_size=64, local_size=64),
        seed=9,
        max_source_pixels=40_000_000,
    )

    first = dataset[0]
    repeated = dataset[0]
    assert first is not None and repeated is not None
    assert first.first_geometry.equal(repeated.first_geometry)


def test_pre_pyramid_dataset_objects_default_missing_cache_field_to_none(tmp_path):
    database, _ = _database(tmp_path)
    output = tmp_path / "manifests"
    metadata = build_pretraining_manifests(
        database,
        output,
        dataset_ids=(3,),
        validation_counts={3: 1},
        minimum_side=32,
        seed=17,
    )
    manifests = open_domain_manifests(output, metadata, "train")
    indexed = ManifestPretrainingDataset(
        manifests[3],
        view_config=ViewConfig(overview_size=64, local_size=64),
        seed=17,
        max_source_pixels=40_000_000,
    )
    balanced = DomainBalancedPretrainingDataset(
        manifests,
        {3: 1.0},
        epoch_samples=1,
        view_config=ViewConfig(overview_size=64, local_size=64),
        seed=17,
        max_source_pixels=40_000_000,
    )
    del indexed.pyramid_cache_root
    del balanced.pyramid_cache_root

    assert indexed[0] is not None
    assert balanced[0] is not None
