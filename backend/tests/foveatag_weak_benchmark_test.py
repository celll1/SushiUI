"""CPU tests for deterministic box-free benchmark construction."""

import sqlite3

from PIL import Image

from core.tagger.foveatag.benchmark import BenchmarkManifest
from core.tagger.foveatag.weak_benchmark import build_weak_benchmark, load_weak_records


def test_weak_benchmark_loads_tags_and_selects_deterministically(tmp_path):
    database = tmp_path / "datasets.db"
    paths = []
    for index in range(4):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (32, 24)).save(path)
        paths.append(str(path))
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE dataset_items (id INTEGER, dataset_id INTEGER, image_path TEXT, width INTEGER, height INTEGER)"
        )
        connection.execute(
            "CREATE TABLE dataset_captions (id INTEGER, item_id INTEGER, tag_data TEXT, content TEXT, is_tags_format INTEGER)"
        )
        connection.executemany(
            "INSERT INTO dataset_items VALUES (?, 7, ?, 32, 24)",
            [(index + 1, path) for index, path in enumerate(paths)],
        )
        connection.executemany(
            "INSERT INTO dataset_captions VALUES (?, ?, NULL, ?, 1)",
            [
                (1, 1, "blue_eyes, 1girl"),
                (2, 2, "red eyes, solo"),
                (3, 3, "blue eyes, standing"),
                (4, 4, "1boy, outdoors"),
            ],
        )

    records = load_weak_records(database, 7)
    hashed = load_weak_records(
        database, 7, limit=2, selection="hashed", seed=9
    )
    assert len(hashed) == 2
    assert hashed == load_weak_records(
        database, 7, limit=2, selection="hashed", seed=9
    )
    first, summary = build_weak_benchmark(
        records, dataset_id=7, name="weak", positive_per_tag=1, negative_count=1,
        seed=9, label_provenance="fixture_ground",
    )
    second, _ = build_weak_benchmark(
        records, dataset_id=7, name="weak", positive_per_tag=1, negative_count=1,
        seed=9, label_provenance="fixture_ground",
    )
    manifest = BenchmarkManifest.from_dict(first)
    assert first == second
    assert manifest.evidence_mode == "dense_oracle"
    assert manifest.provenance["label_provenance"] == "fixture_ground"
    assert summary["fine_tag_coverage"] == {"blue eyes": 1, "red eyes": 1}
    assert "colored contact lenses" in summary["uncovered_fine_tags"]
    assert first["provenance"]["uncovered_fine_tags"] == summary["uncovered_fine_tags"]
    assert summary["negative_items"] == 1
