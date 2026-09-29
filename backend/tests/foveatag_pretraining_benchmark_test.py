"""CPU contracts for the FoveaTag real-data benchmark."""

import sqlite3
import statistics

import pytest

from core.tagger.foveatag.pretraining_benchmark import _summary, registered_image_paths


def test_registered_paths_uses_read_only_database(tmp_path):
    database = tmp_path / "datasets.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE dataset_items (id INTEGER, dataset_id INTEGER, image_path TEXT, width INTEGER, height INTEGER)"
        )
        connection.executemany(
            "INSERT INTO dataset_items VALUES (?, ?, ?, ?, ?)",
            [
                (2, 7, "b.png", 512, 512),
                (1, 7, "a.png", 128, 512),
                (3, 8, "c.png", 1024, 1024),
                (4, 7, "huge.png", 10000, 10000),
            ],
        )

    paths, total = registered_image_paths(database, 7, limit=4, minimum_side=256)
    assert paths == ["b.png"]
    assert total == 1


def test_registered_paths_hash_selection_is_deterministic(tmp_path):
    database = tmp_path / "datasets.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE dataset_items (id INTEGER, dataset_id INTEGER, image_path TEXT, width INTEGER, height INTEGER)"
        )
        connection.executemany(
            "INSERT INTO dataset_items VALUES (?, 7, ?, 512, 512)",
            [(index, f"{index}.png") for index in range(1, 21)],
        )

    first, total = registered_image_paths(
        database, 7, limit=5, minimum_side=256, selection="hashed", seed=11
    )
    second, _ = registered_image_paths(
        database, 7, limit=5, minimum_side=256, selection="hashed", seed=11
    )
    assert first == second
    assert total == 20
    assert first != ["1.png", "2.png", "3.png", "4.png", "5.png"]


def test_timing_summary_rejects_empty_input_naturally():
    assert _summary([3.0, 1.0, 2.0])["p95"] == 3.0
    with pytest.raises(statistics.StatisticsError):
        _summary([])
