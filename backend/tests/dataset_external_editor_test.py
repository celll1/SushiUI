import os

import pytest

from core.datasets.external_editor import build_editor_argv


def test_editor_argv_replaces_dataset_token_without_a_shell(tmp_path):
    executable = tmp_path / "editor.exe"
    executable.write_bytes(b"")
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    argv = build_editor_argv(
        str(executable), ["--root", "{dataset}", "literal value"], str(dataset)
    )

    assert argv == [
        os.path.abspath(executable),
        "--root",
        str(dataset),
        "literal value",
    ]


def test_editor_argv_appends_dataset_when_token_is_absent(tmp_path):
    executable = tmp_path / "editor.exe"
    executable.write_bytes(b"")
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    assert build_editor_argv(str(executable), [], str(dataset))[-1] == str(dataset)


def test_editor_argv_rejects_missing_executable(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    with pytest.raises(ValueError, match="executable"):
        build_editor_argv(str(tmp_path / "missing.exe"), [], str(dataset))
