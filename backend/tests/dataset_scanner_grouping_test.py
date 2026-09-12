from utils.dataset_scanner import relative_group_key, scan_directory_structure


def test_recursive_scan_keeps_equal_stems_in_separate_directories(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "sample.png").write_bytes(b"image")
    (second / "sample.png").write_bytes(b"image")
    (first / "sample_instruction.txt").write_text("first caption", encoding="utf-8")
    (second / "sample_instruction.txt").write_text("second caption", encoding="utf-8")

    groups = scan_directory_structure(str(tmp_path), recursive=True)

    assert set(groups) == {"first/sample", "second/sample"}
    assert groups["first/sample"]["captions"] == [{
        "path": str(first / "sample_instruction.txt"),
        "suffix": "instruction",
        "ext": ".txt",
    }]
    assert groups["second/sample"]["captions"] == [{
        "path": str(second / "sample_instruction.txt"),
        "suffix": "instruction",
        "ext": ".txt",
    }]


def test_root_group_key_remains_backward_compatible(tmp_path):
    assert relative_group_key(str(tmp_path), str(tmp_path), "sample") == "sample"
