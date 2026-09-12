import os

import pytest

from core.datasets.workspaces import DatasetWorkspaceRegistry


def test_workspaces_isolate_roots_and_keep_legacy_compatibility(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    registry = DatasetWorkspaceRegistry()

    first_id, _ = registry.create(str(first))
    registry.set_legacy(first_id)
    second_id, _ = registry.create(str(second), dataset_id=42)

    assert registry.resolve(first_id, "image.png") == os.path.join(str(first), "image.png")
    assert registry.resolve(second_id, "image.png") == os.path.join(str(second), "image.png")
    assert registry.resolve(None, "image.png") == os.path.join(str(first), "image.png")
    assert registry.dataset_id(first_id) is None
    assert registry.dataset_id(second_id) == 42


@pytest.mark.parametrize("rel_path", ["../secret", "..\\secret", "/../secret"])
def test_workspace_rejects_parent_traversal(tmp_path, rel_path):
    registry = DatasetWorkspaceRegistry()
    workspace_id, _ = registry.create(str(tmp_path))

    with pytest.raises(PermissionError):
        registry.resolve(workspace_id, rel_path)


def test_workspace_eviction_is_bounded(tmp_path):
    registry = DatasetWorkspaceRegistry(max_entries=1)
    first_id, _ = registry.create(str(tmp_path))
    second_id, _ = registry.create(str(tmp_path))

    with pytest.raises(KeyError):
        registry.root(first_id)
    assert registry.root(second_id) == str(tmp_path)
