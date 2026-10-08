"""Execute producer-window collection without replacing the active training batch."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.training.concept_exposure import ConceptExposure


SOURCE = Path(__file__).parents[1] / "core" / "training" / "base_trainer.py"


def window_collectors():
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    return [node for node in ast.walk(module) if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Subscript) and isinstance(node.iter.value, ast.Name)
            and node.iter.value.id == "batches" and any(
                isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name) and child.func.value.id == "buffer_items"
                and child.func.attr == "extend" for child in ast.walk(node))]


@pytest.mark.parametrize("current_size,last_size", [(4, 1), (1, 4), (4, 4)])
def test_prefill_and_refill_keep_active_items_and_exposure_counts(tmp_path, current_size, last_size):
    dataset = SimpleNamespace(unique_id="synthetic")
    batches = [[({"image_path": f"{group}-{index}.png", "_concept_exposure_group": f"g{group}"}, dataset)
                for index in range(size)] for group, size in enumerate((current_size, 2, last_size))]
    collectors = window_collectors()
    assert len(collectors) == 4
    tracker = ConceptExposure(tmp_path, "concept")
    for loop in collectors:
        scope = dict(batch=batches[0], batches=batches, buffer_items=[], start_idx=1, end_idx=3,
                     text_encoding_swap_interval=2, latent_encoding_swap_interval=2)
        exposure_items = scope["batch"]
        exec(compile(ast.Module(body=[loop], type_ignores=[]), str(SOURCE), "exec"), scope)
        assert scope["batch"] is exposure_items
        start, end = (0, 2) if loop.iter.slice.lower is None else (1, 3)
        assert scope["buffer_items"] == [item for batch in batches[start:end] for item in batch]
        tracker.record(exposure_items, 257, 1, [.5] * len(scope["batch"]))
    restored = ConceptExposure(tmp_path, "concept", tracker.state())
    assert restored.counts == {"g0": 4 * current_size}
    assert not restored.counts.get("g2")
