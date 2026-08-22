"""``sort_names_by_data_offset`` (offline ConvRot export, HDD read-ordering).

No model, no GPU: exercises the pure reordering helper against synthetic
safetensors-header-shaped dicts only.
"""

import os
import sys

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from core.models.common.convrot_export import sort_names_by_data_offset  # noqa: E402


def test_sorts_by_ascending_data_offset_not_input_order():
    header = {
        "c": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [200, 300]},
        "a": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 100]},
        "b": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [100, 200]},
    }
    assert sort_names_by_data_offset(["c", "a", "b"], header) == ["a", "b", "c"]


def test_falls_back_to_input_order_on_missing_data_offsets():
    header = {
        "a": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 100]},
        "b": {"dtype": "BF16", "shape": [4, 4]},  # malformed: no data_offsets
    }
    assert sort_names_by_data_offset(["b", "a"], header) == ["b", "a"]


def test_falls_back_to_input_order_on_name_missing_from_header():
    header = {"a": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 100]}}
    assert sort_names_by_data_offset(["z", "a"], header) == ["z", "a"]


def test_empty_input_is_a_no_op():
    assert sort_names_by_data_offset([], {}) == []
