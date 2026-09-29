"""CPU contracts for the Phase-3 throughput benchmark."""

import pytest

from core.tagger.foveatag.supervised_benchmark import _validate_arguments


def _values():
    return {
        "image_paths": ["a.png"],
        "num_tags": 10,
        "modes": ("overview_only", "overview_placebo", "source_native"),
        "batch_size": 1,
        "workers": 0,
        "local_views": 4,
        "warmup_steps": 0,
        "measured_steps": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "max_source_pixels": 40_000_000,
        "projected_epoch_samples": 100,
    }


def test_supervised_benchmark_accepts_compute_matched_registered_modes():
    _validate_arguments(**_values())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("modes", ("source_native", "source_native"), "unique registered"),
        ("batch_size", 2, "full batch"),
        ("local_views", 0, "local_views"),
        ("measured_steps", 0, "measured_steps"),
        ("max_source_pixels", 0, "max_source_pixels"),
        ("projected_epoch_samples", 0, "projected_epoch_samples"),
    ],
)
def test_supervised_benchmark_rejects_invalid_contract(field, value, message):
    values = _values()
    values[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_arguments(**values)
