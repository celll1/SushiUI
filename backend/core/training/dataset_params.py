"""
Single source of truth for dataset-level training parameters.

Adding a new dataset-level param requires ONLY:
  1. Add field to DatasetConfigItem (Pydantic) in routes.py
  2. Add field name + default to DATASET_LEVEL_PARAMS below
  3. Handle consumption in train_runner.py
"""

from typing import Any, Dict


# Dataset-level parameters that are propagated through:
#   Frontend -> routes.py -> YAML -> train_runner.py
# Key = field name, Value = default value (omitted from YAML when equal to default)
DATASET_LEVEL_PARAMS: Dict[str, Any] = {
    "caption_types": [],
    "ve_reconstruction_mode": False,
}


def extract_dataset_params(source: dict) -> dict:
    """Extract dataset-level params from a source dict, omitting default values.

    Used when writing to YAML (routes.py -> training_config.py).
    Only includes params whose value differs from the default,
    keeping YAML output clean.
    """
    result = {}
    for key, default in DATASET_LEVEL_PARAMS.items():
        val = source.get(key, default)
        if val != default:
            result[key] = val
    return result


def read_dataset_params(source: dict) -> dict:
    """Read dataset-level params from YAML, filling in defaults for missing keys.

    Used when reading YAML back (get_training_run_params, train_runner.py).
    Always returns all keys so downstream code can rely on their presence.
    """
    return {key: source.get(key, default) for key, default in DATASET_LEVEL_PARAMS.items()}
