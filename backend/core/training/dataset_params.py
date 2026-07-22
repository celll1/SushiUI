"""
Single source of truth for dataset-level training parameters.

Adding a new dataset-level param requires ONLY:
  1. Add field to DatasetConfigItem (Pydantic) in routes.py
  2. Add field name + default to DATASET_LEVEL_PARAMS below
  3. Handle consumption in train_runner.py
"""

from typing import Any, Dict, List, Optional


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


def resolve_dataset_configs_from_yaml(config_yaml: str, datasets_db) -> Optional[List[Dict[str, Any]]]:
    """Derive the ``TrainingRun.dataset_configs`` COLUMN shape from ``config_yaml``.

    ``config_yaml`` is the single source of truth for which datasets a run
    trains on (this is what train_runner.py actually loads). This helper
    re-derives the column's list-of-dicts shape from the YAML so callers can
    keep the (redundant, denormalized) ``dataset_configs`` column in sync
    with it, using the SAME dataset_id-first resolution semantics as
    train_runner.py (core/training/train_runner.py, "Get dataset configs
    from YAML" block): prefer an explicit ``dataset_id`` on the YAML entry,
    falling back to a ``Dataset.path == folder_path`` lookup only when
    ``dataset_id`` is absent.

    Each resolved entry has the exact column shape used everywhere else in
    the codebase (CREATE at routes.py, the pre-existing migration
    add_dataset_configs.py, and get_training_run_params)::

        {"dataset_id": int, "caption_types": [...], "ve_reconstruction_mode": bool, "filters": {}}

    ``filters`` is intentionally always ``{}`` -- the frontend never sends a
    non-empty value and there are no backend consumers of it (see
    routes.py's "TODO: Apply filters here when filter logic is
    implemented"), so there is nothing to round-trip.

    Returns:
        A list of resolved dataset config dicts, or ``None`` if the YAML has
        no ``datasets`` section, is unparsable, or every entry fails to
        resolve to a dataset_id. Callers MUST treat ``None`` as "keep the
        current column value" -- never overwrite with an empty list, since
        that would erase a run's dataset assignment.
    """
    if not config_yaml:
        return None

    try:
        import yaml
        config = yaml.safe_load(config_yaml)
    except Exception:
        return None

    if not isinstance(config, dict):
        return None

    # Mirror train_runner.py's location of the datasets list:
    # config['config']['process'][0]['datasets']
    try:
        process_list = config.get("config", {}).get("process", [])
        process_config = process_list[0] if process_list else {}
    except Exception:
        process_config = {}

    yaml_datasets = process_config.get("datasets", []) if isinstance(process_config, dict) else []
    if not yaml_datasets:
        return None

    from database.models import Dataset

    resolved: List[Dict[str, Any]] = []
    for yaml_ds in yaml_datasets:
        if not isinstance(yaml_ds, dict):
            continue

        dataset_id = yaml_ds.get("dataset_id")

        # dataset_id-first: only fall back to folder_path lookup when the
        # YAML entry has no explicit dataset_id (matches train_runner.py).
        if not dataset_id:
            folder_path = yaml_ds.get("folder_path") or yaml_ds.get("path")
            if folder_path:
                try:
                    dataset = datasets_db.query(Dataset).filter(Dataset.path == folder_path).first()
                except Exception:
                    dataset = None
                if dataset:
                    dataset_id = dataset.id

        if not dataset_id:
            continue

        resolved.append({
            "dataset_id": int(dataset_id),
            "filters": {},
            **read_dataset_params(yaml_ds),
        })

    return resolved if resolved else None
