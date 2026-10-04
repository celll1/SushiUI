"""Checkpoint-bound online collection pools, quotas and pending proposals."""

import json
import os
from pathlib import Path


def _validate(document, *, step):
    if (not isinstance(document, dict) or document.get("version") != 1
            or document.get("step") != step or type(document.get("epoch")) is not int or document["epoch"] < 1):
        raise ValueError("native collection runtime version/step/epoch differs")
    for key in ("dynamic_tags", "cooc_active_tags", "query_tags"):
        values = document[key]
        if not isinstance(values, dict) or any(not isinstance(name, str) or not isinstance(value, (int, float))
            or not 0 <= value < float("inf") for name, value in values.items()):
            raise ValueError("invalid native collection pool")
    for key in ("expanded_tags", "query_expanded", "low_f1_unavailable"):
        if not isinstance(document[key], list) or any(not isinstance(value, str) for value in document[key]):
            raise ValueError("invalid native collection tag set")
    progress = document["epoch_progress"]
    for key in ("collect_count", "cooc_collected"):
        values = progress[key]
        if not isinstance(values, dict) or any(not isinstance(name, str) or type(value) is not int or value < 0
                                               for name, value in values.items()):
            raise ValueError("invalid native collection quota")
    if not isinstance(progress["exhausted_tags"], list) or any(not isinstance(value, str) for value in progress["exhausted_tags"]):
        raise ValueError("invalid native exhausted set")
    if not 0 <= document["query_last_resolve"] < float("inf"):
        raise ValueError("invalid native collection resolver time")
    expander = document["expander"]
    if expander is not None:
        if not isinstance(expander, dict) or any(not isinstance(expander[key], list)
                or any(not isinstance(value, str) for value in expander[key]) for key in ("pending", "already_proposed")):
            raise ValueError("invalid native collection proposals")
        if not set(expander["pending"]).issubset(expander["already_proposed"]):
            raise ValueError("pending native proposals were not recorded")
    return document


def save_collection_runtime(buffer, path, *, step, epoch):
    document = buffer.snapshot_native_runtime()
    document.update(version=1, step=step, epoch=epoch)
    document["epoch_progress"]["epoch"] = epoch
    _validate(document, step=step)
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def load_collection_runtime(path, *, step, epoch=None):
    try:
        document = _validate(json.loads(Path(path).read_text(encoding="utf-8")), step=step)
        if epoch is not None and (document["epoch"] != epoch or document["epoch_progress"]["epoch"] != epoch):
            raise ValueError("native collection epoch differs")
        return document
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("native collection runtime is missing or invalid") from exc
