"""Deterministic SenseNova task-view selection and homogeneous batching."""

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Sequence, Tuple


TEXT_TASKS = frozenset({"i2t_caption", "i2t_tags", "i2t_caption_tags"})
IMAGE_TASKS = frozenset({"t2i", "ti2i"})
TASKS = TEXT_TASKS | IMAGE_TASKS


def required_caption_types(task_views: Sequence[Dict[str, Any]]) -> List[str]:
    """Return the stable union of target and hint sources a dataset must load."""
    result: List[str] = []
    seen = set()
    for view in task_views:
        for field in ("target_caption_types", "hint_caption_types"):
            for value in view.get(field, ()):
                caption_type = str(value).strip()
                if caption_type and caption_type not in seen:
                    seen.add(caption_type)
                    result.append(caption_type)
    return result


def select_task_view(task_views: Sequence[Dict[str, Any]], rng) -> Dict[str, Any]:
    if not task_views:
        raise ValueError("SenseNova task scheduling requires at least one task view")
    weights = [float(view.get("weight", 1.0)) for view in task_views]
    if any(weight <= 0 for weight in weights):
        raise ValueError("SenseNova task-view weights must be greater than zero")
    selected = rng.choices(list(task_views), weights=weights, k=1)[0]
    return dict(selected)


def build_task_homogeneous_batches(
    batches: Sequence[Sequence[Tuple[Dict[str, Any], Any]]],
    batch_size: int,
    rng,
) -> List[List[Tuple[Dict[str, Any], Any]]]:
    """Select one view per drawn item and regroup compatible items by task.

    The caller snapshots and restores ``rng`` before forming the epoch batches.
    This function consumes only that stream, so task draws and the final order
    reproduce exactly on a mid-epoch resume. Item references are shallow-copied
    before the selected view is attached; persistent bucket records stay clean.
    """
    has_views = any(
        item.get("_sensenova_task_views")
        for batch in batches for item, _dataset in batch
    )
    if not has_views:
        return [list(batch) for batch in batches]

    pools: "OrderedDict[tuple, List[Tuple[Dict[str, Any], Any]]]" = OrderedDict()
    for batch in batches:
        for item, dataset in batch:
            views = item.get("_sensenova_task_views") or []
            if not views:
                raise ValueError(
                    "Every dataset item must define task_views in an explicit "
                    "SenseNova task run"
                )
            view = select_task_view(views, rng)
            task = view.get("task")
            if task not in TASKS:
                raise ValueError(f"Unknown SenseNova task: {task!r}")
            scheduled = dict(item)
            scheduled["_sensenova_task"] = task
            scheduled["_sensenova_task_view"] = view
            key = (
                task,
                scheduled.get("bucket_width", scheduled.get("width")),
                scheduled.get("bucket_height", scheduled.get("height")),
                bool(scheduled.get("reference_images")),
                scheduled.get("item_type", "single"),
            )
            pools.setdefault(key, []).append((scheduled, dataset))

    result: List[List[Tuple[Dict[str, Any], Any]]] = []
    size = max(1, int(batch_size))
    for items in pools.values():
        result.extend(items[index:index + size] for index in range(0, len(items), size))
    rng.shuffle(result)
    return result

