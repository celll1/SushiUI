"""Deterministic SenseNova task-view selection and homogeneous batching."""

from collections import OrderedDict
import hashlib
import json
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple


TEXT_TASKS = frozenset({"i2t_caption", "i2t_tags", "i2t_caption_tags"})
IMAGE_TASKS = frozenset({"t2i", "ti2i"})
TASKS = TEXT_TASKS | IMAGE_TASKS
PROMPT_TEMPLATE_VERSION = 1


def canonicalize_tags(contents: Iterable[str]) -> List[str]:
    """Normalize a comma-separated tag target into a versioned stable order."""
    unique: Dict[str, str] = {}
    for content in contents:
        for raw in str(content).split(","):
            tag = raw.strip()
            if tag:
                unique.setdefault(tag.casefold(), tag)
    return [unique[key] for key in sorted(unique)]


def resolve_text_target(
    task: str,
    captions_by_type: Dict[str, Dict[str, Any]],
    target_caption_types: Sequence[str],
) -> str:
    selected = []
    missing = []
    for caption_type in target_caption_types:
        payload = captions_by_type.get(caption_type)
        if payload and str(payload.get("content", "")).strip():
            selected.append(payload)
        else:
            missing.append(caption_type)
    if missing:
        raise ValueError(f"missing target caption source(s): {', '.join(missing)}")
    if not selected:
        raise ValueError(f"{task} requires at least one non-empty target caption source")

    natural = [str(item["content"]).strip() for item in selected
               if not item.get("is_tags_format", False)]
    tag_sources = [str(item["content"]) for item in selected
                   if item.get("is_tags_format", False)]
    if task == "i2t_caption":
        caption_parts = natural or [str(item["content"]).strip() for item in selected]
        return "\n".join(caption_parts)
    tags = canonicalize_tags(tag_sources or [str(item["content"]) for item in selected])
    if not tags:
        raise ValueError(f"{task} requires a non-empty tag target")
    if task == "i2t_tags":
        return json.dumps({"tags": tags}, ensure_ascii=False, separators=(",", ":"))
    if task != "i2t_caption_tags":
        raise ValueError(f"Task {task!r} is not a text-output task")
    if not natural:
        raise ValueError("i2t_caption_tags requires a natural-language target source")
    return json.dumps(
        {"caption": "\n".join(natural), "tags": tags},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def resolve_hint_tags(
    captions_by_type: Dict[str, Dict[str, Any]],
    hint_caption_types: Sequence[str],
) -> List[str]:
    return canonicalize_tags(
        str(captions_by_type[caption_type].get("content", ""))
        for caption_type in hint_caption_types
        if caption_type in captions_by_type
    )


def resolve_generation_caption(
    captions_by_type: Dict[str, Dict[str, Any]],
    target_caption_types: Sequence[str],
) -> str:
    """Join a generation task view's explicitly selected caption sources."""
    parts = []
    for caption_type in target_caption_types:
        payload = captions_by_type.get(caption_type)
        content = str((payload or {}).get("content", "")).strip()
        if not content:
            raise ValueError(f"missing target caption source: {caption_type}")
        parts.append(content)
    if not parts:
        raise ValueError("SenseNova image-output task requires a caption source")
    return "\n".join(parts)


def keep_hints_for_example(
    *, run_seed: int, epoch: int, image_path: str, task: str,
    template_version: int, dropout: float,
) -> bool:
    if dropout <= 0:
        return True
    if dropout >= 1:
        return False
    material = f"{run_seed}|{epoch}|{image_path}|{task}|{template_version}|hint-dropout"
    seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed).random() >= dropout


def _replace_image_token(question: str, grid_hw, downsample_ratio: float) -> str:
    count = int(grid_hw[0, 0] * grid_hw[0, 1] * downsample_ratio ** 2)
    image_tokens = "<img>" + "<IMG_CONTEXT>" * count + "</img>"
    return question.replace("<image>", image_tokens, 1)


def _tokenize_with_assistant_mask(tokenizer, prefix: str, full: str):
    """Tokenize the active chat format and conservatively mask its prompt."""
    try:
        encoded = tokenizer(full, return_tensors="pt", return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        labels = encoded["input_ids"].clone()
        boundary = len(prefix)
        for index, (start, end) in enumerate(offsets):
            if (start == 0 and end == 0) or start < boundary:
                labels[0, index] = -100
    except (NotImplementedError, TypeError, ValueError, KeyError):
        encoded = tokenizer(full, return_tensors="pt")
        prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"]
        labels = encoded["input_ids"].clone()
        labels[:, :min(labels.shape[1], prefix_ids.shape[1])] = -100
    if not bool((labels != -100).any()):
        raise ValueError("SenseNova target produced no supervised assistant tokens")
    return encoded["input_ids"], encoded["attention_mask"], labels


def build_text_supervision(
    transformer,
    tokenizer,
    grid_hw,
    task_view: Dict[str, Any],
    captions_by_type: Dict[str, Dict[str, Any]],
    *,
    image_path: str,
    epoch: int,
    run_seed: int,
):
    """Build active-template input IDs with assistant-only causal-LM labels."""
    from core.models.sensenova.text_output import build_effective_instruction
    from core.models.sensenova.vendor import get_conv_template

    task = str(task_view["task"])
    if task not in TEXT_TASKS:
        raise ValueError(f"Task {task!r} is not a text-output task")
    version = int(task_view.get("prompt_template_version", PROMPT_TEMPLATE_VERSION))
    if version != PROMPT_TEMPLATE_VERSION:
        raise ValueError(f"unsupported SenseNova training prompt template version: {version}")
    hints = resolve_hint_tags(captions_by_type, task_view.get("hint_caption_types", ()))
    if hints and not keep_hints_for_example(
        run_seed=run_seed,
        epoch=epoch,
        image_path=image_path,
        task=task,
        template_version=version,
        dropout=float(task_view.get("hint_dropout", 0.25)),
    ):
        hints = []
    inference_task = {
        "i2t_caption": "caption",
        "i2t_tags": "tags",
        "i2t_caption_tags": "caption_tags",
    }[task]
    instruction = build_effective_instruction(inference_task, None, hints, version)
    target = resolve_text_target(
        task, captions_by_type, task_view.get("target_caption_types", ())
    )
    question = _replace_image_token(
        "<image>\n" + instruction, grid_hw, transformer.downsample_ratio
    )

    prefix_template = get_conv_template(transformer.template)
    prefix_template.system_message = transformer.system_message
    prefix_template.append_message(prefix_template.roles[0], question)
    prefix_template.append_message(prefix_template.roles[1], None)
    prefix = prefix_template.get_prompt()

    full_template = get_conv_template(transformer.template)
    full_template.system_message = transformer.system_message
    full_template.append_message(full_template.roles[0], question)
    full_template.append_message(full_template.roles[1], target)
    full = full_template.get_prompt()
    input_ids, attention_mask, labels = _tokenize_with_assistant_mask(tokenizer, prefix, full)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "target": target,
        "hint_tags": hints,
        "prompt_template_version": version,
        "target_tokens": int((labels != -100).sum().item()),
    }


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
            captions = item.get("_captions_by_type") or {}
            eligible = [
                view for view in views
                if view.get("target_caption_types")
                and (view.get("task") != "ti2i" or item.get("reference_images"))
                and all(
                    caption_type in captions
                    and str(captions[caption_type].get("content", "")).strip()
                    for caption_type in view["target_caption_types"]
                )
            ]
            if not eligible:
                continue
            view = select_task_view(eligible, rng)
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
