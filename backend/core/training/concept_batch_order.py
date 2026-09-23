"""Deterministic concept-focused image batch order."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
import random
import re
import unicodedata
from typing import Any

from api.param_defaults import TRAINING_DEFAULTS


PLAN_VERSION = 1


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold().replace("_", " ")
    return " ".join(normalized.split())


def _item_id(pair: tuple[dict, Any]) -> tuple[str, str]:
    item, dataset = pair
    return str(getattr(dataset, "unique_id", item.get("dataset_unique_id", ""))), str(item["image_path"])


def _bucket_key(item: dict) -> tuple[int, int, bool]:
    return (int(item.get("bucket_width") or item.get("width") or 0),
            int(item.get("bucket_height") or item.get("height") or 0),
            bool(item.get("reference_images") or item.get("_ve_reconstruction_mode")))


def _caption_matches(caption: str, aliases: list[str]) -> bool:
    text = _name(caption)
    for alias in aliases:
        phrase = _name(alias)
        if phrase and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text):
            return True
    return False


@dataclass(frozen=True)
class ConceptOrderConfig:
    category: str
    min_items_per_concept: int
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    focus_batches: int
    local_swap_window: int
    background_placement: str
    background_interval: int
    replay_interval: int
    caption_aliases: dict[str, tuple[str, ...]]
    match_natural_language: bool

    @classmethod
    def parse(cls, value: dict | None) -> ConceptOrderConfig | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("concept_batch_order must be an object")
        if value.get("enabled") is False:
            return None
        defaults = TRAINING_DEFAULTS["concept_batch_order"]
        unknown = set(value) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown concept_batch_order fields: {sorted(unknown)}")
        data = {**defaults, **value}
        if data["enabled"] is not True:
            return None
        if data["category"] not in ("character", "artist"):
            raise ValueError("concept_batch_order.category must be character or artist")
        if data["background_placement"] not in ("front", "spread"):
            raise ValueError("concept_batch_order.background_placement must be front or spread")
        for key, minimum in (("min_items_per_concept", 1), ("focus_batches", 0),
                             ("local_swap_window", 0), ("background_interval", 0),
                             ("replay_interval", 0)):
            if type(data[key]) is not int or data[key] < minimum:
                raise ValueError(f"concept_batch_order.{key} must be an integer >= {minimum}")
        if data["local_swap_window"] > 32 or data["focus_batches"] > 10000:
            raise ValueError("concept_batch_order focus/swap setting is too large")
        if not isinstance(data["include"], list) or not isinstance(data["exclude"], list):
            raise ValueError("concept_batch_order include/exclude must be lists")
        if not isinstance(data["caption_aliases"], dict):
            raise ValueError("concept_batch_order.caption_aliases must be an object")
        aliases = {}
        for key, names in data["caption_aliases"].items():
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise ValueError("concept_batch_order caption aliases must be string lists")
            aliases[_name(key)] = tuple(names)
        return cls(data["category"], data["min_items_per_concept"],
                   tuple(_name(x) for x in data["include"]),
                   tuple(_name(x) for x in data["exclude"]),
                   data["focus_batches"], data["local_swap_window"],
                   data["background_placement"], data["background_interval"],
                   data["replay_interval"], aliases, bool(data["match_natural_language"]))

    def fingerprint(self) -> str:
        return _digest(asdict(self))


@dataclass
class ConceptBatchPlan:
    batches: list[list[tuple[dict, Any]]]
    labels: list[str | None]
    replay: list[bool]
    classification_hash: str
    digest: str
    concept_count: int
    focus_items: int
    background_items: int
    partial_batches: int
    replay_batches: int
    missing_tag_data: int
    uncategorized_items: int
    missing_includes: tuple[str, ...]

    def state(self, config: ConceptOrderConfig, seed: int, crop_fingerprint: str | None) -> dict:
        return {"version": PLAN_VERSION, "seed": seed,
                "config_hash": config.fingerprint(),
                "classification_hash": self.classification_hash,
                "crop_plan_fingerprint": crop_fingerprint,
                "batch_count": len(self.batches), "digest": self.digest}


def _candidate_tags(item: dict, category: str, tag_manager: Any) -> tuple[set[str], bool, bool]:
    if not item.get("is_tags_format", False):
        return set(), False, False
    raw = item.get("tag_data")
    if raw:
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(data, list):
                result = {_name(t["tag"]) for t in data if
                          isinstance(t, dict) and str(t.get("category", "")).casefold() == category
                          and t.get("tag")}
                return result, False, False
        except (ValueError, TypeError, KeyError):
            pass
    raw_caption = item.get("raw_caption") or ""
    if not raw_caption:
        return set(), True, False
    if tag_manager is None:
        from core.training.tag_group_utils import get_tag_group_manager
        tag_manager = get_tag_group_manager()
    tags = [t.strip() for t in raw_caption.split(",") if t.strip()]
    result = {_name(t) for t in tags if
              str(tag_manager.get_tag_group(t) or "").casefold() == category}
    return result, True, not bool(result)


def _classification(items: list[tuple[dict, Any]], config: ConceptOrderConfig,
                    seed: int) -> tuple[dict[str, list[tuple[dict, Any]]],
                                        list[tuple[dict, Any]], str, int, int, tuple[str, ...]]:
    tag_manager = None
    names_by_id: dict[tuple[str, str], set[str]] = {}
    missing, uncategorized = 0, 0
    ordered = sorted(items, key=_item_id)
    classification_digest = sha256()
    for pair in ordered:
        item, _ = pair
        if not item.get("tag_data") and item.get("is_tags_format", False):
            if tag_manager is None:
                from core.training.tag_group_utils import get_tag_group_manager
                tag_manager = get_tag_group_manager()
        tags, was_missing, was_uncategorized = _candidate_tags(
            item, config.category, tag_manager)
        missing += int(was_missing)
        uncategorized += int(was_uncategorized)
        if config.match_natural_language:
            captions = []
            if not item.get("is_tags_format", False):
                captions.append(item.get("raw_caption") or "")
            aux = (item.get("_captions_by_type") or {}).get("natural_language")
            if aux:
                captions.append(aux.get("content") or "")
            for name, aliases in config.caption_aliases.items():
                if any(_caption_matches(caption, [name, *aliases]) for caption in captions):
                    tags.add(name)
        if config.include:
            tags.intersection_update(config.include)
        tags.difference_update(config.exclude)
        names_by_id[_item_id(pair)] = tags
        classification_digest.update(json.dumps(
            (_item_id(pair), sorted(tags), item.get("raw_caption"),
             item.get("tag_data"), item.get("_captions_by_type")),
            ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
        classification_digest.update(b"\n")
    counts = Counter(name for names in names_by_id.values() for name in names)
    missing_includes = tuple(sorted(set(config.include) - set(counts)))
    eligible = {name for name, count in counts.items()
                if count >= config.min_items_per_concept}
    groups: dict[str, list[tuple[dict, Any]]] = defaultdict(list)
    background = []
    assigned = Counter()
    for pair in ordered:
        choices = names_by_id[_item_id(pair)] & eligible
        if not choices:
            background.append(pair)
            continue
        name = min(choices, key=lambda n: (assigned[n], _digest((seed, _item_id(pair), n))))
        groups[name].append(pair)
        assigned[name] += 1
    return dict(groups), background, classification_digest.hexdigest(), missing, uncategorized, missing_includes


def _make_batches(items: list[tuple[dict, Any]], batch_size: int,
                  rng: random.Random) -> list[list[tuple[dict, Any]]]:
    buckets: dict[tuple[int, int, bool], list[tuple[dict, Any]]] = defaultdict(list)
    for pair in items:
        buckets[_bucket_key(pair[0])].append(pair)
    batches = []
    for key in sorted(buckets):
        members = buckets[key]
        rng.shuffle(members)
        batches.extend(members[i:i + batch_size] for i in range(0, len(members), batch_size))
    rng.shuffle(batches)
    return batches


def _batch_digest(batches: list[list[tuple[dict, Any]]],
                  labels: list[str | None], replay: list[bool]) -> str:
    digest = sha256()
    for batch, name, repeated in zip(batches, labels, replay):
        members = []
        for pair in batch:
            item = pair[0]
            crop = item.get("_crop_spec")
            members.append((_item_id(pair), _bucket_key(item),
                            item.get("reference_images"),
                            asdict(crop) if is_dataclass(crop) else str(crop)))
        digest.update(json.dumps((name, repeated, members), ensure_ascii=False,
                                 sort_keys=True, default=str).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_concept_batch_plan(items: list[tuple[dict, Any]], batch_size: int,
                             config: ConceptOrderConfig, epoch: int,
                             seed: int) -> ConceptBatchPlan:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rng = random.Random(int.from_bytes(sha256(
        f"{seed}|{epoch}|{PLAN_VERSION}".encode("utf-8")).digest()[:8], "big"))
    groups, background, classification_hash, missing, uncategorized, missing_includes = _classification(
        items, config, seed)
    queues = {name: _make_batches(members, batch_size, rng)
              for name, members in sorted(groups.items())}
    background_batches = _make_batches(background, batch_size, rng)
    segments: list[tuple[str, list[list[tuple[dict, Any]]]]] = []
    while any(queues.values()):
        names = [name for name, queue in queues.items() if queue]
        rng.shuffle(names)
        for name in names:
            n = config.focus_batches or len(queues[name])
            segment, queues[name] = queues[name][:n], queues[name][n:]
            segments.append((name, segment))

    focus_batches = [batch for _, segment in segments for batch in segment]
    focus_labels = [name for name, segment in segments for _ in segment]
    # A boundary exchange moves each participating batch by at most the window.
    if config.local_swap_window:
        pos = 0
        previous_end = -1
        for i in range(len(segments) - 1):
            pos += len(segments[i][1])
            width = min(config.local_swap_window, len(segments[i][1]),
                        len(segments[i + 1][1]))
            if width and pos - width > previous_end:
                width = rng.randrange(width + 1)
                if width:
                    focus_batches[pos-width:pos+width] = (
                        focus_batches[pos:pos+width] + focus_batches[pos-width:pos])
                    focus_labels[pos-width:pos+width] = (
                        focus_labels[pos:pos+width] + focus_labels[pos-width:pos])
                    previous_end = pos + width - 1

    base_batches: list[list[tuple[dict, Any]]] = []
    base_labels: list[str | None] = []
    bg_idx = 0
    if config.background_placement == "front":
        for focus_idx, (batch, name) in enumerate(zip(focus_batches, focus_labels), 1):
            base_batches.append(batch)
            base_labels.append(name)
            if (config.background_interval and
                    focus_idx % config.background_interval == 0 and
                    bg_idx < len(background_batches)):
                base_batches.append(background_batches[bg_idx])
                base_labels.append(None)
                bg_idx += 1
    else:
        pos = 0
        total_focus = len(focus_batches)
        for i, (_, segment) in enumerate(segments, 1):
            n = len(segment)
            base_batches.extend(focus_batches[pos:pos+n])
            base_labels.extend(focus_labels[pos:pos+n])
            pos += n
            target = (i * len(background_batches)) // len(segments)
            while bg_idx < target:
                base_batches.append(background_batches[bg_idx])
                base_labels.append(None)
                bg_idx += 1
        assert pos == total_focus
    base_batches.extend(background_batches[bg_idx:])
    base_labels.extend([None] * (len(background_batches) - bg_idx))

    batches: list[list[tuple[dict, Any]]] = []
    labels: list[str | None] = []
    replay: list[bool] = []
    seen: dict[str, list[list[tuple[dict, Any]]]] = defaultdict(list)
    last_seen: dict[str, int] = {}
    for idx, (batch, name) in enumerate(zip(base_batches, base_labels)):
        batches.append(batch)
        labels.append(name)
        replay.append(False)
        if name is not None:
            seen[name].append(batch)
            last_seen[name] = idx
        if config.replay_interval and (idx + 1) % config.replay_interval == 0:
            choices = [n for n in seen if idx - last_seen[n] >= config.replay_interval]
            if choices:
                selected = rng.choice(sorted(choices))
                batches.append(rng.choice(seen[selected]))
                labels.append(selected)
                replay.append(True)
                last_seen[selected] = idx
    return ConceptBatchPlan(
        batches, labels, replay, classification_hash,
        _batch_digest(batches, labels, replay), len(groups),
        sum(map(len, groups.values())), len(background),
        sum(len(batch) < batch_size for batch in batches), sum(replay),
        missing, uncategorized, missing_includes)
