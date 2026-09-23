"""Count completed sample passes for focused training groups."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import threading


def item_key(item, dataset):
    return (str(getattr(dataset, "unique_id", item.get("dataset_unique_id", ""))),
            str(item["image_path"]))


class ConceptExposure:
    def __init__(self, output_dir: Path, mode: str, saved: dict | None = None):
        if saved is not None and saved.get("mode") != mode:
            raise ValueError("Cannot resume exposure counts after changing training mode")
        self.path = Path(output_dir) / "concept_exposure.json"
        self.mode = mode
        self.counts = Counter((saved or {}).get("counts", {}))
        self.last_step = dict((saved or {}).get("last_step", {}))
        self.target_items = dict((saved or {}).get("target_items", {}))
        self.names = dict((saved or {}).get("names", {}))
        self.completed_batches = 0
        self._writer = None

    def set_epoch(self, target_items: dict, names: dict):
        for key, name in names.items():
            if key in self.names and self.names[key] != name:
                raise ValueError(f"Cannot resume exposure counts after changing group {key}")
        self.names.update(names)
        self.target_items.update(target_items)

    def record(self, batch, step: int, passes: int):
        if passes <= 0:
            return
        for item, _dataset in batch:
            group = item.get("_concept_exposure_group")
            if group is not None:
                self.counts[group] += passes
                self.last_step[group] = step
        self.completed_batches += 1
        if self.completed_batches % 25 == 0:
            self.publish()

    def state(self):
        return {"mode": self.mode, "counts": dict(self.counts),
                "last_step": self.last_step, "target_items": self.target_items,
                "names": self.names}

    def publish(self, *, wait: bool = False):
        if self._writer is not None and self._writer.is_alive():
            if not wait:
                return
            self._writer.join()
        payload = {"version": 1, **self.state()}
        def write():
            pending = self.path.with_suffix(".json.tmp")
            pending.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            pending.replace(self.path)
        if wait:
            write()
        else:
            self._writer = threading.Thread(target=write, daemon=True)
            self._writer.start()
