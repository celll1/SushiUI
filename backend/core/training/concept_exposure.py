"""Count completed sample passes for focused training groups."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import math
import json
from pathlib import Path
import threading


# Versioned storage geometry, with finer coverage near pure noise.
TIMESTEP_BIN_EDGES = tuple(i / 20 for i in range(17)) + tuple(i / 100 for i in range(81, 101))


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
        edges = (saved or {}).get("timestep_bin_edges")
        if edges is not None and tuple(edges) != TIMESTEP_BIN_EDGES:
            raise ValueError("Cannot resume exposure histogram with incompatible bin edges")
        self.timestep_counts = {
            key: list(values) for key, values in (saved or {}).get("timestep_counts", {}).items()
        }
        for values in self.timestep_counts.values():
            if len(values) != len(TIMESTEP_BIN_EDGES) - 1 or any(
                not isinstance(value, int) or value < 0 for value in values
            ):
                raise ValueError("Invalid saved concept exposure timestep histogram")
        self.completed_batches = 0
        self._writer = None

    def set_epoch(self, target_items: dict, names: dict):
        for key, name in names.items():
            if key in self.names and self.names[key] != name:
                raise ValueError(f"Cannot resume exposure counts after changing group {key}")
        self.names.update(names)
        self.target_items.update(target_items)

    def record(self, batch, step: int, passes: int, timesteps=None, *, convention="t0"):
        if passes <= 0:
            return
        if timesteps is not None and len(timesteps) != len(batch):
            raise ValueError("Exposure timesteps must match the completed sample batch")
        for index, (item, _dataset) in enumerate(batch):
            bin_index = None
            if timesteps is not None:
                value = float(timesteps[index])
                value = value if convention == "t0" else 1 - value
                if math.isfinite(value) and 0 <= value <= 1:
                    bin_index = min(bisect_right(TIMESTEP_BIN_EDGES, value) - 1,
                                    len(TIMESTEP_BIN_EDGES) - 2)
            groups = item.get("_concept_exposure_groups")
            if groups is None:
                group = item.get("_concept_exposure_group")
                groups = (group,) if group is not None else ()
            for group in groups:
                self.counts[group] += passes
                self.last_step[group] = step
                if bin_index is not None:
                    bins = self.timestep_counts.get(group)
                    if bins is None:
                        bins = self.timestep_counts[group] = [0] * (len(TIMESTEP_BIN_EDGES) - 1)
                    bins[bin_index] += passes
        self.completed_batches += 1
        if self.completed_batches % 25 == 0:
            self.publish()

    def state(self):
        return {"mode": self.mode, "counts": dict(self.counts),
                "last_step": dict(self.last_step),
                "target_items": dict(self.target_items),
                "names": dict(self.names),
                "timestep_bin_edges": list(TIMESTEP_BIN_EDGES),
                "timestep_counts": {key: list(values) for key, values in self.timestep_counts.items()}}

    def publish(self, *, wait: bool = False):
        if self._writer is not None and self._writer.is_alive():
            if not wait:
                return
            self._writer.join()
        payload = {"version": 2, **self.state()}
        def write():
            pending = self.path.with_suffix(".json.tmp")
            pending.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            pending.replace(self.path)
        if wait:
            write()
        else:
            self._writer = threading.Thread(target=write, daemon=True)
            self._writer.start()
