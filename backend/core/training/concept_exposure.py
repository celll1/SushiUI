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
LOSS_METRIC = "qwen_prediction_reconstruction_v1"


def loss_histogram(saved, key, timestep_counts):
    """Validate additive loss coverage independently of historical exposure."""
    size = len(TIMESTEP_BIN_EDGES) - 1
    if ((key in saved.get("timestep_loss_sums", {}))
            != (key in saved.get("timestep_loss_counts", {}))):
        raise ValueError("Incomplete concept exposure loss histogram")
    sums = list(saved.get("timestep_loss_sums", {}).get(key, [0.0] * size))
    counts = list(saved.get("timestep_loss_counts", {}).get(key, [0] * size))
    if (saved.get("timestep_loss_metric", LOSS_METRIC) != LOSS_METRIC
            or len(sums) != size or len(counts) != size):
        raise ValueError("Invalid concept exposure loss histogram geometry or metric")
    for total, count, exposed in zip(sums, counts, timestep_counts):
        if (not isinstance(count, int) or not 0 <= count <= exposed
                or not isinstance(total, (float, int)) or not math.isfinite(total)
                or total < 0 or (count == 0 and total != 0)):
            raise ValueError("Invalid concept exposure loss histogram")
    return sums, counts


def loss_moments(saved, key, loss_counts):
    """Mean-only history cannot supply a variance cohort."""
    fields = ("timestep_loss_variance_counts", "timestep_loss_variance_means", "timestep_loss_m2")
    present = [key in saved.get(field, {}) for field in fields]
    if any(present) and not all(present):
        raise ValueError("Incomplete concept exposure loss moments")
    size = len(TIMESTEP_BIN_EDGES) - 1
    counts, means, m2s = (list(saved.get(field, {}).get(key, [0] * size)) for field in fields)
    if any(len(values) != size for values in (counts, means, m2s)):
        raise ValueError("Invalid concept exposure loss moments geometry")
    for count, mean, m2, total_count in zip(counts, means, m2s, loss_counts):
        if (not isinstance(count, int) or not 0 <= count <= total_count
                or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                       or value < 0 for value in (mean, m2))
                or (count == 0 and mean != 0) or (count < 2 and m2 != 0)):
            raise ValueError("Invalid concept exposure loss moments")
    return counts, means, m2s


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
        self.timestep_loss_sums = {}
        self.timestep_loss_counts = {}
        loss_keys = set((saved or {}).get("timestep_loss_sums", {})) | set(
            (saved or {}).get("timestep_loss_counts", {}))
        for key in loss_keys:
            sums, counts = loss_histogram(saved, key, self.timestep_counts.get(
                key, [0] * (len(TIMESTEP_BIN_EDGES) - 1)))
            self.timestep_loss_sums[key] = sums
            self.timestep_loss_counts[key] = counts
        self.timestep_loss_variance_counts = {}
        self.timestep_loss_variance_means = {}
        self.timestep_loss_m2 = {}
        moment_fields = ("timestep_loss_variance_counts", "timestep_loss_variance_means", "timestep_loss_m2")
        moment_keys = set().union(*(set((saved or {}).get(field, {})) for field in moment_fields))
        for key in moment_keys:
            counts, means, m2s = loss_moments(saved, key, self.timestep_loss_counts.get(
                key, [0] * (len(TIMESTEP_BIN_EDGES) - 1)))
            self.timestep_loss_variance_counts[key] = counts
            self.timestep_loss_variance_means[key] = means
            self.timestep_loss_m2[key] = m2s
        self.completed_batches = 0
        self._writer = None

    def set_epoch(self, target_items: dict, names: dict):
        for key, name in names.items():
            if key in self.names and self.names[key] != name:
                raise ValueError(f"Cannot resume exposure counts after changing group {key}")
        self.names.update(names)
        self.target_items.update(target_items)

    def record(self, batch, step: int, passes: int, timesteps=None, *, convention="t0", losses=None):
        if passes <= 0:
            return
        if timesteps is not None and len(timesteps) != len(batch):
            raise ValueError("Exposure timesteps must match the completed sample batch")
        if losses is not None and len(losses) != len(batch):
            raise ValueError("Exposure losses must match the completed sample batch")
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
                    loss = float(losses[index]) if losses is not None else None
                    if loss is not None and math.isfinite(loss) and loss >= 0:
                        if group not in self.timestep_loss_counts:
                            self.timestep_loss_counts[group] = [0] * len(bins)
                            self.timestep_loss_sums[group] = [0.0] * len(bins)
                        self.timestep_loss_counts[group][bin_index] += passes
                        self.timestep_loss_sums[group][bin_index] += loss * passes
                        if group not in self.timestep_loss_variance_counts:
                            self.timestep_loss_variance_counts[group] = [0] * len(bins)
                            self.timestep_loss_variance_means[group] = [0.0] * len(bins)
                            self.timestep_loss_m2[group] = [0.0] * len(bins)
                        n = self.timestep_loss_variance_counts[group][bin_index]
                        delta = loss - self.timestep_loss_variance_means[group][bin_index]
                        new_n = n + passes
                        if n == 0:
                            self.timestep_loss_variance_means[group][bin_index] = loss
                        else:
                            self.timestep_loss_variance_means[group][bin_index] += delta * (passes / new_n)
                            self.timestep_loss_m2[group][bin_index] += delta * delta * (n * passes / new_n)
                        self.timestep_loss_variance_counts[group][bin_index] = new_n
        self.completed_batches += 1
        if self.completed_batches % 25 == 0:
            self.publish()

    def state(self):
        return {"mode": self.mode, "counts": dict(self.counts),
                "last_step": dict(self.last_step),
                "target_items": dict(self.target_items),
                "names": dict(self.names),
                "timestep_bin_edges": list(TIMESTEP_BIN_EDGES),
                "timestep_counts": {key: list(values) for key, values in self.timestep_counts.items()},
                "timestep_loss_metric": LOSS_METRIC,
                "timestep_loss_sums": {key: list(values) for key, values in self.timestep_loss_sums.items()},
                "timestep_loss_counts": {key: list(values) for key, values in self.timestep_loss_counts.items()},
                "timestep_loss_variance_counts": {key: list(values) for key, values in self.timestep_loss_variance_counts.items()},
                "timestep_loss_variance_means": {key: list(values) for key, values in self.timestep_loss_variance_means.items()},
                "timestep_loss_m2": {key: list(values) for key, values in self.timestep_loss_m2.items()}}

    def publish(self, *, wait: bool = False):
        if self._writer is not None and self._writer.is_alive():
            if not wait:
                return
            self._writer.join()
        payload = {"version": 4, **self.state()}
        def write():
            pending = self.path.with_suffix(".json.tmp")
            pending.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            pending.replace(self.path)
        if wait:
            write()
        else:
            self._writer = threading.Thread(target=write, daemon=True)
            self._writer.start()
