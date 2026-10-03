"""Exact training histograms, separate from merged inference calibration files."""

from pathlib import Path
import os

import numpy as np
import torch


HISTOGRAMS = tuple(f"{kind}_hist_{slot}" for kind in ("pos", "total") for slot in ("cur", "prev", "pp"))
VECTORS = ("tag_count", "tag_count_epoch_start", "last_epoch_delta")
COUNTERS = ("total_images_cur", "total_images_prev", "total_images_pp", "total_images_all", "epochs_elapsed")


def _atomic_npz(path, document):
    path = Path(path)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **document)
    os.replace(temporary, path)


def _row_mapping(size, target_size, saved_vocab, current_vocab, lineage):
    if (saved_vocab is None) != (current_vocab is None):
        raise ValueError("native metric vocabulary mapping is incomplete")
    if saved_vocab is None:
        if size != target_size:
            raise ValueError("native metric row changes require vocabulary mappings")
        return list(range(size))
    if set(saved_vocab.values()) != set(range(size)) or set(current_vocab.values()) != set(range(target_size)):
        raise ValueError("native metric vocabulary indices differ")
    rows = [None] * target_size
    for name, index in current_vocab.items():
        sources = ([saved_vocab[name]] if name in saved_vocab else
                   [saved_vocab[source] for source in (lineage or {}).get(name, [name]) if source in saved_vocab])
        sources = list(dict.fromkeys(sources))
        if len(sources) == 1:
            rows[index] = sources[0]
    return rows


def save_metrics_runtime(accumulator, path, *, step, epoch_boundary):
    arrays = {key: getattr(accumulator, key) for key in HISTOGRAMS + VECTORS}
    counters = {key: getattr(accumulator, key) for key in COUNTERS}
    if epoch_boundary:
        for kind in ("pos", "total"):
            arrays[f"{kind}_hist_pp"] = getattr(accumulator, f"{kind}_hist_prev")
            arrays[f"{kind}_hist_prev"] = getattr(accumulator, f"{kind}_hist_cur")
            arrays[f"{kind}_hist_cur"] = np.zeros_like(arrays[f"{kind}_hist_cur"])
        counters.update(total_images_cur=0, total_images_prev=accumulator.total_images_cur,
                        total_images_pp=accumulator.total_images_prev,
                        epochs_elapsed=accumulator.epochs_elapsed + 1)
        arrays["last_epoch_delta"] = (accumulator.tag_count - accumulator.tag_count_epoch_start).astype(np.int32)
        arrays["tag_count_epoch_start"] = accumulator.tag_count
    document = {**arrays, **{key: np.int64(value) for key, value in counters.items()},
                "version": np.int64(1), "global_step": np.int64(step),
                "vocab_size": np.int64(accumulator.vocab_size), "n_bins": np.int64(accumulator.n_bins)}
    _atomic_npz(path, document)


def restore_metrics_runtime(accumulator, path, *, step, saved_vocab=None, current_vocab=None, lineage=None):
    with np.load(path, allow_pickle=False) as data:
        if int(data["version"]) != 1 or int(data["global_step"]) != step:
            raise ValueError("native metric runtime version or step differs")
        size = int(data["vocab_size"])
        if size <= 0 or int(data["n_bins"]) != accumulator.n_bins:
            raise ValueError("native metric runtime dimensions differ")
        arrays = {}
        for key in HISTOGRAMS + VECTORS:
            value = data[key]
            shape = (size, accumulator.n_bins) if key in HISTOGRAMS else (size,)
            if value.shape != shape or value.dtype != np.int32 or np.any(value < 0):
                raise ValueError("invalid native metric array")
            arrays[key] = value.copy()
        counters = {key: int(data[key]) for key in COUNTERS}
        if any(value < 0 for value in counters.values()):
            raise ValueError("invalid native metric counter")
        for slot in ("cur", "prev", "pp"):
            if np.any(arrays[f"pos_hist_{slot}"] > arrays[f"total_hist_{slot}"]):
                raise ValueError("native positives exceed observed counts")
            if np.any(arrays[f"total_hist_{slot}"].sum(axis=1, dtype=np.int64) > counters[f"total_images_{slot}"]):
                raise ValueError("native observations exceed image counts")
    rows = _row_mapping(size, accumulator.vocab_size, saved_vocab, current_vocab, lineage)
    if rows != list(range(size)):
        targets = np.array([index for index, source in enumerate(rows) if source is not None], dtype=np.int64)
        sources = np.array([source for source in rows if source is not None], dtype=np.int64)
        aligned = {key: np.zeros_like(getattr(accumulator, key)) for key in HISTOGRAMS + VECTORS}
        for key in aligned:
            aligned[key][targets] = arrays[key][sources]
        arrays = aligned
    for key, value in {**arrays, **counters}.items():
        setattr(accumulator, key, value)
    return {"reset_rows": sum(source is None for source in rows)}


def masked_metric_inputs(predictions, labels, observation_mask):
    observed = observation_mask.to(device=predictions.device) > 0
    if observed.shape != predictions.shape or labels.shape != predictions.shape:
        raise ValueError("native metric observation shape differs")
    # A negative sentinel excludes unknowns at every supported probability threshold.
    return predictions.masked_fill(~observed, -1), labels.masked_fill(~observed, 0)


def save_f1_runtime(buffer, path, *, step, threshold, vocab_size):
    document = {"version": np.int64(1), "global_step": np.int64(step),
        "threshold": np.float64(threshold), "vocab_size": np.int64(vocab_size),
        "capacity": np.int64(buffer.maxlen), "length": np.int64(len(buffer))}
    for index, (predictions, labels) in enumerate(buffer):
        document[f"predictions_{index}"] = predictions.numpy()
        document[f"labels_{index}"] = labels.numpy()
    _atomic_npz(path, document)


def restore_f1_runtime(buffer, path, *, step, vocab_size, saved_vocab=None, current_vocab=None, lineage=None):
    entries = []
    with np.load(path, allow_pickle=False) as data:
        if int(data["version"]) != 1 or int(data["global_step"]) != step:
            raise ValueError("native F1 runtime version or step differs")
        if int(data["capacity"]) != buffer.maxlen:
            raise ValueError("native F1 buffer capacity differs")
        length, size, threshold = int(data["length"]), int(data["vocab_size"]), float(data["threshold"])
        if length < 0 or length > buffer.maxlen or not 0 <= threshold <= 1:
            raise ValueError("invalid native F1 runtime")
        rows = _row_mapping(size, vocab_size, saved_vocab, current_vocab, lineage)
        targets = np.array([index for index, source in enumerate(rows) if source is not None], dtype=np.int64)
        sources = np.array([source for source in rows if source is not None], dtype=np.int64)
        for index in range(length):
            predictions, labels = data[f"predictions_{index}"], data[f"labels_{index}"]
            if (predictions.ndim != 2 or predictions.shape[1] != size or labels.shape != predictions.shape
                    or predictions.dtype != np.float16 or labels.dtype != np.bool_
                    or not np.isfinite(predictions).all() or np.any(predictions > 1)
                    or np.any((predictions < 0) & (predictions != -1)) or np.any((predictions < 0) & labels)):
                raise ValueError("invalid native F1 buffer entry")
            if rows == list(range(size)):
                aligned_p, aligned_l = predictions.copy(), labels.copy()
            else:
                aligned_p = np.full((len(predictions), vocab_size), -1, dtype=np.float16)
                aligned_l = np.zeros(aligned_p.shape, dtype=np.bool_)
                aligned_p[:, targets], aligned_l[:, targets] = predictions[:, sources], labels[:, sources]
            entries.append((torch.from_numpy(aligned_p), torch.from_numpy(aligned_l)))
    buffer.clear()
    buffer.extend(entries)
    return threshold
