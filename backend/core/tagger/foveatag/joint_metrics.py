"""Common masked tag histograms and rolling F1 for legacy joint-run resumes."""
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ..tag_metrics_accumulator import TagMetricsAccumulator
from .metrics_runtime import (masked_metric_inputs, save_metrics_runtime, restore_metrics_runtime,
    save_f1_runtime, restore_f1_runtime)
from .supervised_run import _file_sha256


class JointMetricsRuntime:
    def __init__(self, vocabulary, config):
        self.vocabulary, self.config = vocabulary, dict(config)
        if config["train_f1_buffer_batches"] < 1 or min(config["train_f1_eval_every_n_steps"], config["train_f1_threshold_search_every_n_steps"]) < 0:
            raise ValueError("invalid joint metric intervals or buffer size")
        if not 0 <= config["train_f1_initial_threshold"] <= 1:
            raise ValueError("joint metric threshold must be a probability")
        self.accumulator = TagMetricsAccumulator(vocabulary.num_tags)
        self.buffer = deque(maxlen=config["train_f1_buffer_batches"])
        self.threshold = config["train_f1_initial_threshold"]
        self.start_step = None

    def update(self, logits, labels, masks, step):
        if self.start_step is None:
            self.start_step = step
        predictions, positive, observed = logits.detach().float().sigmoid().half().cpu(), (labels >= .5).cpu(), (masks > 0).cpu()
        self.accumulator.update(predictions, positive, observation_mask=observed)
        self.buffer.append(masked_metric_inputs(predictions, positive, observed))
        interval = self.config["train_f1_eval_every_n_steps"]
        if not interval or step % interval:
            return None
        from ..tagger_trainer import _find_best_threshold, _compute_all_metrics
        predictions = torch.cat([item[0] for item in self.buffer]).float()
        labels = torch.cat([item[1] for item in self.buffer]).float()
        search = self.config["train_f1_threshold_search_every_n_steps"]
        updated = bool(search and step % search == 0)
        if updated:
            self.threshold, _ = _find_best_threshold(predictions, labels)
        metrics = _compute_all_metrics(predictions, labels, threshold=self.threshold)
        return {"step": step, "train_f1": metrics["f1"], "train_precision": metrics["precision"],
            "train_recall": metrics["recall"], "threshold": self.threshold, "threshold_updated": updated,
            "fp_fn_scatter": self.accumulator.compute_scatter_for_vis(min_npos=20) if updated else None}

    def save(self, root, step, *, epoch_boundary=False):
        root = Path(root)
        with TemporaryDirectory(prefix="joint-metrics-", dir=root) as temporary_dir:
            paths = {"histograms": Path(temporary_dir) / "histograms.npz", "rolling_f1": Path(temporary_dir) / "f1.npz"}
            save_metrics_runtime(self.accumulator, paths["histograms"], step=step, epoch_boundary=epoch_boundary)
            save_f1_runtime(self.buffer, paths["rolling_f1"], step=step, threshold=self.threshold, vocab_size=self.vocabulary.num_tags)
            files = {key: {"data": path.read_bytes(), "sha256": _file_sha256(path)} for key, path in paths.items()}
        if self.config["save_tag_metrics"]:
            temporary = root / "latest_tag_metrics.tmp.npz"
            self.accumulator.save(str(temporary), epoch_boundary=epoch_boundary,
                tag_names=[self.vocabulary.idx_to_tag[i] for i in range(self.vocabulary.num_tags)])
            temporary.replace(root / "latest_tag_metrics.npz")
        return {"version": 1, "start_step": self.start_step, "config": self.config,
            "files": files}

    def restore(self, root, record, step):
        if record["version"] != 1:
            raise ValueError("joint metrics runtime version differs")
        if set(record["files"]) != {"histograms", "rolling_f1"}:
            raise ValueError("joint metrics runtime files differ")
        with TemporaryDirectory(prefix="joint-metrics-", dir=root) as temporary_dir:
            paths = {}
            for key, item in record["files"].items():
                paths[key] = Path(temporary_dir) / (key + ".npz")
                paths[key].write_bytes(item["data"])
                if _file_sha256(paths[key]) != item["sha256"]:
                    raise ValueError("joint metrics runtime hash differs")
            restore_metrics_runtime(self.accumulator, paths["histograms"], step=step)
            saved_buffer = deque(maxlen=record["config"]["train_f1_buffer_batches"])
            self.threshold = restore_f1_runtime(saved_buffer, paths["rolling_f1"], step=step, vocab_size=self.vocabulary.num_tags)
            self.buffer.extend(saved_buffer)
        self.start_step = record["start_step"]
