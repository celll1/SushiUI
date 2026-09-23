"""Run-lifecycle pieces shared by the in-process trainers (tagger, style)."""

import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from api.error_handlers import ValidationError


def validate_gpu_index(gpu_index: Any) -> None:
    """Refuse a gpu_index this machine does not expose; call before the run is moved to "starting"."""
    if gpu_index is None:
        return
    import torch
    device_count = torch.cuda.device_count()
    if not isinstance(gpu_index, int) or not (0 <= gpu_index < device_count):
        raise ValidationError(
            "Selected GPU is not available",
            detail=(
                f"gpu_index={gpu_index!r} but this machine exposes "
                f"{device_count} CUDA device(s)."
            ),
        )


def find_output_dir_owner(training_db, output_dir: str) -> Optional[str]:
    """The training run of any kind (diffusion/VAE, tagger, style) whose output_dir is `output_dir`, as
    "<kind> <id>", else None."""
    from database.models import StyleTrainingRun, TaggerTrainingRun, TrainingRun
    target = os.path.normcase(os.path.abspath(output_dir))
    for kind, key, column in (("training run", TrainingRun.id, TrainingRun.output_dir),
                              ("tagger training run", TaggerTrainingRun.run_id, TaggerTrainingRun.output_dir),
                              ("style training run", StyleTrainingRun.run_id, StyleTrainingRun.output_dir)):
        for run_key, other in training_db.query(key, column).filter(column.isnot(None)).all():
            if other and os.path.normcase(os.path.abspath(other)) == target:
                return f"{kind} {run_key}"
    return None


class LazySerialExecutor:
    """One ordered writer thread, created on first use."""

    def __init__(self, thread_name_prefix: str) -> None:
        self._prefix = thread_name_prefix
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()

    def get(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=self._prefix)
            return self._executor


def next_resume_seq(history_db, metric_model, run_id: str) -> int:
    """0 for a run without metrics, else one past its largest resume_seq."""
    from sqlalchemy import func
    current = history_db.query(
        func.coalesce(func.max(metric_model.resume_seq), -1)
    ).filter(metric_model.run_id == run_id).scalar()
    return int(current) + 1


def upsert_metric_row(history_db, metric_model, run_id: str, resume_seq: int, step: int,
                      values: Dict[str, Any]) -> None:
    """Merge the non-None `values` into the (run_id, resume_seq, step) row; the caller commits."""
    existing = history_db.query(metric_model).filter(
        metric_model.run_id == run_id,
        metric_model.resume_seq == resume_seq,
        metric_model.step == step,
    ).first()
    if existing:
        for key, value in values.items():
            if value is not None:
                setattr(existing, key, value)
    else:
        history_db.add(metric_model(run_id=run_id, resume_seq=resume_seq, step=step, **values))


def decimate_metric_rows(rows: List[Any], max_points: int, is_validation_row: Callable[[Any], bool]) -> List[dict]:
    """`to_dict()` of `rows` (ordered by resume_seq, step), each resume's dense training rows decimated
    uniformly to its share of `max_points` (0 = no limit); validation rows are sparse and always kept."""
    groups: Dict[int, List] = defaultdict(list)
    for m in rows:
        groups[m.resume_seq].append(m)

    data: List[dict] = []
    if max_points > 0 and len(rows) > max_points:
        per_group_max = max(50, max_points // max(1, len(groups)))
        for seq in sorted(groups):
            g_all = groups[seq]
            g_train = [m for m in g_all if not is_validation_row(m)]
            g_val = [m for m in g_all if is_validation_row(m)]
            if len(g_train) > per_group_max:
                step_size = max(1, len(g_train) // per_group_max)
                indices = list(range(0, len(g_train), step_size))
                if indices[-1] != len(g_train) - 1:
                    indices.append(len(g_train) - 1)
                g_train = [g_train[i] for i in indices]
            merged = sorted(g_train + g_val, key=lambda m: m.step)
            data.extend(m.to_dict() for m in merged)
    else:
        for seq in sorted(groups):
            data.extend(m.to_dict() for m in groups[seq])
    return data
