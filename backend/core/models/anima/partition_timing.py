"""Opt-in synchronized region timing without resetting process memory counters."""
from contextlib import contextmanager
import time

import torch

from .partition_training import config_value


class AnimaPartitionTiming:
    def __init__(self, trainer, clean):
        self.trainer, self.device = trainer, clean.device
        self.enabled = bool(config_value(trainer.config, "profile"))
        self.cuda_timing = self.enabled and self.device.type == "cuda"
        self.elapsed = {"forward": 0., "backward": 0.}
        self.start = time.perf_counter()

    @contextmanager
    def measure(self, name):
        if not self.cuda_timing:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(self.device))
        yield
        end.record(torch.cuda.current_stream(self.device))
        end.synchronize()
        self.elapsed[name] += start.elapsed_time(end)

    def finish(self, plan, checkpoint_blocks):
        emit = getattr(self.trainer, "log_extra_metric", None)
        if emit is None:
            return
        values = dict(wall_ms=(time.perf_counter()-self.start)*1000.,
            count=len(plan.regions), largest_tokens=plan.largest_input_tokens,
            total_input_tokens=sum(r.input.tokens for r in plan.regions),
            checkpoint_blocks=checkpoint_blocks)
        if self.cuda_timing:
            values.update({name+"_ms": value for name, value in self.elapsed.items()})
        if self.device.type == "cuda":
            values["peak_allocated_gb"] = torch.cuda.max_memory_allocated(self.device)/(1024**3)
        for name, value in values.items():
            emit("anima_partition_"+name, float(value))
