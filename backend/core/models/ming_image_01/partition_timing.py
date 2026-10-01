"""Opt-in synchronized partition measurements; never reset CUDA memory statistics."""
from contextlib import contextmanager
import time
import torch
from api.param_defaults import TRAINING_DEFAULTS


class MingPartitionTiming:
    def __init__(self,trainer,clean):
        self.trainer=trainer;self.device=clean.device;self.frames=clean.shape[2]
        config=trainer.config
        enabled=config.get('dit_partition_profile')
        if enabled is None:enabled=config.get('qwen_partition_profile')
        if enabled is None:enabled=TRAINING_DEFAULTS['dit_partition_profile']
        self.cuda_timing=self.device.type=='cuda' and bool(enabled)
        self.elapsed={'forward':0.,'backward':0.}
        self.start=time.perf_counter()

    @contextmanager
    def measure(self,name):
        if name not in self.elapsed:raise ValueError('Unsupported Ming partition timing phase')
        if not self.cuda_timing:
            yield
            return
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        stream=torch.cuda.current_stream(self.device)
        start.record(stream)
        yield
        end.record(stream);end.synchronize()
        self.elapsed[name]+=start.elapsed_time(end)

    def finish(self,plan,model):
        emit=self.trainer.log_extra_metric
        emit('ming_partition_wall_ms',(time.perf_counter()-self.start)*1000.)
        emit('ming_partition_count',float(len(plan.regions)))
        emit('ming_partition_largest_tokens',float(self.frames*plan.largest_input_tokens))
        emit('ming_partition_total_input_tokens',float(self.frames*sum(r.input.tokens for r in plan.regions)))
        count=getattr(model,'_training_gradient_checkpointing_blocks',None)
        count=len(model.layers) if count is None else count
        if not getattr(model,'gradient_checkpointing',False) or not model.training:count=0
        emit('ming_partition_checkpoint_blocks',float(count))
        if self.cuda_timing:
            for name,value in self.elapsed.items():emit(f'ming_partition_{name}_ms',value)
        if self.device.type=='cuda':
            emit('ming_partition_peak_allocated_gb',torch.cuda.max_memory_allocated(self.device)/(1024**3))
