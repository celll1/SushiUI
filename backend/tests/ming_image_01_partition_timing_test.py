"""CPU execution equivalence and mocked event ownership; no actual CUDA measurements."""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import pytest
import torch
from core.models.ming_image_01.partition_timing import MingPartitionTiming
from core.models.ming_image_01.partition_training import backward_regions
from core.training.metric_registry import EXTRA_METRIC_DEFS
from ming_image_01_training_debug_test import fixture


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('enabled',[False,True])
def test_cpu_actual_guided_partition_profile_is_gpu_free_and_preserves_all_gradients(monkeypatch,profile,enabled):
    owner,batch,noise,sigma,plan=fixture(profile,True);baseline=deepcopy(owner)
    owner.config['dit_partition_profile']=enabled;baseline.config['dit_partition_profile']=False
    metrics={};owner.log_extra_metric=lambda k,v:metrics.__setitem__(k,v)
    def forbidden(*a,**k):raise AssertionError('CPU timing touched CUDA')
    for name in ('Event','current_stream','max_memory_allocated','reset_peak_memory_stats','synchronize'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    actual=backward_regions(owner,batch,sigma=sigma,noise=noise,plan=plan,backward_scale=.125)
    expected=backward_regions(baseline,batch,sigma=sigma,noise=noise,plan=plan,backward_scale=.125)
    assert actual==expected
    for a,b in zip(owner.transformer.parameters(),baseline.transformer.parameters()):
        if a.grad is None:assert b.grad is None
        else:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    assert metrics['ming_partition_wall_ms']>0 and metrics['ming_partition_count']==4
    frames=batch.clean.shape[2]
    assert metrics['ming_partition_largest_tokens']==frames*plan.largest_input_tokens
    assert metrics['ming_partition_total_input_tokens']==frames*sum(r.input.tokens for r in plan.regions)
    assert metrics['ming_partition_checkpoint_blocks']==0
    assert not any(key in metrics for key in ('ming_partition_forward_ms','ming_partition_backward_ms','ming_partition_peak_allocated_gb'))
    assert not torch.cuda.is_initialized()


def mocked_timer(monkeypatch,config):
    events=[];trace=[];metrics={}
    class Event:
        def __init__(self,enable_timing):assert enable_timing;events.append(self)
        def record(self,stream):assert stream=='training_stream';trace.append('record')
        def synchronize(self):trace.append('sync')
        def elapsed_time(self,end):assert end in events;trace.append('elapsed');return 2.5
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:'training_stream')
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda device:12*1024**3)
    def forbidden(*a,**k):raise AssertionError('Timing reset GPU statistics or initialized CUDA')
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',forbidden)
    monkeypatch.setattr(torch.cuda,'synchronize',forbidden)
    owner=SimpleNamespace(config=config,log_extra_metric=lambda k,v:metrics.__setitem__(k,v))
    clean=SimpleNamespace(device=torch.device('cuda'),shape=(2,16,3,16,16))
    timer=MingPartitionTiming(owner,clean)
    plan=SimpleNamespace(regions=[SimpleNamespace(input=SimpleNamespace(tokens=8)) for _ in range(4)],largest_input_tokens=8)
    model=SimpleNamespace(layers=[None]*6,gradient_checkpointing=True,training=True,_training_gradient_checkpointing_blocks=3)
    return timer,plan,model,events,trace,metrics


@pytest.mark.parametrize('config',[{'dit_partition_profile':True},{'qwen_partition_profile':True},{'dit_partition_profile':None,'qwen_partition_profile':True}])
def test_mocked_cuda_profile_accounts_null_conditional_and_backward_spans_without_global_reset(monkeypatch,config):
    timer,plan,model,events,trace,metrics=mocked_timer(monkeypatch,config)
    for name in ('forward','forward','backward'):
        with timer.measure(name):trace.append(name)
    timer.finish(plan,model)
    assert len(events)==6 and trace.count('sync')==3
    assert metrics['ming_partition_forward_ms']==5. and metrics['ming_partition_backward_ms']==2.5
    assert metrics['ming_partition_peak_allocated_gb']==12 and metrics['ming_partition_checkpoint_blocks']==3
    assert metrics['ming_partition_total_input_tokens']==96 and metrics['ming_partition_largest_tokens']==24
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('config',[{}, {'dit_partition_profile':False}])
def test_mocked_cuda_profile_disabled_emits_no_events_and_keeps_process_peak(monkeypatch,config):
    timer,plan,model,events,trace,metrics=mocked_timer(monkeypatch,config)
    with timer.measure('forward'):pass
    with timer.measure('backward'):pass
    timer.finish(plan,model)
    assert not events and not trace
    assert 'ming_partition_forward_ms' not in metrics and metrics['ming_partition_peak_allocated_gb']==12


@pytest.mark.parametrize('phase',['forward','backward'])
def test_mocked_event_failure_preserves_original_exception_without_success_measurement(monkeypatch,phase):
    timer,plan,model,events,trace,metrics=mocked_timer(monkeypatch,{'dit_partition_profile':True})
    with pytest.raises(RuntimeError,match='model failed'):
        with timer.measure(phase):raise RuntimeError('model failed')
    assert trace==['record'] and timer.elapsed=={'forward':0.,'backward':0.} and not metrics


def test_metric_registry_and_frontend_spec_contract():
    for suffix in ('forward_ms','backward_ms','wall_ms','peak_allocated_gb','count','largest_tokens','total_input_tokens','checkpoint_blocks'):
        ming,qwen=EXTRA_METRIC_DEFS['ming_partition_'+suffix],EXTRA_METRIC_DEFS['qwen_partition_'+suffix]
        assert ming['family']==qwen['family'] and ming['scale_group']==qwen['scale_group']
        assert ming is not qwen and ming['label'].startswith('Ming')
    assert EXTRA_METRIC_DEFS['ming_guidance_loss_normal']['family']=='loss'
    root=Path(__file__).resolve().parents[2]
    source=(root/'frontend/src/components/training/TrainingConfig.tsx').read_text(encoding='utf-8')
    assert 'const isMingImageModel =' in source and 'isMingImage01Model' not in source
    assert '(isQwenImage21Model || isMingImageModel) && <label className="flex items-center gap-2 self-end' in source
    import yaml
    spec=yaml.safe_load((root/'openapi.yaml').read_text(encoding='utf-8'))
    assert 'Ming' in spec['components']['schemas']['TrainingRunCreateRequest']['properties']['dit_partition_profile']['description']
