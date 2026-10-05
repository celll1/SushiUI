import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
_original_cuda = (torch.cuda.get_device_capability, torch.cuda.is_available,
                  torch.cuda._lazy_init, torch._C._cuda_init)
try:
    torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
    torch.cuda.is_available = lambda: False
    torch.cuda._lazy_init = lambda *a, **k: None
    torch._C._cuda_init = lambda *a, **k: None
    import bitsandbytes as bnb
    from core.training.base_trainer import BaseTrainer, optimizer_state_is_host_resident
    from core.training.optimizers.optimizer_state_convert import maybe_convert_optimizer_state, detect_state_format
finally:
    (torch.cuda.get_device_capability, torch.cuda.is_available,
     torch.cuda._lazy_init, torch._C._cuda_init) = _original_cuda


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32])
def test_native_adamw_float_moments_convert_like_fp32(dtype):
    p=torch.nn.Parameter(torch.ones(4096,dtype=dtype))
    optimizer=bnb.optim.AdamW8bit([p])
    first=torch.linspace(-.1,.1,4096).to(dtype)
    second=first.square()
    source={'_sushi_opt_class':'AdamW','param_groups':[{'params':[0]}],
            'state':{0:{'step':torch.tensor(3.),'exp_avg':first,'exp_avg_sq':second}}}
    assert detect_state_format(source)=='torch_adamw'
    result,step=maybe_convert_optimizer_state(source,optimizer)
    assert result is not None and step==3
    baseline={'_sushi_opt_class':'AdamW','param_groups':[{'params':[0]}],
              'state':{0:{'step':torch.tensor(3.),'exp_avg':first.float(),'exp_avg_sq':second.float()}}}
    expected,_=maybe_convert_optimizer_state(baseline,optimizer)
    for key in ('state1','state2','absmax1','absmax2','qmap1','qmap2'):
        torch.testing.assert_close(result['state'][0][key],expected['state'][0][key],rtol=0,atol=0)
    assert result['state'][0]['state1'].dtype==torch.uint8
    assert not optimizer_state_is_host_resident(optimizer)
    assert not result['state'][0]['state1'].is_cuda

def test_only_explicit_host_allocator_is_host_resident():
    assert optimizer_state_is_host_resident(SimpleNamespace(get_state_buffer=lambda *a,**k:None))
    assert not optimizer_state_is_host_resident(SimpleNamespace(get_state_buffer=None))


def test_mislabeled_unconverted_moments_are_refused_before_load():
    parameter = torch.nn.Parameter(torch.ones(4096, dtype=torch.bfloat16))
    optimizer = bnb.optim.AdamW8bit([parameter])
    source = {'_sushi_opt_class': 'AdamW8bit', 'param_groups': [{'params': [0]}],
              'state': {0: {'step': 0, 'exp_avg': torch.zeros_like(parameter),
                            'exp_avg_sq': torch.zeros_like(parameter)}}}
    trainer = SimpleNamespace(log_prefix='[test]', device=torch.device('cpu'))
    assert not BaseTrainer._load_one_optimizer_state(trainer, optimizer, source, 'invalid.pt')
    assert not optimizer.state
