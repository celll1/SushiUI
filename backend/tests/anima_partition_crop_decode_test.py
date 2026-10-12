"""Region crop loss keeps core/context geometry, scaling and resume RNG."""
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

from core.models.anima.partition_training import training_plan, backward_regions, validate_training_options
from core.training.ops import crop_decode_loss as shared
from crop_decode_loss_multi_arch_test import MockVAE


class Pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.2))
        self.blocks = []
        self.calls = []
    def forward(self, x, **kwargs):
        self.calls.append(kwargs['partition_box'])
        return self.scale * x


def owner(weight=.4, count=4):
    return SimpleNamespace(config=dict(dit_partition_training_enabled=True,
        dit_partition_mode='fixed', dit_partition_fixed_count=count, dit_partition_halo_tokens=0),
        transformer=Pointwise(), device=torch.device('cpu'), training_dtype=torch.float32,
        mixed_precision=False, reconstruction_loss_weight=0., vae=MockVAE(4, scale=2),
        crop_decode_loss_enable=True, crop_decode_loss_weight=weight,
        crop_decode_loss_metric='mse', crop_decode_loss_margin_cells=6,
        crop_decode_loss_out_cells=4, crop_decode_loss_snr_range='', log_prefix='[test]')


@pytest.mark.parametrize('count', [2,4])
def test_core_context_one_decode_and_aux_scaling(monkeypatch,count):
    torch.manual_seed(91)
    trainer=owner(count=count)
    metrics={};trainer.log_extra_metric=lambda name,value: metrics.__setitem__(name,value)
    clean,noise=torch.randn(2,4,32,48),torch.randn(2,4,32,48)
    sigma=torch.tensor([.2,.5])
    plan=training_plan(trainer,clean)
    bounds=[];rectangles=[]
    compute=shared.compute_crop_decode_loss
    decode=shared.decode_crop_with_context
    def record(*args,**kwargs):
        bounds.append(kwargs['crop_bounds'])
        return compute(*args,**kwargs)
    def decode_record(vae,latent,rect,**kwargs):
        rectangles.append(rect)
        return decode(vae,latent,rect,**kwargs)
    monkeypatch.setattr(shared,'compute_crop_decode_loss',record)
    monkeypatch.setattr(shared,'decode_crop_with_context',decode_record)
    args=dict(clean=clean,noise=noise,sigma=sigma,plan=plan,prompt_embeds=torch.zeros(2,1,1),
        anima_aux={k:torch.ones(2,1,dtype=torch.long) for k in ('source_mask','t5_input_ids','t5_attn_mask')},
        backward_scale=.25)
    state=torch.get_rng_state()
    result=backward_regions(trainer,**args)
    gradient=trainer.transformer.scale.grad.clone()
    assert len(bounds)==1 and len(rectangles)==2
    assert rectangles[0]==rectangles[1]
    selected=next(r for r in plan.regions if r.input==trainer.transformer.calls[-1])
    core=selected.core_in_input
    assert bounds[0]==(2*core.top,2*core.bottom,2*core.left,2*core.right)
    rect=rectangles[0]
    assert bounds[0][0]<=rect.y0<rect.y1<=bounds[0][1]
    assert bounds[0][2]<=rect.x0<rect.x1<=bounds[0][3]
    assert rect.py0==max(0,rect.y0-6) and rect.px0==max(0,rect.x0-6)
    assert result[0]-result[1]==pytest.approx(.4*float(metrics['crop_decode_loss']),rel=1e-5)
    assert all(p.grad is None and not p.requires_grad for p in trainer.vae.parameters())
    trainer.transformer.scale.grad=None;trainer.transformer.calls.clear();bounds.clear();rectangles.clear()
    torch.set_rng_state(state)
    repeated=backward_regions(trainer,**args)
    assert repeated==result
    torch.testing.assert_close(trainer.transformer.scale.grad,gradient)
    trainer.crop_decode_loss_weight=0.;trainer.transformer.scale.grad=None
    rng=torch.get_rng_state();baseline=backward_regions(trainer,**args)
    assert torch.equal(torch.get_rng_state(),rng)
    assert baseline[0]==baseline[1]
    assert not torch.allclose(trainer.transformer.scale.grad,gradient)


def test_plan_expands_decode_halo_only_when_active():
    trainer=owner()
    clean=torch.zeros(1,4,32,48)
    active=training_plan(trainer,clean)
    trainer.crop_decode_loss_weight=0.
    inactive=training_plan(trainer,clean)
    for a,b in zip(active.regions,inactive.regions):
        assert a.core==b.core
        assert a.input.top<=b.input.top and a.input.bottom>=b.input.bottom
        assert a.input.left<=b.input.left and a.input.right>=b.input.right
    validate_training_options({**trainer.config,'crop_decode_loss_enable':True})


def test_snr_skip_does_not_initialize_or_decode():
    trainer=owner();trainer.crop_decode_loss_snr_range='100,200'
    clean=torch.randn(2,4,16,16);prediction=clean.clone().requires_grad_()
    aux,raw=shared.compute_crop_decode_loss(trainer,prediction,clean,torch.tensor([.5,.7]),clean,
        noise_process='flow',prediction_target='velocity',noise_scheduler=None,
        velocity_sign='eps_minus_x0')
    assert aux is None and raw==0.
    assert not hasattr(trainer,'_crop_decode_loss_module')


def test_vae_residency_restores_dtype_even_on_error():
    trainer=owner()
    with pytest.raises(RuntimeError,match='backward'):
        with shared.preserve_crop_vae_residency(trainer):
            trainer.vae.double()
            raise RuntimeError('backward')
    assert next(trainer.vae.parameters()).dtype==torch.float32


def test_openapi_crop_fields_match_runtime_defaults():
    import yaml
    from api.param_defaults import TRAINING_DEFAULTS
    from api.routes import TrainingRunCreateRequest
    schema=yaml.safe_load((Path(__file__).resolve().parents[2]/'openapi.yaml').read_text(encoding='utf-8'))
    properties=schema['components']['schemas']['TrainingRunCreateRequest']['properties']
    for name in ('enable','weight','margin_cells','out_cells','metric','snr_range'):
        key='crop_decode_loss_'+name
        assert properties[key]['default']==TRAINING_DEFAULTS[key]
        assert TrainingRunCreateRequest.model_fields[key].default==TRAINING_DEFAULTS[key]
