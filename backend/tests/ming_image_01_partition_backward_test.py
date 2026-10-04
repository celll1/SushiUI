"""Real native sequential backward against a single-backward tile oracle."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from core.models.ming_image_01.partition import MingPartitionAdapter
from core.models.ming_image_01.partition_training import backward_regions
from core.models.ming_image_01.conditioning import MingConditioning
from core.models.ming_image_01.model_definition import create_transformer
from core.models.ming_image_01.residual_unet import restore_topology
from ming_image_01_rewiring_test import config,spec_for


@pytest.mark.parametrize('padding',['zero_masked','learned'])
@pytest.mark.parametrize('guided',[False,True])
@pytest.mark.parametrize('grid,rewired',[((8,8),False),((8,8),True),((9,6),False)])
def test_native_sequential_backward_matches_single_backward_oracle(padding,guided,rewired,grid):
    torch.manual_seed(59)
    model=create_transformer(config(padding),device='cpu',dtype=torch.float32)
    if padding=='learned':
        torch.nn.init.constant_(model.x_pad_token,.02)
        torch.nn.init.constant_(model.cap_pad_token,.03)
    if rewired:restore_topology(model,spec_for(model))
    oracle=deepcopy(model);frames=3 if padding=='learned' else 1
    height,width=grid
    clean=torch.randn(2,16,frames,height*2,width*2);noise=torch.randn_like(clean);sigma=torch.tensor([.2,.7])
    weights=torch.tensor([[0.,1.,2.],[0.,2.,1.]]) if frames==3 else torch.ones(2,1)
    q=torch.randn(2,5,8);c=torch.randn(2,3,24)
    refs=[torch.randn(16,1,16,16),torch.randn(16,2,16,16)]
    settings={'qwen_guidance_loss_weight':.4 if guided else 0.,'qwen_guidance_loss_mix_mode':'blend',
              'qwen_guidance_loss_scale':4.,'qwen_guidance_loss_schedule':'sigma',
              'qwen_guidance_loss_extrapolation_max_rms':.25}
    plan=MingPartitionAdapter().build_fixed_plan(height,width,count=4,halo=0,split_ratio_min=.5,split_ratio_max=.5)
    leaves=[];metrics={}
    for current,sequential in ((model,True),(oracle,False)):
        qs=q.clone().requires_grad_(not guided);cs=c.clone().requires_grad_(not guided);leaves.append((qs,cs))
        batch=SimpleNamespace(clean=clean,frame_weights=weights,reference_latents=refs,
            conditioning=MingConditioning(qs*2,cs*2))
        owner=SimpleNamespace(transformer=current,device='cpu',training_dtype=torch.float32,mixed_precision=False,
            config=settings,reconstruction_loss_weight=.3,_collect_concept_sample_loss=True,
            log_extra_metric=lambda k,v:metrics.__setitem__(k,v))
        if sequential:
            actual=backward_regions(owner,batch,sigma=sigma,noise=noise,plan=plan,backward_scale=.125)
            actual_items=owner._pending_concept_sample_loss
            continue
        noisy=(1-sigma[:,None,None,None,None])*clean+sigma[:,None,None,None,None]*noise;target=clean-noise
        def crop(v,b):return v[...,2*b.top:2*b.bottom,2*b.left:2*b.right]
        def error(a,b):return ((a.float()-b.float()).square().mean(dim=(1,3,4))*weights).sum(1)/weights.sum(1)
        def forward(region,zero=False):
            return torch.stack(current(list(crop(noisy,region.input).unbind()),1-sigma,
                list((batch.conditioning.queries*0 if zero else batch.conditioning.queries).unbind()),
                cap_feats_2=list((batch.conditioning.direct*0 if zero else batch.conditioning.direct).unbind()),
                ref_x=refs,target_spatial_position_ids=MingPartitionAdapter().position_ids(height,width,region.input),return_dict=False)[0])
        nulls=[];distance=torch.zeros(2)
        if guided:
            with torch.no_grad():
                for r in plan.regions:
                    u=forward(r,True);nulls.append(u)
                    distance+=r.core.tokens/plan.full_tokens*error(sigma[:,None,None,None,None]*(crop(target,r.core)-u),torch.zeros_like(u))
        scale=1+torch.minimum(3*sigma,.25/distance.sqrt()) if guided else None
        expected_items=torch.zeros(2);expected_pred=torch.zeros(2);expected_recon=torch.zeros(2)
        for i,r in enumerate(plan.regions):
            pred=forward(r);normal=error(pred,crop(target,r.core))
            per=normal if not guided else .6*normal+.4*error(pred,nulls[i]+scale[:,None,None,None,None]*(crop(target,r.core)-nulls[i]))
            recon=error(crop(noisy,r.core)+sigma[:,None,None,None,None]*pred,crop(clean,r.core))
            fraction=r.core.tokens/plan.full_tokens
            expected_pred+=fraction*per;expected_recon+=fraction*recon
            expected_items+=fraction*(.7*per+.3*recon)
        (.125*expected_items.mean()).backward()
        torch.testing.assert_close(torch.tensor(actual),torch.stack((expected_items.mean(),expected_pred.mean(),expected_recon.mean())).detach())
        torch.testing.assert_close(actual_items,expected_items.detach())
    for (_,a),(_,b) in zip(model.named_parameters(),oracle.named_parameters()):
        assert (a.grad is None)==(b.grad is None)
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=2e-6)
    if not guided:
        for a,b in zip(*leaves):torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=2e-6)
    assert metrics['ming_partition_count']==4.
    assert not torch.cuda.is_initialized()
