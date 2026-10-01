"""Independent full-image oracles for halo-free Ming partition losses."""
from types import SimpleNamespace
import pytest
import torch
from core.models.ming_image_01.partition import MingPartitionAdapter,core_result,image_guidance_scale
from core.models.ming_image_01.guidance import prepare_guidance,guidance_loss


@pytest.mark.parametrize('count',[2,4])
@pytest.mark.parametrize('schedule',['constant','sigma'])
@pytest.mark.parametrize('halo',[0,2])
def test_partition_loss_cap_and_gradient_match_full_image_oracle(count,schedule,halo):
    torch.manual_seed(47)
    plan=MingPartitionAdapter().build_fixed_plan(8,12,count=count,halo=halo,seed=8)
    sigma=torch.tensor([.2,.8]);weights=torch.tensor([[0.,1.,2.],[0.,2.,1.]])
    prediction=torch.randn(2,4,3,16,24,requires_grad=True)
    target=torch.randn_like(prediction);uncond=torch.randn_like(prediction)
    uncond[:,:,0]=100000
    owner=SimpleNamespace(config={'qwen_guidance_loss_weight':.4,'qwen_guidance_loss_mix_mode':'blend',
        'qwen_guidance_loss_scale':5.,'qwen_guidance_loss_schedule':schedule,
        'qwen_guidance_loss_extrapolation_max_rms':.25},log_extra_metric=lambda *a:None)
    condition,selection,selected=prepare_guidance(owner,sigma,None)
    def crop(value,box):return value[...,2*box.top:2*box.bottom,2*box.left:2*box.right]
    pairs=[(crop(target,r.core),crop(uncond,r.core)) for r in plan.regions]
    effective=image_guidance_scale(owner,sigma,condition,weights,pairs,plan)
    requested=torch.full_like(sigma,5.) if schedule=='constant' else 1+4*sigma
    distance=(sigma[:,None,None,None,None]*(target-uncond)).square().mean(dim=(1,3,4))
    rms=((distance*weights).sum(1)/weights.sum(1)).sqrt()
    expected_scale=1+torch.minimum(requested-1,.25/rms)
    torch.testing.assert_close(effective,expected_scale)
    normal=((prediction-target).square().mean(dim=(1,3,4))*weights).sum(1)/weights.sum(1)
    guided_target=uncond+expected_scale[:,None,None,None,None]*(target-uncond)
    guided=((prediction-guided_target).square().mean(dim=(1,3,4))*weights).sum(1)/weights.sum(1)
    expected=(.6*normal+.4*guided).mean()
    expected_grad=torch.autograd.grad(expected,prediction)[0]
    detached=prediction.detach().clone().requires_grad_();total=0
    for region,(_,null) in zip(plan.regions,pairs):
        result=core_result(crop(detached,region.input),crop(target,region.input),region,1-sigma,weights)
        loss,_=guidance_loss(owner,result,null,sigma,weights,condition,selection,selected,
            effective_scale_override=effective)
        fraction=region.core.tokens/plan.full_tokens
        total+=fraction*loss.detach()
        (fraction*loss).backward()
    torch.testing.assert_close(total,expected.detach())
    torch.testing.assert_close(detached.grad,expected_grad)
    assert detached.grad[:,:,0].count_nonzero()==0
    assert not torch.cuda.is_initialized()


def test_halo_predictions_never_receive_core_loss_gradient():
    plan=MingPartitionAdapter().build_fixed_plan(8,8,count=4,halo=2)
    region=plan.regions[0]
    prediction=torch.randn(1,4,2,region.input.height*2,region.input.width*2,requires_grad=True)
    target=torch.randn_like(prediction)
    result=core_result(prediction,target,region,torch.tensor([.4]),torch.tensor([[0.,1.]]))
    result.loss.backward()
    box=region.core_in_input
    valid=torch.zeros_like(prediction,dtype=torch.bool)
    valid[:,:,1,2*box.top:2*box.bottom,2*box.left:2*box.right]=True
    assert prediction.grad[~valid].count_nonzero()==0
    assert prediction.grad[valid].abs().sum()>0


def test_cap_requires_complete_core_collection():
    owner=SimpleNamespace(config={'qwen_guidance_loss_extrapolation_max_rms':.25})
    plan=MingPartitionAdapter().build_fixed_plan(8,8,count=2)
    with pytest.raises(ValueError,match='every partition core'):
        image_guidance_scale(owner,torch.tensor([.4]),None,torch.ones(1,1),[],plan)
