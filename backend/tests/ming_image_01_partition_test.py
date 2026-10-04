"""Native partition positions, full reference packing and target-only outputs."""
from copy import deepcopy
import pytest
import torch
from core.models.ming_image_01.partition import MingPartitionAdapter,patchify_partition_image
from core.training.partition import PartitionBox
from core.models.ming_image_01.model_definition import create_transformer
from core.models.ming_image_01.residual_unet import restore_topology
from ming_image_01_rewiring_test import config,spec_for


@pytest.mark.parametrize('padding',['zero_masked','learned'])
@pytest.mark.parametrize('rewired',[False,True])
def test_full_box_matches_native_target_forward_and_all_gradients(padding,rewired):
    torch.manual_seed(29)
    original=create_transformer(config(padding),device='cpu',dtype=torch.float32)
    if padding=='learned':
        torch.nn.init.constant_(original.x_pad_token,.02)
        torch.nn.init.constant_(original.cap_pad_token,.03)
    if rewired:restore_topology(original,spec_for(original))
    tiled=deepcopy(original)
    frames=3 if padding=='learned' else 1
    source=[torch.randn(16,frames,16,16),torch.randn(5,8),torch.randn(35,24),torch.randn(16,2,16,16)]
    position=MingPartitionAdapter().position_ids(8,8,PartitionBox(0,0,8,8))
    predictions=[];leaves=[]
    for model,options in ((original,{}),(tiled,{'target_spatial_position_ids':position})):
        values=[v.clone().requires_grad_() for v in source]
        x,q,c,r=values
        value=model([x],torch.tensor([.4]),[q],cap_feats_2=[c],ref_x=[r],**options).sample[0][:,:frames]
        value.square().mean().backward()
        predictions.append(value);leaves.append(values)
    torch.testing.assert_close(*predictions,rtol=0,atol=0)
    for a,b in zip(*leaves):torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    for (name,a),(other,b) in zip(original.named_parameters(),tiled.named_parameters()):
        assert name==other and (a.grad is None)==(b.grad is None)
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('padding',['zero_masked','learned'])
@pytest.mark.parametrize('rewired',[False,True])
def test_cropped_target_keeps_full_reference_positions_and_gradient(padding,rewired):
    torch.manual_seed(31);model=create_transformer(config(padding),device='cpu',dtype=torch.float32)
    if padding=='learned':
        torch.nn.init.constant_(model.x_pad_token,.02)
        torch.nn.init.constant_(model.cap_pad_token,.03)
    if rewired:restore_topology(model,spec_for(model))
    frames=3 if padding=='learned' else 1
    image=torch.randn(16,frames,16,16,requires_grad=True)
    ref=torch.randn(16,2,32,32,requires_grad=True)
    position=MingPartitionAdapter().position_ids(16,16,PartitionBox(4,6,12,14))
    tokens,size,ids=patchify_partition_image(model,image,ref,64,2,1,position)
    def oracle(value):
        return torch.stack([value[:,f,y:y+2,x:x+2].permute(1,2,0).reshape(-1)
            for f in range(value.shape[1]) for y in range(0,value.shape[2],2) for x in range(0,value.shape[3],2)])
    torch.testing.assert_close(tokens,torch.cat((oracle(image),oracle(ref))),rtol=0,atol=0)
    assert size==(frames,16,16) and tokens.shape[0]==frames*64+512
    assert ids[:frames*64,1:].tolist()==position.repeat(frames,1).tolist()
    assert ids[frames*64].tolist()==[65+frames,0,0]
    assert ids[-1].tolist()==[66+frames,15,15]
    result=model([image],torch.tensor([.4]),[torch.randn(5,8)],cap_feats_2=[torch.randn(35,24)],ref_x=[ref],
        target_spatial_position_ids=position).sample[0]
    assert result.shape==(16,frames,16,16)
    result.square().mean().backward()
    assert image.grad.isfinite().all() and ref.grad.isfinite().all()
    assert ref.grad[:,:,0:2,0:2].abs().sum()>0 and ref.grad[:,:,-2:,-2:].abs().sum()>0
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('invalid',['float','negative','shuffled'])
def test_malformed_target_positions_refused(invalid):
    model=create_transformer(config('zero_masked'),device='cpu',dtype=torch.float32)
    ids=MingPartitionAdapter().position_ids(4,4,PartitionBox(0,0,4,4))
    if invalid=='float':ids=ids.float()
    elif invalid=='negative':ids[0,0]=-1
    else:ids=ids.flip(0)
    with pytest.raises(ValueError,match='coordinates'):
        patchify_partition_image(model,torch.randn(16,1,8,8),None,32,2,1,ids)


def test_adapter_complete_coverage_and_joint_caption_contract():
    adapter=MingPartitionAdapter()
    plan=adapter.build_fixed_plan(16,24,count=4,halo=2,seed=9,epoch=1,occurrence=3)
    coverage=torch.zeros(16,24,dtype=torch.int32)
    for region in plan.regions:
        b=region.core;coverage[b.top:b.bottom,b.left:b.right]+=1
        ids=adapter.position_ids(16,24,region.input)
        assert ids.shape==(region.input.tokens,2)
        assert ids[0].tolist()==[region.input.top,region.input.left]
    assert torch.all(coverage==1)
    assert adapter.topology.prefix_attends_target and adapter.topology.global_position_ids


@pytest.mark.parametrize('height,width', [(105,60),(60,105),(105,61)])
@pytest.mark.parametrize('count', [2,4])
@pytest.mark.parametrize('halo', [0,2])
def test_odd_patch_grid_has_exact_loss_coverage(height,width,count,halo):
    from core.training.partition.core import build_fixed_partition_plan
    adapter=MingPartitionAdapter()
    plan=adapter.build_fixed_plan(height,width,count=count,halo=halo,seed=9)
    coverage=torch.zeros(height,width,dtype=torch.int32)
    for region in plan.regions:
        b=region.core;coverage[b.top:b.bottom,b.left:b.right]+=1
        ids=adapter.position_ids(height,width,region.input)
        assert ids.shape==(region.input.tokens,2)
    assert torch.all(coverage==1)
    with pytest.raises(ValueError,match='even latent sides'):
        build_fixed_partition_plan(height,width,count=count,halo=halo)
