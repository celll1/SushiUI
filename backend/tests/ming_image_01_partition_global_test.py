"""Native global-summary coordinates, gradients and strict artifact contract."""
import pytest
import torch
from core.models.ming_image_01.partition_global import MingPartitionGlobalAdapter,from_payload
from core.training.partition import PartitionBox


@pytest.mark.parametrize('frames',[1,3])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_global_adapter_zero_start_full_context_and_roundtrip(frames,dtype):
    torch.manual_seed(67)
    model=MingPartitionGlobalAdapter(4,24,rank=8,summary_tokens=3,dtype=dtype)
    latents=torch.randn(2,4,frames,8,12,dtype=dtype,requires_grad=True);box=PartitionBox(1,2,3,5)
    result=model(latents,box)
    assert result.shape==(2,frames*6,24) and result.count_nonzero()==0
    result.float().sum().backward()
    assert model.out.weight.grad.abs().sum()>0
    model.zero_grad();latents.grad=None
    torch.nn.init.normal_(model.out.weight,std=.02)
    result=model(latents,box);result.float().square().sum().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum()>0 for p in model.parameters())
    assert latents.grad[:,:,:,:2,:2].abs().sum()>0
    restored=from_payload(model.metadata(),model.export_tensors(),latent_channels=4,hidden_dim=24,dtype=dtype)
    torch.testing.assert_close(restored(latents.detach(),box),result.detach(),rtol=0,atol=0)
    assert not torch.cuda.is_initialized()


def test_native_patch_and_coordinate_axis_order():
    model=MingPartitionGlobalAdapter(2,8,rank=2,summary_tokens=2)
    x=torch.arange(2*3*4*6).float().reshape(1,2,3,4,6)
    features=model._features(x)
    expected=torch.stack([x[0,:,f,y:y+2,w:w+2].permute(1,2,0).reshape(-1)
        for f in range(3) for y in range(0,4,2) for w in range(0,6,2)])
    torch.testing.assert_close(features[0,:,:8],expected,rtol=0,atol=0)
    axes=[torch.linspace(-1,1,n) for n in (3,2,3)]
    coordinates=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
    torch.testing.assert_close(features[0,:,8:],coordinates)
    local=model._features(x,PartitionBox(0,1,2,3))
    torch.testing.assert_close(local,features.reshape(1,3,2,3,11)[:,:,:,1:3].reshape(1,-1,11))


@pytest.mark.parametrize('damage',['missing','extra','shape','integer','nan'])
def test_invalid_payload_is_refused_without_partial_mutation(damage):
    model=MingPartitionGlobalAdapter(4,24,rank=8,summary_tokens=3)
    original=model.export_tensors();state=model.export_tensors()
    if damage=='missing':state.pop('query.weight')
    elif damage=='extra':state['unexpected']=torch.zeros(1)
    elif damage=='shape':state['out.weight']=torch.zeros(1)
    elif damage=='integer':state['value.weight']=state['value.weight'].to(torch.int8)
    else:state['out.weight'][0,0]=float('nan')
    with pytest.raises(ValueError):model.load_tensors(state)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())


@pytest.mark.parametrize('field,value',[('schema_version',True),('schema_version',2),('rank',False),
    ('rank',513),('summary_tokens',257),('hidden_dim',25),('coordinate_axes',['height','width','frame'])])
def test_incompatible_metadata_refused(field,value):
    model=MingPartitionGlobalAdapter(4,24,rank=8,summary_tokens=3);meta=model.metadata();meta[field]=value
    with pytest.raises(ValueError):from_payload(meta,model.export_tensors(),latent_channels=4,hidden_dim=24)


@pytest.mark.parametrize('padding',['zero_masked','learned'])
def test_native_target_residual_is_zero_identity_and_trains_summary(padding):
    from core.models.ming_image_01.model_definition import create_transformer
    from ming_image_01_rewiring_test import config
    torch.manual_seed(71)
    native=create_transformer(config(padding),device='cpu',dtype=torch.float32)
    if padding=='learned':
        torch.nn.init.constant_(native.x_pad_token,.02);torch.nn.init.constant_(native.cap_pad_token,.03)
    frames=3 if padding=='learned' else 1
    full=torch.randn(1,16,frames,32,32,requires_grad=True)
    box=PartitionBox(4,4,12,12)
    from core.models.ming_image_01.partition import MingPartitionAdapter
    position=MingPartitionAdapter().position_ids(16,16,box)
    tile=full[...,8:24,8:24]
    refs=[torch.randn(16,2,32,32)]
    queries=[torch.randn(5,8)];direct=[torch.randn(3,24)]
    branch=MingPartitionGlobalAdapter(16,24,rank=4,summary_tokens=3)
    kwargs=dict(cap_feats_2=direct,ref_x=refs,target_spatial_position_ids=position)
    ordinary=native(list(tile.unbind()),torch.tensor([.4]),queries,**kwargs).sample[0]
    summary=branch(full,box)
    output=native(list(tile.unbind()),torch.tensor([.4]),queries,target_input_residual=summary,**kwargs).sample[0]
    torch.testing.assert_close(output,ordinary,rtol=0,atol=0)
    output.square().mean().backward()
    assert branch.out.weight.grad.isfinite().all() and branch.out.weight.grad.abs().sum()>0
    torch.nn.init.normal_(branch.out.weight,std=.02)
    native.zero_grad();branch.zero_grad();full.grad=None
    output=native(list(tile.unbind()),torch.tensor([.4]),queries,target_input_residual=branch(full,box),**kwargs).sample[0]
    output.square().mean().backward()
    assert full.grad[:,:,:,:4,:4].abs().sum()>0
    assert not torch.cuda.is_initialized()
