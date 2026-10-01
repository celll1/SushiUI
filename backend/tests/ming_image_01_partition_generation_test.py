"""Native partition prediction and sampler cleanup, with no running API or GPU."""
import pytest
import torch
from core.inference.qwen_image_21_partition import Bounds,GenerationPlan
from core.models.ming_image_01.partition_generation import MingPartitionExecutor
from core.models.ming_image_01.partition_global import MingPartitionGlobalAdapter
from core.models.ming_image_01.model_definition import create_transformer
from core.models.ming_image_01.partition import MingPartitionAdapter
from core.training.partition import PartitionBox
from ming_image_01_rewiring_test import config,spec_for,restore_topology


def plan(size,halo=0):
    cores=(Bounds(0,0,size,size//2),Bounds(0,size//2,size,size))
    return GenerationPlan(size,size,cores,tuple(c.expanded(halo,size,size) for c in cores),'manual')


@pytest.mark.parametrize('padding',['zero_masked','learned'])
@pytest.mark.parametrize('nested',[False,True])
@pytest.mark.parametrize('halo',[0,4])
def test_native_partition_matches_independent_tiled_oracle(tmp_path,padding,nested,halo):
    torch.manual_seed(17)
    model=create_transformer(config(padding),device='cpu',dtype=torch.float32).eval()
    if padding=='learned':
        torch.nn.init.constant_(model.x_pad_token,.02);torch.nn.init.constant_(model.cap_pad_token,.03)
    if nested:restore_topology(model,spec_for(model))
    base=MingPartitionGlobalAdapter(16,24,rank=4,summary_tokens=3)
    extra=MingPartitionGlobalAdapter(16,24,rank=4,summary_tokens=3)
    torch.nn.init.normal_(base.out.weight,std=.02);torch.nn.init.normal_(extra.out.weight,std=.03)
    model._ming_partition_global_payload={'metadata':base.metadata(),'state':base.export_tensors()}
    model._ming_partition_global_generation=[{'module':extra,'strength':.5,'active':True}]
    snapshot={k:v.clone() for k,v in model.state_dict().items()}
    frames=3 if padding=='learned' else 1
    grid=torch.randn(2,16,frames,16,16);t=torch.tensor([.2,.8])
    cap=[torch.randn(5,8),torch.randn(7,8)];direct=[torch.randn(3,24),torch.randn(4,24)]
    refs=[torch.randn(16,2,8,8),torch.randn(16,1,12,8)]
    layout=plan(8,halo);executor=MingPartitionExecutor(model,layout)
    with torch.no_grad():
        expected=torch.empty_like(grid)
        for core,box in zip(layout.cores,layout.inputs):
            b=PartitionBox(box.top,box.left,box.bottom,box.right)
            residual=base(grid,b)+.5*extra(grid,b)
            tile=list(grid[...,box.top*2:box.bottom*2,box.left*2:box.right*2].unbind())
            values=model(tile,t,cap,ref_x=refs,cap_feats_2=direct,return_dict=False,
                target_spatial_position_ids=MingPartitionAdapter().position_ids(8,8,b),target_input_residual=residual)[0]
            y,x=2*(core.top-box.top),2*(core.left-box.left)
            expected[...,core.top*2:core.bottom*2,core.left*2:core.right*2]=torch.stack(values)[...,y:y+core.height*2,x:x+core.width*2]
        actual=torch.stack(executor.predict(list(grid.unbind()),t,cap,ref_hidden_states=refs,encoder_hidden_states_2=direct)[0])
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in snapshot.items())
    assert actual.isfinite().all() and not actual.requires_grad
    executor.close();assert executor.base_global is None
    with pytest.raises(RuntimeError,match='closed'):executor.predict([],t,cap)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
def test_sampler_partition_cfg_grouping_cancel_and_reuse(profile):
    from ming_image_01_sampling_test import make_sampler,condition
    from core.models.ming_image_01.sampling import MingGenerationCancelled
    sampler=make_sampler(profile);conditioning=condition(2)
    args=dict(task='layer-decompose' if profile=='layer_decompose' else 'text-to-image',conditioning=conditioning,
        width=64,height=64,steps=2,cfg=2.,seed=29,requested_layers=2 if profile=='layer_decompose' else 1,output_type='latent',partition_plan=plan(4))
    if profile=='layer_decompose':args['reference_pixels']=torch.randn(2,4,64,64)
    trace=[]
    first=sampler.run(**args,step_callback=lambda i,n,t,x:trace.append(x.clone()))
    assert len(trace)==2 and all(x.isfinite().all() for x in trace)
    frames=3 if profile=='layer_decompose' else 1
    assert len(first)==2 and all(len(group.values)==frames for group in first)
    assert sampler.pipe.transformer.partition_executor is None
    cancelled=[]
    with pytest.raises(MingGenerationCancelled):
        sampler.run(**args,cancelled=lambda:bool(cancelled),step_callback=lambda *args:cancelled.append(True))
    assert sampler.pipe.transformer.partition_executor is None
    assert not sampler.components.transformer._forward_pre_hooks
    second=sampler.run(**args)
    assert all(torch.equal(a,b) for x,y in zip(first,second) for a,b in zip(x.values,y.values))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage',['overlap','missing','outside','noninteger'])
def test_invalid_plan_refused_before_native_forward(damage):
    model=create_transformer(config('zero_masked'),device='cpu',dtype=torch.float32)
    p=plan(8)
    if damage=='overlap':p=GenerationPlan(8,8,(p.cores[0],p.cores[0]),(p.inputs[0],p.inputs[0]),'manual')
    elif damage=='missing':p=GenerationPlan(8,8,p.cores[:1],p.inputs[:1],'manual')
    elif damage=='outside':p=GenerationPlan(8,8,p.cores,(Bounds(0,0,9,4),p.inputs[1]),'manual')
    else:p=GenerationPlan(8,8,p.cores,(Bounds(0.,0,8,4),p.inputs[1]),'manual')
    with pytest.raises(ValueError):MingPartitionExecutor(model,p)
