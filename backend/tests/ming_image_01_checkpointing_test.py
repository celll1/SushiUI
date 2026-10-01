"""Native main-stream checkpoint counts and reversible partition overrides."""
from copy import deepcopy
import pytest
import torch
from core.models.ming_image_01.checkpointing import configure,block_limit,partition_scope
from core.models.ming_image_01.model_definition import create_transformer
from core.models.ming_image_01.residual_unet import restore_topology
from ming_image_01_rewiring_test import config,spec_for


@pytest.mark.parametrize('rewired',[False,True])
@pytest.mark.parametrize('limit',[0,2,6])
def test_main_checkpoint_count_and_gradients_match_uncheckpointed(rewired,limit):
    torch.manual_seed(61)
    original=create_transformer(config('learned'),device='cpu',dtype=torch.float32)
    if rewired:restore_topology(original,spec_for(original))
    current=deepcopy(original);current.enable_gradient_checkpointing()
    configure(current,{'dit_gradient_checkpointing_blocks':limit})
    calls=[];real=current._gradient_checkpointing_func
    def record(block,*args,**kwargs):
        if any(block is value for value in current.layers):calls.append(next(i for i,value in enumerate(current.layers) if block is value))
        return real(block,*args,**kwargs)
    current._gradient_checkpointing_func=record
    x=torch.randn(16,3,16,16);q=torch.randn(5,8);c=torch.randn(3,24)
    outputs=[]
    for model in (original,current):
        value=model([x],torch.tensor([.4]),[q],cap_feats_2=[c]).sample[0]
        value.square().mean().backward();outputs.append(value)
    assert calls==list(range(limit))
    torch.testing.assert_close(*outputs)
    for (_,a),(_,b) in zip(original.named_parameters(),current.named_parameters()):
        assert (a.grad is None)==(b.grad is None)
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('present',[False,True])
def test_partition_override_restores_after_exception(present):
    model=create_transformer(config('zero_masked'),device='cpu',dtype=torch.float32)
    if present:configure(model,{'dit_gradient_checkpointing_blocks':4})
    with pytest.raises(RuntimeError,match='deliberate'):
        with partition_scope(model,{'dit_partition_gradient_checkpointing_blocks':1}):
            assert model._training_gradient_checkpointing_blocks==1
            raise RuntimeError('deliberate')
    assert getattr(model,'_training_gradient_checkpointing_blocks',None)==(4 if present else None)


@pytest.mark.parametrize('value',[True,-1,7,1.5])
def test_checkpoint_limit_is_strict_and_bounded(value):
    with pytest.raises(ValueError,match='integer between'):
        block_limit({'dit_partition_gradient_checkpointing_blocks':value},6,partition=True)
