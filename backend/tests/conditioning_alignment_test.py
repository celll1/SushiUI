import pytest
import torch

from core.experiments.conditioning_alignment import conditioning_losses


def loss(sc,su,tc,tu):
    return conditioning_losses(sc,su,tc,tu,floor_fraction=.01)


def test_analytic_delta_error_and_common_offset():
    tc,tu=torch.tensor([[3.,3.]]),torch.tensor([[1.,1.]])
    sc,su=torch.tensor([[4.,4.]]),tu.clone()
    result=loss(sc,su,tc,tu)
    assert result['anchor']==.5 and result['contrast']==.25
    shifted=loss(sc+10,su+10,tc,tu)
    torch.testing.assert_close(shifted['contrast'],result['contrast'])
    assert shifted['anchor']>result['anchor']


def test_only_students_receive_gradients_and_both_branches_learn():
    sc,su,tc,tu=[torch.randn(2,4,3,requires_grad=True) for _ in range(4)]
    result=loss(sc,su,tc,tu)
    (result['anchor']+.1*result['contrast']).backward()
    for value in (sc,su):
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    assert tc.grad is None and tu.grad is None


def test_batch_examples_are_normalized_independently():
    tc=torch.tensor([[2.,2.],[20.,20.]])
    tu=torch.zeros_like(tc)
    result=loss(tc*1.5,tu,tc,tu)
    torch.testing.assert_close(result['contrast'],torch.tensor(.25))


def test_zero_and_near_zero_delta_have_finite_values_and_gradients():
    for common in (0.,10.):
        sc=torch.zeros(1,4,requires_grad=True)
        su=torch.ones(1,4,requires_grad=True)
        teacher=torch.full((1,4),common)
        result=loss(sc,su,teacher,teacher)
        result['contrast'].backward()
        assert torch.isfinite(result['contrast'])
        assert torch.isfinite(sc.grad).all() and torch.isfinite(su.grad).all()
        assert result['denominator']>=torch.finfo(torch.float32).eps


def test_bfloat_inputs_accumulate_in_float32():
    value=torch.ones(1,4,dtype=torch.bfloat16)
    result=loss(value,value,value,value)
    assert result['anchor'].dtype==result['contrast'].dtype==torch.float32
    assert result['anchor']==result['contrast']==0


def test_invalid_layouts_nonfinite_and_floor_are_refused():
    value=torch.ones(1,4)
    for bad in (torch.ones(2,4),torch.ones(1,3),torch.ones(1,4,dtype=torch.int64),torch.full((1,4),float('nan'))):
        with pytest.raises(ValueError):loss(value,bad,value,value)
    for floor in (True,0.,-1.,2.,float('inf')):
        with pytest.raises(ValueError):conditioning_losses(value,value,value,value,floor_fraction=floor)
    with pytest.raises(ValueError):loss(torch.empty(0,4),torch.empty(0,4),torch.empty(0,4),torch.empty(0,4))
