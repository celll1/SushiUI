from copy import deepcopy

import pytest
import torch
from torch import nn

from hybrid_rewiring_test import declaration
from core.models.anima.anima_models import Block
from core.models.anima.anima_attention import AttentionParams
from core.rewiring.hybrid_modules import HybridModules
from core.rewiring.hybrid_spec import build_hybrid_spec, legacy_topology
from core.experiments.anima_projected_block import ProjectedAnimaBlock
from core.experiments.anima_projected_stage import projected_stage_probe, native_fingerprint
from core.rewiring.initialization import frozen_native_teacher


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = declaration()
        spec = build_hybrid_spec(legacy_topology(base), stream_widths={'main':16},
            boundaries=base['boundaries'], refiners=base['refiners'], skips=base['skips'])
        self.blocks = nn.ModuleList([Block(16,12,2,use_adaln_lora=True,adaln_lora_dim=8) for _ in range(7)])
        self.rewiring_modules = HybridModules(spec)
        self._residual_unet_spec = spec
        self.fail_native = False
        with torch.no_grad():
            for name,param in self.named_parameters():
                if 'adaln_modulation' in name:
                    param.normal_(std=.1)

    def forward(self, x, embedding, context, adaln):
        attention = AttentionParams(attn_mode='native')
        if self._residual_unet_spec is None:
            if self.fail_native:
                raise RuntimeError('native sentinel')
            for block in self.blocks:
                x = block(x,embedding,context,attention,adaln_lora_B_T_3D=adaln)
            return x
        def execute(index,value,grid):
            h,w = grid
            return self.blocks[index](value.reshape(x.shape[0],1,h,w,16),embedding,context,
                attention,adaln_lora_B_T_3D=adaln).reshape(x.shape[0],h*w,16)
        return self.rewiring_modules.forward_stream(x.reshape(x.shape[0],96,16),'main',8,12,0,
            (8,12),execute,lambda ctx,value,h,w,prefix:(h,w)).reshape_as(x)


def inputs():
    return (torch.randn(1,1,8,12,16),torch.randn(1,1,16),torch.randn(1,4,12),torch.randn(1,1,48))


def test_teacher_native_parity_student_gradients_and_full_cleanup():
    torch.manual_seed(63)
    model = ToyModel(); original_blocks = list(model.blocks)
    model.blocks[0].eval()
    flags = {id(p):p.requires_grad for p in model.parameters()}
    modes = {id(m):m.training for m in model.modules()}
    before = native_fingerprint(model)
    topology = model._residual_unet_spec
    data = inputs()
    model._residual_unet_spec = None
    with torch.no_grad(): expected = model(*data)
    model._residual_unet_spec = topology
    rng = torch.get_rng_state().clone()
    with projected_stage_probe(model,{1:[0],2:[1]}) as teacher:
        assert isinstance(model.blocks[1],ProjectedAnimaBlock)
        model.blocks[1].self_attn.block_idx = 1
        model.blocks[1].self_attn._style_ctx = None
        assert model.blocks[1].student.self_attn.block_idx == 1
        assert len(teacher.parameters)==len({id(p) for p in teacher.parameters})
        assert all(not p.requires_grad for block in original_blocks for p in block.parameters())
        target = teacher(*data)
        assert not target.requires_grad
        torch.testing.assert_close(target,expected,rtol=0,atol=0)
        optimizer = torch.optim.SGD(teacher.parameters,lr=.01)
        prediction = model(*data)
        assert prediction.requires_grad
        prediction.square().mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.blocks[1].parameters())
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        teacher(*data)
        torch.rand(4)
    assert list(model.blocks)==original_blocks and model._residual_unet_spec is topology
    assert native_fingerprint(model)==before and torch.equal(rng,torch.get_rng_state())
    assert all(p.requires_grad==flags[id(p)] and p.grad is None for p in model.parameters())
    assert all(m.training==modes[id(m)] for m in model.modules())
    with pytest.raises(RuntimeError,match='outside'):
        teacher(*data)


def test_payload_roundtrip_and_invalid_payload_is_nonmutating():
    model = ToyModel()
    with projected_stage_probe(model,{1:[0]}) as teacher:
        payload = teacher.payload()
        with torch.no_grad():model.blocks[1].decoder.add_(.1)
        teacher.load_payload(payload)
        torch.testing.assert_close(model.blocks[1].decoder,payload['blocks']['1']['decoder'],rtol=0,atol=0)
        for corrupt in ('identity','nan','coordinates','manifest'):
            bad = deepcopy(payload)
            if corrupt=='identity':bad['native_sha256']='0'*64
            elif corrupt=='nan':bad['blocks']['1']['decoder'][0,0]=float('nan')
            elif corrupt=='coordinates':bad['blocks']['1']['channels'][0]=3
            else:bad['manifests']['1']['heads']=[1]
            saved = teacher.payload()
            with pytest.raises(ValueError):teacher.load_payload(bad)
            after = teacher.payload()
            for name,value in saved['blocks']['1'].items():
                torch.testing.assert_close(value,after['blocks']['1'][name],rtol=0,atol=0)
            for name,value in saved['structural'].items():
                torch.testing.assert_close(value,after['structural'][name],rtol=0,atol=0)
    with projected_stage_probe(model,{1:[0]}) as restored:
        restored.load_payload(payload)
    with pytest.raises(RuntimeError):teacher.payload()


def test_teacher_failure_restores_active_student_and_outer_failure_cleans():
    model = ToyModel(); original = list(model.blocks)
    with pytest.raises(RuntimeError,match='outer sentinel'):
        with projected_stage_probe(model,{1:[0]}) as teacher:
            student = model.blocks[1]
            model.fail_native = True
            with pytest.raises(RuntimeError,match='native sentinel'):teacher(*inputs())
            assert model.blocks[1] is student and model._residual_unet_spec==teacher.declaration
            with pytest.raises(ValueError,match='exclusively'):
                with projected_stage_probe(model,{2:[0]}):pass
            with pytest.raises(ValueError,match='already active'):
                with frozen_native_teacher(model):pass
            raise RuntimeError('outer sentinel')
    assert (list(model.blocks)==original and not hasattr(model,'_projected_stage_owner')
            and not hasattr(model,'_hybrid_teacher_owner'))
    with projected_stage_probe(model,{1:[0]}):pass


def test_partial_install_failure_restores_all_flags_and_blocks(monkeypatch):
    model = ToyModel(); original = list(model.blocks)
    flags = [(p,p.requires_grad) for p in model.parameters()]
    setter = nn.ModuleList.__setitem__
    def fail(self,index,module):
        if self is model.blocks and index==2 and isinstance(module,ProjectedAnimaBlock):
            raise RuntimeError('install sentinel')
        return setter(self,index,module)
    with monkeypatch.context() as patch:
        patch.setattr(nn.ModuleList,'__setitem__',fail)
        with pytest.raises(RuntimeError,match='install sentinel'):
            with projected_stage_probe(model,{1:[0],2:[0]}):pass
    assert list(model.blocks)==original and all(p.requires_grad==flag for p,flag in flags)
    with projected_stage_probe(model,{1:[0]}):pass


@pytest.mark.parametrize('mapping',[{}, {0:[0]}, {7:[0]}, {True:[0]}, {1:[2]}])
def test_invalid_stage_mapping(mapping):
    model = ToyModel(); original = list(model.blocks)
    with pytest.raises(ValueError):
        with projected_stage_probe(model,mapping):pass
    assert list(model.blocks)==original and not hasattr(model,'_projected_stage_owner')


def test_existing_gradients_refused():
    model = ToyModel();next(model.parameters()).grad=torch.zeros_like(next(model.parameters()))
    with pytest.raises(ValueError,match='existing training gradients'):
        with projected_stage_probe(model,{1:[0]}):pass
