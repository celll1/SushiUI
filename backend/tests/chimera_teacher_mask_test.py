"""Teacher padding cannot improve alignment losses or held-out cosine."""
from types import SimpleNamespace

import pytest
import torch

from core.training.ops.sensenova_sdxl_chimera_ops import bridge_alignment_loss, collate_aux


def _aux(student, teacher, mask):
    return dict(alignment_hidden_states=student,teacher_hidden=teacher,
        pooled_text_embeds=torch.ones(student.shape[0],3),teacher_pooled=torch.ones(student.shape[0],3),
        teacher_attention_mask=mask)


def test_padding_errors_are_excluded_and_receive_no_gradient():
    teacher = torch.tensor([[[1.,0.],[0.,1.],[100.,100.]]])
    student = torch.tensor([[[1.,0.],[0.,1.],[-100.,-100.]]],requires_grad=True)
    mask = torch.tensor([[True,True,False]])
    owner = SimpleNamespace(config=dict(chimera_clip_hidden_weight=1.,chimera_clip_pooled_weight=1.))
    loss,metrics = bridge_alignment_loss(owner,student,_aux(student,teacher,mask))
    assert loss == 0 and metrics['hidden_cosine'] == 1
    loss.backward()
    assert torch.equal(student.grad,torch.zeros_like(student))


def test_equal_padding_cannot_hide_wrong_actual_tokens_and_valid_gradient_trains():
    teacher = torch.ones(2,77,4)
    student = teacher.clone()
    student[:,:2] *= -1
    student.requires_grad_(True)
    mask = torch.zeros(2,77,dtype=torch.bool)
    mask[:,:2] = True
    owner = SimpleNamespace(config=dict(chimera_clip_hidden_weight=1.,chimera_clip_pooled_weight=1.))
    loss,metrics = bridge_alignment_loss(owner,student,_aux(student,teacher,mask))
    assert metrics['hidden_cosine'] == -1
    assert float(loss.detach()) == pytest.approx(1.)
    loss.backward()
    assert torch.equal(student.grad[:,2:],torch.zeros_like(student.grad[:,2:]))
    # Change a valid token's norm so its RMS gradient is nonzero.
    moved = student.detach().clone().requires_grad_(True)
    moved.data[:,:2] *= .5
    bridge_alignment_loss(owner,moved,_aux(moved,teacher,mask))[0].backward()
    assert moved.grad[:,:2].abs().sum() > 0


@pytest.mark.parametrize('mask',[torch.zeros(1,3,dtype=torch.bool),torch.ones(1,2,dtype=torch.bool),torch.ones(1,3)])
def test_invalid_teacher_mask_refuses(mask):
    value = torch.ones(1,3,2)
    owner = SimpleNamespace(config=dict(chimera_clip_hidden_weight=1.,chimera_clip_pooled_weight=1.))
    with pytest.raises(ValueError,match='teacher mask'):
        bridge_alignment_loss(owner,value,_aux(value,value,mask))


def test_teacher_masks_collate_as_fixed_teacher_rows():
    items = [dict(teacher_attention_mask=torch.tensor([[True,True,False]]),
                  context_attention_mask=torch.ones(1,2,dtype=torch.bool)),
             dict(teacher_attention_mask=torch.tensor([[True,False,False]]),
                  context_attention_mask=torch.ones(1,4,dtype=torch.bool))]
    actual = collate_aux(items)['teacher_attention_mask']
    assert torch.equal(actual,torch.cat([item['teacher_attention_mask'] for item in items]))


def test_prompt_encoder_passes_union_of_both_teacher_tokenizer_masks(monkeypatch):
    from core.training.ops import sensenova_sdxl_chimera_ops as ops
    def tokenizer(mask):
        def encode(prompt, **kwargs):
            assert prompt == 'A blue cup.' and kwargs['max_length'] == 6
            return dict(attention_mask=torch.tensor([mask]))
        return encode
    teacher = SimpleNamespace(tokenizer=tokenizer([1,1,0,0,0,0]),
        tokenizer_2=tokenizer([1,1,1,0,0,0]))
    teacher.to = lambda device: teacher
    teacher.encode_prompt = lambda **kwargs: (torch.ones(1,6,4),None,torch.ones(1,3),None)
    bridge_output = SimpleNamespace(encoder_hidden_states=torch.ones(1,5,4),
        pooled_text_embeds=torch.ones(1,3),context_positions=torch.zeros(1,5,3),
        attention_mask=torch.ones(1,5,dtype=torch.bool),alignment_hidden_states=torch.ones(1,6,4))
    monkeypatch.setattr(ops,'encode_chimera_conditioning',lambda *args,**kwargs: bridge_output)
    trainer = SimpleNamespace(device='cpu',weight_dtype=torch.float32,condition_bridge=torch.nn.Linear(2,2),
        chimera_understanding=torch.nn.Linear(2,2),tokenizer=object(),chimera_teacher=teacher,
        config=dict(chimera_training_stage='bridge_align'))
    _,aux = ops.encode_prompt(trainer,'A blue cup.',requires_grad=True)
    assert torch.equal(aux['teacher_attention_mask'],torch.tensor([[True,True,True,False,False,False]]))
