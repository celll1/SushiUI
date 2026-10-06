"""Residency selects a conductor's whole owner before changing any tensors."""

from types import SimpleNamespace
import pytest
import torch

from core.training.mutable_model_residency import move_with_mutable_conductors


def test_cpu_parking_preserves_parameter_and_master_identity():
    model = torch.nn.Linear(3,3)
    owner = model.weight
    pointer = owner.data_ptr()
    calls = []
    conductor = SimpleNamespace(managed_tensor_ids=lambda:{id(owner)},
        park_device_slots=lambda:calls.append("park"), resume_device_slots=lambda:calls.append("resume"))
    assert move_with_mutable_conductors(model,"cpu",[conductor])
    assert calls == ["park"] and model.weight is owner and owner.data_ptr() == pointer


def test_partial_owner_refused_before_parking_or_move():
    model = torch.nn.Linear(3,3)
    calls = []
    conductor = SimpleNamespace(managed_tensor_ids=lambda:{id(model.weight),-1},
        park_device_slots=lambda:calls.append("park"))
    with pytest.raises(ValueError,match="multiple model"):
        move_with_mutable_conductors(model,"cpu",[conductor])
    assert not calls


def test_other_branch_and_absent_owner_leave_normal_move_to_caller():
    model, other = torch.nn.Linear(3,3), torch.nn.Linear(3,3)
    conductor = SimpleNamespace(managed_tensor_ids=lambda:{id(other.weight)})
    assert not move_with_mutable_conductors(model,"cpu",[conductor])
    assert not move_with_mutable_conductors(None,"cpu",[conductor])
