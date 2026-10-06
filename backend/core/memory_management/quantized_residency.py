"""Device ownership for bitsandbytes state outside registered model tensors."""
import torch


def _states(module):
    seen=set()
    pending=[]
    for child in module.modules():
        pending.append(getattr(child,"quant_state",None))
        pending.extend(getattr(parameter,"quant_state",None) for parameter in child.parameters(recurse=False))
    while pending:
        state=pending.pop()
        if state is None or id(state) in seen:
            continue
        seen.add(id(state))
        yield state
        pending.append(getattr(state,"state2",None))


def quant_state_devices(modules):
    return [(state,name,tensor.device) for module in modules for state in _states(module)
            for name,tensor in vars(state).items() if isinstance(tensor,torch.Tensor)]


def move_quant_states(module,device):
    for state in _states(module):
        for name,tensor in list(vars(state).items()):
            if isinstance(tensor,torch.Tensor):
                setattr(state,name,tensor.to(device))


def restore_quant_states(placements):
    for state,name,device in placements:
        setattr(state,name,getattr(state,name).to(device))
