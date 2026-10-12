"""A bias-free Linear whose selected rows or columns are a separate Parameter.

Used to give latent-IO weights that start at zero their own optimizer group while
checkpoints keep the original merged ``weight`` key and shape.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SplitLinear(nn.Module):
    def __init__(self, linear: nn.Linear, fresh, *, axis: int):
        super().__init__()
        if type(linear) is not nn.Linear or linear.bias is not None or axis not in (0, 1):
            raise ValueError("SplitLinear wraps a bias-free nn.Linear along rows (0) or columns (1)")
        size = linear.weight.shape[axis]
        fresh = torch.as_tensor(sorted(set(int(i) for i in fresh)), dtype=torch.long)
        if not len(fresh) or fresh[0] < 0 or fresh[-1] >= size or len(fresh) == size:
            raise ValueError("SplitLinear needs a proper non-empty subset of indices")
        mask = torch.zeros(size, dtype=torch.bool)
        mask[fresh] = True
        kept = torch.nonzero(~mask).flatten()
        order = torch.cat([kept, fresh])
        self.axis, self.in_features, self.out_features = axis, linear.in_features, linear.out_features
        self.register_buffer("_kept", kept, persistent=False)
        self.register_buffer("_fresh", fresh, persistent=False)
        self.register_buffer("_inverse", torch.argsort(order), persistent=False)
        weight = linear.weight.detach()
        self.weight_kept = nn.Parameter(weight.index_select(axis, kept).clone(),
                                        requires_grad=linear.weight.requires_grad)
        self.weight_fresh = nn.Parameter(weight.index_select(axis, fresh).clone(),
                                         requires_grad=linear.weight.requires_grad)
        self._register_state_dict_hook(SplitLinear._merge_state)
        self._register_load_state_dict_pre_hook(self._split_state)

    @property
    def weight(self):
        return torch.cat([self.weight_kept, self.weight_fresh], self.axis).index_select(self.axis, self._inverse)

    def forward(self, value):
        return F.linear(value, self.weight)

    @staticmethod
    def _merge_state(module, state, prefix, local_metadata):
        kept, fresh = state.pop(prefix+"weight_kept"), state.pop(prefix+"weight_fresh")
        state[prefix+"weight"] = torch.cat([kept, fresh], module.axis).index_select(
            module.axis, module._inverse.to(kept.device))
        return state

    def _split_state(self, state, prefix, *args):
        merged = state.pop(prefix+"weight", None)
        if merged is not None:
            state[prefix+"weight_kept"] = merged.index_select(self.axis, self._kept.to(merged.device))
            state[prefix+"weight_fresh"] = merged.index_select(self.axis, self._fresh.to(merged.device))

    def merged(self) -> nn.Linear:
        linear = nn.Linear(self.in_features, self.out_features, bias=False,
                           device=self.weight_kept.device, dtype=self.weight_kept.dtype)
        with torch.no_grad():
            linear.weight.copy_(self.weight)
        linear.weight.requires_grad_(self.weight_kept.requires_grad)
        return linear

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, fresh={len(self._fresh)} along axis {self.axis}"
