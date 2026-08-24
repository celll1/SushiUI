"""Shared hook registration for the ring-buffer optimizers' fused backward pass.

Registration is driven by ``optimizer.param_groups``, never by walking a module.

Under the fused backward pass the trainer never calls ``optimizer.step()``
(base_trainer: ``if not self.use_fused_backward and ...``), so a parameter the
optimizer owns but that carries no hook is updated by nothing at all for the
whole run, while the loss keeps falling. Walking a module cannot give that
guarantee: BaseTrainer passes ``_fused_backward_target_module()``, which is the
transformer or the U-Net, while ``setup_optimizer`` also appends text-encoder and
vision-encoder groups to the same optimizer -- those parameters got no hooks and
no step(). The optimizer's own param_groups are the set that must be updated, so
they are the set that gets hooks.

The module argument is kept for what it can still say: parameter NAMES for the
error messages, and the check in the other direction -- a trainable parameter of
the module that is in no param_group (3a7c9560), which param_groups alone cannot
see.
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch.nn as nn


def _describe(param: nn.Parameter, names: Dict[int, str]) -> str:
    name = names.get(id(param))
    return name if name else f"<unnamed {tuple(param.shape)} {param.dtype}>"


def register_fused_backward_hooks(
    optimizer,
    module: Optional[nn.Module],
    function_name: str,
    make_hook: Callable[[nn.Parameter, dict], Callable],
) -> Tuple[int, List[str]]:
    """Register a per-parameter hook on every parameter the optimizer owns.

    Args:
        optimizer: the ring-buffer optimizer whose param_groups drive registration
        module: optional module, used for parameter names and the orphan check
        function_name: public entry point's name, for the error messages
        make_hook: ``(param, group) -> hook`` factory

    Returns:
        ``(hooked_count, frozen_descriptions)``
    """
    names: Dict[int, str] = {}
    if module is not None:
        names = {id(p): n for n, p in module.named_parameters()}

    # One hook per parameter, even if a parameter appears in two groups (which
    # step() would update twice). The group it resolves to is the last one that
    # lists it, which is what the previous per-parameter id->group map did.
    param_to_group: Dict[int, dict] = {}
    order: List[nn.Parameter] = []
    for group in optimizer.param_groups:
        for param in group['params']:
            if id(param) not in param_to_group:
                order.append(param)
            param_to_group[id(param)] = group
    owned: List[Tuple[nn.Parameter, dict]] = [(p, param_to_group[id(p)]) for p in order]
    seen = set(param_to_group)

    # The hooks implement the 8-bit CUDA update only. A group with use_8bit=False
    # gets FP32 state from _init_param_state (state['is_8bit'] = False), which the
    # hook cannot update, and there is no later step() to do it instead -- the old
    # `return  # Skip FP32 params (updated in optimizer.step())` was that promise,
    # and it was false under fused backward. Refused here, before the run.
    fp32_groups = [_describe(p, names) for p, g in owned if not g.get('use_8bit', True)]
    if fp32_groups:
        raise RuntimeError(
            f"{len(fp32_groups)} parameter(s) passed to {function_name} are in a param_group "
            f"with use_8bit=False, e.g. {fp32_groups[:3]}. The per-parameter fused-backward "
            f"hooks perform the 8-bit CUDA update and cannot update FP32 optimizer state, and "
            f"under the fused backward pass optimizer.step() is never called, so those "
            f"parameters would never be updated. Use use_8bit=True for the fused-backward "
            f"(Block Swap) path, or call optimizer.step() instead of registering these hooks."
        )

    # The other direction, from 3a7c9560: a trainable parameter of the module that
    # the optimizer does not own. param_groups-driven registration cannot see it,
    # and nothing would ever update it.
    if module is not None:
        orphans = [name for name, p in module.named_parameters()
                   if p.requires_grad and id(p) not in seen]
        if orphans:
            raise RuntimeError(
                f"{len(orphans)} trainable parameter(s) of the module passed to "
                f"{function_name} are in no param_group of this optimizer, "
                f"e.g. {orphans[:3]}. Under the fused backward pass optimizer.step() is "
                f"never called, so nothing would ever update them. Add them to the "
                f"optimizer, or freeze them (requires_grad=False)."
            )

    hooked = 0
    frozen: List[str] = []
    for param, group in owned:
        if not param.requires_grad:
            # No gradient is ever accumulated for it, so neither a hook nor step()
            # would move it; not the silent-skip failure. Reported, not refused,
            # because callers do hand whole encoders to the optimizer.
            frozen.append(_describe(param, names))
            continue
        param.register_post_accumulate_grad_hook(make_hook(param, group))
        hooked += 1

    uncovered = [_describe(p, names) for p, _ in owned
                 if p.requires_grad and not getattr(p, '_post_accumulate_grad_hooks', None)]
    if uncovered:
        raise RuntimeError(
            f"{function_name}: {len(uncovered)} parameter(s) the optimizer owns carry no "
            f"post-accumulate-grad hook after registration, e.g. {uncovered[:3]}. Under the "
            f"fused backward pass they would never be updated."
        )

    return hooked, frozen
