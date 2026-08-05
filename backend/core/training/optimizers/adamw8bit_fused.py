"""
AdamW8bit optimizer with fused backward pass

This implementation allows parameter updates to happen immediately after gradients
are computed (via register_post_accumulate_grad_hook), enabling:
- Block Swap compatibility: parameters are updated while still on GPU
- Memory efficiency: gradients are cleared immediately after use
- Works with bitsandbytes AdamW8bit optimizer

Requirements:
- PyTorch 2.1+ (for register_post_accumulate_grad_hook)
- bitsandbytes (for AdamW8bit optimizer)
"""

import torch

from .stochastic_rounding import (
    NATIVE_ATTR,
    Fp32ScratchPool,
    copy_stochastic_bf16,
    should_use_stochastic_rounding,
)


@torch.no_grad()
def adamw8bit_step_param(self, p, group):
    """
    Update a single parameter immediately after its gradient is computed.

    This method is called by the post-accumulate-grad hook, allowing parameter
    updates while the parameter is still on GPU (before Block Swap moves it to CPU).

    Args:
        p: Parameter to update
        group: Optimizer parameter group

    Note:
        This implementation follows bitsandbytes AdamW8bit logic but operates
        on a single parameter instead of all parameters.
    """
    if p.grad is None:
        return

    # Get state for this parameter
    state = self.state[p]

    # State initialization
    if len(state) == 0:
        state['step'] = 0
        # Exponential moving average of gradient values
        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        # Exponential moving average of squared gradient values
        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
    beta1, beta2 = group['betas']

    state['step'] += 1
    bias_correction1 = 1 - beta1 ** state['step']
    bias_correction2 = 1 - beta2 ** state['step']

    # Gradient
    grad = p.grad

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # Compute step
    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
    step_size = group['lr'] / bias_correction1

    # Both writes below land in the parameter's own storage. For a BF16
    # parameter that is round-to-nearest, which discards every update below half
    # a ULP -- deterministically, so such a weight is frozen for the whole run
    # rather than slow (see stochastic_rounding.py). When stochastic rounding is
    # requested, apply the update to a pooled FP32 image of the parameter and
    # round that back instead.
    #
    # The optimizer STATE is deliberately still allocated from ``p`` above, so
    # it keeps the parameter's dtype and this costs no persistent memory. That
    # leaves exp_avg_sq's own accumulation in BF16 (with beta2=0.999 its
    # per-step relative change is 1e-3, just under BF16's 2^-9 half-ULP), which
    # biases the denominator but not the conclusion here: the step size stays of
    # order lr, and it is the parameter write that freezes the weight.
    use_sr = should_use_stochastic_rounding(getattr(self, "stochastic_rounding", False), p)
    if use_sr:
        pool = getattr(self, "_sr_pool", None)
        if pool is None:
            pool = Fp32ScratchPool()
            self._sr_pool = pool
        target = pool.copy_of("master", p)
    else:
        target = p

    # Weight decay (AdamW style - decoupled)
    if group['weight_decay'] != 0:
        target.mul_(1 - group['lr'] * group['weight_decay'])

    # Update parameters
    target.addcdiv_(exp_avg, denom, value=-step_size)

    if use_sr:
        copy_stochastic_bf16(p.data, target)


# Tells attach_stochastic_rounding() not to interpose on this function: it
# applies stochastic rounding itself, above, and doing it twice would also make
# the optimizer state FP32 (this implementation allocates state with
# ``zeros_like(p)``, so it would follow an FP32 view of the parameter).
setattr(adamw8bit_step_param, NATIVE_ATTR, True)


@torch.no_grad()
def adamw8bit_step(self, closure=None):
    """
    Performs a single optimization step (fallback for non-fused mode).

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            # self.step_param, not the module-level function: see the same note
            # in adafactor_fused.adafactor_step. Any interposition on the
            # per-parameter update rebinds the instance attribute, and calling
            # the module-level function bypasses it.
            self.step_param(p, group)

    return loss


def patch_adamw8bit_fused(optimizer, stochastic_rounding: bool = False):
    """
    Patch AdamW8bit optimizer to support per-parameter updates.

    Adds:
    - step_param(): Update single parameter (called by hook)
    - step(): Fallback batch update (if hooks not registered)

    Args:
        optimizer: AdamW8bit optimizer instance to patch
        stochastic_rounding: Write BF16 parameters with stochastic rounding
            instead of round-to-nearest. Only affects BF16 parameters.

    Example:
        >>> import bitsandbytes as bnb
        >>> from core.training.optimizers.adamw8bit_fused import patch_adamw8bit_fused
        >>>
        >>> optimizer = bnb.optim.AdamW8bit(model.parameters())
        >>> patch_adamw8bit_fused(optimizer)
        >>>
        >>> # Register hooks
        >>> for param in model.parameters():
        >>>     if param.requires_grad:
        >>>         param.register_post_accumulate_grad_hook(
        >>>             lambda tensor: optimizer.step_param(tensor, optimizer.param_groups[0])
        >>>         )
    """
    optimizer.stochastic_rounding = bool(stochastic_rounding)
    optimizer.step_param = adamw8bit_step_param.__get__(optimizer)
    # Don't replace step() - keep bitsandbytes implementation for fallback
    # optimizer.step = adamw8bit_step.__get__(optimizer)
    print(f"[AdamW8bitFused] Optimizer patched with fused backward support"
          f"{' (stochastic rounding)' if optimizer.stochastic_rounding else ''}")
