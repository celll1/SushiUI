"""
Adafactor optimizer with fused backward pass

This implementation allows parameter updates to happen immediately after gradients
are computed (via register_post_accumulate_grad_hook), enabling:
- Block Swap compatibility: parameters are updated while still on GPU
- Memory efficiency: gradients are cleared immediately after use
- Works with Gradient Checkpointing without device mismatch errors

Based on sd-scripts implementation:
https://github.com/kohya-ss/sd-scripts/blob/main/library/adafactor_fused.py

Requirements:
- PyTorch 2.1+ (for register_post_accumulate_grad_hook)
- transformers (for Adafactor optimizer)
"""

import math
import torch
from transformers import Adafactor

from .update_census import record_param_update


@torch.no_grad()
def adafactor_step_param(self, p, group):
    """
    Update a single parameter immediately after its gradient is computed.

    This method is called by the post-accumulate-grad hook, allowing parameter
    updates while the parameter is still on GPU (before Block Swap moves it to CPU).

    Args:
        p: Parameter to update
        group: Optimizer parameter group
    """
    if p.grad is None:
        return

    grad = p.grad
    if grad.dtype in {torch.float16, torch.bfloat16}:
        grad = grad.float()
    if grad.is_sparse:
        raise RuntimeError("Adafactor does not support sparse gradients.")

    state = self.state[p]
    grad_shape = grad.shape

    factored, use_first_moment = Adafactor._get_options(group, grad_shape)

    # State Initialization
    if len(state) == 0:
        state["step"] = 0

        if use_first_moment:
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(grad)
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
            state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
        else:
            state["exp_avg_sq"] = torch.zeros_like(grad)

        state["RMS"] = 0
    else:
        if use_first_moment:
            state["exp_avg"] = state["exp_avg"].to(grad)
        if factored:
            state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
            state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
        else:
            state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

    p_data_fp32 = p
    if p.dtype in {torch.float16, torch.bfloat16}:
        p_data_fp32 = p_data_fp32.float()

    state["step"] += 1
    state["RMS"] = Adafactor._rms(p_data_fp32)
    lr = Adafactor._get_lr(group, state)

    beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
    update = (grad ** 2) + group["eps"][0]
    if factored:
        exp_avg_sq_row = state["exp_avg_sq_row"]
        exp_avg_sq_col = state["exp_avg_sq_col"]

        exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
        exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))

        # Approximation of exponential moving average of square of gradient
        update = Adafactor._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
        update.mul_(grad)
    else:
        exp_avg_sq = state["exp_avg_sq"]

        exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
        update = exp_avg_sq.rsqrt().mul_(grad)

    update.div_((Adafactor._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
    update.mul_(lr)

    if use_first_moment:
        exp_avg = state["exp_avg"]
        exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
        update = exp_avg

    if group["weight_decay"] != 0:
        p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

    p_data_fp32.add_(-update)

    if p.dtype in {torch.float16, torch.bfloat16}:
        p.copy_(p_data_fp32)

    # Last, and only on the path that actually wrote: the census is armed by
    # setup_update_census for EVERY fused-backward optimizer, but only the two
    # ring-buffer ones used to call this, so an armed census reported every
    # parameter missing on a correct Adafactor run -- the one optimizer
    # SenseNova's full fine-tune allows. Recorded here rather than in the hook
    # so the early return above (no gradient) stays uncounted.
    record_param_update(self, p)


@torch.no_grad()
def adafactor_step(self, closure=None):
    """
    Performs a single optimization step (fallback for non-fused mode).

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        for p in group["params"]:
            # self.step_param, NOT the module-level function: anything that
            # interposes on the per-parameter update -- stochastic rounding for
            # BF16 parameters -- does so by rebinding the INSTANCE attribute.
            # Calling adafactor_step_param(self, p, group) here bypassed every
            # such interposition, so a run that went through step() (i.e. any
            # Adafactor run without Block Swap, and any run with fused optimizer
            # groups) silently got round-to-nearest while the setup log said
            # stochastic rounding was attached.
            self.step_param(p, group)

    return loss


def patch_adafactor_fused(optimizer: Adafactor):
    """
    Patch Adafactor optimizer to support per-parameter updates.

    Adds:
    - step_param(): Update single parameter (called by hook)
    - step(): Fallback batch update (if hooks not registered)

    Args:
        optimizer: Adafactor optimizer instance to patch

    Example:
        >>> from transformers import Adafactor
        >>> from core.training.optimizers.adafactor_fused import patch_adafactor_fused
        >>>
        >>> optimizer = Adafactor(model.parameters())
        >>> patch_adafactor_fused(optimizer)
        >>>
        >>> # Register hooks
        >>> for param in model.parameters():
        >>>     if param.requires_grad:
        >>>         param.register_post_accumulate_grad_hook(
        >>>             lambda tensor: optimizer.step_param(tensor, optimizer.param_groups[0])
        >>>         )
    """
    optimizer.step_param = adafactor_step_param.__get__(optimizer)
    optimizer.step = adafactor_step.__get__(optimizer)
    print("[AdafactorFused] Optimizer patched with fused backward support")
