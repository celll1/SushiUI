"""Budget persistent Full AdamW state before allocating training activations."""
import torch


def initialize_full_adamw_state(trainer):
    if not getattr(trainer, "trains_base_weights", False) or not isinstance(trainer.optimizer, torch.optim.AdamW):
        return
    optimizer = trainer.optimizer
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not parameter.requires_grad or optimizer.state[parameter]:
                continue
            state = optimizer.state[parameter]
            scalar_dtype = torch.float64 if torch.get_default_dtype() == torch.float64 and not group["fused"] else torch.float32
            step_device = parameter.device if group["capturable"] or group["fused"] else "cpu"
            state["step"] = torch.zeros((), dtype=scalar_dtype, device=step_device)
            state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            if group["amsgrad"]:
                state["max_exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)


def pending_gradient_bytes(trainer):
    if not getattr(trainer, "trains_base_weights", False) or getattr(trainer, "use_fused_backward", False) or getattr(trainer, "fused_optimizer_groups", None) is not None:
        return 0
    seen, total = set(), 0
    for group in trainer.optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in seen and parameter.requires_grad and parameter.grad is None and parameter.device.type == "cuda":
                total += parameter.numel() * parameter.element_size()
            seen.add(id(parameter))
    return total
