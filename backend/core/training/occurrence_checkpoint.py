"""Named pending gradients for checkpoints inside an exposure GAS window."""

import os
from pathlib import Path

import torch


def named_parameters(trainer, optimizers):
    names = trainer._build_ema_param_name_map()
    result = {}
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) not in names:
                    raise ValueError("Exposure resume requires named optimizer parameters")
                if names[id(parameter)] in result and result[names[id(parameter)]] is not parameter:
                    raise ValueError("Exposure parameter names must be unique")
                result[names[id(parameter)]] = parameter
    return result


def save_pending(trainer, optimizers, step):
    if not getattr(trainer, "_concept_multiplier_active", False):
        return None
    if getattr(trainer, "is_ming_image_01", False):
        # Native Ming's optimizer payload already owns gradients and scaler.
        return None
    parameters = named_parameters(trainer, optimizers)
    gradients = {name: p.grad.detach().cpu() for name, p in parameters.items() if p.grad is not None}
    if not gradients:
        return None
    file = trainer.output_dir / f"{trainer.run_name}_step_{step:06d}_gradients.pt"
    pending = file.with_suffix(".pt.tmp")
    scaler = getattr(trainer, "grad_scaler", None)
    payload = {"version": 1, "step": step, "gas": trainer._grad_accum_steps,
               "gradients": gradients, "shapes": {name: list(p.shape) for name, p in parameters.items()},
               "scaler": scaler.state_dict() if getattr(trainer, "use_grad_scaler", False) else None}
    try:
        with pending.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, file)
    finally:
        pending.unlink(missing_ok=True)
    return {"version": 1, "file": file.name, "step": step, "size": file.stat().st_size,
            "gas": trainer._grad_accum_steps, "parameter_count": len(gradients)}


def restore_pending(trainer, optimizers, reference):
    if reference is None:
        return
    name = reference.get("file")
    suffix = f"_step_{reference['step']:06d}_gradients.pt"
    if (reference.get("version") != 1 or not isinstance(name, str)
            or not name.endswith(suffix) or Path(name).name != name):
        raise ValueError("Invalid exposure gradient checkpoint reference")
    file = trainer.output_dir / name
    if not file.exists() and getattr(trainer, "_loaded_checkpoint_path", None):
        file = Path(trainer._loaded_checkpoint_path).parent / name
    if file.stat().st_size != reference["size"]:
        raise ValueError("Exposure gradient checkpoint size mismatch")
    saved = torch.load(file, map_location="cpu", weights_only=True)
    parameters = named_parameters(trainer, optimizers)
    if (saved["version"] != 1 or saved["step"] != reference["step"]
            or saved["gas"] != trainer._grad_accum_steps or reference["gas"] != saved["gas"]
            or saved["shapes"] != {name: list(p.shape) for name, p in parameters.items()}
            or len(saved["gradients"]) != reference["parameter_count"]):
        raise ValueError("Exposure gradient checkpoint settings or parameters changed")
    for name, gradient in saved["gradients"].items():
        if (name not in parameters or list(gradient.shape) != saved["shapes"][name]
                or gradient.dtype != parameters[name].dtype):
            raise ValueError("Invalid exposure pending gradient")
    if (saved["scaler"] is not None) != bool(getattr(trainer, "use_grad_scaler", False)):
        raise ValueError("Cannot change GradScaler with pending exposure gradients")
    if saved["scaler"] is not None:
        if not getattr(trainer, "use_grad_scaler", False):
            raise ValueError("Cannot restore scaled exposure gradients without GradScaler")
        trainer.grad_scaler.load_state_dict(saved["scaler"])
    for name, parameter in parameters.items():
        gradient = saved["gradients"].get(name)
        parameter.grad = None if gradient is None else gradient.to(device=parameter.device, dtype=parameter.dtype)
