"""Ming stochastic and pending-gradient continuation at completed batch seams."""
import random
import numpy as np
import torch

RUNTIME_KEY = "_sushi_ming_runtime"


def _parameters(trainer):
    names = trainer._build_ema_param_name_map()
    result = {}
    from core.training.base_trainer import all_optimizers
    optimizers = all_optimizers(trainer)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                name = names.get(id(parameter))
                if name is None or (name in result and result[name] is not parameter):
                    raise ValueError("Ming resume requires unique model parameter names")
                result[name] = parameter
    return result


def capture_runtime(trainer):
    numpy = np.random.get_state()
    device = torch.device(trainer.device)
    return {"schema_version": 1, "run_seed": trainer.run_seed,
        "gradient_accumulation_steps": trainer._grad_accum_steps,
        "torch_cpu": torch.get_rng_state(),
        "torch_device": device.type,
        "torch_cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
        "numpy": [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]],
        "python": random.getstate(),
        "gradients": {name: parameter.grad.detach().to(device="cpu", copy=True)
            for name, parameter in _parameters(trainer).items() if parameter.grad is not None},
        "grad_scaler": trainer.grad_scaler.state_dict() if trainer.use_grad_scaler else None}


def queue_runtime(trainer, payload):
    runtime = payload.pop(RUNTIME_KEY, None)
    if runtime is None:
        return
    trainer._ming_pending_runtime = runtime
    if not isinstance(runtime,dict) or type(runtime.get("schema_version")) is not int or runtime.get("schema_version") != 1:
        raise ValueError("Unsupported Ming training runtime schema")
    if "state" in payload and not payload["state"]:
        optimizer = trainer.optimizer
        names = trainer._build_ema_param_name_map()
        current = [[names[id(parameter)] for parameter in group["params"]]
                   for group in optimizer.param_groups]
        if payload.get("_sushi_param_names") != current or payload.get("_sushi_opt_class") != type(optimizer).__name__:
            raise ValueError("Ming pending-gradient optimizer ownership differs from this training run")
        # An empty pre-update state is valid; the generic named loader reports no moments restored.
        payload.pop("_sushi_param_names")
    if trainer.run_seed_drawn:
        trainer.run_seed = int(runtime["run_seed"])


def restore_runtime(trainer):
    runtime = getattr(trainer, "_ming_pending_runtime", None)
    if runtime is None:
        return
    device = torch.device(trainer.device)
    if runtime["torch_device"] != device.type:
        raise ValueError("Exact Ming stochastic resume requires the saved execution device type")
    gradients = runtime["gradients"]
    if gradients and runtime["gradient_accumulation_steps"] != trainer._grad_accum_steps:
        raise ValueError("Cannot change Ming gradient accumulation with pending saved gradients")
    parameters = _parameters(trainer)
    if gradients.keys() - parameters.keys():
        raise ValueError("Ming saved gradient ownership differs from this training run")
    for name, value in gradients.items():
        if value.shape != parameters[name].shape or value.dtype != parameters[name].dtype:
            raise ValueError(f"Ming saved gradient shape/precision mismatch: {name}")
    scaler = runtime["grad_scaler"]
    if (scaler is not None) != trainer.use_grad_scaler:
        raise ValueError("Ming saved gradient scaler mode differs from this training run")
    for name, parameter in parameters.items():
        value = gradients.get(name)
        parameter.grad = None if value is None else value.to(device=parameter.device, copy=True)
    if scaler is not None:
        trainer.grad_scaler.load_state_dict(scaler)
    numpy = runtime["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    random.setstate(runtime["python"])
    torch.set_rng_state(runtime["torch_cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(runtime["torch_cuda"].cpu(), device=device)
    trainer._ming_pending_runtime = None
