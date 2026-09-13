"""Run one short YuE2 ABC-planner LoRA update on a production checkpoint."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _require_repo_venv() -> None:
    executable = Path(sys.executable).resolve()
    expected = (REPO_ROOT / "venv" / "Scripts" / "python.exe").resolve()
    if executable != expected:
        raise RuntimeError(
            f"use the repository virtualenv: expected {expected}, got {executable}"
        )


def _memory_gib() -> tuple[float, float]:
    try:
        import psutil
        info = psutil.Process().memory_info()
        return info.rss / 1024**3, getattr(info, "peak_wset", info.rss) / 1024**3
    except Exception:
        return float("nan"), float("nan")


def run(checkpoint: Path, *, device: torch.device, rank: int,
        include_mlp: bool, gradient_checkpointing: bool, updates: int) -> dict:
    from core.models.yue2.loader import load_yue2_from_path
    from core.models.yue2.yue2_lora import iter_yue2_lora_targets
    from core.pipeline_backends.yue2 import YuE2Mixin
    from core.training.adapters.yue2_adapter import YuE2LoRAAdapter
    from core.training.ops.yue2_ops import (
        build_abc_ar_example,
        collate_abc_ar,
        train_abc_ar_step,
    )

    started = time.perf_counter()
    components = load_yue2_from_path(checkpoint, torch.bfloat16)
    model = components["transformer"]
    tokenizer = components["tokenizer"]
    components.pop("vae", None)
    gc.collect()
    after_load = time.perf_counter()

    model.to(device)
    model.requires_grad_(False)
    example = build_abc_ar_example(
        tokenizer,
        style="short acoustic folk song, solo violin",
        lyrics="[Verse]\nMorning light across the hill",
        abc="X:1\nT:Probe\nM:4/4\nL:1/4\nK:C\nC D E F|G A B c|",
    )
    batch = {
        key: value.to(device)
        for key, value in collate_abc_ar([example]).items()
    }
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        base_loss = float(train_abc_ar_step(model, batch).cpu())
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    trainer = SimpleNamespace(
        transformer=model,
        learning_rate=1e-4,
        unet_lr=None,
        arch=None,
        adapter_algorithm="lora",
        weight_decompose=False,
        adapter_config={},
        yue2_model_identity=components["model_identity"],
    )
    scope = {"attention": True, "mlp": include_mlp}
    original_targets = list(iter_yue2_lora_targets(model, half="ar", scope=scope))
    adapter = YuE2LoRAAdapter(
        trainer,
        rank,
        rank,
        torch.float32,
        objective="abc_ar",
        scope=scope,
    )
    layers = {}
    target_count = adapter.apply_lora_to_unet(layers)
    expected = 196 if include_mlp else 112
    if target_count != expected:
        raise AssertionError(f"injected {target_count} targets, expected {expected}")
    trainable = [
        parameter
        for layer in layers.values()
        for parameter in layer.parameters()
        if parameter.requires_grad
    ]
    trainable_ids = {id(parameter) for parameter in trainable}
    if any(parameter.requires_grad and id(parameter) not in trainable_ids
           for parameter in model.parameters()):
        raise AssertionError("a non-LoRA YuE2 parameter became trainable")

    optimizer = torch.optim.SGD(trainable, lr=1e-4)
    model.train()
    losses = []
    for _update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = train_abc_ar_step(model, batch)
        loss.backward()
        losses.append(float(loss.detach().cpu()))
        if _update + 1 < updates:
            optimizer.step()
    finite_grad_tensors = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in trainable
    )
    nonzero_grad_tensors = sum(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in trainable
    )
    expected_nonzero = len(trainable) if updates >= 2 else target_count
    if finite_grad_tensors != len(trainable) or nonzero_grad_tensors < expected_nonzero:
        raise AssertionError(
            f"LoRA gradients finite={finite_grad_tensors}/{len(trainable)}, "
            f"nonzero={nonzero_grad_tensors}, targets={target_count}"
        )
    optimizer.step()
    after_step = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="sushi-yue2-probe-") as directory:
        output = Path(directory) / "probe.safetensors"
        adapter.save_checkpoint(layers, updates, 0, output)
        with safe_open(output, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            saved_tensors = len(handle.keys())
        if metadata.get("yue2_apply_stages") != "abc":
            raise AssertionError("saved adapter lost its ABC-only stage contract")
        if metadata.get("modelspec.license") != "CC-BY-NC-4.0":
            raise AssertionError("saved adapter lost its weight-license metadata")

        for target in original_targets:
            setattr(target.parent, target.attr, target.module)
        model.eval()
        with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16):
            restored_loss = float(train_abc_ar_step(model, batch).cpu())
        if restored_loss != base_loss:
            raise AssertionError(
                f"removing the training adapter changed base loss: {base_loss} -> {restored_loss}"
            )

        backend = YuE2Mixin()
        backend.yue2_components = {"transformer": model}
        backend._yue2_resolve_lora_path = lambda _raw: str(output)
        files = backend._prepare_yue2_loras([{"path": str(output), "strength": 1.0}])
        if backend._set_yue2_adapter_stage(files, "abc") != target_count:
            raise AssertionError("generation session did not apply every ABC LoRA target")
        with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16):
            adapted_loss = float(train_abc_ar_step(model, batch).cpu())
        if adapted_loss == base_loss:
            raise AssertionError("trained ABC adapter did not change planner loss")
        if backend._set_yue2_adapter_stage(files, "semantic") != 0:
            raise AssertionError("ABC-only adapter was applied to the semantic stage")
        with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16):
            semantic_loss = float(train_abc_ar_step(model, batch).cpu())
        if semantic_loss != base_loss:
            raise AssertionError(
                f"semantic stage did not restore base behavior: {base_loss} -> {semantic_loss}"
            )

    rss_gib, peak_rss_gib = _memory_gib()
    result = {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "gradient_checkpointing": gradient_checkpointing,
        "updates": updates,
        "scope": "attention,mlp" if include_mlp else "attention",
        "sequence_tokens": int(example.input_ids.numel()),
        "target_tokens": int(example.target_tokens),
        "lora_targets": target_count,
        "trainable_tensors": len(trainable),
        "finite_grad_tensors": finite_grad_tensors,
        "nonzero_grad_tensors": nonzero_grad_tensors,
        "saved_tensors": saved_tensors,
        "base_loss": base_loss,
        "adapted_loss": adapted_loss,
        "semantic_stage_loss": semantic_loss,
        "losses": losses,
        "load_seconds": after_load - started,
        "step_seconds": after_step - after_load,
        "rss_gib": rss_gib,
        "peak_rss_gib": peak_rss_gib,
    }
    if device.type == "cuda":
        result["peak_cuda_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
    return result


def run_trainer_loop(checkpoint: Path, *, device: torch.device, rank: int) -> dict:
    """Exercise the same LoRATrainer construction and save path used by a run."""
    from core.training.lora_trainer import LoRATrainer

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sushi-yue2-trainer-probe-") as directory:
        output_dir = Path(directory)
        audio = output_dir / "probe.flac"
        audio.write_bytes(b"dataset-placeholder")
        audio.with_suffix(".abc").write_text(
            "X:1\nT:Probe\nM:4/4\nL:1/4\nK:C\nC D E F|G A B c|",
            encoding="utf-8",
        )
        datasets = [SimpleNamespace(items=[{
            "image_path": str(audio),
            "audio_path": str(audio),
            "caption": "short acoustic folk song, solo violin",
            "lyrics": "[Verse]\nMorning light across the hill",
        }])]
        train_config = {
            "gradient_checkpointing": True,
            "yue2_training_objective": "abc_ar",
            "yue2_lora_scope": "attention",
            "yue2_abc_mode": "full",
            "yue2_allow_truncated_targets": False,
        }
        trainer = LoRATrainer(
            lora_rank=rank,
            lora_alpha=rank,
            lora_dtype="fp32",
            train_unet=True,
            train_text_encoder=False,
            model_path=str(checkpoint),
            output_dir=str(output_dir),
            run_name="yue2_probe",
            learning_rate=1e-4,
            device=str(device),
            weight_dtype="bf16",
            training_dtype="bf16",
            output_dtype="fp32",
            vae_dtype="fp16",
            mixed_precision=True,
            attention_backend="native",
            train_config=train_config,
        )
        stopped = trainer.train(
            datasets=datasets,
            total_steps=1,
            batch_size=1,
            save_every_n_steps=0,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            optimizer_type="adamw",
            lr_scheduler_type="constant",
            max_step_saves_to_keep=2,
            max_optimizer_saves_to_keep=1,
        )
        checkpoints = sorted(output_dir.glob("*.safetensors"))
        states = sorted(output_dir.glob("*_state.json"))
        optimizers = sorted(output_dir.glob("*_optimizer.pt"))
        if stopped or len(checkpoints) != 1 or len(states) != 1 or len(optimizers) != 1:
            raise AssertionError(
                f"trainer artifacts checkpoint={len(checkpoints)}, state={len(states)}, "
                f"optimizer={len(optimizers)}, stopped={stopped}"
            )
        with safe_open(checkpoints[0], framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            tensor_count = len(handle.keys())
        state = json.loads(states[0].read_text(encoding="utf-8"))
        if metadata.get("yue2_apply_stages") != "abc" or state.get("global_step") != 1:
            raise AssertionError("full trainer loop saved an invalid YuE2 contract")
        rss_gib, peak_rss_gib = _memory_gib()
        return {
            "checkpoint": str(checkpoint),
            "device": str(device),
            "lora_targets": len(trainer.lora_layers),
            "saved_tensors": tensor_count,
            "saved_step": state["global_step"],
            "saved_batch": state["batch_idx"],
            "elapsed_seconds": time.perf_counter() - started,
            "rss_gib": rss_gib,
            "peak_rss_gib": peak_rss_gib,
        }


def main() -> None:
    _require_repo_venv()
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--include-mlp", action="store_true")
    parser.add_argument("--trainer-loop", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.rank < 1 or args.updates < 1 or args.threads < 1:
        raise ValueError("rank, updates, and threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    result = (
        run_trainer_loop(args.checkpoint, device=device, rank=args.rank)
        if args.trainer_loop
        else run(
            args.checkpoint,
            device=device,
            rank=args.rank,
            include_mlp=args.include_mlp,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            updates=args.updates,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
