"""Measure SenseNova LoRA training against a real converted checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

EXPECTED_TARGETS = 294
EXPECTED_ATTENTION_TARGETS = 168
EXPECTED_MLP_TARGETS = 126
EXPECTED_LAYERS = 42
EXIT_SMOKE_STEPS = 3
EXIT_SMOKE_WIDTH = 64
EXIT_SMOKE_HEIGHT = 64
EXIT_SMOKE_RUN_NAME = "sensenova_exit_smoke"


@dataclass(frozen=True)
class _TensorSnapshot:
    object_id: int
    data_ptr: int
    shape: tuple[int, ...]
    value: torch.Tensor


@dataclass(frozen=True)
class _CacheLayerSnapshot:
    object_id: int
    keys: _TensorSnapshot
    values: _TensorSnapshot


@dataclass(frozen=True)
class _CacheSnapshot:
    object_id: int
    layers_id: int
    seq_length: int
    layers: list[_CacheLayerSnapshot]


def _repo_venv_python() -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return (REPO_ROOT / "venv" / relative).resolve()


def _require_repo_venv() -> None:
    expected = os.path.normcase(str(_repo_venv_python()))
    actual = os.path.normcase(str(Path(sys.executable).resolve()))
    if actual != expected:
        raise RuntimeError(
            "Run this probe with the repository virtualenv: "
            f"{_repo_venv_python()} {Path(__file__).resolve()}"
        )


def _cuda_memory() -> dict[str, int]:
    torch.cuda.synchronize()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
    }


def _snapshot_tensor(tensor: torch.Tensor) -> _TensorSnapshot:
    return _TensorSnapshot(
        object_id=id(tensor),
        data_ptr=tensor.data_ptr(),
        shape=tuple(tensor.shape),
        value=tensor.detach().cpu().clone(),
    )


def _snapshot_cache(cache: Any) -> _CacheSnapshot:
    layers = getattr(cache, "layers", None)
    if layers is None or len(layers) != EXPECTED_LAYERS:
        raise AssertionError(f"expected {EXPECTED_LAYERS} prefix cache layers")
    seq_length = int(cache.get_seq_length())
    snapshots = [
        _CacheLayerSnapshot(
            object_id=id(layer),
            keys=_snapshot_tensor(layer.keys),
            values=_snapshot_tensor(layer.values),
        )
        for layer in layers
    ]
    return _CacheSnapshot(id(cache), id(layers), seq_length, snapshots)


def _assert_tensor_unchanged(tensor: torch.Tensor, before: _TensorSnapshot) -> None:
    assert id(tensor) == before.object_id
    assert tensor.data_ptr() == before.data_ptr
    assert tuple(tensor.shape) == before.shape
    assert torch.equal(tensor.detach().cpu(), before.value)


def _assert_cache_unchanged(
    cache: Any,
    before: _CacheSnapshot,
) -> None:
    assert id(cache) == before.object_id
    assert id(cache.layers) == before.layers_id
    assert int(cache.get_seq_length()) == before.seq_length
    assert len(cache.layers) == len(before.layers)
    for current, original in zip(cache.layers, before.layers):
        assert id(current) == original.object_id
        _assert_tensor_unchanged(current.keys, original.keys)
        _assert_tensor_unchanged(current.values, original.values)


def _wrap_training_lora(transformer: torch.nn.Module):
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    from core.models.ideogram4.vendor.int8_linear import Int8Linear
    from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets
    from core.training.adapters.sd15_adapter import LoRALinearLayer

    targets = list(iter_sensenova_lora_targets(transformer))
    attention_count = sum(".self_attn." in path for path, *_ in targets)
    mlp_count = sum(".mlp_mot_gen." in path for path, *_ in targets)
    counts = {
        "total": len(targets),
        "attention": attention_count,
        "mlp": mlp_count,
    }
    expected = {
        "total": EXPECTED_TARGETS,
        "attention": EXPECTED_ATTENTION_TARGETS,
        "mlp": EXPECTED_MLP_TARGETS,
    }
    if counts != expected:
        raise AssertionError(f"SenseNova LoRA target census mismatch: {counts} != {expected}")

    plain_int8 = sum(type(module) is Int8Linear for module in transformer.modules())
    convrot = sum(isinstance(module, ConvRotInt8Linear) for module in transformer.modules())
    plain_targets = sum(type(current) is Int8Linear for *_, current in targets)
    base_census = {
        "plain_int8": plain_int8,
        "convrot": convrot,
        "plain_int8_targets": plain_targets,
    }
    expected_base = {
        "plain_int8": 588,
        "convrot": 0,
        "plain_int8_targets": EXPECTED_TARGETS,
    }
    if base_census != expected_base:
        raise AssertionError(
            f"probe requires the plain-int8 SenseNova base: {base_census} != {expected_base}"
        )

    wrappers: dict[str, LoRALinearLayer] = {}
    for module_path, parent, attr, current in targets:
        wrapper = LoRALinearLayer(
            current,
            rank=1,
            alpha=1,
            lora_name=module_path,
            lora_dtype=torch.float32,
        )
        setattr(parent, attr, wrapper)
        wrappers[module_path] = wrapper
    return wrappers, counts, base_census


def _build_training_prefix(transformer, tokenizer, caption: str):
    from core.models.sensenova.vendor.utils import SYSTEM_MESSAGE_FOR_GEN

    query = transformer._build_t2i_query(
        caption,
        system_message=SYSTEM_MESSAGE_FOR_GEN,
        append_text="<think>\n\n</think>\n\n<img>",
    )
    with torch.no_grad():
        input_ids, prefix_indexes, prefix_mask = transformer._build_t2i_text_inputs(
            tokenizer, query
        )
        prefix_cache, _ = transformer._t2i_prefix_forward(
            input_ids, prefix_indexes, prefix_mask
        )
    if getattr(prefix_cache, "_kv_cache_streamer", None) is not None:
        raise AssertionError("training prefix unexpectedly installed a KV cache streamer")
    for layer in prefix_cache.layers:
        if (
            getattr(layer, "flash_k_cache", None) is not None
            or getattr(layer, "flash_v_cache", None) is not None
        ):
            raise AssertionError("training prefix unexpectedly prepared flash KV buffers")
    return input_ids, prefix_indexes, prefix_cache


def _build_fixed_inputs(transformer, prefix_indexes: torch.Tensor, seed: int):
    from core.models.sensenova.sensenova_pipeline_ops import (
        _build_step_context,
        compute_noise_scale,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    x0 = torch.rand(
        (1, 3, 64, 64), generator=generator, device=device, dtype=torch.bfloat16
    ).mul_(2).sub_(1)
    eps = torch.randn(
        x0.shape, generator=generator, device=device, dtype=torch.bfloat16
    )
    t = torch.tensor(0.5, device=device, dtype=torch.bfloat16)
    merge_size = int(1 / transformer.downsample_ratio)
    grid_h = grid_w = 64 // transformer.patch_size
    token_h = grid_h // merge_size
    token_w = grid_w // merge_size
    if (grid_h, grid_w, merge_size, token_h, token_w) != (4, 4, 2, 2, 2):
        raise AssertionError("the real checkpoint no longer has the probed 64px geometry")

    noise_scale = compute_noise_scale(transformer, grid_h, grid_w, merge_size)
    x_t = t * x0 + (1 - t) * noise_scale * eps
    prefix_shape = SimpleNamespace(
        batch_size=1,
        merge_size=merge_size,
        grid_h=grid_h,
        grid_w=grid_w,
        token_h=token_h,
        token_w=token_w,
    )
    z, image_embeds, _ = _build_step_context(
        transformer, prefix_shape, x_t, t, noise_scale
    )
    image_indexes = transformer._build_t2i_image_indexes(
        token_h,
        token_w,
        prefix_indexes.shape[1],
        device=prefix_indexes.device,
    )
    x0_tokens = transformer.patchify(x0, transformer.patch_size * merge_size)
    return x0, t, z, image_embeds, image_indexes, x0_tokens, token_h, token_w


def _loss(
    transformer,
    prefix_cache,
    fixed_inputs,
    checkpoint_layers: bool,
) -> torch.Tensor:
    from core.training.ops.sensenova_ops import forward_gen_decoder_layers

    x0, t, z, image_embeds, image_indexes, x0_tokens, token_h, token_w = fixed_inputs
    del x0
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden = forward_gen_decoder_layers(
            transformer.language_model.model,
            image_embeds,
            indexes=image_indexes,
            prefix_cache=prefix_cache,
            attention_mask=None,
            checkpoint_layers=checkpoint_layers,
        )
        image_2d = hidden.view(1, token_h, token_w, -1).permute(0, 3, 1, 2)
        decoded = transformer.fm_modules["fm_head"](image_2d)
        patch = transformer.patch_size * int(1 / transformer.downsample_ratio)
        x0_pred = (
            decoded.view(1, 3, token_h, patch, token_w, patch)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(1, token_h * token_w, patch * patch * 3)
        )
        denominator = (1 - t).clamp_min(transformer.config.t_eps)
        v_pred = (x0_pred - z) / denominator
        v_target = (x0_tokens - z) / denominator
        return F.mse_loss(v_pred.float(), v_target.float())


def _gradient_stats(wrappers: dict[str, torch.nn.Module], field: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    reached = 0
    finite = 0
    nonzero = 0
    squared_norm = 0.0
    for name in sorted(wrappers):
        parameter = getattr(wrappers[name], field).weight
        gradient = parameter.grad
        if gradient is None:
            continue
        reached += 1
        detached = gradient.detach().float()
        is_finite = bool(torch.isfinite(detached).all())
        finite += int(is_finite)
        nonzero += int(bool(torch.count_nonzero(detached)))
        squared_norm += float(detached.square().sum().cpu())
        digest.update(name.encode("utf-8"))
        digest.update(detached.cpu().contiguous().numpy().tobytes())
    return {
        "reached": reached,
        "finite": finite,
        "nonzero": nonzero,
        "l2": squared_norm**0.5,
        "sha256": digest.hexdigest(),
    }


def _run_step(
    transformer,
    wrappers,
    optimizer,
    prefix_cache,
    fixed_inputs,
    checkpoint_layers: bool,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    baseline = _cuda_memory()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss = _loss(transformer, prefix_cache, fixed_inputs, checkpoint_layers)
    if not bool(torch.isfinite(loss)):
        raise AssertionError(f"non-finite SenseNova training loss: {loss.item()}")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    compute_seconds = time.perf_counter() - started
    peak = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }
    up = _gradient_stats(wrappers, "lora_up")
    down = _gradient_stats(wrappers, "lora_down")
    return {
        "loss": float(loss.detach().cpu()),
        "compute_seconds": compute_seconds,
        "baseline": baseline,
        "peak": peak,
        "up_grad": up,
        "down_grad": down,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    from core.attention import AttentionMode
    from core.models.sensenova.loader import load_sensenova_from_path
    from core.models.sensenova.sensenova_pipeline_ops import set_attention_backend

    if not torch.cuda.is_available():
        raise RuntimeError("SenseNova real-checkpoint training probe requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    checkpoint_path = Path(args.model_path)
    components = load_sensenova_from_path(args.model_path, torch_dtype=torch.bfloat16)
    transformer = components["transformer"]
    tokenizer = components["tokenizer"]
    wrappers, target_counts, base_census = _wrap_training_lora(transformer)
    transformer.to("cuda")
    transformer.train()
    attention_layers = set_attention_backend(
        transformer, "native", AttentionMode.TRAINING
    )
    if attention_layers != EXPECTED_LAYERS:
        raise AssertionError(f"expected {EXPECTED_LAYERS} attention layers, got {attention_layers}")
    model_resident = _cuda_memory()

    _, prefix_indexes, prefix_cache = _build_training_prefix(
        transformer, tokenizer, args.caption
    )
    prefix_resident = _cuda_memory()
    cache_before = _snapshot_cache(prefix_cache)
    fixed_inputs = _build_fixed_inputs(transformer, prefix_indexes, args.seed)
    parameters = [
        parameter
        for wrapper in wrappers.values()
        for parameter in (wrapper.lora_down.weight, wrapper.lora_up.weight)
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.0)
    checkpoint_layers = args.checkpointing == "on"
    steps = [
        _run_step(
            transformer,
            wrappers,
            optimizer,
            prefix_cache,
            fixed_inputs,
            checkpoint_layers,
        )
        for _ in range(2)
    ]

    for index, step in enumerate(steps, start=1):
        for direction in ("up_grad", "down_grad"):
            stats = step[direction]
            if stats["reached"] != EXPECTED_TARGETS or stats["finite"] != EXPECTED_TARGETS:
                raise AssertionError(f"step {index} {direction} did not reach all targets: {stats}")
    required_nonzero = (
        (0, "up_grad"),
        (1, "up_grad"),
        (1, "down_grad"),
    )
    for step_index, direction in required_nonzero:
        if steps[step_index][direction]["nonzero"] != EXPECTED_TARGETS:
            raise AssertionError(
                f"step {step_index + 1} {direction} has dead targets: "
                f"{steps[step_index][direction]}"
            )
    _assert_cache_unchanged(prefix_cache, cache_before)

    return {
        "checkpoint": {
            "name": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
        },
        "checkpointing": args.checkpointing,
        "seed": args.seed,
        "geometry": {"batch": 1, "height": 64, "width": 64, "t": 0.5},
        "targets": target_counts,
        "base_census": base_census,
        "attention_layers": attention_layers,
        "prefix_layers": len(prefix_cache.layers),
        "prefix_seq_length": cache_before.seq_length,
        "cache_unchanged": True,
        "memory": {
            "model_resident": model_resident,
            "prefix_resident": prefix_resident,
            "prefix_delta": {
                key: prefix_resident[key] - model_resident[key] for key in model_resident
            },
        },
        "steps": steps,
    }


def trainer_exit_smoke_config() -> dict[str, Any]:
    """Return the fixed, intentionally small Phase 1 exit-smoke contract.

    This is kept as data so the CPU test can pin the contract without loading a
    checkpoint.  The real trainer arm consumes the same mapping below.
    """
    return {
        "constructor": {
            "lora_rank": 1,
            "lora_alpha": 1,
            "lora_dtype": "fp32",
            "weight_dtype": "bf16",
            "training_dtype": "bf16",
            "output_dtype": "fp32",
            "vae_dtype": "bf16",
            "mixed_precision": True,
            "attention_backend": "native",
            "use_flash_attention": False,
            "blocks_to_swap": 0,
        },
        "train_config": {
            "gradient_checkpointing": True,
            "attention_backend": "native",
            "use_flash_attention": False,
            "batch_size": 1,
            "blocks_to_swap": 0,
            "use_reference_images": False,
            "text_encoding_mode": "onthefly_gpu",
            "latent_encoding_mode": "onthefly_gpu",
            "noise_process": "flow",
            "prediction_target": "velocity",
            "gradient_accumulation_steps": 1,
            "multi_noise_timesteps": 1,
        },
        "train": {
            "total_steps": EXIT_SMOKE_STEPS,
            "batch_size": 1,
            "save_every_n_steps": EXIT_SMOKE_STEPS,
            "sample_every_n_steps": 0,
            "optimizer_type": "adamw",
            "lr_scheduler_type": "constant",
            "enable_bucketing": False,
            "base_resolutions": [EXIT_SMOKE_WIDTH],
            "gradient_accumulation_steps": 1,
            "multi_noise_timesteps": 1,
            "text_encoding_mode": "onthefly_gpu",
            "latent_encoding_mode": "onthefly_gpu",
            "use_reference_images": False,
        },
    }


class _ExitSmokeDataset:
    """The smallest dataset object accepted by ``BaseTrainer.train``."""

    unique_id = "sensenova-phase1-exit-smoke"

    def __init__(self, image_path: Path, prompt: str):
        self.items = [{
            "image_path": str(image_path),
            "caption": prompt,
            "width": EXIT_SMOKE_WIDTH,
            "height": EXIT_SMOKE_HEIGHT,
            "dataset_unique_id": self.unique_id,
        }]
        self._reloaded = False

    def reload_for_epoch(self, epoch_num: int, run_id: int | None = None):
        del run_id
        if epoch_num == 0 and not self._reloaded:
            self._reloaded = True
            return None
        return [dict(item) for item in self.items]


def _write_deterministic_smoke_image(path: Path) -> None:
    from PIL import Image

    pixels = bytearray()
    for y in range(EXIT_SMOKE_HEIGHT):
        for x in range(EXIT_SMOKE_WIDTH):
            pixels.extend(((17 * x + 3 * y) % 256,
                           (5 * x + 19 * y) % 256,
                           (x + 11 * y) % 256))
    Image.frombytes("RGB", (EXIT_SMOKE_WIDTH, EXIT_SMOKE_HEIGHT), bytes(pixels)).save(
        path, format="PNG"
    )


def _hash_named_tensors(named_tensors) -> tuple[str, bool]:
    digest = hashlib.sha256()
    finite = True
    for name, tensor in sorted(named_tensors, key=lambda pair: pair[0]):
        if not torch.isfinite(tensor.detach()).all():
            finite = False
        payload = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(payload.dtype).encode("ascii"))
        digest.update(repr(tuple(payload.shape)).encode("ascii"))
        # reshape first: safetensors stores scalar alpha tensors as 0-D values,
        # for which a direct byte view is not valid on all torch versions.
        digest.update(payload.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest(), finite


def _lora_layer_hash(lora_layers: dict[str, torch.nn.Module]) -> tuple[str, bool]:
    named = []
    for name, layer in lora_layers.items():
        named.append((f"{name}.lora_down.weight", layer.lora_down.weight))
        named.append((f"{name}.lora_up.weight", layer.lora_up.weight))
    return _hash_named_tensors(named)


def _inspect_saved_lora(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        metadata = dict(handle.metadata() or {})
        all_tensors = [(key, handle.get_tensor(key)) for key in keys]
    if len(keys) != EXPECTED_TARGETS * 3:
        raise AssertionError(f"expected 882 LoRA tensors, got {len(keys)}")
    target_names = {
        key.rsplit(".lora_down.weight", 1)[0]
        for key in keys
        if key.endswith(".lora_down.weight")
    }
    target_names.update(
        key.rsplit(".lora_up.weight", 1)[0]
        for key in keys
        if key.endswith(".lora_up.weight")
    )
    if len(target_names) != EXPECTED_TARGETS:
        raise AssertionError(
            f"expected {EXPECTED_TARGETS} saved LoRA targets, got {len(target_names)}"
        )
    for target in target_names:
        expected_keys = {
            f"{target}.lora_down.weight",
            f"{target}.lora_up.weight",
            f"{target}.alpha",
        }
        if not expected_keys.issubset(keys):
            raise AssertionError(f"saved LoRA target {target!r} is missing a tensor")
    all_hash, all_finite = _hash_named_tensors(all_tensors)
    parameter_hash, parameter_finite = _hash_named_tensors(
        (key, tensor) for key, tensor in all_tensors if key.endswith(".weight")
    )
    required_metadata = {
        "tensor_kind": "neo_hf_lora",
        "model_type": "sensenova",
        "modelspec.architecture": "sensenova",
        "lora_targets": "generation",
        "lora_rank": "1",
        "lora_alpha": "1",
        "step": str(EXIT_SMOKE_STEPS),
        "epoch": "2",
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise AssertionError(
                f"saved LoRA metadata {key!r}={metadata.get(key)!r}, expected {expected!r}"
            )
    if not (all_finite and parameter_finite):
        raise AssertionError("saved LoRA contains a non-finite tensor")
    return {
        "tensor_count": len(keys),
        "target_count": len(target_names),
        "parameter_tensor_count": sum(key.endswith(".weight") for key in keys),
        "metadata": required_metadata,
        "tensor_sha256": all_hash,
        "parameter_sha256": parameter_hash,
        "finite": True,
    }


def _run_trainer_exit_smoke_arm(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SenseNova trainer exit smoke requires CUDA")

    from core.training.lora_trainer import LoRATrainer

    config = trainer_exit_smoke_config()
    workdir = Path(args.smoke_workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    image_path = workdir / "training_image.png"
    output_dir = workdir / "trainer_output"
    checkpoint_path = output_dir / f"{EXIT_SMOKE_RUN_NAME}_step_{EXIT_SMOKE_STEPS:06d}.safetensors"
    _write_deterministic_smoke_image(image_path)

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()

    trainer_kwargs = dict(config["constructor"])
    trainer = LoRATrainer(
        model_path=args.model_path,
        output_dir=str(output_dir),
        run_name=EXIT_SMOKE_RUN_NAME,
        run_id=None,
        learning_rate=1e-4,
        device="cuda",
        train_config=dict(config["train_config"]),
        **trainer_kwargs,
    )
    losses: list[float] = []
    training_steps: list[int] = []

    def progress_callback(
        phase: str,
        step: int,
        total: int,
        epoch: int = 0,
        loss: float | None = None,
    ) -> None:
        del total, epoch
        if phase != "training":
            return
        if loss is None or not math.isfinite(float(loss)):
            raise AssertionError(f"non-finite SenseNova exit-smoke loss: {loss!r}")
        training_steps.append(int(step))
        losses.append(float(loss))

    train = dict(config["train"])
    train.update({
        "num_epochs": 1,
        "sample_prompts": [],
        "sample_guidance_scale": 1.0,
        "sample_steps": 1,
        "sample_width": EXIT_SMOKE_WIDTH,
        "sample_height": EXIT_SMOKE_HEIGHT,
        "sample_seed": args.seed,
        "max_grad_norm": 1.0,
        "progress_callback": progress_callback,
        "run_id": None,
        "max_step_saves_to_keep": 1,
        "force_recache": False,
    })
    dataset = _ExitSmokeDataset(image_path, args.prompt)
    trainer.train(datasets=[dataset], **train)

    if training_steps != list(range(1, EXIT_SMOKE_STEPS + 1)):
        raise AssertionError(
            f"expected training callback steps [1, 2, 3], got {training_steps}"
        )
    if len(losses) != EXIT_SMOKE_STEPS:
        raise AssertionError(f"expected {EXIT_SMOKE_STEPS} finite training losses, got {len(losses)}")
    if not checkpoint_path.is_file():
        raise AssertionError(f"trainer did not save {checkpoint_path.name}")
    if len(trainer.lora_layers) != EXPECTED_TARGETS:
        raise AssertionError(
            f"trainer LoRA target count {len(trainer.lora_layers)} != {EXPECTED_TARGETS}"
        )

    lora_hash, lora_finite = _lora_layer_hash(trainer.lora_layers)
    if not lora_finite:
        raise AssertionError("trainer LoRA parameters contain a non-finite value")
    saved = _inspect_saved_lora(checkpoint_path)
    if saved["parameter_sha256"] != lora_hash:
        raise AssertionError("saved LoRA tensor hash differs from live trainer parameters")

    torch.cuda.synchronize()
    peak = {
        "allocated": int(torch.cuda.max_memory_allocated()),
        "reserved": int(torch.cuda.max_memory_reserved()),
    }
    result = {
        "checkpoint": {
            "name": checkpoint_path.name,
            **saved,
        },
        "dataset": {
            "unique_id": _ExitSmokeDataset.unique_id,
            "width": EXIT_SMOKE_WIDTH,
            "height": EXIT_SMOKE_HEIGHT,
            "channels": 3,
            "format": "RGB",
        },
        "targets": EXPECTED_TARGETS,
        "seed": args.seed,
        "training_steps": training_steps,
        "losses": losses,
        "losses_finite": True,
        "lora_parameters_finite": True,
        "lora_parameter_sha256": lora_hash,
        "peak_memory": peak,
        "gradient_checkpointing": bool(trainer.gradient_checkpointing),
        "attention_backend": "native",
        "determinism": {
            "python_random": True,
            "numpy_random": True,
            "torch_seed": args.seed,
            "tf32": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms": True,
        },
        "weight_dtype": str(trainer.weight_dtype),
        "training_dtype": str(trainer.training_dtype),
    }
    try:
        trainer.writer.close()
    finally:
        trainer._db_executor.shutdown(wait=True)
    return result


def _runtime_generation_args(args: argparse.Namespace, lora_path: str | None):
    return SimpleNamespace(
        attn_backend="native",
        cfg_scale=args.smoke_cfg_scale,
        timestep_shift=args.smoke_timestep_shift,
        cfg_norm=args.smoke_cfg_norm,
        seed=args.seed,
        prompt=args.prompt,
        negative_prompt=None,
        output=None,
        lora=lora_path,
        lora_strength=0.0,
        return_tensor=True,
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    digest, finite = _hash_named_tensors((("denoise", tensor),))
    if not finite:
        raise AssertionError("runtime denoise tensor is non-finite")
    return digest


def _take_denoise_tensor(result: dict[str, Any]) -> torch.Tensor:
    """Remove the optional tensor before an arm result is written as JSON."""
    try:
        tensor = result.pop("denoise_tensor")
    except KeyError as exc:
        raise AssertionError("runtime generation did not return denoise_tensor") from exc
    if not isinstance(tensor, torch.Tensor):
        raise AssertionError("runtime generation returned a non-tensor denoise_tensor")
    return tensor


def _run_runtime_verification_arm(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SenseNova runtime verification requires CUDA")

    from core.models.sensenova import sensenova_lora
    from core.models.sensenova import smoke as runtime_smoke

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, _config, tokenizer = runtime_smoke._load_converted(
        args.model_path,
        torch.device("cuda"),
        torch.bfloat16,
    )
    model.eval()
    model.requires_grad_(False)
    before_ids = {
        path: id(module)
        for path, _parent, _attr, module in sensenova_lora.iter_sensenova_lora_targets(model)
    }
    base_args = _runtime_generation_args(args, None)
    base_result = runtime_smoke.run_generation(
        model, tokenizer, base_args, EXIT_SMOKE_WIDTH, EXIT_SMOKE_HEIGHT, 1
    )
    base_tensor = _take_denoise_tensor(base_result)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    lora_args = _runtime_generation_args(args, args.smoke_lora_path)
    lora_result = runtime_smoke.run_generation(
        model, tokenizer, lora_args, EXIT_SMOKE_WIDTH, EXIT_SMOKE_HEIGHT, 1
    )
    lora_tensor = _take_denoise_tensor(lora_result)
    after_ids = {
        path: id(module)
        for path, _parent, _attr, module in sensenova_lora.iter_sensenova_lora_targets(model)
    }
    equal = torch.equal(base_tensor, lora_tensor)
    applied = int(lora_result.get("lora_applied", 0))
    restored = int(lora_result.get("lora_restored", 0))
    identity_restored = before_ids == after_ids
    if not equal:
        raise AssertionError("runtime LoRA strength=0 changed the denoise tensor")
    if applied != EXPECTED_TARGETS or restored != EXPECTED_TARGETS:
        raise AssertionError(f"runtime LoRA apply/restore counts are {applied}/{restored}")
    if not identity_restored:
        raise AssertionError("runtime LoRA restore did not recover every module identity")

    result = {
        "settings": {
            "prompt": args.prompt,
            "seed": args.seed,
            "cfg_scale": args.smoke_cfg_scale,
            "timestep_shift": args.smoke_timestep_shift,
            "cfg_norm": args.smoke_cfg_norm,
            "steps": 1,
            "width": EXIT_SMOKE_WIDTH,
            "height": EXIT_SMOKE_HEIGHT,
            "attention_backend": "native",
            "attention_mode": "inference",
            "tf32": False,
            "deterministic_algorithms": True,
        },
        "base_denoise_sha256": _tensor_digest(base_tensor),
        "strength0_denoise_sha256": _tensor_digest(lora_tensor),
        "strength0_equal": True,
        "lora_applied": applied,
        "lora_restored": restored,
        "module_identity_restored": identity_restored,
        "base_peak_vram_gb": base_result.get("peak_vram_gb"),
        "strength0_peak_vram_gb": lora_result.get("peak_vram_gb"),
    }
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _run_exit_smoke_subprocess(
    args: argparse.Namespace,
    arm: str,
    workdir: Path,
    *,
    lora_path: Path | None = None,
) -> dict[str, Any]:
    result_path = workdir / f"{arm}.json"
    cmd = [
        str(_repo_venv_python()),
        str(Path(__file__).resolve()),
        "--model-path", args.model_path,
        "--seed", str(args.seed),
        "--prompt", args.prompt,
        "--smoke-cfg-scale", str(args.smoke_cfg_scale),
        "--smoke-timestep-shift", str(args.smoke_timestep_shift),
        "--smoke-cfg-norm", args.smoke_cfg_norm,
        "--smoke-arm", arm,
        "--smoke-workdir", str(workdir),
        "--smoke-arm-json", str(result_path),
    ]
    if lora_path is not None:
        cmd.extend(("--smoke-lora-path", str(lora_path)))
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=args.smoke_timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"SenseNova exit-smoke {arm} arm timed out") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"SenseNova exit-smoke {arm} arm exited with code {completed.returncode}"
        )
    if not result_path.is_file():
        raise RuntimeError(f"SenseNova exit-smoke {arm} arm wrote no JSON result")
    with result_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_trainer_exit_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run training and runtime verification in separate, short-lived arms."""
    if not torch.cuda.is_available():
        raise RuntimeError("SenseNova trainer exit smoke requires CUDA")
    with tempfile.TemporaryDirectory(prefix="sensenova_phase1_exit_smoke_") as raw_workdir:
        workdir = Path(raw_workdir)
        trainer = _run_exit_smoke_subprocess(args, "trainer", workdir)
        lora_path = workdir / "trainer_output" / (
            f"{EXIT_SMOKE_RUN_NAME}_step_{EXIT_SMOKE_STEPS:06d}.safetensors"
        )
        if not lora_path.is_file():
            raise RuntimeError(f"trainer arm did not produce {lora_path.name}")
        runtime = _run_exit_smoke_subprocess(
            args, "runtime", workdir, lora_path=lora_path
        )
    return {
        "probe": "sensenova_phase1_trainer_exit_smoke",
        "checkpoint": Path(args.model_path).name,
        "trainer": trainer,
        "runtime": runtime,
        "process_isolation": "trainer_then_runtime",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--checkpointing",
        choices=("on", "off"),
        default=None,
        help="Required for the original two-step checkpoint probe; not used by --trainer-exit-smoke.",
    )
    parser.add_argument("--caption", default="a red cube on a white table")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--trainer-exit-smoke",
        action="store_true",
        help="Opt-in Phase 1 real trainer + fresh runtime verification (CUDA, multi-process).",
    )
    parser.add_argument("--prompt", default="a red cube on a white table")
    parser.add_argument(
        "--smoke-cfg-scale",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--smoke-timestep-shift",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--smoke-cfg-norm",
        default=None,
    )
    parser.add_argument("--smoke-timeout-s", type=float, default=3600.0)
    parser.add_argument("--smoke-json-out", default=None)
    parser.add_argument("--smoke-arm", choices=("trainer", "runtime"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-workdir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-arm-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-lora-path", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.trainer_exit_smoke and args.smoke_arm is None and args.checkpointing is None:
        parser.error("--checkpointing is required unless --trainer-exit-smoke is selected")
    if args.smoke_arm is not None and (args.smoke_workdir is None or args.smoke_arm_json is None):
        parser.error("internal smoke arm requires --smoke-workdir and --smoke-arm-json")
    if args.smoke_arm == "runtime" and args.smoke_lora_path is None:
        parser.error("internal runtime smoke arm requires --smoke-lora-path")
    if (args.trainer_exit_smoke or args.smoke_arm is not None) and any(
        value is None
        for value in (args.smoke_cfg_scale, args.smoke_timestep_shift, args.smoke_cfg_norm)
    ):
        from api.param_defaults import SENSENOVA_GENERATION_DEFAULTS

        if args.smoke_cfg_scale is None:
            args.smoke_cfg_scale = SENSENOVA_GENERATION_DEFAULTS["cfg_scale"]
        if args.smoke_timestep_shift is None:
            args.smoke_timestep_shift = SENSENOVA_GENERATION_DEFAULTS["timestep_shift"]
        if args.smoke_cfg_norm is None:
            args.smoke_cfg_norm = SENSENOVA_GENERATION_DEFAULTS["cfg_norm"]
    return args


def _write_smoke_arm_result(args: argparse.Namespace, result: dict[str, Any]) -> None:
    result_path = Path(args.smoke_arm_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


def _write_public_smoke_result(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if args.smoke_json_out:
        result_path = Path(args.smoke_json_out)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)


def main() -> int:
    _require_repo_venv()
    args = _parse_args()
    if args.smoke_arm == "trainer":
        result = _run_trainer_exit_smoke_arm(args)
        _write_smoke_arm_result(args, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.smoke_arm == "runtime":
        result = _run_runtime_verification_arm(args)
        _write_smoke_arm_result(args, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.trainer_exit_smoke:
        result = run_trainer_exit_smoke(args)
        _write_public_smoke_result(args, result)
    else:
        result = run_probe(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
