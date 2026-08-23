"""Measure SenseNova LoRA training against a real converted checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpointing", choices=("on", "off"), required=True)
    parser.add_argument("--caption", default="a red cube on a white table")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    _require_repo_venv()
    args = _parse_args()
    result = run_probe(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
