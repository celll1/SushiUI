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
MIXED_SMOKE_RUN_NAME = "sensenova_mixed_smoke"
MIXED_REFERENCE_DATASET_ID = 23
MIXED_REFERENCE_FREE_DATASET_ID = 37
# Hard per-process cap: an over-budget arm OOMs inside its own process instead of
# filling the shared GPU.
VRAM_GATE_FRACTION = 0.72


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


def _apply_vram_gate(fraction: float = VRAM_GATE_FRACTION) -> dict[str, Any]:
    """Cap this process at ``fraction`` of the device.

    The default is the shared-GPU gate every existing caller relies on. An
    override exists for one case only: an arm whose whole question is whether a
    configuration fits in the WHOLE card. Gated at 0.72 such an arm OOMs on the
    gate rather than on reality, which measures the gate.
    """
    if not torch.cuda.is_available():
        return {"applied": False, "fraction": float(fraction)}
    fraction = float(fraction)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    total = int(torch.cuda.get_device_properties(0).total_memory)
    return {
        "applied": True,
        "fraction": fraction,
        "device_total_bytes": total,
        "budget_bytes": int(total * fraction),
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


def trainer_exit_smoke_config(phase_eviction: bool = False) -> dict[str, Any]:
    """Return the fixed, intentionally small Phase 1 exit-smoke contract.

    This is kept as data so the CPU test can pin the contract without loading a
    checkpoint.  The real trainer arm consumes the same mapping below.
    """
    config = {
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
            "sensenova_mot_phase_eviction": bool(phase_eviction),
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
    return config


class _ExitSmokeDataset:
    """The smallest dataset object accepted by ``BaseTrainer.train``.

    ``width`` / ``height`` are what actually set the training resolution: with
    bucketing off, ``base_resolutions`` only clamps items DOWN into its area
    (``base_trainer`` no-bucketing path), so a caller that raises the resolution
    must raise BOTH this and ``base_resolutions``.
    """

    unique_id = "sensenova-phase1-exit-smoke"

    def __init__(
        self,
        image_path: Path,
        prompt: str,
        width: int = EXIT_SMOKE_WIDTH,
        height: int = EXIT_SMOKE_HEIGHT,
        reference_images: list[str] | None = None,
    ):
        item = {
            "image_path": str(image_path),
            "caption": prompt,
            "width": int(width),
            "height": int(height),
            "dataset_unique_id": self.unique_id,
        }
        # The key production sets from related_images["reference"]; the trainer
        # reads it per item, so presence here is the whole of "this item is
        # reference-conditioned" (7.2 judgement 1).
        if reference_images:
            item["reference_images"] = [str(p) for p in reference_images]
        self.items = [item]
        self._reloaded = False

    def reload_for_epoch(self, epoch_num: int, run_id: int | None = None):
        del run_id
        if epoch_num == 0 and not self._reloaded:
            self._reloaded = True
            return None
        return [dict(item) for item in self.items]


def _lora_grad_digest(lora_layers: dict[str, torch.nn.Module]) -> dict[str, Any]:
    return {
        "up": _gradient_stats(lora_layers, "lora_up"),
        "down": _gradient_stats(lora_layers, "lora_down"),
    }


class _GradDigestCapture:
    """Read the LoRA gradients at every ``optimizer.step()`` without changing it."""

    def __init__(self, layers_getter):
        self._layers_getter = layers_getter
        self.records: list[dict[str, Any]] = []
        self._original = None

    def __enter__(self) -> "_GradDigestCapture":
        original = torch.optim.AdamW.step
        capture = self

        def step(optimizer_self, *args, **kwargs):
            capture.records.append(_lora_grad_digest(capture._layers_getter()))
            return original(optimizer_self, *args, **kwargs)

        torch.optim.AdamW.step = step
        self._original = original
        return self

    def __exit__(self, *exc_info) -> bool:
        torch.optim.AdamW.step = self._original
        return False


class _ProbeDataset:
    """Item dicts straight from the DB, shaped exactly like train_runner's."""

    def __init__(self, unique_id: str, items: list[dict[str, Any]]):
        self.unique_id = unique_id
        self.items = [dict(item, dataset_unique_id=unique_id) for item in items]
        self._reloaded = False

    def reload_for_epoch(self, epoch_num: int, run_id: int | None = None):
        del run_id
        if epoch_num == 0 and not self._reloaded:
            self._reloaded = True
            return None
        return [dict(item) for item in self.items]


def _read_dataset_items(dataset_id: int, limit: int) -> list[dict[str, Any]]:
    """Bounded, read-only mirror of ``train_runner._load_dataset_items_fast``.

    Field derivation is copied from that function, not re-invented:
    the caption auto-select priority (``train_runner.py:889-901``) and
    ``reference_images = related_images["reference"]``
    (``train_runner.py:1387-1388`` / ``:1200-1201``).
    """
    import sqlite3

    database = REPO_ROOT / "datasets.db"
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT id, image_path, width, height, related_images "
            "FROM dataset_items WHERE dataset_id = ? ORDER BY id LIMIT ?",
            (dataset_id, limit),
        )
        rows = cursor.fetchall()
        items = []
        for item_id, image_path, width, height, related_images in rows:
            if not os.path.exists(image_path):
                continue
            cursor.execute(
                "SELECT caption_type, content FROM dataset_captions WHERE item_id = ?",
                (item_id,),
            )
            captions = cursor.fetchall()
            caption = ""
            for caption_type in ("tags", "natural_language"):
                match = [text for kind, text in captions if kind == caption_type]
                if match:
                    caption = match[0]
                    break
            else:
                if captions:
                    caption = captions[0][1]
            item = {
                "image_path": image_path,
                "caption": caption,
                "width": width,
                "height": height,
                "caption_types_available": sorted({kind for kind, _ in captions}),
            }
            related = json.loads(related_images) if related_images else None
            if related and "reference" in related:
                item["reference_images"] = related["reference"]
            items.append(item)
    finally:
        connection.close()
    if len(items) != limit:
        raise AssertionError(
            f"dataset {dataset_id}: wanted {limit} readable items, got {len(items)}"
        )
    return items


class _ReferenceInstrumentation:
    """Record the reference prefix's shape/size without altering any of it."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.records: list[dict[str, Any]] = []
        self._restore: list = []

    def __enter__(self) -> "_ReferenceInstrumentation":
        from core.training.ops import sensenova_ops

        transformer = self.trainer.transformer
        original_encode = sensenova_ops.encode_prompt
        original_build = transformer._build_it2i_inputs
        original_extract = transformer.extract_feature
        pending: dict[str, Any] = {}

        def extract_feature(*args, **kwargs):
            output = original_extract(*args, **kwargs)
            pending["vit_tokens"] = int(output.numel() // output.shape[-1])
            return output

        def build_it2i_inputs(tokenizer, query, pixel_values=None, grid_hw=None):
            transformer.extract_feature = extract_feature
            try:
                embeds, indexes, mask = original_build(
                    tokenizer, query, pixel_values, grid_hw
                )
            finally:
                del transformer.extract_feature
            token_id = int(transformer.img_context_token_id)
            input_ids = tokenizer(query, return_tensors="pt")["input_ids"][0]
            pending.update({
                "img_context_token_id": token_id,
                "img_context_placeholders": int((input_ids == token_id).sum()),
                "prefix_token_count": int(indexes.shape[1]),
                "prefix_t_extent": int(indexes[0].max()) + 1,
                "grid_hw": grid_hw.tolist() if grid_hw is not None else None,
            })
            return embeds, indexes, mask

        def encode_prompt(trainer, prompt, *, requires_grad=False, reference_image_paths=None):
            pending.clear()
            torch.cuda.synchronize()
            before = _cuda_memory()
            torch.cuda.reset_peak_memory_stats()
            prefix = original_encode(
                trainer,
                prompt,
                requires_grad=requires_grad,
                reference_image_paths=reference_image_paths,
            )
            torch.cuda.synchronize()
            after = _cuda_memory()
            sensenova_ops._assert_immutable_prefix_cache(
                prefix.cache, len(trainer.transformer.language_model.model.layers)
            )
            self.records.append({
                "has_reference": bool(reference_image_paths),
                "reference_image_paths": list(reference_image_paths or []),
                "caption_chars": len(prompt),
                "text_length": int(prefix.text_length),
                "prefix_seq_length": int(prefix.cache.get_seq_length()),
                "prefix_layers": len(prefix.cache.layers),
                "immutable_prefix_cache": True,
                "allocated_delta": after["allocated"] - before["allocated"],
                "encode_peak_allocated": int(torch.cuda.max_memory_allocated()),
                **pending,
            })
            return prefix

        sensenova_ops.encode_prompt = encode_prompt
        transformer._build_it2i_inputs = build_it2i_inputs
        self._restore = [
            lambda: setattr(sensenova_ops, "encode_prompt", original_encode),
            lambda: delattr(transformer, "_build_it2i_inputs"),
        ]
        return self

    def __exit__(self, *exc_info) -> bool:
        for restore in self._restore:
            restore()
        return False


def _write_deterministic_smoke_image(
    path: Path,
    width: int = EXIT_SMOKE_WIDTH,
    height: int = EXIT_SMOKE_HEIGHT,
) -> None:
    from PIL import Image

    pixels = bytearray()
    for y in range(int(height)):
        for x in range(int(width)):
            pixels.extend(((17 * x + 3 * y) % 256,
                           (5 * x + 19 * y) % 256,
                           (x + 11 * y) % 256))
    Image.frombytes("RGB", (int(width), int(height)), bytes(pixels)).save(
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

    vram_gate = _apply_vram_gate()
    config = trainer_exit_smoke_config(
        phase_eviction=getattr(args, "smoke_phase_eviction", "off") == "on"
    )
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

    load_started = time.perf_counter()
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
    model_load_wall_time_s = time.perf_counter() - load_started
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
    train_started = time.perf_counter()
    with _GradDigestCapture(lambda: trainer.lora_layers) as grad_capture:
        trainer.train(datasets=[dataset], **train)
    train_wall_time_s = time.perf_counter() - train_started
    wall_time_with_model_load_s = time.perf_counter() - load_started

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
        "grad_digests": grad_capture.records,
        "peak_memory": peak,
        "vram_gate": vram_gate,
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
        "phase_eviction": bool(config["train_config"]["sensenova_mot_phase_eviction"]),
        "wall_time_s": train_wall_time_s,
        "wall_time_with_model_load_s": wall_time_with_model_load_s,
        "model_load_wall_time_s": model_load_wall_time_s,
    }
    try:
        trainer.writer.close()
    finally:
        trainer._db_executor.shutdown(wait=True)
    return result


def _run_mixed_smoke_arm(args: argparse.Namespace) -> dict[str, Any]:
    """Phase 3-4: one run whose items mix reference-carrying and plain datasets."""
    if not torch.cuda.is_available():
        raise RuntimeError("SenseNova mixed smoke requires CUDA")

    from core.training.lora_trainer import LoRATrainer

    vram_gate = _apply_vram_gate()
    config = trainer_exit_smoke_config()
    config["train_config"]["use_reference_images"] = True
    config["train"]["use_reference_images"] = True
    config["train"]["base_resolutions"] = [args.mixed_base_resolution]

    reference_items = _read_dataset_items(args.mixed_reference_dataset_id, args.mixed_items)
    reference_free_items = _read_dataset_items(
        args.mixed_reference_free_dataset_id, args.mixed_items
    )
    for item in reference_items:
        if not item.get("reference_images"):
            raise AssertionError(f"dataset item {item['image_path']} carries no reference")
    for item in reference_free_items:
        if item.get("reference_images"):
            raise AssertionError(
                f"the reference-free dataset item {item['image_path']} carries a reference"
            )

    workdir = Path(args.smoke_workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    output_dir = workdir / "mixed_output"
    checkpoint_path = output_dir / f"{MIXED_SMOKE_RUN_NAME}_step_{EXIT_SMOKE_STEPS:06d}.safetensors"

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    trainer = LoRATrainer(
        model_path=args.model_path,
        output_dir=str(output_dir),
        run_name=MIXED_SMOKE_RUN_NAME,
        run_id=None,
        learning_rate=1e-4,
        device="cuda",
        train_config=dict(config["train_config"]),
        **dict(config["constructor"]),
    )
    model_load_wall_time_s = time.perf_counter() - load_started
    model_resident = _cuda_memory()
    losses: list[float] = []
    training_steps: list[int] = []

    def progress_callback(phase, step, total, epoch=0, loss=None):
        del total, epoch
        if phase != "training":
            return
        if loss is None or not math.isfinite(float(loss)):
            raise AssertionError(f"non-finite SenseNova mixed-smoke loss: {loss!r}")
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
    datasets = [
        _ProbeDataset(f"sensenova-phase3-ref-{args.mixed_reference_dataset_id}", reference_items),
        _ProbeDataset(
            f"sensenova-phase3-noref-{args.mixed_reference_free_dataset_id}",
            reference_free_items,
        ),
    ]
    train_started = time.perf_counter()
    with _ReferenceInstrumentation(trainer) as prefixes:
        with _GradDigestCapture(lambda: trainer.lora_layers) as grad_capture:
            trainer.train(datasets=datasets, **train)
        train_wall_time_s = time.perf_counter() - train_started
        # Same caption, reference on/off: the only A/B where the prefix size
        # difference is attributable to the reference alone.
        from core.training.ops import sensenova_ops

        step_record_count = len(prefixes.records)
        paired_caption = reference_items[0]["caption"]
        paired = sensenova_ops.encode_prompt(
            trainer,
            paired_caption,
            reference_image_paths=reference_items[0]["reference_images"],
        )
        del paired
        torch.cuda.empty_cache()
        paired = sensenova_ops.encode_prompt(trainer, paired_caption)
        del paired
        torch.cuda.empty_cache()
    paired_prefixes = prefixes.records[step_record_count:]

    if training_steps != list(range(1, EXIT_SMOKE_STEPS + 1)):
        raise AssertionError(f"expected training steps [1, 2, 3], got {training_steps}")
    records = prefixes.records[:step_record_count]
    if len(records) != EXIT_SMOKE_STEPS:
        raise AssertionError(
            f"expected one prefix per step, got {len(records)} for {EXIT_SMOKE_STEPS} steps"
        )
    with_reference = [record for record in records if record["has_reference"]]
    without_reference = [record for record in records if not record["has_reference"]]
    if not with_reference or not without_reference:
        raise AssertionError(
            "the mixed run did not exercise both kinds of item: "
            f"{len(with_reference)} with reference, {len(without_reference)} without"
        )
    for record in with_reference:
        if record["vit_tokens"] != record["img_context_placeholders"]:
            raise AssertionError(
                f"ViT emitted {record['vit_tokens']} tokens for "
                f"{record['img_context_placeholders']} <IMG_CONTEXT> placeholders"
            )
        if record["text_length"] != record["prefix_t_extent"]:
            raise AssertionError(
                f"text_length {record['text_length']} != t extent {record['prefix_t_extent']}"
            )
    t_extent_below_tokens = [
        record for record in with_reference
        if record["text_length"] < record["prefix_token_count"]
    ]
    if not t_extent_below_tokens:
        raise AssertionError(
            "no reference prefix had text_length < token count; the t-extent check "
            "would pass on a degenerate case"
        )
    for record in without_reference:
        if record["text_length"] != record["prefix_seq_length"]:
            raise AssertionError(
                f"text-only text_length {record['text_length']} != prefix seq length "
                f"{record['prefix_seq_length']}"
            )
    if not checkpoint_path.is_file():
        raise AssertionError(f"trainer did not save {checkpoint_path.name}")
    lora_hash, lora_finite = _lora_layer_hash(trainer.lora_layers)
    if not lora_finite:
        raise AssertionError("mixed-run LoRA parameters contain a non-finite value")
    saved = _inspect_saved_lora_relaxed(checkpoint_path)
    if saved["parameter_sha256"] != lora_hash:
        raise AssertionError("saved LoRA tensor hash differs from live trainer parameters")
    for index, digest in enumerate(grad_capture.records, start=1):
        for direction in ("up", "down"):
            stats = digest[direction]
            if stats["reached"] != EXPECTED_TARGETS or stats["finite"] != EXPECTED_TARGETS:
                raise AssertionError(f"step {index} {direction} gradients: {stats}")

    torch.cuda.synchronize()
    peak = {
        "allocated": int(torch.cuda.max_memory_allocated()),
        "reserved": int(torch.cuda.max_memory_reserved()),
    }
    result = {
        "probe": "sensenova_phase3_mixed_smoke",
        "checkpoint": {"name": checkpoint_path.name, **saved},
        "datasets": [
            {
                "unique_id": dataset.unique_id,
                "items": [
                    {
                        "image_path": item["image_path"],
                        "width": item["width"],
                        "height": item["height"],
                        "caption_chars": len(item["caption"]),
                        "caption_types_available": item["caption_types_available"],
                        "reference_images": item.get("reference_images", []),
                    }
                    for item in dataset.items
                ],
            }
            for dataset in datasets
        ],
        "seed": args.seed,
        "training_steps": training_steps,
        "losses": losses,
        "losses_finite": True,
        "prefixes": records,
        "paired_caption_prefixes": paired_prefixes,
        "grad_digests": grad_capture.records,
        "lora_parameter_sha256": lora_hash,
        "model_resident": model_resident,
        "peak_memory": peak,
        "vram_gate": vram_gate,
        "wall_time_s": train_wall_time_s,
        "model_load_wall_time_s": model_load_wall_time_s,
    }
    try:
        trainer.writer.close()
    finally:
        trainer._db_executor.shutdown(wait=True)
    return result


def _inspect_saved_lora_relaxed(path: Path) -> dict[str, Any]:
    """``_inspect_saved_lora`` without the exit-smoke's fixed epoch metadata."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        metadata = dict(handle.metadata() or {})
        tensors = [(key, handle.get_tensor(key)) for key in keys]
    if len(keys) != EXPECTED_TARGETS * 3:
        raise AssertionError(f"expected {EXPECTED_TARGETS * 3} LoRA tensors, got {len(keys)}")
    parameter_hash, finite = _hash_named_tensors(
        (key, tensor) for key, tensor in tensors if key.endswith(".weight")
    )
    if not finite:
        raise AssertionError("saved LoRA contains a non-finite tensor")
    return {
        "tensor_count": len(keys),
        "parameter_sha256": parameter_hash,
        "metadata": {
            key: metadata.get(key)
            for key in ("tensor_kind", "model_type", "lora_targets", "step", "epoch")
        },
        "finite": True,
    }


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
        "--smoke-phase-eviction", getattr(args, "smoke_phase_eviction", "off"),
        "--smoke-cfg-scale", str(args.smoke_cfg_scale),
        "--smoke-timestep-shift", str(args.smoke_timestep_shift),
        "--smoke-cfg-norm", args.smoke_cfg_norm,
        "--smoke-arm", arm,
        "--smoke-workdir", str(workdir),
        "--smoke-arm-json", str(result_path),
        "--mixed-reference-dataset-id", str(args.mixed_reference_dataset_id),
        "--mixed-reference-free-dataset-id", str(args.mixed_reference_free_dataset_id),
        "--mixed-items", str(args.mixed_items),
        "--mixed-base-resolution", str(args.mixed_base_resolution),
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


def run_mixed_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SenseNova mixed smoke requires CUDA")
    with tempfile.TemporaryDirectory(prefix="sensenova_phase3_mixed_smoke_") as raw_workdir:
        workdir = Path(raw_workdir)
        return _run_exit_smoke_subprocess(args, "mixed", workdir)


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
    parser.add_argument(
        "--smoke-phase-eviction",
        choices=("off", "on"),
        default="off",
        help="Enable SenseNova MoT half-eviction for the trainer exit-smoke arm.",
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
    parser.add_argument(
        "--mixed-smoke",
        action="store_true",
        help="Opt-in Phase 3 mixed reference / reference-free run (CUDA, real datasets).",
    )
    parser.add_argument(
        "--mixed-reference-dataset-id", type=int, default=MIXED_REFERENCE_DATASET_ID
    )
    parser.add_argument(
        "--mixed-reference-free-dataset-id",
        type=int,
        default=MIXED_REFERENCE_FREE_DATASET_ID,
    )
    parser.add_argument("--mixed-items", type=int, default=1)
    parser.add_argument("--mixed-base-resolution", type=int, default=EXIT_SMOKE_WIDTH)
    parser.add_argument("--smoke-timeout-s", type=float, default=3600.0)
    parser.add_argument("--smoke-json-out", default=None)
    parser.add_argument("--smoke-arm", choices=("trainer", "runtime", "mixed"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-workdir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-arm-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-lora-path", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        not args.trainer_exit_smoke
        and not args.mixed_smoke
        and args.smoke_arm is None
        and args.checkpointing is None
    ):
        parser.error(
            "--checkpointing is required unless --trainer-exit-smoke or --mixed-smoke is selected"
        )
    if args.smoke_arm is not None and (args.smoke_workdir is None or args.smoke_arm_json is None):
        parser.error("internal smoke arm requires --smoke-workdir and --smoke-arm-json")
    if args.smoke_arm == "runtime" and args.smoke_lora_path is None:
        parser.error("internal runtime smoke arm requires --smoke-lora-path")
    if (args.trainer_exit_smoke or args.mixed_smoke or args.smoke_arm is not None) and any(
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
    if args.smoke_arm == "mixed":
        result = _run_mixed_smoke_arm(args)
        _write_smoke_arm_result(args, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.mixed_smoke:
        result = run_mixed_smoke(args)
        _write_public_smoke_result(args, result)
    elif args.trainer_exit_smoke:
        result = run_trainer_exit_smoke(args)
        _write_public_smoke_result(args, result)
    else:
        result = run_probe(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
