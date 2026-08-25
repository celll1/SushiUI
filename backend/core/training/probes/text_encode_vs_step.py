"""Measure text-encode wall against DiT step wall, per architecture.

Answers two questions with numbers rather than argument:

1. Does ``cpu_te_prefetch``'s premise hold?  Its docstring claims "PyTorch CPU
   matmul kernels release the GIL, so the worker really overlaps with GPU
   compute".  The ``te-cpu`` and ``te-cpu-contended`` arms run the SAME
   ``CpuTextEncoderPrefetcher`` worker, differing only in whether the main
   thread is hammering the GPU.  If the premise holds, per-batch CPU encode
   time is unchanged between them.
2. Is ``M3 x batch_size < M2``?  i.e. can a CPU-resident text encoder keep up
   with one DiT step, which is the precondition for ``cpu_prefetch`` hiding
   text encoding entirely.

Every arm runs in its own process (``--arm``); mixing them in one process lets
allocator fragmentation from the DiT arm pollute the encode arms' numbers.

Nothing here mutates training behaviour: the encode arms drive the production
``BaseTrainer.encode_caption`` / ``encode_captions_batched`` code through a
minimal attribute shim, and the ``dit-step`` arm drives the real ``LoRATrainer``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

ARCHS = ("anima", "krea2", "sdxl", "sensenova")
ARMS = ("te-cpu", "te-cpu-contended", "te-gpu", "dit-step")
# SenseNova-only arms, dispatched in main() alongside ARMS.
SENSENOVA_ARMS = ("sensenova", "sensenova-four-phase")

DEFAULT_MODEL_PATHS = {
    "anima": r"M:\model\anima\split_files\diffusion_models\anima-base-v1.0.safetensors",
    "krea2": r"M:\model\krea2",
    "sdxl": r"M:\model\sdxl\Illustrious-XL-v2.0.safetensors",
    "sensenova": r"M:\model\sensenova\sensenova_int8.safetensors",
}
DEFAULT_CAPTION_DIR = r"M:\dataset_working\copyright\kouyoku_senki_exs-tia"


def _repo_venv_python() -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return (REPO_ROOT / "venv" / relative).resolve()


def _require_repo_venv() -> None:
    expected = os.path.normcase(str(_repo_venv_python()))
    actual = os.path.normcase(str(Path(sys.executable).resolve()))
    if actual != expected:
        raise RuntimeError(
            f"Run this probe with the repository virtualenv: {_repo_venv_python()}"
        )


def _apply_vram_gate(fraction: float) -> dict[str, Any]:
    """Cap this process's VRAM so a probe can never fill the whole device.

    An uncapped dit-step arm once reserved 48.3 of 49.1 GiB and starved the
    running backend. Over the cap the process OOMs instead, which is the
    intended failure: it means the configuration is too big, not that the GPU
    needs freeing.
    """
    if not torch.cuda.is_available():
        return {}
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    total = int(torch.cuda.get_device_properties(0).total_memory)
    return {
        "fraction": fraction,
        "device_total_bytes": total,
        "cap_bytes": int(total * fraction),
    }


def _cuda_memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize()
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }


def _rss_bytes() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def _tensor_bytes(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)

    def q(p: float) -> float:
        return ordered[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "min": ordered[0],
        "p50": q(0.5),
        "mean": sum(ordered) / n,
        "p90": q(0.9),
        "max": ordered[-1],
        "n": n,
    }


# ---------------------------------------------------------------- captions


def load_captions(caption_dir: str, count: int, seed: int) -> list[str]:
    """Sample real captions deterministically; caption LENGTH drives M1/M3."""
    directory = Path(caption_dir)
    files = sorted(p for p in directory.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt captions under {directory}")
    rng = random.Random(seed)
    chosen = rng.sample(files, min(count, len(files)))
    captions: list[str] = []
    for path in chosen:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            captions.append(text)
    if not captions:
        raise RuntimeError(f"every sampled caption under {directory} was empty")
    return captions


def caption_stats(captions: list[str], tokenizer=None) -> dict[str, Any]:
    chars = _percentiles([float(len(c)) for c in captions])
    out: dict[str, Any] = {"count": len(captions), "chars": chars}
    if tokenizer is not None:
        try:
            lengths = [
                float(len(tokenizer(c, add_special_tokens=False).input_ids))
                for c in captions
            ]
            out["tokens"] = _percentiles(lengths)
        except Exception as exc:  # tokenizer shapes vary; length is a nicety
            out["tokens_error"] = str(exc)
    return out


# ---------------------------------------------------------------- shim


class _ShimTrainer:
    """The attribute surface ``BaseTrainer.encode_caption`` actually touches.

    Lets the encode arms exercise the production encode code without paying for
    the DiT (krea2's is 26 GB; host RAM here is the binding constraint).
    """

    def __init__(self, arch: str, device: str, dtype: torch.dtype):
        for name in (
            "zimage", "sensenova", "lens", "ideogram4", "minit2i", "krea2",
            "anima", "ltx2", "minimax_h3", "acestep", "flux2", "sdxl", "sd15",
        ):
            setattr(self, f"is_{name}", False)
        setattr(self, f"is_{arch}", True)
        self.device = device
        self.training_dtype = dtype
        self.weight_dtype = dtype
        self.vae_dtype = dtype
        self.log_prefix = "[TEProbe]"
        self.config = {}
        self.text_encoder = None
        self.text_encoder_2 = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.t5_tokenizer = None
        self.transformer = None
        # encode_prompt's "TE is on CPU" warning is aimed at production
        # misconfiguration; cpu_prefetch is the sanctioned exception and this
        # probe measures exactly that case.
        self._text_encoding_mode = "cpu_prefetch"

    def _has_fp8_text_encoder(self) -> bool:
        return False


# ---------------------------------------------------------------- TE loaders


def load_text_encoder(arch: str, model_path: str, device: str,
                      dtype: torch.dtype) -> _ShimTrainer:
    shim = _ShimTrainer(arch, device, dtype)

    if arch == "anima":
        from core.models.anima.anima_loader import (
            discover_anima_components, load_qwen3_text_encoder, load_t5_tokenizer,
        )

        found = discover_anima_components(model_path)
        qwen3_path = found.get("text_encoder") or found.get("qwen3")
        if not qwen3_path:
            raise FileNotFoundError(f"anima Qwen3 text encoder not found near {model_path}: {found}")
        model, tokenizer = load_qwen3_text_encoder(qwen3_path, device=device, dtype=dtype)
        shim.text_encoder = model
        shim.tokenizer = tokenizer
        shim.t5_tokenizer = load_t5_tokenizer()
        return shim

    if arch == "krea2":
        from core.models.krea2 import krea2_loader

        te_dir = krea2_loader._resolve_te_dir(model_path, None)
        if not te_dir:
            raise FileNotFoundError(f"krea2 text encoder dir not resolvable from {model_path}")
        shim.text_encoder = krea2_loader._load_qwen3vl_text_encoder(te_dir, dtype).to(device)
        shim.tokenizer = krea2_loader._load_tokenizer(te_dir)
        shim.krea2_select_layers = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
        return shim

    if arch == "sdxl":
        from diffusers import StableDiffusionXLPipeline

        pipeline = StableDiffusionXLPipeline.from_single_file(
            model_path, torch_dtype=dtype, use_safetensors=True,
        )
        shim.text_encoder = pipeline.text_encoder.to(device).eval().requires_grad_(False)
        shim.text_encoder_2 = pipeline.text_encoder_2.to(device).eval().requires_grad_(False)
        shim.tokenizer = pipeline.tokenizer
        shim.tokenizer_2 = pipeline.tokenizer_2
        shim.sdxl_te_type = "none"
        pipeline.unet = None
        pipeline.vae = None
        del pipeline
        gc.collect()
        return shim

    if arch == "sensenova":
        # SenseNova has no separate TE: "encoding" is the 8B LLM prefix forward
        # that produces the KV cache the denoiser consumes, so the whole
        # checkpoint has to be resident either way.
        from core.attention import AttentionMode
        from core.models.sensenova.loader import load_sensenova_from_path
        from core.models.sensenova.sensenova_pipeline_ops import set_attention_backend

        components = load_sensenova_from_path(model_path, torch_dtype=dtype)
        transformer = components["transformer"].to(device)
        transformer.eval()
        set_attention_backend(transformer, "native", AttentionMode.TRAINING)
        shim.transformer = transformer
        shim.tokenizer = components["tokenizer"]
        return shim

    raise ValueError(f"unsupported arch: {arch}")


def build_encode_fns(shim: _ShimTrainer, arch: str) -> tuple[Callable, Callable]:
    """Return (encode_one, encode_batch) bound to the production code path."""
    from core.training.base_trainer import BaseTrainer

    if arch == "sensenova":
        from core.training.ops import sensenova_ops

        def encode_one(caption: str):
            return sensenova_ops.encode_prompt(shim, caption), None

        def encode_batch(captions, lyrics=None):
            return [encode_one(c) for c in captions]

        return encode_one, encode_batch

    def encode_one(caption: str):
        return BaseTrainer.encode_caption(shim, caption)

    def encode_batch(captions, lyrics=None):
        return BaseTrainer.encode_captions_batched(shim, list(captions), lyrics=lyrics)

    return encode_one, encode_batch


# ---------------------------------------------------------------- GPU busy loop


class _GpuBusyLoop:
    """Saturate the GPU from the calling thread, the way a training step does.

    Used to expose GIL contention: each iteration launches a handful of large
    matmuls (short GIL holds during launch) and periodically synchronizes.
    """

    def __init__(self, size: int = 4096, dtype: torch.dtype = torch.bfloat16,
                 launches_per_iter: int = 8):
        self.a = torch.randn(size, size, device="cuda", dtype=dtype)
        self.b = torch.randn(size, size, device="cuda", dtype=dtype)
        self.launches_per_iter = launches_per_iter
        self.iterations = 0

    def run_for(self, seconds: float, stop: threading.Event | None = None) -> dict[str, float]:
        started = time.perf_counter()
        iterations = 0
        while time.perf_counter() - started < seconds:
            if stop is not None and stop.is_set():
                break
            x = self.a
            for _ in range(self.launches_per_iter):
                x = torch.matmul(x, self.b)
            torch.cuda.synchronize()
            iterations += 1
        elapsed = time.perf_counter() - started
        self.iterations += iterations
        return {
            "iterations": iterations,
            "seconds": elapsed,
            "iters_per_second": iterations / elapsed if elapsed else 0.0,
        }

    def run_until(self, stop: threading.Event, max_seconds: float) -> dict[str, float]:
        started = time.perf_counter()
        iterations = 0
        while not stop.is_set() and time.perf_counter() - started < max_seconds:
            x = self.a
            for _ in range(self.launches_per_iter):
                x = torch.matmul(x, self.b)
            torch.cuda.synchronize()
            iterations += 1
        elapsed = time.perf_counter() - started
        return {
            "iterations": iterations,
            "seconds": elapsed,
            "iters_per_second": iterations / elapsed if elapsed else 0.0,
        }


# ---------------------------------------------------------------- encode arms


def _make_batches(captions: list[str], batch_size: int, batches: int):
    """Shape captions into the [(item, dataset), ...] groups the prefetcher eats."""
    out = []
    for i in range(batches):
        group = []
        for j in range(batch_size):
            caption = captions[(i * batch_size + j) % len(captions)]
            group.append(({"image_path": f"probe/{i}_{j}.png", "caption": caption}, None))
        out.append(group)
    return out


def _run_prefetch_worker(encode_batch, batches, prefetch_depth: int,
                         drain_callback: Callable[[], None] | None = None):
    """Drive the production CpuTextEncoderPrefetcher and return its per-batch stats.

    The main thread drains eagerly so the worker is never back-pressured; its
    ``cpu_encode_seconds`` is then pure encode wall.
    """
    from core.training.cpu_te_prefetch import CpuTextEncoderPrefetcher

    prefetcher = CpuTextEncoderPrefetcher(
        encode_batch_fn=encode_batch,
        batches=batches,
        prefetch_depth=prefetch_depth,
        log_prefix="[TEProbe][cpu_prefetch]",
    )
    per_batch: list[float] = []
    payload_bytes: list[int] = []
    # Wrap record_encode so we get the per-batch series, not just aggregates.
    original_record = prefetcher.stats.record_encode

    def record(seconds: float, n_samples: int) -> None:
        per_batch.append(seconds)
        original_record(seconds, n_samples)

    prefetcher.stats.record_encode = record  # type: ignore[method-assign]
    prefetcher.start()
    pulled = 0
    try:
        while pulled < len(batches):
            index, payload = prefetcher.next(timeout=1800.0)
            if index < 0:
                break
            pulled += 1
            if payload:
                first = next(iter(payload.values()))
                payload_bytes.append(_tensor_bytes(first[0]) + _tensor_bytes(first[1]))
            if drain_callback is not None:
                drain_callback()
    finally:
        prefetcher.stop()
    return prefetcher, per_batch, payload_bytes


def run_encode_arm(args: argparse.Namespace, contended: bool, device: str) -> dict[str, Any]:
    baseline_rss = _rss_bytes()
    dtype = torch.bfloat16
    load_started = time.perf_counter()
    shim = load_text_encoder(args.arch, args.model_path, device, dtype)
    load_seconds = time.perf_counter() - load_started
    resident_rss = _rss_bytes()
    encode_one, encode_batch = build_encode_fns(shim, args.arch)

    captions = load_captions(args.caption_dir, args.captions, args.seed)
    stats = caption_stats(captions, shim.tokenizer)

    total_batches = args.warmup + args.steps
    batches = _make_batches(captions, args.batch_size, total_batches)

    busy_solo = None
    busy_contended = None
    busy: _GpuBusyLoop | None = None
    if contended:
        busy = _GpuBusyLoop(size=args.busy_matmul_size)
        # Calibrate the GPU loop alone first, in this same process, so the
        # comparison isolates GIL contention rather than machine state.
        busy_solo = busy.run_for(args.busy_calibration_seconds)

    stop = threading.Event()
    busy_result: dict[str, float] = {}

    if contended:
        assert busy is not None

        def drain() -> None:
            return None

        # The GPU loop must run on the MAIN thread (as in training) while the
        # encode worker runs on the daemon thread, so spin the drain out instead.
        result_box: dict[str, Any] = {}

        def drain_thread() -> None:
            try:
                result_box["value"] = _run_prefetch_worker(
                    encode_batch, batches, args.prefetch_depth
                )
            except Exception as exc:  # surfaced below
                result_box["error"] = repr(exc)
            finally:
                stop.set()

        worker = threading.Thread(target=drain_thread, name="probe-drain", daemon=True)
        worker.start()
        busy_contended = busy.run_until(stop, max_seconds=args.contended_timeout_s)
        worker.join(timeout=args.contended_timeout_s)
        if "error" in result_box:
            raise RuntimeError(f"contended encode worker failed: {result_box['error']}")
        prefetcher, per_batch, payload_bytes = result_box["value"]
        busy_result = {"solo": busy_solo, "contended": busy_contended}
    else:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        prefetcher, per_batch, payload_bytes = _run_prefetch_worker(
            encode_batch, batches, args.prefetch_depth
        )

    measured = per_batch[args.warmup:]
    per_sample = [value / args.batch_size for value in measured]

    result: dict[str, Any] = {
        "arch": args.arch,
        "arm": args.arm,
        "device": device,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup,
        "measured_batches": len(measured),
        "prefetch_depth": args.prefetch_depth,
        "captions": stats,
        "model_load_seconds": load_seconds,
        "encode_seconds_per_batch": _percentiles(measured),
        "encode_seconds_per_sample": _percentiles(per_sample),
        "payload_bytes_per_sample": _percentiles([float(b) for b in payload_bytes]),
        "rss_bytes": {
            "baseline": baseline_rss,
            "te_resident": resident_rss,
            "delta": resident_rss - baseline_rss,
            "peak_after_encode": _rss_bytes(),
        },
        "worker_errors": list(prefetcher.stats.worker_errors),
        "torch_num_threads": torch.get_num_threads(),
    }
    if device == "cuda":
        result["cuda_peak"] = {
            "allocated": int(torch.cuda.max_memory_allocated()),
            "reserved": int(torch.cuda.max_memory_reserved()),
        }
        result["cuda_resident_after_load"] = _cuda_memory()
    if busy_result:
        result["gpu_busy_loop"] = busy_result
        result["gpu_busy_matmul_size"] = args.busy_matmul_size
    return result


def run_te_gpu_arm(args: argparse.Namespace) -> dict[str, Any]:
    """Same encode, TE on GPU; ``cuda_peak`` is the M4 occupancy figure."""
    if not torch.cuda.is_available():
        raise RuntimeError("te-gpu arm requires CUDA")
    gate = _apply_vram_gate(args.memory_fraction)
    result = run_encode_arm(args, contended=False, device="cuda")
    result["vram_gate"] = gate
    return result


# ---------------------------------------------------------------- dit-step arm


class _ProbeDataset:
    unique_id = "text-encode-vs-step-probe"

    def __init__(self, items: list[dict[str, Any]]):
        self.items = items
        self._served = False

    def reload_for_epoch(self, epoch_num: int, run_id: int | None = None):
        del run_id
        if epoch_num == 0 and not self._served:
            self._served = True
            return None
        return [dict(item) for item in self.items]


def _write_probe_images(workdir: Path, captions: list[str], size: int) -> list[dict[str, Any]]:
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(1234)
    items = []
    for index, caption in enumerate(captions):
        path = workdir / f"probe_{index:03d}.png"
        if not path.is_file():
            array = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
            Image.fromarray(array, "RGB").save(path, format="PNG")
        items.append({
            "image_path": str(path),
            "caption": caption,
            "width": size,
            "height": size,
            "dataset_unique_id": _ProbeDataset.unique_id,
        })
    return items


def run_dit_step_arm(args: argparse.Namespace) -> dict[str, Any]:
    """Time real training steps with text encoding taken off the step path.

    ``text_encoding_mode: pre_encoded_cache`` moves caption encoding into a
    one-off pre-pass, so the wall between consecutive ``progress_callback``
    invocations is the DiT forward+backward+optimizer step (M2).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("dit-step arm requires CUDA")
    vram_gate = _apply_vram_gate(args.memory_fraction)

    from core.training.lora_trainer import LoRATrainer

    workdir = Path(args.workdir or (REPO_ROOT / "tmp" / "text_encode_vs_step" / args.arch))
    workdir.mkdir(parents=True, exist_ok=True)
    captions = load_captions(args.caption_dir, args.captions, args.seed)
    image_dir = workdir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    items = _write_probe_images(image_dir, captions, args.resolution)

    train_config = {
        "gradient_checkpointing": True,
        "attention_backend": args.attention_backend,
        "use_flash_attention": args.attention_backend == "flash",
        "batch_size": args.batch_size,
        "blocks_to_swap": 0,
        "use_reference_images": False,
        "text_encoding_mode": "pre_encoded_cache",
        "latent_encoding_mode": "pre_encoded_cache",
        "noise_process": "auto",
        "prediction_target": "auto",
        "gradient_accumulation_steps": 1,
        "multi_noise_timesteps": 1,
    }
    trainer = LoRATrainer(
        model_path=args.model_path,
        output_dir=str(workdir / "output"),
        run_name="text_encode_vs_step",
        run_id=None,
        learning_rate=1e-5,
        device="cuda",
        train_config=dict(train_config),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dtype="bf16",
        weight_dtype="bf16",
        training_dtype="bf16",
        output_dtype="bf16",
        vae_dtype="bf16",
        mixed_precision=True,
        attention_backend=args.attention_backend,
        use_flash_attention=args.attention_backend == "flash",
        blocks_to_swap=0,
    )

    total_steps = args.warmup + args.steps
    marks: list[float] = []

    def progress_callback(phase, step, total, epoch=0, loss=None):
        del total, epoch, loss
        if phase != "training":
            return
        torch.cuda.synchronize()
        marks.append(time.perf_counter())

    torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    trainer.train(
        datasets=[_ProbeDataset(items)],
        num_epochs=args.epochs,
        total_steps=total_steps,
        batch_size=args.batch_size,
        save_every_n_steps=total_steps * 10,
        sample_every_n_steps=0,
        optimizer_type="adamw8bit",
        lr_scheduler_type="constant",
        enable_bucketing=False,
        base_resolutions=[args.resolution],
        gradient_accumulation_steps=1,
        multi_noise_timesteps=1,
        text_encoding_mode="pre_encoded_cache",
        latent_encoding_mode="pre_encoded_cache",
        use_reference_images=False,
        sample_prompts=[],
        sample_guidance_scale=1.0,
        sample_steps=1,
        sample_width=args.resolution,
        sample_height=args.resolution,
        sample_seed=args.seed,
        max_grad_norm=1.0,
        progress_callback=progress_callback,
        run_id=None,
        max_step_saves_to_keep=1,
        force_recache=False,
    )
    train_wall = time.perf_counter() - train_started

    deltas = [b - a for a, b in zip(marks, marks[1:])]
    measured = deltas[args.warmup:]
    result = {
        "arch": args.arch,
        "arm": "dit-step",
        "batch_size": args.batch_size,
        "resolution": args.resolution,
        "attention_backend": args.attention_backend,
        "gradient_checkpointing": True,
        "callbacks": len(marks),
        "warmup_steps": args.warmup,
        "step_seconds": _percentiles(measured),
        "train_wall_seconds": train_wall,
        "vram_gate": vram_gate,
        "cuda_peak": {
            "allocated": int(torch.cuda.max_memory_allocated()),
            "reserved": int(torch.cuda.max_memory_reserved()),
        },
        "rss_peak_bytes": _rss_bytes(),
    }
    try:
        trainer.writer.close()
    finally:
        trainer._db_executor.shutdown(wait=True)
    return result


# ---------------------------------------------------------------- sensenova


def run_sensenova_arm(args: argparse.Namespace) -> dict[str, Any]:
    """SenseNova sits outside the four text_encoding_modes.

    Its "text encode" is the 8B LLM prefix forward that yields the KV cache the
    denoiser consumes, so prefix wall and step wall are measured against the one
    resident model — which is also how production runs it.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("sensenova arm requires CUDA")
    vram_gate = _apply_vram_gate(args.memory_fraction)

    import torch.nn.functional as F
    from core.attention import AttentionMode
    from core.models.sensenova.loader import load_sensenova_from_path
    from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets
    from core.models.sensenova.sensenova_pipeline_ops import (
        _build_step_context, compute_noise_scale, set_attention_backend,
    )
    from core.training.adapters.sd15_adapter import LoRALinearLayer
    from core.training.ops import sensenova_ops
    from core.training.ops.sensenova_ops import forward_gen_decoder_layers

    baseline_rss = _rss_bytes()
    load_started = time.perf_counter()
    components = load_sensenova_from_path(args.model_path, torch_dtype=torch.bfloat16)
    transformer = components["transformer"]
    tokenizer = components["tokenizer"]

    targets = list(iter_sensenova_lora_targets(transformer))
    wrappers = {}
    for module_path, parent, attr, current in targets:
        wrapper = LoRALinearLayer(
            current, rank=args.lora_rank, alpha=args.lora_rank,
            lora_name=module_path, lora_dtype=torch.float32,
        )
        setattr(parent, attr, wrapper)
        wrappers[module_path] = wrapper
    transformer.to("cuda")
    transformer.train()
    set_attention_backend(transformer, "native", AttentionMode.TRAINING)
    load_seconds = time.perf_counter() - load_started
    resident_rss = _rss_bytes()
    model_resident = _cuda_memory()

    shim = _ShimTrainer("sensenova", "cuda", torch.bfloat16)
    shim.transformer = transformer
    shim.tokenizer = tokenizer

    captions = load_captions(args.caption_dir, args.captions, args.seed)
    stats = caption_stats(captions, tokenizer)

    # --- M1: prefix forward (SenseNova's text encode), one caption at a time.
    prefix_times: list[float] = []
    prefix_bytes: list[int] = []
    prefix_peak = {}
    prefix = None
    for index in range(args.warmup + args.steps):
        caption = captions[index % len(captions)]
        torch.cuda.synchronize()
        before = _cuda_memory()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        prefix = sensenova_ops.encode_prompt(shim, caption)
        torch.cuda.synchronize()
        prefix_times.append(time.perf_counter() - started)
        if index >= args.warmup:
            prefix_peak = {
                "allocated": int(torch.cuda.max_memory_allocated()),
                "reserved": int(torch.cuda.max_memory_reserved()),
                "delta_allocated": int(torch.cuda.memory_allocated() - before["allocated"]),
            }
            prefix_bytes.append(sum(
                _tensor_bytes(layer.keys) + _tensor_bytes(layer.values)
                for layer in prefix.cache.layers
            ))

    # --- M2: DiT step against the last prefix.
    size = args.resolution
    if size % 32:
        raise ValueError("SenseNova requires a /32 resolution")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x0 = torch.rand((1, 3, size, size), generator=generator, device=device,
                    dtype=torch.bfloat16).mul_(2).sub_(1)
    eps = torch.randn(x0.shape, generator=generator, device=device, dtype=torch.bfloat16)
    t = torch.tensor(0.5, device=device, dtype=torch.bfloat16)
    merge_size = int(1 / transformer.downsample_ratio)
    grid_h = grid_w = size // transformer.patch_size
    token_h, token_w = grid_h // merge_size, grid_w // merge_size
    noise_scale = compute_noise_scale(transformer, grid_h, grid_w, merge_size)
    x_t = t * x0 + (1 - t) * noise_scale * eps
    from types import SimpleNamespace

    prefix_shape = SimpleNamespace(
        batch_size=1, merge_size=merge_size, grid_h=grid_h, grid_w=grid_w,
        token_h=token_h, token_w=token_w,
    )
    z, image_embeds, _ = _build_step_context(transformer, prefix_shape, x_t, t, noise_scale)
    prefix_len = int(prefix.cache.get_seq_length())
    image_indexes = transformer._build_t2i_image_indexes(
        token_h, token_w, prefix_len, device=device,
    )
    x0_tokens = transformer.patchify(x0, transformer.patch_size * merge_size)
    patch = transformer.patch_size * merge_size

    parameters = [p for w in wrappers.values()
                  for p in (w.lora_down.weight, w.lora_up.weight)]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.0)

    step_times: list[float] = []
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(args.warmup + args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = forward_gen_decoder_layers(
                transformer.language_model.model, image_embeds,
                indexes=image_indexes, prefix_cache=prefix.cache,
                attention_mask=None, checkpoint_layers=True,
            )
            image_2d = hidden.view(1, token_h, token_w, -1).permute(0, 3, 1, 2)
            decoded = transformer.fm_modules["fm_head"](image_2d)
            x0_pred = (
                decoded.view(1, 3, token_h, patch, token_w, patch)
                .permute(0, 2, 4, 3, 5, 1).contiguous()
                .view(1, token_h * token_w, patch * patch * 3)
            )
            denominator = (1 - t).clamp_min(transformer.config.t_eps)
            loss = F.mse_loss(((x0_pred - z) / denominator).float(),
                              ((x0_tokens - z) / denominator).float())
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - started)
        losses.append(float(loss.detach().cpu()))

    measured_prefix = prefix_times[args.warmup:]
    measured_step = step_times[args.warmup:]
    return {
        "arch": "sensenova",
        "arm": "prefix-vs-step",
        "note": "SenseNova has no separate TE; M1 is the LLM prefix forward",
        "batch_size": 1,
        "resolution": size,
        "lora_rank": args.lora_rank,
        "lora_targets": len(wrappers),
        "gradient_checkpointing": True,
        "captions": stats,
        "vram_gate": vram_gate,
        "model_load_seconds": load_seconds,
        "prefix_seconds": _percentiles(measured_prefix),
        "prefix_seq_length": prefix_len,
        "prefix_kv_bytes_per_sample": _percentiles([float(b) for b in prefix_bytes]),
        "prefix_cuda_peak": prefix_peak,
        "step_seconds": _percentiles(measured_step),
        "losses_finite": all(v == v and abs(v) != float("inf") for v in losses),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "memory": {
            "model_resident_cuda": model_resident,
            "cuda_peak_allocated": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved": int(torch.cuda.max_memory_reserved()),
            "rss_baseline": baseline_rss,
            "rss_after_load": resident_rss,
            "rss_peak": _rss_bytes(),
        },
    }


# ------------------------------------------------- sensenova four-phase gate


def _leaf_prefix_cache(cache: Any) -> tuple[Any, list[torch.Tensor], list[torch.Tensor]]:
    """Cut the und/gen graph at the boundary KV.

    Returns a cache whose K/V are LEAVES that require grad, plus the graph-side
    originals in the same order, so a later
    ``autograd.backward(originals, grad_tensors=[leaf.grad ...])`` resumes the
    understanding backward exactly where the generation one stopped.
    """
    from core.training.ops.sensenova_ops import _TrainingPrefixCache, _TrainingPrefixLayer

    leaves: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []
    layers = []
    for layer in cache.layers:
        keys = layer.keys.detach().requires_grad_(True)
        values = layer.values.detach().requires_grad_(True)
        leaves.extend((keys, values))
        sources.extend((layer.keys, layer.values))
        layers.append(_TrainingPrefixLayer(keys, values))
    return _TrainingPrefixCache(layers), leaves, sources


def run_sensenova_four_phase_arm(args: argparse.Namespace) -> dict[str, Any]:
    """U-2-4's exit gate: is recomputing the und forward cheaper than residency?

    §8.3.2 splits one ``loss.backward()`` into ``prefix`` / ``denoise`` /
    ``und_backward``, and the third phase RECOMPUTES the understanding forward.
    The marginal cost of the split is therefore exactly one extra und prefix
    forward per step; the marginal cost of the eviction it enables is two weight
    round trips per step. Both are measured here against the single-backward
    reference, on the real checkpoint.

    Understanding-branch grad is supplied by LoRA (``branch="both"``), not by a
    materialized bf16 half: the wall times below are therefore for int8
    understanding Linears. A both-branch full fine-tune runs bf16 ones.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("sensenova-four-phase arm requires CUDA")
    vram_gate = _apply_vram_gate(args.memory_fraction)

    import torch.nn.functional as F
    from core.attention import AttentionMode
    from core.models.sensenova.loader import load_sensenova_from_path
    from core.models.sensenova.mot_cpu_staging import stage_modules_to_pinned_cpu
    from core.models.sensenova.mot_weight_selector import select_mot_weight_modules
    from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets
    from core.models.sensenova.sensenova_pipeline_ops import (
        _build_step_context, compute_noise_scale, set_attention_backend,
    )
    from core.training.adapters.sd15_adapter import LoRALinearLayer
    from core.training.ops import sensenova_ops
    from core.training.ops.sensenova_ops import forward_gen_decoder_layers
    from core.training.sensenova_phase_eviction import _move_modules_to_device

    baseline_rss = _rss_bytes()
    load_started = time.perf_counter()
    components = load_sensenova_from_path(args.model_path, torch_dtype=torch.bfloat16)
    transformer = components["transformer"]
    tokenizer = components["tokenizer"]

    wrappers = {}
    for module_path, parent, attr, current in list(
        iter_sensenova_lora_targets(transformer, branch="both")
    ):
        wrapper = LoRALinearLayer(
            current, rank=args.lora_rank, alpha=args.lora_rank,
            lora_name=module_path, lora_dtype=torch.float32,
        )
        setattr(parent, attr, wrapper)
        wrappers[module_path] = wrapper
    transformer.to("cuda")
    transformer.train()
    set_attention_backend(transformer, "native", AttentionMode.TRAINING)
    load_seconds = time.perf_counter() - load_started
    model_resident = _cuda_memory()

    shim = _ShimTrainer("sensenova", "cuda", torch.bfloat16)
    shim.transformer = transformer
    shim.tokenizer = tokenizer
    shim.gradient_checkpointing = True
    shim.train_text_encoder = True

    captions = load_captions(args.caption_dir, args.captions, args.seed)
    stats = caption_stats(captions, tokenizer)

    size = args.resolution
    if size % 32:
        raise ValueError("SenseNova requires a /32 resolution")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x0 = torch.rand((1, 3, size, size), generator=generator, device=device,
                    dtype=torch.bfloat16).mul_(2).sub_(1)
    eps = torch.randn(x0.shape, generator=generator, device=device, dtype=torch.bfloat16)
    t = torch.tensor(0.5, device=device, dtype=torch.bfloat16)
    merge_size = int(1 / transformer.downsample_ratio)
    grid_h = grid_w = size // transformer.patch_size
    token_h, token_w = grid_h // merge_size, grid_w // merge_size
    noise_scale = compute_noise_scale(transformer, grid_h, grid_w, merge_size)
    x_t = t * x0 + (1 - t) * noise_scale * eps
    from types import SimpleNamespace

    prefix_shape = SimpleNamespace(
        batch_size=1, merge_size=merge_size, grid_h=grid_h, grid_w=grid_w,
        token_h=token_h, token_w=token_w,
    )
    patch = transformer.patch_size * merge_size
    x0_tokens = transformer.patchify(x0, patch)
    parameters = [p for w in wrappers.values()
                  for p in (w.lora_down.weight, w.lora_up.weight)]

    def gen_loss(cache: Any, prefix_len: int, *, boundary_leaf: bool = False) -> torch.Tensor:
        z, image_embeds, _ = _build_step_context(
            transformer, prefix_shape, x_t, t, noise_scale
        )
        image_indexes = transformer._build_t2i_image_indexes(
            token_h, token_w, prefix_len, device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = forward_gen_decoder_layers(
                transformer.language_model.model, image_embeds,
                indexes=image_indexes, prefix_cache=cache,
                attention_mask=None, checkpoint_layers=True,
                trainable_prefix=not boundary_leaf,
                boundary_leaf_prefix=boundary_leaf,
            )
            decoded = transformer.fm_modules["fm_head"](
                hidden.view(1, token_h, token_w, -1).permute(0, 3, 1, 2).contiguous()
            )
            x0_pred = (
                decoded.view(1, 3, token_h, patch, token_w, patch)
                .permute(0, 2, 4, 3, 5, 1).contiguous()
                .view(1, token_h * token_w, patch * patch * 3)
            )
            denominator = (1 - t).clamp_min(transformer.config.t_eps)
            return F.mse_loss(((x0_pred - z) / denominator).float(),
                              ((x0_tokens - z) / denominator).float())

    def zero_grads() -> None:
        for parameter in parameters:
            parameter.grad = None

    def clock() -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    single_prefix: list[float] = []
    single_backward: list[float] = []
    split_prefix: list[float] = []
    split_denoise: list[float] = []
    split_recompute: list[float] = []
    split_und_backward: list[float] = []
    losses: list[float] = []
    prefix_len = 0
    torch.cuda.reset_peak_memory_stats()

    for index in range(args.warmup + args.steps):
        caption = captions[index % len(captions)]
        measured = index >= args.warmup

        # --- Reference: one prefix forward, then ONE backward through both halves.
        zero_grads()
        started = clock()
        prefix = sensenova_ops.encode_prompt(shim, caption, requires_grad=True)
        prefix_len = int(prefix.cache.get_seq_length())
        mark = clock()
        loss = gen_loss(prefix.cache, prefix_len)
        loss.backward()
        end = clock()
        if measured:
            single_prefix.append(mark - started)
            single_backward.append(end - mark)
            losses.append(float(loss.detach()))
        del loss, prefix

        # --- Four-phase: prefix / denoise / (recompute + und_backward).
        zero_grads()
        started = clock()
        prefix = sensenova_ops.encode_prompt(shim, caption, requires_grad=True)
        leaf_cache, leaves, sources = _leaf_prefix_cache(prefix.cache)
        mark_prefix = clock()
        loss = gen_loss(leaf_cache, prefix_len, boundary_leaf=True)
        loss.backward()
        kv_grads = [
            leaf.grad if leaf.grad is not None else torch.zeros_like(leaf)
            for leaf in leaves
        ]
        mark_denoise = clock()
        del loss, leaf_cache, leaves, sources, prefix
        recomputed = sensenova_ops.encode_prompt(shim, caption, requires_grad=True)
        recomputed_kv = [
            tensor
            for layer in recomputed.cache.layers
            for tensor in (layer.keys, layer.values)
        ]
        mark_recompute = clock()
        torch.autograd.backward(recomputed_kv, grad_tensors=kv_grads)
        end = clock()
        if measured:
            split_prefix.append(mark_prefix - started)
            split_denoise.append(mark_denoise - mark_prefix)
            split_recompute.append(mark_recompute - mark_denoise)
            split_und_backward.append(end - mark_recompute)
        del recomputed, recomputed_kv, kv_grads
    zero_grads()

    # --- Weight round trip: what half-eviction costs per phase boundary.
    selection = select_mot_weight_modules(transformer)
    und_modules = list(selection.und_modules)
    und_bytes = sum(
        sum(p.numel() * p.element_size() for p in m.parameters(recurse=False))
        + sum(
            b.numel() * b.element_size()
            for name, b in m._buffers.items()
            if b is not None and name not in m._non_persistent_buffers_set
        )
        for m in und_modules
    )
    warn_once: dict[str, bool] = {}
    d2h_times: list[float] = []
    h2d_times: list[float] = []
    # Warmed up like the timing loop above. Without this the FIRST iteration
    # carries the pinned-memory allocation inside stage_modules_to_pinned_cpu,
    # which torch's caching host allocator then pools for every later transfer:
    # unwarmed, the mean landed 57% above the p50 sum and made the round trip
    # look like 1.046 s when the steady-state cost is 0.667 s.
    for index in range(args.warmup + args.steps):
        started = clock()
        stage_modules_to_pinned_cpu(und_modules, warn_once=warn_once)
        mark = clock()
        _move_modules_to_device(und_modules, device)
        finished = clock()
        if index >= args.warmup:
            h2d_times.append(finished - mark)
            d2h_times.append(mark - started)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    def p50(values: list[float]) -> float:
        return _percentiles(values).get("p50", float("nan"))

    # Every ratio is reported on BOTH statistics, never mixed. Quoting a p50
    # component into a mean-derived total is how the first write-up of this arm
    # got a round trip of 1.046 s out of components summing to 0.667 s.
    def totals(stat):
        single = stat(single_prefix) + stat(single_backward)
        split = (
            stat(split_prefix) + stat(split_denoise)
            + stat(split_recompute) + stat(split_und_backward)
        )
        trip = stat(d2h_times) + stat(h2d_times)
        return {
            "single_backward_total": single,
            "four_phase_total": split,
            "prefix_over_step": stat(single_prefix) / stat(single_backward),
            "prefix_over_single_backward_total": stat(single_prefix) / single,
            "four_phase_over_single_backward": split / single,
            "recompute_over_single_backward": stat(split_recompute) / single,
            "weight_round_trip": trip,
            "eviction_overhead_seconds_per_step": 2 * trip,
            "eviction_overhead_over_single_backward": 2 * trip / single,
            # The decomposition identity: the split's denoise + und_backward
            # should reproduce the single backward's one pass through both
            # halves. A drifting value here means the phases are not measuring
            # what their names say.
            "decomposition_residual_frac": (
                (stat(split_denoise) + stat(split_und_backward))
                / stat(single_backward) - 1.0
            ),
        }

    by_statistic = {"mean": totals(mean), "p50": totals(p50)}
    single_total = by_statistic["mean"]["single_backward_total"]
    split_total = by_statistic["mean"]["four_phase_total"]
    round_trip = by_statistic["mean"]["weight_round_trip"]
    return {
        "arch": "sensenova",
        "arm": "sensenova-four-phase",
        "note": "U-2-4 exit gate; und grad supplied by LoRA over int8 und Linears",
        "batch_size": 1,
        "resolution": size,
        "image_tokens": token_h * token_w,
        "lora_rank": args.lora_rank,
        "lora_targets": len(wrappers),
        "gradient_checkpointing": True,
        "captions": stats,
        "prefix_seq_length": prefix_len,
        "vram_gate": vram_gate,
        "model_load_seconds": load_seconds,
        "losses_finite": all(v == v and abs(v) != float("inf") for v in losses),
        "single_backward_seconds": {
            "prefix_forward": _percentiles(single_prefix),
            "gen_forward_and_backward": _percentiles(single_backward),
            "total_mean": single_total,
        },
        "four_phase_seconds": {
            "prefix_forward": _percentiles(split_prefix),
            "denoise_forward_and_backward": _percentiles(split_denoise),
            "und_recompute_forward": _percentiles(split_recompute),
            "und_backward": _percentiles(split_und_backward),
            "total_mean": split_total,
        },
        "ratios_by_statistic": by_statistic,
        "ratios": {
            "statistic": "mean",
            "prefix_over_step": mean(single_prefix) / mean(single_backward),
            "prefix_over_single_backward_total": mean(single_prefix) / single_total,
            "four_phase_over_single_backward": split_total / single_total,
            "recompute_over_single_backward": mean(split_recompute) / single_total,
        },
        "weight_round_trip": {
            "und_half_bytes": und_bytes,
            "d2h_seconds": _percentiles(d2h_times),
            "h2d_seconds": _percentiles(h2d_times),
            "round_trip_seconds": round_trip,
            "round_trips_per_step": 2,
            "eviction_overhead_seconds_per_step": 2 * round_trip,
            "eviction_overhead_over_single_backward": 2 * round_trip / single_total,
        },
        "memory": {
            "model_resident_cuda": model_resident,
            "cuda_peak_allocated": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved": int(torch.cuda.max_memory_reserved()),
            "rss_baseline": baseline_rss,
            "rss_peak": _rss_bytes(),
        },
    }


# ---------------------------------------------------------------- entry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=ARCHS)
    parser.add_argument("--arm", required=True, choices=ARMS + SENSENOVA_ARMS)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--caption-dir", default=DEFAULT_CAPTION_DIR)
    parser.add_argument("--captions", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prefetch-depth", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--attention-backend", default="native")
    parser.add_argument(
        "--memory-fraction", type=float, default=0.72,
        help="Hard per-process VRAM cap; see _apply_vram_gate.",
    )
    parser.add_argument("--busy-matmul-size", type=int, default=4096)
    parser.add_argument("--busy-calibration-seconds", type=float, default=10.0)
    parser.add_argument("--contended-timeout-s", type=float, default=1800.0)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATHS[args.arch]
    return args


def main() -> int:
    _require_repo_venv()
    args = _parse_args()
    if args.arm == "sensenova-four-phase":
        if args.arch != "sensenova":
            raise ValueError("the four-phase arm is SenseNova-only")
        result = run_sensenova_four_phase_arm(args)
    elif args.arch == "sensenova" and args.arm in ("te-gpu", "dit-step", "sensenova"):
        result = run_sensenova_arm(args)
    elif args.arm == "te-cpu":
        result = run_encode_arm(args, contended=False, device="cpu")
    elif args.arm == "te-cpu-contended":
        result = run_encode_arm(args, contended=True, device="cpu")
    elif args.arm == "te-gpu":
        result = run_te_gpu_arm(args)
    elif args.arm == "dit-step":
        result = run_dit_step_arm(args)
    else:
        raise ValueError(f"unsupported arm/arch combination: {args.arm}/{args.arch}")

    payload = json.dumps(result, indent=2, sort_keys=True, default=str)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
