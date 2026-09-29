"""GPU smoke and timing benchmark for Phase-1 control optimization."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from ..siglip2_inference_manager import SigLIP2InferenceManager
from ..tagger_loss import AsymmetricLossOptimized
from .control_training import SigLIP2ControlTrainer
from .pretraining_benchmark import registered_image_paths
from .siglip2_control import SigLIP2ControlTagger
from .siglip2_control_benchmark import _sample
from .siglip2_control_data import collate_siglip2_control_samples
from .siglip2_patch_benchmark import timing_summary


def benchmark_control_training(
    checkpoint: Path | str,
    vocabulary: Path | str,
    image_paths: Sequence[str],
    *,
    local_views: int,
    warmup_steps: int,
    measured_steps: int,
    overview_patch_budget: int = 784,
    local_patch_budget: int = 256,
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("image_paths must not be empty")
    if local_views < 0 or warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("invalid benchmark dimensions")
    manager = SigLIP2InferenceManager()
    manager.load_model(str(checkpoint), str(vocabulary))
    is_naflex = manager.is_naflex

    original = manager.model
    encoder = original.vision_encoder
    hidden_size = int(encoder.config.hidden_size)
    num_tags = len(manager.vocabulary["idx_to_tag"])
    processor = manager.processor
    manager.model = None
    del original
    gc.collect()
    torch.cuda.empty_cache()

    model = SigLIP2ControlTagger(
        encoder,
        hidden_size=hidden_size,
        feature_dim=512,
        num_tags=num_tags,
        evidence_slots_per_view=2,
        is_naflex=is_naflex,
        freeze_encoder=True,
    ).to("cuda")
    criterion = AsymmetricLossOptimized()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4,
        weight_decay=1e-4,
    )
    trainer = SigLIP2ControlTrainer(
        model,
        criterion,
        optimizer,
        device="cuda",
        mixed_precision=torch.bfloat16,
    )

    batches = []
    for index in range(max(warmup_steps + measured_steps, 1)):
        sample = _sample(image_paths[index % len(image_paths)], local_views, index)
        sample = type(sample)(
            views=sample.views,
            label=torch.zeros(num_tags),
            loss_mask=torch.ones(num_tags),
        )
        sample.label[0] = 1.0
        batch = collate_siglip2_control_samples(
            [sample],
            processor=processor,
            is_naflex=is_naflex,
            overview_patch_budget=overview_patch_budget,
            local_patch_budget=local_patch_budget,
        )
        assert batch is not None
        batches.append(batch)

    try:
        for index in range(warmup_steps):
            trainer.train_epoch([batches[index]])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        losses = []
        for index in range(warmup_steps, warmup_steps + measured_steps):
            started = time.perf_counter()
            result = trainer.train_epoch([batches[index]])
            torch.cuda.synchronize()
            durations.append((time.perf_counter() - started) * 1000.0)
            losses.append(result.mean_loss)
        return {
            "format_version": 1,
            "kind": "foveatag_siglip2_control_training",
            "checkpoint": str(Path(checkpoint).resolve()),
            "device": torch.cuda.get_device_properties(0).name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "dtype": "bfloat16",
            "num_tags": num_tags,
            "local_views": local_views,
            "overview_patch_budget": overview_patch_budget,
            "local_patch_budget": local_patch_budget,
            "encoder_input_contract": "naflex" if is_naflex else "fixed",
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "step_ms": timing_summary(durations),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "scope": "Prebuilt real-image views; includes transfer, frozen encoder forward, masked ASL backward, clipping, and AdamW update.",
        }
    finally:
        del trainer, optimizer, criterion, model
        manager.unload()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--datasets-db", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--local-views", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    paths, _ = registered_image_paths(
        args.datasets_db,
        args.dataset_id,
        limit=args.sample_count,
        minimum_side=256,
        selection="hashed",
        seed=20260929,
    )
    report = benchmark_control_training(
        args.checkpoint,
        args.vocabulary,
        paths,
        local_views=args.local_views,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
