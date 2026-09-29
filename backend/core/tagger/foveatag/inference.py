"""Staged overview -> route -> local inference for FoveaTag."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from PIL import Image

from .artifact import FoveaArtifactConfig
from .batch import collate_view_batches
from .model import FoveaTagger
from .router import (
    EvidenceRouter,
    build_scale_space_candidates,
    select_proposals,
    source_candidate_mask,
)
from .views import (
    RegionImageSource,
    SourceBox,
    ViewBatch,
    ViewConfig,
    build_views,
    local_spec,
    render_source_view,
)


@dataclass(frozen=True)
class FoveaInferenceResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    evidence_boxes: Tuple[Optional[SourceBox], ...]
    selected_boxes: Tuple[SourceBox, ...]
    stage_milliseconds: Dict[str, float]


def source_field_sizes(
    image_width: int,
    image_height: int,
    local_canvas_size: int,
    pixels_per_canvas_pixel: Sequence[float],
) -> Tuple[Tuple[float, float], ...]:
    """Convert density levels to normalized square-in-source-pixels fields."""
    if min(image_width, image_height, local_canvas_size) <= 0:
        raise ValueError("image and canvas dimensions must be positive")
    if not pixels_per_canvas_pixel or any(level <= 0 for level in pixels_per_canvas_pixel):
        raise ValueError("pixels_per_canvas_pixel levels must be positive")
    return tuple(
        (
            local_canvas_size * level / image_width,
            local_canvas_size * level / image_height,
        )
        for level in pixels_per_canvas_pixel
    )


@torch.inference_mode()
def predict_image(
    image: Image.Image | RegionImageSource,
    model: FoveaTagger,
    router: EvidenceRouter,
    config: FoveaArtifactConfig,
    *,
    mode: str = "balanced",
    nms_iou: float = 0.5,
    mixed_precision: Optional[torch.dtype] = None,
) -> FoveaInferenceResult:
    """Run the three dependent stages and report their wall-clock costs separately."""
    if mode not in config.inference_modes:
        raise ValueError(f"unknown FoveaTag inference mode: {mode}")
    mode_config = config.inference_modes[mode]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model_was_training = model.training
    router_was_training = router.training
    model.eval()
    router.eval()
    timings: Dict[str, float] = {}
    total_started = time.perf_counter()
    try:
        overview_started = time.perf_counter()
        view_config = ViewConfig(
            overview_size=mode_config.overview_size,
            local_size=config.views.local_size,
            local_views=0,
            field_levels=config.views.field_levels,
            scale_jitter=config.views.scale_jitter,
            pad_color=config.views.pad_color,
        )
        overview_views = build_views(image, view_config, seed=0, training=False)
        overview_batch = collate_view_batches([overview_views]).to(device, dtype=dtype)
        with torch.autocast(
            device_type=device.type,
            dtype=mixed_precision,
            enabled=device.type == "cuda" and mixed_precision is not None,
        ):
            overview = model.encoder(
                overview_batch.overview_images,
                overview_batch.overview_geometry,
            )
        _synchronize(device)
        timings["overview"] = _elapsed_ms(overview_started)

        router_started = time.perf_counter()
        field_sizes = source_field_sizes(
            image.width,
            image.height,
            config.views.local_size,
            config.views.field_levels,
        )
        candidates = build_scale_space_candidates(overview.local_coordinates, field_sizes)
        candidate_mask = source_candidate_mask(candidates)
        if mode_config.local_views:
            with torch.autocast(
                device_type=device.type,
                dtype=mixed_precision,
                enabled=device.type == "cuda" and mixed_precision is not None,
            ):
                routing = router(
                    overview.local_map,
                    overview.local_coordinates,
                    candidates,
                    candidate_mask,
                )
            selected = select_proposals(
                routing.scores,
                candidates,
                budget=mode_config.local_views,
                nms_iou=nms_iou,
                candidate_mask=candidate_mask,
            )[0]
            selected = selected[selected >= 0]
        else:
            selected = torch.empty(0, dtype=torch.long, device=device)
        _synchronize(device)
        timings["router"] = _elapsed_ms(router_started)

        local_started = time.perf_counter()
        local_specs = tuple(
            _candidate_to_spec(candidates[0, index], image.width, image.height, config.views.local_size)
            for index in selected.tolist()
        )
        local_views = tuple(
            render_source_view(image, spec, config.views.pad_color) for spec in local_specs
        )
        staged_views = ViewBatch(
            overview=overview_views.overview,
            overview_spec=overview_views.overview_spec,
            local_views=local_views,
            local_specs=local_specs,
            local_canvas_size=config.views.local_size,
        )
        local_batch = collate_view_batches([staged_views]).to(device, dtype=dtype)
        with torch.autocast(
            device_type=device.type,
            dtype=mixed_precision,
            enabled=device.type == "cuda" and mixed_precision is not None,
        ):
            details = model.classify_encoded_overview(
                overview,
                local_batch.local_images,
                local_batch.local_geometry,
                local_view_mask=local_batch.local_view_mask,
                return_details=True,
            )
        _synchronize(device)
        timings["local_and_head"] = _elapsed_ms(local_started)
        evidence_boxes = _tag_evidence_boxes(details, local_specs)
        timings["total"] = _elapsed_ms(total_started)
        return FoveaInferenceResult(
            logits=details.logits[0].detach().cpu(),
            probabilities=details.logits[0].sigmoid().detach().cpu(),
            evidence_boxes=evidence_boxes,
            selected_boxes=tuple(spec.source_box for spec in local_specs),
            stage_milliseconds=timings,
        )
    finally:
        model.train(model_was_training)
        router.train(router_was_training)


def _candidate_to_spec(
    candidate: torch.Tensor,
    image_width: int,
    image_height: int,
    canvas_size: int,
):
    center_x = float(candidate[0].item()) * image_width
    center_y = float(candidate[1].item()) * image_height
    source_width = float(candidate[2].item()) * image_width
    source_height = float(candidate[3].item()) * image_height
    if not math_isclose(source_width, source_height):
        raise ValueError("router candidate must describe a square in source pixels")
    return local_spec(
        center_x=center_x,
        center_y=center_y,
        canvas_size=canvas_size,
        pixels_per_canvas_pixel=(source_width + source_height) / (2.0 * canvas_size),
        role="routed-local",
    )


def _tag_evidence_boxes(details, local_specs) -> Tuple[Optional[SourceBox], ...]:
    best_slots = details.head_output.best_evidence_indices[0]
    if details.evidence_view_indices.shape[1] == 0:
        return tuple(None for _ in best_slots)
    safe_slots = best_slots.clamp_min(0)
    view_indices = details.evidence_view_indices[0].gather(0, safe_slots)
    return tuple(
        local_specs[int(view_index)].source_box if slot >= 0 and view_index >= 0 else None
        for slot, view_index in zip(best_slots.tolist(), view_indices.tolist())
    )


def math_isclose(first: float, second: float) -> bool:
    tolerance = max(abs(first), abs(second), 1.0) * 1e-5
    return abs(first - second) <= tolerance


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
