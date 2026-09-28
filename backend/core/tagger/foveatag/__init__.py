"""FoveaTag high-resolution tagger building blocks."""

from .benchmark import (
    BENCHMARK_FORMAT_VERSION,
    PREDICTION_FORMAT_VERSION,
    BenchmarkManifest,
    EvidenceBox,
    PredictionSet,
    StageTimer,
    evaluate_predictions,
    load_benchmark_manifest,
    load_prediction_set,
)
from .views import SourceBox, ViewBatch, ViewConfig, ViewSpec, build_views, sample_local_specs

__all__ = [
    "BENCHMARK_FORMAT_VERSION",
    "PREDICTION_FORMAT_VERSION",
    "BenchmarkManifest",
    "EvidenceBox",
    "PredictionSet",
    "StageTimer",
    "evaluate_predictions",
    "load_benchmark_manifest",
    "load_prediction_set",
    "SourceBox",
    "ViewBatch",
    "ViewConfig",
    "ViewSpec",
    "build_views",
    "sample_local_specs",
]
