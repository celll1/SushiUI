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
from .head import EvidenceHeadOutput, EvidenceSlotMixer, EvidenceTagHead
from .encoder import FoveaEncoder, FoveaEncoderConfig, FoveaEncoderOutput
from .model import FoveaTagger, FoveaTaggerOutput
from .objectives import (
    FoveaSelfSupervisedLoss,
    SelfSupervisedLossOutput,
    chroma_reconstruction_loss,
    masked_patch_reconstruction_loss,
    overlap_correspondence_loss,
    rgb_to_chroma,
)
from .siglip2_control import (
    MaskedSlotPooler,
    SigLIP2ControlOutput,
    SigLIP2ControlTagger,
    SigLIP2ViewInputs,
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
    "EvidenceHeadOutput",
    "EvidenceSlotMixer",
    "EvidenceTagHead",
    "FoveaEncoder",
    "FoveaEncoderConfig",
    "FoveaEncoderOutput",
    "FoveaTagger",
    "FoveaTaggerOutput",
    "FoveaSelfSupervisedLoss",
    "SelfSupervisedLossOutput",
    "chroma_reconstruction_loss",
    "masked_patch_reconstruction_loss",
    "overlap_correspondence_loss",
    "rgb_to_chroma",
    "MaskedSlotPooler",
    "SigLIP2ControlOutput",
    "SigLIP2ControlTagger",
    "SigLIP2ViewInputs",
    "SourceBox",
    "ViewBatch",
    "ViewConfig",
    "ViewSpec",
    "build_views",
    "sample_local_specs",
]
