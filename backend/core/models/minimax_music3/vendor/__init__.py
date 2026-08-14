"""Vendored MiniMax Music 3 model code (Apache-2.0, The MiniMax Team / The HuggingFace Team).

Source: https://github.com/huggingface/diffusers, PR #14456 "Add MiniMax Music 3",
commit ``dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d`` (unmerged; targets the
0.40.0.dev0 modular framework). Vendored rather than depending on the pinned
diffusers 0.38.0 build having this architecture, per
``docs/guides/MINIMAX_MUSIC3_DESIGN.md``, "Dependency gate". See each module's
own header for its exact upstream path and the SushiUI modifications applied
(mainly: absolute imports, and attention re-pointed at
``core.attention.dispatch_attention``).

The pipeline blocks (``modular_pipelines/minimax_music3/*.py`` upstream) are
NOT vendored here -- they depend on the 0.40-only modular framework and are
instead reimplemented as a plain pipeline class in
``core.models.minimax_music3.pipeline``.
"""

from .condition_embedder_minimax_music3 import MiniMaxMusic3ConditionEncoder
from .minimax_music3_rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder
from .minimax_music3_vocoder import MiniMaxMusic3Vocoder
from .transformer_minimax_music3 import MiniMaxMusic3Transformer1DModel

__all__ = [
    "MiniMaxMusic3ConditionEncoder",
    "MiniMaxMusic3RVQDepthDecoder",
    "MiniMaxMusic3Vocoder",
    "MiniMaxMusic3Transformer1DModel",
]
