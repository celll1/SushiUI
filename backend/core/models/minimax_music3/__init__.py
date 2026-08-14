"""MiniMax Music 3: lyrics- and caption-conditioned music generation.

Phase 1 (this package's current scope) is vendor + port only: the model
modules under ``vendor/`` and the plain-class ``MiniMaxMusic3Pipeline`` in
``pipeline.py``. The loader, the pipeline-backend wiring, the API routes and
the frontend are later commits -- see ``docs/guides/MINIMAX_MUSIC3_DESIGN.md``,
"Phase plan".
"""

from core.models.minimax_music3.pipeline import (
    MiniMaxMusic3ARResult,
    MiniMaxMusic3GenerationResult,
    MiniMaxMusic3Pipeline,
)

__all__ = [
    "MiniMaxMusic3ARResult",
    "MiniMaxMusic3GenerationResult",
    "MiniMaxMusic3Pipeline",
]
