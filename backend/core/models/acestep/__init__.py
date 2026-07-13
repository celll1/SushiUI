"""ACE-Step 1.5 model package (2B DiT + Oobleck VAE + Qwen3-Embedding-0.6B).

Phase 0+1 (foundation): vendored DiT modeling code + component loader only.
No sampler / generation pipeline yet (Phase 2).
"""

from .loader import detect_acestep_layout, load_acestep_from_path

__all__ = ["detect_acestep_layout", "load_acestep_from_path"]
