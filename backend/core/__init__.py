# Core module
# Import pipeline only when explicitly needed to avoid circular dependencies
# Use: from core.pipeline import pipeline_manager

from .transformers_clip_compat import install as _install_clip_compat

_install_clip_compat()

__all__ = []
