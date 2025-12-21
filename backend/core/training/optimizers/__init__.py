"""
Training optimizers module
"""

from .adafactor_fused import patch_adafactor_fused

__all__ = ["patch_adafactor_fused"]
