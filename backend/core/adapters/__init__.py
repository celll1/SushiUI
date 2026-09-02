"""Architecture-neutral adapter mechanism, shared by training and generation.

Deliberately OUTSIDE ``core.training``. Importing ``core.training.adapters``
executes ``core/training/__init__.py``, which reaches ``api.param_defaults``
and transitively ``api.routes`` -- a ``core -> api`` back-edge that costs about
8.9 s and a CUDA context in a fresh process, and which the twelve generation
modules importing the leaf layer class were all paying. Nothing under this
package may import ``core.training`` or ``api`` at module scope;
``backend/tests/adapter_layering_test.py`` is the gate.

The old ``core.training.adapters`` paths re-export these names during Phase 1;
see ``docs/guides/LYCORIS_ADAPTER_DESIGN.md``.
"""

from .layers import LoRALinearLayer, MiniMaxH3LoRALinearLayer
from .targets import (
    count_quantized_linears,
    is_lora_wrappable_linear,
    lora_branch_dtype,
)

__all__ = [
    "LoRALinearLayer",
    "MiniMaxH3LoRALinearLayer",
    "count_quantized_linears",
    "is_lora_wrappable_linear",
    "lora_branch_dtype",
]
