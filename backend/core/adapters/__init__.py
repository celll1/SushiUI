"""Architecture-neutral adapter mechanism, shared by training and generation.

Deliberately OUTSIDE ``core.training``. Importing ``core.training.adapters``
executes ``core/training/__init__.py``, which reaches ``api.param_defaults``
and transitively ``api.routes`` -- a ``core -> api`` back-edge that costs about
8.9 s and a CUDA context in a fresh process, and which the twelve generation
modules importing the leaf layer class were all paying. Nothing under this
package may import ``core.training`` or ``api`` at module scope;
``backend/tests/adapter_layering_test.py`` is the gate.

Every importer -- generation, training and tests -- reaches these names here;
see ``docs/guides/LYCORIS_ADAPTER_DESIGN.md``.
"""

from .layers import (
    TUCKER_TENSOR_NAMES,
    CompositeAdapterLayer,
    DoRALinearLayer,
    LoHaLinearLayer,
    LoKrLinearLayer,
    LoRALinearLayer,
    MiniMaxH3LoRALinearLayer,
    count_adapter_wrapper_roots,
    dora_magnitude_axis,
    factorization,
    get_module_slot,
    is_adapter_covered,
    is_adapter_wrapper,
    named_modules_outside_adapters,
    refuse_tucker_tensors,
    set_module_slot,
)
from .session import (
    SHAPE_MISMATCH,
    AdapterComponent,
    AdapterFile,
    AdapterFileMissing,
    AdapterIncompatible,
    AdapterLoadFailed,
    AdapterLoadResult,
    AdapterRefusal,
    AdapterSession,
    ApplyCounts,
    BranchRequest,
    PreparedBranch,
)
from .codec import (
    CodecRegistry,
    CodecSpec,
    detect_adapter_codec,
    normalize_adapter_keys,
)
from .spec import (
    ADAPTER_SCHEMA_VERSION,
    ALGORITHMS,
    FORMATS,
    KNOWN_ARCHITECTURES,
    AdapterSpec,
)
from .targets import (
    AdapterTarget,
    count_quantized_linears,
    enumerate_adapter_targets,
    is_lora_wrappable_linear,
    lora_branch_dtype,
    quantization_kind,
)

__all__ = [
    "SHAPE_MISMATCH",
    "AdapterComponent",
    "AdapterFile",
    "AdapterFileMissing",
    "AdapterIncompatible",
    "AdapterLoadFailed",
    "AdapterLoadResult",
    "AdapterRefusal",
    "AdapterSession",
    "ApplyCounts",
    "BranchRequest",
    "PreparedBranch",
    "TUCKER_TENSOR_NAMES",
    "CompositeAdapterLayer",
    "DoRALinearLayer",
    "LoHaLinearLayer",
    "LoKrLinearLayer",
    "LoRALinearLayer",
    "MiniMaxH3LoRALinearLayer",
    "count_adapter_wrapper_roots",
    "dora_magnitude_axis",
    "factorization",
    "get_module_slot",
    "is_adapter_covered",
    "is_adapter_wrapper",
    "named_modules_outside_adapters",
    "refuse_tucker_tensors",
    "set_module_slot",
    "AdapterTarget",
    "count_quantized_linears",
    "enumerate_adapter_targets",
    "is_lora_wrappable_linear",
    "lora_branch_dtype",
    "quantization_kind",
    "ADAPTER_SCHEMA_VERSION",
    "ALGORITHMS",
    "FORMATS",
    "KNOWN_ARCHITECTURES",
    "AdapterSpec",
    "CodecRegistry",
    "CodecSpec",
    "detect_adapter_codec",
    "normalize_adapter_keys",
]
