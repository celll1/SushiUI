"""Execution-backend selection for adapter algebra.

Four boundaries, shaped after ``core/attention/``:

    registry.py   frozen capability descriptors + the callable per backend
    probe.py      the EXECUTED per-region check against the fp32 oracle
    dispatch.py   the conduit every branch's ``forward_delta`` goes through,
                  the process latch, and warm-up
    selection.py  the name vocabulary; explicit, off by default, never auto

Only ``reference`` -- the shipped unfused PyTorch path -- is registered. See
``docs/guides/LYCORIS_ADAPTER_DESIGN.md`` phase 4 for what registering a fused
backend requires, and for what this build deliberately does not claim about it.
"""

from .dispatch import (BACKEND_UNAVAILABLE_CODE, LATCH_CODE, WarmUpReport,
                       active_backend, active_backend_name,
                       adapter_forward_delta, is_latched, latch_off,
                       latched_reason, reset_execution_state,
                       set_active_backend, warm_up_adapter_backend)
from .probe import (ORACLE_TOLERANCE, PROBE_BATCH, PROBE_FACTOR_STD,
                    PROBE_MIN_MOVE, PROBE_ORACLE_BUDGET_BYTES, AdapterRegion,
                    ProbeResult, cached_verdict, clear_probe_cache,
                    probe_region, probed_regions, region_for, region_of)
from .registry import (BACKENDS, REFERENCE, AdapterBackend, declared_support,
                       reference_backend)
from .selection import (BACKEND_ENV_VAR, apply_configured_backend,
                        backend_refusal, configured_adapter_backend,
                        known_adapter_backends, select_adapter_backend,
                        selected_adapter_backend, validate_adapter_backend)

__all__ = [
    "BACKENDS",
    "BACKEND_ENV_VAR",
    "BACKEND_UNAVAILABLE_CODE",
    "LATCH_CODE",
    "ORACLE_TOLERANCE",
    "PROBE_BATCH",
    "PROBE_FACTOR_STD",
    "PROBE_MIN_MOVE",
    "PROBE_ORACLE_BUDGET_BYTES",
    "REFERENCE",
    "AdapterBackend",
    "AdapterRegion",
    "ProbeResult",
    "WarmUpReport",
    "active_backend",
    "active_backend_name",
    "adapter_forward_delta",
    "apply_configured_backend",
    "backend_refusal",
    "cached_verdict",
    "clear_probe_cache",
    "configured_adapter_backend",
    "declared_support",
    "is_latched",
    "known_adapter_backends",
    "latch_off",
    "latched_reason",
    "probe_region",
    "probed_regions",
    "reference_backend",
    "region_for",
    "region_of",
    "reset_execution_state",
    "select_adapter_backend",
    "selected_adapter_backend",
    "set_active_backend",
    "validate_adapter_backend",
    "warm_up_adapter_backend",
]
