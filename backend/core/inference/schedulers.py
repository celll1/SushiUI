"""Sampler and Scheduler management for Stable Diffusion pipelines"""

import inspect

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    UniPCMultistepScheduler,
)

# Available samplers (algorithms) mapped to their scheduler classes
SAMPLER_MAP = {
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
    "dpmpp_sde": DPMSolverSinglestepScheduler,
    "dpm2": KDPM2DiscreteScheduler,
    "dpm2_a": KDPM2AncestralDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "pndm": PNDMScheduler,
    "lms": LMSDiscreteScheduler,
    "unipc": UniPCMultistepScheduler,
}

# Human-readable sampler names
SAMPLER_NAMES = {
    "euler": "Euler",
    "euler_a": "Euler a",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_sde": "DPM++ SDE",
    "dpm2": "DPM2",
    "dpm2_a": "DPM2 a",
    "heun": "Heun",
    "ddim": "DDIM",
    "ddpm": "DDPM",
    "pndm": "PNDM",
    "lms": "LMS",
    "unipc": "UniPC",
}

# Schedule types (noise scheduling strategies)
SCHEDULE_TYPES = {
    "uniform": "Uniform",
    "karras": "Karras",
    # "exponential" selects trailing timestep spacing, not an exponential sigma
    # schedule (diffusers' use_exponential_sigmas); the id is kept for API
    # compatibility, the label states what it does.
    "exponential": "Trailing spacing (not exponential sigmas)",
}

# Config keys this module owns: every one of them is written on every call, so
# a scheduler built here can never inherit a value from the previously selected
# schedule (get_scheduler reads back the scheduler it last returned).
_SCHEDULE_KEYS = ("prediction_type", "use_karras_sigmas", "timestep_spacing")


def _schedule_overrides(schedule_type: str, prediction_type: str) -> dict:
    """Resolved value for every key in ``_SCHEDULE_KEYS``."""
    is_v_prediction = prediction_type == "v_prediction"
    return {
        "prediction_type": prediction_type,
        "use_karras_sigmas": schedule_type == "karras",
        # Karras picks sigmas, not timestep spacing, so it takes the same
        # spacing as "uniform" rather than whichever spacing ran before it.
        "timestep_spacing": (
            "trailing" if (schedule_type == "exponential" or is_v_prediction)
            else "leading"
        ),
    }


# What "uniform on an epsilon model" resolves to. An override equal to this is
# the no-op case, so dropping it on a scheduler class that has no such setting
# changes nothing and is not worth a warning.
_NEUTRAL_OVERRIDES = _schedule_overrides("uniform", "epsilon")


def _accepted_config_keys(scheduler_class) -> set:
    """Config keys ``scheduler_class.__init__`` actually reads.

    Taken from the signature (the same set diffusers' ``extract_init_dict``
    filters ``from_config`` kwargs against) rather than a per-class table, so a
    sampler added to ``SAMPLER_MAP`` cannot silently inherit a stale entry.
    """
    try:
        keys = set(scheduler_class._get_init_keys(scheduler_class))
    except Exception:
        keys = set(inspect.signature(scheduler_class.__init__).parameters)
    # The four names extract_init_dict removes from its expected keys: a kwarg
    # called any of them never reaches __init__. The latter two are empty or
    # absent on all 12 classes in SAMPLER_MAP, so this only matters for a class
    # added later -- which must warn rather than pretend the setting applied.
    keys -= {"self", "kwargs"}
    keys -= set(getattr(scheduler_class, "_flax_internal_args", ()) or ())
    return keys - set(getattr(scheduler_class, "ignore_for_config", ()) or ())


def unsupported_schedule_overrides(sampler: str, schedule_type: str,
                                   prediction_type: str = "epsilon") -> list:
    """Overrides the sampler's scheduler class cannot apply, as warning dicts.

    Building the scheduler succeeds either way — diffusers just drops the
    unknown kwarg — so nothing else reports these.
    """
    scheduler_class = SAMPLER_MAP.get(sampler)
    if scheduler_class is None:
        return []
    accepted = _accepted_config_keys(scheduler_class)
    overrides = _schedule_overrides(schedule_type, prediction_type)
    warnings = []
    for key in _SCHEDULE_KEYS:
        value = overrides[key]
        if key in accepted or value == _NEUTRAL_OVERRIDES[key]:
            continue
        warnings.append({
            "code": "unsupported_param",
            "message": (
                f"Sampler '{sampler}' ({scheduler_class.__name__}) has no "
                f"{key} setting, so the {key}={value!r} that "
                f"schedule_type='{schedule_type}' resolves to for this model "
                f"was ignored."
            ),
        })
    return warnings


def _emit_warnings(warnings: list) -> None:
    try:
        from api.generation_status import add_warning
    except Exception:
        return
    for warning in warnings:
        try:
            add_warning(warning["message"], code=warning["code"])
        except Exception:
            pass


def get_scheduler(pipeline, sampler: str, schedule_type: str = "uniform"):
    """
    Get a scheduler instance for the given pipeline with specified sampler and schedule type

    Overrides go in as ``from_config`` kwargs rather than into the source
    config, which leaves the loaded model's own scheduler untouched and reaches
    keys that ``_use_default_values`` would otherwise drop. See this function's
    tests for why both halves of that are load-bearing.

    Args:
        pipeline: The diffusion pipeline
        sampler: Name of the sampler/algorithm (e.g., "euler", "dpmpp_2m")
        schedule_type: Type of noise schedule (e.g., "uniform", "karras", "exponential")

    Returns:
        Scheduler instance configured with the pipeline's config and schedule type
    """
    if sampler not in SAMPLER_MAP:
        raise ValueError(f"Unknown sampler: {sampler}. Available: {list(SAMPLER_MAP.keys())}")

    scheduler_class = SAMPLER_MAP[sampler]

    try:
        source_config = dict(pipeline.scheduler.config)
        prediction_type = source_config.get("prediction_type") or "epsilon"
        overrides = _schedule_overrides(schedule_type, prediction_type)
        accepted = _accepted_config_keys(scheduler_class)

        base = {k: v for k, v in source_config.items()
                if k not in _SCHEDULE_KEYS or k in accepted}
        applied = {k: v for k, v in overrides.items() if k in accepted}

        _emit_warnings(unsupported_schedule_overrides(
            sampler, schedule_type, prediction_type))

        return scheduler_class.from_config(base, **applied)
    except Exception as e:
        print(f"Warning: Could not create {sampler} scheduler with {schedule_type}: {e}")
        # Fallback to creating with default config
        return scheduler_class()

def get_available_samplers():
    """Get list of available sampler names"""
    return list(SAMPLER_MAP.keys())

def get_sampler_display_names():
    """Get dict mapping sampler IDs to display names"""
    return SAMPLER_NAMES

def get_available_schedule_types():
    """Get list of available schedule types"""
    return list(SCHEDULE_TYPES.keys())

def get_schedule_type_display_names():
    """Get dict mapping schedule type IDs to display names"""
    return SCHEDULE_TYPES
