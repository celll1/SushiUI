"""Sampler and Scheduler management for Stable Diffusion pipelines"""

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
    "exponential": "Exponential",
}

def get_scheduler(pipeline, sampler: str, schedule_type: str = "uniform"):
    """
    Get a scheduler instance for the given pipeline with specified sampler and schedule type

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

    # Get base config from pipeline
    try:
        config = pipeline.scheduler.config

        # Apply schedule type settings
        if schedule_type == "karras":
            # Use Karras sigmas for better quality
            config["use_karras_sigmas"] = True
        elif schedule_type == "exponential":
            # Use exponential schedule
            config["timestep_spacing"] = "trailing"
        else:  # uniform
            config["use_karras_sigmas"] = False
            config["timestep_spacing"] = "leading"

        return scheduler_class.from_config(config)
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
