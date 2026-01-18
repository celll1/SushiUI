"""
Z-Image FlowMatchEulerDiscreteScheduler wrapper for SushiUI

This module provides a simple wrapper around diffusers' FlowMatchEulerDiscreteScheduler
to maintain compatibility with Z-Image loading code.
"""

from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

__all__ = ["FlowMatchEulerDiscreteScheduler"]
