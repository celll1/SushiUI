"""SenseNova-understanding + SDXL-U-Net Chimera building blocks.

The production loader and pipeline land in later implementation phases.  This
package starts with the checkpoint-independent contracts shared by those paths:
donor-equal U-Net construction, conditioning bridge shapes, spatial RoPE, and
the flow objective.
"""

from .conditioning_bridge import (
    ChimeraBridgeConfig,
    ChimeraBridgeOutput,
    ConditioningBridge,
    selected_layer_indices,
)
from .flow import flow_noising, flow_velocity_target, flow_euler_step
from .positional import (
    POSITION_LAYOUT_VERSION,
    apply_sensenova_rope,
    apply_sensenova_rope_qk,
    spatial_query_positions,
)
from .unet import (
    ChimeraUNetBuildReport,
    build_donor_equal_unet,
    parameter_census,
    trainable_parameter_count,
)

__all__ = [
    "POSITION_LAYOUT_VERSION",
    "ChimeraBridgeConfig",
    "ChimeraBridgeOutput",
    "ChimeraUNetBuildReport",
    "ConditioningBridge",
    "apply_sensenova_rope",
    "apply_sensenova_rope_qk",
    "build_donor_equal_unet",
    "flow_euler_step",
    "flow_noising",
    "flow_velocity_target",
    "parameter_census",
    "selected_layer_indices",
    "spatial_query_positions",
    "trainable_parameter_count",
]
