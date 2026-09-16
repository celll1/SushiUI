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
from .artifact import ChimeraArtifactError, MODEL_TYPE, FORMAT_VERSION
from .builder import build_chimera_artifact_from_components, initialize_chimera_atomically
from .flow import flow_noising, flow_velocity_target, flow_euler_step
from .loader import load_chimera_artifact
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
    "MODEL_TYPE",
    "FORMAT_VERSION",
    "ChimeraArtifactError",
    "ChimeraBridgeConfig",
    "ChimeraBridgeOutput",
    "ChimeraUNetBuildReport",
    "ConditioningBridge",
    "apply_sensenova_rope",
    "apply_sensenova_rope_qk",
    "build_donor_equal_unet",
    "build_chimera_artifact_from_components",
    "flow_euler_step",
    "flow_noising",
    "flow_velocity_target",
    "initialize_chimera_atomically",
    "load_chimera_artifact",
    "parameter_census",
    "selected_layer_indices",
    "spatial_query_positions",
    "trainable_parameter_count",
]
