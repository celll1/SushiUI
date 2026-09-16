"""SenseNova-understanding + SDXL-U-Net Chimera runtime and contracts."""

from .conditioning_bridge import (
    ChimeraBridgeConfig,
    ChimeraBridgeOutput,
    ConditioningBridge,
    selected_layer_indices,
)
from .artifact import ChimeraArtifactError, MODEL_TYPE, FORMAT_VERSION, save_chimera_checkpoint
from .attention_processor import (
    ChimeraAttentionContext,
    ChimeraAttnProcessor,
    clear_chimera_attention_caches,
    install_chimera_attention_processors,
    set_chimera_attention_context,
)
from .builder import (
    build_chimera_artifact_from_components,
    initialize_chimera_atomically,
    initialize_chimera_from_paths,
)
from .flow import flow_noising, flow_velocity_target, flow_euler_step
from .loader import load_chimera_artifact, preflight_chimera_artifact
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
from .understanding import UnderstandingPrefix, capture_understanding_prefix, load_understanding_only

__all__ = [
    "POSITION_LAYOUT_VERSION",
    "MODEL_TYPE",
    "FORMAT_VERSION",
    "ChimeraArtifactError",
    "ChimeraAttentionContext",
    "ChimeraAttnProcessor",
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
    "initialize_chimera_from_paths",
    "load_chimera_artifact",
    "preflight_chimera_artifact",
    "save_chimera_checkpoint",
    "parameter_census",
    "selected_layer_indices",
    "spatial_query_positions",
    "trainable_parameter_count",
    "UnderstandingPrefix",
    "capture_understanding_prefix",
    "clear_chimera_attention_caches",
    "install_chimera_attention_processors",
    "load_understanding_only",
    "set_chimera_attention_context",
]
