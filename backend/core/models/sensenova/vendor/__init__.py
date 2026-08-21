"""Vendored SenseNova-U1.5-8B-MoT model code (``neo_unify``).

Source
    https://github.com/OpenSenseNova/SenseNova-U1, branch ``feat/u1.5``
    (commit ``a1ce053d25835e0785a0869ca1c97e717212ef64``), directory
    ``src/sensenova_u1/models/neo_unify/``, retrieved 2026-08-21. Upstream code
    license: Apache-2.0 (see ``LICENSE`` in that repository).

Why vendored (not pinned)
    No ``.py`` modeling code ships in the ``sensenova/SenseNova-U1.5-8B-MoT``
    HF repo (only weights + config, despite ``config.json``'s ``auto_map``
    naming ``modeling_neo_chat``); the modeling code exists only in the
    GitHub source above, which is not a published, pinnable package.

Scope
    Model classes only (Qwen3/Qwen3-MoE backbone, the two-branch NEO-Unify
    attention/norm/MLP layout, the flow-matching head modules, the
    ``NEOVisionModel`` patch embedder, and the top-level ``NEOChatModel``).
    ``NEOChatModel.t2i_generate`` is kept as the ORIGINAL REFERENCE
    implementation, not SushiUI's denoise loop entry point: a later unit
    drives this model's own helper methods (``patchify``/``unpatchify``,
    ``_build_t2i_query``, ``_build_t2i_text_inputs``,
    ``_build_t2i_image_indexes``, ``_t2i_prefix_forward``, ``_t2i_predict_v``,
    ``_apply_time_schedule``, ``prepare_flash_kv_cache``/
    ``clear_flash_kv_cache``, ``_notify_layer_offload_phase``) directly, the
    same reason MiniMax-H3's Modular-Diffusers blocks are not vendored (see
    ``backend/core/models/minimax_h3/vendor/__init__.py``) -- SushiUI owns its
    denoise loop (per-step progress, cancellation, preview, offload
    sequencing).

Modifications
    Every file carries a "SushiUI vendored copy — MODIFIED"/"UNMODIFIED"
    header naming its exact changes, as the source repository's Apache-2.0
    license (NOTICE-file convention) expects for a modified redistribution.
    The one substantive change is in ``modeling_qwen3.py``:
    ``Qwen3Attention.forward_gen``'s sole attention call (``_flash_or_sdpa``)
    now routes through SushiUI's unified attention conduit
    (``core.attention.dispatch_attention``) instead of its own private
    flash-attn/SDPA branch, so the repo-wide ``attention_type`` vocabulary
    (native/flash/sage/tq) selects the kernel here as on every other
    architecture. ``modeling_qwen3_moe.py`` is vendored unmodified, purely
    because ``modeling_neo_chat.py`` imports ``Qwen3MoeForCausalLM``
    unconditionally; the checkpoint this loader targets is the dense
    (non-MoE) backbone and never instantiates those classes.

Weights license
    Apache-2.0 (SenseNova-U1.5-8B-MoT is released under Apache-2.0, unlike
    MiniMax-H3's community license).
"""

from .configuration_neo_chat import NEOChatConfig, NEOLLMConfig, NEOMoELLMConfig
from .configuration_neo_vit import NEOVisionConfig
from .conversation import get_conv_template
from .modeling_neo_chat import NEOChatModel, clear_flash_kv_cache, prepare_flash_kv_cache
from .modeling_neo_vit import NEOVisionModel
from .modeling_qwen3 import (
    _HAS_FLASH_ATTN as has_flash_attn,
    Qwen3Attention,
    Qwen3ForCausalLM,
    effective_attn_backend,
    get_attn_backend,
    set_attn_backend,
)
from .modeling_qwen3_moe import Qwen3MoeForCausalLM
from .utils import SYSTEM_MESSAGE_FOR_GEN, load_image_native

__all__ = [
    "NEOChatConfig",
    "NEOLLMConfig",
    "NEOMoELLMConfig",
    "NEOVisionConfig",
    "NEOChatModel",
    "NEOVisionModel",
    "Qwen3Attention",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "SYSTEM_MESSAGE_FOR_GEN",
    "clear_flash_kv_cache",
    "effective_attn_backend",
    "get_attn_backend",
    "get_conv_template",
    "has_flash_attn",
    "load_image_native",
    "prepare_flash_kv_cache",
    "set_attn_backend",
]
