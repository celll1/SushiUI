"""Vendored MiniMax-H3 model code from the diffusers ``minimax-h3`` branch.

Source
    https://github.com/huggingface/diffusers, branch ``minimax-h3``
    (PR #14355 "Add MiniMax-H3", plus PR #14371 "Minimax h3 follow up
    (review & refactor)" which was merged into that branch, not into ``main``),
    retrieved 2026-08-05. Upstream code license: Apache-2.0.

Why vendored (not pinned)
    The branch is versioned ``0.36.0.dev0`` and the transformer file does not
    exist on diffusers ``main``. This repo runs diffusers 0.38.0 for ten other
    architectures, so pinning every one of them to an unmerged dev branch is
    not acceptable; vendoring also freezes the block signatures the block-loop
    wrapper and the LoRA adapter depend on.

Scope
    Model classes and the scheduler only. The upstream Modular-Diffusers
    blocks and ``MiniMaxH3ModularPipeline`` are deliberately **not** vendored:
    SushiUI owns its denoise loop (per-step progress, cancellation, latent
    preview, block swap, offload sequencing), which is exactly what lives
    inside those blocks upstream.

Modifications
    Every file carries a "SushiUI vendored copy — MODIFIED" header listing its
    changes, as the MiniMax H3 Community License Agreement requires for
    modified files. The substantive one is in ``transformer_minimax_h3.py``:
    support for the released "pruned" checkpoints' AdaLN-curve variant
    (``adaln_curve_grid``). Those checkpoints ship no timestep MLP, only an
    ``adaln_t_table``, while upstream diffusers implements only the
    full-modulation variant.

Weights license
    The MiniMax-H3 weights are covered by the MiniMax H3 Community License
    Agreement (not Apache/MIT): commercial use below USD 20M annual revenue,
    attribution as "MiniMax H3" required in a product UI, and the grant
    excludes the European Union, the United Kingdom, the Republic of Korea and
    the United States of America.

``reference/``
    Not a Python package and never imported: the upstream conversion script,
    kept as the Rosetta stone for the original-checkpoint key mapping.
"""

from .autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from .autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
from .scheduling_minimax_h3 import MiniMaxH3Scheduler
from .transformer_minimax_h3 import (
    MiniMaxH3AdaLayerNormModulation,
    MiniMaxH3AdaLayerNormOut,
    MiniMaxH3Attention,
    MiniMaxH3AttnProcessor,
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TokenRefiner,
    MiniMaxH3TokenRefinerBlock,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerBlock,
    MiniMaxH3TransformerOutput,
    sample_adaln_curve,
)

__all__ = [
    "AutoencoderKLMiniMaxH3",
    "AutoencoderKLMiniMaxH3Audio",
    "MiniMaxH3AdaLayerNormModulation",
    "MiniMaxH3AdaLayerNormOut",
    "MiniMaxH3Attention",
    "MiniMaxH3AttnProcessor",
    "MiniMaxH3RotaryPosEmbed",
    "MiniMaxH3Scheduler",
    "MiniMaxH3TokenRefiner",
    "MiniMaxH3TokenRefinerBlock",
    "MiniMaxH3Transformer3DModel",
    "MiniMaxH3TransformerBlock",
    "MiniMaxH3TransformerOutput",
    "sample_adaln_curve",
]
