"""Component loader for LTX-2.3 (joint audio+video MM-DiT + Gemma-3 + LTX2 VAEs).

Phase 1a: make the model LOADABLE into SushiUI's single-in-memory model slot.
Video generation itself is Phase 1b.

Unlike the other DiT archs (Krea2/Ideogram4), the whole LTX-2 stack already
lives in the pinned venv diffusers 0.38.0, so this loader does NOT rebuild the
transformer from a config + bare state_dict. It simply calls
``LTX2Pipeline.from_pretrained(dir, torch_dtype=bf16)`` (bf16 halves the 46GB
fp32 Gemma-3 text encoder) and returns the resolved components in a dict shaped
like the other archs' return value, which ``PipelineManager.load_model``
consumes.

All components are kept on CPU after load (VRAM discipline). GPU staging happens
at generate time in P1b — either manually per phase, or via the pipeline's
``model_cpu_offload_seq`` (text_encoder -> connectors -> transformer -> vae ->
audio_vae -> vocoder).
"""

from __future__ import annotations

import torch


def load_ltx2_from_diffusers(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Load LTX-2.3 from a diffusers directory (model_index.json + subfolders).

    Returns a component dict consumed by PipelineManager.load_model():
        {
          "type": "ltx2",
          "pipeline": <LTX2Pipeline>,          # the assembled pipeline (P1b entry)
          "transformer": <LTX2VideoTransformer3DModel>,
          "vae": <AutoencoderKLLTX2Video>,
          "audio_vae": <AutoencoderKLLTX2Audio>,
          "text_encoder": <Gemma3ForConditionalGeneration>,
          "tokenizer": <GemmaTokenizerFast>,
          "connectors": <LTX2TextConnectors>,
          "vocoder": <LTX2VocoderWithBWE>,
          "scheduler": <FlowMatchEulerDiscreteScheduler>,
          "vae_scale_factor_spatial": 32,
          "vae_scale_factor_temporal": 8,
          "latent_channels": 128,
          "is_video": True,
        }

    P1b relies on the "pipeline" reference (its __call__ is the generation entry)
    and on the individual component refs for manual GPU staging / offload wiring.
    """
    from core.models.ltx2 import LTX2Pipeline

    print(f"[LTX2Loader] Loading LTX-2.3 diffusers directory: {model_path}")
    print(f"[LTX2Loader] dtype={torch_dtype} (bf16 halves the fp32 Gemma-3 text encoder)")

    pipeline = LTX2Pipeline.from_pretrained(model_path, torch_dtype=torch_dtype)

    # Keep everything on CPU after load; P1b stages to GPU (or uses
    # model_cpu_offload_seq). from_pretrained already leaves modules on CPU.
    for name in (
        "text_encoder", "connectors", "transformer",
        "vae", "audio_vae", "vocoder",
    ):
        comp = getattr(pipeline, name, None)
        if comp is not None and hasattr(comp, "to"):
            try:
                comp.to("cpu")
            except Exception:
                pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    transformer = getattr(pipeline, "transformer", None)
    vae = getattr(pipeline, "vae", None)

    # VAE scale factors (spatial 32x, temporal 8x) — pull from the transformer
    # config when present, else fall back to the confirmed distilled values.
    vae_scale_spatial = 32
    vae_scale_temporal = 8
    latent_channels = 128
    try:
        tcfg = getattr(transformer, "config", None)
        if tcfg is not None:
            factors = getattr(tcfg, "vae_scale_factors", None)
            if factors and len(factors) == 3:
                vae_scale_temporal = int(factors[0])
                vae_scale_spatial = int(factors[1])
            latent_channels = int(getattr(tcfg, "in_channels", latent_channels))
    except Exception:
        pass

    print(
        f"[LTX2Loader] Loaded LTX-2.3 (latent_channels={latent_channels}, "
        f"spatial={vae_scale_spatial}x, temporal={vae_scale_temporal}x)"
    )

    return {
        "type": "ltx2",
        "pipeline": pipeline,
        "transformer": transformer,
        "vae": vae,
        "audio_vae": getattr(pipeline, "audio_vae", None),
        "text_encoder": getattr(pipeline, "text_encoder", None),
        "tokenizer": getattr(pipeline, "tokenizer", None),
        "connectors": getattr(pipeline, "connectors", None),
        "vocoder": getattr(pipeline, "vocoder", None),
        "scheduler": getattr(pipeline, "scheduler", None),
        "vae_scale_factor_spatial": vae_scale_spatial,
        "vae_scale_factor_temporal": vae_scale_temporal,
        "latent_channels": latent_channels,
        "is_video": True,
    }
