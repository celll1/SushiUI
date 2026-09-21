"""Read-only API schemas backed by the canonical default and capability tables."""

from fastapi import APIRouter

from api.param_defaults import (
    AUDIO_GEN_ARCH_OVERLAYS,
    AUD2AUD_DEFAULTS,
    AUD2AUD_GEN_ARCH_OVERLAYS,
    BUNDLE_VAE_DEFAULTS_BY_ARCH,
    CFG_UNCOND_DROP_DEFAULTS_BY_ARCH,
    IMAGE_GEN_ARCH_OVERLAYS,
    IMG2VID_DEFAULTS,
    IMG2IMG_DEFAULTS,
    IMG2TXT_DEFAULTS,
    INPAINT_DEFAULTS,
    INPAINT_VIDEO_ARCH_OVERLAYS,
    INPAINT_VIDEO_DEFAULTS,
    LORA_ITEM_DEFAULTS,
    LR_PREVIEW_DEFAULTS,
    LR_RETARGET_DEFAULTS,
    LR_TRIGGER_DEFAULTS,
    MUSIC_LYRICS_ASSIST_DEFAULTS,
    MUSIC_PROMPT_ASSIST_DEFAULTS,
    OUTPAINT_AUDIO_ARCH_OVERLAYS,
    OUTPAINT_AUDIO_DEFAULTS,
    OUTPAINT_DEFAULTS,
    OUTPAINT_VIDEO_ARCH_OVERLAYS,
    OUTPAINT_VIDEO_DEFAULTS,
    PARAM_BOUNDS,
    PROMPT_ASSIST_DEFAULTS,
    QWEN_IMAGE_21_PROMPT_UPSAMPLE_DEFAULTS,
    REF2VID_DEFAULTS,
    STUDIO_RENDER_DEFAULTS,
    TAGGER_TRAINING_DEFAULTS,
    TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH,
    TRAINING_DEFAULTS,
    TRAINING_SAMPLE_DEFAULTS_BY_ARCH,
    TXT2AUD_DEFAULTS,
    TXT2IMG_DEFAULTS,
    TXT2VID_DEFAULTS,
    UPSCALE_DEFAULTS,
    VAE_TRAINING_DEFAULTS,
    VIDEO_GEN_ARCH_OVERLAYS,
)

router = APIRouter()


@router.get("/schema/generation-defaults")
async def get_generation_defaults():
    """Return canonical defaults and per-architecture generation overlays."""
    return {
        "video_arch_overlays": VIDEO_GEN_ARCH_OVERLAYS,
        "outpaint_video_arch_overlays": OUTPAINT_VIDEO_ARCH_OVERLAYS,
        "inpaint_video_arch_overlays": INPAINT_VIDEO_ARCH_OVERLAYS,
        "audio_arch_overlays": AUDIO_GEN_ARCH_OVERLAYS,
        "aud2aud_arch_overlays": AUD2AUD_GEN_ARCH_OVERLAYS,
        "outpaint_audio_arch_overlays": OUTPAINT_AUDIO_ARCH_OVERLAYS,
        "image_arch_overlays": IMAGE_GEN_ARCH_OVERLAYS,
        "img2txt": IMG2TXT_DEFAULTS,
        "txt2img": TXT2IMG_DEFAULTS,
        "img2img": IMG2IMG_DEFAULTS,
        "inpaint": INPAINT_DEFAULTS,
        "outpaint": OUTPAINT_DEFAULTS,
        "upscale": UPSCALE_DEFAULTS,
        "txt2vid": TXT2VID_DEFAULTS,
        "img2vid": IMG2VID_DEFAULTS,
        "ref2vid": REF2VID_DEFAULTS,
        "outpaint_vid": OUTPAINT_VIDEO_DEFAULTS,
        "inpaint_vid": INPAINT_VIDEO_DEFAULTS,
        "txt2aud": TXT2AUD_DEFAULTS,
        "aud2aud": AUD2AUD_DEFAULTS,
        "outpaint_aud": OUTPAINT_AUDIO_DEFAULTS,
        "studio_render": STUDIO_RENDER_DEFAULTS,
        "lora_item": LORA_ITEM_DEFAULTS,
        "param_bounds": PARAM_BOUNDS,
    }


@router.get("/schema/prompt-assist-defaults")
async def get_prompt_assist_defaults():
    return PROMPT_ASSIST_DEFAULTS


@router.get("/schema/qwen-image-21-prompt-upsample-defaults")
async def get_qwen_image_21_prompt_upsample_defaults():
    return QWEN_IMAGE_21_PROMPT_UPSAMPLE_DEFAULTS


@router.get("/schema/prompt-assist-music-defaults")
async def get_prompt_assist_music_defaults():
    return MUSIC_PROMPT_ASSIST_DEFAULTS


@router.get("/schema/prompt-assist-music-lyrics-defaults")
async def get_prompt_assist_music_lyrics_defaults():
    return MUSIC_LYRICS_ASSIST_DEFAULTS


@router.get("/schema/training-defaults")
async def get_training_defaults():
    return {
        **TRAINING_DEFAULTS,
        "_sample_defaults_by_arch": TRAINING_SAMPLE_DEFAULTS_BY_ARCH,
    }


@router.get("/schema/tagger-training-defaults")
async def get_tagger_training_defaults():
    return TAGGER_TRAINING_DEFAULTS


@router.get("/schema/vae-training-defaults")
async def get_vae_training_defaults():
    return VAE_TRAINING_DEFAULTS


@router.get("/schema/lr-retarget-defaults")
async def get_lr_retarget_defaults():
    return {**LR_RETARGET_DEFAULTS, "n_points": LR_PREVIEW_DEFAULTS["n_points"]}


@router.get("/schema/lr-trigger-defaults")
async def get_lr_trigger_defaults():
    return dict(LR_TRIGGER_DEFAULTS)


@router.get("/schema/timestep-defaults-by-arch")
async def get_timestep_defaults_by_arch():
    return TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH


@router.get("/schema/bundle-vae-defaults-by-arch")
async def get_bundle_vae_defaults_by_arch():
    return BUNDLE_VAE_DEFAULTS_BY_ARCH


@router.get("/schema/arch-capabilities")
async def get_arch_capabilities():
    """Return the canonical generation and training capability matrices."""
    from api.arch_capabilities import (
        ARCH_DISPLAY_NAMES,
        ARCH_SUPPORTED_VALUES,
        ARCH_UNSUPPORTED,
        AUDIO_OUTPAINT_PLACEMENTS,
        AUD2AUD_MUSIC3_REPAINT_MODES,
        CFG_NULL_STAGE_BY_ARCH,
        FEATURE_LABELS,
        FEATURE_PARAMS,
        QUANTIZED_LINEAR_ARCHS,
        RUNTIME_INT8_ARCHS,
        TEXT_OUTPUT_MODES,
        TRAINING_FEATURE_ADVISORY,
        TRAINING_FEATURE_LABELS,
        TRAINING_FEATURE_PARAMS,
        TRAINING_FEATURE_UNSUPPORTED,
        TRAINING_REQUIRED_VALUES,
        TRAINING_SAMPLE_NOTES,
        TRAINING_SAMPLE_SUPPORTED_PARAMS,
        TRAINING_UNSUPPORTED,
        adapter_families_payload,
        chain_context_payload,
        video_constraints_payload,
    )

    return {
        "unsupported": ARCH_UNSUPPORTED,
        "text_output_modes": {key: list(value) for key, value in TEXT_OUTPUT_MODES.items()},
        "supported_values": ARCH_SUPPORTED_VALUES,
        "feature_params": FEATURE_PARAMS,
        "feature_labels": FEATURE_LABELS,
        "training_unsupported": TRAINING_UNSUPPORTED,
        "training_feature_unsupported": TRAINING_FEATURE_UNSUPPORTED,
        "training_feature_params": TRAINING_FEATURE_PARAMS,
        "training_feature_labels": TRAINING_FEATURE_LABELS,
        "training_required_values": TRAINING_REQUIRED_VALUES,
        "training_feature_advisory": TRAINING_FEATURE_ADVISORY,
        "training_sample_supported_params": TRAINING_SAMPLE_SUPPORTED_PARAMS,
        "training_sample_notes": TRAINING_SAMPLE_NOTES,
        "adapter_families": adapter_families_payload(),
        "arch_display_names": ARCH_DISPLAY_NAMES,
        "cfg_null_stage": CFG_NULL_STAGE_BY_ARCH,
        "cfg_uncond_drop_defaults": CFG_UNCOND_DROP_DEFAULTS_BY_ARCH,
        "video_constraints": video_constraints_payload(),
        "chain_context": chain_context_payload(),
        "runtime_int8_archs": list(RUNTIME_INT8_ARCHS),
        "quantized_linear_archs": list(QUANTIZED_LINEAR_ARCHS),
        "audio_outpaint_placements": {
            key: list(value) for key, value in AUDIO_OUTPAINT_PLACEMENTS.items()
        },
        "aud2aud_music3_repaint_modes": {
            key: list(value) for key, value in AUD2AUD_MUSIC3_REPAINT_MODES.items()
        },
    }
