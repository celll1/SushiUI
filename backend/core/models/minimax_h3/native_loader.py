"""Assemble native H3 Full bundles with their explicitly selected companions."""

from pathlib import Path

import torch


def load_native_bundle(path,torch_dtype,*,load_text_encoder=True,te_override=None,
                       te_projection_override=None,video_vae_dtype=None,hybrid=None):
    from . import loader
    from .component_sources import companion_layout
    from .video_artifact import preflight_checkpoint,_restore_checkpoint,_artifact_path
    if hybrid is not None:
        raise ValueError("Native H3 aligned Full bundles do not accept hybrid overlays")
    if video_vae_dtype is not None:
        raise ValueError("Native H3 embedded codec precision is fixed by its checkpoint")
    path = _artifact_path(path)
    preflight = preflight_checkpoint(path)
    metadata,config,record,_ = preflight
    layout = companion_layout(metadata)
    for supplied,declared in ((te_override,layout["text_encoder"]),(te_projection_override,layout["te_projection"])):
        if supplied is not None and (declared is None or Path(supplied).resolve() != Path(declared).resolve()):
            raise ValueError("Native H3 bundle requires its declared text encoder/projection pair")
    declaration,projection = {},None
    text_encoder = text_encoder_config = None
    if load_text_encoder:
        declaration = loader._te_file_declaration(layout["text_encoder"])
        if loader._te_declared_dims(declaration) is not None or layout["te_projection"] is not None:
            projection = loader.resolve_minimax_h3_te_projection(te_path=layout["text_encoder"],
                declared=declaration,root=layout["root"],text_dim=config["text_dim"],override=layout["te_projection"])
        # Windows requires mapping the large encoder before smaller weight files.
        text_encoder,text_encoder_config = loader._build_text_encoder_for(layout["text_encoder"],layout["official"],projection)
    transformer,vae,identity = _restore_checkpoint(path,torch_dtype,preflight)
    audio_vae,audio_config = loader._build_audio_vae(layout["audio_vae"],layout["official"],torch.float32)
    tokenizer,processor = loader._load_tokenizer_and_processor(layout["official"])
    scheduler,audio_scheduler = loader._load_schedulers(layout["official"])
    contract = vae.contract
    components = dict(type="minimax_h3",is_video=True,variant=layout["variant"],
        transformer=transformer,transformer_config=dict(transformer.transformer.config),
        vae=vae,vae_config=dict(vae.config),vae_identity=identity,image_vae=None,image_vae_config=None,
        audio_vae=audio_vae,audio_vae_config=audio_config,text_encoder=text_encoder,text_encoder_config=text_encoder_config,
        tokenizer=tokenizer,processor=processor,scheduler=scheduler,audio_scheduler=audio_scheduler,
        latent_channels=contract.image.latent_channels,audio_latent_channels=config["audio_in_channels"],
        vae_scale_factor_spatial=contract.image.spatial_scale,vae_scale_factor_temporal=contract.temporal_scale,
        audio_sample_rate=loader.MINIMAX_H3_AUDIO_SAMPLE_RATE,audio_latent_rate=loader.MINIMAX_H3_AUDIO_LATENT_RATE,
        fps=loader.MINIMAX_H3_FPS,latents_mean=vae.config.latents_mean,latents_std=vae.config.latents_std,
        audio_latents_mean=audio_config.get("latents_mean"),audio_latents_std=audio_config.get("latents_std"),
        pixel_mean=(.5,)*3,pixel_std=(.5,)*3,vae_tiling_policy=None,video_vae_dtype=str(vae.dtype),
        vae_source="embedded (checkpoint)",vae_origin="embedded_checkpoint",vae_path=str(path),dit_path=str(path),audio_vae_path=layout["audio_vae"],
        source_vae_path=layout["vae"],source_vae_identity=record["declaration"]["source_identity"],
        image_vae_source=None,image_vae_path=None,text_encoder_path=layout["text_encoder"],
        text_encoder_origin="selected_external",te_text_only=str(declaration.get("modalities") or "") == "text",
        te_projection=projection,official_dir=layout["official"])
    from .generation_codec import generation_contract
    generation_contract(components)
    if load_text_encoder:
        from .te_agreement import maybe_measure_substitution
        maybe_measure_substitution(components)
    return components
