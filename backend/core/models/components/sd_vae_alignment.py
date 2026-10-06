"""SD U-Net codec boundaries with explicit VP or flow noise coefficients."""
def resolve_checkpoint_path(path):
    from pathlib import Path
    source=Path(path)
    index=source.with_suffix(".safetensors.index.json")
    if source.name.endswith(".safetensors") and not source.exists() and index.is_file():
        return str(index)
    return str(source)


def read_checkpoint_metadata(path):
    import json
    from pathlib import Path
    from safetensors import safe_open
    source=Path(resolve_checkpoint_path(path))
    if source.name.endswith(".safetensors.index.json"):
        with source.open(encoding="utf-8") as file:
            return dict(json.load(file).get("metadata",{}) or {})
    with safe_open(str(source),framework="pt") as file:
        return dict(file.metadata() or {})


def checkpoint_metadata(model):
    import json
    from .spatial_swap import artifact_metadata
    metadata=artifact_metadata(model)
    if metadata:
        metadata["unet_config"]=json.dumps(dict(model.config),sort_keys=True,allow_nan=False)
    return metadata


def split_unet_state(model):
    state=model.state_dict()
    bridge={key:value for key,value in state.items() if key.startswith("vae_latent_bridge.")}
    return {key:value for key,value in state.items() if key not in bridge},bridge


def pipeline_metadata(trainer,*,arch):
    import json
    metadata={}
    names=[("text_encoder","tokenizer")]
    if arch=="sdxl":names.append(("text_encoder_2","tokenizer_2"))
    for encoder_name,tokenizer_name in names:
        encoder=getattr(trainer,encoder_name,None)
        tokenizer=getattr(trainer,tokenizer_name,None)
        if encoder is None or not hasattr(encoder,"config"):
            raise ValueError(f"Aligned {arch} export requires its native {encoder_name}")
        source=getattr(tokenizer,"name_or_path",None)
        if not isinstance(source,str) or not source.strip():
            raise ValueError(f"Aligned {arch} export requires a restorable {tokenizer_name} source")
        metadata[encoder_name+"_config"]=json.dumps(encoder.config.to_dict(),sort_keys=True,allow_nan=False)
        metadata["component."+tokenizer_name+".source"]=source
        tokenizer_config={key:getattr(tokenizer,key) for key in
            ("model_max_length","padding_side","truncation_side")}
        for key in ("bos_token","eos_token","pad_token","unk_token"):
            value=getattr(tokenizer,key,None)
            if value is not None:tokenizer_config[key]=str(value)
        metadata[tokenizer_name+"_config"]=json.dumps(tokenizer_config,sort_keys=True,allow_nan=False)
    scheduler=getattr(trainer,"original_scheduler",None) or getattr(trainer,"noise_scheduler",None)
    if scheduler is None:
        raise ValueError("Aligned SD export requires its sampling scheduler config")
    config=json.loads(scheduler.to_json_string())
    config["_class_name"]=type(scheduler).__name__
    metadata["scheduler_config"]=json.dumps(config,sort_keys=True,allow_nan=False)
    return metadata


def restore_alignment(model,metadata,state):
    import json
    from .spatial_swap import read_declaration,install,validate_loaded
    declaration=read_declaration(metadata)
    if declaration is None:
        return
    raw=(metadata or {}).get("unet_config")
    if not isinstance(raw,str) or len(raw.encode("utf-8"))>65536:
        raise ValueError("Aligned SD checkpoint requires bounded native U-Net config")
    config=json.loads(raw)
    width=max(declaration["old_channels"],declaration["new_channels"])
    if not isinstance(config,dict) or config.get("in_channels")!=width or config.get("out_channels")!=width:
        raise ValueError("Aligned SD native config disagrees with internal latent width")
    saved={key:value for key,value in config.items() if not key.startswith("_")}
    actual={key:value for key,value in dict(model.config).items() if not key.startswith("_")}
    if json.dumps(saved,sort_keys=True)!=json.dumps(actual,sort_keys=True):
        raise ValueError("Aligned SD checkpoint disagrees with loaded native U-Net config")
    install(model,declaration)
    pair=model.vae_latent_bridge
    parameter=next(model.parameters())
    pair.to(device=parameter.device,dtype=parameter.dtype)
    prefix="model.diffusion_model.vae_latent_bridge."
    bridge={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
    pair.load_state_dict(bridge,strict=True)
    validate_loaded(model)


def load_unet_artifact(path,*,dtype):
    import json
    from accelerate import init_empty_weights
    from diffusers import UNet2DConditionModel
    from core.models.common.single_file_format import read_state_dict
    from core.training.adapters.state_dict_converter import convert_unet_state_dict_to_original
    from .spatial_swap import read_declaration
    state,metadata=read_state_dict(str(path))
    declaration=read_declaration(metadata)
    if declaration is None:
        raise ValueError("Aligned SD artifact requires an alignment declaration")
    raw=metadata.get("unet_config")
    if not isinstance(raw,str) or len(raw.encode("utf-8"))>65536:
        raise ValueError("Aligned SD artifact requires bounded native U-Net config")
    config=json.loads(raw)
    if not isinstance(config,dict):
        raise ValueError("Native U-Net config must be an object")
    width=max(declaration["old_channels"],declaration["new_channels"])
    if config.get("in_channels")!=width or config.get("out_channels")!=width:
        raise ValueError("Aligned SD native config disagrees with internal latent width")
    with init_empty_weights():
        model=UNet2DConditionModel.from_config(config)
        model.to(dtype=dtype)
    names=convert_unet_state_dict_to_original({key:key for key in model.state_dict()})
    if len(names)!=len(model.state_dict()):
        raise ValueError("Saved SD key conversion aliases native U-Net parameters")
    prefix="model.diffusion_model."
    weights={native:state[prefix+saved].to(dtype=dtype) for saved,native in names.items()
        if prefix+saved in state}
    expected={prefix+saved for saved in names}
    unexpected=[key for key in state if key.startswith(prefix) and key not in expected
        and not key.startswith(prefix+"vae_latent_bridge.")]
    if unexpected:
        raise ValueError(f"Aligned SD artifact contains unexpected U-Net weights: {unexpected[:5]}")
    model.load_state_dict(weights,strict=True,assign=True)
    restore_alignment(model,metadata,state)
    return model.eval()


def load_pipeline_artifact(path,*,arch,dtype,unet=None,vae=None):
    import json
    import diffusers
    from accelerate import init_empty_weights
    from transformers import CLIPTextConfig,CLIPTextModel,CLIPTextModelWithProjection,CLIPTokenizer
    from diffusers.loaders.single_file_utils import convert_open_clip_checkpoint
    from core.models.common.single_file_format import read_state_dict
    from core.models.common.vae_source import load_declared_latent_io
    from .spatial_swap import read_declaration,restored_image_codec
    path=resolve_checkpoint_path(path)
    state,metadata=read_state_dict(str(path))
    declaration=read_declaration(metadata)
    if declaration is None or metadata.get("model_type")!=arch or arch not in ("sd15","sdxl"):
        raise ValueError("Aligned SD pipeline requires its matching architecture declaration")

    def config(name):
        raw=metadata.get(name+"_config")
        if not isinstance(raw,str) or len(raw.encode("utf-8"))>65536:
            raise ValueError(f"Aligned SD pipeline requires bounded {name} config")
        value=json.loads(raw)
        if not isinstance(value,dict):raise ValueError(f"Invalid {name} config")
        return value

    def encoder(name,cls,prefix,open_clip=False):
        with init_empty_weights():
            model=cls(CLIPTextConfig.from_dict(config(name)))
            model.to(dtype=dtype)
        weights=(convert_open_clip_checkpoint(model,state,prefix=prefix) if open_clip else
            {key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)})
        model.load_state_dict({key:value.to(dtype=dtype) for key,value in weights.items()},strict=True,assign=True)
        return model.eval()

    def tokenizer(name):
        source=metadata.get("component."+name+".source")
        if not isinstance(source,str) or not source.strip():
            raise ValueError(f"Aligned SD pipeline requires its {name} source")
        return CLIPTokenizer.from_pretrained(source,**config(name))

    if unet is None:unet=load_unet_artifact(path,dtype=dtype)
    if vae is None:
        resolved=load_declared_latent_io(str(path),arch=arch)
        if resolved is None:raise ValueError("Aligned SD pipeline is missing its codec declaration")
        vae,_=restored_image_codec(resolved.load_module(torch_dtype=dtype),resolved,declaration)
    scheduler_config=config("scheduler")
    from core.inference.schedulers import SAMPLER_MAP
    schedulers={cls.__name__:cls for cls in SAMPLER_MAP.values()}
    scheduler_class=schedulers.get(scheduler_config.get("_class_name"))
    if scheduler_class is None:raise ValueError("Unsupported saved SD sampling scheduler")
    kwargs=dict(unet=unet,vae=vae,scheduler=scheduler_class.from_config(scheduler_config),
        text_encoder=encoder("text_encoder",CLIPTextModel,
            "conditioner.embedders.0.transformer." if arch=="sdxl" else "cond_stage_model.transformer."),
        tokenizer=tokenizer("tokenizer"))
    if arch=="sdxl":
        kwargs.update(text_encoder_2=encoder("text_encoder_2",CLIPTextModelWithProjection,
            "conditioner.embedders.1.model.",open_clip=True),tokenizer_2=tokenizer("tokenizer_2"))
        return diffusers.StableDiffusionXLPipeline(**kwargs)
    return diffusers.StableDiffusionPipeline(**kwargs,safety_checker=None,feature_extractor=None,
        requires_safety_checker=False)


def resolve_wiring(handler,trainer):
    from .latent_adapter import NormalizedImageVAE
    vae=getattr(trainer,"vae",None)
    if isinstance(vae,NormalizedImageVAE):
        handler.pixel_align=vae.spatial_scale_factor
        return handler.wiring.replace(latent_channels=vae.config.latent_channels,
            vae_scale_factor=vae.spatial_scale_factor,vae_norm="identity",vae_norm_pack=1)
    return getattr(trainer,"wiring",None) or handler.wiring


def check_codec_compatibility(facts):
    scale=facts.get("scale_factor")
    if facts.get("ndim") not in (4,5) or type(scale) is not int or scale<1:
        return False,"SD VAE swap requires declared image codec geometry"
    if scale%8 and 8%scale:
        return False,"SD VAE swap requires integer spatial scale relative to 8x"
    return True,None


def restore_swap_identity(trainer,resolved,*,handler,module=None):
    from dataclasses import replace
    from .latent_io import ResizeReport
    pair=trainer.unet.vae_latent_bridge
    if module is None and resolved.source!=pair.declaration["source"]:
        raise ValueError("Changing an aligned SD VAE requires a fresh base checkpoint")
    trainer.vae_identity=replace(resolved,state_dict=None)
    trainer.vae_latent_channels=pair.declaration["new_channels"]
    trainer.wiring=handler.resolve_wiring(trainer)
    width=trainer.unet.config.in_channels
    return ResizeReport((),width,width,width,0,0)


def initialize_swap(trainer,resolved,*,handler):
    from .latent_io import resize_latent_io
    from .spatial_swap import initialize_image_swap
    from .vae_wrapper import SDXLVAEWrapper
    from core.training.adapters.base_adapter import reject_quantized_base
    model=trainer.unet
    if model.config.in_channels!=model.config.out_channels:
        raise ValueError("SD spatial swap requires symmetric image-only U-Net latent faces")
    reject_quantized_base(model,model_label="SD U-Net VAE swap")
    original=trainer.vae
    if isinstance(original,SDXLVAEWrapper):
        trainer.vae=original.vae
    try:
        from .gaussian_alignment import POLICY
        source_wiring=handler.resolve_wiring(trainer)
        noise_policy=(POLICY if resolved.scale_factor>source_wiring.vae_scale_factor
            and resolved.latent_channels>=model.config.in_channels else None)
        report=initialize_image_swap(trainer,resolved,handler=handler,model=model,
            old_channels=model.config.in_channels,source_wiring=source_wiring,
            patch_size=1,resize=lambda width:resize_latent_io(trainer,handler.wiring.latent_io,width),
            pending_key="_sd_vae_calibration_source",noise_policy=noise_policy)
        if isinstance(original,SDXLVAEWrapper) and trainer.vae is original.vae:
            trainer.vae=original
        return report
    except Exception:
        if isinstance(original,SDXLVAEWrapper) and trainer.vae is original.vae:
            trainer.vae=original
        raise


def calibrate(trainer,datasets):
    from .spatial_swap import calibrate_images
    calibrate_images(trainer,datasets,model=trainer.unet,pending_key="_sd_vae_calibration_source")


def training_coefficients(model,scheduler,timesteps,*,noise_process,reference,alphas_cumprod=None):
    if getattr(model,"vae_latent_bridge",None) is None:
        return None,None
    import torch
    if noise_process=="flow":
        sigma=timesteps.to(device=reference.device,dtype=reference.dtype)
        return 1-sigma,sigma
    if noise_process!="ddpm":
        raise ValueError("SD spatial alignment requires a declared VP or flow noise process")
    if timesteps.dtype not in (torch.int32,torch.int64):
        raise ValueError("VP training alignment requires discrete scheduler indices")
    table=scheduler.alphas_cumprod if alphas_cumprod is None else alphas_cumprod
    table=table.to(device=reference.device)
    alpha_bar=table[timesteps.to(reference.device).long()].float()
    return alpha_bar.sqrt().to(reference.dtype),(1-alpha_bar).sqrt().to(reference.dtype)


def map_input(model,latent,*,alpha,sigma=None,noise=None):
    pair=getattr(model,"vae_latent_bridge",None)
    if pair is None:
        return latent
    if latent.ndim!=4 or alpha is None:
        raise ValueError("SD spatial alignment requires BCHW latents and an explicit clean coefficient")
    return pair.diffusion_state(latent,alpha=alpha,sigma=sigma,noise=noise)


def map_prediction(model,value,*,noise_process,prediction_target,sigma,alpha=None,noise=None):
    pair=getattr(model,"vae_latent_bridge",None)
    if pair is None:
        return value
    if prediction_target=="velocity":
        kind={"ddpm":"v","flow":"eps_minus_x0"}.get(noise_process)
    else:
        kind={"epsilon":"epsilon","sample":"x0"}.get(prediction_target)
    if kind is None or noise_process not in ("ddpm","flow"):
        raise ValueError("SD spatial alignment requires a declared prediction and noise process")
    return pair.prediction(value,kind=kind,sigma=sigma,alpha=alpha,noise=noise)
