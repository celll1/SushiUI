"""Qwen image-token boundaries for the shared target-native codec bridge."""
import torch

from core.models.components.latent_adapter import SpatialAlignmentPair, NormalizedFrameVAE

NAME = "vae_latent_bridge"


def install(model, declaration):
    if declaration is not None:
        model.add_module(NAME, SpatialAlignmentPair(declaration))
        model.register_to_config(vae_alignment=declaration)


def noise_kwargs(model, noise):
    if getattr(getattr(model,NAME,None),"noise_alignment",None) is None:
        return {}
    if noise is None or noise.ndim != 3:
        raise ValueError("Aligned Qwen requires this exposure's target-token noise")
    return {"alignment_noise":noise}


def _noise_grid(noise, frames, h, w):
    if noise is None:
        return None
    if noise.ndim != 3 or noise.shape[1] != frames*h*w:
        raise ValueError("Aligned Qwen noise must contain only the target image tokens")
    return noise.transpose(1,2).reshape(noise.shape[0],noise.shape[-1],frames,h,w)


def map_input(model, states, timestep, shapes, *, noise=None):
    pair = getattr(model, NAME)
    if not pair.ready:
        raise RuntimeError("Qwen VAE swap requires paired calibration")
    layout = shapes[0]
    if any(value != layout for value in shapes):
        raise ValueError("Aligned Qwen batches must share their image-token layout")
    if sum(f*h*w for f,h,w in layout) != states.shape[1]:
        raise ValueError("Qwen image-token count disagrees with the VAE alignment layout")
    chunks, offset = [], 0
    for i,(frames,h,w) in enumerate(layout):
        count = frames*h*w
        grid = states[:,offset:offset+count].transpose(1,2).reshape(states.shape[0],states.shape[-1],frames,h,w)
        grid = pair.flow_state(grid,timestep,noise=_noise_grid(noise,frames,h,w)) if i == len(layout)-1 else pair.input(grid)
        chunks.append(grid.flatten(2).transpose(1,2))
        offset += count
    return torch.cat(chunks,dim=1)


def map_output(model, output, shapes, *, noise=None):
    frames,h,w = shapes[0][-1]
    count = frames*h*w
    grid = output[:,-count:].transpose(1,2).reshape(output.shape[0],output.shape[-1],frames,h,w)
    mapped = getattr(model,NAME).flow_velocity(grid,convention="eps_minus_x0",
        noise=_noise_grid(noise,frames,h,w)).flatten(2).transpose(1,2)
    # Text/reference outputs are discarded by native Qwen callers.
    return torch.cat((mapped.new_zeros(output.shape[0],output.shape[1]-count,mapped.shape[-1]),mapped),dim=1)


def map_global_grid(model, grid, sigma, *, noise=None):
    pair = getattr(model,NAME,None)
    if pair is None:
        return grid
    return pair.flow_state(grid.permute(0,3,1,2),sigma,
        noise=None if noise is None else noise.permute(0,3,1,2)).permute(0,2,3,1)


def resize_global_payload(model, old_channels, new_channels):
    from .partition_global import GLOBAL_ADAPTER_NAME
    if old_channels == new_channels:
        return
    if getattr(model,GLOBAL_ADAPTER_NAME,None) is not None:
        raise ValueError("Resize the VAE before installing the Qwen global adapter")
    payload = getattr(model,"_qwen_partition_global_payload",None)
    if payload is None:
        return
    for key in ("key.weight","value.weight","query.weight"):
        old = payload["state"][key]
        new = old.new_zeros(old.shape[0],new_channels+2)
        new[:,:min(old_channels,new_channels)] = old[:,:min(old_channels,new_channels)]
        new[:,-2:] = old[:,-2:]
        payload["state"][key] = new


def configure_swap(trainer, components):
    from core.training.vae_swap import resolve_vae_swap_source, check_swap_method, initialization_mode
    from core.training.ops.training_method import resolve_training_method
    from core.models.common.vae_source import resolve_vae_source
    from core.models.components.latent_adapter import ImageCodecContract, SpatialSwapPlan
    from core.models.components.latent_io import resize_latent_io
    from core.models.components.wiring import QWEN_IMAGE_21_LATENT_IO
    from core.models.common.module_identity import module_fingerprint
    from .residual_unet import native_config, restore_topology
    from core.rewiring.spec import rebase_spec
    source = resolve_vae_swap_source(trainer.config)
    if not source:
        return
    check_swap_method(source,resolve_training_method(trainer))
    if initialization_mode(trainer.config,"qwen_image_21") != "paired_spatial":
        raise ValueError("Qwen VAE swaps require paired_spatial initialization")
    model = trainer.transformer
    existing = getattr(model,NAME,None)
    if existing is not None:
        if source == existing.declaration["source"]:
            return
        raise ValueError("Changing an already aligned Qwen VAE requires a fresh base checkpoint")
    # Expanded IO must be materialized before resize, not dropped as INT8 buffers.
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    if any(isinstance(m,ConvRotInt8Linear) for m in model.modules()):
        raise ValueError("Initialize a Qwen VAE swap from a bf16 full checkpoint")
    resolved = resolve_vae_source(source,arch="qwen_image_21")
    codec = ImageCodecContract.from_resolved(resolved)
    old = trainer.vae
    target = resolved.load_module(torch_dtype=trainer.vae_dtype).requires_grad_(False).eval()
    identity = module_fingerprint(old)
    if identity == module_fingerprint(target):
        from core.models.common.vae_source import normalization_config
        if normalization_config(old.config) == normalization_config(target.config):
            return
    declaration = codec.declaration(source=source,source_identity=identity,target_identity=resolved.latent_hash,
                                   old_channels=int(model.config.in_channels),old_scale=16)
    if codec.spatial_scale > declaration["old_scale"] and codec.latent_channels >= declaration["old_channels"]:
        from core.models.components.gaussian_alignment import POLICY
        declaration["noise_policy"] = POLICY
    plan = SpatialSwapPlan.build(declaration,patch_size=2)
    from core.rewiring.training import prepare_topology_io_rebase
    finish_topology = prepare_topology_io_rebase(trainer.config,"qwen_image_21",native_config(dict(model.config)),declaration=declaration)
    topology = getattr(model,"_residual_unet_spec",None)
    resize_latent_io(model,QWEN_IMAGE_21_LATENT_IO,plan.backbone_channels)
    resize_global_payload(model,declaration["old_channels"],plan.backbone_channels)
    model.out_channels = plan.backbone_channels
    if finish_topology is not None:
        finish_topology(native_config(dict(model.config)))
    if topology is not None:
        restore_topology(model,rebase_spec(topology,native_config(dict(model.config))))
    install(model,declaration)
    getattr(model,NAME).to(dtype=trainer.weight_dtype)
    trainer._qwen_vae_calibration_source = old.to("cpu").requires_grad_(False).eval()
    trainer.vae = components["vae"] = NormalizedFrameVAE(target,declaration)
    from dataclasses import replace
    trainer.vae_identity = replace(resolved,state_dict=None)
    trainer.vae_latent_channels = codec.latent_channels
    components.update(vae_scale_factor=codec.spatial_scale,latent_channels=codec.latent_channels,pixel_alignment=plan.pixel_alignment)


def calibrate(trainer,datasets):
    old = getattr(trainer,"_qwen_vae_calibration_source",None)
    if old is None:
        return
    from api.param_defaults import TRAINING_DEFAULTS
    from PIL import Image, ImageOps
    import numpy as np
    from core.models.components.latent_adapter import PairedSpatialCalibration, calibration_source_side
    from core.models.components.vae_registry import normalize
    from core.training.optimizer_residency import encoder_refill_residency
    from core.training.optimizer_residency import park_optimizer_state
    from core.training.base_trainer import all_optimizers
    count = trainer.config.get("vae_swap_calibration_samples",TRAINING_DEFAULTS["vae_swap_calibration_samples"])
    side = trainer.config.get("vae_swap_calibration_resolution",TRAINING_DEFAULTS["vae_swap_calibration_resolution"])
    if type(count) is not int or not 2 <= count <= 256 or side not in (128,256,512):
        raise ValueError("Invalid VAE spatial calibration count/resolution")
    paths = list(dict.fromkeys(item["image_path"] for dataset in datasets for item in dataset.items if item.get("image_path")))
    if not paths:
        raise ValueError("Qwen VAE calibration requires training images")
    paths = [paths[i] for i in torch.linspace(0,len(paths)-1,min(count,len(paths))).long().tolist()]
    pair = getattr(trainer.transformer,NAME)
    source_side = calibration_source_side(side, pair.declaration)
    fit = PairedSpatialCalibration(pair,ridge=1e-2)
    text_device = next(trainer.text_encoder.parameters()).device
    callback = getattr(trainer,"_vae_alignment_progress_callback",None)
    def encode(module,path,*,native):
        with Image.open(path) as image:
            image_side = source_side if native else side
            image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGBA"),(image_side,image_side))
            pixels = torch.from_numpy(np.array(image).copy()).permute(2,0,1).float()/127.5-1
        pixels = pixels[None,:,None].to(device=trainer.device,dtype=trainer.vae_dtype)
        z = module.encode(pixels).latent_dist.mode()
        return (normalize(z,module,trainer.arch.wiring) if native else z).cpu()
    with encoder_refill_residency(trainer,text=False,vae=True), park_optimizer_state(
            all_optimizers(trainer), trainer.text_encoder.parameters()):
        trainer.text_encoder.to("cpu")
        try:
            old.to(trainer.device)
            with torch.no_grad():
                paired = []
                for i,path in enumerate(paths):
                    trainer._check_stop_requested()
                    paired.append(encode(old,path,native=True))
                    if callback: callback(phase="latent_cache",step=i+1,total=2*len(paths))
            old.to("cpu")
            if torch.device(trainer.device).type == "cuda": torch.cuda.empty_cache()
            trainer.vae.to(trainer.device)
            with torch.no_grad():
                for i,(path,legacy) in enumerate(zip(paths,paired)):
                    trainer._check_stop_requested()
                    fit.add(legacy,encode(trainer.vae,path,native=False))
                    if callback: callback(phase="latent_cache",step=len(paths)+i+1,total=2*len(paths))
            fit.finish()
            del trainer._qwen_vae_calibration_source
        finally:
            old.to("cpu")
            trainer.vae.to("cpu")
            trainer.text_encoder.to(text_device)


def load_saved_vae(path,declaration,dtype):
    from core.models.common.vae_source import load_declared_latent_io
    resolved = load_declared_latent_io(path,arch="qwen_image_21")
    if resolved is None:
        raise ValueError("Aligned Qwen checkpoint has no declared target VAE")
    if (resolved.latent_channels,resolved.scale_factor,resolved.norm,resolved.norm_pack,resolved.ndim) != (
        declaration["new_channels"],declaration["new_scale"],declaration["norm"],declaration["norm_pack"],declaration["ndim"]):
        raise ValueError("Saved Qwen VAE disagrees with its alignment contract")
    from dataclasses import replace
    return NormalizedFrameVAE(resolved.load_module(torch_dtype=dtype),declaration),replace(resolved,state_dict=None)
