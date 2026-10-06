"""Affine target-grid boundaries for Chimera's linear-state flow paths."""
from .flow import FLOW_V1_PREDICTION,FLOW_V2_PREDICTION,FLOW_V2_VELOCITY_PREDICTION

PENDING_KEY = "_chimera_vae_calibration_source"


def latent_geometry(unet):
    """External channels, codec scale, and pixel alignment on the target grid."""
    pair = getattr(unet, "vae_latent_bridge", None)
    if pair is None:
        return 4, 8, 8
    declaration = pair.declaration
    from core.models.components.latent_adapter import validate_declaration
    validate_declaration(declaration)
    channels, scale = declaration["new_channels"], declaration["new_scale"]
    blocks = getattr(getattr(unet, "config", None), "block_out_channels", (1,))
    return channels, scale, scale*2**(len(blocks)-1)


class TargetLatentMoments:
    """Pixel-weighted channel moments from the same bounded calibration pass."""
    def __init__(self, channels):
        import torch
        self.channels, self.count, self.images = channels, 0, 0
        self.mean = torch.zeros(channels, dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)

    def add(self, latent):
        import torch
        if latent.ndim != 4 or latent.shape[1] != self.channels:
            raise ValueError("Chimera calibration requires the target BCHW latent geometry")
        values = latent.detach().to(device="cpu", dtype=torch.float64).movedim(1, 0).flatten(1)
        if values.shape[1] == 0 or not torch.isfinite(values).all():
            raise ValueError("Chimera calibration contains empty or nonfinite target latents")
        count = values.shape[1]
        mean = values.mean(1)
        delta = mean - self.mean
        total = self.count + count
        self.m2 += (values - mean[:, None]).square().sum(1) + delta.square()*self.count*count/total
        self.mean += delta*count/total
        self.count, self.images = total, self.images + latent.shape[0]

    def finish(self):
        if not self.count:
            raise ValueError("Chimera target statistics require calibration latents")
        moment = float((self.m2/self.count).mean())
        if not 0 < moment < float("inf"):
            raise ValueError("Chimera target centered second moment must be finite and positive")
        return self.mean.tolist(), moment


def resolve_wiring(handler, trainer):
    from core.models.components.sd_vae_alignment import resolve_wiring as sd_wiring
    wiring = sd_wiring(handler, trainer)
    if getattr(getattr(trainer, "unet", None), "vae_latent_bridge", None) is not None:
        handler.pixel_align = wiring.vae_scale_factor * 2**(len(trainer.unet.config.block_out_channels)-1)
    return wiring


def initialize_swap(trainer, resolved, *, handler):
    from .artifact import prediction_contract
    from core.models.components.sd_vae_alignment import initialize_swap as sd_initialize
    prediction = trainer.chimera_prediction
    if prediction["type"] not in (FLOW_V1_PREDICTION, FLOW_V2_PREDICTION, FLOW_V2_VELOCITY_PREDICTION):
        raise ValueError("Chimera polar VAE swaps require direction/radius calibration wiring")
    prediction = prediction_contract(prediction["type"],
        latent_mean=prediction.get("latent_mean"),
        latent_centered_second_moment=prediction.get("latent_centered_second_moment"))
    if resolved.config.get("in_channels", resolved.config.get("input_channels", 3)) != 3:
        raise ValueError("Chimera VAE swap currently requires an RGB codec")
    report = sd_initialize(trainer, resolved, handler=handler)
    pending = getattr(trainer, "_sd_vae_calibration_source", None)
    if pending is not None:
        setattr(trainer, PENDING_KEY, pending)
        del trainer._sd_vae_calibration_source
        width = trainer.unet.config.in_channels
        if "latent_mean" in prediction:
            prediction["latent_mean"] += [0.]*(width-len(prediction["latent_mean"]))
        trainer.unet._chimera_source_prediction = prediction
    return report


def calibrate(trainer, datasets):
    from copy import deepcopy
    from core.models.components.spatial_swap import calibrate_images
    if getattr(trainer, PENDING_KEY, None) is None:
        return
    pair = trainer.unet.vae_latent_bridge
    moments = TargetLatentMoments(pair.declaration["new_channels"])

    def complete():
        source = trainer.unet._chimera_source_prediction
        target = deepcopy(source)
        mean, moment = moments.finish()
        if target["type"] != FLOW_V1_PREDICTION:
            target.update(latent_mean=mean, latent_centered_second_moment=moment)
        trainer.chimera_prediction = target
        trainer.chimera_vae_calibration = dict(
            images=moments.images, positions_per_channel=moments.count,
            latent_mean=mean, latent_centered_second_moment=moment,
        )

    from core.training.base_trainer import all_optimizers
    from core.training.optimizer_residency import park_optimizer_state
    teacher = getattr(trainer, "chimera_teacher", None)
    extras = [module for module in (getattr(trainer, "condition_bridge", None),
              getattr(teacher, "text_encoder", None), getattr(teacher, "text_encoder_2", None))
              if module is not None]
    devices = [next(module.parameters()).device for module in extras]
    parameters = [parameter for module in extras for parameter in module.parameters()]
    with park_optimizer_state(all_optimizers(trainer), parameters):
        try:
            for module in extras:
                module.to("cpu")
            calibrate_images(trainer, datasets, model=trainer.unet, pending_key=PENDING_KEY,
                             target_observer=moments.add, on_complete=complete)
        finally:
            for module, device in zip(extras, devices):
                module.to(device)


def path_coefficients(prediction,timestep,reference):
    import torch
    kind=prediction["type"]
    if kind==FLOW_V1_PREDICTION:
        alpha=torch.as_tensor(timestep,device=reference.device,dtype=reference.dtype)
        return alpha,torch.ones_like(alpha)
    if kind in (FLOW_V2_PREDICTION,FLOW_V2_VELOCITY_PREDICTION):
        from .flow import endpoint_observable_coefficients
        alpha,_,derivative,_=endpoint_observable_coefficients(timestep,reference)
        return alpha.reshape(reference.shape[0]),derivative.reshape(reference.shape[0])
    raise ValueError("Chimera polar swaps require direction/radius reconstruction, not a linear-state boundary")


def map_state(unet,sample,timestep,*,prediction):
    pair=getattr(unet,"vae_latent_bridge",None)
    if pair is None:return sample
    alpha,_=path_coefficients(prediction,timestep,sample)
    return pair.input.diffusion_state(sample,alpha=alpha)


def map_velocity(unet,velocity,timestep,*,prediction):
    pair=getattr(unet,"vae_latent_bridge",None)
    if pair is None:return velocity
    _,derivative=path_coefficients(prediction,timestep,velocity)
    return pair.output.diffusion_velocity(velocity,clean_derivative=derivative)


def map_residual(unet,source_sample,source_residual,target_sample,timestep,*,source_prediction,target_prediction):
    from .flow import endpoint_observable_reconstruct_velocity,endpoint_observable_preconditioning
    if source_prediction["type"]!=FLOW_V2_PREDICTION or target_prediction["type"]!=FLOW_V2_PREDICTION:
        raise ValueError("Chimera residual alignment requires source and target v2 residual contracts")
    velocity=endpoint_observable_reconstruct_velocity(source_sample,source_residual,timestep,
        latent_mean=source_prediction["latent_mean"],
        latent_centered_second_moment=source_prediction["latent_centered_second_moment"])
    mapped=map_velocity(unet,velocity,timestep,prediction=target_prediction)
    analytic,_=endpoint_observable_preconditioning(target_sample,timestep,
        latent_mean=target_prediction["latent_mean"],
        latent_centered_second_moment=target_prediction["latent_centered_second_moment"])
    return mapped-analytic


def unet_forward(unet, sample, timestep, *, prediction, **kwargs):
    """One boundary shared by training and CFG inference branches."""
    internal = map_state(unet, sample, timestep, prediction=prediction)
    pair = getattr(unet, "vae_latent_bridge", None)
    source_prediction = getattr(unet, "_chimera_source_prediction", None)
    if pair is not None and prediction["type"] == FLOW_V2_PREDICTION and source_prediction is None:
        raise ValueError("Aligned Chimera residual prediction requires its saved source contract")
    kwargs["return_dict"] = False
    raw = unet(internal, timestep, **kwargs)[0]
    if pair is None:
        return raw
    if prediction["type"] == FLOW_V2_PREDICTION:
        return map_residual(
            unet, internal, raw, sample, timestep,
            source_prediction=source_prediction, target_prediction=prediction,
        )
    return map_velocity(unet, raw, timestep, prediction=prediction)
