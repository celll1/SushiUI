"""Affine target-grid boundaries for Chimera's linear-state flow paths."""
from .flow import FLOW_V1_PREDICTION,FLOW_V2_PREDICTION,FLOW_V2_VELOCITY_PREDICTION


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
