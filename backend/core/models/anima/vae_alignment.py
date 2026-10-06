"""Anima image boundaries; padding-mask channels stay outside the codec bridge."""
def map_image(model, value, *, sigma=None, output=False):
    pair=getattr(model,"vae_latent_bridge",None)
    if pair is None:
        return value
    if value.ndim!=5 or value.shape[2]!=1:
        raise ValueError("Anima spatial VAE swap requires a single image frame")
    image=value.squeeze(2)
    if output:
        image=pair.output.flow_velocity(image,convention="eps_minus_x0")
    else:
        if sigma.ndim==2 and sigma.shape[1]==1:
            sigma=sigma[:,0]
        image=pair.input.flow_state(image,sigma)
    return image.unsqueeze(2)
