"""Recover x_0 from a model prediction — one entry point for 11 of the 13 archs.

The math itself is NOT duplicated here: ``predict_x0`` delegates to
``base_trainer.predict_original_latent_unified``, and ``snr_band_mask`` to
``base_trainer._per_sample_snr``.

Velocity sign is the trap: passing the wrong ``velocity_sign`` returns a
plausible-looking tensor that is wrong by ``2*t*v``. Which sign an architecture
trains is declared once, on ``arch/base_arch.ArchHandler.velocity_sign``, and
callers read it from there rather than spelling a literal.

MiniT2I and SenseNova declare ``None`` and are OUT OF SCOPE: their models emit
x_0 directly under the t=1-is-clean convention and their targets carry an extra
scale (``noise_scale``, a clamped denominator), so no argument here returns
their x_0. See ``ops/minit2i_ops.py`` and ``ops/sensenova_ops.py``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

NOISE_PROCESSES = ("ddpm", "flow")
PREDICTION_TARGETS = ("epsilon", "velocity", "sample")
VELOCITY_SIGNS = ("eps_minus_x0", "x0_minus_eps")


def _validate(
    noise_process: str, noise_scheduler: Any, prediction_target: Optional[str] = None
) -> None:
    if noise_process not in NOISE_PROCESSES:
        raise ValueError(
            f"Unsupported noise_process {noise_process!r}; supported: {NOISE_PROCESSES}"
        )
    if prediction_target is not None and prediction_target not in PREDICTION_TARGETS:
        raise ValueError(
            f"Unsupported prediction_target {prediction_target!r}; supported: {PREDICTION_TARGETS}"
        )
    if noise_process == "ddpm" and noise_scheduler is None:
        raise ValueError("noise_process='ddpm' needs noise_scheduler for alphas_cumprod")


def predict_x0(
    noise_process: str,
    prediction_target: str,
    noisy_latents: torch.Tensor,
    model_pred: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: Any = None,
    velocity_sign: Optional[str] = None,
) -> torch.Tensor:
    """Predicted clean latent x_0 from ``model_pred`` at ``noisy_latents``/``timesteps``.

    ``timesteps`` uses the NOISE-AT-THE-TOP convention: flow ``t`` in [0,1] with
    ``t=1`` pure noise, ddpm discrete ``t`` with ``t=num_train_timesteps-1``
    the noisiest — i.e. the noise-mixing coordinate, not the trainer's raw
    ``timestep_sampler`` draw, which ``ArchHandler.timestep_convention`` describes
    and which ``ops/sd_sdxl_ops.py`` flips (``(1-t)*num_train_timesteps``) before
    it reaches here.

    Differentiable in ``model_pred`` and ``noisy_latents``: no ``no_grad``/``detach``
    inside, because the aux-loss callers this exists for need the gradient.

    ``velocity_sign`` has no default on purpose. A wrong sign is silent — no
    exception, no NaN, just a tensor off by ``2*t*v`` — so the caller passes its
    arch's ``ArchHandler.velocity_sign`` rather than inheriting one.
    """
    noise_process = str(noise_process).strip().lower()
    prediction_target = str(prediction_target).strip().lower()
    _validate(noise_process, noise_scheduler, prediction_target)
    if prediction_target == "velocity":
        if velocity_sign is None:
            raise ValueError(
                f"prediction_target='velocity' needs an explicit velocity_sign; "
                f"supported: {VELOCITY_SIGNS}"
            )
        velocity_sign = str(velocity_sign).strip().lower()
        if velocity_sign not in VELOCITY_SIGNS:
            raise ValueError(
                f"Unsupported velocity_sign {velocity_sign!r}; supported: {VELOCITY_SIGNS}"
            )

    if prediction_target == "velocity" and velocity_sign == "x0_minus_eps":
        # -v is exactly the eps_minus_x0 velocity, so the shared formula covers both.
        model_pred = -model_pred

    from core.training.base_trainer import predict_original_latent_unified

    return predict_original_latent_unified(
        noise_process=noise_process,
        prediction_target=prediction_target,
        noise_scheduler=noise_scheduler,
        noisy_latents=noisy_latents,
        model_pred=model_pred,
        timesteps=timesteps,
    )


def snr_band_mask(
    noise_process: str,
    timesteps: torch.Tensor,
    noise_scheduler: Any = None,
    snr_min: Optional[float] = None,
    snr_max: Optional[float] = None,
    alphas_cumprod_cached: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Bool [B] mask of samples whose SNR lies in ``[snr_min, snr_max]`` (both inclusive,
    either may be None for open-ended).

    Bands are expressed in SNR, not raw ``t``: the same ``t`` means a different noise
    level under ddpm and flow, so a t-valued band would not transfer across archs.
    """
    noise_process = str(noise_process).strip().lower()
    _validate(noise_process, noise_scheduler)

    from core.training.base_trainer import _per_sample_snr

    snr = _per_sample_snr(noise_process, timesteps, noise_scheduler, alphas_cumprod_cached)
    mask = torch.ones_like(snr, dtype=torch.bool)
    if snr_min is not None:
        mask &= snr >= float(snr_min)
    if snr_max is not None:
        mask &= snr <= float(snr_max)
    return mask
