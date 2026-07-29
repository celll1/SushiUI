"""Standalone VAE / autoencoder fine-tuning modality (Phase 1: decoder-only).

This package is deliberately NOT built on ``core.training.base_trainer``:
``BaseTrainer`` is a *diffusion* spine (noise scheduler, timestep sampler, SNR
weighting, latent cache) and its ``encode_image`` wraps the VAE forward in
``torch.no_grad()`` — which is exactly the tensor a VAE trainer needs gradients
through. Nothing here subclasses ``BaseTrainer``, edits ``base_trainer.py`` or
registers an arch key; the only integration point is the
``network.type == "vae_decoder"`` branch in ``core/training/train_runner.py``,
which buys the existing TrainingRun/config_yaml storage, subprocess launch,
``.stop_training`` sentinel, checkpoint routes, metrics channel and Training
Monitor UI unchanged.

See ``scratchpad/vae_training/design.md`` (§3, §4, §5 and especially §9, the
Phase-0 measurement outcomes that override the earlier sections).
"""

from core.training.vae.vae_config import (
    VaeConfigError,
    resolve_vae_training_config,
)

__all__ = ["VaeConfigError", "resolve_vae_training_config"]
