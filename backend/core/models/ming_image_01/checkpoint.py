"""Atomic native Ming component-bundle export with topology-aware shards."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from core.models.common.single_file_format import DEFAULT_MAX_SHARD_BYTES, save_single_file_state
from .artifact import inspect_artifact, write_manifest
from .residual_unet import model_topology_metadata, validate_topology_identity


_PROJECTION_PREFIXES = ("query_tokens_dict.", "proj_in.", "proj_out.", "proj_directvlm.")


def _cpu_state(module):
    state = module.state_dict()
    if any(value.device.type != "cpu" for value in state.values()):
        raise ValueError("Ming bundle export requires CPU-offloaded components")
    return {key: value.detach().contiguous() for key, value in state.items()}


def save_bundle(components, destination, *, max_shard_bytes=DEFAULT_MAX_SHARD_BYTES, cancelled=None, training_metadata=None, quantization_metadata=None, progress_callback=None):
    """Publish only a fully preflighted package; never overwrite an existing path."""
    training_json = None if training_metadata is None else json.dumps(training_metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    quantization_json = None if quantization_metadata is None else json.dumps(quantization_metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    def progress(done, component):
        if progress_callback is not None:
            progress_callback(done, 6, component)
    destination = Path(destination).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    if type(max_shard_bytes) is not int or max_shard_bytes <= 0:
        raise ValueError("Ming shard limit must be a positive integer")
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    topology = getattr(components.transformer, "_residual_unet_spec", None)
    if topology is not None:
        validate_topology_identity(topology, components.transformer)
    # Offload is orchestration's responsibility, not an implicit residency change.
    for module in (components.transformer, components.conditioner, components.vae):
        if any(value.device.type != "cpu" for value in module.state_dict().values()):
            raise ValueError("Ming bundle export requires CPU-offloaded components")

    def check_cancel():
        if cancelled is not None and cancelled():
            raise InterruptedError("Ming checkpoint export cancelled")

    check_cancel()
    progress(0, "preflight")
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.ming-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        native = components.conditioner.native
        excluded = ("connector.", *_PROJECTION_PREFIXES)
        mllm_state = {key: value for key, value in _cpu_state(native).items() if not key.startswith(excluded)}
        native.save_pretrained(staging / "mllm", state_dict=mllm_state, max_shard_size=max_shard_bytes)
        del mllm_state
        progress(1, "mllm")
        check_cancel()
        native.connector.save_pretrained(staging / "connector", max_shard_size=max_shard_bytes)
        components.processor.tokenizer.save_pretrained(staging / "mllm")
        components.processor.image_processor.save_pretrained(staging / "mllm")
        progress(2, "connector")
        projections = staging / "mlp"
        projections.mkdir()
        (projections / "config.json").write_text(
            json.dumps(components.artifact.configs["mlp"], indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        from safetensors.torch import save_file
        save_file({key: value.detach().contiguous() for key, value in
                   components.conditioner.projection_state_dict().items()}, str(projections / "model.safetensors"))
        progress(3, "mlp")
        check_cancel()
        transformer_dir = staging / "transformer"
        transformer_dir.mkdir()
        components.transformer.save_config(transformer_dir)
        from .partition_global import GLOBAL_NAME, write_payload
        transformer_state = {key:value for key,value in _cpu_state(components.transformer).items()
                             if not key.startswith((GLOBAL_NAME+".", "vae_latent_bridge."))}
        save_single_file_state(transformer_state, model_topology_metadata(components.transformer),
            str(transformer_dir / "diffusion_pytorch_model.safetensors"), max_shard_bytes=max_shard_bytes)
        del transformer_state
        write_payload(components.transformer, staging)
        alignment = getattr(components.transformer, "vae_latent_bridge", None)
        if alignment is not None:
            from .vae_alignment import WEIGHTS
            if not alignment.ready:
                raise ValueError("Cannot export an uncalibrated Ming VAE alignment")
            save_file(_cpu_state(alignment), str(staging / WEIGHTS))
        check_cancel()
        progress(4, "transformer")
        components.vae.save_pretrained(staging / "vae", max_shard_size=max_shard_bytes)
        progress(5, "vae")
        components.scheduler.save_pretrained(staging / "scheduler")
        progress(6, "scheduler")
        if quantization_json is not None:
            (staging / "quantization.json").write_text(quantization_json, encoding="utf-8")
        if training_json is not None:
            (staging / "training.json").write_text(training_json, encoding="utf-8")
        write_manifest(staging, source=components.artifact.source)
        exported = inspect_artifact(staging)
        if exported.profile != components.artifact.profile:
            raise ValueError("Ming exported profile differs from the loaded package")
        from core.rewiring.artifacts import read_directory_topology
        if read_directory_topology(transformer_dir) != topology:
            raise ValueError("Ming exported shards disagree with the active topology")
        check_cancel()
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        os.rename(staging, destination)
    return destination / "manifest.json"
