"""Atomic bootstrap builder for SenseNova SDXL Chimera artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from core.models.common.single_file_format import dedup_tensors, save_single_file_state
from core.models.common.vae_source import ResolvedVAE
from core.models.sensenova.loader import is_sensenova_state_dict_keys

from .artifact import (
    CONFIG_NAME,
    MANIFEST_NAME,
    MODEL_TYPE,
    WEIGHTS_BASENAME,
    checkpoint_content_hash,
    checkpoint_headers,
    config_hash,
    infer_bridge_config,
    manifest_template,
    prefixed_state,
    runtime_config,
)
from .conditioning_bridge import ConditioningBridge
from .unet import build_donor_equal_unet


def _source_config(source: str) -> dict[str, Any]:
    shapes, metadata = checkpoint_headers(source)
    if not is_sensenova_state_dict_keys(shapes):
        raise ValueError(f"understanding source is not a supported SenseNova checkpoint: {source}")
    embedded = metadata.get("sensenova_config")
    if embedded:
        raw = json.loads(embedded)
    else:
        config_path = Path(source).resolve().parent / CONFIG_NAME
        if not config_path.is_file():
            raise FileNotFoundError(
                f"SenseNova source has no sensenova_config metadata or sibling config.json: {source}"
            )
        with config_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"SenseNova config must be a JSON object: {source}")
    return raw


def _vae_facts(vae: ResolvedVAE) -> dict[str, Any]:
    facts = vae.facts()
    facts.update({"embedded": True, "config_hash": config_hash(vae.config)})
    return facts


def build_chimera_artifact_from_components(
    output_directory: str,
    *,
    understanding_source: str,
    sdxl_source: str,
    donor_unet: torch.nn.Module,
    vae: ResolvedVAE,
    unet_initialization: str = "scratch",
    initialization_seed: int = 0,
    context_tokens: int = 77,
    max_shard_bytes: int = 10 * 1024**3,
) -> dict[str, Any]:
    """Build into an empty directory; callers own atomic target publication."""
    root = Path(output_directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Chimera target must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if context_tokens != 77:
        raise ValueError("Chimera format v1 fixes context_tokens to 77")
    if vae.state_dict is None:
        raise ValueError("builder requires a VAE resolved with load_weights=True")
    if vae.latent_channels != 4 or vae.scale_factor != 8 or vae.ndim != 4:
        raise ValueError(
            "Chimera format v1 requires an SDXL-compatible 4-channel, 8x, 2-D VAE; "
            f"got {vae.latent_channels}ch/{vae.scale_factor}x/ndim={vae.ndim}"
        )

    und_config = _source_config(understanding_source)
    bridge_config = infer_bridge_config(und_config, context_tokens=context_tokens)
    bridge = ConditioningBridge(bridge_config)
    unet, report = build_donor_equal_unet(
        donor_unet, initialization=unet_initialization, seed=initialization_seed
    )
    unet_config = dict(unet.config)
    config = runtime_config(
        unet_config=unet_config,
        bridge_config=bridge_config,
        vae_config=vae.config,
    )
    manifest = manifest_template(
        understanding_source=understanding_source,
        understanding_hash=checkpoint_content_hash(understanding_source),
        understanding_config_hash=config_hash(und_config),
        sdxl_source=sdxl_source,
        sdxl_hash=checkpoint_content_hash(sdxl_source),
        initialization=unet_initialization,
        unet_config=unet_config,
        parameter_count=report.parameter_count,
        bridge_config=bridge_config,
        vae_facts=_vae_facts(vae),
    )

    tensors, dropped = dedup_tensors((
        *prefixed_state(bridge, "condition_bridge."),
        *prefixed_state(unet, "unet."),
        *((f"vae.{name}", tensor) for name, tensor in vae.state_dict.items()),
    ))
    metadata = {
        "model_type": MODEL_TYPE,
        "format": "pt",
        "format_version": "1",
        "tied_weights_dropped": json.dumps(dropped),
    }
    written = save_single_file_state(
        tensors, metadata, str(root / WEIGHTS_BASENAME), max_shard_bytes=max_shard_bytes
    )
    with (root / CONFIG_NAME).open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    # The manifest is the completion marker and is deliberately written last.
    with (root / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return {"directory": str(root), "weights": written, "manifest": manifest, "config": config}


def initialize_chimera_atomically(
    model_root: str,
    output_name: str,
    *,
    build_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(model_root).resolve()
    if not output_name or Path(output_name).name != output_name or output_name in {".", ".."}:
        raise ValueError("output_name must be one relative directory name")
    target = (root / output_name).resolve()
    if target.parent != root:
        raise ValueError("Chimera target escapes the configured model root")
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise FileExistsError(f"Chimera target already exists and is not empty: {target}")
        target.rmdir()
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_name}.building-", dir=str(root)))
    try:
        result = build_chimera_artifact_from_components(str(temporary), **dict(build_kwargs))
        from .loader import load_chimera_artifact

        load_chimera_artifact(str(temporary), load_understanding=False)
        os.replace(temporary, target)
        result["directory"] = str(target)
        result["weights"] = str(target / Path(result["weights"]).name)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
