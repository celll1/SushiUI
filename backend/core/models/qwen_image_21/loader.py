"""Load Qwen-Image 2.1 from a Diffusers directory or SushiUI split artifact."""

from __future__ import annotations

import json
import os

import torch

from .artifact import load_manifest, load_text_encoder, load_transformer as load_transformer_artifact, load_vae
from .vendor import AutoencoderKLQwenImage21, QwenImage21Pipeline, QwenImage21Transformer2DModel


def _load_hybrid_directory(directory, config, dtype, topology):
    from pathlib import Path
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from safetensors.torch import load_file
    from .artifact import _finish_load
    from .residual_unet import restore_topology
    from .partition_global import GLOBAL_ADAPTER_NAME, stash_partition_global_payload

    root = Path(directory).resolve()
    indices = list(root.glob("*.safetensors.index.json"))
    if len(indices) > 1:
        raise ValueError("Qwen hybrid directory has ambiguous shard indices")
    if indices:
        with indices[0].open(encoding="utf-8") as handle:
            index = json.load(handle)
        mapping = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(mapping, dict) or not mapping or any(not isinstance(name, str) for name in mapping.values()):
            raise ValueError("Qwen hybrid directory requires a valid shard weight_map")
        names = sorted(set(mapping.values()))
        paths = [(root / name).resolve() for name in names]
        if any(path.parent != root or path.suffix != ".safetensors" for path in paths):
            raise ValueError("Qwen hybrid shards must stay inside the transformer directory")
    else:
        paths = sorted(root.glob("*.safetensors"))
    config = dict(config)
    declaration = config.pop("vae_alignment", None)
    with init_empty_weights():
        model = QwenImage21Transformer2DModel.from_config(config)
        from .vae_alignment import install
        install(model, declaration)
        restore_topology(model, topology)
    expected = model.state_dict()
    keys, metadata, global_state = set(), None, {}
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if metadata is None:
                metadata = handle.metadata() or {}
            elif (handle.metadata() or {}) != metadata:
                raise ValueError("Qwen hybrid shard metadata disagrees")
            for key in handle.keys():
                if indices and mapping.get(key) != path.name:
                    raise ValueError("Qwen hybrid shard keys disagree with the index")
                if key in keys:
                    raise ValueError("Qwen hybrid directory has duplicate tensor keys")
                keys.add(key)
                if key.startswith(GLOBAL_ADAPTER_NAME + "."):
                    continue
                if key not in expected or tuple(handle.get_slice(key).get_shape()) != tuple(expected[key].shape):
                    raise ValueError(f"Qwen hybrid directory state mismatch: {key}")
    if set(expected) - keys:
        raise ValueError("Qwen hybrid directory is missing native or learned weights")
    if indices and keys != set(mapping):
        raise ValueError("Qwen hybrid shard index has missing tensor keys")
    for path in paths:
        state = load_file(str(path), device="cpu")
        for key, value in state.items():
            if key.startswith(GLOBAL_ADAPTER_NAME + "."):
                global_state[key] = value
            else:
                if expected[key].is_floating_point() and not value.is_floating_point():
                    raise ValueError("Packed Qwen hybrid directory weights require a split artifact")
                set_module_tensor_to_device(model, key, "cpu", value=value,
                    dtype=dtype if expected[key].is_floating_point() else expected[key].dtype)
        del state
    if global_state:
        stash_partition_global_payload(model, global_state, metadata or {})
        if global_state:
            raise ValueError("Qwen hybrid directory global payload was not restored")
    # All bridge buffers must be assigned before their load hooks run.
    model = _finish_load(model, model.state_dict(), "Qwen hybrid directory")
    if declaration is not None and not model.vae_latent_bridge.ready:
        raise ValueError("Saved Qwen VAE alignment is not calibrated")
    return model


def _source_components(
    model_path: str, dtype: torch.dtype, *, load_transformer_component: bool = True,
    load_text_encoder_component: bool = True,
) -> dict:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    required = ("transformer", "text_encoder", "vae", "processor")
    missing = [name for name in required if not os.path.isdir(os.path.join(model_path, name))]
    if missing:
        raise ValueError(f"{model_path}: missing Qwen-Image 2.1 subfolder(s): {', '.join(missing)}")
    from core.rewiring.artifacts import read_directory_topology
    from .residual_unet import restore_topology, validate_topology_identity

    topology = None
    if load_transformer_component:
        topology = read_directory_topology(os.path.join(model_path, "transformer"))
        if topology is not None:
            config = QwenImage21Transformer2DModel.load_config(model_path, subfolder="transformer")
            topology = validate_topology_identity(topology, config)
    if load_transformer_component and (topology or {}).get("method") == "hybrid_unet_ssdit":
        transformer = _load_hybrid_directory(os.path.join(model_path, "transformer"), config, dtype, topology)
    else:
        transformer = QwenImage21Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=dtype, low_cpu_mem_usage=True
        ).eval() if load_transformer_component else None
    if transformer is not None:
        restore_topology(transformer, topology)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval() if load_text_encoder_component else None
    vae = AutoencoderKLQwenImage21.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval()
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_path, "processor"))
    scheduler_dir = os.path.join(model_path, "scheduler")
    scheduler = (
        FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_dir)
        if os.path.isdir(scheduler_dir)
        else FlowMatchEulerDiscreteScheduler()
    )
    return {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "vae": vae,
        "processor": processor,
        "scheduler": scheduler,
        "transformer_variant": "bf16",
        "text_encoder_variant": "bf16",
    }


def _artifact_components(
    model_path: str, dtype: torch.dtype, load_text_encoder_component: bool,
    load_transformer_component: bool = True,
) -> dict:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import Qwen3VLProcessor

    manifest = load_manifest(model_path)
    transformer, transformer_variant = (
        load_transformer_artifact(manifest.transformer, manifest.transformer_config, dtype)
        if load_transformer_component else (None, None)
    )
    text_encoder = None
    text_encoder_variant = None
    if load_text_encoder_component:
        text_encoder, text_encoder_variant = load_text_encoder(
            manifest.text_encoder, manifest.text_encoder_config, dtype
        )
    vae = load_vae(manifest.vae, manifest.vae_config, dtype)
    processor = Qwen3VLProcessor.from_pretrained(manifest.processor)
    scheduler = (
        FlowMatchEulerDiscreteScheduler.from_pretrained(manifest.scheduler)
        if manifest.scheduler and os.path.isdir(manifest.scheduler)
        else FlowMatchEulerDiscreteScheduler()
    )
    return {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "vae": vae,
        "processor": processor,
        "scheduler": scheduler,
        "transformer_variant": transformer_variant,
        "text_encoder_variant": text_encoder_variant,
        "manifest": manifest,
    }


def load_qwen_image_21_components(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    *,
    load_text_encoder: bool = True,
    load_transformer: bool = True,
) -> dict:
    """Return CPU-resident components consumed by ``PipelineManager`` and trainers."""
    manifest_path = os.path.join(model_path, "manifest.json") if os.path.isdir(model_path) else ""
    if manifest_path and os.path.isfile(manifest_path):
        components = _artifact_components(
            model_path, torch_dtype, load_text_encoder, load_transformer
        )
        components["companion_path"] = os.path.abspath(model_path)
    elif os.path.isdir(model_path):
        components = _source_components(
            model_path, torch_dtype,
            load_transformer_component=load_transformer,
            load_text_encoder_component=load_text_encoder,
        )
        components["companion_path"] = os.path.abspath(model_path)
    elif os.path.isfile(model_path):
        from core.models.common.single_file_format import is_index_path
        if is_index_path(model_path):
            with open(model_path, encoding="utf-8") as handle:
                metadata = (json.load(handle).get("metadata") or {})
        else:
            from safetensors import safe_open
            with safe_open(model_path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
        component = metadata.get("component")
        if metadata.get("model_type") != "qwen_image_21" or component not in {
            "transformer", "training_bundle"
        }:
            raise ValueError(f"{model_path}: not a Qwen-Image 2.1 training checkpoint")
        from core.rewiring.artifacts import read_artifact_topology
        from .residual_unet import restore_topology, validate_topology_identity
        topology = read_artifact_topology(model_path)
        config = json.loads(metadata.get("config") or "{}")
        if topology is not None:
            topology = validate_topology_identity(topology, config)
        companion_path = metadata.get("companion_path")
        if not companion_path or os.path.abspath(companion_path) == os.path.abspath(model_path):
            raise ValueError(
                f"{model_path}: training checkpoint has no usable companion_path for TE/VAE/processor"
            )
        components = load_qwen_image_21_components(
            companion_path, torch_dtype=torch_dtype,
            load_text_encoder=load_text_encoder if component == "transformer" or metadata.get("train_text_encoder") == "0" else False,
            load_transformer=component == "transformer",
        )
        config = json.loads(metadata.get("config") or "{}")
        if component == "training_bundle":
            from accelerate import init_empty_weights
            from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
            from core.models.common.single_file_format import read_state_dict
            from .artifact import _finish_load

            state, _ = read_state_dict(model_path)
            transformer_state = {
                key[len("transformer."):]: value for key, value in state.items()
                if key.startswith("transformer.")
            }
            te_state = {
                key[len("text_encoder."):]: value for key, value in state.items()
                if key.startswith("text_encoder.")
            }
            vae_state = {key:value for key,value in state.items() if key.startswith("vae.")}
            declaration = config.pop("vae_alignment", None)
            if (not transformer_state or (not te_state and declaration is None)
                    or len(transformer_state) + len(te_state) + len(vae_state) != len(state)):
                raise ValueError(f"{model_path}: incomplete Qwen-Image 2.1 training bundle")
            with init_empty_weights():
                transformer = QwenImage21Transformer2DModel(**config)
                from .vae_alignment import install
                install(transformer, declaration)
                restore_topology(transformer, topology)
            from .partition_global import stash_partition_global_payload
            stash_partition_global_payload(transformer, transformer_state, metadata)
            transformer = _finish_load(transformer, transformer_state, "Qwen-Image 2.1 transformer")
            if declaration is not None and not transformer.vae_latent_bridge.ready:
                raise ValueError("Saved Qwen VAE alignment is not calibrated")
            transformer.to(dtype=torch_dtype)
            restore_topology(transformer, topology)
            if load_text_encoder and te_state:
                te_config_path = (
                    components["manifest"].text_encoder_config
                    if components.get("manifest") is not None else
                    os.path.join(companion_path, "text_encoder", "config.json")
                )
                with init_empty_weights():
                    text_encoder = Qwen3VLForConditionalGeneration(
                        Qwen3VLConfig.from_json_file(te_config_path)
                    )
                components["text_encoder"] = _finish_load(
                    text_encoder, te_state, "Qwen-Image 2.1 text encoder"
                ).to(dtype=torch_dtype)
                components["text_encoder_variant"] = "bf16"
            variant = "bf16"
        else:
            transformer, variant = load_transformer_artifact(model_path, config, torch_dtype)
        components["transformer"] = transformer
        components["transformer_variant"] = variant
        components["checkpoint_path"] = model_path
        declaration = getattr(transformer, "config", {}).get("vae_alignment")
        if declaration is not None:
            from .vae_alignment import load_saved_vae
            components["vae"], components["vae_identity"] = load_saved_vae(model_path, declaration, torch_dtype)
    else:
        raise ValueError(
            "Qwen-Image 2.1 uses a split component artifact. Select its directory containing manifest.json."
        )

    for name in ("transformer", "text_encoder", "vae"):
        module = components.get(name)
        if module is not None:
            module.to("cpu")
            module.eval()
    components.update(
        type="qwen_image_21",
        vae_scale_factor=16,
        latent_channels=64,
        pixel_alignment=32,
    )
    declaration = getattr(components.get("transformer"), "config", {}).get("vae_alignment")
    if declaration is not None:
        components.update(vae_scale_factor=declaration["new_scale"],
                          latent_channels=declaration["new_channels"], pixel_alignment=declaration["new_scale"]*2)
    return components


def build_pipeline(components: dict) -> QwenImage21Pipeline:
    text_encoder = components.get("text_encoder")
    if text_encoder is None:
        raise RuntimeError("Qwen-Image 2.1 text encoder was not loaded")
    return QwenImage21Pipeline(
        scheduler=components["scheduler"],
        vae=components["vae"],
        text_encoder=text_encoder,
        processor=components["processor"],
        transformer=components["transformer"],
    )
