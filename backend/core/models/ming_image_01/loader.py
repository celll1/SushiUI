"""Native package component loading; no runtime remote-code execution."""

from __future__ import annotations

from dataclasses import dataclass

from .artifact import MingArtifact, inspect_artifact


@dataclass
class MingComponents:
    artifact: MingArtifact
    transformer: object
    conditioner: object
    vae: object
    scheduler: object
    processor: object


def _strict_loading_info(info: dict, component: str) -> None:
    failures = {key: value for key, value in info.items()
                if key in {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"} and value}
    if failures:
        raise ValueError(f"Ming {component} state mismatch: {failures}")


def _restore_native_rope(native) -> None:
    import torch
    from .vendor.modeling_bailing_moe_v2 import BailingMoeV2RotaryEmbedding, BailingMoeV2RotaryEmbedding3D
    from .vendor.qwen2_5_vit import Qwen2_5_VisionRotaryEmbedding
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    # Low-memory HF loading can leave nonpersistent buffers on meta or uninitialized.
    device = next(native.parameters()).device
    for module in native.modules():
        if isinstance(module, (BailingMoeV2RotaryEmbedding, BailingMoeV2RotaryEmbedding3D)):
            inv_freq, scaling = module.rope_init_fn(module.config, device)
            module.register_buffer("inv_freq", inv_freq.float(), persistent=False)
            module.original_inv_freq = module.inv_freq
            module.attention_scaling = scaling
        elif isinstance(module, Qwen2_5_VisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (
                torch.arange(0, module.dim, 2, device=device, dtype=torch.float32) / module.dim))
            module.register_buffer("inv_freq", inv_freq, persistent=False)
        elif isinstance(module, Qwen2RotaryEmbedding):
            fresh = Qwen2RotaryEmbedding(config=module.config, device=device)
            module.register_buffer("inv_freq", fresh.inv_freq.float(), persistent=False)
            module.register_buffer("original_inv_freq", fresh.original_inv_freq.float(), persistent=False)
            module.attention_scaling = fresh.attention_scaling


def load_components(path, *, dtype, conditioner_attention: str) -> MingComponents:
    # Complete preflight precedes construction or residency changes.
    artifact = inspect_artifact(path)
    from core.rewiring.artifacts import read_directory_topology
    from .residual_unet import restore_topology, validate_topology_identity
    topology = read_directory_topology(artifact.components["transformer"])
    if topology is not None:
        topology = validate_topology_identity(topology, artifact.configs["transformer"])
    if conditioner_attention not in {"eager", "flash_attention_2"}:
        raise ValueError("Ming conditioner supports eager or flash_attention_2, not SDPA")
    from diffusers import FlowMatchEulerDiscreteScheduler
    from safetensors.torch import load_file
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from .conditioning import MingConditioner
    from .vendor.configuration_bailingmm2 import BailingMM2Config
    from .vendor.modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
    from .vendor.tokenization_bailing import BailingTokenizer
    from .vendor.image_processing_bailingmm2 import BailingMM2ImageProcessor
    from .vendor.processing_bailingmm2 import BailingMM2Processor
    from .vendor.diffusion.transformer import DiffusionTransformer
    from .vendor.diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

    config = BailingMM2Config.from_dict(artifact.configs["mllm"])
    config._attn_implementation = conditioner_attention
    native, info = BailingMM2NativeForConditionalGeneration.from_pretrained(
        str(artifact.components["mllm"]), config=config, torch_dtype=dtype,
        local_files_only=True, low_cpu_mem_usage=True, output_loading_info=True)
    _strict_loading_info(info, "mllm")
    connector_config = Qwen2Config.from_dict(artifact.configs["connector"])
    connector, info = Qwen2ForCausalLM.from_pretrained(
        str(artifact.components["connector"]), config=connector_config,
        attn_implementation="eager", torch_dtype=dtype, local_files_only=True,
        low_cpu_mem_usage=True, output_loading_info=True)
    _strict_loading_info(info, "connector")
    conditioner = MingConditioner(native, connector, artifact.configs["mlp"])
    conditioner.load_projection_state(load_file(str(artifact.components["mlp"] / "model.safetensors"), device="cpu"))
    conditioner.to(dtype=dtype).eval()
    _restore_native_rope(conditioner)
    conditioner.set_trainable(set())
    tokenizer = BailingTokenizer.from_pretrained(str(artifact.components["mllm"]), local_files_only=True)
    image_processor = BailingMM2ImageProcessor.from_pretrained(str(artifact.components["mllm"]), local_files_only=True)
    processor = BailingMM2Processor(image_processor=image_processor, tokenizer=tokenizer)
    transformer, info = DiffusionTransformer.from_pretrained(
        str(artifact.components["transformer"]), torch_dtype=dtype,
        local_files_only=True, low_cpu_mem_usage=True, output_loading_info=True)
    _strict_loading_info(info, "transformer")
    restore_topology(transformer, topology)
    vae, info = AutoencoderKLQwenImage.from_pretrained(
        str(artifact.components["vae"]), torch_dtype=dtype, local_files_only=True,
        low_cpu_mem_usage=True, output_loading_info=True)
    _strict_loading_info(info, "vae")
    vae.input_channels = artifact.configs["vae"]["input_channels"]
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(artifact.components["scheduler"]), local_files_only=True)
    transformer.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    return MingComponents(artifact, transformer, conditioner, vae, scheduler, processor)
