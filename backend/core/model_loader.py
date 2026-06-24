from typing import Dict, Any, Optional, Literal, Union
import os
import sys
import json
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL
from safetensors.torch import load_file
from pathlib import Path

ModelSource = Literal["safetensors", "diffusers", "huggingface"]
# DEUS support removed - architecture no longer maintained
# ModelType = Literal["sd15", "sdxl", "zimage", "deus", "flux2"]
ModelType = Literal["sd15", "sdxl", "zimage", "flux2", "anima", "lens", "ideogram4", "minit2i"]

class ModelLoader:
    """Handles loading models from various sources"""

    @staticmethod
    def _configure_v_prediction_scheduler(pipeline):
        """Configure scheduler for v-prediction models

        V-prediction models require:
        1. prediction_type = "v_prediction"
        2. timestep_spacing = "trailing" (recommended for v-prediction)

        Note: rescale_betas_zero_snr is intentionally set to False by default.
        While some v-prediction models were trained with zero terminal SNR,
        many SDXL v-prediction models (especially newer ones) work better
        WITHOUT rescale_betas_zero_snr=True, as it can cause extreme sigma
        values (e.g., 4096) leading to black or blurry outputs.

        References:
        - https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/16567
        - https://huggingface.co/docs/diffusers/using-diffusers/scheduler
        - https://github.com/comfyanonymous/ComfyUI/discussions/2794
        """
        try:
            # Register to scheduler config (this modifies the scheduler's configuration)
            # Note: rescale_betas_zero_snr is omitted (defaults to False in most schedulers)
            pipeline.scheduler.register_to_config(
                prediction_type="v_prediction",
                timestep_spacing="trailing"
            )

            print(f"[ModelLoader] V-prediction scheduler configured:")
            print(f"  - prediction_type: v_prediction")
            print(f"  - rescale_betas_zero_snr: False (default, avoids extreme sigma values)")
            print(f"  - timestep_spacing: trailing")

        except Exception as e:
            print(f"[ModelLoader] Warning: Could not configure v-prediction scheduler: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _dir_contains_minit2i(model_path: str, max_depth: int = 2) -> bool:
        """True if a MiniT2I variant dir exists at model_path or within max_depth.

        A variant dir has transformer/config.json with the MiniT2I marker. This lets
        detection accept a repo root (.../MiniT2I) or container (.../minit2i) that holds
        minit2i-b-16 / minit2i-l-16 so the loader can resolve it. JSON-only (no torch).
        """
        def _is_variant(d: str) -> bool:
            cfg = os.path.join(d, "transformer", "config.json")
            if not os.path.isfile(cfg):
                return False
            try:
                with open(cfg, "r", encoding="utf-8") as f:
                    tcfg = json.load(f)
            except Exception:
                return False
            return (tcfg.get("_class_name") == "MiniT2IMMJiTModel"
                    or ("depth_double" in tcfg and "pca_channels" in tcfg))

        def _walk(d: str, depth: int) -> bool:
            if _is_variant(d):
                return True
            if depth >= max_depth:
                return False
            try:
                for name in os.listdir(d):
                    sub = os.path.join(d, name)
                    if os.path.isdir(sub) and _walk(sub, depth + 1):
                        return True
            except OSError:
                return False
            return False

        return _walk(model_path, 0)

    @staticmethod
    def detect_prediction_config(model_path: str, model_type: str) -> Dict[str, str]:
        """Detect prediction configuration from model metadata and state dict

        Supports unified training framework with noise_process and prediction_target.

        Args:
            model_path: Path to model file or directory
            model_type: Model type ("sd15", "sdxl", "zimage")

        Returns:
            Dict with keys:
                - noise_process: "ddpm" or "flow"
                - prediction_target: "epsilon", "velocity", or "sample"
                - source: "modelspec", "state_dict", "legacy", "inferred"
        """
        try:
            metadata = {}
            state_dict_keys = []

            # Load metadata and state dict keys
            if model_path.endswith('.safetensors'):
                from safetensors import safe_open
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}
                    state_dict_keys = list(f.keys())

            elif os.path.isdir(model_path):
                # Check diffusers format config
                config_path = os.path.join(model_path, "scheduler", "scheduler_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        # Map scheduler config to metadata format
                        if config.get('prediction_type'):
                            metadata['prediction_type'] = config['prediction_type']

            # Priority 1: ModelSpec metadata (modelspec.noise_process + modelspec.prediction_type)
            if "modelspec.noise_process" in metadata:
                print(f"[ModelLoader] Detected prediction config from ModelSpec metadata")
                return {
                    "noise_process": metadata["modelspec.noise_process"],
                    "prediction_target": metadata.get("modelspec.prediction_type", "epsilon"),
                    "source": "modelspec"
                }

            # Priority 2: State dict marker (v_pred tensor)
            if "v_pred" in state_dict_keys:
                print(f"[ModelLoader] Detected v-prediction from state_dict marker")
                return {
                    "noise_process": "ddpm",
                    "prediction_target": "velocity",
                    "source": "state_dict"
                }

            # Priority 3: Legacy metadata (prediction_type or v_pred in metadata)
            if metadata.get("v_pred") or metadata.get("prediction_type") == "v_prediction":
                print(f"[ModelLoader] Detected v-prediction from legacy metadata")
                return {
                    "noise_process": "ddpm",
                    "prediction_target": "velocity",
                    "source": "legacy"
                }

            if metadata.get("prediction_type") in ["epsilon", "sample"]:
                print(f"[ModelLoader] Detected prediction_type from legacy metadata: {metadata['prediction_type']}")
                return {
                    "noise_process": "ddpm",
                    "prediction_target": metadata["prediction_type"],
                    "source": "legacy"
                }

            # Priority 4: Infer from model architecture
            if model_type == "zimage":
                print(f"[ModelLoader] Inferred prediction config from Z-Image architecture")
                return {
                    "noise_process": "flow",
                    "prediction_target": "velocity",
                    "source": "inferred"
                }
            elif model_type == "flux2":
                # FLUX.2 uses Flow Matching with velocity prediction
                print(f"[ModelLoader] Inferred prediction config from FLUX.2 architecture")
                return {
                    "noise_process": "flow",
                    "prediction_target": "velocity",
                    "source": "inferred"
                }
            elif model_type == "minit2i":
                # MiniT2I uses flow matching with x0 (sample) prediction.
                print(f"[ModelLoader] Inferred prediction config from MiniT2I architecture")
                return {
                    "noise_process": "flow",
                    "prediction_target": "sample",
                    "source": "inferred"
                }
            # DEUS support removed - architecture no longer maintained
            # elif model_type == "deus":
            #     # DEUS uses DDPM with epsilon prediction (same as SDXL base)
            #     print(f"[ModelLoader] Inferred prediction config from DEUS architecture")
            #     return {
            #         "noise_process": "ddpm",
            #         "prediction_target": "epsilon",
            #         "source": "inferred"
            #     }
            else:  # sd15, sdxl
                print(f"[ModelLoader] Inferred prediction config from {model_type.upper()} architecture")
                return {
                    "noise_process": "ddpm",
                    "prediction_target": "epsilon",
                    "source": "inferred"
                }

        except Exception as e:
            print(f"[ModelLoader] Error detecting prediction config: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to safe defaults
            return {
                "noise_process": "ddpm" if model_type != "zimage" else "flow",
                "prediction_target": "epsilon" if model_type != "zimage" else "velocity",
                "source": "error_fallback"
            }

    @staticmethod
    def detect_v_prediction(model_path: str) -> bool:
        """Legacy method for backward compatibility

        Deprecated: Use detect_prediction_config() instead

        Returns:
            True if v-prediction model, False otherwise
        """
        # Detect model type for prediction config
        model_type = ModelLoader.detect_model_type(model_path)
        config = ModelLoader.detect_prediction_config(model_path, model_type)
        return config["prediction_target"] == "velocity"

    @staticmethod
    def has_embedded_vae(model_path: str) -> bool:
        """Detect if a safetensors model has an embedded VAE

        Returns:
            True if VAE is embedded in the model, False otherwise
        """
        try:
            if model_path.endswith('.safetensors'):
                from safetensors import safe_open
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())

                    # Check for VAE decoder keys (common patterns)
                    vae_patterns = [
                        'first_stage_model.decoder',  # Standard SD/SDXL
                        'vae.decoder',                # Alternative format
                        'first_stage_model.encoder',  # VAE encoder
                        'vae.encoder',
                    ]

                    vae_keys = [k for k in keys if any(pattern in k for pattern in vae_patterns)]
                    has_vae = len(vae_keys) > 0

                    if has_vae:
                        print(f"[ModelLoader] Detected embedded VAE in model (found {len(vae_keys)} VAE keys)")
                        # Show first few keys for debugging
                        sample_keys = vae_keys[:3]
                        print(f"[ModelLoader] Sample VAE keys: {sample_keys}")
                    else:
                        print(f"[ModelLoader] No embedded VAE detected in model")
                        # Show sample of all keys for debugging
                        print(f"[ModelLoader] Total keys in model: {len(keys)}")
                        print(f"[ModelLoader] Sample keys: {keys[:5] if len(keys) > 0 else 'none'}")

                    return has_vae

            return True  # Assume diffusers format has VAE

        except Exception as e:
            print(f"[ModelLoader] ERROR: Could not detect VAE status: {e}")
            import traceback
            traceback.print_exc()
            # Return False to trigger external VAE loading as a safety measure
            # Better to load external VAE unnecessarily than to have None VAE
            return False

    @staticmethod
    def is_valid_diffusers_directory(path: str) -> bool:
        """Check if a directory is a valid diffusers-format model directory.

        A valid diffusers directory must have either:
        - model_index.json (standard diffusers pipeline), OR
        - transformer/config.json with axes_dims + rope_theta (Z-Image format)

        Non-model directories (tensorboard logs, training output, etc.) are excluded.
        """
        if not os.path.isdir(path):
            return False
        # Z-Image: transformer/config.json with Z-Image-specific keys
        transformer_config = os.path.join(path, "transformer", "config.json")
        if os.path.exists(transformer_config):
            try:
                with open(transformer_config, 'r') as f:
                    config = json.load(f)
                if "axes_dims" in config and "rope_theta" in config:
                    return True
            except Exception:
                pass
        # Standard diffusers pipeline requires model_index.json
        return os.path.exists(os.path.join(path, "model_index.json"))

    @staticmethod
    def detect_model_type(model_path: str) -> ModelType:
        """Detect if model is SD1.5, SDXL, Z-Image, DEUS, or FLUX.2 based on config or structure

        Supports:
        - Z-Image diffusers format (directory with transformer/, vae/, etc.)
        - Z-Image Comfy format (single safetensors with transformer weights only)
        - FLUX.2 Klein (single safetensors with Flux2Transformer2DModel weights)
        - SD1.5/SDXL diffusers and safetensors
        Note: DEUS support has been removed (architecture no longer maintained)
        """
        # From-scratch MiniT2I sentinel ("scratch:minit2i:<variant>:<vae_type>"):
        # not a filesystem path — handled by the in-memory build path in the loader.
        if isinstance(model_path, str) and model_path.startswith("scratch:minit2i:"):
            return "minit2i"

        # Lens detection (microsoft/Lens diffusers directory or HF repo)
        if os.path.isdir(model_path):
            model_index_path = os.path.join(model_path, "model_index.json")
            if os.path.exists(model_index_path):
                try:
                    with open(model_index_path, "r") as f:
                        idx = json.load(f)
                    if idx.get("_class_name") == "LensPipeline":
                        return "lens"
                except Exception:
                    pass
            # Fallback: transformer/config.json with LensTransformer2DModel architecture
            transformer_config_path = os.path.join(model_path, "transformer", "config.json")
            if os.path.exists(transformer_config_path):
                try:
                    with open(transformer_config_path, "r") as f:
                        tcfg = json.load(f)
                    if "LensTransformer2DModel" in tcfg.get("architectures", []):
                        return "lens"
                except Exception:
                    pass

            # Ideogram 4 detection (diffusers directory: Ideogram4Pipeline / Ideogram4Transformer2DModel)
            if os.path.exists(model_index_path):
                try:
                    with open(model_index_path, "r") as f:
                        idx = json.load(f)
                    if idx.get("_class_name") == "Ideogram4Pipeline":
                        return "ideogram4"
                except Exception:
                    pass
            if os.path.exists(transformer_config_path):
                try:
                    with open(transformer_config_path, "r") as f:
                        tcfg = json.load(f)
                    # Ideogram4 single-stream DiT: unique mrope_section + llm_features_dim config keys.
                    if tcfg.get("_class_name") == "Ideogram4Transformer2DModel" or (
                        "mrope_section" in tcfg and "llm_features_dim" in tcfg
                    ):
                        return "ideogram4"
                    # MiniT2I MM-JiT (pixel-space): unique config class / keys.
                    if tcfg.get("_class_name") == "MiniT2IMMJiTModel" or (
                        "depth_double" in tcfg and "pca_channels" in tcfg
                    ):
                        return "minit2i"
                except Exception:
                    pass

            # MiniT2I repo root (.../MiniT2I with MiniT2IPipeline model_index) or a
            # container (.../minit2i with MiniT2I/ inside): detect by scanning for a
            # variant dir within 2 levels so the loader can resolve it to a variant.
            if os.path.isdir(model_path) and ModelLoader._dir_contains_minit2i(model_path):
                return "minit2i"

        # Anima detection (split-files layout or single DiT safetensors)
        try:
            from core.models.anima.anima_loader import (
                detect_anima_split_layout, is_anima_safetensors,
            )
            if os.path.isdir(model_path):
                if detect_anima_split_layout(model_path):
                    return "anima"
            elif model_path.endswith(".safetensors"):
                # If the file is inside a split_files/diffusion_models/ tree, treat as Anima.
                if detect_anima_split_layout(model_path):
                    return "anima"
                # Otherwise inspect keys.
                if is_anima_safetensors(model_path):
                    return "anima"
        except Exception as e:
            print(f"[ModelLoader] Anima detection skipped: {e}")

        # Z-Image detection (diffusers format)
        if os.path.isdir(model_path):
            # Z-Image has transformer/ directory with unique config
            transformer_config = os.path.join(model_path, "transformer", "config.json")
            if os.path.exists(transformer_config):
                try:
                    with open(transformer_config, 'r') as f:
                        config = json.load(f)
                        # Z-Image has unique structure with axes_dims, rope_theta
                        if "axes_dims" in config and "rope_theta" in config:
                            return "zimage"
                except Exception as e:
                    pass

            # Check for SDXL indicators
            config_path = os.path.join(model_path, "model_index.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # SDXL uses different UNet config
                    if "_class_name" in config and "XL" in config["_class_name"]:
                        return "sdxl"

        # Check safetensors files
        if model_path.endswith('.safetensors'):
            try:
                from safetensors import safe_open
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    metadata = f.metadata() or {}

                    # MiniT2I single-file (bundled FLAN-T5 + MM-JiT transformer).
                    if (metadata.get("model_type", "").lower() == "minit2i"
                            or any(k.startswith("transformer.model.net.") for k in keys)
                            or any(k.startswith("model.net.double_blocks.") for k in keys)):
                        return "minit2i"

                    # Priority 1: Check metadata for explicit model_type
                    if "model_type" in metadata:
                        model_type = metadata["model_type"].lower() if isinstance(metadata["model_type"], str) else str(metadata["model_type"]).lower()

                        # SigLIP2 Vision Encoder detection (training checkpoint format)
                        if model_type == "siglip2_vision_encoder":
                            return "vision_encoder"

                        if model_type in ["flux2", "flux.2", "flux2-klein", "flux.2-klein"]:
                            return "flux2"
                        elif model_type in ["sdxl", "sd-xl", "stable-diffusion-xl", "stable_diffusion_xl"]:
                            return "sdxl"
                        elif model_type in ["sd15", "sd-1.5", "sd_1.5", "stable-diffusion", "stable_diffusion", "sd"]:
                            return "sd15"
                        elif model_type == "zimage":
                            return "zimage"

                    # Priority 2: FLUX.2 detection by state_dict keys
                    # FLUX.2 has two key formats:
                    # - Diffusers format: time_guidance_embed, double_stream_modulation_*, single_stream_modulation
                    # - BFL/Comfy format: double_blocks.*.img_attn, single_blocks.*, guidance_in

                    # Diffusers format detection
                    has_time_guidance_embed = any(k.startswith('time_guidance_embed.') for k in keys)
                    has_double_stream_modulation = any(k.startswith('double_stream_modulation_') for k in keys)
                    has_single_stream_modulation = any(k.startswith('single_stream_modulation.') for k in keys)
                    has_single_transformer_blocks = any(k.startswith('single_transformer_blocks.') for k in keys)

                    if has_time_guidance_embed and has_double_stream_modulation and has_single_stream_modulation:
                        return "flux2"

                    # BFL/Comfy format detection (double_blocks.*.img_attn, single_blocks.*)
                    has_double_blocks = any(k.startswith('double_blocks.') for k in keys)
                    has_single_blocks = any(k.startswith('single_blocks.') for k in keys)
                    has_img_attn = any('.img_attn.' in k for k in keys)
                    has_guidance_in = any(k.startswith('guidance_in.') for k in keys)

                    if has_double_blocks and has_single_blocks and has_img_attn:
                        return "flux2"

                    # DEUS support removed - architecture no longer maintained
                    # Priority 3: DEUS detection by state_dict keys
                    # DEUS uses SigLIP-2 text encoder with key prefix "conditioner.embedders.0.model."
                    # AND it has U-Net keys (unlike Z-Image which uses transformer)
                    has_unet_keys = any(k.startswith('model.diffusion_model.') for k in keys)

                    # DEUS-specific detection removed:
                    # has_siglip2_keys = any(
                    #     k.startswith('conditioner.embedders.0.model.text_model.embeddings') for k in keys
                    # )
                    # has_siglip2_layers = any(
                    #     k.startswith('conditioner.embedders.0.model.text_model.encoder.layers') for k in keys
                    # )
                    # has_dual_clip = (
                    #     any(k.startswith('conditioner.embedders.0.transformer') for k in keys) and
                    #     any(k.startswith('conditioner.embedders.1.') for k in keys)
                    # )
                    # if has_unet_keys and has_siglip2_keys and has_siglip2_layers and not has_dual_clip:
                    #     print(f"[ModelLoader] Detected DEUS model (SigLIP-2 text encoder): {model_path}")
                    #     return "deus"

                    # Priority 3: SD/SDXL detection
                    if has_unet_keys:
                        # This is SD or SDXL, not Z-Image
                        # SDXL detection by file size (>6GB) or specific keys
                        file_size = os.path.getsize(model_path) / (1024**3)  # GB
                        if file_size > 6:
                            return "sdxl"
                        else:
                            return "sd15"

                    # Z-Image Comfy format detection
                    # Z-Image transformer has unique keys WITHOUT U-Net structure
                    # Check for required indicators (cap_embedder, t_embedder, context_refiner)
                    # Note: x_embedder may be "x_embedder" or "all_x_embedder" depending on architecture
                    required_indicators = ['cap_embedder', 't_embedder', 'context_refiner']
                    has_required = all(any(k.startswith(indicator) for k in keys) for indicator in required_indicators)
                    has_x_embedder = any(k.startswith('x_embedder') or k.startswith('all_x_embedder') for k in keys)

                    if has_required and has_x_embedder:
                        return "zimage"

                    # SigLIP2 Vision Encoder detection by key structure
                    # Our saved format: embeddings.patch_embedding.weight (no prefix)
                    # HuggingFace format: vision_model.embeddings.patch_embedding.weight
                    has_ve_direct = any(k == 'embeddings.patch_embedding.weight' or k.startswith('embeddings.patch_embedding.') for k in keys)
                    has_ve_prefixed = any(k.startswith('vision_model.embeddings.') for k in keys)
                    has_header_token = 'header_token' in keys
                    has_encoder_layers = any(('encoder.layers.' in k) for k in keys)
                    # VE files have no U-Net, flux, or zimage keys
                    if (has_ve_direct or has_ve_prefixed or has_header_token) and has_encoder_layers and not has_unet_keys:
                        return "vision_encoder"

                    # Fallback: SDXL detection by file size
                    file_size = os.path.getsize(model_path) / (1024**3)  # GB
                    if file_size > 6:
                        return "sdxl"
            except Exception as e:
                print(f"[ModelLoader] Warning: Could not read safetensors: {e}")
                # Fallback to file size check
                file_size = os.path.getsize(model_path) / (1024**3)  # GB
                if file_size > 6:
                    return "sdxl"

        return "sd15"

    @staticmethod
    def _convert_comfy_to_official_state_dict(
        comfy_state_dict: dict,
        n_heads: int,
        n_kv_heads: int,
        dim: int
    ) -> dict:
        """Convert ComfyUI's state dict to official Z-Image format

        ComfyUI format:
            - attention.qkv.weight: [n_heads*head_dim + 2*n_kv_heads*head_dim, dim] (fused QKV)
            - attention.out.weight: [dim, n_heads*head_dim]
            - attention.q_norm.weight / k_norm.weight
            - x_embedder.weight/bias: Single embedder
            - final_layer.linear.weight/bias: Single final layer

        Official format:
            - attention.to_q/to_k/to_v.weight: Split Q/K/V
            - attention.to_out.0.weight: Output projection
            - attention.norm_q/norm_k.weight: Norm layers
            - all_x_embedder.{patch_size}-{aspect}.weight/bias: Multi-resolution embedders
            - all_final_layer.{patch_size}-{aspect}.linear.weight/bias: Multi-resolution final layers

        Args:
            comfy_state_dict: State dict from Comfy-format safetensors
            n_heads: Number of attention heads
            n_kv_heads: Number of key/value heads
            dim: Model dimension

        Returns:
            Converted state dict in official format
        """
        head_dim = dim // n_heads
        official_state_dict = {}

        # Default resolution key (patch_size=2, aspect_ratio=1:1)
        default_resolution_key = "2-1"

        print(f"[ModelLoader] Converting ComfyUI format to official Z-Image format")
        print(f"  - Attention layers: n_heads={n_heads}, n_kv_heads={n_kv_heads}, dim={dim}, head_dim={head_dim}")
        print(f"  - Using default resolution key: {default_resolution_key}")

        for key, value in comfy_state_dict.items():
            # Split fused QKV weights
            if ".qkv.weight" in key:
                q_dim = n_heads * head_dim
                kv_dim = n_kv_heads * head_dim

                q_weight = value[:q_dim, :]
                k_weight = value[q_dim:q_dim + kv_dim, :]
                v_weight = value[q_dim + kv_dim:q_dim + 2*kv_dim, :]

                base_key = key.replace(".qkv.weight", "")
                official_state_dict[f"{base_key}.to_q.weight"] = q_weight
                official_state_dict[f"{base_key}.to_k.weight"] = k_weight
                official_state_dict[f"{base_key}.to_v.weight"] = v_weight

            # Rename output projection
            elif ".out.weight" in key:
                new_key = key.replace(".out.weight", ".to_out.0.weight")
                official_state_dict[new_key] = value

            # Rename norm layers
            elif ".q_norm.weight" in key:
                new_key = key.replace(".q_norm.weight", ".norm_q.weight")
                official_state_dict[new_key] = value
            elif ".k_norm.weight" in key:
                new_key = key.replace(".k_norm.weight", ".norm_k.weight")
                official_state_dict[new_key] = value

            # Map x_embedder to all_x_embedder with resolution key
            elif key.startswith("x_embedder."):
                param_name = key.replace("x_embedder.", "")
                new_key = f"all_x_embedder.{default_resolution_key}.{param_name}"
                official_state_dict[new_key] = value
                print(f"  Mapped {key} -> {new_key}")

            # Map final_layer to all_final_layer with resolution key
            elif key.startswith("final_layer."):
                param_name = key.replace("final_layer.", "")
                new_key = f"all_final_layer.{default_resolution_key}.{param_name}"
                official_state_dict[new_key] = value
                print(f"  Mapped {key} -> {new_key}")

            # Copy all other keys as-is
            else:
                official_state_dict[key] = value

        print(f"[ModelLoader] Conversion complete: {len(comfy_state_dict)} keys -> {len(official_state_dict)} keys")
        return official_state_dict

    @staticmethod
    def load_zimage_from_comfy_safetensors(
        file_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        base_model_repo: str = "Tongyi-MAI/Z-Image-Turbo"
    ) -> Dict[str, Any]:
        """Load Z-Image from ComfyUI Lumina format with weight conversion

        This loads ComfyUI-format safetensors and converts the weights to match
        the official Z-Image transformer structure by:
        1. Splitting fused QKV weights into separate Q/K/V layers
        2. Mapping single-resolution embedders to multi-resolution format
        3. Converting key names to match official structure

        Args:
            file_path: Path to Comfy-format Z-Image safetensors
            device: Device to load models on
            torch_dtype: Data type for model weights (bfloat16 recommended)
            base_model_repo: HuggingFace repo ID for base components (VAE, text encoder, etc.)

        Returns:
            Dict containing transformer, vae, text_encoder, tokenizer, scheduler
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Z-Image Comfy safetensors not found: {file_path}")

        print(f"[ModelLoader] Loading Z-Image from Comfy safetensors: {file_path}")
        print(f"[ModelLoader] Base components will be downloaded from: {base_model_repo}")

        # Use SushiUI's internal Z-Image implementation (standalone)
        try:
            from transformers import AutoModel, AutoTokenizer
            from safetensors.torch import load_file
            import importlib.util

            # Load SushiUI's internal Z-Image modules (standalone implementation)
            sushiui_models_path = Path(__file__).parent / "models"

            # Load transformer module (Block Swap integrated)
            transformer_spec = importlib.util.spec_from_file_location(
                "zimage_transformer",
                sushiui_models_path / "zimage_transformer.py"
            )
            transformer_module = importlib.util.module_from_spec(transformer_spec)
            # Register in sys.modules BEFORE exec_module (required for Flash Attention setup)
            import sys as _sys
            _sys.modules['zimage_transformer'] = transformer_module
            transformer_spec.loader.exec_module(transformer_module)
            ZImageTransformer2DModel = transformer_module.ZImageTransformer2DModel
            print(f"[ModelLoader] Loaded SushiUI Z-Image Transformer (standalone, Block Swap integrated)")

            # Load autoencoder module
            autoencoder_spec = importlib.util.spec_from_file_location(
                "zimage_autoencoder",
                sushiui_models_path / "zimage_autoencoder.py"
            )
            autoencoder_module = importlib.util.module_from_spec(autoencoder_spec)
            autoencoder_spec.loader.exec_module(autoencoder_module)
            AutoencoderKL = autoencoder_module.AutoencoderKL
            print(f"[ModelLoader] Loaded SushiUI Z-Image AutoencoderKL (standalone)")

            # Load scheduler module
            scheduler_spec = importlib.util.spec_from_file_location(
                "zimage_scheduler",
                sushiui_models_path / "zimage_scheduler.py"
            )
            scheduler_module = importlib.util.module_from_spec(scheduler_spec)
            scheduler_spec.loader.exec_module(scheduler_module)
            FlowMatchEulerDiscreteScheduler = scheduler_module.FlowMatchEulerDiscreteScheduler
            print(f"[ModelLoader] Loaded SushiUI Z-Image Scheduler (standalone)")

            # Step 1: Download base components from HuggingFace
            print(f"[ModelLoader] Downloading base components from {base_model_repo}...")
            from huggingface_hub import snapshot_download
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            base_model_path = snapshot_download(
                base_model_repo,
                cache_dir=cache_dir,
                allow_patterns=["vae/*", "text_encoder/*", "tokenizer/*", "scheduler/*", "transformer/config.json"]
            )
            print(f"[ModelLoader] Base components downloaded to: {base_model_path}")

            # Step 2: Load transformer config from base model
            transformer_config_path = os.path.join(base_model_path, "transformer", "config.json")
            with open(transformer_config_path, 'r') as f:
                transformer_config = json.load(f)

            # Step 3: Detect actual layer count from safetensors file
            print(f"[ModelLoader] Loading Comfy transformer weights from: {file_path}")
            comfy_state_dict = load_file(file_path, device="cpu")

            # Auto-detect layer count from state_dict (supports pruned models)
            layer_indices = set()
            for key in comfy_state_dict.keys():
                if "layers." in key:
                    parts = key.split(".")
                    if len(parts) > 1 and parts[0] == "layers":
                        try:
                            layer_idx = int(parts[1])
                            layer_indices.add(layer_idx)
                        except ValueError:
                            pass

            actual_n_layers = max(layer_indices) + 1 if layer_indices else transformer_config["n_layers"]

            if actual_n_layers != transformer_config["n_layers"]:
                print(f"[ModelLoader] WARNING: Detected {actual_n_layers} layers in model file, "
                      f"but config specifies {transformer_config['n_layers']} layers.")
                print(f"[ModelLoader] Using detected layer count: {actual_n_layers}")

            # Step 4: Detect in_channels from actual safetensors file
            # If the model has x_embedder with 4 input channels, use SDXL VAE
            # If it has 16 input channels, use FLUX VAE (standard Z-Image)
            actual_in_channels = transformer_config["in_channels"]  # Default from config (16)

            # Get patch size from config for x_embedder shape calculation
            # all_patch_size can be [patch_h, patch_w] = [4, 4] or single value [4]
            all_patch_size = transformer_config["all_patch_size"]
            if isinstance(all_patch_size, (list, tuple)):
                if len(all_patch_size) == 2:
                    patch_h, patch_w = all_patch_size
                elif len(all_patch_size) == 1:
                    patch_h = patch_w = all_patch_size[0]
                else:
                    patch_h = patch_w = 4  # Default fallback
            else:
                patch_h = patch_w = all_patch_size  # Single integer
            patch_product = patch_h * patch_w  # 16 for standard Z-Image (4x4)

            # Try to detect from state_dict (x_embedder weight shape)
            # ComfyUI format: x_embedder.weight shape is [dim, in_channels * patch_h * patch_w]
            # - Standard Z-Image (FLUX VAE): [3840, 16 * 4 * 4] = [3840, 256]
            # - SDXL VAE version: [3840, 4 * 4 * 4] = [3840, 64]
            for key in comfy_state_dict.keys():
                if "x_embedder" in key and "weight" in key:
                    weight_shape = comfy_state_dict[key].shape
                    if len(weight_shape) == 2:
                        # 2D weight: [dim, in_channels * patch_h * patch_w]
                        flattened_in = weight_shape[1]
                        detected_in_channels = flattened_in // patch_product
                        if detected_in_channels != actual_in_channels:
                            print(f"[ModelLoader] Detected in_channels={detected_in_channels} from x_embedder "
                                  f"(weight shape {weight_shape}, patch={patch_h}x{patch_w}, "
                                  f"config has {actual_in_channels})")
                            actual_in_channels = detected_in_channels
                    elif len(weight_shape) == 4:
                        # 4D weight: [dim, in_channels, patch_h, patch_w]
                        detected_in_channels = weight_shape[1]
                        if detected_in_channels != actual_in_channels:
                            print(f"[ModelLoader] Detected in_channels={detected_in_channels} from x_embedder "
                                  f"(weight shape {weight_shape}, config has {actual_in_channels})")
                            actual_in_channels = detected_in_channels
                    break

            # Determine VAE type based on in_channels
            use_sdxl_vae = (actual_in_channels == 4)
            if use_sdxl_vae:
                print(f"[ModelLoader] Z-Image model uses 4-channel latents (SDXL VAE)")
            else:
                print(f"[ModelLoader] Z-Image model uses {actual_in_channels}-channel latents (FLUX VAE)")

            # Step 5: Create transformer model with detected layer count and in_channels
            print("[ModelLoader] Creating Z-Image transformer...")
            with torch.device("meta"):
                transformer = ZImageTransformer2DModel(
                    all_patch_size=tuple(transformer_config["all_patch_size"]),
                    all_f_patch_size=tuple(transformer_config["all_f_patch_size"]),
                    in_channels=actual_in_channels,
                    dim=transformer_config["dim"],
                    n_layers=actual_n_layers,
                    n_refiner_layers=transformer_config["n_refiner_layers"],
                    n_heads=transformer_config["n_heads"],
                    n_kv_heads=transformer_config["n_kv_heads"],
                    norm_eps=transformer_config["norm_eps"],
                    qk_norm=transformer_config["qk_norm"],
                    cap_feat_dim=transformer_config["cap_feat_dim"],
                    rope_theta=transformer_config["rope_theta"],
                    t_scale=transformer_config["t_scale"],
                    axes_dims=transformer_config["axes_dims"],
                    axes_lens=transformer_config["axes_lens"],
                ).to(torch_dtype)

            # Convert Comfy format (fused QKV) to official format (separate Q/K/V)
            print("[ModelLoader] Converting Comfy format to official format...")
            state_dict = ModelLoader._convert_comfy_to_official_state_dict(
                comfy_state_dict,
                transformer_config["n_heads"],
                transformer_config["n_kv_heads"],
                transformer_config["dim"]
            )
            del comfy_state_dict

            transformer.load_state_dict(state_dict, strict=True, assign=True)
            del state_dict

            print(f"[ModelLoader] Moving transformer to {device}...")
            transformer = transformer.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            transformer.eval()

            # Step 6: Load VAE based on in_channels
            if use_sdxl_vae:
                # Load SDXL VAE (4-channel latents)
                print("[ModelLoader] Loading SDXL VAE (4-channel latents)...")
                sdxl_vae_repo = "madebyollin/sdxl-vae-fp16-fix"
                from diffusers import AutoencoderKL as DiffusersAutoencoderKL
                vae = DiffusersAutoencoderKL.from_pretrained(
                    sdxl_vae_repo,
                    torch_dtype=torch.float32  # VAE in fp32 for quality
                )
                vae.to(device=device)
                vae.eval()
                print(f"[ModelLoader] SDXL VAE loaded: latent_channels={vae.config.latent_channels}, "
                      f"scaling_factor={vae.config.scaling_factor}")
            else:
                # Load FLUX VAE from base model (16-channel latents)
                print("[ModelLoader] Loading FLUX VAE (16-channel latents)...")
                vae_path = os.path.join(base_model_path, "vae")
                vae_config_path = os.path.join(vae_path, "config.json")
                with open(vae_config_path, 'r') as f:
                    vae_config = json.load(f)

                vae = AutoencoderKL(
                    in_channels=vae_config["in_channels"],
                    out_channels=vae_config["out_channels"],
                    down_block_types=tuple(vae_config["down_block_types"]),
                    up_block_types=tuple(vae_config["up_block_types"]),
                    block_out_channels=tuple(vae_config["block_out_channels"]),
                    layers_per_block=vae_config["layers_per_block"],
                    latent_channels=vae_config["latent_channels"],
                    norm_num_groups=vae_config["norm_num_groups"],
                    scaling_factor=vae_config["scaling_factor"],
                    shift_factor=vae_config.get("shift_factor"),
                    use_quant_conv=vae_config.get("use_quant_conv", True),
                    use_post_quant_conv=vae_config.get("use_post_quant_conv", True),
                    mid_block_add_attention=vae_config.get("mid_block_add_attention", True),
                )

                vae_weights_path = os.path.join(vae_path, "diffusion_pytorch_model.safetensors")
                vae_state_dict = load_file(vae_weights_path, device="cpu")
                vae.load_state_dict(vae_state_dict, strict=False)
                del vae_state_dict
                vae.to(device=device, dtype=torch.float32)  # VAE uses fp32
                vae.eval()
                print(f"[ModelLoader] FLUX VAE loaded: latent_channels={vae.config.latent_channels}, "
                      f"scaling_factor={vae.config.scaling_factor}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[ModelLoader] Loading text encoder...")
            text_encoder_path = os.path.join(base_model_path, "text_encoder")
            text_encoder = AutoModel.from_pretrained(
                text_encoder_path,
                dtype=torch_dtype,
                trust_remote_code=True,
            )
            print(f"[ModelLoader] Moving text encoder to {device}...")
            text_encoder.to(device)
            text_encoder.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[ModelLoader] Loading tokenizer...")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            tokenizer_path = os.path.join(base_model_path, "tokenizer")
            if not os.path.exists(tokenizer_path):
                tokenizer_path = text_encoder_path
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
            )

            print("[ModelLoader] Loading scheduler...")
            scheduler_path = os.path.join(base_model_path, "scheduler")
            scheduler_config_path = os.path.join(scheduler_path, "scheduler_config.json")
            with open(scheduler_config_path, 'r') as f:
                scheduler_config = json.load(f)

            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=scheduler_config.get("num_train_timesteps", 1000),
                shift=scheduler_config.get("shift", 1.0),
                use_dynamic_shifting=scheduler_config.get("use_dynamic_shifting", False),
            )

            vae_type = "sdxl" if use_sdxl_vae else "flux"
            print("[ModelLoader] Z-Image Comfy format loaded successfully")
            print(f"  - Transformer: {actual_n_layers} layers, in_channels={actual_in_channels}")
            print(f"  - VAE: {vae_type.upper()} (latent_channels={vae.config.latent_channels})")
            print(f"  - Text Encoder, Tokenizer, Scheduler: Loaded from {base_model_repo}")

            return {
                "transformer": transformer,
                "vae": vae,
                "text_encoder": text_encoder,
                "tokenizer": tokenizer,
                "scheduler": scheduler,
                "vae_type": vae_type,  # "sdxl" or "flux" - for TAESD preview selection
            }

        except Exception as e:
            print(f"[ModelLoader] Error loading Z-Image Comfy format: {e}")
            import traceback
            traceback.print_exc()
            raise

    # DEUS support removed - architecture no longer maintained
    # @staticmethod
    # def load_deus_from_safetensors(
    #     file_path: str,
    #     device: str = "cuda",
    #     torch_dtype: torch.dtype = torch.float16
    # ) -> Dict[str, Any]:
    #     """Load DEUS model from safetensors file using diffusers DeusPipeline
    #
    #     DEUS architecture uses:
    #     - SigLIP-2 text encoder (1152d, variable sequence length)
    #     - U-Net with RoPE 2D positional encoding
    #     - SDXL VAE (same as SDXL)
    #     - 2-Pass CFG for inference
    #
    #     Args:
    #         file_path: Path to DEUS safetensors file
    #         device: Device to load models on
    #         torch_dtype: Data type for model weights
    #
    #     Returns:
    #         Dict containing unet, vae, text_encoder, tokenizer, scheduler, processor
    #     """
    #     if not os.path.exists(file_path):
    #         raise FileNotFoundError(f"DEUS model file not found: {file_path}")
    #
    #     print(f"[ModelLoader] Loading DEUS model from: {file_path}")
    #     print(f"[ModelLoader] Using diffusers DeusPipeline.from_single_file()")
    #
    #     try:
    #         # Import diffusers DEUS pipeline from custom location
    #         # Note: This uses the local diffusers installation at D:\celll1\diffusers
    #         sys.path.insert(0, "D:\\celll1\\diffusers\\src")
    #         from diffusers.pipelines.deus import DeusPipeline
    #         from diffusers.schedulers import EulerDiscreteScheduler
    #
    #         # Load pipeline using from_single_file
    #         pipeline = DeusPipeline.from_single_file(
    #             file_path,
    #             torch_dtype=torch_dtype,
    #         )
    #
    #         # Move components to device
    #         print(f"[ModelLoader] Moving DEUS components to {device}...")
    #         pipeline.text_encoder.to(device)
    #         pipeline.unet.to(device)
    #         # VAE uses fp32 for better quality (same as SDXL)
    #         pipeline.vae.to(device, dtype=torch.float32)
    #
    #         print(f"[ModelLoader] DEUS model loaded successfully")
    #         print(f"  - U-Net: {type(pipeline.unet).__name__}")
    #         print(f"  - Text Encoder: {type(pipeline.text_encoder).__name__}")
    #         print(f"  - VAE: {type(pipeline.vae).__name__}")
    #         print(f"  - Scheduler: {type(pipeline.scheduler).__name__}")
    #
    #         # Return components in dict format (consistent with Z-Image)
    #         return {
    #             "unet": pipeline.unet,
    #             "vae": pipeline.vae,
    #             "text_encoder": pipeline.text_encoder,
    #             "tokenizer": pipeline.processor.tokenizer if hasattr(pipeline.processor, 'tokenizer') else None,
    #             "processor": pipeline.processor,
    #             "scheduler": pipeline.scheduler,
    #             "pipeline": pipeline,  # Keep reference to pipeline for encode_prompt etc.
    #         }
    #
    #     except Exception as e:
    #         print(f"[ModelLoader] Error loading DEUS model: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         raise

    @staticmethod
    def load_flux2_from_safetensors(
        file_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        base_model_repo: str = None
    ) -> Dict[str, Any]:
        """Load FLUX.2 Klein from safetensors file (transformer weights only)

        The safetensors file should contain only transformer weights.
        VAE, text encoder, tokenizer, and scheduler are downloaded from HuggingFace.

        Args:
            file_path: Path to FLUX.2 transformer safetensors
            device: Device to load models on
            torch_dtype: Data type for model weights (bfloat16 recommended)
            base_model_repo: HuggingFace repo ID for base components (auto-detected if None)

        Returns:
            Dict with transformer, vae, text_encoder, tokenizer, scheduler components
        """
        try:
            from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
            from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
            from huggingface_hub import snapshot_download
            from safetensors import safe_open
            import os

            print(f"[ModelLoader] Loading FLUX.2 Klein from safetensors: {file_path}")

            # Auto-detect base_model_repo from safetensors metadata if not specified
            if base_model_repo is None:
                print(f"[ModelLoader] Auto-detecting HuggingFace repo from metadata...")
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}

                    # Priority 1: Check metadata for base_model_repo (from finetuned models)
                    if "base_model_repo" in metadata:
                        base_model_repo = metadata["base_model_repo"]
                        print(f"[ModelLoader] Found base_model_repo in metadata: {base_model_repo}")
                    else:
                        # Priority 2: Detect from layer count (for original models)
                        # Klein Base 4B: 48 single layers
                        # Klein 4B (distilled): 24 single layers
                        # Klein 9B (distilled): 36 single layers
                        num_single_layers = None
                        for key in f.keys():
                            if "single_blocks.47." in key or "single_transformer_blocks.47." in key:
                                num_single_layers = 48
                                break
                            elif "single_blocks.35." in key or "single_transformer_blocks.35." in key:
                                num_single_layers = 36
                                break
                            elif "single_blocks.23." in key or "single_transformer_blocks.23." in key:
                                num_single_layers = 24
                                break

                        # Determine repo based on layer count
                        if num_single_layers == 48:
                            base_model_repo = "black-forest-labs/FLUX.2-klein-base-4B"
                            print(f"[ModelLoader] Detected Klein Base 4B (48 single layers)")
                        elif num_single_layers == 36:
                            base_model_repo = "black-forest-labs/FLUX.2-klein-9B"
                            print(f"[ModelLoader] Detected Klein 9B (36 single layers)")
                        elif num_single_layers == 24:
                            base_model_repo = "black-forest-labs/FLUX.2-klein-4B"
                            print(f"[ModelLoader] Detected Klein 4B (24 single layers)")
                        else:
                            # Fallback: use Base 4B as default
                            base_model_repo = "black-forest-labs/FLUX.2-klein-base-4B"
                            print(f"[ModelLoader] Could not detect model variant, defaulting to Klein Base 4B")

                print(f"[ModelLoader] Using HuggingFace repo: {base_model_repo}")

            # Step 1: Download base components from HuggingFace
            print(f"[ModelLoader] Downloading base components from {base_model_repo}...")
            cache_dir = snapshot_download(
                base_model_repo,
                allow_patterns=["vae/*", "text_encoder/*", "tokenizer/*", "scheduler/*", "transformer/config.json", "model_index.json"],
            )
            print(f"[ModelLoader] Base components downloaded to: {cache_dir}")

            # Step 2: Load transformer config
            transformer_config_path = os.path.join(cache_dir, "transformer", "config.json")
            with open(transformer_config_path, 'r') as f:
                transformer_config = json.load(f)
            print(f"[ModelLoader] Transformer config loaded:")
            print(f"  - in_channels: {transformer_config.get('in_channels', 128)}")
            print(f"  - num_layers: {transformer_config.get('num_layers', 8)} (dual stream)")
            print(f"  - num_single_layers: {transformer_config.get('num_single_layers', 48)} (single stream)")
            print(f"  - num_attention_heads: {transformer_config.get('num_attention_heads', 48)}")
            print(f"  - attention_head_dim: {transformer_config.get('attention_head_dim', 128)}")

            # Load is_distilled flag
            # Priority 1: Check safetensors metadata (from finetuned models)
            # Priority 2: Check model_index.json from HuggingFace repo
            is_distilled = False
            with safe_open(file_path, framework="pt", device="cpu") as f:
                metadata = f.metadata() or {}
                if "is_distilled" in metadata:
                    is_distilled = metadata["is_distilled"].lower() == "true"
                    print(f"  - is_distilled (from metadata): {is_distilled}")
                else:
                    # Fallback to model_index.json
                    model_index_path = os.path.join(cache_dir, "model_index.json")
                    if os.path.exists(model_index_path):
                        with open(model_index_path, 'r') as f_index:
                            model_index = json.load(f_index)
                            is_distilled = model_index.get("is_distilled", False)
                    print(f"  - is_distilled (from model_index.json): {is_distilled}")

            # Step 3: Create transformer and load weights from safetensors
            print(f"[ModelLoader] Loading FLUX.2 transformer weights from: {file_path}")
            transformer_state_dict = load_file(file_path)
            print(f"[ModelLoader] Loaded {len(transformer_state_dict)} tensors from safetensors")

            # Detect state_dict format and convert if needed
            # FLUX.2 state_dict can be in 3 formats:
            # 1. BFL/Comfy format: double_blocks.*, single_blocks.* (original BFL weights)
            # 2. Diffusers format: time_guidance_embed.*, double_stream_modulation_*, single_transformer_blocks.*
            # 3. SushiUI/musubi training format: model.diffusion_model.* prefix (ComfyUI-style but with diffusers keys inside)
            sample_keys = list(transformer_state_dict.keys())[:5]
            is_bfl_format = any(k.startswith('double_blocks.') for k in transformer_state_dict.keys())
            is_sushiui_format = any(k.startswith('model.diffusion_model.') for k in transformer_state_dict.keys())

            if is_bfl_format:
                print(f"[ModelLoader] Detected BFL/Comfy format state_dict, converting to diffusers format...")
                from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
                transformer_state_dict = convert_flux2_transformer_checkpoint_to_diffusers(transformer_state_dict)
                print(f"[ModelLoader] Converted to diffusers format ({len(transformer_state_dict)} tensors)")
            elif is_sushiui_format:
                # SushiUI/musubi training saves with "model.diffusion_model." prefix
                # Extract only transformer keys (skip VAE "first_stage_model.*" and TE "text_encoders.*")
                print(f"[ModelLoader] Detected SushiUI/musubi training format state_dict, stripping prefix...")
                original_count = len(transformer_state_dict)
                new_state_dict = {}
                for key, value in transformer_state_dict.items():
                    if key.startswith('model.diffusion_model.'):
                        new_key = key.replace('model.diffusion_model.', '', 1)
                        new_state_dict[new_key] = value
                transformer_state_dict = new_state_dict
                print(f"[ModelLoader] Extracted {len(transformer_state_dict)} transformer tensors from {original_count} total tensors")
            else:
                print(f"[ModelLoader] State dict is already in diffusers format")

            # Create transformer model
            print(f"[ModelLoader] Creating Flux2Transformer2DModel...")
            transformer = Flux2Transformer2DModel(**transformer_config)

            # Load weights
            missing_keys, unexpected_keys = transformer.load_state_dict(transformer_state_dict, strict=False)
            if missing_keys:
                print(f"[ModelLoader] WARNING: Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"[ModelLoader] WARNING: Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"[ModelLoader] WARNING: Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"[ModelLoader] WARNING: Unexpected keys: {unexpected_keys}")

            transformer = transformer.to(dtype=torch_dtype)
            print(f"[ModelLoader] Transformer loaded with {sum(p.numel() for p in transformer.parameters()):,} parameters")

            # Step 4: Load VAE
            print(f"[ModelLoader] Loading FLUX.2 VAE...")
            vae = AutoencoderKLFlux2.from_pretrained(
                cache_dir,
                subfolder="vae",
                torch_dtype=torch.float32  # VAE in fp32 for quality
            )
            print(f"[ModelLoader] VAE loaded: latent_channels={vae.config.latent_channels}")

            # Step 5: Load Text Encoder (Qwen3)
            print(f"[ModelLoader] Loading Qwen3 text encoder...")
            text_encoder = Qwen3ForCausalLM.from_pretrained(
                cache_dir,
                subfolder="text_encoder",
                torch_dtype=torch_dtype
            )
            print(f"[ModelLoader] Text encoder loaded: Qwen3ForCausalLM")

            # Step 6: Load Tokenizer
            print(f"[ModelLoader] Loading tokenizer...")
            tokenizer = Qwen2TokenizerFast.from_pretrained(
                cache_dir,
                subfolder="tokenizer"
            )
            print(f"[ModelLoader] Tokenizer loaded: Qwen2TokenizerFast")

            # Step 7: Load Scheduler
            print(f"[ModelLoader] Loading scheduler...")
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                cache_dir,
                subfolder="scheduler"
            )
            print(f"[ModelLoader] Scheduler loaded: FlowMatchEulerDiscreteScheduler")

            print(f"[ModelLoader] FLUX.2 Klein loaded successfully")
            print(f"  - Transformer: {transformer_config.get('num_layers', 8)} dual + {transformer_config.get('num_single_layers', 48)} single layers")
            print(f"  - VAE: AutoencoderKLFlux2 (latent_channels={vae.config.latent_channels})")
            print(f"  - Text Encoder: Qwen3ForCausalLM")
            print(f"  - Tokenizer: Qwen2TokenizerFast (max_length=512)")
            print(f"  - Scheduler: FlowMatchEulerDiscreteScheduler")

            # Add is_distilled and base_model_repo to config dict (for inference and training)
            config_dict = transformer_config.copy()
            config_dict["is_distilled"] = is_distilled
            config_dict["base_model_repo"] = base_model_repo

            return {
                "transformer": transformer,
                "vae": vae,
                "text_encoder": text_encoder,
                "tokenizer": tokenizer,
                "scheduler": scheduler,
                "config": config_dict,
                "model_type": "flux2",  # Distinguish from Z-Image
            }

        except Exception as e:
            print(f"[ModelLoader] Error loading FLUX.2 model: {e}")
            import traceback
            traceback.print_exc()
            raise

    @staticmethod
    def load_from_safetensors(
        file_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ) -> Union[StableDiffusionPipeline, Dict[str, Any]]:
        """Load model from .safetensors file

        Returns:
            - StableDiffusionPipeline for SD1.5/SDXL
            - Dict of components for Z-Image/DEUS
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        model_type = ModelLoader.detect_model_type(file_path)
        print(f"[ModelLoader] Detected model type: {model_type}")

        # FLUX.2 format
        if model_type == "flux2":
            print(f"[ModelLoader] Loading as FLUX.2 Klein (Flux2Transformer2DModel)")
            return ModelLoader.load_flux2_from_safetensors(file_path, device, torch.bfloat16)

        # Anima (DiT + Qwen3 + Qwen-Image VAE)
        if model_type == "anima":
            print(f"[ModelLoader] Loading as Anima (Cosmos-Predict2 DiT)")
            return ModelLoader.load_anima_from_files(file_path, device, torch.bfloat16)

        # DEUS support removed - architecture no longer maintained
        # if model_type == "deus":
        #     print(f"[ModelLoader] Loading as DEUS (SigLIP-2 text encoder)")
        #     return ModelLoader.load_deus_from_safetensors(file_path, device, torch_dtype)

        # Z-Image Comfy format
        if model_type == "zimage":
            print(f"[ModelLoader] Loading as Z-Image (Comfy safetensors format)")
            return ModelLoader.load_zimage_from_comfy_safetensors(file_path, device, torch.bfloat16)

        # MiniT2I single-file (bundled FLAN-T5 + MM-JiT transformer)
        if model_type == "minit2i":
            print(f"[ModelLoader] Loading as MiniT2I (single-file)")
            return ModelLoader.load_minit2i_from_path(file_path, torch.bfloat16)

        is_v_prediction = ModelLoader.detect_v_prediction(file_path)

        # Custom SDXL architecture (SushiUI): non-standard latent VAE (e.g. FLUX.1 16ch).
        # Read sushi.vae_type / sushi.in_channels so the U-Net conv_in/out and the VAE
        # are reconstructed after load. Absent => standard SDXL (unchanged path).
        custom_vae_type = None
        custom_in_channels = None
        if model_type == "sdxl":
            try:
                from safetensors import safe_open
                with safe_open(file_path, framework="pt") as _f:
                    _md = _f.metadata() or {}
                _vt = (_md.get("sushi.vae_type") or "").strip().lower()
                if _vt and _vt not in ("none", "sdxl"):
                    custom_vae_type = _vt
                    custom_in_channels = int(_md.get("sushi.in_channels", "0") or 0) or None
                    print(f"[ModelLoader] Custom SDXL arch: vae_type={custom_vae_type}, "
                          f"in_channels={custom_in_channels}")
            except Exception as _e:
                print(f"[ModelLoader] custom-arch metadata read failed (standard load): {_e}")

        # Check if VAE is embedded
        print(f"[ModelLoader] Checking if model has embedded VAE...")
        has_vae = ModelLoader.has_embedded_vae(file_path)
        print(f"[ModelLoader] VAE detection result: {'embedded' if has_vae else 'not embedded'}")

        # Load external VAE only if not embedded
        external_vae = None
        if custom_vae_type:
            # Custom high-spec VAE is registry-referenced (not embedded); load it here.
            from core.models.sdxl_custom_arch import load_alt_vae
            print(f"[ModelLoader] Loading custom registry VAE: {custom_vae_type}")
            external_vae = load_alt_vae(custom_vae_type, torch_dtype=torch_dtype)
            has_vae = False
        elif not has_vae:
            if model_type == "sdxl":
                vae_repo = "madebyollin/sdxl-vae-fp16-fix"
            else:  # SD1.5
                vae_repo = "stabilityai/sd-vae-ft-mse-original"

            print(f"[ModelLoader] Model without embedded VAE detected")
            print(f"[ModelLoader] Loading external VAE: {vae_repo}")
            try:
                external_vae = AutoencoderKL.from_pretrained(
                    vae_repo,
                    torch_dtype=torch_dtype,
                    use_safetensors=True
                )
                print(f"[ModelLoader] External VAE loaded successfully")
            except Exception as e:
                print(f"[ModelLoader] ERROR: Failed to load external VAE: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to load external VAE: {e}")

        # Use single_file loading which is the standard way to load safetensors
        print(f"[ModelLoader] Loading as {'SDXL' if model_type == 'sdxl' else 'SD1.5'} (standard pipeline)")
        try:
            if model_type == "sdxl":
                # Only pass vae parameter if external VAE was loaded
                if external_vae is not None:
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        file_path,
                        num_in_channels=custom_in_channels,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        vae=external_vae,
                    )
                else:
                    # Use embedded VAE (don't pass vae parameter)
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        file_path,
                        num_in_channels=custom_in_channels,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                    )
            else:
                pipeline = StableDiffusionPipeline.from_single_file(
                    file_path,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                )
        except Exception as e:
            # Fallback: try with float32
            print(f"Failed to load with fp16, trying with fp32: {e}")
            if model_type == "sdxl":
                # Only pass vae parameter if external VAE was loaded
                if external_vae is not None:
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        file_path,
                        num_in_channels=custom_in_channels,
                        torch_dtype=torch.float32,
                        use_safetensors=True,
                        vae=external_vae,
                    )
                else:
                    # Use embedded VAE (don't pass vae parameter)
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        file_path,
                        num_in_channels=custom_in_channels,
                        torch_dtype=torch.float32,
                        use_safetensors=True,
                    )
            else:
                pipeline = StableDiffusionPipeline.from_single_file(
                    file_path,
                    torch_dtype=torch.float32,
                    use_safetensors=True,
                )

        # Custom SDXL arch: ensure conv_in/conv_out match the custom latent channels and
        # carry the trained weights (from_single_file overrides in_channels but not
        # out_channels). num_in_channels above loaded conv_in; resize fixes conv_out, then
        # both convs are assigned directly from the file for correctness.
        if custom_vae_type and custom_in_channels and hasattr(pipeline, "unet"):
            try:
                from core.models.sdxl_custom_arch import (
                    resize_unet_in_out, load_custom_convs_from_single_file,
                )
                resize_unet_in_out(pipeline.unet, custom_in_channels)
                load_custom_convs_from_single_file(pipeline.unet, file_path)
                print(f"[ModelLoader] Custom SDXL reconstructed: {custom_in_channels}ch latents "
                      f"({custom_vae_type} VAE)")
            except Exception as _re:
                print(f"[ModelLoader] ERROR reconstructing custom SDXL: {_re}")
                import traceback
                traceback.print_exc()

        # Configure scheduler for v-prediction if detected
        if is_v_prediction:
            print(f"[ModelLoader] Configuring scheduler for v-prediction model")
            ModelLoader._configure_v_prediction_scheduler(pipeline)

        # Move components to device individually (avoid pipeline.to() which can cause issues)
        print(f"[ModelLoader] Moving pipeline components to {device}...")
        print(f"[ModelLoader] DEBUG: Before any moves - pipeline.vae is not None: {pipeline.vae is not None}")
        print(f"[ModelLoader] DEBUG: Before any moves - 'vae' in components: {'vae' in pipeline.components}")

        # Move each component individually
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
            print(f"[ModelLoader] DEBUG: Moving text_encoder...")
            pipeline.text_encoder.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After text_encoder move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
            print(f"[ModelLoader] DEBUG: Moving text_encoder_2...")
            pipeline.text_encoder_2.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After text_encoder_2 move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
            print(f"[ModelLoader] DEBUG: Moving unet...")
            pipeline.unet.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After unet move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            print(f"[ModelLoader] DEBUG: Moving vae...")
            pipeline.vae.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After vae move - pipeline.vae is not None: {pipeline.vae is not None}")

        print(f"[ModelLoader] All components moved to {device}")
        print(f"[ModelLoader] DEBUG: After all moves - pipeline.vae is not None: {pipeline.vae is not None}")

        # Verify VAE exists after moving to device
        if not hasattr(pipeline, 'vae') or pipeline.vae is None:
            print(f"[ModelLoader] ERROR: VAE is missing after component move")
            print(f"[ModelLoader] Pipeline components: {list(pipeline.components.keys())}")
            raise RuntimeError("VAE is missing after loading model")
        else:
            print(f"[ModelLoader] VAE verified: {type(pipeline.vae).__name__}")

        return pipeline

    @staticmethod
    def load_zimage_from_diffusers(
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ) -> Dict[str, Any]:
        """Load Z-Image from diffusers format directory

        Returns:
            Dict containing transformer, vae, text_encoder, tokenizer, scheduler
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Z-Image model directory not found: {model_path}")

        print(f"[ModelLoader] Loading Z-Image from: {model_path}")

        # Check if model_path is a single safetensors file (Comfy format) or directory (diffusers format)
        is_single_file = os.path.isfile(model_path) and model_path.endswith('.safetensors')

        if is_single_file:
            print(f"[ModelLoader] Detected Comfy format safetensors, delegating to Comfy loader")
            # Use existing Comfy loader (already handles weight conversion)
            components = ModelLoader.load_zimage_from_comfy_safetensors(
                file_path=model_path,
                device=device,
                torch_dtype=torch_dtype
            )
            return components
        else:
            # Diffusers format directory
            raise NotImplementedError(
                f"Z-Image diffusers format directory loading is not yet implemented.\n"
                f"Please use Comfy format (.safetensors) instead.\n"
                f"Convert your model using ComfyUI's 'Save Model' feature."
            )

    @staticmethod
    def load_from_diffusers(
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ) -> Union[StableDiffusionPipeline, Dict[str, Any]]:
        """Load model from diffusers format directory

        Returns:
            - StableDiffusionPipeline for SD1.5/SDXL
            - Dict of components for Z-Image/DEUS
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        model_type = ModelLoader.detect_model_type(model_path)

        # DEUS support removed - architecture no longer maintained
        # if model_type == "deus":
        #     # DEUS diffusers format directory loading
        #     # For now, raise NotImplementedError (safetensors is the primary format)
        #     raise NotImplementedError(
        #         f"DEUS diffusers format directory loading is not yet implemented.\n"
        #         f"Please use safetensors format instead."
        #     )

        # Z-Image uses component-based loading
        if model_type == "zimage":
            return ModelLoader.load_zimage_from_diffusers(model_path, device, torch.bfloat16)

        # Anima split-files directory layout
        if model_type == "anima":
            print(f"[ModelLoader] Loading as Anima (split-files layout)")
            return ModelLoader.load_anima_from_files(model_path, device, torch.bfloat16)

        # Lens (microsoft/Lens) diffusers directory
        if model_type == "lens":
            print(f"[ModelLoader] Loading as Lens (diffusers directory)")
            return ModelLoader.load_lens_from_path(model_path, torch.bfloat16)

        # Ideogram 4 diffusers directory (dual-branch DiT + Qwen3-VL + AutoencoderKLFlux2)
        if model_type == "ideogram4":
            print(f"[ModelLoader] Loading as Ideogram 4 (diffusers directory)")
            return ModelLoader.load_ideogram4_from_path(model_path, torch.bfloat16)

        # MiniT2I diffusers directory (pixel-space MM-JiT + FLAN-T5, no VAE)
        if model_type == "minit2i":
            print(f"[ModelLoader] Loading as MiniT2I (diffusers directory)")
            return ModelLoader.load_minit2i_from_path(model_path, torch.bfloat16)

        is_v_prediction = ModelLoader.detect_v_prediction(model_path)

        if model_type == "sdxl":
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )

        # Configure scheduler for v-prediction if detected
        if is_v_prediction:
            print(f"[ModelLoader] Configuring scheduler for v-prediction model")
            ModelLoader._configure_v_prediction_scheduler(pipeline)

        # Move components to device individually (avoid pipeline.to() which can cause issues)
        print(f"[ModelLoader] Moving pipeline components to {device}...")
        print(f"[ModelLoader] DEBUG: Before any moves - pipeline.vae is not None: {pipeline.vae is not None}")
        print(f"[ModelLoader] DEBUG: Before any moves - 'vae' in components: {'vae' in pipeline.components}")

        # Move each component individually
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
            print(f"[ModelLoader] DEBUG: Moving text_encoder...")
            pipeline.text_encoder.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After text_encoder move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
            print(f"[ModelLoader] DEBUG: Moving text_encoder_2...")
            pipeline.text_encoder_2.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After text_encoder_2 move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
            print(f"[ModelLoader] DEBUG: Moving unet...")
            pipeline.unet.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After unet move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            print(f"[ModelLoader] DEBUG: Moving vae...")
            pipeline.vae.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After vae move - pipeline.vae is not None: {pipeline.vae is not None}")

        print(f"[ModelLoader] All components moved to {device}")
        print(f"[ModelLoader] DEBUG: After all moves - pipeline.vae is not None: {pipeline.vae is not None}")

        # Verify VAE exists after moving to device
        if not hasattr(pipeline, 'vae') or pipeline.vae is None:
            print(f"[ModelLoader] ERROR: VAE is missing after component move")
            print(f"[ModelLoader] Pipeline components: {list(pipeline.components.keys())}")
            raise RuntimeError("VAE is missing after loading model")
        else:
            print(f"[ModelLoader] VAE verified: {type(pipeline.vae).__name__}")

        return pipeline

    @staticmethod
    def load_from_huggingface(
        repo_id: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        revision: Optional[str] = None
    ) -> StableDiffusionPipeline:
        """Load model from HuggingFace repository"""
        # Lens (microsoft/Lens): detect by repo_id prefix
        if "microsoft/lens" in repo_id.lower() or repo_id.lower().endswith("/lens"):
            print(f"[ModelLoader] Loading as Lens (HuggingFace Hub): {repo_id}")
            return ModelLoader.load_lens_from_path(repo_id, torch.bfloat16)

        # Detect model type from repo_id or try loading
        if "xl" in repo_id.lower() or "sdxl" in repo_id.lower():
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch_dtype,
                revision=revision,
                use_safetensors=True,
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch_dtype,
                revision=revision,
                use_safetensors=True,
            )

        # Move components to device individually (avoid pipeline.to() which can cause issues)
        print(f"[ModelLoader] Moving pipeline components to {device}...")
        print(f"[ModelLoader] DEBUG: Before any moves - pipeline.vae is not None: {pipeline.vae is not None}")
        print(f"[ModelLoader] DEBUG: Before any moves - 'vae' in components: {'vae' in pipeline.components}")

        # Move each component individually
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
            print(f"[ModelLoader] DEBUG: Moving text_encoder...")
            pipeline.text_encoder.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After text_encoder move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
            print(f"[ModelLoader] DEBUG: Moving text_encoder_2...")
            pipeline.text_encoder_2.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After text_encoder_2 move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
            print(f"[ModelLoader] DEBUG: Moving unet...")
            pipeline.unet.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After unet move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            print(f"[ModelLoader] DEBUG: Moving vae...")
            pipeline.vae.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After vae move - pipeline.vae is not None: {pipeline.vae is not None}")

        print(f"[ModelLoader] All components moved to {device}")
        print(f"[ModelLoader] DEBUG: After all moves - pipeline.vae is not None: {pipeline.vae is not None}")

        # Verify VAE exists after moving to device
        if not hasattr(pipeline, 'vae') or pipeline.vae is None:
            print(f"[ModelLoader] ERROR: VAE is missing after component move")
            print(f"[ModelLoader] Pipeline components: {list(pipeline.components.keys())}")
            raise RuntimeError("VAE is missing after loading model")
        else:
            print(f"[ModelLoader] VAE verified: {type(pipeline.vae).__name__}")

        return pipeline

    @staticmethod
    def load_model(
        source_type: ModelSource,
        source: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs
    ) -> Union[StableDiffusionPipeline, Dict[str, Any]]:
        """Universal model loading method

        Returns:
            - StableDiffusionPipeline for SD1.5/SDXL
            - Dict of components for Z-Image
        """
        if source_type == "safetensors":
            return ModelLoader.load_from_safetensors(source, device, torch_dtype)
        elif source_type == "diffusers":
            return ModelLoader.load_from_diffusers(source, device, torch_dtype)
        elif source_type == "huggingface":
            return ModelLoader.load_from_huggingface(
                source,
                device,
                torch_dtype,
                revision=kwargs.get("revision")
            )
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    @staticmethod
    def load_anima_from_files(
        path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        text_encoder_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        models_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load Anima (DiT + Qwen3 + Qwen-Image VAE) from either a single DiT
        safetensors file or a split-files directory layout.

        Returns a component dict consumed by PipelineManager.load_model().
        """
        from core.models.anima.anima_loader import (
            load_anima_components, detect_anima_split_layout, discover_anima_components,
        )

        # If the user pointed at a directory (split layout), pick the DiT file
        dit_path = path
        if os.path.isdir(path):
            split = detect_anima_split_layout(path)
            if not split or not split.get("dit"):
                raise FileNotFoundError(
                    f"Anima split layout expected at {path} but no DiT safetensors found."
                )
            dit_path = split["dit"]
            if text_encoder_path is None:
                text_encoder_path = split.get("text_encoder")
            if vae_path is None:
                vae_path = split.get("vae")

        # Resolve models_root: the parent of the dit_path's "models" ancestor, or settings.
        if models_root is None:
            try:
                from config.settings import settings
                models_root = getattr(settings, "models_dir", None)
            except Exception:
                models_root = None
            if models_root is None:
                # Walk up to find a `models` directory ancestor
                p = os.path.abspath(dit_path)
                for _ in range(6):
                    p = os.path.dirname(p)
                    if not p:
                        break
                    if os.path.basename(p).lower() == "models":
                        models_root = p
                        break

        return load_anima_components(
            dit_path=dit_path,
            text_encoder_path=text_encoder_path,
            vae_path=vae_path,
            models_root=models_root,
            device="cpu",  # Loaded to CPU; pipeline.py moves to GPU per stage
            dit_dtype=torch_dtype,
            te_dtype=torch_dtype,
            vae_dtype=torch_dtype,
        )

    @staticmethod
    def load_lens_from_path(
        path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Load Microsoft/Lens from a local diffusers directory or HF Hub ID.

        Returns a component dict consumed by PipelineManager.load_model().
        """
        from core.models.lens.lens_loader import load_lens_components
        return load_lens_components(model_path=path, torch_dtype=torch_dtype)

    @staticmethod
    def load_ideogram4_from_path(
        path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Load Ideogram 4 from a local diffusers directory.

        Returns a component dict consumed by PipelineManager.load_model().
        """
        from core.models.ideogram4.ideogram4_loader import load_ideogram4_components
        return load_ideogram4_components(model_path=path, torch_dtype=torch_dtype)

    @staticmethod
    def load_minit2i_from_path(
        path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Load MiniT2I from a diffusers directory or a single-file safetensors.

        Returns a component dict consumed by PipelineManager.load_model().
        """
        from core.models.minit2i.minit2i_loader import load_minit2i_components
        return load_minit2i_components(model_path=path, torch_dtype=torch_dtype)
