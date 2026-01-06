from typing import Dict, Any, Optional, Literal, Union
import os
import sys
import json
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL
from safetensors.torch import load_file
from pathlib import Path

ModelSource = Literal["safetensors", "diffusers", "huggingface"]
ModelType = Literal["sd15", "sdxl", "zimage", "original"]

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
    def detect_model_type(model_path: str) -> ModelType:
        """Detect if model is SD1.5, SDXL, Z-Image, or Original based on config or structure

        Supports:
        - Original architecture (directory with unet/, encoder/, vae/)
        - Z-Image diffusers format (directory with transformer/, vae/, etc.)
        - Z-Image Comfy format (single safetensors with transformer weights only)
        - SD1.5/SDXL diffusers and safetensors
        """
        # Original architecture detection (diffusers-like format)
        if os.path.isdir(model_path):
            # Original has unet/ directory with specific config markers
            unet_config = os.path.join(model_path, "unet", "config.json")
            if os.path.exists(unet_config):
                try:
                    with open(unet_config, 'r') as f:
                        config = json.load(f)
                        # Original has unique markers: skip_connection_interval, variant
                        if "skip_connection_interval" in config and "variant" in config:
                            print(f"[ModelLoader] Detected Original architecture: {model_path}")
                            return "original"
                except Exception as e:
                    print(f"[ModelLoader] Warning: Could not read unet config: {e}")

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
                            print(f"[ModelLoader] Detected Z-Image model (diffusers format): {model_path}")
                            return "zimage"
                except Exception as e:
                    print(f"[ModelLoader] Warning: Could not read transformer config: {e}")

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

                    # SD/SDXL detection (priority check)
                    # SD/SDXL models have U-Net keys starting with "model.diffusion_model."
                    has_unet_keys = any(k.startswith('model.diffusion_model.') for k in keys)

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
                        print(f"[ModelLoader] Detected Z-Image model (Comfy safetensors format): {model_path}")
                        return "zimage"

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

            # Step 4: Create transformer model with detected layer count
            print("[ModelLoader] Creating Z-Image transformer...")
            with torch.device("meta"):
                transformer = ZImageTransformer2DModel(
                    all_patch_size=tuple(transformer_config["all_patch_size"]),
                    all_f_patch_size=tuple(transformer_config["all_f_patch_size"]),
                    in_channels=transformer_config["in_channels"],
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

            # Step 5: Load other components from base model
            print("[ModelLoader] Loading VAE...")
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

            print("[ModelLoader] Z-Image Comfy format loaded successfully")
            print(f"  - Transformer: Loaded from {file_path}")
            print(f"  - VAE, Text Encoder, Tokenizer, Scheduler: Loaded from {base_model_repo}")

            return {
                "transformer": transformer,
                "vae": vae,
                "text_encoder": text_encoder,
                "tokenizer": tokenizer,
                "scheduler": scheduler,
            }

        except Exception as e:
            print(f"[ModelLoader] Error loading Z-Image Comfy format: {e}")
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
            - Dict of components for Z-Image
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        model_type = ModelLoader.detect_model_type(file_path)
        print(f"[ModelLoader] Detected model type: {model_type}")

        # Z-Image Comfy format
        if model_type == "zimage":
            print(f"[ModelLoader] Loading as Z-Image (Comfy safetensors format)")
            return ModelLoader.load_zimage_from_comfy_safetensors(file_path, device, torch.bfloat16)

        is_v_prediction = ModelLoader.detect_v_prediction(file_path)

        # Check if VAE is embedded
        print(f"[ModelLoader] Checking if model has embedded VAE...")
        has_vae = ModelLoader.has_embedded_vae(file_path)
        print(f"[ModelLoader] VAE detection result: {'embedded' if has_vae else 'not embedded'}")

        # Load external VAE only if not embedded
        external_vae = None
        if not has_vae:
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
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        vae=external_vae,
                    )
                else:
                    # Use embedded VAE (don't pass vae parameter)
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        file_path,
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
                        torch_dtype=torch.float32,
                        use_safetensors=True,
                        vae=external_vae,
                    )
                else:
                    # Use embedded VAE (don't pass vae parameter)
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        file_path,
                        torch_dtype=torch.float32,
                        use_safetensors=True,
                    )
            else:
                pipeline = StableDiffusionPipeline.from_single_file(
                    file_path,
                    torch_dtype=torch.float32,
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
            - Dict of components for Z-Image
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        model_type = ModelLoader.detect_model_type(model_path)

        # Z-Image uses component-based loading
        if model_type == "zimage":
            return ModelLoader.load_zimage_from_diffusers(model_path, device, torch.bfloat16)

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
            - OriginalPipeline for Original architecture
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
    def load_original_architecture(
        model_path: Optional[str] = None,
        unet_variant: str = "medium",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ):
        """Load or create Original architecture pipeline

        Args:
            model_path: Path to unified safetensors checkpoint (if None, create new pipeline)
            unet_variant: "small", "medium", or "large" (for new pipeline or checkpoint structure)
            device: Device to load on
            torch_dtype: Data type

        Returns:
            OriginalPipeline instance
        """
        from core.pipelines.pipeline_original import create_original_pipeline, load_original_pipeline_from_checkpoint

        print(f"[ModelLoader] Loading Original architecture...")

        if model_path is None:
            # Create new pipeline with random weights
            print(f"[ModelLoader] Creating new pipeline (variant: {unet_variant})")
            pipeline = create_original_pipeline(
                unet_variant=unet_variant,
                dtype=torch_dtype,
                device=device
            )
        else:
            # Load from unified safetensors checkpoint
            if not model_path.endswith('.safetensors'):
                raise ValueError(
                    f"Original architecture requires .safetensors checkpoint. "
                    f"Got: {model_path}"
                )

            print(f"[ModelLoader] Loading from checkpoint: {model_path}")
            pipeline = load_original_pipeline_from_checkpoint(
                checkpoint_path=model_path,
                unet_variant=unet_variant,
                dtype=torch_dtype,
                device=device
            )

        return pipeline
