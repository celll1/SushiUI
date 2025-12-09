from typing import Dict, Any, Optional, Literal, Union
import os
import sys
import json
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from safetensors.torch import load_file
from pathlib import Path

ModelSource = Literal["safetensors", "diffusers", "huggingface"]
ModelType = Literal["sd15", "sdxl", "zimage"]

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
    def detect_v_prediction(model_path: str) -> bool:
        """Detect if a model is v-prediction by checking for 'v_pred' key in state_dict

        V-prediction models have a 'v_pred' key in their state_dict or config.
        This is different from standard epsilon-prediction models.

        Returns:
            True if v-prediction model, False otherwise
        """
        try:
            if model_path.endswith('.safetensors'):
                # Check safetensors metadata
                from safetensors import safe_open
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    # Check metadata first
                    metadata = f.metadata()
                    if metadata:
                        # Check for v_pred in metadata
                        if metadata.get('v_pred') or metadata.get('prediction_type') == 'v_prediction':
                            print(f"[ModelLoader] Detected v-prediction model from metadata: {model_path}")
                            return True

                    # Also check state dict keys for v_pred indicator
                    keys = list(f.keys())
                    if 'v_pred' in keys:
                        print(f"[ModelLoader] Detected v-prediction model from state_dict keys: {model_path}")
                        return True

            elif os.path.isdir(model_path):
                # Check diffusers format config
                config_path = os.path.join(model_path, "scheduler", "scheduler_config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if config.get('prediction_type') == 'v_prediction':
                            print(f"[ModelLoader] Detected v-prediction model from scheduler config: {model_path}")
                            return True

            return False

        except Exception as e:
            print(f"[ModelLoader] Warning: Could not detect v-prediction status: {e}")
            return False

    @staticmethod
    def detect_model_type(model_path: str) -> ModelType:
        """Detect if model is SD1.5, SDXL, or Z-Image based on config or structure

        Supports:
        - Z-Image diffusers format (directory with transformer/, vae/, etc.)
        - Z-Image Comfy format (single safetensors with transformer weights only)
        - SD1.5/SDXL diffusers and safetensors
        """
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

                    # Z-Image Comfy format detection
                    # Z-Image transformer has unique keys: cap_embedder, t_embedder, x_embedder, layers, context_refiner
                    zimage_indicators = ['cap_embedder', 't_embedder', 'x_embedder', 'context_refiner']
                    if all(any(k.startswith(indicator) for k in keys) for indicator in zimage_indicators):
                        print(f"[ModelLoader] Detected Z-Image model (Comfy safetensors format): {model_path}")
                        return "zimage"

                    # SDXL detection by file size (>6GB)
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

        # Add Z-Image source to Python path
        zimage_src_path = Path(__file__).parent.parent.parent.parent / "Z-Image" / "src"
        if not zimage_src_path.exists():
            raise FileNotFoundError(
                f"Z-Image source code not found at: {zimage_src_path}\n"
                f"Please clone Z-Image repository to: {zimage_src_path.parent}"
            )

        # Temporarily replace sys.path to prioritize Z-Image modules
        original_sys_path = sys.path.copy()
        sys.path = [str(zimage_src_path)] + sys.path

        try:
            from transformers import AutoModel, AutoTokenizer
            from safetensors.torch import load_file
            import importlib.util

            # CRITICAL: Load Z-Image's config module first and inject it into sys.modules
            # This prevents transformer.py from importing SushiUI's config
            config_spec = importlib.util.spec_from_file_location(
                "config",
                zimage_src_path / "config" / "__init__.py"
            )
            config_module = importlib.util.module_from_spec(config_spec)

            # Temporarily inject Z-Image config into sys.modules
            import sys as _sys
            original_config = _sys.modules.get('config')
            _sys.modules['config'] = config_module
            config_spec.loader.exec_module(config_module)

            # Now load Z-Image modules (they will import the correct config)
            # CRITICAL: Load SushiUI's custom transformer module (Block Swap integrated)
            # This replaces the original Z-Image transformer.py with our modified version
            sushiui_transformer_path = Path(__file__).parent / "models" / "zimage_transformer.py"
            transformer_spec = importlib.util.spec_from_file_location(
                "zimage_transformer",
                sushiui_transformer_path
            )
            transformer_module = importlib.util.module_from_spec(transformer_spec)
            transformer_spec.loader.exec_module(transformer_module)
            ZImageTransformer2DModel = transformer_module.ZImageTransformer2DModel
            print(f"[ModelLoader] Loaded SushiUI Z-Image Transformer (Block Swap integrated) from: {sushiui_transformer_path}")

            # Load autoencoder module
            autoencoder_spec = importlib.util.spec_from_file_location(
                "zimage_autoencoder",
                zimage_src_path / "zimage" / "autoencoder.py"
            )
            autoencoder_module = importlib.util.module_from_spec(autoencoder_spec)
            autoencoder_spec.loader.exec_module(autoencoder_module)
            AutoencoderKL = autoencoder_module.AutoencoderKL

            # Load scheduler module
            scheduler_spec = importlib.util.spec_from_file_location(
                "zimage_scheduler",
                zimage_src_path / "zimage" / "scheduler.py"
            )
            scheduler_module = importlib.util.module_from_spec(scheduler_spec)
            scheduler_spec.loader.exec_module(scheduler_module)
            FlowMatchEulerDiscreteScheduler = scheduler_module.FlowMatchEulerDiscreteScheduler

            # Restore original config module
            if original_config is not None:
                _sys.modules['config'] = original_config
            else:
                del _sys.modules['config']

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

            # Step 3: Create transformer model
            print("[ModelLoader] Creating Z-Image transformer...")
            with torch.device("meta"):
                transformer = ZImageTransformer2DModel(
                    all_patch_size=tuple(transformer_config["all_patch_size"]),
                    all_f_patch_size=tuple(transformer_config["all_f_patch_size"]),
                    in_channels=transformer_config["in_channels"],
                    dim=transformer_config["dim"],
                    n_layers=transformer_config["n_layers"],
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

            # Step 4: Load Comfy safetensors weights into transformer
            print(f"[ModelLoader] Loading Comfy transformer weights from: {file_path}")
            comfy_state_dict = load_file(file_path, device="cpu")

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

            print("[ModelLoader] Moving transformer to GPU...")
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

            print("[ModelLoader] Loading text encoder...")
            text_encoder_path = os.path.join(base_model_path, "text_encoder")
            text_encoder = AutoModel.from_pretrained(
                text_encoder_path,
                dtype=torch_dtype,
                trust_remote_code=True,
            )
            text_encoder.to(device)
            text_encoder.eval()

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
        finally:
            # Restore original sys.path
            sys.path = original_sys_path

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

        # Z-Image Comfy format
        if model_type == "zimage":
            return ModelLoader.load_zimage_from_comfy_safetensors(file_path, device, torch.bfloat16)

        is_v_prediction = ModelLoader.detect_v_prediction(file_path)

        # Use single_file loading which is the standard way to load safetensors
        try:
            if model_type == "sdxl":
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

        # Move to device
        pipeline = pipeline.to(device, dtype=torch_dtype)
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

        # Add Z-Image source to Python path
        zimage_src_path = Path(__file__).parent.parent.parent.parent / "Z-Image" / "src"
        if not zimage_src_path.exists():
            raise FileNotFoundError(
                f"Z-Image source code not found at: {zimage_src_path}\n"
                f"Please clone Z-Image repository to: {zimage_src_path.parent}"
            )

        # Temporarily replace sys.path to prioritize Z-Image modules
        original_sys_path = sys.path.copy()
        sys.path = [str(zimage_src_path)] + sys.path

        try:
            import importlib.util

            # CRITICAL: Load Z-Image's config module and inject into sys.modules
            config_spec = importlib.util.spec_from_file_location(
                "config",
                zimage_src_path / "config" / "__init__.py"
            )
            config_module = importlib.util.module_from_spec(config_spec)
            original_config = sys.modules.get('config')
            sys.modules['config'] = config_module
            config_spec.loader.exec_module(config_module)

            # CRITICAL: Load SushiUI's custom transformer module and inject as zimage.transformer
            # This ensures load_from_local_dir() uses our Block Swap integrated version
            sushiui_transformer_path = Path(__file__).parent / "models" / "zimage_transformer.py"
            transformer_spec = importlib.util.spec_from_file_location(
                "zimage.transformer",
                sushiui_transformer_path
            )
            transformer_module = importlib.util.module_from_spec(transformer_spec)
            sys.modules['zimage.transformer'] = transformer_module
            transformer_spec.loader.exec_module(transformer_module)
            print(f"[ModelLoader] Injected SushiUI Z-Image Transformer (Block Swap integrated) into sys.modules")

            # Now load Z-Image components (will use our custom transformer)
            from utils.loader import load_from_local_dir

            # Check if model_path is a single safetensors file (Comfy format) or directory (diffusers format)
            is_single_file = os.path.isfile(model_path) and model_path.endswith('.safetensors')

            if is_single_file:
                print(f"[ModelLoader] Detected Comfy format safetensors, using existing Comfy loader")
                # Use existing Comfy loader (already handles weight conversion)
                # Restore sys.path first before calling Comfy loader
                sys.path = original_sys_path

                components = ModelLoader.load_zimage_from_comfy_safetensors(
                    file_path=model_path,
                    device=device,
                    torch_dtype=torch_dtype
                )

                # Don't restore config module here - Comfy loader handles it
                return components
            else:
                print(f"[ModelLoader] Loading from diffusers directory")
                components = load_from_local_dir(
                    model_path,
                    device=device,
                    dtype=torch_dtype,
                    verbose=True,
                    compile=False  # Disable compile for now
                )

            # Restore original config module
            if original_config is not None:
                sys.modules['config'] = original_config
            else:
                if 'config' in sys.modules:
                    del sys.modules['config']

            print(f"[ModelLoader] Z-Image components loaded successfully")
            print(f"  - Transformer: {type(components['transformer']).__name__}")
            print(f"  - VAE: {type(components['vae']).__name__}")
            print(f"  - Text Encoder: {type(components['text_encoder']).__name__}")
            print(f"  - Scheduler: {type(components['scheduler']).__name__}")

            return components

        except Exception as e:
            print(f"[ModelLoader] Error loading Z-Image: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Restore original sys.path
            sys.path = original_sys_path

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

        # Move to device
        pipeline = pipeline.to(device, dtype=torch_dtype)
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

        # Move to device
        pipeline = pipeline.to(device, dtype=torch_dtype)
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
