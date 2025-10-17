from typing import Dict, Any, Optional, Literal
import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from safetensors.torch import load_file
from pathlib import Path

ModelSource = Literal["safetensors", "diffusers", "huggingface"]

class ModelLoader:
    """Handles loading models from various sources"""

    @staticmethod
    def _configure_v_prediction_scheduler(pipeline):
        """Configure scheduler for v-prediction models

        V-prediction models require:
        1. prediction_type = "v_prediction"
        2. rescale_betas_zero_snr = True (rescale betas for zero terminal SNR)
        3. timestep_spacing = "trailing" (recommended for v-prediction)

        References:
        - https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/16567
        - https://huggingface.co/docs/diffusers/using-diffusers/scheduler
        """
        try:
            # Register to scheduler config (this modifies the scheduler's configuration)
            pipeline.scheduler.register_to_config(
                prediction_type="v_prediction",
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing"
            )

            print(f"[ModelLoader] V-prediction scheduler configured:")
            print(f"  - prediction_type: v_prediction")
            print(f"  - rescale_betas_zero_snr: True")
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
    def detect_model_type(model_path: str) -> str:
        """Detect if model is SD1.5 or SDXL based on config or structure"""
        # Check for SDXL indicators
        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, "model_index.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # SDXL uses different UNet config
                    if "_class_name" in config and "XL" in config["_class_name"]:
                        return "sdxl"

        # Check file size for safetensors (SDXL is typically >6GB)
        if model_path.endswith('.safetensors'):
            file_size = os.path.getsize(model_path) / (1024**3)  # GB
            if file_size > 6:
                return "sdxl"

        return "sd15"

    @staticmethod
    def load_from_safetensors(
        file_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ) -> StableDiffusionPipeline:
        """Load model from .safetensors file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        model_type = ModelLoader.detect_model_type(file_path)
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
    def load_from_diffusers(
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ) -> StableDiffusionPipeline:
        """Load model from diffusers format directory"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        model_type = ModelLoader.detect_model_type(model_path)
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
    ) -> StableDiffusionPipeline:
        """Universal model loading method"""
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
