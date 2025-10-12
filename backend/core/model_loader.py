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

        # Move to device and ensure all components have the same dtype
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
