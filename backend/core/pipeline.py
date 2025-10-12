from typing import Dict, Any, Optional, List
from PIL import Image
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline
)
from config.settings import settings
from extensions.base_extension import BaseExtension
from core.model_loader import ModelLoader, ModelSource

class DiffusionPipelineManager:
    """Manages Stable Diffusion pipelines and extensions"""

    def __init__(self):
        self.txt2img_pipeline: Optional[StableDiffusionPipeline] = None
        self.img2img_pipeline: Optional[StableDiffusionImg2ImgPipeline] = None
        self.inpaint_pipeline: Optional[StableDiffusionInpaintPipeline] = None
        self.current_model: Optional[str] = None
        self.current_model_info: Optional[Dict[str, str]] = None
        self.extensions: List[BaseExtension] = []
        self.device = settings.device

    def load_model(
        self,
        source_type: ModelSource,
        source: str,
        pipeline_type: str = "txt2img",
        **kwargs
    ):
        """Load a Stable Diffusion model from various sources"""
        model_id = f"{source_type}:{source}"

        if self.current_model == model_id:
            return

        try:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

            # Load base pipeline
            base_pipeline = ModelLoader.load_model(
                source_type=source_type,
                source=source,
                device=self.device,
                torch_dtype=torch_dtype,
                **kwargs
            )

            # Determine if SDXL
            is_sdxl = isinstance(base_pipeline, StableDiffusionXLPipeline)

            # Convert to appropriate pipeline type
            if pipeline_type == "txt2img":
                self.txt2img_pipeline = base_pipeline
            elif pipeline_type == "img2img":
                # Convert to img2img pipeline using components method
                if is_sdxl:
                    self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline(**base_pipeline.components)
                else:
                    self.img2img_pipeline = StableDiffusionImg2ImgPipeline(**base_pipeline.components)
                self.img2img_pipeline = self.img2img_pipeline.to(self.device)
            elif pipeline_type == "inpaint":
                # Convert to inpaint pipeline using components method
                if is_sdxl:
                    self.inpaint_pipeline = StableDiffusionXLInpaintPipeline(**base_pipeline.components)
                else:
                    self.inpaint_pipeline = StableDiffusionInpaintPipeline(**base_pipeline.components)
                self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)

            self.current_model = model_id
            self.current_model_info = {
                "source_type": source_type,
                "source": source,
                "type": ModelLoader.detect_model_type(source) if source_type != "huggingface" else "unknown"
            }

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def register_extension(self, extension: BaseExtension):
        """Register a new extension"""
        self.extensions.append(extension)

    def generate_txt2img(self, params: Dict[str, Any]) -> Image.Image:
        """Generate image from text"""
        if not self.txt2img_pipeline:
            raise RuntimeError("txt2img pipeline not loaded. Please load a model first.")

        # Apply extensions before generation
        for ext in self.extensions:
            if ext.enabled:
                params = ext.process_before_generation(self.txt2img_pipeline, params)

        # Check if SDXL
        is_sdxl = isinstance(self.txt2img_pipeline, StableDiffusionXLPipeline)

        # Prepare generation parameters
        gen_params = {
            "prompt": params["prompt"],
            "negative_prompt": params.get("negative_prompt", ""),
            "num_inference_steps": params.get("steps", settings.default_steps),
            "guidance_scale": params.get("cfg_scale", settings.default_cfg_scale),
        }

        # Add size parameters only if not SDXL (SDXL has different parameter names)
        if not is_sdxl:
            gen_params["width"] = params.get("width", settings.default_width)
            gen_params["height"] = params.get("height", settings.default_height)
        else:
            # SDXL uses different size parameters
            gen_params["width"] = params.get("width", 1024)
            gen_params["height"] = params.get("height", 1024)

        # Add generator if seed is specified
        if params.get("seed", -1) >= 0:
            gen_params["generator"] = torch.Generator(device=self.device).manual_seed(params["seed"])

        # Generate image
        try:
            result = self.txt2img_pipeline(**gen_params)
            image = result.images[0]
        except Exception as e:
            print(f"Generation error: {e}")
            print(f"Parameters used: {gen_params}")
            raise

        # Apply extensions after generation
        for ext in self.extensions:
            if ext.enabled:
                image = ext.process_after_generation(image, params)

        return image

    def generate_img2img(self, params: Dict[str, Any], init_image: Image.Image) -> Image.Image:
        """Generate image from image"""
        if not self.img2img_pipeline:
            raise RuntimeError("img2img pipeline not loaded")

        # Apply extensions before generation
        for ext in self.extensions:
            if ext.enabled:
                params = ext.process_before_generation(self.img2img_pipeline, params)

        result = self.img2img_pipeline(
            prompt=params["prompt"],
            negative_prompt=params.get("negative_prompt", ""),
            image=init_image,
            strength=params.get("denoising_strength", 0.75),
            num_inference_steps=params.get("steps", settings.default_steps),
            guidance_scale=params.get("cfg_scale", settings.default_cfg_scale),
            generator=torch.Generator(device=self.device).manual_seed(params.get("seed", -1)) if params.get("seed", -1) >= 0 else None,
        )

        image = result.images[0]

        # Apply extensions after generation
        for ext in self.extensions:
            if ext.enabled:
                image = ext.process_after_generation(image, params)

        return image

    def generate_inpaint(
        self,
        params: Dict[str, Any],
        init_image: Image.Image,
        mask_image: Image.Image
    ) -> Image.Image:
        """Generate inpainted image"""
        if not self.inpaint_pipeline:
            raise RuntimeError("inpaint pipeline not loaded")

        # Apply extensions before generation
        for ext in self.extensions:
            if ext.enabled:
                params = ext.process_before_generation(self.inpaint_pipeline, params)

        result = self.inpaint_pipeline(
            prompt=params["prompt"],
            negative_prompt=params.get("negative_prompt", ""),
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=params.get("steps", settings.default_steps),
            guidance_scale=params.get("cfg_scale", settings.default_cfg_scale),
            generator=torch.Generator(device=self.device).manual_seed(params.get("seed", -1)) if params.get("seed", -1) >= 0 else None,
        )

        image = result.images[0]

        # Apply extensions after generation
        for ext in self.extensions:
            if ext.enabled:
                image = ext.process_after_generation(image, params)

        return image

# Global pipeline manager instance
pipeline_manager = DiffusionPipelineManager()
