from typing import Dict, Any, Optional, List
from PIL import Image
import torch
import json
import os
from pathlib import Path
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
from core.schedulers import get_scheduler
# Prompt parser imports are done locally in methods to avoid circular imports

LAST_MODEL_CONFIG_FILE = Path("last_model.json")

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

        # Auto-load last used model on startup
        self._auto_load_last_model()

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

            # Save this model as the last loaded model
            self._save_last_model(source_type, source, pipeline_type)

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _save_last_model(self, source_type: str, source: str, pipeline_type: str):
        """Save the last loaded model configuration to file"""
        try:
            config = {
                "source_type": source_type,
                "source": source,
                "pipeline_type": pipeline_type
            }
            with open(LAST_MODEL_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save last model config: {e}")

    def _auto_load_last_model(self):
        """Auto-load the last used model on startup"""
        if not LAST_MODEL_CONFIG_FILE.exists():
            print("No previous model to load")
            return

        try:
            with open(LAST_MODEL_CONFIG_FILE, 'r') as f:
                config = json.load(f)

            source_type = config.get("source_type")
            source = config.get("source")
            pipeline_type = config.get("pipeline_type", "txt2img")

            if source_type and source:
                print(f"Auto-loading last model: {source_type}:{source}")
                self.load_model(
                    source_type=source_type,
                    source=source,
                    pipeline_type=pipeline_type
                )
                print(f"Successfully loaded last model: {source}")
        except Exception as e:
            print(f"Warning: Failed to auto-load last model: {e}")

    def register_extension(self, extension: BaseExtension):
        """Register a new extension"""
        self.extensions.append(extension)

    def _encode_prompt_with_weights(self, prompt: str, negative_prompt: str = "", pipeline=None):
        """
        Encode prompts with A1111-style emphasis weights.

        Returns:
            For SD1.5: (prompt_embeds, negative_prompt_embeds)
            For SDXL: (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        """
        # Check if prompt or negative prompt contains emphasis syntax
        has_pos_emphasis = '(' in prompt or '[' in prompt
        has_neg_emphasis = '(' in negative_prompt or '[' in negative_prompt

        if not has_pos_emphasis and not has_neg_emphasis:
            # No emphasis - use normal prompt encoding
            return None, None, None, None

        # Use provided pipeline or default to txt2img_pipeline
        if pipeline is None:
            pipeline = self.txt2img_pipeline

        if pipeline is None:
            return None, None, None, None

        # Check if SDXL
        is_sdxl = isinstance(pipeline, StableDiffusionXLPipeline) or isinstance(pipeline, StableDiffusionXLImg2ImgPipeline)

        device = self.device
        dtype = pipeline.dtype if hasattr(pipeline, 'dtype') else torch.float16

        # NEW APPROACH: Use pipeline's encode_prompt, then apply weights
        # This ensures our embeddings match exactly what the pipeline would generate
        from .prompt_parser import parse_prompt_attention, apply_emphasis_to_embeds

        # Parse to get clean text and weights
        parsed_pos = parse_prompt_attention(prompt) if has_pos_emphasis else [(prompt, 1.0)]
        clean_prompt = "".join([text for text, _ in parsed_pos])

        # Use pipeline's native encode_prompt to get correct embeddings
        base_embeds = pipeline.encode_prompt(
            prompt=clean_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False  # We'll handle negative separately
        )

        # Extract embeddings (format depends on pipeline type)
        prompt_embeds = base_embeds[0]  # prompt_embeds
        pooled_prompt_embeds = base_embeds[2] if len(base_embeds) > 2 and is_sdxl else None

        # Apply weights to prompt_embeds
        if has_pos_emphasis:
            prompt_embeds = apply_emphasis_to_embeds(prompt, prompt_embeds, pipeline.tokenizer if not is_sdxl else pipeline.tokenizer_2, device, dtype)

        # Encode negative prompt similarly
        if negative_prompt:
            parsed_neg = parse_prompt_attention(negative_prompt) if has_neg_emphasis else [(negative_prompt, 1.0)]
            clean_neg_prompt = "".join([text for text, _ in parsed_neg])

            # Use pipeline's encode_prompt for negative
            neg_embeds = pipeline.encode_prompt(
                prompt=clean_neg_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )

            negative_prompt_embeds = neg_embeds[0]
            negative_pooled_prompt_embeds = neg_embeds[2] if len(neg_embeds) > 2 and is_sdxl else None

            # Apply weights if negative has emphasis
            if has_neg_emphasis:
                negative_prompt_embeds = apply_emphasis_to_embeds(negative_prompt, negative_prompt_embeds, pipeline.tokenizer if not is_sdxl else pipeline.tokenizer_2, device, dtype)
        else:
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def generate_txt2img(self, params: Dict[str, Any], progress_callback=None) -> tuple[Image.Image, int]:
        """Generate image from text

        Returns:
            tuple: (image, actual_seed)
        """
        if not self.txt2img_pipeline:
            raise RuntimeError("txt2img pipeline not loaded. Please load a model first.")

        # Apply extensions before generation
        for ext in self.extensions:
            if ext.enabled:
                params = ext.process_before_generation(self.txt2img_pipeline, params)

        # Set sampler and schedule type if specified
        sampler = params.get("sampler", "euler")
        schedule_type = params.get("schedule_type", "uniform")
        if sampler:
            try:
                self.txt2img_pipeline.scheduler = get_scheduler(self.txt2img_pipeline, sampler, schedule_type)
            except Exception as e:
                print(f"Warning: Could not set sampler to {sampler} with schedule {schedule_type}: {e}")

        # Check if SDXL
        is_sdxl = isinstance(self.txt2img_pipeline, StableDiffusionXLPipeline)

        # Encode prompts with weights if emphasis syntax is present
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt_with_weights(
            params["prompt"],
            params.get("negative_prompt", ""),
            pipeline=self.txt2img_pipeline
        )

        # Prepare generation parameters
        gen_params = {
            "num_inference_steps": params.get("steps", settings.default_steps),
            "guidance_scale": params.get("cfg_scale", settings.default_cfg_scale),
        }

        # Use embeds if weights are present, otherwise use text prompts
        if prompt_embeds is not None:
            gen_params["prompt_embeds"] = prompt_embeds
            if negative_prompt_embeds is not None:
                gen_params["negative_prompt_embeds"] = negative_prompt_embeds
            # Add pooled embeds for SDXL
            if is_sdxl:
                if pooled_prompt_embeds is not None:
                    gen_params["pooled_prompt_embeds"] = pooled_prompt_embeds
                if negative_pooled_prompt_embeds is not None:
                    gen_params["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
        else:
            gen_params["prompt"] = params["prompt"]
            gen_params["negative_prompt"] = params.get("negative_prompt", "")

        # Add size parameters only if not SDXL (SDXL has different parameter names)
        if not is_sdxl:
            gen_params["width"] = params.get("width", settings.default_width)
            gen_params["height"] = params.get("height", settings.default_height)
        else:
            # SDXL uses different size parameters
            gen_params["width"] = params.get("width", 1024)
            gen_params["height"] = params.get("height", 1024)

        # Create generator and get actual seed
        seed = params.get("seed", -1)
        if seed < 0:
            # Generate random seed
            import random
            actual_seed = random.randint(0, 2**32 - 1)
        else:
            actual_seed = seed

        gen_params["generator"] = torch.Generator(device=self.device).manual_seed(actual_seed)

        # Add progress callback if provided
        if progress_callback:
            gen_params["callback"] = progress_callback
            gen_params["callback_steps"] = 1

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

        return image, actual_seed

    def generate_img2img(self, params: Dict[str, Any], init_image: Image.Image, progress_callback=None) -> tuple[Image.Image, int]:
        """Generate image from image

        Returns:
            tuple: (image, actual_seed)
        """
        # If img2img pipeline is not loaded, create it from txt2img pipeline
        if not self.img2img_pipeline:
            if not self.txt2img_pipeline:
                raise RuntimeError("No model loaded. Please load a model first.")

            print("Creating img2img pipeline from txt2img pipeline...")
            # Check if SDXL
            is_sdxl = isinstance(self.txt2img_pipeline, StableDiffusionXLPipeline)

            # Create img2img pipeline from txt2img components
            if is_sdxl:
                self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline(**self.txt2img_pipeline.components)
            else:
                self.img2img_pipeline = StableDiffusionImg2ImgPipeline(**self.txt2img_pipeline.components)

            self.img2img_pipeline = self.img2img_pipeline.to(self.device)
            print("img2img pipeline created successfully")

        # Apply extensions before generation
        for ext in self.extensions:
            if ext.enabled:
                params = ext.process_before_generation(self.img2img_pipeline, params)

        # Set sampler and schedule type if specified
        sampler = params.get("sampler", "euler")
        schedule_type = params.get("schedule_type", "uniform")
        if sampler:
            try:
                self.img2img_pipeline.scheduler = get_scheduler(self.img2img_pipeline, sampler, schedule_type)
            except Exception as e:
                print(f"Warning: Could not set sampler to {sampler} with schedule {schedule_type}: {e}")

        # Create generator and get actual seed
        seed = params.get("seed", -1)
        if seed < 0:
            # Generate random seed
            import random
            actual_seed = random.randint(0, 2**32 - 1)
        else:
            actual_seed = seed

        # Check if SDXL
        is_sdxl = isinstance(self.img2img_pipeline, StableDiffusionXLImg2ImgPipeline)

        # Get resize parameters
        target_width = params.get("width")
        target_height = params.get("height")
        resize_mode = params.get("resize_mode", "image")
        resampling_method = params.get("resampling_method", "lanczos")

        # Resize input image if width/height are specified and mode is "image"
        if target_width and target_height and resize_mode == "image":
            if init_image.size != (target_width, target_height):
                print(f"Resizing input image from {init_image.size} to {target_width}x{target_height} using {resampling_method}")

                # Map resampling method name to PIL constant
                resampling_map = {
                    "lanczos": Image.Resampling.LANCZOS,
                    "bicubic": Image.Resampling.BICUBIC,
                    "bilinear": Image.Resampling.BILINEAR,
                    "nearest": Image.Resampling.NEAREST,
                }
                resampling = resampling_map.get(resampling_method, Image.Resampling.LANCZOS)

                init_image = init_image.resize((target_width, target_height), resampling)

        # Encode prompts with weights if emphasis syntax is present
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt_with_weights(
            params["prompt"],
            params.get("negative_prompt", ""),
            pipeline=self.img2img_pipeline
        )

        # Handle latent resize mode by encoding, resizing latent, then decoding
        if resize_mode == "latent" and target_width and target_height:
            if init_image.size != (target_width, target_height):
                print(f"Using latent resize mode: {init_image.size} -> {target_width}x{target_height} with {resampling_method}")

                # Encode image to latent space
                import torch.nn.functional as F

                # Prepare image for VAE encoding
                image_tensor = self.img2img_pipeline.image_processor.preprocess(init_image)
                image_tensor = image_tensor.to(device=self.device, dtype=self.img2img_pipeline.vae.dtype)

                # Encode to latent
                with torch.no_grad():
                    latent = self.img2img_pipeline.vae.encode(image_tensor).latent_dist.sample()
                    latent = latent * self.img2img_pipeline.vae.config.scaling_factor

                # Calculate target latent size (VAE downsamples by 8x)
                latent_height = target_height // 8
                latent_width = target_width // 8

                # Resize latent with selected resampling method
                if resampling_method == "lanczos":
                    # Use scipy for Lanczos (not available in PyTorch)
                    from scipy.ndimage import zoom
                    import numpy as np

                    # Convert to numpy for scipy processing
                    latent_np = latent.cpu().numpy()
                    batch, channels, h, w = latent_np.shape

                    # Calculate zoom factors
                    zoom_h = latent_height / h
                    zoom_w = latent_width / w

                    # Apply Lanczos resampling (order=3 for Lanczos-3)
                    resized_list = []
                    for b in range(batch):
                        resized_channels = []
                        for c in range(channels):
                            # zoom with Lanczos kernel (order=3)
                            resized_channel = zoom(latent_np[b, c], (zoom_h, zoom_w), order=3, mode='reflect')
                            resized_channels.append(resized_channel)
                        resized_list.append(np.stack(resized_channels))

                    resized_np = np.stack(resized_list)
                    resized_latent = torch.from_numpy(resized_np).to(device=latent.device, dtype=latent.dtype)
                else:
                    # Use PyTorch's built-in interpolation
                    torch_mode_map = {
                        "nearest": "nearest",
                        "bilinear": "bilinear",
                        "bicubic": "bicubic",
                    }
                    torch_mode = torch_mode_map.get(resampling_method, "bicubic")

                    resized_latent = F.interpolate(
                        latent,
                        size=(latent_height, latent_width),
                        mode=torch_mode,
                        align_corners=False if torch_mode != "nearest" else None
                    )

                # Decode latent back to image
                with torch.no_grad():
                    resized_latent = resized_latent / self.img2img_pipeline.vae.config.scaling_factor
                    decoded = self.img2img_pipeline.vae.decode(resized_latent).sample

                # Convert back to PIL Image
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
                decoded = (decoded * 255).round().astype("uint8")
                init_image = Image.fromarray(decoded[0])

        # Prepare generation parameters
        gen_params = {
            "image": init_image,
            "strength": params.get("denoising_strength", 0.75),
            "num_inference_steps": params.get("steps", settings.default_steps),
            "guidance_scale": params.get("cfg_scale", settings.default_cfg_scale),
            "generator": torch.Generator(device=self.device).manual_seed(actual_seed),
        }

        # Use embeds if weights are present, otherwise use text prompts
        if prompt_embeds is not None:
            gen_params["prompt_embeds"] = prompt_embeds
            if negative_prompt_embeds is not None:
                gen_params["negative_prompt_embeds"] = negative_prompt_embeds
            # Add pooled embeds for SDXL
            if is_sdxl:
                if pooled_prompt_embeds is not None:
                    gen_params["pooled_prompt_embeds"] = pooled_prompt_embeds
                if negative_pooled_prompt_embeds is not None:
                    gen_params["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
        else:
            gen_params["prompt"] = params["prompt"]
            gen_params["negative_prompt"] = params.get("negative_prompt", "")

        # Add progress callback if provided
        if progress_callback:
            gen_params["callback"] = progress_callback
            gen_params["callback_steps"] = 1

        # Generate image
        try:
            result = self.img2img_pipeline(**gen_params)
            image = result.images[0]
        except Exception as e:
            print(f"Generation error: {e}")
            print(f"Parameters used: {gen_params}")
            raise

        # Apply extensions after generation
        for ext in self.extensions:
            if ext.enabled:
                image = ext.process_after_generation(image, params)

        return image, actual_seed

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
