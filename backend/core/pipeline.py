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
from core.schedulers import get_scheduler
from core.prompt_parser import parse_prompt_attention, get_weighted_prompt_embeds

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

        print(f"[DEBUG] Used pipeline encode_prompt, applied custom weights")

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

    def OLD_encode_prompt_with_weights_BACKUP(self, prompt: str, negative_prompt: str = "", pipeline=None):
        """OLD VERSION - KEPT FOR REFERENCE"""
        if False:  # Disabled
            if is_sdxl:
                # SDXL has two text encoders
                tokenizer = pipeline.tokenizer
                tokenizer_2 = pipeline.tokenizer_2
                text_encoder = pipeline.text_encoder
                text_encoder_2 = pipeline.text_encoder_2

                # Encode with text_encoder (CLIP ViT-L, 768-dim)
                negative_prompt_embeds_1 = get_weighted_prompt_embeds(
                    negative_prompt, tokenizer, text_encoder, device, dtype, return_pooled=False
                )

                # Encode with text_encoder_2 (OpenCLIP ViT-bigG, 1280-dim) + pooled
                negative_prompt_embeds_2, negative_pooled_prompt_embeds = get_weighted_prompt_embeds(
                    negative_prompt, tokenizer_2, text_encoder_2, device, dtype, return_pooled=True
                )

                # Concatenate embeddings from both encoders (768 + 1280 = 2048)
                negative_prompt_embeds = torch.cat([negative_prompt_embeds_1, negative_prompt_embeds_2], dim=-1)
            else:
                # SD1.5 - single text encoder
                tokenizer = pipeline.tokenizer
                text_encoder = pipeline.text_encoder
                negative_prompt_embeds = get_weighted_prompt_embeds(
                    negative_prompt, tokenizer, text_encoder, device, dtype, return_pooled=False
                )
                negative_pooled_prompt_embeds = None
        elif negative_prompt and not has_neg_emphasis:
            # No weights for negative prompt - use normal encoding
            if is_sdxl:
                tokenizer = pipeline.tokenizer
                tokenizer_2 = pipeline.tokenizer_2
                text_encoder = pipeline.text_encoder
                text_encoder_2 = pipeline.text_encoder_2

                # Encode with text_encoder (768-dim)
                text_inputs = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)

                with torch.no_grad():
                    encoder_output_1 = text_encoder(text_input_ids, return_dict=True)
                    negative_prompt_embeds_1 = encoder_output_1.last_hidden_state

                # Encode with text_encoder_2 (1280-dim) + pooled
                text_inputs_2 = tokenizer_2(
                    negative_prompt,
                    padding="max_length",
                    max_length=tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids_2 = text_inputs_2.input_ids.to(device)

                with torch.no_grad():
                    encoder_output_2 = text_encoder_2(text_input_ids_2, return_dict=True)
                    negative_prompt_embeds_2 = encoder_output_2.last_hidden_state

                    # Get pooled output - handle different encoder types
                    # CLIP ViT-L has 'pooler_output', OpenCLIP ViT-bigG has 'text_embeds'
                    if hasattr(encoder_output_2, 'pooler_output') and encoder_output_2.pooler_output is not None:
                        negative_pooled_prompt_embeds = encoder_output_2.pooler_output
                    elif hasattr(encoder_output_2, 'text_embeds') and encoder_output_2.text_embeds is not None:
                        negative_pooled_prompt_embeds = encoder_output_2.text_embeds
                    else:
                        raise ValueError(f"Cannot find pooled embeddings from text_encoder_2, type: {type(encoder_output_2)}")

                # Concatenate
                negative_prompt_embeds = torch.cat([negative_prompt_embeds_1, negative_prompt_embeds_2], dim=-1)
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(dtype=dtype)
            else:
                tokenizer = pipeline.tokenizer
                text_encoder = pipeline.text_encoder

                text_inputs = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)

                with torch.no_grad():
                    negative_prompt_embeds = text_encoder(text_input_ids, return_dict=False)[0]
                    negative_pooled_prompt_embeds = None

                negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype)
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

        # DEBUG: Compare with pipeline's native encode_prompt for weight=1.0 case
        # Use the PARSED text (without emphasis syntax) for fair comparison
        if prompt_embeds is not None and hasattr(self.txt2img_pipeline, 'encode_prompt'):
            try:
                # Parse to remove emphasis syntax for comparison
                from backend.core.prompt_parser import parse_prompt_attention
                parsed_prompt = parse_prompt_attention(params["prompt"])
                clean_prompt = "".join([text for text, _ in parsed_prompt])

                print(f"[DEBUG] Comparing: Original='{params['prompt']}', Cleaned='{clean_prompt}'")

                native_embeds = self.txt2img_pipeline.encode_prompt(
                    prompt=clean_prompt,  # Use cleaned prompt for fair comparison
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=params.get("negative_prompt", "")
                )
                print(f"[DEBUG] Native encode_prompt returned {len(native_embeds)} values")
                print(f"[DEBUG] Custom prompt_embeds: {prompt_embeds.shape} dtype={prompt_embeds.dtype}, Native: {native_embeds[0].shape} dtype={native_embeds[0].dtype}")
                print(f"[DEBUG] Custom mean: {prompt_embeds.mean():.6f}, Native mean: {native_embeds[0].mean():.6f}")
                print(f"[DEBUG] Custom std: {prompt_embeds.std():.6f}, Native std: {native_embeds[0].std():.6f}")
                print(f"[DEBUG] Max diff: {(prompt_embeds - native_embeds[0]).abs().max():.6f}")
                print(f"[DEBUG] Are prompt_embeds equal? {torch.allclose(prompt_embeds, native_embeds[0], atol=1e-3)}")
                if len(native_embeds) > 2:
                    print(f"[DEBUG] Custom pooled: {pooled_prompt_embeds.shape}, Native: {native_embeds[2].shape}")
                    print(f"[DEBUG] Custom pooled mean: {pooled_prompt_embeds.mean():.6f}, Native: {native_embeds[2].mean():.6f}")
                    print(f"[DEBUG] Pooled max diff: {(pooled_prompt_embeds - native_embeds[2]).abs().max():.6f}")
                    print(f"[DEBUG] Are pooled_embeds equal? {torch.allclose(pooled_prompt_embeds, native_embeds[2], atol=1e-3)}")
            except Exception as e:
                print(f"[DEBUG] Could not compare with native encode_prompt: {e}")

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
            print(f"[DEBUG] Generation parameters keys: {list(gen_params.keys())}")
            if "prompt_embeds" in gen_params:
                print(f"[DEBUG] Using custom embeddings")
                print(f"[DEBUG] prompt_embeds shape: {gen_params['prompt_embeds'].shape}")
                if "negative_prompt_embeds" in gen_params:
                    print(f"[DEBUG] negative_prompt_embeds shape: {gen_params['negative_prompt_embeds'].shape}")
                else:
                    print(f"[DEBUG] WARNING: negative_prompt_embeds is MISSING!")
                if "pooled_prompt_embeds" in gen_params:
                    print(f"[DEBUG] pooled_prompt_embeds shape: {gen_params['pooled_prompt_embeds'].shape}")
                else:
                    print(f"[DEBUG] WARNING: pooled_prompt_embeds is MISSING!")
                if "negative_pooled_prompt_embeds" in gen_params:
                    print(f"[DEBUG] negative_pooled_prompt_embeds shape: {gen_params['negative_pooled_prompt_embeds'].shape}")
                else:
                    print(f"[DEBUG] WARNING: negative_pooled_prompt_embeds is MISSING!")
            else:
                print(f"[DEBUG] Using text prompts: {gen_params.get('prompt', '')[:50]}")

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

        # Encode prompts with weights if emphasis syntax is present
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt_with_weights(
            params["prompt"],
            params.get("negative_prompt", ""),
            pipeline=self.img2img_pipeline
        )

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
