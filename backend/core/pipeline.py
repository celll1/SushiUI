from typing import Dict, Any, Optional, List, Callable
from PIL import Image
import torch
import json
import os
import gc
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel
)
from config.settings import settings
from extensions.base_extension import BaseExtension
from core.model_loader import ModelLoader, ModelSource
from core.prompt_processors import PromptEditingProcessor
from core.schedulers import get_scheduler
from core.custom_sampling import custom_sampling_loop, custom_img2img_sampling_loop, custom_inpaint_sampling_loop
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

        # Prompt chunking settings
        self.prompt_chunking_mode: str = "a1111"  # Options: a1111, sd_scripts, nobos
        self.max_prompt_chunks: int = 0  # 0 = unlimited, 1-4 = limit chunks

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
            # === Step 1: Complete cleanup of existing pipelines ===
            print("[Pipeline] Cleaning up existing pipelines and releasing resources...")

            # Get list of all existing pipelines
            pipelines_to_cleanup = [self.txt2img_pipeline, self.img2img_pipeline, self.inpaint_pipeline]

            # Keep track of already-freed components to avoid double-freeing
            freed_components = set()

            for pipeline in pipelines_to_cleanup:
                if pipeline is not None:
                    # Remove offload hooks if present
                    if hasattr(pipeline, '_all_hooks') and pipeline._all_hooks:
                        print(f"[Pipeline] Removing {len(pipeline._all_hooks)} hooks from pipeline")
                        pipeline._all_hooks.clear()
                    if hasattr(pipeline, 'remove_all_hooks'):
                        pipeline.remove_all_hooks()

                    # Move each component to CPU and free from CUDA memory
                    component_names = ['unet', 'text_encoder', 'text_encoder_2', 'vae']
                    for comp_name in component_names:
                        if hasattr(pipeline, comp_name):
                            comp = getattr(pipeline, comp_name)
                            if comp is not None and id(comp) not in freed_components:
                                # Move to CPU to free CUDA memory
                                if hasattr(comp, 'to'):
                                    comp.to('cpu')
                                # Delete the component
                                delattr(pipeline, comp_name)
                                freed_components.add(id(comp))
                                del comp

            # Delete pipeline references
            if self.txt2img_pipeline is not None:
                del self.txt2img_pipeline
                self.txt2img_pipeline = None
            if self.img2img_pipeline is not None:
                del self.img2img_pipeline
                self.img2img_pipeline = None
            if self.inpaint_pipeline is not None:
                del self.inpaint_pipeline
                self.inpaint_pipeline = None

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                # Synchronize to ensure all operations are complete
                torch.cuda.synchronize()

            print("[Pipeline] Cleanup complete, VRAM released")

            # === Step 2: Load new model ===
            # Always use fp16 (default in ModelLoader)
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

            # Load base pipeline
            print("[Pipeline] Loading new model...")
            base_pipeline = ModelLoader.load_model(
                source_type=source_type,
                source=source,
                device=self.device,
                torch_dtype=torch_dtype,
                **kwargs
            )

            # Log component devices after loading
            self._log_component_devices(base_pipeline, "After model loading")

            # Determine if SDXL
            is_sdxl = isinstance(base_pipeline, StableDiffusionXLPipeline)

            # === Step 3: Create all pipeline variants from base ===
            print("[Pipeline] Creating pipeline variants...")

            # Set txt2img pipeline
            self.txt2img_pipeline = base_pipeline

            # Create img2img pipeline
            if is_sdxl:
                self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline(**base_pipeline.components)
            else:
                self.img2img_pipeline = StableDiffusionImg2ImgPipeline(**base_pipeline.components)

            # Create inpaint pipeline
            if is_sdxl:
                self.inpaint_pipeline = StableDiffusionXLInpaintPipeline(**base_pipeline.components)
            else:
                self.inpaint_pipeline = StableDiffusionInpaintPipeline(**base_pipeline.components)

            print(f"[Pipeline] All pipelines created successfully on device: {self.device}")

            self.current_model = model_id

            # Detect v-prediction status
            is_v_prediction = False
            if hasattr(base_pipeline, 'scheduler') and hasattr(base_pipeline.scheduler, 'config'):
                is_v_prediction = base_pipeline.scheduler.config.get("prediction_type") == "v_prediction"

            # Calculate model hash for local files
            model_hash = ""
            if source_type in ["safetensors", "diffusers"] and os.path.exists(source):
                print(f"[Pipeline] Calculating model hash for: {source}")
                from utils import calculate_file_hash
                model_hash = calculate_file_hash(source)
                print(f"[Pipeline] Model hash: {model_hash[:16]}...")

            self.current_model_info = {
                "source_type": source_type,
                "source": source,
                "type": ModelLoader.detect_model_type(source) if source_type != "huggingface" else "unknown",
                "is_v_prediction": is_v_prediction,
                "model_hash": model_hash
            }

            # Save this model as the last loaded model
            self._save_last_model(source_type, source, pipeline_type)

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _setup_img2img_steps(self, requested_steps: int, denoising_strength: float, fix_steps: bool = None) -> tuple[int, int]:
        """Calculate proper steps for img2img/inpaint to ensure full denoising

        Args:
            requested_steps: The number of steps the user wants to perform
            denoising_strength: Denoising strength (0.0 to 1.0)
            fix_steps: Override for img2img_fix_steps setting (defaults to settings value)

        Returns:
            tuple: (total_steps, t_enc) where total_steps is the adjusted step count
                   and t_enc is the encoding timestep
        """
        # Use parameter if provided, otherwise fall back to settings
        if fix_steps is None:
            fix_steps = settings.img2img_fix_steps

        if fix_steps:
            # Adjust total steps so that actual denoising steps = requested_steps
            # This ensures proper denoising even with low denoising strength
            total_steps = int(requested_steps / min(denoising_strength, 0.999)) if denoising_strength > 0 else 0
            t_enc = requested_steps - 1
        else:
            # Standard behavior: steps * strength
            total_steps = requested_steps
            t_enc = int(min(denoising_strength, 0.999) * requested_steps)

        return total_steps, t_enc

    def _log_component_devices(self, pipeline, context: str):
        """Log the device placement of all pipeline components"""
        print(f"\n[Pipeline] Component devices - {context}:")

        # Check U-Net
        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
            try:
                unet_device = next(pipeline.unet.parameters()).device
                print(f"  U-Net: {unet_device}")
            except StopIteration:
                print(f"  U-Net: No parameters found (meta device?)")

        # Check Text Encoder
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
            try:
                te_device = next(pipeline.text_encoder.parameters()).device
                print(f"  Text Encoder: {te_device}")
            except StopIteration:
                print(f"  Text Encoder: No parameters found (meta device?)")

        # Check Text Encoder 2 (SDXL)
        if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
            try:
                te2_device = next(pipeline.text_encoder_2.parameters()).device
                print(f"  Text Encoder 2: {te2_device}")
            except StopIteration:
                print(f"  Text Encoder 2: No parameters found (meta device?)")

        # Check VAE
        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            try:
                vae_device = next(pipeline.vae.parameters()).device
                print(f"  VAE: {vae_device}")
            except StopIteration:
                print(f"  VAE: No parameters found (meta device?)")

        # Check for hooks
        if hasattr(pipeline, '_all_hooks'):
            print(f"  Offload hooks: {len(pipeline._all_hooks)} hooks registered")
        else:
            print(f"  Offload hooks: None")

        print()

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

    def _build_token_weights(self, clean_text: str, parsed_fragments, tokenizer, device, dtype):
        """Build per-token weight array from parsed emphasis fragments"""
        # Build token weight array
        token_weights = []
        current_text = ""
        previous_token_count = 0

        for text, weight in parsed_fragments:
            if not text:
                continue

            # Add this fragment to accumulated text
            current_text += text

            # Tokenize accumulated text
            current_tokens = tokenizer(
                current_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
            current_token_count = current_tokens.input_ids.shape[1]

            # Add weights for the NEW tokens
            num_new_tokens = current_token_count - previous_token_count
            token_weights.extend([weight] * num_new_tokens)

            previous_token_count = current_token_count

        # Convert to tensor
        if len(token_weights) == 0:
            return None

        return torch.tensor(token_weights, device=device, dtype=dtype)

    def _apply_controlnets(self, pipeline, controlnet_images, width, height, is_sdxl):
        """Apply ControlNets to the pipeline"""
        from core.controlnet_manager import controlnet_manager

        if not controlnet_images:
            return pipeline

        try:
            # Load ControlNet models
            controlnets = []
            control_images = []

            for cn_config in controlnet_images:
                # Load ControlNet model
                controlnet = controlnet_manager.load_controlnet(
                    cn_config["model_path"],
                    device=self.device,
                    dtype=pipeline.dtype if hasattr(pipeline, 'dtype') else torch.float16,
                    is_lllite=cn_config.get("is_lllite", False)
                )

                if controlnet is None:
                    print(f"Warning: Could not load ControlNet {cn_config['model_path']}")
                    continue

                # Apply layer weights if specified
                layer_weights = cn_config.get("layer_weights")
                print(f"[Pipeline] ControlNet config: model_path={cn_config.get('model_path')}, layer_weights={layer_weights}")
                if layer_weights:
                    print(f"[Pipeline] Applying layer weights to ControlNet: {layer_weights}")
                    controlnet_manager.apply_layer_weights(controlnet, layer_weights)
                else:
                    print(f"[Pipeline] No layer weights specified for this ControlNet")

                controlnets.append(controlnet)

                # Prepare control image
                control_image = controlnet_manager.prepare_controlnet_image(
                    cn_config["image"],
                    width,
                    height
                )
                control_images.append(control_image)

            if not controlnets:
                print("No ControlNets loaded, using original pipeline")
                return pipeline

            # Create ControlNet pipeline
            if is_sdxl:
                if len(controlnets) == 1:
                    cn_pipeline = StableDiffusionXLControlNetPipeline(
                        vae=pipeline.vae,
                        text_encoder=pipeline.text_encoder,
                        text_encoder_2=pipeline.text_encoder_2,
                        tokenizer=pipeline.tokenizer,
                        tokenizer_2=pipeline.tokenizer_2,
                        unet=pipeline.unet,
                        controlnet=controlnets[0],
                        scheduler=pipeline.scheduler,
                    )
                else:
                    # Multiple ControlNets
                    cn_pipeline = StableDiffusionXLControlNetPipeline(
                        vae=pipeline.vae,
                        text_encoder=pipeline.text_encoder,
                        text_encoder_2=pipeline.text_encoder_2,
                        tokenizer=pipeline.tokenizer,
                        tokenizer_2=pipeline.tokenizer_2,
                        unet=pipeline.unet,
                        controlnet=controlnets,  # Pass list for multi-controlnet
                        scheduler=pipeline.scheduler,
                    )
            else:
                if len(controlnets) == 1:
                    cn_pipeline = StableDiffusionControlNetPipeline(
                        vae=pipeline.vae,
                        text_encoder=pipeline.text_encoder,
                        tokenizer=pipeline.tokenizer,
                        unet=pipeline.unet,
                        controlnet=controlnets[0],
                        scheduler=pipeline.scheduler,
                        safety_checker=getattr(pipeline, 'safety_checker', None),
                        feature_extractor=getattr(pipeline, 'feature_extractor', None),
                    )
                else:
                    # Multiple ControlNets
                    cn_pipeline = StableDiffusionControlNetPipeline(
                        vae=pipeline.vae,
                        text_encoder=pipeline.text_encoder,
                        tokenizer=pipeline.tokenizer,
                        unet=pipeline.unet,
                        controlnet=controlnets,  # Pass list for multi-controlnet
                        scheduler=pipeline.scheduler,
                        safety_checker=getattr(pipeline, 'safety_checker', None),
                        feature_extractor=getattr(pipeline, 'feature_extractor', None),
                    )

            # Store control images for later use
            cn_pipeline.control_images = control_images
            cn_pipeline.controlnet_configs = controlnet_images

            print(f"ControlNet pipeline created with {len(controlnets)} ControlNet(s)")
            return cn_pipeline

        except Exception as e:
            print(f"Error applying ControlNets: {e}")
            import traceback
            traceback.print_exc()
            return pipeline

    def _encode_prompt_chunked(self, prompt: str, negative_prompt: str = "", pipeline=None):
        """
        Encode prompts with chunking support for long prompts (>75 tokens).
        Uses pipeline.encode_prompt() for each chunk to ensure correct encoding.

        Returns:
            For SD1.5: (prompt_embeds, negative_prompt_embeds, None, None)
            For SDXL: (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        """
        from .prompt_parser import parse_prompt_attention, apply_emphasis_to_embeds

        # Use provided pipeline or default to txt2img_pipeline
        if pipeline is None:
            pipeline = self.txt2img_pipeline

        if pipeline is None:
            return None, None, None, None

        # Check if SDXL by checking if text_encoder_2 exists (more reliable than isinstance for ControlNet pipelines)
        is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None

        device = self.device
        dtype = pipeline.dtype if hasattr(pipeline, 'dtype') else torch.float16

        # Parse prompts for emphasis syntax
        # Note: Escaped parentheses like \( and \) should not be counted as emphasis
        import re
        # Check for unescaped ( or [ (not preceded by \)
        has_pos_emphasis = bool(re.search(r'(?<!\\)[\(\[]', prompt))
        has_neg_emphasis = bool(re.search(r'(?<!\\)[\(\[]', negative_prompt))

        # Get clean prompts
        clean_prompt = prompt
        if has_pos_emphasis:
            parsed = parse_prompt_attention(prompt)
            clean_prompt = "".join([text for text, _ in parsed])

        clean_neg_prompt = negative_prompt
        if negative_prompt and has_neg_emphasis:
            parsed_neg = parse_prompt_attention(negative_prompt)
            clean_neg_prompt = "".join([text for text, _ in parsed_neg])

        # Tokenize to split into chunks
        tokenizer = pipeline.tokenizer_2 if is_sdxl else pipeline.tokenizer
        tokens = tokenizer(clean_prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]

        # Split tokens into 75-token chunks
        chunk_size = 75
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunks.append(chunk_tokens)

        # Limit chunks if max_chunks is set
        if self.max_prompt_chunks > 0 and len(chunks) > self.max_prompt_chunks:
            chunks = chunks[:self.max_prompt_chunks]

        # Encode each chunk using pipeline.encode_prompt
        chunk_embeds_list = []
        pooled_prompt_embeds = None

        for idx, chunk_tokens in enumerate(chunks):
            # Decode tokens back to text
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Encode using pipeline.encode_prompt
            embeds = pipeline.encode_prompt(
                prompt=chunk_text,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )

            # For SDXL, use pooled embeds from first chunk only
            if is_sdxl and idx == 0:
                pooled_prompt_embeds = embeds[2]

            chunk_embeds_list.append(embeds[0])

        # Concatenate chunk embeddings based on mode
        if self.prompt_chunking_mode == "a1111":
            # A1111 mode: concatenate all chunks
            prompt_embeds = torch.cat(chunk_embeds_list, dim=1)
        elif self.prompt_chunking_mode == "sd_scripts":
            # sd-scripts mode: strip BOS/EOS between chunks
            # First chunk: keep all, middle chunks: strip BOS/EOS, last chunk: keep all
            processed_chunks = []
            for idx, chunk_emb in enumerate(chunk_embeds_list):
                if len(chunk_embeds_list) == 1:
                    processed_chunks.append(chunk_emb)
                elif idx == 0:
                    # First chunk: remove EOS (last token before padding)
                    processed_chunks.append(chunk_emb[:, :-1, :])
                elif idx == len(chunk_embeds_list) - 1:
                    # Last chunk: remove BOS (first token)
                    processed_chunks.append(chunk_emb[:, 1:, :])
                else:
                    # Middle chunks: remove both BOS and EOS
                    processed_chunks.append(chunk_emb[:, 1:-1, :])
            prompt_embeds = torch.cat(processed_chunks, dim=1)
        else:  # nobos
            # NoBOS mode: strip all BOS/EOS tokens
            processed_chunks = []
            for chunk_emb in chunk_embeds_list:
                # Remove first (BOS) and last (EOS) tokens
                processed_chunks.append(chunk_emb[:, 1:-1, :])
            prompt_embeds = torch.cat(processed_chunks, dim=1)

        # Apply emphasis weights if present
        if has_pos_emphasis:
            prompt_embeds = apply_emphasis_to_embeds(
                prompt, prompt_embeds,
                tokenizer,
                device, dtype
            )

        # Encode negative prompt similarly
        if negative_prompt:
            neg_tokens = tokenizer(clean_neg_prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
            neg_chunks = []
            for i in range(0, len(neg_tokens), chunk_size):
                neg_chunk_tokens = neg_tokens[i:i + chunk_size]
                neg_chunks.append(neg_chunk_tokens)

            if self.max_prompt_chunks > 0 and len(neg_chunks) > self.max_prompt_chunks:
                neg_chunks = neg_chunks[:self.max_prompt_chunks]

            neg_chunk_embeds_list = []
            negative_pooled_prompt_embeds = None

            for idx, neg_chunk_tokens in enumerate(neg_chunks):
                neg_chunk_text = tokenizer.decode(neg_chunk_tokens, skip_special_tokens=True)

                neg_embeds = pipeline.encode_prompt(
                    prompt=neg_chunk_text,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False
                )

                if is_sdxl and idx == 0:
                    negative_pooled_prompt_embeds = neg_embeds[2]

                neg_chunk_embeds_list.append(neg_embeds[0])

            # Concatenate based on mode
            if self.prompt_chunking_mode == "a1111":
                negative_prompt_embeds = torch.cat(neg_chunk_embeds_list, dim=1)
            elif self.prompt_chunking_mode == "sd_scripts":
                processed_chunks = []
                for idx, chunk_emb in enumerate(neg_chunk_embeds_list):
                    if len(neg_chunk_embeds_list) == 1:
                        processed_chunks.append(chunk_emb)
                    elif idx == 0:
                        processed_chunks.append(chunk_emb[:, :-1, :])
                    elif idx == len(neg_chunk_embeds_list) - 1:
                        processed_chunks.append(chunk_emb[:, 1:, :])
                    else:
                        processed_chunks.append(chunk_emb[:, 1:-1, :])
                negative_prompt_embeds = torch.cat(processed_chunks, dim=1)
            else:  # nobos
                processed_chunks = []
                for chunk_emb in neg_chunk_embeds_list:
                    processed_chunks.append(chunk_emb[:, 1:-1, :])
                negative_prompt_embeds = torch.cat(processed_chunks, dim=1)

            # Apply emphasis weights
            if has_neg_emphasis:
                negative_prompt_embeds = apply_emphasis_to_embeds(
                    negative_prompt, negative_prompt_embeds,
                    tokenizer,
                    device, dtype
                )
        else:
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None

        # Ensure prompt_embeds and negative_prompt_embeds have the same shape
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.size(1) != negative_prompt_embeds.size(1):
                max_len = max(prompt_embeds.size(1), negative_prompt_embeds.size(1))

                if prompt_embeds.size(1) < max_len:
                    pad_size = max_len - prompt_embeds.size(1)
                    padding = torch.zeros(
                        (prompt_embeds.size(0), pad_size, prompt_embeds.size(2)),
                        device=device,
                        dtype=dtype
                    )
                    prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)

                if negative_prompt_embeds.size(1) < max_len:
                    pad_size = max_len - negative_prompt_embeds.size(1)
                    padding = torch.zeros(
                        (negative_prompt_embeds.size(0), pad_size, negative_prompt_embeds.size(2)),
                        device=device,
                        dtype=dtype
                    )
                    negative_prompt_embeds = torch.cat([negative_prompt_embeds, padding], dim=1)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _encode_prompt_with_weights(self, prompt: str, negative_prompt: str = "", pipeline=None):
        """
        Encode prompts with A1111-style emphasis weights and/or chunking.

        Returns:
            For SD1.5: (prompt_embeds, negative_prompt_embeds)
            For SDXL: (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        """
        # Use provided pipeline or default to txt2img_pipeline
        if pipeline is None:
            pipeline = self.txt2img_pipeline

        if pipeline is None:
            return None, None, None, None

        # Check if prompt or negative prompt contains emphasis syntax
        # Note: Escaped parentheses like \( and \) should not be counted as emphasis
        import re
        # Check for unescaped ( or [ (not preceded by \)
        has_pos_emphasis = bool(re.search(r'(?<!\\)[\(\[]', prompt))
        has_neg_emphasis = bool(re.search(r'(?<!\\)[\(\[]', negative_prompt))

        # Tokenize to check length
        tokenizer = pipeline.tokenizer if hasattr(pipeline, 'tokenizer') else None
        if tokenizer:
            from .prompt_parser import parse_prompt_attention

            # Get clean prompt for length check
            clean_prompt = prompt
            if has_pos_emphasis:
                parsed = parse_prompt_attention(prompt)
                clean_prompt = "".join([text for text, _ in parsed])

            prompt_tokens = tokenizer(clean_prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
            needs_chunking = len(prompt_tokens) > 75
        else:
            needs_chunking = False

        # Use chunked encoding for long prompts
        if needs_chunking:
            return self._encode_prompt_chunked(prompt, negative_prompt, pipeline)

        # For short prompts (<=75 tokens), use pipeline.encode_prompt for correct encoding
        # Then apply emphasis weights if needed
        device = self.device
        dtype = pipeline.dtype if hasattr(pipeline, 'dtype') else torch.float16
        # Check if SDXL by checking if text_encoder_2 exists (more reliable than isinstance for ControlNet pipelines)
        is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None

        # If no emphasis syntax, just encode normally
        if not has_pos_emphasis and not has_neg_emphasis:
            # Use pipeline's encode_prompt for correct embeddings
            base_embeds = pipeline.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )

            # Extract embeddings
            prompt_embeds = base_embeds[0]
            pooled_prompt_embeds = base_embeds[2] if len(base_embeds) > 2 and is_sdxl else None

            # Encode negative prompt
            if negative_prompt:
                neg_embeds = pipeline.encode_prompt(
                    prompt=negative_prompt,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False
                )

                negative_prompt_embeds = neg_embeds[0]
                negative_pooled_prompt_embeds = neg_embeds[2] if len(neg_embeds) > 2 and is_sdxl else None
            else:
                negative_prompt_embeds = None
                negative_pooled_prompt_embeds = None

            return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

        # Has emphasis but fits in single chunk - use pipeline.encode_prompt then apply weights
        from .prompt_parser import parse_prompt_attention, apply_emphasis_to_embeds

        # Parse to get clean text
        parsed_pos = parse_prompt_attention(prompt) if has_pos_emphasis else [(prompt, 1.0)]
        clean_prompt = "".join([text for text, _ in parsed_pos])

        # Use pipeline's encode_prompt for correct embeddings
        base_embeds = pipeline.encode_prompt(
            prompt=clean_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False
        )

        # Extract embeddings
        prompt_embeds = base_embeds[0]
        pooled_prompt_embeds = base_embeds[2] if len(base_embeds) > 2 and is_sdxl else None

        # Apply emphasis weights
        if has_pos_emphasis:
            prompt_embeds = apply_emphasis_to_embeds(
                prompt, prompt_embeds,
                pipeline.tokenizer_2 if is_sdxl else pipeline.tokenizer,
                device, dtype
            )

        # Encode negative prompt
        if negative_prompt:
            parsed_neg = parse_prompt_attention(negative_prompt) if has_neg_emphasis else [(negative_prompt, 1.0)]
            clean_neg_prompt = "".join([text for text, _ in parsed_neg])

            neg_embeds = pipeline.encode_prompt(
                prompt=clean_neg_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )

            negative_prompt_embeds = neg_embeds[0]
            negative_pooled_prompt_embeds = neg_embeds[2] if len(neg_embeds) > 2 and is_sdxl else None

            if has_neg_emphasis:
                negative_prompt_embeds = apply_emphasis_to_embeds(
                    negative_prompt, negative_prompt_embeds,
                    pipeline.tokenizer_2 if is_sdxl else pipeline.tokenizer,
                    device, dtype
                )
        else:
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def generate_txt2img(self, params: Dict[str, Any], progress_callback=None, step_callback=None) -> tuple[Image.Image, int]:
        """Generate image from text

        Args:
            params: Generation parameters
            progress_callback: Legacy callback for progress (step, timestep, latents)
            step_callback: New style callback for step-based control (pipe, step, timestep, callback_kwargs)

        Returns:
            tuple: (image, actual_seed)
        """
        if not self.txt2img_pipeline:
            raise RuntimeError("txt2img pipeline not loaded. Please load a model first.")

        # Log component devices before generation
        self._log_component_devices(self.txt2img_pipeline, "Before txt2img generation")

        # Debug: Check ControlNet presence
        print(f"[Pipeline] Before extensions - controlnet_images in params: {'controlnet_images' in params}, value: {bool(params.get('controlnet_images'))}")

        # Apply extensions before generation
        for ext in self.extensions:
            if ext.enabled:
                params = ext.process_before_generation(self.txt2img_pipeline, params)

        # Debug: Check ControlNet presence after extensions
        print(f"[Pipeline] After extensions - controlnet_images in params: {'controlnet_images' in params}, value: {bool(params.get('controlnet_images'))}")

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

        # Check for prompt editing syntax
        prompt_processor = None
        has_prompt_editing = '[' in params["prompt"] and ':' in params["prompt"] and ']' in params["prompt"]

        if has_prompt_editing:
            print("[PromptEditing] Detected prompt editing syntax")
            prompt_processor = PromptEditingProcessor()
            num_steps = params.get("steps", settings.default_steps)
            prompt_processor.parse(params["prompt"], num_steps)

            # Use the initial (cleaned) prompt for encoding
            initial_prompt = prompt_processor.current_prompt
        else:
            initial_prompt = params["prompt"]

        # Encode prompts with weights if emphasis syntax is present
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt_with_weights(
            initial_prompt,
            params.get("negative_prompt", ""),
            pipeline=self.txt2img_pipeline
        )

        # Log embedding shapes for debugging
        if prompt_embeds is not None:
            print(f"Prompt embeddings shape: {prompt_embeds.shape}")
        if negative_prompt_embeds is not None:
            print(f"Negative prompt embeddings shape: {negative_prompt_embeds.shape}")
        if pooled_prompt_embeds is not None:
            print(f"Pooled prompt embeddings shape: {pooled_prompt_embeds.shape}")
        if negative_pooled_prompt_embeds is not None:
            print(f"Negative pooled prompt embeddings shape: {negative_pooled_prompt_embeds.shape}")

        # Handle ControlNet if specified
        controlnet_images = params.get("controlnet_images", [])
        pipeline_to_use = self.txt2img_pipeline

        if controlnet_images:
            print(f"Applying {len(controlnet_images)} ControlNet(s)")
            pipeline_to_use = self._apply_controlnets(
                self.txt2img_pipeline,
                controlnet_images,
                params.get("width", settings.default_width),
                params.get("height", settings.default_height),
                is_sdxl
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

        generator = torch.Generator(device=self.device).manual_seed(actual_seed)

        # Create ancestral generator for stochastic samplers
        ancestral_seed = params.get("ancestral_seed", -1)
        if ancestral_seed == -1:
            # Use main seed for ancestral sampling (default behavior)
            ancestral_generator = None  # Will use generator in custom_sampling_loop
        else:
            # Use separate seed for ancestral sampling
            ancestral_generator = torch.Generator(device=self.device).manual_seed(ancestral_seed)
            print(f"[Pipeline] Using separate ancestral seed: {ancestral_seed}")

        # Add ControlNet images if using ControlNet pipeline
        if hasattr(pipeline_to_use, 'control_images'):
            gen_params["image"] = pipeline_to_use.control_images

            # Add controlnet_conditioning_scale for strength control
            controlnet_scales = [cn["strength"] for cn in pipeline_to_use.controlnet_configs]
            if len(controlnet_scales) == 1:
                gen_params["controlnet_conditioning_scale"] = controlnet_scales[0]
            else:
                gen_params["controlnet_conditioning_scale"] = controlnet_scales

            # Add control_guidance_start and control_guidance_end for step range control
            # Convert from 0-1000 range to 0.0-1.0 fraction
            total_steps = params.get("steps", 20)
            guidance_starts = [cn.get("start_step", 0) / 1000.0 for cn in pipeline_to_use.controlnet_configs]
            guidance_ends = [cn.get("end_step", 1000) / 1000.0 for cn in pipeline_to_use.controlnet_configs]

            if len(guidance_starts) == 1:
                gen_params["control_guidance_start"] = guidance_starts[0]
                gen_params["control_guidance_end"] = guidance_ends[0]
            else:
                gen_params["control_guidance_start"] = guidance_starts
                gen_params["control_guidance_end"] = guidance_ends

            print(f"[Pipeline] ControlNet guidance: start={guidance_starts}, end={guidance_ends}")

        # Add progress callback if provided
        if progress_callback:
            gen_params["callback"] = progress_callback
            gen_params["callback_steps"] = 1

        # Create combined step callback for prompt editing and LoRA step range
        if prompt_processor or step_callback:
            # Store embeds cache for prompt editing
            embeds_cache = {}

            def combined_step_callback(pipe, step_index, timestep, callback_kwargs):
                # Handle prompt editing
                if prompt_processor:
                    new_prompt = prompt_processor.get_prompt_at_step(step_index, params.get("steps", settings.default_steps))

                    if new_prompt is not None:
                        print(f"[PromptEditing] Step {step_index}: Re-encoding prompt")

                        # Check if we've already encoded this prompt
                        if new_prompt not in embeds_cache:
                            # Re-encode the new prompt
                            new_embeds, new_neg_embeds, new_pooled, new_neg_pooled = self._encode_prompt_with_weights(
                                new_prompt,
                                params.get("negative_prompt", ""),
                                pipeline=self.txt2img_pipeline
                            )
                            embeds_cache[new_prompt] = (new_embeds, new_neg_embeds, new_pooled, new_neg_pooled)
                        else:
                            new_embeds, new_neg_embeds, new_pooled, new_neg_pooled = embeds_cache[new_prompt]

                        # Update the embeddings in callback_kwargs
                        if 'prompt_embeds' in callback_kwargs:
                            callback_kwargs['prompt_embeds'] = new_embeds
                        if 'negative_prompt_embeds' in callback_kwargs:
                            callback_kwargs['negative_prompt_embeds'] = new_neg_embeds
                        if new_pooled is not None and 'pooled_prompt_embeds' in callback_kwargs:
                            callback_kwargs['pooled_prompt_embeds'] = new_pooled
                        if new_neg_pooled is not None and 'negative_pooled_prompt_embeds' in callback_kwargs:
                            callback_kwargs['negative_pooled_prompt_embeds'] = new_neg_pooled

                # Handle LoRA step range callback
                if step_callback:
                    callback_kwargs = step_callback(pipe, step_index, timestep, callback_kwargs)

                return callback_kwargs

            gen_params["callback_on_step_end"] = combined_step_callback

        # Generate image
        try:
            # Always use custom sampling loop for consistent behavior
            print("[Pipeline] Using custom sampling loop")

            # Prepare prompt embeddings callback for prompt editing
            prompt_embeds_callback_fn = None
            if prompt_processor:
                embeds_cache = {}

                def prompt_embeds_callback_fn(step_index):
                    new_prompt = prompt_processor.get_prompt_at_step(step_index, params.get("steps", settings.default_steps))
                    if new_prompt is not None:
                        if new_prompt not in embeds_cache:
                            new_embeds, new_neg_embeds, new_pooled, new_neg_pooled = self._encode_prompt_with_weights(
                                new_prompt,
                                params.get("negative_prompt", ""),
                                pipeline=pipeline_to_use
                            )
                            embeds_cache[new_prompt] = (new_embeds, new_neg_embeds, new_pooled, new_neg_pooled)
                        return embeds_cache[new_prompt]
                    return None

            # Prepare ControlNet parameters
            controlnet_kwargs = {}
            print(f"[Pipeline] ControlNet check: controlnet_images={bool(controlnet_images)}, has_control_images={hasattr(pipeline_to_use, 'control_images')}, pipeline_type={type(pipeline_to_use).__name__}")
            if controlnet_images and hasattr(pipeline_to_use, 'control_images'):
                print(f"[Pipeline] Preparing ControlNet kwargs with {len(pipeline_to_use.control_images)} control images")
                controlnet_kwargs['controlnet_images'] = pipeline_to_use.control_images
                controlnet_scales = [cn["strength"] for cn in pipeline_to_use.controlnet_configs]
                controlnet_kwargs['controlnet_conditioning_scale'] = controlnet_scales if len(controlnet_scales) > 1 else controlnet_scales[0]

                total_steps = params.get("steps", settings.default_steps)
                guidance_starts = [cn.get("start_step", 0) / 1000.0 for cn in pipeline_to_use.controlnet_configs]
                guidance_ends = [cn.get("end_step", 1000) / 1000.0 for cn in pipeline_to_use.controlnet_configs]
                controlnet_kwargs['control_guidance_start'] = guidance_starts if len(guidance_starts) > 1 else guidance_starts[0]
                controlnet_kwargs['control_guidance_end'] = guidance_ends if len(guidance_ends) > 1 else guidance_ends[0]
                print(f"[Pipeline] ControlNet kwargs prepared: scales={controlnet_kwargs['controlnet_conditioning_scale']}, start={controlnet_kwargs['control_guidance_start']}, end={controlnet_kwargs['control_guidance_end']}")
            else:
                if controlnet_images:
                    print(f"[Pipeline] WARNING: ControlNet images specified but pipeline_to_use doesn't have control_images attribute")

            # Call custom sampling loop
            image = custom_sampling_loop(
                pipeline=pipeline_to_use,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=params.get("steps", settings.default_steps),
                guidance_scale=params.get("cfg_scale", settings.default_cfg_scale),
                width=params.get("width", 1024 if is_sdxl else settings.default_width),
                height=params.get("height", 1024 if is_sdxl else settings.default_height),
                generator=generator,
                ancestral_generator=ancestral_generator,
                latents=None,
                prompt_embeds_callback=prompt_embeds_callback_fn,
                progress_callback=progress_callback,
                step_callback=step_callback,
                **controlnet_kwargs,
            )

        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Clear intermediate tensors
            if hasattr(self, 'device') and self.device == "cuda":
                torch.cuda.empty_cache()

        # Apply extensions after generation
        for ext in self.extensions:
            if ext.enabled:
                image = ext.process_after_generation(image, params)

        return image, actual_seed

    def generate_img2img(self, params: Dict[str, Any], init_image: Image.Image, progress_callback=None, step_callback=None) -> tuple[Image.Image, int]:
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

        # Check for prompt editing syntax
        prompt_processor = None
        has_prompt_editing = '[' in params["prompt"] and ':' in params["prompt"] and ']' in params["prompt"]

        if has_prompt_editing:
            print("[PromptEditing] Detected prompt editing syntax in img2img")
            prompt_processor = PromptEditingProcessor()
            num_steps = params.get("steps", settings.default_steps)
            prompt_processor.parse(params["prompt"], num_steps)
            initial_prompt = prompt_processor.current_prompt
        else:
            initial_prompt = params["prompt"]

        # Handle ControlNet if specified
        controlnet_images = params.get("controlnet_images", [])
        pipeline_to_use = self.img2img_pipeline

        if controlnet_images:
            print(f"Applying {len(controlnet_images)} ControlNet(s) to img2img")
            pipeline_to_use = self._apply_controlnets(
                self.img2img_pipeline,
                controlnet_images,
                target_width or settings.default_width,
                target_height or settings.default_height,
                is_sdxl
            )

        # Encode prompts with weights if emphasis syntax is present
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt_with_weights(
            initial_prompt,
            params.get("negative_prompt", ""),
            pipeline=pipeline_to_use
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
                    # scipy doesn't support float16, so convert to float32
                    original_dtype = latent.dtype
                    latent_np = latent.cpu().float().numpy()  # Convert to float32
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
                    # Convert back to original dtype
                    resized_latent = torch.from_numpy(resized_np).to(device=latent.device, dtype=original_dtype)
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

        # Calculate proper steps for img2img
        requested_steps = params.get("steps", settings.default_steps)
        denoising_strength = params.get("denoising_strength", 0.75)
        fix_steps = params.get("img2img_fix_steps", True)
        total_steps, t_enc = self._setup_img2img_steps(requested_steps, denoising_strength, fix_steps)

        if fix_steps:
            print(f"[img2img] Do full steps enabled: {requested_steps} requested steps -> {total_steps} total steps (t_enc: {t_enc})")

        # Prepare generation parameters
        gen_params = {
            "image": init_image,
            "strength": denoising_strength,
            "num_inference_steps": total_steps,
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

        # Add step callback for LoRA step range if provided
        if step_callback:
            gen_params["callback_on_step_end"] = step_callback

        # Generate image using custom sampling loop
        try:
            print("[Pipeline] Using custom img2img sampling loop")

            # Prepare prompt embeddings callback for prompt editing
            prompt_embeds_callback_fn = None
            if prompt_processor:
                embeds_cache = {}

                def prompt_embeds_callback_fn(step_index):
                    new_prompt = prompt_processor.get_prompt_at_step(step_index, total_steps)
                    if new_prompt is not None:
                        if new_prompt not in embeds_cache:
                            new_embeds, new_neg_embeds, new_pooled, new_neg_pooled = self._encode_prompt_with_weights(
                                new_prompt,
                                params.get("negative_prompt", ""),
                                pipeline=pipeline_to_use
                            )
                            embeds_cache[new_prompt] = (new_embeds, new_neg_embeds, new_pooled, new_neg_pooled)
                        return embeds_cache[new_prompt]
                    return None

            # Prepare ControlNet parameters
            controlnet_kwargs = {}
            if controlnet_images and hasattr(pipeline_to_use, 'control_images'):
                controlnet_kwargs['controlnet_images'] = pipeline_to_use.control_images
                controlnet_scales = [cn["strength"] for cn in pipeline_to_use.controlnet_configs]
                controlnet_kwargs['controlnet_conditioning_scale'] = controlnet_scales if len(controlnet_scales) > 1 else controlnet_scales[0]

                guidance_starts = [cn.get("start_step", 0) / 1000.0 for cn in pipeline_to_use.controlnet_configs]
                guidance_ends = [cn.get("end_step", 1000) / 1000.0 for cn in pipeline_to_use.controlnet_configs]
                controlnet_kwargs['control_guidance_start'] = guidance_starts if len(guidance_starts) > 1 else guidance_starts[0]
                controlnet_kwargs['control_guidance_end'] = guidance_ends if len(guidance_ends) > 1 else guidance_ends[0]

            # Create ancestral generator for stochastic samplers
            ancestral_seed = params.get("ancestral_seed", -1)
            if ancestral_seed == -1:
                ancestral_generator = None
            else:
                ancestral_generator = torch.Generator(device=self.device).manual_seed(ancestral_seed)
                print(f"[Pipeline] Using separate ancestral seed: {ancestral_seed}")

            # Call custom img2img sampling loop
            image = custom_img2img_sampling_loop(
                pipeline=pipeline_to_use,
                init_image=init_image,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=total_steps,
                strength=denoising_strength,
                guidance_scale=params.get("cfg_scale", settings.default_cfg_scale),
                generator=torch.Generator(device=self.device).manual_seed(actual_seed),
                ancestral_generator=ancestral_generator,
                prompt_embeds_callback=prompt_embeds_callback_fn,
                progress_callback=progress_callback,
                step_callback=step_callback,
                **controlnet_kwargs,
            )

        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
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
        mask_image: Image.Image,
        progress_callback=None,
        step_callback=None
    ) -> tuple[Image.Image, int]:
        """Generate inpainted image

        Returns:
            tuple: (image, actual_seed)
        """
        # If inpaint pipeline is not loaded, create it from txt2img pipeline
        if not self.inpaint_pipeline:
            if not self.txt2img_pipeline:
                raise RuntimeError("No model loaded. Please load a model first.")

            # Check if current model is SDXL
            is_sdxl = isinstance(self.txt2img_pipeline, StableDiffusionXLPipeline)

            if is_sdxl:
                self.inpaint_pipeline = StableDiffusionXLInpaintPipeline(**self.txt2img_pipeline.components)
            else:
                self.inpaint_pipeline = StableDiffusionInpaintPipeline(**self.txt2img_pipeline.components)

            self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)

        # Apply extensions before generation
        for ext in self.extensions:
            if ext.enabled:
                params = ext.process_before_generation(self.inpaint_pipeline, params)

        # Set scheduler (sampler + schedule type)
        sampler_name = params.get("sampler", "euler")
        schedule_type = params.get("schedule_type", "uniform")

        self.inpaint_pipeline.scheduler = get_scheduler(
            pipeline=self.inpaint_pipeline,
            sampler=sampler_name,
            schedule_type=schedule_type
        )

        # Handle seed
        seed = params.get("seed", -1)
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Resize images if needed
        target_width = params.get("width", settings.default_width)
        target_height = params.get("height", settings.default_height)

        if init_image.size != (target_width, target_height):
            init_image = init_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        if mask_image.size != (target_width, target_height):
            mask_image = mask_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Calculate proper steps for inpaint
        requested_steps = params.get("steps", settings.default_steps)
        denoising_strength = params.get("denoising_strength", 0.75)
        fix_steps = params.get("img2img_fix_steps", True)
        total_steps, t_enc = self._setup_img2img_steps(requested_steps, denoising_strength, fix_steps)

        if fix_steps:
            print(f"[inpaint] Do full steps enabled: {requested_steps} requested steps -> {total_steps} total steps (t_enc: {t_enc})")

        # Check for prompt editing syntax
        prompt_processor = None
        has_prompt_editing = '[' in params["prompt"] and ':' in params["prompt"] and ']' in params["prompt"]

        if has_prompt_editing:
            print("[PromptEditing] Detected prompt editing syntax in inpaint")
            prompt_processor = PromptEditingProcessor()
            prompt_processor.parse(params["prompt"], total_steps)
            initial_prompt = prompt_processor.current_prompt
        else:
            initial_prompt = params["prompt"]

        # Determine if SDXL
        is_sdxl = isinstance(self.inpaint_pipeline, StableDiffusionXLInpaintPipeline)

        # Handle ControlNet if specified
        controlnet_images = params.get("controlnet_images", [])
        pipeline_to_use = self.inpaint_pipeline

        if controlnet_images:
            print(f"Applying {len(controlnet_images)} ControlNet(s) to inpaint")
            pipeline_to_use = self._apply_controlnets(
                self.inpaint_pipeline,
                controlnet_images,
                target_width,
                target_height,
                is_sdxl
            )

        # Encode initial prompt
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt_with_weights(
            initial_prompt,
            params.get("negative_prompt", ""),
            pipeline=pipeline_to_use
        )

        # Prepare callback for prompt editing
        prompt_embeds_callback_fn = None
        if prompt_processor:
            embeds_cache = {}
            def prompt_embeds_callback_fn(step_index):
                new_prompt = prompt_processor.get_prompt_at_step(step_index, total_steps)
                if new_prompt is not None:
                    if new_prompt not in embeds_cache:
                        print(f"[PromptEditing] Step {step_index}/{total_steps}: Switching to prompt: {new_prompt}")
                        new_embeds, new_neg_embeds, new_pooled, new_neg_pooled = self._encode_prompt_with_weights(
                            new_prompt,
                            params.get("negative_prompt", ""),
                            pipeline=pipeline_to_use
                        )
                        embeds_cache[new_prompt] = (new_embeds, new_neg_embeds, new_pooled, new_neg_pooled)
                    return embeds_cache[new_prompt]
                return None

        # Prepare ControlNet parameters
        controlnet_kwargs = {}
        if controlnet_images and hasattr(pipeline_to_use, 'control_images'):
            controlnet_kwargs['controlnet_images'] = pipeline_to_use.control_images
            controlnet_scales = [cn["strength"] for cn in pipeline_to_use.controlnet_configs]
            controlnet_kwargs['controlnet_conditioning_scale'] = controlnet_scales if len(controlnet_scales) > 1 else controlnet_scales[0]

            guidance_starts = [cn.get("start_step", 0) / 1000.0 for cn in pipeline_to_use.controlnet_configs]
            guidance_ends = [cn.get("end_step", 1000) / 1000.0 for cn in pipeline_to_use.controlnet_configs]
            controlnet_kwargs['control_guidance_start'] = guidance_starts if len(guidance_starts) > 1 else guidance_starts[0]
            controlnet_kwargs['control_guidance_end'] = guidance_ends if len(guidance_ends) > 1 else guidance_ends[0]

        # Create ancestral generator for stochastic samplers
        ancestral_seed = params.get("ancestral_seed", -1)
        if ancestral_seed == -1:
            ancestral_generator = None
        else:
            ancestral_generator = torch.Generator(device=self.device).manual_seed(ancestral_seed)
            print(f"[Pipeline] Using separate ancestral seed: {ancestral_seed}")

        # Use custom inpaint sampling loop
        image = custom_inpaint_sampling_loop(
            pipeline=pipeline_to_use,
            init_image=init_image,
            mask_image=mask_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=total_steps,
            strength=denoising_strength,
            guidance_scale=params.get("cfg_scale", settings.default_cfg_scale),
            generator=torch.Generator(device=self.device).manual_seed(seed),
            ancestral_generator=ancestral_generator,
            prompt_embeds_callback=prompt_embeds_callback_fn,
            progress_callback=progress_callback,
            step_callback=step_callback,
            **controlnet_kwargs,
        )

        # Apply extensions after generation
        for ext in self.extensions:
            if ext.enabled:
                image = ext.process_after_generation(image, params)

        return image, seed

# Global pipeline manager instance
pipeline_manager = DiffusionPipelineManager()
