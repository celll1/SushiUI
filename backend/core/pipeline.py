from typing import Dict, Any, Optional, List, Callable
from PIL import Image
import torch
import json
import os
import sys
import gc
import random
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
from core.prompts.processors import PromptEditingProcessor
from core.inference.schedulers import get_scheduler
from core.inference.custom_sampling import custom_sampling_loop, custom_img2img_sampling_loop, custom_inpaint_sampling_loop
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

        # Z-Image components (component-based, not pipeline-based)
        self.zimage_components: Optional[Dict[str, Any]] = None
        self.is_zimage_model: bool = False

        # FLUX.2 Klein components (MMDiT with Qwen3 text encoder)
        # Key features: 8 dual + 48 single stream blocks, 32ch VAE with BatchNorm, Flow Matching
        self.flux2_components: Optional[Dict[str, Any]] = None
        self.is_flux2_model: bool = False

        # Anima components (Cosmos-Predict2 DiT + Qwen3 + Qwen-Image VAE, Rectified Flow)
        self.anima_components: Optional[Dict[str, Any]] = None
        self.is_anima_model: bool = False

        # Lens components (Microsoft/Lens MMDiT + GPT-OSS + AutoencoderKLFlux2)
        self.lens_components: Optional[Dict[str, Any]] = None
        self.is_lens_model: bool = False

        # Ideogram 4 components (dual-branch single-stream DiT + Qwen3-VL + AutoencoderKLFlux2)
        # Two transformers (conditional + unconditional) for asymmetric CFG; Flow Matching.
        self.ideogram4_components: Optional[Dict[str, Any]] = None
        self.is_ideogram4_model: bool = False

        # MiniT2I components (pixel-space MM-JiT + FLAN-T5, NO VAE). Flow matching, x0 pred.
        self.minit2i_components: Optional[Dict[str, Any]] = None
        self.is_minit2i_model: bool = False

        # SigLIP2 Vision Encoder (optional, for SD/SDXL vision-conditioned generation)
        self.vision_encoder: Optional[Any] = None
        self._vision_encoder_path: Optional[str] = None

        # Prompt chunking settings
        self.prompt_chunking_mode: str = "a1111"  # Options: a1111, sd_scripts, nobos
        self.max_prompt_chunks: int = 0  # 0 = unlimited, 1-4 = limit chunks

        # Attention processor settings (dynamically loaded from localStorage via API)
        self.original_processors: Optional[dict] = None  # Store original processors
        self.current_attention_type: str = "normal"  # Track current attention type to avoid redundant switching

        # Cancellation flag
        self.cancel_requested = False

        # Model loading state
        self.is_loading = False
        self.load_error: Optional[str] = None

        # Note: Auto-load is now triggered by startup event in main.py

    @property
    def current_pipeline_kind(self) -> str:
        """Coarse identifier for the currently loaded pipeline.

        Used by the GPU coordinator to estimate peak VRAM for an incoming
        generation request.  Returns one of:
          "flux2", "zimage", "sdxl", "sd15", or "unknown".
        """
        if self.is_flux2_model:
            return "flux2"
        if self.is_zimage_model:
            return "zimage"
        if self.is_anima_model:
            return "anima"
        if self.is_lens_model:
            return "lens"
        if self.is_ideogram4_model:
            return "ideogram4"
        if self.is_minit2i_model:
            return "minit2i"
        # Detect SDXL vs SD1.5 by inspecting the loaded pipeline class
        pipe = self.txt2img_pipeline
        if pipe is not None:
            try:
                from diffusers import StableDiffusionXLPipeline
                if isinstance(pipe, StableDiffusionXLPipeline):
                    return "sdxl"
                return "sd15"
            except ImportError:
                pass
        return "unknown"

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

                    # Clear quantization cache
                    if hasattr(pipeline, '_quantized_unet_cache'):
                        print(f"[Pipeline] Clearing quantization cache ({len(pipeline._quantized_unet_cache)} cached models)")
                        pipeline._quantized_unet_cache.clear()
                        delattr(pipeline, '_quantized_unet_cache')
                    if hasattr(pipeline, '_original_unet'):
                        delattr(pipeline, '_original_unet')

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

            # Clean up Z-Image components
            if self.zimage_components is not None:
                print("[Pipeline] Cleaning up Z-Image components...")
                for comp_name, comp in self.zimage_components.items():
                    if comp is not None and hasattr(comp, 'to'):
                        comp.to('cpu')
                    del comp
                self.zimage_components = None
                self.is_zimage_model = False

            # Clean up FLUX.2 components
            if self.flux2_components is not None:
                print("[Pipeline] Cleaning up FLUX.2 components...")
                for comp_name, comp in self.flux2_components.items():
                    if comp is not None and hasattr(comp, 'to'):
                        comp.to('cpu')
                    del comp
                self.flux2_components = None
                self.is_flux2_model = False

            # Clean up Anima components
            if self.anima_components is not None:
                print("[Pipeline] Cleaning up Anima components...")
                for comp_name, comp in self.anima_components.items():
                    if comp is not None and hasattr(comp, 'to'):
                        try:
                            comp.to('cpu')
                        except Exception:
                            pass
                    del comp
                self.anima_components = None
                self.is_anima_model = False

            # Clean up Lens components
            if self.lens_components is not None:
                print("[Pipeline] Cleaning up Lens components...")
                for comp_name, comp in self.lens_components.items():
                    if comp is not None and hasattr(comp, 'to'):
                        try:
                            comp.to('cpu')
                        except Exception:
                            pass
                    del comp
                self.lens_components = None
                self.is_lens_model = False

            # Clean up Ideogram 4 components
            if self.ideogram4_components is not None:
                print("[Pipeline] Cleaning up Ideogram 4 components...")
                for comp_name, comp in self.ideogram4_components.items():
                    if comp is not None and hasattr(comp, 'to'):
                        try:
                            comp.to('cpu')
                        except Exception:
                            pass
                    del comp
                self.ideogram4_components = None
                self.is_ideogram4_model = False

            # Clean up MiniT2I components
            if self.minit2i_components is not None:
                print("[Pipeline] Cleaning up MiniT2I components...")
                for comp_name, comp in self.minit2i_components.items():
                    if comp is not None and hasattr(comp, 'to'):
                        try:
                            comp.to('cpu')
                        except Exception:
                            pass
                    del comp
                self.minit2i_components = None
                self.is_minit2i_model = False

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

            # Load base pipeline or Z-Image components
            print("[Pipeline] Loading new model...")
            model_result = ModelLoader.load_model(
                source_type=source_type,
                source=source,
                device=self.device,
                torch_dtype=torch_dtype,
                **kwargs
            )

            # Check if FLUX.2 (must check before Z-Image since both have "transformer" key)
            if isinstance(model_result, dict) and model_result.get("model_type") == "flux2":
                # FLUX.2 component-based model
                print("[Pipeline] FLUX.2 model detected (component-based dict returned)")
                self.flux2_components = model_result
                self.is_flux2_model = True
                self.is_zimage_model = False
                self.current_model = model_id
                self.current_attention_type = "normal"  # Reset on model load

                # Initialize VRAM optimization: Move all components to CPU
                print("[VRAM] Initializing sequential loading strategy for FLUX.2...")
                if self.flux2_components.get("text_encoder") is not None:
                    self.flux2_components["text_encoder"].to("cpu")
                if self.flux2_components.get("transformer") is not None:
                    self.flux2_components["transformer"].to("cpu")
                if self.flux2_components.get("vae") is not None:
                    self.flux2_components["vae"].to("cpu")
                torch.cuda.empty_cache()
                print("[VRAM] All FLUX.2 components moved to CPU. Will load to GPU as needed.")

                # FLUX.2 info
                model_type = "flux2"
                is_v_prediction = False  # FLUX.2 uses flow matching
                model_hash = ""
                if source_type in ["safetensors", "diffusers"] and os.path.exists(source):
                    from utils.hash_cache import get_cached_file_hash
                    model_hash = get_cached_file_hash(source)
                    print(f"[Pipeline] Model hash: {model_hash[:16]}...")

                self.current_model_info = {
                    "source_type": source_type,
                    "source": source,
                    "type": model_type,
                    "is_v_prediction": is_v_prediction,
                    "model_hash": model_hash,
                }

                # Save this model as the last loaded model
                self._save_last_model(source_type, source, pipeline_type)

                print("[Pipeline] FLUX.2 model loaded successfully")
                return

            # Check if Anima (must come before Z-Image since both have "transformer")
            if isinstance(model_result, dict) and model_result.get("type") == "anima":
                print("[Pipeline] Anima model detected (component-based dict returned)")
                self.anima_components = model_result
                self.is_anima_model = True
                self.is_zimage_model = False
                self.is_flux2_model = False
                self.current_model = model_id
                self.current_attention_type = "normal"

                # All components start on CPU; pipeline.py moves per stage.
                for comp_name in ("text_encoder", "transformer", "vae"):
                    comp = self.anima_components.get(comp_name)
                    if comp is not None and hasattr(comp, "to"):
                        try:
                            comp.to("cpu")
                        except Exception:
                            pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[VRAM] All Anima components moved to CPU. Will load to GPU as needed.")

                model_hash = ""
                if source_type in ["safetensors", "diffusers"] and os.path.exists(source):
                    try:
                        from utils.hash_cache import get_cached_file_hash
                        model_hash = get_cached_file_hash(source)
                        print(f"[Pipeline] Model hash: {model_hash[:16]}...")
                    except Exception as e:
                        print(f"[Pipeline] Hash compute skipped: {e}")

                self.current_model_info = {
                    "source_type": source_type,
                    "source": source,
                    "type": "anima",
                    "is_v_prediction": False,  # flow matching
                    "model_hash": model_hash,
                }
                self._save_last_model(source_type, source, pipeline_type)
                print("[Pipeline] Anima model loaded successfully")
                return

            # Check if Lens (microsoft/Lens MMDiT)
            if isinstance(model_result, dict) and model_result.get("type") == "lens":
                print("[Pipeline] Lens model detected (component-based dict returned)")
                self.lens_components = model_result
                self.is_lens_model = True
                self.is_anima_model = False
                self.is_zimage_model = False
                self.is_flux2_model = False
                self.current_model = model_id
                self.current_attention_type = "normal"

                for comp_name in ("transformer", "vae"):
                    comp = self.lens_components.get(comp_name)
                    if comp is not None and hasattr(comp, "to"):
                        try:
                            comp.to("cpu")
                        except Exception:
                            pass
                # Free text encoder mxfp4 CUDA buffers immediately after load.
                # Reloaded lazily before each generation's encoding stage.
                import gc as _gc
                self.lens_components["text_encoder"] = None
                _gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[VRAM] Lens transformer/VAE on CPU; text encoder freed (reloaded per generation).")

                model_hash = ""
                if source_type in ["safetensors", "diffusers"] and os.path.exists(source):
                    try:
                        from utils.hash_cache import get_cached_file_hash
                        model_hash = get_cached_file_hash(source)
                        print(f"[Pipeline] Model hash: {model_hash[:16]}...")
                    except Exception as e:
                        print(f"[Pipeline] Hash compute skipped: {e}")

                self.current_model_info = {
                    "source_type": source_type,
                    "source": source,
                    "type": "lens",
                    "is_v_prediction": False,
                    "model_hash": model_hash,
                }
                self._save_last_model(source_type, source, pipeline_type)
                print("[Pipeline] Lens model loaded successfully")
                return

            # Check if Ideogram 4 (dual-branch DiT). Must come before the generic
            # Z-Image check below, which matches any dict carrying a "transformer" key.
            if isinstance(model_result, dict) and model_result.get("type") == "ideogram4":
                print("[Pipeline] Ideogram 4 model detected (component-based dict returned)")
                self.ideogram4_components = model_result
                self.is_ideogram4_model = True
                self.is_lens_model = False
                self.is_anima_model = False
                self.is_zimage_model = False
                self.is_flux2_model = False
                self.current_model = model_id
                self.current_attention_type = "normal"

                # All components start on CPU; pipeline.py stages them to GPU per phase.
                for comp_name in ("transformer", "unconditional_transformer", "text_encoder", "vae"):
                    comp = self.ideogram4_components.get(comp_name)
                    if comp is not None and hasattr(comp, "to"):
                        try:
                            comp.to("cpu")
                        except Exception:
                            pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[VRAM] All Ideogram 4 components moved to CPU. Will load to GPU as needed.")

                model_hash = ""
                if source_type in ["safetensors", "diffusers"] and os.path.exists(source):
                    try:
                        from utils.hash_cache import get_cached_file_hash
                        model_hash = get_cached_file_hash(source)
                        print(f"[Pipeline] Model hash: {model_hash[:16]}...")
                    except Exception as e:
                        print(f"[Pipeline] Hash compute skipped: {e}")

                self.current_model_info = {
                    "source_type": source_type,
                    "source": source,
                    "type": "ideogram4",
                    "is_v_prediction": False,  # flow matching
                    "model_hash": model_hash,
                }
                self._save_last_model(source_type, source, pipeline_type)
                print("[Pipeline] Ideogram 4 model loaded successfully")
                return

            # Check if MiniT2I (pixel-space MM-JiT, no VAE). Before the generic
            # Z-Image check (which matches any dict carrying a "transformer" key).
            if isinstance(model_result, dict) and model_result.get("type") == "minit2i":
                print("[Pipeline] MiniT2I model detected (component-based dict returned)")
                self.minit2i_components = model_result
                self.is_minit2i_model = True
                self.is_ideogram4_model = False
                self.is_lens_model = False
                self.is_anima_model = False
                self.is_zimage_model = False
                self.is_flux2_model = False
                self.current_model = model_id
                self.current_attention_type = "normal"

                for comp_name in ("transformer", "text_encoder"):
                    comp = self.minit2i_components.get(comp_name)
                    if comp is not None and hasattr(comp, "to"):
                        try:
                            comp.to("cpu")
                        except Exception:
                            pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[VRAM] MiniT2I components on CPU. Will load to GPU as needed.")

                model_hash = ""
                if source_type in ["safetensors", "diffusers"] and os.path.exists(source):
                    try:
                        from utils.hash_cache import get_cached_file_hash
                        model_hash = get_cached_file_hash(source)
                    except Exception as e:
                        print(f"[Pipeline] Hash compute skipped: {e}")

                self.current_model_info = {
                    "source_type": source_type,
                    "source": source,
                    "type": "minit2i",
                    "is_v_prediction": False,  # flow matching, x0 prediction
                    "model_hash": model_hash,
                    "variant": self.minit2i_components.get("variant"),
                }
                self._save_last_model(source_type, source, pipeline_type)
                print("[Pipeline] MiniT2I model loaded successfully")
                return

            # Check if Z-Image
            if isinstance(model_result, dict) and "transformer" in model_result:
                # Z-Image component-based model
                print("[Pipeline] Z-Image model detected (component-based dict returned)")
                self.zimage_components = model_result
                self.is_zimage_model = True
                self.is_flux2_model = False
                self.is_anima_model = False
                self.current_model = model_id
                self.current_attention_type = "normal"  # Reset on model load

                # Initialize VRAM optimization: Move all components to CPU
                print("[VRAM] Initializing sequential loading strategy for Z-Image...")
                from core.vram_optimization import (
                    move_zimage_text_encoder_to_cpu,
                    move_zimage_transformer_to_cpu,
                    move_zimage_vae_to_cpu
                )
                move_zimage_text_encoder_to_cpu(self.zimage_components["text_encoder"])
                move_zimage_transformer_to_cpu(self.zimage_components["transformer"])
                move_zimage_vae_to_cpu(self.zimage_components["vae"])
                torch.cuda.empty_cache()
                print("[VRAM] All Z-Image components moved to CPU. Will load to GPU as needed.")

                # Z-Image info
                model_type = "zimage"
                is_v_prediction = False
                model_hash = ""
                if source_type in ["safetensors", "diffusers"] and os.path.exists(source):
                    from utils.hash_cache import get_cached_file_hash
                    model_hash = get_cached_file_hash(source)
                    print(f"[Pipeline] Model hash: {model_hash[:16]}...")

                # Get VAE type from loaded components (flux or sdxl)
                zimage_vae_type = model_result.get("vae_type", "flux")

                self.current_model_info = {
                    "source_type": source_type,
                    "source": source,
                    "type": model_type,
                    "is_v_prediction": is_v_prediction,
                    "model_hash": model_hash,
                    "vae_type": zimage_vae_type  # "flux" (16ch) or "sdxl" (4ch)
                }

                # Save this model as the last loaded model
                self._save_last_model(source_type, source, pipeline_type)

                print("[Pipeline] Z-Image model loaded successfully")
                return

            # Check if FLUX.2 (detected by "transformer" key with Flux2Transformer2DModel-specific keys)
            if isinstance(model_result, dict) and "transformer" in model_result and "scheduler" in model_result:
                # Check if it's FLUX.2 by looking at config or class name
                transformer = model_result.get("transformer")
                is_flux2 = (
                    transformer is not None and
                    hasattr(transformer, 'config') and
                    hasattr(transformer.config, 'num_single_layers')  # FLUX.2-specific config
                )

                if is_flux2:
                    print("[Pipeline] FLUX.2 Klein model detected (Flux2Transformer2DModel)")
                    self.flux2_components = model_result
                    self.is_flux2_model = True
                    self.is_zimage_model = False
                    self.current_model = model_id
                    self.current_attention_type = "normal"  # Reset on model load

                    # Initialize VRAM optimization: Move all components to CPU
                    print("[VRAM] Initializing sequential loading strategy for FLUX.2...")
                    if self.flux2_components.get("text_encoder") is not None:
                        self.flux2_components["text_encoder"].to("cpu")
                    if self.flux2_components.get("transformer") is not None:
                        self.flux2_components["transformer"].to("cpu")
                    if self.flux2_components.get("vae") is not None:
                        self.flux2_components["vae"].to("cpu")
                    torch.cuda.empty_cache()
                    print("[VRAM] All FLUX.2 components moved to CPU. Will load to GPU as needed.")

                    # FLUX.2 info
                    model_type = "flux2"
                    is_v_prediction = False  # FLUX.2 uses Flow Matching with velocity prediction
                    model_hash = ""
                    if source_type in ["safetensors", "diffusers"] and os.path.exists(source):
                        from utils.hash_cache import get_cached_file_hash
                        model_hash = get_cached_file_hash(source)
                        print(f"[Pipeline] Model hash: {model_hash[:16]}...")

                    self.current_model_info = {
                        "source_type": source_type,
                        "source": source,
                        "type": model_type,
                        "is_v_prediction": is_v_prediction,
                        "model_hash": model_hash
                    }

                    # Save this model as the last loaded model
                    self._save_last_model(source_type, source, pipeline_type)

                    print("[Pipeline] FLUX.2 Klein model loaded successfully")
                    return

            # Standard SD1.5/SDXL pipeline
            base_pipeline = model_result
            self.is_zimage_model = False
            self.is_flux2_model = False
            self.is_anima_model = False

            # Determine if SDXL
            is_sdxl = isinstance(base_pipeline, StableDiffusionXLPipeline)
            model_arch = "SDXL" if is_sdxl else "SD1.5"
            print(f"[Pipeline] Standard {model_arch} pipeline detected (NOT Z-Image)")

            # Log component devices after loading
            self._log_component_devices(base_pipeline, "After model loading")

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

            # Initialize VRAM optimization: Move all components to CPU except what's immediately needed
            print("[VRAM] Initializing sequential loading strategy...")
            from core.vram_optimization import move_text_encoders_to_cpu, move_unet_to_cpu, move_vae_to_cpu, log_device_status
            move_text_encoders_to_cpu(self.txt2img_pipeline)
            move_unet_to_cpu(self.txt2img_pipeline)
            move_vae_to_cpu(self.txt2img_pipeline)
            torch.cuda.empty_cache()
            log_device_status("Initial load complete, all components on CPU", self.txt2img_pipeline)

            self.current_model = model_id
            self.current_attention_type = "normal"  # Reset on model load

            # Detect v-prediction status
            is_v_prediction = False
            if hasattr(base_pipeline, 'scheduler') and hasattr(base_pipeline.scheduler, 'config'):
                is_v_prediction = base_pipeline.scheduler.config.get("prediction_type") == "v_prediction"

            # Calculate model hash for local files (with caching)
            model_hash = ""
            if source_type in ["safetensors", "diffusers"] and os.path.exists(source):
                from utils.hash_cache import get_cached_file_hash
                model_hash = get_cached_file_hash(source)
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

    def _setup_img2img_steps(self, requested_steps: int, denoising_strength: float, fix_steps: bool = None) -> tuple[int, int, int]:
        """Calculate proper steps for img2img/inpaint to ensure full denoising

        Args:
            requested_steps: The number of steps the user wants to perform
            denoising_strength: Denoising strength (0.0 to 1.0)
            fix_steps: Override for img2img_fix_steps setting (defaults to settings value)

        Returns:
            tuple: (total_steps, t_start, actual_steps) where:
                - total_steps: Total steps to set for scheduler
                - t_start: Starting timestep index
                - actual_steps: Actual number of denoising steps that will be performed
        """
        # Use parameter if provided, otherwise fall back to settings
        if fix_steps is None:
            fix_steps = settings.img2img_fix_steps

        if fix_steps:
            # Execute exactly requested_steps loops
            # Formula: total_steps - t_start = requested_steps
            total_steps = int(requested_steps / max(denoising_strength, 0.001))
            t_start = total_steps - requested_steps
            actual_steps = requested_steps
        else:
            # Standard behavior: steps * strength
            total_steps = requested_steps
            actual_steps = int(min(denoising_strength, 0.999) * requested_steps)
            t_start = total_steps - actual_steps

        return total_steps, t_start, actual_steps

    def _load_lora_zimage(self, lora_configs: List[Dict]):
        """Load LoRAs for Z-Image Transformer

        Args:
            lora_configs: List of LoRA configurations

        Note:
            Z-Image uses component-based architecture (not pipeline-based).
            LoRAs wrap original linear layers (forward-time addition, not weight merging).
            This allows LoRAs to be unloaded by restoring original modules.
            Based on training implementation in lora_trainer.py:674-708
        """
        if not lora_configs:
            return

        if not self.zimage_components:
            print("[Z-Image LoRA] WARNING: Z-Image components not loaded")
            return

        transformer = self.zimage_components["transformer"]

        # Store original modules for unloading (first time only)
        if not hasattr(self, '_zimage_lora_original_modules'):
            self._zimage_lora_original_modules = {}
            self._zimage_lora_wrapped_modules = set()  # Track which modules have LoRA

        # Use global lora_manager instance (has user-configured additional_dirs)
        from core.extensions.lora_manager import lora_manager

        print(f"[Z-Image LoRA] Loading {len(lora_configs)} LoRA(s)...")

        for i, lora_config in enumerate(lora_configs):
            lora_path = lora_config.get("path", "")
            lora_strength = lora_config.get("strength", 1.0)

            # Resolve path using LoRAManager (checks lora_dir + additional_dirs)
            resolved_path = lora_manager._resolve_lora_path(lora_path)

            if resolved_path is None:
                print(f"[Z-Image LoRA] WARNING: LoRA file not found: {lora_path}")
                print(f"[Z-Image LoRA]   Searched in: {lora_manager.lora_dir}")
                print(f"[Z-Image LoRA]   Additional dirs: {lora_manager.additional_dirs}")
                continue

            print(f"[Z-Image LoRA] Loading LoRA {i+1}/{len(lora_configs)}: {lora_path} (strength={lora_strength})")

            # Load LoRA weights
            from safetensors import safe_open

            try:
                with safe_open(str(resolved_path), framework="pt", device="cpu") as f:
                    lora_state_dict = {key: f.get_tensor(key) for key in f.keys()}

                print(f"[Z-Image LoRA] Loaded {len(lora_state_dict)} tensors from {lora_path}")

                # Apply LoRA to transformer attention modules
                # Target modules: to_q, to_k, to_v, to_out.0 in ZImageAttention
                applied_count = 0

                # Find all attention modules
                for attn_name, attn_module in transformer.named_modules():
                    if "ZImageAttention" not in attn_module.__class__.__name__:
                        continue

                    # Apply to to_q, to_k, to_v
                    for attr_name in ["to_q", "to_k", "to_v"]:
                        if hasattr(attn_module, attr_name):
                            original_linear = getattr(attn_module, attr_name)

                            if isinstance(original_linear, torch.nn.Linear):
                                # Build LoRA key prefix
                                lora_key_prefix = f"transformer.{attn_name}.{attr_name}"
                                lora_down_key = f"{lora_key_prefix}.lora_down.weight"
                                lora_up_key = f"{lora_key_prefix}.lora_up.weight"

                                # Check if LoRA weights exist for this module
                                if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                    lora_down_weight = lora_state_dict[lora_down_key]
                                    lora_up_weight = lora_state_dict[lora_up_key]

                                    # Load alpha if present
                                    lora_alpha_key = f"{lora_key_prefix}.alpha"
                                    lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                    # Wrap with LoRA layer
                                    module_key = f"{attn_name}.{attr_name}"
                                    wrapped_module = self._wrap_with_lora(
                                        attn_module,
                                        attr_name,
                                        original_linear,
                                        lora_down_weight,
                                        lora_up_weight,
                                        lora_strength,
                                        lora_alpha,
                                        module_key
                                    )
                                    if wrapped_module is not None:
                                        applied_count += 1

                    # Apply to to_out.0 (ModuleList)
                    if hasattr(attn_module, "to_out") and isinstance(attn_module.to_out, torch.nn.ModuleList):
                        if len(attn_module.to_out) > 0 and isinstance(attn_module.to_out[0], torch.nn.Linear):
                            original_linear = attn_module.to_out[0]

                            lora_key_prefix = f"transformer.{attn_name}.to_out.0"
                            lora_down_key = f"{lora_key_prefix}.lora_down.weight"
                            lora_up_key = f"{lora_key_prefix}.lora_up.weight"

                            if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                lora_down_weight = lora_state_dict[lora_down_key]
                                lora_up_weight = lora_state_dict[lora_up_key]

                                # Load alpha if present
                                lora_alpha_key = f"{lora_key_prefix}.alpha"
                                lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                # Wrap with LoRA layer (to_out is ModuleList, replace [0])
                                module_key = f"{attn_name}.to_out.0"
                                wrapped_module = self._wrap_with_lora(
                                    attn_module.to_out,
                                    0,  # ModuleList index
                                    original_linear,
                                    lora_down_weight,
                                    lora_up_weight,
                                    lora_strength,
                                    lora_alpha,
                                    module_key
                                )
                                if wrapped_module is not None:
                                    applied_count += 1

                print(f"[Z-Image LoRA] Applied LoRA to {applied_count} modules")

            except Exception as e:
                print(f"[Z-Image LoRA] ERROR: Failed to load LoRA {lora_path}: {e}")
                import traceback
                traceback.print_exc()

    def _wrap_with_lora(self, parent_module, attr_name, original_linear, lora_down_weight, lora_up_weight, strength, alpha, module_key):
        """Wrap a linear layer with LoRA

        Args:
            parent_module: Parent module containing the linear layer
            attr_name: Attribute name or index (for ModuleList)
            original_linear: Original linear layer
            lora_down_weight: LoRA down projection weight [rank, in_features]
            lora_up_weight: LoRA up projection weight [out_features, rank]
            strength: LoRA strength multiplier
            alpha: LoRA alpha parameter
            module_key: Unique key for this module (for tracking)

        Returns:
            Wrapped LoRA module or None if failed
        """
        # Import LoRALinearLayer from training adapters (model-agnostic wrapper class)
        from core.training.adapters.sd15_adapter import LoRALinearLayer
        import numpy as np

        # Get true original module (unwrap if it's already a LoRA wrapper)
        LoRALinearLayerClass = LoRALinearLayer  # Same class, just alias for clarity

        if isinstance(original_linear, LoRALinearLayerClass):
            # Already wrapped - extract the original module
            true_original = original_linear.original_module
            print(f"[Z-Image LoRA DEBUG] Detected existing LoRA wrapper, extracting original module")
        else:
            true_original = original_linear

        # Save original module (first time only)
        if module_key not in self._zimage_lora_original_modules:
            self._zimage_lora_original_modules[module_key] = true_original

        # Compute rank and alpha value
        rank = lora_down_weight.shape[0]
        alpha_value = alpha.item() if alpha is not None else rank

        # Create LoRA wrapper using the true original module
        # lora_name is required parameter, use module_key for identification
        lora_wrapper = LoRALinearLayer(
            true_original, rank=rank, alpha=alpha_value, lora_name=module_key
        )

        # Load pretrained LoRA weights
        device = true_original.weight.device
        dtype = true_original.weight.dtype

        with torch.no_grad():
            lora_wrapper.lora_down.weight.data = lora_down_weight.to(device=device, dtype=dtype)
            lora_wrapper.lora_up.weight.data = lora_up_weight.to(device=device, dtype=dtype)

        # Apply strength by adjusting scaling (override the default scale)
        lora_wrapper.scale = (alpha_value / rank) * strength

        # Replace in parent module
        if isinstance(attr_name, int):
            # ModuleList index
            parent_module[attr_name] = lora_wrapper
        else:
            # Attribute name
            setattr(parent_module, attr_name, lora_wrapper)

        # Track wrapped modules
        self._zimage_lora_wrapped_modules.add(module_key)

        print(f"[Z-Image LoRA DEBUG] Wrapped {module_key}: alpha={alpha_value:.1f}, rank={rank}, strength={strength:.2f}, scaling={lora_wrapper.scaling:.4f}")

        return lora_wrapper

    def _unload_lora_zimage(self):
        """Unload LoRAs from Z-Image Transformer

        Restores original linear layers by removing LoRA wrappers.
        """
        if not hasattr(self, '_zimage_lora_original_modules'):
            print("[Z-Image LoRA] No LoRAs loaded")
            return

        if not self.zimage_components:
            print("[Z-Image LoRA] WARNING: Z-Image components not loaded")
            return

        transformer = self.zimage_components["transformer"]
        unloaded_count = 0

        print(f"[Z-Image LoRA] Unloading LoRAs ({len(self._zimage_lora_wrapped_modules)} modules)...")

        # Restore original modules
        for attn_name, attn_module in transformer.named_modules():
            if "ZImageAttention" not in attn_module.__class__.__name__:
                continue

            # Restore to_q, to_k, to_v
            for attr_name in ["to_q", "to_k", "to_v"]:
                module_key = f"{attn_name}.{attr_name}"
                if module_key in self._zimage_lora_original_modules:
                    original_module = self._zimage_lora_original_modules[module_key]
                    setattr(attn_module, attr_name, original_module)
                    unloaded_count += 1

            # Restore to_out.0 (ModuleList)
            if hasattr(attn_module, "to_out") and isinstance(attn_module.to_out, torch.nn.ModuleList):
                module_key = f"{attn_name}.to_out.0"
                if module_key in self._zimage_lora_original_modules:
                    original_module = self._zimage_lora_original_modules[module_key]
                    attn_module.to_out[0] = original_module
                    unloaded_count += 1

        # Clear wrapped modules tracking (but keep original modules for future loads)
        self._zimage_lora_wrapped_modules.clear()

        print(f"[Z-Image LoRA] Unloaded {unloaded_count} LoRA modules")
        print(f"[Z-Image LoRA] Original modules preserved for future LoRA loads")

    def _load_lora_flux2(self, lora_configs: List[Dict]):
        """Load LoRAs for FLUX.2 Transformer

        Args:
            lora_configs: List of LoRA configurations

        Note:
            FLUX.2 uses component-based architecture (not pipeline-based).
            LoRAs wrap original linear layers (forward-time addition, not weight merging).
            This allows LoRAs to be unloaded by restoring original modules.
            Based on training implementation in flux2_adapter.py

            FLUX.2 has two block types:
            1. Dual stream blocks: Flux2Attention (to_q, to_k, to_v, to_out[0], add_q_proj, add_k_proj, add_v_proj, to_add_out)
            2. Single stream blocks: Flux2ParallelSelfAttention (to_qkv_mlp_proj, to_out)
        """
        if not lora_configs:
            return

        if not self.flux2_components:
            print("[FLUX.2 LoRA] WARNING: FLUX.2 components not loaded")
            return

        transformer = self.flux2_components["transformer"]

        # Store original modules for unloading (first time only)
        if not hasattr(self, '_flux2_lora_original_modules'):
            self._flux2_lora_original_modules = {}
            self._flux2_lora_wrapped_modules = set()

        # Use global lora_manager instance (has user-configured additional_dirs)
        from core.extensions.lora_manager import lora_manager

        print(f"[FLUX.2 LoRA] Loading {len(lora_configs)} LoRA(s)...")

        for i, lora_config in enumerate(lora_configs):
            lora_path = lora_config.get("path", "")
            lora_strength = lora_config.get("strength", 1.0)
            layer_weights = lora_config.get("unet_layer_weights", {})

            # Resolve path using LoRAManager
            resolved_path = lora_manager._resolve_lora_path(lora_path)

            if resolved_path is None:
                print(f"[FLUX.2 LoRA] WARNING: LoRA file not found: {lora_path}")
                print(f"[FLUX.2 LoRA]   Searched in: {lora_manager.lora_dir}")
                print(f"[FLUX.2 LoRA]   Additional dirs: {lora_manager.additional_dirs}")
                continue

            print(f"[FLUX.2 LoRA] Loading LoRA {i+1}/{len(lora_configs)}: {lora_path} (strength={lora_strength})")
            if layer_weights:
                print(f"[FLUX.2 LoRA] Layer weights: {layer_weights}")

            # Load LoRA weights
            from safetensors import safe_open

            try:
                with safe_open(str(resolved_path), framework="pt", device="cpu") as f:
                    lora_state_dict = {key: f.get_tensor(key) for key in f.keys()}

                print(f"[FLUX.2 LoRA] Loaded {len(lora_state_dict)} tensors from {lora_path}")

                # Apply LoRA to transformer modules
                applied_count = 0

                # Debug: Print first few LoRA keys
                lora_keys_sample = list(lora_state_dict.keys())[:5]
                print(f"[FLUX.2 LoRA] Sample LoRA keys: {lora_keys_sample}")

                # Debug: Print module class names found
                module_classes_found = set()
                for name, module in transformer.named_modules():
                    module_classes_found.add(module.__class__.__name__)
                print(f"[FLUX.2 LoRA] Module classes in transformer: {module_classes_found}")

                for name, module in transformer.named_modules():
                    # Flux2Attention (dual stream blocks)
                    if module.__class__.__name__ == "Flux2Attention":
                        # Get block name for layer-wise weight lookup
                        block_name = self._get_flux2_block_name(name)
                        block_weight = layer_weights.get(block_name, 1.0)
                        effective_strength = lora_strength * block_weight

                        # Standard QKV projections
                        for attr_name in ["to_q", "to_k", "to_v"]:
                            if hasattr(module, attr_name):
                                original_linear = getattr(module, attr_name)
                                if isinstance(original_linear, torch.nn.Linear):
                                    # Build LoRA key using training adapter's naming convention
                                    lora_name = f"lora_transformer_{name.replace('.', '_')}_{attr_name}"
                                    lora_down_key = f"{lora_name}.lora_down.weight"
                                    lora_up_key = f"{lora_name}.lora_up.weight"

                                    if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                        lora_down_weight = lora_state_dict[lora_down_key]
                                        lora_up_weight = lora_state_dict[lora_up_key]
                                        lora_alpha_key = f"{lora_name}.alpha"
                                        lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                        module_key = f"{name}.{attr_name}"
                                        wrapped = self._wrap_with_lora_flux2(
                                            module, attr_name, original_linear,
                                            lora_down_weight, lora_up_weight, effective_strength, lora_alpha, module_key
                                        )
                                        if wrapped:
                                            applied_count += 1

                        # to_out (ModuleList) - uses same effective_strength computed above
                        if hasattr(module, "to_out") and isinstance(module.to_out, torch.nn.ModuleList):
                            if len(module.to_out) > 0 and isinstance(module.to_out[0], torch.nn.Linear):
                                lora_name = f"lora_transformer_{name.replace('.', '_')}_to_out_0"
                                lora_down_key = f"{lora_name}.lora_down.weight"
                                lora_up_key = f"{lora_name}.lora_up.weight"

                                if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                    lora_down_weight = lora_state_dict[lora_down_key]
                                    lora_up_weight = lora_state_dict[lora_up_key]
                                    lora_alpha_key = f"{lora_name}.alpha"
                                    lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                    module_key = f"{name}.to_out.0"
                                    wrapped = self._wrap_with_lora_flux2(
                                        module.to_out, 0, module.to_out[0],
                                        lora_down_weight, lora_up_weight, effective_strength, lora_alpha, module_key
                                    )
                                    if wrapped:
                                        applied_count += 1

                        # Additional projections for encoder cross attention - uses same effective_strength
                        for attr_name in ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]:
                            if hasattr(module, attr_name):
                                original_linear = getattr(module, attr_name)
                                if isinstance(original_linear, torch.nn.Linear):
                                    lora_name = f"lora_transformer_{name.replace('.', '_')}_{attr_name}"
                                    lora_down_key = f"{lora_name}.lora_down.weight"
                                    lora_up_key = f"{lora_name}.lora_up.weight"

                                    if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                        lora_down_weight = lora_state_dict[lora_down_key]
                                        lora_up_weight = lora_state_dict[lora_up_key]
                                        lora_alpha_key = f"{lora_name}.alpha"
                                        lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                        module_key = f"{name}.{attr_name}"
                                        wrapped = self._wrap_with_lora_flux2(
                                            module, attr_name, original_linear,
                                            lora_down_weight, lora_up_weight, effective_strength, lora_alpha, module_key
                                        )
                                        if wrapped:
                                            applied_count += 1

                    # Flux2ParallelSelfAttention (single stream blocks)
                    elif module.__class__.__name__ == "Flux2ParallelSelfAttention":
                        # Get block name for layer-wise weight lookup
                        block_name = self._get_flux2_block_name(name)
                        block_weight = layer_weights.get(block_name, 1.0)
                        effective_strength = lora_strength * block_weight

                        # Fused QKV + MLP projection
                        if hasattr(module, "to_qkv_mlp_proj"):
                            original_linear = module.to_qkv_mlp_proj
                            if isinstance(original_linear, torch.nn.Linear):
                                lora_name = f"lora_transformer_{name.replace('.', '_')}_to_qkv_mlp_proj"
                                lora_down_key = f"{lora_name}.lora_down.weight"
                                lora_up_key = f"{lora_name}.lora_up.weight"

                                if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                    lora_down_weight = lora_state_dict[lora_down_key]
                                    lora_up_weight = lora_state_dict[lora_up_key]
                                    lora_alpha_key = f"{lora_name}.alpha"
                                    lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                    module_key = f"{name}.to_qkv_mlp_proj"
                                    wrapped = self._wrap_with_lora_flux2(
                                        module, "to_qkv_mlp_proj", original_linear,
                                        lora_down_weight, lora_up_weight, effective_strength, lora_alpha, module_key
                                    )
                                    if wrapped:
                                        applied_count += 1

                        # Output projection (fused attention + MLP) - uses same effective_strength
                        if hasattr(module, "to_out") and isinstance(module.to_out, torch.nn.Linear):
                            lora_name = f"lora_transformer_{name.replace('.', '_')}_to_out"
                            lora_down_key = f"{lora_name}.lora_down.weight"
                            lora_up_key = f"{lora_name}.lora_up.weight"

                            if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                lora_down_weight = lora_state_dict[lora_down_key]
                                lora_up_weight = lora_state_dict[lora_up_key]
                                lora_alpha_key = f"{lora_name}.alpha"
                                lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                module_key = f"{name}.to_out"
                                wrapped = self._wrap_with_lora_flux2(
                                    module, "to_out", module.to_out,
                                    lora_down_weight, lora_up_weight, effective_strength, lora_alpha, module_key
                                )
                                if wrapped:
                                    applied_count += 1

                    # Flux2FeedForward (dual stream blocks)
                    elif module.__class__.__name__ == "Flux2FeedForward":
                        # Get block name for layer-wise weight lookup
                        block_name = self._get_flux2_block_name(name)
                        block_weight = layer_weights.get(block_name, 1.0)
                        effective_strength = lora_strength * block_weight

                        for attr_name in ["linear_in", "linear_out"]:
                            if hasattr(module, attr_name):
                                original_linear = getattr(module, attr_name)
                                if isinstance(original_linear, torch.nn.Linear):
                                    lora_name = f"lora_transformer_{name.replace('.', '_')}_{attr_name}"
                                    lora_down_key = f"{lora_name}.lora_down.weight"
                                    lora_up_key = f"{lora_name}.lora_up.weight"

                                    if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                        lora_down_weight = lora_state_dict[lora_down_key]
                                        lora_up_weight = lora_state_dict[lora_up_key]
                                        lora_alpha_key = f"{lora_name}.alpha"
                                        lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                        module_key = f"{name}.{attr_name}"
                                        wrapped = self._wrap_with_lora_flux2(
                                            module, attr_name, original_linear,
                                            lora_down_weight, lora_up_weight, effective_strength, lora_alpha, module_key
                                        )
                                        if wrapped:
                                            applied_count += 1

                print(f"[FLUX.2 LoRA] Applied LoRA to {applied_count} modules")

            except Exception as e:
                print(f"[FLUX.2 LoRA] ERROR: Failed to load LoRA {lora_path}: {e}")
                import traceback
                traceback.print_exc()

    def _get_flux2_block_name(self, module_name: str) -> str:
        """Get the block name (DUAL{XX} or SING{XX}) from module name for layer-wise weight lookup

        Args:
            module_name: Module name like 'transformer_blocks.0.attn' or 'single_transformer_blocks.5.attn'

        Returns:
            Block name like 'DUAL00', 'SING05', or 'BASE' if no match
        """
        import re

        # Dual stream blocks: transformer_blocks.X.* (but not single_transformer_blocks)
        if 'transformer_blocks' in module_name and 'single_transformer_blocks' not in module_name:
            match = re.search(r'transformer_blocks\.(\d+)', module_name)
            if match:
                block_num = int(match.group(1))
                return f"DUAL{block_num:02d}"

        # Single stream blocks: single_transformer_blocks.X.*
        match = re.search(r'single_transformer_blocks\.(\d+)', module_name)
        if match:
            block_num = int(match.group(1))
            return f"SING{block_num:02d}"

        return "BASE"

    def _wrap_with_lora_flux2(self, parent_module, attr_name, original_linear, lora_down_weight, lora_up_weight, strength, alpha, module_key):
        """Wrap a linear layer with LoRA for FLUX.2

        Args:
            parent_module: Parent module containing the linear layer
            attr_name: Attribute name or index (for ModuleList)
            original_linear: Original linear layer
            lora_down_weight: LoRA down projection weight
            lora_up_weight: LoRA up projection weight
            strength: LoRA strength multiplier (already adjusted with layer weight)
            alpha: LoRA alpha parameter
            module_key: Unique key for tracking

        Returns:
            True if wrapped successfully, False otherwise
        """
        # Import LoRALinearLayer from training adapters (model-agnostic wrapper class)
        from core.training.adapters.sd15_adapter import LoRALinearLayer

        # Handle already wrapped modules
        if isinstance(original_linear, LoRALinearLayer):
            true_original = original_linear.original_module
        else:
            true_original = original_linear

        # Save original module (first time only)
        if module_key not in self._flux2_lora_original_modules:
            self._flux2_lora_original_modules[module_key] = true_original

        # Compute rank and alpha value
        rank = lora_down_weight.shape[0]
        alpha_value = alpha.item() if alpha is not None else rank

        # Create LoRA wrapper
        # lora_name is required parameter, use module_key for identification
        lora_wrapper = LoRALinearLayer(
            true_original, rank=rank, alpha=alpha_value, lora_name=module_key
        )

        # Load pretrained weights
        device = true_original.weight.device
        dtype = true_original.weight.dtype

        with torch.no_grad():
            lora_wrapper.lora_down.weight.data = lora_down_weight.to(device=device, dtype=dtype)
            lora_wrapper.lora_up.weight.data = lora_up_weight.to(device=device, dtype=dtype)

        # Apply strength (override the default scale)
        lora_wrapper.scale = (alpha_value / rank) * strength

        # Replace in parent module
        if isinstance(attr_name, int):
            parent_module[attr_name] = lora_wrapper
        else:
            setattr(parent_module, attr_name, lora_wrapper)

        self._flux2_lora_wrapped_modules.add(module_key)
        return True

    def _unload_lora_flux2(self):
        """Unload LoRAs from FLUX.2 Transformer"""
        if not hasattr(self, '_flux2_lora_original_modules'):
            print("[FLUX.2 LoRA] No LoRAs loaded")
            return

        if not self.flux2_components:
            print("[FLUX.2 LoRA] WARNING: FLUX.2 components not loaded")
            return

        transformer = self.flux2_components["transformer"]
        unloaded_count = 0

        print(f"[FLUX.2 LoRA] Unloading LoRAs ({len(self._flux2_lora_wrapped_modules)} modules)...")

        for name, module in transformer.named_modules():
            # Flux2Attention
            if module.__class__.__name__ == "Flux2Attention":
                for attr_name in ["to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]:
                    module_key = f"{name}.{attr_name}"
                    if module_key in self._flux2_lora_original_modules:
                        setattr(module, attr_name, self._flux2_lora_original_modules[module_key])
                        unloaded_count += 1

                # to_out (ModuleList)
                module_key = f"{name}.to_out.0"
                if module_key in self._flux2_lora_original_modules and hasattr(module, "to_out"):
                    module.to_out[0] = self._flux2_lora_original_modules[module_key]
                    unloaded_count += 1

            # Flux2ParallelSelfAttention
            elif module.__class__.__name__ == "Flux2ParallelSelfAttention":
                for attr_name in ["to_qkv_mlp_proj", "to_out"]:
                    module_key = f"{name}.{attr_name}"
                    if module_key in self._flux2_lora_original_modules:
                        setattr(module, attr_name, self._flux2_lora_original_modules[module_key])
                        unloaded_count += 1

            # Flux2FeedForward
            elif module.__class__.__name__ == "Flux2FeedForward":
                for attr_name in ["linear_in", "linear_out"]:
                    module_key = f"{name}.{attr_name}"
                    if module_key in self._flux2_lora_original_modules:
                        setattr(module, attr_name, self._flux2_lora_original_modules[module_key])
                        unloaded_count += 1

        self._flux2_lora_wrapped_modules.clear()
        print(f"[FLUX.2 LoRA] Unloaded {unloaded_count} LoRA modules")

    def _get_zimage_scheduler(self, sampler: str):
        """
        Get appropriate Flow Match scheduler for Z-Image based on sampler selection

        Z-Image uses Flow Matching schedulers (different from SD/SDXL).
        Maps user-selected sampler to compatible Flow Match scheduler.

        Sampler mapping:
        - euler → FlowMatchEulerDiscreteScheduler (stochastic_sampling=False)
        - euler_a → FlowMatchEulerDiscreteScheduler (stochastic_sampling=True)
        - heun → FlowMatchHeunDiscreteScheduler

        Args:
            sampler: User-selected sampler name (e.g., "euler", "heun")

        Returns:
            Configured Flow Match scheduler instance
        """
        from diffusers.schedulers import (
            FlowMatchEulerDiscreteScheduler,
            FlowMatchHeunDiscreteScheduler,
        )

        base_scheduler = self.zimage_components["scheduler"]
        config = base_scheduler.config

        # Map sampler to Flow Match scheduler class
        if sampler == "heun":
            scheduler_class = FlowMatchHeunDiscreteScheduler
            print(f"[Z-Image] Using FlowMatchHeunDiscreteScheduler for sampler '{sampler}'")
            return scheduler_class.from_config(config)
        else:
            # Euler/Euler a: use FlowMatchEulerDiscreteScheduler with stochastic_sampling flag
            is_ancestral = sampler in ["euler_a", "dpm2_a"]
            print(f"[Z-Image] Using FlowMatchEulerDiscreteScheduler for sampler '{sampler}' (stochastic={is_ancestral})")

            # Create config dict and enable stochastic_sampling for ancestral samplers
            scheduler_config = dict(config)
            scheduler_config["stochastic_sampling"] = is_ancestral

            return FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    def _generate_txt2img_zimage(self, params: Dict[str, Any], progress_callback=None, step_callback=None) -> tuple[Image.Image, int]:
        """Generate image from text using Z-Image

        Args:
            params: Generation parameters
            progress_callback: Legacy callback (not used for Z-Image)
            step_callback: Step callback (not used for Z-Image)

        Returns:
            tuple: (image, actual_seed)
        """
        if not self.zimage_components:
            raise RuntimeError("Z-Image components not loaded. Please load a Z-Image model first.")

        print("[Z-Image] Starting txt2img generation")

        try:

            # Extract components
            transformer = self.zimage_components["transformer"]
            vae = self.zimage_components["vae"]
            text_encoder = self.zimage_components["text_encoder"]
            tokenizer = self.zimage_components["tokenizer"]

            # Get scheduler based on user-selected sampler
            # Z-Image uses Flow Match schedulers (different from SD/SDXL)
            sampler = params.get("sampler", "euler")
            scheduler = self._get_zimage_scheduler(sampler)

            # Set attention backend based on global settings or params
            attention_type = params.get("attention_type", settings.attention_type)

            # Only switch if attention type has changed (avoid redundant switching overhead)
            if attention_type != self.current_attention_type:
                print(f"[Z-Image] Switching attention backend: {self.current_attention_type} -> {attention_type}")
                from core.models.zimage_transformer import ZImageAttention
                ZImageAttention._attention_backend = attention_type
                self.current_attention_type = attention_type
            else:
                print(f"[Z-Image] Attention backend already set to: {attention_type} (skipping)")
                from core.models.zimage_transformer import ZImageAttention
                ZImageAttention._attention_backend = attention_type  # Ensure it's set (for safety)

            # Load or unload LoRAs
            lora_configs = params.get("loras", [])
            print(f"[Z-Image] DEBUG: lora_configs received: {lora_configs}")
            print(f"[Z-Image] DEBUG: lora_configs type: {type(lora_configs)}")
            print(f"[Z-Image] DEBUG: lora_configs length: {len(lora_configs) if lora_configs else 0}")

            if lora_configs:
                # Unload previous LoRAs first (if any)
                if hasattr(self, '_zimage_lora_wrapped_modules') and self._zimage_lora_wrapped_modules:
                    self._unload_lora_zimage()
                # Load new LoRAs
                self._load_lora_zimage(lora_configs)
            else:
                # No LoRAs requested - unload if any are loaded
                if hasattr(self, '_zimage_lora_wrapped_modules') and self._zimage_lora_wrapped_modules:
                    self._unload_lora_zimage()

            # Prepare generator
            seed = params.get("seed", -1)
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

            # Determine ancestral seed for database storage (stochastic_sampling uses internal RNG)
            ancestral_seed = params.get("ancestral_seed", -1)
            if ancestral_seed == -1:
                # Generate random seed for reproducibility tracking
                actual_ancestral_seed = random.randint(0, 2147483647)
                print(f"[Z-Image] Generated random ancestral seed: {actual_ancestral_seed}")
            else:
                # Use specified seed
                actual_ancestral_seed = ancestral_seed
                print(f"[Z-Image] Using specified ancestral seed: {ancestral_seed}")

            # Z-Image parameters
            prompt = params.get("prompt", "")
            negative_prompt = params.get("negative_prompt", "")
            height = params.get("height", 1024)
            width = params.get("width", 1024)
            num_inference_steps = params.get("steps", 8)  # Turbo default: 8 steps
            max_sequence_length = params.get("max_sequence_length", 512)

            # Z-Image supports CFG (guidance_scale)
            # CFG=1.0: no CFG (positive only)
            # CFG!=1.0: CFG enabled
            guidance_scale = params.get("cfg_scale", 3.5)

            print(f"[Z-Image] Generating {width}x{height} image")
            print(f"[Z-Image] Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")
            print(f"[Z-Image] Prompt: {prompt[:100]}...")

            # Import VRAM optimization functions
            from core.vram_optimization import (
                log_device_status,
                move_zimage_text_encoder_to_gpu,
                move_zimage_text_encoder_to_cpu,
                move_zimage_transformer_to_gpu,
                move_zimage_transformer_to_cpu,
                move_zimage_vae_to_gpu,
                move_zimage_vae_to_cpu
            )

            # Get quantization parameters
            transformer_quantization = params.get("unet_quantization")  # Transformer (U-Net equivalent)
            text_encoder_quantization = params.get("text_encoder_quantization")  # Text Encoder (Z-Image only)

            # ============================================================
            # Stage 1: Text Encoding
            # ============================================================
            text_encoder = move_zimage_text_encoder_to_gpu(text_encoder, text_encoder_quantization)
            log_device_status("Ready for Z-Image text encoding", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            prompt_embeds_list, negative_prompt_embeds_list, do_classifier_free_guidance = \
                self._zimage_encode_prompt(
                    text_encoder, tokenizer, prompt, negative_prompt,
                    guidance_scale, max_sequence_length, text_encoder_quantization
                )

            # Offload Text Encoder to CPU to free VRAM
            move_zimage_text_encoder_to_cpu(text_encoder)
            log_device_status("Text encoding complete, Text Encoder offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 2: Denoising Loop
            # ============================================================
            # Block Swap parameters
            enable_block_swap = params.get("enable_block_swap", False)
            blocks_to_swap = params.get("blocks_to_swap", 20)
            use_pinned_memory = params.get("use_pinned_memory", False)

            if not enable_block_swap:
                # Normal mode: move entire Transformer to GPU
                transformer = move_zimage_transformer_to_gpu(transformer, transformer_quantization)

                # DEBUG: Verify LoRA is still applied after GPU move
                if lora_configs:
                    for attn_name, attn_module in transformer.named_modules():
                        if "ZImageAttention" in attn_module.__class__.__name__:
                            if hasattr(attn_module, "to_q"):
                                weight_norm = attn_module.to_q.weight.data.norm().item()
                                print(f"[Z-Image LoRA DEBUG] After GPU move, first attention to_q weight norm: {weight_norm:.4f}")
                            break

                log_device_status("Ready for Z-Image denoising loop", None, zimage_components={
                    "text_encoder": text_encoder,
                    "transformer": transformer,
                    "vae": vae
                })
            else:
                # Block Swap mode: keep Transformer on CPU for Block Swap initialization
                print("[Z-Image] Block Swap enabled - keeping Transformer on CPU for Block Swap initialization")

                # Create block offloader
                from core.memory_management import create_block_offloader_for_model

                block_offloader = create_block_offloader_for_model(
                    transformer=transformer,
                    blocks_to_swap=blocks_to_swap,
                    device=torch.device(self.device),
                    target_dtype=torch.bfloat16,
                    use_pinned_memory=use_pinned_memory
                )

                # Attach block offloader to transformer
                transformer._block_offloader = block_offloader

                # Prepare block devices (this moves blocks to GPU/CPU according to strategy)
                block_offloader.prepare_block_devices_before_forward()

                log_device_status("Ready for Z-Image denoising loop (Block Swap enabled)", None, zimage_components={
                    "text_encoder": text_encoder,
                    "transformer": transformer,
                    "vae": vae
                })

            latents = self._zimage_denoising_loop(
                transformer, scheduler, prompt_embeds_list, negative_prompt_embeds_list,
                height, width, num_inference_steps, guidance_scale, do_classifier_free_guidance,
                generator, progress_callback, step_callback
            )

            # Offload Transformer to CPU to free VRAM for VAE
            move_zimage_transformer_to_cpu(transformer)
            log_device_status("Denoising complete, Transformer offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 3: VAE Decode
            # ============================================================
            move_zimage_vae_to_gpu(vae)
            log_device_status("Ready for Z-Image VAE decode", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            images = self._zimage_decode_latents(vae, latents)

            # Offload VAE to CPU after decoding
            move_zimage_vae_to_cpu(vae)

            # Clear intermediate tensors from GPU memory
            del prompt_embeds_list, negative_prompt_embeds_list, latents
            torch.cuda.empty_cache()  # Release PyTorch's VRAM cache

            log_device_status("VAE decode complete, all components offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            print("[Z-Image] Generation completed")

            return images[0], seed, actual_ancestral_seed

        except Exception as e:
            print(f"[Z-Image] Generation error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Z-Image generation failed: {str(e)}")

    def _generate_txt2img_flux2(self, params: Dict[str, Any], progress_callback=None, step_callback=None) -> tuple[Image.Image, int, int]:
        """Generate image from text using FLUX.2 Klein

        Args:
            params: Generation parameters
            progress_callback: Callback for progress (step, total_steps, latent)
            step_callback: Step callback (not used for FLUX.2)

        Returns:
            tuple: (image, actual_seed, actual_ancestral_seed)
        """
        if not self.flux2_components:
            raise RuntimeError("FLUX.2 components not loaded. Please load a FLUX.2 model first.")

        print("[FLUX.2] Starting txt2img generation")

        try:
            import numpy as np

            # Load LoRAs if specified
            lora_configs = params.get("loras", [])
            print(f"[FLUX.2] DEBUG: lora_configs from params = {lora_configs}")
            if lora_configs:
                # Unload previous LoRAs first (if any)
                if hasattr(self, '_flux2_lora_wrapped_modules') and self._flux2_lora_wrapped_modules:
                    self._unload_lora_flux2()
                # Load new LoRAs
                print(f"[FLUX.2] Loading {len(lora_configs)} LoRA(s)...")
                self._load_lora_flux2(lora_configs)
            else:
                # No LoRAs requested - unload if any are loaded
                if hasattr(self, '_flux2_lora_wrapped_modules') and self._flux2_lora_wrapped_modules:
                    print(f"[FLUX.2] No LoRAs in params, unloading existing LoRAs")
                    self._unload_lora_flux2()
                else:
                    print(f"[FLUX.2] DEBUG: No LoRAs in params, skipping LoRA loading")

            # Extract components
            transformer = self.flux2_components["transformer"]
            vae = self.flux2_components["vae"]
            text_encoder = self.flux2_components["text_encoder"]
            tokenizer = self.flux2_components["tokenizer"]
            scheduler = self.flux2_components["scheduler"]
            config = self.flux2_components.get("config", {})

            # Prepare generator
            seed = params.get("seed", -1)
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

            # Ancestral seed (for stochastic samplers)
            ancestral_seed = params.get("ancestral_seed", -1)
            if ancestral_seed == -1:
                actual_ancestral_seed = random.randint(0, 2147483647)
                print(f"[FLUX.2] Generated random ancestral seed: {actual_ancestral_seed}")
            else:
                actual_ancestral_seed = ancestral_seed
                print(f"[FLUX.2] Using specified ancestral seed: {ancestral_seed}")

            # FLUX.2 parameters
            prompt = params.get("prompt", "")
            negative_prompt = params.get("negative_prompt", "")
            height = params.get("height", 1024)
            width = params.get("width", 1024)
            num_inference_steps = params.get("steps", 50)
            guidance_scale = params.get("cfg_scale", 4.0)
            max_sequence_length = 512  # FLUX.2 uses Qwen3 with max 512 tokens

            # Check if distilled model (no CFG)
            is_distilled = config.get("is_distilled", False)
            do_classifier_free_guidance = guidance_scale > 1.0 and not is_distilled

            print(f"[FLUX.2] Generating {width}x{height} image")
            print(f"[FLUX.2] Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")
            print(f"[FLUX.2] CFG enabled: {do_classifier_free_guidance}")
            print(f"[FLUX.2] Prompt: {prompt[:100]}...")

            # Import VRAM optimization functions
            from core.vram_optimization import (
                move_flux2_text_encoder_to_gpu,
                move_flux2_transformer_to_gpu
            )

            # Get quantization parameters
            transformer_quantization = params.get("unet_quantization")  # Transformer (U-Net equivalent)
            text_encoder_quantization = params.get("text_encoder_quantization")  # Text Encoder (Qwen3)

            # ============================================================
            # Stage 1: Text Encoding (Qwen3)
            # ============================================================
            print("[FLUX.2] Stage 1: Text encoding...")
            text_encoder = move_flux2_text_encoder_to_gpu(text_encoder, text_encoder_quantization)

            prompt_embeds, text_ids = self._flux2_encode_prompt(
                text_encoder, tokenizer, prompt, max_sequence_length
            )

            if do_classifier_free_guidance:
                negative_prompt_embeds, negative_text_ids = self._flux2_encode_prompt(
                    text_encoder, tokenizer, negative_prompt, max_sequence_length
                )
            else:
                negative_prompt_embeds = None
                negative_text_ids = None

            # Offload text encoder to CPU
            text_encoder.to("cpu")
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 1.5: Encode Reference Images (Image Edit)
            # ============================================================
            ref_images = params.get("ref_images", [])
            ref_tokens = None
            ref_ids = None

            if ref_images:
                print(f"[FLUX.2 Image Edit] Encoding {len(ref_images)} reference image(s)...")
                ref_tokens, ref_ids = self.encode_flux2_image_refs(ref_images, device=self.device)
                if ref_tokens is not None:
                    ref_tokens = ref_tokens.to(prompt_embeds.dtype)
                    ref_ids = ref_ids.to(self.device)
                    print(f"[FLUX.2 Image Edit] Reference tokens: {ref_tokens.shape}, IDs: {ref_ids.shape}")

            # ============================================================
            # Stage 2: Prepare Latents
            # ============================================================
            print("[FLUX.2] Stage 2: Preparing latents...")

            # VAE scale factor (8) * patch size (2) = 16
            vae_scale_factor = 8
            patch_size = 2

            # Ensure height/width divisible by vae_scale_factor * patch_size
            latent_height = 2 * (int(height) // (vae_scale_factor * patch_size))
            latent_width = 2 * (int(width) // (vae_scale_factor * patch_size))

            # FLUX.2 has 32 latent channels, but patchified to 128
            num_channels_latents = transformer.config.in_channels // 4  # 32

            # Create random latents
            latent_shape = (1, num_channels_latents * 4, latent_height // 2, latent_width // 2)
            latents = torch.randn(latent_shape, generator=generator, device=self.device, dtype=prompt_embeds.dtype)

            # Prepare latent position IDs
            latent_ids = self._flux2_prepare_latent_ids(latents).to(self.device)

            # Pack latents: (B, C, H, W) -> (B, H*W, C)
            latents = self._flux2_pack_latents(latents)

            print(f"[FLUX.2] Latents shape: {latents.shape}, Latent IDs shape: {latent_ids.shape}")

            # ============================================================
            # Stage 3: Denoising Loop
            # ============================================================
            print("[FLUX.2] Stage 3: Denoising loop...")

            # Block Swap setup
            enable_block_swap = params.get("enable_block_swap", False)
            blocks_to_swap = params.get("blocks_to_swap", 0) if enable_block_swap else 0
            use_pinned_memory = params.get("use_pinned_memory", False)
            block_offloader = None

            if enable_block_swap and blocks_to_swap > 0:
                print(f"[FLUX.2] Block Swap enabled: {blocks_to_swap} blocks to swap")
                from core.memory_management import create_flux_block_offloader
                from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper

                # Create block offloader
                block_offloader = create_flux_block_offloader(
                    transformer=transformer,
                    blocks_to_swap=blocks_to_swap,
                    device=torch.device(self.device),
                    target_dtype=torch.bfloat16,
                    use_pinned_memory=use_pinned_memory,
                    supports_backward=False
                )

                # Prepare block devices
                block_offloader.prepare_block_devices_before_forward()

                # Wrap transformer
                transformer_wrapper = Flux2BlockSwapWrapper(transformer, block_offloader)
                print("[FLUX.2] Using Block Swap wrapper for denoising")
            else:
                # No Block Swap - ensure ALL weights are on GPU
                # This is important when switching from Block Swap ON to OFF
                from core.memory_management.block_offloading import weighs_to_device
                transformer = move_flux2_transformer_to_gpu(transformer, transformer_quantization)
                # Move all block weights to GPU (in case they were on CPU from previous Block Swap)
                for block in transformer.transformer_blocks:
                    weighs_to_device(block, torch.device(self.device))
                for block in transformer.single_transformer_blocks:
                    weighs_to_device(block, torch.device(self.device))
                transformer_wrapper = transformer

            # Prepare timesteps
            image_seq_len = latents.shape[1]
            mu = self._flux2_compute_empirical_mu(image_seq_len, num_inference_steps)

            # Set timesteps with sigmas
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
            timesteps = scheduler.timesteps
            scheduler.set_begin_index(0)

            # Determine input dtype for transformer (FP8 quantized uses BF16 input)
            transformer_has_fp8 = False
            for module in transformer.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        transformer_has_fp8 = True
                        break

            if transformer_has_fp8:
                transformer_input_dtype = torch.bfloat16
            else:
                transformer_input_dtype = transformer.dtype

            print(f"[FLUX.2] Transformer FP8 detection: {transformer_has_fp8}, input dtype = {transformer_input_dtype}")

            # Denoising loop
            for i, t in enumerate(timesteps):
                if self.cancel_requested:
                    print("[FLUX.2] Generation cancelled")
                    self.cancel_requested = False
                    # Cleanup block offloader if used
                    if block_offloader is not None:
                        block_offloader.cleanup()
                    raise RuntimeError("Generation cancelled by user")

                # Expand timestep
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                latent_model_input = latents.to(transformer_input_dtype)
                latent_image_ids = latent_ids

                # Concatenate reference tokens/IDs if present (Image Edit)
                if ref_tokens is not None:
                    # Temporarily move to GPU for concatenation
                    ref_tokens = ref_tokens.to(device=latent_model_input.device, dtype=transformer_input_dtype)
                    ref_ids = ref_ids.to(device=latent_image_ids.device)
                    latent_model_input = torch.cat([latent_model_input, ref_tokens], dim=1)
                    latent_image_ids = torch.cat([latent_image_ids, ref_ids], dim=1)

                # Batch CFG: Concatenate unconditional and conditional for single forward pass
                if do_classifier_free_guidance:
                    # Double the batch: [uncond, cond]
                    latent_model_input_doubled = torch.cat([latent_model_input, latent_model_input], dim=0)
                    timestep_doubled = torch.cat([timestep, timestep], dim=0)
                    prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    text_ids_combined = torch.cat([negative_text_ids, text_ids], dim=0)
                    latent_image_ids_doubled = torch.cat([latent_image_ids, latent_image_ids], dim=0)

                    # Single forward pass for both unconditional and conditional
                    # For FP8 quantized models, use autocast for mixed precision
                    with torch.no_grad():
                        if transformer_has_fp8:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                noise_pred_combined = transformer_wrapper(
                                    hidden_states=latent_model_input_doubled,
                                    timestep=timestep_doubled / 1000,
                                    guidance=None,
                                    encoder_hidden_states=prompt_embeds_combined,
                                    txt_ids=text_ids_combined,
                                    img_ids=latent_image_ids_doubled,
                                    return_dict=False,
                                )[0]
                        else:
                            noise_pred_combined = transformer_wrapper(
                                hidden_states=latent_model_input_doubled,
                                timestep=timestep_doubled / 1000,
                                guidance=None,
                                encoder_hidden_states=prompt_embeds_combined,
                                txt_ids=text_ids_combined,
                                img_ids=latent_image_ids_doubled,
                                return_dict=False,
                            )[0]

                    # Extract generation part only (remove reference tokens)
                    if ref_tokens is not None:
                        seq_len = latents.shape[1]
                        noise_pred_combined = noise_pred_combined[:, :seq_len, :]

                    # Split and apply CFG formula
                    noise_pred_uncond, noise_pred_cond = noise_pred_combined.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    # Distilled model: Use guidance vector (not CFG)
                    guidance_vec = torch.full(
                        (latent_model_input.shape[0],),
                        guidance_scale,
                        device=latent_model_input.device,
                        dtype=latent_model_input.dtype
                    )
                    # For FP8 quantized models, use autocast for mixed precision
                    with torch.no_grad():
                        if transformer_has_fp8:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                noise_pred = transformer_wrapper(
                                    hidden_states=latent_model_input,
                                    timestep=timestep / 1000,
                                    guidance=guidance_vec,
                                    encoder_hidden_states=prompt_embeds,
                                    txt_ids=text_ids,
                                    img_ids=latent_image_ids,
                                    return_dict=False,
                                )[0]
                        else:
                            noise_pred = transformer_wrapper(
                                hidden_states=latent_model_input,
                                timestep=timestep / 1000,
                                guidance=guidance_vec,
                                encoder_hidden_states=prompt_embeds,
                                txt_ids=text_ids,
                                img_ids=latent_image_ids,
                                return_dict=False,
                            )[0]

                    # Extract generation part only (remove reference tokens)
                    if ref_tokens is not None:
                        seq_len = latents.shape[1]
                        noise_pred = noise_pred[:, :seq_len, :]

                # Predicted clean latent for preview, computed from the
                # pre-step latents + noise_pred. x_t = (1-σ)·x_0 + σ·noise,
                # v = noise - x_0, σ = t / 1000 -> pred_x0 = x_t - σ·v.
                # The progress callback receives this as the 5th positional
                # arg (pred_original_sample) and the factory uses it when
                # preview_predicted_x0=True (defaulted on for FLUX.2 below).
                try:
                    sigma = (
                        t.float() / 1000.0 if isinstance(t, torch.Tensor)
                        else float(t) / 1000.0
                    )
                    preview_pred_x0 = (latents.float() - sigma * noise_pred.float()).to(latents.dtype)
                except Exception:
                    preview_pred_x0 = None

                # Scheduler step
                latents_dtype = latents.dtype
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                # Progress callback (step is 0-indexed, generation_utils will add +1 for display)
                if progress_callback:
                    try:
                        progress_callback(i, len(timesteps), latents, None, preview_pred_x0)
                    except Exception as e:
                        print(f"[FLUX.2] Progress callback error: {e}")

                if (i + 1) % 10 == 0 or i == len(timesteps) - 1:
                    print(f"[FLUX.2] Step {i + 1}/{len(timesteps)}")

            # Cleanup block offloader and offload transformer to CPU
            if block_offloader is not None:
                block_offloader.cleanup()
            transformer.to("cpu")
            torch.cuda.empty_cache()

            # Clean up reference tokens/IDs (Image Edit)
            if ref_tokens is not None:
                del ref_tokens, ref_ids
                torch.cuda.empty_cache()

            # ============================================================
            # Stage 4: VAE Decode
            # ============================================================
            print("[FLUX.2] Stage 4: VAE decoding...")
            vae = vae.to(self.device)

            # Unpack latents with IDs
            latents = self._flux2_unpack_latents_with_ids(latents, latent_ids)

            # Apply BatchNorm scaling (FLUX.2-specific)
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_bn_std + latents_bn_mean

            # Unpatchify
            latents = self._flux2_unpatchify_latents(latents)

            # Decode - convert latents to VAE dtype (bfloat16 -> float32)
            latents = latents.to(dtype=vae.dtype)
            with torch.no_grad():
                image = vae.decode(latents, return_dict=False)[0]

            # Convert to PIL
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image[0] * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)

            # Offload VAE to CPU
            vae.to("cpu")
            torch.cuda.empty_cache()

            print("[FLUX.2] Generation completed")
            return pil_image, seed, actual_ancestral_seed

        except Exception as e:
            print(f"[FLUX.2] Generation error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"FLUX.2 generation failed: {str(e)}")

    def _flux2_encode_prompt(
        self,
        text_encoder,
        tokenizer,
        prompt: str,
        max_sequence_length: int = 512,
        hidden_states_layers: tuple = (9, 18, 27),
    ):
        """Encode prompt using Qwen3 text encoder

        FLUX.2 extracts hidden states from layers 9, 18, 27 of Qwen3 and concatenates them.
        """
        device = text_encoder.device

        # Check if Text Encoder has FP8 weights
        has_fp8_weights = False
        for module in text_encoder.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    has_fp8_weights = True
                    break

        # For FP8 quantized models, use BF16 for output dtype (not FP8)
        if has_fp8_weights:
            dtype = torch.bfloat16
        else:
            dtype = text_encoder.dtype

        print(f"[FLUX.2] FP8 weight detection: has_fp8_weights = {has_fp8_weights}, output dtype = {dtype}")

        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass
        # For FP8 quantized Text Encoder, use autocast for mixed precision
        with torch.no_grad():
            if has_fp8_weights:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
            else:
                output = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

        # Extract and stack hidden states from specified layers
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        # Reshape: (B, num_layers, seq_len, hidden_dim) -> (B, seq_len, num_layers * hidden_dim)
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        # Prepare text IDs (4D position coordinates)
        text_ids = self._flux2_prepare_text_ids(prompt_embeds).to(device)

        return prompt_embeds, text_ids

    def _flux2_prepare_text_ids(self, x: torch.Tensor):
        """Prepare 4D position IDs for text embeddings"""
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1)
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    def _flux2_prepare_latent_ids(self, latents: torch.Tensor):
        """Prepare 4D position IDs for latents"""
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)

        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    def _flux2_pack_latents(self, latents: torch.Tensor):
        """Pack latents: (B, C, H, W) -> (B, H*W, C)"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    def _flux2_unpack_latents_with_ids(self, x: torch.Tensor, x_ids: torch.Tensor):
        """Unpack latents using position IDs"""
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def _flux2_patchify_latents(self, latents: torch.Tensor):
        """Patchify latents for 2x2 patches"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
        return latents

    def _flux2_unpatchify_latents(self, latents: torch.Tensor):
        """Unpatchify latents from 2x2 patches"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
        return latents

    def _flux2_compute_empirical_mu(self, image_seq_len: int, num_steps: int) -> float:
        """Compute empirical mu for FLUX.2 scheduler"""
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            mu = a2 * image_seq_len + b2
            return float(mu)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1

        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_steps + b

        return float(mu)

    # ── Vision Encoder management ────────────────────────────────────────────

    def load_vision_encoder(self, safetensors_path: str):
        """Load (or reload) the SigLIP2 vision encoder from a safetensors file."""
        if self._vision_encoder_path == safetensors_path and self.vision_encoder is not None:
            print(f"[VisionEncoder] Already loaded: {safetensors_path}")
            return
        from core.vision_encoder import SigLIP2VisionEncoderWrapper
        if self.vision_encoder is not None:
            print("[VisionEncoder] Replacing existing vision encoder.")
            self.unload_vision_encoder()
        self.vision_encoder = SigLIP2VisionEncoderWrapper(safetensors_path, device="cpu")
        self._vision_encoder_path = safetensors_path

    def unload_vision_encoder(self):
        """Unload the vision encoder and free memory."""
        if self.vision_encoder is not None:
            self.vision_encoder.to("cpu")
            del self.vision_encoder
            self.vision_encoder = None
            self._vision_encoder_path = None
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[VisionEncoder] Unloaded.")

    def _apply_vision_encoder(
        self,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        ref_images: List[Image.Image],
        nag_negative_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        """
        Encode reference images and concatenate vision embeddings to text embeddings.

        Args:
            prompt_embeds:          [B, 77, D]
            negative_prompt_embeds: [B, 77, D]
            ref_images:             list of PIL Images
            nag_negative_prompt_embeds: optional [B, 77, D] for NAG

        Returns updated (prompt_embeds, negative_prompt_embeds, nag_negative_prompt_embeds).
        All outputs have shape [B, 77 + 1 + 256*N, D].
        """
        from core.vram_optimization import move_vision_encoder_to_gpu, move_vision_encoder_to_cpu

        device = prompt_embeds.device
        dtype  = prompt_embeds.dtype
        target_dim = prompt_embeds.shape[-1]

        move_vision_encoder_to_gpu(self.vision_encoder, str(device))

        ve_pos, ve_neg = self.vision_encoder.encode(
            ref_images, target_dim=target_dim, dtype=dtype
        )
        ve_pos = ve_pos.to(device)
        ve_neg = ve_neg.to(device)

        # Expand to match batch size
        batch = prompt_embeds.shape[0]
        if ve_pos.shape[0] != batch:
            ve_pos = ve_pos.expand(batch, -1, -1)
            ve_neg = ve_neg.expand(batch, -1, -1)

        prompt_embeds          = torch.cat([prompt_embeds,          ve_pos], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, ve_neg], dim=1)

        if nag_negative_prompt_embeds is not None:
            nag_negative_prompt_embeds = torch.cat(
                [nag_negative_prompt_embeds, ve_neg.clone()], dim=1
            )

        move_vision_encoder_to_cpu(self.vision_encoder)

        print(f"[VisionEncoder] Combined embeds: prompt={list(prompt_embeds.shape)}, "
              f"negative={list(negative_prompt_embeds.shape)}")

        return prompt_embeds, negative_prompt_embeds, nag_negative_prompt_embeds

    def encode_flux2_image_refs(self, images: List[Image.Image], device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode reference images for FLUX.2 Image Edit feature

        This encodes reference images into latent tokens with position IDs,
        allowing them to be used as sequence-level conditioning in the transformer.
        Reference images are concatenated with generation latents in the sequence dimension.

        Args:
            images: List of reference images (max 10)
            device: Device to encode on

        Returns:
            ref_tokens: [1, K, 128] Encoded reference image tokens
            ref_ids: [1, K, 4] Position IDs [t, h, w, l]
                     Returns (None, None) if no images provided
        """
        if not images:
            return None, None

        if not self.flux2_components:
            raise RuntimeError("FLUX.2 components not loaded")

        import numpy as np

        # Pixel limits based on number of images
        limit_pixels = 2024**2 if len(images) == 1 else 1024**2

        vae = self.flux2_components["vae"]
        vae_device = next(vae.parameters()).device
        vae_dtype = next(vae.parameters()).dtype

        print(f"[FLUX.2 Image Edit] Encoding {len(images)} reference image(s)...")

        # Preprocess and encode each image
        encoded_refs = []
        for idx, img in enumerate(images[:10]):  # Max 10 images
            # Convert to RGB
            img = img.convert("RGB")

            # Resize to fit pixel limit (preserve aspect ratio)
            w, h = img.size
            if w * h > limit_pixels:
                scale = (limit_pixels / (w * h)) ** 0.5
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                print(f"[FLUX.2 Image Edit] Image {idx+1}: Resized from {w}x{h} to {new_w}x{new_h}")

            # Crop to multiple of 16
            w, h = img.size
            new_w = (w // 16) * 16
            new_h = (h // 16) * 16
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            img = img.crop((left, top, left + new_w, top + new_h))

            # Convert to tensor
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = (img_array - 0.5) * 2.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(device=vae_device, dtype=vae_dtype)

            # VAE encode
            with torch.no_grad():
                latent_dist = vae.encode(img_tensor).latent_dist
                encoded = latent_dist.sample()

                # Patchify: (1, 32, H, W) -> (1, 128, H/2, W/2)
                encoded = self._flux2_patchify_latents(encoded)

                # BatchNorm normalization
                latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(encoded.device, encoded.dtype)
                latents_bn_std = torch.sqrt(
                    vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
                ).to(encoded.device, encoded.dtype)
                encoded = (encoded - latents_bn_mean) / latents_bn_std

                encoded_refs.append(encoded[0])  # [128, H, W]
                print(f"[FLUX.2 Image Edit] Image {idx+1}: Encoded to latent {encoded[0].shape}")

        # Generate position IDs for each reference image
        ref_tokens_list = []
        ref_ids_list = []

        scale = 10  # Time offset scale
        for idx, encoded in enumerate(encoded_refs):
            c, h, w = encoded.shape

            # Time offset: 10, 20, 30, ...
            t_coord = torch.tensor([scale + scale * idx], dtype=torch.long, device=device)

            # Position IDs: [t, h, w, l]
            t_ids = t_coord.expand(h * w)
            h_ids = torch.arange(h, device=device).repeat_interleave(w)
            w_ids = torch.arange(w, device=device).repeat(h)
            l_ids = torch.zeros(h * w, dtype=torch.long, device=device)

            pos_ids = torch.stack([t_ids, h_ids, w_ids, l_ids], dim=1)  # [H*W, 4]

            # Flatten spatial dimensions
            tokens = encoded.view(c, -1).permute(1, 0)  # [H*W, 128]

            ref_tokens_list.append(tokens)
            ref_ids_list.append(pos_ids)

        # Concatenate all references
        ref_tokens = torch.cat(ref_tokens_list, dim=0)  # [K, 128]
        ref_ids = torch.cat(ref_ids_list, dim=0)        # [K, 4]

        # Add batch dimension
        ref_tokens = ref_tokens.unsqueeze(0)  # [1, K, 128]
        ref_ids = ref_ids.unsqueeze(0)        # [1, K, 4]

        print(f"[FLUX.2 Image Edit] Total reference tokens: {ref_tokens.shape[1]}, shape: {ref_tokens.shape}")

        # Offload VAE to CPU after encoding reference images
        vae.to("cpu")
        torch.cuda.empty_cache()

        return ref_tokens, ref_ids

    def _generate_img2img_flux2(self, params: Dict[str, Any], init_image: Image.Image, progress_callback=None, step_callback=None) -> tuple[Image.Image, int, int]:
        """Generate image from image using FLUX.2 Klein

        FLUX.2 supports image conditioning by encoding input images to latents
        and using them as reference during denoising.

        Args:
            params: Generation parameters
            init_image: Input PIL image
            progress_callback: Callback for progress
            step_callback: Step callback (not used)

        Returns:
            tuple: (image, actual_seed, actual_ancestral_seed)
        """
        if not self.flux2_components:
            raise RuntimeError("FLUX.2 components not loaded. Please load a FLUX.2 model first.")

        print("[FLUX.2] Starting img2img generation")

        try:
            import numpy as np

            # Load LoRAs if specified
            lora_configs = params.get("loras", [])
            if lora_configs:
                # Unload previous LoRAs first (if any)
                if hasattr(self, '_flux2_lora_wrapped_modules') and self._flux2_lora_wrapped_modules:
                    self._unload_lora_flux2()
                # Load new LoRAs
                print(f"[FLUX.2] Loading {len(lora_configs)} LoRA(s)...")
                self._load_lora_flux2(lora_configs)
            else:
                # No LoRAs requested - unload if any are loaded
                if hasattr(self, '_flux2_lora_wrapped_modules') and self._flux2_lora_wrapped_modules:
                    print(f"[FLUX.2] No LoRAs in params, unloading existing LoRAs")
                    self._unload_lora_flux2()

            # Extract components
            transformer = self.flux2_components["transformer"]
            vae = self.flux2_components["vae"]
            text_encoder = self.flux2_components["text_encoder"]
            tokenizer = self.flux2_components["tokenizer"]
            scheduler = self.flux2_components["scheduler"]
            config = self.flux2_components.get("config", {})

            # Prepare generator
            seed = params.get("seed", -1)
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

            # Ancestral seed
            ancestral_seed = params.get("ancestral_seed", -1)
            if ancestral_seed == -1:
                actual_ancestral_seed = random.randint(0, 2147483647)
            else:
                actual_ancestral_seed = ancestral_seed

            # Parameters
            prompt = params.get("prompt", "")
            negative_prompt = params.get("negative_prompt", "")
            denoising_strength = params.get("denoising_strength", 0.75)
            num_inference_steps = params.get("steps", 50)
            guidance_scale = params.get("cfg_scale", 4.0)
            max_sequence_length = 512

            # Get image dimensions (use input image size)
            width, height = init_image.size

            # VAE scale factor
            vae_scale_factor = 8
            patch_size = 2
            multiple_of = vae_scale_factor * patch_size

            # Resize if needed
            width = (width // multiple_of) * multiple_of
            height = (height // multiple_of) * multiple_of
            if init_image.size != (width, height):
                init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)

            print(f"[FLUX.2] img2img: {width}x{height}, strength: {denoising_strength}")

            # Check CFG
            is_distilled = config.get("is_distilled", False)
            do_classifier_free_guidance = guidance_scale > 1.0 and not is_distilled

            # Import VRAM optimization functions
            from core.vram_optimization import (
                move_flux2_text_encoder_to_gpu,
                move_flux2_transformer_to_gpu
            )

            # Get quantization parameters
            transformer_quantization = params.get("unet_quantization")
            text_encoder_quantization = params.get("text_encoder_quantization")

            # ============================================================
            # Stage 1: Text Encoding
            # ============================================================
            print("[FLUX.2] Stage 1: Text encoding...")
            text_encoder = move_flux2_text_encoder_to_gpu(text_encoder, text_encoder_quantization)

            prompt_embeds, text_ids = self._flux2_encode_prompt(
                text_encoder, tokenizer, prompt, max_sequence_length
            )

            if do_classifier_free_guidance:
                negative_prompt_embeds, negative_text_ids = self._flux2_encode_prompt(
                    text_encoder, tokenizer, negative_prompt, max_sequence_length
                )
            else:
                negative_prompt_embeds = None
                negative_text_ids = None

            text_encoder.to("cpu")
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 1.5: Encode Reference Images (Image Edit)
            # ============================================================
            ref_images = params.get("ref_images", [])
            ref_tokens = None
            ref_ids = None

            if ref_images:
                print(f"[FLUX.2 Image Edit] Encoding {len(ref_images)} reference image(s)...")
                ref_tokens, ref_ids = self.encode_flux2_image_refs(ref_images, device=self.device)
                if ref_tokens is not None:
                    ref_tokens = ref_tokens.to(prompt_embeds.dtype)
                    ref_ids = ref_ids.to(self.device)
                    print(f"[FLUX.2 Image Edit] Reference tokens: {ref_tokens.shape}, IDs: {ref_ids.shape}")

            # ============================================================
            # Stage 2: Encode input image
            # ============================================================
            print("[FLUX.2] Stage 2: Encoding input image...")
            vae = vae.to(self.device)

            # Preprocess image
            image_tensor = torch.from_numpy(np.array(init_image)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            image_tensor = (image_tensor - 0.5) * 2  # Normalize to [-1, 1]
            image_tensor = image_tensor.to(self.device, dtype=vae.dtype)

            # Encode
            with torch.no_grad():
                latent_dist = vae.encode(image_tensor).latent_dist
                init_latents = latent_dist.mode()  # Use mode for img2img

            # Patchify
            init_latents = self._flux2_patchify_latents(init_latents)

            # Apply BatchNorm normalization
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(init_latents.device, init_latents.dtype)
            latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
            init_latents = (init_latents - latents_bn_mean) / latents_bn_std

            vae.to("cpu")
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 3: Prepare latents with noise
            # ============================================================
            print("[FLUX.2] Stage 3: Preparing latents...")

            # Prepare position IDs
            latent_ids = self._flux2_prepare_latent_ids(init_latents).to(self.device)

            # Pack latents
            init_latents = self._flux2_pack_latents(init_latents)

            # Prepare timesteps
            image_seq_len = init_latents.shape[1]
            mu = self._flux2_compute_empirical_mu(image_seq_len, num_inference_steps)
            scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
            timesteps = scheduler.timesteps

            # Calculate start timestep based on denoising strength
            t_start = max(int(len(timesteps) * (1 - denoising_strength)), 1)
            timesteps = timesteps[t_start:]

            # Add noise at start timestep (Flow Matching linear interpolation)
            # t ranges from 1.0 (pure noise) to 0.0 (clean image)
            # scheduler.timesteps is in [0, 1000] range, normalize to [0, 1]
            t_value = timesteps[0].item() / 1000.0
            noise = torch.randn(init_latents.shape, generator=generator, device=init_latents.device, dtype=init_latents.dtype)
            latents = (1 - t_value) * init_latents + t_value * noise

            print(f"[FLUX.2] Denoising from step {t_start} ({len(timesteps)} steps, t={t_value:.4f})")

            # ============================================================
            # Stage 4: Denoising Loop
            # ============================================================
            print("[FLUX.2] Stage 4: Denoising loop...")

            # Block Swap setup
            enable_block_swap = params.get("enable_block_swap", False)
            blocks_to_swap = params.get("blocks_to_swap", 0) if enable_block_swap else 0
            use_pinned_memory = params.get("use_pinned_memory", False)
            block_offloader = None

            if enable_block_swap and blocks_to_swap > 0:
                print(f"[FLUX.2] Block Swap enabled: {blocks_to_swap} blocks to swap")
                from core.memory_management import create_flux_block_offloader
                from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper

                block_offloader = create_flux_block_offloader(
                    transformer=transformer,
                    blocks_to_swap=blocks_to_swap,
                    device=torch.device(self.device),
                    target_dtype=torch.bfloat16,
                    use_pinned_memory=use_pinned_memory,
                    supports_backward=False
                )
                block_offloader.prepare_block_devices_before_forward()
                transformer_wrapper = Flux2BlockSwapWrapper(transformer, block_offloader)
                print("[FLUX.2] Using Block Swap wrapper for denoising")
            else:
                # No Block Swap - ensure ALL weights are on GPU
                from core.memory_management.block_offloading import weighs_to_device
                transformer = move_flux2_transformer_to_gpu(transformer, transformer_quantization)
                for block in transformer.transformer_blocks:
                    weighs_to_device(block, torch.device(self.device))
                for block in transformer.single_transformer_blocks:
                    weighs_to_device(block, torch.device(self.device))
                transformer_wrapper = transformer

            scheduler.set_begin_index(t_start)

            # Determine input dtype for transformer (FP8 quantized uses BF16 input)
            transformer_has_fp8 = False
            for module in transformer.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        transformer_has_fp8 = True
                        break

            if transformer_has_fp8:
                transformer_input_dtype = torch.bfloat16
            else:
                transformer_input_dtype = transformer.dtype

            print(f"[FLUX.2] Transformer FP8 detection: {transformer_has_fp8}, input dtype = {transformer_input_dtype}")

            for i, t in enumerate(timesteps):
                if self.cancel_requested:
                    print("[FLUX.2] Generation cancelled")
                    self.cancel_requested = False
                    if block_offloader is not None:
                        block_offloader.cleanup()
                    raise RuntimeError("Generation cancelled by user")

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = latents.to(transformer_input_dtype)
                latent_image_ids = latent_ids

                # Concatenate reference tokens/IDs if present (Image Edit)
                if ref_tokens is not None:
                    # Temporarily move to GPU for concatenation
                    ref_tokens = ref_tokens.to(device=latent_model_input.device, dtype=transformer_input_dtype)
                    ref_ids = ref_ids.to(device=latent_image_ids.device)
                    latent_model_input = torch.cat([latent_model_input, ref_tokens], dim=1)
                    latent_image_ids = torch.cat([latent_image_ids, ref_ids], dim=1)

                # Batch CFG: Concatenate unconditional and conditional for single forward pass
                if do_classifier_free_guidance:
                    # Double the batch: [uncond, cond]
                    latent_model_input_doubled = torch.cat([latent_model_input, latent_model_input], dim=0)
                    timestep_doubled = torch.cat([timestep, timestep], dim=0)
                    prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    text_ids_combined = torch.cat([negative_text_ids, text_ids], dim=0)
                    latent_image_ids_doubled = torch.cat([latent_image_ids, latent_image_ids], dim=0)

                    # Single forward pass for both unconditional and conditional
                    # For FP8 quantized models, use autocast for mixed precision
                    with torch.no_grad():
                        if transformer_has_fp8:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                noise_pred_combined = transformer_wrapper(
                                    hidden_states=latent_model_input_doubled,
                                    timestep=timestep_doubled / 1000,
                                    guidance=None,
                                    encoder_hidden_states=prompt_embeds_combined,
                                    txt_ids=text_ids_combined,
                                    img_ids=latent_image_ids_doubled,
                                    return_dict=False,
                                )[0]
                        else:
                            noise_pred_combined = transformer_wrapper(
                                hidden_states=latent_model_input_doubled,
                                timestep=timestep_doubled / 1000,
                                guidance=None,
                                encoder_hidden_states=prompt_embeds_combined,
                                txt_ids=text_ids_combined,
                                img_ids=latent_image_ids_doubled,
                                return_dict=False,
                            )[0]

                    # Extract generation part only (remove reference tokens)
                    if ref_tokens is not None:
                        seq_len = latents.shape[1]
                        noise_pred_combined = noise_pred_combined[:, :seq_len, :]

                    # Split and apply CFG formula
                    noise_pred_uncond, noise_pred_cond = noise_pred_combined.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    # Distilled model: Use guidance vector (not CFG)
                    guidance_vec = torch.full(
                        (latent_model_input.shape[0],),
                        guidance_scale,
                        device=latent_model_input.device,
                        dtype=latent_model_input.dtype
                    )
                    # For FP8 quantized models, use autocast for mixed precision
                    with torch.no_grad():
                        if transformer_has_fp8:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                noise_pred = transformer_wrapper(
                                    hidden_states=latent_model_input,
                                    timestep=timestep / 1000,
                                    guidance=guidance_vec,
                                    encoder_hidden_states=prompt_embeds,
                                    txt_ids=text_ids,
                                    img_ids=latent_image_ids,
                                    return_dict=False,
                                )[0]
                        else:
                            noise_pred = transformer_wrapper(
                                hidden_states=latent_model_input,
                                timestep=timestep / 1000,
                                guidance=guidance_vec,
                                encoder_hidden_states=prompt_embeds,
                                txt_ids=text_ids,
                                img_ids=latent_image_ids,
                                return_dict=False,
                            )[0]

                    # Extract generation part only (remove reference tokens)
                    if ref_tokens is not None:
                        seq_len = latents.shape[1]
                        noise_pred = noise_pred[:, :seq_len, :]

                # Step
                latents_dtype = latents.dtype
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                # Progress callback (step is 0-indexed, generation_utils will add +1 for display)
                if progress_callback:
                    try:
                        progress_callback(i, len(timesteps), latents)
                    except Exception:
                        pass

            # Cleanup block offloader and offload transformer to CPU (img2img)
            if block_offloader is not None:
                block_offloader.cleanup()
            transformer.to("cpu")
            torch.cuda.empty_cache()

            # Clean up reference tokens/IDs (Image Edit)
            if ref_tokens is not None:
                del ref_tokens, ref_ids
                torch.cuda.empty_cache()

            # ============================================================
            # Stage 5: VAE Decode (img2img)
            # ============================================================
            print("[FLUX.2] Stage 5: VAE decoding...")
            vae = vae.to(self.device)

            latents = self._flux2_unpack_latents_with_ids(latents, latent_ids)

            # Denormalize
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_bn_std + latents_bn_mean

            latents = self._flux2_unpatchify_latents(latents)

            with torch.no_grad():
                image = vae.decode(latents, return_dict=False)[0]

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image[0] * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)

            vae.to("cpu")
            torch.cuda.empty_cache()

            print("[FLUX.2] img2img generation completed")
            return pil_image, seed, actual_ancestral_seed

        except Exception as e:
            print(f"[FLUX.2] img2img error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"FLUX.2 img2img failed: {str(e)}")

    def _generate_inpaint_flux2(
        self,
        params: Dict[str, Any],
        init_image: Image.Image,
        mask_image: Image.Image,
        progress_callback=None,
        step_callback=None
    ) -> tuple[Image.Image, int, int]:
        """Generate inpainted image using FLUX.2 Klein

        FLUX.2 inpainting works by blending masked regions during denoising.

        Args:
            params: Generation parameters
            init_image: Input PIL image
            mask_image: Mask PIL image (white = inpaint, black = keep)
            progress_callback: Callback for progress
            step_callback: Step callback (not used)

        Returns:
            tuple: (image, actual_seed, actual_ancestral_seed)
        """
        if not self.flux2_components:
            raise RuntimeError("FLUX.2 components not loaded. Please load a FLUX.2 model first.")

        print("[FLUX.2] Starting inpaint generation")

        try:
            import numpy as np

            # Load LoRAs if specified
            lora_configs = params.get("loras", [])
            if lora_configs:
                # Unload previous LoRAs first (if any)
                if hasattr(self, '_flux2_lora_wrapped_modules') and self._flux2_lora_wrapped_modules:
                    self._unload_lora_flux2()
                # Load new LoRAs
                print(f"[FLUX.2] Loading {len(lora_configs)} LoRA(s)...")
                self._load_lora_flux2(lora_configs)
            else:
                # No LoRAs requested - unload if any are loaded
                if hasattr(self, '_flux2_lora_wrapped_modules') and self._flux2_lora_wrapped_modules:
                    print(f"[FLUX.2] No LoRAs in params, unloading existing LoRAs")
                    self._unload_lora_flux2()

            # Extract components
            transformer = self.flux2_components["transformer"]
            vae = self.flux2_components["vae"]
            text_encoder = self.flux2_components["text_encoder"]
            tokenizer = self.flux2_components["tokenizer"]
            scheduler = self.flux2_components["scheduler"]
            config = self.flux2_components.get("config", {})

            # Prepare generator
            seed = params.get("seed", -1)
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

            # Ancestral seed
            ancestral_seed = params.get("ancestral_seed", -1)
            if ancestral_seed == -1:
                actual_ancestral_seed = random.randint(0, 2147483647)
            else:
                actual_ancestral_seed = ancestral_seed

            # Parameters
            prompt = params.get("prompt", "")
            negative_prompt = params.get("negative_prompt", "")
            denoising_strength = params.get("denoising_strength", 1.0)
            num_inference_steps = params.get("steps", 50)
            guidance_scale = params.get("cfg_scale", 4.0)
            mask_blur = params.get("mask_blur", 4)
            max_sequence_length = 512

            # Get dimensions
            width, height = init_image.size

            vae_scale_factor = 8
            patch_size = 2
            multiple_of = vae_scale_factor * patch_size

            # Resize if needed
            width = (width // multiple_of) * multiple_of
            height = (height // multiple_of) * multiple_of
            if init_image.size != (width, height):
                init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)
                mask_image = mask_image.resize((width, height), Image.Resampling.LANCZOS)

            print(f"[FLUX.2] inpaint: {width}x{height}, strength: {denoising_strength}")

            # Apply mask blur
            if mask_blur > 0:
                from PIL import ImageFilter
                mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))

            # Check CFG
            is_distilled = config.get("is_distilled", False)
            do_classifier_free_guidance = guidance_scale > 1.0 and not is_distilled

            # Import VRAM optimization functions
            from core.vram_optimization import (
                move_flux2_text_encoder_to_gpu,
                move_flux2_transformer_to_gpu
            )

            # Get quantization parameters
            transformer_quantization = params.get("unet_quantization")
            text_encoder_quantization = params.get("text_encoder_quantization")

            # ============================================================
            # Stage 1: Text Encoding
            # ============================================================
            print("[FLUX.2] Stage 1: Text encoding...")
            text_encoder = move_flux2_text_encoder_to_gpu(text_encoder, text_encoder_quantization)

            prompt_embeds, text_ids = self._flux2_encode_prompt(
                text_encoder, tokenizer, prompt, max_sequence_length
            )

            if do_classifier_free_guidance:
                negative_prompt_embeds, negative_text_ids = self._flux2_encode_prompt(
                    text_encoder, tokenizer, negative_prompt, max_sequence_length
                )
            else:
                negative_prompt_embeds = None
                negative_text_ids = None

            text_encoder.to("cpu")
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 1.5: Encode Reference Images (Image Edit)
            # ============================================================
            ref_images = params.get("ref_images", [])
            ref_tokens = None
            ref_ids = None

            if ref_images:
                print(f"[FLUX.2 Image Edit] Encoding {len(ref_images)} reference image(s)...")
                ref_tokens, ref_ids = self.encode_flux2_image_refs(ref_images, device=self.device)
                if ref_tokens is not None:
                    ref_tokens = ref_tokens.to(prompt_embeds.dtype)
                    ref_ids = ref_ids.to(self.device)
                    print(f"[FLUX.2 Image Edit] Reference tokens: {ref_tokens.shape}, IDs: {ref_ids.shape}")

            # ============================================================
            # Stage 2: Encode input image and prepare mask
            # ============================================================
            print("[FLUX.2] Stage 2: Encoding input image and mask...")
            vae = vae.to(self.device)

            # Preprocess image
            image_tensor = torch.from_numpy(np.array(init_image)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            image_tensor = (image_tensor - 0.5) * 2
            image_tensor = image_tensor.to(self.device, dtype=vae.dtype)

            # Encode
            with torch.no_grad():
                latent_dist = vae.encode(image_tensor).latent_dist
                init_latents = latent_dist.mode()

            # Prepare mask in latent space
            mask_tensor = torch.from_numpy(np.array(mask_image.convert("L"))).float() / 255.0
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            # Resize mask to latent size
            latent_h = height // vae_scale_factor
            latent_w = width // vae_scale_factor
            mask_latent = torch.nn.functional.interpolate(
                mask_tensor, size=(latent_h, latent_w), mode='bilinear', align_corners=False
            )
            mask_latent = mask_latent.to(self.device, dtype=init_latents.dtype)

            # Patchify
            init_latents = self._flux2_patchify_latents(init_latents)

            # Apply BatchNorm normalization
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(init_latents.device, init_latents.dtype)
            latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
            init_latents_normalized = (init_latents - latents_bn_mean) / latents_bn_std

            vae.to("cpu")
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 3: Prepare latents
            # ============================================================
            print("[FLUX.2] Stage 3: Preparing latents...")

            # Patchify mask (same spatial transform as latents)
            # Mask for patchified latents needs special handling
            mask_patchified = torch.nn.functional.interpolate(
                mask_latent, size=(latent_h // 2, latent_w // 2), mode='bilinear', align_corners=False
            )

            # Prepare position IDs
            latent_ids = self._flux2_prepare_latent_ids(init_latents).to(self.device)

            # Pack latents
            init_latents_packed = self._flux2_pack_latents(init_latents_normalized)

            # Pack mask (1, 1, H/2, W/2) -> (1, H*W/4, 1)
            mask_packed = mask_patchified.reshape(1, 1, -1).permute(0, 2, 1)

            # Prepare timesteps
            image_seq_len = init_latents_packed.shape[1]
            mu = self._flux2_compute_empirical_mu(image_seq_len, num_inference_steps)
            scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
            timesteps = scheduler.timesteps

            # Calculate start timestep
            t_start = max(int(len(timesteps) * (1 - denoising_strength)), 1)
            timesteps = timesteps[t_start:]

            # Add noise (Flow Matching linear interpolation)
            # t ranges from 1.0 (pure noise) to 0.0 (clean image)
            # scheduler.timesteps is in [0, 1000] range, normalize to [0, 1]
            t_value = timesteps[0].item() / 1000.0
            noise = torch.randn(init_latents_packed.shape, generator=generator, device=init_latents_packed.device, dtype=init_latents_packed.dtype)
            latents = (1 - t_value) * init_latents_packed + t_value * noise

            print(f"[FLUX.2] Inpainting from step {t_start} ({len(timesteps)} steps, t={t_value:.4f})")

            # ============================================================
            # Stage 4: Denoising Loop with mask blending
            # ============================================================
            print("[FLUX.2] Stage 4: Denoising loop with mask blending...")

            # Block Swap setup
            enable_block_swap = params.get("enable_block_swap", False)
            blocks_to_swap = params.get("blocks_to_swap", 0) if enable_block_swap else 0
            use_pinned_memory = params.get("use_pinned_memory", False)
            block_offloader = None

            if enable_block_swap and blocks_to_swap > 0:
                print(f"[FLUX.2] Block Swap enabled: {blocks_to_swap} blocks to swap")
                from core.memory_management import create_flux_block_offloader
                from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper

                block_offloader = create_flux_block_offloader(
                    transformer=transformer,
                    blocks_to_swap=blocks_to_swap,
                    device=torch.device(self.device),
                    target_dtype=torch.bfloat16,
                    use_pinned_memory=use_pinned_memory,
                    supports_backward=False
                )
                block_offloader.prepare_block_devices_before_forward()
                transformer_wrapper = Flux2BlockSwapWrapper(transformer, block_offloader)
                print("[FLUX.2] Using Block Swap wrapper for denoising")
            else:
                # No Block Swap - ensure ALL weights are on GPU
                from core.memory_management.block_offloading import weighs_to_device
                transformer = move_flux2_transformer_to_gpu(transformer, transformer_quantization)
                for block in transformer.transformer_blocks:
                    weighs_to_device(block, torch.device(self.device))
                for block in transformer.single_transformer_blocks:
                    weighs_to_device(block, torch.device(self.device))
                transformer_wrapper = transformer

            scheduler.set_begin_index(t_start)

            # Determine input dtype for transformer (FP8 quantized uses BF16 input)
            transformer_has_fp8 = False
            for module in transformer.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        transformer_has_fp8 = True
                        break

            if transformer_has_fp8:
                transformer_input_dtype = torch.bfloat16
            else:
                transformer_input_dtype = transformer.dtype

            print(f"[FLUX.2] Transformer FP8 detection: {transformer_has_fp8}, input dtype = {transformer_input_dtype}")

            for i, t in enumerate(timesteps):
                if self.cancel_requested:
                    print("[FLUX.2] Generation cancelled")
                    self.cancel_requested = False
                    if block_offloader is not None:
                        block_offloader.cleanup()
                    raise RuntimeError("Generation cancelled by user")

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = latents.to(transformer_input_dtype)
                latent_image_ids = latent_ids

                # Concatenate reference tokens/IDs if present (Image Edit)
                if ref_tokens is not None:
                    # Temporarily move to GPU for concatenation
                    ref_tokens = ref_tokens.to(device=latent_model_input.device, dtype=transformer_input_dtype)
                    ref_ids = ref_ids.to(device=latent_image_ids.device)
                    latent_model_input = torch.cat([latent_model_input, ref_tokens], dim=1)
                    latent_image_ids = torch.cat([latent_image_ids, ref_ids], dim=1)

                # Batch CFG: Concatenate unconditional and conditional for single forward pass
                if do_classifier_free_guidance:
                    # Double the batch: [uncond, cond]
                    latent_model_input_doubled = torch.cat([latent_model_input, latent_model_input], dim=0)
                    timestep_doubled = torch.cat([timestep, timestep], dim=0)
                    prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    text_ids_combined = torch.cat([negative_text_ids, text_ids], dim=0)
                    latent_image_ids_doubled = torch.cat([latent_image_ids, latent_image_ids], dim=0)

                    # Single forward pass for both unconditional and conditional
                    # For FP8 quantized models, use autocast for mixed precision
                    with torch.no_grad():
                        if transformer_has_fp8:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                noise_pred_combined = transformer_wrapper(
                                    hidden_states=latent_model_input_doubled,
                                    timestep=timestep_doubled / 1000,
                                    guidance=None,
                                    encoder_hidden_states=prompt_embeds_combined,
                                    txt_ids=text_ids_combined,
                                    img_ids=latent_image_ids_doubled,
                                    return_dict=False,
                                )[0]
                        else:
                            noise_pred_combined = transformer_wrapper(
                                hidden_states=latent_model_input_doubled,
                                timestep=timestep_doubled / 1000,
                                guidance=None,
                                encoder_hidden_states=prompt_embeds_combined,
                                txt_ids=text_ids_combined,
                                img_ids=latent_image_ids_doubled,
                                return_dict=False,
                            )[0]

                    # Extract generation part only (remove reference tokens)
                    if ref_tokens is not None:
                        seq_len = latents.shape[1]
                        noise_pred_combined = noise_pred_combined[:, :seq_len, :]

                    # Split and apply CFG formula
                    noise_pred_uncond, noise_pred_cond = noise_pred_combined.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    # Distilled model: Use guidance vector (not CFG)
                    guidance_vec = torch.full(
                        (latent_model_input.shape[0],),
                        guidance_scale,
                        device=latent_model_input.device,
                        dtype=latent_model_input.dtype
                    )
                    # For FP8 quantized models, use autocast for mixed precision
                    with torch.no_grad():
                        if transformer_has_fp8:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                noise_pred = transformer_wrapper(
                                    hidden_states=latent_model_input,
                                    timestep=timestep / 1000,
                                    guidance=guidance_vec,
                                    encoder_hidden_states=prompt_embeds,
                                    txt_ids=text_ids,
                                    img_ids=latent_image_ids,
                                    return_dict=False,
                                )[0]
                        else:
                            noise_pred = transformer_wrapper(
                                hidden_states=latent_model_input,
                                timestep=timestep / 1000,
                                guidance=guidance_vec,
                                encoder_hidden_states=prompt_embeds,
                                txt_ids=text_ids,
                                img_ids=latent_image_ids,
                                return_dict=False,
                            )[0]

                    # Extract generation part only (remove reference tokens)
                    if ref_tokens is not None:
                        seq_len = latents.shape[1]
                        noise_pred = noise_pred[:, :seq_len, :]

                # Step
                latents_dtype = latents.dtype
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                # Blend with original in unmasked regions
                # Noise original latents to current timestep using Flow Matching interpolation
                if i < len(timesteps) - 1:
                    # Flow Matching: normalize timestep [0, 1000] -> [0.0, 1.0]
                    t_value = timesteps[i + 1].item() / 1000.0
                    # Linear interpolation: x_t = (1 - t) * x_0 + t * noise
                    init_latents_noised = (1 - t_value) * init_latents_packed + t_value * noise
                else:
                    init_latents_noised = init_latents_packed

                # Blend: mask=1 -> use new latents, mask=0 -> use original
                latents = mask_packed * latents + (1 - mask_packed) * init_latents_noised

                # Progress callback (step is 0-indexed, generation_utils will add +1 for display)
                if progress_callback:
                    try:
                        progress_callback(i, len(timesteps), latents)
                    except Exception:
                        pass

            # Cleanup block offloader and offload transformer to CPU (inpaint)
            if block_offloader is not None:
                block_offloader.cleanup()
            transformer.to("cpu")
            torch.cuda.empty_cache()

            # Clean up reference tokens/IDs (Image Edit)
            if ref_tokens is not None:
                del ref_tokens, ref_ids
                torch.cuda.empty_cache()

            # ============================================================
            # Stage 5: VAE Decode (inpaint)
            # ============================================================
            print("[FLUX.2] Stage 5: VAE decoding...")
            vae = vae.to(self.device)

            latents = self._flux2_unpack_latents_with_ids(latents, latent_ids)

            # Denormalize
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_bn_std + latents_bn_mean

            latents = self._flux2_unpatchify_latents(latents)

            with torch.no_grad():
                image = vae.decode(latents, return_dict=False)[0]

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image[0] * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)

            vae.to("cpu")
            torch.cuda.empty_cache()

            print("[FLUX.2] inpaint generation completed")
            return pil_image, seed, actual_ancestral_seed

        except Exception as e:
            print(f"[FLUX.2] inpaint error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"FLUX.2 inpaint failed: {str(e)}")

    def _generate_img2img_zimage(self, params: Dict[str, Any], init_image: Image.Image, progress_callback=None, step_callback=None) -> tuple[Image.Image, int]:
        """Generate image from image using Z-Image

        Args:
            params: Generation parameters
            init_image: Input PIL image
            progress_callback: Legacy callback (not used for Z-Image)
            step_callback: Step callback (not used for Z-Image)

        Returns:
            tuple: (image, actual_seed)
        """
        if not self.zimage_components:
            raise RuntimeError("Z-Image components not loaded. Please load a Z-Image model first.")

        print("[Z-Image] Starting img2img generation")

        try:
            # Extract components
            transformer = self.zimage_components["transformer"]
            vae = self.zimage_components["vae"]
            text_encoder = self.zimage_components["text_encoder"]
            tokenizer = self.zimage_components["tokenizer"]

            # Get scheduler based on user-selected sampler
            # Z-Image uses Flow Match schedulers (different from SD/SDXL)
            sampler = params.get("sampler", "euler")
            scheduler = self._get_zimage_scheduler(sampler)

            # Set attention backend
            attention_type = params.get("attention_type", settings.attention_type)
            if attention_type != self.current_attention_type:
                print(f"[Z-Image] Switching attention backend: {self.current_attention_type} -> {attention_type}")
                from core.models.zimage_transformer import ZImageAttention
                ZImageAttention._attention_backend = attention_type
                self.current_attention_type = attention_type
            else:
                print(f"[Z-Image] Attention backend already set to: {attention_type} (skipping)")
                from core.models.zimage_transformer import ZImageAttention
                ZImageAttention._attention_backend = attention_type

            # Load or unload LoRAs
            lora_configs = params.get("loras", [])
            if lora_configs:
                if hasattr(self, '_zimage_lora_wrapped_modules') and self._zimage_lora_wrapped_modules:
                    self._unload_lora_zimage()
                self._load_lora_zimage(lora_configs)
            else:
                if hasattr(self, '_zimage_lora_wrapped_modules') and self._zimage_lora_wrapped_modules:
                    self._unload_lora_zimage()

            # Prepare generator
            seed = params.get("seed", -1)
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

            # Determine ancestral seed for database storage (stochastic_sampling uses internal RNG)
            ancestral_seed = params.get("ancestral_seed", -1)
            if ancestral_seed == -1:
                # Generate random seed for reproducibility tracking
                actual_ancestral_seed = random.randint(0, 2147483647)
                print(f"[Z-Image] Generated random ancestral seed: {actual_ancestral_seed}")
            else:
                # Use specified seed
                actual_ancestral_seed = ancestral_seed
                print(f"[Z-Image] Using specified ancestral seed: {ancestral_seed}")

            # Z-Image parameters
            prompt = params.get("prompt", "")
            negative_prompt = params.get("negative_prompt", "")
            height = params.get("height", 1024)
            width = params.get("width", 1024)
            num_inference_steps = params.get("steps", 8)
            max_sequence_length = params.get("max_sequence_length", 512)
            guidance_scale = params.get("cfg_scale", 3.5)

            # img2img specific parameters
            denoising_strength = params.get("denoising_strength", 0.75)

            print(f"[Z-Image] Generating {width}x{height} image from input image")
            print(f"[Z-Image] Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}, Strength: {denoising_strength}")
            print(f"[Z-Image] Prompt: {prompt[:100]}...")

            # Import VRAM optimization functions
            from core.vram_optimization import (
                log_device_status,
                move_zimage_text_encoder_to_gpu,
                move_zimage_text_encoder_to_cpu,
                move_zimage_transformer_to_gpu,
                move_zimage_transformer_to_cpu,
                move_zimage_vae_to_gpu,
                move_zimage_vae_to_cpu
            )

            # Get quantization parameters
            transformer_quantization = params.get("unet_quantization")
            text_encoder_quantization = params.get("text_encoder_quantization")

            # ============================================================
            # Stage 1: Text Encoding
            # ============================================================
            text_encoder = move_zimage_text_encoder_to_gpu(text_encoder, text_encoder_quantization)
            log_device_status("Ready for Z-Image text encoding", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            prompt_embeds_list, negative_prompt_embeds_list, do_classifier_free_guidance = \
                self._zimage_encode_prompt(
                    text_encoder, tokenizer, prompt, negative_prompt,
                    guidance_scale, max_sequence_length, text_encoder_quantization
                )

            # Offload Text Encoder to CPU
            move_zimage_text_encoder_to_cpu(text_encoder)
            log_device_status("Text encoding complete, Text Encoder offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 2: VAE Encode Input Image
            # ============================================================
            move_zimage_vae_to_gpu(vae)
            log_device_status("Ready for Z-Image VAE encode (img2img)", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # Resize input image if needed
            if init_image.size != (width, height):
                print(f"[Z-Image] Resizing input image from {init_image.size} to {width}x{height}")
                init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)

            # Prepare image tensor
            import numpy as np
            image_array = np.array(init_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            image_tensor = image_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
            image_tensor = image_tensor.to(device=self.device, dtype=vae.dtype)

            # Encode to latent space
            # Z-Image VAE uses encoder -> quant_conv -> sample (not encode method)
            with torch.no_grad():
                h = vae.encoder(image_tensor)
                if vae.quant_conv is not None:
                    h = vae.quant_conv(h)
                mean, logvar = torch.chunk(h, 2, dim=1)
                std = torch.exp(0.5 * logvar)

                # Generate noise with generator
                noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
                init_latents = mean + std * noise

                # Z-Image VAE scaling factor (apply scaling and shift)
                if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
                    init_latents = init_latents * vae.config.scaling_factor
                else:
                    # Fallback: assume standard scaling
                    init_latents = init_latents * 0.13025

                # Clean up intermediate tensors
                del h, mean, logvar, std

            print(f"[Z-Image] Encoded input image to latents: {init_latents.shape}")

            # Offload VAE to CPU after encoding
            move_zimage_vae_to_cpu(vae)

            # ============================================================
            # Stage 3: Add Noise to Latents (Flow Matching Style)
            # ============================================================
            device = torch.device(self.device)

            # Calculate VAE scale factor for dynamic shift
            if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            else:
                vae_scale_factor = 8

            # Calculate dynamic shift
            from core.zimage_utils import calculate_shift
            image_seq_len = (init_latents.shape[2] // 2) * (init_latents.shape[3] // 2)
            mu = calculate_shift(
                image_seq_len,
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.15),
            )

            # Set scheduler parameters
            scheduler.sigma_min = 0.0
            scheduler_kwargs = {"mu": mu}

            # Prepare full timesteps first
            scheduler.set_timesteps(num_inference_steps, device=device, **scheduler_kwargs)
            timesteps = scheduler.timesteps

            # Calculate timestep to start from (based on strength)
            init_timestep = int(num_inference_steps * denoising_strength)
            t_start = max(num_inference_steps - init_timestep, 0)

            # Get partial timesteps for img2img
            timesteps_img2img = timesteps[t_start:]

            print(f"[Z-Image] img2img: Using {len(timesteps_img2img)}/{len(timesteps)} timesteps (t_start={t_start}, strength={denoising_strength})")

            # Add noise to init_latents at the starting timestep
            noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=torch.float32)

            # Flow Matching noise addition
            # Check if scheduler has add_noise method
            if hasattr(scheduler, 'add_noise'):
                print(f"[Z-Image] Using scheduler.add_noise() for noise addition")
                noised_latents = scheduler.add_noise(init_latents, noise, timesteps_img2img[0:1])
            else:
                # Manual flow matching noise addition: x_t = (1 - t) * x_0 + t * noise
                # Normalize timestep to [0, 1] range (Z-Image: 1000=start/noisy, 0=end/clean)
                t_normalized = timesteps_img2img[0].item() / 1000.0
                print(f"[Z-Image] Manual flow matching noise addition: t={timesteps_img2img[0].item():.1f}, t_norm={t_normalized:.3f}")
                noised_latents = (1.0 - t_normalized) * init_latents + t_normalized * noise

            print(f"[Z-Image] Noised latents shape: {noised_latents.shape}, dtype: {noised_latents.dtype}")

            # ============================================================
            # Stage 4: Denoising Loop
            # ============================================================
            enable_block_swap = params.get("enable_block_swap", False)
            blocks_to_swap = params.get("blocks_to_swap", 20)
            use_pinned_memory = params.get("use_pinned_memory", False)

            if not enable_block_swap:
                transformer = move_zimage_transformer_to_gpu(transformer, transformer_quantization)
                log_device_status("Ready for Z-Image denoising loop (img2img)", None, zimage_components={
                    "text_encoder": text_encoder,
                    "transformer": transformer,
                    "vae": vae
                })
            else:
                print("[Z-Image] Block Swap enabled - keeping Transformer on CPU for Block Swap initialization")
                from core.memory_management import create_block_offloader_for_model
                block_offloader = create_block_offloader_for_model(
                    transformer=transformer,
                    blocks_to_swap=blocks_to_swap,
                    device=torch.device(self.device),
                    target_dtype=torch.bfloat16,
                    use_pinned_memory=use_pinned_memory
                )
                transformer._block_offloader = block_offloader
                block_offloader.prepare_block_devices_before_forward()
                log_device_status("Ready for Z-Image denoising loop (Block Swap enabled, img2img)", None, zimage_components={
                    "text_encoder": text_encoder,
                    "transformer": transformer,
                    "vae": vae
                })

            # Run denoising loop with noised latents and partial timesteps
            latents = self._zimage_denoising_loop(
                transformer, scheduler, prompt_embeds_list, negative_prompt_embeds_list,
                height, width, num_inference_steps, guidance_scale, do_classifier_free_guidance,
                generator, progress_callback, step_callback,
                init_latents=noised_latents,
                timesteps_override=timesteps_img2img
            )

            # Offload Transformer to CPU
            move_zimage_transformer_to_cpu(transformer)
            log_device_status("Denoising complete, Transformer offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 5: VAE Decode
            # ============================================================
            move_zimage_vae_to_gpu(vae)
            log_device_status("Ready for Z-Image VAE decode", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            images = self._zimage_decode_latents(vae, latents)

            # Offload VAE to CPU after decoding
            move_zimage_vae_to_cpu(vae)

            # Clear intermediate tensors
            del prompt_embeds_list, negative_prompt_embeds_list, init_latents, noised_latents, latents
            torch.cuda.empty_cache()

            log_device_status("VAE decode complete, all components offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            print("[Z-Image] img2img generation completed")

            return images[0], seed, actual_ancestral_seed

        except Exception as e:
            print(f"[Z-Image] img2img generation error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Z-Image img2img generation failed: {str(e)}")

    def _generate_inpaint_zimage(
        self, params: dict, init_image, mask_image, progress_callback=None, step_callback=None
    ) -> tuple:
        """
        Generate inpainted image using Z-Image model.

        Inpaint = img2img + mask blending
        - Encode init_image to latents
        - Add noise based on denoising_strength
        - Denoise with mask blending at each step
        - Decode back to image

        Args:
            params: Generation parameters (prompt, steps, cfg_scale, etc.)
            init_image: PIL Image (area to inpaint)
            mask_image: PIL Image (white = inpaint, black = keep)
            progress_callback: Progress callback function
            step_callback: Step callback function

        Returns:
            (generated_image, seed)
        """
        try:
            # Get components
            text_encoder = self.zimage_components["text_encoder"]
            tokenizer = self.zimage_components["tokenizer"]
            transformer = self.zimage_components["transformer"]
            vae = self.zimage_components["vae"]
            scheduler = self.zimage_components["scheduler"]

            # Get parameters
            prompt = params.get("prompt", "")
            negative_prompt = params.get("negative_prompt", "")
            num_inference_steps = params.get("steps", 8)
            guidance_scale = params.get("cfg_scale", 3.5)
            height = params.get("height", 1024)
            width = params.get("width", 1024)
            seed = params.get("seed", -1)
            denoising_strength = params.get("denoising_strength", 0.75)
            mask_blur = params.get("mask_blur", 0)
            max_sequence_length = params.get("max_sequence_length", 256)

            # Generate seed
            if seed == -1:
                seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device=self.device).manual_seed(seed)

            print(f"[Z-Image] Starting inpaint generation")
            print(f"[Z-Image] Generating {width}x{height} inpainted image")
            print(f"[Z-Image] Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}, Strength: {denoising_strength}")
            print(f"[Z-Image] Mask blur: {mask_blur}")
            print(f"[Z-Image] Prompt: {prompt[:100]}...")

            # Import VRAM optimization functions
            from core.vram_optimization import (
                log_device_status,
                move_zimage_text_encoder_to_gpu,
                move_zimage_text_encoder_to_cpu,
                move_zimage_transformer_to_gpu,
                move_zimage_transformer_to_cpu,
                move_zimage_vae_to_gpu,
                move_zimage_vae_to_cpu
            )

            # Get quantization parameters
            transformer_quantization = params.get("unet_quantization")
            text_encoder_quantization = params.get("text_encoder_quantization")

            # ============================================================
            # Stage 1: Text Encoding
            # ============================================================
            text_encoder = move_zimage_text_encoder_to_gpu(text_encoder, text_encoder_quantization)
            log_device_status("Ready for Z-Image text encoding", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            prompt_embeds_list, negative_prompt_embeds_list, do_classifier_free_guidance = \
                self._zimage_encode_prompt(
                    text_encoder, tokenizer, prompt, negative_prompt,
                    guidance_scale, max_sequence_length, text_encoder_quantization
                )

            # Offload Text Encoder to CPU
            move_zimage_text_encoder_to_cpu(text_encoder)
            log_device_status("Text encoding complete, Text Encoder offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 2: VAE Encode Input Image and Mask
            # ============================================================
            move_zimage_vae_to_gpu(vae)
            log_device_status("Ready for Z-Image VAE encode (inpaint)", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # Resize input image and mask if needed
            if init_image.size != (width, height):
                print(f"[Z-Image] Resizing input image from {init_image.size} to {width}x{height}")
                init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)

            if mask_image.size != (width, height):
                print(f"[Z-Image] Resizing mask from {mask_image.size} to {width}x{height}")
                mask_image = mask_image.resize((width, height), Image.Resampling.LANCZOS)

            # Apply mask blur if requested
            if mask_blur > 0:
                from PIL import ImageFilter
                mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))
                print(f"[Z-Image] Applied Gaussian blur to mask (radius={mask_blur})")

            # Prepare image tensor
            import numpy as np
            image_array = np.array(init_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            image_tensor = image_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
            image_tensor = image_tensor.to(device=self.device, dtype=vae.dtype)

            # Prepare mask tensor (white = 1 = inpaint, black = 0 = keep)
            mask_array = np.array(mask_image.convert('L')).astype(np.float32) / 255.0  # Grayscale
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)  # 1CHW
            mask_tensor = mask_tensor.to(device=self.device, dtype=vae.dtype)

            # Encode input image to latent space
            with torch.no_grad():
                h = vae.encoder(image_tensor)
                if vae.quant_conv is not None:
                    h = vae.quant_conv(h)
                mean, logvar = torch.chunk(h, 2, dim=1)
                std = torch.exp(0.5 * logvar)

                # Generate noise with generator
                noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
                init_latents = mean + std * noise

                # Z-Image VAE scaling factor
                if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
                    init_latents = init_latents * vae.config.scaling_factor
                else:
                    init_latents = init_latents * 0.13025

                # Store original latents for mask blending
                original_latents = init_latents.clone()

                # Clean up intermediate tensors
                del h, mean, logvar, std

            # Resize mask to latent dimensions (downsample by VAE scale factor)
            # Z-Image VAE: 8x downsampling -> latent is 1/8 of image size
            latent_height = init_latents.shape[2]
            latent_width = init_latents.shape[3]
            mask_latent = torch.nn.functional.interpolate(
                mask_tensor, size=(latent_height, latent_width), mode='nearest'
            )

            print(f"[Z-Image] Encoded input image to latents: {init_latents.shape}")
            print(f"[Z-Image] Mask latent shape: {mask_latent.shape}")

            # Offload VAE to CPU after encoding
            move_zimage_vae_to_cpu(vae)

            # ============================================================
            # Stage 3: Add Noise to Latents (Flow Matching Style)
            # ============================================================
            device = torch.device(self.device)

            # Calculate VAE scale factor for dynamic shift
            if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            else:
                vae_scale_factor = 8

            # Calculate dynamic shift
            from core.zimage_utils import calculate_shift
            image_seq_len = (init_latents.shape[2] // 2) * (init_latents.shape[3] // 2)
            mu = calculate_shift(
                image_seq_len,
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.15),
            )

            # Set scheduler parameters
            scheduler.sigma_min = 0.0
            scheduler_kwargs = {"mu": mu}

            # Prepare full timesteps first
            scheduler.set_timesteps(num_inference_steps, device=device, **scheduler_kwargs)
            timesteps = scheduler.timesteps

            # Calculate timestep to start from (based on strength)
            init_timestep = int(num_inference_steps * denoising_strength)
            t_start = max(num_inference_steps - init_timestep, 0)

            # Get partial timesteps for inpaint
            timesteps_inpaint = timesteps[t_start:]

            print(f"[Z-Image] inpaint: Using {len(timesteps_inpaint)}/{len(timesteps)} timesteps (t_start={t_start}, strength={denoising_strength})")

            # Save original unnoised latents (for mask blending in loop)
            original_latents = init_latents.clone()

            # Add noise to init_latents at the starting timestep
            noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=torch.float32)

            # Flow Matching noise addition (apply to entire image, mask blending happens in loop)
            if hasattr(scheduler, 'add_noise'):
                print(f"[Z-Image] Using scheduler.add_noise() for noise addition")
                noised_latents = scheduler.add_noise(init_latents, noise, timesteps_inpaint[0:1])
            else:
                # Manual flow matching noise addition: x_t = (1 - t) * x_0 + t * noise
                t_normalized = timesteps_inpaint[0].item() / 1000.0
                print(f"[Z-Image] Manual flow matching noise addition: t={timesteps_inpaint[0].item():.1f}, t_norm={t_normalized:.3f}")
                noised_latents = (1.0 - t_normalized) * init_latents + t_normalized * noise

            print(f"[Z-Image] Noised latents shape: {noised_latents.shape}, dtype: {noised_latents.dtype}")

            # ============================================================
            # Stage 4: Denoising Loop with Mask Blending
            # ============================================================
            enable_block_swap = params.get("enable_block_swap", False)
            blocks_to_swap = params.get("blocks_to_swap", 20)
            use_pinned_memory = params.get("use_pinned_memory", False)

            if not enable_block_swap:
                transformer = move_zimage_transformer_to_gpu(transformer, transformer_quantization)
                log_device_status("Ready for Z-Image denoising loop (inpaint)", None, zimage_components={
                    "text_encoder": text_encoder,
                    "transformer": transformer,
                    "vae": vae
                })
            else:
                print("[Z-Image] Block Swap enabled - keeping Transformer on CPU for Block Swap initialization")
                from core.memory_management import create_block_offloader_for_model
                block_offloader = create_block_offloader_for_model(
                    transformer=transformer,
                    blocks_to_swap=blocks_to_swap,
                    device=torch.device(self.device),
                    target_dtype=torch.bfloat16,
                    use_pinned_memory=use_pinned_memory
                )
                transformer._block_offloader = block_offloader
                block_offloader.prepare_block_devices_before_forward()
                log_device_status("Ready for Z-Image denoising loop (Block Swap enabled, inpaint)", None, zimage_components={
                    "text_encoder": text_encoder,
                    "transformer": transformer,
                    "vae": vae
                })

            # Run denoising loop with mask blending
            latents = self._zimage_denoising_loop(
                transformer, scheduler, prompt_embeds_list, negative_prompt_embeds_list,
                height, width, num_inference_steps, guidance_scale, do_classifier_free_guidance,
                generator, progress_callback, step_callback,
                init_latents=noised_latents,
                timesteps_override=timesteps_inpaint,
                mask_latent=mask_latent,
                original_latents=original_latents
            )

            # Offload Transformer to CPU
            move_zimage_transformer_to_cpu(transformer)
            log_device_status("Denoising complete, Transformer offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            # ============================================================
            # Stage 5: VAE Decode
            # ============================================================
            move_zimage_vae_to_gpu(vae)
            log_device_status("Ready for Z-Image VAE decode", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            images = self._zimage_decode_latents(vae, latents)

            # Offload VAE to CPU after decoding
            move_zimage_vae_to_cpu(vae)

            # Clear intermediate tensors
            del prompt_embeds_list, negative_prompt_embeds_list, init_latents, original_latents, noised_latents, mask_latent, latents
            torch.cuda.empty_cache()

            log_device_status("VAE decode complete, all components offloaded to CPU", None, zimage_components={
                "text_encoder": text_encoder,
                "transformer": transformer,
                "vae": vae
            })

            print("[Z-Image] inpaint generation completed")

            return images[0], seed, actual_ancestral_seed

        except Exception as e:
            print(f"[Z-Image] inpaint generation error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Z-Image inpaint generation failed: {str(e)}")

    def _zimage_encode_prompt(
        self, text_encoder, tokenizer, prompt, negative_prompt,
        guidance_scale, max_sequence_length, text_encoder_quantization=None
    ):
        """
        Stage 1: Text Encoding for Z-Image
        Encodes prompt and negative prompt using Qwen text encoder.
        Text encoder is on GPU when this is called, and will be moved to CPU after.

        Returns:
            prompt_embeds_list: List of text embeddings (one per image)
            negative_prompt_embeds_list: List of negative embeddings (if CFG enabled)
            do_classifier_free_guidance: bool
        """
        device = next(text_encoder.parameters()).device

        # Check if Text Encoder has FP8 weights
        has_fp8_weights = False
        if text_encoder_quantization and text_encoder_quantization.startswith('fp8_'):
            for module in text_encoder.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        has_fp8_weights = True
                        break

        # Format prompts using Qwen chat template
        if isinstance(prompt, str):
            prompt = [prompt]

        # CFG is enabled when guidance_scale is not 1.0 (consistent with SD/SDXL)
        # CFG=1.0 or CFG=0.0: no CFG (positive only)
        # CFG!=1.0 and CFG!=0.0: CFG enabled
        # Note: CFG=0.0 is treated as "positive only" (same as CFG=1.0)
        do_classifier_free_guidance = abs(guidance_scale - 1.0) > 1e-5 and abs(guidance_scale) > 1e-5

        print(f"[Z-Image] Encoding prompt with Text Encoder on {device}")

        formatted_prompts = []
        for p in prompt:
            messages = [{"role": "user", "content": p}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            formatted_prompts.append(formatted_prompt)

        # Tokenize prompts
        text_inputs = tokenizer(
            formatted_prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        # Encode prompts (use penultimate layer output)
        # For FP8 quantized Text Encoder, use autocast for mixed precision
        with torch.no_grad():
            if has_fp8_weights:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    prompt_embeds = text_encoder(
                        input_ids=text_input_ids,
                        attention_mask=prompt_masks,
                        output_hidden_states=True,
                    ).hidden_states[-2]
            else:
                prompt_embeds = text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=prompt_masks,
                    output_hidden_states=True,
                ).hidden_states[-2]

        # Extract embeddings per prompt (masked by attention mask)
        prompt_embeds_list = []
        for i in range(len(prompt_embeds)):
            prompt_embeds_list.append(prompt_embeds[i][prompt_masks[i]])

        # Encode negative prompts if CFG is enabled
        negative_prompt_embeds_list = []
        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]

            neg_formatted = []
            for p in negative_prompt:
                messages = [{"role": "user", "content": p}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                neg_formatted.append(formatted_prompt)

            neg_inputs = tokenizer(
                neg_formatted,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )

            neg_input_ids = neg_inputs.input_ids.to(device)
            neg_masks = neg_inputs.attention_mask.to(device).bool()

            with torch.no_grad():
                if has_fp8_weights:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        neg_embeds = text_encoder(
                            input_ids=neg_input_ids,
                            attention_mask=neg_masks,
                            output_hidden_states=True,
                        ).hidden_states[-2]
                else:
                    neg_embeds = text_encoder(
                        input_ids=neg_input_ids,
                        attention_mask=neg_masks,
                        output_hidden_states=True,
                    ).hidden_states[-2]

            for i in range(len(neg_embeds)):
                negative_prompt_embeds_list.append(neg_embeds[i][neg_masks[i]])

        print(f"[Z-Image] Text encoding complete: {len(prompt_embeds_list)} prompts encoded")

        return prompt_embeds_list, negative_prompt_embeds_list, do_classifier_free_guidance

    def _zimage_denoising_loop(
        self, transformer, scheduler, prompt_embeds_list, negative_prompt_embeds_list,
        height, width, num_inference_steps, guidance_scale, do_classifier_free_guidance,
        generator, progress_callback, step_callback,
        init_latents: Optional[torch.Tensor] = None,
        timesteps_override: Optional[torch.Tensor] = None,
        mask_latent: Optional[torch.Tensor] = None,
        original_latents: Optional[torch.Tensor] = None
    ):
        """
        Stage 2: Denoising Loop for Z-Image
        Runs the transformer denoising loop with flow matching.
        Transformer is on GPU when this is called, and will be moved to CPU after.

        Args:
            init_latents: Optional initial latents for img2img/inpaint (already noised)
            timesteps_override: Optional timesteps for img2img/inpaint (partial timesteps from t_start)
            mask_latent: Optional mask for inpainting (1 = inpaint, 0 = keep original)
            original_latents: Optional original unnoised latents for inpaint blending

        Returns:
            latents: Denoised latents (torch.Tensor)
        """
        # Import calculate_shift from local zimage_utils (with fallback)
        try:
            from core.zimage_utils import calculate_shift
        except ImportError:
            # Fallback implementation if zimage_utils is not available
            def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
                m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
                b = base_shift - m * base_seq_len
                mu = image_seq_len * m + b
                return mu

        # Use self.device instead of transformer device (Block Swap may have weights on CPU)
        device = torch.device(self.device)

        print(f"[Z-Image] Starting denoising loop on {device}")

        # Calculate VAE scale factor
        vae = self.zimage_components["vae"]
        if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        else:
            vae_scale_factor = 8
        vae_scale = vae_scale_factor * 2

        # Calculate latent dimensions
        height_latent = 2 * (int(height) // vae_scale)
        width_latent = 2 * (int(width) // vae_scale)
        batch_size = len(prompt_embeds_list)
        shape = (batch_size, transformer.in_channels, height_latent, width_latent)

        # Initialize latents (use init_latents if provided for img2img, otherwise random for txt2img)
        if init_latents is not None:
            latents = init_latents.to(device=device, dtype=torch.float32)
            print(f"[Z-Image] Starting from noised input image latents (img2img)")
        else:
            latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
            print(f"[Z-Image] Starting from random latents (txt2img)")

        # Calculate dynamic shift for flow matching
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

        # Use local calculate_shift implementation (from zimage_utils.py or fallback)
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )

        # Set scheduler parameters
        scheduler.sigma_min = 0.0

        # Prepare timesteps (use override if provided for img2img, otherwise calculate normally)
        if timesteps_override is not None:
            timesteps = timesteps_override
            print(f"[Z-Image] Using {len(timesteps)} timesteps for img2img (strength-based, from t_start)")
        else:
            # Only FlowMatchEulerDiscreteScheduler supports 'mu' parameter
            # FlowMatchHeunDiscreteScheduler does not support it
            if hasattr(scheduler, '__class__') and 'Euler' in scheduler.__class__.__name__:
                scheduler_kwargs = {"mu": mu}
                scheduler.set_timesteps(num_inference_steps, device=device, **scheduler_kwargs)
                print(f"[Z-Image] Denoising loop: {num_inference_steps} steps requested, {len(scheduler.timesteps)} timesteps generated, shift={mu:.3f}")
            else:
                # Heun or other schedulers: no mu parameter
                scheduler.set_timesteps(num_inference_steps, device=device)
                print(f"[Z-Image] Denoising loop: {num_inference_steps} steps requested, {len(scheduler.timesteps)} timesteps generated (scheduler: {scheduler.__class__.__name__})")
            timesteps = scheduler.timesteps

        # Detect FP8 quantization (check once before loop)
        has_fp8_weights = False
        for module in transformer.modules():
            if isinstance(module, torch.nn.Linear):
                if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    has_fp8_weights = True
                    print(f"[Z-Image] Detected FP8 quantized Transformer (dtype: {module.weight.dtype})")
                    print(f"[Z-Image] Will use autocast for mixed precision inference")
                    break
        if not has_fp8_weights:
            print(f"[Z-Image] Transformer not quantized (BF16 inference)")

        # Denoising loop with progress callback
        # Note: Heun scheduler generates 2*steps-1 timesteps (39 for 20 steps)
        # We normalize progress to user-requested num_inference_steps for UI consistency
        for i, t in enumerate(timesteps):
            if self.cancel_requested:
                print("[Z-Image] Generation cancelled by user")
                raise RuntimeError("Generation cancelled by user")
            # Skip last step if t=0 (flow matching termination)
            if t == 0 and i == len(timesteps) - 1:
                print(f"[Z-Image] Step {i+1}/{len(timesteps)} | t={t.item():.2f} | Skipping last step (flow matching termination)")
                continue

            # Calculate normalized step for progress bar (map timestep index to user-requested steps)
            # For Heun: len(timesteps)=39, num_inference_steps=20 → normalize i to 0-19 range
            normalized_step = int((i / len(timesteps)) * num_inference_steps)

            # step_callback fires before the model forward so step-range LoRA
            # hooks see the next step index correctly. progress_callback (which
            # carries the preview payload) is deferred until after the forward
            # so we can hand it pred_x0 in addition to the raw latents.
            if step_callback:
                step_callback(normalized_step, num_inference_steps)

            # Normalize timestep to [0, 1]
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000
            t_norm = timestep[0].item()

            # CFG truncation logic (disable CFG after certain timestep)
            # Default value from Z-Image: DEFAULT_CFG_TRUNCATION = 1.0
            current_guidance_scale = guidance_scale
            cfg_truncation = 1.0  # Z-Image default
            if do_classifier_free_guidance and cfg_truncation is not None and float(cfg_truncation) <= 1:
                if t_norm > cfg_truncation:
                    current_guidance_scale = 1.0  # Set to 1.0 (no CFG) instead of 0.0

            # Apply CFG when guidance_scale is not 1.0 (consistent with SD/SDXL)
            apply_cfg = do_classifier_free_guidance and abs(current_guidance_scale - 1.0) > 1e-5

            # Prepare model input (concat positive + negative if CFG)
            # Note: For FP8 quantization, keep input in BF16/FP16, don't convert to FP8
            if has_fp8_weights:
                # FP8 quantized: use BF16 input (autocast will handle conversion)
                input_dtype = torch.bfloat16
            else:
                # Normal case: use transformer's dtype
                transformer_dtype = next(transformer.parameters()).dtype
                input_dtype = transformer_dtype

            if apply_cfg:
                latent_model_input = latents.to(input_dtype).repeat(2, 1, 1, 1)
                # CFG input order: [negative, positive] (consistent with SD/SDXL)
                prompt_embeds_model_input = negative_prompt_embeds_list + prompt_embeds_list
                timestep_model_input = timestep.repeat(2)
            else:
                latent_model_input = latents.to(input_dtype)
                prompt_embeds_model_input = prompt_embeds_list
                timestep_model_input = timestep

            # Add channel dimension and split into list
            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))

            # Transformer forward pass
            # For FP8 quantized models, use autocast to handle mixed precision
            with torch.no_grad():
                if has_fp8_weights:
                    # FP8: use autocast for automatic mixed precision
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        model_out_list = transformer(
                            latent_model_input_list,
                            timestep_model_input,
                            prompt_embeds_model_input,
                        )[0]
                else:
                    # Normal: no autocast needed
                    model_out_list = transformer(
                        latent_model_input_list,
                        timestep_model_input,
                        prompt_embeds_model_input,
                    )[0]

            # Apply CFG if enabled
            if apply_cfg:
                # CFG output order matches input: [negative, positive]
                neg_out = model_out_list[:batch_size]  # negative (uncond)
                pos_out = model_out_list[batch_size:]  # positive (cond)
                noise_pred = []
                for j in range(batch_size):
                    neg = neg_out[j].float()
                    pos = pos_out[j].float()
                    # Standard CFG formula (consistent with SD/SDXL)
                    # pred = uncond + guidance_scale * (cond - uncond)
                    pred = neg + current_guidance_scale * (pos - neg)
                    noise_pred.append(pred)
                noise_pred = torch.stack(noise_pred, dim=0)
            else:
                noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)

            # Scheduler step (flow matching with stochastic_sampling if enabled)
            noise_pred = -noise_pred.squeeze(2)

            # Predicted clean latent for preview: x_t = (1-σ)·x_0 + σ·noise,
            # v = noise - x_0, so x_0 = x_t - σ·v. σ is t_norm (the timestep
            # already normalised to [0, 1] above). Note that the sign flip on
            # noise_pred above gives us the standard-direction velocity, so the
            # straight subtraction is the right formula here.
            try:
                preview_pred_x0 = (latents.float() - t_norm * noise_pred.float()).to(latents.dtype)
            except Exception:
                preview_pred_x0 = None

            if progress_callback:
                try:
                    progress_callback(normalized_step, num_inference_steps, latents,
                                       None, preview_pred_x0)
                except Exception as e:
                    print(f"[Z-Image] Progress callback error: {e}")

            latents = scheduler.step(
                noise_pred.to(torch.float32), t, latents,
                return_dict=False
            )[0]

            # Inpaint mask blending: blend denoised latents with noised original latents
            if mask_latent is not None and original_latents is not None:
                # For inpaint, non-masked area should also be noised at current timestep
                # then blended with denoised latents
                original_latents_device = original_latents.to(device=latents.device, dtype=latents.dtype)
                mask_latent_device = mask_latent.to(device=latents.device, dtype=latents.dtype)

                # Add noise to original latents at current timestep
                # This ensures non-masked area follows the same noise schedule
                if i < len(timesteps) - 1:  # Not the last step
                    next_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor([0.0], device=device)
                    # Generate noise for original latents
                    noise_for_original = torch.randn_like(original_latents_device)

                    # Flow Matching: add noise at next timestep level
                    t_next_normalized = next_t.item() / 1000.0
                    noised_original = (1.0 - t_next_normalized) * original_latents_device + t_next_normalized * noise_for_original
                else:
                    # Last step: use clean original latents
                    noised_original = original_latents_device

                # Blend: mask * denoised + (1 - mask) * noised_original
                latents = mask_latent_device * latents + (1.0 - mask_latent_device) * noised_original

            if normalized_step % 5 == 0 or normalized_step == num_inference_steps - 1:
                print(f"[Z-Image] Step {normalized_step+1}/{num_inference_steps} | t={t_norm:.3f} | CFG={current_guidance_scale:.1f}")

        print(f"[Z-Image] Denoising loop complete")

        return latents

    def _zimage_decode_latents(self, vae, latents):
        """
        Stage 3: VAE Decode for Z-Image
        Decodes latents to images using VAE.
        VAE is on GPU when this is called, and will be moved to CPU after.

        Returns:
            images: List of PIL images
        """
        device = next(vae.parameters()).device

        print(f"[Z-Image] Decoding latents with VAE on {device}")

        # Apply VAE scaling and shift
        shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
        latents = (latents.to(vae.dtype) / vae.config.scaling_factor) + shift_factor

        # Decode latents
        with torch.no_grad():
            image = vae.decode(latents, return_dict=False)[0]

        # Convert to PIL images
        from PIL import Image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        images = [Image.fromarray(img) for img in image]

        print(f"[Z-Image] VAE decode complete: {len(images)} images generated")

        return images

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
        from core.extensions.controlnet_manager import controlnet_manager

        if not controlnet_images:
            return pipeline

        try:
            # Load ControlNet models - separate LLLite from standard ControlNets
            controlnets = []
            control_images = []
            lllite_models = []

            for cn_config in controlnet_images:
                # Detect if model is LLLite
                model_path = cn_config["model_path"]
                is_lllite = controlnet_manager.is_lllite_model(model_path)

                # Load ControlNet model
                controlnet = controlnet_manager.load_controlnet(
                    model_path,
                    device=self.device,
                    dtype=pipeline.dtype if hasattr(pipeline, 'dtype') else torch.float16,
                    is_lllite=is_lllite
                )

                if controlnet is None:
                    print(f"Warning: Could not load ControlNet {cn_config['model_path']}")
                    continue

                # Apply layer weights if specified (only for standard ControlNets)
                layer_weights = cn_config.get("layer_weights")
                print(f"[Pipeline] ControlNet config: model_path={cn_config.get('model_path')}, is_lllite={is_lllite}, layer_weights={layer_weights}")
                if layer_weights and not is_lllite:
                    print(f"[Pipeline] Applying layer weights to ControlNet: {layer_weights}")
                    controlnet_manager.apply_layer_weights(controlnet, layer_weights)
                elif is_lllite:
                    print(f"[Pipeline] Skipping layer weights for LLLite model (not supported)")
                else:
                    print(f"[Pipeline] No layer weights specified for this ControlNet")

                # Prepare control image
                control_image = controlnet_manager.prepare_controlnet_image(
                    cn_config["image"],
                    width,
                    height
                )

                # Separate LLLite from standard ControlNets
                if is_lllite:
                    lllite_models.append({
                        'model': controlnet,
                        'image': control_image,
                        'config': cn_config
                    })
                else:
                    controlnets.append(controlnet)
                    control_images.append(control_image)

            # Apply LLLite models directly to U-Net
            if lllite_models:
                print(f"Applying {len(lllite_models)} LLLite model(s) to U-Net")
                for lllite_data in lllite_models:
                    controlnet_manager.apply_lllite_to_unet(
                        pipeline.unet,
                        lllite_data['model'],
                        lllite_data['image']
                    )

            if not controlnets:
                print("No standard ControlNets loaded, using original pipeline" +
                      (f" (with {len(lllite_models)} LLLite(s))" if lllite_models else ""))
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

            # Ensure all pipeline components are on the correct device
            cn_pipeline = cn_pipeline.to(self.device)

            # Move VAE back to CPU to preserve VRAM optimization
            # (VAE will be moved to GPU only when needed for encode/decode)
            if hasattr(cn_pipeline, 'vae') and cn_pipeline.vae is not None:
                cn_pipeline.vae.to('cpu')

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
        from core.prompts.prompt_parser import parse_prompt_attention, apply_emphasis_to_embeds

        # Use provided pipeline or default to txt2img_pipeline
        if pipeline is None:
            pipeline = self.txt2img_pipeline

        if pipeline is None:
            return None, None, None, None

        # Check if SDXL by checking if text_encoder_2 exists (more reliable than isinstance for ControlNet pipelines)
        is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None

        # Use the text encoder's actual device so cpu_text_encoding works correctly:
        # when TE is kept on CPU, encode_prompt must also receive device="cpu" to avoid
        # a device mismatch between text_input_ids.to(device) and TE weights.
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
            device = next(pipeline.text_encoder.parameters()).device
        else:
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

    def _encode_prompt_nobos_single_chunk(self, prompt: str, negative_prompt: str = "", pipeline=None):
        """
        Encode prompts with NoBOS mode for single chunk (<=75 tokens).
        Strips BOS and EOS tokens from embeddings.

        Returns:
            For SD1.5: (prompt_embeds, negative_prompt_embeds, None, None)
            For SDXL: (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        """
        from core.prompts.prompt_parser import parse_prompt_attention, apply_emphasis_to_embeds

        # Use provided pipeline or default to txt2img_pipeline
        if pipeline is None:
            pipeline = self.txt2img_pipeline

        if pipeline is None:
            return None, None, None, None

        # Check if SDXL by checking if text_encoder_2 exists
        is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None

        device = self.device
        dtype = pipeline.dtype if hasattr(pipeline, 'dtype') else torch.float16

        # Parse prompts for emphasis syntax
        import re
        has_pos_emphasis = bool(re.search(r'(?<!\\)[\(\[]', prompt))
        has_neg_emphasis = bool(re.search(r'(?<!\\)[\(\[]', negative_prompt))

        tokenizer = pipeline.tokenizer_2 if is_sdxl else pipeline.tokenizer

        # Encode positive prompt
        embeds = pipeline.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False
        )

        prompt_embeds = embeds[0]
        pooled_prompt_embeds = embeds[2] if is_sdxl else None

        # Strip BOS (first token) and EOS (last token) for NoBOS mode
        # For prompts <=75 tokens, embedding shape is typically [1, 77, hidden_dim]
        # Remove first and last tokens: [1, 75, hidden_dim]
        if prompt_embeds.shape[1] > 2:  # Ensure there are enough tokens
            prompt_embeds = prompt_embeds[:, 1:-1, :]

        # Apply emphasis weights if present
        if has_pos_emphasis:
            prompt_embeds = apply_emphasis_to_embeds(
                prompt, prompt_embeds,
                tokenizer,
                device, dtype
            )

        # Encode negative prompt
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None

        if negative_prompt:
            neg_embeds = pipeline.encode_prompt(
                prompt=negative_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )

            negative_prompt_embeds = neg_embeds[0]
            negative_pooled_prompt_embeds = neg_embeds[2] if is_sdxl else None

            # Strip BOS and EOS for NoBOS mode
            if negative_prompt_embeds.shape[1] > 2:
                negative_prompt_embeds = negative_prompt_embeds[:, 1:-1, :]

            # Apply emphasis weights if present
            if has_neg_emphasis:
                negative_prompt_embeds = apply_emphasis_to_embeds(
                    negative_prompt, negative_prompt_embeds,
                    tokenizer,
                    device, dtype
                )

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
            from core.prompts.prompt_parser import parse_prompt_attention

            # Get clean prompt for length check
            clean_prompt = prompt
            if has_pos_emphasis:
                parsed = parse_prompt_attention(prompt)
                clean_prompt = "".join([text for text, _ in parsed])

            prompt_tokens = tokenizer(clean_prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
            needs_chunking = len(prompt_tokens) > 75
        else:
            needs_chunking = False

        # Check if NoBOS mode is enabled
        needs_nobos_processing = self.prompt_chunking_mode == "nobos"

        # Use chunked encoding for long prompts
        if needs_chunking:
            return self._encode_prompt_chunked(prompt, negative_prompt, pipeline)
        elif needs_nobos_processing:
            # Even for <=75 tokens, apply NoBOS processing
            return self._encode_prompt_nobos_single_chunk(prompt, negative_prompt, pipeline)

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
        from core.prompts.prompt_parser import parse_prompt_attention, apply_emphasis_to_embeds

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

    def generate_txt2img(self, params: Dict[str, Any], progress_callback=None, step_callback=None) -> tuple[Image.Image, int, int]:
        """Generate image from text

        Args:
            params: Generation parameters
            progress_callback: Legacy callback for progress (step, timestep, latents)
            step_callback: New style callback for step-based control (pipe, step, timestep, callback_kwargs)

        Returns:
            tuple: (image, actual_seed, actual_ancestral_seed)
        """
        # Z-Image handling
        if self.is_zimage_model:
            return self._generate_txt2img_zimage(params, progress_callback, step_callback)

        # FLUX.2 Klein handling (MMDiT with Qwen3 text encoder)
        if self.is_flux2_model:
            return self._generate_txt2img_flux2(params, progress_callback, step_callback)

        # Anima handling (Cosmos-Predict2 DiT + Qwen3 + Qwen-Image VAE)
        if self.is_anima_model:
            return self._generate_txt2img_anima(params, progress_callback, step_callback)

        # Lens handling (Microsoft/Lens MMDiT)
        if self.is_lens_model:
            return self._generate_txt2img_lens(params, progress_callback, step_callback)

        # Ideogram 4 handling (dual-branch single-stream DiT)
        if self.is_ideogram4_model:
            return self._generate_txt2img_ideogram4(params, progress_callback, step_callback)

        # MiniT2I handling (pixel-space MM-JiT)
        if self.is_minit2i_model:
            return self._generate_txt2img_minit2i(params, progress_callback, step_callback)

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

        # ===== STAGE 1: TEXT ENCODING =====
        from core.vram_optimization import log_device_status, move_text_encoders_to_gpu, move_text_encoders_to_cpu

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        if not cpu_text_encoding:
            move_text_encoders_to_gpu(self.txt2img_pipeline)
        log_device_status("Ready for text encoding", self.txt2img_pipeline, vision_encoder=getattr(self, 'vision_encoder', None))

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

        # Encode NAG negative prompt if NAG is enabled
        nag_negative_prompt_embeds = None
        nag_negative_pooled_prompt_embeds = None
        if params.get("nag_enable", False):
            nag_negative_prompt = params.get("nag_negative_prompt", "")
            # If NAG negative prompt is empty, use the main negative prompt
            if not nag_negative_prompt:
                nag_negative_prompt = params.get("negative_prompt", "")

            print(f"[NAG] Encoding NAG negative prompt: '{nag_negative_prompt[:100]}...'")
            # Encode NAG negative prompt (positive part is ignored, only need negative)
            _, nag_negative_prompt_embeds, _, nag_negative_pooled_prompt_embeds = self._encode_prompt_with_weights(
                "",  # Empty positive prompt
                nag_negative_prompt,
                pipeline=self.txt2img_pipeline
            )
            print(f"[NAG] NAG negative embeddings shape: {nag_negative_prompt_embeds.shape if nag_negative_prompt_embeds is not None else None}")

        # Pre-calculate all prompt editing embeddings if needed
        embeds_cache = {}
        if prompt_processor:
            print("[PromptEditing] Pre-calculating all prompt variations...")
            all_prompts = prompt_processor.get_all_prompts(params.get("steps", settings.default_steps))
            for prompt_text in all_prompts:
                if prompt_text not in embeds_cache:
                    edit_embeds, edit_neg_embeds, edit_pooled, edit_neg_pooled = self._encode_prompt_with_weights(
                        prompt_text,
                        params.get("negative_prompt", ""),
                        pipeline=self.txt2img_pipeline
                    )
                    # Keep prompt editing embeddings on CPU to save VRAM
                    # They will be moved to GPU on-demand in the callback
                    embeds_cache[prompt_text] = (
                        edit_embeds.to('cpu') if edit_embeds is not None else None,
                        edit_neg_embeds.to('cpu') if edit_neg_embeds is not None else None,
                        edit_pooled.to('cpu') if edit_pooled is not None else None,
                        edit_neg_pooled.to('cpu') if edit_neg_pooled is not None else None
                    )
            print(f"[PromptEditing] Pre-calculated {len(embeds_cache)} prompt variations (stored on CPU)")

        # Ensure main embeddings are on GPU before offloading text encoders
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)
        if nag_negative_prompt_embeds is not None:
            nag_negative_prompt_embeds = nag_negative_prompt_embeds.to(device)
        if nag_negative_pooled_prompt_embeds is not None:
            nag_negative_pooled_prompt_embeds = nag_negative_pooled_prompt_embeds.to(device)

        # Offload text encoders to CPU after all encoding is complete
        move_text_encoders_to_cpu(self.txt2img_pipeline)

        # ===== STAGE 1.5: VISION ENCODER (optional) =====
        # Apply vision encoder if loaded and reference images are provided.
        # Skipped for FLUX.2 (handled separately via encode_flux2_image_refs).
        _ve_ref_images = params.get("ref_images", [])
        if (
            self.vision_encoder is not None
            and _ve_ref_images
            and prompt_embeds is not None
            and negative_prompt_embeds is not None
        ):
            prompt_embeds, negative_prompt_embeds, nag_negative_prompt_embeds = \
                self._apply_vision_encoder(
                    prompt_embeds,
                    negative_prompt_embeds,
                    _ve_ref_images,
                    nag_negative_prompt_embeds=nag_negative_prompt_embeds,
                )
            print(f"[txt2img][VE] Combined prompt embeddings shape: {prompt_embeds.shape}")
            print(f"[txt2img][VE] Combined negative embeddings shape: {negative_prompt_embeds.shape}")

        # ===== STAGE 2: U-NET INFERENCE =====
        from core.vram_optimization import move_unet_to_gpu

        # Get quantization option from params
        unet_quantization = params.get("unet_quantization", None)
        use_torch_compile = params.get("use_torch_compile", False)
        print(f"[Pipeline] U-Net quantization parameter: {repr(unet_quantization)}")
        print(f"[Pipeline] torch.compile parameter: {use_torch_compile}")
        if unet_quantization and unet_quantization != "none":
            print(f"[Pipeline] Applying U-Net quantization: {unet_quantization}")
        move_unet_to_gpu(self.txt2img_pipeline, quantization=unet_quantization, use_torch_compile=use_torch_compile)

        log_device_status("Ready for U-Net inference", self.txt2img_pipeline, vision_encoder=getattr(self, 'vision_encoder', None))

        # Handle ControlNet and Reference Guide
        all_controlnet_images = params.get("controlnet_images", [])
        # Separate Reference Guide entries from ControlNet entries
        ref_guide_configs = [c for c in all_controlnet_images if c.get("is_reference_guide")]
        controlnet_images = [c for c in all_controlnet_images if not c.get("is_reference_guide")]
        pipeline_to_use = self.txt2img_pipeline

        if ref_guide_configs:
            print(f"[RefGuide] Found {len(ref_guide_configs)} reference guide(s)")

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
            actual_seed = random.randint(0, 2**32 - 1)
        else:
            actual_seed = seed

        generator = torch.Generator(device=self.device).manual_seed(actual_seed)

        # Create ancestral generator for stochastic samplers
        ancestral_seed = params.get("ancestral_seed", -1)
        if ancestral_seed == -1:
            # Generate random seed for ancestral sampling (reproducible when saved)
            actual_ancestral_seed = random.randint(0, 2147483647)
            ancestral_generator = torch.Generator(device=self.device).manual_seed(actual_ancestral_seed)
            print(f"[Pipeline] Generated random ancestral seed: {actual_ancestral_seed}")
        else:
            # Use specified seed for ancestral sampling
            actual_ancestral_seed = ancestral_seed
            ancestral_generator = torch.Generator(device=self.device).manual_seed(ancestral_seed)
            print(f"[Pipeline] Using specified ancestral seed: {ancestral_seed}")

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
            # embeds_cache is already pre-calculated above with all variations
            prompt_embeds_callback_fn = None
            if prompt_processor:
                def prompt_embeds_callback_fn(step_index):
                    new_prompt = prompt_processor.get_prompt_at_step(step_index, params.get("steps", settings.default_steps))
                    if new_prompt is not None and new_prompt in embeds_cache:
                        # Move embeddings from CPU to GPU on-demand
                        cpu_embeds = embeds_cache[new_prompt]
                        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                        gpu_embeds = (
                            cpu_embeds[0].to(device) if cpu_embeds[0] is not None else None,
                            cpu_embeds[1].to(device) if cpu_embeds[1] is not None else None,
                            cpu_embeds[2].to(device) if cpu_embeds[2] is not None else None,
                            cpu_embeds[3].to(device) if cpu_embeds[3] is not None else None
                        )
                        return gpu_embeds
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

            # Detect v-prediction and apply guidance_rescale if needed
            is_v_prediction = pipeline_to_use.scheduler.config.get("prediction_type") == "v_prediction"
            guidance_rescale = 0.7 if is_v_prediction else 0.0
            if is_v_prediction:
                print(f"[Pipeline] V-prediction model detected, applying guidance_rescale={guidance_rescale}")

            # Set attention processor based on attention_type (unless NAG is enabled)
            # NAG has its own processors that will be set in custom_sampling_loop
            attention_type = params.get("attention_type", "normal")

            # Only switch if attention type has changed and NAG is not enabled (avoid redundant switching overhead)
            if not params.get("nag_enable", False):
                if attention_type != "normal" and attention_type != self.current_attention_type:
                    print(f"[Pipeline] Switching attention processor: {self.current_attention_type} -> {attention_type}")
                    from core.inference.attention_processors import set_attention_processor
                    self.original_processors = set_attention_processor(pipeline_to_use.unet, attention_type)
                    self.current_attention_type = attention_type
                elif attention_type == "normal" and self.current_attention_type != "normal":
                    print(f"[Pipeline] Restoring original attention processors (normal mode)")
                    if self.original_processors is not None:
                        pipeline_to_use.unet.set_attn_processor(self.original_processors)
                        self.original_processors = None
                    self.current_attention_type = "normal"
                else:
                    print(f"[Pipeline] Attention processor already set to: {attention_type} (skipping)")

            # Call custom sampling loop
            image = custom_sampling_loop(
                pipeline=pipeline_to_use,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=params.get("steps", settings.default_steps),
                guidance_scale=params.get("cfg_scale", settings.default_cfg_scale),
                guidance_rescale=guidance_rescale,
                width=params.get("width", 1024 if is_sdxl else settings.default_width),
                height=params.get("height", 1024 if is_sdxl else settings.default_height),
                generator=generator,
                ancestral_generator=ancestral_generator,
                latents=None,
                prompt_embeds_callback=prompt_embeds_callback_fn,
                progress_callback=progress_callback,
                step_callback=step_callback,
                developer_mode=params.get("developer_mode", False),
                cfg_schedule_type=params.get("cfg_schedule_type", "constant"),
                cfg_schedule_min=params.get("cfg_schedule_min", 1.0),
                cfg_schedule_max=params.get("cfg_schedule_max", None),
                cfg_schedule_power=params.get("cfg_schedule_power", 2.0),
                cfg_rescale_snr_alpha=params.get("cfg_rescale_snr_alpha", 0.0),
                dynamic_threshold_percentile=params.get("dynamic_threshold_percentile", 0.0),
                dynamic_threshold_mimic_scale=params.get("dynamic_threshold_mimic_scale", 1.0),
                nag_enable=params.get("nag_enable", False),
                nag_scale=params.get("nag_scale", 5.0),
                nag_tau=params.get("nag_tau", 3.5),
                nag_alpha=params.get("nag_alpha", 0.25),
                nag_sigma_end=params.get("nag_sigma_end", 0.0),
                nag_negative_prompt_embeds=nag_negative_prompt_embeds,
                nag_negative_pooled_prompt_embeds=nag_negative_pooled_prompt_embeds,
                attention_type=attention_type,
                ref_guide_configs=ref_guide_configs if ref_guide_configs else None,
                vision_encoder=getattr(self, 'vision_encoder', None),
                **controlnet_kwargs,
            )

        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Restore original attention processors if they were changed
            if self.original_processors is not None:
                from core.inference.attention_processors import restore_processors
                restore_processors(pipeline_to_use.unet, self.original_processors)
                self.original_processors = None

            # Delete GPU embed tensors
            prompt_embeds = None
            negative_prompt_embeds = None
            pooled_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            nag_negative_prompt_embeds = None
            nag_negative_pooled_prompt_embeds = None

            # Offload all components to CPU to free VRAM
            from core.vram_optimization import move_text_encoders_to_cpu, move_unet_to_cpu, move_vae_to_cpu
            move_text_encoders_to_cpu(pipeline_to_use)
            move_unet_to_cpu(pipeline_to_use)
            move_vae_to_cpu(pipeline_to_use)

            # Move TAESD preview decoder to CPU
            from core.utils.taesd import taesd_manager
            taesd_manager.offload_to_cpu()

            print("[VRAM] All components offloaded to CPU after txt2img generation")

            # Clear embeds_cache to prevent VRAM leak from prompt editing closures
            if 'embeds_cache' in dir() and embeds_cache:
                for key in list(embeds_cache.keys()):
                    tensors = embeds_cache[key]
                    if tensors:
                        for tensor in tensors:
                            if tensor is not None:
                                del tensor
                    del embeds_cache[key]
                embeds_cache.clear()
                print("[VRAM] Cleared embeds_cache for prompt editing")

            # Final cache clear
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        # Apply extensions after generation
        for ext in self.extensions:
            if ext.enabled:
                image = ext.process_after_generation(image, params)

        return image, actual_seed, actual_ancestral_seed

    def generate_img2img(self, params: Dict[str, Any], init_image: Image.Image, progress_callback=None, step_callback=None) -> tuple[Image.Image, int]:
        """Generate image from image

        Returns:
            tuple: (image, actual_seed)
        """
        # Z-Image handling
        if self.is_zimage_model:
            return self._generate_img2img_zimage(params, init_image, progress_callback, step_callback)

        # FLUX.2 Klein handling
        if self.is_flux2_model:
            return self._generate_img2img_flux2(params, init_image, progress_callback, step_callback)

        # Anima handling
        if self.is_anima_model:
            return self._generate_img2img_anima(params, init_image, progress_callback, step_callback)

        # Lens handling
        if self.is_lens_model:
            return self._generate_img2img_lens(params, init_image, progress_callback, step_callback)

        if self.is_ideogram4_model:
            return self._generate_img2img_ideogram4(params, init_image, progress_callback, step_callback)

        if self.is_minit2i_model:
            return self._generate_img2img_minit2i(params, init_image, progress_callback, step_callback)

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

        # Ensure dimensions are multiples of 8 (required for VAE)
        if target_width:
            target_width = round(target_width / 8) * 8
            params["width"] = target_width
        if target_height:
            target_height = round(target_height / 8) * 8
            params["height"] = target_height

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

        # ===== STAGE 1: TEXT ENCODING =====
        from core.vram_optimization import log_device_status, move_text_encoders_to_gpu, move_text_encoders_to_cpu, move_vae_to_gpu, move_vae_to_cpu

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        if not cpu_text_encoding:
            move_text_encoders_to_gpu(self.img2img_pipeline)
        log_device_status("Ready for text encoding (img2img)", self.img2img_pipeline, vision_encoder=getattr(self, 'vision_encoder', None))

        # Handle ControlNet and Reference Guide
        all_controlnet_images = params.get("controlnet_images", [])
        ref_guide_configs = [c for c in all_controlnet_images if c.get("is_reference_guide")]
        controlnet_images = [c for c in all_controlnet_images if not c.get("is_reference_guide")]
        pipeline_to_use = self.img2img_pipeline

        if ref_guide_configs:
            print(f"[RefGuide] Found {len(ref_guide_configs)} reference guide(s) for img2img")

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

        # Log embedding shapes for debugging
        if prompt_embeds is not None:
            print(f"[img2img] Prompt embeddings shape: {prompt_embeds.shape}")
        if negative_prompt_embeds is not None:
            print(f"[img2img] Negative prompt embeddings shape: {negative_prompt_embeds.shape}")
        if pooled_prompt_embeds is not None:
            print(f"[img2img] Pooled prompt embeddings shape: {pooled_prompt_embeds.shape}")
        if negative_pooled_prompt_embeds is not None:
            print(f"[img2img] Negative pooled prompt embeddings shape: {negative_pooled_prompt_embeds.shape}")

        # Encode NAG negative prompt if NAG is enabled
        nag_negative_prompt_embeds = None
        nag_negative_pooled_prompt_embeds = None
        if params.get("nag_enable", False):
            nag_negative_prompt = params.get("nag_negative_prompt", "")
            # If NAG negative prompt is empty, use the main negative prompt
            if not nag_negative_prompt:
                nag_negative_prompt = params.get("negative_prompt", "")

            print(f"[NAG] Encoding NAG negative prompt: '{nag_negative_prompt[:100]}...'")
            # Encode NAG negative prompt (positive part is ignored, only need negative)
            _, nag_negative_prompt_embeds, _, nag_negative_pooled_prompt_embeds = self._encode_prompt_with_weights(
                "",  # Empty positive prompt
                nag_negative_prompt,
                pipeline=pipeline_to_use
            )
            print(f"[NAG] NAG negative embeddings shape: {nag_negative_prompt_embeds.shape}")

        # Pre-calculate all prompt editing embeddings if needed
        embeds_cache = {}
        if prompt_processor:
            print("[PromptEditing] Pre-calculating all prompt variations...")
            all_prompts = prompt_processor.get_all_prompts(params.get("steps", settings.default_steps))
            for prompt_text in all_prompts:
                if prompt_text not in embeds_cache:
                    edit_embeds, edit_neg_embeds, edit_pooled, edit_neg_pooled = self._encode_prompt_with_weights(
                        prompt_text,
                        params.get("negative_prompt", ""),
                        pipeline=pipeline_to_use
                    )
                    # Keep prompt editing embeddings on CPU to save VRAM
                    embeds_cache[prompt_text] = (
                        edit_embeds.to('cpu') if edit_embeds is not None else None,
                        edit_neg_embeds.to('cpu') if edit_neg_embeds is not None else None,
                        edit_pooled.to('cpu') if edit_pooled is not None else None,
                        edit_neg_pooled.to('cpu') if edit_neg_pooled is not None else None
                    )
            print(f"[PromptEditing] Pre-calculated {len(embeds_cache)} prompt variations (stored on CPU)")

        # Ensure main embeddings are on GPU before offloading text encoders
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)
        if nag_negative_prompt_embeds is not None:
            nag_negative_prompt_embeds = nag_negative_prompt_embeds.to(device)
        if nag_negative_pooled_prompt_embeds is not None:
            nag_negative_pooled_prompt_embeds = nag_negative_pooled_prompt_embeds.to(device)

        # Offload text encoders to CPU after all encoding is complete
        move_text_encoders_to_cpu(pipeline_to_use)

        # ===== STAGE 1.5: VISION ENCODER (optional) =====
        _ve_ref_images = params.get("ref_images", [])
        if (
            self.vision_encoder is not None
            and _ve_ref_images
            and prompt_embeds is not None
            and negative_prompt_embeds is not None
        ):
            prompt_embeds, negative_prompt_embeds, nag_negative_prompt_embeds = \
                self._apply_vision_encoder(
                    prompt_embeds,
                    negative_prompt_embeds,
                    _ve_ref_images,
                    nag_negative_prompt_embeds=nag_negative_prompt_embeds,
                )
            print(f"[img2img][VE] Combined prompt embeddings shape: {prompt_embeds.shape}")
            print(f"[img2img][VE] Combined negative embeddings shape: {negative_prompt_embeds.shape}")

        # ===== STAGE 2: U-NET INFERENCE (after VAE operations) =====
        # Note: For img2img, we need VAE first for initial latent encoding

        # Handle latent resize mode by encoding, resizing latent, then decoding
        if resize_mode == "latent" and target_width and target_height:
            if init_image.size != (target_width, target_height):
                print(f"Using latent resize mode: {init_image.size} -> {target_width}x{target_height} with {resampling_method}")

                # Encode image to latent space
                import torch.nn.functional as F

                # Move VAE to GPU for latent resize encoding/decoding
                move_vae_to_gpu(pipeline_to_use)

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

                # Clean up intermediate tensors from latent resize
                del image_tensor, latent, resized_latent, decoded

                # Move VAE back to CPU after latent resize operations
                move_vae_to_cpu(pipeline_to_use)
                torch.cuda.empty_cache()

        # Calculate proper steps for img2img
        requested_steps = params.get("steps", settings.default_steps)
        denoising_strength = params.get("denoising_strength", 0.75)
        fix_steps = params.get("img2img_fix_steps", True)
        total_steps, t_start, actual_steps = self._setup_img2img_steps(requested_steps, denoising_strength, fix_steps)

        if fix_steps:
            print(f"[img2img] Do full steps enabled: {requested_steps} requested -> {total_steps} scheduler steps, t_start={t_start}, actual={actual_steps}")

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
            # embeds_cache is already pre-calculated above with all variations
            prompt_embeds_callback_fn = None
            if prompt_processor:
                def prompt_embeds_callback_fn(step_index):
                    new_prompt = prompt_processor.get_prompt_at_step(step_index, total_steps)
                    if new_prompt is not None and new_prompt in embeds_cache:
                        # Move embeddings from CPU to GPU on-demand
                        cpu_embeds = embeds_cache[new_prompt]
                        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                        gpu_embeds = (
                            cpu_embeds[0].to(device) if cpu_embeds[0] is not None else None,
                            cpu_embeds[1].to(device) if cpu_embeds[1] is not None else None,
                            cpu_embeds[2].to(device) if cpu_embeds[2] is not None else None,
                            cpu_embeds[3].to(device) if cpu_embeds[3] is not None else None
                        )
                        return gpu_embeds
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
                # Generate random ancestral seed for reproducibility tracking
                actual_ancestral_seed = random.randint(0, 2147483647)
                ancestral_generator = torch.Generator(device=self.device).manual_seed(actual_ancestral_seed)
                print(f"[Pipeline] Generated random ancestral seed: {actual_ancestral_seed}")
            else:
                # Use specified ancestral seed
                actual_ancestral_seed = ancestral_seed
                ancestral_generator = torch.Generator(device=self.device).manual_seed(ancestral_seed)
                print(f"[Pipeline] Using specified ancestral seed: {ancestral_seed}")

            # Detect v-prediction and apply guidance_rescale if needed
            is_v_prediction = pipeline_to_use.scheduler.config.get("prediction_type") == "v_prediction"
            guidance_rescale = 0.7 if is_v_prediction else 0.0
            if is_v_prediction:
                print(f"[Pipeline] V-prediction model detected, applying guidance_rescale={guidance_rescale}")

            # Set attention processor based on attention_type (unless NAG is enabled)
            # NAG has its own processors that will be set in custom_sampling_loop
            attention_type = params.get("attention_type", "normal")

            # Only switch if attention type has changed and NAG is not enabled (avoid redundant switching overhead)
            if not params.get("nag_enable", False):
                if attention_type != "normal" and attention_type != self.current_attention_type:
                    print(f"[Pipeline] Switching attention processor: {self.current_attention_type} -> {attention_type}")
                    from core.inference.attention_processors import set_attention_processor
                    self.original_processors = set_attention_processor(pipeline_to_use.unet, attention_type)
                    self.current_attention_type = attention_type
                elif attention_type == "normal" and self.current_attention_type != "normal":
                    print(f"[Pipeline] Restoring original attention processors (normal mode)")
                    if self.original_processors is not None:
                        pipeline_to_use.unet.set_attn_processor(self.original_processors)
                        self.original_processors = None
                    self.current_attention_type = "normal"
                else:
                    print(f"[Pipeline] Attention processor already set to: {attention_type} (skipping)")

            # Use t_start directly for custom sampling loop
            t_start_override = t_start if fix_steps else None
            if fix_steps:
                print(f"[img2img] Using t_start={t_start_override} for Do full steps mode")

            # Move U-Net to GPU for inference
            from core.vram_optimization import move_unet_to_gpu

            # Get quantization option from params
            unet_quantization = params.get("unet_quantization", None)
            use_torch_compile = params.get("use_torch_compile", False)
            move_unet_to_gpu(pipeline_to_use, quantization=unet_quantization, use_torch_compile=use_torch_compile)

            log_device_status("Ready for U-Net inference (img2img)", pipeline_to_use, vision_encoder=getattr(self, 'vision_encoder', None))

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
                guidance_rescale=guidance_rescale,
                generator=torch.Generator(device=self.device).manual_seed(actual_seed),
                ancestral_generator=ancestral_generator,
                t_start_override=t_start_override,
                prompt_embeds_callback=prompt_embeds_callback_fn,
                progress_callback=progress_callback,
                step_callback=step_callback,
                developer_mode=params.get("developer_mode", False),
                cfg_schedule_type=params.get("cfg_schedule_type", "constant"),
                cfg_schedule_min=params.get("cfg_schedule_min", 1.0),
                cfg_schedule_max=params.get("cfg_schedule_max", None),
                cfg_schedule_power=params.get("cfg_schedule_power", 2.0),
                cfg_rescale_snr_alpha=params.get("cfg_rescale_snr_alpha", 0.0),
                dynamic_threshold_percentile=params.get("dynamic_threshold_percentile", 0.0),
                dynamic_threshold_mimic_scale=params.get("dynamic_threshold_mimic_scale", 1.0),
                nag_enable=params.get("nag_enable", False),
                nag_scale=params.get("nag_scale", 5.0),
                nag_tau=params.get("nag_tau", 3.5),
                nag_alpha=params.get("nag_alpha", 0.25),
                nag_sigma_end=params.get("nag_sigma_end", 3.0),
                nag_negative_prompt_embeds=nag_negative_prompt_embeds,
                nag_negative_pooled_prompt_embeds=nag_negative_pooled_prompt_embeds,
                attention_type=attention_type,
                ref_guide_configs=ref_guide_configs if ref_guide_configs else None,
                vision_encoder=getattr(self, 'vision_encoder', None),
                **controlnet_kwargs,
            )

        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Restore original attention processors if they were changed
            if self.original_processors is not None:
                from core.inference.attention_processors import restore_processors
                restore_processors(pipeline_to_use.unet, self.original_processors)
                self.original_processors = None

            # Delete GPU embed tensors
            prompt_embeds = None
            negative_prompt_embeds = None
            pooled_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            nag_negative_prompt_embeds = None
            nag_negative_pooled_prompt_embeds = None

            # Offload all components to CPU to free VRAM
            from core.vram_optimization import move_text_encoders_to_cpu, move_unet_to_cpu, move_vae_to_cpu
            move_text_encoders_to_cpu(pipeline_to_use)
            move_unet_to_cpu(pipeline_to_use)
            move_vae_to_cpu(pipeline_to_use)

            # Move TAESD preview decoder to CPU
            from core.utils.taesd import taesd_manager
            taesd_manager.offload_to_cpu()

            print("[VRAM] All components offloaded to CPU after img2img generation")

            # Clear embeds_cache to prevent VRAM leak from prompt editing closures
            if 'embeds_cache' in dir() and embeds_cache:
                for key in list(embeds_cache.keys()):
                    tensors = embeds_cache[key]
                    if tensors:
                        for tensor in tensors:
                            if tensor is not None:
                                del tensor
                    del embeds_cache[key]
                embeds_cache.clear()
                print("[VRAM] Cleared embeds_cache for prompt editing")

            # Final cache clear
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        # Apply extensions after generation
        for ext in self.extensions:
            if ext.enabled:
                image = ext.process_after_generation(image, params)

        return image, actual_seed, actual_ancestral_seed

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
        # Z-Image inpaint support
        if self.is_zimage_model:
            return self._generate_inpaint_zimage(params, init_image, mask_image, progress_callback, step_callback)

        # FLUX.2 Klein inpaint support
        if self.is_flux2_model:
            return self._generate_inpaint_flux2(params, init_image, mask_image, progress_callback, step_callback)

        # Anima inpaint support
        if self.is_anima_model:
            return self._generate_inpaint_anima(params, init_image, mask_image, progress_callback, step_callback)

        # Lens inpaint (repaint approach)
        if self.is_lens_model:
            return self._generate_inpaint_lens(params, init_image, mask_image, progress_callback, step_callback)

        # Ideogram 4 inpaint (repaint approach)
        if self.is_ideogram4_model:
            return self._generate_inpaint_ideogram4(params, init_image, mask_image, progress_callback, step_callback)

        # MiniT2I inpaint (repaint approach)
        if self.is_minit2i_model:
            return self._generate_inpaint_minit2i(params, init_image, mask_image, progress_callback, step_callback)

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

        # Create ancestral generator for stochastic samplers
        ancestral_seed = params.get("ancestral_seed", -1)
        if ancestral_seed == -1:
            # Generate random seed for ancestral sampling (reproducible when saved)
            actual_ancestral_seed = random.randint(0, 2147483647)
            ancestral_generator = torch.Generator(device=self.device).manual_seed(actual_ancestral_seed)
            print(f"[Pipeline] Generated random ancestral seed: {actual_ancestral_seed}")
        else:
            # Use specified seed for ancestral sampling
            actual_ancestral_seed = ancestral_seed
            ancestral_generator = torch.Generator(device=self.device).manual_seed(ancestral_seed)
            print(f"[Pipeline] Using specified ancestral seed: {ancestral_seed}")

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
        total_steps, t_start, actual_steps = self._setup_img2img_steps(requested_steps, denoising_strength, fix_steps)

        if fix_steps:
            print(f"[inpaint] Do full steps enabled: {requested_steps} requested -> {total_steps} scheduler steps, t_start={t_start}, actual={actual_steps}")

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

        # ===== STAGE 1: TEXT ENCODING =====
        from core.vram_optimization import log_device_status, move_text_encoders_to_gpu, move_text_encoders_to_cpu, move_vae_to_gpu, move_vae_to_cpu

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        if not cpu_text_encoding:
            move_text_encoders_to_gpu(self.inpaint_pipeline)
        log_device_status("Ready for text encoding (inpaint)", self.inpaint_pipeline, vision_encoder=getattr(self, 'vision_encoder', None))

        # Determine if SDXL
        is_sdxl = isinstance(self.inpaint_pipeline, StableDiffusionXLInpaintPipeline)

        # Handle ControlNet and Reference Guide
        all_controlnet_images = params.get("controlnet_images", [])
        ref_guide_configs = [c for c in all_controlnet_images if c.get("is_reference_guide")]
        controlnet_images = [c for c in all_controlnet_images if not c.get("is_reference_guide")]
        pipeline_to_use = self.inpaint_pipeline

        if ref_guide_configs:
            print(f"[RefGuide] Found {len(ref_guide_configs)} reference guide(s) for inpaint")

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

        # Log embedding shapes for debugging
        if prompt_embeds is not None:
            print(f"[inpaint] Prompt embeddings shape: {prompt_embeds.shape}")
        if negative_prompt_embeds is not None:
            print(f"[inpaint] Negative prompt embeddings shape: {negative_prompt_embeds.shape}")
        if pooled_prompt_embeds is not None:
            print(f"[inpaint] Pooled prompt embeddings shape: {pooled_prompt_embeds.shape}")
        if negative_pooled_prompt_embeds is not None:
            print(f"[inpaint] Negative pooled prompt embeddings shape: {negative_pooled_prompt_embeds.shape}")

        # Pre-calculate all prompt editing embeddings if needed
        embeds_cache = {}
        if prompt_processor:
            print("[PromptEditing] Pre-calculating all prompt variations...")
            all_prompts = prompt_processor.get_all_prompts(total_steps)
            for prompt_text in all_prompts:
                if prompt_text not in embeds_cache:
                    edit_embeds, edit_neg_embeds, edit_pooled, edit_neg_pooled = self._encode_prompt_with_weights(
                        prompt_text,
                        params.get("negative_prompt", ""),
                        pipeline=pipeline_to_use
                    )
                    # Keep prompt editing embeddings on CPU to save VRAM
                    embeds_cache[prompt_text] = (
                        edit_embeds.to('cpu') if edit_embeds is not None else None,
                        edit_neg_embeds.to('cpu') if edit_neg_embeds is not None else None,
                        edit_pooled.to('cpu') if edit_pooled is not None else None,
                        edit_neg_pooled.to('cpu') if edit_neg_pooled is not None else None
                    )
            print(f"[PromptEditing] Pre-calculated {len(embeds_cache)} prompt variations (stored on CPU)")

        # Encode NAG negative prompt if NAG is enabled
        nag_negative_prompt_embeds = None
        nag_negative_pooled_prompt_embeds = None
        if params.get("nag_enable", False):
            nag_negative_prompt = params.get("nag_negative_prompt", "")
            if not nag_negative_prompt:
                nag_negative_prompt = params.get("negative_prompt", "")

            print(f"[NAG] Encoding NAG negative prompt: '{nag_negative_prompt[:100]}...'")
            _, nag_negative_prompt_embeds, _, nag_negative_pooled_prompt_embeds = self._encode_prompt_with_weights(
                "",  # Empty positive prompt
                nag_negative_prompt,
                pipeline=pipeline_to_use
            )
            print(f"[NAG] NAG negative embeddings shape: {nag_negative_prompt_embeds.shape}")

        # Move embeddings to device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        prompt_embeds = prompt_embeds.to(device)
        negative_prompt_embeds = negative_prompt_embeds.to(device)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)
        if nag_negative_prompt_embeds is not None:
            nag_negative_prompt_embeds = nag_negative_prompt_embeds.to(device)
        if nag_negative_pooled_prompt_embeds is not None:
            nag_negative_pooled_prompt_embeds = nag_negative_pooled_prompt_embeds.to(device)

        # Offload text encoders to CPU after all encoding is complete
        move_text_encoders_to_cpu(pipeline_to_use)

        # ===== STAGE 1.5: VISION ENCODER (optional) =====
        _ve_ref_images = params.get("ref_images", [])
        if (
            self.vision_encoder is not None
            and _ve_ref_images
            and prompt_embeds is not None
            and negative_prompt_embeds is not None
        ):
            prompt_embeds, negative_prompt_embeds, nag_negative_prompt_embeds = \
                self._apply_vision_encoder(
                    prompt_embeds,
                    negative_prompt_embeds,
                    _ve_ref_images,
                    nag_negative_prompt_embeds=nag_negative_prompt_embeds,
                )
            print(f"[inpaint][VE] Combined prompt embeddings shape: {prompt_embeds.shape}")
            print(f"[inpaint][VE] Combined negative embeddings shape: {negative_prompt_embeds.shape}")

        # Prepare callback for prompt editing
        prompt_embeds_callback_fn = None
        if prompt_processor:
            def prompt_embeds_callback_fn(step_index):
                new_prompt = prompt_processor.get_prompt_at_step(step_index, total_steps)
                if new_prompt is not None and new_prompt in embeds_cache:
                    # Move embeddings from CPU to GPU on-demand
                    cpu_embeds = embeds_cache[new_prompt]
                    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                    gpu_embeds = (
                        cpu_embeds[0].to(device) if cpu_embeds[0] is not None else None,
                        cpu_embeds[1].to(device) if cpu_embeds[1] is not None else None,
                        cpu_embeds[2].to(device) if cpu_embeds[2] is not None else None,
                        cpu_embeds[3].to(device) if cpu_embeds[3] is not None else None
                    )
                    return gpu_embeds
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

        # Detect v-prediction and apply guidance_rescale if needed
        is_v_prediction = pipeline_to_use.scheduler.config.get("prediction_type") == "v_prediction"
        guidance_rescale = 0.7 if is_v_prediction else 0.0
        if is_v_prediction:
            print(f"[Pipeline] V-prediction model detected, applying guidance_rescale={guidance_rescale}")

        # Set attention processor based on attention_type (unless NAG is enabled)
        # NAG has its own processors that will be set in custom_sampling_loop
        attention_type = params.get("attention_type", "normal")

        # Only switch if attention type has changed and NAG is not enabled (avoid redundant switching overhead)
        if not params.get("nag_enable", False):
            if attention_type != "normal" and attention_type != self.current_attention_type:
                print(f"[Pipeline] Switching attention processor: {self.current_attention_type} -> {attention_type}")
                from core.inference.attention_processors import set_attention_processor
                self.original_processors = set_attention_processor(pipeline_to_use.unet, attention_type)
                self.current_attention_type = attention_type
            elif attention_type == "normal" and self.current_attention_type != "normal":
                print(f"[Pipeline] Restoring original attention processors (normal mode)")
                if self.original_processors is not None:
                    pipeline_to_use.unet.set_attn_processor(self.original_processors)
                    self.original_processors = None
                self.current_attention_type = "normal"
            else:
                print(f"[Pipeline] Attention processor already set to: {attention_type} (skipping)")

        # Use t_start directly for custom sampling loop
        t_start_override = t_start if fix_steps else None
        if fix_steps:
            print(f"[inpaint] Using t_start={t_start_override} for Do full steps mode")

        # ===== STAGE 2: U-NET INFERENCE =====
        from core.vram_optimization import move_unet_to_gpu

        # Get quantization option from params
        unet_quantization = params.get("unet_quantization", None)
        use_torch_compile = params.get("use_torch_compile", False)
        move_unet_to_gpu(pipeline_to_use, quantization=unet_quantization, use_torch_compile=use_torch_compile)

        log_device_status("Ready for U-Net inference (inpaint)", pipeline_to_use, vision_encoder=getattr(self, 'vision_encoder', None))

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
            guidance_rescale=guidance_rescale,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            ancestral_generator=ancestral_generator,
            t_start_override=t_start_override,
            prompt_embeds_callback=prompt_embeds_callback_fn,
            progress_callback=progress_callback,
            step_callback=step_callback,
            inpaint_fill_mode=params.get("inpaint_fill_mode", "original"),
            inpaint_fill_strength=params.get("inpaint_fill_strength", 1.0),
            inpaint_blur_strength=params.get("inpaint_blur_strength", 1.0),
            developer_mode=params.get("developer_mode", False),
            cfg_schedule_type=params.get("cfg_schedule_type", "constant"),
            cfg_schedule_min=params.get("cfg_schedule_min", 1.0),
            cfg_schedule_max=params.get("cfg_schedule_max", None),
            cfg_schedule_power=params.get("cfg_schedule_power", 2.0),
            cfg_rescale_snr_alpha=params.get("cfg_rescale_snr_alpha", 0.0),
            dynamic_threshold_percentile=params.get("dynamic_threshold_percentile", 0.0),
            dynamic_threshold_mimic_scale=params.get("dynamic_threshold_mimic_scale", 1.0),
            nag_enable=params.get("nag_enable", False),
            nag_scale=params.get("nag_scale", 5.0),
            nag_tau=params.get("nag_tau", 3.5),
            nag_alpha=params.get("nag_alpha", 0.25),
            nag_sigma_end=params.get("nag_sigma_end", 3.0),
            nag_negative_prompt_embeds=nag_negative_prompt_embeds,
            nag_negative_pooled_prompt_embeds=nag_negative_pooled_prompt_embeds,
            attention_type=params.get("attention_type", "normal"),
            ref_guide_configs=ref_guide_configs if ref_guide_configs else None,
            vision_encoder=getattr(self, 'vision_encoder', None),
            **controlnet_kwargs,
        )

        # Restore original attention processors if they were changed
        if self.original_processors is not None:
            from core.inference.attention_processors import restore_processors
            restore_processors(pipeline_to_use.unet, self.original_processors)
            self.original_processors = None

        # Delete GPU embed tensors
        prompt_embeds = None
        negative_prompt_embeds = None
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        nag_negative_prompt_embeds = None
        nag_negative_pooled_prompt_embeds = None

        # Offload all components to CPU to free VRAM
        from core.vram_optimization import move_text_encoders_to_cpu, move_unet_to_cpu, move_vae_to_cpu
        move_text_encoders_to_cpu(pipeline_to_use)
        move_unet_to_cpu(pipeline_to_use)
        move_vae_to_cpu(pipeline_to_use)

        # Move TAESD preview decoder to CPU
        from core.utils.taesd import taesd_manager
        taesd_manager.offload_to_cpu()

        print("[VRAM] All components offloaded to CPU after inpaint generation")

        # Clear embeds_cache to prevent VRAM leak from prompt editing closures
        if 'embeds_cache' in dir() and embeds_cache:
            for key in list(embeds_cache.keys()):
                tensors = embeds_cache[key]
                if tensors:
                    for tensor in tensors:
                        if tensor is not None:
                            del tensor
                del embeds_cache[key]
            embeds_cache.clear()
            print("[VRAM] Cleared embeds_cache for prompt editing")

        # Final cache clear
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Apply extensions after generation
        for ext in self.extensions:
            if ext.enabled:
                image = ext.process_after_generation(image, params)

        return image, seed, actual_ancestral_seed

    # =============================================================
    # Anima generation methods
    # =============================================================

    def _anima_resolve_dtype(self, dtype_str: Optional[str] = None) -> torch.dtype:
        if dtype_str == "fp16":
            return torch.float16
        if dtype_str == "fp32":
            return torch.float32
        return torch.bfloat16

    def _load_lora_anima(self, lora_configs: List[Dict]) -> int:
        """Wrap target Linear modules of the Anima DiT with LoRA adapters.

        Supports stacking multiple LoRAs on the same module (each subsequent
        wrap takes the existing wrapper's true original as its base, so
        unload always returns to the un-LoRA'd model).
        """
        from core.models.anima.anima_lora import (
            load_lora_safetensors, normalise_lora_state_dict, apply_lora_group,
        )
        from core.extensions.lora_manager import lora_manager

        if not lora_configs:
            return 0
        if not self.anima_components:
            print("[Anima LoRA] WARNING: components not loaded")
            return 0

        transformer = self.anima_components["transformer"]
        if not hasattr(self, "_anima_lora_original_modules"):
            self._anima_lora_original_modules: Dict[str, torch.nn.Linear] = {}
            self._anima_lora_wrapped_keys: set = set()

        total_applied = 0
        for i, cfg in enumerate(lora_configs):
            lora_path = cfg.get("path", "")
            strength = float(cfg.get("strength", 1.0))
            resolved = lora_manager._resolve_lora_path(lora_path)
            if resolved is None:
                print(f"[Anima LoRA] WARNING: file not found: {lora_path}")
                continue
            try:
                raw, fmt = load_lora_safetensors(str(resolved))
                grouped = normalise_lora_state_dict(raw)
                print(f"[Anima LoRA] {i+1}/{len(lora_configs)}: {lora_path} "
                      f"format={fmt} keys={len(raw)} matched_modules={len(grouped)} strength={strength}")
                applied = apply_lora_group(
                    transformer, grouped, strength,
                    self._anima_lora_original_modules, self._anima_lora_wrapped_keys,
                )
                print(f"[Anima LoRA]   wrapped {applied} module(s)")
                total_applied += applied
            except Exception as e:
                print(f"[Anima LoRA] ERROR loading {lora_path}: {e}")
                import traceback; traceback.print_exc()
        return total_applied

    def _unload_lora_anima(self) -> int:
        """Restore every Anima DiT Linear to its pre-LoRA original."""
        from core.models.anima.anima_lora import restore_originals
        if not getattr(self, "_anima_lora_wrapped_keys", None):
            return 0
        if not self.anima_components:
            return 0
        transformer = self.anima_components["transformer"]
        restored = restore_originals(
            transformer, self._anima_lora_original_modules, self._anima_lora_wrapped_keys,
        )
        print(f"[Anima LoRA] Unloaded {restored} LoRA wrappers")
        return restored

    @staticmethod
    def _anima_advanced_cfg(params: Dict[str, Any]) -> Dict[str, Any]:
        """Collect Advanced-CFG knobs from a generation params dict.

        Returns a dict consumed by anima_pipeline_ops._apply_advanced_cfg.
        Missing keys fall back to no-op defaults (constant CFG, no rescale,
        no threshold).
        """
        return {
            "cfg_schedule_type": params.get("cfg_schedule_type", "constant"),
            "cfg_schedule_min": params.get("cfg_schedule_min", 1.0),
            "cfg_schedule_max": params.get("cfg_schedule_max"),
            "cfg_schedule_power": params.get("cfg_schedule_power", 2.0),
            "cfg_rescale_snr_alpha": params.get("cfg_rescale_snr_alpha", 0.0),
            "dynamic_threshold_percentile": params.get("dynamic_threshold_percentile", 0.0),
            "dynamic_threshold_mimic_scale": params.get("dynamic_threshold_mimic_scale", 1.0),
            "developer_mode": params.get("developer_mode", False),
        }

    def _load_lora_lens(self, lora_configs: List[Dict]) -> int:
        """Wrap target Linear modules of the Lens transformer with LoRA adapters.

        Must be called after the transformer is on GPU (and optionally quantised).
        Supports stacking multiple LoRAs on the same module.
        """
        from core.models.lens.lens_lora import (
            load_lora_safetensors, normalise_lora_state_dict, apply_lora_group,
        )
        from core.extensions.lora_manager import lora_manager

        if not lora_configs:
            return 0
        if not self.lens_components:
            print("[Lens LoRA] WARNING: components not loaded")
            return 0

        transformer = self.lens_components["transformer"]
        if not hasattr(self, "_lens_lora_original_modules"):
            self._lens_lora_original_modules: Dict[str, torch.nn.Linear] = {}
            self._lens_lora_wrapped_keys: set = set()

        total_applied = 0
        for i, cfg in enumerate(lora_configs):
            lora_path = cfg.get("path", "")
            strength  = float(cfg.get("strength", 1.0))
            resolved  = lora_manager._resolve_lora_path(lora_path)
            if resolved is None:
                print(f"[Lens LoRA] WARNING: file not found: {lora_path}")
                continue
            try:
                raw, fmt = load_lora_safetensors(str(resolved))
                grouped  = normalise_lora_state_dict(raw)
                print(f"[Lens LoRA] {i+1}/{len(lora_configs)}: {lora_path} "
                      f"format={fmt} keys={len(raw)} matched_modules={len(grouped)} "
                      f"strength={strength}")
                applied = apply_lora_group(
                    transformer, grouped, strength,
                    self._lens_lora_original_modules, self._lens_lora_wrapped_keys,
                )
                print(f"[Lens LoRA]   wrapped {applied} module(s)")
                total_applied += applied
            except Exception as e:
                print(f"[Lens LoRA] ERROR loading {lora_path}: {e}")
                import traceback; traceback.print_exc()
        return total_applied

    def _unload_lora_lens(self) -> int:
        """Restore every Lens transformer Linear to its pre-LoRA original."""
        from core.models.lens.lens_lora import restore_originals
        if not getattr(self, "_lens_lora_wrapped_keys", None):
            return 0
        if not self.lens_components:
            return 0
        transformer = self.lens_components["transformer"]
        restored = restore_originals(
            transformer, self._lens_lora_original_modules, self._lens_lora_wrapped_keys,
        )
        print(f"[Lens LoRA] Unloaded {restored} LoRA wrappers")
        return restored

    def _load_lora_ideogram4(self, lora_configs: List[Dict]) -> int:
        """Wrap Ideogram 4 transformer Linear/Fp8Linear modules with LoRA adapters.

        Applies the conditional-branch LoRA to `transformer`; if the checkpoint
        also carries unconditional-branch keys (lora_uncond_*) and the
        unconditional transformer is loaded, those are applied to it too.
        Must be called after the transformer(s) are staged on GPU.
        """
        from core.models.ideogram4.ideogram4_lora import (
            load_lora_safetensors, normalise_lora_state_dict, apply_lora_group,
        )
        from core.extensions.lora_manager import lora_manager

        if not lora_configs or not self.ideogram4_components:
            return 0

        transformer = self.ideogram4_components["transformer"]
        uncond = self.ideogram4_components.get("unconditional_transformer")
        if not hasattr(self, "_ideogram4_lora_orig"):
            self._ideogram4_lora_orig: Dict[str, torch.nn.Module] = {}
            self._ideogram4_lora_keys: set = set()
            self._ideogram4_lora_orig_uncond: Dict[str, torch.nn.Module] = {}
            self._ideogram4_lora_keys_uncond: set = set()

        total = 0
        for i, cfg in enumerate(lora_configs):
            lora_path = cfg.get("path", "")
            strength = float(cfg.get("strength", 1.0))
            resolved = lora_manager._resolve_lora_path(lora_path)
            if resolved is None:
                print(f"[Ideogram4 LoRA] WARNING: file not found: {lora_path}")
                continue
            try:
                raw, fmt = load_lora_safetensors(str(resolved))
                grouped = normalise_lora_state_dict(raw, branch="cond")
                applied = apply_lora_group(
                    transformer, grouped, strength,
                    self._ideogram4_lora_orig, self._ideogram4_lora_keys,
                )
                print(f"[Ideogram4 LoRA] {i+1}/{len(lora_configs)}: {lora_path} "
                      f"format={fmt} cond_modules={len(grouped)} wrapped={applied} strength={strength}")
                total += applied

                if uncond is not None:
                    grouped_u = normalise_lora_state_dict(raw, branch="uncond")
                    if grouped_u:
                        applied_u = apply_lora_group(
                            uncond, grouped_u, strength,
                            self._ideogram4_lora_orig_uncond, self._ideogram4_lora_keys_uncond,
                        )
                        print(f"[Ideogram4 LoRA]   uncond wrapped {applied_u} module(s)")
                        total += applied_u
            except Exception as e:
                print(f"[Ideogram4 LoRA] ERROR loading {lora_path}: {e}")
                import traceback; traceback.print_exc()
        return total

    def _unload_lora_ideogram4(self) -> int:
        """Restore every Ideogram 4 transformer Linear to its pre-LoRA original."""
        from core.models.ideogram4.ideogram4_lora import restore_originals
        if not self.ideogram4_components:
            return 0
        restored = 0
        if getattr(self, "_ideogram4_lora_keys", None):
            restored += restore_originals(
                self.ideogram4_components["transformer"],
                self._ideogram4_lora_orig, self._ideogram4_lora_keys,
            )
        uncond = self.ideogram4_components.get("unconditional_transformer")
        if uncond is not None and getattr(self, "_ideogram4_lora_keys_uncond", None):
            restored += restore_originals(
                uncond, self._ideogram4_lora_orig_uncond, self._ideogram4_lora_keys_uncond,
            )
        if restored:
            print(f"[Ideogram4 LoRA] Unloaded {restored} LoRA wrappers")
        return restored

    @staticmethod
    def _lens_advanced_cfg(params: Dict[str, Any]) -> Dict[str, Any]:
        """Collect Advanced-CFG knobs for Lens generation.

        Returns a dict consumed by lens_pipeline_ops._apply_advanced_cfg_lens.
        """
        return {
            "cfg_schedule_type": params.get("cfg_schedule_type", "constant"),
            "cfg_schedule_min": params.get("cfg_schedule_min", 1.0),
            "cfg_schedule_max": params.get("cfg_schedule_max"),
            "cfg_schedule_power": params.get("cfg_schedule_power", 2.0),
            "cfg_rescale_snr_alpha": params.get("cfg_rescale_snr_alpha", 0.0),
            "dynamic_threshold_percentile": params.get("dynamic_threshold_percentile", 0.0),
            "dynamic_threshold_mimic_scale": params.get("dynamic_threshold_mimic_scale", 1.0),
            "developer_mode": params.get("developer_mode", False),
        }

    def _anima_move(self, component_name: str, target_device: str,
                     quantization: Optional[str] = None):
        """Move a named Anima component to the given device.

        For GPU moves the specialized helpers in core.vram_optimization apply
        optional FP8 quantization to text_encoder / transformer; CPU moves use
        the plain .to('cpu') path. The (possibly quantized) component is
        written back into self.anima_components so subsequent calls see the
        quantized copy.
        """
        from core.vram_optimization import (
            move_anima_text_encoder_to_gpu, move_anima_text_encoder_to_cpu,
            move_anima_transformer_to_gpu, move_anima_transformer_to_cpu,
            move_anima_vae_to_gpu, move_anima_vae_to_cpu,
        )

        comp = self.anima_components.get(component_name)
        if comp is None:
            return comp

        try:
            if component_name == "text_encoder":
                if target_device == "cpu":
                    move_anima_text_encoder_to_cpu(comp)
                else:
                    comp = move_anima_text_encoder_to_gpu(comp, quantization)
                    self.anima_components["text_encoder"] = comp
            elif component_name == "transformer":
                if target_device == "cpu":
                    move_anima_transformer_to_cpu(comp)
                else:
                    comp = move_anima_transformer_to_gpu(comp, quantization)
                    self.anima_components["transformer"] = comp
            elif component_name == "vae":
                if target_device == "cpu":
                    move_anima_vae_to_cpu(comp)
                else:
                    move_anima_vae_to_gpu(comp)
            else:
                # Fallback for unknown components
                if hasattr(comp, "to"):
                    comp.to(target_device)
        except Exception as e:
            print(f"[Anima] Warning: could not move {component_name} to {target_device}: {e}")
        return comp

    def _generate_txt2img_anima(self, params: Dict[str, Any],
                                 progress_callback=None, step_callback=None
                                 ) -> tuple[Image.Image, int, int]:
        if not self.anima_components:
            raise RuntimeError("Anima components not loaded. Please load an Anima model first.")

        print("[Anima] Starting txt2img generation")
        from core.models.anima.anima_pipeline_ops import (
            encode_prompt, sample_txt2img, vae_decode_latents,
        )

        device = self.device
        compute_dtype = torch.bfloat16

        transformer = self.anima_components["transformer"]
        text_encoder = self.anima_components["text_encoder"]
        qwen3_tokenizer = self.anima_components["tokenizer"]
        t5_tokenizer = self.anima_components["t5_tokenizer"]
        vae = self.anima_components["vae"]
        scheduler = self.anima_components["scheduler"]

        # Seed
        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        ancestral_seed = params.get("ancestral_seed", -1)
        if ancestral_seed == -1:
            ancestral_seed = random.randint(0, 2147483647)

        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        height = int(params.get("height", 512))
        width = int(params.get("width", 512))
        num_inference_steps = int(params.get("steps", 28))
        guidance_scale = float(params.get("cfg_scale", 4.0))

        # Optional FP8 quantization (matches FLUX.2 / Z-Image pattern)
        transformer_quantization = params.get("unet_quantization")
        text_encoder_quantization = params.get("text_encoder_quantization")

        # Snap to patch_spatial * vae_scale_factor
        snap = transformer.patch_spatial * 8
        height = (height // snap) * snap
        width = (width // snap) * snap

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        try:
            # Stage 1: text encoding
            if not cpu_text_encoding:
                text_encoder = self._anima_move("text_encoder", device, text_encoder_quantization)
            cond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                  prompt, device=enc_device, dtype=compute_dtype)
            uncond = None
            if guidance_scale > 1.0:
                uncond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                       negative_prompt, device=enc_device, dtype=compute_dtype)
            if not cpu_text_encoding:
                self._anima_move("text_encoder", "cpu")
            if cpu_text_encoding:
                # Move CPU-encoded embeddings to GPU for denoising
                def _embeds_to_gpu(d):
                    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
                cond = _embeds_to_gpu(cond)
                if uncond is not None:
                    uncond = _embeds_to_gpu(uncond)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 2: denoising
            transformer = self._anima_move("transformer", device, transformer_quantization)

            # Apply user-supplied LoRAs after the transformer is on GPU (and
            # after any optional quantization). LoRA wrappers point at the
            # current Linear modules; they survive .to() but not deepcopy,
            # so the order must be: quantize -> wrap LoRA -> sample -> unwrap.
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_anima(lora_configs) if lora_configs else 0
            transformer = self.anima_components["transformer"]
            latents = sample_txt2img(
                transformer=transformer, scheduler=scheduler,
                cond_embeds=cond, uncond_embeds=uncond,
                height=height, width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator, device=device, dtype=compute_dtype,
                step_callback=(progress_callback or step_callback),
                advanced_cfg=self._anima_advanced_cfg(params),
            )
            if applied_lora_count:
                self._unload_lora_anima()
            self._anima_move("transformer", "cpu")
            del cond, uncond
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 3: VAE decode
            self._anima_move("vae", device)
            images = vae_decode_latents(vae, latents)
            del latents
            self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Anima] txt2img completed")
            return images[0], seed, ancestral_seed
        except Exception as e:
            print(f"[Anima] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            # Ensure all components are back on CPU even if an error occurred
            for _comp in ("text_encoder", "transformer", "vae"):
                try:
                    self._anima_move(_comp, "cpu")
                except Exception:
                    pass

    def _generate_img2img_anima(self, params: Dict[str, Any], init_image: Image.Image,
                                 progress_callback=None, step_callback=None
                                 ) -> tuple[Image.Image, int]:
        if not self.anima_components:
            raise RuntimeError("Anima components not loaded.")

        print("[Anima] Starting img2img generation")
        from core.models.anima.anima_pipeline_ops import (
            encode_prompt, sample_img2img, vae_encode_image, vae_decode_latents,
        )

        device = self.device
        compute_dtype = torch.bfloat16

        transformer = self.anima_components["transformer"]
        text_encoder = self.anima_components["text_encoder"]
        qwen3_tokenizer = self.anima_components["tokenizer"]
        t5_tokenizer = self.anima_components["t5_tokenizer"]
        vae = self.anima_components["vae"]
        scheduler = self.anima_components["scheduler"]

        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        num_inference_steps = int(params.get("steps", 28))
        guidance_scale = float(params.get("cfg_scale", 4.0))
        denoising_strength = float(params.get("denoising_strength", 0.7))
        transformer_quantization = params.get("unet_quantization")
        text_encoder_quantization = params.get("text_encoder_quantization")

        # Resize init image to match desired width/height (snapped)
        width = int(params.get("width", init_image.width))
        height = int(params.get("height", init_image.height))
        snap = transformer.patch_spatial * 8
        width = (width // snap) * snap
        height = (height // snap) * snap
        if (init_image.width, init_image.height) != (width, height):
            init_image = init_image.resize((width, height), Image.LANCZOS)

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        try:
            # Encode init image
            self._anima_move("vae", device)
            init_latents = vae_encode_image(vae, init_image, device, compute_dtype)
            self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Text encoding
            if not cpu_text_encoding:
                text_encoder = self._anima_move("text_encoder", device, text_encoder_quantization)
            cond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                  prompt, device=enc_device, dtype=compute_dtype)
            uncond = None
            if guidance_scale > 1.0:
                uncond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                       negative_prompt, device=enc_device, dtype=compute_dtype)
            if not cpu_text_encoding:
                self._anima_move("text_encoder", "cpu")
            if cpu_text_encoding:
                def _embeds_to_gpu(d):
                    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
                cond = _embeds_to_gpu(cond)
                if uncond is not None:
                    uncond = _embeds_to_gpu(uncond)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Denoise
            transformer = self._anima_move("transformer", device, transformer_quantization)

            # Apply user-supplied LoRAs after the transformer is on GPU (and
            # after any optional quantization). LoRA wrappers point at the
            # current Linear modules; they survive .to() but not deepcopy,
            # so the order must be: quantize -> wrap LoRA -> sample -> unwrap.
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_anima(lora_configs) if lora_configs else 0
            transformer = self.anima_components["transformer"]
            latents = sample_img2img(
                transformer=transformer, scheduler=scheduler,
                init_latents=init_latents,
                cond_embeds=cond, uncond_embeds=uncond,
                num_inference_steps=num_inference_steps,
                denoising_strength=denoising_strength,
                guidance_scale=guidance_scale,
                generator=generator, device=device, dtype=compute_dtype,
                step_callback=(progress_callback or step_callback),
                advanced_cfg=self._anima_advanced_cfg(params),
            )
            if applied_lora_count:
                self._unload_lora_anima()
            self._anima_move("transformer", "cpu")
            del cond, uncond, init_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Decode
            self._anima_move("vae", device)
            images = vae_decode_latents(vae, latents)
            del latents
            self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Anima] img2img completed")
            return images[0], seed
        except Exception as e:
            print(f"[Anima] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            for _comp in ("text_encoder", "transformer", "vae"):
                try:
                    self._anima_move(_comp, "cpu")
                except Exception:
                    pass

    def _generate_inpaint_anima(self, params: Dict[str, Any],
                                 init_image: Image.Image, mask_image: Image.Image,
                                 progress_callback=None, step_callback=None
                                 ) -> tuple[Image.Image, int]:
        if not self.anima_components:
            raise RuntimeError("Anima components not loaded.")

        print("[Anima] Starting inpaint generation")
        from core.models.anima.anima_pipeline_ops import (
            encode_prompt, sample_inpaint, vae_encode_image, vae_decode_latents,
            make_mask_latents,
        )

        device = self.device
        compute_dtype = torch.bfloat16

        transformer = self.anima_components["transformer"]
        text_encoder = self.anima_components["text_encoder"]
        qwen3_tokenizer = self.anima_components["tokenizer"]
        t5_tokenizer = self.anima_components["t5_tokenizer"]
        vae = self.anima_components["vae"]
        scheduler = self.anima_components["scheduler"]

        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        num_inference_steps = int(params.get("steps", 28))
        guidance_scale = float(params.get("cfg_scale", 4.0))
        denoising_strength = float(params.get("denoising_strength", 0.8))
        mask_blur = int(params.get("mask_blur", 4))
        transformer_quantization = params.get("unet_quantization")
        text_encoder_quantization = params.get("text_encoder_quantization")

        width = int(params.get("width", init_image.width))
        height = int(params.get("height", init_image.height))
        snap = transformer.patch_spatial * 8
        width = (width // snap) * snap
        height = (height // snap) * snap
        if (init_image.width, init_image.height) != (width, height):
            init_image = init_image.resize((width, height), Image.LANCZOS)
        if (mask_image.width, mask_image.height) != (width, height):
            mask_image = mask_image.resize((width, height), Image.NEAREST)

        if mask_blur > 0:
            from PIL import ImageFilter
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(mask_blur))

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        try:
            # Encode init image
            self._anima_move("vae", device)
            init_latents = vae_encode_image(vae, init_image, device, compute_dtype)
            self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            mask_latents = make_mask_latents(
                mask_image, init_latents.shape[-2], init_latents.shape[-1],
                device, compute_dtype,
            )

            # Text encoding
            if not cpu_text_encoding:
                text_encoder = self._anima_move("text_encoder", device, text_encoder_quantization)
            cond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                  prompt, device=enc_device, dtype=compute_dtype)
            uncond = None
            if guidance_scale > 1.0:
                uncond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                       negative_prompt, device=enc_device, dtype=compute_dtype)
            if not cpu_text_encoding:
                self._anima_move("text_encoder", "cpu")
            if cpu_text_encoding:
                def _embeds_to_gpu(d):
                    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
                cond = _embeds_to_gpu(cond)
                if uncond is not None:
                    uncond = _embeds_to_gpu(uncond)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Denoise
            transformer = self._anima_move("transformer", device, transformer_quantization)

            # Apply user-supplied LoRAs after the transformer is on GPU (and
            # after any optional quantization). LoRA wrappers point at the
            # current Linear modules; they survive .to() but not deepcopy,
            # so the order must be: quantize -> wrap LoRA -> sample -> unwrap.
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_anima(lora_configs) if lora_configs else 0
            transformer = self.anima_components["transformer"]
            latents = sample_inpaint(
                transformer=transformer, scheduler=scheduler,
                init_latents=init_latents, mask_latents=mask_latents,
                cond_embeds=cond, uncond_embeds=uncond,
                num_inference_steps=num_inference_steps,
                denoising_strength=denoising_strength,
                guidance_scale=guidance_scale,
                generator=generator, device=device, dtype=compute_dtype,
                step_callback=(progress_callback or step_callback),
                advanced_cfg=self._anima_advanced_cfg(params),
            )
            if applied_lora_count:
                self._unload_lora_anima()
            self._anima_move("transformer", "cpu")
            del cond, uncond, init_latents, mask_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Decode
            self._anima_move("vae", device)
            images = vae_decode_latents(vae, latents)
            del latents
            self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Anima] inpaint completed")
            return images[0], seed
        except Exception as e:
            print(f"[Anima] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            for _comp in ("text_encoder", "transformer", "vae"):
                try:
                    self._anima_move(_comp, "cpu")
                except Exception:
                    pass

    # ================================================================
    # Lens (Microsoft/Lens MMDiT) generation methods
    # ================================================================

    def _reload_lens_text_encoder(self) -> None:
        """Reload the Lens text encoder from disk (~4 s).

        Called lazily at the start of each generation when the text encoder has
        been freed after the previous encoding stage to reclaim ~9.7 GB of mxfp4
        CUDA memory.
        """
        from core.models.lens.lens_loader import reload_lens_text_encoder
        model_path = (self.current_model_info or {}).get("source", "")
        transformer = self.lens_components.get("transformer")
        selected_layers = (
            tuple(transformer.config.selected_layer_index)
            if transformer is not None else None
        )
        te = reload_lens_text_encoder(
            model_path,
            torch_dtype=torch.bfloat16,
            selected_layers=selected_layers,
        )
        self.lens_components["text_encoder"] = te

    def _lens_move(self, component_name: str, target_device: str,
                   quantization: Optional[str] = None):
        """Move a Lens component to the target device.

        GPU moves delegate to specialized helpers in core.vram_optimization
        that apply optional FP8 quantization.  The (possibly quantized)
        component is written back into self.lens_components.
        """
        from core.vram_optimization import (
            move_lens_text_encoder_to_gpu, move_lens_text_encoder_to_cpu,
            move_lens_transformer_to_gpu, move_lens_transformer_to_cpu,
            move_lens_vae_to_gpu, move_lens_vae_to_cpu,
        )

        comp = self.lens_components.get(component_name)
        if comp is None:
            return comp

        try:
            if component_name == "text_encoder":
                if target_device == "cpu":
                    move_lens_text_encoder_to_cpu(comp)
                else:
                    comp = move_lens_text_encoder_to_gpu(comp, quantization)
                    self.lens_components["text_encoder"] = comp
            elif component_name == "transformer":
                if target_device == "cpu":
                    move_lens_transformer_to_cpu(comp)
                else:
                    comp = move_lens_transformer_to_gpu(comp, quantization)
                    self.lens_components["transformer"] = comp
            elif component_name == "vae":
                if target_device == "cpu":
                    move_lens_vae_to_cpu(comp)
                else:
                    move_lens_vae_to_gpu(comp)
        except Exception as e:
            print(f"[Lens] Warning: could not move {component_name} to {target_device}: {e}")
        return comp

    def _generate_txt2img_lens(self, params: Dict[str, Any],
                                progress_callback=None, step_callback=None,
                                ) -> tuple:
        if not self.lens_components:
            raise RuntimeError("Lens components not loaded. Please load a Lens model first.")

        from core.models.lens.lens_pipeline_ops import (
            encode_prompt, prepare_latents, denoise_loop, vae_decode,
        )
        from core.models.lens.lens_resolution import align_to_grid

        print("[Lens] Starting txt2img generation")

        device = self.device
        dtype = torch.bfloat16

        # Lazy reload: text encoder is freed after each generation to reclaim
        # the ~9.7 GB of mxfp4 CUDA memory.  Reload it here before encoding.
        if self.lens_components.get("text_encoder") is None:
            self._reload_lens_text_encoder()

        transformer = self.lens_components["transformer"]
        text_encoder = self.lens_components["text_encoder"]
        tokenizer = self.lens_components["tokenizer"]
        vae = self.lens_components["vae"]
        scheduler = self.lens_components["scheduler"]

        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        num_inference_steps = int(params.get("steps", 28))
        guidance_scale = float(params.get("cfg_scale", 4.0))
        transformer_quantization = params.get("unet_quantization")
        text_encoder_quantization = params.get("text_encoder_quantization")
        max_sequence_length = 512

        req_width = int(params.get("width", 1024))
        req_height = int(params.get("height", 1024))
        width, height = align_to_grid(req_width, req_height)
        if (width, height) != (req_width, req_height):
            print(f"[Lens] Resolution aligned: {req_width}×{req_height} → {width}×{height}")

        latent_h = height // 16
        latent_w = width // 16

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        try:
            # Stage 1: Text encoding
            print("[Lens] Stage 1: Text encoding...")
            if not cpu_text_encoding:
                text_encoder = self._lens_move("text_encoder", device, text_encoder_quantization)
            encoder_features, encoder_mask = encode_prompt(
                text_encoder, tokenizer, prompt, negative_prompt,
                device=enc_device, dtype=dtype, max_length=max_sequence_length,
            )
            if not cpu_text_encoding:
                self._lens_move("text_encoder", "cpu")
            if cpu_text_encoding:
                encoder_features = [f.to(device) for f in encoder_features]
                encoder_mask = encoder_mask.to(device)

            # Free mxfp4 CUDA buffers (~9.7 GB) — not needed during denoising.
            # Will be reloaded lazily at the start of the next generation.
            import gc as _gc
            self.lens_components["text_encoder"] = None
            text_encoder = None
            _gc.collect()
            torch.cuda.empty_cache()

            # Stage 2: Prepare latents
            latents = prepare_latents(height, width, dtype=dtype, device=device, seed=seed)

            # Stage 3: Denoising
            print("[Lens] Stage 3: Denoising...")
            transformer = self._lens_move("transformer", device, transformer_quantization)
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_lens(lora_configs) if lora_configs else 0
            transformer = self.lens_components["transformer"]
            try:
                latents = denoise_loop(
                    transformer=transformer, scheduler=scheduler,
                    latents=latents, encoder_features=encoder_features, encoder_mask=encoder_mask,
                    guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                    latent_h=latent_h, latent_w=latent_w,
                    progress_callback=progress_callback,
                    advanced_cfg=self._lens_advanced_cfg(params),
                )
            finally:
                if applied_lora_count:
                    self._unload_lora_lens()
            self._lens_move("transformer", "cpu")
            del encoder_features, encoder_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 4: VAE decode
            print("[Lens] Stage 4: VAE decode...")
            self._lens_move("vae", device)
            vae_gpu = self.lens_components["vae"]
            image = vae_decode(vae_gpu, latents, latent_h, latent_w)
            del latents
            self._lens_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Lens] txt2img completed")
            return image, seed, 0

        except Exception as e:
            print(f"[Lens] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            # Always free text encoder CUDA buffers on exit (normal or exception).
            # Next generation will reload it lazily before encoding.
            if self.lens_components.get("text_encoder") is not None:
                import gc as _gc
                self.lens_components["text_encoder"] = None
                _gc.collect()
            for _comp in ("transformer", "vae"):
                try:
                    self._lens_move(_comp, "cpu")
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate_img2img_lens(self, params: Dict[str, Any], init_image: Image.Image,
                                progress_callback=None, step_callback=None,
                                ) -> tuple:
        if not self.lens_components:
            raise RuntimeError("Lens components not loaded.")

        from core.models.lens.lens_pipeline_ops import (
            encode_prompt, vae_encode, denoise_loop_img2img, vae_decode,
        )
        from core.models.lens.lens_resolution import align_to_grid

        print("[Lens] Starting img2img generation")

        device = self.device
        dtype = torch.bfloat16

        if self.lens_components.get("text_encoder") is None:
            self._reload_lens_text_encoder()

        transformer = self.lens_components["transformer"]
        text_encoder = self.lens_components["text_encoder"]
        tokenizer = self.lens_components["tokenizer"]
        vae = self.lens_components["vae"]
        scheduler = self.lens_components["scheduler"]

        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        num_inference_steps = int(params.get("steps", 28))
        guidance_scale = float(params.get("cfg_scale", 4.0))
        denoising_strength = float(params.get("denoising_strength", 0.7))
        transformer_quantization = params.get("unet_quantization")
        text_encoder_quantization = params.get("text_encoder_quantization")
        max_sequence_length = 512

        req_width = int(params.get("width", init_image.width))
        req_height = int(params.get("height", init_image.height))
        width, height = align_to_grid(req_width, req_height)
        if (width, height) != (req_width, req_height):
            print(f"[Lens] Resolution aligned: {req_width}×{req_height} → {width}×{height}")
        latent_h = height // 16
        latent_w = width // 16

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        try:
            # Stage 1: Text encoding
            print("[Lens] Stage 1: Text encoding...")
            if not cpu_text_encoding:
                text_encoder = self._lens_move("text_encoder", device, text_encoder_quantization)
            encoder_features, encoder_mask = encode_prompt(
                text_encoder, tokenizer, prompt, negative_prompt,
                device=enc_device, dtype=dtype, max_length=max_sequence_length,
            )
            if not cpu_text_encoding:
                self._lens_move("text_encoder", "cpu")
            if cpu_text_encoding:
                encoder_features = [f.to(device) for f in encoder_features]
                encoder_mask = encoder_mask.to(device)

            # Free mxfp4 CUDA buffers (~9.7 GB) — not needed during denoising.
            import gc as _gc
            self.lens_components["text_encoder"] = None
            text_encoder = None
            _gc.collect()
            torch.cuda.empty_cache()

            # Stage 2: Encode init image
            print("[Lens] Stage 2: Encoding init image...")
            self._lens_move("vae", device)
            vae_gpu = self.lens_components["vae"]
            init_latents = vae_encode(vae_gpu, init_image, height, width, device=device, dtype=dtype)
            self._lens_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 3: Denoising (SDEdit)
            print("[Lens] Stage 3: Denoising...")
            transformer = self._lens_move("transformer", device, transformer_quantization)
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_lens(lora_configs) if lora_configs else 0
            transformer = self.lens_components["transformer"]
            try:
                latents = denoise_loop_img2img(
                    transformer=transformer, scheduler=scheduler,
                    init_latents=init_latents, denoising_strength=denoising_strength,
                    encoder_features=encoder_features, encoder_mask=encoder_mask,
                    guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                    latent_h=latent_h, latent_w=latent_w, seed=seed,
                    progress_callback=progress_callback,
                    advanced_cfg=self._lens_advanced_cfg(params),
                )
            finally:
                if applied_lora_count:
                    self._unload_lora_lens()
            self._lens_move("transformer", "cpu")
            del encoder_features, encoder_mask, init_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 4: VAE decode
            print("[Lens] Stage 4: VAE decode...")
            self._lens_move("vae", device)
            vae_gpu = self.lens_components["vae"]
            image = vae_decode(vae_gpu, latents, latent_h, latent_w)
            del latents
            self._lens_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Lens] img2img completed")
            return image, seed, 0

        except Exception as e:
            print(f"[Lens] img2img error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            if self.lens_components.get("text_encoder") is not None:
                import gc as _gc
                self.lens_components["text_encoder"] = None
                _gc.collect()
            for _comp in ("transformer", "vae"):
                try:
                    self._lens_move(_comp, "cpu")
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate_inpaint_lens(self, params: Dict[str, Any],
                                init_image: Image.Image, mask_image: Image.Image,
                                progress_callback=None, step_callback=None,
                                ) -> tuple:
        if not self.lens_components:
            raise RuntimeError("Lens components not loaded.")

        from core.models.lens.lens_pipeline_ops import (
            encode_prompt, vae_encode, denoise_loop_inpaint, vae_decode, prepare_mask_latent,
        )
        from core.models.lens.lens_resolution import align_to_grid

        print("[Lens] Starting inpaint generation (repaint)")

        device = self.device
        dtype = torch.bfloat16

        if self.lens_components.get("text_encoder") is None:
            self._reload_lens_text_encoder()

        transformer = self.lens_components["transformer"]
        text_encoder = self.lens_components["text_encoder"]
        tokenizer = self.lens_components["tokenizer"]
        vae = self.lens_components["vae"]
        scheduler = self.lens_components["scheduler"]

        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        num_inference_steps = int(params.get("steps", 28))
        guidance_scale = float(params.get("cfg_scale", 4.0))
        denoising_strength = float(params.get("denoising_strength", 0.8))
        mask_blur = int(params.get("mask_blur", 4))
        transformer_quantization = params.get("unet_quantization")
        text_encoder_quantization = params.get("text_encoder_quantization")
        max_sequence_length = 512

        req_width = int(params.get("width", init_image.width))
        req_height = int(params.get("height", init_image.height))
        width, height = align_to_grid(req_width, req_height)
        if (width, height) != (req_width, req_height):
            print(f"[Lens] Resolution aligned: {req_width}×{req_height} → {width}×{height}")
        latent_h = height // 16
        latent_w = width // 16

        if (init_image.width, init_image.height) != (width, height):
            init_image = init_image.resize((width, height), Image.LANCZOS)
        if (mask_image.width, mask_image.height) != (width, height):
            mask_image = mask_image.resize((width, height), Image.NEAREST)

        if mask_blur > 0:
            from PIL import ImageFilter
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(mask_blur))

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        try:
            # Stage 1: Text encoding
            print("[Lens] Stage 1: Text encoding...")
            if not cpu_text_encoding:
                text_encoder = self._lens_move("text_encoder", device, text_encoder_quantization)
            encoder_features, encoder_mask = encode_prompt(
                text_encoder, tokenizer, prompt, negative_prompt,
                device=enc_device, dtype=dtype, max_length=max_sequence_length,
            )
            if not cpu_text_encoding:
                self._lens_move("text_encoder", "cpu")
            if cpu_text_encoding:
                encoder_features = [f.to(device) for f in encoder_features]
                encoder_mask = encoder_mask.to(device)

            # Free mxfp4 CUDA buffers (~9.7 GB) — not needed during denoising.
            import gc as _gc
            self.lens_components["text_encoder"] = None
            text_encoder = None
            _gc.collect()
            torch.cuda.empty_cache()

            # Stage 2: Encode init image + prepare mask
            print("[Lens] Stage 2: Encoding init image...")
            self._lens_move("vae", device)
            vae_gpu = self.lens_components["vae"]
            init_latents = vae_encode(vae_gpu, init_image, height, width, device=device, dtype=dtype)
            self._lens_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            mask_latent = prepare_mask_latent(mask_image, latent_h, latent_w, device=device, dtype=dtype)

            # Stage 3: Denoising with repaint
            print("[Lens] Stage 3: Denoising (repaint)...")
            transformer = self._lens_move("transformer", device, transformer_quantization)
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_lens(lora_configs) if lora_configs else 0
            transformer = self.lens_components["transformer"]
            try:
                latents = denoise_loop_inpaint(
                    transformer=transformer, scheduler=scheduler,
                    init_latents=init_latents, mask_latent=mask_latent,
                    denoising_strength=denoising_strength,
                    encoder_features=encoder_features, encoder_mask=encoder_mask,
                    guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                    latent_h=latent_h, latent_w=latent_w, seed=seed,
                    progress_callback=progress_callback,
                    advanced_cfg=self._lens_advanced_cfg(params),
                )
            finally:
                if applied_lora_count:
                    self._unload_lora_lens()
            self._lens_move("transformer", "cpu")
            del encoder_features, encoder_mask, init_latents, mask_latent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 4: VAE decode
            print("[Lens] Stage 4: VAE decode...")
            self._lens_move("vae", device)
            vae_gpu = self.lens_components["vae"]
            image = vae_decode(vae_gpu, latents, latent_h, latent_w)
            del latents
            self._lens_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Lens] inpaint completed")
            return image, seed, 0

        except Exception as e:
            print(f"[Lens] inpaint error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            if self.lens_components.get("text_encoder") is not None:
                import gc as _gc
                self.lens_components["text_encoder"] = None
                _gc.collect()
            for _comp in ("transformer", "vae"):
                try:
                    self._lens_move(_comp, "cpu")
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Ideogram 4 (dual-branch single-stream DiT, asymmetric CFG)
    # ------------------------------------------------------------------

    def _ideogram4_move(self, component_name: str, target_device: str):
        """Move a named Ideogram 4 component to the target device.

        Weights are already weight-only FP8 (Fp8Linear buffers move with .to());
        plain .to() is sufficient — no extra quantization step is applied.
        """
        comp = self.ideogram4_components.get(component_name)
        if comp is None or not hasattr(comp, "to"):
            return comp
        try:
            comp.to(target_device)
        except Exception as e:
            print(f"[Ideogram4] Warning: could not move {component_name} to {target_device}: {e}")
        return comp

    @staticmethod
    def _ideogram4_advanced_cfg(params: Dict[str, Any]) -> Dict[str, Any]:
        """Collect Advanced-CFG knobs for Ideogram 4 generation.

        Consumed by ideogram4_pipeline_ops._blend_guidance (standard CFG blend
        plus optional schedule / SNR-rescale / dynamic thresholding).
        """
        return {
            "cfg_schedule_type": params.get("cfg_schedule_type", "constant"),
            "cfg_schedule_min": params.get("cfg_schedule_min", 1.0),
            "cfg_schedule_max": params.get("cfg_schedule_max"),
            "cfg_schedule_power": params.get("cfg_schedule_power", 2.0),
            "cfg_rescale_snr_alpha": params.get("cfg_rescale_snr_alpha", 0.0),
            "dynamic_threshold_percentile": params.get("dynamic_threshold_percentile", 0.0),
            "dynamic_threshold_mimic_scale": params.get("dynamic_threshold_mimic_scale", 1.0),
            "developer_mode": params.get("developer_mode", False),
        }

    def _ideogram4_common_params(self, params: Dict[str, Any], default_w: int, default_h: int):
        """Resolve shared Ideogram 4 generation parameters."""
        from core.models.ideogram4.ideogram4_resolution import normalize_resolution, latent_grid

        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        req_width = int(params.get("width", default_w))
        req_height = int(params.get("height", default_h))
        width, height = normalize_resolution(req_width, req_height)
        if (width, height) != (req_width, req_height):
            print(f"[Ideogram4] Resolution aligned: {req_width}x{req_height} -> {width}x{height}")
        grid_h, grid_w = latent_grid(width, height)

        return {
            "seed": seed,
            "prompt": params.get("prompt", ""),
            "num_inference_steps": int(params.get("steps", 28)),
            "guidance_scale": float(params.get("cfg_scale", 7.0)),
            "mu": float(params.get("ideogram4_mu", 0.0)),
            "std": float(params.get("ideogram4_std", 1.5)),
            "max_sequence_length": int(params.get("ideogram4_max_seq_len", 512)),
            "width": width,
            "height": height,
            "grid_h": grid_h,
            "grid_w": grid_w,
        }

    @torch.no_grad()
    def _ideogram4_encode(self, prompt, grid_h, grid_w, max_sequence_length, device, dtype):
        """Stage the text encoder to GPU, encode the prompt, then free it back to CPU."""
        from core.models.ideogram4.ideogram4_pipeline_ops import encode_prompt

        self._ideogram4_move("text_encoder", device)
        text_encoder = self.ideogram4_components["text_encoder"]
        tokenizer = self.ideogram4_components["tokenizer"]
        cond = encode_prompt(
            text_encoder, tokenizer, prompt,
            grid_h=grid_h, grid_w=grid_w,
            max_sequence_length=max_sequence_length, device=device,
        )
        self._ideogram4_move("text_encoder", "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Cast conditioning to the transformer compute dtype (halves memory; matches RMSNorm dtype).
        cond["llm_features"] = cond["llm_features"].to(dtype)
        cond["neg_llm_features"] = cond["neg_llm_features"].to(dtype)
        return cond

    def _ideogram4_setup_block_swap(self, transformer, blocks_to_swap: int,
                                    use_pinned_memory: bool, device: str):
        """Attach a block-swap offloader to one Ideogram 4 transformer.

        The transformer starts on CPU; the offloader keeps the first
        (num_layers - blocks_to_swap) blocks resident on GPU and streams the rest
        per forward. Non-block (auxiliary) modules are moved to GPU here since the
        shared offloader only auto-moves Z-Image-named aux modules.
        """
        from core.memory_management import create_block_offloader_for_model

        # Auxiliary modules (everything except the swappable block list) stay on GPU.
        for name, child in transformer.named_children():
            if name != "layers":
                child.to(device)
        for p in transformer.parameters(recurse=False):
            if p.device.type != "cuda":
                p.data = p.data.to(device)
        for b in transformer.buffers(recurse=False):
            if b.device.type != "cuda":
                b.data = b.data.to(device)

        offloader = create_block_offloader_for_model(
            transformer=transformer,
            blocks_to_swap=blocks_to_swap,
            device=torch.device(device),
            target_dtype=torch.bfloat16,
            use_pinned_memory=use_pinned_memory,
        )
        transformer._block_offloader = offloader
        offloader.prepare_block_devices_before_forward()
        return offloader

    def _ideogram4_stage_transformers(self, device: str, params: Optional[Dict[str, Any]] = None):
        """Place both transformers on GPU for the denoise loop.

        With block swap enabled, each transformer streams its blocks (per-model
        offloader) instead of being fully resident, roughly halving the resident
        footprint of the two 9.3B FP8 transformers at the cost of CPU<->GPU traffic.
        """
        params = params or {}
        enable_block_swap = bool(params.get("enable_block_swap", False))
        num_layers = len(self.ideogram4_components["transformer"].layers)
        blocks_to_swap = int(params.get("blocks_to_swap", 20))
        blocks_to_swap = max(0, min(blocks_to_swap, num_layers - 1))
        use_pinned_memory = bool(params.get("use_pinned_memory", False))

        self._ideogram4_offloaders = []
        if enable_block_swap and blocks_to_swap > 0:
            print(f"[Ideogram4] Block swap enabled: {blocks_to_swap}/{num_layers} blocks per transformer "
                  f"(pinned_memory={use_pinned_memory})")
            for comp_name in ("transformer", "unconditional_transformer"):
                t = self.ideogram4_components[comp_name]
                off = self._ideogram4_setup_block_swap(t, blocks_to_swap, use_pinned_memory, device)
                self._ideogram4_offloaders.append((comp_name, off))
        else:
            self._ideogram4_move("transformer", device)
            self._ideogram4_move("unconditional_transformer", device)
        return (
            self.ideogram4_components["transformer"],
            self.ideogram4_components["unconditional_transformer"],
        )

    def _ideogram4_unstage_transformers(self):
        # Tear down any block-swap offloaders, then return both transformers to CPU.
        offloaders = getattr(self, "_ideogram4_offloaders", [])
        for comp_name, off in offloaders:
            t = self.ideogram4_components.get(comp_name)
            if t is not None and hasattr(t, "_block_offloader"):
                try:
                    delattr(t, "_block_offloader")
                except Exception:
                    pass
            cleanup = getattr(off, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass
        self._ideogram4_offloaders = []
        self._ideogram4_move("transformer", "cpu")
        self._ideogram4_move("unconditional_transformer", "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ideogram4_cleanup(self):
        # Strip any leftover block-swap offloaders (e.g. if setup raised mid-way).
        for _comp in ("transformer", "unconditional_transformer"):
            t = (self.ideogram4_components or {}).get(_comp)
            if t is not None and hasattr(t, "_block_offloader"):
                try:
                    delattr(t, "_block_offloader")
                except Exception:
                    pass
        self._ideogram4_offloaders = []
        for _comp in ("text_encoder", "transformer", "unconditional_transformer", "vae"):
            try:
                self._ideogram4_move(_comp, "cpu")
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_txt2img_ideogram4(self, params: Dict[str, Any],
                                    progress_callback=None, step_callback=None) -> tuple:
        if not self.ideogram4_components:
            raise RuntimeError("Ideogram 4 components not loaded. Please load an Ideogram 4 model first.")

        from core.models.ideogram4.ideogram4_pipeline_ops import (
            prepare_latents, denoise_loop, vae_decode,
        )

        print("[Ideogram4] Starting txt2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._ideogram4_common_params(params, 1024, 1024)
        scheduler = self.ideogram4_components["scheduler"]
        advanced_cfg = self._ideogram4_advanced_cfg(params)

        try:
            print("[Ideogram4] Stage 1: Text encoding...")
            cond = self._ideogram4_encode(
                cfg["prompt"], cfg["grid_h"], cfg["grid_w"],
                cfg["max_sequence_length"], device, dtype,
            )

            print("[Ideogram4] Stage 2: Prepare latents...")
            latents = prepare_latents(
                cfg["grid_h"], cfg["grid_w"], dtype=torch.float32, device=device, seed=cfg["seed"],
            )

            print("[Ideogram4] Stage 3: Denoising (dual-branch)...")
            transformer, uncond_transformer = self._ideogram4_stage_transformers(device, params)
            applied_lora = self._load_lora_ideogram4(params.get("loras") or [])
            try:
                latents = denoise_loop(
                    transformer=transformer, unconditional_transformer=uncond_transformer,
                    scheduler=scheduler, latents=latents, cond=cond,
                    guidance_scale=cfg["guidance_scale"], num_inference_steps=cfg["num_inference_steps"],
                    grid_h=cfg["grid_h"], grid_w=cfg["grid_w"], height=cfg["height"], width=cfg["width"],
                    mu=cfg["mu"], std=cfg["std"],
                    progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                )
            finally:
                if applied_lora:
                    self._unload_lora_ideogram4()
                self._ideogram4_unstage_transformers()
            del cond

            print("[Ideogram4] Stage 4: VAE decode...")
            self._ideogram4_move("vae", device)
            image = vae_decode(self.ideogram4_components["vae"], latents, cfg["grid_h"], cfg["grid_w"])
            del latents
            self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Ideogram4] txt2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Ideogram4] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._ideogram4_cleanup()

    def _generate_img2img_ideogram4(self, params: Dict[str, Any], init_image: Image.Image,
                                    progress_callback=None, step_callback=None) -> tuple:
        if not self.ideogram4_components:
            raise RuntimeError("Ideogram 4 components not loaded.")

        from core.models.ideogram4.ideogram4_pipeline_ops import (
            vae_encode, denoise_loop_img2img, vae_decode,
        )

        print("[Ideogram4] Starting img2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._ideogram4_common_params(params, init_image.width, init_image.height)
        denoising_strength = float(params.get("denoising_strength", 0.7))
        scheduler = self.ideogram4_components["scheduler"]
        advanced_cfg = self._ideogram4_advanced_cfg(params)

        try:
            print("[Ideogram4] Stage 1: Text encoding...")
            cond = self._ideogram4_encode(
                cfg["prompt"], cfg["grid_h"], cfg["grid_w"],
                cfg["max_sequence_length"], device, dtype,
            )

            print("[Ideogram4] Stage 2: Encoding init image...")
            self._ideogram4_move("vae", device)
            init_latents = vae_encode(
                self.ideogram4_components["vae"], init_image, cfg["height"], cfg["width"],
                device=device, dtype=torch.float32,
            )
            self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Ideogram4] Stage 3: Denoising (SDEdit)...")
            transformer, uncond_transformer = self._ideogram4_stage_transformers(device, params)
            applied_lora = self._load_lora_ideogram4(params.get("loras") or [])
            try:
                latents = denoise_loop_img2img(
                    transformer=transformer, unconditional_transformer=uncond_transformer,
                    scheduler=scheduler, init_latents=init_latents, denoising_strength=denoising_strength,
                    cond=cond, guidance_scale=cfg["guidance_scale"],
                    num_inference_steps=cfg["num_inference_steps"],
                    grid_h=cfg["grid_h"], grid_w=cfg["grid_w"], height=cfg["height"], width=cfg["width"],
                    mu=cfg["mu"], std=cfg["std"], seed=cfg["seed"],
                    progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                )
            finally:
                if applied_lora:
                    self._unload_lora_ideogram4()
                self._ideogram4_unstage_transformers()
            del cond, init_latents

            print("[Ideogram4] Stage 4: VAE decode...")
            self._ideogram4_move("vae", device)
            image = vae_decode(self.ideogram4_components["vae"], latents, cfg["grid_h"], cfg["grid_w"])
            del latents
            self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Ideogram4] img2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Ideogram4] img2img error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._ideogram4_cleanup()

    def _generate_inpaint_ideogram4(self, params: Dict[str, Any],
                                    init_image: Image.Image, mask_image: Image.Image,
                                    progress_callback=None, step_callback=None) -> tuple:
        if not self.ideogram4_components:
            raise RuntimeError("Ideogram 4 components not loaded.")

        from core.models.ideogram4.ideogram4_pipeline_ops import (
            vae_encode, denoise_loop_inpaint, vae_decode, prepare_mask_latent,
        )

        print("[Ideogram4] Starting inpaint generation (repaint)")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._ideogram4_common_params(params, init_image.width, init_image.height)
        denoising_strength = float(params.get("denoising_strength", 0.8))
        mask_blur = int(params.get("mask_blur", 4))
        scheduler = self.ideogram4_components["scheduler"]
        advanced_cfg = self._ideogram4_advanced_cfg(params)

        width, height = cfg["width"], cfg["height"]
        if (init_image.width, init_image.height) != (width, height):
            init_image = init_image.resize((width, height), Image.LANCZOS)
        if (mask_image.width, mask_image.height) != (width, height):
            mask_image = mask_image.resize((width, height), Image.NEAREST)
        if mask_blur > 0:
            from PIL import ImageFilter
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(mask_blur))

        try:
            print("[Ideogram4] Stage 1: Text encoding...")
            cond = self._ideogram4_encode(
                cfg["prompt"], cfg["grid_h"], cfg["grid_w"],
                cfg["max_sequence_length"], device, dtype,
            )

            print("[Ideogram4] Stage 2: Encoding init image + mask...")
            self._ideogram4_move("vae", device)
            init_latents = vae_encode(
                self.ideogram4_components["vae"], init_image, height, width,
                device=device, dtype=torch.float32,
            )
            self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mask_latent = prepare_mask_latent(
                mask_image, cfg["grid_h"], cfg["grid_w"], device=device, dtype=torch.float32,
            )

            print("[Ideogram4] Stage 3: Denoising (repaint)...")
            transformer, uncond_transformer = self._ideogram4_stage_transformers(device, params)
            applied_lora = self._load_lora_ideogram4(params.get("loras") or [])
            try:
                latents = denoise_loop_inpaint(
                    transformer=transformer, unconditional_transformer=uncond_transformer,
                    scheduler=scheduler, init_latents=init_latents, mask_latent=mask_latent,
                    denoising_strength=denoising_strength, cond=cond,
                    guidance_scale=cfg["guidance_scale"], num_inference_steps=cfg["num_inference_steps"],
                    grid_h=cfg["grid_h"], grid_w=cfg["grid_w"], height=height, width=width,
                    mu=cfg["mu"], std=cfg["std"], seed=cfg["seed"],
                    progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                )
            finally:
                if applied_lora:
                    self._unload_lora_ideogram4()
                self._ideogram4_unstage_transformers()
            del cond, init_latents, mask_latent

            print("[Ideogram4] Stage 4: VAE decode...")
            self._ideogram4_move("vae", device)
            image = vae_decode(self.ideogram4_components["vae"], latents, cfg["grid_h"], cfg["grid_w"])
            del latents
            self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Ideogram4] inpaint completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Ideogram4] inpaint error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._ideogram4_cleanup()

    # ------------------------------------------------------------------
    # MiniT2I (pixel-space MM-JiT, flow matching, x0 prediction, no VAE)
    # ------------------------------------------------------------------

    def _minit2i_move(self, component_name: str, target_device: str):
        comp = self.minit2i_components.get(component_name)
        if comp is None or not hasattr(comp, "to"):
            return comp
        try:
            comp.to(target_device)
        except Exception as e:
            print(f"[MiniT2I] Warning: could not move {component_name} to {target_device}: {e}")
        return comp

    def _minit2i_common_params(self, params: Dict[str, Any], default_w: int, default_h: int):
        from core.models.minit2i.minit2i_pipeline_ops import normalize_resolution
        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        req_w = int(params.get("width", default_w))
        req_h = int(params.get("height", default_h))
        width, height = normalize_resolution(req_w, req_h)
        if (width, height) != (req_w, req_h):
            print(f"[MiniT2I] Resolution aligned: {req_w}x{req_h} -> {width}x{height}")
        cfg = self.minit2i_components["transformer"].mmjit_config
        scheduler = self.minit2i_components["scheduler"]
        return {
            "seed": seed,
            "prompt": params.get("prompt", ""),
            "negative_prompt": params.get("negative_prompt", "") or "",
            "num_inference_steps": int(params.get("steps", scheduler.config.num_inference_steps)),
            "cfg_scale": float(params.get("cfg_scale", 6.0)),
            "cfg_interval": tuple(cfg.cfg_interval),
            "prompt_length": int(cfg.prompt_length),
            "width": width,
            "height": height,
        }

    @torch.no_grad()
    def _minit2i_encode(self, prompt, negative_prompt, prompt_length, device, dtype):
        """Encode prompt (+ optional negative) with FLAN-T5, then free TE to CPU."""
        from core.models.minit2i.minit2i_pipeline_ops import encode_prompt
        self._minit2i_move("text_encoder", device)
        te = self.minit2i_components["text_encoder"]
        tok = self.minit2i_components["tokenizer"]
        text, mask = encode_prompt(te, tok, prompt, prompt_length, device)
        neg_text = neg_mask = None
        if negative_prompt and negative_prompt.strip():
            neg_text, neg_mask = encode_prompt(te, tok, negative_prompt, prompt_length, device)
            neg_text = neg_text.to(dtype)
        self._minit2i_move("text_encoder", "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text.to(dtype), mask, neg_text, neg_mask

    def _load_lora_minit2i(self, lora_configs: List[Dict]) -> int:
        from core.models.minit2i.minit2i_lora import (
            load_lora_safetensors, normalise_lora_state_dict, apply_lora_group,
            apply_te_lora_group, TE_NAMESPACE,
        )
        from core.extensions.lora_manager import lora_manager
        if not lora_configs or not self.minit2i_components:
            return 0
        transformer = self.minit2i_components["transformer"]
        text_encoder = self.minit2i_components.get("text_encoder")
        if not hasattr(self, "_minit2i_lora_orig"):
            self._minit2i_lora_orig: Dict[str, torch.nn.Module] = {}
            self._minit2i_lora_keys: set = set()
        total = 0
        for i, cfg in enumerate(lora_configs):
            lora_path = cfg.get("path", "")
            strength = float(cfg.get("strength", 1.0))
            resolved = lora_manager._resolve_lora_path(lora_path)
            if resolved is None:
                print(f"[MiniT2I LoRA] WARNING: file not found: {lora_path}")
                continue
            try:
                raw, fmt = load_lora_safetensors(str(resolved))
                grouped = normalise_lora_state_dict(raw)
                # Transformer LoRA (lora_unet_) and TE LoRA (lora_te_) auto-route by key.
                applied = apply_lora_group(transformer, grouped, strength,
                                           self._minit2i_lora_orig, self._minit2i_lora_keys)
                applied_te = 0
                has_te_keys = any(k.startswith(TE_NAMESPACE) for k in grouped)
                if has_te_keys and text_encoder is not None:
                    applied_te = apply_te_lora_group(text_encoder, grouped, strength,
                                                     self._minit2i_lora_orig, self._minit2i_lora_keys)
                elif has_te_keys:
                    print(f"[MiniT2I LoRA] WARNING: {lora_path} has TE-LoRA keys but no text encoder is loaded; "
                          f"TE-LoRA skipped")
                print(f"[MiniT2I LoRA] {i+1}/{len(lora_configs)}: {lora_path} fmt={fmt} "
                      f"matched={len(grouped)} wrapped(transformer)={applied} wrapped(te)={applied_te} "
                      f"strength={strength}")
                total += applied + applied_te
            except Exception as e:
                print(f"[MiniT2I LoRA] ERROR loading {lora_path}: {e}")
                import traceback; traceback.print_exc()
        return total

    def _unload_lora_minit2i(self) -> int:
        from core.models.minit2i.minit2i_lora import restore_originals
        if not self.minit2i_components or not getattr(self, "_minit2i_lora_keys", None):
            return 0
        restored = restore_originals(
            self.minit2i_components["transformer"], self._minit2i_lora_orig, self._minit2i_lora_keys,
            text_encoder=self.minit2i_components.get("text_encoder"),
        )
        if restored:
            print(f"[MiniT2I LoRA] Unloaded {restored} LoRA wrappers")
        return restored

    def _minit2i_cleanup(self):
        try:
            self._unload_lora_minit2i()
        except Exception:
            pass
        for _c in ("text_encoder", "transformer"):
            try:
                self._minit2i_move(_c, "cpu")
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_txt2img_minit2i(self, params, progress_callback=None, step_callback=None) -> tuple:
        if not self.minit2i_components:
            raise RuntimeError("MiniT2I components not loaded.")
        from core.models.minit2i.minit2i_pipeline_ops import denoise_loop, tensor_to_image
        print("[MiniT2I] Starting txt2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._minit2i_common_params(params, 512, 512)
        try:
            text, mask, neg_text, neg_mask = self._minit2i_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg["prompt_length"], device, dtype)
            transformer = self._minit2i_move("transformer", device)
            applied_lora = self._load_lora_minit2i(params.get("loras") or [])
            try:
                x = denoise_loop(
                    transformer, text, mask, cfg["height"], cfg["width"],
                    cfg["num_inference_steps"], cfg["cfg_scale"], cfg["cfg_interval"],
                    device, dtype, seed=cfg["seed"], neg_text=neg_text, neg_mask=neg_mask,
                    progress_callback=progress_callback,
                )
            finally:
                if applied_lora:
                    self._unload_lora_minit2i()
            image = tensor_to_image(x)
            print("[MiniT2I] txt2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[MiniT2I] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._minit2i_cleanup()

    def _generate_img2img_minit2i(self, params, init_image, progress_callback=None, step_callback=None) -> tuple:
        if not self.minit2i_components:
            raise RuntimeError("MiniT2I components not loaded.")
        from core.models.minit2i.minit2i_pipeline_ops import (
            denoise_loop_img2img, image_to_tensor, tensor_to_image,
        )
        print("[MiniT2I] Starting img2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._minit2i_common_params(params, init_image.width, init_image.height)
        denoising_strength = float(params.get("denoising_strength", 0.7))
        try:
            text, mask, neg_text, neg_mask = self._minit2i_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg["prompt_length"], device, dtype)
            init_t = image_to_tensor(init_image, cfg["height"], cfg["width"], device, dtype)
            transformer = self._minit2i_move("transformer", device)
            applied_lora = self._load_lora_minit2i(params.get("loras") or [])
            try:
                x = denoise_loop_img2img(
                    transformer, init_t, denoising_strength, text, mask,
                    cfg["num_inference_steps"], cfg["cfg_scale"], cfg["cfg_interval"],
                    device, dtype, seed=cfg["seed"], neg_text=neg_text, neg_mask=neg_mask,
                    progress_callback=progress_callback,
                )
            finally:
                if applied_lora:
                    self._unload_lora_minit2i()
            image = tensor_to_image(x)
            print("[MiniT2I] img2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[MiniT2I] img2img error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._minit2i_cleanup()

    def _generate_inpaint_minit2i(self, params, init_image, mask_image, progress_callback=None, step_callback=None) -> tuple:
        if not self.minit2i_components:
            raise RuntimeError("MiniT2I components not loaded.")
        from core.models.minit2i.minit2i_pipeline_ops import (
            denoise_loop_inpaint, image_to_tensor, tensor_to_image, prepare_mask,
        )
        print("[MiniT2I] Starting inpaint generation (repaint)")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._minit2i_common_params(params, init_image.width, init_image.height)
        denoising_strength = float(params.get("denoising_strength", 0.8))
        try:
            text, mask, neg_text, neg_mask = self._minit2i_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg["prompt_length"], device, dtype)
            init_t = image_to_tensor(init_image, cfg["height"], cfg["width"], device, dtype)
            mask_latent = prepare_mask(mask_image, cfg["height"], cfg["width"], device, dtype)
            transformer = self._minit2i_move("transformer", device)
            applied_lora = self._load_lora_minit2i(params.get("loras") or [])
            try:
                x = denoise_loop_inpaint(
                    transformer, init_t, mask_latent, denoising_strength, text, mask,
                    cfg["num_inference_steps"], cfg["cfg_scale"], cfg["cfg_interval"],
                    device, dtype, seed=cfg["seed"], neg_text=neg_text, neg_mask=neg_mask,
                    progress_callback=progress_callback,
                )
            finally:
                if applied_lora:
                    self._unload_lora_minit2i()
            image = tensor_to_image(x)
            print("[MiniT2I] inpaint completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[MiniT2I] inpaint error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._minit2i_cleanup()

    def cancel_generation(self):
        """Request cancellation of current generation"""
        self.cancel_requested = True
        print("[Pipeline] Generation cancellation requested")

    def reset_cancel_flag(self):
        """Reset cancellation flag before starting new generation"""
        self.cancel_requested = False

# Global pipeline manager instance
pipeline_manager = DiffusionPipelineManager()
