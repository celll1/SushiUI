from typing import Dict, Any, Optional, List, Callable
from PIL import Image
import torch
import json
import os
import sys
import gc
import time
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
from core.inference.generation_timing import generation_timer
from core.pipeline_backends import ZImageMixin, Flux2Mixin, AnimaMixin, LensMixin, Ideogram4Mixin, MiniT2IMixin, Krea2Mixin

LAST_MODEL_CONFIG_FILE = Path("last_model.json")

class DiffusionPipelineManager(ZImageMixin, Flux2Mixin, AnimaMixin, LensMixin, Ideogram4Mixin, MiniT2IMixin, Krea2Mixin):
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

        # Krea 2 components (single-stream MMDiT + Qwen3-VL + Qwen-Image VAE). Flow matching.
        self.krea2_components: Optional[Dict[str, Any]] = None
        self.is_krea2_model: bool = False

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
        if self.is_krea2_model:
            return "krea2"
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

            # Clean up Krea 2 components
            if self.krea2_components is not None:
                print("[Pipeline] Cleaning up Krea 2 components...")
                for comp_name, comp in self.krea2_components.items():
                    if comp is not None and hasattr(comp, 'to'):
                        try:
                            comp.to('cpu')
                        except Exception:
                            pass
                    del comp
                self.krea2_components = None
                self.is_krea2_model = False

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

            # Check if Krea 2 (single-stream MMDiT + Qwen3-VL + Qwen-Image VAE). Before the
            # generic Z-Image check (which matches any dict carrying a "transformer" key).
            if isinstance(model_result, dict) and model_result.get("type") == "krea2":
                print("[Pipeline] Krea 2 model detected (component-based dict returned)")
                self.krea2_components = model_result
                self.is_krea2_model = True
                self.is_minit2i_model = False
                self.is_ideogram4_model = False
                self.is_lens_model = False
                self.is_anima_model = False
                self.is_zimage_model = False
                self.is_flux2_model = False
                self.current_model = model_id
                self.current_attention_type = "normal"

                for comp_name in ("transformer", "text_encoder", "vae"):
                    comp = self.krea2_components.get(comp_name)
                    if comp is not None and hasattr(comp, "to"):
                        try:
                            comp.to("cpu")
                        except Exception:
                            pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[VRAM] Krea 2 components on CPU. Will load to GPU as needed.")

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
                    "type": "krea2",
                    "is_v_prediction": False,  # flow matching, velocity prediction
                    "model_hash": model_hash,
                    "is_distilled": self.krea2_components.get("is_distilled", False),
                }
                self._save_last_model(source_type, source, pipeline_type)
                print("[Pipeline] Krea 2 model loaded successfully")
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

            model_type_detected = ModelLoader.detect_model_type(source) if source_type != "huggingface" else "unknown"

            # Unified prediction config (SushiUI modelspec.* > v_pred marker > legacy >
            # architecture inference). Lets the loader know the objective (epsilon / v /
            # x / flow) the model was trained with, so the right scheduler is selected.
            noise_process = None
            prediction_target = None
            pred_source = None
            if source_type in ["safetensors", "diffusers"] and os.path.exists(source):
                try:
                    _pc = ModelLoader.detect_prediction_config(source, model_type_detected)
                    noise_process = _pc.get("noise_process")
                    prediction_target = _pc.get("prediction_target")
                    pred_source = _pc.get("source")
                    # Keep is_v_prediction consistent with the unified config.
                    if prediction_target == "velocity" and noise_process == "ddpm":
                        is_v_prediction = True
                    print(f"[Pipeline] Prediction config: noise_process={noise_process}, "
                          f"prediction_target={prediction_target} (source={pred_source})")
                    if noise_process == "flow":
                        # Flow-matching inference sampling is not yet wired into the custom
                        # SDXL sampler (which assumes DDPM/v sigmas). Recognized here so the
                        # model is correctly tagged; a dedicated flow sampler is a follow-up.
                        print("[Pipeline] WARNING: flow-matching (FM) prediction detected — "
                              "FM inference sampling is not yet implemented; output may be incorrect.")
                except Exception as _pc_e:
                    print(f"[Pipeline] detect_prediction_config failed (using scheduler default): {_pc_e}")

            self.current_model_info = {
                "source_type": source_type,
                "source": source,
                "type": model_type_detected,
                "is_v_prediction": is_v_prediction,
                "noise_process": noise_process,
                "prediction_target": prediction_target,
                "prediction_source": pred_source,
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

    def _apply_vae_tiling(self, vae, enabled: bool):
        """Enable (or disable) VAE tiling + slicing for the upcoming decode.

        Tiling decodes the latent in overlapping tiles so the VAE decode peak is
        bounded by the tile size rather than the full image -- the main lever for
        decoding large images without OOM. The setting persists on the VAE object,
        so we explicitly enable or DISABLE it every time to honour the per-request
        option. Routes through wrappers (SDXLVAEWrapper / FluxVAEWrapper) that hold
        an inner diffusers AutoencoderKL.

        The THRESHOLD (image size above which tiling kicks in) is configurable via
        self._vae_tile_threshold (px). diffusers couples the threshold and the tile
        size through tile_latent_min_size, so we set both to the threshold: below it
        the decode runs whole (bit-identical, no quality/speed cost), above it the
        image is split into threshold-sized tiles -- i.e. tiles as big as the size
        you're comfortable decoding un-tiled. 0 -> auto = VAE sample_size * 1.5.
        """
        if vae is None:
            return
        threshold_px = int(getattr(self, "_vae_tile_threshold", 0) or 0)
        targets = [vae]
        inner = getattr(vae, "vae", None)
        if inner is not None and inner is not vae:
            targets.append(inner)
        for t in targets:
            for on_name, off_name in (("enable_tiling", "disable_tiling"),
                                      ("enable_slicing", "disable_slicing")):
                method = on_name if enabled else off_name
                if hasattr(t, method):
                    try:
                        getattr(t, method)()
                    except Exception as e:
                        print(f"[VAE Tiling] {method} failed: {e}")
            if enabled and hasattr(t, "tile_latent_min_size"):
                try:
                    cfg = getattr(t, "config", None)
                    boc = getattr(cfg, "block_out_channels", None) if cfg else None
                    scale = 2 ** (len(boc) - 1) if boc else 8
                    sample = (getattr(cfg, "sample_size", None) if cfg else None) or 1024
                    thr = threshold_px if threshold_px > 0 else int(sample * 1.5)
                    thr = max(int(scale), thr)            # at least one tile
                    t.tile_sample_min_size = int(thr)
                    t.tile_latent_min_size = max(1, int(thr / scale))
                except Exception as e:
                    print(f"[VAE Tiling] threshold setup failed: {e}")
        if enabled:
            _thr = threshold_px if threshold_px > 0 else "auto(sample_size*1.5)"
            print(f"[VAE Tiling] enabled (tiles when image > {_thr}px; tile size = threshold)")

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

    def _negpip_eligible(self, prompt: str, negative_prompt: str, pipeline) -> bool:
        """Whether NegPip should auto-activate for this prompt pair.

        Trigger: any negative emphasis weight (e.g. (worst:-1)) in either prompt.
        Supports all chunking modes (a1111 / sd_scripts / nobos), single- and multi-
        chunk: the signed-weight builder mirrors each mode's BOS/EOS layout. Only the
        custom swapped text encoder (which bypasses CLIP/emphasis/chunking) falls back.
        """
        from core.prompts.prompt_parser import prompt_has_negative_weight
        if not (prompt_has_negative_weight(prompt) or prompt_has_negative_weight(negative_prompt)):
            return False
        if getattr(pipeline, "_sushi_te", None) is not None:
            print("[NegPip] Negative weight detected, but a custom text encoder is active -> "
                  "falling back (NegPip needs the standard CLIP path)")
            try:
                from api.generation_status import add_warning
                add_warning(
                    "NegPip (negative-weight emphasis) disabled: a custom text encoder is "
                    "active and NegPip needs the standard CLIP path",
                    code="feature_auto_disabled",
                )
            except Exception:
                pass
            return False
        if getattr(pipeline, "tokenizer", None) is None:
            return False
        return True

    def _build_negpip_weights(self, prompt: str, negative_prompt: str, pipeline,
                              prompt_embeds, negative_prompt_embeds, dtype,
                              nag_negative_prompt: str = None, nag_negative_prompt_embeds=None):
        """Build signed per-token weight vectors aligned with the encoded embeddings.

        Returns {"pos","neg"[, "nag_neg"]} where each value is a 1-D tensor of length
        equal to the corresponding embedding sequence (1.0 on BOS/EOS/padding).
        """
        from core.prompts.prompt_parser import build_signed_weight_vector_chunked
        is_sdxl = hasattr(pipeline, "text_encoder_2") and pipeline.text_encoder_2 is not None
        tokenizer = pipeline.tokenizer_2 if is_sdxl else pipeline.tokenizer
        device = self.device
        mode = getattr(self, "prompt_chunking_mode", "a1111")
        max_chunks = getattr(self, "max_prompt_chunks", 0)

        def _wv(text, embeds):
            return build_signed_weight_vector_chunked(
                text or "", embeds.shape[1], tokenizer, device, dtype,
                mode=mode, max_chunks=max_chunks,
            )

        pos_w = _wv(prompt, prompt_embeds)
        neg_w = _wv(negative_prompt, negative_prompt_embeds) if negative_prompt_embeds is not None else None
        weights = {"pos": pos_w, "neg": neg_w}
        if nag_negative_prompt_embeds is not None:
            weights["nag_neg"] = _wv(nag_negative_prompt, nag_negative_prompt_embeds)
        return weights

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
                    if layer_weights:
                        try:
                            from api.generation_status import add_warning
                            add_warning(
                                "ControlNet layer weights are not supported for LLLite models "
                                "and were ignored",
                                code="not_implemented",
                            )
                        except Exception:
                            pass
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

    def _encode_prompt_chunked(self, prompt: str, negative_prompt: str = "", pipeline=None, skip_emphasis: bool = False):
        """
        Encode prompts with chunking support for long prompts (>75 tokens).
        Uses pipeline.encode_prompt() for each chunk to ensure correct encoding.

        skip_emphasis: when True, return CLEAN embeddings (no emphasis scaling) for
        NegPip. Only correct for a1111 chunking (77 tokens/chunk), which is what the
        signed-weight position mapping assumes; the caller (_negpip_eligible) gates
        chunked NegPip to a1111 mode.

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

        # Apply emphasis weights if present (skipped for NegPip, which weights V)
        if has_pos_emphasis and not skip_emphasis:
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

            # Apply emphasis weights (skipped for NegPip, which weights V)
            if has_neg_emphasis and not skip_emphasis:
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

    def _encode_prompt_nobos_single_chunk(self, prompt: str, negative_prompt: str = "", pipeline=None, skip_emphasis: bool = False):
        """
        Encode prompts with NoBOS mode for single chunk (<=75 tokens).
        Strips BOS and EOS tokens from embeddings.

        skip_emphasis: when True, return CLEAN embeddings (no emphasis scaling) for NegPip.

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

        # Apply emphasis weights if present (skipped for NegPip, which weights V)
        if has_pos_emphasis and not skip_emphasis:
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

            # Apply emphasis weights if present (skipped for NegPip, which weights V)
            if has_neg_emphasis and not skip_emphasis:
                negative_prompt_embeds = apply_emphasis_to_embeds(
                    negative_prompt, negative_prompt_embeds,
                    tokenizer,
                    device, dtype
                )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _custom_te_encode(self, pipeline, prompt: str, negative_prompt: str = ""):
        """Encode prompts with a swapped SDXL text encoder + bridge adapters (inference).

        Returns the SDXL 4-tuple (prompt_embeds[1,L,2048], negative_prompt_embeds,
        pooled[1,1280], negative_pooled). Fixed-length; emphasis weights / chunking are
        not applied for custom encoders in this first version.
        """
        from core.models.sdxl_te_registry import encode_text
        enc = pipeline._sushi_te
        tok = pipeline._sushi_te_tokenizer
        ad = pipeline._sushi_te_adapters
        max_len = getattr(pipeline, "_sushi_te_max_len", 256)
        hidden_layer = getattr(pipeline, "_sushi_te_hidden_layer", -2)
        ad_dtype = next(ad.parameters()).dtype
        with torch.no_grad():
            h, p = encode_text(enc, tok, [prompt or ""], max_len=max_len,
                               hidden_layer=hidden_layer, device=self.device)
            nh, np_ = encode_text(enc, tok, [negative_prompt or ""], max_len=max_len,
                                  hidden_layer=hidden_layer, device=self.device)
            pe, pp = ad(h.to(ad_dtype), p.to(ad_dtype))
            ne, npp = ad(nh.to(ad_dtype), np_.to(ad_dtype))
        return pe, ne, pp, npp

    def _encode_prompt_with_weights(self, prompt: str, negative_prompt: str = "", pipeline=None, skip_emphasis: bool = False):
        """
        Encode prompts with A1111-style emphasis weights and/or chunking.

        Args:
            skip_emphasis: When True, return CLEAN embeddings (no emphasis scaling) for
                the single-chunk path. Used by NegPip, which applies signed per-token
                weights to V instead of scaling the embedding. Only honored on the
                single-chunk path (NegPip v1 scope).

        Returns:
            For SD1.5: (prompt_embeds, negative_prompt_embeds)
            For SDXL: (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        """
        # Use provided pipeline or default to txt2img_pipeline
        if pipeline is None:
            pipeline = self.txt2img_pipeline

        if pipeline is None:
            return None, None, None, None

        # Custom SDXL text encoder (swapped at train time): bypass CLIP / emphasis /
        # chunking and use the attached encoder + bridge adapters at fixed length.
        if getattr(pipeline, "_sushi_te", None) is not None:
            return self._custom_te_encode(pipeline, prompt, negative_prompt)

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
            return self._encode_prompt_chunked(prompt, negative_prompt, pipeline, skip_emphasis=skip_emphasis)
        elif needs_nobos_processing:
            # Even for <=75 tokens, apply NoBOS processing
            return self._encode_prompt_nobos_single_chunk(prompt, negative_prompt, pipeline, skip_emphasis=skip_emphasis)

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

        # Apply emphasis weights (skipped for NegPip, which weights V instead)
        if has_pos_emphasis and not skip_emphasis:
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

            if has_neg_emphasis and not skip_emphasis:
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
        # VAE tiling flag for this request (read by all decode paths, incl. the
        # per-architecture handlers dispatched just below).
        self._vae_tiling = bool(params.get("vae_tiling", False))
        self._vae_tile_threshold = int(params.get("vae_tile_threshold", 0) or 0)
        # Color Flatten (chroma smoothing) strength for this request; read by all
        # decode funnels via getattr(self, "_color_flatten_strength", 0). <=0 is a no-op.
        self._color_flatten_strength = int(params.get("color_flatten_strength", 0) or 0)
        # In-loop hard-flatten (SD1.5/SDXL): master switch + last-N steps + region gate.
        self._flatten_in_loop = bool(params.get("flatten_in_loop", False))
        self._flatten_in_loop_last_steps = int(params.get("flatten_in_loop_last_steps", 3) or 3)
        self._flatten_in_loop_min_region = float(params.get("flatten_in_loop_min_region", 0.02) or 0.02)
        # VAE DC-drift correction is img2img/inpaint only; force off for txt2img.
        self._vae_drift_correction = False

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
        if self.is_krea2_model:
            return self._generate_txt2img_krea2(params, progress_callback, step_callback)

        if not self.txt2img_pipeline:
            raise RuntimeError("txt2img pipeline not loaded. Please load a model first.")

        # VAE tiling option: decode bounded by tile size (large-image OOM relief).
        self._apply_vae_tiling(getattr(self.txt2img_pipeline, "vae", None),
                               bool(params.get("vae_tiling", False)))

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

        # NegPip auto-activation: when the prompt(s) carry negative emphasis weights,
        # encode CLEAN embeddings (skip embedding scaling) and apply the signed weights
        # to V inside attention instead. Disabled when prompt editing is active (the
        # per-step edit embeds path is not yet NegPip-aware -- v1 scope).
        _negpip_neg_prompt = params.get("negative_prompt", "")
        use_negpip = (prompt_processor is None) and self._negpip_eligible(
            initial_prompt, _negpip_neg_prompt, self.txt2img_pipeline
        )

        # Encode prompts with weights if emphasis syntax is present
        with generation_timer.phase("text_encode"):
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt_with_weights(
                initial_prompt,
                params.get("negative_prompt", ""),
                pipeline=self.txt2img_pipeline,
                skip_emphasis=use_negpip,
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

        # Build NegPip signed per-token weights (clean embeds were encoded above)
        negpip_weights = None
        if use_negpip:
            _negpip_dtype = self.txt2img_pipeline.dtype if hasattr(self.txt2img_pipeline, "dtype") else torch.float16
            negpip_weights = self._build_negpip_weights(
                initial_prompt, _negpip_neg_prompt, self.txt2img_pipeline,
                prompt_embeds, negative_prompt_embeds, _negpip_dtype,
                nag_negative_prompt=params.get("nag_negative_prompt", "") or params.get("negative_prompt", ""),
                nag_negative_prompt_embeds=nag_negative_prompt_embeds,
            )
            print(f"[NegPip] Auto-activated (negative emphasis weights detected)")

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

            # Call custom sampling loop. The legacy SD/SDXL path folds VAE decode
            # into this loop, so denoise and decode are not separable here — the
            # combined span is recorded as the "denoise" phase.
            _t_denoise = time.perf_counter()
            image = custom_sampling_loop(
                pipeline=pipeline_to_use,
                color_flatten_strength=getattr(self, "_color_flatten_strength", 0),
                flatten_in_loop=getattr(self, "_flatten_in_loop", False),
                flatten_in_loop_last_steps=getattr(self, "_flatten_in_loop_last_steps", 3),
                flatten_in_loop_min_region=getattr(self, "_flatten_in_loop_min_region", 0.02),
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
                original_size_w=params.get("original_size_w", 0),
                original_size_h=params.get("original_size_h", 0),
                original_size_scale=params.get("original_size_scale", 1.0),
                negpip_weights=negpip_weights,
                spectrum_enable=params.get("spectrum_enable", False),
                spectrum_w=params.get("spectrum_w", 0.5),
                spectrum_w_decay=params.get("spectrum_w_decay", 0.0),
                spectrum_delta_cap=params.get("spectrum_delta_cap", 0.0),
                spectrum_m=params.get("spectrum_m", 4),
                spectrum_lam=params.get("spectrum_lam", 0.1),
                spectrum_warmup_steps=params.get("spectrum_warmup_steps", 3),
                spectrum_window_size=params.get("spectrum_window_size", 4),
                spectrum_flex_window=params.get("spectrum_flex_window", 0.75),
                spectrum_tail=params.get("spectrum_tail", 0.12),
                spectrum_feature_mode=params.get("spectrum_feature_mode", "output"),
                spectrum_cache_branch=params.get("spectrum_cache_branch", 1),
                spectrum_max_cache=params.get("spectrum_max_cache", 0),
                fbcache_enable=params.get("fbcache_enable", False),
                fbcache_threshold=params.get("fbcache_threshold", 0.12),
                fbcache_warmup_steps=params.get("fbcache_warmup_steps", 1),
                fbcache_cache_branch=params.get("fbcache_cache_branch", 1),
                **controlnet_kwargs,
            )
            generation_timer.add("denoise", time.perf_counter() - _t_denoise)

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
        self._vae_tiling = bool(params.get("vae_tiling", False))
        self._vae_tile_threshold = int(params.get("vae_tile_threshold", 0) or 0)
        self._color_flatten_strength = int(params.get("color_flatten_strength", 0) or 0)
        # In-loop hard-flatten (SD1.5/SDXL): master switch + last-N steps + region gate.
        self._flatten_in_loop = bool(params.get("flatten_in_loop", False))
        self._flatten_in_loop_last_steps = int(params.get("flatten_in_loop_last_steps", 3) or 3)
        self._flatten_in_loop_min_region = float(params.get("flatten_in_loop_min_region", 0.02) or 0.02)
        self._vae_drift_correction = bool(params.get("vae_drift_correction", False))

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
        if self.is_krea2_model:
            return self._generate_img2img_krea2(params, init_image, progress_callback, step_callback)

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

        # VAE tiling option: decode bounded by tile size (large-image OOM relief).
        self._apply_vae_tiling(getattr(self.img2img_pipeline, "vae", None),
                               bool(params.get("vae_tiling", False)))

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

        # NegPip auto-activation (same as txt2img): clean embeds + signed V weights
        # when a negative emphasis weight is present (and not prompt-editing / chunked).
        _negpip_neg_prompt = params.get("negative_prompt", "")
        use_negpip = (prompt_processor is None) and self._negpip_eligible(
            initial_prompt, _negpip_neg_prompt, pipeline_to_use
        )

        # Encode prompts with weights if emphasis syntax is present
        with generation_timer.phase("text_encode"):
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt_with_weights(
                initial_prompt,
                params.get("negative_prompt", ""),
                pipeline=pipeline_to_use,
                skip_emphasis=use_negpip,
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

        # Build NegPip signed per-token weights (clean embeds were encoded above)
        negpip_weights = None
        if use_negpip:
            _negpip_dtype = pipeline_to_use.dtype if hasattr(pipeline_to_use, "dtype") else torch.float16
            negpip_weights = self._build_negpip_weights(
                initial_prompt, _negpip_neg_prompt, pipeline_to_use,
                prompt_embeds, negative_prompt_embeds, _negpip_dtype,
                nag_negative_prompt=params.get("nag_negative_prompt", "") or params.get("negative_prompt", ""),
                nag_negative_prompt_embeds=nag_negative_prompt_embeds,
            )
            print(f"[NegPip] Auto-activated (img2img, negative emphasis weights detected)")

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

            # Call custom img2img sampling loop. VAE decode is folded into the loop
            # on this legacy path, so the combined span is recorded as "denoise".
            _t_denoise = time.perf_counter()
            image = custom_img2img_sampling_loop(
                pipeline=pipeline_to_use,
                color_flatten_strength=getattr(self, "_color_flatten_strength", 0),
                flatten_in_loop=getattr(self, "_flatten_in_loop", False),
                flatten_in_loop_last_steps=getattr(self, "_flatten_in_loop_last_steps", 3),
                flatten_in_loop_min_region=getattr(self, "_flatten_in_loop_min_region", 0.02),
                vae_drift_correction=getattr(self, "_vae_drift_correction", False),
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
                original_size_w=params.get("original_size_w", 0),
                original_size_h=params.get("original_size_h", 0),
                original_size_scale=params.get("original_size_scale", 1.0),
                negpip_weights=negpip_weights,
                spectrum_enable=params.get("spectrum_enable", False),
                spectrum_w=params.get("spectrum_w", 0.5),
                spectrum_w_decay=params.get("spectrum_w_decay", 0.0),
                spectrum_delta_cap=params.get("spectrum_delta_cap", 0.0),
                spectrum_m=params.get("spectrum_m", 4),
                spectrum_lam=params.get("spectrum_lam", 0.1),
                spectrum_warmup_steps=params.get("spectrum_warmup_steps", 3),
                spectrum_window_size=params.get("spectrum_window_size", 4),
                spectrum_flex_window=params.get("spectrum_flex_window", 0.75),
                spectrum_tail=params.get("spectrum_tail", 0.12),
                spectrum_feature_mode=params.get("spectrum_feature_mode", "output"),
                spectrum_cache_branch=params.get("spectrum_cache_branch", 1),
                spectrum_max_cache=params.get("spectrum_max_cache", 0),
                fbcache_enable=params.get("fbcache_enable", False),
                fbcache_threshold=params.get("fbcache_threshold", 0.12),
                fbcache_warmup_steps=params.get("fbcache_warmup_steps", 1),
                fbcache_cache_branch=params.get("fbcache_cache_branch", 1),
                **controlnet_kwargs,
            )
            generation_timer.add("denoise", time.perf_counter() - _t_denoise)

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
        self._vae_tiling = bool(params.get("vae_tiling", False))
        self._vae_tile_threshold = int(params.get("vae_tile_threshold", 0) or 0)
        self._color_flatten_strength = int(params.get("color_flatten_strength", 0) or 0)
        # In-loop hard-flatten (SD1.5/SDXL): master switch + last-N steps + region gate.
        self._flatten_in_loop = bool(params.get("flatten_in_loop", False))
        self._flatten_in_loop_last_steps = int(params.get("flatten_in_loop_last_steps", 3) or 3)
        self._flatten_in_loop_min_region = float(params.get("flatten_in_loop_min_region", 0.02) or 0.02)
        self._vae_drift_correction = bool(params.get("vae_drift_correction", False))

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
        if self.is_krea2_model:
            return self._generate_inpaint_krea2(params, init_image, mask_image, progress_callback, step_callback)

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

        # VAE tiling option: decode bounded by tile size (large-image OOM relief).
        self._apply_vae_tiling(getattr(self.inpaint_pipeline, "vae", None),
                               bool(params.get("vae_tiling", False)))

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

        # NegPip auto-activation (same as txt2img): clean embeds + signed V weights
        # when a negative emphasis weight is present (and not prompt-editing / chunked).
        _negpip_neg_prompt = params.get("negative_prompt", "")
        use_negpip = (prompt_processor is None) and self._negpip_eligible(
            initial_prompt, _negpip_neg_prompt, pipeline_to_use
        )

        # Encode initial prompt
        with generation_timer.phase("text_encode"):
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt_with_weights(
                initial_prompt,
                params.get("negative_prompt", ""),
                pipeline=pipeline_to_use,
                skip_emphasis=use_negpip,
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

        # Build NegPip signed per-token weights (clean embeds were encoded above)
        negpip_weights = None
        if use_negpip:
            _negpip_dtype = pipeline_to_use.dtype if hasattr(pipeline_to_use, "dtype") else torch.float16
            negpip_weights = self._build_negpip_weights(
                initial_prompt, _negpip_neg_prompt, pipeline_to_use,
                prompt_embeds, negative_prompt_embeds, _negpip_dtype,
                nag_negative_prompt=params.get("nag_negative_prompt", "") or params.get("negative_prompt", ""),
                nag_negative_prompt_embeds=nag_negative_prompt_embeds,
            )
            print(f"[NegPip] Auto-activated (inpaint, negative emphasis weights detected)")

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

        # Use custom inpaint sampling loop. VAE decode is folded into the loop on
        # this legacy path, so the combined span is recorded as "denoise".
        _t_denoise = time.perf_counter()
        image = custom_inpaint_sampling_loop(
            pipeline=pipeline_to_use,
            color_flatten_strength=getattr(self, "_color_flatten_strength", 0),
            vae_drift_correction=getattr(self, "_vae_drift_correction", False),
            flatten_in_loop=getattr(self, "_flatten_in_loop", False),
            flatten_in_loop_last_steps=getattr(self, "_flatten_in_loop_last_steps", 3),
            flatten_in_loop_min_region=getattr(self, "_flatten_in_loop_min_region", 0.02),
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
            original_size_w=params.get("original_size_w", 0),
            original_size_h=params.get("original_size_h", 0),
            original_size_scale=params.get("original_size_scale", 1.0),
            negpip_weights=negpip_weights,
            spectrum_enable=params.get("spectrum_enable", False),
            spectrum_w=params.get("spectrum_w", 0.5),
            spectrum_w_decay=params.get("spectrum_w_decay", 0.0),
            spectrum_delta_cap=params.get("spectrum_delta_cap", 0.0),
            spectrum_m=params.get("spectrum_m", 4),
            spectrum_lam=params.get("spectrum_lam", 0.1),
            spectrum_warmup_steps=params.get("spectrum_warmup_steps", 3),
            spectrum_window_size=params.get("spectrum_window_size", 4),
            spectrum_flex_window=params.get("spectrum_flex_window", 0.75),
            spectrum_tail=params.get("spectrum_tail", 0.12),
            spectrum_feature_mode=params.get("spectrum_feature_mode", "output"),
            spectrum_cache_branch=params.get("spectrum_cache_branch", 1),
            spectrum_max_cache=params.get("spectrum_max_cache", 0),
            fbcache_enable=params.get("fbcache_enable", False),
            fbcache_threshold=params.get("fbcache_threshold", 0.12),
            fbcache_warmup_steps=params.get("fbcache_warmup_steps", 1),
            fbcache_cache_branch=params.get("fbcache_cache_branch", 1),
            **controlnet_kwargs,
        )
        generation_timer.add("denoise", time.perf_counter() - _t_denoise)

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

    def cancel_generation(self):
        """Request cancellation of current generation"""
        self.cancel_requested = True
        print("[Pipeline] Generation cancellation requested")

    def reset_cancel_flag(self):
        """Reset cancellation flag before starting new generation"""
        self.cancel_requested = False

# Global pipeline manager instance
pipeline_manager = DiffusionPipelineManager()
