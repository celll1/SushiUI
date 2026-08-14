from typing import Dict, Any, Optional, List, Callable, Union
from PIL import Image
import torch
import json
import os
import sys
import gc
import time
import random
import threading
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
from core.pipeline_backends import ZImageMixin, Flux2Mixin, AnimaMixin, LensMixin, Ideogram4Mixin, MiniT2IMixin, Krea2Mixin, LTX2Mixin, AceStepMixin, MiniMaxH3Mixin, MiniMaxMusic3Mixin

LAST_MODEL_CONFIG_FILE = Path("last_model.json")

class DiffusionPipelineManager(ZImageMixin, Flux2Mixin, AnimaMixin, LensMixin, Ideogram4Mixin, MiniT2IMixin, Krea2Mixin, LTX2Mixin, AceStepMixin, MiniMaxH3Mixin, MiniMaxMusic3Mixin):
    def __init__(self):
        self.txt2img_pipeline: Optional[StableDiffusionPipeline] = None
        self.img2img_pipeline: Optional[StableDiffusionImg2ImgPipeline] = None
        self.inpaint_pipeline: Optional[StableDiffusionInpaintPipeline] = None
        self.current_model: Optional[str] = None
        self.current_model_info: Optional[Dict[str, Any]] = None
        self.extensions: List[BaseExtension] = []
        self.device = settings.device
        self.model_revision = 0
        self.component_revision = 0
        self.component_health = "unloaded"

        # Serializes model loading. The boot-time auto-load runs in a background
        # thread (main.py _auto_load_last_model) and can race a concurrent manual
        # POST /models/load: interleaved cleanup (which resets every is_<arch>_model
        # flag) + component/current_model assignment left is_<arch>_model False while
        # current_model/current_model_info pointed at a loaded model, so a repeat
        # /models/load with the same source hit the model-id early-return no-op and
        # never restored the flag. Holding this lock makes each load atomic, so the
        # early-return only fires against fully, consistently loaded state.
        self._load_model_lock = threading.Lock()

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

        # LTX-2.3 components (joint audio+video MM-DiT + Gemma-3 + LTX2 VAEs). Video
        # model; flow matching. P1a: loadable/slot-switchable only. Video generation
        # (txt2vid/img2vid) is P1b — image endpoints reject a loaded LTX-2.3 model.
        self.ltx2_components: Optional[Dict[str, Any]] = None
        self.is_ltx2_model: bool = False
        self._ltx2_offload_enabled: bool = False

        # ACE-Step 1.5 components (2B DiT + Oobleck VAE + Qwen3-Embedding-0.6B
        # text encoder). Audio/music model; flow matching. Phase 0+1: loadable/
        # slot-switchable only — no sampler/generation entry point yet (Phase 2).
        self.acestep_components: Optional[Dict[str, Any]] = None
        self.is_acestep_model: bool = False

        # MiniMax-H3 components (pruned joint video+audio DiT + Qwen3-VL-32B text
        # encoder + a 24ch video VAE and a 32ch audio VAE). Video model; flow
        # matching. Phase 1: loadable/slot-switchable only — video generation is
        # Phase 2, so the image and audio endpoints reject a loaded H3 model.
        #
        # NOTE for every future consumer: nothing may call `.to(device, dtype)`
        # on `minimax_h3_components["text_encoder"]`. Its 48 GiB of CPU weights
        # are memory-mapped from the file by `load_state_dict(assign=True)`, and
        # writing them back detaches every parameter from the mapping (MEASURED:
        # 73.08 GB peak RSS + pagefile growth, against 49.82 GB flat for the
        # `torch.func.functional_call` streaming Phase 2 uses).
        self.minimax_h3_components: Optional[Dict[str, Any]] = None
        self.is_minimax_h3_model: bool = False
        # The (text_encoder_file, clip_projection_file) the loaded H3 pairing was
        # requested with, so `last_model.json` can replay the same choice and the
        # DiT-only reload (which rebuilds nothing but the DiT) does not erase it.
        self._minimax_h3_te_request: tuple = (None, None)

        # MiniMax Music 3 components (2.4B flow-matching DiT + 8B Qwen3
        # language model + 0.6B RVQ depth decoder + condition encoder +
        # vocoder). Loadable/slot-switchable only; generation is
        # pipeline_backends/minimax_music3.py (a later commit).
        self.minimax_music3_components: Optional[Dict[str, Any]] = None
        self.is_minimax_music3_model: bool = False

        # SigLIP2 Vision Encoder (optional, for SD/SDXL vision-conditioned generation)
        self.vision_encoder: Optional[Any] = None
        self._vision_encoder_path: Optional[str] = None

        # Per-generation component overrides (RP2b). Idempotent, path-keyed. The
        # ORIGINAL component ref is kept so clearing the override restores WITHOUT
        # a reload. Applied component is CPU-resident (the vram_optimization
        # move_vae_to_gpu/cpu funnel stages it per generation).
        self._override_vae_path: Optional[str] = None
        self._original_vae: Optional[Any] = None
        self._override_vae_targets: List[Any] = []
        self._override_te_path: Optional[str] = None
        self._original_te: Optional[Dict[str, Any]] = None

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
        if self.is_ltx2_model:
            return "ltx2"
        if self.is_acestep_model:
            return "acestep"
        if self.is_minimax_h3_model:
            return "minimax_h3"
        if self.is_minimax_music3_model:
            return "minimax_music3"
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
        force_reload: bool = False,
        text_encoder_file: Optional[str] = None,
        clip_projection_file: Optional[str] = None,
        hybrid: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Load a model, holding the lifecycle gate for the duration.

        Concurrent loads no longer queue behind each other: the gate refuses
        the second one with ModelStateBusyError (409) rather than serializing
        it, so a manual /models/load arriving during the boot auto-load is
        rejected and retried by the caller instead of silently applying after
        it. ``_load_model_lock`` is still taken inside the gate, as the
        in-process mutual exclusion the component switcher also relies on.

        ``force_reload`` bypasses the same-model early return below. It exists
        because that early return made "reload the model" -- the documented (and
        only) way to undo an in-place runtime INT8 conversion, a VAE/TE override,
        or any other per-session mutation of the loaded components -- a silent
        no-op when the user re-selected the SAME checkpoint.

        ``text_encoder_file``/``clip_projection_file`` choose MiniMax-H3's
        load-time text encoder and its trained projection. Naming an encoder
        other than the loaded one reloads on its own: the model id does not
        change with the encoder, so depending on the caller to send
        ``force_reload`` would make an ignored request look like a successful
        one.

        ``hybrid`` is MiniMax-H3's base+overlay DiT request -- a mapping of
        ``HYBRID_REQUEST_KEYS`` (``overlay_file`` and the recipe). It IS part of
        the model id (see ``_load_model_locked``), so a different overlay or a
        different block range reloads without ``force_reload``. There is no API
        surface for it yet; C5 owns that."""
        from core.model_state_coordinator import model_state_coordinator
        previous_info = self.current_model_info
        mutation_started = False
        try:
            with model_state_coordinator.mutation("model load"):
                mutation_started = True
                with self._load_model_lock:
                    result = self._load_model_locked(
                        source_type, source, pipeline_type, force_reload=force_reload,
                        text_encoder_file=text_encoder_file,
                        clip_projection_file=clip_projection_file,
                        hybrid=hybrid, **kwargs)
        except Exception:
            if mutation_started:
                # Degraded means "the live components are not trustworthy", so
                # decide it from what the failure actually left behind, not from
                # whether a model was loaded beforehand. The H3 DiT-only path
                # builds the replacement before it swaps and keeps the current
                # model on failure (_reload_minimax_h3_dit_only); calling that
                # degraded would disable generation on a model that is fine.
                if self.current_model_info is None:
                    self.component_health = "unloaded"
                elif self.current_model_info is not previous_info:
                    self.component_health = "degraded"
            raise

        if self.current_model_info is not None and self.current_model_info is not previous_info:
            self.model_revision += 1
            self.component_revision += 1
            self.component_health = "ready"

        # Auto-discover a per-model `loras/` sibling directory (e.g.
        # M:/model/minimax_h3/loras next to diffusion_models/text_encoders/vae)
        # and register it as a seeded LoRA search dir. Only runs after a
        # confirmed-successful load of THIS source (current_model_info can be
        # left stale/None on failure, or reflect a different model when the
        # same-model early return fired above).
        try:
            if self.current_model_info and self.current_model_info.get("source") == source:
                from core.extensions.lora_manager import lora_manager
                lora_manager.register_model_sibling_loras(source)
        except Exception as exc:
            print(f"[Pipeline] LoRA sibling directory auto-discovery skipped: {exc}")

        return result

    def _load_model_locked(
        self,
        source_type: ModelSource,
        source: str,
        pipeline_type: str = "txt2img",
        force_reload: bool = False,
        text_encoder_file: Optional[str] = None,
        clip_projection_file: Optional[str] = None,
        hybrid: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Load a Stable Diffusion model from various sources"""
        model_id = f"{source_type}:{source}"
        hybrid_preflight = None
        if hybrid is not None:
            # HEADER-ONLY, and before anything is torn down: a refused hybrid
            # pair leaves the live model exactly as it was. It also has to
            # precede the same-model early return, because the identity below
            # needs its digest.
            from core.models.minimax_h3.hybrid_spec import (
                hybrid_model_identity, preflight_hybrid_request,
            )

            hybrid_preflight = preflight_hybrid_request(source, hybrid)

            # The base alone is NOT the identity of a hybrid: the same base with
            # another overlay, or the same pair over another block range, is a
            # different model and must not be answered by the early return.
            model_id = f"{model_id}#{hybrid_model_identity(hybrid_preflight.spec)}"

        # A different H3 text encoder or projection is a different set of loaded
        # components under the SAME model id, so neither the same-model early
        # return nor the DiT-only fast path below may fire for it. The fast path
        # in particular recomputes both layouts without the override, so the
        # encoder would compare equal and the request would vanish silently.
        h3_te_selection_changed = self._minimax_h3_te_selection_differs(
            text_encoder_file, clip_projection_file)

        # Degraded means a component switch or a failed load left a hole in the
        # live components. Re-selecting the same checkpoint is then a repair
        # request, not a no-op, and it is the obvious thing a user reaches for:
        # returning early would leave generation disabled with no way out of the
        # UI short of loading some other model first.
        if (self.current_model == model_id and not force_reload
                and not h3_te_selection_changed
                and self.component_health != "degraded"):
            return

        # MiniMax-H3's selectable files differ only in the DiT. Rebuilding its
        # shared 48 GiB memory-mapped text encoder on every partition/format
        # switch can terminate a Windows process inside torch_cpu.dll before
        # Python can report an allocation error. Build the new DiT first and
        # retain the current model if that build fails; only then swap it in.
        # Not while degraded: this path carries the existing text encoder over
        # untouched, and a failed TE switch is exactly what leaves that slot
        # empty. Repairing through it would report success over the same hole.
        if (self.is_minimax_h3_model and source_type in ("safetensors", "diffusers")
                and not h3_te_selection_changed
                and self.component_health != "degraded"):
            current_source = (self.current_model_info or {}).get("source")
            # The DiT is the only thing a hybrid changes, so this path serves a
            # hybrid request too -- and keeps the 48 GiB encoder mapped while
            # doing it. Base-only keeps the exact five-argument call it had.
            if current_source and self._reload_minimax_h3_dit_only(
                    source_type, source, current_source, pipeline_type, model_id,
                    **({} if hybrid_preflight is None else {"hybrid": hybrid_preflight})):
                return

        self._minimax_h3_te_request = (text_encoder_file, clip_projection_file)

        # A model (re)load invalidates any keep-models-hot resident set from the
        # previous model — the components about to be freed/replaced are exactly
        # what the resident set refers to.
        from core.keep_hot import clear_resident
        clear_resident(self)

        # A model (re)load is also the ONLY invalidation point for the in-place
        # runtime INT8 conversion (vram_optimization.apply_runtime_int8_quantization):
        # it drops the source bf16 weights, so the freshly loaded transformer is
        # the only way back to full precision. Reached for the SAME checkpoint
        # only via force_reload -- which is why POST /models/load takes it.
        self._runtime_int8_converted = False
        # The checkpoint-provenance latch is per LOADED MODEL too: the next
        # checkpoint may be an unquantized one, and a stale True would key
        # keep-hot as "quantized" for a bf16 transformer.
        self._runtime_int8_from_checkpoint = False
        self._runtime_int8_partial = False
        self._runtime_int8_partial_rows = []
        self._runtime_int8_partial_done = 0
        self._runtime_int8_audit = None

        # Clear any TE/VAE override state: the new model replaces the components
        # the override swapped, so the previous override refs are now stale.
        self._override_vae_path = None
        self._original_vae = None
        self._override_vae_targets = []
        self._override_te_path = None
        self._original_te = []

        # Everything below destroys the previously loaded model before the new
        # one is built.  Do not keep advertising that old model if cleanup or
        # the replacement load fails partway through.
        self.current_model = None
        self.current_model_info = None

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

            # Clean up LTX-2.3 components
            if self.ltx2_components is not None:
                print("[Pipeline] Cleaning up LTX-2.3 components...")
                for comp_name, comp in self.ltx2_components.items():
                    if comp is not None and hasattr(comp, 'to'):
                        try:
                            comp.to('cpu')
                        except Exception:
                            pass
                    del comp
                self.ltx2_components = None
                self.is_ltx2_model = False
                # Reset offload guard so a later LTX-2.3 load re-attaches the
                # cpu-offload hooks on the fresh pipeline.
                self._ltx2_offload_enabled = False

            # Clean up ACE-Step 1.5 components
            if self.acestep_components is not None:
                print("[Pipeline] Cleaning up ACE-Step 1.5 components...")
                for comp_name, comp in self.acestep_components.items():
                    if comp is not None and hasattr(comp, 'to'):
                        try:
                            comp.to('cpu')
                        except Exception:
                            pass
                    del comp
                self.acestep_components = None
                self.is_acestep_model = False

            # Clean up MiniMax-H3 components
            if self.minimax_h3_components is not None:
                print("[Pipeline] Cleaning up MiniMax-H3 components...")
                for comp_name, comp in self.minimax_h3_components.items():
                    # DELIBERATELY no `comp.to('cpu')` here, unlike every branch
                    # above. Every H3 component is already CPU-resident (the
                    # loader never stages to GPU in Phase 1), and the text
                    # encoder's parameters are memory-mapped from a 48 GiB file:
                    # a `.to()` on it would copy 48 GiB into anonymous memory
                    # moments before the reference is dropped. Phase 2, which
                    # does stage components to the GPU, must move them back
                    # WITHOUT a dtype argument and must exempt the text encoder.
                    del comp
                self.minimax_h3_components = None
                self.is_minimax_h3_model = False

            # Clean up MiniMax Music 3 components
            if self.minimax_music3_components is not None:
                print("[Pipeline] Cleaning up MiniMax Music 3 components...")
                for comp_name, comp in self.minimax_music3_components.items():
                    # No `comp.to('cpu')`: every component is already CPU-resident
                    # (the loader never stages to GPU), so it is a no-op, not a
                    # safety requirement -- unlike MiniMax-H3's text encoder, this
                    # loader's components are not memory-mapped.
                    del comp
                self.minimax_music3_components = None
                self.is_minimax_music3_model = False

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
                text_encoder_file=text_encoder_file,
                clip_projection_file=clip_projection_file,
                # Base-only keeps the exact call it always had, as at every
                # other site this commit touches.
                **({} if hybrid_preflight is None else {"hybrid": hybrid_preflight}),
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
                self._save_last_model(
                    source_type, source, pipeline_type,
                    text_encoder_path=kwargs.get("text_encoder_path"),
                    vae_path=kwargs.get("vae_path"))
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

            # Check if LTX-2.3 (joint audio+video MM-DiT). Before the generic Z-Image
            # check (which matches any dict carrying a "transformer" key). P1a:
            # loadable/slot-switchable only; video generation is P1b.
            if isinstance(model_result, dict) and model_result.get("type") == "ltx2":
                print("[Pipeline] LTX-2.3 video model detected (component-based dict returned)")
                self.ltx2_components = model_result
                self.is_ltx2_model = True
                self.is_krea2_model = False
                self.is_minit2i_model = False
                self.is_ideogram4_model = False
                self.is_lens_model = False
                self.is_anima_model = False
                self.is_zimage_model = False
                self.is_flux2_model = False
                self.current_model = model_id
                self.current_attention_type = "normal"

                # Keep components on CPU (VRAM discipline; GPU staging is P1b).
                for comp_name in ("text_encoder", "connectors", "transformer",
                                  "vae", "audio_vae", "vocoder"):
                    comp = self.ltx2_components.get(comp_name)
                    if comp is not None and hasattr(comp, "to"):
                        try:
                            comp.to("cpu")
                        except Exception:
                            pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[VRAM] LTX-2.3 components on CPU. GPU staging happens at generate time (P1b).")

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
                    "type": "ltx2",
                    "is_v_prediction": False,  # flow matching, velocity prediction
                    "model_hash": model_hash,
                    "is_video": True,
                    "latent_channels": self.ltx2_components.get("latent_channels", 128),
                    "vae_scale_factor_spatial": self.ltx2_components.get("vae_scale_factor_spatial", 32),
                    "vae_scale_factor_temporal": self.ltx2_components.get("vae_scale_factor_temporal", 8),
                }
                self._save_last_model(source_type, source, pipeline_type)
                print("[Pipeline] LTX-2.3 model loaded successfully")
                return

            # Check if ACE-Step 1.5 (2B DiT + Oobleck VAE + Qwen3-Embedding-0.6B
            # text encoder). Before the generic Z-Image check (which matches any
            # dict carrying a "transformer" key — ACE-Step's DiT key is "dit", so
            # it would not collide, but the explicit type-tag check runs first
            # for clarity, matching the LTX-2.3 pattern above). Phase 0+1:
            # loadable/slot-switchable only; no sampler/generation entry point yet.
            if isinstance(model_result, dict) and model_result.get("type") == "acestep":
                print("[Pipeline] ACE-Step 1.5 audio model detected (component-based dict returned)")
                self.acestep_components = model_result
                self.is_acestep_model = True
                self.is_ltx2_model = False
                self.is_krea2_model = False
                self.is_minit2i_model = False
                self.is_ideogram4_model = False
                self.is_lens_model = False
                self.is_anima_model = False
                self.is_zimage_model = False
                self.is_flux2_model = False
                self.current_model = model_id
                self.current_attention_type = "normal"

                # Keep components on CPU (VRAM discipline; GPU staging is Phase 2).
                for comp_name in ("dit", "vae", "text_encoder"):
                    comp = self.acestep_components.get(comp_name)
                    if comp is not None and hasattr(comp, "to"):
                        try:
                            comp.to("cpu")
                        except Exception:
                            pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[VRAM] ACE-Step 1.5 components on CPU. GPU staging happens at generate time (Phase 2).")

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
                    "type": "acestep",
                    "is_v_prediction": False,  # flow matching, velocity prediction
                    "model_hash": model_hash,
                    "is_audio": True,
                    "sample_rate": self.acestep_components.get("sample_rate", 48000),
                    "latent_frame_rate": self.acestep_components.get("latent_frame_rate", 25),
                    "latent_channels": self.acestep_components.get("latent_channels", 64),
                }
                self._save_last_model(source_type, source, pipeline_type)
                print("[Pipeline] ACE-Step 1.5 model loaded successfully")
                return

            # Check if MiniMax-H3 (pruned joint video+audio DiT + Qwen3-VL-32B +
            # a video and an audio VAE). MUST be before the generic Z-Image check
            # below, which matches any dict carrying a "transformer" key — H3's
            # does. Phase 1: loadable/slot-switchable only; generation is Phase 2.
            if isinstance(model_result, dict) and model_result.get("type") == "minimax_h3":
                print("[Pipeline] MiniMax-H3 video model detected (component-based dict returned)")
                self.minimax_h3_components = model_result
                self.is_minimax_h3_model = True
                self.is_acestep_model = False
                self.is_ltx2_model = False
                self.is_krea2_model = False
                self.is_minit2i_model = False
                self.is_ideogram4_model = False
                self.is_lens_model = False
                self.is_anima_model = False
                self.is_zimage_model = False
                self.is_flux2_model = False
                self.current_model = model_id
                self.current_attention_type = "normal"

                # The loader already leaves every component on the CPU, and the
                # text encoder is EXCLUDED from any `.to()` on purpose: its 48 GiB
                # of parameters are memory-mapped from the file, and moving the
                # module (even "to cpu", even more so with a dtype) detaches them
                # from that mapping — MEASURED at 73.08 GB peak RSS against
                # 49.82 GB for the mapping-preserving path.
                for comp_name in ("transformer", "vae", "audio_vae"):
                    comp = self.minimax_h3_components.get(comp_name)
                    if comp is not None and hasattr(comp, "to"):
                        try:
                            comp.to("cpu")
                        except Exception:
                            pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[VRAM] MiniMax-H3 components on CPU. GPU staging happens at generate "
                      "time (Phase 2); no two of them fit in 48 GB together.")

                model_hash = ""
                if source_type in ["safetensors", "diffusers"] and os.path.exists(source):
                    try:
                        from utils.hash_cache import get_cached_file_hash
                        model_hash = get_cached_file_hash(source)
                    except Exception as e:
                        print(f"[Pipeline] Hash compute skipped: {e}")

                from core.models.minimax_h3.hybrid_spec import hybrid_model_info_fields
                from core.models.minimax_h3.loader import minimax_h3_te_model_info_fields
                self.current_model_info = {
                    "source_type": source_type,
                    "source": source,
                    "type": "minimax_h3",
                    "is_v_prediction": False,  # flow matching, velocity prediction
                    "model_hash": model_hash,
                    "is_video": True,
                    # "hybrid" for a merged DiT -- the loader sets it explicitly,
                    # never from the base's filename (see hybrid_component_fields).
                    "variant": self.minimax_h3_components.get("variant"),
                    "latent_channels": self.minimax_h3_components.get("latent_channels", 24),
                    "vae_scale_factor_spatial": self.minimax_h3_components.get(
                        "vae_scale_factor_spatial", 16),
                    "vae_scale_factor_temporal": self.minimax_h3_components.get(
                        "vae_scale_factor_temporal", 4),
                    # Which text encoder produced the conditioning, and (when it
                    # is a converted small one) the projection it is only usable
                    # through -- a client cannot tell a substitution apart from
                    # the released encoder otherwise.
                    **minimax_h3_te_model_info_fields(self.minimax_h3_components),
                    # Sanitised base/overlay provenance; empty for a base-only load.
                    **hybrid_model_info_fields(self.minimax_h3_components),
                }
                # Base-only keeps the exact call it always had; a hybrid persists
                # the request that reproduces it.
                self._save_last_model(
                    source_type, source, pipeline_type, *self._minimax_h3_te_request,
                    **({} if hybrid_preflight is None
                       else {"hybrid": self.minimax_h3_components.get("hybrid_request")}))
                print("[Pipeline] MiniMax-H3 model loaded successfully")
                return

            # Check if MiniMax Music 3 (2.4B flow-matching DiT + 8B Qwen3
            # language model + 0.6B RVQ depth decoder + condition encoder +
            # vocoder). MUST be before the generic Z-Image check below, which
            # matches any dict carrying a "transformer" key -- Music 3's does.
            # Design doc phase 2: loadable/slot-switchable only; generation is
            # a later commit (pipeline_backends/minimax_music3.py).
            if isinstance(model_result, dict) and model_result.get("type") == "minimax_music3":
                print("[Pipeline] MiniMax Music 3 audio model detected (component-based dict returned)")
                self.minimax_music3_components = model_result
                self.is_minimax_music3_model = True
                self.is_minimax_h3_model = False
                self.is_acestep_model = False
                self.is_ltx2_model = False
                self.is_krea2_model = False
                self.is_minit2i_model = False
                self.is_ideogram4_model = False
                self.is_lens_model = False
                self.is_anima_model = False
                self.is_zimage_model = False
                self.is_flux2_model = False
                self.current_model = model_id
                self.current_attention_type = "normal"

                # The loader already leaves every component on the CPU, so this is a
                # no-op today; language_model is skipped only because it is large and
                # moving it here buys nothing that being already-CPU does not.
                for comp_name in ("transformer", "condition_encoder", "rvq_depth_decoder", "vocoder"):
                    comp = self.minimax_music3_components.get(comp_name)
                    if comp is not None and hasattr(comp, "to"):
                        try:
                            comp.to("cpu")
                        except Exception:
                            pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[VRAM] MiniMax Music 3 components on CPU. GPU staging happens at "
                      "generate time (a later commit); the language model and depth decoder "
                      "must be resident together for the autoregressive stage.")

                model_hash = ""
                if source_type in ["safetensors", "diffusers"] and os.path.exists(source):
                    try:
                        from utils.hash_cache import get_cached_file_hash
                        model_hash = get_cached_file_hash(source)
                    except Exception as e:
                        print(f"[Pipeline] Hash compute skipped: {e}")

                from core.models.minimax_music3.defaults import (
                    FALLBACK_FRAME_RATE, FALLBACK_NUM_CHANNELS_LATENTS, FALLBACK_SAMPLING_RATE,
                )

                self.current_model_info = {
                    "source_type": source_type,
                    "source": source,
                    "type": "minimax_music3",
                    "is_v_prediction": False,  # flow matching, velocity prediction (flow stage only)
                    "model_hash": model_hash,
                    "is_audio": True,
                    "sample_rate": self.minimax_music3_components.get("sample_rate", FALLBACK_SAMPLING_RATE),
                    "frame_rate": self.minimax_music3_components.get("frame_rate", FALLBACK_FRAME_RATE),
                    "latent_channels": self.minimax_music3_components.get(
                        "latent_channels", FALLBACK_NUM_CHANNELS_LATENTS),
                }
                self._save_last_model(source_type, source, pipeline_type)
                print("[Pipeline] MiniMax Music 3 model loaded successfully")
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

    def _minimax_h3_te_selection_differs(
        self,
        text_encoder_file: Optional[str],
        clip_projection_file: Optional[str],
    ) -> bool:
        """Whether a requested H3 encoder/projection is not the loaded one.

        Compared against the LOADED component paths, not against what the last
        request asked for: a request that names the file already in use must
        stay a no-op, and one that names any other file must reload.
        """
        if text_encoder_file is None and clip_projection_file is None:
            return False
        from core.models.minimax_h3.reload import same_path

        components = self.minimax_h3_components or {}
        loaded_te = components.get("text_encoder_path")
        loaded_projection = (components.get("te_projection") or {}).get("path")
        if text_encoder_file is not None and not same_path(text_encoder_file, loaded_te):
            return True
        return (clip_projection_file is not None
                and not same_path(clip_projection_file, loaded_projection))

    def _reload_minimax_h3_dit_only(
        self,
        source_type: str,
        source: str,
        current_source: str,
        pipeline_type: str,
        model_id: str,
        *,
        hybrid: Optional[Any] = None,
    ) -> bool:
        """Atomically replace only the DiT for two checkpoints in one H3 tree.

        ``hybrid`` is a validated preflight whose base is ``source``; the
        replacement is then the merged DiT. Atomicity is unchanged either way --
        the replacement is fully built before anything is swapped, so a failed
        merge leaves the live transformer, TE, VAEs and schedulers untouched.
        """
        from core.models.minimax_h3.hybrid_spec import hybrid_model_info_fields
        from core.models.minimax_h3.loader import minimax_h3_te_model_info_fields
        from core.models.minimax_h3.reload import build_dit_only_reload

        # Base-only keeps the exact three-argument call it always had.
        replacement = build_dit_only_reload(
            self.minimax_h3_components, current_source, source,
            **({} if hybrid is None else {"hybrid": hybrid}))
        if replacement is None:
            return False

        model_hash = ""
        if os.path.exists(source):
            try:
                from utils.hash_cache import get_cached_file_hash
                model_hash = get_cached_file_hash(source)
            except Exception as exc:
                print(f"[Pipeline] Hash compute skipped: {exc}")

        from core.keep_hot import clear_resident
        clear_resident(self)
        self._runtime_int8_converted = False
        self._runtime_int8_from_checkpoint = False
        self._runtime_int8_partial = False
        self._runtime_int8_partial_rows = []
        self._runtime_int8_partial_done = 0
        self._runtime_int8_audit = None
        self._override_vae_path = None
        self._original_vae = None
        self._override_vae_targets = []
        self._override_te_path = None
        self._original_te = []

        self.minimax_h3_components = replacement
        self.current_model = model_id
        self.current_attention_type = "normal"
        self.current_model_info = {
            "source_type": source_type,
            "source": source,
            "type": "minimax_h3",
            "is_v_prediction": False,
            "model_hash": model_hash,
            "is_video": True,
            "variant": replacement.get("variant"),
            "latent_channels": replacement.get("latent_channels", 24),
            "vae_scale_factor_spatial": replacement.get("vae_scale_factor_spatial", 16),
            "vae_scale_factor_temporal": replacement.get("vae_scale_factor_temporal", 4),
            # The DiT-only reload keeps the mapped text encoder, so this reports
            # the same encoder/projection the full load did.
            **minimax_h3_te_model_info_fields(replacement),
            **hybrid_model_info_fields(replacement),
        }
        # This path rebuilt only the DiT, so the encoder/projection request the
        # retained bundle came from still describes it. The hybrid recipe does
        # NOT carry over: it belongs to the DiT that was just replaced.
        self._save_last_model(
            source_type, source, pipeline_type, *self._minimax_h3_te_request,
            **({} if hybrid is None else {"hybrid": replacement.get("hybrid_request")}))

        # The replacement dict shares every ancillary object but not the old
        # transformer. Collect after the attribute swap so its large CPU storage
        # is released without ever dropping/remapping the text encoder.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Pipeline] MiniMax-H3 DiT reloaded; shared text encoder, VAEs, "
              "tokenizer/processor and schedulers retained")
        return True

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

    def _resize_latent(self, latent: torch.Tensor, latent_height: int, latent_width: int,
                        resampling_method: str = "lanczos") -> torch.Tensor:
        """Resize a latent tensor [B,C,H,W] to a target LATENT-space size.

        Shared by img2img's "latent resize mode" (quality-preserving upscale via
        a VAE round-trip: encode -> resize latent -> decode) and the
        input_latent_id latent-passthrough path (loop-generation latent
        upscale between steps, with NO VAE round-trip at all -- the cached
        latent is resized directly and fed straight into the denoise loop).

        Args:
            latent: [B,C,H,W] latent tensor (any device/dtype).
            latent_height, latent_width: Target size in LATENT space (i.e.
                already pixel_size // 8, the VAE's downsample factor) -- NOT
                pixel dimensions.
            resampling_method: "lanczos" (scipy, CPU round-trip) | "nearest" |
                "bilinear" | "bicubic" (torch.nn.functional.interpolate).

        Returns:
            Resized latent tensor, same device/dtype as the input.
        """
        if resampling_method == "lanczos":
            # scipy for Lanczos (not available in PyTorch); scipy doesn't
            # support float16, so round-trip through float32.
            from scipy.ndimage import zoom
            import numpy as np

            original_dtype = latent.dtype
            latent_np = latent.cpu().float().numpy()
            batch, channels, h, w = latent_np.shape
            zoom_h = latent_height / h
            zoom_w = latent_width / w

            resized_list = []
            for b in range(batch):
                resized_channels = []
                for c in range(channels):
                    resized_channel = zoom(latent_np[b, c], (zoom_h, zoom_w), order=3, mode='reflect')
                    resized_channels.append(resized_channel)
                resized_list.append(np.stack(resized_channels))

            resized_np = np.stack(resized_list)
            return torch.from_numpy(resized_np).to(device=latent.device, dtype=original_dtype)

        import torch.nn.functional as F
        torch_mode_map = {
            "nearest": "nearest",
            "bilinear": "bilinear",
            "bicubic": "bicubic",
        }
        torch_mode = torch_mode_map.get(resampling_method, "bicubic")
        return F.interpolate(
            latent,
            size=(latent_height, latent_width),
            mode=torch_mode,
            align_corners=False if torch_mode != "nearest" else None,
        )

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

    # ------------------------------------------------------------------
    # Per-generation VAE / TE overrides (RP2b)
    # ------------------------------------------------------------------
    def _vae_override_targets(self) -> List[tuple]:
        """Return the (kind, container, key) slots currently holding the active
        VAE. ``kind`` is "attr" (setattr) for the diffusers image pipelines or
        "item" (setitem) for a component-dict arch (zimage/flux2/lens/...)."""
        targets: List[tuple] = []
        for pipe in (self.txt2img_pipeline, self.img2img_pipeline, self.inpaint_pipeline):
            if pipe is not None and getattr(pipe, "vae", None) is not None:
                targets.append(("attr", pipe, "vae"))
        for comps in (self.zimage_components, self.flux2_components,
                      self.anima_components, self.lens_components,
                      self.ideogram4_components, self.krea2_components):
            if isinstance(comps, dict) and comps.get("vae") is not None:
                targets.append(("item", comps, "vae"))
        return targets

    @staticmethod
    def _set_slot(kind: str, container: Any, key: str, value: Any) -> None:
        if kind == "attr":
            setattr(container, key, value)
        else:
            container[key] = value

    def override_vae_identity(self) -> tuple:
        """Return (source, path) describing the active override VAE, for metadata."""
        if not self._override_vae_path:
            return ("none", None)
        try:
            from core.models.common.vae_store import vae_identity
            from core.models.pid.pid_vae_wrapper import PidVaeWrapper
            active = None
            for kind, container, key in self._vae_override_targets():
                active = getattr(container, key) if kind == "attr" else container.get(key)
                break
            if isinstance(active, PidVaeWrapper):
                return ("PiD SDXL decoder (pixel-diffusion super-resolution)", self._override_vae_path)
            src, path = vae_identity(active)
            return (src, path or self._override_vae_path)
        except Exception:
            return (self._override_vae_path, self._override_vae_path)

    def set_pid_prompt(self, prompt: Optional[str]) -> None:
        """Forward the current generation's raw text prompt to an active
        ``PidVaeWrapper`` (consulted only when ``pid_use_gemma=True``); a no-op
        when no PiD override is active. Call once per generation, before the
        sampling loop, so the wrapper has it ready for ``pid_final_decode``."""
        try:
            from core.models.pid.pid_vae_wrapper import PidVaeWrapper
        except Exception:
            return
        for kind, container, key in self._vae_override_targets():
            active = getattr(container, key) if kind == "attr" else container.get(key)
            if isinstance(active, PidVaeWrapper):
                active.set_prompt(prompt)
                return

    @staticmethod
    def _upcast_vae_if_needed(new_vae, vae_path: str):
        """Replicate diffusers' native VAE fp16-overflow guard for override VAEs.

        The stock SDXL pipelines (e.g. ``StableDiffusionXLPipeline.decode_latents``)
        upcast the VAE to float32 before decode whenever
        ``vae.dtype == torch.float16 and vae.config.force_upcast`` — the original
        (non "fp16-fix") SDXL VAE overflows to NaN inside its decoder attention
        blocks at fp16. ``AutoencoderKL.decode()`` itself does NOT perform this
        upcast; it is pipeline-level logic. This VAE-override path bypasses that
        pipeline entirely (the VRAM funnel in ``vram_optimization.py`` only moves
        device, never dtype), so a bare original-format SDXL VAE loaded here
        (``force_upcast`` defaults to ``True`` when there is no accompanying
        ``config.json`` to override it) would silently decode to an all-gray/NaN
        image. Force float32 in that case; leave fp16-fix-style VAEs
        (``force_upcast=False``) and bf16-loaded VAEs alone.
        """
        if new_vae.dtype == torch.float16 and getattr(new_vae.config, "force_upcast", False):
            print(f"[VAEOverride] {vae_path}: force_upcast=True + fp16 would overflow to NaN on "
                  f"decode (matches diffusers' native VAE fp16 guard) - loading in float32 instead.")
            new_vae = new_vae.to(dtype=torch.float32)
        return new_vae

    def load_override_vae(
        self,
        vae_path: Optional[str],
        override_kind: Optional[str] = None,
        pid_sr_output: str = "4x",
        pid_use_gemma: bool = False,
        pid_low_vram: bool = False,
        pid_tile_native: int = 512,
        pid_tile_overlap_ratio: float = 0.25,
        pid_fast_large_decode: bool = False,
    ):
        """Swap the model's VAE for the one at ``vae_path`` (idempotent).

        ``vae_path`` None/empty RESTORES the original VAE (kept in
        ``self._original_vae``) without a reload. The new VAE is loaded to CPU;
        the existing move_vae_to_gpu/cpu funnel stages it per generation.

        ``override_kind`` (from ``generation_overrides.classify_vae_candidate``'s
        ``"kind"`` field) selects the construction path: ``"pid_decoder"`` builds
        a ``PidVaeWrapper`` around the currently-loaded model's OWN SDXL VAE
        (reused by reference — no extra download/VRAM) plus the PiD ``.pth`` at
        ``vae_path``; anything else (including ``None``, for backward
        compatibility) is a normal ``AutoencoderKL``-family swap.

        ``pid_tile_native``/``pid_tile_overlap_ratio``/``pid_fast_large_decode``
        only matter for a PiD override — see ``PidVaeWrapper``'s F9 docstring
        (tiled decode is the default large-output path; ``pid_fast_large_decode``
        opts back into the original whole-latent cap+bicubic path).
        """
        from core.models.pid.pid_vae_wrapper import PidVaeWrapper

        if not vae_path:
            self._restore_override_vae()
            return
        if self._override_vae_path == vae_path:
            # Idempotent on the path, but pid_sr_output/pid_use_gemma/pid_low_vram/
            # pid_tile_native/pid_tile_overlap_ratio/pid_fast_large_decode may have
            # changed between generations on the SAME PiD checkpoint — update the
            # live wrapper's flags in place rather than silently ignoring them.
            if override_kind == "pid_decoder":
                for _slot_kind, container, key in self._vae_override_targets():
                    active = getattr(container, key) if _slot_kind == "attr" else container.get(key)
                    if isinstance(active, PidVaeWrapper):
                        active.pid_sr_output = pid_sr_output
                        active.pid_use_gemma = pid_use_gemma
                        active.low_vram_decode = pid_low_vram
                        active.tile_native = pid_tile_native
                        active.tile_overlap_ratio = pid_tile_overlap_ratio
                        active.fast_large_decode = pid_fast_large_decode
                    break
            return  # idempotent — already applied

        targets = self._vae_override_targets()
        if not targets:
            # Silent no-op until now: `apply_overrides` saw no exception and
            # still recorded `vae_override_path`, so the row CLAIMED an
            # override that never reached the decoder. Reported with the same
            # code as a load failure (both mean "the model's own VAE decoded
            # this image"), which is also what `GeneratedImage.to_dict` keys
            # its override-label derivation on.
            print("[VAEOverride] No VAE slot on the loaded model; override skipped.")
            try:
                from api.generation_status import add_warning
                add_warning(
                    "VAE override could not be applied: the loaded model exposes no VAE "
                    "slot to swap; the model's own VAE decoded this image",
                    code="vae_override_error",
                )
            except Exception:
                pass
            return

        # Snapshot the originals on the FIRST override so a later clear restores.
        if self._override_vae_path is None:
            first_kind, first_c, first_k = targets[0]
            self._original_vae = (getattr(first_c, first_k) if first_kind == "attr"
                                  else first_c.get(first_k))
            self._override_vae_targets = targets

        if override_kind == "pid_decoder":
            print(f"[VAEOverride] Building PidVaeWrapper around the loaded model's own SDXL VAE "
                  f"+ PiD checkpoint {vae_path} (pid_sr_output={pid_sr_output}, pid_use_gemma={pid_use_gemma}, "
                  f"pid_low_vram={pid_low_vram}, pid_tile_native={pid_tile_native}, "
                  f"pid_tile_overlap_ratio={pid_tile_overlap_ratio}, pid_fast_large_decode={pid_fast_large_decode})")
            new_vae = PidVaeWrapper(
                self._original_vae,
                pid_pth_path=vae_path,
                pid_sr_output=pid_sr_output,
                pid_use_gemma=pid_use_gemma,
                low_vram_decode=pid_low_vram,
                tile_native=pid_tile_native,
                tile_overlap_ratio=pid_tile_overlap_ratio,
                fast_large_decode=pid_fast_large_decode,
            )
        else:
            # Resolve the candidate VAE directory + class.
            from api.generation_overrides import _vae_config_dir, _read_json
            cfg_dir = _vae_config_dir(vae_path)

            # Match the original VAE's dtype so downstream device/dtype staging is a no-op.
            dtype = None
            try:
                dtype = next(self._original_vae.parameters()).dtype
            except Exception:
                dtype = torch.float16

            if cfg_dir is None and isinstance(vae_path, str) and vae_path.endswith(".safetensors") and os.path.isfile(vae_path):
                # Bare original/LDM-format AutoencoderKL VAE (no config.json next
                # to it — e.g. a plain `sdxl_vae.safetensors`). There is no
                # diffusers directory to `from_pretrained()`; load the state dict
                # directly, which infers the AutoencoderKL architecture from the
                # header shapes alone.
                from diffusers import AutoencoderKL
                print(f"[VAEOverride] Loading bare AutoencoderKL from single file {vae_path} (dtype={dtype})")
                new_vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
                # `from_single_file` cannot tell SDXL from SD1.5 apart for a bare
                # AutoencoderKL — both share the identical architecture — and
                # silently defaults to the SD1.5 scaling_factor (0.18215). The
                # override VAE decodes the SAME latent space as the model's OWN
                # (original) VAE, so copy scaling_factor/shift_factor from it
                # rather than trusting from_single_file's guess.
                try:
                    orig_cfg = self._original_vae.config
                    new_vae.register_to_config(
                        scaling_factor=getattr(orig_cfg, "scaling_factor", new_vae.config.scaling_factor),
                        shift_factor=getattr(orig_cfg, "shift_factor", None) or 0.0,
                    )
                except Exception as e:
                    print(f"[VAEOverride] scaling_factor copy from original VAE failed (non-fatal): {e}")
                new_vae = self._upcast_vae_if_needed(new_vae, vae_path)
                new_vae = new_vae.to("cpu")
            else:
                if cfg_dir is None:
                    raise ValueError(f"No loadable VAE config.json found under: {vae_path}")
                cfg = _read_json(os.path.join(cfg_dir, "config.json")) or {}
                class_name = cfg.get("_class_name") or "AutoencoderKL"
                import diffusers
                vae_cls = getattr(diffusers, class_name, None)
                if vae_cls is None:
                    from diffusers import AutoencoderKL as vae_cls  # fallback

                print(f"[VAEOverride] Loading {class_name} from {cfg_dir} (dtype={dtype})")
                new_vae = vae_cls.from_pretrained(cfg_dir, torch_dtype=dtype)
                new_vae = self._upcast_vae_if_needed(new_vae, vae_path)
                new_vae = new_vae.to("cpu")

        # If an outgoing PiD wrapper occupies the slot (switching override A->B
        # WITHOUT an intervening restore), release its cached PiD net before
        # replacing it, rather than leaving a CPU-side orphan for the GC.
        for _k, _c, _key in targets:
            _cur = getattr(_c, _key) if _k == "attr" else _c.get(_key)
            if isinstance(_cur, PidVaeWrapper) and _cur is not new_vae:
                try:
                    _cur.unload()
                except Exception as _e:
                    print(f"[VAEOverride] outgoing PiD wrapper unload failed (non-fatal): {_e}")
            break

        for kind, container, key in targets:
            self._set_slot(kind, container, key, new_vae)
        self._override_vae_path = vae_path

    def _restore_override_vae(self):
        if self._override_vae_path is None:
            return
        print("[VAEOverride] Restoring original VAE.")
        for kind, container, key in self._override_vae_targets:
            try:
                active = getattr(container, key) if kind == "attr" else container.get(key)
                try:
                    from core.models.pid.pid_vae_wrapper import PidVaeWrapper
                    if isinstance(active, PidVaeWrapper):
                        active.unload()  # drop the cached PiD net (CPU+GPU) immediately
                except Exception:
                    pass
                self._set_slot(kind, container, key, self._original_vae)
            except Exception as e:
                print(f"[VAEOverride] restore slot failed: {e}")
        self._override_vae_path = None
        self._original_vae = None
        self._override_vae_targets = []
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_override_te(self, te_path: Optional[str]):
        """Swap the text encoder for SD1.5/SDXL (idempotent).

        Two sound cases (FABLE decision): (1) a custom-TE checkpoint
        (``pipeline._sushi_te`` present) reloads its encoder body from ``te_path``
        via ``te_registry.load_te`` while the trained bridge adapters stay; (2) a
        stock CLIP encoder is substituted. ``te_path`` None RESTORES the original.
        """
        if not te_path:
            self._restore_override_te()
            return
        if self._override_te_path == te_path:
            return

        primary = self.txt2img_pipeline
        if primary is None:
            print("[TEOverride] No SD/SDXL pipeline loaded; override skipped.")
            return

        # SD/SDXL share text-encoder module objects across txt2img/img2img/inpaint,
        # but each pipeline holds its OWN attribute reference — rebinding one does
        # not rebind the others. Apply (and snapshot) on every present pipeline so
        # img2img/inpaint use the override too, not the original.
        pipes = [p for p in (self.txt2img_pipeline, self.img2img_pipeline,
                             self.inpaint_pipeline) if p is not None]

        if getattr(primary, "_sushi_te", None) is not None:
            # Custom-TE checkpoint: reload the encoder body, keep the adapters.
            arch_info = getattr(primary, "_sushi_arch", {}) or {}
            te_type = arch_info.get("te_type")
            if not te_type:
                raise ValueError("Custom-TE checkpoint has no recorded te_type")
            from core.models.components.te_registry import load_te
            dtype = torch.float16
            try:
                dtype = next(primary._sushi_te.parameters()).dtype
            except Exception:
                pass
            max_len = getattr(primary, "_sushi_te_max_len", 256)
            encoder, tokenizer, _dim = load_te(te_type, repo=te_path, dtype=dtype,
                                               device="cpu", max_len=max_len)
            encoder.eval()
            snapshot = self._override_te_path is None
            for pipe in pipes:
                if getattr(pipe, "_sushi_te", None) is None:
                    continue
                if snapshot:
                    self._original_te.append({
                        "mode": "sushi", "pipe": pipe,
                        "_sushi_te": pipe._sushi_te,
                        "_sushi_te_tokenizer": pipe._sushi_te_tokenizer,
                    })
                pipe._sushi_te = encoder
                pipe._sushi_te_tokenizer = tokenizer
        else:
            # Stock CLIP substitution.
            from api.generation_overrides import _te_config_dir
            cfg_dir = _te_config_dir(te_path)
            if cfg_dir is None:
                raise ValueError(f"No loadable text_encoder config.json found under: {te_path}")
            from transformers import CLIPTextModel
            dtype = torch.float16
            try:
                dtype = next(primary.text_encoder.parameters()).dtype
            except Exception:
                pass
            new_te = CLIPTextModel.from_pretrained(cfg_dir, torch_dtype=dtype).to("cpu")
            snapshot = self._override_te_path is None
            for pipe in pipes:
                if getattr(pipe, "text_encoder", None) is None:
                    continue
                if snapshot:
                    self._original_te.append({
                        "mode": "clip", "pipe": pipe,
                        "text_encoder": pipe.text_encoder,
                    })
                pipe.text_encoder = new_te

        self._override_te_path = te_path
        print(f"[TEOverride] Applied text-encoder override on {len(pipes)} pipeline(s): {te_path}")

    def _restore_override_te(self):
        if self._override_te_path is None or not self._original_te:
            self._override_te_path = None
            self._original_te = []
            return
        print("[TEOverride] Restoring original text encoder.")
        for orig in self._original_te:
            pipe = orig.get("pipe")
            try:
                if orig.get("mode") == "sushi" and pipe is not None:
                    pipe._sushi_te = orig["_sushi_te"]
                    pipe._sushi_te_tokenizer = orig["_sushi_te_tokenizer"]
                elif orig.get("mode") == "clip" and pipe is not None:
                    pipe.text_encoder = orig["text_encoder"]
            except Exception as e:
                print(f"[TEOverride] restore failed: {e}")
        self._override_te_path = None
        self._original_te = []
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        size, so we set both to the threshold: below it the decode runs whole
        (bit-identical, no quality/speed cost), above it the image is split into
        threshold-sized tiles -- i.e. tiles as big as the size you're comfortable
        decoding un-tiled.

        Two diffusers autoencoder APIs express this differently and BOTH must be
        handled -- only wiring the first silently ignored the threshold on the
        Qwen-family VAEs (Anima / Krea2), which then tiled at the diffusers
        default 256px regardless of the user's setting:
          * square API (AutoencoderKL, AutoencoderKLFlux2 -- SD1.5/SDXL/Z-Image/
            Lens/Ideogram4/FLUX.2): tile_sample_min_size + tile_latent_min_size,
            overlap via tile_overlap_factor. 0 -> auto = VAE sample_size * 1.5.
          * height/width API (AutoencoderKLQwenImage -- Anima/Krea2):
            tile_sample_min_{height,width} + tile_sample_stride_{height,width},
            where the stride is the tile advance and (min - stride) is the blend
            band. 0 -> auto = leave the class defaults untouched.

        MODE (self._vae_tile_mode):
          * "blend"   -- diffusers' own overlap + linear cross-fade, as above.
          * "context" -- core/inference/context_tiled_decode.py: each tile is
            decoded with a real neighbouring-latent margin that is discarded
            afterwards, so tiles abut exactly with no blending. Installed as an
            instance-level `decode` override on the object whose decode actually
            runs the decoder (the inner AutoencoderKL behind SDXLVAEWrapper /
            FluxVAEWrapper, never PidVaeWrapper -- PiD is a 4-step pixel
            diffusion decoder with its own tiling, not an autoencoder decode).
            diffusers' own use_tiling is turned OFF in this mode so the latent
            is not tiled twice.

        GLOBAL GROUPNORM (self._vae_tile_global_norm, opt-in, default off):
          core/inference/global_group_norm.py wraps the WHOLE decode (whichever
          mode is active) in two passes -- one recording every decoder
          GroupNorm's per-group statistics across the tiles, one re-decoding with
          the accumulated whole-image statistics forced. Removes the per-tile
          tint that neither join mode addresses (measured 1.32 -> 0.037 /255
          peak-to-peak on SDXL, blend, 512px budget) for one extra decoder pass
          and +0.02-0.03 GB peak. Skipped entirely when the decoder has no
          GroupNorm (Qwen-family: Anima / Krea2) or when the latent fits the
          budget un-tiled.
        """
        if vae is None:
            return
        from core.inference.context_tiled_decode import (
            DEFAULT_MARGIN_CELLS,
            install_context_tiled_decode,
            supports_context_tiling,
            uninstall_context_tiled_decode,
        )
        from core.inference.global_group_norm import (
            install_global_group_norm_decode,
            supports_global_group_norm,
            uninstall_global_group_norm_decode,
        )
        threshold_px = int(getattr(self, "_vae_tile_threshold", 0) or 0)
        mode = str(getattr(self, "_vae_tile_mode", "blend") or "blend").lower()
        global_norm = bool(getattr(self, "_vae_tile_global_norm", False))
        if mode not in ("blend", "context"):
            # Unknown value falls back to the DEFAULT mode, not to "context" --
            # otherwise a typo would silently opt a user into the non-default
            # decode path.
            mode = "blend"
        targets = [vae]
        # Wrapper objects hold their real autoencoder under different names:
        # SDXLVAEWrapper / FluxVAEWrapper use `.vae`, PidVaeWrapper uses
        # `.real_vae`. BOTH must be walked, and specifically for uninstall:
        # PidVaeWrapper.decode delegates straight to `real_vae.decode`, and that
        # object is the same one a previous non-PiD generation may have patched.
        # If discovery cannot reach it, the stale override survives a switch to
        # PiD (or to tiling off) and silently keeps running with the previous
        # request's threshold. Unlike a plain VAE swap this does not self-heal.
        for attr in ("vae", "real_vae"):
            inner = getattr(vae, attr, None)
            if inner is not None and inner is not vae and inner not in targets:
                targets.append(inner)
        context_mode = enabled and mode == "context"
        for t in targets:
            # The two-pass global-GroupNorm override WRAPS whichever decode is
            # installed below it, so it must come off FIRST: otherwise the
            # context install/uninstall below would snapshot (or strip) this
            # wrapper as if it were the VAE's original bound decode, and the two
            # would stack. Re-installed last, after the mode is settled.
            uninstall_global_group_norm_decode(t)
            # diffusers' own tiling stays OFF in context mode (otherwise the
            # padded tiles we hand it would be tiled a second time).
            for on_name, off_name, want_on in (
                ("enable_tiling", "disable_tiling", enabled and not context_mode),
                ("enable_slicing", "disable_slicing", enabled),
            ):
                method = on_name if want_on else off_name
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
            elif enabled and hasattr(t, "tile_sample_min_height"):
                # Qwen-family API. Snapshot the as-loaded values once so a later
                # request with threshold=0 (auto) restores them instead of
                # inheriting the previous request's tile size -- these live on
                # the VAE object, and this method's contract is to honour the
                # per-request option every time.
                try:
                    if not hasattr(t, "_sushi_tile_defaults"):
                        t._sushi_tile_defaults = (
                            int(t.tile_sample_min_height), int(t.tile_sample_min_width),
                            int(t.tile_sample_stride_height), int(t.tile_sample_stride_width),
                        )
                    if threshold_px > 0:
                        scale = int(getattr(t, "spatial_compression_ratio", 8) or 8)
                        # Tile size and stride must be whole latent cells, else
                        # the // in tiled_decode truncates them apart.
                        # Floor at 2 cells: this API blends over (min - stride),
                        # so a 1-cell tile would leave no blend band at all.
                        thr = max(2 * scale, (int(threshold_px) // scale) * scale)
                        # Keep the class's 192/256 = 0.75 stride ratio so the
                        # blend band scales with the tile instead of vanishing.
                        stride = max(scale, (int(thr * 0.75) // scale) * scale)
                        stride = min(stride, thr - scale)
                        t.tile_sample_min_height = thr
                        t.tile_sample_min_width = thr
                        t.tile_sample_stride_height = stride
                        t.tile_sample_stride_width = stride
                    else:
                        (t.tile_sample_min_height, t.tile_sample_min_width,
                         t.tile_sample_stride_height, t.tile_sample_stride_width) = t._sushi_tile_defaults
                except Exception as e:
                    print(f"[VAE Tiling] threshold setup failed: {e}")

            # Resolve the decode budget from whatever the threshold setup above
            # landed on, so "auto" means the same thing in every mode. Shared by
            # the context install and the global-GroupNorm install (the latter
            # uses it only to decide whether this decode is actually tiled).
            resolved = 0
            try:
                if hasattr(t, "tile_sample_min_size"):
                    resolved = int(t.tile_sample_min_size or 0)
                elif hasattr(t, "tile_sample_min_height"):
                    resolved = int(t.tile_sample_min_height or 0)
            except Exception:
                resolved = 0
            if resolved <= 0:
                resolved = threshold_px if threshold_px > 0 else 1536

            # Context-margin mode: install/uninstall the decode override.
            # Reversible and idempotent -- uninstall runs on every non-context
            # call so switching modes or disabling tiling restores the original
            # bound method, and install always rebuilds from the snapshot so
            # wrappers never stack. Only lands on the object whose decode
            # actually runs the decoder (see supports_context_tiling).
            if context_mode and supports_context_tiling(t):
                try:
                    install_context_tiled_decode(t, resolved, DEFAULT_MARGIN_CELLS)
                except Exception as e:
                    print(f"[VAE Tiling] context-mode install failed: {e}")
                    uninstall_context_tiled_decode(t)
                    # Context mode turned diffusers' own tiling OFF above, so
                    # without this the fallback would run a WHOLE un-tiled decode
                    # -- on the code path a user enabled tiling to avoid an OOM.
                    # Mirrors context_tiled_decode._fallback_to_blend.
                    try:
                        if hasattr(t, "enable_tiling"):
                            t.enable_tiling()
                    except Exception as e2:
                        print(f"[VAE Tiling] could not re-enable blend tiling "
                              f"for the fallback: {e2}")
            else:
                uninstall_context_tiled_decode(t)

            # Two-pass global GroupNorm statistics (opt-in). Installed LAST so it
            # wraps the mode-specific decode chosen just above, and only when
            # tiling is on -- with tiling off there is nothing to correct.
            # supports_global_group_norm() is the mandatory GroupNorm gate: on a
            # decoder without any (Qwen family) the second pass is a bit-exact
            # no-op that would still cost a full extra decode, so it is skipped
            # silently rather than warned about.
            if enabled and global_norm and supports_global_group_norm(t):
                try:
                    install_global_group_norm_decode(t, resolved)
                except Exception as e:
                    print(f"[VAE Tiling] global-GroupNorm install failed: {e}")
                    uninstall_global_group_norm_decode(t)
        if enabled:
            _thr = threshold_px if threshold_px > 0 else "auto(per-VAE default)"
            if context_mode:
                print(f"[VAE Tiling] enabled, mode=context (decode-area budget "
                      f"{_thr}px; {DEFAULT_MARGIN_CELLS}-cell real-context margin "
                      f"decoded then discarded; no blending)")
            else:
                print(f"[VAE Tiling] enabled, mode=blend (tiles when image > {_thr}px; "
                      f"tile size = threshold)")
            if global_norm:
                _gn_on = [t for t in targets if supports_global_group_norm(t)]
                if _gn_on:
                    print("[VAE Tiling] global GroupNorm statistics: ON "
                          "(decode runs twice when the image is actually tiled)")
                else:
                    print("[VAE Tiling] global GroupNorm statistics: not applicable "
                          "(this decoder has no GroupNorm) - single pass")

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

    def _save_last_model(self, source_type: str, source: str, pipeline_type: str,
                         text_encoder_file: Optional[str] = None,
                         clip_projection_file: Optional[str] = None,
                         text_encoder_path: Optional[str] = None,
                         vae_path: Optional[str] = None,
                         hybrid: Optional[Dict[str, Any]] = None):
        """Save the last loaded model configuration to file.

        The optional fields are written only when they were requested, so a
        file written before they existed and one written by a default load look
        the same. ``text_encoder_path``/``vae_path`` are Anima's explicit
        companions: without them a live component switch would last only until
        the next restart, which reads this file back.

        ``hybrid`` is MiniMax-H3's overlay REQUEST (overlay file + recipe), not
        its digest: a restore re-derives the digest from the files as they are
        then, so a replaced overlay is refused instead of silently loaded.
        """
        try:
            config = {
                "source_type": source_type,
                "source": source,
                "pipeline_type": pipeline_type
            }
            if text_encoder_file is not None:
                config["text_encoder_file"] = text_encoder_file
            if clip_projection_file is not None:
                config["clip_projection_file"] = clip_projection_file
            if text_encoder_path is not None:
                config["text_encoder_path"] = text_encoder_path
            if vae_path is not None:
                config["vae_path"] = vae_path
            if hybrid is not None:
                config["hybrid"] = dict(hybrid)
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
            # Absent in a file written before these existed, and in one written
            # by a load that made no explicit choice.
            text_encoder_file = config.get("text_encoder_file")
            clip_projection_file = config.get("clip_projection_file")
            # Anima's explicit companions, recorded when they were chosen --
            # either at load time or by a live component switch.
            companions = {
                key: config[key]
                for key in ("text_encoder_path", "vae_path")
                if config.get(key)
            }
            # MiniMax-H3's overlay request. Absent for every base-only load;
            # re-validated (and re-digested) by the load it is passed to. NOT
            # `or None`: an empty/malformed entry in a hand-edited file must
            # reach normalize_hybrid_request and be refused by name, not degrade
            # into a base-only load that reports success.
            hybrid = config.get("hybrid")

            if source_type and source:
                print(f"Auto-loading last model: {source_type}:{source}")
                self.load_model(
                    source_type=source_type,
                    source=source,
                    pipeline_type=pipeline_type,
                    text_encoder_file=text_encoder_file,
                    clip_projection_file=clip_projection_file,
                    hybrid=hybrid,
                    **companions,
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

        # Encode negative prompt similarly. Note: this branch runs even when
        # negative_prompt is "" — an empty negative prompt must still be encoded
        # (BOS/EOS-only CLIP embedding) so SD1.5/SDXL CFG sampling never receives
        # None for the negative embeds (see custom_sampling.py CFG concat sites).
        neg_tokens = tokenizer(clean_neg_prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
        neg_chunks = []
        for i in range(0, len(neg_tokens), chunk_size):
            neg_chunk_tokens = neg_tokens[i:i + chunk_size]
            neg_chunks.append(neg_chunk_tokens)

        if self.max_prompt_chunks > 0 and len(neg_chunks) > self.max_prompt_chunks:
            neg_chunks = neg_chunks[:self.max_prompt_chunks]

        # Zero-token negative prompt (empty string): keep a single empty chunk so
        # pipeline.encode_prompt("") still runs and produces the uncond embedding.
        if not neg_chunks:
            neg_chunks = [neg_tokens[:0]]

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
                    # Single chunk (including the empty-string case): keep the
                    # full BOS+EOS embedding — slicing would zero-length it.
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
                # Zero-token chunk (empty negative prompt) is BOS+EOS only (len 2);
                # stripping both would zero-length it, so keep it as-is in that case.
                # Non-empty chunks keep the pre-existing unconditional strip.
                if chunk_emb.shape[1] > 2:
                    processed_chunks.append(chunk_emb[:, 1:-1, :])
                else:
                    processed_chunks.append(chunk_emb)
            negative_prompt_embeds = torch.cat(processed_chunks, dim=1)

        # Apply emphasis weights (skipped for NegPip, which weights V)
        if negative_prompt and has_neg_emphasis and not skip_emphasis:
            negative_prompt_embeds = apply_emphasis_to_embeds(
                negative_prompt, negative_prompt_embeds,
                tokenizer,
                device, dtype
            )

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

        # Encode negative prompt. Runs even when negative_prompt is "" — an empty
        # negative prompt must still be encoded (BOS/EOS-only CLIP embedding) so
        # SD1.5/SDXL CFG sampling never receives None for the negative embeds
        # (see custom_sampling.py CFG concat sites).
        neg_embeds = pipeline.encode_prompt(
            prompt=negative_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False
        )

        negative_prompt_embeds = neg_embeds[0]
        negative_pooled_prompt_embeds = neg_embeds[2] if is_sdxl else None

        # Strip BOS and EOS for NoBOS mode (only if there's more than BOS+EOS,
        # i.e. skip stripping the empty-prompt case to avoid a zero-length tensor)
        if negative_prompt_embeds.shape[1] > 2:
            negative_prompt_embeds = negative_prompt_embeds[:, 1:-1, :]

        # Apply emphasis weights if present (skipped for NegPip, which weights V)
        if negative_prompt and has_neg_emphasis and not skip_emphasis:
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

            # Encode negative prompt. Runs even when negative_prompt is "" — an
            # empty negative prompt must still be encoded (BOS/EOS-only CLIP
            # embedding) so SD1.5/SDXL CFG sampling never receives None for the
            # negative embeds (see custom_sampling.py CFG concat sites).
            neg_embeds = pipeline.encode_prompt(
                prompt=negative_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )

            negative_prompt_embeds = neg_embeds[0]
            negative_pooled_prompt_embeds = neg_embeds[2] if len(neg_embeds) > 2 and is_sdxl else None

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

        # Encode negative prompt. Runs even when negative_prompt is "" — an empty
        # negative prompt must still be encoded (BOS/EOS-only CLIP embedding) so
        # SD1.5/SDXL CFG sampling never receives None for the negative embeds
        # (see custom_sampling.py CFG concat sites).
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

        if negative_prompt and has_neg_emphasis and not skip_emphasis:
            negative_prompt_embeds = apply_emphasis_to_embeds(
                negative_prompt, negative_prompt_embeds,
                pipeline.tokenizer_2 if is_sdxl else pipeline.tokenizer,
                device, dtype
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def generate_txt2vid(self, params: Dict[str, Any], progress_callback=None, step_callback=None):
        """Generate a video from text (LTX-2.3 or MiniMax-H3).

        Args:
            params: Generation parameters (see TXT2VID_DEFAULTS, resolved
                against the loaded arch's overlay by the route).
            progress_callback: Called as (step, total_steps) at each denoise step.
            step_callback: Per-step latent preview hook. Consumed by MiniMax-H3
                as `(i, total, latents, None, pred_x0)`, where both tensors are
                unpatchified `[1, C, T_lat, H_lat, W_lat]` latents (NOT packed
                rows) and `pred_x0` is `x_t + sigma_t * v_t` off the latent the
                step's velocity was predicted from; unused for LTX-2.3,
                whose pipeline exposes no equivalent. The video routes pass
                None today — there is no video preview surface yet.

        Returns:
            tuple: (frames, audio, audio_sample_rate, actual_seed) where frames is
            a np.uint8 array [T, H, W, 3] and audio is a torch.FloatTensor
            [channels, samples] on CPU (or None when audio disabled).
        """
        if self.is_ltx2_model:
            return self._generate_txt2vid_ltx2(params, progress_callback, step_callback)
        if self.is_minimax_h3_model:
            return self._generate_txt2vid_minimax_h3(params, progress_callback, step_callback)

        from api.error_handlers import ValidationError
        raise ValidationError(
            "Text-to-video generation requires a video model",
            detail="The currently loaded model is not a video model. Load an LTX-2.3 or MiniMax-H3 "
                   "model to use /generate/txt2vid.",
        )

    def generate_img2vid(self, params: Dict[str, Any], input_image, progress_callback=None,
                         step_callback=None, last_frame_image=None, keyframes=None,
                         input_audio=None):
        """Generate a video from uploaded media (LTX-2.3 or MiniMax-H3).

        Args:
            params: Generation parameters (see IMG2VID_DEFAULTS, resolved
                against the loaded arch's overlay by the route).
            input_image: PIL.Image used as a keyframe (the first frame unless
                the route placed it elsewhere). May be None on MiniMax-H3 when
                `input_audio` is supplied -- that request conditions on the
                prompt and the audio alone. LTX-2.3 always needs one.
            progress_callback: Called as (step, total_steps) at each denoise step.
            step_callback: Per-step latent preview hook. Consumed by MiniMax-H3
                exactly as in generate_txt2vid; unused for LTX-2.3.
            last_frame_image: Optional PIL.Image used as the LAST-frame keyframe
                (MiniMax-H3's `fl2va` two-condition workflow). LTX-2.3's
                image-to-video pipeline conditions on the first frame only and
                ignores it -- the route warns via arch_capabilities rather than
                refusing, because one endpoint serves both architectures.
            keyframes: The RESOLVED keyframe placement plan for MiniMax-H3:
                `(anchor, PIL.Image)` in packed order, anchor being "first",
                "last" or an integer pixel frame. Built by the route
                (`generation_utils.plan_keyframe_placements`), which resolves
                the `-1` sentinel against the SNAPPED clip length -- something
                only the route can do, since the snap happens there. None means
                the pre-placement shape (input_image first, last_frame_image
                last). Ignored by LTX-2.3, which pins the uploaded image as
                frame 0 and declares `keyframe_placement` unsupported.
            input_audio: ia2v track for MiniMax-H3 -- a `[2, samples]` float32
                waveform, already at the audio VAE's rate and at the exact
                length this clip needs (`h3_references.prepare_pinned_audio`,
                called by the route so a too-short track is a 400 rather than a
                failed generation). Its rows are pinned clean across the whole
                clip and the video is generated against them; the returned
                audio is the source waveform, not a decode. Ignored by LTX-2.3,
                which declares `audio_conditioning` unsupported.

        Returns:
            tuple: (frames, audio, audio_sample_rate, actual_seed) — identical
            contract to generate_txt2vid.
        """
        if self.is_ltx2_model:
            return self._generate_img2vid_ltx2(params, input_image, progress_callback, step_callback)
        if self.is_minimax_h3_model:
            return self._generate_img2vid_minimax_h3(
                params, input_image, last_frame_image=last_frame_image,
                progress_callback=progress_callback, step_callback=step_callback,
                keyframes=keyframes, input_audio=input_audio)

        from api.error_handlers import ValidationError
        raise ValidationError(
            "Image-to-video generation requires a video model",
            detail="The currently loaded model is not a video model. Load an LTX-2.3 or MiniMax-H3 "
                   "model to use /generate/img2vid.",
        )

    def generate_ref2vid(self, params: Dict[str, Any], references, progress_callback=None,
                         step_callback=None, keyframes=None):
        """Generate a video from omni-references (MiniMax-H3 `ref2va` only).

        Args:
            params: Generation parameters (see REF2VID_DEFAULTS, resolved
                against the loaded arch's overlay by the route).
            references: `core.models.minimax_h3.h3_references.MiniMaxH3Reference`
                list, IN THE ORDER THE MODEL READS THEM -- the order labels the
                references in the prompt presentation and lays them out on the
                packed sequence's rotary clock.
            progress_callback: Called as (step, total_steps) at each denoise step.
            step_callback: Per-step latent preview hook, exactly as in
                generate_txt2vid.
            keyframes: Optional (C5) `(anchor, PIL.Image)` placement plan, same
                shape as `generate_img2vid`'s -- laid out AFTER the reference
                blocks. None/empty is a plain ref2vid request.

        Returns:
            tuple: (frames, audio, audio_sample_rate, actual_seed) -- identical
            contract to generate_txt2vid.

        Raises:
            ValidationError: if the loaded model is not MiniMax-H3. There is no
                second architecture to dispatch to: LTX-2.3 has no
                omni-reference workflow, which is why this endpoint is not one
                of the two-architecture video routes.
        """
        if self.is_minimax_h3_model:
            return self._generate_ref2vid_minimax_h3(
                params, references, progress_callback=progress_callback,
                step_callback=step_callback, keyframes=keyframes or ())

        from api.error_handlers import ValidationError
        raise ValidationError(
            "Reference-to-video generation requires a MiniMax-H3 ref2va model",
            detail="The currently loaded model is not MiniMax-H3. Omni-reference conditioning "
                   "(up to 9 images, 3 videos and 3 audio clips in one packed sequence) is a "
                   "MiniMax-H3 ref2va workflow; no other architecture in this repo implements it.",
        )

    def generate_vid_outpaint(
        self,
        params: Dict[str, Any],
        video_frames,
        fps: float,
        input_audio,
        progress_callback=None,
        step_callback=None,
        bridge_frames=None,
        bridge_fps=None,
        bridge_audio=None,
        reference_images=(),
    ):
        """Video temporal outpaint: place a (trimmed) input clip inside a
        LONGER output timeline and generate the frames before/after.

        The two video architectures do this by DIFFERENT mechanisms, because
        their conditioning differs, and the difference is visible in the
        contract rather than hidden behind it:

        * **LTX-2.3** — pure orchestration over the stock
          `diffusers.LTX2ConditionPipeline` (no new denoise loop): the whole
          timeline is generated with the clip pinned at an arbitrary latent
          index, and the input is pasted back frame-exact afterwards. Any
          offset is placeable. See `_generate_vidoutpaint_ltx2` and
          `scratchpad/outpaint_design.md` section 4.
        * **MiniMax-H3** — only the MISSING span is generated, anchored on the
          preserved clip's boundary frame(s), and the result is concatenated
          with the untouched input. It conditions on first/last frames only, so
          the clip must abut a timeline boundary or bridge two clips; a
          mid-timeline placement is refused with that reason. See
          `_generate_vidoutpaint_minimax_h3`.

        Args:
            params: see `OUTPAINT_VIDEO_DEFAULTS`.
            video_frames: np.uint8 [T, H, W, 3] decoded input clip, as
                returned by `utils.video_utils.load_video_frames`.
            fps: the input clip's own probed frame rate.
            input_audio: WAV bytes of the input clip's original audio track
                (see `utils.video_utils.extract_audio_stream`), or None.
            progress_callback: Called as (step, total_steps) at each denoise step.
            step_callback: Per-step latent preview hook. Consumed by MiniMax-H3
                exactly as in generate_txt2vid; unused for LTX-2.3.
            bridge_frames / bridge_fps / bridge_audio: the same three things for
                an optional SECOND clip preserved at the END of the timeline,
                which turns the request into a bridge. Only an architecture
                whose `TemporalSpec` lists the `bridge` placement accepts them.
            reference_images: optional PIL images, MiniMax-H3 ref2va only
                (extend_forward). The route's own gate refuses this on any
                other architecture/variant/placement; this is the defensive
                re-check for an internal caller that bypasses the route.

        Returns:
            tuple: (frames, audio, audio_sample_rate, actual_seed) --
            identical contract to generate_img2vid/generate_txt2vid.
        """
        from api.error_handlers import ValidationError

        if self.is_ltx2_model:
            if bridge_frames is not None:
                raise ValidationError(
                    "this model has no bridge placement",
                    detail="bridge_video adds a SECOND preserved clip at the end of the timeline, "
                           "which is a placement only an architecture that conditions on boundary "
                           "frames needs. LTX-2.3 places one clip at an arbitrary offset instead.",
                )
            if reference_images:
                raise ValidationError(
                    "reference_images on outpaint is a MiniMax-H3 ref2va capability",
                    detail="LTX-2.3 has no reference-conditioned outpaint path.",
                )
            return self._generate_vidoutpaint_ltx2(
                params, video_frames, fps, input_audio, progress_callback, step_callback
            )
        if self.is_minimax_h3_model:
            return self._generate_vidoutpaint_minimax_h3(
                params, video_frames, fps, input_audio, progress_callback, step_callback,
                bridge_frames=bridge_frames, bridge_fps=bridge_fps, bridge_audio=bridge_audio,
                reference_images=reference_images,
            )

        raise ValidationError(
            "Video outpaint requires a video model",
            detail="The currently loaded model is not a video model. Load an LTX-2.3 or MiniMax-H3 "
                   "model to use /generate/outpaint/video.",
        )

    def generate_vid_inpaint(
        self,
        params: Dict[str, Any],
        video_frames,
        fps: float,
        input_audio,
        progress_callback=None,
        step_callback=None,
        spatial_mask_timeline=None,
        spatial_mask_arrays=None,
        references=(),
    ):
        """Video temporal inpaint: regenerate one time range of a clip in place.

        ONE architecture, unlike the other video routes: the mechanism is a
        permutation of MiniMax-H3's packed video rows that puts the kept latent
        frames in the conditioning prefix, which LTX-2.3's pipeline has no
        equivalent of -- its conditions carry whole clips at latent indices, not
        a per-frame pin of the target itself. So this refuses rather than
        dispatching, with that reason.

        Args:
            params: see `INPAINT_VIDEO_DEFAULTS`.
            video_frames: np.uint8 [T, H, W, 3] decoded input clip.
            fps: the input clip's own probed frame rate.
            input_audio: WAV bytes of the clip's original audio, or None.
            progress_callback: Called as (step, total_steps) at each denoise step.
            step_callback: Per-step latent preview hook.
            spatial_mask_timeline: Optional spatial mask timeline for H3 inpaint.
            spatial_mask_arrays: Optional decoded spatial mask arrays for H3 inpaint.
            references: PHASE B-3-open (`minimax_h3_inpaint_refs_design.md`,
                Option B): a `ref2va` reference list, same convention as
                `generate_ref2vid`'s. Threaded straight through; allowed on
                `ref2va` and refused on `fl2va`/`hybrid` by
                `resolve_minimax_h3_inpaint_reference_gate` (see
                `_generate_vidinpaint_minimax_h3`'s own docstring for the
                unmeasured-shape caveat).

        Returns:
            tuple: (frames, audio, audio_sample_rate, actual_seed) -- identical
            contract to generate_vid_outpaint.
        """
        from api.error_handlers import ValidationError

        if self.is_minimax_h3_model:
            if spatial_mask_timeline is None and spatial_mask_arrays is None:
                return self._generate_vidinpaint_minimax_h3(
                    params, video_frames, fps, input_audio, progress_callback, step_callback,
                    references=references)
            return self._generate_vidinpaint_minimax_h3(
                params, video_frames, fps, input_audio, progress_callback, step_callback,
                spatial_mask_timeline=spatial_mask_timeline,
                spatial_mask_arrays=spatial_mask_arrays,
                references=references,
            )

        raise ValidationError(
            "Video temporal inpaint requires a MiniMax-H3 model",
            detail="Regenerating a time range in place pins the kept frames' own latents as "
                   "conditioning inside one packed sequence, which is a MiniMax-H3 mechanism; "
                   "LTX-2.3 has no equivalent and no other architecture in this repo implements "
                   "it. Load a MiniMax-H3 fl2va model, or use /generate/outpaint/video to extend "
                   "a clip.",
        )

    def generate_txt2aud(self, params: Dict[str, Any], progress_callback=None, step_callback=None):
        """Generate music/audio from text (ACE-Step 1.5 or MiniMax Music 3).

        Args:
            params: Generation parameters. ACE-Step: caption/prompt, lyrics,
                audio_duration, seed, inference_steps, guidance_scale, shift,
                sampler_mode, bpm, key_scale, time_signature, vocal_language.
                MiniMax Music 3: prompt, lyrics, seed, audio_duration,
                num_inference_steps, flow_guidance_scale (all three required,
                no default -- see `MiniMaxMusic3Mixin._generate_txt2aud_minimax_music3`).
            progress_callback: Called as (step, total_steps).
            step_callback: Reserved (unused for txt2aud on either arch).

        Returns:
            ACE-Step: (waveform, sample_rate, actual_seed) tuple, waveform a
            torch.FloatTensor [2, samples] on CPU, sample_rate 48000.
            MiniMax Music 3: `MiniMaxMusic3Txt2AudResult` (NOT the same
            3-tuple shape -- see its own docstring for why: the design doc's
            per-generation frame-code state contract must survive this call
            for a later commit's route to persist).
        """
        if self.is_acestep_model:
            return self._generate_txt2aud_acestep(params, progress_callback, step_callback)
        if self.is_minimax_music3_model:
            return self._generate_txt2aud_minimax_music3(params, progress_callback, step_callback)

        from api.error_handlers import ValidationError
        raise ValidationError(
            "Text-to-audio generation requires an ACE-Step or MiniMax Music 3 model",
            detail="The currently loaded model is not an audio model. Load an ACE-Step or MiniMax "
                   "Music 3 model to use /generate/txt2aud.",
        )

    def generate_aud2aud(self, params: Dict[str, Any], reference_audio, progress_callback=None, step_callback=None):
        """Generate a cover OR repaint (audio-to-audio) from a reference clip
        + text conditioning (ACE-Step 1.5 only). See
        `AceStepMixin._generate_aud2aud_acestep` docstring for the full
        mode contract.

        Args:
            params: Generation parameters -- caption/prompt, lyrics, mode
                ("cover"|"repaint", default "cover"), cover_strength (cover
                only), repaint_start/repaint_end (seconds, repaint only),
                seed, inference_steps, guidance_scale, shift,
                vocal_language/bpm/key_scale/time_signature.
            reference_audio: a file path (str) or raw audio bytes for the
                cover/repaint reference clip.
            progress_callback: Called as (step, total_steps).
            step_callback: Reserved (unused for ACE-Step aud2aud).

        Returns:
            tuple: (waveform, sample_rate, actual_seed) -- identical contract
            to generate_txt2aud.
        """
        if self.is_acestep_model:
            return self._generate_aud2aud_acestep(params, reference_audio, progress_callback, step_callback)

        from api.error_handlers import ValidationError
        raise ValidationError(
            "Audio-to-audio generation requires an ACE-Step model",
            detail="The currently loaded model is not an audio model. Load an ACE-Step model to use /generate/aud2aud.",
        )

    def generate_aud_outpaint(self, params: Dict[str, Any], reference_audio, progress_callback=None, step_callback=None):
        """Audio temporal outpaint (extend): ACE-Step 1.5 places a (trimmed)
        input clip at a time offset inside a LONGER output timeline and
        generates the audio before/and-or after it (see
        `AceStepMixin._generate_audoutpaint_acestep` -- the structural
        inverse of `generate_aud2aud`'s `mode="repaint"`). MiniMax Music 3
        instead forward-extends a SushiUI-generated song by resuming its
        autoregressive stage from a stored frame-code sidecar -- backward
        extension and mid-song infill are refused (causal LM); see
        `MiniMaxMusic3Mixin._generate_audoutpaint_minimax_music3`'s
        docstring for the full mechanism and its `MiniMaxMusic3ExtendResult`
        return shape (NOT the plain 3-tuple below).

        Args:
            params: ACE-Step -- see `OUTPAINT_AUDIO_DEFAULTS`: prompt/lyrics,
                seed, inference_steps, guidance_scale, shift, vocal_language,
                loras, total_duration (seconds, output timeline length),
                input_offset_sec (seconds, where the trimmed input is
                placed), input_trim_start_sec/input_trim_end_sec (seconds,
                trims the UPLOADED clip itself before placement). MiniMax
                Music 3 -- `placement` (required, only `"extend_forward"`),
                `extend_duration_sec`/`num_inference_steps`/
                `flow_guidance_scale` (required, no fallback), `seed`,
                `content_hash` (optional).
            reference_audio: ACE-Step -- a file path (str) or raw audio
                bytes for the input clip to place. MiniMax Music 3 -- a
                server-side file PATH (str) ONLY (the sidecar must already
                sit next to it; raw upload bytes are refused).
            progress_callback: Called as (step, total_steps).
            step_callback: Reserved (unused on either architecture).

        Returns:
            ACE-Step: (waveform, sample_rate, actual_seed) tuple -- identical
            contract to generate_txt2aud/generate_aud2aud. MiniMax Music 3:
            `MiniMaxMusic3ExtendResult` (see its own docstring for why this
            is not the plain 3-tuple).
        """
        if self.is_acestep_model:
            return self._generate_audoutpaint_acestep(params, reference_audio, progress_callback, step_callback)
        if self.is_minimax_music3_model:
            return self._generate_audoutpaint_minimax_music3(params, reference_audio, progress_callback, step_callback)

        from api.error_handlers import ValidationError
        raise ValidationError(
            "Audio outpaint requires an ACE-Step or MiniMax Music 3 model",
            detail="The currently loaded model is not an audio model. Load an ACE-Step or MiniMax Music 3 model "
                   "to use /generate/outpaint/audio.",
        )

    def generate_txt2img(self, params: Dict[str, Any], progress_callback=None, step_callback=None) -> tuple[Union[Image.Image, torch.Tensor], int, int]:
        """Generate image from text

        Args:
            params: Generation parameters
            progress_callback: Legacy callback for progress (step, timestep, latents)
            step_callback: New style callback for step-based control (pipe, step, timestep, callback_kwargs)

        Returns:
            tuple: (image, actual_seed, actual_ancestral_seed). `image` is a raw
            torch.Tensor (the pre-unscale final latent) instead of a PIL.Image
            when params["loop_decode"] == "none" (SD1.5/SDXL legacy path only,
            see custom_sampling_loop's Stage-3 site) -- callers must check
            isinstance(image, Image.Image) before treating it as a decoded image.
        """
        # VAE tiling flag for this request (read by all decode paths, incl. the
        # per-architecture handlers dispatched just below).
        self._vae_tiling = bool(params.get("vae_tiling", False))
        self._vae_tile_threshold = int(params.get("vae_tile_threshold", 0) or 0)
        self._vae_tile_mode = str(params.get("vae_tile_mode", "blend") or "blend")
        self._vae_tile_global_norm = bool(params.get("vae_tile_global_norm", False))
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

        # LTX-2.3 is a video model — image endpoints must not run it (P1b adds
        # /generate/txt2vid, /generate/img2vid).
        if self.is_ltx2_model:
            from api.error_handlers import ValidationError
            raise ValidationError(
                "LTX-2.3 is a video model — use /generate/txt2vid or /generate/img2vid",
                detail="The currently loaded model is LTX-2.3, which produces video, not still images.",
            )

        # MiniMax-H3 likewise. The route-level `_reject_if_video_model` fires
        # first for an API request; this is the second line, for every internal
        # caller that reaches the pipeline directly.
        if self.is_minimax_h3_model:
            from api.error_handlers import ValidationError
            raise ValidationError(
                "MiniMax-H3 is a video model — use /generate/txt2vid or /generate/img2vid",
                detail="The currently loaded model is MiniMax-H3, which produces video with a "
                       "joint audio track, not still images. Its shortest decodable clip is 22 "
                       "frames; there is no single-image path.",
            )

        if not self.txt2img_pipeline:
            raise RuntimeError("txt2img pipeline not loaded. Please load a model first.")

        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident, clear_resident,
            discard_resident, should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        from core.vram_optimization import move_text_encoders_to_cpu as _kh_te_to_cpu, \
            move_unet_to_cpu as _kh_unet_to_cpu, move_vae_to_cpu as _kh_vae_to_cpu
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        _kh_cpu_text_encoding = bool(params.get("cpu_text_encoding", False))
        _kh_has_loras = bool(params.get("loras") or [])
        # If a resident set exists from a previous generation but is no longer valid
        # for THIS request's model_key (checkpoint/LoRA/quantization/dtype changed),
        # force a full offload before staging anything.
        invalidate_if_model_changed(
            self, params,
            offload_fn=lambda: (
                _kh_te_to_cpu(self.txt2img_pipeline),
                _kh_unet_to_cpu(self.txt2img_pipeline),
                _kh_vae_to_cpu(self.txt2img_pipeline),
            ),
        )
        _kh_total_bytes = 0
        if _kh_requested:
            if not _kh_cpu_text_encoding:
                _kh_total_bytes += component_nbytes(getattr(self.txt2img_pipeline, "text_encoder", None))
                _kh_total_bytes += component_nbytes(getattr(self.txt2img_pipeline, "text_encoder_2", None))
            # LoRA hazard gate (Phase A): LoRA mutates the U-Net per generation, so
            # keeping it resident is only safe when the next gen's LoRA set is
            # guaranteed identical. Routes.py currently reloads/unloads LoRA around
            # every generation regardless of keep-hot, so provably-safe skip-reload
            # coordination is NOT wired yet in this phase -- gate U-Net-hot to the
            # no-LoRA case only. TODO(Phase A follow-up / Phase B): once routes.py
            # skips the LoRA unload/reload for an unchanged model_key, drop this gate.
            if not _kh_has_loras:
                _kh_total_bytes += component_nbytes(getattr(self.txt2img_pipeline, "unet", None))
            _kh_total_bytes += component_nbytes(getattr(self.txt2img_pipeline, "vae", None))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        _kh_keep_te = _kh_requested and _kh_guard_ok and not _kh_cpu_text_encoding
        _kh_keep_unet = _kh_requested and _kh_guard_ok and not _kh_has_loras
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

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
        if not cpu_text_encoding and not is_resident(self, "text_encoder", _kh_model_key):
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

        # Offload text encoders to CPU after all encoding is complete (unless kept hot)
        if _kh_keep_te:
            mark_resident(self, "text_encoder", _kh_model_key)
        else:
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
        if not is_resident(self, "unet", _kh_model_key):
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

            # Training-free reference-style transfer (StyleAligned/VSP-style KV
            # injection): build the (config, ref_x0, ref_noise) triple from
            # params["style_transfer"] (assembled by process_controlnet_configs from
            # an is_style_transfer ControlNet-shaped entry), or (None, None, None)
            # when no style reference is attached -- fully gated OFF by default.
            # build_style_transfer_all also covers multi-reference (N>1):
            # params["style_transfers"] with 2+ entries populates style_refs instead
            # (style_cfg/style_ref_x0/style_eps_ref stay None in that case) and a
            # single-entry style_transfers routes through the single-ref triple, so
            # single-ref behavior is unaffected either way.
            from core.inference.custom_sampling import build_style_transfer_all
            _unet_dtype_for_style = next(pipeline_to_use.unet.parameters()).dtype
            style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = build_style_transfer_all(
                params, pipeline_to_use,
                width=gen_params["width"], height=gen_params["height"],
                device=device, dtype=_unet_dtype_for_style, seed=actual_seed,
            )
            # The style KV-injection hook lives ONLY in UnifiedAttnProcessor, but the
            # default attention_type "normal" leaves diffusers' stock AttnProcessor2_0
            # on the U-Net -> style would be a SILENT no-op. When style is active and
            # the stock processor is still installed (no attention-backend swap above),
            # force-install UnifiedAttnProcessor (backend "normal" == native SDPA,
            # byte-identical to stock when no _style_ctx is attached). Restored via
            # self.original_processors in the finally. Skip under NAG (it owns the
            # processors and style yields to it downstream).
            if (style_cfg is not None or style_refs is not None) and self.original_processors is None and not params.get("nag_enable", False):
                from core.inference.attention_processors import set_attention_processor
                print("[Pipeline] Style transfer active with attention_type=normal; installing UnifiedAttnProcessor so the KV-injection hook is present")
                self.original_processors = set_attention_processor(pipeline_to_use.unet, "normal")
                self.current_attention_type = "normal"

            # Call custom sampling loop. The legacy SD/SDXL path folds VAE decode
            # into this loop, so denoise and decode are not separable here — the
            # combined span is recorded as the "denoise" phase.
            _t_denoise = time.perf_counter()
            image = custom_sampling_loop(
                pipeline=pipeline_to_use,
                style_cfg=style_cfg,
                style_ref_x0=style_ref_x0,
                style_eps_ref=style_eps_ref,
                style_refs=style_refs,
                style_combine_mode=style_combine_mode,
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
                loop_decode=params.get("loop_decode", "full"),
                **controlnet_kwargs,
            )
            generation_timer.add("denoise", time.perf_counter() - _t_denoise)
            _kh_gen_succeeded = True

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

            # Offload all components to CPU to free VRAM -- EXCEPT components kept
            # hot on a SUCCESSFUL generation. On an exception, ALWAYS force a full
            # offload + clear residency (never trust the pipeline state after an
            # error going into the next generation).
            from core.vram_optimization import move_text_encoders_to_cpu, move_unet_to_cpu, move_vae_to_cpu
            if not _kh_gen_succeeded:
                clear_resident(self)
                move_text_encoders_to_cpu(pipeline_to_use)
                move_unet_to_cpu(pipeline_to_use)
                move_vae_to_cpu(pipeline_to_use)
            else:
                # A component that is NOT kept hot must be dropped from the
                # resident set (discard_resident) in addition to being offloaded,
                # so state never claims a component is GPU-resident after it was
                # moved to CPU (that would make the next same-model generation
                # skip its ->GPU stage -> device mismatch).
                if _kh_keep_te:
                    mark_resident(self, "text_encoder", _kh_model_key)
                else:
                    move_text_encoders_to_cpu(pipeline_to_use)
                    discard_resident(self, "text_encoder")
                if _kh_keep_unet:
                    mark_resident(self, "unet", _kh_model_key)
                else:
                    move_unet_to_cpu(pipeline_to_use)
                    discard_resident(self, "unet")
                if _kh_keep_vae:
                    mark_resident(self, "vae", _kh_model_key)
                else:
                    move_vae_to_cpu(pipeline_to_use)
                    discard_resident(self, "vae")

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

        # Apply extensions after generation -- skipped when loop_decode="none"
        # returned a raw latent tensor instead of a decoded image (nothing to
        # post-process; the next loop step's img2img denoise runs on it instead).
        if isinstance(image, Image.Image):
            for ext in self.extensions:
                if ext.enabled:
                    image = ext.process_after_generation(image, params)

        return image, actual_seed, actual_ancestral_seed

    def generate_img2img(self, params: Dict[str, Any], init_image: Optional[Image.Image] = None, progress_callback=None, step_callback=None) -> tuple[Union[Image.Image, torch.Tensor], int, int]:
        """Generate image from image

        Returns:
            tuple: (image, actual_seed, actual_ancestral_seed). `image` is a raw
            torch.Tensor (pre-unscale final latent) instead of a PIL.Image when
            params["loop_decode"] == "none" (SD1.5/SDXL legacy path only) --
            see generate_txt2img's docstring for the same contract.
        """
        self._vae_tiling = bool(params.get("vae_tiling", False))
        self._vae_tile_threshold = int(params.get("vae_tile_threshold", 0) or 0)
        self._vae_tile_mode = str(params.get("vae_tile_mode", "blend") or "blend")
        self._vae_tile_global_norm = bool(params.get("vae_tile_global_norm", False))
        self._color_flatten_strength = int(params.get("color_flatten_strength", 0) or 0)
        # In-loop hard-flatten (SD1.5/SDXL): master switch + last-N steps + region gate.
        self._flatten_in_loop = bool(params.get("flatten_in_loop", False))
        self._flatten_in_loop_last_steps = int(params.get("flatten_in_loop_last_steps", 3) or 3)
        self._flatten_in_loop_min_region = float(params.get("flatten_in_loop_min_region", 0.02) or 0.02)
        self._vae_drift_correction = bool(params.get("vae_drift_correction", False))

        # Loop-generation latent passthrough (input_latent_id) is only wired
        # into the legacy SD1.5/SDXL path below (custom_img2img_sampling_loop's
        # init_latents_override). Refuse it up front for every other
        # architecture rather than dispatching into a handler that expects a
        # real PIL image and would fail deep inside with a confusing error.
        if params.get("input_latent_id") and (
            self.is_zimage_model or self.is_flux2_model or self.is_anima_model
            or self.is_lens_model or self.is_ideogram4_model or self.is_minit2i_model
            or self.is_krea2_model or self.is_ltx2_model
        ):
            from api.error_handlers import ValidationError
            raise ValidationError(
                "input_latent_id (loop latent passthrough) is only supported for SD1.5/SDXL models",
                detail="The currently loaded model architecture does not support latent-passthrough "
                       "img2img; upload an image instead, or use loop_decode='cheap' for lower-cost "
                       "intermediate loop steps.",
            )

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

        # LTX-2.3 is a video model — image endpoints must not run it (P1b adds
        # /generate/txt2vid, /generate/img2vid).
        if self.is_ltx2_model:
            from api.error_handlers import ValidationError
            raise ValidationError(
                "LTX-2.3 is a video model — use /generate/txt2vid or /generate/img2vid",
                detail="The currently loaded model is LTX-2.3, which produces video, not still images.",
            )

        # MiniMax-H3 likewise. The route-level `_reject_if_video_model` fires
        # first for an API request; this is the second line, for every internal
        # caller that reaches the pipeline directly.
        if self.is_minimax_h3_model:
            from api.error_handlers import ValidationError
            raise ValidationError(
                "MiniMax-H3 is a video model — use /generate/txt2vid or /generate/img2vid",
                detail="The currently loaded model is MiniMax-H3, which produces video with a "
                       "joint audio track, not still images. Its shortest decodable clip is 22 "
                       "frames; there is no single-image path.",
            )

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

        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident, clear_resident,
            discard_resident, should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        from core.vram_optimization import move_text_encoders_to_cpu as _kh_te_to_cpu, \
            move_unet_to_cpu as _kh_unet_to_cpu, move_vae_to_cpu as _kh_vae_to_cpu
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        _kh_cpu_text_encoding = bool(params.get("cpu_text_encoding", False))
        _kh_has_loras = bool(params.get("loras") or [])
        invalidate_if_model_changed(
            self, params,
            offload_fn=lambda: (
                _kh_te_to_cpu(self.img2img_pipeline),
                _kh_unet_to_cpu(self.img2img_pipeline),
                _kh_vae_to_cpu(self.img2img_pipeline),
            ),
        )
        _kh_total_bytes = 0
        if _kh_requested:
            if not _kh_cpu_text_encoding:
                _kh_total_bytes += component_nbytes(getattr(self.img2img_pipeline, "text_encoder", None))
                _kh_total_bytes += component_nbytes(getattr(self.img2img_pipeline, "text_encoder_2", None))
            # LoRA hazard gate (Phase A) -- see generate_txt2img for rationale.
            if not _kh_has_loras:
                _kh_total_bytes += component_nbytes(getattr(self.img2img_pipeline, "unet", None))
            _kh_total_bytes += component_nbytes(getattr(self.img2img_pipeline, "vae", None))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        _kh_keep_te = _kh_requested and _kh_guard_ok and not _kh_cpu_text_encoding
        _kh_keep_unet = _kh_requested and _kh_guard_ok and not _kh_has_loras
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

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

        # Loop-generation latent passthrough: resolve input_latent_id (if any)
        # to a cached latent BEFORE the normal image-resize logic below, since
        # it replaces the input image entirely -- see custom_img2img_sampling_loop's
        # init_latents_override and this module's _resize_latent helper.
        from api.error_handlers import ValidationError
        _input_latent_id = params.get("input_latent_id")
        init_latents_override = None
        if _input_latent_id:
            from core.inference.latent_cache import get_latent
            _cached = get_latent(_input_latent_id)
            if _cached is None:
                raise ValidationError(
                    "Input latent expired or not found; restart the loop",
                    detail=f"latent_id={_input_latent_id}",
                )
            _cached_latent, _cached_meta = _cached
            _target_lat_h = (target_height or settings.default_height) // 8
            _target_lat_w = (target_width or settings.default_width) // 8
            if _cached_latent.shape[-2:] != (_target_lat_h, _target_lat_w):
                print(f"[img2img] Latent passthrough: resizing cached latent "
                      f"{_cached_latent.shape[-1]}x{_cached_latent.shape[-2]} -> "
                      f"{_target_lat_w}x{_target_lat_h} ({resampling_method})")
                _cached_latent = self._resize_latent(_cached_latent, _target_lat_h, _target_lat_w, resampling_method)
            init_latents_override = _cached_latent

        if init_image is None:
            if init_latents_override is None:
                raise ValidationError("img2img requires either an input image or input_latent_id")
            # Size-only placeholder: its pixels are never read (init_latents_override
            # feeds custom_img2img_sampling_loop directly) -- matching its size to
            # the target makes every init_image.size-driven block below (resize
            # blocks, style-transfer width/height, ...) a correct no-op/no-encode.
            init_image = Image.new("RGB", (
                target_width or settings.default_width,
                target_height or settings.default_height,
            ))

        # Resize input image if width/height are specified and mode is "image"
        # (never applies to latent passthrough -- see the latent resize above).
        if target_width and target_height and resize_mode == "image" and init_latents_override is None:
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
        if not cpu_text_encoding and not is_resident(self, "text_encoder", _kh_model_key):
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

        # Offload text encoders to CPU after all encoding is complete (unless kept hot)
        if _kh_keep_te:
            mark_resident(self, "text_encoder", _kh_model_key)
        else:
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

        # Handle latent resize mode by encoding, resizing latent, then decoding.
        # Never applies to latent passthrough (init_latents_override is resized
        # directly, with no VAE round-trip at all -- see above).
        if resize_mode == "latent" and target_width and target_height and init_latents_override is None:
            if init_image.size != (target_width, target_height):
                print(f"Using latent resize mode: {init_image.size} -> {target_width}x{target_height} with {resampling_method}")

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

                resized_latent = self._resize_latent(latent, latent_height, latent_width, resampling_method)

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
            if not is_resident(self, "unet", _kh_model_key):
                move_unet_to_gpu(pipeline_to_use, quantization=unet_quantization, use_torch_compile=use_torch_compile)

            log_device_status("Ready for U-Net inference (img2img)", pipeline_to_use, vision_encoder=getattr(self, 'vision_encoder', None))

            # Training-free reference-style transfer (StyleAligned/VSP-style KV
            # injection): build the (config, ref_x0, ref_noise) triple from
            # params["style_transfer"] (assembled by process_controlnet_configs from
            # an is_style_transfer ControlNet-shaped entry), or (None, None, None)
            # when no style reference is attached -- fully gated OFF by default.
            # build_style_transfer_all also covers multi-reference (N>1) -- see the
            # txt2img site for the full rationale.
            from core.inference.custom_sampling import build_style_transfer_all
            _unet_dtype_for_style = next(pipeline_to_use.unet.parameters()).dtype
            style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = build_style_transfer_all(
                params, pipeline_to_use,
                width=init_image.width, height=init_image.height,
                device=self.device, dtype=_unet_dtype_for_style, seed=actual_seed,
            )
            # Force-install UnifiedAttnProcessor when style is active but the stock
            # processor is still on the U-Net (default attention_type="normal") so the
            # KV-injection hook is present (see the txt2img site for the full rationale).
            if (style_cfg is not None or style_refs is not None) and self.original_processors is None and not params.get("nag_enable", False):
                from core.inference.attention_processors import set_attention_processor
                print("[Pipeline] Style transfer (img2img) active with attention_type=normal; installing UnifiedAttnProcessor")
                self.original_processors = set_attention_processor(pipeline_to_use.unet, "normal")
                self.current_attention_type = "normal"

            # Call custom img2img sampling loop. VAE decode is folded into the loop
            # on this legacy path, so the combined span is recorded as "denoise".
            _t_denoise = time.perf_counter()
            image = custom_img2img_sampling_loop(
                pipeline=pipeline_to_use,
                style_cfg=style_cfg,
                style_ref_x0=style_ref_x0,
                style_eps_ref=style_eps_ref,
                style_refs=style_refs,
                style_combine_mode=style_combine_mode,
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
                loop_decode=params.get("loop_decode", "full"),
                init_latents_override=init_latents_override,
                **controlnet_kwargs,
            )
            generation_timer.add("denoise", time.perf_counter() - _t_denoise)
            _kh_gen_succeeded = True

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

            # Offload all components to CPU to free VRAM -- EXCEPT components kept
            # hot on a SUCCESSFUL generation (see generate_txt2img for the contract).
            from core.vram_optimization import move_text_encoders_to_cpu, move_unet_to_cpu, move_vae_to_cpu
            if not _kh_gen_succeeded:
                clear_resident(self)
                move_text_encoders_to_cpu(pipeline_to_use)
                move_unet_to_cpu(pipeline_to_use)
                move_vae_to_cpu(pipeline_to_use)
            else:
                # Non-kept components are dropped from the resident set (see the
                # txt2img finally for why) as well as offloaded.
                if _kh_keep_te:
                    mark_resident(self, "text_encoder", _kh_model_key)
                else:
                    move_text_encoders_to_cpu(pipeline_to_use)
                    discard_resident(self, "text_encoder")
                if _kh_keep_unet:
                    mark_resident(self, "unet", _kh_model_key)
                else:
                    move_unet_to_cpu(pipeline_to_use)
                    discard_resident(self, "unet")
                if _kh_keep_vae:
                    mark_resident(self, "vae", _kh_model_key)
                else:
                    move_vae_to_cpu(pipeline_to_use)
                    discard_resident(self, "vae")

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

        # Apply extensions after generation -- skipped when loop_decode="none"
        # returned a raw latent tensor instead of a decoded image.
        if isinstance(image, Image.Image):
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
    ) -> tuple[Image.Image, int, int]:
        """Generate inpainted image

        Returns:
            tuple: (image, actual_seed, actual_ancestral_seed). Unlike
            generate_txt2img/generate_img2img, `image` here is ALWAYS a decoded
            PIL.Image -- loop_decode="none" (latent passthrough) is not
            supported for inpaint (see the guard below) because its
            pixel-space mask compositing needs a decoded image.
        """
        self._vae_tiling = bool(params.get("vae_tiling", False))
        self._vae_tile_threshold = int(params.get("vae_tile_threshold", 0) or 0)
        self._vae_tile_mode = str(params.get("vae_tile_mode", "blend") or "blend")
        self._vae_tile_global_norm = bool(params.get("vae_tile_global_norm", False))
        self._color_flatten_strength = int(params.get("color_flatten_strength", 0) or 0)
        # In-loop hard-flatten (SD1.5/SDXL): master switch + last-N steps + region gate.
        self._flatten_in_loop = bool(params.get("flatten_in_loop", False))
        self._flatten_in_loop_last_steps = int(params.get("flatten_in_loop_last_steps", 3) or 3)
        self._flatten_in_loop_min_region = float(params.get("flatten_in_loop_min_region", 0.02) or 0.02)
        self._vae_drift_correction = bool(params.get("vae_drift_correction", False))

        # Loop-generation latent passthrough is NOT supported for inpaint:
        # pixel-space mask compositing (blending the generated region back
        # into the original image) needs a decoded image, so there is no
        # correct latent to hand back. routes.py already rejects both of
        # these up front; this is defense-in-depth for any other caller.
        if params.get("input_latent_id"):
            from api.error_handlers import ValidationError
            raise ValidationError(
                "input_latent_id (loop latent passthrough) is not supported for inpaint",
                detail="Inpaint's mask compositing requires a real source image. "
                       "Use loop_decode='cheap' for lower-cost intermediate loop steps instead.",
            )
        if params.get("loop_decode", "full") == "none":
            from api.error_handlers import ValidationError
            raise ValidationError(
                "loop_decode='none' is not supported for inpaint",
                detail="Inpaint's mask compositing requires a decoded image. "
                       "Use loop_decode='cheap' for lower-cost intermediate loop steps instead.",
            )

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

        # LTX-2.3 is a video model — image endpoints must not run it (P1b adds
        # /generate/txt2vid, /generate/img2vid).
        if self.is_ltx2_model:
            from api.error_handlers import ValidationError
            raise ValidationError(
                "LTX-2.3 is a video model — use /generate/txt2vid or /generate/img2vid",
                detail="The currently loaded model is LTX-2.3, which produces video, not still images.",
            )

        # MiniMax-H3 likewise. The route-level `_reject_if_video_model` fires
        # first for an API request; this is the second line, for every internal
        # caller that reaches the pipeline directly.
        if self.is_minimax_h3_model:
            from api.error_handlers import ValidationError
            raise ValidationError(
                "MiniMax-H3 is a video model — use /generate/txt2vid or /generate/img2vid",
                detail="The currently loaded model is MiniMax-H3, which produces video with a "
                       "joint audio track, not still images. Its shortest decodable clip is 22 "
                       "frames; there is no single-image path.",
            )

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

        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident, clear_resident,
            discard_resident, should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        from core.vram_optimization import move_text_encoders_to_cpu as _kh_te_to_cpu, \
            move_unet_to_cpu as _kh_unet_to_cpu, move_vae_to_cpu as _kh_vae_to_cpu
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        _kh_cpu_text_encoding = bool(params.get("cpu_text_encoding", False))
        _kh_has_loras = bool(params.get("loras") or [])
        invalidate_if_model_changed(
            self, params,
            offload_fn=lambda: (
                _kh_te_to_cpu(self.inpaint_pipeline),
                _kh_unet_to_cpu(self.inpaint_pipeline),
                _kh_vae_to_cpu(self.inpaint_pipeline),
            ),
        )
        _kh_total_bytes = 0
        if _kh_requested:
            if not _kh_cpu_text_encoding:
                _kh_total_bytes += component_nbytes(getattr(self.inpaint_pipeline, "text_encoder", None))
                _kh_total_bytes += component_nbytes(getattr(self.inpaint_pipeline, "text_encoder_2", None))
            # LoRA hazard gate (Phase A) -- see generate_txt2img for rationale.
            if not _kh_has_loras:
                _kh_total_bytes += component_nbytes(getattr(self.inpaint_pipeline, "unet", None))
            _kh_total_bytes += component_nbytes(getattr(self.inpaint_pipeline, "vae", None))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        _kh_keep_te = _kh_requested and _kh_guard_ok and not _kh_cpu_text_encoding
        _kh_keep_unet = _kh_requested and _kh_guard_ok and not _kh_has_loras
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

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
        if not cpu_text_encoding and not is_resident(self, "text_encoder", _kh_model_key):
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

        # Regional additional prompt (STAGE R1, method "cfg"): an additional
        # positive/negative prompt that conditions ONLY the generated region
        # (outpaint = mask_latent==1; inpaint = the repaint mask), leaving the
        # main whole-image prompt and the preserved region untouched. Encoded
        # the SAME way as the main prompt (reuse _encode_prompt_with_weights)
        # -- see scratchpad/regional_prompt_synthesis.md. Only encoded when
        # active (strength>0 AND at least one region string non-empty) --
        # otherwise this whole block is skipped (no extra encode pass, byte-
        # identical to before this feature).
        region_prompt_text = params.get("region_prompt", "") or ""
        region_negative_prompt_text = params.get("region_negative_prompt", "") or ""
        region_prompt_strength = params.get("region_prompt_strength", 1.0)
        region_prompt_method = params.get("region_prompt_method", "cfg")
        region_mask_feather = params.get("region_mask_feather", 0.0)
        region_has_positive = bool(region_prompt_text.strip())
        region_has_negative = bool(region_negative_prompt_text.strip())
        region_prompt_active = region_prompt_strength > 0 and (region_has_positive or region_has_negative)
        region_prompt_embeds = None
        region_negative_prompt_embeds = None
        region_pooled_prompt_embeds = None
        region_negative_pooled_prompt_embeds = None
        if region_prompt_active:
            print(f"[RegionalPrompt] Encoding region prompt (method={region_prompt_method}): "
                  f"positive={region_has_positive}, negative={region_has_negative}")
            (region_prompt_embeds, region_negative_prompt_embeds,
             region_pooled_prompt_embeds, region_negative_pooled_prompt_embeds) = self._encode_prompt_with_weights(
                region_prompt_text,
                region_negative_prompt_text,
                pipeline=pipeline_to_use,
            )

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
        if region_prompt_embeds is not None:
            region_prompt_embeds = region_prompt_embeds.to(device)
        if region_negative_prompt_embeds is not None:
            region_negative_prompt_embeds = region_negative_prompt_embeds.to(device)
        if region_pooled_prompt_embeds is not None:
            region_pooled_prompt_embeds = region_pooled_prompt_embeds.to(device)
        if region_negative_pooled_prompt_embeds is not None:
            region_negative_pooled_prompt_embeds = region_negative_pooled_prompt_embeds.to(device)

        # Offload text encoders to CPU after all encoding is complete (unless kept hot)
        if _kh_keep_te:
            mark_resident(self, "text_encoder", _kh_model_key)
        else:
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
        if not is_resident(self, "unet", _kh_model_key):
            move_unet_to_gpu(pipeline_to_use, quantization=unet_quantization, use_torch_compile=use_torch_compile)

        log_device_status("Ready for U-Net inference (inpaint)", pipeline_to_use, vision_encoder=getattr(self, 'vision_encoder', None))

        # Training-free reference-style transfer (StyleAligned/VSP-style KV
        # injection): build the (config, ref_x0, ref_noise) triple from
        # params["style_transfer"] (assembled by process_controlnet_configs from
        # an is_style_transfer ControlNet-shaped entry), or (None, None, None)
        # when no style reference is attached -- fully gated OFF by default.
        # build_style_transfer_all also covers multi-reference (N>1) -- see the
        # txt2img site for the full rationale.
        from core.inference.custom_sampling import build_style_transfer_all
        _unet_dtype_for_style = next(pipeline_to_use.unet.parameters()).dtype
        style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = build_style_transfer_all(
            params, pipeline_to_use,
            width=init_image.width, height=init_image.height,
            device=self.device, dtype=_unet_dtype_for_style, seed=seed,
        )
        # Force-install UnifiedAttnProcessor when style is active but the stock
        # processor is still on the U-Net (default attention_type="normal") so the
        # KV-injection hook is present (see the txt2img site for the full rationale).
        if (style_cfg is not None or style_refs is not None) and self.original_processors is None and not params.get("nag_enable", False):
            from core.inference.attention_processors import set_attention_processor
            print("[Pipeline] Style transfer (inpaint) active with attention_type=normal; installing UnifiedAttnProcessor")
            self.original_processors = set_attention_processor(pipeline_to_use.unet, "normal")
            self.current_attention_type = "normal"

        # Use custom inpaint sampling loop. VAE decode is folded into the loop on
        # this legacy path, so the combined span is recorded as "denoise".
        try:
            _t_denoise = time.perf_counter()
            image = custom_inpaint_sampling_loop(
            pipeline=pipeline_to_use,
            style_cfg=style_cfg,
            style_ref_x0=style_ref_x0,
            style_eps_ref=style_eps_ref,
            style_refs=style_refs,
            style_combine_mode=style_combine_mode,
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
            loop_decode=params.get("loop_decode", "full"),
            outpaint_noise_init=bool(params.get("_outpaint_noise_init", False)),
            outpaint_boundary_color_strength=params.get("outpaint_boundary_color_strength", 0.25),
            outpaint_resample_count=params.get("outpaint_resample_count", 2),
            outpaint_jump_length=params.get("outpaint_jump_length", 4),
            outpaint_reference_strength=params.get("outpaint_reference_strength", 0.0),
            outpaint_commit_strength=params.get("outpaint_commit_strength", 0.0),
            outpaint_commit_near=params.get("outpaint_commit_near", 0.35),
            outpaint_commit_far=params.get("outpaint_commit_far", 0.80),
            outpaint_commit_distance=params.get("outpaint_commit_distance", 32.0),
            region_prompt_embeds=region_prompt_embeds,
            region_negative_prompt_embeds=region_negative_prompt_embeds,
            region_pooled_prompt_embeds=region_pooled_prompt_embeds,
            region_negative_pooled_prompt_embeds=region_negative_pooled_prompt_embeds,
            region_has_positive=region_has_positive,
            region_has_negative=region_has_negative,
            region_prompt_strength=region_prompt_strength,
            region_prompt_method=region_prompt_method,
            region_mask_feather=region_mask_feather,
            seam_structure_strength=params.get("seam_structure_strength", 0.0),
            seam_structure_depth=params.get("seam_structure_depth", 6.0),
            seam_structure_end=params.get("seam_structure_end", 0.70),
            seam_structure_saliency=params.get("seam_structure_saliency", 2.0),
            seam_structure_max_area=params.get("seam_structure_max_area", 0.25),
            boundary_relax_strength=params.get("boundary_relax_strength", 0.0),
            boundary_relax_width=params.get("boundary_relax_width", 3.0),
            boundary_relax_noise=params.get("boundary_relax_noise", 0.35),
            boundary_relax_full_until=params.get("boundary_relax_full_until", 0.37),
            boundary_relax_end=params.get("boundary_relax_end", 0.55),
            boundary_relax_paste=params.get("boundary_relax_paste", "feather"),
            outpaint_controlnet_gate=params.get("outpaint_controlnet_gate"),
            outpaint_pin_corner_relax=params.get("outpaint_pin_corner_relax"),
            outpaint_preview_unpinned_x0=bool(params.get("outpaint_preview_unpinned_x0", False)),
            paste_feather_px=float(params.get("outpaint_paste_feather_px", 0) or 0),
            outpaint_preserve_mode=str(params.get("outpaint_preserve_mode", "exact") or "exact"),
            outpaint_seam_offset_prop=float(params.get("outpaint_seam_offset_prop", 0.0) or 0.0),
            **controlnet_kwargs,
            )
            generation_timer.add("denoise", time.perf_counter() - _t_denoise)
            _kh_gen_succeeded = True

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
            region_prompt_embeds = None
            region_negative_prompt_embeds = None
            region_pooled_prompt_embeds = None
            region_negative_pooled_prompt_embeds = None

            # Offload all components to CPU to free VRAM -- EXCEPT components kept
            # hot on a SUCCESSFUL generation (see generate_txt2img for the contract).
            from core.vram_optimization import move_text_encoders_to_cpu, move_unet_to_cpu, move_vae_to_cpu
            if not _kh_gen_succeeded:
                clear_resident(self)
                move_text_encoders_to_cpu(pipeline_to_use)
                move_unet_to_cpu(pipeline_to_use)
                move_vae_to_cpu(pipeline_to_use)
            else:
                # Non-kept components are dropped from the resident set (see the
                # txt2img finally for why) as well as offloaded.
                if _kh_keep_te:
                    mark_resident(self, "text_encoder", _kh_model_key)
                else:
                    move_text_encoders_to_cpu(pipeline_to_use)
                    discard_resident(self, "text_encoder")
                if _kh_keep_unet:
                    mark_resident(self, "unet", _kh_model_key)
                else:
                    move_unet_to_cpu(pipeline_to_use)
                    discard_resident(self, "unet")
                if _kh_keep_vae:
                    mark_resident(self, "vae", _kh_model_key)
                else:
                    move_vae_to_cpu(pipeline_to_use)
                    discard_resident(self, "vae")

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

    def _outpaint_controlnet_conditioning_channels(self, model_path: str) -> int:
        """Return the ControlNet's conditioning-input channel count (3 for a
        normal RGB/edge ControlNet, 4 for an outpaint-native crop+mask model).

        A trained outpaint CN is a diffusers DIRECTORY whose config.json records
        conditioning_channels=4; anything else (single-file edge CN, unresolvable
        path) is treated as 3. Best-effort, never raises -- used only to give a
        clear error before the cryptic conv2d channel-mismatch."""
        try:
            from core.extensions.controlnet_manager import controlnet_manager
            resolved = controlnet_manager._resolve_controlnet_path(model_path)
            if resolved is not None and resolved.is_dir():
                cfg = resolved / "config.json"
                if cfg.exists():
                    import json as _json
                    with open(cfg, "r", encoding="utf-8") as _f:
                        return int(_json.load(_f).get("conditioning_channels", 3) or 3)
        except Exception:
            pass
        return 3

    def _outpaint_controlnet_model_is_lllite(self, model_path: str) -> bool:
        """Best-effort LLLite check for Outpaint ControlNet's OWN configured
        model (see the mutual-exclusion note in generate_outpaint). Reuses
        controlnet_manager.is_lllite_model (a cheap safetensors-header peek in
        the common case); never raises -- an unresolvable/invalid path is
        treated as "not LLLite" so it falls through to the normal load path,
        which will surface its own error."""
        try:
            from core.extensions.controlnet_manager import controlnet_manager
            return bool(controlnet_manager.is_lllite_model(model_path))
        except Exception:
            return False

    def generate_outpaint(
        self,
        params: Dict[str, Any],
        input_image: Image.Image,
        progress_callback=None,
        step_callback=None,
    ) -> tuple[Image.Image, int, int]:
        """Generate an outpainted image: place ``input_image`` inside a larger
        canvas and generate everything outside it, preserving the placed
        region byte-exact.

        Pure orchestration -- no new sampling loop, no arch-specific code.
        Builds the enlarged canvas + an outward-only-blurred mask (see
        ``core.inference.outpaint_utils``), delegates to the existing
        all-architecture ``generate_inpaint``, then performs an UNCONDITIONAL
        final pixel paste of the placed rectangle. That paste -- not
        generate_inpaint's own (gated/latent-only) compositing -- is the
        strict-preservation guarantee, so it holds regardless of loaded
        architecture or denoising_strength.

        The canvas is built 16-aligned (not 8): 7 of 9 image architectures
        re-round their working resolution to their own 16px grid internally
        (FLUX.2/Anima floor down, Lens nearest-16, Ideogram4/MiniT2I round to
        16, Krea2 rounds up), so an only-8-aligned canvas can come back from
        generate_inpaint at a DIFFERENT size, silently misaligning (or
        clipping) the preserved rect. 16-alignment is a fixed point for
        every architecture's grid. ``reconcile_and_paste`` is the defensive
        second half of this fix -- it re-squares the result to the canvas
        size before pasting, in case some arch still returns a different
        size despite the 16-aligned request.

        Returns:
            tuple: (image, actual_seed, actual_ancestral_seed) -- same shape
            as generate_inpaint's return contract.
        """
        from core.inference.outpaint_utils import (
            build_outpaint_canvas,
            build_outpaint_mask,
            reconcile_and_paste,
        )

        canvas_img, placed_img, rect = build_outpaint_canvas(input_image, params, align=16)
        mask_blur = params.get("mask_blur", 4)
        # Boundary Determinism Relaxation owns the seam transition (a keep-side,
        # latent-space, scheduled soft-pin) -- so when it is active, bypass the
        # legacy OUTWARD FILL-BLEND (which blends generated content toward the
        # synthetic replicate/reflect fill outside the rect and is the source of
        # the "bleed band"). Use a HARD outpaint mask instead. The default
        # mask_blur (4) and the legacy fill-blend path are UNCHANGED when
        # boundary relaxation is off (byte-identical). See
        # scratchpad/boundary_relaxation_synthesis.md Q2.
        _bdr_hard_mask = bool(
            params.get("boundary_relax_strength", 0.0)
            and float(params.get("boundary_relax_strength", 0.0)) > 0.0
        )
        # Trained crop_mask ControlNet: the CN was trained (loss weight 1.0, cond
        # mask channel 0, no residual gate) that the first generate-side cell is
        # 100% its to render. A soft inference mask (default mask_blur=4) instead
        # overwrites ~45% of that first latent cell / first ~8 px with
        # encode(replicate fill) via the x0-projection AND the post-decode pixel
        # blend -- injecting the "bleed band" into the exact band the CN owns, so
        # the seam can never be model-determined. Force the hard mask here too, so
        # the mask_latent is binary and the CN's rendering survives untouched. See
        # scratchpad/outpaint_vae_boundary_alignment.md Q2.
        _crop_mask_cn = (
            bool(params.get("outpaint_controlnet_enable", False))
            and str(params.get("outpaint_controlnet_mode", "edge_extrapolate")) == "crop_mask"
        )
        if _bdr_hard_mask or _crop_mask_cn:
            if mask_blur and mask_blur > 0:
                from api.generation_status import add_warning as _bdr_add_warning
                _bdr_add_warning(
                    "Trained crop_mask ControlNet active: using a hard outpaint mask so the "
                    "seam band is fully model-determined; the legacy outward fill-blend "
                    "(mask_blur) is bypassed."
                    if _crop_mask_cn and not _bdr_hard_mask else
                    "Boundary relaxation active: using a hard outpaint mask; the legacy outward "
                    "fill-blend (mask_blur) is bypassed for the seam transition.",
                    code="outpaint_crop_mask_hard_mask" if _crop_mask_cn and not _bdr_hard_mask
                    else "boundary_relax_hard_mask",
                )
            mask_blur = 0
        mask_img = build_outpaint_mask(canvas_img.size, rect, mask_blur)

        # The canvas IS the actual output size for this generation -- must
        # match canvas_img.size exactly, otherwise generate_inpaint's own
        # width/height resize (see custom_sampling target_width/target_height)
        # would resize our carefully-built canvas and destroy `rect`'s
        # correspondence to the preserved content. This mutates the caller's
        # `params` (not a copy) -- routes.py relies on params["width"]/["height"]
        # reflecting the resolved canvas size after this call returns.
        params["width"], params["height"] = canvas_img.size

        # From here on, work on a COPY of params for the internal-only
        # noise-init / mask-blur adjustments below -- these must NOT leak
        # into the caller's `params` (which is persisted to the DB and PNG
        # metadata as the user's requested parameters).
        work = dict(params)

        # Noise-init gate: at the default (and any >=1.0) denoising_strength,
        # the GENERATE region is initialized from pure architecture-native
        # noise -- independent of the canvas fill -- instead of a noised
        # encode(fill), which otherwise leaves a visible extended-edge
        # artifact in the generated region and drives the ~25% exposure
        # mismatch at the rect boundary (see
        # core.inference.outpaint_utils.compose_outpaint_start and
        # custom_inpaint_sampling_loop's outpaint_noise_init kwarg). Below
        # 1.0 the user is deliberately requesting the legacy SDEdit-from-fill
        # behavior ("guided outpaint"), so noise-init stays off and a warning
        # is recorded.
        requested_strength = float(work.get("denoising_strength", 1.0))
        # Internal-only bookkeeping (underscore-prefixed -- not a user param,
        # not in param_defaults.py/openapi.yaml): preserves the user's
        # ORIGINAL requested strength for any downstream diagnostics, since
        # `work["denoising_strength"]` itself is overwritten to 1.0 below
        # when noise-init is active.
        work["_outpaint_requested_strength"] = requested_strength
        work["_outpaint_noise_init"] = requested_strength >= 1.0
        if work["_outpaint_noise_init"]:
            work["denoising_strength"] = 1.0
        else:
            try:
                from api.generation_status import add_warning
                add_warning(
                    "Outpaint denoising_strength < 1.0 uses the legacy "
                    "guided-from-fill mode: the generated region is denoised "
                    "starting from the canvas fill instead of pure noise, so "
                    "it stays influenced by the extended-edge fill content",
                    code="outpaint_guided_lowstrength",
                )
            except Exception:
                pass

        # The outpaint mask (build_outpaint_mask, above) already has its blur
        # baked in AND is hard-clamped to 0 over the preserved rect. Several
        # backends re-Gaussian-blur the incoming mask using `mask_blur` --
        # doing that again here would reintroduce nonzero mask weight INSIDE
        # the rect, breaking the outward-only-blur contract. Force it off for
        # the delegated generate_inpaint call only; the real blur radius
        # (`mask_blur`, above) is still used for `build_outpaint_mask` and for
        # the exposure harmonizer's transition-band skip below.
        work["mask_blur"] = 0

        # ============================================================
        # OUTPAINT-CONTROLNET (PART A -- edge-extrapolation ControlNet;
        # scratchpad/outpaint_controlnet_synthesis.md). Detects structures in
        # the PRESERVED region that cross the rect boundary and extrapolates
        # them a short, confidence-tapered distance into the generate region
        # as a synthetic ControlNet control image, injected into
        # work["controlnet_images"] exactly like a real user ControlNet
        # entry. The same confidence field is ALSO threaded through as
        # work["outpaint_controlnet_gate"] -- a per-residual spatial gate
        # applied in the shared ControlNet block of custom_inpaint_sampling_
        # loop (RESIDUAL MASKING), so the ControlNet's influence tapers to 0
        # with distance from the boundary instead of applying at a flat
        # conditioning_scale everywhere.
        #
        # v1 scope: SD/SDXL only; mutually exclusive with a user-supplied
        # ControlNet/LLLite (never overrides the user's own request -- see
        # the _ocn_user_cn check below); forces the byte-exact boundary paste
        # variant (BDR's "feather" leaves a ~24px non-exact strip, which
        # would defeat this feature's own boundary-geometry enforcement);
        # disables Seam Structure Continuity (both mechanisms extrapolate the
        # same boundary-crossing structures into the generate region --
        # running both would double-enforce the same geometry).
        #
        # Fully gated on outpaint_controlnet_enable: when False (default),
        # none of this runs, work["outpaint_controlnet_gate"] stays unset
        # (generate_inpaint's params.get(...) below returns None), and the
        # whole ControlNet/inpaint path is byte-identical to before this
        # feature existed.
        # ============================================================
        if work.get("outpaint_controlnet_enable", False):
            from api.generation_status import add_warning as _ocn_warn

            _ocn_sd_family = self.current_pipeline_kind in ("sd15", "sdxl")
            _ocn_user_cn = [
                c for c in (work.get("controlnet_images") or [])
                if not c.get("is_reference_guide")
            ]

            if not _ocn_sd_family:
                _ocn_warn(
                    "Outpaint ControlNet (edge extrapolation) is implemented for SD/SDXL "
                    "only; skipped on the currently loaded architecture.",
                    code="outpaint_controlnet_arch_unsupported",
                )
            elif _ocn_user_cn:
                _ocn_warn(
                    "Outpaint ControlNet (edge extrapolation) was skipped: a user-supplied "
                    "ControlNet/LLLite is already active for this generation, and this "
                    "feature never overrides a user's own ControlNet request.",
                    code="outpaint_controlnet_user_cn_conflict",
                )
            elif not work.get("outpaint_controlnet_model"):
                _ocn_warn(
                    "Outpaint ControlNet (edge extrapolation) was skipped: no ControlNet "
                    "model path was provided.",
                    code="outpaint_controlnet_no_model",
                )
            elif self._outpaint_controlnet_model_is_lllite(work["outpaint_controlnet_model"]):
                # LLLite ControlNets are applied directly to the U-Net's attention
                # layers and never return down/mid residuals -- there is nothing
                # for RESIDUAL MASKING to gate, so this feature cannot drive one.
                _ocn_warn(
                    "Outpaint ControlNet (edge extrapolation) was skipped: the configured "
                    "ControlNet model is an LLLite model, which does not produce the "
                    "residuals this feature masks.",
                    code="outpaint_controlnet_lllite_unsupported",
                )
            else:
                _ocn_mode = str(work.get("outpaint_controlnet_mode", "edge_extrapolate"))
                # Fail loud on a channel/mode mismatch BEFORE the ControlNet forward
                # (a 4-ch crop-mask model fed a 3-ch edge image, or vice-versa,
                # otherwise crashes deep inside conv2d with an opaque "expected
                # input to have 4 channels, but got 3" message).
                _ocn_ch = self._outpaint_controlnet_conditioning_channels(work["outpaint_controlnet_model"])
                if _ocn_ch == 4 and _ocn_mode != "crop_mask":
                    raise ValueError(
                        f"Outpaint ControlNet model '{work['outpaint_controlnet_model']}' is a "
                        f"4-channel outpaint-trained ControlNet, which requires Mode = "
                        f"'crop_mask'. The current Mode is '{_ocn_mode}' (edge extrapolation "
                        f"builds a 3-channel control image). Switch the Outpaint ControlNet "
                        f"Mode to 'Crop mask (trained outpaint CN)'."
                    )
                if _ocn_ch != 4 and _ocn_mode == "crop_mask":
                    raise ValueError(
                        f"Outpaint ControlNet Mode is 'crop_mask' (4-channel crop+mask "
                        f"conditioning) but the selected model "
                        f"'{work['outpaint_controlnet_model']}' is a 3-channel ControlNet. "
                        f"Use a ControlNet trained with conditioning_mode='outpaint' (a "
                        f"diffusers directory with conditioning_channels=4), or switch Mode "
                        f"to 'Edge extrapolate'."
                    )
                if _ocn_mode == "crop_mask":
                    # PART B: trained outpaint-native 4-channel conditioning (crop RGB
                    # + binary known-mask), the EXACT format the ControlNet was trained
                    # on (core.utils.crop_mask_condition, shared with training -> no
                    # skew). The net LEARNED the continuation, so there is no edge
                    # extrapolation / termination heuristic; the gate is flat 1.0 over
                    # the whole generate region (no distance taper). Requires a
                    # ControlNet trained with conditioning_mode="outpaint" (4-ch); a
                    # 3-ch model will raise a channel mismatch at the ControlNet forward.
                    from core.utils.crop_mask_condition import build_crop_mask_condition
                    import numpy as _np
                    # R1 (scratchpad/outpaint_boundary_structure_fix.md D3-R1): a
                    # FIXED inward feather -- see param_defaults.py OUTPAINT_DEFAULTS
                    # outpaint_controlnet_edge_feather_px for the no-skew rationale.
                    # The fallback is single-sourced from OUTPAINT_DEFAULTS (default
                    # 0.0 = razor-sharp, matching the current live pre-R1 CN); a
                    # request may override it, and it flips to the training-range
                    # midpoint only once an R1-retrained soft-edge CN is live.
                    from api.param_defaults import OUTPAINT_DEFAULTS as _OUTPAINT_DEFAULTS
                    _ocn_edge_feather_px = float(work.get(
                        "outpaint_controlnet_edge_feather_px",
                        _OUTPAINT_DEFAULTS["outpaint_controlnet_edge_feather_px"],
                    ))
                    # Feature #3a (secondary, independent lever to Feature #2 below):
                    # opt-in rounded-corner CN conditioning geometry. 0.0 default =
                    # byte-identical (see crop_mask_condition.py docstring).
                    _ocn_corner_radius_px = float(work.get(
                        "outpaint_controlnet_corner_radius_px",
                        _OUTPAINT_DEFAULTS["outpaint_controlnet_corner_radius_px"],
                    ))
                    _cond_np, _gate_np = build_crop_mask_condition(
                        _np.array(canvas_img.convert("RGB")), rect, canvas_img.size,
                        edge_feather_px=_ocn_edge_feather_px,
                        corner_radius_px=_ocn_corner_radius_px,
                    )
                    _ocn_result = (_cond_np, _gate_np)
                else:
                    from core.inference.outpaint_control import build_outpaint_control_image
                    _ocn_result = build_outpaint_control_image(
                        placed_img, rect, canvas_img.size,
                        detector=work.get("outpaint_controlnet_detector", "canny"),
                        depth_px=int(work.get("outpaint_controlnet_depth", 160)),
                        taper_power=float(work.get("outpaint_controlnet_taper", 2.0)),
                    )
                if _ocn_result is None:
                    _ocn_warn(
                        "Outpaint ControlNet (edge extrapolation): no eligible "
                        "boundary-crossing structure was found in the preserved region; "
                        "skipped.",
                        code="outpaint_controlnet_no_crossings",
                    )
                else:
                    _ocn_control_img, _ocn_gate = _ocn_result
                    _ocn_scale = float(work.get("outpaint_controlnet_scale", 0.6))
                    _ocn_gstart = float(work.get("outpaint_controlnet_guidance_start", 0.0))
                    _ocn_gend = float(work.get("outpaint_controlnet_guidance_end", 0.55))
                    # Match the exact ControlNet config dict keys _apply_controlnets +
                    # generate_inpaint's controlnet_kwargs builder read (model_path,
                    # image, strength, start_step/end_step in 0-1000 units, etc.) --
                    # see backend/api/generation_utils.py's process_controlnet_configs,
                    # the schema this dict must match.
                    work["controlnet_images"] = list(work.get("controlnet_images") or []) + [{
                        "model_path": work["outpaint_controlnet_model"],
                        "image": _ocn_control_img,
                        "strength": _ocn_scale,
                        "start_step": _ocn_gstart * 1000.0,
                        "end_step": _ocn_gend * 1000.0,
                        "layer_weights": None,
                        "prompt": None,
                        "is_lllite": False,
                        "is_reference_guide": False,
                    }]
                    # Residual gate: edge-extrapolation NEEDS it (residual masking
                    # IS its mechanism -- confine the untrained edge CN to the
                    # generate region). The trained crop_mask CN was trained
                    # GATELESS (residuals applied over the full field, keep region
                    # only loss-down-weighted 0.3), so a keep/gen gate at inference
                    # would feed the UNet a residual field it never saw in training
                    # (keep side zeroed + fractional coarse-block cells at the
                    # seam). By default the gate stays UNSET (None -> inert) for
                    # crop_mask to match training; the keep region stays exact via
                    # the mask_latent x0-pin + the final byte-exact paste. See
                    # outpaint_vae_boundary_alignment.md Q2b. The ONE opt-in
                    # exception is Feature #2 below (per-corner residual gate),
                    # which is a 1.0-base field with only local corner dips, not a
                    # keep/gen split -- it stays inert (all-1.0, byte-identical) at
                    # its own defaults.
                    if _ocn_mode != "crop_mask":
                        work["outpaint_controlnet_gate"] = _ocn_gate
                    else:
                        # Feature #2 (PRIMARY corner-seam fix; H1 vertex-feature-lock,
                        # scratchpad/outpaint_seam_diagnosis.md): attenuate the CN
                        # residual ONLY in small disks at the 4 rect vertices -- NOT
                        # the flat keep/gen gate above (_ocn_gate, unused here), which
                        # would zero the whole keep-side residual (strong OOD vs.
                        # training's gateless flat-1.0 field). Default radius=0.0 /
                        # g_min=1.0 leaves the gate UNSET (None -> inert), exactly as
                        # before this feature existed.
                        _ocn_corner_gate_radius_px = float(work.get(
                            "outpaint_controlnet_corner_gate_radius_px",
                            _OUTPAINT_DEFAULTS["outpaint_controlnet_corner_gate_radius_px"],
                        ))
                        _ocn_corner_gate_min = float(work.get(
                            "outpaint_controlnet_corner_gate_min",
                            _OUTPAINT_DEFAULTS["outpaint_controlnet_corner_gate_min"],
                        ))
                        if _ocn_corner_gate_radius_px > 0.0 and _ocn_corner_gate_min < 1.0:
                            from core.utils.outpaint_corner_gate import build_corner_gate
                            work["outpaint_controlnet_gate"] = build_corner_gate(
                                rect, canvas_img.size,
                                _ocn_corner_gate_radius_px, _ocn_corner_gate_min,
                            )

                        # L1 four-corner x0-pin softening (independent of the CN
                        # residual gate above -- this touches the per-step x0-pin
                        # composite in custom_inpaint_sampling_loop, NOT the CN
                        # residual). Root cause: image_latents is fixed across all
                        # steps, so the hard rectangular mask_latent re-stamps the
                        # same re-entrant 90-degree seed at each rect vertex every
                        # step; the CN's structure-completion regime extends that
                        # seed outward and the pin re-enforces it each step (a
                        # feedback loop). This relaxes the pin's keep-weight to
                        # outpaint_pin_corner_relax_min (instead of the full 1.0)
                        # in small radius-px disks at the 4 rect vertices only, so
                        # the seed seam is blunted while the straight edges (away
                        # from corners) keep the full pin. The preserved rect
                        # stays byte-exact regardless: the FINAL byte-exact paste
                        # (see boundary_relax_paste="exact" above) restores its
                        # pixels from the untouched input, so relaxing only the
                        # intermediate latent trajectory near the corners cannot
                        # leak into the final output. Default radius=0.0 / min=1.0
                        # leaves work["outpaint_pin_corner_relax"] UNSET (None ->
                        # inert), exactly as before this feature existed.
                        _ocn_pin_relax_radius_px = float(work.get(
                            "outpaint_pin_corner_relax_radius_px",
                            _OUTPAINT_DEFAULTS["outpaint_pin_corner_relax_radius_px"],
                        ))
                        _ocn_pin_relax_min = float(work.get(
                            "outpaint_pin_corner_relax_min",
                            _OUTPAINT_DEFAULTS["outpaint_pin_corner_relax_min"],
                        ))
                        if _ocn_pin_relax_radius_px > 0.0 and _ocn_pin_relax_min < 1.0:
                            from core.utils.outpaint_corner_gate import build_corner_gate
                            work["outpaint_pin_corner_relax"] = build_corner_gate(
                                rect, canvas_img.size,
                                _ocn_pin_relax_radius_px, _ocn_pin_relax_min,
                            )

                    if str(work.get("boundary_relax_paste", "feather")) != "exact":
                        work["boundary_relax_paste"] = "exact"

                    if float(work.get("seam_structure_strength", 0.0) or 0.0) > 0.0:
                        work["seam_structure_strength"] = 0.0
                        _ocn_warn(
                            "Outpaint ControlNet (edge extrapolation) disabled Seam "
                            "Structure Continuity (seam_structure_strength) to avoid "
                            "double geometry enforcement at the boundary.",
                            code="outpaint_controlnet_ssc_disabled",
                        )

                    _ocn_warn(
                        (
                            "Outpaint ControlNet (crop_mask, trained) is active: driving a "
                            "trained outpaint-native ControlNet from the 4-channel crop+mask "
                            "conditioning."
                            if _ocn_mode == "crop_mask" else
                            "Outpaint ControlNet (edge extrapolation) is active: extrapolating "
                            "structures crossing the preserved boundary into the generate "
                            "region."
                        ),
                        code="outpaint_controlnet_active",
                    )

        result_image, actual_seed, actual_ancestral_seed = self.generate_inpaint(
            work, canvas_img, mask_img,
            progress_callback=progress_callback, step_callback=step_callback,
        )

        # BDR Variant B (feather): when boundary relaxation is active AND the
        # paste mode is "feather", erode/feather a thin strip at the rect's
        # generate-adjacent edges so the model's bridged seam rendering survives
        # instead of the exact input (the interior stays byte-exact). See
        # scratchpad/boundary_relaxation_synthesis.md Q3 variant B.
        # NOTE: reads `work` (not the caller's `params`) for boundary_relax_paste
        # specifically, since Outpaint ControlNet (above) forces work["boundary_
        # relax_paste"] = "exact" without leaking that override into `params`
        # (which is persisted to the DB/PNG metadata as the user's requested
        # parameters). Byte-identical to reading `params` whenever that override
        # never fired -- `work` starts as an unmodified copy of `params` for this
        # key otherwise.
        _paste_alpha = None
        # Option E band width (px). Computed here (before the BDR branch) because
        # when it is active it OVERRIDES any BDR paste_alpha in
        # reconcile_and_paste -- so we must skip building the BDR alpha AND its
        # warning below, otherwise the user sees a BDR "~24 px model-rendered"
        # notice for a strip that is actually discarded in favour of Option E's
        # N-px band (different width and semantics).
        _paste_feather_px = int(params.get("outpaint_paste_feather_px", 0) or 0)
        _bdr_on = float(params.get("boundary_relax_strength", 0.0) or 0.0) > 0.0
        if _bdr_on and _paste_feather_px <= 0 and str(work.get("boundary_relax_paste", "feather")) == "feather":
            from core.inference.outpaint_utils import build_paste_alpha
            # erosion/feather in pixels: tie to the soft strip (W_soft=2 latent
            # cells -> ~16 px) plus an 8 px feather. VAE scale factor 8.
            _erode_px = 8.0 * 2.0
            _feather_px = 8.0
            _paste_alpha = build_paste_alpha(rect, canvas_img.size, _erode_px, _feather_px)
            from api.generation_status import add_warning as _bdr_fx_warn
            _bdr_fx_warn(
                "Boundary relaxation feather paste: a thin strip (~24 px) inside the placed rect at "
                "generated-adjacent edges is model-rendered, not byte-identical to the input; the "
                "interior beyond the strip is byte-exact.",
                code="boundary_relax_feather_nonexact",
            )

        # Option E (paste-band reconciliation feather, scratchpad/
        # outpaint_seam_latent_stage.md section 4.1): reads `params` (NOT
        # `work`), so the CN "force boundary_relax_paste=exact" guard above
        # (which only ever mutates `work["boundary_relax_paste"]`) has no
        # effect on this independent parameter -- it applies to crop_mask CN
        # runs too. `reconcile_and_paste` itself gives this precedence over
        # any BDR `paste_alpha` when both are set (see its docstring).
        # `_paste_feather_px` was resolved above (before the BDR branch).
        if _paste_feather_px > 0:
            from api.generation_status import add_warning as _pf_warn
            _pf_warn(
                "Paste-band reconciliation feather: the last "
                f"{_paste_feather_px} row(s)/column(s) of the preserved rect at its "
                "generate-adjacent edges are blended (raised-cosine) toward the "
                "decoded canvas underneath instead of pasted byte-exact; the rest of "
                "the preserved rect is unaffected.",
                code="outpaint_paste_feather_nonexact",
            )

        # G_prop16 boundary-offset propagation (core.inference.seam_membrane.
        # apply_seam_offset_propagation, opt-in): generated-side-only by
        # construction -- see param_defaults.py OUTPAINT_DEFAULTS and
        # scratchpad/outpaint_seamless_vae_native.md for the mechanism.
        _seam_offset_prop = float(params.get("outpaint_seam_offset_prop", 0.0) or 0.0)
        if _seam_offset_prop > 0:
            from api.generation_status import add_warning as _sop_warn
            _sop_warn(
                "Seam offset propagation (G_prop16) modifies generated-side pixels "
                "near the seam to match the preserved boundary's own tone; the "
                "preserved region itself is unchanged.",
                code="outpaint_seam_offset_prop_engaged",
            )

        def _seam_membrane_warn(message: str, code: str) -> None:
            from api.generation_status import add_warning as _sm_warn
            _sm_warn(message, code=code)

        # Preserved-region compositing mode (opt-in, default "exact" = the
        # unchanged byte-exact paste; see param_defaults.py OUTPAINT_DEFAULTS
        # for the full mode descriptions and scratchpad/vae_native_ab for the
        # validated recipe this trades against). Read from `params` (not
        # `work`) -- nothing above mutates this key, mirroring how
        # `_paste_feather_px` is resolved.
        _preserve_mode = str(params.get("outpaint_preserve_mode", "exact") or "exact")
        if _preserve_mode != "exact":
            from api.generation_status import add_warning as _pm_warn
            _sd_family = self.current_pipeline_kind in ("sd15", "sdxl")
            if _preserve_mode == "vae_reconstruct_hf" and not _sd_family:
                _pm_warn(
                    "outpaint_preserve_mode='vae_reconstruct_hf' is implemented for "
                    "SD1.5/SDXL only; the currently loaded architecture falls back to "
                    "'vae_reconstruct' behavior (no high-frequency detail restoration).",
                    code="outpaint_preserve_mode_hf_unsupported_arch",
                )
            _pm_warn(
                "outpaint_preserve_mode is not 'exact': the preserved region is a "
                + (
                    "VAE reconstruction of the input"
                    if _preserve_mode == "vae_reconstruct" or (_preserve_mode == "vae_reconstruct_hf" and not _sd_family)
                    else "VAE reconstruction of the input with its high-frequency detail restored"
                )
                + ", NOT byte-identical to it -- this trades exact preservation for a "
                "seamless boundary (no hard raw/decoded pixel discontinuity).",
                code="outpaint_preserve_mode_nonexact",
            )

        result_image = reconcile_and_paste(
            result_image, placed_img, rect, canvas_img.size,
            mask_blur=mask_blur,
            outpaint_seam_fix=bool(params.get("outpaint_seam_fix", True)),
            paste_alpha=_paste_alpha,
            paste_feather_px=float(_paste_feather_px),
            seam_membrane=bool(params.get("outpaint_seam_membrane", False)),
            seam_membrane_band=int(params.get("outpaint_seam_membrane_band", 0) or 0),
            seam_tone_strength=float(params.get("outpaint_seam_tone_strength", 0.0) or 0.0),
            seam_tone_band=int(params.get("outpaint_seam_tone_band", 0) or 0),
            seam_offset_prop=_seam_offset_prop,
            outpaint_preserve_mode=_preserve_mode,
            warn_callback=_seam_membrane_warn,
        )

        return result_image, actual_seed, actual_ancestral_seed

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
