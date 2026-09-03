"""LTX-2.3 video backend mixin for DiffusionPipelineManager.

Holds the text-to-video generation entry point for the LTX-2.3 joint
audio+video MM-DiT. The assembled `LTX2Pipeline` is stored in
`self.ltx2_components["pipeline"]` by the loader (P1a). This mixin stages the
components onto the GPU via `enable_model_cpu_offload` and drives the pipeline's
`__call__` to produce a video (and optional audio) tensor.

Output contract of `_generate_txt2vid_ltx2`:
    (frames, audio, audio_sample_rate, actual_seed)
where:
    frames  = np.uint8 array of shape [T, H, W, 3] (RGB, value range 0-255).
              Derived from the pipeline's `output_type="np"` result
              (shape [B, T, H, W, C], float in [0, 1]); batch index 0 is taken.
    audio   = torch.FloatTensor of shape [channels, samples] on CPU, or None
              when audio is disabled / unavailable. From `audio[0]`.
    audio_sample_rate = int (vocoder output sampling rate, typically 24000) or None.
    actual_seed = the concrete seed used (random draw resolved when seed < 0).
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
import os
import random
import weakref

import numpy as np
import torch
from PIL import Image


# The preserved-span preprocessing is shared with every other video backend's
# temporal outpaint (MiniMax-H3's, today) and therefore lives in the
# architecture-neutral `outpaint_utils` next to the image-outpaint convention it
# mirrors. Imported under its historical private name so this module's call
# sites and comments still read the same.
from core.inference.outpaint_utils import center_crop_resize_frames as _center_crop_resize_frames


def _snap_offset_to_latent_index(
    offset_frames: int, frame_scale_factor: int, latent_num_frames: int
) -> Tuple[int, int]:
    """Resolve a desired pixel-frame offset to the nearest VALID latent frame
    index (and its corresponding pixel start), per `LTX2VideoCondition.index`'s
    contract (`preprocess_conditions`, pipeline_ltx2_condition.py:732-746):
    valid latent indices L map to pixel starts {0 (L=0), 1, 1+scale, 1+2*scale,
    ...} i.e. `start_idx = max((L - 1) * scale + 1, 0)`.

    Returns:
        (latent_index, pixel_start), with `latent_index` clamped to
        [0, latent_num_frames - 1].
    """
    offset_frames = max(0, int(offset_frames))
    max_index = max(0, latent_num_frames - 1)
    if offset_frames <= 0 or max_index <= 0:
        return 0, 0

    # Invert pixel_start(L) = (L - 1) * scale + 1  =>  L = (offset - 1) / scale + 1.
    candidate = int(round((offset_frames - 1) / frame_scale_factor)) + 1
    candidate = max(1, min(candidate, max_index))
    candidate_pixel_start = (candidate - 1) * frame_scale_factor + 1

    dist_candidate = abs(candidate_pixel_start - offset_frames)
    dist_zero = offset_frames  # abs(0 - offset_frames)
    if dist_candidate <= dist_zero:
        return candidate, candidate_pixel_start
    return 0, 0


def _trim_conditioning_sequence_frames(
    start_frame: int, sequence_num_frames: int, target_num_frames: int, scale_factor: int
) -> int:
    """Pure re-implementation of `LTX2ConditionPipeline.trim_conditioning_sequence`
    (pipeline_ltx2_condition.py:657-672), computed locally (rather than via a
    pipeline instance) so the placement/paste-span math doesn't require the
    condition pipeline to be built (and block-swap-wrapped) before the
    fits-in-timeline check is known.
    """
    num_frames = min(sequence_num_frames, target_num_frames - start_frame)
    if num_frames <= 0:
        return 0
    num_frames = (num_frames - 1) // scale_factor * scale_factor + 1
    return max(0, num_frames)


def _ltx2_invalidate_block_swap_caches(offloader) -> None:
    """Drop every offloader cache that describes the PREVIOUS block tree.

    LTX-2.3's offloader is persistent across generations (torn down only by a
    change of ``blocks_to_swap``) while a LoRA wrap/unwrap changes the tree under
    it, and BOTH swap paths cache a description of that tree: the H2D masters
    hold module references and a flat layout, and the standard swap's staging
    buffers are sized from the FIRST job list and then ``zip``-paired with every
    later one -- a wrap adds two Linears per target and renames the base to
    ``.original_module``, so a stale list mispairs a buffer with a differently
    shaped tensor and drops every job past its end.

    Cost: rebuilding the H2D masters re-pins one flat CPU buffer per swappable
    block while the old ones are still referenced by the weights, i.e. a
    transient ~2x pinned host footprint on the first forward afterwards.
    """
    for attr in ("h2d_masters", "h2d_ring", "h2d_slot_futures", "h2d_loaded_block",
                 "staging_buffer_a", "staging_buffer_b", "pinned_buffer",
                 "_dtype_split_paths"):
        if hasattr(offloader, attr):
            setattr(offloader, attr, None)


class LTX2Mixin:
    """LTX2Mixin: LTX-2.3 text-to-video generation backend."""

    def _ltx2_runtime_int8(self, params: Dict[str, Any], progress_callback=None,
                           force_wrap: bool = False):
        """Apply the one-time in-place INT8 conversion, if this request asks for it.

        No-op for every ``unet_quantization`` value other than ``"int8"``, for an
        already-converted transformer, and for a checkpoint that already carries
        weight-only quantized Linears (see
        ``vram_optimization.apply_runtime_int8_quantization`` for the full
        contract). The conversion replaces child modules in place and never
        builds a second module -- an 18.98 G-parameter DiT would not tolerate the
        ``copy.deepcopy`` the legacy FP8 path uses.

        SCOPE. Only ``LTX2VideoTransformer3DModel`` is converted. The Gemma-3
        text encoder and the text connectors are separate components and are not
        reached by this walk; ``arch_capabilities`` declares
        ``text_encoder_quantization`` unsupported for ltx2 and nothing here
        changes that.

        ORDERING, load-bearing rather than incidental, and the reason this is
        called at the TOP of every generate path:

        * BEFORE ``_ensure_ltx2_swap_and_offload``. In block-swap mode that
          builds a ``TransformerBlockOffloader`` over
          ``transformer.transformer_blocks``, which captures each block's Linear
          modules and pinned CPU masters; converting afterwards would replace the
          modules it holds and leave it streaming pre-conversion bf16 weights
          into modules nothing reads. Checked and raised as a ``RuntimeError``,
          not asserted -- ``python -O`` strips an assert, and this is the one
          violation that is invisible at runtime.

          The check is handed to the converter as its ``precheck`` rather than
          run here, and that placement is the whole point. LTX-2.3's offloader is
          PERSISTENT state on ``Ltx2BlockLoopWrapper``: it is created once
          (``_ensure_ltx2_block_swap_wrapper``) and torn down only by that same
          function on the NEXT generation, i.e. strictly after this runs. FLUX.2's
          look-alike guard is safe where this one is not, because
          ``_flux2_active_block_offloader`` is cleared in a ``finally`` at the end
          of every generation. Raised before the request was consulted, this guard
          fired on the second block-swap generation of a session with
          ``unet_quantization`` unset -- block swap being the standard mode for a
          37 GB model, that broke the primary workflow for users who never enable
          INT8 at all. Inside ``precheck`` it can only fire when a conversion is
          genuinely about to touch the first layer.

          The refusal's ADVICE is made true rather than merely printed. A
          session that generated once with ``blocks_to_swap=22`` leaves the
          offloader live, so an INT8 request made afterwards WITH
          ``blocks_to_swap=0`` -- exactly what the message tells the user to do
          -- would have hit the refusal, microseconds before
          ``_ensure_ltx2_swap_and_offload(0)`` tore that same wrapper down. When
          the incoming request asks for no block swap (and no ``force_wrap``,
          i.e. no FBCache/Spectrum/style wrap needs the block loop) the unwrap is
          simply performed FIRST, here, which is the same call the generate path
          makes a few lines later and in the same order (unwrap, then offload).
          The refusal then fires only where it is unavoidable: an INT8 request
          that also asks for block swap in the same generation, where the two
          orderings genuinely conflict.
        * BEFORE staging. The converter is device-aware, so running here (with
          the components still on CPU, or wherever the previous generation left
          them) is correct either way; running it after the accelerate
          ``enable_model_cpu_offload`` chain has staged the transformer would
          quantize a GPU-resident module for no reason.
        * The wrapper is unwrapped for the walk. A FBCache-/Spectrum-/style-only
          wrapper (``block_offloader=None``) reads ``self.transformer`` and its
          ``transformer_blocks`` at call time and caches no Linear references, so
          converting the inner module underneath it is safe; a wrapper with a
          live offloader is not, and is refused above.
        """
        from core.vram_optimization import apply_runtime_int8_quantization

        components = getattr(self, "ltx2_components", None)
        if not components:
            return
        current = components.get("transformer")
        if current is None:
            return

        from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper
        from core.vram_optimization import runtime_int8_requested

        blocks_to_swap = int(params.get("blocks_to_swap", 0) or 0)
        if (runtime_int8_requested(params.get("unet_quantization"))
                and blocks_to_swap <= 0 and not force_wrap
                and isinstance(current, Ltx2BlockLoopWrapper)
                and current._block_offloader is not None):
            # This request wants no block swap, so the LEFTOVER offloader from an
            # earlier generation is about to be torn down anyway (by
            # _ensure_ltx2_swap_and_offload, below). Do it first so the conversion
            # is legal instead of refused -- see ORDERING above.
            print("[LTX-2.3] INT8 requested with blocks_to_swap=0: tearing down the "
                  "block offloader left by an earlier generation before converting")
            self._ensure_ltx2_block_swap_wrapper(0)
            current = components.get("transformer")
            if current is None:
                return

        wrapper = current if isinstance(current, Ltx2BlockLoopWrapper) else None
        inner = wrapper.transformer if wrapper is not None else current

        def _refuse_if_offloader_live():
            if wrapper is None or getattr(wrapper, "_block_offloader", None) is None:
                return
            raise RuntimeError(
                "LTX-2.3 INT8 conversion was reached while a block offloader is still "
                "active. It must run BEFORE the offloader is created: the offloader holds "
                "references to each transformer block's Linear modules, and the conversion "
                "replaces those modules, so afterwards it would stream the original bf16 "
                "weights into modules nothing reads. Request INT8 in a generation that "
                "does not itself enable block swap (blocks_to_swap=0) -- a stale "
                "offloader left by an EARLIER generation is torn down automatically in "
                "that case -- or on a freshly loaded model.")

        model, _converted = apply_runtime_int8_quantization(
            self, inner, "ltx2", params.get("unet_quantization"),
            label="LTX-2.3 Transformer", progress_callback=progress_callback,
            precheck=_refuse_if_offloader_live)

        # The converter mutates in place and returns the SAME object, so the
        # component/pipeline references stay valid; re-assigned anyway so a future
        # converter that returned a new module could not silently strand them.
        if wrapper is not None:
            wrapper.transformer = model
        else:
            components["transformer"] = model
            # The base pipeline and the two cached derived pipelines share every
            # module rather than owning weights, so all three must point at the
            # converted object (same rule _ensure_ltx2_block_swap_wrapper follows
            # when it wraps/unwraps).
            for key in ("pipeline", "i2v_pipeline", "cond_pipeline"):
                pipe = components.get(key)
                if pipe is not None and getattr(pipe, "transformer", None) is current:
                    pipe.transformer = model

    # ------------------------------------------------------------------
    # Generation-time LoRA
    # ------------------------------------------------------------------

    def _ltx2_lora_transformer(self):
        """The stock ``LTX2VideoTransformer3DModel``, block-loop wrapper peeled off.

        ``Ltx2BlockLoopWrapper.__getattr__`` would forward ``transformer_blocks``
        anyway, but the target iterator REPLACES child modules and must do so on
        the object the wrapper and the block offloader actually hold.
        """
        components = getattr(self, "ltx2_components", None)
        if not components:
            return None
        current = components.get("transformer")
        if current is None:
            return None
        from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper
        return current.transformer if isinstance(current, Ltx2BlockLoopWrapper) else current

    def _ltx2_block_offloader(self):
        """The block offloader driving this generation, or None (no block swap)."""
        components = getattr(self, "ltx2_components", None)
        if not components:
            return None
        from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper
        current = components.get("transformer")
        if not isinstance(current, Ltx2BlockLoopWrapper):
            return None
        return current._block_offloader

    @staticmethod
    def _ltx2_lora_warn(message: str, code: str) -> None:
        print(f"[LTX-2.3 LoRA] WARNING: {message}")
        try:
            from api.generation_status import add_warning
            add_warning(message, code=code)
        except Exception:
            pass

    @property
    def _ltx2_lora_session(self):
        session = getattr(self, "_ltx2_lora_session_instance", None)
        if session is None:
            from core.adapters import AdapterSession
            session = AdapterSession(
                resolve_path=self._ltx2_resolve_lora_path,
                warn=self._ltx2_lora_warn,
                label="LTX-2.3 LoRA",
                message_label="LTX-2.3 LoRA",
                count_declared_branches=self._ltx2_count_declared_branches,
                missing_file=self._ltx2_missing_lora,
                prepare_file=self._ltx2_prepare_lora_file,
                describe_zero_targets=self._ltx2_zero_target_message,
            )
            self._ltx2_lora_session_instance = session
        return session

    @staticmethod
    def _ltx2_resolve_lora_path(raw_path: Any):
        from core.extensions.lora_manager import lora_manager
        return lora_manager._resolve_lora_path(raw_path)

    @staticmethod
    def _ltx2_missing_lora(lora_file: str, raw_path: Any):
        from api.error_handlers import ValidationError
        from core.adapters import AdapterFileMissing

        class Ltx2FileMissing(AdapterFileMissing, ValidationError):
            def __init__(self, message: str, detail: str = None, code: Optional[str] = None):
                code = code or "lora_not_found"
                AdapterFileMissing.__init__(self, message, code=code)
                ValidationError.__init__(self, message, detail=detail, code=code)

        print(f"[LTX-2.3 LoRA] ERROR: {lora_file} not found")
        return Ltx2FileMissing(
            f"LoRA file not found: {lora_file}",
            detail="No such file in the configured LoRA directory or in any additional directory.",
            code="lora_not_found",
        )

    @classmethod
    def _ltx2_count_declared_branches(cls, tensors, _components) -> int:
        from core.models.ltx2.ltx2_lora import normalise_lora_state_dict
        return len(normalise_lora_state_dict(tensors))

    @staticmethod
    def _ltx2_zero_target_message(file, counts):
        from api.error_handlers import ValidationError
        from core.adapters import AdapterIncompatible

        class Ltx2Incompatible(AdapterIncompatible, ValidationError):
            def __init__(self, message: str, detail: str = None, code: Optional[str] = None):
                code = code or "lora_incompatible"
                AdapterIncompatible.__init__(self, message, code=code)
                ValidationError.__init__(self, message, detail=detail, code=code)

        return Ltx2Incompatible(
            f"LoRA '{file.name}' matched 0 target modules in the LTX-2.3 DiT",
            detail=f"{len(file.prepared)} adapter(s) in the file, none of whose module paths exist in this transformer.",
            code="lora_incompatible",
        )

    def _ltx2_prepare_lora_file(self, file):
        from api.error_handlers import ValidationError
        from core.adapters import AdapterIncompatible
        from core.models.ltx2.ltx2_lora import normalise_lora_state_dict, detect_lora_format

        class Ltx2ArchMismatch(AdapterIncompatible, ValidationError):
            def __init__(self, message: str, detail: str = None, code: Optional[str] = None):
                code = code or "ltx2_lora_arch_mismatch"
                AdapterIncompatible.__init__(self, message, code=code)
                ValidationError.__init__(self, message, detail=detail, code=code)

        class Ltx2Incompatible(AdapterIncompatible, ValidationError):
            def __init__(self, message: str, detail: str = None, code: Optional[str] = None):
                code = code or "lora_incompatible"
                AdapterIncompatible.__init__(self, message, code=code)
                ValidationError.__init__(self, message, detail=detail, code=code)

        if not file.config.get("apply_to_unet", True):
            self._ltx2_lora_warn(
                f"LoRA '{file.name}' was selected with apply_to_unet disabled; an "
                f"LTX-2.3 LoRA only targets the video DiT, so nothing was applied.",
                "ltx2_lora_unet_disabled")
            return {}

        declared = str(file.metadata.get("model_type")
                       or file.metadata.get("modelspec.architecture") or "").strip()
        if declared and declared != "ltx2":
            self._ltx2_lora_warn(
                f"LoRA '{file.name}' declares architecture '{declared}', not ltx2.",
                "ltx2_lora_arch_mismatch")
            raise Ltx2ArchMismatch(
                f"LoRA '{file.name}' was trained for '{declared}', not LTX-2.3",
                detail="Its tensors index a different architecture's module tree; "
                       "applying it would either match nothing or match the wrong layers.",
                code="ltx2_lora_arch_mismatch")

        grouped = normalise_lora_state_dict(file.tensors)
        fmt = detect_lora_format(file.tensors)
        if not grouped:
            self._ltx2_lora_warn(
                f"LoRA '{file.name}' carries no recognisable LoRA tensors (key format '{fmt}').",
                "lora_incompatible")
            raise Ltx2Incompatible(
                f"LoRA '{file.name}' carries no recognisable LoRA tensors",
                detail=f"{len(file.tensors)} tensor(s), key format '{fmt}'. Expected "
                       f"sd-scripts native keys (lora_unet_<module path>.lora_down.weight / "
                       f"lora_up.weight / alpha) as written by SushiUI's LTX-2.3 trainer.",
                code="lora_incompatible")

        if file.config.get("unet_layer_weights"):
            self._ltx2_lora_warn(
                f"per-block LoRA weights are not applied for LTX-2.3; LoRA "
                f"'{file.name}' uses its plain strength ({file.strength}) on every block.",
                "ltx2_lora_layer_weights_ignored")
        step_range = file.config.get("step_range")
        if step_range is not None and list(step_range) != [0, 1000]:
            self._ltx2_lora_warn(
                f"LoRA step_range is not applied for LTX-2.3; LoRA '{file.name}' is "
                f"active for the whole denoise loop.",
                "ltx2_lora_step_range_ignored")

        return grouped

    def _ltx2_build_lora_branch(self, request):
        from core.adapters import LoRALinearLayer, PreparedBranch, SHAPE_MISMATCH, lora_branch_dtype
        from core.models.ltx2.ltx2_lora import flatten_module_path, metadata_alpha

        stem = flatten_module_path(request.module_path)
        weights = request.prepared.get(stem)
        if weights is None:
            return None

        down = weights.get("down")
        up = weights.get("up")
        if down is None or up is None:
            return None

        base = request.base
        expected_in = getattr(base, "in_features", None)
        expected_out = getattr(base, "out_features", None)
        lora_in = down.shape[-1]
        lora_out = up.shape[0]
        if lora_in != expected_in or lora_out != expected_out or down.shape[0] != up.shape[1]:
            return SHAPE_MISMATCH

        rank = down.shape[0]
        alpha = weights.get("alpha")
        fallback_alpha = metadata_alpha(request.file.metadata)
        if alpha is not None:
            alpha_val = float(alpha.item()) if torch.is_tensor(alpha) else float(alpha)
        elif fallback_alpha is not None:
            alpha_val = float(fallback_alpha)
        else:
            alpha_val = float(rank)

        branch = LoRALinearLayer(base, rank=rank, alpha=alpha_val, lora_name=request.module_path)
        device = base.weight.device
        dtype = lora_branch_dtype(base)
        with torch.no_grad():
            branch.lora_down.weight.data = down.to(device=device, dtype=dtype)
            branch.lora_up.weight.data = up.to(device=device, dtype=dtype)

        return PreparedBranch(branch, request.file.strength)

    def _ltx2_lora_components(self):
        from core.adapters import AdapterComponent
        from core.models.ltx2.ltx2_lora import iter_lora_slots

        def iter_ltx2_targets(transformer):
            for module_path, parent, slot in iter_lora_slots(transformer):
                yield parent, slot, module_path

        transformer = self._ltx2_lora_transformer()
        return [AdapterComponent(
            name="transformer",
            module=transformer,
            iter_targets=iter_ltx2_targets,
            build_branch=self._ltx2_build_lora_branch,
        )]

    @property
    def _ltx2_lora_original_modules(self):
        return self._ltx2_lora_session.state("transformer").originals

    @property
    def _ltx2_lora_wrapped_keys(self):
        return self._ltx2_lora_session.state("transformer").wrapped

    def _load_lora_ltx2(self, lora_configs: Optional[List[Dict]]) -> int:
        transformer = self._ltx2_lora_transformer()
        self._unload_lora_ltx2()

        if not lora_configs:
            return 0
        if transformer is None:
            raise RuntimeError(
                "LTX-2.3 components not loaded; cannot apply the selected LoRA(s)")

        print(f"[LTX-2.3 LoRA] Loading {len(lora_configs)} LoRA(s)...")
        applied = self._ltx2_lora_session.load(lora_configs, self._ltx2_lora_components()).applied
        if applied:
            self._ltx2_sync_block_swap_after_lora()
        return applied

    def _unload_lora_ltx2(self) -> int:
        restored = self._ltx2_lora_session.unload(self._ltx2_lora_components())
        if restored:
            offloader = self._ltx2_block_offloader()
            if offloader is not None and getattr(offloader, "blocks_to_swap", 0):
                _ltx2_invalidate_block_swap_caches(offloader)
            print(f"[LTX-2.3 LoRA] Unloaded {restored} LoRA wrapper(s)")
        return restored

    def _ltx2_sync_block_swap_after_lora(self) -> None:
        """Reconcile the block offloader with the adapters just installed.

        Both cases come from LTX-2.3's offloader being PERSISTENT state on
        ``Ltx2BlockLoopWrapper`` -- FLUX.2's and MiniMax-H3's are rebuilt per
        generation, so neither has to deal with this:

        1. Every cache the offloader holds describes the PREVIOUS block tree, on
           BOTH swap paths -- see ``_ltx2_invalidate_block_swap_caches``, which
           is why the drop is unconditional rather than H2D-only.
        2. A LoRA that does not cover every swappable block leaves those blocks
           structurally unequal, which the coalesced path asserts against
           (``_h2d_setup``: "H2D-only requires identically-structured swappable
           blocks"). Fall back to the standard per-module swap, which pairs by
           name and tolerates an unpaired module, rather than let that assert
           fire mid-denoise.
        """
        offloader = self._ltx2_block_offloader()
        if offloader is None or not getattr(offloader, "blocks_to_swap", 0):
            return
        _ltx2_invalidate_block_swap_caches(offloader)

        if not getattr(offloader, "h2d_only", False):
            return

        from core.models.ltx2.ltx2_lora import swappable_block_weight_footprints
        footprints = swappable_block_weight_footprints(
            offloader.blocks, offloader.blocks_to_swap)
        if len(set(footprints)) > 1:
            offloader.h2d_only = False
            self._ltx2_lora_warn(
                "the selected LoRA(s) do not cover every swapped transformer block, so the "
                "coalesced host-to-device block-swap path (which requires identically sized "
                "blocks) was disabled for this generation; the standard block swap is used "
                "instead. That path pairs an outgoing and an incoming block's Linears BY "
                "NAME, so the modules a wrapped block has and an unwrapped one does not (the "
                "adapter branches, and the wrapped base Linear itself) cannot pair: they are "
                "moved onto the GPU without the outgoing block's counterpart being evicted, "
                "and so accumulate there over the denoise loop. Applying the LoRA to every "
                "swapped block, or lowering blocks_to_swap, avoids both.",
                "ltx2_lora_h2d_disabled")
            return

        print("[LTX-2.3 LoRA] block-swap caches dropped; they will be rebuilt on the next "
              "forward so the adapters stream with their blocks")

    def _ltx2_build_fbcache(self, params: Dict[str, Any], block_swap_on: bool):
        """Build a FirstBlockCache for LTX-2.3, or None when inactive/guarded.

        FBCache is mutually exclusive with Block Swap (a cache hit skips the
        block loop -- including its wait_for_block/submit_move_blocks_forward
        calls -- which would desync the per-block swap prefetch rotation), so it
        is force-disabled (with a logged reason) whenever ``blocks_to_swap > 0``.
        FBCache is also mutually exclusive with Spectrum (same
        trajectory-redundancy target as FBCache -- both skip a full forward on
        selected steps); Spectrum takes precedence, mirroring the FLUX.2 policy
        (``_flux2_build_fbcache``), so FBCache is disabled whenever
        ``spectrum_enable`` is set."""
        from core.inference.fbcache import build_fbcache, fbcache_active
        if not fbcache_active(params):
            return None
        if block_swap_on:
            print("[FBCache] LTX-2.3 disabled: Block Swap is enabled (layer skip desyncs rotation)")
            return None
        if params.get("spectrum_enable", False):
            print("[FBCache] LTX-2.3 disabled: Spectrum is enabled (same redundancy target)")
            return None
        return build_fbcache(params, label="LTX-2.3")

    def _ltx2_build_spectrum(self, params: Dict[str, Any], num_inference_steps: int, block_swap_on: bool):
        """Build the (video, audio) Spectrum output-forecaster pair for LTX-2.3, or (None, None).

        Two forecasters, built from IDENTICAL config, so ``is_anchor(step)``
        agrees for both streams (the anchor schedule is a pure function of the
        step index + config, not of the tensor data) -- this is required for the
        wrapper's single skip/anchor branch to cover both streams consistently.

        Mutually exclusive with Block Swap (a forecast-skip step returns from
        the wrapper's ``forward`` without running the block loop, so the
        offloader's per-block wait/submit calls never fire, desyncing the swap
        prefetch rotation) -- disabled whenever ``blocks_to_swap > 0``.

        Forecasting requires exactly ONE transformer call per denoise step (the
        forecast is fit against, and skips, that single call). LTX-2.3
        generation today issues exactly one call per step (CFG is a single
        batched 2B-batch call, not two separate calls) and never sets
        Spatio-Temporal Guidance (``spatio_temporal_guidance_blocks`` /
        ``perturbation_mask`` are not wired into ``params`` anywhere in this
        module). Defensively check for that STG param anyway: if a future
        change threads it through, disable Spectrum rather than silently
        forecasting an inconsistent multi-call step."""
        if not params.get("spectrum_enable", False):
            return None, None
        if block_swap_on:
            print("[Spectrum] LTX-2.3 disabled: Block Swap is enabled (forecast skip desyncs swap rotation)")
            return None, None
        if params.get("stg_scale") or params.get("audio_stg_scale"):
            print("[Spectrum] LTX-2.3 disabled: Spatio-Temporal Guidance would require more "
                  "than one transformer call per step (not currently supported alongside Spectrum)")
            return None, None
        from core.inference.spectrum_forecaster import build_output_forecaster
        video_fc = build_output_forecaster(params, num_inference_steps, label="LTX-2.3 video")
        if video_fc is None:
            return None, None
        audio_fc = build_output_forecaster(params, num_inference_steps, label="LTX-2.3 audio")
        # Same config -> build_output_forecaster's warmup-length gate is
        # deterministic, so audio_fc is None iff video_fc is None; this is just
        # a defensive symmetry check, not expected to fire in practice.
        if audio_fc is None:
            print("[Spectrum] LTX-2.3: audio forecaster failed to build; disabling Spectrum entirely")
            return None, None
        max_cache = video_fc.max_cache
        print(f"[Spectrum] LTX-2.3: {len(video_fc.anchors)}/{num_inference_steps} actual passes "
              f"(video + audio forecasters, each caching up to {max_cache} anchor tensor(s))")
        return video_fc, audio_fc

    def _ltx2_prepare_style_reference(self, style_image, width: int, height: int, device) -> "torch.Tensor":
        """VAE-encode a still-image style reference as a ONE-FRAME LTX-2.3 video
        latent, packed into the SAME token layout attn1 sees for the target's
        own frame-0 spatial tokens (see ``core.inference.style_ltx2`` module
        docstring: "Still -> single-frame video-latent reference"). Returns a
        ``[1, H*W, C]`` float32 tensor (pre ``proj_in``, pre re-noising) --
        analogous to FLUX.2's ``_flux2_prepare_style_reference`` but through the
        LTX-2.3 VIDEO VAE with ``num_frames=1`` instead of an image VAE.
        """
        from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline

        vae = self.ltx2_components["vae"]
        vae_device = next(vae.parameters()).device
        vae_dtype = next(vae.parameters()).dtype

        img = style_image.convert("RGB").resize((int(width), int(height)), Image.LANCZOS)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) * 2.0
        # [1, C, F=1, H, W]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        img_tensor = img_tensor.to(device=vae_device, dtype=vae_dtype)

        with torch.no_grad():
            latent_dist = vae.encode(img_tensor).latent_dist
            latent = latent_dist.mode() if hasattr(latent_dist, "mode") else latent_dist.sample()
            latent = LTX2Pipeline._normalize_latents(
                latent, vae.latents_mean, vae.latents_std, vae.config.scaling_factor
            )
            packed = LTX2Pipeline._pack_latents(latent, patch_size=1, patch_size_t=1)

        return packed.to(device=device, dtype=torch.float32)

    def _ltx2_style_triple(self, style_dict: Dict[str, Any], width: int, height: int, device,
                            seed, ref_index: int = 0):
        """Build a single ``(StyleTransferConfig, ref_x0, eps_ref)`` triple from
        one ``style_transfer`` dict. ``axes_dims`` is intentionally left unset
        (LTX-2.3's interleaved RoPE does not match ``frequency_scale_vector``'s
        layout -- see ``core.inference.style_ltx2`` module docstring; the attn1
        hook always passes an all-ones frequency vector regardless).

        ``value_mode`` is forced to ``"ref_raw"`` here (overriding whatever
        ``style_config_from_dict`` parsed from the request) -- see
        ``core.inference.style_ltx2`` module docstring's "Value-mode
        deviation" section: ``make_ref_value``'s "target_adain" blend requires
        the target's own image-token region and the reference to share the
        SAME token count, which never holds for LTX-2.3 (still ref = ``H*W``
        tokens; target video = ``num_frames*H*W`` tokens). The single-ref
        attention hook (``style_ltx2._apply_style_hook``) works around this by
        bypassing ``make_ref_value`` entirely and is UNAFFECTED by this
        override (it never reads ``cfg.value_mode``); the multi-reference hook
        instead goes through the shared ``StyleContext.collect_block_refs``/
        ``make_ref_value`` API (not modified for this port), so forcing
        ``"ref_raw"`` here is what keeps that call shape-safe.

        ``ref_index`` decorrelates the fixed re-noising noise tensor across
        multiple simultaneous references (each ref would otherwise draw the
        EXACT same noise from the ``seed+991`` offset, since that offset does
        not depend on which reference is being prepared). ``ref_index=0`` (the
        default, used by the single-ref path) reproduces the pre-multi-ref
        ``seed+991`` offset exactly.
        """
        from diffusers.utils.torch_utils import randn_tensor
        from core.inference.reference_style import style_config_from_dict

        cfg = style_config_from_dict(style_dict)
        cfg.value_mode = "ref_raw"

        ref_x0 = self._ltx2_prepare_style_reference(style_dict["image"], width, height, device)

        try:
            seed_i = int(seed)
        except (TypeError, ValueError):
            seed_i = -1
        ref_seed = None if seed_i < 0 else (seed_i + 991 + ref_index) % (2 ** 32)
        generator = torch.Generator(device=device).manual_seed(ref_seed) if ref_seed is not None else None
        eps_ref = randn_tensor(ref_x0.shape, generator=generator, device=device, dtype=ref_x0.dtype)
        return cfg, ref_x0, eps_ref

    def _ltx2_style_config(self, params: Dict[str, Any], width: int, height: int, device):
        """Build a ``(StyleTransferConfig, ref_x0, eps_ref)`` triple from
        ``params["style_transfer"]`` (assembled by
        ``generation_utils.process_controlnet_configs``), or ``(None, None,
        None)`` when no style reference is attached. Single-reference path,
        BYTE-IDENTICAL to the pre-multi-ref implementation (delegates to
        ``_ltx2_style_triple`` with ``ref_index=0``, which reproduces the
        original ``seed+991`` re-noising offset exactly).
        """
        style_dict = params.get("style_transfer")
        if not style_dict or not style_dict.get("image"):
            return None, None, None

        seed = params.get("seed", -1)
        return self._ltx2_style_triple(style_dict, width, height, device, seed, ref_index=0)

    def _ltx2_style_configs(self, params: Dict[str, Any], width: int, height: int, device):
        """Build the full style-transfer configuration for LTX-2.3 generation,
        covering both the single-reference path (legacy ``(style_cfg,
        style_ref_x0, style_eps_ref)`` triple, exactly as ``_ltx2_style_config``
        would return) and the multi-reference path (``style_refs``, a list of
        per-ref triples, populated ONLY when ``params["style_transfers"]`` has
        more than one entry). A single-entry ``style_transfers`` list is
        intentionally routed through the single-ref triple instead (``style_refs``
        stays ``None``), so the pre-multi-ref code path executes
        byte-identically end to end. Mirrors ``_krea2_style_configs`` /
        ``_anima_style_configs``.

        Returns ``(style_cfg, style_ref_x0, style_eps_ref, style_refs,
        style_combine_mode)``.
        """
        style_list = params.get("style_transfers")
        if style_list and len(style_list) > 1:
            seed = params.get("seed", -1)
            combine_mode = str(params.get("style_combine_mode", "stack") or "stack")
            refs = []
            for idx, style_dict in enumerate(style_list):
                if not style_dict or not style_dict.get("image"):
                    continue
                refs.append(self._ltx2_style_triple(style_dict, width, height, device, seed, ref_index=idx))
            if len(refs) > 1:
                return None, None, None, refs, combine_mode
            if len(refs) == 1:
                cfg, x0, eps = refs[0]
                return cfg, x0, eps, None, combine_mode
            return None, None, None, None, combine_mode

        style_cfg, style_ref_x0, style_eps_ref = self._ltx2_style_config(params, width, height, device)
        return style_cfg, style_ref_x0, style_eps_ref, None, "stack"

    def _ltx2_resolve_style(self, params: Dict[str, Any]) -> bool:
        """Returns whether style transfer is requested for this generation and,
        if so, force-disables the features that are mutually exclusive with it
        DIRECTLY ON ``params`` (mirrors every other arch's precedence policy --
        see ``core.inference.style_ltx2`` module docstring's interop section):
        FBCache and Spectrum are disabled (a cache hit / forecast skip would
        desync the per-block style capture/inject store), and Block Swap is
        forced off (``blocks_to_swap = 0``; the ref-capture sub-pass does not
        thread the block-offloader's wait/submit calls).

        ``style_active`` is true when EITHER ``params["style_transfer"]``
        (singular, legacy single-ref key) carries an image OR
        ``params["style_transfers"]`` (plural, 0+ entries) is non-empty --
        mirrors the Anima/Krea2 ``params.get("style_transfer") or
        params.get("style_transfers")`` truthy check.
        """
        style_active = bool((params.get("style_transfer") or {}).get("image")) or bool(params.get("style_transfers"))
        if not style_active:
            return False
        if params.get("stg_scale") or params.get("audio_stg_scale"):
            # Mirrors _ltx2_build_spectrum's identical guard: Spatio-Temporal
            # Guidance issues EXTRA (non-doubled) transformer calls per step
            # that core.inference.style_ltx2's CFG row-split heuristic does not
            # model correctly (see that module's docstring). Not reachable
            # today (no caller wires stg_scale/audio_stg_scale for LTX-2.3),
            # kept as a defensive guard against a future STG wiring.
            print("[StyleLTX2] Style transfer disabled: Spatio-Temporal Guidance is active "
                  "(extra per-step transformer calls are not modeled by the CFG row-split logic)")
            params["style_transfer"] = None
            params["style_transfers"] = None
            return False
        if params.get("fbcache_enable"):
            print("[StyleLTX2] FBCache disabled: style transfer is active (capture-forward cache pollution)")
            params["fbcache_enable"] = False
        if params.get("spectrum_enable"):
            print("[StyleLTX2] Spectrum disabled: style transfer is active (capture-forward cache pollution)")
            params["spectrum_enable"] = False
        if int(params.get("blocks_to_swap", 0) or 0) > 0:
            print("[StyleLTX2] Block Swap disabled: style transfer is active "
                  "(ref-capture sub-pass does not thread the block-offloader rotation)")
            params["blocks_to_swap"] = 0
        return True

    def _ensure_ltx2_offload(self, blocks_to_swap: int = 0, force_block_swap_mode: bool = False):
        """Attach model_cpu_offload hooks, in either the stock ("normal") mode or
        the block-swap-compatible ("block_swap") mode, re-attaching whenever the
        requested mode differs from what is currently attached.

        Stock mode (``blocks_to_swap == 0``, UNCHANGED): `enable_model_cpu_offload`
        stages the offload sequence (text_encoder -> connectors -> transformer ->
        vae -> audio_vae -> vocoder) so the 19B transformer + Gemma-3 text encoder
        move through the GPU one component at a time.

        Block-swap mode (``blocks_to_swap > 0``): `enable_model_cpu_offload`
        normally moves each staged component to GPU via an accelerate forward
        hook that owns the WHOLE module (`cpu_offload_with_hook`), which
        conflicts with block-swap (some transformer blocks must stay resident on
        CPU while the rest stream through GPU). diffusers 0.38.0's
        `enable_model_cpu_offload` (`pipeline_utils.py` ~1189-1282) builds its
        per-model hook chain by walking ``model_cpu_offload_seq.split("->")``;
        remaining `self.components` entries are either given a plain
        `.to(device)` (if listed in `self._exclude_from_cpu_offload`) or another
        accelerate hook. Both are plain-Python attributes read off the pipeline
        INSTANCE (falling back to the class default), so reassigning them on the
        instance does not mutate the class or other pipeline instances. We drop
        the ``"transformer"`` token from the instance's ``model_cpu_offload_seq``
        AND add ``"transformer"`` to ``self._exclude_from_cpu_offload`` so it
        takes the plain-`.to(device)` branch (one unconditional move, not an
        accelerate hook) instead of being hook-managed. The block offloader
        (built afterward in `_ensure_ltx2_block_swap_wrapper`) then repositions
        the swappable blocks' weights back to CPU. Every other component keeps
        its stock accelerate-hook behavior in both modes.
        """
        pipeline = self.ltx2_components.get("pipeline")
        if pipeline is None:
            raise RuntimeError("LTX-2.3 pipeline reference missing from components")

        desired_mode = "block_swap" if (blocks_to_swap > 0 or force_block_swap_mode) else "normal"
        current_mode = getattr(self, "_ltx2_offload_mode", None)
        if getattr(self, "_ltx2_offload_enabled", False) and current_mode == desired_mode:
            return pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            try:
                # model_cpu_offload_seq is a CLASS attribute; reading it via
                # type(pipeline) always yields the stock diffusers-defined
                # string, independent of any previous instance-level override.
                stock_seq = type(pipeline).model_cpu_offload_seq
                exclude = list(getattr(pipeline, "_exclude_from_cpu_offload", []) or [])
                if desired_mode == "block_swap":
                    pipeline.model_cpu_offload_seq = "->".join(
                        tok for tok in stock_seq.split("->") if tok != "transformer"
                    )
                    if "transformer" not in exclude:
                        exclude.append("transformer")
                else:
                    pipeline.model_cpu_offload_seq = stock_seq
                    exclude = [tok for tok in exclude if tok != "transformer"]
                pipeline._exclude_from_cpu_offload = exclude

                pipeline.enable_model_cpu_offload(device=device)
                print(f"[LTX-2.3] enable_model_cpu_offload attached (mode={desired_mode}, "
                      f"seq={pipeline.model_cpu_offload_seq})")
            except Exception as e:
                print(f"[LTX-2.3] enable_model_cpu_offload failed ({e}); "
                      f"falling back to whole-pipeline .to(cuda)")
                pipeline.to(device)
        else:
            print("[LTX-2.3] CUDA unavailable; running pipeline on CPU")

        # VRAM: enable the LTX video VAE's temporal/spatial tiling so decoding the
        # full [T,H,W] latent does not spike VRAM (the dominant generation peak,
        # especially for long clips). Idempotent; diffusers picks tile sizes.
        vae = self.ltx2_components.get("vae")
        if vae is not None and hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling()
                print("[LTX-2.3] video VAE tiling enabled (decode VRAM headroom)")
            except Exception as e:
                print(f"[LTX-2.3] VAE tiling enable failed ({e}); continuing")

        self._ltx2_offload_enabled = True
        self._ltx2_offload_mode = desired_mode
        return pipeline

    def _ensure_ltx2_block_swap_wrapper(self, blocks_to_swap: int, force_wrap: bool = False):
        """Wrap (or unwrap) the LTX-2.3 transformer for AP1 block-swap GENERATION.

        ``blocks_to_swap <= 0`` and ``force_wrap=False``: unwraps back to the
        stock ``LTX2VideoTransformer3DModel`` (byte-identical current behavior —
        the wrapper is NOT applied in this case).

        ``blocks_to_swap <= 0`` and ``force_wrap=True``: FBCache (AP2) needs the
        wrapper's custom block loop even with no real block-swap; wraps with a
        NULL block offloader (``block_offloader=None``, mirroring
        ``Flux2BlockSwapWrapper``'s FBCache-only path) so ``_any_feature_active()``
        is decided solely by ``_fbcache``. The caller must still route
        ``_ensure_ltx2_offload`` through ``force_block_swap_mode=True`` in this
        case, since the custom forward bypasses the whole-transformer accelerate
        offload hook (it calls submodules directly, never ``inner.forward()``).

        ``blocks_to_swap > 0``: builds a ``TransformerBlockOffloader`` over
        ``transformer.transformer_blocks`` (generic block_offloading.py,
        ``supports_backward=False`` — inference only) and wraps the transformer
        with ``Ltx2BlockLoopWrapper``. Both ``pipeline.transformer`` and
        ``self.ltx2_components["transformer"]`` are updated to the wrapper so
        every consumer (base pipeline and a later-built i2v/cond pipeline)
        sees the same object. An already-cached i2v pipeline AND/OR an
        already-cached cond (video-outpaint) pipeline (if either exists) have
        their ``transformer`` ref updated too, since they share every module
        with the base pipeline rather than owning their own weights -- this
        is what keeps a 2nd+ vid_outpaint call (whose requested block-swap/
        FBCache/Spectrum state can differ from cond_pipeline's build-time
        state) from running against a stale/unwrapped transformer.

        Must be called AFTER `_ensure_ltx2_offload(blocks_to_swap)` so the
        transformer's device placement (whole-GPU via the offload-exclusion
        above) is settled before the offloader repositions swappable blocks.
        """
        pipeline = self.ltx2_components.get("pipeline")
        if pipeline is None:
            raise RuntimeError("LTX-2.3 pipeline reference missing from components")

        from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper

        current = pipeline.transformer
        inner = current.transformer if isinstance(current, Ltx2BlockLoopWrapper) else current

        if blocks_to_swap <= 0:
            if force_wrap:
                # FBCache-only wrap: no real block offloader. Idempotent no-op if
                # already wrapped this way (no swap count, wrapper present).
                already = (
                    isinstance(current, Ltx2BlockLoopWrapper)
                    and current._block_offloader is None
                    and getattr(self, "_ltx2_block_swap_count", 0) == 0
                )
                if already:
                    return
                if isinstance(current, Ltx2BlockLoopWrapper) and current._block_offloader is not None:
                    current._block_offloader.cleanup()
                wrapper = current if isinstance(current, Ltx2BlockLoopWrapper) else Ltx2BlockLoopWrapper(inner, block_offloader=None)
                wrapper._block_offloader = None
                pipeline.transformer = wrapper
                self.ltx2_components["transformer"] = wrapper
                i2v = self.ltx2_components.get("i2v_pipeline")
                if i2v is not None:
                    i2v.transformer = wrapper
                cond = self.ltx2_components.get("cond_pipeline")
                if cond is not None:
                    cond.transformer = wrapper
                self._ltx2_block_swap_count = 0
                print("[LTX-2.3] FBCache-only wrap active (Ltx2BlockLoopWrapper, no block offloader)")
                return
            if isinstance(current, Ltx2BlockLoopWrapper):
                offloader = current._block_offloader
                if offloader is not None:
                    offloader.cleanup()
                pipeline.transformer = inner
                self.ltx2_components["transformer"] = inner
                i2v = self.ltx2_components.get("i2v_pipeline")
                if i2v is not None:
                    i2v.transformer = inner
                cond = self.ltx2_components.get("cond_pipeline")
                if cond is not None:
                    cond.transformer = inner
                print("[LTX-2.3] Block Swap disabled; transformer unwrapped (stock forward)")
            self._ltx2_block_swap_count = 0
            return

        prev_count = getattr(self, "_ltx2_block_swap_count", 0)
        if isinstance(current, Ltx2BlockLoopWrapper) and prev_count == blocks_to_swap:
            return  # already wired for this exact swap count

        if isinstance(current, Ltx2BlockLoopWrapper) and current._block_offloader is not None:
            current._block_offloader.cleanup()

        from core.memory_management import TransformerBlockOffloader
        device = next(inner.parameters()).device
        offloader = TransformerBlockOffloader(
            blocks=inner.transformer_blocks,
            blocks_to_swap=blocks_to_swap,
            device=device,
            target_dtype=inner.dtype,
            use_pinned_memory=False,
            transformer=inner,
            supports_backward=False,
            # Generation weights are frozen: use the H2D-only fast path (permanent
            # pinned CPU masters + coalesced single copy per block) so we skip the
            # pointless device->host eviction of read-only weights (halves PCIe
            # traffic vs the standard swap). Auto-disables if backward is ever on.
            h2d_only=True,
        )
        offloader.prepare_block_devices_before_forward()

        wrapper = Ltx2BlockLoopWrapper(inner, block_offloader=offloader)
        pipeline.transformer = wrapper
        self.ltx2_components["transformer"] = wrapper
        i2v = self.ltx2_components.get("i2v_pipeline")
        if i2v is not None:
            i2v.transformer = wrapper
        cond = self.ltx2_components.get("cond_pipeline")
        if cond is not None:
            cond.transformer = wrapper
        self._ltx2_block_swap_count = blocks_to_swap
        print(f"[LTX-2.3] Block Swap enabled: {blocks_to_swap} blocks to swap "
              f"(Ltx2BlockLoopWrapper active)")

    def _ensure_ltx2_swap_and_offload(self, blocks_to_swap: int, force_wrap: bool = False):
        """Bring the shared transformer to the requested block-swap state with the
        CORRECT ordering relative to the model-offload hook attach, and return the
        base pipeline.

        Enabling (``blocks_to_swap > 0``): offload FIRST (which excludes the
        transformer from the accelerate hook chain and gives it a plain
        `.to(device)`), THEN wrap + build the block offloader that repositions the
        swappable blocks to CPU.

        ``force_wrap`` (FBCache-only, ``blocks_to_swap == 0``): same ordering as
        the enabling path (offload first, in ``force_block_swap_mode`` so the
        transformer is offload-excluded even though no real block-swap is
        requested), then wrap with a null block offloader. Needed because the
        wrapper's custom forward bypasses the whole-transformer accelerate hook
        (see ``_ensure_ltx2_block_swap_wrapper``).

        Disabling (``blocks_to_swap <= 0`` and ``force_wrap=False``): UNWRAP
        FIRST, then re-attach offload. `enable_model_cpu_offload` moves the whole
        pipeline to CPU and binds a streaming forward-hook to
        ``pipeline.transformer``; if we re-offloaded while the wrapper were still
        installed, the hook would bind to the wrapper object that the subsequent
        unwrap discards, leaving the inner transformer stranded on CPU with no
        hook (device-mismatch on the next call). Unwrapping first makes the hook
        bind to the restored inner transformer.
        """
        if blocks_to_swap > 0:
            pipeline = self._ensure_ltx2_offload(blocks_to_swap=blocks_to_swap)
            self._ensure_ltx2_block_swap_wrapper(blocks_to_swap)
        elif force_wrap:
            pipeline = self._ensure_ltx2_offload(blocks_to_swap=0, force_block_swap_mode=True)
            self._ensure_ltx2_block_swap_wrapper(0, force_wrap=True)
        else:
            self._ensure_ltx2_block_swap_wrapper(0)
            pipeline = self._ensure_ltx2_offload(blocks_to_swap=0)
        return pipeline

    def _ensure_ltx2_i2v_pipeline(self):
        """Build (once) and return the LTX2ImageToVideoPipeline.

        The image-to-video pipeline shares EVERY module with the base
        `LTX2Pipeline` — it only changes the denoise loop (first-frame keyframe
        conditioning). We therefore construct it from the already-loaded
        components WITHOUT reloading any weights, and cache it under
        `ltx2_components["i2v_pipeline"]` so it survives across calls and is
        freed by the load_model eviction loop (which iterates the components
        dict) on a model swap.

        Offload: we never call `enable_model_cpu_offload` on the i2v pipeline.
        The base pipeline owns the cpu-offload hooks (attached to the shared
        module objects via `_ensure_ltx2_offload`). Because the i2v pipeline
        drives those same module objects, the per-module accelerate hooks fire
        during its `__call__` exactly as they do for the base pipeline. This
        avoids double-hooking the shared modules.
        """
        cached = self.ltx2_components.get("i2v_pipeline")
        if cached is not None:
            return cached

        base = self.ltx2_components.get("pipeline")
        if base is None:
            raise RuntimeError("LTX-2.3 pipeline reference missing from components")

        from core.models.ltx2 import LTX2ImageToVideoPipeline

        i2v = LTX2ImageToVideoPipeline(
            scheduler=self.ltx2_components.get("scheduler"),
            vae=self.ltx2_components.get("vae"),
            audio_vae=self.ltx2_components.get("audio_vae"),
            text_encoder=self.ltx2_components.get("text_encoder"),
            tokenizer=self.ltx2_components.get("tokenizer"),
            connectors=self.ltx2_components.get("connectors"),
            transformer=self.ltx2_components.get("transformer"),
            vocoder=self.ltx2_components.get("vocoder"),
            processor=getattr(base, "processor", None),
        )
        self.ltx2_components["i2v_pipeline"] = i2v
        print("[LTX-2.3] LTX2ImageToVideoPipeline constructed from shared components "
              "(no weight reload; offload owned by base pipeline)")
        return i2v

    def _generate_img2vid_ltx2(
        self,
        params: Dict[str, Any],
        input_image,
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ):
        """Image-to-video generation with LTX-2.3 (first-frame keyframe).

        `input_image` is a PIL.Image used as the first-frame keyframe. The
        pipeline's `video_processor.preprocess(image, height, width)` resizes it
        to the (÷32) target resolution and the denoise loop pins it as frame 0
        via `conditioning_mask[:, :, 0] = 1`.

        Returns (frames, audio, audio_sample_rate, actual_seed) — identical
        contract to `_generate_txt2vid_ltx2` (see module docstring).
        """
        if not self.ltx2_components:
            raise RuntimeError("LTX-2.3 components not loaded. Please load an LTX-2.3 model first.")

        if input_image is None:
            raise RuntimeError("img2vid requires an input image for the first-frame keyframe")

        # Resolve parameters (mirrors _generate_txt2vid_ltx2; moved up: style
        # transfer needs width/height before the offload/wrap decision below).
        prompt = params.get("prompt", "") or ""
        negative_prompt = params.get("negative_prompt", "") or ""
        width = int(params.get("width", 768))
        height = int(params.get("height", 512))
        num_frames = int(params.get("num_frames", 121))
        frame_rate = float(params.get("frame_rate", 24.0))
        num_inference_steps = int(params.get("num_inference_steps", 8))
        guidance_scale = float(params.get("guidance_scale", 1.0))
        num_videos_per_prompt = int(params.get("num_videos_per_prompt", 1))
        max_sequence_length = int(params.get("max_sequence_length", 1024))
        audio_enable = bool(params.get("audio_enable", True))

        # Training-free reference-style transfer: resolve BEFORE FBCache/
        # Spectrum/Block-Swap (see _generate_txt2vid_ltx2 / _ltx2_resolve_style).
        style_active = self._ltx2_resolve_style(params)

        blocks_to_swap = int(params.get("blocks_to_swap", 0) or 0)

        # AP2 First-Block-Cache: build before the offload/wrap step so the wrap
        # decision (force the wrapper on for FBCache-only / Spectrum-only /
        # style-only) is known up front. Mutually exclusive with Block Swap and
        # Spectrum (see _ltx2_build_fbcache; Spectrum takes precedence over
        # FBCache) and with style transfer (force-disabled above).
        fbcache = self._ltx2_build_fbcache(params, blocks_to_swap > 0)
        num_inference_steps_probe = num_inference_steps
        spectrum_video, spectrum_audio = self._ltx2_build_spectrum(
            params, num_inference_steps_probe, blocks_to_swap > 0
        )
        force_wrap = (fbcache is not None or spectrum_video is not None or style_active) and blocks_to_swap <= 0

        # One-time in-place INT8 conversion (unet_quantization="int8"). MUST be
        # before the block-swap/offload step below: the block offloader captures
        # the transformer blocks' Linear modules and the conversion replaces them
        # (see _ltx2_runtime_int8). No-op for every other value and for an
        # already-quantized model.
        self._ltx2_runtime_int8(params, progress_callback=progress_callback,
                                force_wrap=force_wrap)

        # Base pipeline owns the offload hooks on the shared modules. This brings
        # the shared transformer to the requested block-swap state (wrap/unwrap +
        # offload) in the correct order, BEFORE the i2v pipeline is built (or
        # re-cached) so it always references the correct (wrapped or stock) object.
        self._ensure_ltx2_swap_and_offload(blocks_to_swap, force_wrap=force_wrap)
        pipeline = self._ensure_ltx2_i2v_pipeline()

        from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper
        style_target = pipeline.transformer if isinstance(pipeline.transformer, Ltx2BlockLoopWrapper) else None
        fbcache_target = pipeline.transformer if isinstance(pipeline.transformer, Ltx2BlockLoopWrapper) else None
        if fbcache is not None and fbcache_target is not None:
            fbcache_target.attach_fbcache(fbcache)
        elif fbcache is not None:
            print("[FBCache] LTX-2.3: could not attach (transformer not wrapped)")
            fbcache = None
        spectrum_target = None
        if spectrum_video is not None and fbcache_target is not None:
            spectrum_target = fbcache_target
            spectrum_target.attach_spectrum(spectrum_video, spectrum_audio)
        elif spectrum_video is not None:
            print("[Spectrum] LTX-2.3: could not attach (transformer not wrapped)")
            spectrum_video = spectrum_audio = None

        style_processors: List[Any] = []
        style_saved_processors: List[Any] = []
        style_cfg = None
        style_refs = None
        style_combine_mode = "stack"
        if style_active and style_target is not None:
            device = next(style_target.transformer.parameters()).device
            style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                self._ltx2_style_configs(params, width, height, device)
            if style_cfg is not None or style_refs is not None:
                from core.inference.style_ltx2 import install_ltx2_style_processors
                style_processors, style_saved_processors = install_ltx2_style_processors(style_target.transformer)
                style_target.attach_style(
                    style_processors, style_cfg, style_ref_x0, style_eps_ref,
                    style_refs=style_refs, combine_mode=style_combine_mode,
                    guidance_scale=guidance_scale,
                )
                style_target._style_total_steps = num_inference_steps
                ref_suffix = f" ({len(style_refs)} references, combine={style_combine_mode})" if style_refs else ""
                print(f"[StyleLTX2] Style transfer active: {len(style_processors)} attn1 processors stamped{ref_suffix}")
        elif style_active:
            print("[StyleLTX2]: could not attach (transformer not wrapped)")
            style_cfg = None
            style_refs = None

        # Normalize the keyframe to RGB PIL; the pipeline's video_processor
        # handles the resize/fit to (width, height).
        if not isinstance(input_image, Image.Image):
            raise RuntimeError("img2vid input_image must be a PIL.Image")
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        # Seed: -1 (or negative/None) -> random draw, recorded back for the caller.
        seed = params.get("seed", -1)
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = -1
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Progress + FBCache/Spectrum/style step advance: LTX2Pipeline invokes
        # callback_on_step_end(pipe, i, t, kwargs) AFTER every denoise step. We
        # advance _fbcache_step / _spectrum_step / _style_step_idx to i+1 there
        # (primed for the NEXT step's forward call); step 0 uses the wrapper's
        # default (0) from attach_fbcache() / attach_spectrum() / attach_style().
        style_wants_step_advance = style_cfg is not None or style_refs is not None
        if progress_callback is not None or fbcache_target is not None or spectrum_target is not None or style_wants_step_advance:
            def _cb(pipe, step_index, timestep, callback_kwargs):
                if progress_callback is not None:
                    try:
                        progress_callback(step_index + 1, num_inference_steps)
                    except Exception:
                        pass
                if fbcache_target is not None:
                    fbcache_target._fbcache_step = step_index + 1
                if spectrum_target is not None:
                    spectrum_target._spectrum_step = step_index + 1
                if style_wants_step_advance and style_target is not None:
                    style_target._style_step_idx = step_index + 1
                return callback_kwargs
            callback = _cb
        else:
            callback = None

        print(f"[LTX-2.3] img2vid: {width}x{height} num_frames={num_frames} "
              f"fps={frame_rate} steps={num_inference_steps} cfg={guidance_scale} "
              f"seed={seed} audio={audio_enable}")

        try:
            # LoRA: see _generate_txt2vid_ltx2's note (same ordering contract).
            self._load_lora_ltx2(params.get("loras"))

            video, audio = pipeline(
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                output_type="np",
                return_dict=False,
                max_sequence_length=max_sequence_length,
                callback_on_step_end=callback,
            )
        finally:
            self._unload_lora_ltx2()
            if fbcache_target is not None:
                print(f"[FBCache] LTX-2.3 summary: {fbcache.n_hits} hit(s), {fbcache.n_miss} miss(es)")
                fbcache_target.attach_fbcache(None)
            if style_saved_processors:
                from core.inference.style_ltx2 import restore_ltx2_style_processors
                if style_target is not None:
                    style_target.attach_style(None, None, None, None)
                restore_ltx2_style_processors(style_saved_processors)
                print("[StyleLTX2] processors restored (generation complete)")
            if spectrum_target is not None:
                v_stats = spectrum_video.stats()
                print(f"[Spectrum] LTX-2.3 summary: {v_stats['anchors']} anchor(s), "
                      f"{v_stats['forecasts']} forecast(s) of {v_stats['total']} step(s)")
                spectrum_target.attach_spectrum(None, None)

        frames_np = video[0]  # [T, H, W, C]
        frames = (np.clip(frames_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        audio_out = None
        audio_sample_rate = None
        if audio_enable and audio is not None:
            try:
                audio_sample_rate = int(pipeline.vocoder.config.output_sampling_rate)
            except Exception:
                audio_sample_rate = 24000
            try:
                audio_out = audio[0].detach().float().cpu()
            except Exception as e:
                print(f"[LTX-2.3] audio extraction failed ({e}); saving video without audio")
                audio_out = None
                audio_sample_rate = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return frames, audio_out, audio_sample_rate, seed

    def _ensure_ltx2_condition_pipeline(self):
        """Build (once) and return the LTX2ConditionPipeline.

        Same "no weight reload" construction as `_ensure_ltx2_i2v_pipeline`:
        every module is shared with the base `LTX2Pipeline` (and the i2v
        pipeline, if already built) -- only the denoise loop differs
        (arbitrary latent-index frame/video conditioning via
        `LTX2VideoCondition`, vs. the i2v pipeline's fixed first-frame
        keyframe). Cached under `ltx2_components["cond_pipeline"]` so it
        survives across calls and is freed by the load_model eviction loop
        (which iterates the components dict) on a model swap.

        Offload: identical policy to `_ensure_ltx2_i2v_pipeline` -- this
        pipeline never gets its own `enable_model_cpu_offload` call; it
        drives the SAME shared module objects the base pipeline's offload
        hooks are already attached to.

        NOTE: unlike `LTX2ImageToVideoPipeline.__init__` (which accepts a
        `processor` kwarg -- see `_ensure_ltx2_i2v_pipeline`),
        `LTX2ConditionPipeline.__init__` (pipeline_ltx2_condition.py:248-258)
        does NOT have a `processor` parameter at all -- passing one raises a
        TypeError. Verified directly against the venv's
        `pipeline_ltx2_condition.py`, not assumed from the i2v pipeline's
        signature.
        """
        cached = self.ltx2_components.get("cond_pipeline")
        if cached is not None:
            return cached

        base = self.ltx2_components.get("pipeline")
        if base is None:
            raise RuntimeError("LTX-2.3 pipeline reference missing from components")

        from core.models.ltx2 import LTX2ConditionPipeline

        cond_pipeline = LTX2ConditionPipeline(
            scheduler=self.ltx2_components.get("scheduler"),
            vae=self.ltx2_components.get("vae"),
            audio_vae=self.ltx2_components.get("audio_vae"),
            text_encoder=self.ltx2_components.get("text_encoder"),
            tokenizer=self.ltx2_components.get("tokenizer"),
            connectors=self.ltx2_components.get("connectors"),
            transformer=self.ltx2_components.get("transformer"),
            vocoder=self.ltx2_components.get("vocoder"),
        )
        self.ltx2_components["cond_pipeline"] = cond_pipeline
        print("[LTX-2.3] LTX2ConditionPipeline constructed from shared components "
              "(no weight reload; offload owned by base pipeline)")
        return cond_pipeline

    def _generate_vidoutpaint_ltx2(
        self,
        params: Dict[str, Any],
        video_frames: np.ndarray,
        fps: float,
        input_audio: Optional[bytes],
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ):
        """Video temporal outpaint with LTX-2.3: place a (trimmed) input clip
        at a latent-frame offset inside a LONGER output timeline and generate
        the frames before/after, preserving the placed input frames
        byte-exact.

        Pure orchestration over the stock `diffusers.LTX2ConditionPipeline`
        (no new denoise loop) -- see `core.pipeline.generate_vid_outpaint` /
        `scratchpad/outpaint_design.md` section 4.

        Args:
            params: see `OUTPAINT_VIDEO_DEFAULTS`.
            video_frames: np.uint8 [T, H, W, 3] decoded input clip (RGB), as
                returned by `utils.video_utils.load_video_frames`.
            fps: the input clip's own probed frame rate -- used ONLY to
                convert `input_trim_start_frames` into a real-time offset
                into the ORIGINAL audio track for `preserve_input` mode
                (placement of the VIDEO frames themselves is purely
                frame-index based and does not depend on this value).
            input_audio: WAV bytes of the input clip's original (untrimmed)
                audio track (see `utils.video_utils.extract_audio_stream`),
                or None if the clip has no audio stream. Only consulted when
                `outpaint_video_audio_mode == "preserve_input"`.
            progress_callback / step_callback: see `_generate_img2vid_ltx2`.

        Returns:
            (frames, audio, audio_sample_rate, actual_seed) -- identical
            contract to `_generate_img2vid_ltx2` / `_generate_txt2vid_ltx2`.
        """
        if not self.ltx2_components:
            raise RuntimeError("LTX-2.3 components not loaded. Please load an LTX-2.3 model first.")
        if video_frames is None or len(video_frames) == 0:
            raise RuntimeError("vid_outpaint requires a decoded input video clip")

        from api.error_handlers import ValidationError

        prompt = params.get("prompt", "") or ""
        negative_prompt = params.get("negative_prompt", "") or ""
        width = int(params.get("width", 768))
        height = int(params.get("height", 512))
        total_frames = int(params.get("total_frames", 121))
        frame_rate = float(params.get("frame_rate", 24.0))
        num_inference_steps = int(params.get("num_inference_steps", 8))
        guidance_scale = float(params.get("guidance_scale", 1.0))
        num_videos_per_prompt = int(params.get("num_videos_per_prompt", 1))
        max_sequence_length = int(params.get("max_sequence_length", 1024))
        audio_enable = bool(params.get("audio_enable", True))
        audio_mode = params.get("outpaint_video_audio_mode", "regenerate") or "regenerate"

        # ---- Trim the decoded clip (pixel frames) BEFORE preprocessing ----
        trim_start = max(0, int(params.get("input_trim_start_frames", 0) or 0))
        trim_end = max(0, int(params.get("input_trim_end_frames", 0) or 0))
        total_src_frames = video_frames.shape[0]
        end_idx = total_src_frames - trim_end if trim_end > 0 else total_src_frames
        trimmed_frames = video_frames[trim_start:end_idx]
        if trimmed_frames.shape[0] < 1:
            raise ValidationError(
                "vid_outpaint input trim leaves no frames",
                detail=f"input has {total_src_frames} frames; "
                       f"trim_start={trim_start}, trim_end={trim_end}",
            )

        # ---- Preprocess ONCE to the working (÷32) resolution -- the RESULT
        # is the exact-preserved content (mirrors the image-outpaint RESIZE
        # convention), not the raw upload. ----
        canonical_input_frames = _center_crop_resize_frames(trimmed_frames, width, height)

        # ---- Placement math (pure, no pipeline/GPU dependency) ----
        vae_component = self.ltx2_components.get("vae")
        frame_scale_factor = int(getattr(vae_component, "temporal_compression_ratio", 8) or 8)
        latent_num_frames = (total_frames - 1) // frame_scale_factor + 1

        desired_offset = max(0, int(params.get("input_offset_frames", 0) or 0))
        latent_index, pixel_start = _snap_offset_to_latent_index(
            desired_offset, frame_scale_factor, latent_num_frames
        )
        if pixel_start != desired_offset:
            try:
                from api.generation_status import add_warning
                add_warning(
                    f"input_offset_frames snapped from {desired_offset} to {pixel_start} "
                    f"(nearest valid latent frame index {latent_index})",
                    code="outpaint_video_offset_snapped",
                )
            except Exception:
                pass

        cond_num_frames = canonical_input_frames.shape[0]
        t_eff = _trim_conditioning_sequence_frames(
            pixel_start, cond_num_frames, total_frames, frame_scale_factor
        )
        if t_eff < 1:
            raise ValidationError(
                "vid_outpaint placement leaves no room for the input clip",
                detail=f"pixel_start={pixel_start}, total_frames={total_frames}, "
                       f"cond_num_frames={cond_num_frames}",
            )

        # Frame-drop transparency: `t_eff` is `cond_num_frames` rounded DOWN to
        # LTX-2.3's 8k+1 grid (`_trim_conditioning_sequence_frames`) -- when
        # that rounds below the full (trimmed) clip length, tail frames are
        # silently dropped from the preserved span unless surfaced here. The
        # user requires the input preserved exactly, so this must be visible:
        # warn, and record the EFFECTIVE preserved span (frame count + pixel
        # range) into `params` in place, so routes.py's `params.copy()` ->
        # gallery metadata/DB path picks it up automatically.
        params["outpaint_effective_preserved_frames"] = t_eff
        params["outpaint_effective_pixel_start"] = pixel_start
        params["outpaint_effective_pixel_end"] = pixel_start + t_eff
        if t_eff < cond_num_frames:
            dropped = cond_num_frames - t_eff
            try:
                from api.generation_status import add_warning
                add_warning(
                    f"{dropped} tail frame(s) of the (trimmed) input clip were dropped to fit "
                    f"LTX-2.3's 8k+1 frame grid; the effective preserved span is frames "
                    f"[{pixel_start}, {pixel_start + t_eff}) of the output ({t_eff} of "
                    f"{cond_num_frames} clip frames)",
                    code="outpaint_video_tail_frames_dropped",
                )
            except Exception:
                pass

        # LTX2VideoCondition's non-PIL (ndarray/tensor) preprocessing branch
        # inside VaeImageProcessor.preprocess() does NOT auto-divide by 255
        # (only the PIL path does, via pil_to_numpy), and its resize() step
        # calls F.interpolate, which requires a floating dtype -- passing our
        # raw uint8 array directly would raise. Verified against the venv's
        # image_processor.py, not assumed.
        cond_frames_float = canonical_input_frames.astype(np.float32) / 255.0

        from core.models.ltx2 import LTX2VideoCondition
        condition = LTX2VideoCondition(frames=cond_frames_float, index=latent_index, strength=1.0)

        # ---- FBCache/Spectrum/Block-Swap: identical wiring to _generate_img2vid_ltx2 ----
        blocks_to_swap = int(params.get("blocks_to_swap", 0) or 0)
        fbcache = self._ltx2_build_fbcache(params, blocks_to_swap > 0)
        spectrum_video, spectrum_audio = self._ltx2_build_spectrum(
            params, num_inference_steps, blocks_to_swap > 0
        )
        force_wrap = (fbcache is not None or spectrum_video is not None) and blocks_to_swap <= 0

        # One-time in-place INT8 conversion, before the offloader is built (see
        # _ltx2_runtime_int8; same ordering as the other two generate paths).
        self._ltx2_runtime_int8(params, progress_callback=progress_callback,
                                force_wrap=force_wrap)

        self._ensure_ltx2_swap_and_offload(blocks_to_swap, force_wrap=force_wrap)
        cond_pipeline = self._ensure_ltx2_condition_pipeline()

        from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper
        fbcache_target = cond_pipeline.transformer if isinstance(cond_pipeline.transformer, Ltx2BlockLoopWrapper) else None
        if fbcache is not None and fbcache_target is not None:
            fbcache_target.attach_fbcache(fbcache)
        elif fbcache is not None:
            print("[FBCache] LTX-2.3 vid_outpaint: could not attach (transformer not wrapped)")
            fbcache = None
        spectrum_target = None
        if spectrum_video is not None and fbcache_target is not None:
            spectrum_target = fbcache_target
            spectrum_target.attach_spectrum(spectrum_video, spectrum_audio)
        elif spectrum_video is not None:
            print("[Spectrum] LTX-2.3 vid_outpaint: could not attach (transformer not wrapped)")
            spectrum_video = spectrum_audio = None

        # Seed: -1 (or negative/None) -> random draw, recorded back for the caller.
        seed = params.get("seed", -1)
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = -1
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        if progress_callback is not None or fbcache_target is not None or spectrum_target is not None:
            def _cb(pipe, step_index, timestep, callback_kwargs):
                if progress_callback is not None:
                    try:
                        progress_callback(step_index + 1, num_inference_steps)
                    except Exception:
                        pass
                if fbcache_target is not None:
                    fbcache_target._fbcache_step = step_index + 1
                if spectrum_target is not None:
                    spectrum_target._spectrum_step = step_index + 1
                return callback_kwargs
            callback = _cb
        else:
            callback = None

        print(f"[LTX-2.3] vid_outpaint: {width}x{height} total_frames={total_frames} "
              f"placed at latent_idx={latent_index} (pixel {pixel_start}..{pixel_start + t_eff}) "
              f"fps={frame_rate} steps={num_inference_steps} cfg={guidance_scale} "
              f"seed={seed} audio_mode={audio_mode}")

        try:
            # LoRA: see _generate_txt2vid_ltx2's note (same ordering contract).
            self._load_lora_ltx2(params.get("loras"))

            video, audio = cond_pipeline(
                conditions=[condition],
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=total_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                output_type="np",
                return_dict=False,
                max_sequence_length=max_sequence_length,
                callback_on_step_end=callback,
            )
        finally:
            self._unload_lora_ltx2()
            if fbcache_target is not None:
                print(f"[FBCache] LTX-2.3 vid_outpaint summary: {fbcache.n_hits} hit(s), {fbcache.n_miss} miss(es)")
                fbcache_target.attach_fbcache(None)
            if spectrum_target is not None:
                v_stats = spectrum_video.stats()
                print(f"[Spectrum] LTX-2.3 vid_outpaint summary: {v_stats['anchors']} anchor(s), "
                      f"{v_stats['forecasts']} forecast(s) of {v_stats['total']} step(s)")
                spectrum_target.attach_spectrum(None, None)

        frames_np = video[0]  # [T, H, W, C], float in [0, 1]
        frames_out = (np.clip(frames_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        # ---- STRICT preservation: unconditional frame-exact paste over the
        # EFFECTIVE (possibly truncated by trim_conditioning_sequence) span. ----
        frames_out[pixel_start:pixel_start + t_eff] = canonical_input_frames[:t_eff]

        audio_out = None
        audio_sample_rate = None
        if audio_enable and audio is not None:
            try:
                audio_sample_rate = int(cond_pipeline.vocoder.config.output_sampling_rate)
            except Exception:
                audio_sample_rate = 24000
            try:
                audio_out = audio[0].detach().float().cpu()
            except Exception as e:
                print(f"[LTX-2.3] vid_outpaint audio extraction failed ({e}); saving video without audio")
                audio_out = None
                audio_sample_rate = None

            if audio_out is not None and audio_mode == "preserve_input":
                if input_audio is None:
                    print("[LTX-2.3] vid_outpaint: outpaint_video_audio_mode='preserve_input' requested "
                          "but the input clip has no audio stream -- falling back to 'regenerate'")
                    try:
                        from api.generation_status import add_warning
                        add_warning(
                            "preserve_input audio mode requested but the input clip has no audio "
                            "stream; falling back to regenerate",
                            code="outpaint_video_no_input_audio",
                        )
                    except Exception:
                        pass
                else:
                    try:
                        from utils.video_utils import extract_audio_window, mux_audio_over_span

                        # Placement in the OUTPUT timeline (what the pasted
                        # frames actually occupy, at the OUTPUT frame_rate).
                        offset_sec = pixel_start / frame_rate if frame_rate else 0.0
                        target_dur_sec = t_eff / frame_rate if frame_rate else 0.0
                        # The SAME t_eff frames, measured in the SOURCE clip's
                        # OWN real time (fps probed from the upload). When
                        # source fps != output frame_rate, these two durations
                        # differ -- the frames are reused 1:1 at frame_rate,
                        # so pasting the source audio at its native tempo
                        # would drift out of sync with the placed video.
                        # extract_audio_window pitch-preservingly time-stretches
                        # the source window to close that gap.
                        src_start_sec = trim_start / fps if fps else 0.0
                        src_dur_sec = t_eff / fps if fps else target_dur_sec

                        if target_dur_sec > 0 and abs(src_dur_sec - target_dur_sec) / target_dur_sec > 0.005:
                            try:
                                from api.generation_status import add_warning
                                add_warning(
                                    f"preserve_input audio was time-stretched ({src_dur_sec:.3f}s -> "
                                    f"{target_dur_sec:.3f}s) because the input clip's frame rate "
                                    f"({fps:.3f}) differs from the output frame_rate ({frame_rate:.3f}); "
                                    "preserve_input assumes matching fps for an untouched splice",
                                    code="outpaint_video_audio_stretched",
                                )
                            except Exception:
                                pass

                        generated_audio_np = audio_out.numpy()

                        # Pad the generated (vocoder) track to the AUTHORITATIVE
                        # video duration first -- the vocoder's own temporal
                        # grid can fall short of total_frames/frame_rate, and
                        # splicing against a too-short array would silently
                        # clamp/lose part of the placed window.
                        full_video_samples = int(round((total_frames / frame_rate) * audio_sample_rate)) if frame_rate else generated_audio_np.shape[1]
                        if full_video_samples > generated_audio_np.shape[1]:
                            pad_amount = full_video_samples - generated_audio_np.shape[1]
                            generated_audio_np = np.pad(
                                generated_audio_np, ((0, 0), (0, pad_amount)), mode="constant"
                            )

                        input_window = extract_audio_window(
                            input_audio, src_start_sec, src_dur_sec, target_dur_sec,
                            sample_rate=audio_sample_rate,
                            channels=generated_audio_np.shape[0],
                        )
                        if input_window is None:
                            # NEVER overwrite with silence -- keep the
                            # regenerated audio untouched on extraction failure.
                            print("[LTX-2.3] vid_outpaint audio window extraction failed; "
                                  "keeping the regenerated audio track (no splice)")
                            try:
                                from api.generation_status import add_warning
                                add_warning(
                                    "preserve_input audio window extraction failed; kept the "
                                    "regenerated audio track unspliced",
                                    code="outpaint_video_audio_extract_failed",
                                )
                            except Exception:
                                pass
                        else:
                            spliced = mux_audio_over_span(
                                generated_audio_np, input_window,
                                offset_sec=offset_sec, dur_sec=target_dur_sec,
                                sample_rate=audio_sample_rate, crossfade_ms=50.0,
                            )
                            audio_out = torch.from_numpy(spliced)
                    except Exception as e:
                        print(f"[LTX-2.3] vid_outpaint audio preserve_input mux failed ({e}); "
                              "keeping the regenerated audio track")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return frames_out, audio_out, audio_sample_rate, seed

    def _generate_txt2vid_ltx2(
        self,
        params: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ):
        """Text-to-video generation with LTX-2.3.

        Returns (frames, audio, audio_sample_rate, actual_seed) — see module docstring.
        """
        if not self.ltx2_components:
            raise RuntimeError("LTX-2.3 components not loaded. Please load an LTX-2.3 model first.")

        # Resolve parameters (moved up: style transfer needs width/height to
        # VAE-encode the reference at the target resolution before the
        # offload/wrap decision below).
        prompt = params.get("prompt", "") or ""
        negative_prompt = params.get("negative_prompt", "") or ""
        width = int(params.get("width", 768))
        height = int(params.get("height", 512))
        num_frames = int(params.get("num_frames", 121))
        frame_rate = float(params.get("frame_rate", 24.0))
        num_inference_steps = int(params.get("num_inference_steps", 8))
        guidance_scale = float(params.get("guidance_scale", 1.0))
        num_videos_per_prompt = int(params.get("num_videos_per_prompt", 1))
        max_sequence_length = int(params.get("max_sequence_length", 1024))
        audio_enable = bool(params.get("audio_enable", True))

        # Training-free reference-style transfer: resolve BEFORE FBCache/
        # Spectrum/Block-Swap so their build calls see the (possibly
        # force-disabled) params (see core.inference.style_ltx2 module
        # docstring's interop section / _ltx2_resolve_style).
        style_active = self._ltx2_resolve_style(params)

        blocks_to_swap = int(params.get("blocks_to_swap", 0) or 0)

        # AP2 First-Block-Cache: build before the offload/wrap step so the wrap
        # decision (force the wrapper on for FBCache-only / Spectrum-only /
        # style-only) is known up front. Mutually exclusive with Block Swap and
        # Spectrum (see _ltx2_build_fbcache; Spectrum takes precedence over
        # FBCache) and with style transfer (force-disabled by
        # _ltx2_resolve_style above, so fbcache_active(params) is already False
        # here whenever style_active).
        fbcache = self._ltx2_build_fbcache(params, blocks_to_swap > 0)
        num_inference_steps_probe = num_inference_steps
        spectrum_video, spectrum_audio = self._ltx2_build_spectrum(
            params, num_inference_steps_probe, blocks_to_swap > 0
        )
        force_wrap = (fbcache is not None or spectrum_video is not None or style_active) and blocks_to_swap <= 0

        # One-time in-place INT8 conversion, before the offloader is built (see
        # _ltx2_runtime_int8; same ordering as the other two generate paths).
        self._ltx2_runtime_int8(params, progress_callback=progress_callback,
                                force_wrap=force_wrap)

        pipeline = self._ensure_ltx2_swap_and_offload(blocks_to_swap, force_wrap=force_wrap)

        from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper
        style_target = pipeline.transformer if isinstance(pipeline.transformer, Ltx2BlockLoopWrapper) else None
        fbcache_target = pipeline.transformer if isinstance(pipeline.transformer, Ltx2BlockLoopWrapper) else None
        if fbcache is not None and fbcache_target is not None:
            fbcache_target.attach_fbcache(fbcache)
        elif fbcache is not None:
            print("[FBCache] LTX-2.3: could not attach (transformer not wrapped)")
            fbcache = None
        spectrum_target = None
        if spectrum_video is not None and fbcache_target is not None:
            spectrum_target = fbcache_target
            spectrum_target.attach_spectrum(spectrum_video, spectrum_audio)
        elif spectrum_video is not None:
            print("[Spectrum] LTX-2.3: could not attach (transformer not wrapped)")
            spectrum_video = spectrum_audio = None

        # Training-free reference-style transfer: install patched attn1
        # processors on the INNER (unwrapped) transformer and attach the
        # (cfg, ref_x0, eps_ref) triple to the wrapper. See
        # core.inference.style_ltx2 module docstring for the full design.
        style_processors: List[Any] = []
        style_saved_processors: List[Any] = []
        style_cfg = None
        style_refs = None
        style_combine_mode = "stack"
        if style_active and style_target is not None:
            device = next(style_target.transformer.parameters()).device
            style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                self._ltx2_style_configs(params, width, height, device)
            if style_cfg is not None or style_refs is not None:
                from core.inference.style_ltx2 import install_ltx2_style_processors
                style_processors, style_saved_processors = install_ltx2_style_processors(style_target.transformer)
                style_target.attach_style(
                    style_processors, style_cfg, style_ref_x0, style_eps_ref,
                    style_refs=style_refs, combine_mode=style_combine_mode,
                    guidance_scale=guidance_scale,
                )
                style_target._style_total_steps = num_inference_steps
                ref_suffix = f" ({len(style_refs)} references, combine={style_combine_mode})" if style_refs else ""
                print(f"[StyleLTX2] Style transfer active: {len(style_processors)} attn1 processors stamped{ref_suffix}")
        elif style_active:
            print("[StyleLTX2]: could not attach (transformer not wrapped)")
            style_cfg = None
            style_refs = None

        # Seed: -1 (or negative/None) -> random draw, recorded back for the caller.
        seed = params.get("seed", -1)
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = -1
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Progress + FBCache/Spectrum/style step advance: LTX2Pipeline invokes
        # callback_on_step_end(pipe, i, t, kwargs) AFTER every denoise step. We
        # advance _fbcache_step / _spectrum_step / _style_step_idx to i+1 there
        # (primed for the NEXT step's forward call); step 0 uses the wrapper's
        # default (0) from attach_fbcache() / attach_spectrum() / attach_style().
        style_wants_step_advance = style_cfg is not None or style_refs is not None
        if progress_callback is not None or fbcache_target is not None or spectrum_target is not None or style_wants_step_advance:
            def _cb(pipe, step_index, timestep, callback_kwargs):
                if progress_callback is not None:
                    try:
                        progress_callback(step_index + 1, num_inference_steps)
                    except Exception:
                        pass
                if fbcache_target is not None:
                    fbcache_target._fbcache_step = step_index + 1
                if spectrum_target is not None:
                    spectrum_target._spectrum_step = step_index + 1
                if style_wants_step_advance and style_target is not None:
                    style_target._style_step_idx = step_index + 1
                return callback_kwargs
            callback = _cb
        else:
            callback = None

        print(f"[LTX-2.3] txt2vid: {width}x{height} num_frames={num_frames} "
              f"fps={frame_rate} steps={num_inference_steps} cfg={guidance_scale} "
              f"seed={seed} audio={audio_enable}")

        try:
            # LoRA: applied here (inside the try) so a failure in _load_lora_ltx2
            # still restores the FBCache/Spectrum/style state attached above, and
            # unloaded in the finally below. See _load_lora_ltx2 for the ordering
            # and block-swap contracts.
            self._load_lora_ltx2(params.get("loras"))

            video, audio = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                output_type="np",
                return_dict=False,
                max_sequence_length=max_sequence_length,
                callback_on_step_end=callback,
            )
        finally:
            self._unload_lora_ltx2()
            if fbcache_target is not None:
                print(f"[FBCache] LTX-2.3 summary: {fbcache.n_hits} hit(s), {fbcache.n_miss} miss(es)")
                fbcache_target.attach_fbcache(None)
            if style_saved_processors:
                # Restore/patch-removal + context clear MUST run on exception too
                # (finally), else style state leaks into the next generation --
                # mirrors the FLUX.2 audit finding.
                from core.inference.style_ltx2 import restore_ltx2_style_processors
                if style_target is not None:
                    style_target.attach_style(None, None, None, None)
                restore_ltx2_style_processors(style_saved_processors)
                print("[StyleLTX2] processors restored (generation complete)")
            if spectrum_target is not None:
                v_stats = spectrum_video.stats()
                print(f"[Spectrum] LTX-2.3 summary: {v_stats['anchors']} anchor(s), "
                      f"{v_stats['forecasts']} forecast(s) of {v_stats['total']} step(s)")
                spectrum_target.attach_spectrum(None, None)

        # video: np.ndarray [B, T, H, W, C] float in [0, 1] (output_type="np").
        frames_np = video[0]  # [T, H, W, C]
        frames = (np.clip(frames_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        # audio: torch.Tensor [B, channels, samples]; take batch 0 -> [channels, samples].
        audio_out = None
        audio_sample_rate = None
        if audio_enable and audio is not None:
            try:
                audio_sample_rate = int(pipeline.vocoder.config.output_sampling_rate)
            except Exception:
                audio_sample_rate = 24000
            try:
                audio_out = audio[0].detach().float().cpu()
            except Exception as e:
                print(f"[LTX-2.3] audio extraction failed ({e}); saving video without audio")
                audio_out = None
                audio_sample_rate = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return frames, audio_out, audio_sample_rate, seed
