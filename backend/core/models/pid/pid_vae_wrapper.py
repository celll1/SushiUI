"""VAE-like adapter that routes SushiUI's per-generation VAE-override slot to the
PiD (Pixel Diffusion Decoder) SDXL 4-step distilled student, with a real SDXL
``AutoencoderKL`` held underneath for everything that is NOT the final Stage-3
decode.

Not vendored from NVIDIA source — original SushiUI integration code. See
``scratchpad/pid_integration_design.md`` §6/§7b/§8 (F1/F2/F3/F7/F8) for the full
design rationale; the bullet list below is the implementation of those findings.

F2 — FAIL-SAFE ROUTING (HIGH): a VAE override hijacks EVERY ``pipeline.vae`` call,
not just the final image decode. Mid-pipeline ``.encode()``/``.decode()`` calls
(inloop_hard_flatten's decode+encode roundtrip, compute_vae_dc_bias, ref-guide/
style encodes, img2img/inpaint init + mask-blur encodes) all go through
``pipeline.vae``. PiD has NO encoder and its decode is only valid as the FINAL
Stage-3 super-resolution step, so:
  - ``.encode(...)``          -> ALWAYS delegates to the real held SDXL VAE.
  - ``.decode(...)`` (default) -> ALSO delegates to the real held SDXL VAE.
  - PiD runs ONLY via the distinct ``.pid_final_decode(...)`` method, which the
    3 Stage-3 decode sites in ``custom_sampling.py`` call explicitly when
    ``isinstance(pipeline.vae, PidVaeWrapper)``. A call site that is NOT updated
    to check for ``PidVaeWrapper`` (a missed site) safely falls through to the
    real VAE instead of silently running PiD on a non-final latent.

F1 — RE-NORMALIZE (HIGH correctness): the 3 Stage-3 call sites already unscale
the diffusion-space latent BEFORE calling decode (``latents/scaling_factor +
shift``), because a plain ``AutoencoderKL.decode()`` expects the RAW (unscaled)
latent. PiD, however, was trained on the diffusers-NORMALIZED latent (``z' = 0.13025
* (z - shift)``). So ``pid_final_decode`` re-applies that same normalization
internally to recover PiD's expected frame — see the module-level docstring on
``pid_final_decode`` for the assertion this produces (std ~0.6-1.0, not ~5-8).

F3 — VRAM staging: the PiD net (~2.7GB bf16) is built lazily on CPU on first use
and cached on the wrapper instance across calls (never reloaded per-generation);
it is staged to GPU only for the duration of ``pid_final_decode`` and staged back
to CPU in a ``finally`` block, mirroring ``move_vae_to_gpu``/``move_vae_to_cpu``'s
timing. The real held VAE's device moves are NOT touched by this — ``.to()``
delegates to the real VAE only, so the existing ``move_vae_to_gpu``/
``move_vae_to_cpu`` funnel keeps working transparently for the non-PiD encode/
decode traffic. When ``pid_use_gemma`` is set, Gemma is loaded, used once, and
freed BEFORE the PiD net is staged to GPU (sequential, never resident together).

F7 — input-resolution cap (HIGH correctness, A/B-confirmed 2026-07): PiD's SDXL
checkpoint is trained for the canonical "1024 LDM -> 4K" range. Decoding at
NATIVE resolution (``latent_h/w * 8``) above ``SAFE_NATIVE`` is
out-of-distribution and produces real artifacts, not just a quality-vs-speed
tradeoff — A/B-tested via the real held VAE on the identical latent: native
1024px is clean; native 1216px produces full-frame rainbow streaking; native
1344px produces a green-yellow gradient band across the bottom ~15-20% of the
frame (per-band G-R flips from -19 to +20). The same latent decoded through the
plain SDXL ``AutoencoderKL`` at native 1344px is clean, so this is PiD-specific
and resolution-driven, not a base-model or scaling-code defect. ``native_cap``
(default ``SAFE_NATIVE``, 1024px) is therefore a REAL cap on the native
resolution PiD is ever asked to decode at, not a warn-and-proceed threshold.
When the FULL latent's native exceeds it, the DEFAULT response is the F9
TILED decode below (every individual tile's native stays <= ``native_cap``,
so PiD itself never runs OOD, while the full requested output size is still
produced at true super-resolution detail). The original whole-latent
downscale-then-decode-then-upscale (bicubic + antialias, aspect-preserving)
is kept as the ``fast_large_decode=True`` opt-out for callers who want the
cheaper, blurrier single-pass path. Passing a larger ``native_cap`` (e.g. a
very high value) is a trivial opt-out for callers who want uncapped OOD
native on either path.

F8: the caller's generation seed is threaded into ``generate_samples_from_batch``
so PiD's noise draw is reproducible per-seed like the rest of the pipeline.

F9 — TILED decode for native > native_cap (R&D-proven, replaces cap+bicubic as
the DEFAULT large-output path): downscaling the whole LQ latent (F7's
cap+bicubic path) keeps PiD in-distribution but throws away detail — it is
now the FAST OPT-OUT (``fast_large_decode=True``), not the default. The
default for native > ``native_cap`` is instead to split the (renormalized) LQ
latent into OVERLAPPING tiles, each tile's own native resolution
(``tile_lat * 8``) capped at ``tile_native`` (<= ``native_cap``, so every tile
individually stays in-distribution), run the SAME 4-step PiD decode on each
tile (same seed across tiles, per the prototype — identical shape means an
identical initial-noise draw, which is what gives adjacent tiles their
boundary continuity), and feather-blend the 4x tile outputs back into the
single full-resolution canvas at pixel scale (overlap ratio
``tile_overlap_ratio``, 25% confirmed seam-free on both busy and smooth
backgrounds in the R&D probe). This gives true super-resolution detail at
the full requested output size, at the cost of ~6x more decode passes
(bounded per-tile VRAM, ~6.5GB) instead of one. Tile geometry + blend math
are shared with ``core.upscaler``'s diffusion tile upscale via
``core.utils.tile_blend`` (see that module's docstring for why it is kept
free of ``api.*`` imports).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

# PiD's `sr_scale` is baked into the SDXL distilled checkpoint this wrapper
# targets (`PiD_res2kto4k_sr4x_official_sdxl_distill_4step.pth`, PID_SR4X net
# config `sr_scale=4`) — not read dynamically because the "4x" output size is
# needed before the (lazily-constructed) net object exists.
SR_SCALE = 4

# A/B-confirmed (2026-07) safe native-resolution ceiling for this checkpoint's
# canonical "1024 LDM -> 4K" training range (see the F7 note in the module
# docstring for the artifact evidence). Named module constant so it is easy to
# retune if a future PiD checkpoint ships with a different trained range.
SAFE_NATIVE = 1024

# R&D-confirmed (2026-07, F9) defaults for the tiled large-output decode path:
# each tile's own native resolution (well inside SAFE_NATIVE) and the feather
# overlap ratio (in latent space, as a fraction of the tile size) that was
# seam-free on both busy and smooth backgrounds in the tiling probe.
DEFAULT_TILE_NATIVE = 512
DEFAULT_TILE_OVERLAP_RATIO = 0.25

_NULL_ASSET_PATH = Path(__file__).resolve().parent / "assets" / "pid_sdxl_null_caption.npy"


def _warn_pid(message: str, code: str = "pid_decoder_warning") -> None:
    try:
        from api.generation_status import add_warning
        add_warning(message, code=code)
    except Exception:
        pass
    print(f"[PidVaeWrapper] WARNING: {message}")


class PidVaeWrapper:
    """Drop-in ``pipeline.vae`` replacement: real SDXL VAE + PiD final decode.

    Constructed once by ``pipeline.load_override_vae`` when the override
    candidate's ``kind == "pid_decoder"``; swapped onto the same
    ``pipeline.vae`` slots used by a normal VAE override.
    """

    def __init__(
        self,
        real_vae,
        pid_pth_path: str,
        pid_sr_output: str = "4x",
        pid_use_gemma: bool = False,
        native_cap: int = SAFE_NATIVE,
        low_vram_decode: bool = False,
        tile_native: int = DEFAULT_TILE_NATIVE,
        tile_overlap_ratio: float = DEFAULT_TILE_OVERLAP_RATIO,
        fast_large_decode: bool = False,
    ):
        if pid_sr_output not in ("4x", "original"):
            raise ValueError(f"pid_sr_output must be '4x' or 'original', got {pid_sr_output!r}")

        self.real_vae = real_vae
        # Defensive re-apply: the caller (pipeline.load_override_vae) should have
        # already fixed a bare-.safetensors SDXL VAE's scaling_factor (see
        # `from_single_file` gotcha in the module docstring / design doc §7b), but
        # re-assert it here too so `.config.scaling_factor` is right even if a
        # future call site constructs the wrapper directly with an unfixed VAE.
        try:
            real_vae.register_to_config(scaling_factor=0.13025, shift_factor=0.0)
        except Exception:
            pass

        self.pid_pth_path = pid_pth_path
        self.pid_sr_output = pid_sr_output
        self.pid_use_gemma = pid_use_gemma
        # F7 — real cap (not a warn-only soft ceiling): native above this triggers
        # downscale-then-decode-then-upscale in `pid_final_decode` (see module
        # docstring). Default is `SAFE_NATIVE`; pass a larger value to opt out.
        self.native_cap = native_cap
        # SushiUI VRAM deviation (not upstream): opt-in low-VRAM decode.
        # False (default) = PiTBlock/FinalLayer run their exact original,
        # unchunked forward (bit-identical to a plain PiD checkout). True
        # enables the row-chunked activation path (see
        # `pixeldit_official.PixDiT_T2I.set_vram_chunk_rows` /
        # `_DEFAULT_VRAM_CHUNK_ROWS`): ~6.6GB/42% less activation peak at
        # 4096px decode (measured), at the cost of bf16 GEMM-tiling rounding
        # drift that is NOT bit-identical (verified bit-identical in fp32;
        # amplified by bf16's coarse precision through the 4-step SDE
        # sampler — see scratchpad/pid_vram_proposal.md). Applied fresh on
        # every `pid_final_decode` call (not just at construction) so an
        # idempotent update (see `pipeline.load_override_vae`) takes effect
        # on the very next decode.
        self.low_vram_decode = low_vram_decode

        # F9 — tiled large-output decode (default path when native > native_cap):
        # tile_native is each tile's own native-resolution ceiling (must stay
        # <= native_cap so every tile is individually in-distribution; clamped
        # defensively in `_tiled_decode` with a warning if misconfigured above
        # it). tile_overlap_ratio is the feather overlap as a fraction of the
        # tile size, applied in latent space before scaling to pixel space for
        # the blend. fast_large_decode=True opts OUT of tiling back to the
        # original whole-latent cap+bicubic path (F7) for large outputs.
        self.tile_native = tile_native
        self.tile_overlap_ratio = tile_overlap_ratio
        self.fast_large_decode = fast_large_decode

        self.current_prompt: Optional[str] = None  # set via set_prompt() before a pid_use_gemma decode

        self._pid_model = None          # lazily-built PidInferenceModel, cached across calls
        self._pid_device = "cpu"        # PiD net's current device (staged independently of real_vae)
        self._null_embs_cache: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # VAE-like surface — delegates to the real held VAE (F2 default routing)
    # ------------------------------------------------------------------

    @property
    def config(self):
        return self.real_vae.config

    @property
    def dtype(self):
        # FAIL-SAFE (F2): delegate to the real VAE's dtype rather than hardcoding
        # bf16. Every non-final call site casts `latents.to(dtype=pipeline.vae.dtype)`
        # BEFORE dispatching to encode/decode; if this reported a fixed bf16, a
        # non-PiD decode/encode against the real (fp16/fp32) VAE would receive
        # mismatched-dtype latents. `pid_final_decode` re-casts to bf16 internally
        # regardless of the dtype it is handed, so delegating here is correct for
        # both paths.
        return self.real_vae.dtype

    @property
    def device(self):
        # FAIL-SAFE (F2): several call sites read pipeline.vae.device (e.g.
        # move_vae_to_gpu, inloop_hard_flatten). Delegate to the real held VAE so
        # those keep working; without this they hit AttributeError (silently
        # swallowed at some sites, disabling the feature).
        try:
            return next(self.real_vae.parameters()).device
        except StopIteration:
            return getattr(self.real_vae, "device", torch.device("cpu"))

    def to(self, *args, **kwargs):
        """Delegates to the real VAE only — the existing move_vae_to_gpu/cpu
        funnel keeps staging it transparently. The PiD net is staged separately
        (see `pid_final_decode`), only for the duration of the final decode."""
        self.real_vae.to(*args, **kwargs)
        return self

    def parameters(self, *args, **kwargs):
        return self.real_vae.parameters(*args, **kwargs)

    def eval(self):
        self.real_vae.eval()
        return self

    def encode(self, *args, **kwargs):
        """F2: PiD has no encoder — always delegates to the real held SDXL VAE."""
        return self.real_vae.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """F2 default routing: delegates to the real held SDXL VAE. PiD only runs
        through the distinct `pid_final_decode` entry point."""
        return self.real_vae.decode(*args, **kwargs)

    def set_prompt(self, prompt: Optional[str]) -> None:
        """Record the current generation's raw text prompt, consulted by
        `pid_final_decode` only when `pid_use_gemma=True`. Call this once per
        generation before the sampling loop runs (see generation_overrides.apply_overrides)."""
        self.current_prompt = prompt

    # ------------------------------------------------------------------
    # Lazy PiD net construction + GPU/CPU staging (F3)
    # ------------------------------------------------------------------

    def _ensure_pid_model(self):
        if self._pid_model is None:
            from core.models.pid.loader import load_pid_sdxl_decoder
            print(f"[PidVaeWrapper] Loading PiD SDXL decoder (lazy, CPU) from {self.pid_pth_path}")
            self._pid_model = load_pid_sdxl_decoder(self.pid_pth_path, device="cpu", load_text_encoder=False)
            self._pid_device = "cpu"
        return self._pid_model

    def _stage_pid_gpu(self):
        model = self._ensure_pid_model()
        if self._pid_device != "cuda":
            import time
            t0 = time.time()
            print("[PidVaeWrapper] Staging PiD net to GPU for decode...")
            # Flag set BEFORE the move: nn.Module.to() moves parameters one at a
            # time, so a mid-move OOM leaves some on cuda -- a flag still reading
            # "cpu" would make the offload below skip them permanently.
            self._pid_device = "cuda"
            model.net.to("cuda")
            print(f"[PidVaeWrapper] PiD net staged to GPU in {time.time() - t0:.2f}s")
        return model

    def _stage_pid_cpu(self):
        # Unconditional on purpose: .to("cpu") on an already-CPU module is a
        # no-op, and the flag cannot be trusted after a partial stage.
        if self._pid_model is not None:
            if self._pid_device != "cpu":
                print("[PidVaeWrapper] Offloading PiD net to CPU...")
            self._pid_model.net.to("cpu")
            self._pid_device = "cpu"
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def unload(self):
        """Fully drop the cached PiD net (both CPU and GPU memory)."""
        if self._pid_model is not None:
            del self._pid_model
            self._pid_model = None
            self._pid_device = "cpu"
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Caption embedding sources
    # ------------------------------------------------------------------

    def _load_null_caption_embs(self) -> torch.Tensor:
        if self._null_embs_cache is None:
            arr = np.load(str(_NULL_ASSET_PATH))
            self._null_embs_cache = torch.from_numpy(arr).to(torch.bfloat16)
        return self._null_embs_cache.to("cuda")

    def _encode_with_gemma(self, prompt: str) -> Optional[torch.Tensor]:
        """Load Gemma-2-2b-it (ungated mirror), encode `prompt` through the
        vendored `_encode_text_raw` path for real, then free Gemma immediately
        (sequential VRAM staging — never resident at the same time as the PiD
        net's own GPU stage). Returns None (caller falls back to null caption)
        on any resolution/load/encode failure — never raises."""
        try:
            from core.models.common.te_store import resolve_te_dir
            te_dir = resolve_te_dir("gemma-2-2b-it")
            if te_dir is None:
                _warn_pid(
                    "pid_use_gemma requested but Efficient-Large-Model/gemma-2-2b-it could not be "
                    "resolved (no local cache and download unavailable/failed); used null caption instead"
                )
                return None

            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[PidVaeWrapper] Loading Gemma-2-2b-it from {te_dir} for prompt encode...")
            tokenizer = None
            text_encoder = None
            model = self._ensure_pid_model()
            try:
                tokenizer = AutoTokenizer.from_pretrained(te_dir)
                tokenizer.padding_side = "right"
                text_encoder = (
                    AutoModelForCausalLM.from_pretrained(te_dir, dtype=torch.bfloat16).get_decoder().to("cuda")
                )
                text_encoder.eval()

                # Temporarily install the real encoder so `_encode_text_raw` runs the
                # exact vendored path (chi_prompt prepend + tokenizer + select_index),
                # bypassing PixelDiTModel's `load_text_encoder=False` construction.
                object.__setattr__(model, "tokenizer", tokenizer)
                object.__setattr__(model, "text_encoder", text_encoder)
                # `_num_chi_tokens` was 0 at construction (no tokenizer was loaded);
                # recompute now that a real tokenizer exists, or chi_prompt truncation
                # math (`max_length_all = num_chi_tokens + model_max_length - 2`) is wrong.
                model._num_chi_tokens = (
                    len(tokenizer.encode(model._chi_prompt_str)) if model._chi_prompt_str else 0
                )

                with torch.no_grad():
                    embs, _ = model._encode_text_raw([prompt])
                embs = embs.detach().to(torch.bfloat16).cpu()
                return embs.to(device="cuda", dtype=torch.bfloat16)
            finally:
                # ALWAYS detach + free Gemma, even if from_pretrained / tokenization /
                # encoding raised or OOM'd — otherwise the ~5GB decoder stays attached
                # to the cached PiD model (resident on GPU) and OOMs the PiD stage next.
                object.__setattr__(model, "tokenizer", None)
                object.__setattr__(model, "text_encoder", None)
                del tokenizer, text_encoder
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"[PidVaeWrapper] Gemma encode failed: {e}")
            _warn_pid(f"pid_use_gemma failed ({e}); used null caption instead")
            return None

    # ------------------------------------------------------------------
    # PiD final decode (the only place PiD actually runs)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_pid_pass(
        model, lq: torch.Tensor, B: int, caption_embs: torch.Tensor, image_size: tuple, seed: int,
        step_callback=None,
    ) -> torch.Tensor:
        """Run one PiD 4-step decode pass on an already in-distribution LQ
        latent (the whole image, the F7 whole-latent cap, or a single F9
        tile — same call shape either way). Returns a `[B, 3, H, W]` float
        tensor in [-1, 1] (still on the model's device).

        step_callback: optional (i:int, total:int) callable forwarded into
            `generate_samples_from_batch` for per-step decode progress."""
        from core.inference.cancellation import raise_if_cancelled

        raise_if_cancelled()
        lq_bf16 = lq.to(dtype=torch.bfloat16, device="cuda")
        data_batch = {
            model.config.input_caption_key: [""] * B,
            "LQ_latent": lq_bf16,
            "degrade_sigma": torch.zeros(B),
        }
        prior_override = model._injected_caption_embs
        try:
            model.set_injected_caption_embs(caption_embs)
            with torch.no_grad():
                out = model.generate_samples_from_batch(
                    data_batch,
                    num_steps=model.config.student_sample_steps,
                    seed=int(seed),
                    image_size=image_size,
                    step_callback=step_callback,
                )
        finally:
            model.set_injected_caption_embs(prior_override)
        return out.squeeze(2)  # [B, 3, 1, H, W] -> [B, 3, H, W]

    @staticmethod
    def _tensor_to_pil_uint8(t: torch.Tensor):
        """`[1, 3, H, W]` float tensor in [-1, 1] -> PIL RGB image (uint8)."""
        from PIL import Image
        arr = t.detach().float().clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy()
        arr = ((arr + 1.0) * 127.5).clip(0, 255).astype("uint8")
        return Image.fromarray(arr, mode="RGB")

    @staticmethod
    def _pil_uint8_to_tensor(img) -> torch.Tensor:
        """PIL RGB image (uint8) -> `[1, 3, H, W]` float tensor in [-1, 1] (CPU)."""
        arr = np.asarray(img.convert("RGB"), dtype="float32")
        arr = (arr / 127.5) - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()

    def _tiled_decode(
        self,
        model,
        lq: torch.Tensor,
        lat_h: int,
        lat_w: int,
        B: int,
        caption_embs: torch.Tensor,
        seed: int,
        target_out_h: int,
        target_out_w: int,
        native_px: int,
        progress_callback=None,
    ) -> torch.Tensor:
        """F9 — DEFAULT path for native > native_cap. Split `lq` into
        overlapping latent-space tiles (each individually in-distribution),
        run `_run_pid_pass` on each, feather-blend the 4x tile outputs into
        the full `(target_out_h, target_out_w)` canvas. Processes each batch
        item independently (tile-blend is a single-image operation) and
        concatenates the results. Returns a `[B, 3, H, W]` float tensor in
        [-1, 1] on CPU.

        progress_callback: optional (cur:int,total:int,label:str) callable;
            reported in GLOBAL units across all tiles (see `total_tiles` below)
            so the bar advances smoothly tile-by-tile instead of resetting.
        """
        from core.utils.tile_blend import compute_tile_boxes, feather_blend_tiles
        from core.inference.cancellation import raise_if_cancelled

        effective_tile_native = min(self.tile_native, self.native_cap)
        if effective_tile_native != self.tile_native:
            _warn_pid(
                f"pid_tile_native={self.tile_native} exceeds native_cap={self.native_cap}; "
                f"clamped to {effective_tile_native} so every tile stays in-distribution.",
                code="pid_tile_native_clamped",
            )
        # Latent-space tile size + overlap (tile_native/8 latent cells; overlap
        # is tile_overlap_ratio of that, R&D-confirmed 0.25 = seam-free).
        tile_lat = max(1, effective_tile_native // 8)
        overlap_lat = max(0, int(round(tile_lat * self.tile_overlap_ratio)))

        boxes_lat = compute_tile_boxes(lat_w, lat_h, tile_lat, overlap_lat)

        # Latent-space boxes -> pixel-space boxes on the 4x output canvas
        # (px_scale = VAE's 8x spatial compression * PiD's baked-in 4x SR).
        px_scale = 8 * SR_SCALE
        boxes_px = [(x1 * px_scale, y1 * px_scale, x2 * px_scale, y2 * px_scale) for (x1, y1, x2, y2) in boxes_lat]
        overlap_px = overlap_lat * px_scale

        total_tiles = B * len(boxes_lat)
        tile_idx = 0  # 0-based, monotonic across the whole b_idx/tile nesting

        batch_out = []
        for b_idx in range(B):
            tile_images = []
            for (x1, y1, x2, y2) in boxes_lat:
                raise_if_cancelled()
                lq_tile = lq[b_idx : b_idx + 1, :, y1:y2, x1:x2]
                th, tw = y2 - y1, x2 - x1
                image_size = (th * 8 * SR_SCALE, tw * 8 * SR_SCALE)
                cap = caption_embs[b_idx : b_idx + 1]

                def _tile_step_cb(i, total, _base=tile_idx):
                    if progress_callback is not None:
                        progress_callback(
                            _base * total + (i + 1),
                            total_tiles * total,
                            f"PiD decode (tile {_base + 1}/{total_tiles})",
                        )

                out_tile = self._run_pid_pass(model, lq_tile, 1, cap, image_size, seed, step_callback=_tile_step_cb)
                tile_images.append(self._tensor_to_pil_uint8(out_tile))
                # Free per-tile transients before the next tile decode (F9 keeps
                # per-tile VRAM bounded — this is the whole point of tiling).
                del out_tile
                torch.cuda.empty_cache()
                tile_idx += 1

            blended = feather_blend_tiles(target_out_w, target_out_h, boxes_px, tile_images, overlap_px)
            batch_out.append(self._pil_uint8_to_tensor(blended))

        _warn_pid(
            f"Large output {target_out_w}x{target_out_h}px decoded via tiled PiD "
            f"({len(boxes_lat)} tiles, native {native_px}px > cap {self.native_cap}px) for true "
            "super-resolution detail.",
            code="pid_tiled_decode",
        )

        return torch.cat(batch_out, dim=0)

    def pid_final_decode(self, latents: torch.Tensor, seed: int = 0, progress_callback=None) -> Any:
        """Run the PiD SDXL 4-step distilled decoder on `latents`.

        Args:
            latents: the SAME already-unscaled tensor the caller was about to
                hand to a plain `vae.decode()` (i.e. `latents/scaling_factor +
                shift`, computed by the Stage-3 site BEFORE this call — see the
                module docstring's F1 note). This method re-applies the SDXL
                normalization internally to recover PiD's expected frame.
            seed: the generation's seed (F8), forwarded to
                `generate_samples_from_batch` for a reproducible noise draw.
            progress_callback: optional (cur:int,total:int,label:str) callable
                for decode-phase progress. None (default) disables decode
                progress reporting entirely (fully backward compatible).

        Native resolution above `native_cap` (F7) is handled one of two ways:
        the DEFAULT (F9) tiles the latent into overlapping in-distribution
        chunks and feather-blends the 4x decode of each back into the full
        requested output size (true super-resolution detail); setting
        `fast_large_decode=True` opts back into the original single-pass
        whole-latent downscale-then-decode-then-upscale (cheaper, blurrier).

        Returns:
            A `diffusers.models.autoencoders.vae.DecoderOutput`-shaped object
            (`.sample` attribute) holding a `[B, 3, H, W]` tensor in [-1, 1] —
            matching the `pipeline.vae.decode(...).sample` contract used at
            every Stage-3 call site.
        """
        from diffusers.models.autoencoders.vae import DecoderOutput
        from core.inference.cancellation import raise_if_cancelled

        B, _C, lat_h, lat_w = latents.shape

        # Cancel check + "preparing" progress emit as early as possible, so both
        # cover the Gemma-encode (~5GB load, opt-in) and GPU-staging window below
        # — not just the per-step decode passes further down.
        raise_if_cancelled()
        if progress_callback is not None:
            progress_callback(0, 1, "PiD decode: preparing")

        # F1 — re-normalize: the caller pre-unscaled (raw AutoencoderKL frame);
        # recover PiD's normalized training frame (z' = scaling_factor * (z - shift)).
        shift_factor = getattr(self.config, "shift_factor", None) or 0.0
        scaling_factor = self.config.scaling_factor
        lq = (latents.float() - shift_factor) * scaling_factor
        lq_std = lq.std().item()
        print(
            f"[PidVaeWrapper] F1 re-normalize: incoming latents std={latents.float().std().item():.4f} "
            f"-> LQ_latent std={lq_std:.4f} (expect ~0.6-1.0)"
        )
        if not (0.4 <= lq_std <= 1.3):
            _warn_pid(
                f"PiD LQ_latent std={lq_std:.3f} is outside the expected ~0.6-1.0 band; the held SDXL "
                "VAE's scaling_factor/shift_factor may not be the standard SDXL values (0.13025/0.0)"
            )

        # F7/F9 — native-resolution cap (A/B-confirmed real fix, not a soft
        # warning): native above `native_cap` is out-of-distribution for this
        # checkpoint (see module docstring). Branch below decides the response.
        native_px = max(lat_h, lat_w) * 8
        target_out_h, target_out_w = lat_h * 8 * SR_SCALE, lat_w * 8 * SR_SCALE
        over_cap = native_px > self.native_cap

        # Caption embedding: real Gemma prompt (opt-in) or the shipped null asset.
        caption_embs = None
        if self.pid_use_gemma:
            if self.current_prompt:
                caption_embs = self._encode_with_gemma(self.current_prompt)
            else:
                _warn_pid("pid_use_gemma is enabled but no prompt was recorded at decode time; used null caption instead")
        if caption_embs is None:
            caption_embs = self._load_null_caption_embs()

        # Keep caption/LQ batch dims aligned. SushiUI's per-image decode is B=1, but
        # the null asset (and a single-prompt Gemma encode) is [1, T, C]; expand so a
        # B>1 latent never diverges from the caption batch.
        if caption_embs.shape[0] == 1 and B > 1:
            caption_embs = caption_embs.expand(B, *caption_embs.shape[1:]).contiguous()

        model = self._stage_pid_gpu()
        # SushiUI VRAM deviation (not upstream): apply the current
        # low_vram_decode flag on every decode (not just at construction) so
        # an idempotent flag update on an already-cached net takes effect
        # immediately. None (disabled) restores the exact original,
        # unchunked PiTBlock/FinalLayer forward.
        if self.low_vram_decode:
            from core.models.pid.networks.pixeldit_official import _DEFAULT_VRAM_CHUNK_ROWS
            model.net.set_vram_chunk_rows(_DEFAULT_VRAM_CHUNK_ROWS)
        else:
            model.net.set_vram_chunk_rows(None)
        try:
            if over_cap and self.fast_large_decode:
                # F7 FAST OPT-OUT: whole-latent downscale-then-decode-then-upscale
                # (opt-in via fast_large_decode=True; default is F9 tiling below).
                scale = self.native_cap / native_px
                cap_lat_h = max(1, round(lat_h * scale))
                cap_lat_w = max(1, round(lat_w * scale))
                capped_native_px = max(cap_lat_h, cap_lat_w) * 8
                _warn_pid(
                    f"PiD native capped to {capped_native_px}px for quality (was {native_px}px, out of the "
                    f"checkpoint's trained ~{SAFE_NATIVE}px range); output resized to the requested "
                    f"{target_out_w}x{target_out_h}.",
                    code="pid_native_capped",
                )
                lq_capped = F.interpolate(
                    lq, size=(cap_lat_h, cap_lat_w), mode="bicubic", align_corners=False, antialias=True
                )
                image_size = (cap_lat_h * 8 * SR_SCALE, cap_lat_w * 8 * SR_SCALE)

                def _step_cb(i, total):
                    if progress_callback is not None:
                        progress_callback(i + 1, total, f"PiD decode (step {i + 1}/{total})")

                out = self._run_pid_pass(model, lq_capped, B, caption_embs, image_size, seed, step_callback=_step_cb)
                # F7 top-up: PiD ran on the capped (in-distribution) latent above;
                # upscale its clean output back to the originally-requested 4x
                # output size. Bicubic + antialias (torch has no native "lanczos"
                # interpolate mode); this is a minor top-up on top of PiD's own
                # 4x super-resolution detail, not a substitute for it.
                out = F.interpolate(
                    out, size=(target_out_h, target_out_w), mode="bicubic", align_corners=False, antialias=True
                )
            elif over_cap:
                # F9 DEFAULT: tiled decode for true super-resolution detail at
                # the full requested output size (see module docstring / class
                # docstring above `_tiled_decode`).
                out = self._tiled_decode(
                    model, lq, lat_h, lat_w, B, caption_embs, seed, target_out_h, target_out_w, native_px,
                    progress_callback=progress_callback,
                )
            else:
                # Direct in-distribution single-pass decode (native <= native_cap).
                image_size = (lat_h * 8 * SR_SCALE, lat_w * 8 * SR_SCALE)

                def _step_cb(i, total):
                    if progress_callback is not None:
                        progress_callback(i + 1, total, f"PiD decode (step {i + 1}/{total})")

                out = self._run_pid_pass(model, lq, B, caption_embs, image_size, seed, step_callback=_step_cb)

            if self.pid_sr_output == "original":
                # F7: this is output-size control, NOT a cheaper mode — the full
                # 4x super-resolution decode always runs; sr_scale=4 is baked
                # into this checkpoint. Downscale AFTER the full decode (and
                # after the F7/F9 branch above, so this always targets the
                # ORIGINALLY requested native size, `target_out / SR_SCALE`,
                # regardless of which branch produced `out`).
                out = F.interpolate(
                    out, size=(target_out_h // SR_SCALE, target_out_w // SR_SCALE),
                    mode="bilinear", align_corners=False, antialias=True,
                )
        finally:
            self._stage_pid_cpu()

        return DecoderOutput(sample=out)
