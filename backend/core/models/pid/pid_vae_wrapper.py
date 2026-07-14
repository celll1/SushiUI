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

F7 — input-resolution cap: native SDXL resolution (``latent_h/w * 8``) above
``native_cap`` (default ~1280px) triggers a warning (not a hard refuse — this
override is opt-in and generation should still complete); quadratic attention
cost and PiD's 2k-4k training range make very large native inputs both slow and
out-of-distribution.

F8: the caller's generation seed is threaded into ``generate_samples_from_batch``
so PiD's noise draw is reproducible per-seed like the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# PiD's `sr_scale` is baked into the SDXL distilled checkpoint this wrapper
# targets (`PiD_res2kto4k_sr4x_official_sdxl_distill_4step.pth`, PID_SR4X net
# config `sr_scale=4`) — not read dynamically because the "4x" output size is
# needed before the (lazily-constructed) net object exists.
SR_SCALE = 4

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
        native_cap: int = 1280,
        low_vram_decode: bool = False,
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
            model.net.to("cuda")
            self._pid_device = "cuda"
            print(f"[PidVaeWrapper] PiD net staged to GPU in {time.time() - t0:.2f}s")
        return model

    def _stage_pid_cpu(self):
        if self._pid_model is not None and self._pid_device != "cpu":
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

    def pid_final_decode(self, latents: torch.Tensor, seed: int = 0) -> Any:
        """Run the PiD SDXL 4-step distilled decoder on `latents`.

        Args:
            latents: the SAME already-unscaled tensor the caller was about to
                hand to a plain `vae.decode()` (i.e. `latents/scaling_factor +
                shift`, computed by the Stage-3 site BEFORE this call — see the
                module docstring's F1 note). This method re-applies the SDXL
                normalization internally to recover PiD's expected frame.
            seed: the generation's seed (F8), forwarded to
                `generate_samples_from_batch` for a reproducible noise draw.

        Returns:
            A `diffusers.models.autoencoders.vae.DecoderOutput`-shaped object
            (`.sample` attribute) holding a `[B, 3, H, W]` tensor in [-1, 1] —
            matching the `pipeline.vae.decode(...).sample` contract used at
            every Stage-3 call site.
        """
        from diffusers.models.autoencoders.vae import DecoderOutput

        B, _C, lat_h, lat_w = latents.shape

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

        native_px = max(lat_h, lat_w) * 8
        if native_px > self.native_cap:
            _warn_pid(
                f"PiD decode requested at native {native_px}px (> {self.native_cap}px cap); proceeding, "
                "but attention cost grows quadratically and this exceeds PiD's ~2k-4k training range"
            )

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
            lq_bf16 = lq.to(dtype=torch.bfloat16, device="cuda")
            data_batch = {
                model.config.input_caption_key: [""] * B,
                "LQ_latent": lq_bf16,
                "degrade_sigma": torch.zeros(B),
            }
            image_size = (lat_h * 8 * SR_SCALE, lat_w * 8 * SR_SCALE)

            prior_override = model._injected_caption_embs
            try:
                model.set_injected_caption_embs(caption_embs)
                with torch.no_grad():
                    out = model.generate_samples_from_batch(
                        data_batch,
                        num_steps=model.config.student_sample_steps,
                        seed=int(seed),
                        image_size=image_size,
                    )
            finally:
                model.set_injected_caption_embs(prior_override)

            out = out.squeeze(2)  # [B, 3, 1, H, W] -> [B, 3, H, W]

            if self.pid_sr_output == "original":
                # F7: this is output-size control, NOT a cheaper mode — the full
                # 4x super-resolution decode always runs; sr_scale=4 is baked
                # into this checkpoint. Downscale AFTER the full decode.
                import torch.nn.functional as F
                out = F.interpolate(
                    out, size=(lat_h * 8, lat_w * 8), mode="bilinear", align_corners=False, antialias=True
                )
        finally:
            self._stage_pid_cpu()

        return DecoderOutput(sample=out)
