"""ACE-Step 1.5 (turbo) txt2aud backend mixin for DiffusionPipelineManager.

Phase 2: assembles the conditioning tensors the vendored
`AceStepConditionGenerationModel.generate_audio`
(`backend/core/models/acestep/vendor/modeling_acestep_v15_turbo.py`) expects,
and drives its INTERNAL flow-matching sampling loop. The loop itself
(Euler/ODE integration over the turbo distilled 8-step schedule) is vendored
model code -- it is called, not reimplemented, here.

Conditioning-assembly citations (official `ace-step/ACE-Step-1.5` HEAD 6d467e4,
cached under `scratchpad/ace/` during Phase 2 research; see also
`scratchpad/acestep_txt2aud_recipe.md`):
  - Caption prompt template: `acestep/constants.py` (`SFT_GEN_PROMPT`,
    `DEFAULT_DIT_INSTRUCTION`), mirrored into
    `core.models.acestep.defaults`.
  - Caption tokenize + encode: `conditioning_text.py::_prepare_text_conditioning_inputs`
    (tokenizer max_length=256) + `conditioning_embed.py::infer_text_embeddings`
    (`text_encoder(input_ids=...).last_hidden_state`).
  - Lyric format + embed: `prompt_utils.py::_format_lyrics` (verbatim) +
    `conditioning_text.py` (tokenizer max_length=2048) +
    `conditioning_embed.py::infer_lyric_embeddings`
    (`text_encoder.embed_tokens(ids)` -- embedding table only, no transformer
    forward, no separate lyric tokenizer/LM).
  - Silence timbre + src_latents (plain text2music, no reference audio):
    `conditioning_embed.py::infer_refer_latent` (all-zero refer_audio ->
    `self.silence_latent[:, :750, :]`) and
    `conditioning_target.py::_get_silence_latent_slice` /
    `conditioning_masks.py::_build_chunk_masks_and_src_latents` (no target
    audio -> `src_latents = silence_latent_tiled`). No precomputed
    `silence_latent` asset ships with the local checkpoint (see
    `core.models.acestep.loader` docstring), so it is derived once, per
    loaded model, by VAE-encoding literal zero-amplitude audio (30s / 750
    frames @ 25Hz, matching `AceStepConfig.timbre_fix_frame`).
  - chunk_masks shape: the *vendored* `generate_audio`/`prepare_condition`
    concatenates `chunk_masks.to(dtype)` directly onto the 64-channel
    `src_latents` (`context_latents = cat([src_latents, chunk_masks], -1)`),
    so `chunk_masks` must already be `[B, T, 64]` (not `[B, T]` bool) --
    confirmed against the vendored file's own `test_forward()` fixture.
  - VAE decode + peak normalization:
    `generate_music_decode.py::_decode_generate_music_pred_latents`
    (`vae.decode(pred.transpose(1,2)).sample`; no `scaling_factor`; peak
    normalize only when `amax > 1`).

IMPORTANT: the locally VENDORED `generate_audio` (this repo's
`vendor/modeling_acestep_v15_turbo.py`) is a simpler variant than the
official reference above -- it has no `infer_steps` / `sampler_mode` /
`diffusion_guidance_scale` / DCW / repaint kwargs (turbo bakes CFG into the
distillation and never runs a twin forward pass; the 8-step schedule comes
from the baked-in `SHIFT_TIMESTEPS` table keyed by `shift`, or from an
explicit `timesteps=` override that gets snapped to the nearest valid
distilled timestep). This mixin only passes kwargs the vendored signature
actually declares.
"""

from typing import Dict, Any, Optional, Tuple, Callable
import os
import random
import re

import torch


def _is_lora_target(module) -> bool:
    """True for a module a generation-time ACE-Step LoRA may wrap.

    Thin re-export of the shared
    ``core.training.adapters.base_adapter.is_lora_wrappable_linear``, and it
    exists as a MODULE-LEVEL name for two reasons. First, `Int8Linear` /
    `Fp8Linear` are ``nn.Module``s but NOT ``nn.Linear`` subclasses, so the
    ``isinstance(x, torch.nn.Linear)`` these two call sites used to spell skipped
    every quantized layer silently -- an INT8-converted DiT (which
    ``_acestep_runtime_int8`` now produces on request) would have reported
    "0 of N modules applied" and told the user their LoRA was for a different
    ACE-Step generation. Second, ``quantized_capability_parity_test`` locates an
    arch's predicate BY NAME in this module and exercises it on real
    ``Int8Linear``/``Fp8Linear``/``nn.Linear`` instances; an inline test cannot be
    checked that way.
    """
    from core.training.adapters.base_adapter import is_lora_wrappable_linear

    return is_lora_wrappable_linear(module)


class AceStepMixin:
    """AceStepMixin: ACE-Step 1.5 (2B DiT + Oobleck VAE + Qwen3-Embedding
    text encoder) text-to-music generation backend."""

    # ------------------------------------------------------------------
    # Component staging (sequential text_encoder -> DiT -> VAE; mirrors the
    # `_move` helper pattern used by the other single-file-loaded backends,
    # e.g. MiniT2IMixin._minit2i_move).
    # ------------------------------------------------------------------

    def _acestep_runtime_int8(self, params: Dict[str, Any], progress_callback=None):
        """Apply the one-time in-place INT8 conversion, if this request asks for it.

        No-op for every ``unet_quantization`` value other than ``"int8"``, for an
        already-converted DiT, and for a checkpoint that already carries
        weight-only quantized Linears (see
        ``vram_optimization.apply_runtime_int8_quantization`` for the full
        contract). The conversion replaces child modules in place and never
        builds a second module.

        SCOPE. Only the DiT (``acestep_components["dit"]``, 392 ``nn.Linear``
        modules holding 2.3922 G parameters) is converted. The Oobleck VAE holds
        no 2-D Linear weight at all and the Qwen3-Embedding text encoder is a
        separate component this walk cannot reach; ``arch_capabilities`` declares
        ``text_encoder_quantization`` unsupported for acestep and nothing here
        changes that.

        ORDERING, and why this is called at the top of every generate path but
        AFTER ``_apply_or_clear_lora_acestep``:

        * AFTER the LoRA gate, not before. The converter refuses a LoRA-wrapped
          module (the wrappers hide the Linears, so the selection would differ
          from the offline audit) -- and ACE-Step's wrappers are PERSISTENT state:
          ``_apply_or_clear_lora_acestep`` unloads whatever the PREVIOUS
          generation wrapped and only then loads this request's set. Running the
          conversion first would therefore see a stale wrapper from an earlier
          generation and refuse an INT8 request that carries no LoRAs at all --
          the LTX-2.3 stale-offloader shape, where the refusal's own advice
          ("remove the LoRAs") had already been followed. Running after the gate
          means the wrappers present are exactly the ones THIS request asked for,
          which is the only case where refusing is the right answer.
        * BEFORE staging. The converter is device-aware, so running here (with
          the components still on CPU, which is where ``load_model`` leaves them
          and where each generate path's ``_acestep_move`` finds them) is
          correct; running it after the DiT was moved to CUDA would quantize a
          GPU-resident module for no reason.

        No ``precheck`` is passed: unlike FLUX.2/Ideogram 4/Krea 2/LTX-2.3 there
        is no block offloader on this architecture (no ``blocks_to_swap`` path
        exists in this backend at all), so there is no caller-owned invariant
        that only applies to a real conversion. The LoRA invariant is the shared
        converter's own, checked over the whole component set before the first
        layer is touched.
        """
        from core.vram_optimization import apply_runtime_int8_quantization

        components = getattr(self, "acestep_components", None)
        if not components:
            return
        dit = components.get("dit")
        if dit is None:
            return

        model, converted = apply_runtime_int8_quantization(
            self, dit, "acestep", params.get("unet_quantization"),
            label="ACE-Step DiT", progress_callback=progress_callback)

        # The converter mutates in place and returns the SAME object, so the
        # component reference stays valid; re-assigned anyway so a future
        # converter that returned a new module could not silently strand it.
        components["dit"] = model

        if converted or getattr(self, "_runtime_int8_partial", False):
            # The LoRA loader caches the module it wrapped under
            # ``_acestep_lora_original_modules`` so ``_unload_lora_acestep`` can
            # put it back, and that cache is keyed by module path and never
            # overwritten (``if module_key not in ...``). After a conversion the
            # cached entries are the PRE-conversion bf16 Linears, which are still
            # alive precisely because that dict holds them: a later
            # load-then-unload cycle would restore them, silently un-quantizing
            # those layers (and having kept their bf16 weights resident the whole
            # time). The conversion only runs when nothing is wrapped, so
            # dropping the cache here is safe and simply lets the next LoRA load
            # record the quantized modules as the originals.
            #
            # ``_runtime_int8_partial`` as well as ``converted``, because
            # ``converted`` is False for a PARTIAL conversion too -- the
            # CUDA-OOM-at-layer-N path the converter explicitly designs for
            # (vram_optimization.apply_runtime_int8_quantization sets the latch
            # and returns False). Those layers ARE Int8Linear, so gating on
            # ``converted`` alone left the pre-conversion bf16 modules cached for
            # exactly the layers that were converted: reproduced as
            # "after a LoRA load/unload cycle the converted module is an
            # nn.Linear again", plus 2.4 GB of bf16 held resident by the cache.
            # The same latch is what Anima's hook consults, for the same reason.
            stale = getattr(self, "_acestep_lora_original_modules", None)
            if stale:
                print(f"[AceStep] Dropping {len(stale)} cached pre-quantization LoRA base "
                      f"module(s); future LoRA loads restore the INT8 modules instead.")
                stale.clear()

    def _acestep_move(self, component_name: str, target_device: str):
        comp = self.acestep_components.get(component_name)
        if comp is None or not hasattr(comp, "to"):
            return comp
        try:
            comp.to(target_device)
        except Exception as e:
            print(f"[AceStep] Warning: could not move {component_name} to {target_device}: {e}")
        return comp

    def _acestep_empty_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Silence-latent asset (lazy, cached on self.acestep_components so it
    # survives across generate calls for the currently loaded model, and is
    # naturally invalidated on model reload since acestep_components is
    # replaced wholesale by load_model()).
    # ------------------------------------------------------------------

    def _acestep_ensure_silence_latent(self, device: str) -> torch.Tensor:
        """Return the cached [1, 750, 64] VAE-encoded silence latent, building
        it on first use. See module docstring for the citation trail."""
        cached = self.acestep_components.get("silence_latent")
        if cached is not None:
            return cached.to(device)

        from core.models.acestep.defaults import SAMPLE_RATE, SILENCE_LATENT_FRAMES

        vae = self.acestep_components["vae"]
        was_on = next(vae.parameters()).device
        self._acestep_move("vae", device)
        vae_dtype = next(vae.parameters()).dtype
        duration_sec = SILENCE_LATENT_FRAMES / 25.0  # 30s
        zeros = torch.zeros(
            1, 2, int(round(duration_sec * SAMPLE_RATE)), device=device, dtype=vae_dtype
        )
        with torch.inference_mode():
            # .mode() (deterministic mean), not .sample() -- silence should
            # encode to a fixed, reproducible latent, not a stochastic draw.
            silence_latent = vae.encode(zeros).latent_dist.mode()  # [1, 64, 750]
        silence_latent = silence_latent.transpose(1, 2).contiguous()  # [1, 750, 64]
        self.acestep_components["silence_latent"] = silence_latent.detach().to("cpu")
        self._acestep_move("vae", str(was_on) if was_on is not None else "cpu")
        self._acestep_empty_cache()
        print(f"[AceStep] Built silence-latent asset: shape={tuple(silence_latent.shape)}")
        return silence_latent.to(device)

    @staticmethod
    def _acestep_silence_slice(silence_latent: torch.Tensor, length: int) -> torch.Tensor:
        """Slice or tile the cached silence latent to exactly `length` frames.
        Mirrors `conditioning_target.py::_get_silence_latent_slice`."""
        available = silence_latent.shape[1]
        if length <= available:
            return silence_latent[:, :length, :]
        repeats = (length + available - 1) // available
        tiled = silence_latent[0].repeat(repeats, 1)
        return tiled[:length, :].unsqueeze(0)

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _acestep_build_text_prompt(
        caption: str,
        duration_sec: float,
        bpm: Optional[int],
        key_scale: str,
        time_signature: str,
        vocal_language: str,
    ) -> str:
        from core.models.acestep.defaults import DEFAULT_DIT_INSTRUCTION, SFT_GEN_PROMPT

        # NOTE: the official `_build_metadata_dict`/`_parse_metas` source that
        # renders bpm/key/time_signature/duration/language into the "# Metas"
        # block was not recoverable from the cached reference snapshot used
        # for this port (only its *callers*/consumers were cached, not its
        # body). This key: value block is a best-effort reconstruction from
        # the documented field list (recipe section 2c) -- it is valid text
        # the Qwen3 encoder will happily embed, but may not byte-match the
        # training-time format exactly. Flagged for follow-up if generation
        # quality suggests the metas block is being ignored/misread.
        metas_lines = []
        if bpm:
            metas_lines.append(f"BPM: {bpm}")
        metas_lines.append(f"Duration: {duration_sec:.1f}s")
        if key_scale:
            metas_lines.append(f"Key: {key_scale}")
        if time_signature:
            metas_lines.append(f"Time Signature: {time_signature}")
        metas_lines.append(f"Language: {vocal_language}")
        metas_block = "\n".join(metas_lines)
        return SFT_GEN_PROMPT.format(DEFAULT_DIT_INSTRUCTION, caption, metas_block)

    @staticmethod
    def _acestep_format_lyrics(lyrics: str, vocal_language: str) -> str:
        # Verbatim port of prompt_utils.py::_format_lyrics.
        return f"# Languages\n{vocal_language}\n\n# Lyric\n{lyrics}<|endoftext|>"

    # ------------------------------------------------------------------
    # Reference-audio loading (aud2aud / cover). Citations:
    # `core/generation/handler/io_audio.py::_read_audio_file` (soundfile
    # primary, torchaudio.load() fallback) and
    # `_normalize_audio_to_stereo_48k` (mono->stereo by duplication, [:2],
    # resample iff sr!=48000, clamp[-1,1], NO loudness norm). See
    # `scratchpad/acestep_aud2aud_recipe.md` section 0/5.
    # ------------------------------------------------------------------

    @staticmethod
    def _acestep_load_reference_audio(source) -> Tuple[torch.Tensor, int]:
        """Load a reference audio file (path or raw bytes) as
        ([channels, samples] float32 CPU tensor, sample_rate). soundfile
        (libsndfile) first, torchaudio.load() fallback."""
        try:
            import soundfile as sf
            if isinstance(source, (bytes, bytearray)):
                import io as _io
                data, sr = sf.read(_io.BytesIO(source), dtype="float32", always_2d=True)
            else:
                data, sr = sf.read(source, dtype="float32", always_2d=True)  # [samples, channels]
            wav = torch.from_numpy(data.T).contiguous()  # [channels, samples]
            return wav, int(sr)
        except Exception as e:  # noqa: BLE001 - any failure -> fall through to torchaudio
            print(f"[AceStep] soundfile failed to load reference audio ({type(e).__name__}: {e}); trying torchaudio")

        import torchaudio
        if isinstance(source, (bytes, bytearray)):
            import io as _io
            wav, sr = torchaudio.load(_io.BytesIO(source))
        else:
            wav, sr = torchaudio.load(source)
        return wav.float(), int(sr)

    @staticmethod
    def _acestep_normalize_stereo_48k(wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Mirrors `io_audio.py::_normalize_audio_to_stereo_48k`: mono->stereo
        by duplication, take [:2] channels, resample iff sr != 48000, clamp
        to [-1, 1]. No loudness/RMS normalization."""
        if wav.shape[0] == 1:
            wav = torch.cat([wav, wav], dim=0)
        wav = wav[:2]
        if sr != 48000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, 48000)
            wav = resampler(wav)
        return torch.clamp(wav, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Repaint (inpaint analog) helpers. Citations:
    # `scratchpad/acestep_repaint_feasibility.md` section 5 (recipe) and
    # `core/generation/handler/conditioning_masks.py` /
    # `repaint_step_injection.py` / `repaint_waveform_splice.py` (official
    # reference, NOT vendored -- these are small, pure-tensor, orchestration-
    # layer utilities we reimplement directly here, same pattern already
    # used for the cover-mode conditioning-mask math above).
    # ------------------------------------------------------------------

    @staticmethod
    def _acestep_repaint_frame_range(
        repaint_start: float, repaint_end: float, latent_frames: int
    ) -> Tuple[int, int]:
        """Convert repaint_start/end (seconds) to a [start, end) latent-frame
        range, clamped to the reference's actual length (25 Hz latent rate,
        matching `conditioning_masks.py`'s `sec * sample_rate // 1920` with
        `sample_rate=48000` -- `48000 // 1920 == 25`). Negative starts (the
        official service's "extend before the reference" mode) are NOT
        supported here -- out of scope for "regenerate a time-range of a
        reference clip, keeping the rest"; a negative start simply clamps to 0."""
        s = int(round(max(0.0, repaint_start) * 25.0))
        e = int(round(max(0.0, repaint_end) * 25.0))
        s = max(0, min(s, latent_frames - 1))
        e = max(s + 1, min(e, latent_frames))
        return s, e

    @staticmethod
    def _acestep_apply_repaint_waveform_splice(
        pred_wav: torch.Tensor,
        ref_wav: torch.Tensor,
        start_sec: float,
        end_sec: float,
        sample_rate: int = 48000,
        crossfade_duration: float = 0.01,
    ) -> torch.Tensor:
        """Re-insert the ORIGINAL reference waveform into the KEPT (non-
        repaint) region, with a short linear crossfade at each boundary, so
        only the repainted `[start_sec, end_sec)` region actually changes.
        Reimplementation (not a vendor copy) of the official
        `repaint_waveform_splice.py::apply_repaint_waveform_splice` +
        `_build_waveform_crossfade_mask` -- necessary because the VAE
        reconstruction of the "kept" region still carries reconstruction
        error the latent-level repaint hold (`repaint_mask`/
        `clean_src_latents`, applied inside the vendored `generate_audio`)
        doesn't fully erase.

        Args:
            pred_wav: VAE-decoded waveform, [channels, samples].
            ref_wav: original (pre-VAE) reference waveform, [channels, samples].
            start_sec/end_sec: repaint region boundaries in seconds --
                pass the LATENT-frame-derived boundaries (`frame / 25.0`),
                not the raw user input, so the sample-domain splice boundary
                exactly matches the latent-domain repaint_mask boundary.
            sample_rate: audio sample rate (48000 for ACE-Step).
            crossfade_duration: crossfade length in seconds (default 0.01 = 10ms,
                matching the official default).

        Returns:
            Spliced waveform, same shape as `pred_wav`.
        """
        min_samples = min(pred_wav.shape[-1], ref_wav.shape[-1])
        pred = pred_wav[..., :min_samples]
        ref = ref_wav[..., :min_samples].to(device=pred.device, dtype=pred.dtype)

        start_sample = max(0, min(int(round(start_sec * sample_rate)), min_samples))
        end_sample = max(start_sample, min(int(round(end_sec * sample_rate)), min_samples))
        crossfade_samples = int(crossfade_duration * sample_rate)

        if start_sample == 0 and end_sample >= min_samples:
            # Whole-clip repaint -- nothing outside the region to splice back in.
            result = pred.clone()
        else:
            mask = torch.zeros(min_samples, device=pred.device, dtype=pred.dtype)
            mask[start_sample:end_sample] = 1.0
            if crossfade_samples > 0:
                fade_start = max(start_sample - crossfade_samples, 0)
                ramp_len = start_sample - fade_start
                if ramp_len > 0:
                    mask[fade_start:start_sample] = torch.linspace(
                        0.0, 1.0, ramp_len + 2, device=pred.device
                    )[1:-1]
                fade_end = min(end_sample + crossfade_samples, min_samples)
                ramp_len = fade_end - end_sample
                if ramp_len > 0:
                    mask[end_sample:fade_end] = torch.linspace(
                        1.0, 0.0, ramp_len + 2, device=pred.device
                    )[1:-1]
            m = mask.unsqueeze(0).expand_as(pred)
            result = m * pred + (1.0 - m) * ref

        if pred_wav.shape[-1] > min_samples:
            result = torch.cat([result, pred_wav[..., min_samples:]], dim=-1)
        return result

    # ------------------------------------------------------------------
    # Outpaint (extend) helper -- the structural INVERSE of the repaint
    # waveform splice above. Repaint holds the region OUTSIDE its window and
    # its 10ms crossfade deliberately encroaches INTO that kept region
    # (`fade_start = start_sample - crossfade_samples` above, i.e. the ramp
    # eats into the region that is otherwise pure `ref`). Outpaint instead
    # holds the ENTIRE placed-input span with NO encroachment: mask=0 (=
    # original input) covers every sample in `[start_sample, end_sample)`
    # unconditionally, and the short declick ramp is placed ENTIRELY on the
    # GENERATED side of each boundary (outside the span). Unlike repaint
    # (whose `ref_wav` covers the FULL clip, so the crossfade can blend
    # against real reference samples on both sides of its window), outpaint's
    # `input_wave` covers ONLY the placed span itself -- there is no
    # reference audio beyond its edges to blend against. The ramp therefore
    # fades the generated waveform's amplitude toward the exact boundary
    # sample value (a DC/level-match fade, not a content cross-mix), which
    # removes the audible click at the hard seam without touching a single
    # preserved sample.
    # ------------------------------------------------------------------

    @staticmethod
    def _acestep_apply_outpaint_waveform_splice(
        generated_wave: torch.Tensor,
        input_wave: torch.Tensor,
        offset_sec: float,
        dur_sec: float,
        sample_rate: int = 48000,
        crossfade_ms: float = 10.0,
    ) -> torch.Tensor:
        """Re-insert the ORIGINAL (pre-VAE, trimmed) input waveform into the
        HELD span starting at `offset_sec` of the generated (outpainted)
        track, sample-exact to the decoded 48kHz/16-bit representation, with
        a short declick ramp confined entirely to the GENERATED side of each
        boundary.

        Args:
            generated_wave: VAE-decoded full-timeline waveform,
                [channels, samples].
            input_wave: the original (pre-VAE), already-trimmed input
                waveform that was placed, [channels, samples]. Its OWN
                sample count is authoritative for how many samples get
                overwritten -- `dur_sec` (the LATENT-domain hold length,
                `t_ref / 25.0`) is used only to derive `offset_sec`'s
                counterpart in the caller and is NOT used here to cap the
                spliced length. This matters because the ACE-Step VAE
                encoder may FLOOR when computing `t_ref` from the input's
                true sample count: `t_ref * (sample_rate / 25)` can then be
                *shorter* than `input_wave.shape[-1]` by up to ~1 latent
                frame (~40ms @ 48kHz). Capping the splice at that
                latent-derived length would silently drop the input's TAIL
                (replacing it with generated audio) -- a violation of the
                exact-preservation contract. Splicing the FULL
                `input_wave` instead preserves it entirely whether the VAE
                floors or ceils (in the ceil case the extra length simply
                doesn't exist and this is a no-op). If the VAE floored, the
                ~1-latent-frame region just past the latent hold window was
                free-running (generated) content in `pred_latents`/
                `generated_wave`; this splice overwrites it with the exact
                input tail, and the right-side crossfade below smooths the
                resulting seam -- a sub-40ms generated-side seam shift is
                accepted so the ENTIRE input is preserved sample-exact.
            offset_sec/dur_sec: the LATENT-frame-derived placement boundary
                (`off / 25.0`, `t_ref / 25.0`), not raw user input -- mirrors
                `_acestep_apply_repaint_waveform_splice`'s convention of
                using the mask-aligned boundary, not the raw seconds param.
                Only `offset_sec` (the start) is actually load-bearing here;
                `dur_sec` is accepted for call-site symmetry with the
                repaint splice but is intentionally NOT used to bound the
                spliced length (see above).
            sample_rate: audio sample rate (48000 for ACE-Step).
            crossfade_ms: declick ramp length in milliseconds on EACH side
                that has generated content (10ms default, matching repaint).

        Returns:
            Spliced waveform, same shape as `generated_wave`.
        """
        total_samples = generated_wave.shape[-1]
        start_sample = max(0, min(int(round(offset_sec * sample_rate)), total_samples))
        # `input_wave`'s FULL length is authoritative (see docstring) -- the
        # ONLY cap is running off the end of `generated_wave` itself. No
        # `dur_sec`/latent-frame-count cap: that would risk dropping the
        # input's tail whenever the VAE encoder floored `t_ref`.
        in_len = max(0, min(input_wave.shape[-1], total_samples - start_sample))
        end_sample = start_sample + in_len

        result = generated_wave.clone()
        if in_len <= 0:
            # Degenerate placement (nothing to preserve) -- return the
            # untouched generated track rather than raise; the caller
            # (`_generate_audoutpaint_acestep`) already validates that the
            # placed input has >=1 latent frame before reaching decode.
            return result

        ref = input_wave[..., :in_len].to(device=result.device, dtype=result.dtype)
        # Unconditional overwrite of the ENTIRE held span -- mask=0 (original)
        # everywhere in [start_sample, end_sample), no exceptions, no
        # crossfade encroachment (contrast the repaint splice above, whose
        # ramp starts BEFORE its window boundary).
        result[..., start_sample:end_sample] = ref

        crossfade_samples = int(round((crossfade_ms / 1000.0) * sample_rate))
        if crossfade_samples <= 0:
            return result

        # Left boundary: only meaningful if there is generated content BEFORE
        # the span (start_sample > 0). The ramp occupies
        # [start_sample - n, start_sample) -- strictly GENERATED-side samples,
        # entirely outside the held span -- fading the raw generated
        # amplitude UP TOWARD the exact first preserved sample as it
        # approaches the seam (a level-match declick, not a content blend,
        # since no reference audio exists before the span to blend against).
        if start_sample > 0:
            n = min(crossfade_samples, start_sample)
            boundary_value = ref[..., :1]  # exact first preserved sample, per channel
            frac = torch.linspace(0.0, 1.0, n + 2, device=result.device, dtype=result.dtype)[1:-1]
            seg = result[..., start_sample - n:start_sample]
            result[..., start_sample - n:start_sample] = seg * (1.0 - frac) + boundary_value * frac

        # Right boundary: only meaningful if there is generated content AFTER
        # the span (end_sample < total_samples). Mirrors the left boundary,
        # fading DOWN FROM the exact last preserved sample as we move away
        # from the seam.
        if end_sample < total_samples:
            n = min(crossfade_samples, total_samples - end_sample)
            boundary_value = ref[..., -1:]  # exact last preserved sample, per channel
            frac = torch.linspace(1.0, 0.0, n + 2, device=result.device, dtype=result.dtype)[1:-1]
            seg = result[..., end_sample:end_sample + n]
            result[..., end_sample:end_sample + n] = seg * (1.0 - frac) + boundary_value * frac

        return result

    # ------------------------------------------------------------------
    # LoRA (generation-time apply/restore for a trained ACE-Step LoRA).
    #
    # ACE-Step uses the same component-based (not diffusers-pipeline-based)
    # architecture as Z-Image/FLUX.2, so this mirrors
    # `ZImageMixin._load_lora_zimage`/`_wrap_with_lora`/`_unload_lora_zimage`
    # and `FluxMixin._load_lora_flux2` (pipeline_backends/zimage.py,
    # pipeline_backends/flux2.py): LoRAs wrap the original nn.Linear via
    # forward-time addition (`core.training.adapters.sd15_adapter.LoRALinearLayer`),
    # not weight merging, so they can be unloaded by restoring the original
    # module reference (no drift, no leak across generations).
    #
    # Key format matches the training adapter
    # (`core.training.adapters.acestep_adapter.AceStepLoRAAdapter`/
    # `iter_acestep_lora_targets`) exactly: sd-scripts native
    # `lora_unet_decoder_layers_{i}_{self_attn|cross_attn}_{q,k,v,o}_proj`,
    # mapping onto `dit.decoder.layers[i].{self_attn,cross_attn}.{q,k,v,o}_proj`.
    # Feed-forward (`mlp`) LoRA (opt-in on the training side) is intentionally
    # NOT applied at inference in this phase -- only the always-on
    # attention scope is supported here; unmatched keys (mlp scope, or an
    # unrelated-architecture LoRA) are skipped with a warning, not a crash.
    #
    # EXTERNAL diffusers/PEFT-format LoRAs (e.g. community checkpoints trained
    # with `peft`/`diffusers` tooling instead of sd-scripts) use a different
    # key convention entirely -- `transformer_blocks.{i}.attn.to_{q,k,v}` /
    # `transformer_blocks.{i}.attn.to_out.0` (DiT self-attn),
    # `transformer_blocks.{i}.cross_attn.to_{q,k,v}` / `...cross_attn.to_out.0`
    # (DiT cross-attn), and `lyric_encoder.encoders.{i}.self_attn.linear_{q,k,v}`
    # (lyric encoder self-attn; no `linear_out`/output-projection LoRA target
    # observed in the wild, but mapped too in case one ever ships), with
    # `.lora_A.weight` (down, [rank, in]) / `.lora_B.weight` (up, [out, rank])
    # instead of `.lora_down.weight` / `.lora_up.weight`, and typically NO
    # per-tensor `.alpha` (alpha defaults to rank in that case -- PEFT usually
    # carries alpha in a sidecar `adapter_config.json`, not in the safetensors
    # file itself, but any `.alpha` tensor found alongside the source prefix
    # is still honored). `_load_lora_acestep` auto-detects the format per file
    # and dispatches to `_load_lora_acestep_diffusers_format` for this case.
    #
    # IMPORTANT: key-convention remapping only bridges NAMING differences for
    # LoRAs trained against the SAME underlying architecture/hidden sizes as
    # this repo's vendored ACE-Step 1.5 checkpoints. A diffusers/PEFT LoRA
    # trained against a different ACE-Step generation (e.g. the original
    # ACE-Step v1 3.5B model, whose DiT hidden_size=2560 and lyric encoder
    # hidden_size=1024/6 layers, vs this repo's 1.5 checkpoints'
    # hidden_size=2048 DiT / 2048-dim 8-layer lyric encoder) is dimensionally
    # incompatible no matter how the keys are renamed -- `_wrap_with_lora_acestep`
    # shape-validates in/out features against the target Linear and skips
    # (with a warning) any module whose LoRA tensor shape does not match,
    # rather than crashing on a matmul dimension error at generation time.
    # ------------------------------------------------------------------

    _ACESTEP_LORA_ATTN_NAMES = ("self_attn", "cross_attn")
    _ACESTEP_LORA_ATTN_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")

    # -- diffusers/PEFT key-format regexes (see comment block above) --
    _ACESTEP_LORA_DIFFUSERS_DIT_QKV_RE = re.compile(
        r"^transformer_blocks\.(\d+)\.(attn|cross_attn)\.to_(q|k|v)\.(lora_A|lora_B)\.weight$"
    )
    _ACESTEP_LORA_DIFFUSERS_DIT_OUT_RE = re.compile(
        r"^transformer_blocks\.(\d+)\.(attn|cross_attn)\.to_out\.0\.(lora_A|lora_B)\.weight$"
    )
    _ACESTEP_LORA_DIFFUSERS_LYRIC_RE = re.compile(
        r"^lyric_encoder\.encoders\.(\d+)\.self_attn\.linear_(q|k|v|out)\.(lora_A|lora_B)\.weight$"
    )
    # DiT: diffusers names the DiT's own self-attention scope "attn" (not
    # "self_attn"); our vendored model's attribute is "self_attn" for that
    # same scope. Cross-attn is named "cross_attn" on both sides.
    _ACESTEP_LORA_DIFFUSERS_DIT_SCOPE = {"attn": "self_attn", "cross_attn": "cross_attn"}
    _ACESTEP_LORA_DIFFUSERS_LEAF = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "out": "o_proj"}

    @staticmethod
    def _acestep_walk_module_path(root, dotted_path: str):
        """Resolve a dotted module path (e.g.
        `"decoder.layers.0.self_attn.q_proj"` or
        `"encoder.lyric_encoder.layers.3.self_attn.q_proj"`) against `root`,
        returning `(parent_module, leaf_attr_name)` so the caller can
        getattr/setattr the leaf on the parent. Numeric path segments index
        into an `nn.ModuleList` (e.g. the `.layers.<i>.` hop). Returns
        `(None, None)` if any hop along the path fails to resolve (missing
        attribute, out-of-range index, etc.) -- never raises."""
        parts = dotted_path.split(".")
        obj = root
        for part in parts[:-1]:
            if obj is None:
                return None, None
            if part.isdigit():
                try:
                    obj = obj[int(part)]
                except (TypeError, IndexError, KeyError):
                    return None, None
            else:
                obj = getattr(obj, part, None)
        if obj is None:
            return None, None
        return obj, parts[-1]

    @staticmethod
    def _acestep_lora_emit_compat_warning(
        lora_path: str, applied_count: int, total_count: int, mismatch_note: str = "shape mismatch",
    ) -> None:
        """Surface a user-facing warning when a LoRA matched fewer target
        modules than were present in the file -- i.e. it had no effect (0
        applied) or only a partial effect (some modules skipped) on the
        currently loaded ACE-Step model.

        Mirrors the `add_warning` pattern used by every other pipeline
        backend (e.g. `Flux2Mixin`'s attention-fallback warnings): a plain
        module-level call into `api.generation_status`, backed by an
        in-process, per-request store. `routes.py`'s `generate_txt2aud`/
        `generate_aud2aud` call `start_generation()` before invoking
        `pipeline_manager.generate_txt2aud`/`generate_aud2aud` (which is what
        eventually calls `_load_lora_acestep` via
        `_apply_or_clear_lora_acestep`), so this warning is already recorded
        by the time the endpoint calls `get_warnings()` for the response --
        no change to this method's/its callers' return values is needed.

        No-op (does not call `add_warning`) when `applied_count ==
        total_count` -- a fully-compatible LoRA must not produce a warning.
        """
        if applied_count >= total_count:
            return
        lora_name = os.path.basename(lora_path)
        if applied_count == 0:
            message = (
                f"LoRA '{lora_name}': 0 of {total_count} modules applied -- incompatible with the "
                f"loaded ACE-Step model (likely a different ACE-Step generation, e.g. v1 3.5B vs "
                f"this v1.5 2B, or an unrecognized key format). It had no effect."
            )
        else:
            message = (
                f"LoRA '{lora_name}': applied {applied_count} of {total_count} modules; "
                f"{total_count - applied_count} skipped on {mismatch_note}."
            )
        try:
            from api.generation_status import add_warning
            add_warning(message, code="lora_incompatible")
        except Exception:
            pass

    def _load_lora_acestep(self, lora_configs: list):
        """Load LoRAs onto the ACE-Step DiT's decoder attention Linears.

        Args:
            lora_configs: list of {"path": str, "strength": float, ...}
                (same shape as every other arch's `params["loras"]`).
        """
        if not lora_configs:
            return

        if not self.acestep_components:
            print("[AceStep LoRA] WARNING: ACE-Step components not loaded")
            return

        dit = self.acestep_components.get("dit")
        decoder = getattr(dit, "decoder", None)
        layers = getattr(decoder, "layers", None) if decoder is not None else None
        if layers is None:
            print("[AceStep LoRA] WARNING: DiT has no decoder.layers -- cannot apply LoRA")
            return

        if not hasattr(self, "_acestep_lora_original_modules"):
            self._acestep_lora_original_modules = {}
            self._acestep_lora_wrapped_modules = set()

        # Use global lora_manager instance (has user-configured additional_dirs)
        from core.extensions.lora_manager import lora_manager

        print(f"[AceStep LoRA] Loading {len(lora_configs)} LoRA(s)...")

        for i, lora_config in enumerate(lora_configs):
            lora_path = lora_config.get("path", "")
            lora_strength = lora_config.get("strength", 1.0)

            resolved_path = lora_manager._resolve_lora_path(lora_path)
            if resolved_path is None:
                print(f"[AceStep LoRA] WARNING: LoRA file not found: {lora_path}")
                print(f"[AceStep LoRA]   Searched in: {lora_manager.lora_dir}")
                print(f"[AceStep LoRA]   Additional dirs: {lora_manager.additional_dirs}")
                continue

            print(f"[AceStep LoRA] Loading LoRA {i+1}/{len(lora_configs)}: {lora_path} (strength={lora_strength})")

            from safetensors import safe_open

            try:
                with safe_open(str(resolved_path), framework="pt", device="cpu") as f:
                    lora_state_dict = {key: f.get_tensor(key) for key in f.keys()}

                print(f"[AceStep LoRA] Loaded {len(lora_state_dict)} tensors from {lora_path}")

                # ---- format detection (sd-scripts native vs external diffusers/PEFT) ----
                is_sdscripts_format = any(
                    k.startswith("lora_unet_decoder_layers_") for k in lora_state_dict
                )
                is_diffusers_format = (not is_sdscripts_format) and any(
                    (".lora_A." in k) or (".lora_B." in k) for k in lora_state_dict
                )

                if is_diffusers_format:
                    self._load_lora_acestep_diffusers_format(
                        dit, lora_state_dict, lora_strength, lora_path
                    )
                    continue

                if not is_sdscripts_format:
                    sample_keys = list(lora_state_dict.keys())[:5]
                    print(
                        f"[AceStep LoRA] WARNING: unrecognized LoRA key format in {lora_path!r} -- neither "
                        f"sd-scripts native ('lora_unet_decoder_layers_...') nor diffusers/PEFT "
                        f"('transformer_blocks....lora_A/lora_B.weight') naming was detected; skipping this "
                        f"file entirely (not an error). Sample keys found in file: {sample_keys}"
                    )
                    try:
                        from api.generation_status import add_warning
                        add_warning(
                            f"LoRA '{os.path.basename(lora_path)}': unrecognized key format (neither "
                            f"sd-scripts native nor diffusers/PEFT ACE-Step naming) -- it had no effect.",
                            code="lora_incompatible",
                        )
                    except Exception:
                        pass
                    continue

                total_pairs = sum(1 for k in lora_state_dict if k.endswith(".lora_down.weight"))
                applied_count = 0

                for layer_idx, layer in enumerate(layers):
                    for attn_name in self._ACESTEP_LORA_ATTN_NAMES:
                        attn = getattr(layer, attn_name, None)
                        if attn is None:
                            continue

                        for leaf in self._ACESTEP_LORA_ATTN_LEAVES:
                            original_linear = getattr(attn, leaf, None)
                            if not _is_lora_target(original_linear):
                                # Either absent, or already LoRA-wrapped (the call
                                # site always unloads before reloading -- see
                                # `_generate_txt2aud_acestep`/`_generate_aud2aud_acestep`).
                                continue

                            lora_key_prefix = f"lora_unet_decoder_layers_{layer_idx}_{attn_name}_{leaf}"
                            lora_down_key = f"{lora_key_prefix}.lora_down.weight"
                            lora_up_key = f"{lora_key_prefix}.lora_up.weight"

                            if lora_down_key not in lora_state_dict or lora_up_key not in lora_state_dict:
                                continue

                            lora_down_weight = lora_state_dict[lora_down_key]
                            lora_up_weight = lora_state_dict[lora_up_key]
                            lora_alpha_key = f"{lora_key_prefix}.alpha"
                            lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                            module_key = f"decoder.layers.{layer_idx}.{attn_name}.{leaf}"
                            wrapped = self._wrap_with_lora_acestep(
                                attn, leaf, original_linear,
                                lora_down_weight, lora_up_weight, lora_strength, lora_alpha, module_key,
                            )
                            if wrapped:
                                applied_count += 1

                print(f"[AceStep LoRA] Applied LoRA to {applied_count} modules "
                      f"({total_pairs} lora_down/up pair(s) present in file)")
                if applied_count == 0:
                    sample_keys = list(lora_state_dict.keys())[:5]
                    print(
                        f"[AceStep LoRA] WARNING: 0 modules matched in {lora_path!r} -- this LoRA does not "
                        f"target the ACE-Step decoder.layers.*.{{self_attn,cross_attn}}.* scope this backend "
                        f"applies (skipped entirely, not an error). It was likely trained for a different "
                        f"component/architecture entirely (e.g. a lyric/text encoder-only LoRA). Expected key "
                        f"prefix: 'lora_unet_decoder_layers_<i>_<self_attn|cross_attn>_<q,k,v,o>_proj.lora_down.weight'. "
                        f"Sample keys found in file: {sample_keys}"
                    )
                elif applied_count < total_pairs:
                    sample_keys = list(lora_state_dict.keys())[:5]
                    print(
                        f"[AceStep LoRA] WARNING: {total_pairs - applied_count} LoRA tensor pair(s) in "
                        f"{lora_path!r} did not match any ACE-Step decoder.layers.*.{{self_attn,cross_attn}}.* "
                        f"module (skipped, not an error -- e.g. mlp-scope keys, or a different architecture's "
                        f"LoRA entirely). Expected key prefix: "
                        f"'lora_unet_decoder_layers_<i>_<self_attn|cross_attn>_<q,k,v,o>_proj'. "
                        f"Sample keys found in file: {sample_keys}"
                    )

                if total_pairs > 0:
                    self._acestep_lora_emit_compat_warning(
                        lora_path, applied_count, total_pairs,
                        mismatch_note="shape mismatch or unmatched target module",
                    )

            except Exception as e:
                print(f"[AceStep LoRA] ERROR: Failed to load LoRA {lora_path}: {e}")
                import traceback
                traceback.print_exc()

    def _wrap_with_lora_acestep(self, parent_module, attr_name, original_linear,
                                 lora_down_weight, lora_up_weight, strength, alpha, module_key):
        """Wrap a decoder attention Linear with a LoRA forward-addition layer.

        Mirrors `ZImageMixin._wrap_with_lora`/`FluxMixin._wrap_with_lora_flux2`.

        Shape-validates the LoRA tensors against the target Linear's
        `in_features`/`out_features` before wrapping and returns False
        (skipped, not applied) on a mismatch instead of crashing later at a
        matmul inside the wrapped forward pass. This guard is a no-op for
        this repo's own sd-scripts-trained LoRAs (their shapes always match,
        since they were trained against this exact model); it matters for
        externally-sourced diffusers/PEFT-format LoRAs that may have been
        trained against a different-sized ACE-Step variant (see the
        `_load_lora_acestep_diffusers_format` docstring).
        """
        from core.training.adapters.sd15_adapter import LoRALinearLayer

        if isinstance(original_linear, LoRALinearLayer):
            true_original = original_linear.original_module
        else:
            true_original = original_linear

        expected_in = true_original.in_features
        expected_out = true_original.out_features
        lora_in = lora_down_weight.shape[-1]
        lora_out = lora_up_weight.shape[0]
        if lora_in != expected_in or lora_out != expected_out:
            print(
                f"[AceStep LoRA] WARNING: shape mismatch for {module_key!r} -- LoRA tensor "
                f"in/out=({lora_in}, {lora_out}) vs this model's module in/out="
                f"({expected_in}, {expected_out}); skipping this module (not an error -- the "
                f"LoRA was very likely trained against a different-sized ACE-Step checkpoint "
                f"variant, which no key-name remap can bridge)."
            )
            return False

        if module_key not in self._acestep_lora_original_modules:
            self._acestep_lora_original_modules[module_key] = true_original

        rank = lora_down_weight.shape[0]
        alpha_value = alpha.item() if alpha is not None else rank

        lora_wrapper = LoRALinearLayer(
            true_original, rank=rank, alpha=alpha_value, lora_name=module_key
        )

        # The BRANCH's dtype must never be taken from the base weight: over an
        # Int8Linear that would store the LoRA at 8 uniform levels, and over an
        # Fp8Linear at e4m3 precision. `lora_branch_dtype` returns the base's
        # dtype only when it is a real floating-point one and falls back to
        # bfloat16 otherwise. The DEVICE is still the base's -- that is a
        # placement, not a precision.
        from core.training.adapters.base_adapter import lora_branch_dtype

        device = true_original.weight.device
        dtype = lora_branch_dtype(true_original)

        with torch.no_grad():
            lora_wrapper.lora_down.weight.data = lora_down_weight.to(device=device, dtype=dtype)
            lora_wrapper.lora_up.weight.data = lora_up_weight.to(device=device, dtype=dtype)

        # Apply strength by overriding the default alpha/rank scale.
        lora_wrapper.scale = (alpha_value / rank) * strength

        setattr(parent_module, attr_name, lora_wrapper)

        self._acestep_lora_wrapped_modules.add(module_key)
        return True

    def _load_lora_acestep_diffusers_format(self, dit, lora_state_dict, lora_strength, lora_path):
        """Load an EXTERNAL diffusers/PEFT-format ACE-Step LoRA (e.g. community
        checkpoints exported from `peft`/`diffusers` training tooling, as
        opposed to this repo's own sd-scripts-native training format).

        Key convention (see the class-level "LoRA" comment block for the full
        citation/rationale) -- remapped onto this repo's vendored module
        attribute paths:

            transformer_blocks.{i}.attn.to_{q,k,v}            -> decoder.layers.{i}.self_attn.{q,k,v}_proj
            transformer_blocks.{i}.attn.to_out.0               -> decoder.layers.{i}.self_attn.o_proj
            transformer_blocks.{i}.cross_attn.to_{q,k,v}       -> decoder.layers.{i}.cross_attn.{q,k,v}_proj
            transformer_blocks.{i}.cross_attn.to_out.0         -> decoder.layers.{i}.cross_attn.o_proj
            lyric_encoder.encoders.{i}.self_attn.linear_{q,k,v} -> encoder.lyric_encoder.layers.{i}.self_attn.{q,k,v}_proj
            lyric_encoder.encoders.{i}.self_attn.linear_out    -> encoder.lyric_encoder.layers.{i}.self_attn.o_proj
                                                                   (mapped defensively; no LoRA in the wild has been
                                                                   observed to actually target this leaf)

        `.lora_A.weight` is the down-projection ([rank, in]), `.lora_B.weight`
        is the up-projection ([out, rank]) -- diffusers/PEFT naming, vs this
        repo's own `.lora_down.weight`/`.lora_up.weight`. A per-tensor
        `<source_prefix>.alpha` is honored if present; otherwise alpha
        defaults to rank (PEFT LoRAs typically carry alpha in a sidecar
        `adapter_config.json` that a bare safetensors dump does not include).

        IMPORTANT: this only bridges a NAMING-convention difference. A
        diffusers/PEFT LoRA trained against a different ACE-Step generation
        (different hidden_size / layer counts) is dimensionally incompatible
        no matter how its keys are renamed -- `_wrap_with_lora_acestep`
        shape-validates every module against this model's actual
        `in_features`/`out_features` and skips (with a per-module warning)
        anything that does not match, so a mismatched file degrades to
        "0 modules applied" rather than crashing.

        Unmatched keys (e.g. an incomplete lora_A/lora_B pair, or a target
        module this backend does not expose) are skipped with a warning, not
        an error -- mirrors the sd-scripts-format path's contract.
        """
        # module_key -> {"kind": "dit_self"|"dit_cross"|"lyric", "source_prefix": str,
        #                "down": Tensor|None, "up": Tensor|None, "alpha": Tensor|None}
        groups: Dict[str, Dict[str, Any]] = {}
        matched_source_keys = set()

        def _bucket(module_key, kind, source_prefix, slot, tensor):
            g = groups.setdefault(
                module_key, {"kind": kind, "source_prefix": source_prefix, "down": None, "up": None, "alpha": None}
            )
            g[slot] = tensor

        for key, tensor in lora_state_dict.items():
            m = self._ACESTEP_LORA_DIFFUSERS_DIT_QKV_RE.match(key)
            if m:
                idx, scope_raw, qkv, ab = m.groups()
                scope = self._ACESTEP_LORA_DIFFUSERS_DIT_SCOPE[scope_raw]
                leaf = self._ACESTEP_LORA_DIFFUSERS_LEAF[qkv]
                module_key = f"decoder.layers.{idx}.{scope}.{leaf}"
                source_prefix = key.rsplit(".", 2)[0]  # strip ".lora_A.weight" / ".lora_B.weight"
                kind = "dit_self" if scope == "self_attn" else "dit_cross"
                _bucket(module_key, kind, source_prefix, "down" if ab == "lora_A" else "up", tensor)
                matched_source_keys.add(key)
                continue

            m = self._ACESTEP_LORA_DIFFUSERS_DIT_OUT_RE.match(key)
            if m:
                idx, scope_raw, ab = m.groups()
                scope = self._ACESTEP_LORA_DIFFUSERS_DIT_SCOPE[scope_raw]
                module_key = f"decoder.layers.{idx}.{scope}.o_proj"
                source_prefix = key.rsplit(".", 2)[0]
                kind = "dit_self" if scope == "self_attn" else "dit_cross"
                _bucket(module_key, kind, source_prefix, "down" if ab == "lora_A" else "up", tensor)
                matched_source_keys.add(key)
                continue

            m = self._ACESTEP_LORA_DIFFUSERS_LYRIC_RE.match(key)
            if m:
                idx, qkv, ab = m.groups()
                leaf = self._ACESTEP_LORA_DIFFUSERS_LEAF[qkv]
                module_key = f"encoder.lyric_encoder.layers.{idx}.self_attn.{leaf}"
                source_prefix = key.rsplit(".", 2)[0]
                _bucket(module_key, "lyric", source_prefix, "down" if ab == "lora_A" else "up", tensor)
                matched_source_keys.add(key)
                continue

        # Second pass: pick up an optional per-tensor alpha alongside each
        # matched group's source prefix (rare for bare PEFT safetensors dumps,
        # but honored if present).
        for module_key, info in groups.items():
            alpha_key = f"{info['source_prefix']}.alpha"
            if alpha_key in lora_state_dict:
                info["alpha"] = lora_state_dict[alpha_key]
                matched_source_keys.add(alpha_key)

        total_groups = len(groups)
        applied_count = 0
        dit_self_applied = 0
        dit_cross_applied = 0
        lyric_applied = 0
        skipped_shape = 0
        skipped_missing = 0

        for module_key, info in groups.items():
            down = info["down"]
            up = info["up"]
            if down is None or up is None:
                print(
                    f"[AceStep LoRA] WARNING: incomplete lora_A/lora_B pair for {module_key!r} "
                    f"(source prefix {info['source_prefix']!r}) -- skipping."
                )
                skipped_missing += 1
                continue

            parent, leaf = self._acestep_walk_module_path(dit, module_key)
            if parent is None or leaf is None:
                print(
                    f"[AceStep LoRA] WARNING: no module at {module_key!r} on this ACE-Step model "
                    f"(remapped from {info['source_prefix']!r}) -- skipping."
                )
                skipped_missing += 1
                continue

            original_linear = getattr(parent, leaf, None)
            if not _is_lora_target(original_linear):
                skipped_missing += 1
                continue

            wrapped = self._wrap_with_lora_acestep(
                parent, leaf, original_linear, down, up, lora_strength, info["alpha"], module_key,
            )
            if wrapped:
                applied_count += 1
                if info["kind"] == "dit_self":
                    dit_self_applied += 1
                elif info["kind"] == "dit_cross":
                    dit_cross_applied += 1
                elif info["kind"] == "lyric":
                    lyric_applied += 1
            else:
                skipped_shape += 1

        unmatched_keys = [k for k in lora_state_dict if k not in matched_source_keys]

        print(
            f"[AceStep LoRA] Loading (diffusers/PEFT format detected) {lora_path!r} -- "
            f"{total_groups} module group(s) parsed from key names"
        )
        print(
            f"[AceStep LoRA] Applied LoRA to {applied_count} modules "
            f"(DiT self_attn={dit_self_applied}, DiT cross_attn={dit_cross_applied}, "
            f"lyric_encoder self_attn={lyric_applied})"
        )
        if skipped_shape:
            print(
                f"[AceStep LoRA] WARNING: {skipped_shape} module(s) skipped due to a hidden-dimension "
                f"shape mismatch against the currently loaded ACE-Step model (see per-module warnings "
                f"above) -- this LoRA was very likely trained against a different-sized ACE-Step "
                f"checkpoint variant; renaming keys cannot bridge a real architecture/dimension mismatch."
            )
        if skipped_missing:
            print(
                f"[AceStep LoRA] WARNING: {skipped_missing} module group(s) skipped -- either an "
                f"incomplete lora_A/lora_B pair, or no matching module on this model (e.g. "
                f"lyric_encoder self-attn output-projection LoRA, which this backend's lyric encoder "
                f"does not expose as a separately-trained-target in the wild)."
            )
        if unmatched_keys:
            print(
                f"[AceStep LoRA] {len(unmatched_keys)} key(s) in {lora_path!r} did not match any known "
                f"diffusers/PEFT ACE-Step key pattern (e.g. FF/mlp-scope LoRA, or an unrelated "
                f"architecture) and were ignored entirely. Sample: {unmatched_keys[:5]}"
            )
        if applied_count == 0:
            sample_keys = list(lora_state_dict.keys())[:5]
            print(
                f"[AceStep LoRA] WARNING: 0 modules matched in {lora_path!r} after diffusers/PEFT key "
                f"remap (skipped entirely, not an error). Sample keys found in file: {sample_keys}"
            )

        if total_groups > 0:
            if skipped_shape and skipped_missing:
                mismatch_note = "shape mismatch or unmatched target module"
            elif skipped_shape:
                mismatch_note = "shape mismatch"
            else:
                mismatch_note = "unmatched target module"
            self._acestep_lora_emit_compat_warning(
                lora_path, applied_count, total_groups, mismatch_note=mismatch_note,
            )

    def _unload_lora_acestep(self):
        """Restore original (un-wrapped) Linear modules on the ACE-Step DiT.

        Walks `self._acestep_lora_wrapped_modules` (the set of `module_key`
        dotted paths actually wrapped by either `_load_lora_acestep`'s
        sd-scripts-format path -- always under `decoder.layers.*` -- or
        `_load_lora_acestep_diffusers_format`'s remap path -- which can ALSO
        wrap `encoder.lyric_encoder.layers.*` modules) via
        `_acestep_walk_module_path`, rather than re-deriving a fixed
        `decoder.layers` scope here; this keeps unload correct regardless of
        which scopes a given LoRA touched."""
        if not hasattr(self, "_acestep_lora_original_modules"):
            print("[AceStep LoRA] No LoRAs loaded")
            return

        if not self.acestep_components:
            print("[AceStep LoRA] WARNING: ACE-Step components not loaded")
            return

        dit = self.acestep_components.get("dit")
        if dit is None:
            print("[AceStep LoRA] WARNING: ACE-Step DiT not loaded -- cannot unload LoRA")
            return

        unloaded_count = 0
        print(f"[AceStep LoRA] Unloading LoRAs ({len(self._acestep_lora_wrapped_modules)} modules)...")

        for module_key in list(self._acestep_lora_wrapped_modules):
            original = self._acestep_lora_original_modules.get(module_key)
            if original is None:
                continue
            parent, leaf = self._acestep_walk_module_path(dit, module_key)
            if parent is None:
                print(f"[AceStep LoRA] WARNING: could not re-resolve {module_key!r} to unload -- skipping")
                continue
            setattr(parent, leaf, original)
            unloaded_count += 1

        self._acestep_lora_wrapped_modules.clear()
        print(f"[AceStep LoRA] Unloaded {unloaded_count} LoRA modules")
        print("[AceStep LoRA] Original modules preserved for future LoRA loads")

    def _apply_or_clear_lora_acestep(self, lora_configs: list):
        """Shared load/unload gate, called at the top of both txt2aud and
        aud2aud before the DiT forward pass. Mirrors the call-site pattern in
        `_generate_txt2img_zimage`/`_generate_txt2img_flux2`: always unload
        any previously-wrapped modules first (so switching to a different
        LoRA, a different strength, or no LoRA at all never leaves stale
        wrappers around), then load the newly requested set (if any)."""
        if lora_configs:
            if getattr(self, "_acestep_lora_wrapped_modules", None):
                self._unload_lora_acestep()
            self._load_lora_acestep(lora_configs)
        else:
            if getattr(self, "_acestep_lora_wrapped_modules", None):
                print("[AceStep LoRA] No LoRAs in params, unloading existing LoRAs")
                self._unload_lora_acestep()

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def _generate_txt2aud_acestep(
        self,
        params: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """Generate a music waveform from text conditioning (ACE-Step 1.5, turbo).

        Args:
            params: caption/prompt (str), lyrics (str, default ""),
                audio_duration (float sec), seed (int, -1 = random),
                inference_steps (int, default 8), guidance_scale (float,
                forced 1.0 -- turbo is CFG-distilled), shift (float, default
                3.0), sampler_mode (accepted for forward-compat; the vendored
                `generate_audio` has no sampler_mode knob so this is
                currently a no-op -- logged if non-default), bpm/key_scale/
                time_signature/vocal_language (folded into the "# Metas"
                text block, see `_acestep_build_text_prompt`).
            progress_callback: called as (step, total_steps). The vendored
                `generate_audio` runs its Euler loop internally (not
                reimplemented here), so this is only invoked coarsely
                (start / end) for Phase 2 -- fine-grained per-step progress
                is a Phase 3 concern.
            step_callback: reserved, unused (mirrors LTX2Mixin's contract).

        Returns:
            (waveform, sample_rate, actual_seed) where waveform is a
            torch.FloatTensor [2, samples] on CPU, sample_rate is 48000.
        """
        from api.error_handlers import ValidationError

        if not self.is_acestep_model or self.acestep_components is None:
            raise ValidationError(
                "Text-to-audio generation requires an ACE-Step model",
                detail="The currently loaded model is not an ACE-Step audio model.",
            )

        comps = self.acestep_components
        dit = comps.get("dit")
        vae = comps.get("vae")
        text_encoder = comps.get("text_encoder")
        tokenizer = comps.get("tokenizer")
        if dit is None or vae is None or text_encoder is None or tokenizer is None:
            raise ValidationError(
                "ACE-Step model is missing a required component",
                detail=f"dit={dit is not None}, vae={vae is not None}, "
                       f"text_encoder={text_encoder is not None}, tokenizer={tokenizer is not None}",
            )

        # ---- optional LoRA (see the "LoRA" section above for the apply/restore contract) ----
        self._apply_or_clear_lora_acestep(params.get("loras") or [])

        # ---- one-time in-place INT8 conversion (unet_quantization="int8") ----
        # After the LoRA gate and before staging; see _acestep_runtime_int8 for
        # why that order is the load-bearing one. No-op for every other value and
        # for an already-converted / already-quantized DiT.
        self._acestep_runtime_int8(params, progress_callback=progress_callback)

        device = self.device
        model_dtype = next(dit.parameters()).dtype

        caption = params.get("prompt") or params.get("caption") or ""
        lyrics = params.get("lyrics", "") or ""
        audio_duration = float(params.get("audio_duration", 10.0) or 10.0)
        seed = params.get("seed", -1)
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)
        inference_steps = int(params.get("inference_steps", 8) or 8)
        guidance_scale = float(params.get("guidance_scale", 1.0) or 1.0)
        shift = float(params.get("shift", 3.0) or 3.0)
        sampler_mode = params.get("sampler_mode", "euler")
        infer_method = params.get("infer_method", "ode")
        bpm = params.get("bpm")
        key_scale = params.get("key_scale", "") or ""
        time_signature = params.get("time_signature", "") or ""
        vocal_language = params.get("vocal_language", "en") or "en"

        if guidance_scale != 1.0:
            print(
                f"[AceStep] Turbo is CFG-distilled (no twin forward pass); "
                f"overriding guidance_scale {guidance_scale} -> 1.0."
            )
        if sampler_mode not in ("euler", None):
            print(
                f"[AceStep] sampler_mode={sampler_mode!r} requested, but the vendored "
                f"generate_audio has no sampler_mode knob (Euler-only ODE/SDE integration); ignored."
            )

        text_prompt = self._acestep_build_text_prompt(
            caption, audio_duration, bpm, key_scale, time_signature, vocal_language
        )
        lyrics_text = self._acestep_format_lyrics(lyrics, vocal_language)

        if progress_callback:
            try:
                progress_callback(0, inference_steps)
            except Exception:
                pass

        # ---- one-time silence-latent asset (VAE encode of literal silence) ----
        silence_latent = self._acestep_ensure_silence_latent(device)  # [1, 750, 64] on device

        # ---- text encoder stage ----
        self._acestep_move("text_encoder", device)
        try:
            tt = tokenizer(
                text_prompt, padding="longest", truncation=True, max_length=256, return_tensors="pt"
            )
            text_ids = tt.input_ids.to(device)
            text_attention_mask = tt.attention_mask.to(device).bool()

            lt = tokenizer(
                lyrics_text, padding="longest", truncation=True, max_length=2048, return_tensors="pt"
            )
            lyric_ids = lt.input_ids.to(device)
            lyric_attention_mask = lt.attention_mask.to(device).bool()

            with torch.inference_mode():
                text_hidden_states = text_encoder(input_ids=text_ids).last_hidden_state.to(model_dtype)
                lyric_hidden_states = text_encoder.embed_tokens(lyric_ids).to(model_dtype)
        finally:
            self._acestep_move("text_encoder", "cpu")
            self._acestep_empty_cache()

        # ---- latent-space conditioning (src_latents / chunk_masks / timbre) ----
        latent_frames = int(round(round(audio_duration, 1) * 25))
        latent_frames = max(latent_frames, 1)

        src_latents = self._acestep_silence_slice(silence_latent, latent_frames).to(model_dtype)  # [1, T, 64]
        chunk_masks = torch.ones(1, latent_frames, 64, dtype=model_dtype, device=device)
        is_covers = torch.zeros(1, dtype=torch.bool, device=device)
        # Silence timbre (no reference audio): matches infer_refer_latent's
        # all-zero-refer_audio branch (self.silence_latent[:, :750, :]).
        timbre_packed = silence_latent[:, :silence_latent.shape[1], :].to(model_dtype)  # [1, 750, 64]
        refer_audio_order_mask = torch.zeros(1, dtype=torch.long, device=device)

        # ---- optional custom timestep schedule for inference_steps != 8 ----
        # The vendored generate_audio has no `infer_steps` kwarg -- only a
        # baked-in 8-step SHIFT_TIMESTEPS[shift] table, or an explicit
        # `timesteps=` override (snapped to the nearest valid distilled
        # timestep). Mirrors the official reference's variable-step formula
        # (recipe section 1c) so callers can still trade steps for speed.
        custom_timesteps = None
        if inference_steps and inference_steps != 8:
            n = min(max(int(inference_steps), 1), 20)
            raw = [1.0 - i / n for i in range(n)]
            if shift != 1.0:
                raw = [shift * t / (1.0 + (shift - 1.0) * t) for t in raw]
            custom_timesteps = torch.tensor(raw, device=device, dtype=model_dtype)

        # ---- DiT stage: call the vendored generate_audio (internal sampling loop) ----
        self._acestep_move("dit", device)
        try:
            with torch.inference_mode():
                outputs = dit.generate_audio(
                    text_hidden_states=text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    lyric_hidden_states=lyric_hidden_states,
                    lyric_attention_mask=lyric_attention_mask,
                    refer_audio_acoustic_hidden_states_packed=timbre_packed,
                    refer_audio_order_mask=refer_audio_order_mask,
                    src_latents=src_latents,
                    chunk_masks=chunk_masks,
                    is_covers=is_covers,
                    silence_latent=silence_latent,
                    seed=seed,
                    infer_method=infer_method,
                    shift=shift,
                    timesteps=custom_timesteps,
                    # DCW (Differential Correction in Wavelet domain) defaults to
                    # ON (dcw_enabled=True, scaler=0.05) in the newer vendored
                    # generate_audio. pytorch_wavelets is not an installed
                    # dependency, so DCW would otherwise silently no-op after a
                    # per-process warning log; disable it explicitly to keep
                    # txt2aud behavior identical to the pre-re-vendor model
                    # (which had no DCW code path at all) with no warning noise.
                    dcw_enabled=False,
                )
            pred_latents = outputs["target_latents"]  # [1, T, 64]
        finally:
            self._acestep_move("dit", "cpu")
            self._acestep_empty_cache()

        if progress_callback:
            try:
                progress_callback(inference_steps, inference_steps)
            except Exception:
                pass

        # ---- validate latents (mirrors generate_music_decode.py's guards) ----
        if torch.isnan(pred_latents).any() or torch.isinf(pred_latents).any():
            raise RuntimeError(
                f"ACE-Step generation produced NaN/Inf latents "
                f"(shape={list(pred_latents.shape)}, dtype={pred_latents.dtype})."
            )
        if pred_latents.numel() > 0 and pred_latents.abs().sum() == 0:
            raise RuntimeError("ACE-Step generation produced all-zero latents.")

        # ---- VAE decode stage ----
        self._acestep_move("vae", device)
        try:
            vae_dtype = next(vae.parameters()).dtype
            pred_for_decode = pred_latents.transpose(1, 2).contiguous().to(vae_dtype)  # [1, 64, T]
            with torch.inference_mode():
                waveform = vae.decode(pred_for_decode).sample  # [1, 2, samples]
            waveform = waveform.float()
            peak = waveform.abs().amax(dim=[1, 2], keepdim=True)
            if torch.any(peak > 1.0):
                waveform = waveform / peak.clamp(min=1.0)
        finally:
            self._acestep_move("vae", "cpu")
            self._acestep_empty_cache()

        sample_rate = int(comps.get("sample_rate", 48000))
        return waveform[0].detach().cpu(), sample_rate, seed

    # ------------------------------------------------------------------
    # aud2aud: COVER (audio-to-audio, img2img analog) and REPAINT (inpaint
    # analog -- regenerate a time-range of a reference clip, keeping the
    # rest). Both call the same vendored `generate_audio`
    # (`vendor/modeling_acestep_v15_turbo.py`, re-vendored from the official
    # main branch -- see `scratchpad/acestep_repaint_feasibility.md` for the
    # GO/NO-GO state-dict-parity verification of that re-vendor), which now
    # declares `repaint_mask` / `clean_src_latents` / `repaint_crossfade_
    # frames` / `repaint_injection_ratio` for real (the OLDER vendored
    # snapshot swallowed them silently via `**kwargs` -- see
    # `scratchpad/acestep_aud2aud_recipe.md` sections 2 and 6 for that
    # earlier NO-GO state, now superseded).
    # ------------------------------------------------------------------

    def _generate_aud2aud_acestep(
        self,
        params: Dict[str, Any],
        reference_audio,
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """Cover or repaint generation (audio-to-audio) for ACE-Step 1.5 turbo.

        `params["mode"]` (or `params["audio_task"]`) selects the sub-mode,
        default `"cover"`:

        COVER (`mode="cover"`): the reference audio is VAE-encoded and fed
        back to the DiT as `src_latents` with `is_covers=True`; the vendored
        `generate_audio` internally tokenizes/detokenizes it through the
        model's FSQ codec to get a "semantic re-render" context (see
        `scratchpad/acestep_aud2aud_recipe.md` section 1b) -- this is a
        semantic-only cover (timbre stays silence, same as txt2aud), not a
        raw-latent passthrough. `cover_strength` (`audio_cover_strength` on
        the vendored model) is a STEP-COUNT blend, NOT an img2img
        start-timestep / partial-denoise knob: `xt` always starts from full
        noise and runs every step; the first `int(num_steps * cover_strength)`
        steps use the reference's semantic context, the remaining steps
        switch to a text2music-style (silence src_latents) context built
        from the SAME caption/lyric text. Higher `cover_strength` => closer
        to the reference. A true img2img-style partial-denoise
        (`cover_noise_strength` on the vendored model) is NOT wired here --
        out of scope for this phase.

        REPAINT (`mode="repaint"`): the `[repaint_start, repaint_end)`
        (seconds) window of the reference is blanked to silence in
        `src_latents` and marked `True` in `chunk_masks`/`repaint_mask`
        (`is_covers=False`); the FULL, unmodified reference latent is passed
        as `clean_src_latents`. The vendored sampler then (a) holds every
        non-repaint frame to the correctly-noised original at each early
        step (`repaint_mask`/`clean_src_latents`, default
        `repaint_injection_ratio=0.5` => first half of steps), (b)
        soft-blends generated vs. source at the boundaries post-loop
        (default `repaint_crossfade_frames=10` LATENT frames), and (c) after
        VAE decode we additionally splice the ORIGINAL reference waveform
        back into the kept region with a short (10ms) crossfade
        (`_acestep_apply_repaint_waveform_splice`) -- the VAE reconstruction
        of the kept region is not bit-identical to the source even though
        its latent was never touched by the sampler. See
        `scratchpad/acestep_repaint_feasibility.md` section 5 for the full
        recipe this mirrors.

        Args:
            params: prompt/caption (str), lyrics (str), mode
                ("cover"|"repaint", default "cover"), cover_strength (float
                in [0, 1], default 1.0, cover only), repaint_start/
                repaint_end (float seconds, repaint only, required),
                seed (int, -1 = random), inference_steps (int, default 8),
                guidance_scale (forced 1.0 -- turbo is CFG-distilled), shift
                (float, default 3.0), vocal_language / bpm / key_scale /
                time_signature (folded into the "# Metas" text block, see
                `_acestep_build_text_prompt`). `audio_duration` is NOT a
                user param here -- duration is derived from the reference
                audio's length in BOTH modes (recipe section 4).
            reference_audio: a file path (str) or raw audio bytes for the
                cover/repaint reference clip.
            progress_callback: called as (step, total_steps); coarse
                (start/end) only, see `_generate_txt2aud_acestep`.
            step_callback: reserved, unused.

        Returns:
            (waveform, sample_rate, actual_seed) -- identical contract to
            `_generate_txt2aud_acestep`.
        """
        from api.error_handlers import ValidationError

        if not self.is_acestep_model or self.acestep_components is None:
            raise ValidationError(
                "Audio-to-audio generation requires an ACE-Step model",
                detail="The currently loaded model is not an ACE-Step audio model.",
            )

        comps = self.acestep_components
        dit = comps.get("dit")
        vae = comps.get("vae")
        text_encoder = comps.get("text_encoder")
        tokenizer = comps.get("tokenizer")
        if dit is None or vae is None or text_encoder is None or tokenizer is None:
            raise ValidationError(
                "ACE-Step model is missing a required component",
                detail=f"dit={dit is not None}, vae={vae is not None}, "
                       f"text_encoder={text_encoder is not None}, tokenizer={tokenizer is not None}",
            )
        if reference_audio is None:
            raise ValidationError(
                "Audio-to-audio generation requires a reference audio file",
                detail="No reference_audio was provided.",
            )

        mode = (params.get("mode") or params.get("audio_task") or "cover").strip().lower()
        if mode not in ("cover", "repaint"):
            raise ValidationError(
                "Invalid aud2aud mode",
                detail=f"mode must be 'cover' or 'repaint', got {mode!r}.",
            )
        is_repaint = mode == "repaint"

        # ---- optional LoRA (see the "LoRA" section above for the apply/restore contract) ----
        self._apply_or_clear_lora_acestep(params.get("loras") or [])

        # ---- one-time in-place INT8 conversion (unet_quantization="int8") ----
        # After the LoRA gate and before staging; see _acestep_runtime_int8 for
        # why that order is the load-bearing one. No-op for every other value and
        # for an already-converted / already-quantized DiT.
        self._acestep_runtime_int8(params, progress_callback=progress_callback)

        device = self.device
        model_dtype = next(dit.parameters()).dtype

        caption = params.get("prompt") or params.get("caption") or ""
        lyrics = params.get("lyrics", "") or ""
        seed = params.get("seed", -1)
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)
        inference_steps = int(params.get("inference_steps", 8) or 8)
        guidance_scale = float(params.get("guidance_scale", 1.0) or 1.0)
        shift = float(params.get("shift", 3.0) or 3.0)
        infer_method = params.get("infer_method", "ode")
        cover_strength = float(params.get("cover_strength", 1.0) or 1.0)
        cover_strength = min(max(cover_strength, 0.0), 1.0)
        repaint_start = float(params.get("repaint_start", 0.0) or 0.0)
        repaint_end = float(params.get("repaint_end", 0.0) or 0.0)
        if is_repaint and repaint_end <= repaint_start:
            raise ValidationError(
                "Invalid repaint range",
                detail=f"repaint_end ({repaint_end}) must be greater than repaint_start ({repaint_start}).",
            )
        bpm = params.get("bpm")
        key_scale = params.get("key_scale", "") or ""
        time_signature = params.get("time_signature", "") or ""
        vocal_language = params.get("vocal_language", "en") or "en"

        if guidance_scale != 1.0:
            print(
                f"[AceStep] Turbo is CFG-distilled (no twin forward pass); "
                f"overriding guidance_scale {guidance_scale} -> 1.0."
            )

        if progress_callback:
            try:
                progress_callback(0, inference_steps)
            except Exception:
                pass

        # ---- one-time silence-latent asset (shared with txt2aud) ----
        silence_latent = self._acestep_ensure_silence_latent(device)  # [1, 750, 64] on device

        # ---- load + normalize the reference audio, then VAE-encode it ----
        ref_wav, ref_sr = self._acestep_load_reference_audio(reference_audio)
        ref_wav = self._acestep_normalize_stereo_48k(ref_wav, ref_sr)  # [2, samples], CPU float32

        self._acestep_move("vae", device)
        try:
            vae_dtype = next(vae.parameters()).dtype
            ref_wav_dev = ref_wav.unsqueeze(0).to(device=device, dtype=vae_dtype)  # [1, 2, samples]
            with torch.inference_mode():
                # .mode() (deterministic mean), matching the silence-latent asset.
                ref_latent = vae.encode(ref_wav_dev).latent_dist.mode()  # [1, 64, T]
            ref_latent = ref_latent.transpose(1, 2).contiguous().to(model_dtype)  # [1, T, 64]
        finally:
            self._acestep_move("vae", "cpu")
            self._acestep_empty_cache()

        latent_frames = int(ref_latent.shape[1])
        if latent_frames < 1:
            raise ValidationError(
                "Reference audio is too short to encode",
                detail=f"VAE-encoded reference latent has {latent_frames} frames (need >= 1).",
            )
        # Duration is DERIVED from the reference length (recipe section 4),
        # not a user-facing param; only used for the "# Metas" text block.
        audio_duration = latent_frames / 25.0

        text_prompt = self._acestep_build_text_prompt(
            caption, audio_duration, bpm, key_scale, time_signature, vocal_language
        )
        lyrics_text = self._acestep_format_lyrics(lyrics, vocal_language)

        # ---- text encoder stage ----
        self._acestep_move("text_encoder", device)
        try:
            tt = tokenizer(
                text_prompt, padding="longest", truncation=True, max_length=256, return_tensors="pt"
            )
            text_ids = tt.input_ids.to(device)
            text_attention_mask = tt.attention_mask.to(device).bool()

            lt = tokenizer(
                lyrics_text, padding="longest", truncation=True, max_length=2048, return_tensors="pt"
            )
            lyric_ids = lt.input_ids.to(device)
            lyric_attention_mask = lt.attention_mask.to(device).bool()

            with torch.inference_mode():
                text_hidden_states = text_encoder(input_ids=text_ids).last_hidden_state.to(model_dtype)
                lyric_hidden_states = text_encoder.embed_tokens(lyric_ids).to(model_dtype)
        finally:
            self._acestep_move("text_encoder", "cpu")
            self._acestep_empty_cache()

        # ---- cover / repaint conditioning ----
        repaint_frame_range = None  # (s, e) in latent frames, repaint only
        clean_src_latents = None
        repaint_mask = None
        if is_repaint:
            # ---- repaint conditioning (recipe section 5: mask/src_latents/repaint_mask) ----
            s, e = self._acestep_repaint_frame_range(repaint_start, repaint_end, latent_frames)
            repaint_frame_range = (s, e)

            silence_slice = self._acestep_silence_slice(silence_latent, latent_frames).to(model_dtype)

            src_latents = ref_latent.clone()  # [1, T, 64]
            src_latents[:, s:e, :] = silence_slice[:, s:e, :]

            chunk_masks = torch.zeros(1, latent_frames, 64, dtype=model_dtype, device=device)
            chunk_masks[:, s:e, :] = 1.0  # True/1 INSIDE the repaint region (opposite fill from cover)

            is_covers = torch.zeros(1, dtype=torch.bool, device=device)

            repaint_mask = torch.zeros(1, latent_frames, dtype=torch.bool, device=device)
            repaint_mask[:, s:e] = True  # True = generate (free), False = preserve (held to ref)
            clean_src_latents = ref_latent  # FULL, unmodified reference latent (the "hold" target)
        else:
            # ---- cover conditioning (recipe section 1a/1e) ----
            src_latents = ref_latent  # [1, T, 64] -- the reference latent (NOT silence)
            chunk_masks = torch.ones(1, latent_frames, 64, dtype=model_dtype, device=device)
            is_covers = torch.ones(1, dtype=torch.bool, device=device)

        # Silence timbre (semantic-only cover/repaint): matches txt2aud's timbre condition.
        timbre_packed = silence_latent[:, :silence_latent.shape[1], :].to(model_dtype)  # [1, 750, 64]
        refer_audio_order_mask = torch.zeros(1, dtype=torch.long, device=device)

        # ---- optional custom timestep schedule for inference_steps != 8 ----
        custom_timesteps = None
        if inference_steps and inference_steps != 8:
            n = min(max(int(inference_steps), 1), 20)
            raw = [1.0 - i / n for i in range(n)]
            if shift != 1.0:
                raw = [shift * t / (1.0 + (shift - 1.0) * t) for t in raw]
            custom_timesteps = torch.tensor(raw, device=device, dtype=model_dtype)

        # ---- mode-specific generate_audio kwargs ----
        # cover: audio_cover_strength + the non_cover_* fallback conditioning
        #   (used only when strength<1.0; harmless to always pass).
        # repaint: repaint_mask + clean_src_latents (audio_cover_strength stays
        #   at the vendored default of 1.0 -- with is_covers all-False and no
        #   repaint use of the cover-strength step-switch, leaving it at 1.0
        #   also guarantees `encoder_hidden_states_non_cover` is never built
        #   and the cover step-switch never fires, so non_cover_text_* must
        #   NOT be passed here).
        mode_kwargs: Dict[str, Any] = {}
        if is_repaint:
            mode_kwargs["repaint_mask"] = repaint_mask
            mode_kwargs["clean_src_latents"] = clean_src_latents
            # repaint_crossfade_frames (10 latent frames) / repaint_injection_ratio
            # (0.5) intentionally left at the vendored generate_audio's own
            # defaults -- not yet exposed as user params (recipe section 5/6).
        else:
            mode_kwargs["audio_cover_strength"] = cover_strength
            mode_kwargs["non_cover_text_hidden_states"] = text_hidden_states
            mode_kwargs["non_cover_text_attention_mask"] = text_attention_mask

        # ---- DiT stage: call the vendored generate_audio (internal sampling loop) ----
        self._acestep_move("dit", device)
        try:
            with torch.inference_mode():
                outputs = dit.generate_audio(
                    text_hidden_states=text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    lyric_hidden_states=lyric_hidden_states,
                    lyric_attention_mask=lyric_attention_mask,
                    refer_audio_acoustic_hidden_states_packed=timbre_packed,
                    refer_audio_order_mask=refer_audio_order_mask,
                    src_latents=src_latents,
                    chunk_masks=chunk_masks,
                    is_covers=is_covers,
                    silence_latent=silence_latent,
                    seed=seed,
                    infer_method=infer_method,
                    shift=shift,
                    timesteps=custom_timesteps,
                    # See the txt2aud call site for why this is explicit: DCW
                    # defaults to ON in the newer vendored generate_audio, but
                    # pytorch_wavelets isn't an installed dependency -- disable
                    # explicitly to keep cover/repaint behavior identical to
                    # pre-re-vendor (no DCW code path existed before) with no
                    # warning noise.
                    dcw_enabled=False,
                    **mode_kwargs,
                )
            pred_latents = outputs["target_latents"]  # [1, T, 64]
        finally:
            self._acestep_move("dit", "cpu")
            self._acestep_empty_cache()

        if progress_callback:
            try:
                progress_callback(inference_steps, inference_steps)
            except Exception:
                pass

        # ---- validate latents (mirrors generate_music_decode.py's guards) ----
        if torch.isnan(pred_latents).any() or torch.isinf(pred_latents).any():
            raise RuntimeError(
                f"ACE-Step {mode} generation produced NaN/Inf latents "
                f"(shape={list(pred_latents.shape)}, dtype={pred_latents.dtype})."
            )
        if pred_latents.numel() > 0 and pred_latents.abs().sum() == 0:
            raise RuntimeError(f"ACE-Step {mode} generation produced all-zero latents.")

        # ---- VAE decode stage ----
        self._acestep_move("vae", device)
        try:
            vae_dtype = next(vae.parameters()).dtype
            pred_for_decode = pred_latents.transpose(1, 2).contiguous().to(vae_dtype)  # [1, 64, T]
            with torch.inference_mode():
                waveform = vae.decode(pred_for_decode).sample  # [1, 2, samples]
            waveform = waveform.float()
            peak = waveform.abs().amax(dim=[1, 2], keepdim=True)
            if torch.any(peak > 1.0):
                waveform = waveform / peak.clamp(min=1.0)
        finally:
            self._acestep_move("vae", "cpu")
            self._acestep_empty_cache()

        waveform_out = waveform[0]  # [2, samples]

        # ---- repaint-only: post-decode waveform splice (recipe section 5 step 8) ----
        if is_repaint:
            s, e = repaint_frame_range
            waveform_out = self._acestep_apply_repaint_waveform_splice(
                waveform_out,
                ref_wav.to(device=waveform_out.device, dtype=waveform_out.dtype),
                start_sec=s / 25.0,  # latent-frame-derived boundary, NOT the raw
                end_sec=e / 25.0,    # user seconds -- exactly matches the mask.
                sample_rate=48000,
                crossfade_duration=0.01,
            )

        sample_rate = int(comps.get("sample_rate", 48000))
        return waveform_out.detach().cpu(), sample_rate, seed

    # ------------------------------------------------------------------
    # Audio temporal outpaint (extend): place a (trimmed) input clip at a
    # time offset inside a LONGER output timeline and generate the audio
    # before and/or after it, preserving the placed input sample-exact.
    #
    # This is the structural INVERSE of the REPAINT branch above:
    #   - repaint:  window [s, e) is FREE (generate); everything OUTSIDE is
    #     HELD to the reference. `repaint_mask` True INSIDE the window.
    #   - outpaint: window [off, off+T_ref) is HELD to the placed input;
    #     everything OUTSIDE (before AND after) is FREE (generate).
    #     `repaint_mask` True OUTSIDE the window, False INSIDE.
    #
    # Confirmed against the vendored sampler itself (NOT assumed) --
    # `vendor/modeling_acestep_v15_turbo.py`:
    #   `_repaint_step_injection(xt, clean_src, mask, t_next, noise)`:
    #       zt = t_next * noise + (1 - t_next) * clean_src
    #       m = mask.unsqueeze(-1).expand_as(xt)
    #       return torch.where(m, xt, zt)
    #   i.e. mask=True -> keep the free-running `xt` (generate); mask=False
    #   -> replace with the correctly-noised `clean_src` (hold to
    #   reference). `repaint_mask` therefore ALWAYS means "True = generate
    #   freely, False = hold to clean_src_latents" -- outpaint only flips
    #   WHICH latent frames get which value relative to repaint, it does not
    #   change the mask's semantics.
    #   `_repaint_boundary_blend(x_gen, clean_src, mask, cf_frames)` performs
    #   the analogous post-loop boundary soft-blend using the same mask
    #   polarity (`m * x_gen + (1 - m) * clean_src`), with the crossfade
    #   ramp straddling the True/False boundary -- this can let a small
    #   amount of `x_gen` leak into the edge of the HELD (False) region at
    #   the LATENT level (exactly like repaint's own boundary blend does to
    #   its held region), which is why the post-decode WAVEFORM splice below
    #   is still required for a sample-exact guarantee: it unconditionally
    #   overwrites the ENTIRE held span with the original waveform,
    #   regardless of what the latent-level blend did near its edges.
    # ------------------------------------------------------------------

    def _generate_audoutpaint_acestep(
        self,
        params: Dict[str, Any],
        reference_audio,
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """Audio temporal outpaint (extend) for ACE-Step 1.5 turbo. See the
        class-level comment block immediately above for the full mask-
        polarity/inversion argument (verified against the vendored sampler).

        Args:
            params: see `OUTPAINT_AUDIO_DEFAULTS` -- prompt/caption (str),
                lyrics (str), seed (int, -1 = random), inference_steps (int,
                default 8), guidance_scale (forced 1.0 -- turbo is
                CFG-distilled), shift (float, default 3.0), vocal_language /
                bpm / key_scale / time_signature (folded into the "# Metas"
                text block, see `_acestep_build_text_prompt`; NOT exposed via
                the `/generate/outpaint/audio` route Form params today,
                mirroring aud2aud), loras, total_duration (float seconds,
                the OUTPUT timeline length), input_offset_sec (float
                seconds, where the trimmed input clip is placed within that
                timeline), input_trim_start_sec/input_trim_end_sec (float
                seconds, trim the UPLOADED clip itself before placement).
            reference_audio: a file path (str) or raw audio bytes for the
                input clip to place.
            progress_callback: called as (step, total_steps); coarse
                (start/end) only, see `_generate_txt2aud_acestep`.
            step_callback: reserved, unused.

        Returns:
            (waveform, sample_rate, actual_seed) -- identical contract to
            `_generate_txt2aud_acestep` / `_generate_aud2aud_acestep`.
        """
        from api.error_handlers import ValidationError

        if not self.is_acestep_model or self.acestep_components is None:
            raise ValidationError(
                "Audio outpaint requires an ACE-Step model",
                detail="The currently loaded model is not an ACE-Step audio model.",
            )

        comps = self.acestep_components
        dit = comps.get("dit")
        vae = comps.get("vae")
        text_encoder = comps.get("text_encoder")
        tokenizer = comps.get("tokenizer")
        if dit is None or vae is None or text_encoder is None or tokenizer is None:
            raise ValidationError(
                "ACE-Step model is missing a required component",
                detail=f"dit={dit is not None}, vae={vae is not None}, "
                       f"text_encoder={text_encoder is not None}, tokenizer={tokenizer is not None}",
            )
        if reference_audio is None:
            raise ValidationError(
                "Audio outpaint requires an input audio file to place",
                detail="No reference_audio was provided.",
            )

        # ---- optional LoRA (see the "LoRA" section below for the apply/restore contract) ----
        self._apply_or_clear_lora_acestep(params.get("loras") or [])

        # ---- one-time in-place INT8 conversion (unet_quantization="int8") ----
        # After the LoRA gate and before staging; see _acestep_runtime_int8 for
        # why that order is the load-bearing one. No-op for every other value and
        # for an already-converted / already-quantized DiT.
        self._acestep_runtime_int8(params, progress_callback=progress_callback)

        device = self.device
        model_dtype = next(dit.parameters()).dtype

        caption = params.get("prompt") or params.get("caption") or ""
        lyrics = params.get("lyrics", "") or ""
        seed = params.get("seed", -1)
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)
        inference_steps = int(params.get("inference_steps", 8) or 8)
        guidance_scale = float(params.get("guidance_scale", 1.0) or 1.0)
        shift = float(params.get("shift", 3.0) or 3.0)
        infer_method = params.get("infer_method", "ode")
        bpm = params.get("bpm")
        key_scale = params.get("key_scale", "") or ""
        time_signature = params.get("time_signature", "") or ""
        vocal_language = params.get("vocal_language", "en") or "en"

        total_duration = float(params.get("total_duration", 60.0) or 60.0)
        if total_duration <= 0:
            raise ValidationError(
                "Invalid total_duration",
                detail=f"total_duration must be > 0, got {total_duration}.",
            )
        input_offset_sec = max(0.0, float(params.get("input_offset_sec", 0.0) or 0.0))
        trim_start_sec = max(0.0, float(params.get("input_trim_start_sec", 0.0) or 0.0))
        trim_end_sec = max(0.0, float(params.get("input_trim_end_sec", 0.0) or 0.0))

        if guidance_scale != 1.0:
            print(
                f"[AceStep] Turbo is CFG-distilled (no twin forward pass); "
                f"overriding guidance_scale {guidance_scale} -> 1.0."
            )

        if progress_callback:
            try:
                progress_callback(0, inference_steps)
            except Exception:
                pass

        # ---- one-time silence-latent asset (shared with txt2aud/aud2aud) ----
        silence_latent = self._acestep_ensure_silence_latent(device)  # [1, 750, 64] on device

        # ---- load + normalize + trim the input clip, then VAE-encode it ----
        ref_wav, ref_sr = self._acestep_load_reference_audio(reference_audio)
        ref_wav = self._acestep_normalize_stereo_48k(ref_wav, ref_sr)  # [2, samples], CPU float32

        total_src_samples = ref_wav.shape[-1]
        start_trim_samples = int(round(trim_start_sec * 48000))
        end_trim_samples = int(round(trim_end_sec * 48000))
        trim_end_idx = (total_src_samples - end_trim_samples) if end_trim_samples > 0 else total_src_samples
        trim_end_idx = max(0, min(trim_end_idx, total_src_samples))
        start_trim_samples = max(0, min(start_trim_samples, trim_end_idx))
        trimmed_wav = ref_wav[:, start_trim_samples:trim_end_idx]  # [2, samples], the EXACT content to preserve
        if trimmed_wav.shape[-1] < 1:
            raise ValidationError(
                "Audio outpaint input trim leaves no samples",
                detail=f"input has {total_src_samples} samples @ 48kHz; "
                       f"input_trim_start_sec={trim_start_sec}, input_trim_end_sec={trim_end_sec}.",
            )

        self._acestep_move("vae", device)
        try:
            vae_dtype = next(vae.parameters()).dtype
            ref_wav_dev = trimmed_wav.unsqueeze(0).to(device=device, dtype=vae_dtype)  # [1, 2, samples]
            with torch.inference_mode():
                # .mode() (deterministic mean), matching the silence-latent asset / aud2aud.
                ref_latent = vae.encode(ref_wav_dev).latent_dist.mode()  # [1, 64, T_ref]
            ref_latent = ref_latent.transpose(1, 2).contiguous().to(model_dtype)  # [1, T_ref, 64]
        finally:
            self._acestep_move("vae", "cpu")
            self._acestep_empty_cache()

        t_ref = int(ref_latent.shape[1])
        if t_ref < 1:
            raise ValidationError(
                "Input audio is too short to encode",
                detail=f"VAE-encoded (trimmed) input latent has {t_ref} frames (need >= 1).",
            )

        # ---- output timeline placement math (25 Hz latent rate) ----
        t_total = max(1, int(round(total_duration * 25.0)))
        if t_ref > t_total:
            raise ValidationError(
                "Input audio (after trim) does not fit inside total_duration",
                detail=f"Trimmed input is {t_ref / 25.0:.3f}s ({t_ref} latent frames); "
                       f"total_duration is {total_duration:.3f}s ({t_total} latent frames). "
                       f"Trim the input further or increase total_duration.",
            )
        desired_off = int(round(input_offset_sec * 25.0))
        off = max(0, min(desired_off, t_total - t_ref))
        if off != desired_off:
            try:
                from api.generation_status import add_warning
                add_warning(
                    f"input_offset_sec clamped from {desired_off / 25.0:.3f}s to {off / 25.0:.3f}s "
                    f"so the placed input ({t_ref / 25.0:.3f}s) fits inside total_duration "
                    f"({t_total / 25.0:.3f}s).",
                    code="outpaint_audio_offset_clamped",
                )
            except Exception:
                pass

        # Surface the EFFECTIVE preserved span into params -- routes.py's
        # params.copy() -> gallery metadata/DB path picks this up
        # automatically, mirroring the video-outpaint `outpaint_effective_*`
        # convention (`core.pipeline_backends.ltx2._generate_vidoutpaint_ltx2`).
        params["outpaint_effective_offset_sec"] = off / 25.0
        params["outpaint_effective_duration_sec"] = t_ref / 25.0
        params["outpaint_effective_total_duration_sec"] = t_total / 25.0

        audio_duration = t_total / 25.0

        text_prompt = self._acestep_build_text_prompt(
            caption, audio_duration, bpm, key_scale, time_signature, vocal_language
        )
        lyrics_text = self._acestep_format_lyrics(lyrics, vocal_language)

        # ---- text encoder stage ----
        self._acestep_move("text_encoder", device)
        try:
            tt = tokenizer(
                text_prompt, padding="longest", truncation=True, max_length=256, return_tensors="pt"
            )
            text_ids = tt.input_ids.to(device)
            text_attention_mask = tt.attention_mask.to(device).bool()

            lt = tokenizer(
                lyrics_text, padding="longest", truncation=True, max_length=2048, return_tensors="pt"
            )
            lyric_ids = lt.input_ids.to(device)
            lyric_attention_mask = lt.attention_mask.to(device).bool()

            with torch.inference_mode():
                text_hidden_states = text_encoder(input_ids=text_ids).last_hidden_state.to(model_dtype)
                lyric_hidden_states = text_encoder.embed_tokens(lyric_ids).to(model_dtype)
        finally:
            self._acestep_move("text_encoder", "cpu")
            self._acestep_empty_cache()

        # ---- outpaint conditioning (INVERTED repaint -- see the class-level
        # comment block above for the mask-polarity proof) ----
        silence_full = self._acestep_silence_slice(silence_latent, t_total).to(model_dtype)  # [1, T_total, 64]

        src_latents = silence_full.clone()  # [1, T_total, 64]
        src_latents[:, off:off + t_ref, :] = ref_latent

        chunk_masks = torch.ones(1, t_total, 64, dtype=model_dtype, device=device)
        chunk_masks[:, off:off + t_ref, :] = 0.0  # 0 INSIDE the held/placed span (opposite fill from repaint)

        is_covers = torch.zeros(1, dtype=torch.bool, device=device)

        repaint_mask = torch.ones(1, t_total, dtype=torch.bool, device=device)
        repaint_mask[:, off:off + t_ref] = False  # True = generate (free) OUTSIDE, False = hold to input INSIDE
        # The FULL [silence-elsewhere + placed-input] tensor is the "hold"
        # target `clean_src_latents` -- but the silence-filled frames are
        # NEVER actually read as a hold value: `repaint_mask` is True
        # everywhere outside [off, off+t_ref), so `_repaint_step_injection`'s
        # `torch.where(mask, xt, zt)` always takes the free-running `xt`
        # branch there (see the class-level comment block above). Only the
        # placed-input sub-range of `clean_src_latents` is ever selected by
        # the False (hold) branch of the mask.
        clean_src_latents = src_latents

        # Silence timbre (semantic-only, same as txt2aud/aud2aud).
        timbre_packed = silence_latent[:, :silence_latent.shape[1], :].to(model_dtype)  # [1, 750, 64]
        refer_audio_order_mask = torch.zeros(1, dtype=torch.long, device=device)

        # ---- optional custom timestep schedule for inference_steps != 8 ----
        custom_timesteps = None
        if inference_steps and inference_steps != 8:
            n = min(max(int(inference_steps), 1), 20)
            raw = [1.0 - i / n for i in range(n)]
            if shift != 1.0:
                raw = [shift * t / (1.0 + (shift - 1.0) * t) for t in raw]
            custom_timesteps = torch.tensor(raw, device=device, dtype=model_dtype)

        # ---- DiT stage: call the vendored generate_audio (internal sampling loop) ----
        self._acestep_move("dit", device)
        try:
            with torch.inference_mode():
                outputs = dit.generate_audio(
                    text_hidden_states=text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    lyric_hidden_states=lyric_hidden_states,
                    lyric_attention_mask=lyric_attention_mask,
                    refer_audio_acoustic_hidden_states_packed=timbre_packed,
                    refer_audio_order_mask=refer_audio_order_mask,
                    src_latents=src_latents,
                    chunk_masks=chunk_masks,
                    is_covers=is_covers,
                    silence_latent=silence_latent,
                    seed=seed,
                    infer_method=infer_method,
                    shift=shift,
                    timesteps=custom_timesteps,
                    repaint_mask=repaint_mask,
                    clean_src_latents=clean_src_latents,
                    # repaint_crossfade_frames (10 latent frames) /
                    # repaint_injection_ratio (0.5) intentionally left at the
                    # vendored generate_audio's own defaults, same as
                    # aud2aud's repaint mode -- not yet exposed as user
                    # params.
                    dcw_enabled=False,
                )
            pred_latents = outputs["target_latents"]  # [1, T_total, 64]
        finally:
            self._acestep_move("dit", "cpu")
            self._acestep_empty_cache()

        if progress_callback:
            try:
                progress_callback(inference_steps, inference_steps)
            except Exception:
                pass

        # ---- validate latents (mirrors generate_music_decode.py's guards) ----
        if torch.isnan(pred_latents).any() or torch.isinf(pred_latents).any():
            raise RuntimeError(
                f"ACE-Step outpaint generation produced NaN/Inf latents "
                f"(shape={list(pred_latents.shape)}, dtype={pred_latents.dtype})."
            )
        if pred_latents.numel() > 0 and pred_latents.abs().sum() == 0:
            raise RuntimeError("ACE-Step outpaint generation produced all-zero latents.")

        # ---- VAE decode stage ----
        self._acestep_move("vae", device)
        try:
            vae_dtype = next(vae.parameters()).dtype
            pred_for_decode = pred_latents.transpose(1, 2).contiguous().to(vae_dtype)  # [1, 64, T_total]
            with torch.inference_mode():
                waveform = vae.decode(pred_for_decode).sample  # [1, 2, samples]
            waveform = waveform.float()
            peak = waveform.abs().amax(dim=[1, 2], keepdim=True)
            if torch.any(peak > 1.0):
                waveform = waveform / peak.clamp(min=1.0)
        finally:
            self._acestep_move("vae", "cpu")
            self._acestep_empty_cache()

        waveform_out = waveform[0]  # [2, samples]

        # ---- strict preservation: splice the ORIGINAL (pre-VAE, trimmed)
        # input waveform back into [off_sec, off_sec + ref_dur_sec), with
        # crossfade ramps confined to the GENERATED side of each boundary ----
        waveform_out = self._acestep_apply_outpaint_waveform_splice(
            waveform_out,
            trimmed_wav.to(device=waveform_out.device, dtype=waveform_out.dtype),
            offset_sec=off / 25.0,   # latent-frame-derived boundary, NOT the
            dur_sec=t_ref / 25.0,    # raw user input -- exactly matches the mask.
            sample_rate=48000,
            crossfade_ms=10.0,
        )

        sample_rate = int(comps.get("sample_rate", 48000))
        return waveform_out.detach().cpu(), sample_rate, seed
