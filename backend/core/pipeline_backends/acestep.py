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
import random

import torch


class AceStepMixin:
    """AceStepMixin: ACE-Step 1.5 (2B DiT + Oobleck VAE + Qwen3-Embedding
    text encoder) text-to-music generation backend."""

    # ------------------------------------------------------------------
    # Component staging (sequential text_encoder -> DiT -> VAE; mirrors the
    # `_move` helper pattern used by the other single-file-loaded backends,
    # e.g. MiniT2IMixin._minit2i_move).
    # ------------------------------------------------------------------

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
    # ------------------------------------------------------------------

    _ACESTEP_LORA_ATTN_NAMES = ("self_attn", "cross_attn")
    _ACESTEP_LORA_ATTN_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")

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

                total_pairs = sum(1 for k in lora_state_dict if k.endswith(".lora_down.weight"))
                applied_count = 0

                for layer_idx, layer in enumerate(layers):
                    for attn_name in self._ACESTEP_LORA_ATTN_NAMES:
                        attn = getattr(layer, attn_name, None)
                        if attn is None:
                            continue

                        for leaf in self._ACESTEP_LORA_ATTN_LEAVES:
                            original_linear = getattr(attn, leaf, None)
                            if not isinstance(original_linear, torch.nn.Linear):
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
                        f"applies (skipped entirely, not an error). Either it uses a different key convention "
                        f"(e.g. `.lora_A.`/`.lora_B.` PEFT-style instead of `.lora_down.`/`.lora_up.`), or it "
                        f"was trained for a different component/architecture entirely (e.g. a lyric/text "
                        f"encoder LoRA). Expected key prefix: "
                        f"'lora_unet_decoder_layers_<i>_<self_attn|cross_attn>_<q,k,v,o>_proj.lora_down.weight'. "
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

            except Exception as e:
                print(f"[AceStep LoRA] ERROR: Failed to load LoRA {lora_path}: {e}")
                import traceback
                traceback.print_exc()

    def _wrap_with_lora_acestep(self, parent_module, attr_name, original_linear,
                                 lora_down_weight, lora_up_weight, strength, alpha, module_key):
        """Wrap a decoder attention Linear with a LoRA forward-addition layer.

        Mirrors `ZImageMixin._wrap_with_lora`/`FluxMixin._wrap_with_lora_flux2`.
        """
        from core.training.adapters.sd15_adapter import LoRALinearLayer

        if isinstance(original_linear, LoRALinearLayer):
            true_original = original_linear.original_module
        else:
            true_original = original_linear

        if module_key not in self._acestep_lora_original_modules:
            self._acestep_lora_original_modules[module_key] = true_original

        rank = lora_down_weight.shape[0]
        alpha_value = alpha.item() if alpha is not None else rank

        lora_wrapper = LoRALinearLayer(
            true_original, rank=rank, alpha=alpha_value, lora_name=module_key
        )

        device = true_original.weight.device
        dtype = true_original.weight.dtype

        with torch.no_grad():
            lora_wrapper.lora_down.weight.data = lora_down_weight.to(device=device, dtype=dtype)
            lora_wrapper.lora_up.weight.data = lora_up_weight.to(device=device, dtype=dtype)

        # Apply strength by overriding the default alpha/rank scale.
        lora_wrapper.scale = (alpha_value / rank) * strength

        setattr(parent_module, attr_name, lora_wrapper)

        self._acestep_lora_wrapped_modules.add(module_key)
        return True

    def _unload_lora_acestep(self):
        """Restore original (un-wrapped) Linear modules on the ACE-Step DiT."""
        if not hasattr(self, "_acestep_lora_original_modules"):
            print("[AceStep LoRA] No LoRAs loaded")
            return

        if not self.acestep_components:
            print("[AceStep LoRA] WARNING: ACE-Step components not loaded")
            return

        dit = self.acestep_components.get("dit")
        decoder = getattr(dit, "decoder", None)
        layers = getattr(decoder, "layers", None) if decoder is not None else None
        if layers is None:
            print("[AceStep LoRA] WARNING: DiT has no decoder.layers -- cannot unload LoRA")
            return

        unloaded_count = 0
        print(f"[AceStep LoRA] Unloading LoRAs ({len(self._acestep_lora_wrapped_modules)} modules)...")

        for layer_idx, layer in enumerate(layers):
            for attn_name in self._ACESTEP_LORA_ATTN_NAMES:
                attn = getattr(layer, attn_name, None)
                if attn is None:
                    continue
                for leaf in self._ACESTEP_LORA_ATTN_LEAVES:
                    module_key = f"decoder.layers.{layer_idx}.{attn_name}.{leaf}"
                    if module_key in self._acestep_lora_original_modules:
                        setattr(attn, leaf, self._acestep_lora_original_modules[module_key])
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
    # aud2aud (audio-to-audio COVER, img2img analog). Repaint (inpaint
    # analog) is intentionally NOT implemented -- the vendored
    # `generate_audio` (this repo's `vendor/modeling_acestep_v15_turbo.py`,
    # HEAD 6d467e4) has no `repaint_mask` / `clean_src_latents` /
    # `repaint_crossfade_frames` / `repaint_injection_ratio` kwargs; they are
    # silently swallowed by its trailing `**kwargs` (would need re-vendoring
    # the newer main-branch modeling file). See
    # `scratchpad/acestep_aud2aud_recipe.md` sections 2 and 6.
    # ------------------------------------------------------------------

    def _generate_aud2aud_acestep(
        self,
        params: Dict[str, Any],
        reference_audio,
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """Cover generation (audio-to-audio, img2img analog) for ACE-Step 1.5 turbo.

        The reference audio is VAE-encoded and fed back to the DiT as
        `src_latents` with `is_covers=True`; the vendored `generate_audio`
        internally tokenizes/detokenizes it through the model's FSQ codec to
        get a "semantic re-render" context (see
        `scratchpad/acestep_aud2aud_recipe.md` section 1b) -- this is a
        semantic-only cover (timbre stays silence, same as txt2aud), not a
        raw-latent passthrough.

        `cover_strength` (`audio_cover_strength` on the vendored model) is a
        STEP-COUNT blend, NOT an img2img start-timestep / partial-denoise
        knob: `xt` always starts from full noise and runs every step; the
        first `int(num_steps * cover_strength)` steps use the reference's
        semantic context, the remaining steps switch to a
        text2music-style (silence src_latents) context built from the SAME
        caption/lyric text. Higher `cover_strength` => closer to the
        reference. A true img2img-style partial-denoise (`cover_noise_strength`
        on the official main-branch model) is NOT available on the vendored
        model (recipe section 1c/6) and is out of scope here.

        Args:
            params: prompt/caption (str), lyrics (str), cover_strength
                (float in [0, 1], default 1.0), seed (int, -1 = random),
                inference_steps (int, default 8), guidance_scale (forced
                1.0 -- turbo is CFG-distilled), shift (float, default 3.0),
                vocal_language / bpm / key_scale / time_signature (folded
                into the "# Metas" text block, see
                `_acestep_build_text_prompt`). `audio_duration` is NOT a
                user param here -- duration is derived from the reference
                audio's length (recipe section 4).
            reference_audio: a file path (str) or raw audio bytes for the
                cover reference clip.
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
                "Audio-to-audio (cover) generation requires a reference audio file",
                detail="No reference_audio was provided.",
            )

        # ---- optional LoRA (see the "LoRA" section above for the apply/restore contract) ----
        self._apply_or_clear_lora_acestep(params.get("loras") or [])

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

        # ---- cover conditioning (recipe section 1a/1e) ----
        src_latents = ref_latent  # [1, T, 64] -- the reference latent (NOT silence)
        chunk_masks = torch.ones(1, latent_frames, 64, dtype=model_dtype, device=device)
        is_covers = torch.ones(1, dtype=torch.bool, device=device)
        # Silence timbre (semantic-only cover): matches txt2aud's timbre condition.
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
                    audio_cover_strength=cover_strength,
                    # Required whenever audio_cover_strength < 1.0 (the vendored
                    # model builds a 2nd, text2music-style conditioning from
                    # these + silence src_latents for the post-cover_steps
                    # portion of the schedule -- None + strength<1 crashes).
                    # Harmless to always pass (unused at strength==1.0).
                    non_cover_text_hidden_states=text_hidden_states,
                    non_cover_text_attention_mask=text_attention_mask,
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
                f"ACE-Step cover generation produced NaN/Inf latents "
                f"(shape={list(pred_latents.shape)}, dtype={pred_latents.dtype})."
            )
        if pred_latents.numel() > 0 and pred_latents.abs().sum() == 0:
            raise RuntimeError("ACE-Step cover generation produced all-zero latents.")

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
