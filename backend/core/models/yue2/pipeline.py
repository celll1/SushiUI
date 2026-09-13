"""SushiUI lifecycle for Apache-derived YuE2 AR, NAR, and VAE stages."""
from __future__ import annotations

import time
from dataclasses import replace
from typing import NamedTuple
import torch

from .vendor.protocol import SongRequest, GenerationConfig, token_prefixes, negative_prefix, CODEC_OFFSET
from .vendor.sampling import generate_tokens
from .vendor.nar import CachedNAR, song_chunks


class YuE2Txt2AudResult(NamedTuple):
    waveform: torch.Tensor
    sample_rate: int
    seed: int
    abc_text: str | None
    abc_ids: list[int]
    semantic_tokens: list[int]
    latents: torch.Tensor
    truncated: dict
    effective_config: dict
    timings: dict
    model_identity: dict


def ar_modules(model):
    yield model.model.embed_tokens
    yield model.model.norm
    yield model.lm_head
    for layer in model.model.layers:
        yield from (layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp)


def nar_modules(model):
    yield from (model.vae2llm, model.llm2vae, model.time_embedder, model.latent_pos_embed, model.nar_norm)
    for layer in model.model.layers:
        yield from (layer.nar_input_layernorm, layer.nar_self_attn, layer.nar_pre_mlp_layernorm, layer.nar_mlp)


def move(modules, device):
    for module in modules:
        module.to(device=device)


def validate_context_budget(request, tokenizer, config):
    """Reserve both declared token budgets before staging any model weights."""
    prefix = token_prefixes(request, tokenizer)
    budget = config.semantic.max_tokens
    if request.cot != "off" and request.abc is None:
        budget += config.abc.max_tokens + 2  # ABC_END and MUSIC_START.
    if len(prefix) + budget > config.context:
        raise ValueError("Prefix + requested generation budget exceeds 24576; no implicit truncation")
    return prefix


@torch.inference_mode()
def generate(components, params, device, seed, cancelled, progress, set_adapter_stage=None):
    model, vae, tokenizer = (components[key] for key in ("transformer", "vae", "tokenizer"))
    request = SongRequest(style=params["prompt"], lyrics=params["lyrics"], cot=params["yue2_cot"],
                          seed=seed, abc=params["yue2_abc"] or None, cfg_scale=params["guidance_scale"])
    config = GenerationConfig()
    maximum = round(float(params["audio_duration"]) * 25)
    semantic = replace(config.semantic, temperature=float(params["temperature"]), top_p=float(params["top_p"]),
                       top_k=int(params["top_k"]), repetition_penalty=float(params["repetition_penalty"]),
                       max_tokens=maximum, min_tokens=min(config.semantic.min_tokens, maximum))
    abc_sampling = replace(config.abc, max_tokens=int(params["yue2_abc_max_tokens"]),
                           min_tokens=min(config.abc.min_tokens, int(params["yue2_abc_max_tokens"])))
    config = replace(config, abc=abc_sampling, semantic=semantic)
    timings, abc_ids, abc_text = {}, [], None
    abc_truncated = False
    decode_mode = params["vae_decode_mode"]
    if decode_mode not in {"full", "tiled"}:
        raise ValueError("YuE2 VAE decode mode must be full or tiled")
    initial_prefix = validate_context_budget(request, tokenizer, config)
    def check():
        if cancelled():
            raise InterruptedError("YuE2 generation cancelled")
    counts = {"abc": 0, "semantic": 0}
    def token_progress(stage, token):
        counts[stage] += 1
        progress(stage, counts[stage], abc_sampling.max_tokens if stage == "abc" else semantic.max_tokens)
    try:
        check()
        move(ar_modules(model), device)
        if request.cot != "off":
            if request.abc is None:
                if set_adapter_stage is not None:
                    set_adapter_stage("abc")
                abc_ids, timings["abc"], abc_truncated = generate_tokens(
                    model, initial_prefix, abc_sampling, seed, "abc",
                    cancelled=cancelled, on_token=token_progress, use_cuda_graph=False)
                abc_text = tokenizer.decode(abc_ids)
            else:
                abc_text, abc_ids = request.abc, tokenizer.encode(request.abc)
        prefix = token_prefixes(request, tokenizer, abc_ids if request.cot != "off" else None)
        negative = negative_prefix(request, tokenizer, abc_ids if request.cot != "off" else None)
        if set_adapter_stage is not None:
            set_adapter_stage("semantic")
        tokens, timings["semantic"], semantic_truncated = generate_tokens(
            model, prefix, semantic, seed, "semantic", negative=negative, cfg_scale=request.guidance,
            legacy_off=request.cot == "off", cancelled=cancelled, on_token=token_progress, use_cuda_graph=False)
        codec = [token - CODEC_OFFSET for token in tokens]
        chunks = song_chunks(prefix, codec, seed)
        latents, started = [], time.perf_counter()
        move(ar_modules(model), "cpu")
        if set_adapter_stage is not None:
            set_adapter_stage("nar")
        for index, chunk in enumerate(chunks):
            check()
            # CachedNAR reads acoustic projection dtype/device while building the AR prefix.
            move(ar_modules(model), device)
            move((model.vae2llm, model.latent_pos_embed), device)
            engine = CachedNAR(model, chunk)
            try:
                move(ar_modules(model), "cpu")
                move(nar_modules(model), device)
                latents.append(engine.solve(steps=32, cancelled=cancelled,
                    on_progress=lambda n, total: progress("nar", index * total + n, len(chunks) * total)))
            finally:
                engine.close()
                move(nar_modules(model), "cpu")
        latent = torch.cat(latents)
        timings["nar"] = {"seconds": time.perf_counter() - started, "chunks": len(chunks), "steps": 32}
        check()
        vae.decoder.to(device=device)
        started = time.perf_counter()
        def tile_progress(n, total):
            check()
            progress("vae", n, total)
        effective_decode_mode, oom_fallback = decode_mode, False
        if decode_mode == "full":
            try:
                waveform = vae.decode(latent.T[None]).float().cpu()[0]
            except torch.cuda.OutOfMemoryError:
                if torch.device(device).type != "cuda":
                    raise
                torch.cuda.empty_cache()
                effective_decode_mode, oom_fallback = "tiled", True
        if effective_decode_mode == "tiled":
            waveform = vae.decode_tiled(latent.T[None], core_frames=int(params["vae_tile_frames"]),
                                       halo_frames=16, on_progress=tile_progress)[0]
        waveform = waveform.float().cpu()
        check()
        if not torch.isfinite(waveform).all():
            raise FloatingPointError("YuE2 VAE produced nonfinite audio")
        waveform = waveform.clamp(-1, 1)
        progress("vae", 1, 1)
        timings["vae"] = {"seconds": time.perf_counter() - started}
        return YuE2Txt2AudResult(waveform, 48000, seed, abc_text, abc_ids, codec, latent,
            {"abc": abc_truncated, "semantic": semantic_truncated},
            {**config.to_dict(), "request": request.to_dict(), "guidance_scale": request.guidance,
             "audio_duration": float(params["audio_duration"]), "vae_decode_mode": effective_decode_mode,
             "requested_vae_decode_mode": decode_mode, "vae_oom_fallback": oom_fallback,
             "vae_tile_frames": int(params["vae_tile_frames"]), "vae_halo_frames": 16,
             "prefix_ids": prefix}, timings, components["model_identity"])
    finally:
        if set_adapter_stage is not None:
            set_adapter_stage(None)
        model.to(device="cpu")
        vae.to(device="cpu")
        model.model.rotary_emb._inv_freq = None
