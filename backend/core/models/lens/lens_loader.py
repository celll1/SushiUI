"""Component loader for Microsoft/Lens models.

Loads each sub-model to CPU; pipeline.py stages them to GPU per generation phase.
Supports both HuggingFace Hub IDs (str) and local directory paths.

NOTE — Lens text encoder VRAM:
  The text encoder (LensGptOssEncoder) uses mxfp4 quantization via the `kernels`
  library.  During from_pretrained the library allocates ~9.7 GB of CUDA memory
  for packed FP4 weight buffers.  These buffers are NOT tracked by PyTorch's
  named_parameters() / named_buffers(), so .to('cpu') cannot free them.
  ~9.7 GB of VRAM is therefore permanently consumed while a Lens model is loaded.
"""

import os

import torch


def load_lens_components(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Load Lens components from a HF repo or local diffusers directory.

    Returns a component dict consumed by PipelineManager.load_model():
        {
            "type": "lens",
            "transformer": LensTransformer2DModel,
            "text_encoder": LensGptOssEncoder,
            "tokenizer": PreTrainedTokenizerFast,
            "vae": AutoencoderKLFlux2,
            "scheduler": FlowMatchEulerDiscreteScheduler,
        }
    """
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler

    # Vendor import — also registers classes in diffusers/transformers namespaces
    # so that from_pretrained can resolve LensTransformer2DModel / LensGptOssEncoder.
    from core.models.lens.vendor import LensGptOssEncoder, LensTransformer2DModel

    print(f"[LensLoader] Loading components from: {model_path}")

    print("[LensLoader] Loading transformer (LensTransformer2DModel)...")
    transformer = LensTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    transformer.eval()
    transformer.to("cpu")

    # The mxfp4 text encoder allocates ~9.7 GB of CUDA memory via the `kernels`
    # library during from_pretrained.  This is unavoidable — .to('cpu') moves only
    # the regular (non-quantized) PyTorch params; the FP4 weight buffers remain on
    # GPU.  During generation the non-quantized params are moved to GPU by
    # _lens_move(); they are returned to CPU afterwards to reclaim that small slice.
    print("[LensLoader] Loading text encoder (LensGptOssEncoder, mxfp4 — allocates ~9.7 GB VRAM)...")
    text_encoder = LensGptOssEncoder.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    text_encoder.eval()
    text_encoder.to("cpu")
    # Ensure layer-selection is configured (matches transformer config)
    selected_layers = tuple(transformer.config.selected_layer_index)
    text_encoder.set_selected_layers(selected_layers)

    print("[LensLoader] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Sanity encode: surface corrupted vocabulary files at load time rather than
    # at the first generation step where the error message is less informative.
    try:
        tokenizer.encode("validation", add_special_tokens=False)
    except Exception as e:
        raise RuntimeError(
            f"[LensLoader] Tokenizer sanity encode failed — vocabulary files may be corrupted "
            f"({model_path}/tokenizer): {e}"
        ) from e

    print("[LensLoader] Loading VAE (AutoencoderKLFlux2)...")
    # Primary: the model's own vae/ subfolder (Lens ships its own). Fallback: the
    # shared Apache-2.0 FLUX.2 store (black-forest-labs/FLUX.2-klein-4B vae).
    own_vae = os.path.join(model_path, "vae") if os.path.isdir(model_path) else None
    vae_source = None
    vae_path = None
    if own_vae and os.path.isfile(os.path.join(own_vae, "config.json")):
        vae = AutoencoderKLFlux2.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        )
        vae_source = own_vae
        vae_path = own_vae
    else:
        store_dir = None
        try:
            from core.models.common.vae_store import resolve_vae_dir
            store_dir = resolve_vae_dir("flux2", model_own_vae=own_vae)
        except Exception as e:
            print(f"[LensLoader] FLUX.2 VAE store resolution failed: {e}")
        if store_dir and os.path.isdir(store_dir):
            print(f"[LensLoader] Loading VAE from shared FLUX.2 store: {store_dir}")
            vae = AutoencoderKLFlux2.from_pretrained(
                store_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
            )
            vae_source = str(store_dir)
            vae_path = str(store_dir)
        else:
            vae = AutoencoderKLFlux2.from_pretrained(
                model_path, subfolder="vae", torch_dtype=torch_dtype, low_cpu_mem_usage=True,
            )
            fallback_vae = os.path.join(model_path, "vae")
            vae_source = fallback_vae
            vae_path = fallback_vae if os.path.isdir(fallback_vae) else None
    vae.eval()
    vae.to("cpu")

    print("[LensLoader] Loading scheduler (FlowMatchEulerDiscreteScheduler)...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path,
        subfolder="scheduler",
    )

    print("[LensLoader] All components loaded successfully.")
    return {
        "type": "lens",
        "transformer": transformer,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "vae": vae,
        "vae_source": vae_source,
        "vae_path": vae_path,
        "scheduler": scheduler,
        "base_dir": model_path,
    }


def _resolve_lens_base_dir(dit_path: str, base_dir_hint: str = None) -> str:
    """Resolve a base Lens diffusers directory (transformer/text_encoder/vae/tokenizer/
    scheduler subfolders) for a single-file DiT save.

    Search order:
      1. ``base_dir_hint`` (from the DiT file metadata / caller)
      2. ``settings.models_dir`` entries whose name contains "lens"
      3. sibling / ancestor directories of the DiT file (up to 4 levels)
      4. sibling SUBDIRECTORIES of the DiT file's parent (one level down): a
         root-level single file at ``M:/model/lens/lens_transformer.safetensors``
         thus finds the base dir ``M:/model/lens/microsoft-lens/``.
    A directory qualifies when it contains ``transformer/config.json``.
    """
    import os

    def _is_lens_dir(d: str) -> bool:
        return bool(d) and os.path.isdir(d) and os.path.isfile(
            os.path.join(d, "transformer", "config.json")
        )

    searched = []
    if base_dir_hint:
        searched.append(base_dir_hint)
        if _is_lens_dir(base_dir_hint):
            return base_dir_hint

    models_root = None
    try:
        from config.settings import settings
        models_root = getattr(settings, "models_dir", None)
    except Exception:
        models_root = None
    if models_root and os.path.isdir(models_root):
        for name in os.listdir(models_root):
            if "lens" in name.lower():
                cand = os.path.join(models_root, name)
                searched.append(cand)
                if _is_lens_dir(cand):
                    return cand

    p = os.path.abspath(dit_path)
    for _ in range(4):
        p = os.path.dirname(p)
        if not p:
            break
        searched.append(p)
        if _is_lens_dir(p):
            return p

    # Step 4: probe the immediate child subdirectories of the DiT file's parent
    # (one level down). A root-level single file sits next to the base diffusers
    # dir rather than inside it, so the ancestor walk above cannot reach it.
    # Prefer a child whose name contains "lens" when several qualify.
    parent = os.path.dirname(os.path.abspath(dit_path))
    if parent and os.path.isdir(parent):
        sibling_matches = []
        for name in sorted(os.listdir(parent)):
            cand = os.path.join(parent, name)
            searched.append(cand)
            if _is_lens_dir(cand):
                sibling_matches.append(cand)
        if sibling_matches:
            sibling_matches.sort(key=lambda d: 0 if "lens" in os.path.basename(d).lower() else 1)
            return sibling_matches[0]

    raise FileNotFoundError(
        "Lens single-file DiT requires a base Lens diffusers directory for its "
        "text encoder / VAE / tokenizer / scheduler, but none was found.\n"
        f"  DiT file: {dit_path}\n"
        "Searched (need a 'transformer/config.json' inside):\n  - "
        + "\n  - ".join(searched or ["(nothing to search)"])
        + "\nProvide the original Lens model directory next to the DiT file, "
        "or under <models_dir>/ with 'lens' in its name."
    )


def load_lens_single_file(
    dit_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    base_dir_hint: str = None,
) -> dict:
    """Load Lens from a single-file full-FT DiT save (``net.*``-prefixed weights).

    The DiT weights come from ``dit_path``; the text encoder, VAE, tokenizer and
    scheduler are resolved from a base Lens diffusers directory (see
    ``_resolve_lens_base_dir``). The base transformer is loaded and then its
    weights are overridden by the trained single-file DiT.
    """
    import os
    from core.models.common.single_file_format import read_state_dict, reattach_embedded_weights

    # Single consolidated read (also tolerates a <stem>.safetensors.index.json path).
    raw, md = read_state_dict(dit_path)
    md = md or {}
    hint = base_dir_hint or md.get("component.base_dir") or md.get("sushi.base_model_path")

    base_dir = _resolve_lens_base_dir(dit_path, hint)
    print(f"[LensLoader] Single-file DiT: {dit_path}")
    print(f"[LensLoader] Resolved base Lens directory: {base_dir}")

    components = load_lens_components(model_path=base_dir, torch_dtype=torch_dtype)

    # Split off an embedded VAE section (bundle_vae saves under first_stage_model.*).
    embedded_vae_sd = {
        k[len("first_stage_model."):]: v
        for k, v in raw.items() if k.startswith("first_stage_model.")
    } or None

    # Override the base transformer weights with the trained single-file DiT
    # (net.*-stripped; the first_stage_model.* VAE keys are excluded here).
    dit_sd = {
        (k[len("net."):] if k.startswith("net.") else k): v
        for k, v in raw.items() if not k.startswith("first_stage_model.")
    }
    # strict=False is right here (the base transformer is being OVERRIDDEN, and a
    # bundle may legitimately omit sections) and fatal for one input: a
    # weight-only quantized DiT would land every .weight_scale in unexpected_keys
    # and cast its int8/e4m3 codes into the base bf16 parameters, i.e. load
    # "successfully" into a silently wrong model. Lens has no quantized-Linear
    # swap, so refuse instead.
    from core.models.common.quantized_checkpoint_guard import refuse_quantized_state_dict
    refuse_quantized_state_dict(dit_sd, arch="lens", path=dit_path, label="DiT")

    info = components["transformer"].load_state_dict(dit_sd, strict=False)
    missing = getattr(info, "missing_keys", [])
    unexpected = getattr(info, "unexpected_keys", [])
    print(f"[LensLoader] Applied single-file DiT: missing={len(missing)}, unexpected={len(unexpected)}")
    if unexpected:
        print(f"[LensLoader]   unexpected (first 5): {list(unexpected)[:5]}")
    components["transformer"].to(torch_dtype).to("cpu").eval()

    # Embedded (trained) VAE overrides the base-dir VAE. Absent -> keep base VAE.
    if embedded_vae_sd is not None and components.get("vae") is not None:
        print("[LensLoader] Reattaching embedded VAE weights from single-file bundle")
        reattach_embedded_weights(components["vae"], embedded_vae_sd, "VAE")
        components["vae"].to(torch_dtype).to("cpu").eval()
        components["vae_source"] = "embedded (checkpoint)"
        components["vae_path"] = None

    return components


def reload_lens_text_encoder(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    selected_layers: tuple = None,
):
    """Reload only the Lens text encoder from disk (~4 s).

    Called at the start of each generation when the text encoder has been freed
    after the previous encoding stage to reclaim its ~9.7 GB of mxfp4 CUDA memory.
    """
    from core.models.lens.vendor import LensGptOssEncoder

    print("[LensLoader] Reloading text encoder (mxfp4, ~4 s)...")
    text_encoder = LensGptOssEncoder.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    text_encoder.eval()
    text_encoder.to("cpu")
    if selected_layers is not None:
        text_encoder.set_selected_layers(selected_layers)
    return text_encoder
