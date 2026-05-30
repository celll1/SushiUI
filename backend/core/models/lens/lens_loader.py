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
    vae = AutoencoderKLFlux2.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
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
        "scheduler": scheduler,
    }
