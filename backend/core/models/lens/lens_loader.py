"""Component loader for Microsoft/Lens models.

Loads each sub-model to CPU; pipeline.py stages them to GPU per generation phase.
Supports both HuggingFace Hub IDs (str) and local directory paths.
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

    print("[LensLoader] Loading text encoder (LensGptOssEncoder)...")
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
