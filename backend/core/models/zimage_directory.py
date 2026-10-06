"""Local official Z-Image packages using SushiUI's trainable transformer."""
import json
from pathlib import Path


def validate_component_directory(value):
    if not isinstance(value, str) or len(value) > 4096 or not Path(value).is_absolute():
        raise ValueError("Z-Image saved component directory must be an absolute path")
    root = Path(value)
    index = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
    if index.get("_class_name") != "ZImagePipeline":
        raise ValueError("Z-Image saved component directory is not a ZImagePipeline package")
    for component in ("text_encoder", "tokenizer", "scheduler", "transformer", "vae"):
        if not (root / component).is_dir():
            raise ValueError(f"Z-Image saved component directory is missing {component}")
    return root


def load_directory(path, *, device, torch_dtype, build_transformer):
    import torch
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from transformers import AutoModel, AutoTokenizer
    from .common.single_file_format import read_state_dict
    from .common.quantized_checkpoint_guard import quantized_state_dict_report

    root = Path(path)
    index = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
    component = index.get("transformer")
    if (index.get("_class_name") != "ZImagePipeline" or not isinstance(component, list)
            or len(component) != 2 or component[1] != "ZImageTransformer2DModel"):
        raise ValueError("Expected an official ZImagePipeline package")
    config = json.loads((root / "transformer/config.json").read_text(encoding="utf-8"))
    if config.get("_class_name") != "ZImageTransformer2DModel":
        raise ValueError("Unexpected Z-Image transformer configuration")
    if config.get("siglip_feat_dim") is not None:
        raise ValueError("Z-Image image-encoder conditioning is not supported by this transformer")
    weights = root / "transformer/diffusion_pytorch_model.safetensors"
    if not weights.is_file():
        weights = weights.with_suffix(".safetensors.index.json")
    state, _ = read_state_dict(str(weights))
    if quantized_state_dict_report(state) is None:
        state = {key: value.to(torch_dtype) if value.is_floating_point() else value for key, value in state.items()}
    transformer = build_transformer(config, config["n_layers"], config["in_channels"],
        state, "official", torch_dtype, path=str(root))
    del state
    transformer.to(device).eval()
    vae_path = root / "vae"
    vae = AutoencoderKL.from_pretrained(str(vae_path), torch_dtype=torch.float32, local_files_only=True)
    if vae.config.latent_channels != config["in_channels"]:
        raise ValueError("Z-Image VAE and transformer latent channels disagree")
    vae.to(device).eval()
    text_encoder = AutoModel.from_pretrained(str(root / "text_encoder"),
        torch_dtype=torch_dtype, local_files_only=True, trust_remote_code=False)
    if text_encoder.config.hidden_size != config["cap_feat_dim"]:
        raise ValueError("Z-Image text encoder and caption feature width disagree")
    text_encoder.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"),
        local_files_only=True, trust_remote_code=False)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(root / "scheduler"), local_files_only=True)
    return dict(transformer=transformer, vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, scheduler=scheduler, vae_source=str(vae_path), vae_path=str(vae_path),
        vae_type="sdxl" if config["in_channels"] == 4 else "flux", declared_vae=None, vae_identity=None,
        base_component_directory=str(root.resolve()))
