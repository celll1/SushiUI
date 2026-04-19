"""
SigLIP2 Encoder Extraction Utilities.

Extracts the vision or text encoder sub-module from a HuggingFace SigLIP2 model
and saves it as a safetensors file for use in tagger training or inference.

Replaces the ad-hoc scripts in test/siglip2_extract.py and
test/siglip2_extract_text_encoder.py with a proper importable module.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import torch
from safetensors.torch import save_file


def extract_vision_encoder(repo_id: str, output_path: str) -> Dict[str, Any]:
    """Extract the vision encoder sub-module from a HF SigLIP2 model.

    Parameters
    ----------
    repo_id     : HuggingFace repo ID, e.g. ``"google/siglip2-so400m-patch16-naflex"``
    output_path : Destination ``.safetensors`` path; parent directory is created if needed.

    Returns
    -------
    dict with keys:
        output_path  – absolute path to the saved file
        num_params   – total parameter count of the vision encoder
        hidden_size  – encoder hidden dimension
        num_layers   – number of transformer layers
    """
    from transformers import AutoModel

    output_path = os.path.abspath(output_path.strip().strip('"').strip("'"))
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[SigLIP2Extractor] Loading {repo_id} (vision encoder)...")
    try:
        full_model = AutoModel.from_pretrained(repo_id, torch_dtype=torch.float32, local_files_only=True)
    except Exception:
        full_model = AutoModel.from_pretrained(repo_id, torch_dtype=torch.float32)
    full_model.eval()

    if not hasattr(full_model, "vision_model"):
        raise AttributeError(f"Model {repo_id!r} has no .vision_model attribute")
    encoder = full_model.vision_model
    cfg = encoder.config

    num_params = sum(p.numel() for p in encoder.parameters())
    hidden_size = int(getattr(cfg, "hidden_size", 0))
    num_layers  = int(getattr(cfg, "num_hidden_layers", 0))

    sd = {k: v.contiguous() for k, v in encoder.state_dict().items()}
    save_file(sd, output_path)
    print(f"[SigLIP2Extractor] Vision encoder saved → {output_path}  ({num_params:,} params, hidden={hidden_size}, layers={num_layers})")

    return {
        "output_path": output_path,
        "num_params":  num_params,
        "hidden_size": hidden_size,
        "num_layers":  num_layers,
    }


def extract_text_encoder(repo_id: str, output_path: str) -> Dict[str, Any]:
    """Extract the text encoder sub-module from a HF SigLIP2 model.

    Parameters
    ----------
    repo_id     : HuggingFace repo ID, e.g. ``"google/siglip2-so400m-patch16-naflex"``
    output_path : Destination ``.safetensors`` path; parent directory is created if needed.

    Returns
    -------
    dict with keys:
        output_path  – absolute path to the saved file
        num_params   – total parameter count of the text encoder
        hidden_size  – encoder hidden dimension
        num_layers   – number of transformer layers
    """
    from transformers import AutoModel

    output_path = os.path.abspath(output_path.strip().strip('"').strip("'"))
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[SigLIP2Extractor] Loading {repo_id} (text encoder)...")
    try:
        full_model = AutoModel.from_pretrained(repo_id, torch_dtype=torch.float32, local_files_only=True)
    except Exception:
        full_model = AutoModel.from_pretrained(repo_id, torch_dtype=torch.float32)
    full_model.eval()

    if not hasattr(full_model, "text_model"):
        raise AttributeError(f"Model {repo_id!r} has no .text_model attribute")
    encoder = full_model.text_model
    cfg = encoder.config

    num_params = sum(p.numel() for p in encoder.parameters())
    hidden_size = int(getattr(cfg, "hidden_size", 0))
    num_layers  = int(getattr(cfg, "num_hidden_layers", 0))

    sd = {k: v.contiguous() for k, v in encoder.state_dict().items()}
    save_file(sd, output_path)
    print(f"[SigLIP2Extractor] Text encoder saved → {output_path}  ({num_params:,} params, hidden={hidden_size}, layers={num_layers})")

    return {
        "output_path": output_path,
        "num_params":  num_params,
        "hidden_size": hidden_size,
        "num_layers":  num_layers,
    }
