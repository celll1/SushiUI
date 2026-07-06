"""TE component registry — arch-independent text-encoder component layer (plan A.4).

Generalizes ``core/models/sdxl_te_registry.py``: it hosts the canonical
implementation of the custom-TE registry (``TE_REGISTRY``, ``is_custom_te``,
``load_sdxl_te``, ``encode_text``, positional-embedding extension). The old
``core.models.sdxl_te_registry`` module is now a thin re-export shim of this one,
so every existing import keeps resolving to the SAME objects (identity preserved).

BEHAVIOR FREEZE: ``load_sdxl_te`` / ``encode_text`` and their helpers are moved
byte-identically — signatures and semantics are unchanged. The generalized,
spec-driven ``load_te`` entry point is ADDITIVE (no existing caller uses it yet;
arch handlers wire it in later phases).

Swaps a backbone's native text encoders for a single alternative encoder whose
hidden states + a pooled vector are bridged to the backbone interface by trainable
adapters (``components/bridge_adapter.py``). The encoder body is referenced (HF
repo), not embedded in the checkpoint; the adapters are saved/reloaded.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# te_type -> default HF repo (hidden_dim is read from the loaded config at runtime).
TE_REGISTRY = {
    "siglip2_text": {"repo": "google/siglip2-so400m-patch16-512", "kind": "siglip2"},
    "flan_t5": {"repo": "google/flan-t5-large", "kind": "t5"},          # added later
    "qwen3": {"repo": "Qwen/Qwen3.5-0.8B", "kind": "qwen"},             # added later
}

_DEFAULT_MAX_LEN = 256


def is_custom_te(te_type: Optional[str]) -> bool:
    return bool(te_type) and te_type not in ("none", "clip") and te_type in TE_REGISTRY


def _find_position_embedding(module: nn.Module):
    """Return (parent_module, attr_name, embedding) for the abs-position nn.Embedding."""
    for name, m in module.named_modules():
        if name.endswith("position_embedding") and isinstance(m, nn.Embedding):
            parent = module
            *parents, attr = name.split(".")
            for p in parents:
                parent = getattr(parent, p)
            return parent, attr, m
    return None, None, None


def _extend_position_embeddings(text_model: nn.Module, max_len: int) -> None:
    """Grow the learned absolute position embedding to `max_len` (linear interpolation),
    so prompts longer than the encoder's native limit (SigLIP text = 64) are supported."""
    parent, attr, pe = _find_position_embedding(text_model)
    if pe is None:
        print("[SDXL-TE] No absolute position_embedding found; skipping pos-emb extension.")
        return
    cur = pe.num_embeddings
    if max_len <= cur:
        return
    dim = pe.embedding_dim
    new = nn.Embedding(max_len, dim).to(pe.weight.device, pe.weight.dtype)
    with torch.no_grad():
        old = pe.weight.data.float()  # [cur, dim]
        interp = F.interpolate(old.t().unsqueeze(0), size=max_len, mode="linear",
                               align_corners=False).squeeze(0).t()  # [max_len, dim]
        new.weight.data.copy_(interp.to(pe.weight.dtype))
    setattr(parent, attr, new)
    # Refresh any cached position_ids buffer + config so forward() accepts the new length.
    if hasattr(parent, "position_ids"):
        parent.position_ids = torch.arange(max_len).expand((1, -1)).to(
            parent.position_ids.device if hasattr(parent.position_ids, "device") else "cpu")
    cfg = getattr(text_model, "config", None)
    if cfg is not None and hasattr(cfg, "max_position_embeddings"):
        cfg.max_position_embeddings = max_len
    print(f"[SDXL-TE] Extended position embeddings {cur} -> {max_len} (interpolated).")


def load_sdxl_te(
    te_type: str,
    *,
    repo: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
    max_len: int = _DEFAULT_MAX_LEN,
) -> Tuple[nn.Module, object, int]:
    """Load a custom SDXL text encoder + tokenizer. Returns (encoder, tokenizer, hidden_dim).

    The encoder is returned in eval mode; the caller decides requires_grad (default the
    body is frozen and only the adapters train).
    """
    te_type = (te_type or "").strip().lower()
    if te_type not in TE_REGISTRY:
        raise ValueError(f"Unknown sdxl_te_type '{te_type}' (expected one of {list(TE_REGISTRY)})")
    entry = TE_REGISTRY[te_type]
    kind = entry["kind"]
    repo = (repo or entry["repo"]).strip()

    from transformers import AutoTokenizer

    if kind == "siglip2":
        from transformers import AutoModel
        full = AutoModel.from_pretrained(repo, dtype=torch.float32)
        text_model = full.text_model
        tokenizer = AutoTokenizer.from_pretrained(repo)
        _extend_position_embeddings(text_model, max_len)
        # Some text models only emit hidden_states when the config flag is set (the
        # forward kwarg alone is ignored), so enable it for the penultimate-layer tap.
        try:
            text_model.config.output_hidden_states = True
        except Exception:
            pass
        hidden_dim = int(text_model.config.hidden_size)
        encoder = text_model.to(device=device, dtype=dtype).eval()
        print(f"[SDXL-TE] Loaded SigLIP2 text tower: {repo} (hidden={hidden_dim}, max_len={max_len})")
        return encoder, tokenizer, hidden_dim

    if kind == "t5":
        # FLAN-T5 encoder (same family as MiniT2I). T5 uses relative position bias (no
        # learned absolute position embedding), so it handles any length and needs no
        # pos-emb extension; max_len only controls tokenization (pad/truncate).
        from transformers import T5EncoderModel
        encoder = T5EncoderModel.from_pretrained(repo, dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(repo)
        try:
            encoder.config.output_hidden_states = True
        except Exception:
            pass
        hidden_dim = int(getattr(encoder.config, "d_model", None) or encoder.config.hidden_size)
        encoder = encoder.to(device=device, dtype=dtype).eval()
        print(f"[SDXL-TE] Loaded FLAN-T5 encoder: {repo} (hidden={hidden_dim}, max_len={max_len})")
        return encoder, tokenizer, hidden_dim

    if kind == "qwen":
        # Qwen3 decoder-only LLM as a text encoder. RoPE (no learned absolute pos-emb):
        # any length, no pos-emb extension; max_len controls tokenization. Causal
        # attention is kept (LLM-as-encoder); pooled = masked mean, hidden = penultimate.
        from transformers import AutoModel
        try:
            encoder = AutoModel.from_pretrained(repo, dtype=torch.float32, trust_remote_code=True)
        except TypeError:
            encoder = AutoModel.from_pretrained(repo, dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            encoder.config.output_hidden_states = True
        except Exception:
            pass
        hidden_dim = int(encoder.config.hidden_size)
        encoder = encoder.to(device=device, dtype=dtype).eval()
        print(f"[SDXL-TE] Loaded Qwen3 encoder: {repo} (hidden={hidden_dim}, max_len={max_len})")
        return encoder, tokenizer, hidden_dim

    raise NotImplementedError(
        f"sdxl_te_type '{te_type}' (kind={kind}) is registered but not yet implemented."
    )


def encode_text(
    encoder: nn.Module,
    tokenizer,
    prompts,
    *,
    max_len: int = _DEFAULT_MAX_LEN,
    hidden_layer: int = -2,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts to (hidden[B,L,D_te], pooled[B,D_te]) at fixed length max_len.

    hidden = the chosen hidden-states layer (penultimate by default); pooled = masked
    mean of the last hidden state. requires_grad follows the encoder (frozen => no grad).
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    toks = tokenizer(
        list(prompts), padding="max_length", truncation=True, max_length=max_len,
        return_tensors="pt",
    )
    input_ids = toks["input_ids"].to(device)
    attn = toks.get("attention_mask")
    attn = attn.to(device) if attn is not None else None

    try:
        encoder.config.output_hidden_states = True
    except Exception:
        pass
    out = encoder(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
    if getattr(out, "hidden_states", None):
        hidden = out.hidden_states[hidden_layer]        # [B, L, D] (penultimate by default)
    else:
        hidden = out.last_hidden_state                  # fallback if hidden_states unavailable
    last = out.last_hidden_state                         # [B, L, D]
    if attn is not None:
        m = attn.unsqueeze(-1).to(last.dtype)            # [B, L, 1]
        pooled = (last * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    else:
        pooled = last.mean(dim=1)
    return hidden, pooled


# --- Generalized, spec-driven entry point (ADDITIVE — plan A.4) -----------------

def load_te(
    spec_or_type,
    *,
    repo: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
    max_len: int = _DEFAULT_MAX_LEN,
) -> Tuple[nn.Module, object, int]:
    """Arch-independent TE loader. Accepts a te_type string OR a wiring spec that
    carries a ``te_type`` attribute, and delegates to ``load_sdxl_te`` (which is the
    frozen canonical loader). Returns (encoder, tokenizer, hidden_dim).

    This is the plan A.4 spec-driven entry point; existing callers keep using
    ``load_sdxl_te`` (byte-identical). No behavior is added beyond the te_type
    resolution — the loading path is unchanged.
    """
    if isinstance(spec_or_type, str):
        te_type = spec_or_type
    else:
        te_type = getattr(spec_or_type, "te_type", None)
        if te_type is None:
            raise ValueError(
                "load_te requires a te_type string or a spec with a 'te_type' attribute"
            )
    return load_sdxl_te(te_type, repo=repo, dtype=dtype, device=device, max_len=max_len)
