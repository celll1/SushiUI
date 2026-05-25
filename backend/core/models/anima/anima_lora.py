"""LoRA support for the Anima DiT.

Loads LoRA safetensors and wraps the target Linear layers with LoRALinearLayer
(forward-time addition, not weight merge — fully reversible by restoring the
original module).

Two key conventions are accepted:

  1. sd-scripts native ("lora_unet_" prefix, dots flattened to underscores in
     the module path, weight suffixes lora_down.weight / lora_up.weight /
     alpha):
       lora_unet_blocks_0_self_attn_q_proj.lora_down.weight
       lora_unet_blocks_0_self_attn_q_proj.lora_up.weight
       lora_unet_blocks_0_self_attn_q_proj.alpha

  2. Interchange format (used by various third-party tooling): dot-path module
     names under a "diffusion_model." prefix, with peft-style weight suffixes
     (lora_A = down, lora_B = up):
       diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight
       diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight
       diffusion_model.blocks.0.self_attn.q_proj.alpha

Anima target modules per Block:
    blocks.<N>.self_attn.{q_proj, k_proj, v_proj, output_proj}
    blocks.<N>.cross_attn.{q_proj, k_proj, v_proj, output_proj}
"""

from typing import Dict, Tuple, List, Optional, Any

import torch
from torch import nn
from safetensors import safe_open


# Module-path tokens that the underscore-flattened native format may map back to.
# Order matters: longer compound names first so they bind before their substrings.
_SDSCRIPTS_REVERSE_TOKENS = (
    ("llm.adapter", "llm_adapter"),
    ("t.embedding.norm", "t_embedding_norm"),
    ("x.embedder", "x_embedder"),
    ("adaln.modulation.self.attn", "adaln_modulation_self_attn"),
    ("adaln.modulation.cross.attn", "adaln_modulation_cross_attn"),
    ("adaln.modulation.mlp", "adaln_modulation_mlp"),
    ("adaln.modulation", "adaln_modulation"),
    ("norm.cross.attn", "norm_cross_attn"),
    ("norm.self.attn", "norm_self_attn"),
    ("norm.mlp", "norm_mlp"),
    ("cross.attn", "cross_attn"),
    ("self.attn", "self_attn"),
    ("final.layer", "final_layer"),
    ("k.proj", "k_proj"),
    ("k.norm", "k_norm"),
    ("q.proj", "q_proj"),
    ("q.norm", "q_norm"),
    ("v.proj", "v_proj"),
    ("v.norm", "v_norm"),
    ("o.proj", "o_proj"),
    ("output.proj", "output_proj"),
    ("out.proj", "out_proj"),
)


INTERCHANGE_DIT_PREFIX = "diffusion_model."


def _restore_sdscripts_dots(flat: str) -> str:
    """Convert underscore-flattened module path back to the canonical dotted path."""
    dotted = flat.replace("_", ".")
    for compound_dot, original in _SDSCRIPTS_REVERSE_TOKENS:
        dotted = dotted.replace(compound_dot, original)
    return dotted


def _parse_key(key: str) -> Optional[Tuple[str, str]]:
    """Return (module_path, suffix) for a recognised LoRA key, or None.

    suffix is one of {"down", "up", "alpha"}.
    """
    # Interchange format first (unambiguous prefix)
    if key.startswith(INTERCHANGE_DIT_PREFIX):
        rest = key[len(INTERCHANGE_DIT_PREFIX):]
        if rest.endswith(".lora_A.weight"):
            return rest[:-len(".lora_A.weight")], "down"
        if rest.endswith(".lora_B.weight"):
            return rest[:-len(".lora_B.weight")], "up"
        if rest.endswith(".alpha"):
            return rest[:-len(".alpha")], "alpha"
        return None

    # sd-scripts native: lora_unet_<flattened>.<suffix>
    if key.startswith("lora_unet_"):
        rest = key[len("lora_unet_"):]
        if "." not in rest:
            return None
        flat_module, weight_name = rest.split(".", 1)
        module_path = _restore_sdscripts_dots(flat_module)
        if weight_name == "lora_down.weight":
            return module_path, "down"
        if weight_name == "lora_up.weight":
            return module_path, "up"
        if weight_name == "alpha":
            return module_path, "alpha"
        return None

    return None


def normalise_lora_state_dict(raw_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Group raw LoRA tensors by module path into {module_path: {down, up, alpha?}}.

    Unrecognised keys are silently dropped (typically text-encoder LoRA keys
    when only the DiT side is targeted).
    """
    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in raw_state_dict.items():
        parsed = _parse_key(key)
        if parsed is None:
            continue
        module_path, suffix = parsed
        grouped.setdefault(module_path, {})[suffix] = tensor
    # Drop entries missing the required down/up pair
    return {m: v for m, v in grouped.items() if "down" in v and "up" in v}


def detect_lora_format(raw_state_dict: Dict[str, torch.Tensor]) -> str:
    """Return a label describing the dominant LoRA key format."""
    n_sd = sum(1 for k in raw_state_dict if k.startswith("lora_unet_"))
    n_ix = sum(1 for k in raw_state_dict if k.startswith(INTERCHANGE_DIT_PREFIX))
    if n_sd > n_ix:
        return "sd-scripts"
    if n_ix > 0:
        return "interchange"
    return "unknown"


def load_lora_safetensors(path: str) -> Tuple[Dict[str, torch.Tensor], str]:
    """Load a LoRA safetensors file and return (raw_state_dict, format_label)."""
    raw: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)
    return raw, detect_lora_format(raw)


# --------- Target enumeration ---------

_ANIMA_ATTENTION_CLASS_NAME = "Attention"  # from core.models.anima.anima_models.Attention
_LINEAR_ATTRS = ("q_proj", "k_proj", "v_proj", "output_proj")


def _iter_anima_attention_targets(transformer: nn.Module):
    """Yield (module_path, parent_attention_module, attr_name, current_module)
    for each LoRA-targetable Linear in the Anima DiT.

    `current_module` is whatever currently sits at parent.attr — typically an
    nn.Linear on the un-LoRA'd model, or a LoRALinearLayer once wrapped.
    Callers that need the true original should unwrap explicitly. Including
    wrapped modules here is essential so that restore_originals() and LoRA
    stacking can find the slots after a previous load.

    We restrict to top-level Block attention (self_attn / cross_attn). The
    LLM Adapter's internal attention is intentionally skipped here: LoRA
    training on it is uncommon and its naming differs (o_proj vs output_proj).
    """
    # Import lazily so anima_lora can be inspected without dragging in the
    # training adapters at module import time.
    from core.training.adapters.sd15_adapter import LoRALinearLayer

    for name, module in transformer.named_modules():
        if module.__class__.__name__ != _ANIMA_ATTENTION_CLASS_NAME:
            continue
        if not (".self_attn" in name or ".cross_attn" in name):
            continue
        for attr in _LINEAR_ATTRS:
            current = getattr(module, attr, None)
            if isinstance(current, (nn.Linear, LoRALinearLayer)):
                module_path = f"{name}.{attr}"
                yield module_path, module, attr, current


# --------- Apply / restore ---------

def apply_lora_group(
    transformer: nn.Module,
    grouped: Dict[str, Dict[str, torch.Tensor]],
    strength: float,
    lora_original_modules: Dict[str, nn.Linear],
    wrapped_keys: set,
) -> int:
    """Wrap matching Linear modules in the transformer with LoRALinearLayer.

    Args:
        transformer: Anima DiT instance.
        grouped: Output of normalise_lora_state_dict().
        strength: User-supplied scalar; multiplies the (alpha/rank) scaling.
        lora_original_modules: Dict to record original modules for unload.
            Keyed by module_path; the first wrap wins so multiple LoRAs over
            the same module always restore back to the true original.
        wrapped_keys: Set of module_paths currently wrapped (for unload).

    Returns:
        Number of modules wrapped (or rewrapped with a stacked LoRA).
    """
    from core.training.adapters.sd15_adapter import LoRALinearLayer

    applied = 0
    for module_path, parent, attr, linear in _iter_anima_attention_targets(transformer):
        weights = grouped.get(module_path)
        if weights is None:
            continue

        down = weights["down"]
        up = weights["up"]
        alpha_tensor = weights.get("alpha")

        # Unwrap existing wrapper to get the true original for stacking.
        if isinstance(linear, LoRALinearLayer):
            true_original = linear.original_module
        else:
            true_original = linear

        # First time we touch this slot — preserve the genuine original for unload.
        lora_original_modules.setdefault(module_path, true_original)

        rank = int(down.shape[0])
        alpha_value = float(alpha_tensor.item()) if alpha_tensor is not None else float(rank)

        wrapper = LoRALinearLayer(true_original, rank=rank, alpha=alpha_value,
                                   lora_name=module_path)
        device = true_original.weight.device
        # LoRA matrices must match the base model's *compute* dtype, not the
        # LoRA file's stored dtype. The base may carry FP8 weights with an
        # on-the-fly dequant patch (Phase B.1-d), in which case the actual
        # compute happens at the bias dtype or — when bias is absent — falls
        # back to bfloat16.
        if true_original.bias is not None and true_original.bias.dtype.is_floating_point:
            compute_dtype = true_original.bias.dtype
        elif true_original.weight.dtype.is_floating_point and not (
            'float8' in str(true_original.weight.dtype)
        ):
            compute_dtype = true_original.weight.dtype
        else:
            compute_dtype = torch.bfloat16

        with torch.no_grad():
            wrapper.lora_down.weight.data = down.to(device=device, dtype=compute_dtype)
            wrapper.lora_up.weight.data = up.to(device=device, dtype=compute_dtype)
        # Also cast the wrapper's children explicitly (LoRALinearLayer init
        # creates float32 weights and we just overwrote the .data, but the
        # Parameter object's dtype tracking still says float32 in some torch
        # builds — re-create as the right dtype).
        wrapper.lora_down = wrapper.lora_down.to(dtype=compute_dtype)
        wrapper.lora_up = wrapper.lora_up.to(dtype=compute_dtype)
        wrapper.scale = (alpha_value / rank) * strength

        setattr(parent, attr, wrapper)
        wrapped_keys.add(module_path)
        applied += 1

    return applied


def restore_originals(
    transformer: nn.Module,
    lora_original_modules: Dict[str, nn.Linear],
    wrapped_keys: set,
) -> int:
    """Revert every Block attention Linear to its original (pre-LoRA) module."""
    restored = 0
    for module_path, parent, attr, _linear in _iter_anima_attention_targets(transformer):
        if module_path in lora_original_modules:
            setattr(parent, attr, lora_original_modules[module_path])
            restored += 1
    wrapped_keys.clear()
    return restored
