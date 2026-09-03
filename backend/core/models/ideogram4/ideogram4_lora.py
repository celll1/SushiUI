"""LoRA support for the Ideogram 4 DiT (training target enumeration + inference apply).

Wraps target Linear / Fp8Linear layers with LoRALinearLayer (forward-time addition,
fully reversible — never merges into the fp8 base). Mirrors the Lens LoRA design.

Accepted key formats:
  1. sd-scripts native ("lora_unet_" prefix, dots flattened to underscores):
       lora_unet_layers_0_attention_to_q.lora_down.weight / .lora_up.weight / .alpha
  2. Interchange (dot-path under "diffusion_model." prefix):
       diffusion_model.layers.0.attention.to_q.lora_A.weight / .lora_B.weight / .alpha

Ideogram 4 target modules (per block N, `transformer.layers[N]`):
  layers.{N}.attention.to_q / to_k / to_v        attention Q/K/V
  layers.{N}.attention.to_out.0                   attention output projection
  layers.{N}.feed_forward.w1 / w2 / w3            SwiGLU MLP
  layers.{N}.adaln_modulation                     AdaLN modulation Linear (opt, scope "mod")

Conditional and unconditional transformers share this structure; the same helpers
apply to both (the inference loader uses distinct sub-prefixes per branch).
"""

from __future__ import annotations

from typing import Any, Dict, Generator, Optional, Set, Tuple

import torch
from torch import nn
from safetensors import safe_open

from .vendor.fp8_linear import Fp8Linear
from .vendor.int8_linear import Int8Linear


# ---------------------------------------------------------------------------
# sd-scripts key format reverse token table
# ---------------------------------------------------------------------------
# Each entry: (dotted_form_after_naive_replace, original_identifier).
# Compound identifiers contain underscores that the naive "_"<->"." flatten would
# mangle; list longest-first so shorter entries don't bind inside longer ones.
_SDSCRIPTS_REVERSE_TOKENS = (
    ("adaln.modulation", "adaln_modulation"),
    ("feed.forward",     "feed_forward"),
    ("to.out",           "to_out"),
    ("to.q",             "to_q"),
    ("to.k",             "to_k"),
    ("to.v",             "to_v"),
)

INTERCHANGE_DIT_PREFIX = "diffusion_model."


def _restore_sdscripts_dots(flat: str) -> str:
    dotted = flat.replace("_", ".")
    for compound_dot, original in _SDSCRIPTS_REVERSE_TOKENS:
        dotted = dotted.replace(compound_dot, original)
    return dotted


def _flatten_to_sdscripts(module_path: str) -> str:
    intermediate = module_path
    for compound_dot, original in reversed(_SDSCRIPTS_REVERSE_TOKENS):
        intermediate = intermediate.replace(original, compound_dot)
    return intermediate.replace(".", "_")


def _parse_key(key: str) -> Optional[Tuple[str, str]]:
    """Return (module_path, suffix in {down, up, alpha}) for a LoRA key, else None."""
    if key.startswith(INTERCHANGE_DIT_PREFIX):
        rest = key[len(INTERCHANGE_DIT_PREFIX):]
        if rest.endswith(".lora_A.weight"):
            return rest[: -len(".lora_A.weight")], "down"
        if rest.endswith(".lora_B.weight"):
            return rest[: -len(".lora_B.weight")], "up"
        if rest.endswith(".alpha"):
            return rest[: -len(".alpha")], "alpha"
        return None

    for prefix in ("lora_unet_", "lora_uncond_"):
        if key.startswith(prefix):
            rest = key[len(prefix):]
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


def normalise_lora_state_dict(
    raw: Dict[str, torch.Tensor],
    branch: str = "cond",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Group raw LoRA tensors for one branch → {module_path: {down, up, alpha?}}.

    branch="cond" reads "lora_unet_" / interchange keys; branch="uncond" reads
    "lora_uncond_" keys. Entries missing a down/up pair are dropped.
    """
    want_prefix = "lora_uncond_" if branch == "uncond" else "lora_unet_"
    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in raw.items():
        if branch == "uncond":
            if not key.startswith("lora_uncond_"):
                continue
        else:
            # cond: accept sd-scripts lora_unet_ and interchange, but NOT lora_uncond_
            if key.startswith("lora_uncond_"):
                continue
        parsed = _parse_key(key)
        if parsed is None:
            continue
        module_path, suffix = parsed
        grouped.setdefault(module_path, {})[suffix] = tensor
    return {m: v for m, v in grouped.items() if "down" in v and "up" in v}


def load_lora_safetensors(path: str) -> Tuple[Dict[str, torch.Tensor], str, Dict[str, str]]:
    """Load a LoRA safetensors file → (raw_state_dict, format_label, metadata)."""
    raw: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = dict(f.metadata() or {})
        for k in f.keys():
            raw[k] = f.get_tensor(k)
    n_sd = sum(1 for k in raw if k.startswith("lora_unet_") or k.startswith("lora_uncond_"))
    n_ix = sum(1 for k in raw if k.startswith(INTERCHANGE_DIT_PREFIX))
    fmt = "sd-scripts" if n_sd >= n_ix else "interchange"
    if n_sd == 0 and n_ix == 0:
        fmt = "unknown"
    return raw, fmt, metadata


def alpha_from_metadata(metadata: Optional[Dict[str, str]]) -> Optional[float]:
    """File-level LoRA alpha, or None.

    Second rung of the precedence per-key ``.alpha`` tensor -> file metadata ->
    rank. Without it a trainer that records alpha only in metadata (kohya's
    ``ss_network_alpha``) silently applies at scale 1.0 instead of alpha/rank.
    """
    if not metadata:
        return None
    for key in ("lora_alpha", "ss_network_alpha"):
        value = metadata.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Scope and target enumeration
# ---------------------------------------------------------------------------

DEFAULT_SCOPE: Dict[str, bool] = {
    "attn": True,
    "mlp":  True,
    "mod":  False,
}

_FULL_SCOPE: Dict[str, bool] = {k: True for k in DEFAULT_SCOPE}


def parse_scope_csv(scope_csv: Optional[str]) -> Dict[str, bool]:
    """Parse a comma-separated scope string (e.g. "attn,mlp") into a scope dict."""
    scope = {k: False for k in DEFAULT_SCOPE}
    if not scope_csv:
        return dict(DEFAULT_SCOPE)
    for tok in scope_csv.split(","):
        tok = tok.strip()
        if tok in scope:
            scope[tok] = True
    if not any(scope.values()):
        return dict(DEFAULT_SCOPE)
    return scope


def _set_module(parent: Any, attr: Any, module: nn.Module) -> None:
    if isinstance(attr, int):
        parent[attr] = module
    else:
        setattr(parent, attr, module)


def _is_lora_target(m: Any) -> bool:
    """True for a module a LoRA can wrap or re-wrap on the Ideogram 4 DiT.

    A plain ``nn.Linear``, EITHER weight-only quantized Linear, or an
    already-wrapped ``LoRALinearLayer`` (yielded so re-application and stacking
    find the slot). ``Fp8Linear`` and ``Int8Linear`` are ``nn.Module``s, NOT
    ``nn.Linear`` subclasses, so both have to be named or their layers are
    skipped SILENTLY -- no target, no warning, and a LoRA that appears to do
    nothing on a quantized checkpoint. Ideogram 4's published checkpoints are
    quantized, so that would be the normal case here, not an edge case.

    MODULE-LEVEL rather than the closure it used to be, so that
    ``quantized_capability_parity_test`` can find it by convention and check it
    against ``Int8Linear``/``Fp8Linear`` the same way it checks Anima's
    ``_is_lora_target`` and Krea 2's ``_is_target``. The behaviour is unchanged.
    """
    from core.adapters import LoRALinearLayer

    return isinstance(m, (nn.Linear, Fp8Linear, Int8Linear, LoRALinearLayer))


def iter_ideogram4_lora_targets(
    transformer: nn.Module,
    scope: Optional[Dict[str, bool]] = None,
) -> Generator[Tuple[str, Any, Any, nn.Module], None, None]:
    """Yield (module_path, parent, attr_or_idx, current_module) per LoRA target.

    attr_or_idx is a str for normal attributes or an int for ModuleList children
    (e.g. to_out[0]). Use _set_module() for assignment. Targets include both plain
    nn.Linear, weight-only-quantized Fp8Linear / Int8Linear (the e4m3 and int8
    bases), plus already-wrapped LoRALinearLayer (for stacking).
    """
    scope = scope if scope is not None else DEFAULT_SCOPE
    want_attn = bool(scope.get("attn", False))
    want_mlp = bool(scope.get("mlp", False))
    want_mod = bool(scope.get("mod", False))

    is_target = _is_lora_target

    blocks = getattr(transformer, "layers", None)
    if blocks is None:
        return

    for block_idx, block in enumerate(blocks):
        prefix = f"layers.{block_idx}"

        if want_attn:
            attn = getattr(block, "attention", None)
            if attn is not None:
                for attr_name in ("to_q", "to_k", "to_v"):
                    m = getattr(attn, attr_name, None)
                    if is_target(m):
                        yield f"{prefix}.attention.{attr_name}", attn, attr_name, m
                to_out = getattr(attn, "to_out", None)
                if isinstance(to_out, nn.ModuleList) and len(to_out) > 0 and is_target(to_out[0]):
                    yield f"{prefix}.attention.to_out.0", to_out, 0, to_out[0]

        if want_mlp:
            mlp = getattr(block, "feed_forward", None)
            if mlp is not None:
                for wname in ("w1", "w2", "w3"):
                    m = getattr(mlp, wname, None)
                    if is_target(m):
                        yield f"{prefix}.feed_forward.{wname}", mlp, wname, m

        if want_mod:
            m = getattr(block, "adaln_modulation", None)
            if is_target(m):
                yield f"{prefix}.adaln_modulation", block, "adaln_modulation", m


# ---------------------------------------------------------------------------
# Apply / restore (inference)
# ---------------------------------------------------------------------------

def apply_lora_group(
    transformer: nn.Module,
    grouped: Dict[str, Dict[str, torch.Tensor]],
    strength: float,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: Set[str],
    scope: Optional[Dict[str, bool]] = None,
    default_alpha: Optional[float] = None,
) -> Tuple[int, int]:
    """Wrap matching modules with LoRALinearLayer -> (applied, already_wrapped).

    ``default_alpha`` is the file-metadata alpha used for a module with no
    per-key ``.alpha`` tensor (see alpha_from_metadata).

    A target already wrapped by an earlier LoRA is counted, not re-wrapped:
    LoRALinearLayer cannot wrap a wrapper (no in_features/out_features). The
    caller turns the count into a refusal or a warning.
    """
    from core.adapters import LoRALinearLayer

    effective_scope = scope if scope is not None else _FULL_SCOPE
    applied = 0
    occupied = 0

    for module_path, parent, attr, linear in iter_ideogram4_lora_targets(transformer, effective_scope):
        weights = grouped.get(module_path)
        if weights is None:
            continue

        if isinstance(linear, LoRALinearLayer):
            occupied += 1
            continue

        down = weights["down"]
        up = weights["up"]
        alpha_tensor = weights.get("alpha")

        true_original = linear
        lora_original_modules.setdefault(module_path, true_original)

        rank = int(down.shape[0])
        if alpha_tensor is not None:
            alpha_value = float(alpha_tensor.item())
        elif default_alpha is not None:
            alpha_value = float(default_alpha)
        else:
            alpha_value = float(rank)

        wrapper = LoRALinearLayer(true_original, rank=rank, alpha=alpha_value, lora_name=module_path)
        device = true_original.weight.device

        # Match the base compute dtype (fp8 base -> bf16 compute).
        if getattr(true_original, "bias", None) is not None and true_original.bias.dtype.is_floating_point:
            compute_dtype = true_original.bias.dtype
        elif (true_original.weight.dtype.is_floating_point and
              "float8" not in str(true_original.weight.dtype)):
            compute_dtype = true_original.weight.dtype
        else:
            compute_dtype = torch.bfloat16

        with torch.no_grad():
            wrapper.lora_down.weight.data = down.to(device=device, dtype=compute_dtype)
            wrapper.lora_up.weight.data = up.to(device=device, dtype=compute_dtype)
        wrapper.lora_down = wrapper.lora_down.to(dtype=compute_dtype)
        wrapper.lora_up = wrapper.lora_up.to(dtype=compute_dtype)
        wrapper.scale = (alpha_value / rank) * strength

        _set_module(parent, attr, wrapper)
        wrapped_keys.add(module_path)
        applied += 1

    return applied, occupied


def restore_originals(
    transformer: nn.Module,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: Set[str],
) -> int:
    """Revert every wrapped module to its pre-LoRA original.

    Restores only paths this session actually wrapped, and drops their
    bookkeeping afterwards: a surviving ``lora_original_modules`` entry would be
    written into the NEXT model loaded at the same path, i.e. one model's
    Linear installed into another.
    """
    restored = 0
    restored_keys: Set[str] = set()
    for module_path, parent, attr, _linear in iter_ideogram4_lora_targets(transformer, _FULL_SCOPE):
        if module_path in wrapped_keys and module_path in lora_original_modules:
            _set_module(parent, attr, lora_original_modules[module_path])
            restored_keys.add(module_path)
            restored += 1
    for key in restored_keys:
        lora_original_modules.pop(key, None)
    wrapped_keys -= restored_keys
    return restored
