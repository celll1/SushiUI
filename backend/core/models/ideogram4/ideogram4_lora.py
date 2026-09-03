"""LoRA support for the Ideogram 4 DiT (training target enumeration + inference apply).

Covers each target Linear / Fp8Linear with a CompositeAdapterLayer holding one
named branch per selected LoRA (forward-time addition, fully reversible — never
merges into the fp8 base), so two LoRAs over one module SUM. Mirrors Lens.

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

from typing import Any, Callable, Dict, Generator, Mapping, Optional, Tuple

import torch
from torch import nn

from core.adapters.groups import (TensorGroup, declared_groups,
                                  group_adapter_tensors, split_adapter_suffix)

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


def _ideogram4_stem(raw_stem: str) -> Optional[str]:
    """Suffix-stripped key -> module path, or None for a foreign key."""
    if raw_stem.startswith(INTERCHANGE_DIT_PREFIX):
        return raw_stem[len(INTERCHANGE_DIT_PREFIX):]
    for prefix in ("lora_unet_", "lora_uncond_"):
        if raw_stem.startswith(prefix):
            flat = raw_stem[len(prefix):]
            # The native codec flattens every dot away, so a dot left in the
            # stem means an unrecognised weight name, not a module path.
            return None if "." in flat else _restore_sdscripts_dots(flat)
    return None


def _branch_stem(branch: str) -> Callable[[str], Optional[str]]:
    """``_ideogram4_stem`` restricted to one asymmetric-CFG branch's keys."""
    def stem_of(raw_stem: str) -> Optional[str]:
        if branch == "uncond":
            if not raw_stem.startswith("lora_uncond_"):
                return None
        elif raw_stem.startswith("lora_uncond_"):
            # cond: sd-scripts lora_unet_ and interchange, but NOT lora_uncond_.
            return None
        return _ideogram4_stem(raw_stem)
    return stem_of


def _parse_key(key: str) -> Optional[Tuple[str, str]]:
    """``(module_path, canonical tensor name)`` for a LoRA key, else None."""
    split = split_adapter_suffix(key)
    if split is None:
        return None
    module_path = _ideogram4_stem(split[0])
    return None if module_path is None else (module_path, split[1])


def normalise_lora_state_dict(
    raw: Dict[str, torch.Tensor],
    branch: str = "cond",
) -> Dict[str, TensorGroup]:
    """Group raw LoRA tensors for one branch → {module_path: TensorGroup}.

    branch="cond" reads "lora_unet_" / interchange keys; branch="uncond" reads
    "lora_uncond_" keys. Only down/up groups are returned -- any other algebra
    is counted (``count_declared_pairs``) and refused unapplied rather than
    handed to a builder that cannot express it. ``TensorGroup`` answers to
    ``["down"]``/``["up"]``/``.get("alpha")``, which is what the builder reads.
    """
    grouped = group_adapter_tensors(raw, _branch_stem(branch)).groups
    return {m: g for m, g in grouped.items() if "down" in g and "up" in g}


def detect_lora_format(raw: Mapping[str, Any]) -> str:
    """"sd-scripts" / "interchange" / "unknown", from the key names alone."""
    n_sd = sum(1 for k in raw if k.startswith("lora_unet_") or k.startswith("lora_uncond_"))
    n_ix = sum(1 for k in raw if k.startswith(INTERCHANGE_DIT_PREFIX))
    if n_sd == 0 and n_ix == 0:
        return "unknown"
    return "sd-scripts" if n_sd >= n_ix else "interchange"


def count_declared_pairs(raw: Mapping[str, torch.Tensor]) -> int:
    """Branches this file declares across BOTH branches; see ``declared_groups``."""
    return sum(len(declared_groups(raw, _branch_stem(b)))
               for b in ("cond", "uncond"))


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


def _is_lora_target(m: Any) -> bool:
    """True for a module a LoRA can wrap or re-wrap on the Ideogram 4 DiT.

    A plain ``nn.Linear``, EITHER weight-only quantized Linear, or an adapter
    wrapper already sitting in the slot (yielded so a second selected LoRA finds
    the composite instead of skipping every occupied target and reporting zero
    matches as if its keys were wrong). ``Fp8Linear`` and ``Int8Linear`` are
    ``nn.Module``s, NOT ``nn.Linear`` subclasses, so both have to be named or
    their layers are skipped SILENTLY -- no target, no warning, and a LoRA that
    appears to do nothing on a quantized checkpoint. Ideogram 4's published
    checkpoints are quantized, so that would be the normal case here, not an
    edge case.

    MODULE-LEVEL rather than the closure it used to be, so that
    ``quantized_capability_parity_test`` can find it by convention and check it
    against ``Int8Linear``/``Fp8Linear`` the same way it checks Anima's
    ``_is_lora_target`` and Krea 2's ``_is_target``. The behaviour is unchanged.
    """
    from core.adapters import CompositeAdapterLayer, LoRALinearLayer

    return isinstance(m, (nn.Linear, Fp8Linear, Int8Linear, LoRALinearLayer,
                          CompositeAdapterLayer))


def iter_ideogram4_lora_targets(
    transformer: nn.Module,
    scope: Optional[Dict[str, bool]] = None,
) -> Generator[Tuple[str, Any, Any, nn.Module], None, None]:
    """Yield (module_path, parent, attr_or_idx, current_module) per LoRA target.

    ONE enumerator for both load and unload, so the two cannot disagree about a
    slot once a target can hold more than one branch. Callers materialise it
    before mutating.

    attr_or_idx is a str for normal attributes or an int for ModuleList children
    (e.g. to_out[0]) -- address it with ``core.adapters.get_module_slot`` /
    ``set_module_slot``, which take either; ``setattr(parent, 0, module)`` raises
    TypeError. Targets include plain nn.Linear, weight-only-quantized Fp8Linear /
    Int8Linear (the e4m3 and int8 bases), plus an adapter wrapper already in the
    slot.
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
# Inference: the two callables AdapterSession needs
# ---------------------------------------------------------------------------

def iter_ideogram4_lora_slots(transformer: nn.Module):
    """``(parent, slot, module_path)`` over the FULL scope, for ``AdapterSession``.

    Same traversal as ``iter_ideogram4_lora_targets``, re-shaped to the session's
    tuple order; the generation loader has never scoped its apply, so load and
    unload both see every target.
    """
    for module_path, parent, attr, _current in iter_ideogram4_lora_targets(
            transformer, _FULL_SCOPE):
        yield parent, attr, module_path


def build_lora_branch(
    base: nn.Module,
    weights: Dict[str, torch.Tensor],
    module_path: str,
    default_alpha: Optional[float] = None,
) -> nn.Module:
    """One branch for one target, built and not installed.

    Alpha precedence: per-key ``.alpha`` tensor, then file metadata
    (``default_alpha``), then rank. The strength is NOT folded here --
    ``CompositeAdapterLayer.add_branch(strength=)`` does it, and multiplying it
    onto the delta instead loses bit-identity with the single-LoRA numerics.
    """
    from core.adapters import LoRALinearLayer

    down, up = weights["down"], weights["up"]
    alpha_tensor = weights.get("alpha")
    rank = int(down.shape[0])
    if alpha_tensor is not None:
        alpha_value = float(alpha_tensor.item())
    elif default_alpha is not None:
        alpha_value = float(default_alpha)
    else:
        alpha_value = float(rank)

    branch = LoRALinearLayer(base, rank=rank, alpha=alpha_value, lora_name=module_path)
    device = base.weight.device

    # Match the base compute dtype (fp8 base -> bf16 compute).
    if getattr(base, "bias", None) is not None and base.bias.dtype.is_floating_point:
        compute_dtype = base.bias.dtype
    elif base.weight.dtype.is_floating_point and "float8" not in str(base.weight.dtype):
        compute_dtype = base.weight.dtype
    else:
        compute_dtype = torch.bfloat16

    with torch.no_grad():
        branch.lora_down.weight.data = down.to(device=device, dtype=compute_dtype)
        branch.lora_up.weight.data = up.to(device=device, dtype=compute_dtype)
    branch.lora_down = branch.lora_down.to(dtype=compute_dtype)
    branch.lora_up = branch.lora_up.to(dtype=compute_dtype)
    return branch
