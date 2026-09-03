"""LoRA support for the Microsoft Lens DiT.

Loads LoRA safetensors and wraps target Linear layers with LoRALinearLayer
(forward-time addition, not weight merge — fully reversible).

Two key conventions are accepted:

  1. sd-scripts native ("lora_unet_" prefix, dots/underscores flattened):
       lora_unet_transformer_blocks_0_attn_img_qkv.lora_down.weight
       lora_unet_transformer_blocks_0_attn_img_qkv.lora_up.weight
       lora_unet_transformer_blocks_0_attn_img_qkv.alpha

  2. Interchange format (dot-path under "diffusion_model." prefix):
       diffusion_model.transformer_blocks.0.attn.img_qkv.lora_A.weight
       diffusion_model.transformer_blocks.0.attn.img_qkv.lora_B.weight

Lens target modules (per block N):
  transformer_blocks.{N}.attn.img_qkv         combined Q+K+V for image stream
  transformer_blocks.{N}.attn.txt_qkv         combined Q+K+V for text stream
  transformer_blocks.{N}.attn.to_out[0]       image output projection
  transformer_blocks.{N}.attn.to_add_out      text output projection
  transformer_blocks.{N}.img_mlp.{w1,w2,w3}  image GateMLP
  transformer_blocks.{N}.txt_mlp.{w1,w2,w3}  text GateMLP
  transformer_blocks.{N}.img_mod[1]           image AdaLN modulation Linear (opt)
  transformer_blocks.{N}.txt_mod[1]           text  AdaLN modulation Linear (opt)
"""

from __future__ import annotations

from typing import Any, Dict, Generator, Mapping, Optional, Tuple

import torch
from torch import nn

from core.adapters.groups import (TensorGroup, declared_groups,
                                  group_adapter_tensors, split_adapter_suffix)


# ---------------------------------------------------------------------------
# sd-scripts key format reverse token table
# ---------------------------------------------------------------------------

# Each entry: (dotted_form_after_naive_replace, original_identifier).
# Listed longest-first to prevent shorter entries matching inside longer ones.
_SDSCRIPTS_REVERSE_TOKENS = (
    ("transformer.blocks", "transformer_blocks"),
    ("to.add.out",         "to_add_out"),
    ("img.qkv",            "img_qkv"),
    ("txt.qkv",            "txt_qkv"),
    ("to.out",             "to_out"),
    ("img.mlp",            "img_mlp"),
    ("txt.mlp",            "txt_mlp"),
    ("img.mod",            "img_mod"),
    ("txt.mod",            "txt_mod"),
)

INTERCHANGE_DIT_PREFIX = "diffusion_model."


def _restore_sdscripts_dots(flat: str) -> str:
    """Convert underscore-flattened sd-scripts module key back to dotted path."""
    dotted = flat.replace("_", ".")
    for compound_dot, original in _SDSCRIPTS_REVERSE_TOKENS:
        dotted = dotted.replace(compound_dot, original)
    return dotted


def _lens_stem(raw_stem: str) -> Optional[str]:
    """Suffix-stripped key -> module path, or None for a foreign key."""
    if raw_stem.startswith(INTERCHANGE_DIT_PREFIX):
        return raw_stem[len(INTERCHANGE_DIT_PREFIX):]
    if raw_stem.startswith("lora_unet_"):
        flat = raw_stem[len("lora_unet_"):]
        # The native codec flattens every dot away, so a dot left in the stem
        # means the key carried an unrecognised weight name, not a module path.
        return None if "." in flat else _restore_sdscripts_dots(flat)
    return None


def _parse_key(key: str) -> Optional[Tuple[str, str]]:
    """``(module_path, canonical tensor name)`` for a recognised key, else None."""
    split = split_adapter_suffix(key)
    if split is None:
        return None
    module_path = _lens_stem(split[0])
    return None if module_path is None else (module_path, split[1])


def declared_branch_count(raw: Mapping[str, torch.Tensor]) -> int:
    """Branches this file declares to Lens; see ``declared_groups``."""
    return len(declared_groups(raw, _lens_stem))


def _flatten_to_sdscripts(module_path: str) -> str:
    """Convert canonical dotted Lens module path to sd-scripts underscore key.

    Inverse of _restore_sdscripts_dots: re-inserts dots inside known compound
    identifiers, then replaces remaining dots with underscores.
    """
    intermediate = module_path
    # Apply in REVERSE order so longer/more-specific entries bind before sub-strings.
    for compound_dot, original in reversed(_SDSCRIPTS_REVERSE_TOKENS):
        intermediate = intermediate.replace(original, compound_dot)
    return intermediate.replace(".", "_")


def normalise_lora_state_dict(
    raw: Dict[str, torch.Tensor],
) -> Dict[str, TensorGroup]:
    """COMPLETE factor groups by module path, whatever the algebra.

    ``group_adapter_tensors`` already drops the incomplete ones; a down/up
    filter on top would silently drop every LoHa and LoKr group.
    """
    return group_adapter_tensors(raw, _lens_stem).groups


def detect_lora_format(raw: Mapping[str, torch.Tensor]) -> str:
    """The key-format label for one already-read file.

    ``AdapterSession`` reads the safetensors and its metadata; this and the
    mixed-format note below are the only parts of the load that were Lens's.
    """
    n_sd = sum(1 for k in raw if k.startswith("lora_unet_"))
    n_ix = sum(1 for k in raw if k.startswith(INTERCHANGE_DIT_PREFIX))
    if n_sd == 0 and n_ix == 0:
        return "unknown"
    return "sd-scripts" if n_sd >= n_ix else "interchange"


def mixed_format_note(raw: Mapping[str, torch.Tensor]) -> Optional[str]:
    """A console line when a file carries BOTH codecs, else None.

    The minority keys are dropped by the per-format parser, which otherwise
    looks like a LoRA that half applied.
    """
    n_sd = sum(1 for k in raw if k.startswith("lora_unet_"))
    n_ix = sum(1 for k in raw if k.startswith(INTERCHANGE_DIT_PREFIX))
    if n_sd > 0 and n_ix > 0:
        return (f"[LensLoRA] WARNING: mixed-format file (sd-scripts={n_sd}, "
                f"interchange={n_ix}), loading dominant format "
                f"{detect_lora_format(raw)!r}")
    return None


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
    "img_attn": True,
    "txt_attn": True,
    "img_mlp":  True,
    "txt_mlp":  True,
    "mod":      False,
}

_FULL_SCOPE: Dict[str, bool] = {k: True for k in DEFAULT_SCOPE}


def parse_scope_csv(scope_csv: Optional[str]) -> Dict[str, bool]:
    """Parse a comma-separated scope string (e.g. "img_attn,txt_attn") into a scope dict.

    Builds from an ALL-FALSE dict, so a scope can narrow as well as widen. The
    caller used to start from ``DEFAULT_SCOPE`` (four of five groups already
    True) and only ever set True, which made every narrowing selection a no-op:
    a user who unticked img_mlp/txt_mlp still trained all four groups.

    Empty input, or input naming nothing recognised, returns ``DEFAULT_SCOPE``
    rather than an empty scope -- a LoRA with no target modules trains nothing.
    Same contract as ``core/models/ideogram4/ideogram4_lora.parse_scope_csv``.
    """
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


def iter_lens_lora_targets(
    transformer: nn.Module,
    scope: Optional[Dict[str, bool]] = None,
) -> Generator[Tuple[str, Any, Any, nn.Module], None, None]:
    """Yield (module_path, parent, attr_or_idx, current_module) per LoRA target.

    ONE enumerator for both load and unload, so the two cannot disagree about a
    slot once a target can hold more than one branch.

    attr_or_idx is a str for normal attributes or an int for ModuleList/Sequential
    children (`attn.to_out[0]`, `img_mod[1]`, `txt_mod[1]`) -- address it with
    ``core.adapters.get_module_slot`` / ``set_module_slot``, which take either;
    ``setattr(parent, 1, module)`` raises TypeError.
    """
    from core.adapters import CompositeAdapterLayer, LoRALinearLayer

    scope = scope if scope is not None else DEFAULT_SCOPE
    want_img_attn = bool(scope.get("img_attn", False))
    want_txt_attn = bool(scope.get("txt_attn", False))
    want_img_mlp  = bool(scope.get("img_mlp",  False))
    want_txt_mlp  = bool(scope.get("txt_mlp",  False))
    want_mod      = bool(scope.get("mod",       False))

    # CompositeAdapterLayer is a target too: drop it and a second selected LoRA
    # skips every occupied slot and reports zero matches as if its keys were wrong.
    is_target = lambda m: isinstance(
        m, (nn.Linear, LoRALinearLayer, CompositeAdapterLayer))

    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        return

    for block_idx, block in enumerate(blocks):
        prefix = f"transformer_blocks.{block_idx}"
        attn = getattr(block, "attn", None)

        if attn is not None:
            if want_img_attn:
                m = getattr(attn, "img_qkv", None)
                if is_target(m):
                    yield f"{prefix}.attn.img_qkv", attn, "img_qkv", m

                to_out = getattr(attn, "to_out", None)
                if isinstance(to_out, nn.ModuleList):
                    m = to_out[0]
                    if is_target(m):
                        yield f"{prefix}.attn.to_out.0", to_out, 0, m

            if want_txt_attn:
                m = getattr(attn, "txt_qkv", None)
                if is_target(m):
                    yield f"{prefix}.attn.txt_qkv", attn, "txt_qkv", m

                m = getattr(attn, "to_add_out", None)
                if is_target(m):
                    yield f"{prefix}.attn.to_add_out", attn, "to_add_out", m

        if want_img_mlp:
            img_mlp = getattr(block, "img_mlp", None)
            if img_mlp is not None:
                for wname in ("w1", "w2", "w3"):
                    m = getattr(img_mlp, wname, None)
                    if m is not None and is_target(m):
                        yield f"{prefix}.img_mlp.{wname}", img_mlp, wname, m

        if want_txt_mlp:
            txt_mlp = getattr(block, "txt_mlp", None)
            if txt_mlp is not None:
                for wname in ("w1", "w2", "w3"):
                    m = getattr(txt_mlp, wname, None)
                    if m is not None and is_target(m):
                        yield f"{prefix}.txt_mlp.{wname}", txt_mlp, wname, m

        if want_mod:
            for mod_name in ("img_mod", "txt_mod"):
                mod_seq = getattr(block, mod_name, None)
                if isinstance(mod_seq, nn.Sequential):
                    m = mod_seq[1]
                    if is_target(m):
                        yield f"{prefix}.{mod_name}.1", mod_seq, 1, m


# ---------------------------------------------------------------------------
# Apply (inference). The LIFETIME -- resolve, parse, refuse, install, restore --
# belongs to ``core.adapters.AdapterSession``; what is Lens's is the target
# scope, the two key codecs and one branch.
# ---------------------------------------------------------------------------

def iter_lens_lora_slots(transformer: nn.Module):
    """``(parent, slot, module_path)`` over the FULL scope, for ``AdapterSession``.

    Full scope on both the load and the unload path, not DEFAULT_SCOPE.
    Application is lookup-driven -- a target the file names no key for gets no
    branch -- so the checkpoint's own keys pick the targets, and the narrower
    default would silently drop the ``mod`` group that training can opt into.
    """
    for module_path, parent, attr, _current in iter_lens_lora_targets(
            transformer, _FULL_SCOPE):
        yield parent, attr, module_path


def branch_dtype(base: nn.Module) -> torch.dtype:
    """Lens's own branch dtype: the base model's compute dtype, which an
    FP8-quantised base states through its bias rather than its weight.

    Deliberately not ``core.adapters.lora_branch_dtype``, which has no bias
    tier and would send every biased fp8 target to the bfloat16 default.
    """
    bias = getattr(base, "bias", None)
    if bias is not None and bias.dtype.is_floating_point:
        return bias.dtype
    weight_dtype = base.weight.dtype
    if weight_dtype.is_floating_point and "float8" not in str(weight_dtype):
        return weight_dtype
    return torch.bfloat16


def build_lora_branch(base: nn.Module, group: TensorGroup,
                      module_path: str,
                      default_alpha: Optional[float] = None) -> nn.Module:
    """One branch over ``base``, or ``SHAPE_MISMATCH``.

    The algebra is the group's: ``build_adapter_branch`` dispatches on the tensor
    names. Alpha precedence: the per-key ``.alpha`` tensor, then ``default_alpha``
    (the file metadata's, see ``alpha_from_metadata``), then the rank. The request
    strength is NOT folded in here: ``add_branch(strength=)`` refolds it into
    this branch's own scale, and multiplying it onto the delta instead is
    different arithmetic that loses bit-identity with the single-LoRA numerics.
    """
    from core.adapters import build_adapter_branch

    return build_adapter_branch(base, group, metadata_alpha=default_alpha,
                                lora_dtype=branch_dtype(base),
                                lora_name=module_path)
