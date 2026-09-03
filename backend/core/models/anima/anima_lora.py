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

Target modules are whatever the checkpoint carries (see iter_anima_lora_targets
for the per-scope enumeration): DiT block attention, block MLP, AdaLN modulation,
and the LLM Adapter.
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
    ("in.proj", "in_proj"),  # LLM Adapter input projection (when not Identity)
)


INTERCHANGE_DIT_PREFIX = "diffusion_model."


def _restore_sdscripts_dots(flat: str) -> str:
    """Convert underscore-flattened module path back to the canonical dotted path."""
    dotted = flat.replace("_", ".")
    for compound_dot, original in _SDSCRIPTS_REVERSE_TOKENS:
        dotted = dotted.replace(compound_dot, original)
    return dotted


def _flatten_to_sdscripts(module_path: str) -> str:
    """Inverse of _restore_sdscripts_dots.

    Maps canonical Anima module paths back to the underscore-flattened form
    used by sd-scripts native LoRA file keys. The trick is to insert dots
    inside the known multi-token names (q_proj -> q.proj, self_attn ->
    self.attn, llm_adapter -> llm.adapter, ...) BEFORE flattening, so that
    the final underscores-to-dots pass on the consumer side reproduces the
    same canonical path.
    """
    intermediate = module_path
    # Apply the reverse mapping in REVERSE-priority order so that longer
    # compound names (which include shorter ones as substrings) are matched
    # before their constituents are touched.
    for compound_dot, original in reversed(_SDSCRIPTS_REVERSE_TOKENS):
        intermediate = intermediate.replace(original, compound_dot)
    return intermediate.replace(".", "_")


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


def unmatched_source_keys(
    raw_state_dict: Dict[str, torch.Tensor],
    grouped: Dict[str, Dict[str, torch.Tensor]],
) -> List[str]:
    """Raw keys that carry nothing into `grouped` — unparseable, or part of a
    module group missing its down/up pair. Callers must surface these: a dropped
    key is a silently weaker LoRA, not a no-op.
    """
    dropped: List[str] = []
    for key in raw_state_dict:
        parsed = _parse_key(key)
        if parsed is None or parsed[0] not in grouped:
            dropped.append(key)
    return sorted(dropped)


def detect_lora_format(raw_state_dict: Dict[str, torch.Tensor]) -> str:
    """Return a label describing the dominant LoRA key format.

    Warns when both sd-scripts ("lora_unet_*") and interchange
    (INTERCHANGE_DIT_PREFIX*) keys are present — that almost certainly
    indicates a malformed checkpoint and the minority keys will be
    silently dropped by the per-format parser.
    """
    n_sd = sum(1 for k in raw_state_dict if k.startswith("lora_unet_"))
    n_ix = sum(1 for k in raw_state_dict if k.startswith(INTERCHANGE_DIT_PREFIX))
    if n_sd > 0 and n_ix > 0:
        # Mixed-format LoRA: pick the dominant one but warn the user so
        # they can re-export the file in a single consistent format.
        dominant = "sd-scripts" if n_sd >= n_ix else "interchange"
        minority = "interchange" if dominant == "sd-scripts" else "sd-scripts"
        minority_count = n_ix if dominant == "sd-scripts" else n_sd
        print(f"[AnimaLoRA] WARNING: mixed-format LoRA detected "
              f"(sd-scripts keys={n_sd}, interchange keys={n_ix}). "
              f"Loading as {dominant!r}; the {minority_count} {minority!r} "
              f"keys will be ignored.")
        return dominant
    if n_sd > 0:
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

_ANIMA_ATTENTION_CLASS_NAME = "Attention"            # DiT Block attention class
_LLM_ADAPTER_ATTN_CLASS_NAME = "LLMAdapterAttention"  # inside the 6-layer LLM Adapter
_BLOCK_ATTN_ATTRS = ("q_proj", "k_proj", "v_proj", "output_proj")
_LLM_ADAPTER_ATTN_ATTRS = ("q_proj", "k_proj", "v_proj", "o_proj")  # note: o_proj


# Default scope for TRAINING only. Inference has no default: it derives the scope
# from the checkpoint's own keys (see derive_scope_from_keys), so any scope the
# trainer can save wraps in full at generation time.
DEFAULT_TRAINING_SCOPE = {
    "attention": True,
    "mlp": True,
    "mod": False,
    "llm_adapter": True,
}


_LORA_TARGET_TYPES: Optional[tuple] = None


def _lora_target_types() -> tuple:
    """The ``isinstance`` tuple used by ``_is_lora_target``, resolved ONCE.

    Resolved lazily rather than at module import because the quantized Linear
    classes live under ``core.models.*``: hoisting them would give this module
    an import-time edge into the model loaders. Lazy + cached gets the same
    result as hoisting for the cost that matters: the imports run once per
    process instead of once per module on a 515-module walk, twice per
    iteration.

    If the quantized classes cannot be imported the fallback is announced, not
    silent — it is exactly the pre-fix predicate (56 targets instead of 224 on
    the shipped int8 artifact), and a LoRA that quietly wraps a third of its
    intended layers looks like a LoRA that "just has no effect". Same reporting
    contract as ``anima_loader.anima_state_dict_is_quantized``.
    """
    global _LORA_TARGET_TYPES
    if _LORA_TARGET_TYPES is not None:
        return _LORA_TARGET_TYPES
    from core.adapters import LoRALinearLayer
    try:
        from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
        from core.models.ideogram4.vendor.int8_linear import Int8Linear
        _LORA_TARGET_TYPES = (nn.Linear, Fp8Linear, Int8Linear, LoRALinearLayer)
    except Exception as e:
        print(f"[AnimaLoRA] weight-only quantized Linear classes unavailable ({e}); "
              f"only plain nn.Linear layers can be wrapped. On a quantized Anima DiT "
              f"this yields FAR fewer LoRA targets than intended and the LoRA will "
              f"appear to have little or no effect.")
        _LORA_TARGET_TYPES = (nn.Linear, LoRALinearLayer)
    return _LORA_TARGET_TYPES


def _is_lora_target(m) -> bool:
    """True for a module a LoRA can wrap: a plain ``nn.Linear``, EITHER
    weight-only quantized Linear (e4m3 ``Fp8Linear`` or ``Int8Linear``), or an
    already-wrapped ``LoRALinearLayer``.

    ``Fp8Linear`` and ``Int8Linear`` are ``nn.Module``s, NOT ``nn.Linear``
    subclasses, so both must be named explicitly. Omitting them is SILENT: on a
    quantized DiT the iterator simply yields no targets for those layers,
    ``apply_lora_group`` returns a small ``applied`` count without raising, and
    the generation proceeds looking exactly as if no LoRA had been selected.
    Same fix, same reasoning as ``krea2_lora._is_target`` and
    ``ideogram4_lora``'s ``is_target``.

    ``LoRALinearLayer`` reads only ``in_features`` / ``out_features`` /
    ``weight.device`` off the module it wraps, and calls it as a callable, all of
    which both quantized classes provide -- so wrapping one needs nothing else.
    """
    return isinstance(m, _lora_target_types())


def iter_anima_lora_targets(
    transformer: nn.Module,
    scope: Optional[Dict[str, bool]] = None,
):
    """Yield (module_path, parent_module, attr_name_or_int, current_module) for
    each LoRA-targetable Linear in the Anima DiT under the requested scope.

    `current_module` is whatever currently sits at `getattr(parent, attr)`
    (or `parent[attr]` for ModuleList children) — typically an nn.Linear on
    the un-LoRA'd model, or a LoRALinearLayer once wrapped. Including wrapped
    modules here lets restore_originals() and LoRA stacking find the slots
    after a previous load.

    Scope flags (all default False if not present in the dict; see
    DEFAULT_TRAINING_SCOPE for the typical training preset):

      "attention":   blocks.<N>.{self_attn,cross_attn}.{q,k,v,output}_proj
      "mlp":         blocks.<N>.mlp.{layer1, layer2}
      "mod":         blocks.<N>.adaln_modulation_{self_attn,cross_attn,mlp}.{1,2}
                     (AdaLN-LoRA dim 256 -> hidden, the two Linears inside the
                      Sequential)
      "llm_adapter": llm_adapter.blocks.<N>.{self_attn,cross_attn}.{q,k,v,o}_proj
                     + llm_adapter.blocks.<N>.mlp.{0,2}
                     + llm_adapter.in_proj (if Linear) + llm_adapter.out_proj
    """
    from core.adapters import LoRALinearLayer

    scope = scope or {}
    want_attn = bool(scope.get("attention", False))
    want_mlp = bool(scope.get("mlp", False))
    want_mod = bool(scope.get("mod", False))
    want_adapter = bool(scope.get("llm_adapter", False))

    is_linear_or_wrap = _is_lora_target

    for name, module in transformer.named_modules():
        cls_name = module.__class__.__name__

        # 1. DiT Block attention modules
        if want_attn and cls_name == _ANIMA_ATTENTION_CLASS_NAME:
            # Inside Block (top-level self_attn / cross_attn), NOT inside the
            # LLM Adapter (its inner attention is LLMAdapterAttention).
            if not name.startswith("llm_adapter") and (
                ".self_attn" in name or ".cross_attn" in name
            ):
                for attr in _BLOCK_ATTN_ATTRS:
                    current = getattr(module, attr, None)
                    if is_linear_or_wrap(current):
                        yield f"{name}.{attr}", module, attr, current
            continue  # don't double-visit the children below

        # 2. LLM Adapter inner attention (under llm_adapter.blocks.<N>)
        if want_adapter and cls_name == _LLM_ADAPTER_ATTN_CLASS_NAME:
            for attr in _LLM_ADAPTER_ATTN_ATTRS:
                current = getattr(module, attr, None)
                if is_linear_or_wrap(current):
                    yield f"{name}.{attr}", module, attr, current
            continue

    # The remaining scopes (mlp, mod, adapter mlp/projections) are easier to
    # enumerate via parameter-path inspection rather than class detection.
    for name, module in transformer.named_modules():
        if not is_linear_or_wrap(module):
            continue

        # 3. Block MLP: blocks.<N>.mlp.{layer1, layer2}
        if want_mlp and ".mlp." in name and name.split(".")[-1] in ("layer1", "layer2"):
            # Skip if this is actually inside the LLM Adapter
            if not name.startswith("llm_adapter"):
                parent, attr = _resolve_parent(transformer, name)
                if parent is not None:
                    yield name, parent, attr, module

        # 4. AdaLN modulation Linears
        # adaln_modulation_self_attn / cross_attn / mlp are nn.Sequential with
        # [SiLU, Linear(x_dim -> adaln_lora_dim), Linear(adaln_lora_dim -> 3*x_dim)]
        if want_mod and ".adaln_modulation_" in name and name.split(".")[-1] in ("1", "2"):
            if not name.startswith("llm_adapter"):
                parent, attr = _resolve_parent(transformer, name)
                if parent is not None:
                    yield name, parent, attr, module

        # 5. LLM Adapter MLP + outer projections
        if want_adapter and name.startswith("llm_adapter"):
            tail = name.split(".")[-1]
            # llm_adapter.blocks.<N>.mlp.{0, 2}  (nn.Sequential Linear, GELU, Linear)
            if ".mlp." in name and tail in ("0", "2"):
                parent, attr = _resolve_parent(transformer, name)
                if parent is not None:
                    yield name, parent, attr, module
            # llm_adapter.in_proj (only when Linear, not Identity)
            elif name == "llm_adapter.in_proj":
                parent, attr = _resolve_parent(transformer, name)
                if parent is not None:
                    yield name, parent, attr, module
            # llm_adapter.out_proj
            elif name == "llm_adapter.out_proj":
                parent, attr = _resolve_parent(transformer, name)
                if parent is not None:
                    yield name, parent, attr, module


def _resolve_parent(root: nn.Module, dotted_name: str):
    """Walk to the parent of `root.<dotted_name>`. Returns (parent, last_attr).

    last_attr is a str (attribute name) for nn.Module attrs, or an int for
    nn.Sequential / nn.ModuleList children — chosen so that setattr(parent, ...)
    or parent[int] works in apply/restore code.
    """
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit() and hasattr(parent, "__getitem__"):
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p, None)
            if parent is None:
                return None, None
    last = parts[-1]
    if last.isdigit() and hasattr(parent, "__getitem__"):
        return parent, int(last)
    return parent, last


def derive_scope_from_keys(module_paths) -> Dict[str, bool]:
    """Scope flags covering exactly the module paths a checkpoint carries.

    Classification order mirrors the gating in iter_anima_lora_targets (the
    llm_adapter prefix wins over the mlp/attention shapes nested under it). A
    path matching nothing enables no scope and therefore shows up in
    apply_lora_group's unmatched list instead of being dropped in silence.
    """
    scope = {"attention": False, "mlp": False, "mod": False, "llm_adapter": False}
    for path in module_paths:
        if path.startswith("llm_adapter"):
            scope["llm_adapter"] = True
        elif ".adaln_modulation_" in path:
            scope["mod"] = True
        elif ".mlp." in path:
            scope["mlp"] = True
        elif ".self_attn." in path or ".cross_attn." in path:
            scope["attention"] = True
    return scope


# --------- Apply / restore ---------

def apply_lora_group(
    transformer: nn.Module,
    grouped: Dict[str, Dict[str, torch.Tensor]],
    strength: float,
    lora_original_modules: Dict[str, nn.Linear],
    wrapped_keys: set,
    scope: Optional[Dict[str, bool]] = None,
) -> Tuple[int, List[str]]:
    """Wrap matching Linear modules in the transformer with LoRALinearLayer.

    Args:
        transformer: Anima DiT instance.
        grouped: Output of normalise_lora_state_dict().
        strength: User-supplied scalar; multiplies the (alpha/rank) scaling.
        lora_original_modules: Dict to record original modules for unload.
            Keyed by module_path; the first wrap wins so multiple LoRAs over
            the same module always restore back to the true original.
        wrapped_keys: Set of module_paths currently wrapped (for unload).
        scope: Target scope; defaults to the one derive_scope_from_keys reads
            off `grouped`, i.e. exactly what the checkpoint contains.

    Returns:
        (wrapped_count, unmatched_module_paths). `unmatched` is every module
        path the checkpoint carries that no target in the scope matched — the
        caller reports it; this function never drops one silently.
    """
    from core.adapters import LoRALinearLayer

    if scope is None:
        scope = derive_scope_from_keys(grouped.keys())

    applied = 0
    matched: set = set()
    for module_path, parent, attr, linear in iter_anima_lora_targets(transformer, scope):
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
        #
        # Fp8Linear / Int8Linear state their compute dtype outright, so ask them
        # rather than inferring it: an int8 weight is not floating point at all
        # and a bias-less quantized layer would otherwise fall through to the
        # bfloat16 default, which is right today only by coincidence.
        declared = getattr(true_original, "compute_dtype", None)
        if isinstance(declared, torch.dtype) and declared.is_floating_point:
            compute_dtype = declared
        elif true_original.bias is not None and true_original.bias.dtype.is_floating_point:
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

        # adaln_modulation_* and llm_adapter MLP targets sit inside an
        # nn.Sequential, so the slot is an index; block mlp.layer1/layer2 and
        # the attention projections are attribute-named.
        if isinstance(attr, int):
            parent[attr] = wrapper
        else:
            setattr(parent, attr, wrapper)
        matched.add(module_path)
        wrapped_keys.add(module_path)
        applied += 1

    return applied, sorted(set(grouped) - matched)


def restore_originals(
    transformer: nn.Module,
    lora_original_modules: Dict[str, nn.Linear],
    wrapped_keys: set,
) -> Tuple[int, List[str]]:
    """Revert every wrapped Linear to its original (pre-LoRA) module.

    Driven by the wrapped paths rather than by a target scope: the wrapped set
    spans whatever scopes the applied checkpoints carried, and unload must not
    depend on re-deriving them.

    Returns (restored_count, unresolved_paths); the caller must surface the
    latter, since those wrappers are still installed. Both bookkeeping
    structures are cleared, but only on a clean return -- an exception mid-loop
    skips that, so the caller must additionally key them to the CURRENT
    transformer (see ``AnimaMixin._anima_lora_state``) rather than rely on it.
    """
    restored = 0
    unresolved: List[str] = []
    for module_path in sorted(wrapped_keys):
        original = lora_original_modules.get(module_path)
        parent, attr = (None, None)
        if original is not None:
            parent, attr = _resolve_parent(transformer, module_path)
        if original is None or parent is None:
            unresolved.append(module_path)
            continue
        if isinstance(attr, int):
            parent[attr] = original
        else:
            setattr(parent, attr, original)
        restored += 1
    if unresolved:
        print(f"[AnimaLoRA] WARNING: {len(unresolved)} wrapped module(s) could not be "
              f"restored (first few: {unresolved[:5]}); the DiT may still carry LoRA "
              f"weights.")
    wrapped_keys.clear()
    lora_original_modules.clear()
    return restored, unresolved
