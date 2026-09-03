"""LoRA support for the SenseNova-U1.5-8B-MoT distillation LoRA.

This is a RUNTIME-ONLY LoRA (forward-time addition via ``LoRALinearLayer``,
never merged into the base weights). The base checkpoint stays canonical for
a later training phase, so this module never writes into the base's
``state_dict`` -- see ``apply_lora_group``/``restore_originals`` below.
Mirrors ``core.models.ideogram4.ideogram4_lora`` (the exact precedent: its
published checkpoints are quantized too, so LoRA-over-quantized is its
NORMAL case, not an edge case) rather than upstream's own
``load_and_merge_lora_weight_from_safetensors`` helper, which only merges.

KEY FORMAT
----------
The one real distillation checkpoint this repo has seen
(``SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors``, metadata
``tensor_kind: "neo_hf_lora"``) uses the module path VERBATIM -- no
``lora_unet_``/``diffusion_model.`` wrapping prefix, unlike every other
architecture's LoRA in this repo:

    language_model.model.layers.{N}.self_attn.q_proj_mot_gen.lora_down.weight
    language_model.model.layers.{N}.self_attn.q_proj_mot_gen.lora_up.weight
    language_model.model.layers.{N}.self_attn.q_proj_mot_gen.alpha
    language_model.model.layers.{N}.mlp_mot_gen.gate_proj.lora_down.weight
    ...

Verified against the real file: 882 tensors = 294 modules x
{lora_down.weight, lora_up.weight, alpha}, rank 128, ``lora_alpha: "8.0"``
(scale = alpha/rank = 0.0625), delta computed
``(alpha/rank) * (lora_up.weight @ lora_down.weight)`` in fp32 (upstream
``utils/lora.py:30-33``).

TARGET ASYMMETRY
-----------------
The ``_mot_gen`` suffix lands in TWO different places depending on branch:

  * ``self_attn``: the suffix is on the LINEAR's own attribute name --
    ``block.self_attn.q_proj_mot_gen`` (the ``Qwen3Attention`` instance
    itself, ``self_attn``, is shared between the understanding and
    generation paths; only the four projections are duplicated).
  * ``mlp``: the suffix is on the PARENT module's attribute name --
    ``block.mlp_mot_gen.gate_proj`` (the whole ``Qwen3MLP`` is duplicated;
    ``gate_proj``/``up_proj``/``down_proj`` inside it keep their plain
    names).

Getting this backwards silently drops every mlp target (the parsed module
path would end in a name that is never a real attribute) or every attn
target the same way, so the module path returned by ``_parse_key`` is used
UNCHANGED as the live-tree navigation path below rather than re-derived --
one source of truth for both directions.

The asymmetry is the GENERATION side's alone: the understanding branch is
``self_attn.{q,k,v,o}_proj`` and ``mlp.{gate,up,down}_proj``, plain names
throughout. ``iter_sensenova_lora_targets(transformer, branch=...)`` is the
one enumerator for both, used by training injection AND inference
application -- a second enumerator is how the two drift.

QUANTIZED BASE
--------------
Every one of these 294 target Linears is loaded as ``Int8Linear``
(``core.models.ideogram4.vendor.int8_linear``, the SAME weight-only
per-row-scaled int8 layout Ideogram 4/Krea 2/FLUX.2/Anima already use -- see
``core/models/sensenova/loader.py``'s module docstring). ``Int8Linear`` is an
``nn.Module``, NOT an ``nn.Linear`` subclass, so a naive
``isinstance(x, nn.Linear)`` scan silently drops every one of them -- the
run "succeeds" with a smaller applied count that merely looks like a
narrower scope. This exact trap has already been found and fixed on four
other architectures in this repo. ``_is_lora_target`` below is
MODULE-LEVEL (not a closure) so ``quantized_capability_parity_test`` can
find it by convention and check it against ``Int8Linear``/``Fp8Linear`` the
same way it checks Ideogram 4's.
"""

from __future__ import annotations

from typing import Any, Dict, Generator, Optional, Set, Tuple

import torch
from torch import nn
from safetensors import safe_open

from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
from core.models.ideogram4.vendor.int8_linear import Int8Linear


# ---------------------------------------------------------------------------
# Key parsing
# ---------------------------------------------------------------------------

_LORA_DOWN_SUFFIX = ".lora_down.weight"
_LORA_UP_SUFFIX = ".lora_up.weight"
_ALPHA_SUFFIX = ".alpha"

# The layer-index-bearing prefix every real target module path starts with;
# used only as a cheap file-format sniff (`load_lora_safetensors`), not by
# the per-key parser itself (`_parse_key` accepts any module path shape so a
# future non-MoT-doubled target -- should this checkpoint's scope ever widen
# -- is not silently dropped by an over-narrow prefix check).
LAYER_PREFIX = "language_model.model.layers."


def _parse_key(key: str) -> Optional[Tuple[str, str]]:
    """Return (module_path, suffix in {down, up, alpha}) for a LoRA key, else None."""
    if key.endswith(_LORA_DOWN_SUFFIX):
        return key[: -len(_LORA_DOWN_SUFFIX)], "down"
    if key.endswith(_LORA_UP_SUFFIX):
        return key[: -len(_LORA_UP_SUFFIX)], "up"
    if key.endswith(_ALPHA_SUFFIX):
        return key[: -len(_ALPHA_SUFFIX)], "alpha"
    return None


def normalise_lora_state_dict(
    raw: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Group raw LoRA tensors -> {module_path: {down, up, alpha?}}.

    Entries missing a down/up pair are dropped (mirrors
    ``ideogram4_lora.normalise_lora_state_dict``). There is no cond/uncond
    split here (unlike Ideogram 4's), so every key that parses is kept --
    including understanding-branch ones, which ``apply_lora_group`` reaches
    because it enumerates both MoT halves.
    """
    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in raw.items():
        parsed = _parse_key(key)
        if parsed is None:
            continue
        module_path, suffix = parsed
        grouped.setdefault(module_path, {})[suffix] = tensor
    return {m: v for m, v in grouped.items() if "down" in v and "up" in v}


def _looks_like_sensenova_key(key: str) -> bool:
    """Key-shape sniff covering BOTH branches.

    ``"mot_gen" in key`` alone recognises a generation-only or mixed file but
    drops a metadata-less understanding-bearing one into ``"unknown"``, so the
    understanding attribute names are matched too.
    """
    if not key.startswith(LAYER_PREFIX):
        return False
    if "mot_gen" in key:
        return True
    parsed = _parse_key(key)
    if parsed is None:
        return False
    module_path = parsed[0]
    return any(
        module_path.endswith(f".self_attn.{attr}") for attr in _UND_ATTN_LINEAR_ATTRS
    ) or any(
        module_path.endswith(f".{_UND_MLP_PARENT_ATTR}.{attr}")
        for attr in _MLP_LINEAR_ATTRS
    )


def load_lora_safetensors(path: str) -> Tuple[Dict[str, torch.Tensor], str, Dict[str, str]]:
    """Load a LoRA safetensors file -> (raw_state_dict, format_label, metadata).

    ``format_label`` is ``"neo_hf_lora"`` when the file carries that
    ``tensor_kind`` metadata (the real checkpoint does), when it declares a
    known ``lora_targets`` scope, or when its keys look like either branch's
    layer path even without the metadata (a re-saved copy might drop
    ``__metadata__``); otherwise ``"unknown"`` -- callers decide how loudly to
    reject an unrecognised file, this function never guesses.

    The metadata is returned rather than discarded so the caller can turn
    ``lora_targets`` into an EXPECTED applied count: an understanding-bearing
    file loaded by a build that only enumerates the generation branch applies
    fewer modules without raising (see ``check_lora_application``).
    """
    raw: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = dict(f.metadata() or {})
        for k in f.keys():
            raw[k] = f.get_tensor(k)
    recognised = (
        metadata.get("tensor_kind") == "neo_hf_lora"
        or metadata.get("lora_targets") in EXPECTED_MODULE_COUNTS
        or any(_looks_like_sensenova_key(k) for k in raw)
    )
    return raw, ("neo_hf_lora" if recognised else "unknown"), metadata


# ---------------------------------------------------------------------------
# Target enumeration
# ---------------------------------------------------------------------------

# Per decoder layer: (parent_attr, linear_attr) pairs. The generation
# self_attn four are suffix-on-the-linear; its mlp three are
# suffix-on-the-parent (see module docstring's TARGET ASYMMETRY section) --
# both are pre-resolved here so `iter_sensenova_lora_targets` need not
# special-case either shape. The understanding branch carries no suffix at all.
_ATTN_LINEAR_ATTRS = ("q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen")
_MLP_LINEAR_ATTRS = ("gate_proj", "up_proj", "down_proj")
_UND_ATTN_LINEAR_ATTRS = ("q_proj", "k_proj", "v_proj", "o_proj")
_GEN_MLP_PARENT_ATTR = "mlp_mot_gen"
_UND_MLP_PARENT_ATTR = "mlp"

# branch -> (attention Linear attrs, mlp parent attr, mlp Linear attrs)
_BRANCH_LAYOUT = {
    "gen": (_ATTN_LINEAR_ATTRS, _GEN_MLP_PARENT_ATTR, _MLP_LINEAR_ATTRS),
    "und": (_UND_ATTN_LINEAR_ATTRS, _UND_MLP_PARENT_ATTR, _MLP_LINEAR_ATTRS),
}
LORA_BRANCHES = ("gen", "und", "both")

# `lora_targets` checkpoint metadata -> the module count that scope implies.
# An understanding-only file is never produced (the training adapter refuses to
# save one) and is therefore not a scope this map recognises.
LORA_TARGET_LABELS = {"gen": "generation", "both": "generation+understanding"}
EXPECTED_MODULE_COUNTS = {"generation": 294, "generation+understanding": 588}


def und_gradient_unreachable_paths(num_layers: int = 42) -> Set[str]:
    """The understanding targets a t2i image loss structurally cannot reach.

    A prefix forward keeps ``past_key_values`` and discards
    ``last_hidden_state``, so the LAST layer's post-attention half feeds
    nothing: its ``q_proj``/``o_proj`` and all three MLP projections receive no
    gradient, while its ``k_proj``/``v_proj`` do (generation layer N-1 consumes
    their K/V). Inference discards the same tensor, so this is the model's
    shape, not a defect -- measured on the real checkpoint in Phase U-0.

    Enumeration keeps all 294 anyway, so the five stay at their zero
    ``lora_up`` init and contribute nothing at inference. This exists so a
    census can predict them BY NAME instead of asserting that every trained
    tensor moved, which fails on exactly these five.
    """
    last = f"language_model.model.layers.{num_layers - 1}"
    return {
        f"{last}.self_attn.q_proj",
        f"{last}.self_attn.o_proj",
        *(f"{last}.{_UND_MLP_PARENT_ATTR}.{attr}" for attr in _MLP_LINEAR_ATTRS),
    }


def _set_module(parent: Any, attr: str, module: nn.Module) -> None:
    setattr(parent, attr, module)


def _is_lora_target(m: Any) -> bool:
    """True for a module a LoRA can wrap or re-wrap on the SenseNova gen branch.

    A plain ``nn.Linear``, EITHER weight-only quantized Linear (only
    ``Int8Linear`` is ever produced by this arch's loader, but ``Fp8Linear``
    is accepted identically -- the parity test checks both, and a predicate
    that only recognised the one class this checkpoint happens to use would
    still be the exact isinstance(x, nn.Linear)-shaped trap for the other),
    or an already-wrapped ``LoRALinearLayer`` (yielded so re-application and
    stacking find the slot). See the module docstring's QUANTIZED BASE
    section for why ``Int8Linear``/``Fp8Linear`` cannot be reached through a
    bare ``isinstance(m, nn.Linear)``.
    """
    from core.adapters import LoRALinearLayer

    return isinstance(m, (nn.Linear, Fp8Linear, Int8Linear, LoRALinearLayer))


def iter_sensenova_lora_targets(
    transformer: nn.Module,
    *,
    branch: str = "gen",
) -> Generator[Tuple[str, Any, str, nn.Module], None, None]:
    """Yield (module_path, parent, attr, current_module) per LoRA target.

    ``transformer`` is the live ``NEOChatModel`` (the ``"transformer"``
    component key the loader returns). Walks
    ``transformer.language_model.model.layers`` -- absent on any tree that
    is not this arch's, in which case this yields nothing rather than
    raising, matching ``iter_ideogram4_lora_targets``'s "return silently on
    an unexpected tree shape" convention.

    ``branch`` selects the MoT half: ``"gen"`` (default, the distillation
    LoRA's scope), ``"und"``, or ``"both"`` (every generation target, then
    every understanding one). This is the ONLY target enumerator -- training
    injection and inference application both drive it.
    """
    if branch not in LORA_BRANCHES:
        raise ValueError(
            f"Unknown SenseNova LoRA branch {branch!r} (expected one of {list(LORA_BRANCHES)})"
        )
    is_target = _is_lora_target

    language_model = getattr(transformer, "language_model", None)
    llm_core = getattr(language_model, "model", None) if language_model is not None else None
    layers = getattr(llm_core, "layers", None)
    if layers is None:
        return

    selected = ("gen", "und") if branch == "both" else (branch,)
    for branch_name in selected:
        attn_attrs, mlp_parent_attr, mlp_attrs = _BRANCH_LAYOUT[branch_name]
        for layer_idx, block in enumerate(layers):
            prefix = f"language_model.model.layers.{layer_idx}"

            attn = getattr(block, "self_attn", None)
            if attn is not None:
                for attr_name in attn_attrs:
                    m = getattr(attn, attr_name, None)
                    if is_target(m):
                        yield f"{prefix}.self_attn.{attr_name}", attn, attr_name, m

            mlp = getattr(block, mlp_parent_attr, None)
            if mlp is not None:
                for attr_name in mlp_attrs:
                    m = getattr(mlp, attr_name, None)
                    if is_target(m):
                        yield f"{prefix}.{mlp_parent_attr}.{attr_name}", mlp, attr_name, m


# ---------------------------------------------------------------------------
# Apply / restore (inference)
# ---------------------------------------------------------------------------

def metadata_alpha(metadata: Optional[Dict[str, str]]) -> Optional[float]:
    """File-level ``lora_alpha``/``ss_network_alpha``, or ``None``.

    The middle tier of the alpha precedence per-key ``.alpha`` tensor -> file
    metadata -> rank. This repo's own trainer and the real distillation
    checkpoint both write per-key alphas, so this only rescues a file that
    carries the scale in its metadata alone -- which would otherwise apply at
    1.0 instead of its trained scale.
    """
    for key in ("lora_alpha", "ss_network_alpha"):
        value = (metadata or {}).get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def apply_lora_group(
    transformer: nn.Module,
    grouped: Dict[str, Dict[str, torch.Tensor]],
    strength: float,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: Set[str],
    *,
    branch: str = "both",
    file_alpha: Optional[float] = None,
    shadowed: Optional[list] = None,
) -> int:
    """Wrap matching modules with ``LoRALinearLayer`` (reversible).

    NEVER merges into the ``Int8Linear`` base -- the wrapper computes
    ``base(x) + scale * lora_up(lora_down(x))`` at forward time, exactly the
    ``ideogram4_lora.apply_lora_group`` shape, so the quantized base tensors
    are never touched and ``restore_originals`` can always recover the
    pre-LoRA module by reference.

    ``branch`` defaults to ``"both"`` because application is LOOKUP-driven: a
    generation-only file simply misses on every understanding slot, so the
    existing distillation checkpoint keeps applying exactly 294 modules while
    a gen+und file stops being silently truncated to its generation half.

    A target an EARLIER LoRA in the same request already wrapped is SKIPPED
    and appended to ``shadowed`` when the caller supplies a list.
    ``LoRALinearLayer`` exposes ``weight``/``bias`` but not
    ``in_features``/``out_features``, so it cannot wrap a wrapper; re-wrapping
    the recovered original instead silently DISCARDED the earlier LoRA.
    Additive composition is ``CompositeAdapterLinear`` work
    (LYCORIS_ADAPTER_DESIGN Phase 1); the caller refuses a fully shadowed
    stack rather than faking it.

    ``file_alpha`` is the middle tier of the alpha precedence (see
    ``metadata_alpha``); it is used only for a module carrying no ``.alpha``.
    """
    from core.adapters import LoRALinearLayer

    applied = 0

    for module_path, parent, attr, linear in iter_sensenova_lora_targets(
        transformer, branch=branch
    ):
        weights = grouped.get(module_path)
        if weights is None:
            continue

        if isinstance(linear, LoRALinearLayer):
            if shadowed is not None:
                shadowed.append(module_path)
            continue

        down = weights["down"]
        up = weights["up"]
        alpha_tensor = weights.get("alpha")

        true_original = linear
        lora_original_modules.setdefault(module_path, true_original)

        rank = int(down.shape[0])
        if alpha_tensor is not None:
            alpha_value = float(alpha_tensor.item())
        elif file_alpha is not None:
            alpha_value = file_alpha
        else:
            alpha_value = float(rank)

        wrapper = LoRALinearLayer(true_original, rank=rank, alpha=alpha_value, lora_name=module_path)
        device = true_original.weight.device

        # Match the base compute dtype (int8 base -> bf16 compute), same
        # fallback chain as ideogram4_lora.apply_lora_group: a real floating
        # bias wins, else a real floating (non-quantized) weight dtype, else
        # bf16 for a quantized-only base with no bias.
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

    return applied


def restore_originals(
    transformer: nn.Module,
    lora_original_modules: Dict[str, nn.Module],
    wrapped_keys: Set[str],
    *,
    branch: str = "both",
) -> int:
    """Revert every wrapped module to its pre-LoRA original.

    Defaults to ``"both"`` for the same reason ``apply_lora_group`` does:
    restoration must cover every branch application could have touched, or an
    understanding wrapper survives the generation that installed it.

    Clears ``wrapped_keys`` but NOT ``lora_original_modules`` (a caller may
    inspect what was restored). The owner of a map that outlives one
    generation must clear it -- restoration is by map membership and
    ``apply_lora_group`` records with ``setdefault``, so a retained entry for
    an unloaded transformer would be written into the next model loaded.
    ``SenseNovaMixin._unload_lora_sensenova`` is that owner.
    """
    restored = 0
    for module_path, parent, attr, _linear in iter_sensenova_lora_targets(
        transformer, branch=branch
    ):
        if module_path in lora_original_modules:
            _set_module(parent, attr, lora_original_modules[module_path])
            restored += 1
    wrapped_keys.clear()
    return restored


def check_lora_application(
    grouped: Dict[str, Dict[str, torch.Tensor]],
    applied: int,
    metadata: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Return a message when a LoRA did not fully apply, else ``None``.

    Two independent checks, because a partial application raises nothing on
    its own -- it just returns a smaller count that looks like a narrower
    training scope:

    * every module the FILE carries must have reached a live module;
    * when the file declares a ``lora_targets`` scope, its module count must
      match what that scope implies (this is what catches a gen+und
      checkpoint read by a build that only knows the generation branch).
    """
    problems = []
    if applied != len(grouped):
        problems.append(
            f"{len(grouped) - applied} of {len(grouped)} module(s) in the file "
            f"matched no target in the loaded transformer"
        )
    declared = (metadata or {}).get("lora_targets")
    expected = EXPECTED_MODULE_COUNTS.get(declared) if declared else None
    if expected is not None and len(grouped) != expected:
        problems.append(
            f"metadata declares lora_targets={declared!r} ({expected} modules) "
            f"but the file carries {len(grouped)}"
        )
    if not problems:
        return None
    return (
        f"[SenseNova LoRA] applied {applied} module(s): " + "; ".join(problems) +
        ". The LoRA is only partially in effect."
    )
