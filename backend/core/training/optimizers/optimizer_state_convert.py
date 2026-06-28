"""Cross-implementation optimizer-state conversion for resume.

A run can resume with a different optimizer than it was checkpointed with. The
8-bit families share an identical quantization scheme but use different state-dict
key names, so a raw ``load_state_dict`` either crashes (KeyError on the renamed
state during the first ``step``) or silently resets momentum/variance.

Verified equivalence (see docs / bitsandbytes ref):
- bitsandbytes ``AdamW8bit``/``Lion8bit`` (block_wise) and the Ring Buffer
  ``AdamW8bit_RingBuffer``/``Lion8bit_RingBuffer`` both use **blocksize 256**, the
  **same sorted dynamic/udynamic quantization maps**, and the **same blockwise
  dequant convention** ``value = qmap[code] * absmax[block]``.
- The ONLY differences are state-dict key names and bnb's extra per-state fields
  (``qmap1``/``qmap2``/``max1``/``new_max1``/``step``).

So conversion is a key remap + absmax copy (bit-exact for the quantized codes),
plus carrying the step counter. No dequant/requant round-trip is needed.

Supported pairs (both directions):
- bnb AdamW8bit  <-> AdamW8bit_RingBuffer   (standard 8-bit, not schedule-free)
- bnb Lion8bit   <-> Lion8bit_RingBuffer    (standard 8-bit, not schedule-free)

Anything else (different algorithm, 32-bit state, schedule-free/RAdam targets,
non-8-bit) returns None so the caller falls back to a fresh optimizer state.
"""

from typing import Optional, Tuple

import torch


# ---- format detection ------------------------------------------------------

def _first_param_state(state_dict: dict) -> Optional[dict]:
    state = state_dict.get("state") if isinstance(state_dict, dict) else None
    if not isinstance(state, dict) or not state:
        return None
    for v in state.values():
        if isinstance(v, dict):
            return v
    return None


def detect_state_format(state_dict: dict) -> Optional[str]:
    """Return one of bnb_adamw8bit / bnb_lion8bit / rb_adamw8bit / rb_lion8bit,
    or None if the format is unrecognised / not 8-bit."""
    st = _first_param_state(state_dict)
    if st is None:
        return None
    keys = set(st.keys())

    def is_uint8(name: str) -> bool:
        t = st.get(name)
        return isinstance(t, torch.Tensor) and t.dtype == torch.uint8

    # bitsandbytes: state1 (+ state2 for Adam), qmap1 present
    if "state1" in keys and "qmap1" in keys:
        if "state2" in keys:
            return "bnb_adamw8bit" if is_uint8("state1") else None
        return "bnb_lion8bit" if is_uint8("state1") else None

    # Ring Buffer: exp_avg (+ exp_avg_sq for Adam)
    if "exp_avg" in keys:
        if "exp_avg_sq" in keys:
            return "rb_adamw8bit" if is_uint8("exp_avg") else None
        # Lion ring buffer: exp_avg + absmax (no exp_avg_sq); schedule-free uses state_z
        if "absmax" in keys and is_uint8("exp_avg"):
            return "rb_lion8bit"
    return None


def _target_format(optimizer) -> Optional[str]:
    """Map the live target optimizer to a format tag, or None if unsupported
    (schedule-free / RAdam / non-8-bit targets cannot ingest a plain 8-bit dump)."""
    cls = type(optimizer).__name__
    if getattr(optimizer, "schedule_free", False) or getattr(optimizer, "use_radam", False):
        return None
    if cls == "AdamW8bit_RingBuffer":
        return "rb_adamw8bit"
    if cls == "Lion8bit_RingBuffer":
        return "rb_lion8bit"
    # bitsandbytes targets
    if cls in ("AdamW8bit", "PagedAdamW8bit"):
        return "bnb_adamw8bit"
    if cls in ("Lion8bit", "PagedLion8bit"):
        return "bnb_lion8bit"
    return None


# ---- per-param state remaps ------------------------------------------------

def _algo(fmt: str) -> str:
    return "adamw" if "adamw" in fmt else "lion"


def _impl(fmt: str) -> str:
    return "bnb" if fmt.startswith("bnb_") else "rb"


def _make_qmaps(device):
    """bnb signed (dynamic) + unsigned (udynamic) maps, for rb -> bnb."""
    try:
        from bitsandbytes.functional import create_dynamic_map
        q1 = create_dynamic_map(signed=True)
        q2 = create_dynamic_map(signed=False)
    except Exception:
        from .quantization_map import create_quantization_map
        q1 = create_quantization_map(signed=True)
        q2 = create_quantization_map(signed=False)
    return q1.to(device), q2.to(device)


def _param_is_8bit(src: dict, src_impl: str) -> bool:
    """A single parameter may be 8-bit (large) or 32-bit (small, below the
    optimizer's min_8bit_size threshold). Detect from the actual state."""
    name = "state1" if src_impl == "bnb" else "exp_avg"
    t = src.get(name)
    return isinstance(t, torch.Tensor) and t.dtype == torch.uint8


def _convert_param_state(src: dict, src_fmt: str, dst_fmt: str) -> dict:
    """Remap a single parameter's state between bnb and ring buffer, handling
    both 8-bit (quantized + absmax) and 32-bit (small-param fallback) params."""
    out: dict = {}
    si, di = _impl(src_fmt), _impl(dst_fmt)
    is_adam = _algo(src_fmt) == "adamw"
    eight = _param_is_8bit(src, si)

    # canonical pull from source (m = first moment, a = its absmax or None)
    if si == "bnb":
        m1 = src["state1"]
        a1 = src.get("absmax1")
        m2 = src.get("state2") if is_adam else None
        a2 = src.get("absmax2") if is_adam else None
    else:  # ring buffer
        m1 = src["exp_avg"]
        a1 = src.get("absmax1") if is_adam else src.get("absmax")
        m2 = src.get("exp_avg_sq") if is_adam else None
        a2 = src.get("absmax2") if is_adam else None

    # push into destination layout
    if di == "bnb":
        out["state1"] = m1
        if eight:
            out["absmax1"] = a1
            device = m1.device if isinstance(m1, torch.Tensor) else torch.device("cpu")
            q1, q2 = _make_qmaps(device)
            out["qmap1"] = q1
        if is_adam:
            out["state2"] = m2
            if eight:
                out["absmax2"] = a2
                out["qmap2"] = q2
        out["step"] = int(src.get("step", 0)) if "step" in src else 0
    else:  # ring buffer
        if is_adam:
            out["exp_avg"] = m1
            out["exp_avg_sq"] = m2
            if eight:
                out["absmax1"] = a1
                out["absmax2"] = a2
        else:
            out["exp_avg"] = m1
            if eight:
                out["absmax"] = a1
        out["is_8bit"] = bool(eight)
    return out


def maybe_convert_optimizer_state(
    saved_state_dict: dict,
    target_optimizer,
    log_prefix: str = "[Trainer]",
) -> Tuple[Optional[dict], int]:
    """If the saved optimizer state comes from a compatible-but-different 8-bit
    implementation than the target, return a converted state dict (using the
    TARGET's param_groups) plus the step counter to carry. Otherwise return
    (None, 0) and the caller should load the saved dict directly.

    Returns:
        (converted_state_dict_or_None, carry_step)
    """
    try:
        src_fmt = detect_state_format(saved_state_dict)
        dst_fmt = _target_format(target_optimizer)
    except Exception as e:
        print(f"{log_prefix} [OptConvert] detection failed: {e}")
        return None, 0

    if src_fmt is None or dst_fmt is None:
        return None, 0
    if src_fmt == dst_fmt:
        return None, 0  # same format -> normal load path
    if _algo(src_fmt) != _algo(dst_fmt):
        # adamw <-> lion cannot share state (different moment semantics)
        print(f"{log_prefix} [OptConvert] cannot convert {src_fmt} -> {dst_fmt} "
              f"(different algorithm); optimizer state will reset")
        return None, 0

    saved_state = saved_state_dict.get("state", {})
    converted_state = {}
    carry_step = 0
    try:
        for pid, st in saved_state.items():
            if not isinstance(st, dict):
                converted_state[pid] = st
                continue
            converted_state[pid] = _convert_param_state(st, src_fmt, dst_fmt)
            if "step" in st:
                try:
                    carry_step = max(carry_step, int(st["step"]))
                except Exception:
                    pass
    except KeyError as e:
        print(f"{log_prefix} [OptConvert] missing key {e} converting {src_fmt} -> "
              f"{dst_fmt}; optimizer state will reset")
        return None, 0

    # Use the TARGET optimizer's param_groups so group-level keys always match.
    target_groups = target_optimizer.state_dict().get("param_groups", [])
    converted = {"state": converted_state, "param_groups": target_groups}
    print(f"{log_prefix} [OptConvert] converted optimizer state {src_fmt} -> {dst_fmt} "
          f"({len(converted_state)} params, carry step={carry_step})")
    return converted, carry_step
