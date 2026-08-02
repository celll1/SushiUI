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

The one supported 32-bit -> 8-bit migration is:
- torch ``AdamW`` -> bitsandbytes ``AdamW8bit``

For that migration the fp32 first and second moments are quantized with bnb's
native blockwise quantizer (block size 256, signed first moment and unsigned
second moment). Parameters below the TARGET optimizer's ``min_8bit_size`` keep
fp32 ``state1`` / ``state2`` buffers, exactly as a fresh AdamW8bit state would.

Anything else (different algorithm, unsupported 32-bit state,
schedule-free/RAdam targets, non-8-bit) returns None. Callers that deliberately
requested a cross-optimizer resume must treat None as a refusal, not feed the
foreign raw state to ``load_state_dict``: PyTorch's loader can accept renamed
state keys and only fail on the first optimizer step.
"""

import copy
from typing import Optional, Tuple

import torch


_SOURCE_IDENTITY_ALIASES = {
    "adamw": "torch_adamw",
    "torch.optim.adamw": "torch_adamw",
    "adamw8bit": "bnb_adamw8bit",
    "pagedadamw8bit": "bnb_adamw8bit",
    "paged_adamw8bit": "bnb_adamw8bit",
    "lion8bit": "bnb_lion8bit",
    "pagedlion8bit": "bnb_lion8bit",
    "paged_lion8bit": "bnb_lion8bit",
    "adamw8bit_ringbuffer": "rb_adamw8bit",
    "lion8bit_ringbuffer": "rb_lion8bit",
}


# ---- format detection ------------------------------------------------------

def _first_param_state(state_dict: dict) -> Optional[dict]:
    state = state_dict.get("state") if isinstance(state_dict, dict) else None
    if not isinstance(state, dict) or not state:
        return None
    for v in state.values():
        if isinstance(v, dict):
            return v
    return None


def detect_state_format(
    state_dict: dict,
    source_optimizer_name: Optional[str] = None,
) -> Optional[str]:
    """Return one of bnb_adamw8bit / bnb_lion8bit / rb_adamw8bit / rb_lion8bit,
    or None if the format is unrecognised / not 8-bit."""
    # ``exp_avg`` / ``exp_avg_sq`` alone identifies the Adam family, not AdamW
    # specifically. Only accept it when the checkpoint class tag (new saves) or
    # the VAE train_state config (legacy saves, supplied as a trusted hint) says
    # AdamW. This prevents an Adam state from being silently reinterpreted as
    # decoupled-weight-decay AdamW state.
    source_class = state_dict.get("_sushi_opt_class")
    tag_key = (str(source_class).strip().lower()
               if source_class is not None else None)
    hint_key = (str(source_optimizer_name).strip().lower()
                if source_optimizer_name is not None else None)
    tag_identity = _SOURCE_IDENTITY_ALIASES.get(tag_key)
    hint_identity = _SOURCE_IDENTITY_ALIASES.get(hint_key)
    if source_class is not None:
        # The checkpoint's own tag is authoritative. A known tag that disagrees
        # with a known external/config hint makes the state untrustworthy; an
        # unknown tag must not borrow the hint's identity either.
        if tag_identity is not None and hint_identity is not None and \
                tag_identity != hint_identity:
            return None
        source_identity = tag_identity
    else:
        source_identity = hint_identity
    claims_torch_adamw = source_identity == "torch_adamw"
    st = _first_param_state(state_dict)
    if st is None:
        # A checkpoint taken before the first optimizer step legitimately has
        # no per-parameter state. The class/config identity plus a structurally
        # valid empty state is sufficient; group compatibility is validated by
        # the converter before it returns the target's empty state dict.
        state = state_dict.get("state")
        groups = state_dict.get("param_groups")
        if claims_torch_adamw and isinstance(state, dict) and not state and \
                isinstance(groups, list):
            return "torch_adamw"
        return None
    keys = set(st.keys())

    if claims_torch_adamw:
        exp_avg = st.get("exp_avg")
        exp_avg_sq = st.get("exp_avg_sq")
        if isinstance(exp_avg, torch.Tensor) and \
                isinstance(exp_avg_sq, torch.Tensor) and \
                exp_avg.dtype == torch.float32 and \
                exp_avg_sq.dtype == torch.float32:
            return "torch_adamw"
        # Do not fall through and reinterpret a state that contradicts its
        # AdamW identity as bnb/ring-buffer merely because those keys happen to
        # match another supported converter format.
        return None

    def is_uint8(name: str) -> bool:
        t = st.get(name)
        return isinstance(t, torch.Tensor) and t.dtype == torch.uint8

    # bitsandbytes: state1 (+ state2 for Adam), qmap1 present
    if "state1" in keys and "qmap1" in keys:
        if "state2" in keys:
            detected = "bnb_adamw8bit" if is_uint8("state1") else None
        else:
            detected = "bnb_lion8bit" if is_uint8("state1") else None
        return detected if source_identity in (None, detected) else None

    # Ring Buffer: exp_avg (+ exp_avg_sq for Adam)
    if "exp_avg" in keys:
        if "exp_avg_sq" in keys:
            detected = "rb_adamw8bit" if is_uint8("exp_avg") else None
            return detected if source_identity in (None, detected) else None
        # Lion ring buffer: exp_avg + absmax (no exp_avg_sq); schedule-free uses state_z
        if "absmax" in keys and is_uint8("exp_avg"):
            detected = "rb_lion8bit"
            return detected if source_identity in (None, detected) else None
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
    if fmt.startswith("bnb_"):
        return "bnb"
    if fmt.startswith("torch_"):
        return "torch"
    return "rb"


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


def _convert_param_state(src: dict, src_fmt: str, dst_fmt: str) -> Optional[dict]:
    """Remap a single 8-bit parameter's state between bnb and ring buffer.

    Returns None for non-8-bit (small, below-threshold) params: the two
    implementations disagree on which params are quantized (bnb skips params
    below min_8bit_size; the Ring Buffer optimizer quantizes every param in a
    use_8bit group), so a 32-bit source param cannot be safely placed into the
    target without re-quantization. Skipping it lets the target fresh-init that
    param on its first step (these are tiny, e.g. biases -- negligible momentum).
    """
    si, di = _impl(src_fmt), _impl(dst_fmt)
    if not _param_is_8bit(src, si):
        return None

    is_adam = _algo(src_fmt) == "adamw"
    out: dict = {}

    # canonical pull from source (m = first moment, a = its absmax)
    if si == "bnb":
        m1, a1 = src["state1"], src["absmax1"]
        m2 = src.get("state2") if is_adam else None
        a2 = src.get("absmax2") if is_adam else None
    else:  # ring buffer
        m1 = src["exp_avg"]
        a1 = src["absmax1"] if is_adam else src["absmax"]
        m2 = src.get("exp_avg_sq") if is_adam else None
        a2 = src.get("absmax2") if is_adam else None

    # push into destination layout
    if di == "bnb":
        out["state1"] = m1
        out["absmax1"] = a1
        device = m1.device if isinstance(m1, torch.Tensor) else torch.device("cpu")
        q1, q2 = _make_qmaps(device)
        out["qmap1"] = q1
        if is_adam:
            out["state2"] = m2
            out["absmax2"] = a2
            out["qmap2"] = q2
        out["step"] = int(src.get("step", 0)) if "step" in src else 0
    else:  # ring buffer
        if is_adam:
            out["exp_avg"] = m1
            out["exp_avg_sq"] = m2
            out["absmax1"] = a1
            out["absmax2"] = a2
        else:
            out["exp_avg"] = m1
            out["absmax"] = a1
        out["is_8bit"] = True
    return out


def _as_non_negative_step(value) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("AdamW step must be a scalar")
        value = value.detach().cpu().item()
    step = int(value)
    if step < 0:
        raise ValueError(f"AdamW step must be non-negative, got {step}")
    return step


def _convert_torch_adamw_to_bnb(
    saved_state_dict: dict,
    target_optimizer,
) -> Tuple[dict, int]:
    """Convert torch AdamW fp32 moments into a bnb AdamW8bit state dict.

    Parameter IDs are optimizer-state serialization details, so source IDs are
    mapped to target IDs by group and position. The group layout and every
    populated moment shape are validated before conversion. Any mismatch raises
    instead of producing a dict that might load successfully and fail later.
    """
    if type(target_optimizer).__name__ != "AdamW8bit":
        raise ValueError("torch AdamW conversion requires an AdamW8bit target")

    saved_groups = saved_state_dict.get("param_groups")
    saved_state = saved_state_dict.get("state")
    target_dict = target_optimizer.state_dict()
    target_groups = target_dict.get("param_groups")
    live_groups = getattr(target_optimizer, "param_groups", None)
    if not isinstance(saved_groups, list) or not isinstance(saved_state, dict):
        raise ValueError("saved AdamW state has no state/param_groups mapping")
    if not isinstance(target_groups, list) or not isinstance(live_groups, list):
        raise ValueError("target AdamW8bit has no param_groups mapping")
    if len(saved_groups) != len(target_groups) or len(target_groups) != len(live_groups):
        raise ValueError(
            "optimizer param-group count differs "
            f"(checkpoint={len(saved_groups)}, target={len(target_groups)})"
        )

    qmap1, qmap2 = _make_qmaps(torch.device("cpu"))
    try:
        from bitsandbytes.functional import quantize_blockwise
    except Exception as e:
        raise RuntimeError(f"bitsandbytes blockwise quantizer is unavailable: {e}") from e

    converted_state = {}
    carry_step = 0
    seen_saved_ids = set()
    seen_target_ids = set()
    for gindex, (saved_group, target_group, live_group) in enumerate(
            zip(saved_groups, target_groups, live_groups)):
        saved_ids = saved_group.get("params")
        target_ids = target_group.get("params")
        live_params = live_group.get("params")
        if not isinstance(saved_ids, list) or not isinstance(target_ids, list) or \
                not isinstance(live_params, list):
            raise ValueError(f"optimizer param group {gindex} has no params list")
        if len(saved_ids) != len(target_ids) or len(target_ids) != len(live_params):
            raise ValueError(
                f"optimizer param group {gindex} size differs "
                f"(checkpoint={len(saved_ids)}, target={len(target_ids)})"
            )

        for pindex, (saved_id, target_id, param) in enumerate(
                zip(saved_ids, target_ids, live_params)):
            if saved_id in seen_saved_ids or target_id in seen_target_ids:
                raise ValueError("optimizer param_groups contain duplicate parameter IDs")
            seen_saved_ids.add(saved_id)
            seen_target_ids.add(target_id)

            src = saved_state.get(saved_id)
            if src is None:
                # Torch creates state lazily. An untouched parameter should stay
                # lazy in the target too, rather than receiving invented moments.
                continue
            if not isinstance(src, dict):
                raise ValueError(
                    f"optimizer state for group {gindex} param {pindex} is not a mapping"
                )
            exp_avg = src.get("exp_avg")
            exp_avg_sq = src.get("exp_avg_sq")
            if not isinstance(exp_avg, torch.Tensor) or \
                    not isinstance(exp_avg_sq, torch.Tensor):
                raise ValueError(
                    f"AdamW state for group {gindex} param {pindex} lacks moments"
                )
            expected_shape = tuple(param.shape)
            if tuple(exp_avg.shape) != expected_shape or \
                    tuple(exp_avg_sq.shape) != expected_shape:
                raise ValueError(
                    f"AdamW moment shape differs for group {gindex} param {pindex} "
                    f"(checkpoint={tuple(exp_avg.shape)}/{tuple(exp_avg_sq.shape)}, "
                    f"target={expected_shape})"
                )
            if exp_avg.dtype != torch.float32 or exp_avg_sq.dtype != torch.float32:
                raise ValueError(
                    f"AdamW moments for group {gindex} param {pindex} must be fp32"
                )

            try:
                config = target_optimizer.get_config(gindex, pindex, live_group)
            except Exception as e:
                raise ValueError(
                    f"cannot read AdamW8bit config for group {gindex} param {pindex}: {e}"
                ) from e
            if int(config.get("optim_bits", 0)) != 8:
                raise ValueError("target AdamW8bit parameter is not configured for 8-bit state")
            if not bool(config.get("block_wise", False)):
                raise ValueError("only block-wise AdamW8bit state is supported")
            min_8bit_size = int(config.get("min_8bit_size", 4096))
            if min_8bit_size < 0:
                raise ValueError(f"invalid target min_8bit_size={min_8bit_size}")

            step = _as_non_negative_step(src.get("step", 0))
            carry_step = max(carry_step, step)
            out = {"step": step}
            first = exp_avg.detach().to(device="cpu", dtype=torch.float32).contiguous()
            second = exp_avg_sq.detach().to(device="cpu", dtype=torch.float32).contiguous()
            if param.numel() < min_8bit_size:
                out["state1"] = first.clone()
                out["state2"] = second.clone()
            else:
                state1, quant1 = quantize_blockwise(
                    first, code=qmap1, blocksize=256
                )
                state2, quant2 = quantize_blockwise(
                    second, code=qmap2, blocksize=256
                )
                out.update({
                    "state1": state1,
                    "qmap1": quant1.code,
                    "absmax1": quant1.absmax,
                    "state2": state2,
                    "qmap2": quant2.code,
                    "absmax2": quant2.absmax,
                })
            if int(config.get("percentile_clipping", 100)) < 100:
                out["gnorm_vec"] = torch.zeros(100, dtype=torch.float32)
            if float(config.get("max_unorm", 0.0)) > 0.0:
                out["unorm_vec"] = torch.zeros(1, dtype=torch.float32)
            converted_state[target_id] = out

    unknown_state_ids = set(saved_state) - seen_saved_ids
    if unknown_state_ids:
        raise ValueError(
            f"saved optimizer state contains {len(unknown_state_ids)} parameter ID(s) "
            "not present in param_groups"
        )

    return {
        "state": converted_state,
        # Deep-copy the TARGET groups: loading source hyperparameters would undo
        # the current run's optimizer configuration before LR/WD reassertion.
        "param_groups": copy.deepcopy(target_groups),
    }, carry_step


def maybe_convert_optimizer_state(
    saved_state_dict: dict,
    target_optimizer,
    log_prefix: str = "[Trainer]",
    source_optimizer_name: Optional[str] = None,
) -> Tuple[Optional[dict], int]:
    """If the saved optimizer state comes from a compatible-but-different 8-bit
    implementation than the target, return a converted state dict (using the
    TARGET's param_groups) plus the step counter to carry. Otherwise return
    (None, 0) and the caller should load the saved dict directly.

    Returns:
        (converted_state_dict_or_None, carry_step)
    """
    try:
        src_fmt = detect_state_format(saved_state_dict, source_optimizer_name)
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

    if src_fmt == "torch_adamw":
        if dst_fmt != "bnb_adamw8bit" or type(target_optimizer).__name__ != "AdamW8bit":
            print(f"{log_prefix} [OptConvert] cannot convert {src_fmt} -> {dst_fmt}; "
                  f"optimizer state will reset")
            return None, 0
        try:
            converted, carry_step = _convert_torch_adamw_to_bnb(
                saved_state_dict, target_optimizer
            )
        except Exception as e:
            print(f"{log_prefix} [OptConvert] conversion failed: {e}")
            return None, 0
        print(f"{log_prefix} [OptConvert] converted optimizer state "
              f"torch_adamw -> bnb_adamw8bit "
              f"({len(converted['state'])} params, carry step={carry_step})")
        return converted, carry_step

    saved_state = saved_state_dict.get("state", {})
    converted_state = {}
    carry_step = 0
    skipped = 0
    try:
        for pid, st in saved_state.items():
            if not isinstance(st, dict):
                continue
            conv = _convert_param_state(st, src_fmt, dst_fmt)
            if conv is None:
                # non-8-bit (small) param -> let the target fresh-init it
                skipped += 1
            else:
                converted_state[pid] = conv
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
          f"({len(converted_state)} 8-bit params converted, {skipped} small params "
          f"fresh-init, carry step={carry_step})")
    return converted, carry_step
