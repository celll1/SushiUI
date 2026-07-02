"""
Runnable equivalence / guard tests for the unified attention conduit.

Run:
    d:/celll1/webui_cl/venv/Scripts/python.exe backend/core/attention/tests_conduit.py

Covers (per design R1 / R3):
    * sage-vs-native numerical equivalence at head_dim 64 and 128 for BOTH
      a Z-Image-shaped BSHD call and an SDXL-shaped layout='BHSD' call (R1).
    * flash-vs-native equivalence (bonus; same two layouts) when flash_attn is
      available.
    * GQA (n_kv < n_q) on the native path, incl. R3 auto-enable_gqa.
    * Guard downgrades: sage in TRAINING -> native, mask present -> native,
      head_dim > max -> native, GQA -> native; plus normalize_backend aliases
      and the 'sla' passthrough.

Kernel-equivalence subtests require CUDA + the respective library; when
unavailable they are reported SKIPPED (not FAILED). Guard / normalization
subtests run on CPU and always execute.
"""

import os
import sys

# Make ``core.*`` importable: backend/ is the package root (this file lives at
# backend/core/attention/tests_conduit.py -> parents[2] == backend/).
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from core.attention import (  # noqa: E402
    AttentionMode,
    dispatch_attention,
    normalize_backend,
    resolve_backend,
)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results = []


def record(name, status, detail=""):
    _results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def _rel_err(a, b):
    """Relative L2 error ||a-b|| / ||b||, computed in fp32."""
    a32, b32 = a.float(), b.float()
    return (torch.linalg.vector_norm(a32 - b32) / torch.linalg.vector_norm(b32).clamp_min(1e-12)).item()


def _has_cuda():
    return torch.cuda.is_available()


def _lib_available(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _make_qkv(layout, B, S, H, D, device, dtype, h_kv=None):
    """Build (q, k, v) in the requested layout. h_kv enables GQA (k/v heads)."""
    h_kv = h_kv or H
    if layout == "BSHD":
        q = torch.randn(B, S, H, D, device=device, dtype=dtype)
        k = torch.randn(B, S, h_kv, D, device=device, dtype=dtype)
        v = torch.randn(B, S, h_kv, D, device=device, dtype=dtype)
    else:  # BHSD
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, h_kv, S, D, device=device, dtype=dtype)
        v = torch.randn(B, h_kv, S, D, device=device, dtype=dtype)
    return q, k, v


# --------------------------------------------------------------------------
# Kernel equivalence: <backend> vs native
# --------------------------------------------------------------------------
def test_backend_equivalence(backend, layout, head_dim, lib_mod, tol):
    name = f"{backend}-vs-native | layout={layout} D={head_dim}"
    if not _has_cuda():
        record(name, SKIP, "no CUDA")
        return
    if not _lib_available(lib_mod):
        record(name, SKIP, f"{lib_mod} unavailable")
        return

    device = "cuda"
    dtype = torch.float16
    B, S, H = 2, 256, 8
    torch.manual_seed(0)
    q, k, v = _make_qkv(layout, B, S, H, head_dim, device, dtype)

    try:
        out_native = dispatch_attention(q, k, v, backend="native", layout=layout,
                                        mode=AttentionMode.INFERENCE)
        out_backend = dispatch_attention(q, k, v, backend=backend, layout=layout,
                                         mode=AttentionMode.INFERENCE)
    except Exception as e:  # noqa: BLE001
        record(name, FAIL, f"raised: {e}")
        return

    if out_backend.shape != out_native.shape:
        record(name, FAIL, f"shape mismatch {tuple(out_backend.shape)} vs {tuple(out_native.shape)}")
        return

    err = _rel_err(out_backend, out_native)
    status = PASS if err < tol else FAIL
    record(name, status, f"rel_L2_err={err:.4e} (tol {tol})")


# --------------------------------------------------------------------------
# GQA on native path (R3)
# --------------------------------------------------------------------------
def test_gqa_native(layout):
    name = f"GQA native (n_kv<n_q) | layout={layout}"
    if not _has_cuda():
        record(name, SKIP, "no CUDA")
        return

    device = "cuda"
    dtype = torch.float16
    B, S, H, H_KV, D = 2, 128, 8, 2, 64
    torch.manual_seed(1)
    q, k, v = _make_qkv(layout, B, S, H, D, device, dtype, h_kv=H_KV)

    try:
        # enable_gqa deliberately left False -> conduit must auto-enable (R3).
        out = dispatch_attention(q, k, v, backend="native", layout=layout,
                                 mode=AttentionMode.INFERENCE, enable_gqa=False)
    except Exception as e:  # noqa: BLE001
        record(name, FAIL, f"raised (auto-enable_gqa missing?): {e}")
        return

    # Reference: manual SDPA in BHSD with enable_gqa=True.
    if layout == "BSHD":
        qb, kb, vb = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    else:
        qb, kb, vb = q, k, v
    ref = F.scaled_dot_product_attention(qb, kb, vb, enable_gqa=True)
    ref = ref.transpose(1, 2) if layout == "BSHD" else ref

    if out.shape != ref.shape:
        record(name, FAIL, f"shape {tuple(out.shape)} vs ref {tuple(ref.shape)}")
        return
    err = _rel_err(out, ref)
    status = PASS if err < 1e-2 else FAIL
    record(name, status, f"rel_L2_err={err:.4e}")


# --------------------------------------------------------------------------
# Guard downgrades (CPU-only; always run)
# --------------------------------------------------------------------------
def test_guards():
    dev = "cpu"
    # equal-head tensors, head_dim 64 (sage-allowed) unless noted
    q = torch.randn(1, 16, 8, 64, device=dev)
    k = torch.randn(1, 16, 8, 64, device=dev)

    # TRAINING -> sage downgraded (no backward)
    r = resolve_backend("sage", AttentionMode.TRAINING, q, k, None, "BSHD")
    record("guard: sage in TRAINING -> native", PASS if r == "native" else FAIL, f"got {r}")

    # mask present -> sage downgraded
    mask = torch.ones(1, 16, dtype=torch.bool, device=dev)
    r = resolve_backend("sage", AttentionMode.INFERENCE, q, k, mask, "BSHD")
    record("guard: mask present -> native", PASS if r == "native" else FAIL, f"got {r}")

    # head_dim > sage max(128) -> downgraded
    q256 = torch.randn(1, 16, 8, 256, device=dev)
    k256 = torch.randn(1, 16, 8, 256, device=dev)
    r = resolve_backend("sage", AttentionMode.INFERENCE, q256, k256, None, "BSHD")
    record("guard: head_dim>max -> native (sage)", PASS if r == "native" else FAIL, f"got {r}")

    # head_dim not in sage allowed_set (40) -> downgraded
    q40 = torch.randn(1, 16, 8, 40, device=dev)
    k40 = torch.randn(1, 16, 8, 40, device=dev)
    r = resolve_backend("sage", AttentionMode.INFERENCE, q40, k40, None, "BSHD")
    record("guard: head_dim=40 not allowed -> native (sage, SD1.5)", PASS if r == "native" else FAIL, f"got {r}")

    # GQA -> sage downgraded (unequal heads), BSHD (heads dim 2)
    q_g = torch.randn(1, 16, 8, 64, device=dev)
    k_g = torch.randn(1, 16, 2, 64, device=dev)
    r = resolve_backend("sage", AttentionMode.INFERENCE, q_g, k_g, None, "BSHD")
    record("guard: GQA -> native (sage)", PASS if r == "native" else FAIL, f"got {r}")

    # GQA guard is layout-aware: BHSD (heads dim 1)
    q_gb = torch.randn(1, 8, 16, 64, device=dev)
    k_gb = torch.randn(1, 2, 16, 64, device=dev)
    r = resolve_backend("sage", AttentionMode.INFERENCE, q_gb, k_gb, None, "BHSD")
    record("guard: GQA -> native (sage, BHSD layout)", PASS if r == "native" else FAIL, f"got {r}")

    # flash: head_dim 64 inference, equal heads, no mask -> stays flash
    r = resolve_backend("flash", AttentionMode.INFERENCE, q, k, None, "BSHD")
    record("guard: flash inference D=64 -> flash", PASS if r == "flash" else FAIL, f"got {r}")

    # flash TRAINING allowed (has backward)
    r = resolve_backend("flash", AttentionMode.TRAINING, q, k, None, "BSHD")
    record("guard: flash in TRAINING -> flash", PASS if r == "flash" else FAIL, f"got {r}")

    # flash head_dim 256 (Ideogram4) within max -> flash (mask guard handles masked case)
    r = resolve_backend("flash", AttentionMode.INFERENCE, q256, k256, None, "BSHD")
    record("guard: flash D=256 -> flash", PASS if r == "flash" else FAIL, f"got {r}")


def test_normalize():
    cases = {
        None: "native",
        "normal": "native",
        "none": "native",
        "sdpa": "native",
        "NATIVE": "native",
        "Flash": "flash",
        "SAGE": "sage",
        "sla": "sla",          # passthrough, must NOT be clobbered (R2)
        "bogus_xyz": "native",  # unknown -> native
    }
    ok = True
    detail = []
    for inp, exp in cases.items():
        got = normalize_backend(inp)
        if got != exp:
            ok = False
            detail.append(f"{inp!r}->{got!r} (exp {exp!r})")
    record("normalize_backend aliases + sla passthrough (R2)", PASS if ok else FAIL,
           "; ".join(detail))


def test_sla_short_circuit():
    """'sla' must run without crashing and produce a correct-shape output."""
    dev = "cuda" if _has_cuda() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    q, k, v = _make_qkv("BSHD", 1, 32, 4, 64, dev, dtype)
    try:
        out = dispatch_attention(q, k, v, backend="sla", layout="BSHD",
                                 mode=AttentionMode.INFERENCE)
        ok = out.shape == q.shape
        record("sla passthrough short-circuit runs (R2)", PASS if ok else FAIL,
               f"out shape {tuple(out.shape)}")
    except Exception as e:  # noqa: BLE001
        record("sla passthrough short-circuit runs (R2)", FAIL, f"raised: {e}")


def main():
    print("=" * 70)
    print("Unified attention conduit tests")
    print(f"CUDA available: {_has_cuda()} | "
          f"flash_attn: {_lib_available('flash_attn')} | "
          f"sageattention: {_lib_available('sageattention')}")
    print("=" * 70)

    # R1: sage vs native, both layouts, head_dim 64 & 128.
    for layout in ("BSHD", "BHSD"):
        for D in (64, 128):
            test_backend_equivalence("sage", layout, D, "sageattention", tol=0.05)

    # Bonus: flash vs native, both layouts, head_dim 64 & 128.
    for layout in ("BSHD", "BHSD"):
        for D in (64, 128):
            test_backend_equivalence("flash", layout, D, "flash_attn", tol=0.02)

    # R3: GQA native path, both layouts.
    for layout in ("BSHD", "BHSD"):
        test_gqa_native(layout)

    # Guards + normalization + SLA passthrough (CPU-safe where possible).
    test_guards()
    test_normalize()
    test_sla_short_circuit()

    print("=" * 70)
    n_pass = sum(1 for _, s, _ in _results if s == PASS)
    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    n_skip = sum(1 for _, s, _ in _results if s == SKIP)
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed, {n_skip} skipped "
          f"(total {len(_results)})")
    print("=" * 70)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
