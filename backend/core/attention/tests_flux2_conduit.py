"""
Equivalence gate for the conduit-routed FLUX.2 attention processors.

Asserts that ConduitFlux2AttnProcessor / ConduitFlux2ParallelSelfAttnProcessor with
backend='native' are numerically identical to diffusers' default Flux2AttnProcessor /
Flux2ParallelSelfAttnProcessor (the clone differs by ONE line: the kernel call), and
that tq engages with a finite backward at FLUX.2's head_dim=128.

Run: "<venv>/python.exe" backend/core/attention/test_flux2_conduit.py
Skips (does not fail) when CUDA / diffusers FLUX.2 are unavailable.
"""

import os
import sys

# Make ``core.*`` importable: backend/ is the package root (this file lives at
# backend/core/attention/test_flux2_conduit.py -> parents[2] == backend/).
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

_results = []
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def record(name, status, detail=""):
    _results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def _has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    print("=" * 70)
    print("FLUX.2 conduit-processor equivalence gate")
    print("=" * 70)
    if not _has_cuda():
        record("cuda available", SKIP, "no CUDA; skipping FLUX.2 equivalence")
        return 0
    try:
        import torch
        from diffusers.models.transformers.transformer_flux2 import (
            Flux2Attention,
            Flux2AttnProcessor,
            Flux2ParallelSelfAttention,
            Flux2ParallelSelfAttnProcessor,
        )
        from core.attention import AttentionMode
        from core.inference.conduit_flux2 import (
            ConduitFlux2AttnProcessor,
            ConduitFlux2ParallelSelfAttnProcessor,
        )
    except Exception as e:  # noqa: BLE001
        record("imports", SKIP, f"diffusers FLUX.2 unavailable: {e}")
        return 0

    dev = "cuda"
    torch.manual_seed(0)
    H, dh = 8, 128  # FLUX.2 head_dim = 128
    qd = H * dh

    # --- dual-stream (joint text+image) native parity ---
    attn = Flux2Attention(query_dim=qd, heads=H, dim_head=dh, added_kv_proj_dim=qd, bias=True).to(dev).eval()
    B, Si, St = 1, 64, 16
    hs = torch.randn(B, Si, qd, device=dev)
    ehs = torch.randn(B, St, qd, device=dev)
    L = St + Si
    rope = (torch.randn(L, dh, device=dev).cos(), torch.randn(L, dh, device=dev).sin())
    with torch.no_grad():
        attn.set_processor(Flux2AttnProcessor())
        r_img, r_txt = attn(hs, ehs, image_rotary_emb=rope)
        attn.set_processor(ConduitFlux2AttnProcessor("native", AttentionMode.INFERENCE))
        c_img, c_txt = attn(hs, ehs, image_rotary_emb=rope)
    di = (c_img - r_img).abs().max().item()
    dt = (c_txt - r_txt).abs().max().item()
    record("dual-stream native == diffusers", PASS if max(di, dt) < 1e-4 else FAIL,
           f"img={di:.3e} txt={dt:.3e} (atol 1e-4)")

    # --- single-stream native parity ---
    attn2 = Flux2ParallelSelfAttention(query_dim=qd, heads=H, dim_head=dh, mlp_ratio=4.0).to(dev).eval()
    hs2 = torch.randn(B, Si, qd, device=dev)
    rope2 = (torch.randn(Si, dh, device=dev).cos(), torch.randn(Si, dh, device=dev).sin())
    with torch.no_grad():
        attn2.set_processor(Flux2ParallelSelfAttnProcessor())
        r = attn2(hs2, image_rotary_emb=rope2)
        attn2.set_processor(ConduitFlux2ParallelSelfAttnProcessor("native", AttentionMode.INFERENCE))
        c = attn2(hs2, image_rotary_emb=rope2)
    ds = (c - r).abs().max().item()
    record("single-stream native == diffusers", PASS if ds < 1e-4 else FAIL, f"max_diff={ds:.3e} (atol 1e-4)")

    # --- tq backward at head_dim 128 (single-stream) ---
    try:
        attn2 = Flux2ParallelSelfAttention(query_dim=qd, heads=H, dim_head=dh, mlp_ratio=4.0).to(dev).to(torch.bfloat16)
        attn2.train()
        attn2.set_processor(ConduitFlux2ParallelSelfAttnProcessor("tq", AttentionMode.TRAINING))
        hs3 = torch.randn(B, Si, qd, device=dev, dtype=torch.bfloat16)
        rope3 = (torch.randn(Si, dh, device=dev, dtype=torch.bfloat16).cos(),
                 torch.randn(Si, dh, device=dev, dtype=torch.bfloat16).sin())
        out = attn2(hs3, image_rotary_emb=rope3)
        out.float().pow(2).mean().backward()
        g = attn2.to_qkv_mlp_proj.weight.grad
        ok = bool(torch.isfinite(g).all().item()) and bool((g != 0).any().item())
        record("tq backward D=128 finite grads", PASS if ok else FAIL,
               f"finite={torch.isfinite(g).all().item()} nonzero={(g != 0).any().item()}")
    except Exception as e:  # noqa: BLE001
        record("tq backward D=128 finite grads", FAIL, f"raised: {e}")

    print("=" * 70)
    n_pass = sum(1 for _, s, _ in _results if s == PASS)
    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    n_skip = sum(1 for _, s, _ in _results if s == SKIP)
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed, {n_skip} skipped (total {len(_results)})")
    print("=" * 70)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
