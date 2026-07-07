"""Color Flatten (chroma smoothing / 色ムラ除去) for decoded images.

RGB-guided guided filter applied to the two YCoCg chroma channels (Co, Cg); the
luma (Y) channel is left untouched so edges and detail carried by luminance
survive while low-frequency chroma mottling is smoothed away.

This is the backend torch port of the frontend reference
``frontend/src/utils/postEdit.ts::flattenChroma`` and MUST stay numerically
equivalent to it. The math (YCoCg transform, clamped-window separable box
filter, 3x3 guide-covariance closed-form inverse, the second box pass over the
coefficients, the blend cap) mirrors that reference exactly. Domain note: like
the reference it operates on the sRGB-coded [0,1] values directly (no
linearization / gamma conversion).

Applied to the DECODED image tensor (in [0,1]) right after VAE decode, before
PIL conversion. ``strength <= 0`` is a hard, zero-cost no-op.
"""

from __future__ import annotations

import torch


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _box_filter_last(src: torch.Tensor, r: int) -> torch.Tensor:
    """Clamped-window box filter along the last dim (running sum via prefix
    sums). For each output index the average is taken over the valid clamped
    window [i-r, i+r] and divided by the actual (clamped) sample count -
    identical to the reference boxfilter (normalize by window count, not 2r+1).

    src: [..., L] fp32. Returns same shape.
    """
    L = src.shape[-1]
    # Prefix sum with a leading zero: cs[..., k] = sum(src[..., :k]); length L+1.
    cs = torch.cumsum(src, dim=-1)
    zeros = torch.zeros(*src.shape[:-1], 1, dtype=src.dtype, device=src.device)
    cs = torch.cat([zeros, cs], dim=-1)  # [..., L+1]

    idx = torch.arange(L, device=src.device)
    lo = (idx - r).clamp(min=0)
    hi = (idx + r + 1).clamp(max=L)
    cnt = (hi - lo).to(src.dtype)
    out = (cs.index_select(-1, hi) - cs.index_select(-1, lo)) / cnt
    return out


def _box_filter_2d(src: torch.Tensor, r: int) -> torch.Tensor:
    """Separable box filter over a [H, W] plane. Horizontal pass then vertical
    pass, matching the reference (src -> tmp -> dst)."""
    tmp = _box_filter_last(src, r)                 # horizontal (over W)
    tmp = _box_filter_last(tmp.transpose(-1, -2), r)  # vertical (over H)
    return tmp.transpose(-1, -2)


def _flatten_chroma_single(img: torch.Tensor, f: float) -> torch.Tensor:
    """Process a single [3, H, W] fp32 image in [0,1]. Returns [3, H, W]."""
    C, H, W = img.shape
    long_side = max(W, H)

    # f > 1 extrapolates radius/eps toward "flatten everything"; blend MUST cap
    # at 1.0 (lerp past the smoothed value would invert colors).
    radius = round(_lerp(12.0, 40.0, f) * (long_side / 1024.0))
    if radius < 4:
        radius = 4
    max_radius = -(-long_side // 2)  # ceil(long_side / 2)
    if radius > max_radius:
        radius = max_radius
    eps = _lerp(1.5e-3, 8e-3, f)
    blend = min(1.0, _lerp(0.4, 1.0, f))

    R = img[0]
    G = img[1]
    B = img[2]
    Y = 0.25 * R + 0.5 * G + 0.25 * B
    Co = 0.5 * R - 0.5 * B
    Cg = -0.25 * R + 0.5 * G - 0.25 * B

    box = lambda t: _box_filter_2d(t, radius)

    mR = box(R)
    mG = box(G)
    mB = box(B)

    mRR = box(R * R)
    mRG = box(R * G)
    mRB = box(R * B)
    mGG = box(G * G)
    mGB = box(G * B)
    mBB = box(B * B)

    # Symmetric 3x3 (Sigma + eps*I) per pixel.
    a = mRR - mR * mR + eps  # vrr
    b = mRG - mR * mG        # vrg
    c = mRB - mR * mB        # vrb
    d = mGG - mG * mG + eps  # vgg
    e = mGB - mG * mB        # vgb
    g = mBB - mB * mB + eps  # vbb

    co00 = d * g - e * e
    co01 = c * e - b * g
    co02 = b * e - c * d
    co11 = a * g - c * c
    co12 = b * c - a * e
    co22 = a * d - b * b
    det = a * co00 + b * co01 + c * co02

    # Singular guide window -> inverse entries 0 (a-vector collapses to zero,
    # q -> box(mean(p))). Matches the reference |det| < 1e-12 branch.
    inv_det = torch.where(det.abs() < 1e-12, torch.zeros_like(det), 1.0 / det)
    i00 = co00 * inv_det
    i01 = co01 * inv_det
    i02 = co02 * inv_det
    i11 = co11 * inv_det
    i12 = co12 * inv_det
    i22 = co22 * inv_det

    def process_channel(p: torch.Tensor) -> torch.Tensor:
        mp = box(p)
        mRp = box(R * p)
        mGp = box(G * p)
        mBp = box(B * p)

        covR = mRp - mR * mp
        covG = mGp - mG * mp
        covB = mBp - mB * mp
        aR = i00 * covR + i01 * covG + i02 * covB
        aG = i01 * covR + i11 * covG + i12 * covB
        aB = i02 * covR + i12 * covG + i22 * covB
        bb = mp - (aR * mR + aG * mG + aB * mB)

        maR = box(aR)
        maG = box(aG)
        maB = box(aB)
        mb = box(bb)

        q = maR * R + maG * G + maB * B + mb
        return p * (1.0 - blend) + q * blend

    Co = process_channel(Co)
    Cg = process_channel(Cg)

    # Reconstruct RGB from Y + smoothed chroma.
    out = torch.stack([Y + Co - Cg, Y + Cg, Y - Co - Cg], dim=0)
    return out


def flatten_chroma(t: torch.Tensor, strength: int) -> torch.Tensor:
    """Apply Color Flatten to a decoded image tensor in [0,1].

    Args:
        t: image tensor, BCHW or CHW, float, values in [0,1]. Any device.
        strength: 0-100 (design range). ``<= 0`` is a hard no-op (returns ``t``
            unchanged, zero cost). Values > 100 extrapolate (blend still caps).

    Returns:
        Tensor of the same shape / dtype / device as ``t`` with the chroma
        channels smoothed. Output is NOT clamped here beyond the transform; the
        caller's existing clamp-to-range handles final quantization.
    """
    if strength is None or strength <= 0:
        return t

    f = strength / 100.0

    squeeze = False
    if t.dim() == 3:
        t = t.unsqueeze(0)
        squeeze = True
    if t.dim() != 4:
        raise ValueError(f"flatten_chroma expects CHW or BCHW, got shape {tuple(t.shape)}")

    orig_dtype = t.dtype
    work = t.float()
    outs = [_flatten_chroma_single(work[i], f) for i in range(work.shape[0])]
    out = torch.stack(outs, dim=0).to(orig_dtype)

    if squeeze:
        out = out.squeeze(0)
    return out
