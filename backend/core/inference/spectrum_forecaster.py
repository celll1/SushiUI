"""Spectrum: Adaptive Spectral Feature Forecasting (training-free DiT/U-Net accel).

Implements arXiv 2603.01623. The denoiser output is treated as a function of the
(normalized) diffusion timestep and approximated by Chebyshev polynomials whose
coefficients are fit online by ridge regression over the outputs of ACTUAL network
passes (anchor steps). On skipped steps the network forward is replaced by a cheap
polynomial forecast of the output, giving a large speedup at high step counts with a
non-compounding error bound (vs the local Taylor/cache approximations).

Math (paper):
    g(t) = 2t - 1                          # normalized timestep [0,1] -> [-1,1]
    T0=1, T1=τ, Tm = 2τ T(m-1) - T(m-2)    # Chebyshev, 1st kind
    φ(τ) = [T0(τ), ..., T_{M}(τ)]          # M+1 basis  (Eq. 9)
    Φ = stack φ(g(t_k)) over cached anchors # [K, M+1]  (Eq. 10)
    H = stack h_{t_k}                       # [K, F]    (Eq. 11), F = flattened output
    C = (ΦᵀΦ + λI)^{-1} Φᵀ H                # ridge close-form (Eq. 13)
    h_{t_j} = φ(g(t_j)) C                   # forecast  (Eq. 14)

This module is model-agnostic: ``record``/``forecast`` operate on whatever output
tensor the caller hands them (here, the raw pre-CFG U-Net output).
"""

import torch


# M2b -- trajectory speed limiter safety factor. A forecast may advance past the last
# real anchor by at most K times the distance implied by the recently-observed real
# per-step trajectory speed. Directly bounds time-direction overshoot (the empirically
# identified oversaturation mechanism) without shrinking toward the anchor. User-tunable
# via SpectrumForecaster(delta_cap=...); this constant is only the default value. <=0
# disables the cap entirely (restores pre-cap oversaturation-prone behavior).
DELTA_CAP_K = 1.25


def _cheb_row(tau: float, num_basis: int, device, dtype) -> torch.Tensor:
    """Chebyshev (1st kind) basis row [T0..T_{num_basis-1}] evaluated at tau in [-1,1]."""
    row = torch.empty(num_basis, device=device, dtype=dtype)
    row[0] = 1.0
    if num_basis > 1:
        row[1] = tau
    for m in range(2, num_basis):
        row[m] = 2.0 * tau * row[m - 1] - row[m - 2]
    return row


def build_output_forecaster(params, num_steps, label=""):
    """Build an output-mode SpectrumForecaster from generation params, or return None.

    Shared entry point for the DiT backends (Z-Image / FLUX.2 / Anima / Lens / Ideogram4
    / MiniT2I), which forecast their final per-step model output (velocity/noise). Honors
    the same spectrum_* params and auto-disable rules as the SD/SDXL path. block mode is
    U-Net-only, so this always uses output mode (logs if block was requested). Defaults the
    sliding window to a small local size for extrapolation stability.
    """
    if not params.get("spectrum_enable", False):
        return None
    warmup = int(params.get("spectrum_warmup_steps", 3))
    if num_steps < warmup + 3:
        print(f"[Spectrum] {label}: requested but disabled ({num_steps} steps < warmup+3; "
              f"little benefit at low step counts)")
        return None
    if params.get("spectrum_feature_mode", "output") == "block":
        print(f"[Spectrum] {label}: block mode is U-Net (SD/SDXL) only; using output mode on this DiT model")
    max_cache = params.get("spectrum_max_cache", 0)
    fc = SpectrumForecaster(
        int(num_steps),
        num_basis=int(params.get("spectrum_m", 4)),
        lam=float(params.get("spectrum_lam", 0.1)),
        w=float(params.get("spectrum_w", 0.5)),
        w_decay=float(params.get("spectrum_w_decay", 1.0)),
        warmup_steps=warmup,
        window_size=int(params.get("spectrum_window_size", 4)),
        flex_window=float(params.get("spectrum_flex_window", 0.75)),
        tail_fraction=float(params.get("spectrum_tail", 0.12)),
        max_cache=int(max_cache) if int(max_cache) > 0 else 5,
        delta_cap=float(params.get("spectrum_delta_cap", 1.25)),
    )
    print(f"[Spectrum] {label}: enabled (output mode) {len(fc.anchors)}/{num_steps} actual passes")
    return fc


def build_anchor_schedule(num_steps: int, warmup_steps: int, window_size: int,
                          flex_window: float, tail_fraction: float = 0.12):
    """Decide which step indices are anchors (actual forward) vs forecast (skipped).

    Faithful to the paper's adaptive scheduler (dense early, sparse late) with a
    TaylorSeer-style warm-up: the first ``warmup_steps`` are always anchors, then the
    skip interval starts at ``window_size`` and grows, damped by ``flex_window`` (0 =
    skip the full window, 1 = never skip). The first and last steps are always anchors.
    ``tail_fraction`` of the final steps (>=2) are forced to actual passes to protect
    fine detail at low noise. Returns a set of anchor step indices.
    """
    anchors = set()
    if num_steps <= 0:
        return anchors
    warmup = max(1, min(warmup_steps, num_steps))
    for i in range(warmup):
        anchors.add(i)
    anchors.add(0)

    # Reserve the tail steps as actual passes. Every forecast extrapolates beyond the
    # most recent anchor, and the LOW-noise tail carries the fine detail, so a large
    # skip gap there visibly degrades sharpness. Keep the last tail_fraction (>=2).
    tail_frac = max(0.0, min(1.0, float(tail_fraction)))
    tail = max(2, int(round(tail_frac * num_steps)))
    tail = min(tail, num_steps)
    tail_start = num_steps - tail
    for i in range(tail_start, num_steps):
        anchors.add(i)

    keep = max(0.0, min(1.0, float(flex_window)))  # fraction of the window kept as actual passes
    interval = max(1, int(window_size))
    i = warmup
    grow = 0
    while i < tail_start:
        anchors.add(i)                  # actual pass
        # number of steps to skip after this anchor; flex_window damps it down
        win = interval + grow
        skip = int(round(win * (1.0 - keep)))
        skip = max(0, skip)
        i += 1 + skip
        grow += 1                       # intervals grow over the schedule (sparser late)
    return anchors


class SpectrumForecaster:
    """Online Chebyshev feature forecaster over actual-pass denoiser outputs.

    Usage per sampling step i (normalized time t_i = i/(N-1)):
        if forecaster.is_anchor(i):
            out = unet(...)                 # actual pass
            forecaster.record(i, out)
        else:
            out = forecaster.forecast(i)    # skip the forward
    """

    def __init__(self, num_steps, num_basis=4, lam=0.1, w=0.5, w_decay=1.0,
                 warmup_steps=3, window_size=4, flex_window=0.75, tail_fraction=0.12,
                 max_cache=0, delta_cap=DELTA_CAP_K):
        self.num_steps = int(num_steps)
        self.num_basis = max(1, int(num_basis))
        self.lam = float(lam)
        # Clamp w to [0,1]: the UI accepts free-typed values, and an out-of-range w flips
        # the sign of the cheb/linear mix (w<0) or over-weights the extrapolation (w>1).
        self.w = min(1.0, max(0.0, float(w)))
        # Per-step decay exponent for the spectral mix weight (M1). w is scaled by
        # (1 - step_frac)**w_decay so the extrapolated Chebyshev term (evaluated at tau>1)
        # contributes less at low-noise late steps, where overshoot injects ghosting.
        # 0.0 disables the decay (M1 off); the overshoot clamp (M2) in forecast()
        # still applies regardless, so 0.0 is not bit-identical to pre-mitigation
        # output whenever a forecast overshoots the last anchor's norm.
        self.w_decay = float(w_decay)
        # Keep only the most recent ``max_cache`` anchors (0 = unlimited). A finite
        # window caps memory (important for the large block-mode features) and makes the
        # fit local, which stabilizes extrapolation beyond the most recent anchor.
        self.max_cache = int(max_cache)
        # M2b delta-cap multiplier K (see module comment). <=0 disables the cap entirely.
        self.delta_cap = float(delta_cap)
        self.anchors = build_anchor_schedule(self.num_steps, warmup_steps, window_size,
                                             flex_window, tail_fraction)
        # caches of actual passes
        self._steps = []           # step indices of cached anchors (oldest..newest)
        self._H = []               # flattened outputs [F] (stored in source dtype)
        self._shape = None
        self._dtype = None
        self._device = None
        self._coeffs = None        # [M+1, F]
        self._n_forecast = 0
        self._n_anchor = 0

    def is_anchor(self, step_idx: int) -> bool:
        return step_idx in self.anchors

    def _window_tau(self, step_idx: int) -> float:
        """Map a step index to [-1,1] using the CURRENT cache window [oldest, newest].

        Local renormalization (rather than a fixed global g(t)=2t-1) keeps the Chebyshev
        fit well-conditioned: a sliding window of recent anchors spans a narrow slice of
        the global trajectory, and fitting Chebyshev there directly is ill-conditioned.
        The forecast step is newer than the newest cached anchor, so it maps to >1
        (short-range extrapolation whose distance is bounded by the window span).
        """
        if len(self._steps) < 2:
            return 0.0
        lo, hi = self._steps[0], self._steps[-1]
        if hi == lo:
            return 0.0
        return -1.0 + 2.0 * (step_idx - lo) / (hi - lo)

    def record(self, step_idx: int, output: torch.Tensor):
        """Record an actual-pass output and refit the Chebyshev coefficients (Eq.13)."""
        self._shape = output.shape
        self._dtype = output.dtype
        self._device = output.device
        self._steps.append(int(step_idx))
        # Store in the source dtype (e.g. fp16) to halve memory; the fit casts to fp32.
        self._H.append(output.detach().reshape(-1))
        # Sliding window: drop the oldest anchors beyond max_cache.
        if self.max_cache > 0 and len(self._H) > self.max_cache:
            drop = len(self._H) - self.max_cache
            self._steps = self._steps[drop:]
            self._H = self._H[drop:]
        self._n_anchor += 1
        self._refit()

    def _refit(self):
        K = len(self._H)
        if K == 0:
            self._coeffs = None
            return
        device = self._device
        M1 = min(self.num_basis, K)   # can't fit more basis than samples
        taus = [self._window_tau(s) for s in self._steps]
        Phi = torch.stack([_cheb_row(tau, M1, device, torch.float32) for tau in taus], dim=0)  # [K, M1]
        H = torch.stack(self._H, dim=0).float()  # [K, F] (cast to fp32 for the fit)
        # Ridge close-form: C = (ΦᵀΦ + λI)^-1 Φᵀ H, via Cholesky-backed solve.
        A = Phi.transpose(0, 1) @ Phi                       # [M1, M1]
        A = A + self.lam * torch.eye(M1, device=device, dtype=A.dtype)
        B = Phi.transpose(0, 1) @ H                          # [M1, F]
        try:
            self._coeffs = torch.linalg.solve(A, B)          # [M1, F]
        except Exception:
            self._coeffs = torch.linalg.lstsq(A, B).solution
        self._cur_basis = M1

    def _linear_extrap(self, step_idx: int) -> torch.Tensor:
        """Linear extrapolation from the two most recent anchors (stabilizer)."""
        if len(self._H) == 1:
            return self._H[-1]
        s0, s1 = self._steps[-2], self._steps[-1]
        h0, h1 = self._H[-2], self._H[-1]
        if s1 == s0:
            return h1
        alpha = (step_idx - s1) / (s1 - s0)
        return h1 + alpha * (h1 - h0)

    def forecast(self, step_idx: int) -> torch.Tensor:
        """Forecast the denoiser output at a skipped step (Eq.14), mixed with linear."""
        if self._coeffs is None:
            raise RuntimeError("SpectrumForecaster.forecast called before any record()")
        tau = self._window_tau(step_idx)
        phi = _cheb_row(tau, self._cur_basis, self._device, torch.float32)  # [M1]
        cheb = phi @ self._coeffs                                           # [F] float32
        # M1 -- per-step decay of the spectral mix weight. The Chebyshev term is an
        # extrapolation (tau>1) whose overshoot grows with |w|; damping w toward the
        # end of sampling keeps late (low-noise, detail-bearing) steps close to the
        # stable linear extrapolation. w_decay=0 => w_eff==w (M1 off; M2 below still runs).
        frac = step_idx / max(1, self.num_steps - 1)
        w_eff = self.w * (1.0 - frac) ** self.w_decay
        if w_eff >= 0.999:
            out = cheb
        else:
            lin = self._linear_extrap(step_idx).float()
            out = w_eff * cheb + (1.0 - w_eff) * lin
        # M2 -- overshoot clamp (shrink-only). Never let the forecast carry more energy
        # than the most recent real anchor; extrapolation overshoot injects unrenormalized
        # energy that persists as ghosting at low-noise steps. Only shrinks, never amplifies.
        if len(self._H) > 0:
            ref = self._H[-1].float().norm()
            cur = out.norm()
            if cur > 0:
                out = out * torch.clamp(ref / cur, max=1.0)
        # M2b -- delta-cap (trajectory speed limiter). Bounds how far the forecast may
        # ADVANCE past the last real anchor relative to the actually-observed trajectory
        # speed. The norm cap above is inert here (epsilon norm ~constant); the residual
        # oversaturation is time-direction overshoot -- the Chebyshev extrapolation
        # effectively predicts eps(t+delta), advancing the trajectory too fast. This caps
        # the advance distance while PRESERVING direction (unlike shrinking toward the
        # anchor, so it does not force ghosting-by-staleness).
        if self.delta_cap > 0 and len(self._H) >= 2:
            s0, s1 = self._steps[-2], self._steps[-1]
            if s1 != s0:
                v = (self._H[-1].float() - self._H[-2].float()).norm() / abs(s1 - s0)
                dist = step_idx - s1
                if dist > 0 and v > 0:
                    max_delta = float(v) * dist * self.delta_cap
                    h_last = self._H[-1].float()
                    delta = out - h_last
                    dn = delta.norm()
                    if dn > max_delta and dn > 0:
                        out = h_last + delta * (max_delta / dn)
        self._n_forecast += 1
        return out.reshape(self._shape).to(self._dtype)

    def stats(self):
        return {"anchors": self._n_anchor, "forecasts": self._n_forecast,
                "total": self.num_steps}
