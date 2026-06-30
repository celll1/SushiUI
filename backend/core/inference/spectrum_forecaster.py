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


def _cheb_row(tau: float, num_basis: int, device, dtype) -> torch.Tensor:
    """Chebyshev (1st kind) basis row [T0..T_{num_basis-1}] evaluated at tau in [-1,1]."""
    row = torch.empty(num_basis, device=device, dtype=dtype)
    row[0] = 1.0
    if num_basis > 1:
        row[1] = tau
    for m in range(2, num_basis):
        row[m] = 2.0 * tau * row[m - 1] - row[m - 2]
    return row


def build_anchor_schedule(num_steps: int, warmup_steps: int, window_size: int,
                          flex_window: float):
    """Decide which step indices are anchors (actual forward) vs forecast (skipped).

    Faithful to the paper's adaptive scheduler (dense early, sparse late) with a
    TaylorSeer-style warm-up: the first ``warmup_steps`` are always anchors, then the
    skip interval starts at ``window_size`` and grows, damped by ``flex_window`` (0 =
    skip the full window, 1 = never skip). The first and last steps are always anchors.

    Returns a set of anchor step indices.
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
    # skip gap there visibly degrades sharpness. Keep the last ~12% (>=2) as anchors.
    tail = max(2, int(round(0.12 * num_steps)))
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

    def __init__(self, num_steps, num_basis=4, lam=0.1, w=1.0,
                 warmup_steps=3, window_size=4, flex_window=0.75):
        self.num_steps = int(num_steps)
        self.num_basis = max(1, int(num_basis))
        self.lam = float(lam)
        self.w = float(w)
        self.anchors = build_anchor_schedule(self.num_steps, warmup_steps, window_size, flex_window)
        # caches of actual passes
        self._taus = []            # normalized g(t) in [-1,1]
        self._H = []               # flattened outputs [F]
        self._shape = None
        self._dtype = None
        self._device = None
        self._coeffs = None        # [M+1, F]
        self._n_forecast = 0
        self._n_anchor = 0

    def is_anchor(self, step_idx: int) -> bool:
        return step_idx in self.anchors

    def _g(self, step_idx: int) -> float:
        # normalized timestep i/(N-1) in [0,1] -> g(t)=2t-1 in [-1,1]
        denom = max(1, self.num_steps - 1)
        t = step_idx / denom
        return 2.0 * t - 1.0

    def record(self, step_idx: int, output: torch.Tensor):
        """Record an actual-pass output and refit the Chebyshev coefficients (Eq.13)."""
        self._shape = output.shape
        self._dtype = output.dtype
        self._device = output.device
        self._taus.append(self._g(step_idx))
        self._H.append(output.detach().reshape(-1).float())
        self._n_anchor += 1
        self._refit()

    def _refit(self):
        K = len(self._H)
        if K == 0:
            self._coeffs = None
            return
        device = self._device
        M1 = min(self.num_basis, K)   # can't fit more basis than samples
        Phi = torch.stack([_cheb_row(tau, M1, device, torch.float32) for tau in self._taus], dim=0)  # [K, M1]
        H = torch.stack(self._H, dim=0)  # [K, F] float32
        # Ridge close-form: C = (ΦᵀΦ + λI)^-1 Φᵀ H, via Cholesky-backed solve.
        A = Phi.transpose(0, 1) @ Phi                       # [M1, M1]
        A = A + self.lam * torch.eye(M1, device=device, dtype=A.dtype)
        B = Phi.transpose(0, 1) @ H                          # [M1, F]
        try:
            self._coeffs = torch.linalg.solve(A, B)          # [M1, F]
        except Exception:
            self._coeffs = torch.linalg.lstsq(A, B).solution
        self._cur_basis = M1

    def _linear_extrap(self, tau: float) -> torch.Tensor:
        """Linear extrapolation from the two most recent anchors (stabilizer)."""
        if len(self._H) == 1:
            return self._H[-1]
        t0, t1 = self._taus[-2], self._taus[-1]
        h0, h1 = self._H[-2], self._H[-1]
        if abs(t1 - t0) < 1e-8:
            return h1
        alpha = (tau - t1) / (t1 - t0)
        return h1 + alpha * (h1 - h0)

    def forecast(self, step_idx: int) -> torch.Tensor:
        """Forecast the denoiser output at a skipped step (Eq.14), mixed with linear."""
        if self._coeffs is None:
            raise RuntimeError("SpectrumForecaster.forecast called before any record()")
        tau = self._g(step_idx)
        phi = _cheb_row(tau, self._cur_basis, self._device, torch.float32)  # [M1]
        cheb = phi @ self._coeffs                                           # [F] float32
        if self.w >= 0.999:
            out = cheb
        else:
            lin = self._linear_extrap(tau)
            out = self.w * cheb + (1.0 - self.w) * lin
        self._n_forecast += 1
        return out.reshape(self._shape).to(self._dtype)

    def stats(self):
        return {"anchors": self._n_anchor, "forecasts": self._n_forecast,
                "total": self.num_steps}
