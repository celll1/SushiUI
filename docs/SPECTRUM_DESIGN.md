# Spectrum (Adaptive Spectral Feature Forecasting) — Design

Paper: arXiv 2603.01623 "spectral diffusion feature forecaster (Spectrum)".
Training-free inference accelerator. Treats the denoiser output as a function of
time, fits Chebyshev coefficients by ridge regression over actual-pass outputs, and
forecasts the output at skipped steps — no extra forward.

## Math (verbatim from paper)
- Time map: `g(t) = 2t - 1`, `[0,1] -> [-1,1]` (t = normalized timestep i/(N-1)).
- Chebyshev (1st kind): `T0=1, T1=τ, Tm = 2τ·T(m-1) - T(m-2)`.
- Basis row: `φ(τ) = [T0(τ),...,TM(τ)]  ∈ R^{M+1}`  (Eq.9). M+1 = number of basis = `spectrum_m`.
- Design matrix `Φ = [φ(g(t_k))]_{k}  ∈ R^{K×(M+1)}` over cached anchors k (Eq.10).
- Feature matrix `H = [h_{t_k}]  ∈ R^{K×F}`, F = flattened network output dim (Eq.11).
- Ridge close-form (Eq.13): `C = (ΦᵀΦ + λI)^{-1} Φᵀ H ∈ R^{(M+1)×F}`, λ = `spectrum_lam`.
  Solved by Cholesky. Inversion is (M+1)×(M+1) — negligible vs F.
- Forecast (Eq.14): `h_{t_j} = φ(g(t_j)) · C`.

## Forecast target (black-box, v1)
`h` = the RAW U-Net output `noise_pred` BEFORE CFG combine (batch [2,...] for CFG,
[3,...] for NAG; NAG/NegPip/ControlNet effects are baked into the recorded anchor
output, so skipped steps inherit them). Downstream CFG combine + scheduler.step are
unchanged. DEUS 2-pass path is excluded in v1 (separate uncond/text tensors).

## Anchor / skip schedule (U = actual pass, V = forecast)
Paper adaptive scheduler: `U = {⌊α·r(r+1)/2⌋}` — dense early, sparse late — plus a
TaylorSeer-style warm-up W (full eval every step at the start) and an initial
interval that grows. Param mapping (impl naming from the reference node):
- `spectrum_warmup_steps` (W): leading steps that are always actual passes (build cache).
- `spectrum_window_size`: initial skip interval; subsequent intervals grow.
- `spectrum_flex_window` ∈ [0,1]: damps the actual skip count. 0 = skip up to the full
  window; 0.75 = conservative (fewer skips, higher quality).
- The FIRST step and the LAST step are always anchors.
- A step is forced to be an anchor when conditioning changes (prompt-editing embeds,
  ControlNet on/off boundary) — v1 simply disables Spectrum if prompt-editing or
  ControlNet is active.

## Output mixing (spectrum_w)
`out = w · cheb_forecast + (1-w) · linear`, where `linear` is linear extrapolation
from the two most recent anchors (stabilizer). w=1.0 → pure spectral. Recommended 0.5–1.0.

## Params (SSoT: param_defaults GENERATION_DEFAULTS)
| param | default | note |
|---|---|---|
| spectrum_enable | False | master toggle |
| spectrum_w | 1.0 | spectral/linear mix (1.0 = spectral only) |
| spectrum_m | 4 | number of Chebyshev basis (M+1) |
| spectrum_lam | 0.1 | ridge λ |
| spectrum_warmup_steps | 3 | leading full-eval steps |
| spectrum_window_size | 4 | initial skip interval |
| spectrum_flex_window | 0.75 | skip damping (0 = max skip) |

## Scope
- v1: SDXL `custom_sampling_loop` (txt2img), standard CFG/NAG path. Little benefit on
  low-step/distilled (warmup dominates) — auto-disable when N < warmup + a few.
- v2: img2img/inpaint, then DiT loops (Z-Image/FLUX.2/...).

## Integration point
`custom_sampling.py` "Predict noise residual" block (~937): wrap so anchor steps run
`unet(...)` + `forecaster.record(...)`, skip steps call `forecaster.forecast(...)`.

## Risks
- Prompt editing / ControlNet per-step changes -> auto-disable v1.
- VRAM: cache K×F output tensors (SDXL 1024²: ~2.6MB total, negligible).
- Schedulers: scheduler.step still runs each step; only the forward is skipped.
