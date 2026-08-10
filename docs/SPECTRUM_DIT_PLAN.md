# Phase 1: Spectrum output mode → DiT backends

## Goal
Apply Spectrum **output mode** (black-box: forecast the per-step model output over the
denoising trajectory, skip the transformer on forecast steps) to the 6 non-SD/SDXL
backends. Block mode is U-Net-specific and stays SDXL-only.

## Why portable
Each backend has its own denoising loop but every loop produces a single per-step model
output (velocity / noise) that feeds `scheduler.step`. Forecasting that FINAL (post-CFG)
output is architecture-agnostic: skipping a step skips the whole model eval regardless of
the CFG scheme (batched / separate-pass / dual-branch / interval).

## Shared piece
`spectrum_forecaster.build_output_forecaster(params, num_steps, label)` — builds an
output-mode `SpectrumForecaster` from the existing `spectrum_*` generation params with the
standard auto-disable (needs `spectrum_enable`, `num_steps >= warmup+3`), forces output
mode (logs if block requested), defaults max_cache to a small local window. Returns None
when disabled. Reused by all 6 backends.

## Hook pattern (per loop, at the model-eval site)
```python
spectrum_skip = fc is not None and not fc.is_anchor(i)
if spectrum_skip:
    model_out = fc.forecast(i)          # skip transformer + CFG
else:
    model_out = <run transformer + CFG as before>
    if fc is not None:
        fc.record(i, model_out)         # record the final post-CFG output, refit
# ... scheduler.step(model_out) as before
```
`fc.is_anchor(i)`/`record`/`forecast` use the loop's 0-based index `i` over `len(timesteps)`.

## Order (separate commit each)
1. **Z-Image** — one shared `_zimage_denoising_loop` covers txt2img/img2img/inpaint.
2. **FLUX.2** — inline loops in `_generate_txt2img_flux2` and `_generate_img2img_flux2` (+inpaint).
3. **Anima** — `anima_pipeline_ops.sample_txt2img/img2img/inpaint`.
4. **Lens** — `lens_pipeline_ops.denoise_loop*`.
5. **Ideogram4** — `ideogram4_pipeline_ops._run_loop` (dual-branch velocity).
6. **MiniT2I** — `minit2i_pipeline_ops._euler_run` (interval CFG).
7. **MiniMax-H3** — `h3_pipeline_ops.denoise`; two forecasters share one
   anchor decision and forecast paired final video/audio velocities.

## Per-backend wiring
- The `_generate_*_<arch>` method reads `params` (already carries `spectrum_*` from SSoT) and
  either builds the forecaster (if it owns the loop count) or passes `params` into the loop
  which builds it once `len(timesteps)` is known.
- The loop takes an optional forecaster and applies the hook.

## Auto-disable / caveats
- Too few steps → disabled (warmup dominates; matches SD/SDXL).
- Ideogram4 dual-branch: forecast the final combined velocity (both transformers skipped).
- MiniT2I interval CFG: forecast the final per-step velocity; the interval only changes how
  the recorded output was produced (baked in) — the local window adapts across the boundary.
- CFG-truncation / guidance-vector transitions: local windowed fit adapts.
- MiniMax-H3 block swap: disabled for that request because a forecast skips the
  block offloader's wait/submit rotation.

## Approximate-quality gate
Same-seed LPIPS/SSIM measure trajectory distance and remain useful diagnostics,
but do not alone decide whether an approximate generation is usable. A release
arm must first have no black/non-finite frames, subject loss, unexpected
duplication, topology collapse, unrequested cut/freeze, conditioning violation
or audio discontinuity. Surviving clips are reviewed blind for prompt adherence,
subject consistency, temporal coherence and practical usability, and accepted
only when that non-inferiority result forms a useful Pareto point with the
measured denoise speedup. Exact/reproducible generation keeps Spectrum off.

## Verification (per commit)
- py_compile + real import of core.pipeline.
- Numeric: forecaster already unit-tested; here confirm the hook wiring (shapes, is_anchor).
- User runtime test: 1 image per backend with spectrum_enable, confirm `[Spectrum] enabled`
  log and no shape errors; compare quality/speed.

## Out of scope (later phases)
- Phase 2: NAG on DiT (per-model joint-attention hooks).
- Phase 3: NegPip on DiT (per-model signed-V + CFG-batch alignment).
- DiT block-level caching (TeaCache/FORA analog) — not the Spectrum U-Net block mode.
