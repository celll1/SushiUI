# Training diagnostics and auxiliary losses

This guide describes the shipped monitoring and auxiliary-objective boundary.
It does not predict future convergence or record unimplemented alternatives.
API defaults remain authoritative in `backend/api/param_defaults.py`.

## Convergence diagnostics

`convergence_diagnostics_enable` opts a run into periodic comparison of its
single-step clean prediction, normal sampling rollout, and dataset round trip.
`convergence_diagnostics_interval` defaults to 100 steps and is normalized by
the shared periodic-interval policy.

The trainer stores the companion images under the run's sample directory and
records these extra metrics when their required inputs exist:

- latent mean and standard-deviation error;
- rollout low-frequency power and 8-pixel periodicity;
- pixel luminance error; and
- the gap between single-step and rollout luminance error.

These are defect indicators, not a scalar model-quality or time-to-convergence
score. A stable value can coexist with semantic, spatial, or trajectory errors;
interpret the metrics with the saved images.

`ArchHandler.supplies_predicted_latent` declares whether an architecture can
provide the single-step prediction. Image architectures through SenseNova do;
MiniT2I, LTX-2.3, MiniMax-H3, and ACE-Step currently provide rollout-side
diagnostics only.

## Crop-decode auxiliary loss

`crop_decode_loss_enable` plus a positive `crop_decode_loss_weight` adds an
opt-in pixel-space reconstruction term. The implementation recovers the clean
latent, selects a latent crop, decodes it with surrounding context, and compares
it with the detached clean target through `VaeLossBank`.

| Parameter | Default | Meaning |
|---|---:|---|
| `crop_decode_loss_weight` | `0.0` | Multiplier added to the architecture's main objective. |
| `crop_decode_loss_margin_cells` | `16` | Decoder context around the scored region. |
| `crop_decode_loss_out_cells` | `32` | Scored latent-region width and height. |
| `crop_decode_loss_metric` | `lpips` | `VaeLossBank` comparison term. |
| `crop_decode_loss_snr_range` | empty | Optional inclusive SNR band. |

The raw auxiliary value is logged as `crop_decode_loss`. Where the comparison
is available, `crop_decode_grad_norm_ratio` reports auxiliary-gradient norm over
main-objective gradient norm for coefficient calibration.

Support is declared by `ArchHandler.consumes_crop_decode_loss`; callers must not
infer it from latent rank. SD1.5, SDXL, Z-Image, FLUX.2, Anima, Lens, Krea 2,
Ideogram 4, and SenseNova consume the loss. MiniT2I, LTX-2.3, MiniMax-H3, and
ACE-Step refuse it. SenseNova scores its pixel-space prediction without a VAE
and requires an output size compatible with its patch grid.

## REPA target source

REPA uses the existing pixel teacher by default. `repa_target_source` may select
the shipped `latent_stem` target only when `repa_latent_stem_path` resolves to a
compatible distilled artifact. Architecture support remains owned by
`ArchHandler.repa_tap`; the target-source choice does not widen that set.

The distilled stem reproduces its teacher artifact, not an independently
established quality improvement. Keep `pixel` as the default unless the chosen
artifact has been evaluated for the run's domain.

## Ownership

- Metric definitions: `backend/core/training/metric_registry.py`
- Diagnostic orchestration: `BaseTrainer._run_convergence_diagnostics`
- Crop decode: `backend/core/training/ops/crop_decode_loss.py`
- Architecture declarations: `backend/core/training/arch/`
- REPA latent stem: `backend/core/training/repa_latent_stem.py`
