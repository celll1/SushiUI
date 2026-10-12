# Owner 176 full-passthrough gradient and weight-update audit

Date: 2026-10-12. Read-only CPU investigation while run 176 continued training.
No GPU context, dataset image access, run stop, parameter change or backend
restart was required. This audit does not assess sample quality.

## Sources and method

- Latest available saved full-passthrough checkpoint and named AdamW8bit state:
  step 138000.
- Legacy backup: step 136490. Apply the exact checked-in function-preserving
  conversion in memory to reconstruct the full-passthrough starting weights.
- Live `per_parameter_grad_probe.jsonl` after conversion. This stores top-k
  pre-clipping gradient norms, not every parameter's gradient.
- Current API parameters and startup/warmup logs.

Read only the latent-facing tensors from safetensors. Read optimizer state via
CPU mmap; expand the selected moments using their saved quantization maps and
256-element block scales. Unlike raw uint8 code norms, these report actual
dequantized moment values. Verify that CUDA remains uninitialized.

Input patch columns are channel-major `(c r m n)`; output rows are phase-major
`(p1 p2 t C)`. Compare source channels 0-15, newly introduced direct channels
16-31 (target channels 0-15), and inherited direct channels 32-79 (target 16-63).

## Results: new paths are receiving updates

The checkpoint is 1,510 updates after conversion, including the initial
200-update re-warmup. Both split `weight_fresh` parameters have saved optimizer
step 1510, actual LR **5e-5**, and finite nonzero first/second moments.

| Weight section | Initial L2 norm | Step-138000 L2 norm | L2 norm of change | Changed coefficients |
|---|---:|---:|---:|---:|
| Input, newly introduced target 0-15 | 0 | 0.214213 | 0.214213 | 100% |
| Output, newly introduced target 0-15 | 0 | 0.640659 | 0.640659 | 100% |
| Input, all 64 direct channels | 0.826994 | 1.024768 | 0.413194 | 99.6275% |
| Output, all 64 direct channels | 1.110400 | 1.747390 | 1.044636 | 99.7404% |

All 64 direct input/output channel norms are nonzero. No nonfinite weight or
selected optimizer moment was found. The new target-0-15 input/output weight
RMS values are 0.000591684 / 0.001769582 respectively.

The live log directly observes nonzero input `weight_fresh` gradients; the
initial read contained 197 such observations among 205 top-k log records since
conversion. At step 138530 its pre-clipping L2 gradient norm is 4.91745.
Output `weight_fresh` is not present in those top-k lists; do not interpret that
absence as a zero gradient. Its nonzero saved Adam moments and actual changes
from zero independently establish gradient/update activity over the saved
interval. This is not a full instantaneous-gradient census of the live model.
Later records continue observing nonzero input-path gradients.

For the direct input/output tensors, dequantized first-moment RMS is
1.05946e-4 / 2.30142e-5; second-moment RMS is 1.74626e-6 / 1.99614e-8.
These are moment statistics, not raw instantaneous gradient norms.

The LR=0 startup messages belong to the re-warmup anchor at 136490. Saved
optimizer LR=5e-5 at 138000 rules out remaining stuck at zero after warmup.
The `.lr_schedule.json` file is a startup snapshot and must not be mistaken
for the current live learning rate.

## Separate finding: some BF16 updates are being rounded away

Bridge weight changes over the same interval:

| Parameter | Changed coefficient fraction | L2 norm of change |
|---|---:|---:|
| Input bridge weight | 68.3941% | 0.050107 |
| Input bridge bias | 75% | 0.003475 |
| Output bridge weight | 50.3494% | 0.045690 |
| Output bridge bias | **0%** | **0** |

Output bridge bias is not disconnected: its live top-k log includes nonzero
gradient observations, its saved moments are finite and nonzero, and its
optimizer step advances to 138000. Its LR is 5e-6, but the BF16 parameter
values equal the conversion baseline exactly.

The API value of `optimizer_stochastic_rounding` is null. On this Anima route
it is not forced on; `BaseTrainer._attach_stochastic_rounding` returns for a
false-like value. The current run therefore uses ordinary BF16 rounding for
these updates, with no persistent FP32 master accumulation.

Compute a single AdamW proposal using the step-138000 bias, saved moments,
betas, epsilon, LR and weight decay, in FP32. This is a stored-moment diagnostic,
not the next observed live optimizer step. Its delta RMS is **1.38636e-6** and
maximum absolute delta is **3.12924e-6**. After BF16 conversion, **0 of 64 bias
elements change**. This supports BF16 update quantization as the explanation
for this parameter's observed stagnation. It is not a measured frozen fraction
for the complete DiT.

## Interpretation and next action

There is no observed disconnect, missing optimizer group, persistent zero LR
or zero-weight failure in the new direct IO paths. They have grown substantially
since conversion. This proves update activity, not adequate detail learning or
an appropriate globally optimal LR. Only 1,510 saved updates belong to the new
layout; the preceding 136,490 belong to the legacy layout.

The output-bias stagnation prevents declaring the entire update path healthy.
The repository already supplies stochastic rounding for AdamW8bit, including
fused update seams. Enabling it for this run is a concrete numerical remedy to
verify next; increasing the whole-body LR is not required to address rounding.
This investigation does not change that setting or claim it will cure jaggies.

[Numeric report](data/anima_owner176_new_path_learning_2026-10.json).
The private CPU probe is `local/verification/owner176_new_path_audit.py`;
its small weight snapshot and raw report remain in ignored
`local/verification/owner176_new_path_audit/`. No owner checkpoint, optimizer
tensor, image or caption is committed.
