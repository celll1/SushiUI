# Phase 2: NAG → DiT backends

## Goal
Apply NAG (Normalized Attention Guidance) to the 6 DiT backends. NAG extrapolates the
image-token attention output away from a NAG-negative text context, in attention-OUTPUT
space (not the final noise, which is CFG). Per-architecture, because each model vendors
its own attention.

## Shared core (done)
`core/inference/nag_dit.py`:
- `nag_guidance(z_pos, z_neg, scale, tau, alpha, feature_dim=-1)` — the model-agnostic
  guidance math (identical to the SDXL cross-attention NAG; unit-tested to match).
- `nag_active(nag_enable, nag_scale, nag_neg_embeds)` — gating.

Each per-model hook only has to produce z_pos (image queries vs positive text) and z_neg
(image queries vs NAG-negative text), then call `nag_guidance`.

## NAG × Spectrum (orthogonal, stackable)
On anchor steps the NAG-modified output is recorded; on Spectrum skip steps it is
forecast. NAG's extra attention cost is paid only on anchor steps. No special handling —
Spectrum records/forecasts whatever the (NAG-modified) forward produces.

## Per-model interception (from the code survey)
| Model | Attention | Text tokens | Hook |
|---|---|---|---|
| FLUX.2 | diffusers `Flux2AttnProcessor` (dual) + `Flux2ParallelSelfAttnProcessor` (single) | PREFIX `[0:num_txt]` | processor swap (set_processor); pass nag-neg text via joint_attention_kwargs |
| Z-Image | `ZImageAttention.forward` (vendored, no processor) | SUFFIX `[img_len:img_len+cap_len]` | edit/subclass forward; slice text K/V |
| Anima | vendored attention (`anima_attention.py`) | joint context | edit attention forward or block |
| Lens | `LensTransformer2DModel` (vendored) | joint | edit attention forward |
| Ideogram4 | vendored transformer (dual-branch) | packed [text][image] | edit attention forward |
| MiniT2I | mmjit transformer | joint | edit attention forward |

## Mechanism (per model)
When NAG active, run each attention block twice for the IMAGE query rows:
  - z_pos = attn(img_q, keys/values from [positive text (+ image)])  ← the normal output
  - z_neg = attn(img_q, keys/values from [nag-negative text (+ image)])
  - image output rows := nag_guidance(z_pos, z_neg, scale, tau, alpha)
  - text output rows unchanged (positive)
Requires threading nag_negative_prompt_embeds into the transformer forward so the block
can compute the nag-negative text K/V. Extra cost ≈ one extra text→image attention per
block (NAG's "+1 attention", on anchor steps only when combined with Spectrum).

## Order (separate commit each; runtime-tested by the user between models)
0. Shared core + plan (this).
1. FLUX.2 — processor swap (cleanest; text is a known-length prefix).
2. Z-Image — forward edit/subclass (text is a known-length suffix).
3. Anima → 4. Lens → 5. Ideogram4 → 6. MiniT2I.

## SSoT / params
Reuse the existing nag_* generation params (nag_enable/scale/tau/alpha/sigma_end/
nag_negative_prompt). Each `_generate_*` encodes the nag-negative prompt (like SDXL) and
threads it into the loop/transformer.

## Risk / verification
- nag_guidance math: unit-tested (matches SDXL).
- Per-model integration: NOT runtime-verifiable here (backend is user-run). Each model
  is a separate commit; user tests 1 image with nag_enable before the next model.
- Core inference path → per-model regression risk; keep NAG strictly gated behind
  nag_enable so default behaviour is unchanged.
