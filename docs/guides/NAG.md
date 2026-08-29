# Normalized Attention Guidance (NAG)

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

## Mechanism (from the official MIT impl — ChenDarYen/Normalized-Attention-Guidance)
The official `NAGFluxAttnProcessor2_0` shows canonical NAG on MM-DiT WITHOUT a separate
evolving image stream (my earlier "parallel-stream" worry was wrong):
  - Carry the TEXT as a doubled batch `[positive_text; negative_text]` (both evolve via
    each block's own projections — the negative "context" evolves as the batch half).
  - The IMAGE is shared: tile it x2 for the attention, run the [pos_text+img] and
    [neg_text+img] attentions, extrapolate the IMAGE output rows via nag_guidance, then
    write the SAME NAG-guided image back into both batch halves (image stays single).
  - Text output rows keep their own pos/neg halves for the next block.
Norm is **L2** (p=2), tau default 2.5 (nag_dit.nag_guidance(..., norm_p=2)).

Extra cost ≈ one extra attention per block (NAG's "+1 attention"), on anchor steps only
when combined with Spectrum.

## CFG + NAG (based on how SDXL already does it)
SDXL's canonical CFG+NAG (custom_sampling.py + nag_processor.py):
- latent(image) batch = 2 `[uncond, cond]`; text(context) batch = 3
  `[cfg_negative, cfg_positive, nag_negative]`.
- The attention expands the query `[uncond, cond]` -> `[uncond, cond, cond]` to pair with
  the 3 texts. Results: idx0 = uncond->cfg_neg, idx1 = cond->cfg_pos, idx2 = cond->nag_neg.
- NAG on the COND query only: `A_cond = nag_guidance(idx1, idx2)`, `A_uncond = idx0`.
- Output `[A_uncond, A_cond]` (batch 2) -> standard CFG combine.

Generalize to DiT (unified batch-ratio logic in one processor):
- **distilled** (no CFG): image batch B, text batch 2B `[pos, nag_neg]`. Expand image
  B->2B (duplicate). NAG on all: `guided = nag(img_pos, img_neg)`. Image out = B.
- **CFG**: image batch 2k `[uncond, cond]`, text batch 3k `[cfg_neg, cfg_pos, nag_neg]`.
  Expand image `[uncond, cond]` -> `[uncond, cond, cond]` (3k). NAG cond-only:
  `A_cond = nag(cond->cfg_pos, cond->nag_neg)`, `A_uncond = uncond->cfg_neg`. Image out = 2k.
- Detect: `txt_b == 2*img_b` -> distilled (k=img_b); `2*txt_b == 3*img_b` -> CFG (k=img_b/2).
- One SDPA over the full (text_b) batch (each element attends within its own [text;image]),
  then slice image portions per group and reduce. Image out batch stays = img_b.
- temb (from the shared timestep) is batch 1 so it broadcasts to both the image (img_b)
  and text (txt_b) modulation.

## Port note (our Flux.2 != official Flux.1)
The official targets diffusers Flux.1 (`FluxAttnProcessor2_0`). Our repo uses
`transformer_flux2.py` (Flux.2) with `Flux2AttnProcessor` (dual-stream) +
`Flux2ParallelSelfAttnProcessor` (single-stream). Porting = adapt the tile/chunk/
extrapolate logic to Flux.2's projection + rotary + split structure, plus a wrapper that
doubles the text batch through the forward. The official demo uses the DISTILLED path
(guidance_scale=0, NAG only) — start there; CFG(non-distilled)+NAG batch interaction is a
follow-up.

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
