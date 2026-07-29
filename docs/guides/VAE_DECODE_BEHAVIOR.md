# VAE Decode Behavior

What this repo's VAE decoders measurably do, and the two inference-side options
that act on it. Written so a future maintainer does not have to re-derive any of
it — several numbers here overturned an earlier internal record precisely
because that record had lost its provenance.

> **Provenance for every measurement below.** Measured 2026-07-28,
> inference-only, `torch.no_grad()`, **fp32 unless a dtype is stated**, single
> GPU, no backend running, no training. Four VAEs: two independent SDXL
> `AutoencoderKL` checkpoints, `AutoencoderKLQwenImage` (Anima/Krea2), and the
> 16-channel FLUX.1 `AutoencoderKL` (Z-Image). **Sample sizes are n = 1–3
> images per cell** except where a larger n is stated. The end-to-end
> verification of `vae_tile_global_norm` was run against a live backend in
> fp16, 6 arms × 3 repeats. Working notes were untracked; numbers are restated
> here rather than linked. Ratios and orderings are stable across VAEs and
> stimuli; a third significant figure of any single cell is not defensible.

## The two options

Both are generation parameters in `GENERATION_DEFAULTS`, both off/neutral by
default, both wired into `PipelineManager._apply_vae_tiling`
(`backend/core/pipeline.py:1506`). Neither has any effect unless the decode
actually tiles (`vae_tiling: true` and the image above `vae_tile_threshold`).

### `vae_tile_mode`: `"blend"` (default) | `"context"`

- **`blend`** — diffusers' own `tiled_decode`: overlapping tiles, linear
  cross-fade of the overlap band. Unchanged historical behavior.
- **`context`** — `backend/core/inference/context_tiled_decode.py`. Each output
  tile is decoded together with a margin of **real neighbouring latent cells**
  which is then cropped off. Tiles join with no blend at all. `vae_tile_threshold`
  is interpreted as the **decode-area budget** in this mode (output tile =
  threshold − 2·margin), so peak memory tracks the threshold the user set. When
  the budget cannot hold a useful margin, the **margin** shrinks — never the
  decode window — and if it floors, the decode is handed back to blend rather
  than tiled with a margin too short to do its job.

**Why the default stays `blend`, deliberately.** Measuring the join band against
the tile interior for both modes shows the correctness win is currently
invisible:

- **SDXL family** — blend has no coherent join line to remove (join/interior
  1.02–1.07, and its peak join-line row-mean never exceeds the peak interior
  row-mean). Its real tiling artifact is a per-tile *tint*, which a join/interior
  split is structurally blind to — and blend is *better* at it, because a 25%
  cross-fade ramps the tint step instead of stepping it.
- **Qwen family** — blend does leave a coherent line (join/interior 1.30–1.51,
  peak join-line row-mean 2.5–2.6× context's) and context removes it — but its
  amplitude is **0.10–0.23 /255**, 4–10× below one 8-bit level.

The one measured reason to opt in today is **memory**: a lower decode peak at the
same threshold (2.67 vs 3.35 GB SDXL, 5.73 vs 7.28 GB Qwen-family at threshold
1024), because diffusers retains every decoded tile before blending while this
path writes each tile into the canvas and frees it. The cost is more decoder
calls at small thresholds (36 vs 16 at threshold 512); break-even is around
threshold 1024.

### `vae_tile_global_norm`: bool, default `false`

`backend/core/inference/global_group_norm.py`. When a decode is actually tiled,
the decode runs **twice**: pass 1 records each decoder `GroupNorm`'s per-group
sum/sumsq/count across whatever tiles the decode produces internally; pass 2
re-decodes forcing those accumulated whole-image statistics. It wraps the whole
decode and never needs to know the tile geometry, so it composes with either
`vae_tile_mode`.

Measured per-tile tint peak-to-peak, flag off → on, SDXL, n = 1 image:

| dtype | blend @512 | context @512 |
|---|---|---|
| fp32 | 1.321 → 0.037 (35.5×) | 1.799 → 0.085 |
| fp16 | 1.347 → 0.038 (35.2×) | 1.802 → 0.095 |
| bf16 | 1.361 → 0.183 (**7.4×**) | 1.890 → 0.198 |

Across 24 fp32 cells (3 images × 2 budgets × 2 modes × {SDXL, FLUX.1}) two
passes recover a **median ~91%** of the ceiling set by an exact transplant of
statistics captured from a whole-image decode, and 47–89% of the whole-image
mean gap. **24/24 cells improved; none regressed.**

Cost: exactly **2.0×** the decode wall time and **+0.00003 GB** peak VRAM. Note
the doubling applies to *every* VAE decode in the request — on SD1.5/SDXL
`_apply_vae_tiling` runs before the sampling loop, so the in-loop decodes of
`flatten_in_loop` and `vae_drift_correction` are doubled too.

It is a **silent no-op when the decoder has no `nn.GroupNorm`** (Qwen family),
gated on that rather than left to run: every arm there was bit-exact identical
to the plain decode while still costing 2× decode time (3.03 s → 6.05 s for a
byte-identical image).

## Decoder non-locality decomposes into exactly three terms

This is the load-bearing result. An independently decoded latent crop disagrees
with the corresponding region of a whole-image decode; that disagreement is the
sum of three mechanisms with different fixes.

**(a) Finite receptive field + zero padding.** Boundary-local. The crop's
outermost pixels are simply wrong: **12–25 /255** at the boundary pixel
(individual pixels peaking at 71–218 /255), decaying to a floor by d ≈ 32–48 px.
Supplying real neighbouring context and discarding it **extinguishes this term
exactly at 14–16 latent cells (112–128 px)** — measured 0.0000 /255 once (b) and
(c) are ablated away. The theoretical receptive field is ~17.25 cells / 138 px;
the effective figure is ~2.5× smaller than the theoretical bound would suggest
for the practical threshold, and 14–16 is the exact extinction point. The
shipped `context` margin default is 16.

**(b) GroupNorm spatial statistics.** Whole-tile, not boundary-local — a
per-tile **tint**, up to **1.8 /255 peak-to-peak** at a 512 px tile on SDXL.
Present **30×** in every decoder of the SDXL family (`AutoencoderKL` and
`AutoencoderKLFlux2`: SD1.5, SDXL, Z-Image, Lens, Ideogram4, FLUX.2);
**absent — literally zero — in `AutoencoderKLQwenImage`** (Anima/Krea2), which
uses RMSNorm over channels. Transplanting whole-image statistics removes 42–50%
of the deep-interior floor on SDXL and **89%** on FLUX.1. It does *not* reduce
the boundary error at zero context: (a) and (b) are independent levers and both
are needed.

**(c) Mid-block global self-attention.** Exactly one per decoder on all four
VAEs. **0.05–0.39 /255** and irreducible: a crop's K/V is a strict subset of the
full map's, so no padding or statistic transplant can make it exact. It is not
*functionally* negligible — bypassing it changes the full decode by 0.96–15.94
/255 — it just does nearly the same work for a crop as for the full map.

**The closure test is what makes this attribution trustworthy.** With GroupNorm
statistics transplanted *and* the mid-block attention identity-bypassed, a
padded crop decode equals the full decode **bit-exactly** (`e_floor = 0.0000`
from an fp32 mean over 147k pixels, at every margin including zero). On the
Qwen family, which has no (b), bypassing the attention alone reaches 0.0000 with
*zero* padding. That closure is why the decomposition can be called exhaustive
rather than merely plausible.

Deep-interior floor by family (mean |Δ| beyond 192 px from the crop border,
per-image): SDXL 0.41 / 0.67 / 1.12; FLUX.1 0.43 / 0.67; Qwen 0.052 / 0.114 /
0.071 — i.e. GroupNorm-bearing decoders are **6–16×** the GroupNorm-free one.

**The whole thing is small in absolute terms.** With both practical levers
applied (≥16 cells of context + global GroupNorm statistics), the residual is
**0.03–0.16 /255** on the SDXL family and 0.02 on Qwen — 6–50× under a 1/255
visibility bar. That is precisely why the training-side crop-consistency loss
was **not** built (see `docs/guides/VAE_TRAINING.md`).

## Implementation constraints that are load-bearing, not style

These come from measurement and re-litigating them will cost more than reading
them. All are documented in-code in `global_group_norm.py`.

1. **Exactly two passes. Never iterate.** The fixed-point iteration **diverges**:
   a third pass is worse than the second in 5 of 6 measured SDXL cells, and a
   fourth reaches mean **14.5 / max 358 per 255**. The map "decode under
   statistics S, re-accumulate" has a fixed point, and it is *not* the
   whole-image statistics — the union of tile activations is not the whole-image
   activation field, so iterating walks toward a wrong attractor. On FLUX.1 the
   iteration is stable but plateaus 10–17× above the exact transplant. Ruled out
   as the mechanism (MEASURED): it is *not* blend's overlapping tiles
   double-counting — `context` has zero overlap and diverges just as hard. An
   `iterations` parameter would be a footgun, not a quality dial; the divergence
   table is kept as a comment block next to the code.
2. **Fold the forced statistics into `F.group_norm`'s own per-channel weight and
   bias** (`w' = w·sd_t/sd_g`, `b' = b + w·(mu_t−mu_g)/sd_g`) and call the fused
   kernel once. Writing the normalisation as explicit elementwise ops measured
   **+1.76 GB** peak (3.32 → 5.08) from unfused transients — more memory than
   tiling saves. `addcmul`, `mul().add_()` and `empty_like` were all equally bad.
   The identity check (forcing a decode's own statistics) then passes at
   1e-4 /255, three orders of magnitude below the smallest effect measured, with
   `w'` deviating from `w` by ≤6.7e-7 (≈5.6 fp32 ulps).
3. **Reduce with `mean(dtype=fp32)` + `linalg.vector_norm(dtype=fp32)`,
   accumulate in fp32/fp64, cast only the finished `w'`/`b'`.** Measured
   alternatives: `.float()` first costs +512 MB; `(x*x).mean()` in fp16 returns
   `inf` on a constant-300 tensor (SDXL VAEs are fp16-marginal); `var_mean`
   returns fp16 and quantises the very statistic being corrected; a float64
   variant costs +1024 MB. `var = E[x²] − mu²` can cancel, so its conditioning
   was measured on the real decoder rather than assumed: `max(mu²/var) = 7.8`
   over every GroupNorm call of a tiled SDXL decode, identical across dtypes.
4. **Two-pass, not layer-interleaved.** Exact global statistics (the
   multidiffusion Tiled-VAE approach) require every tile's layer-L activations
   resident at once, returning peak memory to whole-image scale — which defeats
   the reason tiling exists. Two-pass retains ~23–30 KB of scalars.
5. **Install hooks inside a `try`, restore in `finally`, restoring the exact
   prior state.** A leaked GroupNorm hook corrupts every later decode in the
   process — far worse than the artifact being fixed.
6. **Split a `B>1` latent and recurse per sample.** Slicing does not make
   batching safe: it pools every image's moments into one entry and normalises
   the images to each other. Sensitivity-tested — with the split disabled the
   check reports 36.0 / 88.6 per 255 per-sample deviation; with it, exactly 0.

### The dtype lesson

**bf16 is materially weaker than fp16/fp32 for this correction** (7.4× vs 35×
tint reduction). The statistics are fine; the correction can only be applied
through the normalisation's weight/bias in the activation dtype, and bf16's 8
mantissa bits quantise it at ~4e-3 relative. That is the floor, and it is stated
in the API description and the UI because several architectures decode in bf16.

This has a second, sharper form. A 24-cell measurement probe, the implementer's
production-path checks, and a code audit that correctly validated the fold
algebra to 1.9e-6 **all agreed the feature worked — and it could not execute at
all**: every one of them ran fp32 while production runs fp16, so `F.group_norm`
rejected the fp32 folded weights against `Half` activations. Only running it
against the live backend found that. See `docs/guides/ADD_A_PARAMETER.md`.

## Facts about what the decoders do, independent of tiling

These close questions that were open (or wrongly recorded) before, and set the
realistic ceiling for any decoder work.

**The decoder does invent high frequency.** `invented_share` (the fraction of
output HF surviving heavy latent smoothing) is **not** ≈0: SDXL **0.13–0.25**,
Qwen **0.04–0.09**, FLUX.1 **0.013–0.022** (n = 3 encoded real images per VAE,
plus 2 real generated SDXL latents). A spectral control that measures only
wavelengths the content cannot occupy shows a 28× / 23× / 6× excess over the
content ceiling at σ = 8 cells, with an assumption-free floor at σ = 64 of
**0.40 / 0.42 / 0.17 /255**. The sharpest single signature: **blurring an SDXL
latent can *raise* output HF by +37%** — a faithful decoder cannot produce more
detail from less latent detail. In a flat window of a real generated image the
fine-scale energy stays at ≈**1.2 /255** while the window's content collapses
13×.

The honest magnitude: **1–3 /255 RMS at fine scales**, i.e. visible only under
an 8–16× brightness lift. A counter-arm went the other way and is kept for that
reason: against a ground truth, an encode→decode roundtrip *removes* 10–25% of
shadow HF (`shadow_hf_excess` 0.68–0.90 in all 9 cases). Both are true — in
generation there is no ground truth to lose, so the first regime is what a user
sees. Unexpectedly, **FLUX.1's decoder is the quietest** on this axis (≈1/10 of
SDXL, 1/3–1/6 of Qwen) despite identical topology and the same 30 GroupNorms —
so invented HF is a *weights* property, not a topology property (n = 1 for the
spectral control; indicative, not settled).

**The 8 px grid artifact does not exist at a measurable level.** Ratio ≈1.0 on
four VAEs under three independent metric definitions, on synthetic, real and
generated-latent inputs. This is a **non-reproduction** of an earlier internal
record of 3.0–5.5 — not a refutation, since the original raw artifacts and
metric definition are gone. What *did* reproduce from that record is SDXL's edge
residual being **3.7×** Qwen's (9.44 vs 2.54 /255), which is broadband
softness/ringing, not grid structure.

**SDXL converts latent noise into pixel HF ~70× more efficiently than Qwen**
(and ~3× more than FLUX.1) at ε = 0.10 of latent std. This is why latent-side
mottle *reads* worse through the SDXL decoder: the same latent perturbation buys
much more visible pixel HF.

**A constant latent decodes to an exactly flat interior** (std 0.000) on all four
VAEs — but only once the canvas border is excluded. There is a **64–128 px
zero-padding border band** the prior record does not mention; measuring the whole
frame makes that control look like a 0.9-level failure. All measurements here
inset by ≥48 px.

## The outpaint seam is not a VAE problem

Worth stating explicitly, because a large measured number points the wrong way
if read without the mechanism.

The VAE's own isolated contribution at an outpaint preserved/generated boundary
is **23.5 /255 at the boundary row** (MEASURED, **n = 9 real outpaint runs**,
range 13.8–31.3; a conservative edge-replicate arm gives 16.2), staying above
4.8 /255 for the first 8 px and crossing 2 /255 only at d ≈ 16 px.

**This is not fixable by VAE work.** The whole canvas is decoded in a **single**
`vae.decode` on the whole-canvas latent — nothing is context-starved; the
generated side already has the preserved latent as context and vice versa. And
`vae_tiling` defaults to `false`. The preserved pixels are a byte-exact paste of
fixed input data, produced in a world where the new outpainted surroundings did
not exist; no decode path can retroactively give them that context. Corroborated
by the repo's own A/B: `outpaint_preserve_mode=vae_reconstruct` (decode the whole
canvas, do not paste) measures crossing 0.635 ≈ background, while the byte-exact
paste measures 2.03.

So the 23.5 /255 figure is the **quantitative explanation of why a byte-exact
paste must step**, measured for the first time — not a defect awaiting a fix.
It also independently validates a shipped default: `outpaint_paste_feather_px`
= 24 was set empirically, and 24 px is exactly where the boundary-localised term
merges into the feather-immune floor (16 px would terminate on a ~2/255 step
with no margin; 32 px buys 0.45 /255; beyond 32 px buys nothing, because the
remaining 0.70 /255 is a whole-region tint a feather cannot remove — that is
what `outpaint_seam_fix` / `match_generated_exposure` already harmonises).
**Keep 24.**

A tiled outpaint decode *is* helped by these features, but only if the user
turns `vae_tiling` on (off by default; it would engage on 7 of the 9 sampled
canvases at the auto threshold), and the benefit is a ~5× cut of a **sub-1/255**
tiling penalty (0.31–0.77 → 0.06–0.17 mean |Δ|).

## Limits on these numbers

- **Small n.** Most cells are n = 1–3 images, one crop, one seed, one GPU. The
  outpaint arm is the best powered at n = 9 runs (6 seeds, 3 canvas geometries —
  not 9 independent scenes). `gn4` (the fourth-pass divergence figure) is n = 1.
- **The ablations are not orthogonal.** Bypassing the attention changes the
  activations feeding all 30 GroupNorms, so the attention-only ablation *raises*
  the SDXL floor (0.41 → 0.62). Attribution is always phrased as "what remains
  after removing X", never "X = A − B"; the closure test is what licenses the
  decomposition.
- **Only `sdxl` / `qwen` / `flux1` (+ a second SDXL checkpoint) were measured.**
  SD1.5, Lens, Ideogram4, MiniT2I and FLUX.2 are covered by inference from
  topology, not by measurement.
- **FLUX.2's packed latent is untested.** `AutoencoderKLFlux2.decode` receives
  the natural 8× grid (patchify/BatchNorm live outside `decode`), so 16 cells
  should be the right unit; `context_tiled_decode.py` carries a runtime
  scale-mismatch guard that falls back rather than writing a wrong canvas.
- **The outpaint measurement is counterfactual** (with-neighbour vs
  without-neighbour decode); no ground-truth original exists for those runs.
- **LTX-2 (video), ACE-Step (audio) and PiD are out of scope** and were not
  measured.
- **The earlier records that these numbers contradict cannot be re-derived** —
  their raw artifacts and exact metric definitions are gone. Treat those as
  non-reproductions, not as proofs of error, and do not let the numbers *here*
  lose their conditions the same way.
</content>
</invoke>
