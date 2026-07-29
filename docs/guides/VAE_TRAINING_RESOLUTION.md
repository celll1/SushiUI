# VAE Training: Crop Geometry and Resolution Scaling

What the `vae_decoder` crop policy actually feeds the decoder, and what raising
`resolution` costs. Two measurement campaigns are distilled here because their
working notes are untracked and this repo has already had to retract two numbers
whose provenance was lost. Every claim below carries its sample size and its
conditions; anything not measured is labelled INFERRED.

This doc is the **training-side** companion to
`docs/guides/VAE_DECODE_BEHAVIOR.md` (inference-side decoder facts) and to
`docs/guides/VAE_TRAINING.md` (config surface, loss defaults, refusals, export).
Config keys and their current defaults live in `backend/api/param_defaults.py`
(`VAE_TRAINING_DEFAULTS`) and are described in `VAE_TRAINING.md`; they are
deliberately not restated here.

> **Provenance.**
> **Campaign A — crop geometry.** Inference-only, `torch.no_grad()`, fp32, one
> VAE resident at a time (peak 7.42 GiB), no training, no backend call. Run
> against run 113's own init checkpoint (`vae_dec_IL02_v1`, an SDXL-family
> 4-channel `AutoencoderKL`) and its own 22 datasets, after that run had been
> stopped by its owner at step 52,527. The training-distribution arm calls the
> production loader (`vae_dataset.load_image_tensor`) read-only, so it is the
> real crop policy rather than a re-implementation.
> **Campaign B — memory/resolution scaling.** 2026-07-30, RTX 6000 Ada
> (47.99 GB, WDDM), bf16 autocast over fp32 master, AdamW 1e-5,
> `decoder_blocks: all` (49.49 M trainable), `mse 1.0 + lpips 1.0 (VGG) +
> ycbcr_dc 0.1`, EMA on, real images from the same 22 datasets. Every arm drives
> the production `VaeTrainer._train_micro_step` verbatim. Four early arms were
> timed under GPU contention from another process and were **discarded**; every
> number here was measured with **46.1 GB free**. The 512 control reproduced a
> prior benchmark to 2 % (0.1704 vs 0.1732 s/image; peak 6.697 vs 6.70 GB).
> Neither campaign modified `backend/`, started/stopped a run, or restarted the
> backend.

## 1. What the loader does, and what the corpus can supply

How much an image is resampled before its square crop is taken is decided by
`backend/core/training/vae/vae_dataset.py::resolve_crop_scale`, whose allowed
values are the enum `vae_config.py::VALID_CROP_SCALE_POLICIES`
(`downscale` / `native` / `mixed`); the crop itself (random offset for training,
centred for validation) is in `load_image_tensor`. Nothing is ever squashed, and
there is no bucketing under any policy.

Everything measured below was measured against the **`downscale`** policy, which
is the historical behaviour and the current default: `scale = resolution /
min(w, h)` applied **whether it is above or below 1**, so a large image is
LANCZOS-**downscaled** until its short side equals `resolution`, and only then
cropped. The current default and the semantics of each value live in
`VAE_TRAINING_DEFAULTS` / `docs/guides/VAE_TRAINING.md` and are not restated
here. An image whose short side is below `resolution` is upscaled under every
policy, because there is no `resolution`-sized window to crop otherwise.

Over the 22 datasets configured for run 113 — **3,842,897 items, every one with
recorded width and height** (read-only query against `datasets.db`, MEASURED):

| statistic | value |
|---|---|
| short side: median / mean | **1131 px** / 1285 px |
| short side p05 / p25 / p75 / p95 | 536 / 783 / 1528 / 2732 px |
| short side ≥ 512 (i.e. **downscaled** before cropping) | **95.79 %** |
| short side < 512 (upscaled) | **4.21 %** |
| downscale factor among the 3,681,110 downscaled items: median / mean / p75 / p95 | **2.30×** / 2.58× / 3.00× / 5.42× |
| aspect ratio: median / mean | **1.412** / 1.434 |
| aspect ≥ 1.5 / ≥ 2.0 / ≥ 3.0 | 25.24 % / 3.46 % / 0.68 % |
| source area covered by one square crop (1/aspect): mean / median | 72.7 % / 70.8 % |

So at `resolution: 512` the decoder is trained almost entirely on
LANCZOS-downscaled content, typically a **2-3× reduction**, and one crop leaves
~27 % of each source reachable only through the random-crop offset across epochs.

An independent 6,000-image sample of the same pool (header reads only, 0
unreadable, MEASURED) gives the fraction that a *native* crop at each size could
take without any upscaling — median short side 1122 px:

| native crop target | 512 | 768 | 1024 | 1280 | 1536 | 2048 | 2560 |
|---|---|---|---|---|---|---|---|
| fraction of corpus with short side ≥ target | 95.9 % | 76.9 % | 56.5 % | 40.6 % | 23.8 % | **11.8 %** | 5.7 % |

**The corpus cannot feed 2048.** A fixed 2048 square crop *upscales* 88 % of the
data, teaching the decoder to reproduce LANCZOS ringing. The honest fixed-size
native options are 1024 (56 % native) or 1280 (41 %).

For comparison, what inference asks of the decoder (last 3,000 rows of
`gallery.db`, MEASURED): median generated short side **960 px**, median latent
token count **20,736 = 5.06×** the 4,096 tokens of a 512 crop, p95 11.3×,
**97.4 %** of generations above 2×, median aspect 1.48, square (<1.05) only
11.1 %.

## 2. The intuitive hypothesis is INVERTED: downscaling concentrates high frequency

The plausible premise — "the decoder never sees native-resolution edges, so it
is calibrated to blunted content" — was tested directly and is **wrong**.
Record this prominently: it is the kind of argument that gets re-proposed.

`n = 300` source images with short side ≥ 1024, sampled across the 22 datasets.
Two 512×512 **centre crops of the same image**: `native` (native pixels, no
resampling) vs `prod` (the production loader's output; median downscale factor
for this pool 2.89×). Pure CPU/NumPy/PIL — **no model is involved**. Median of
the per-image ratio; `t` on log-ratios. MEASURED.

| statistic | prod / native (median) | images where prod > native | t |
|---|---|---|---|
| mean Sobel gradient | **1.674×** | 90.3 % | +19.8 |
| RMS Laplacian | 1.722× | 93.0 % | +19.9 |
| luma std (contrast control) | 1.061× | 62.3 % | +7.9 |
| band 0.10–0.18 c/px | 1.870× | 85.3 % | +15.0 |
| band 0.18–0.28 | 2.631× | 92.7 % | +18.8 |
| band 0.28–0.38 | 3.705× | 93.3 % | +20.6 |
| **band 0.38–0.50 (top octave)** | **4.060×** (+306 %) | **93.3 %** | **+21.6** |

A LANCZOS downscale does not remove high frequency — it **concentrates** it:
the same structures are packed into 2.9× fewer pixels, so everything moves up in
normalised frequency. The excess grows monotonically with frequency. **The
training distribution is HF-*enriched* relative to inference, not starved**, by
roughly 4× in the band a decoder fine-tune is most likely to attenuate. This is
the strongest leg of Campaign A and it does not depend on any VAE measurement.

## 3. Dose-response: how much a fine-tune softens tracks the downscale factor

`n = 19` source images, all with short side ≥ 2048 so every arm is real native
pixel data — **16 `anime`, 3 `comic`, 0 `photo`** (see [Limitations](#limitations)).
Five 512×512 inputs per image differing **only** in how much LANCZOS downscaling
produced them; the encoder is frozen, so each input is encoded once and
re-decoded through base and the step-52,361 EMA decoder, making every comparison
exactly paired. Medians. MEASURED.

| arm | downscale | base PSNR | ΔPSNR (ft−base) | **edge softening (Δsobel_ratio)** | t | images softened | Δ top-octave power |
|---|---|---|---|---|---|---|---|
| `native512` | **1× (none)** | 36.31 | +1.202 | **−0.44 %** | −1.83 | **63 %** | −16.0 % |
| `ds2` | 2× | 32.96 | +1.007 | −0.87 % | −2.91 | 68 % | −25.0 % |
| `ds3` | 3× | 29.48 | +1.136 | −1.29 % | −3.42 | 84 % | −22.8 % |
| `ds4` | 4× | 28.97 | +1.176 | −3.16 % | −4.33 | 89 % | −33.1 % |
| `dstrain` (the real loader) | ≥4× | 28.02 | +1.163 | **−3.79 %** | −5.97 | **100 %** | −37.4 % |

Monotone across all five arms with no inversions. Paired
difference-of-differences on the identical source image, `native512` vs
`dstrain`: Δsobel_ratio **+3.20 %** (t = +2.36, positive on 89 % of 19),
Δfine0_ratio +2.19 % (t = +3.22; on native content that metric shows a very
slight *sharpening*, +0.41 %), Δ top-octave +8.0 % (t = +2.48).

Two consequences, both load-bearing:

1. **Softening measurements taken on the training distribution overstate
   native-resolution generation by roughly an order of magnitude.** The
   companion edge-softness study's headline of **−4.8 % edge gradient energy**
   (n = 60, 512 px, LANCZOS-downscaled crops) is a *training-distribution*
   number; the same fine-tune measures **−0.44 %** on a native 512 crop. Treat
   "roughly 8×" as an order of magnitude, not a coefficient (n = 19). That study
   also reported the effect at ~60 % magnitude when repeated at 1024 px and read
   it as a canvas-size gradient; §4 shows canvas size on its own does nothing,
   so that gradient was a **downscale-factor artifact** (its 512 arm downscaled
   by a median ~2.3×, its 1024 arm by ~1.1×).
2. **The real measured cost of the crop policy is fidelity, not sharpness.** The
   fine-tune's accuracy gain is ~**30 % smaller on native content**: edge
   residual −7.70 % (native) vs −12.48 % (training distribution), a gap positive
   on **19/19** images, t = **+7.49** — the strongest statistic in Campaign A.
   PSNR agrees (+0.81 vs +1.15 dB). The decoder is genuinely better calibrated to
   the distribution it saw, and it never saw native detail.

Dataloader cost is not a constraint on choosing a different policy (n = 120 real
images, OS cache pre-warmed identically, median per-image wall time): the
`downscale` policy costs 30.8 ms and a native crop **17.4 ms** — native is 44 %
*cheaper*, because it skips the LANCZOS resample of a multi-megapixel image while
the JPEG decode dominates either way. Every policy delivers 8-30× what the GPU
consumes (5.2 img/s at 512). MEASURED. Neither the VRAM nor the s/image figures
in §5 depend on the policy: a 512 native crop measured 6.70 GB / 0.1704 s against
6.70 GB / 0.1732 s for a downscale crop, i.e. identical within noise.

## 4. Aspect-ratio bucketing is correctly skipped — but not for the reason first recorded

`vae_dataset.py`'s module docstring used to justify skipping bucketing with "a
fixed square crop makes every batch the same shape by construction". That is a
statement about batch collation, and as a justification it is circular:
bucketing exists precisely to avoid forcing a fixed shape. The docstring now
carries the argument below instead; this section is the measurement behind it.

The defensible argument is architectural. Inspecting run 113's own init
checkpoint (MEASURED by loading it): decoder + `post_quant_conv` =
**49,490,199** parameters, **exactly one** attention module anywhere in the
decoder (`mid_block.attentions.0`, `diffusers.Attention`, **1 head**, dim 512),
and **30 GroupNorms**. Both non-local classes reduce over the spatial axes —
`AttnProcessor2_0` flattens `[B, C, H, W] → [B, H·W, C]` before attending, and
GroupNorm reduces over `(C_group, H, W)` — so they observe a latent **area**,
never a shape; everything else is convolution. A 64×64 latent and a 32×128
latent are the same problem to every layer. Aspect-ratio bucketing holds area
approximately constant by construction, so it would leave the only perceivable
variable unchanged.

**This is INFERRED from code, not measured.** The experiment that would settle
it (native crops at matched token counts, e.g. 1024×512 vs 720×720) was written
but **never ran** — the GPU was contended. What the code argument does not cover
is a second-order shape effect: a long thin canvas has more perimeter per unit
area, so the decoder's zero-pad border band covers a larger fraction of it (see
`VAE_DECODE_BEHAVIOR.md` for that band).

The *area* gap is real and large (§1: median generation = 5.06× the training
token count) — and it is **measurably harmless**. Concentric **native** crops
with no resampling anywhere, so per-pixel frequency statistics are identical
across arms and canvas size is the only variable (n = 19, same images as §3,
MEASURED):

| arm | latent grid | tokens | ΔPSNR (ft−base) | softening (median) | Δ top-octave |
|---|---|---|---|---|---|
| `n512` | 64×64 | 4,096 (= training) | **+1.202** | −0.44 % | −16.0 % |
| `n768` | 96×96 | 9,216 (2.3×) | +0.927 | −0.36 % | −12.0 % |
| `n1024` | 128×128 | 16,384 (4×) | +0.948 | −0.65 % | −11.8 % |
| `n1536` | 192×192 | 36,864 (**9×**) | +0.971 | −0.57 % | −8.0 % |

Paired `n1536` vs `n512`: Δsobel_ratio −0.324 % (**t = −1.75, n.s.**),
Δfine0_ratio −0.270 % (n.s.), Δ top-octave −0.78 % (n.s.), ΔPSNR +0.051 dB
(n.s.). The fine-tune's advantage neither shrinks nor inverts across a 9× token
range; if anything the top-octave attenuation *falls* with canvas size.

**This closes off "train at 1024 so the attention sees the right token count"**
as a motivation — nothing in the fine-tune's behaviour degrades between 4,096 and
36,864 tokens, so a resolution increase has to be justified on resampling or
field-of-view grounds instead, at the costs in §5. What genuine bucketing would
buy here is **dataset coverage** (§1: ~27 % of each source is off-crop), which is
a data argument, not an architectural one. The repo already has a
`BucketManager` (`backend/core/training/bucketing.py:200`); what it would need is
a bucket-aware batch sampler, which `VaeRawImageDataset` (a plain map-style
`Dataset`) does not have.

## 5. Memory and time scaling with `resolution`

**None of the three mitigations in §5-§7 is currently a `vae_decoder` config
option** — there is no `gradient_checkpointing` key in `VAE_TRAINING_DEFAULTS`
(the diffusion and tagger trainers have one; this trainer does not), no
activation-dispatch wiring in the VAE path, and no tiled training step. Campaign
B enabled GC by calling `vae.enable_gradient_checkpointing()` on the loaded VAE,
wrapped the unmodified micro-step in `offload_activations(...)`, and implemented
tiling in its harness. Wiring any of them in is unbuilt work; §6 records what
that costs for ActDispatch.

Fits over the measured arms (batch 1, accumulation 1, `num_workers 2`, native
crop, 8 measured steps after 3 discarded warm-up steps — 6 after 2 for the
2048/ActDispatch arms; step-time stdev 0.8-1.6 % of the median for arms that
fit, 1-5 % for spilling arms):

```
no mitigation   peak_alloc = 1.045 + 2.1553e-5 × res²  GB   (n=8, max residual 0.004 GB)
grad. ckpt      peak_alloc = 0.956 + 1.0119e-5 × res²  GB   (n=6, max residual 0.003 GB)
ActDispatch     peak_alloc = 0.992 + 3.3464e-6 × res²  GB   (n=4, max residual 0.019 GB)
GC + ActDisp    peak_alloc = 0.951 + 9.7933e-6 × res²  GB   (n=3, max residual 0.001 GB)

time, while the configuration FITS:
no mitigation   s/image ≈ 1.213e-7 × res^2.270   (n=7)
grad. ckpt      s/image ≈ 1.353e-7 × res^2.283   (n=5)
```

**Memory is exactly quadratic** (residuals under 20 MB), so extrapolation is
safe; the ~1.0 GB constant is fp32 master weights + AdamW state + EMA + LPIPS
weights and is resolution-independent. **Time is not quadratic — the exponent is
2.27.** The excess is the decoder mid-block's single self-attention, which is
O(tokens²) with tokens = (res/8)²: from 512 to 2048 the convolutions grow 16×
while the attention grows **256×**. That term is what makes §7 work.

Representative arms (peak allocated / peak reserved GB, s/image), MEASURED:

| res | strategy | peak alloc | peak reserved | s/image | fits? |
|---|---|---|---|---|---|
| 512 | none | 6.70 | 6.94 | 0.1704 | ✅ |
| 1024 | none | 23.65 | 25.41 | 0.8107 | ✅ |
| 1280 | none | 36.36 | 39.26 | 1.3707 | ✅ **last one** |
| 1536 | none | 51.89 | **56.15** | **44.2952** | ❌ spills |
| 1536 | GC | 24.83 | 31.55 | 2.5770 | ✅ **last GC one** |
| 2048 | GC | 43.40 | **55.54** | **53.7414** | ❌ spills |
| 2048 | ActDispatch (4 MB) | 15.04 | 23.73 | **195.5208** | ✅ |
| 2048 | tiled 4×4 k=19 | 14.14 | 16.65 | **6.2314** | ✅ |
| 2048 | none | *91.4 (INFERRED)* | — | *≫100 s (INFERRED)* | ❌ |

`res_2048_base` was **not executed**; 91.4 GB comes from the 8-point fit and is
labelled INFERRED throughout.

**`peak_allocated` is the wrong number to plan with near the boundary;
`peak_reserved` is the one that decides**, and it ran 22-28 % above allocated in
these arms. Both failure cases above are WDDM spills, not OOMs: 1536 without
mitigation allocates 51.9 GB and reserves 56.2 GB on a 48 GB device, and the
step goes from a fitted 2.07 s to a measured 44.30 s (**21×**); 2048 with GC
allocates 43.4 GB — under 48 — but reserves 55.5 GB and runs **11× slower than
its own compute** (53.74 s vs a fitted 4.90 s). No exception is raised. A run in
this state looks alive and delivers 67-81 images/hour.

Pixel throughput while a configuration fits is nearly flat — 1.54 Mpx/s at 512,
1.20 at 1280, i.e. 512→1280 costs only **22 %** — while crossing the VRAM
boundary costs **95 %** (1536 unmitigated: 0.053 Mpx/s). Every strategy question
here is "how do I stay under the device", not "how do I make convolutions
faster".

## 6. Gradient checkpointing and ActDispatch are SUBSTITUTES, not stackable

Gradient checkpointing buys a consistent **2.13×** memory reduction for a flat
**21-22 %** time cost (−46 % at 512, −51 % at 1024, −52 % at 1280/1536, −53 % at
2048). Largest resolution that fits with it: **1536**. It is only 2.13× because
`diffusers 0.38.0` checkpoints the decoder at *block* granularity — mid-block
plus each up-block, 5 segments
(`diffusers/models/autoencoders/vae.py:287-294`) — so the transient working set
inside the largest up-block survives, and `conv_in` / `conv_norm_out` /
`conv_act` / `conv_out` are outside the checkpointed region entirely. A
finer-grained wrapper is a design option, not something measured.

`vae.enable_gradient_checkpointing()` flags 4 modules and applies to the
**decoder**; the encoder flag is **inert by construction**, because
`backend/core/training/vae/vae_trainer.py::_train_micro_step` encodes under
`torch.no_grad()` and `Encoder.forward`'s guard is
`torch.is_grad_enabled() and self.gradient_checkpointing`. Confirmed
empirically — encoder-side checkpointing changed nothing measurable.

**The natural assumption that checkpointing should be applied regardless is
contradicted by measurement.** At 1024, MEASURED:

| 1024 px | peak alloc | s/image | bytes offloaded / micro-step |
|---|---|---|---|
| none | 23.65 | 0.8107 | — |
| GC alone | 11.57 | 0.9897 | — |
| **ActDispatch alone** | **4.48** | 6.3118 | 23.0 GB |
| GC + ActDispatch | 11.22 | 2.7130 | 6.3 GB |

Stacking is **2.5× worse than ActDispatch alone** on memory and 3.3× slower than
GC alone. The mechanism: ActDispatch offloads *saved* tensors, and GC's entire
purpose is to stop saving them, so with GC on there are only 6.3 GB left to
offload and the residual peak is the transient working set of a recomputed
up-block, which neither mechanism can reach. **Choose one.** At 2048 the
combination does not rescue the spill either (42.03 GB allocated but 51.75 GB
reserved, 43.0 s/step).

### ActDispatch is reusable standalone

`backend/core/memory_management/` exports exactly two symbols
(`__init__.py:17`): `ActivationDispatcher` and `offload_activations`. The
offloader hooks `torch.autograd.graph.saved_tensors_hooks` and **nothing else**
(`activation_dispatcher.py:67-80`) — no module forward hooks, no
`autograd.Function`, no module patching, no module selection, no `nn.Module`
import, and no architecture knowledge; copies are synchronous, so gradients are
bit-identical to not offloading. `ActivationDispatcher.decide(...)`
(`activation_dispatcher.py:206`) returns `fast` / `offload` / `escalate` from a
self-calibrating area fit refined by `record(...)` (`:239`); there is no
recompute mode, which is why it is orthogonal to GC. `BaseTrainer` coupling is
three glue methods (`base_trainer.py:4954`, `:5077`, `:5117`) reading ~12
`self.*` attributes, five of which are the config keys in
`param_defaults.py:1102-1106`; one of them (`self.debug_vram`) is a bare
attribute access that would `AttributeError` on `VaeTrainer`. Campaign B
therefore measured the real feature **without modifying anything**, by wrapping
the unmodified `_train_micro_step`. Wiring it into `VaeTrainer` needs the five
keys, a lazily-built dispatcher, a wider `with` block (the VAE loss and
`.backward()` currently sit outside the autocast context), a `record(...)` call,
and — for `escalate` — a small VAE-specific micro-split helper, since
`_microbatch_two_stage` (`base_trainer.py:4733`) is bound to the diffusion
argument set. Note `base_trainer.py` sets
`torch.cuda.set_per_process_memory_fraction` process-globally and never resets
it; that is a deliberate decision a VAE path would have to make too.

**It fits 2048 in 15.04 GB and costs 195.5 s/image** — 18 images/hour — because
it moves **91.9 GB per micro-step** over synchronous pageable PCIe, twice. Its
cost is worse than linear in offloaded bytes (1024: 23.0 GB / 6.31 s → 2048:
91.9 GB / 195.5 s, i.e. 4× the bytes for 31× the time). The knobs do not help:
`threshold_mb 64` barely changes the volume (91.2 vs 91.9 GB) because the
decoder's saved tensors at 2048 are individually larger than any useful
threshold; `threshold_mb 1` is slower (59.2 vs 43.0 s at 2048+GC); and
`use_pinned=True` is **30 % slower** (55.9 vs 43.0 s) for identical memory. It
is a fit-or-die lever, never a speed feature.

## 7. True tiled forward/backward wins 2048 — at a gradient cost the loss curve hides

Harness form: encode the whole image once under `no_grad`, split the latent into
a grid, decode each tile **with gradients** from a window padded by `k` latent
cells, crop to the core, compute the loss on the core and `backward()` it
immediately, so only one tile's graph is ever alive. `k = 19` cells (152 px) is
above the 14-16 cells at which receptive-field truncation extinguishes (see
`VAE_DECODE_BEHAVIOR.md`). MEASURED at 2048, batch 1:

| variant | padded decode | area ratio | peak alloc | s/image | img/h |
|---|---|---|---|---|---|
| **4×4, k=19** | 816 px | 2.54× | **14.14** | **6.2314** | **578** |
| 4×4, k=0 | 512 px | 1.00× | 11.04 | 3.5002 | 1,029 |
| 2×2, k=19 | 1328 px | 1.68× | 29.86 | 4.9643 | 725 |

4×4 k=19 is **8.6× faster than GC and 31× faster than ActDispatch, in a third of
GC's memory**, and it beats its own area arithmetic (2.54× the decoded pixels for
1.57× the fitted whole-image time) because splitting the latent turns one
65,536-token attention into sixteen 4,624-token attentions. At ≥1536 px tiling is
not only a memory strategy for this decoder; it is also a time strategy. Do not
cost it with an area-ratio model — that overestimates by ~60 % at 2048.

**The price is a different gradient, and the loss value does not reveal it.**
`n = 4` real images at 1024, 2×2 grid, against the whole-image forward/backward
on the identical latent; **B2** = tiled forward with one whole-image loss and one
backward (no memory saving), **B1** = tiled forward with per-tile loss and
per-tile backward (the only memory-saving form). Medians. MEASURED.

| | k = 0 | k = 19 |
|---|---|---|
| reconstruction MAE vs whole decode | 1.84 /255 | **0.96 /255** |
| MSE term vs whole-image MSE | +38 %…+70 % | **+1 %…+5 %** |
| LPIPS term vs whole-image LPIPS | +25 %…+38 % | +0.2 %…+2.9 % |
| grad cosine / rel-L2, B2 | 0.938 / 0.358 | **0.963 / 0.279** |
| grad cosine / rel-L2, B1 | 0.944 / 0.359 | **0.960 / 0.289** |
| peak alloc A / B2 / B1 (GB) | 23.65 / 23.65 / 6.90 | 23.65 / 36.49 / 10.11 |

1. **The trap:** at k=19 the tiled **loss value stays within 1-5 %** of the
   whole-image loss while the **gradient is cos 0.963 / 28 % relative L2 away
   from the whole-image gradient**. A tiled run's loss curve will look
   indistinguishable from a whole-image run's while it descends a *different*
   objective ("reconstruct each 512 px core given 152 px of context"). That is a
   modelling decision to take deliberately, not "the same result in less memory".
   The divergence is systematic and reproducible per image, not noise, and it is
   image-dependent (cos ranges **0.912-0.985** at k=19 over 4 images — a small
   sample; treat the medians as indicative).
2. **All of the divergence comes from tiling the FORWARD, none from tiling the
   loss** (B1 0.960/0.289 vs B2 0.963/0.279 are indistinguishable). So the
   memory-saving form costs nothing extra in fidelity, and per-tile LPIPS on
   512 px cores is fine.
3. **`k = 0` is not acceptable**: 1.84 /255 reconstruction error is above the
   1 /255 visibility bar this repo uses, and the MSE term is inflated by up to
   70 % — the trainer would chase a seam artifact it created itself.
4. The **30 GroupNorms compute their statistics per tile**, and the
   inference-side fix for that (whole-image statistics, two passes — see
   `VAE_DECODE_BEHAVIOR.md`) has no measured training-side analogue and would
   break the single-backward structure. The residual 0.96 /255 and the 28 %
   gradient gap are consistent with that term still being present. Expect
   GroupNorm, not padding, to be the floor.

## 8. LPIPS never becomes the binding constraint

The opposite of the standing hypothesis. With gradient checkpointing on, LPIPS
costs a **constant +0.06 GB from 512 to 2048** (3.61→3.55, 11.57→11.51,
24.83→24.77, 43.40→43.35 with/without) — that is its fp32 VGG16 weights and
nothing else, because with GC the peak occurs during the backward recompute of
the largest up-block, by which time LPIPS's activations are already freed.
Without GC it is a real but sub-dominant 12-13 % of peak (+0.83 GB at 512,
+3.15 GB at 1024). Time cost is 6-8 % of the step at every resolution.
MEASURED. Cropping LPIPS while the reconstruction loss stays whole-image is
therefore *sound* (§7 point 2) but buys 0.06 GB with GC — not worth design
effort, and free anyway under a tiled scheme.

## 9. Batch shape and budget

`batch_size: 1` + `gradient_accumulation_steps` measures cheaper per image and in
memory than a real batch at the same effective batch, by a wider margin than at
512. MEASURED at 1024 with GC: batch 1 → 0.9960 s/image at
11.57 GB; batch 1 × accum 2 → **0.9923 s/image** at 11.76 GB; batch 2 →
1.1554 s/image at 22.18 GB, i.e. **16 % slower per image for 1.9× the memory**
at the same effective batch (the penalty was 10 % at 512). Above 1280 it is not
a preference but the only shape that fits. Changing the images-per-step also
changes the recipe — `total_steps`, learning rate, `max_grad_norm` and the EMA
horizon are all counted in optimizer steps and have to move with it.

**Step budgets stop being meaningful at high resolution.** 500,000 optimizer
steps at batch 1 / accum 1, from the measured s/image:

| configuration | s/image | 500 k steps |
|---|---|---|
| 512, no mitigation | 0.1704 | 23.7 h |
| *run 113 as configured (512, batch 2 → 1 M images)* | 0.1921 | **53.4 h** |
| 1024, no mitigation | 0.8107 | 4.7 days |
| 1536, GC | 2.5770 | 14.9 days |
| **2048, tiled 4×4 k=19** | 6.2314 | **36.1 days** |
| 2048, GC (spilling) | 53.7414 | 311 days |
| 2048, ActDispatch | 195.5208 | 3.1 years |

**The honest budget unit is pixels, not steps.** 500 k images at 512 px is
131 Gpx; the same 131 Gpx is 31 k images at 2048, which tiled takes **54 h** —
barely more than the current recipe's 53 h. *Per pixel*, high resolution is only
~2.3× less efficient (tiled). A design that keeps the pixel budget and cuts the
step count is defensible; one that keeps `total_steps: 500000` and raises
`resolution` is a 36-day-to-311-day run.

## Limitations

- **The VAE arms of Campaign A are `n = 19`, not the 47 planned, and the sample
  is category-imbalanced: 16 anime, 3 comic, 0 photo.** The machine went into
  heavy GPU/system-RAM contention from another process partway through; the job
  was stopped rather than risk squeezing the machine, and a second pass to
  collect the photo category stalled before its first image and was stopped too.
  **Every §3/§4 conclusion should be read as "on anime/comic content".** The
  direction is not in doubt (the ladder was monotone at n = 4, at n = 19, and
  within each category separately) but the *magnitude* is not supported to a
  coefficient.
- **The canvas-shape-at-matched-token-count experiment was written but never
  ran.** §4's shape-vs-area claim is INFERRED from code (`AttnProcessor2_0`
  flattens; GroupNorm reduces over H, W), not measured.
- **No training was run in either campaign.** Every "this policy would teach the
  decoder X" statement is an inference from measured input statistics and
  measured base/fine-tune behaviour, not a result. Absolute loss values in
  Campaign B carry no meaning (fresh base VAE per arm, no resume).
- **Campaign A's numbers are for one SDXL-family 4-channel `AutoencoderKL`** —
  the base of run 113 — at 512 px crops, with the `dstrain` arm using
  `random_crop=False` (validation semantics) so it pairs with the other arms on
  the same centre region.
- **Campaign B is one machine, one device size** (RTX 6000 Ada 47.99 GB, WDDM).
  The spill behaviour in §5 is specifically WDDM's: on a driver that OOMs
  instead, those arms fail rather than run slowly. The `nvidia-smi` peaks in the
  raw notes are sampled maxima, i.e. lower bounds; `peak_reserved` is the
  authoritative allocator number.
- **Campaign B deliberately omitted periodic work** — metrics DB writes,
  validation, checkpoint saves — so its s/image figures are micro-step costs and
  run slightly faster than a live run at the same settings.
- **Tiled-gradient statistics are n = 4 images at 1024 px, 2×2 grid**, with a
  genuinely image-dependent spread.
- **Everything above assumes the decoder-only default** (`train_decoder: true`,
  encoder frozen, `decoder_blocks: all`, 49.49 M trainable). Narrowing
  `decoder_blocks`, or training the encoder, changes both the memory slope and
  what the crop policy means.

## See also

- `docs/guides/VAE_TRAINING.md` — the `vae_decoder` config surface, loss defaults
  and their falsification criteria, refusals, export. The current `resolution`,
  `validation_resolution` and `crop_scale_policy` defaults are documented there
  and in `VAE_TRAINING_DEFAULTS`, not duplicated here. One interaction worth
  knowing: `make_validation_batch` always takes a deterministic centre crop under
  the `downscale` policy whatever the training policy is, so
  `validation_resolution` is the only axis that makes the held-out metric
  representative — and a validation crop from the heavily-downscaled regime is
  measured where the fine-tune's accuracy gain is *largest* (§3), which is why
  `vae_val_psnr` alone cannot judge a change of objective.
- `docs/guides/VAE_DECODE_BEHAVIOR.md` — inference-side decoder facts: the
  three-term non-locality decomposition, the 14-16 cell context extinction point
  used as `k` in §7, the GroupNorm tint, and invented high frequency.
- `backend/core/memory_management/BLOCK_SWAP.md` — the other CPU-offload
  mechanism in this repo, for diffusion training.
