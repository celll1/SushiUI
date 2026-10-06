# VAE swap alignment probe

This opt-in experiment compares normalized Ming RGBA latents (16 channels,
spatial factor 8) and Qwen-Image 2.1 RGBA latents (64 channels, factor 16).
It does not enable a production VAE swap, load a DiT, or change a training run.
The current swap gate still refuses different spatial compression factors.

`examples/vae_swap_alignment_probe.py` loads only one FP32 VAE at a time.
Its dry-run is the default; `--execute` explicitly enables model execution.
Pass `--ming-vae`, `--qwen-model`, approved image records via `--image-list`, and
`--output`. Use the repository venv. GPU residency must first be released by
stopping the active run through the API.

Real images are restricted to explicit rating-general metadata. Each JSON
record carries `path`, `rating: "general"`, and a `tags` list containing the
exact token `general`, without sensitive/questionable/explicit rating tokens.
All records are validated before opening any image. `--synthetic-only` reads
no dataset images. Do not select unfiltered training images for this probe.

Ming's 2x2 cells are packed losslessly into 64 channels on the Qwen grid.
Paired posterior modes use each VAE's native normalization, not raw encoder
moments. Train and evaluation images are disjoint. Synthetic RGBA probes add
varying alpha; opaque JPEG evaluation alone cannot establish alpha alignment.

Compared initializations:

- Zero, target mean, and blind packed identity.
- Existing RGB preview coefficients composed with a pseudoinverse (Qwen to
  Ming only). Wan coefficients approximate Ming and do not encode alpha.
- Train-set calibrated RGB bottleneck, to separate calibration from rank loss.
- Full affine ridge fit between paired latents, independently in both directions.
- A 3x3 spatial affine fit, to test whether neighboring features recover detail
  that a per-cell channel map misses. This remains on the smaller token grid.
- Orthogonal Procrustes, which preserves isotropic Gaussian noise covariance.

The output contains held-out latent errors, RGB/alpha reconstruction PSNR,
noise covariance distortion, small alignment matrices and visual comparisons.
Images display white-composited RGB with an alpha strip underneath. The native
VAE reconstruction is the reference for separating alignment error from each
VAE's own reconstruction error.

CPU tests verify packing, held-out affine recovery, composition into an input
Linear, spatial recovery, rating restrictions, RGB rank loss and orthogonal noise preservation. Affine bias belongs
to clean features: composing it into an epsilon/velocity output is not justified.
For flow state mapping its bias is multiplied by the clean fraction, not added
unchanged at every noise level. Spatial diagnostics only measure covariance at
one output position; overlapping neighborhoods also correlate adjacent positions.

Even good clean-latent reconstruction is not acceptance for diffusion sampling.
Noise distributions, velocity targets, patch folding, token grid positions and
teacher/student intermediate features remain separate checks. Mapping back to
the old dense token grid does not produce the proposed attention speedup.

Ming's patch-2 embedder sees one packed 64-feature cell at image scale 16.
The Qwen VAE's raw 64-channel cell is already at that scale. Preserving that
token correspondence would require patch-1 on the new latent, and would not
reduce token count. Keeping patch-2 on the new latent merges four old token
locations; that changes the backbone's spatial processing and needs additional
feature alignment/distillation. This probe tests the VAE bridge, not that step.

## General-only GPU measurement

The accepted measurement uses 48 rating-general images: 32 for fitting and
16 held out, plus 8 fitting and 4 held-out synthetic RGBA patterns. No other
rating is used. Reconstruction scores below are pooled MSE PSNR over four
held-out natural images and two synthetic alpha probes. Higher is better.
Latent metrics evaluate all 20 held-out inputs. FP32 posterior modes are
encoded and decoded with each native VAE's normalization, one VAE on GPU at
a time. These measurements do not evaluate generated diffusion samples.

| Qwen latent to Ming decoder, 256px | RGB PSNR | Alpha PSNR |
|---|---:|---:|
| Ming native reconstruction | 19.78 | 30.65 |
| Zero initialization | 8.36 | 8.90 |
| Fixed preview RGB pseudoinverse | 11.66 | 10.05 |
| Calibrated RGB bottleneck | 14.83 | 8.43 |
| Paired per-cell affine | 16.87 | 20.79 |
| Paired 3x3 spatial affine | 19.21 | 23.52 |

The spatial bridge reduces held-out relative latent RMSE from 0.72 to 0.58.
It also amplifies white-noise RMS by 2.51, versus 1.16 for the per-cell bridge.
That reconstruction improvement does not justify passing noisy diffusion
states through the bridge unchanged. Orthogonal fitting preserves noise
covariance but reconstructs these clean inputs poorly (RGB PSNR 13.72).
The data subset is small and drawn from one illustrated-image dataset;
generalization to photographic data and other model pairs is unmeasured.

The unchanged 256px maps were also applied to 512px inputs without refitting:

| Qwen latent to Ming decoder, 512px | RGB PSNR | Alpha PSNR |
|---|---:|---:|
| Ming native reconstruction | 20.25 | 29.00 |
| Calibrated RGB bottleneck | 16.11 | 8.43 |
| Paired per-cell affine | 17.38 | 18.04 |
| Paired 3x3 spatial affine | 20.12 | 22.05 |

The reverse Ming-to-Qwen bridge also improves with spatial context at 512px:
RGB PSNR 16.86 to 18.24, versus Qwen native reconstruction 18.82. Alpha remains
substantially below native reconstruction in both directions.

`--embedding-only` reads only the saved bf16 Ming input projection and composes
the fitted per-cell bridge into a patch-1 experimental input. It explicitly
converts channel-major `pixel_unshuffle` features to Ming's position-major
patch ordering. On the real saved projection, FP64 fold equivalence has max
absolute error 1.24e-14. The artifact is only an input initialization, not a
loadable full model, and retains the original token count.

## Training integration contract (2026-10-06)

The requested default for a new VAE swap is **paired_spatial**, not the
per-cell linear probe. This is a target for the integration, not a shipped
API default: `vae_swap_new_channel_init` still accepts only `zero` until the
backbone, checkpoint reader and generation paths can restore the new layers.

`core/models/components/vae_alignment.py::SpatialLatentBridge` implements the
shared trainable 3x3 operator. It imports the probe's channel-major `unfold`
coefficients, supports channel expansion/reduction and lossless spatial
packing in either direction, and handles independent ordered image frames.
It does not implement temporal alignment or install itself on a backbone.
CPU tests check fp32/bf16, non-square grids, gradients, frame independence,
state restoration and flow-state/velocity semantics against independent
oracles. The operator refuses execution without calibration.
The state includes versioned method/packing geometry and rejects loading a
same-width projection from a different spatial domain. The streaming fitter
retains only float64 sufficient statistics; no image paths or pixels.

Calibration is a **one-time preparation for a particular pair of VAE weights
and normalization domains**, not a dataset prerequisite on every start.
Initializers can reuse the fitted matrices; a trained checkpoint must embed
the bridge's learned weights and geometry. Resume and inference must load
those weights without opening the calibration images or fitting again.
The existing approved general-only Ming/Qwen matrices can seed this pair;
they are not interchangeable with matrices for another VAE, even when the
channel counts match. The general-only restriction applies to images the
coding agent uses for validation or for producing fixed bundled matrices.
Production calibration on a user's training dataset has no rating filter.
At first swap, the training integration should sample the selected run's
dataset and fit the bridges; subsequent resume/inference restores tensors.
Calibration images themselves must not be included in checkpoints.

The architecture-neutral integration must:

1. Bind calibration to both VAEs' weight/normalization identities and packing
   geometry. A missing or mismatched calibration must surface in preflight;
   it must not silently use zero initialization or another pair's matrix.
2. Use separately fitted new-to-old input and old-to-new output bridges.
   A transpose or pseudoinverse is not the fitted reverse direction.
3. Install bridges before optimizer construction and include them in full
   fine-tune parameters, offload ownership, optimizer restoration and export.
4. Apply the mean offset to clean states only. For
   `x_s=(1-s)*clean+s*noise`, the mapped state is `A*x_s+(1-s)*b`.
   `eps-clean` velocity maps to `A*v-b`; Ming's `clean-eps` maps to `A*v+b`.
   Epsilon prediction maps without a clean bias. This does not whiten the
   transformed noise: the measured 2.51x RMS gain still needs diffusion-level
   validation, separate from reconstruction and fold-equivalence tests.
5. Preserve the existing backbone token geometry initially. Ming/Qwen uses
   a packed 16-channel Ming domain against Qwen's 64-channel domain; that
   preserves Ming's original token count, not a fourfold token reduction.
   Reference latents, U-Net scales, partition paths and sampling must obey
   the same bridge geometry as the target stream.
6. Store a versioned bridge declaration and tensors with the checkpoint,
   restore before strict state loading, and load the declared target VAE in
   generation as well as training. Existing checkpoints retain their saved
   IO weights and must never be initialized again merely by Edit config.

Other image VAE pairs can use the same operator after pair-specific
calibration if their grids can be related by integer spatial packing and
their pixel preprocessing is explicit (RGB/RGBA, alpha and normalization).
The operator alone does not establish architecture-level compatibility.
Non-integer grid relationships, real temporal compression, audio codecs,
and pixel-space backbones require separate adapters and validation. Current
production spatial/temporal compatibility refusals remain in force.
