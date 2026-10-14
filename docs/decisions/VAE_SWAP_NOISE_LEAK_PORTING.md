# Porting the exposure-noise leak fix to the other VAE-swap architectures

Status: plan, 2026-10-14. Anima is done; this document is the brief for the remaining
architectures. Background and evidence: [VAE_SWAP_NOISE_LEAK.md](VAE_SWAP_NOISE_LEAK.md).

## Invariant

The backbone input must be a function of `(x_t, t, c)` only.

Gaussian noise alignment may still correct the source channels toward native noise
statistics, but its **input** side must use `E[eps | x_t]` from the calibrated
`RadialNoisePrior`, never the exposure noise.

The **output** residual is added after the network, which never reads it, so it keeps
the exposure noise in training and the initial noise at inference.

A declaration with `noise_policy`, no `noise_estimate` and a backbone wider than the
source channels (`leaks_exposure_noise`) is refused at training start (`base_trainer`).
That is the current state of every architecture below. Inference is unaffected.

## Reference implementation (Anima)

| Piece | Where |
|---|---|
| Prior: per-channel mean + radial power spectrum, `capture`, `estimate` | `core/models/components/noise_prior.py` |
| Declaration key `noise_estimate: wiener_radial_v1`, `leaks_exposure_noise`, pair owns `noise_prior`, `ready` includes it | `core/models/components/latent_adapter.py` |
| `initialize_image_swap(..., noise_estimate=)` | `core/models/components/spatial_swap.py` |
| Calibration fills the prior from the normalised target latents (`calibrate_images`) | `core/models/components/spatial_swap.py` |
| Declares the estimate for wide swaps | `core/models/anima/vae_alignment.py::initialize_swap` |
| Input side uses `pair.noise_prior.estimate(x_t, sigma)`; output keeps the given noise; the global canvas too | `core/models/anima/anima_models.py::forward_mini_train_dit` |
| Tests | `backend/tests/vae_swap_noise_leak_test.py` |

Commits: `dd87890b`, `d0bef1ed`.

## Recipe per architecture

1. **Declare.** Where the init sets `noise_policy`, also set
   `noise_estimate = NOISE_ESTIMATE` whenever `backbone_width(declaration) > old_channels`.
   - Shared-init archs pass `noise_estimate=` to `initialize_image_swap`.
   - Archs that build the declaration themselves set the key before `validate_declaration`.
2. **Calibrate.**
   - Archs using `calibrate_images` get `pair.noise_prior.capture(...)` for free.
   - Archs with their own calibration must call `pair.noise_prior.capture(targets)`
     with the normalised target-codec latents of the calibration images, on the
     external grid before any packing.
3. **Input side.** At every call that passes the exposure or initial noise into
   `flow_state` / `diffusion_state` / `map_packed_image(..., output=False)`, pass
   `pair.noise_prior.estimate(x_t, sigma)` computed from that same `x_t` instead.
   - `x_t` must be on the external grid (unpack first, or estimate before packing).
   - Leave the output-side calls (`flow_velocity`, `prediction`,
     `correct_diffusion_velocity`, `map_packed_image(..., output=True)`) unchanged.
4. **DDPM paths** (`diffusion_state`, SD1.5/SDXL and Chimera):
   - `RadialNoisePrior.estimate` assumes flow, `x_t = (1 - sigma) x0 + sigma eps`. Add a
     general form `estimate(x_t, alpha, sigma)` with gain
     `sigma / (alpha^2 P + sigma^2)` on `x_t - alpha * mean`.
   - The flow call is `alpha = 1 - sigma`.
   - Keep one implementation and make the flow call delegate to it.
5. **Global / partition canvases.** Any second map of a full canvas (Ming
   `partition_training.py`, `partition_generation.py`; Qwen global payloads) uses the
   estimate from that canvas.
6. **Tests.**
   - The input map is identical for two different `noise` arguments.
   - The declaration of a fresh wide swap carries `noise_estimate`.
   - Calibration leaves the prior `_ready`.
   - The existing Gaussian-swap tests that the training refusal now fails pass again:
     `ming_image_01_training_resume_test` (`gaussian_swap_pending_resume_*`) and
     `ming_image_01_partition_global_full_test` (`True-*`).

## Sites

| Architecture | Init (declaration) | Input-side noise sites |
|---|---|---|
| Z-Image | `zimage_vae_alignment.py:53` | `batched_zimage_wrapper.py:370`, `zimage_transformer.py:677` |
| FLUX.2 | `flux2/vae_alignment.py:56` | `map_packed_image(..., output=False)` callers in `flux2/vae_alignment.py` (`forward_with_noise` / `alignment_context`) |
| Krea 2 | `krea2/vae_alignment.py:53` | `krea2/vae_alignment.py` packed-map input calls |
| Lens | `lens/vae_alignment.py:48` | `lens/vae_alignment.py` input map |
| SD1.5 / SDXL | `components/sd_vae_alignment.py:241` | `sd_vae_alignment.py:284` (`map_input` → `diffusion_state`); DDPM form, item 4 |
| SenseNova SDXL Chimera | via the SD init; confirm `noise_policy` reaches it | `sensenova_sdxl_chimera/vae_alignment.py:150`, `polar_vae_alignment.py:87`; DDPM form |
| Qwen-Image 2.1 | `qwen_image_21/vae_alignment.py:122` (own declaration) | `qwen_image_21/vae_alignment.py:44`, `:64` |
| Ming-Image 0.1 | `ming_image_01/vae_alignment.py:192` (own declaration) | `ming_image_01/vae_alignment.py:108`; `partition_training.py:60`; `partition_generation.py:53` |

Line numbers are as of `d0bef1ed`; search for `flow_state(` / `diffusion_state(` to
re-locate them.

Out of scope:

- Patch noise alignment (MiniT2I / SenseNova pixel swaps). It replaces every channel's
  noise, so there is no second view of `x_t`.
- Audio and video alignments (no noise argument).
- Ideogram 4 (its swap sets no Gaussian policy).

## Acceptance

- Unit tests above. Compile and import (`python -c "import ..."`).
- R2 probe per architecture: a linear readout of the clean projection from the bridge
  input must be no better than from `x_t` alone. The recipe is in `VAE_SWAP_NOISE_LEAK.md`.
- GPU, owner's call: the step-0 sample of a fresh wide swap shows the subject (Anima:
  close to the old leaking init), and samples do not lose structure past step 10k.
- One commit per architecture.
