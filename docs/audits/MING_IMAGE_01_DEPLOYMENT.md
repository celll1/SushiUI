# Ming official model deployment and live UI verification

Date: 2026-10-04. Owner explicitly authorized main GPU/API use and API restart.

## Deployed packages

| Profile | Local model directory | HF revision | Converted package bytes |
|---|---|---|---:|
| Design (`generation_edit`) | `M:/model/ming_image_01/design_int8_convrot` | `208087ada1486931692c1896f38d4cd16ff3df82` | 26,739,650,742 before license/provenance additions |
| Layer (`layer_decompose`) | `M:/model/ming_image_01/layer_int8_convrot` | `9fabca8b62a67f1f53a957d46389c00451c11e52` | 27,017,075,174 before license/provenance additions |

Official originals are in the independent worktree's ignored
`local/models/ming-image-0.1-sources/{Design,Layer}` directories. Earlier work
had not created these full models; tiny CPU fixtures were never release models.
Both packages now contain native six-component manifests, quantization reports,
MIT notices, and pinned HF repository/revision provenance. The main model-search
configuration includes `M:/model/ming_image_01`, preserving the other search roots.
Both models are discovered through `/api/v1/models` and selected in Generate.

Conversion ran on CPU with CUDA initialization forbidden. Each profile converted
15,261 Linear layers: 238 DiT, 14,824 MLLM, 196 connector, and 3 projections.
Embeddings, tied outputs, queries, normalization and other protected tensors stay
floating. Eight official weight files with identical SHA256, totaling 34.26 GB,
are shared by hardlinks between original downloads on the same volume.

## Real model findings and fixes

The first live Design load inherited generic FP16 from public model dispatch;
its generated RGBA PNG contained zero in every channel. Public Ming loads now
use BF16, and the native RGBA VAE loads in FP32. Conditioning, scheduler latents
and decoded pixels reject nonfinite values before a black PNG can be saved.
The unchanged source package was used for subsequent successful inference.

Live Layer review found the API selected auxiliary frame zero as `image` while
preview/training semantics expect Layer 1. Primary decoded output, normal gallery
responses and skip-gallery responses now select Layer 1; ordered `images`,
`groups` and canvas ownership remain unchanged. OpenAPI records this contract.
The component graphic explicitly labels floating parameter objects: packed
weights are buffers, so that census is not the full architectural parameter count.
Serialized storage is displayed separately.

## Verification scope

- Main integration: `90869f1d`; precision correction: `958e9dff`; Layer primary,
  notices and accounting wording: `cb1277e7`.
- Main API restart used its supported endpoint; no server was launched directly.
- Main integration CPU checks: 59 cases. Precision/manager/conversion checks:
  37 cases. Updated conversion-notice checks: 9 cases. Layer-primary/persistence/
  manager checks: 42 cases. These sets overlap and must not be added as unique cases.
- Changed backend files compile and import with CUDA forbidden; OpenAPI parses.
- Frontend contracts and primary-output transfer consumers were reviewed
  statically. No agent frontend build/type-check was run; owner compiler and
  full interactive acceptance remain separate.
- Design: UI selection/load and Generate, 1024x1024, 12 steps, CFG 1, seed 123456,
  RGBA PNG persisted and displayed. Both a plain caption and native-schema JSON
  were exercised after precision correction.
- Layer: UI selection/load, Design result sent to img2img, 512x512, 12 native
  steps, CFG 2, seed 123456, two requested layers. Three RGBA PNGs persisted and
  displayed with `auxiliary_canvas`, `layer_1`, `layer_2` roles; their alpha ranges
  were 234-255, 137-255, 242-255 in this sample.

These are functional inference checks. Design samples contain unintended
lettering and do not satisfy every requested object; the Layer sample separates
subject and background but does not establish clean transparency or exact
recomposition. Numerical parity against original full-precision inference,
quantization quality acceptance, official-model GPU training/resume and high-
resolution Layer memory limits have not been established by these runs.

## Final primary-output and preview verification

The final Layer API run completed without error and reported `_1.png` as
`last_result.filename` (gallery ID 11619); its output group retained canvas and
both layers. The follow-up Preview selection change is `4304b481`, integrated
into main. Existing results verified Layer 2 / Layer 1 switching in the browser,
with main image URLs and active indicators changing together. This UI verification
required no additional GPU inference. See the frontend audit for review scope.
