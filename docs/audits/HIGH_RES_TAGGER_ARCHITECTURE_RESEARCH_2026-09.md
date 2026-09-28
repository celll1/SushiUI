# High-resolution tagger architecture research

Status: **proposed and unimplemented**

Date: 2026-09-29

Scope: a successor to the trainable SigLIP2 tagger for fine-detail, multi-label
classification at arbitrary source resolutions. This document makes no measured
quality, latency, or memory claim about SushiUI.

## 1. Decision

Build a new, weight-independent architecture, provisionally named
**FoveaTag**, rather than replacing SigLIP2 with another fixed-resolution
foundation encoder.

FoveaTag processes these inputs together:

1. one bounded, aspect-preserving **overview** of the complete image;
2. a small, fixed-budget set of **local views** selected from an aligned image
   pyramid; and
3. continuous coordinates and field-of-view metadata for every local view.

The overview preserves composition and supplies routing context. Local views
preserve source pixels for eyes, contact lenses, jewelry, text, and other small
evidence. A shared encoder and a multi-instance classification head combine the
views without forming attention over every source-image patch.

Do not upscale the complete image. Permit magnification only for selected local
views, initially at `1x` and `2x` field-of-view levels. Upscaling cannot create
missing information, but a bounded local magnification can place existing small
features across enough learned cells to classify them. Whole-image upscaling
adds cost without adding evidence and creates a particularly easy scale
shortcut.

FoveaTag is fully from scratch: original architecture, random initialization,
and self-supervised pretraining on project data, with no third-party model
weights. The goal is a representation that keeps local detail, not a
replacement for an existing encoder, so the encoder should not inherit the
inductive biases of a fixed-resolution pretrained model.

Image-level tags alone are not expected to train the encoder end to end from
random initialization. For a small eye or colored contact lens, the tag gradient
is diluted across the whole image, and a learned router and an untrained encoder
each wait on the other. Training is therefore staged (section 5.1):
self-supervised pretraining, tag training on random and stratified views,
router training from a dense-candidate teacher, then joint fine-tuning.

Pretrained encoders such as SigLIP2 or PE are used only as diagnostic controls
and as a reference upper bound (section 8.2). They are never used for
initialization, and a gap to them does not trigger a switch to pretrained
weights. If the from-scratch encoder does not meet its pre-registered gate,
work stops and the cause is measured (section 9).

## 2. Current failure mechanism

The current trainable path uses
`google/siglip2-so400m-patch16-naflex` by default
(`backend/core/tagger/siglip2_tagger_model.py`). Each dataset item calls the
processor once, without an explicit patch budget
(`backend/core/tagger/tagger_dataset.py`). The upstream processor default is 256
patches. NaFlex retains aspect ratio, but it still resizes the complete image to
fit that budget; it does not preserve native pixels. For a square image this is
effectively a 256 x 256 patch canvas at patch size 16, irrespective of how large
the source was.

Inference already accepts `max_num_patches` (request default 256,
`backend/api/routes.py`; applied in
`backend/core/tagger/siglip2_inference_manager.py`), so a larger inference
budget can be measured without retraining. Such a measurement runs a checkpoint
trained at 256 patches at a different budget, and must be reported as a
train/inference budget mismatch.

The encoder output is reduced to one pooled vector and then passed through one
linear multi-label head. The optional custom pooler is also a single learned
query. This creates two independent bottlenecks:

- small evidence may disappear before tokenization because the whole image was
  reduced to a fixed patch budget;
- evidence that survives tokenization must compete for one global pooled
  representation shared by every tag.

Increasing `max_num_patches` alone is not the proposed fix. It increases
self-attention cost with the number of retained patches, does not teach which
small regions matter, and risks making the training resolution a shortcut that
does not extrapolate. This risk is a hypothesis; the project has no recorded
measurement of it, and Phase 0 should measure it on a resolution ladder.

## 3. What recent work contributes

The table separates reusable ideas from suitability as the production base.
Paper results are the authors' results in their own regimes; they are not
expected SushiUI gains.

| Work | Relevant result or mechanism | License finding | Verdict here |
|---|---|---|---|
| SigLIP2 / NaFlex (2025-02) | Variable aspect ratio and patch budget; native aspect ratio is not native-pixel preservation. | Google checkpoints and Transformers implementation are Apache-2.0. | Current baseline. Raising its patch budget is a useful ablation, not the target architecture. |
| PS3 (CVPR 2025) | Low-resolution global encoding plus selective high-resolution regions; multi-scale high-resolution pretraining; fixed selected-token budgets can improve as source resolution rises. Its largest reported selection gains come from local caption and box supervision. | Code package is Apache-2.0, but released PS3-SigLIP2 weights use NVIDIA's non-commercial license. | Closest mechanism to this proposal (global low-resolution plus selected high-resolution regions). Reuse the principle, not the weights or implementation. Image-level tag labels require a different router objective. |
| Perception Encoder (2025-04) | Global contrastive features, intermediate-layer alignment, spatial variants, and tiling-tuned variants. Released sizes still have preferred 224--512 pixel inputs. | The PE page states its checkpoints are Apache-2.0; the containing repository also includes separate non-commercial PLM material, so provenance must remain checkpoint-specific. | Permissive benchmark candidate, but tiling and fine-detail multi-label aggregation would still need to be designed. |
| RADIOv2.5 / C-RADIOv3-v4 | Multi-resolution training, dense features, non-square inputs, and multi-teacher distillation. C-RADIOv4 incorporates newer SigLIP2, DINOv3, and SAM3 teachers. | RADIO/E-RADIO are non-commercial. C-RADIO weights use the NVIDIA Open Model License and allow commercial use, but repository code and model terms are not a simple Apache dependency. | Teacher/benchmark candidate. Multi-teacher lineage and licensing complexity make it a poor default dependency. |
| DINOv3 (2025-08) | Dense-feature objective, a RoPE variant on normalized patch coordinates with box jittering, high-resolution adaptation stage, and Gram anchoring that prevents dense-feature degradation in long training. | Custom DINOv3 license, gated weights, redistribution conditions, and use restrictions; not OSI Apache/MIT. | Dense-feature benchmark or research teacher. Do not make it a required production base. |
| Charm (CVPR 2025) | Fixed-length multi-scale tokenization that retains aspect ratio, composition, and selected detail. | Apache-2.0 implementation. | Relevant tokenizer ablation, but evidence is from aesthetic assessment and it still depends on a pretrained ViT. |
| Token Cropr (CVPR 2025) | End-to-end task-aware token selection; reported 1.5--4x speedups with task-dependent accuracy trade-offs. | License must be rechecked before adapting code. | Supports learned pruning after shallow features. It does not solve regions erased before tokenization. |

The most transferable findings are:

- high-resolution learning needs local high-resolution supervision, not merely
  high-resolution inference;
- a constant selected-region budget decouples inference cost from source
  dimensions;
- multi-resolution training is required to prevent a resolution mode switch;
- dense features and global classification features are different objectives;
- routing failures must be supervised, not hidden behind the final tag loss.

PS3 used 75 million high-resolution images, local captions, and boxes. Its
headline results therefore cannot be assumed to transfer to fine-tuning on
SushiUI's image-level tag captions. DINOv3 and RADIO likewise rely on
large-scale pretraining or distillation. The proposed model must prove that
weakly supervised multiple-instance learning is sufficient for this dataset.

## 4. FoveaTag v0

### 4.1 View construction

For source image `I` with size `(H, W)`, construct a coordinate-aligned pyramid
without materializing every level at once:

- **overview**: the entire image, letterboxed to a bounded canvas, initially
  384 or 448 square;
- **context candidates**: regions sampled from an intermediate level whose long
  side is around twice the overview size;
- **native candidates**: crops read at source pixel density;
- **magnified candidates**: selected regions with half the native field of view,
  resized to the same local-view canvas. This is the only default `2x` path.

All local views use a common canvas, initially 256 square. Coordinate transforms
remain exact so an evidence slot can be mapped back to the source image.

Training must not always contain an untouched native view. For each item, sample
the pyramid's base pixel density log-uniformly from a bounded range, include the
overview, and draw local views from at least two field-of-view levels. Source
resolution, resampling kernel, and JPEG quality should be varied independently.
This makes the simultaneous views agree on content without making one exact
resolution a privileged signature.

Images smaller than a local canvas are padded. They are not routinely enlarged
as a whole. A mild train-time resize jitter may cross `1x`; it exists for scale
robustness, not as a source of detail.

### 4.2 Budgeted coarse-to-fine router

The router has no text prompt at inference and cannot score every vocabulary tag
against every source patch. It therefore predicts a small set of generic
**evidence proposals**:

1. the overview encoder emits a coarse spatial map;
2. learned routing slots score spatial locations and field-of-view levels;
3. scale-space non-maximum suppression retains `K` proposals;
4. a proposal may be refined once at the next-finer pyramid level.

At training time, reserve part of the budget for stratified random coverage.
Otherwise an early, incorrect router can permanently hide rare evidence. On a
small fraction of batches, evaluate a denser candidate set and train the router
to rank regions that improve positive-tag margins. The dense pass is a
training-only oracle and is detached from the main classifier when producing
router targets. Its cost is one local-encoder pass per dense candidate, so the
candidate count and batch fraction must be set from the Phase 0 timing harness
and reported with training throughput.

Inference is sequential: the overview must be encoded before local views can
be selected, and an optional refinement adds a third dependent stage. Batched
inference therefore runs as staged batches (all overviews, then all selected
views), and latency must be measured per stage rather than derived from the
pixel counts below.

The initial inference modes should differ only in view budget, not model weights:

| Mode | Overview | Local views | Purpose |
|---|---:|---:|---|
| fast | 384 | 2 | bulk dataset triage |
| balanced | 448 | 4 | default |
| detail | 448 | 8 | small-object and color-sensitive tagging |

These are experiment starting points, not API defaults. The approximate encoded
pixel counts are 0.28, 0.46, and 0.72 megapixels respectively. They are useful
for comparing workloads but do not predict wall-clock latency.

### 4.3 Shared local/global encoder

Use one encoder for overview and local canvases so that evidence is comparable
across scale. The proposed original backbone has:

- an anti-aliased convolutional stem to stride 4;
- two early stages of depthwise-gated residual blocks for color, edge, and
  texture retention;
- windowed attention at the middle stage, with alternating shifted or dilated
  neighborhood mixing;
- one low-token global stage after spatial merging;
- RMS normalization and gated MLPs; and
- continuous 2-D rotary coordinates, augmented by normalized source
  coordinates and continuous `log2(field_of_view / source_size)` features.

There is no learned absolute position table and no resolution-bucket embedding.
Continuous scale features are dropped on some training views so they help
geometry without becoming a label shortcut. Global attention is applied only
after spatial reduction; compute is approximately linear in the number of
encoded view pixels rather than quadratic in all source patches.

The first size target is 25--50 million parameters. The number of effective
fine-detail supervision signals is much smaller than the number of images, so
capacity is increased only after the smaller model is shown to work. Width and
depth are selected after throughput measurement; the document intentionally
does not invent a final configuration.

### 4.4 Evidence slots and tag head

Each view emits several evidence slots rather than one CLS vector. Each slot
carries its feature, source rectangle, and field-of-view. A small slot mixer lets
local evidence consult the overview while keeping the token count fixed.

For tag `t`, use a shared tag vector with a global term and a smooth
multiple-instance reduction over local evidence:

```text
logit_t = bias_t
        + dot(global_summary, global_weight_t)
        + temperature * logsumexp_j(dot(evidence_j, local_weight_t) / temperature)
```

The local reduction lets different tags select different views. It also exposes
the highest-contributing source rectangle for debugging. Quality and Rating keep
their current group-aware selection rules downstream; the architecture still
produces ordinary per-tag logits.

The vocabulary head remains expandable by appending rows. Checkpoints must store
the view contract, coordinate convention, and router budget independently of the
vocabulary, so vocabulary growth does not alter the image representation.

## 5. Training objectives

### 5.1 Training stages

1. **Self-supervised pretraining** on the same dataset, with no external
   weights or teachers:
   - coordinate-aware correspondence between the overview and overlapping local
     views;
   - masked local-patch reconstruction; and
   - chroma-preserving reconstruction. Pixel- or color-space targets must keep
     color differences; a semantic-only target can discard the contact-lens
     color this project is trying to recover.
2. **Tag training on random and stratified local views.** No learned router.
   The encoder and evidence head are established first.
3. **Router training from dense candidates.** The stage-2 head evaluates a dense
   candidate set, and the regions that contribute to positive-tag margins become
   router targets.
4. **Joint fine-tuning with the router.** Random coverage is kept, never fully
   annealed away, so rare tags continue to be explored.

The stage-3 teacher is biased: the head learned only what random views exposed,
so evidence it does not yet recognize yields no router target. Retained random
coverage is the mitigation, and its effect is checked with router recall and
with rare-tag results split by positives and negatives.

**Local-view teacher distillation (ablation only).** A variant of stage 1 adds
a feature-alignment term to SigLIP2 (`google/siglip2-so400m-patch16-naflex`,
Apache-2.0) evaluated on local views only. A local crop keeps fine detail
visible at the teacher's own input size, whereas the teacher's whole-image
input discards it. No teacher weights enter the FoveaTag checkpoint, but a
distilled checkpoint records the teacher as provenance. The chroma
reconstruction term is kept, and color-sensitive tags are checked for
regression, because a semantic teacher can discard color differences. This
variant is not part of the default plan: it is the Phase 3 cause measurement
that separates missing semantic knowledge from other causes. Weight transplant
from SigLIP2 is not used; its patch16 linear embedding and absolute position
embedding do not match this encoder, and matching them would reintroduce the
fixed-resolution assumptions this design avoids.

The alignment is defined as follows. The teacher encodes each 256-square local
canvas with `max_num_patches=256`, giving a 16 x 16 grid of 16-pixel cells, and
its last hidden state (before pooling) is the target. After self-attention each
teacher token also carries context from the whole canvas, so a cell is a 16 x 16
spatial anchor at a source coordinate, not a description of those pixels alone.
The student map used for distillation must be at least 16 x 16 on a 256 canvas
(stride 16 or finer), so it is taken from the windowed-attention stage before
spatial merging, not from the low-token global stage. It is average-pooled onto
the teacher's anchors through the exact source coordinate transform. A
training-only projection head maps the pooled student features to the teacher
dimension, and the loss is cosine distance between L2-normalized features per
anchor. The projection head weights and optimizer state are saved in resumable
training checkpoints and excluded from inference and export artifacts.

The comparison against pure self-supervised pretraining uses the same number
of images presented and the same number of optimizer steps. The teacher forward
pass makes equal GPU time incompatible with equal steps, so GPU time including
the teacher and peak VRAM are reported separately rather than equalized.

Self-supervised pretraining on one GPU is a large time cost. Its budget is set
from the Phase 0 timing harness (time per epoch at the chosen model size) before
it starts.

### 5.2 Losses

The minimum viable objective is a weighted sum whose coefficients must be
measured rather than copied from another regime:

- **bag-level asymmetric multi-label loss** on the final logits, retaining the
  current tag masks and category rules;
- **router ranking loss** from the training-only dense candidate oracle;
- **overlap consistency**, aligning features from two views only in their shared
  source rectangle;
- **prediction consistency** between two independently sampled view sets when
  both retain adequate source pixel density; and
- **coverage/diversity regularization**, preventing every routing slot from
  selecting the same face or high-contrast corner.

Do not force a deliberately blurred view to predict a fine-detail tag merely to
obtain scale consistency. Consistency is gated by the known pixel footprint of
the corresponding evidence. Otherwise the loss teaches the desired information
away.

## 6. Why this should extrapolate better

FoveaTag does not promise unlimited detail from a fixed budget. It changes the
failure mode from deterministic whole-image destruction to measurable routing
recall. Extrapolation is improved by four explicit constraints:

- the expensive view count is independent of `(H, W)`;
- positions and scale are continuous rather than learned per resolution;
- the same weights see overview, context, native, and magnified views;
- train-time pixel density and view membership vary, so exact original size is
  never a guaranteed cue.

An image larger than the training range can therefore expose finer native crops
without lengthening the encoder sequence. Accuracy may still fall if `K` misses
the relevant region; this is why router recall is a first-class metric.

## 7. Rejected alternatives

### Full image at several resolutions

This directly implements simultaneous multi-scale input, but the native branch
still grows with source area and dominates compute. If every branch instead has
a fixed patch budget, the native detail is destroyed before the model sees it.
Use it as an ablation, not the production design.

### Only overview plus random crops

This is robust and simple, but wastes the small inference budget and gives no
reason to sample an eye rather than a background patch. Random crops remain
necessary exploration during training.

### Replace SigLIP2 with DINOv3, PE, or C-RADIO

All are usable as baselines. None supplies a tagger-trained, weak-label router
and per-tag local evidence reduction. DINOv3 and RADIO also add material license
constraints. Encoder replacement alone does not remove the pooling bottleneck.

### Whole-image upscaling

It creates no source information, increases compute, and makes interpolation
artifacts resolution-dependent. Local magnification is bounded and ablated
separately.

### Hard top-k routing from the first step

This starves the local encoder and router of positive examples. Begin with
coverage and random exploration, then anneal toward selected views.

## 8. Evaluation plan and acceptance gates

### 8.1 Benchmark slices

Create a fixed, versioned validation suite with:

- eye color, colored contacts, heterochromia, makeup details, earrings, nail
  color, small logos/text, handheld objects, and distant characters;
- coarse tags that must not regress, including count, scene, pose, and broad
  character attributes;
- source-resolution ladders made from the same images;
- aspect-ratio extremes; and
- at least a small manually boxed subset for measuring whether the router saw
  the evidence. Boxes are evaluation data, not required production labels.

Report macro average precision and F1 by tag group, tag prevalence, evidence-box
pixel area, and source resolution. Also report calibration error, prediction
stability across the resolution ladder, router recall at `K`, images/second,
batch-size scaling, peak VRAM, and host preprocessing time.

### 8.2 Required baselines

1. current SigLIP2 NaFlex at 256 patches;
2. the same checkpoint at larger patch budgets;
3. current encoder with naive global-plus-random-crop aggregation;
4. FoveaTag overview only;
5. FoveaTag with random local views;
6. FoveaTag with learned routing; and
7. one permissively usable recent encoder benchmark, such as PE (Apache-2.0
   checkpoints), with the
   same downstream evidence head when technically possible.

Baselines 3 and 7 are **encoder-swap controls**. No pretrained weights are
loaded into the FoveaTag encoder; instead, the whole encoder is replaced by an
existing model, and the FoveaTag view construction and evidence head consume
its patch-token map. The views and the head need only a spatial token map, so
this works with any such encoder. These controls answer whether the view and
head design helps independently of the encoder, and they give a reference
upper bound for the from-scratch encoder. They cannot test the encoder's
internal choices (continuous coordinate RoPE, scale features, stem design);
those are ablated within the from-scratch encoder.

The router, local `2x` magnification, continuous scale feature, dense-oracle
frequency, and local `logsumexp` head each require an ablation. Local-view
teacher distillation (section 5.1) is compared against pure self-supervised
pretraining at equal images presented and optimizer steps.

### 8.3 Go/no-go gates

Do not integrate the new architecture into the API until measurements show:

- a material gain on the fine-detail slice at matched false-positive rate;
- no material loss on the coarse slice;
- stable or improving results on held-out source resolutions above the training
  range;
- router recall high enough that the classification head, rather than missing
  views, is the dominant remaining error;
- balanced inference within the owner-approved latency and VRAM budget; and
- an explicit checkpoint license/provenance decision.

No numeric quality or latency threshold is set here because it has not been
measured on project hardware.

## 9. Implementation phases

| Phase | Work | Exit condition |
|---|---|---|
| 0 | Add the fixed benchmark, evidence-area annotations, and timing harness. Measure current 256-patch and enlarged-patch SigLIP2, and time per epoch of candidate from-scratch model sizes. | The current failure is quantified by tag group and evidence size; the self-supervised pretraining budget is set. |
| 1 | Encoder-swap control: FoveaTag random multi-scale views and evidence-slot head on the existing SigLIP2 encoder, without a learned router. | The view and head design improves fine-detail tags at a known cost. The from-scratch gate for Phase 3 is written into this document before Phase 2 starts. |
| 2 | Original shared encoder (25--50M) with the continuous coordinate/scale contract; self-supervised pretraining (stage 1). | Pretraining losses converge, and overlap correspondence and chroma reconstruction are measured on held-out images. |
| 3 | Tag training on random and stratified views (stage 2). | The pre-registered gate against the Phase 1 control, including resolution-ladder extrapolation, is met. If not, stop and measure the cause (pretraining objectives, model capacity, router timing, and missing semantic knowledge via the local-view teacher distillation ablation); do not switch to pretrained weights. |
| 4 | Router training from dense candidates (stage 3), then joint fine-tuning with retained random coverage (stage 4). | Learned routing beats random views at the same `K`; router recall and rare-tag results are measured. |
| 5 | Add checkpoint bundles, resume, inference manager, export, API, defaults, OpenAPI, and UI. | Existing tagger lifecycle tests plus new view-contract tests pass. |
| 6 | Optional capacity increase and further objective ablations. | Retain only components with isolated improvements. |

The Phase 3 gate is set from Phase 0 and Phase 1 measurements before the
from-scratch run, so the criterion cannot be adjusted to fit its result.

Phase 0 must precede implementation. The project does not currently have an
evaluation that can distinguish a better high-resolution architecture from a
model that merely changes thresholds or prevalence bias.

## 10. Integration surface after acceptance

FoveaTag should be a sibling stack under `backend/core/tagger/`, not conditionals
inside `siglip2_tagger_model.py`. The shared pieces are vocabulary, label/mask
construction, tag selection, checkpoint retention policy, and run lifecycle.
Image preprocessing, model loading, checkpoint format, and inference batching
are architecture-specific.

Any implementation phase that exposes parameters must update, together:

- `TAGGER_TRAINING_DEFAULTS` in `backend/api/param_defaults.py`;
- the Pydantic and `Form()` request surfaces in `backend/api/routes.py`;
- `/schema/tagger-training-defaults` consumers;
- `openapi.yaml`; and
- the frontend API type and tagger panels.

Model metadata must fail closed on an unknown architecture or view contract.
Existing SigLIP2 and legacy WD checkpoints remain loadable and are not silently
interpreted as FoveaTag.

## 11. Primary sources

- SigLIP2 paper: <https://arxiv.org/abs/2502.14786>
- Transformers SigLIP2 / NaFlex documentation and defaults:
  <https://huggingface.co/docs/transformers/model_doc/siglip2>
- PS3 paper and implementation: <https://arxiv.org/abs/2503.19903>,
  <https://github.com/NVlabs/PS3>
- PS3-SigLIP2 weight terms:
  <https://huggingface.co/nvidia/PS3-4K-SigLIP2>
- Perception Encoder paper and checkpoint documentation:
  <https://arxiv.org/abs/2504.13181>,
  <https://github.com/facebookresearch/perception_models/tree/main/apps/pe>
- RADIOv2.5 paper and repository license matrix:
  <https://openaccess.thecvf.com/content/CVPR2025/html/Heinrich_RADIOv2.5_Improved_Baselines_for_Agglomerative_Vision_Foundation_Models_CVPR_2025_paper.html>,
  <https://github.com/NVlabs/RADIO>
- DINOv3 paper and license: <https://arxiv.org/abs/2508.10104>,
  <https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md>
- Charm paper and implementation: <https://arxiv.org/abs/2504.02522>,
  <https://github.com/FBehrad/Charm>
- Token Cropr paper:
  <https://openaccess.thecvf.com/content/CVPR2025/html/Bergner_Token_Cropr_Faster_ViTs_for_Quite_a_Few_Tasks_CVPR_2025_paper.html>
