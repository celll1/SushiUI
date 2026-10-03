# High-resolution tagger architecture research

Status: **Phase 2 accepted; A/B/C abandoned by owner; direct joint router training for owner visual review**

Date: 2026-09-29

Scope: a successor to the trainable SigLIP2 tagger for fine-detail, multi-label
classification at arbitrary source resolutions. Measurements in section 9 are
specific to the listed checkpoints, datasets, and RTX 6000 Ada.

## 1. Decision

### Current owner-directed revision (supersedes the staged A/B/C gates below)

The owner has explicitly rejected further A/B/C training as unnecessary for the
desired decision. The comparison stopped at step 43,322; its checkpoint,
optimizer, scheduler and RNG remain recoverable. It is neither a passed gate
nor evidence that the final architecture works. Historical Phase 3/4 tables
and measurements below record the abandoned protocol, not current prerequisites.

The active path jointly trains the encoder, head and generic router. It starts
from this project's scratch-derived supervised weights and a newly initialized
router; no SigLIP2 weights or external teacher are introduced. Because the input
policy changes, this is a new run with a new optimizer/scheduler, not continuation
of the old A optimizer under a silently changed contract. All provenance and
data/initialization hashes are saved in `review_plan.json`.

The initial overview is a fixed 448x448 full-image resize, without an artificial
RGB letterbox/pillarbox. This preserves a fixed compute budget rather than
letting native source resolution dictate the token count. Aspect distortion is
an explicit tradeoff, not a solved extrapolation guarantee. Native local windows
remain square, use a fixed 256x256 canvas, and shift/shrink inside the actual
source bounds to avoid invented local border content. Original image borders
remain real image content. Geometry describes the actual crop after adjustment.

The initial joint budget has 16 generic multiscale candidates, K=4 selected
windows, and one retained random training slot. Dense-oracle router supervision
is refreshed every 16 updates in chunks of four candidates; ordinary steps
transfer/encode only the selected K local inputs. Hard top-K is not falsely
described as differentiable: the tagger learns from classification and the
router learns from the periodically refreshed ranking/diversity objective.
There are no eye-specific candidates, routing rules or manual evidence boxes.
An untrained router can be wrong; retaining exploration prevents exclusive
dependence on its early choices. Oracle frequency and gradient balance remain
measured optimization questions, not evidence of useful learned focus.

Each start initially runs at most 1,000 updates, saves model/router/optimizer/
scheduler/RNG/offset, then stops for owner review. Checkpoints are saved every
250 updates. The owner can continue the same new run after review. The review
budget is not a quality gate or a claim that 1,000 steps suffice for convergence.

Optional CPU diagnostics run the actual learned-router inference path from a
saved joint checkpoint. A single source image panel overlays the selected
native regions and is paired with anonymous TP/FP/FN probabilities; TN is only
a count. No individual tag names or IDs are published. The first annotated
Rating:General example is pinned; missing labels remain a limitation. This
workflow explicitly enables diagnostics, but the owner can disable them.
Generated diagnostic images are not opened or read by the agent.

Acceptance is owner visual review of (a) whether selected regions are useful,
(b) whether the anonymous prediction diagnostic improves the desired decisions,
and (c) whether the fixed overview causes unacceptable distortion-related errors.
Aggregate metrics are supporting diagnostics, not automatic pass/fail rules.
All-tag averages cannot substitute for the small-detail objective. AP/F1 slices
may still be recorded, but paired confidence/power analysis is not a prerequisite
for this owner-directed visual experiment. No A/B/C or learned-vs-random quality
gate is fabricated to unlock this path. Existing legacy gated CLIs remain gated.

Remaining after the first visual review: adjust budgets/optimization if requested,
add bounded repeated/refine visits (current routed inference is one selection
pass), expand reviewed examples/domains, and connect accepted combined artifacts
to the normal inference manager. A live joint-training run is not a completed
production replacement and does not prove superiority to the existing tagger.

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

Arbitrary source resolution does not mean an unbounded model input or an
unbounded full-image decode. The production input contract is a bounded
overview proxy followed by at most `K` region reads from the original or an
aligned tiled pyramid. Extremely large sources use a configurable raw-decode
cap; they must be served by that region-readable representation or skipped,
not silently reduced to the overview and treated as if local evidence survived.

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
random initialization. For any compact evidence region, the tag gradient is
diluted across the whole image, and a learned router and an untrained encoder
each wait on the other. Eye color and colored contact lenses are evaluation
examples, not an anatomical prior or a router specialization. Training is
therefore staged (section 5.1):
self-supervised pretraining, tag training on random and stratified views,
router training from a dense-candidate teacher, then joint fine-tuning.

Pretrained encoders such as SigLIP2 or PE are used only as diagnostic controls
and as a reference upper bound (section 8.2). They are never used for
initialization, and a gap to them does not trigger a switch to pretrained
weights. If the from-scratch encoder does not meet its pre-registered gate,
work stops and the cause is measured (section 9).

Three weight lineages must remain separate:

1. **fixed-384 baseline**: the existing trained encoder adaptation and tag head,
   evaluated unchanged through its original single-image input contract;
2. **multi-view diagnostic control**: a frozen copy of that baseline evaluates
   the original source through its unchanged fixed-384 processor and evaluates
   fixed local canvases independently, while a new zero-initialized local
   residual learns whether those views expose usable evidence; and
3. **FoveaTag mainline**: the original encoder, evidence head, and router are
   randomly initialized and trained without importing any SigLIP2 tagger
   tensor.

The diagnostic control is not an initialization proposal for the mainline. It
answers only whether the proposed source-to-view transformation can add useful
information to a known representation. A positive result de-risks the view
contract; a negative result does not reject the scratch architecture because a
fixed-resolution encoder was never trained to represent the new local-view
distribution. Existing tagger logits are not used as scratch-model labels.

The input contracts intentionally differ. The diagnostic keeps the old
source-to-384 path unchanged so that its initial logits exactly reproduce the
existing tagger. The mainline instead builds a bounded overview proxy, selects
at most `K` source-coordinate regions, and reads only those regions from the
original or tiled pyramid into fixed local canvases. It never feeds a native
full-image patch sequence to the encoder. Consequently, the diagnostic weights
are neither shape-compatible evidence for the final architecture nor a useful
initialization without importing the fixed-resolution assumptions being
replaced.

## 2. Current failure mechanism

The trainable path defaults to
`google/siglip2-so400m-patch16-naflex`
(`backend/core/tagger/siglip2_tagger_model.py`). Each dataset item calls the
processor once, without an explicit patch budget
(`backend/core/tagger/tagger_dataset.py`). The upstream processor default is 256
patches. NaFlex retains aspect ratio, but it still resizes the complete image to
fit that budget; it does not preserve native pixels. For a square image this is
effectively a 256 x 256 patch canvas at patch size 16, irrespective of how large
the source was.

That default is not the strongest project result. The owner's prior tagger
training found NaFlex materially weaker than resizing every input to the fixed
SigLIP2 resolution, especially for extrapolation. Fixed-resolution checkpoints
based on `google/siglip2-so400m-patch14-384` are therefore the primary existing
baseline. NaFlex remains useful only for isolating whether extra global patches
recover detail; it must not stand in for the model FoveaTag has to beat.

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
| SigLIP2 / NaFlex (2025-02) | Variable aspect ratio and patch budget; native aspect ratio is not native-pixel preservation. | Google checkpoints and Transformers implementation are Apache-2.0. | Diagnostic failed-route control. Raising its patch budget is a useful ablation, not the target architecture or primary baseline. |
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

The storage-side contract has two paths. Ordinary sources below a configurable
pixel cap may use the existing full-decode fallback. Sources above the cap use
a persistent tiled pyramid: a bounded proxy supplies the overview and router,
then only the selected source rectangles are decoded from the closest pyramid
level. Header dimensions are read before decoding. If no region-readable form
exists during training, the sample is skipped and the skip is recorded;
downscaling the sole copy to fit the cap is not an acceptable fallback because
it defeats the fine-detail objective. The prototype control runner initially uses a 40 MP
full-decode cap; this is an operational safety value to tune from decode and
coverage measurements, not an architectural resolution limit.

The isolated backend stores aligned 2x levels as independently readable,
lossless RGB PNG tiles. Building that cache from an ordinary JPEG/PNG still
requires one full decode, so cache construction is a preprocessing operation,
not a hidden first-inference fallback. A source above the raw-decode cap is
usable only when this region-readable cache already exists. Staged inference
reads a coarse level for the overview and reads level-zero or another
non-upsampled level only for the router-selected rectangles.

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

The proposals are deliberately category- and anatomy-agnostic. The router has
no face detector, eye prior, object-class branch, tag embedding, or region type
such as `face`, `text`, or `hand`. "Stratified" sampling means uniform coverage
over source-space cells and field-of-view levels, not semantic strata. Eye
color is only one motivating failure case; the same mechanism must discover
any compact evidence that changes an image-level tag margin, including texture,
accessories, text, distant objects, and details not named during design.

At training time, reserve part of the budget for stratified random coverage.
Otherwise an early, incorrect router can permanently hide rare evidence. On a
small fraction of batches, evaluate a denser candidate set and train the router
to rank regions by positive-tag margin improvement relative to the overview-only
prediction. Absolute local confidence is not the target, and unknown labels are
masked before tag identities are reduced into one candidate utility. The dense pass is a
training-only oracle and is detached from the main classifier when producing
router targets. Its cost is one local-encoder pass per dense candidate, so the
candidate count and batch fraction must be set from the Phase 0 timing harness
and reported with training throughput.

Inference is sequential: the overview must be encoded before local views can
be selected, and an optional refinement adds a third dependent stage. Batched
inference therefore runs as staged batches (all overviews, then all selected
views), and latency must be measured per stage rather than derived from the
pixel counts below.

No stage concatenates all source patches into one attention sequence. With
`T_o` overview tokens, `T_l` tokens per local canvas, and fixed budget `K`,
ordinary self-attention is `O(T_o^2 + K T_l^2)`, rather than quadratic in the
source pixel count or in one combined high-resolution sequence. View rendering
and local encoding grow approximately linearly with `K`; `T_o`, `T_l`, and `K`
are independently capped in both training and inference.

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
        + temperature * logmeanexp_j(dot(evidence_j, local_weight_t) / temperature)
```

The local reduction lets different tags select different views. It also exposes
the highest-contributing source rectangle for debugging. Quality and Rating keep
their current group-aware selection rules downstream; the architecture still
produces ordinary per-tag logits. The normalization by the number of valid
slots is required: adding views with zero local evidence must not shift every
tag logit merely because the inference budget changed.

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
2. **Tag training on random and spatial/scale-stratified local views.** No
   learned router, detector, or semantic region sampler is used; the strata are
   source-space cells and field-of-view levels only.
   The encoder and evidence head are established first. Ground-truth
   illustration tags are the main supervised source. Real-photo captions have
   mixed provenance: some were corrected by hand and some originate from an
   earlier tagger, but the corrected subset is not identified exhaustively.
   Photo samples are therefore domain-balanced rather than allowed to vanish
   inside the much larger illustration corpus, and use a separately reported
   loss weight. Fine-detail photo negatives are masked unless their provenance
   is known; observed positives remain weak supervision. Unlabeled photo views
   still participate in the self-supervised and cross-view objectives.
   The existing ASL reduction multiplies the mask and then averages over every
   batch-by-tag element; it does not renormalize by active mask weight. Phase 3
   records active-label fraction, mean mask weight, and weighted positive count
   from a bounded deterministic sample of each domain. This keeps the historical
   loss definition while making any extra photo-gradient dilution measurable.
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
reason to sample a useful compact region rather than a background patch. Random crops remain
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
- aspect-ratio extremes;
- a Danbooru-derived illustration slice whose image-level tags are treated as
  ground labels for the illustration domain; and
- a deterministic weak-label slice built from the existing image-level tags.
  Manual evidence boxes are not a prerequisite because their annotation cost
  is not justified for Phase 0.

The photo benchmark fixes the manifest, tag normalization, negative cohort, and
split assignment. Its captions have mixed provenance: part was manually
corrected, but the checked subset and completeness are not recorded per item.
It cannot establish where the visual evidence lies, and a missing caption tag
is not a guaranteed negative. Results are therefore partially curated
weak-label measurements, not complete ground truth. Report macro
average precision and F1 by tag group, tag prevalence, and source resolution.
Also report calibration error, prediction stability across the resolution
ladder, dense-oracle utility recovery and oracle top-`K` recall, images/second,
batch-size scaling, peak VRAM, and host preprocessing time. The dense oracle
scores candidate regions with the image-level tag loss during evaluation; it
does not require human boxes and is not available at production inference.

### 8.2 Required baselines

1. the strongest existing fixed-resolution SigLIP2 tagger at its trained input
   size; this is the primary production baseline because earlier project
   training found materially better accuracy and extrapolation than NaFlex;
2. that fixed-resolution checkpoint over the source-resolution ladder, retaining
   its trained resize contract;
3. NaFlex at 256/512/1,024 patches as a diagnostic failed-architecture control,
   not as the primary baseline;
4. the complete trained fixed-resolution tagger with its original source-to-384
   overview path unchanged, plus a zero-initialized local residual fed by
   random crops through the same frozen encoder/LoRA;
5. scratch FoveaTag overview only;
6. scratch FoveaTag with overview-derived placebo local views: use the same
   coordinates, local-view count, network shape, and encoded-pixel budget as
   baseline 7, but crop from the already reduced overview so no source detail
   is restored. This branch reads only the overview from the source; it samples
   the shared local specifications without first reading and discarding native
   local regions;
7. scratch FoveaTag with source-native random local views, forked from the same
   self-supervised checkpoint as baselines 5 and 6 and trained on the same data
   and optimizer steps;
8. FoveaTag with learned routing; and
9. one permissively usable recent encoder benchmark, such as PE (Apache-2.0
   checkpoints), with the
   same downstream evidence head when technically possible.

Baselines 4 and 9 are **pretrained diagnostic controls**, not candidate
initializations. Baseline 4 must reproduce baseline 1 before any local update:
the same source image enters the same fixed-384 processor, trained encoder/LoRA,
and trained classifier. Only the added local residual is new. This can show
that selected source crops contain information usable by the old
representation, but it retains that tagger's blind spots and cannot establish
the value of the scratch encoder. Baseline 9 supplies a second external
reference under the same restriction. No tensor from either control enters the
FoveaTag mainline.

The operational input-architecture comparison is baseline 5 versus 7. Both start
from the same randomly initialized FoveaTag encoder after the same
self-supervised pretraining; only the presence of local views differs. This
paired scratch comparison prevents a pretrained tagger's errors or semantic
head from deciding whether the new input scheme works. Encoder internals
(continuous coordinate RoPE, scale features, and stem design) are ablated only
within this scratch lineage. The stricter causal comparison is baseline 6
versus 7: architecture, local-token count, trainable parameter count, and
encoder work are held fixed, while only access to source-native detail changes.
Baseline 5 remains necessary to measure whether the accuracy gain justifies the
additional inference cost.

The three scratch branches also share an immutable data contract. The first
branch records the dataset-database identity, complete holdout-manifest hashes,
and ordered train/validation hashes over domain, resolved source path, source
file size/mtime, and canonical supervision tags. Later branches must reproduce
that contract exactly. Each branch recomputes it before validation, so a changed
database, caption set, selected source, source file, split order, holdout, seed,
or source-pixel cap fails the experiment instead of becoming an input-mode
effect. The contract hash is included in each inference artifact and the paired
report rejects branches with different hashes.
The same comparison root separately freezes epochs, presented samples, batch
size, worker count, view budget, optimizer and scheduler values, ASL parameters,
photo policy, seed, and source-pixel cap. This shared training-contract hash is
also carried by every branch artifact and paired report. Matching weights and
data are therefore insufficient if any branch received a different optimization
budget.

The router, local `2x` magnification, continuous scale feature, dense-oracle
frequency, and local `logsumexp` head each require an ablation. Local-view
teacher distillation (section 5.1) is compared against pure self-supervised
pretraining at equal images presented and optimizer steps.

### 8.3 Historical numeric gate proposal (superseded by owner visual review)

The abandoned staged protocol proposed the following requirements. They are
not prerequisites for the current experimental joint-review API; the current
owner-directed acceptance process is specified in section 1:

- a material gain on the ground-illustration fine-detail slice at matched
  false-positive rate;
- no regression on the partially curated photo slice, reported separately and
  treated as supporting rather than decisive evidence;
- no material loss on the coarse slice;
- stable or improving results on held-out source resolutions above the training
  range;
- learned routing recovers enough of the positive dense-oracle utility at the
  same `K` that candidate selection is not the dominant remaining error;
- balanced inference within the owner-approved latency and VRAM budget; and
- an explicit checkpoint license/provenance decision.

No numeric quality or latency threshold is set here because it has not been
measured on project hardware.

## 9. Implementation phases

| Phase | Work | Exit condition |
|---|---|---|
| 0 | Add the fixed weak-label benchmark and timing harness. Measure current 256-patch and enlarged-patch SigLIP2, and time per epoch of candidate from-scratch model sizes. | The current failure is quantified by tag group and resolution ladder; the self-supervised pretraining budget is set. |
| 1 | Pretrained diagnostic control: preserve the complete trained fixed-384 source-to-logit path and add random local crops only through a zero-initialized residual. Do not reuse these weights in FoveaTag. | Measure whether the old representation can use local evidence and its cost. This is diagnostic and does not gate the scratch architecture. |
| 2 | Original shared encoder (25--50M) with the continuous coordinate/scale contract; self-supervised pretraining (stage 1). | Pretraining losses converge; overlap correspondence and chroma reconstruction are measured on held-out images; and the exact encoder proposed for Phase 3 passes a hash-bound review against its same-seed scratch feature-health reference. |
| 3 | Tag training on overview-only, overview-derived placebo-local, and source-native random/stratified views (stage 2), with all branches forked from the same accepted scratch-pretrained checkpoint. | Source-native multi-view beats the compute-matched placebo, improves sufficiently over overview-only to justify its cost, and meets the operational gate against the untouched fixed-384 baseline, including resolution-ladder extrapolation. If not, stop and measure the cause (pretraining objectives, model capacity, router timing, and missing semantic knowledge via the local-view teacher distillation ablation); do not switch to pretrained weights. |
| 4 | Router training from dense candidates (stage 3), then joint fine-tuning with retained random coverage (stage 4). | Learned routing beats random views at the same `K`; dense-oracle utility recovery, oracle top-`K` recall, and rare-tag results are measured. |
| 5 | Add checkpoint bundles, resume, inference manager, export, API, defaults, OpenAPI, and UI. | Existing tagger lifecycle tests plus new view-contract tests pass. |
| 6 | Optional capacity increase and further objective ablations. | Retain only components with isolated improvements. |

Phase 3 refuses to start without an explicit Phase-2 pass artifact. That
artifact binds the exact encoder and metadata plus both held-out manifest
files and indexes, and the completed-run summary to the post-run encoder
evaluation. It rejects an incomplete run and an encoder that is not the
completed run's best held-out checkpoint. The evaluation compares
masked-local and overview correspondence, variance, pairwise cosine, and rank
with the same-seed untrained initialization. There is no post-hoc automatic
rank threshold: the reviewer records the reason for accepting or rejecting the
measured representation, and any changed evidence invalidates the decision.
The bounded feature-health token budget is distributed deterministically over
every held-out batch. It is not filled from the first batch alone, which would
make the collapse evidence depend on one narrow prefix of each domain.
The evaluation also records a canonical SHA-256 over the same-seed scratch
encoder's named tensors, dtypes, shapes, and bytes, so the exact untrained
reference is identifiable rather than implied only by a seed value.
Gate creation and every downstream gate load re-parse the encoder metadata and
completed summary rather than trusting paths copied into the evaluation or a
handwritten pass JSON. They require the same run
contract, best held-out loss, epoch and step identity, finite correspondence,
reconstruction and chroma metrics, and exactly the validation manifests named
by that contract.

The Phase 3 gate is set before the paired scratch runs, so the criterion cannot
be adjusted to fit their result. Phase 1 can inform feasibility and cost, but
its pretrained weights and result do not define that causal gate.

Because the gate has no post-hoc numeric materiality threshold, its final
pass/fail remains an explicit repository-owner decision. The decision artifact
records the rationale and hashes the completed evaluation summary, manifests,
fixed-baseline reports, all A/B/C predictions, and paired reports. Phase 4 must
refuse a failed decision or any evidence changed after the decision. Each
prediction also records the model and metadata SHA-256; the passed gate must
resolve one identical source-native model hash across both benchmarks before
that model can become the frozen Phase-4 teacher.
Gate creation and every load rebuild each paired report from its registered
manifest, fixed baseline, and exact three branch predictions. The rebuilt
report must be data-equivalent to the saved JSON, every benchmark must name the
same source-native artifact, and the gate evidence set may contain neither
omissions nor handwritten substitutes.

Phase 4 has a second explicit repository-owner gate before any production
trainer, manager, API, or UI integration. Its decision artifact hashes the
complete Phase-4 evaluation summary and, for every registered benchmark, the
manifest, fixed baseline, freshly generated source/routed predictions, and
comparison report. A passed decision must resolve one identical routed model
and metadata hash across all benchmarks. Router utility recovery alone cannot
pass this gate: the owner decision also considers fine-detail and coarse-tag
quality, extrapolation across the resolution ladder, and measured staged
latency/VRAM.
Gate creation and every load rebuild each routed comparison from the registered
manifest, fixed baseline, source-native predictions, and routed predictions.
They require one source and one routed artifact across all benchmarks, an
inference mode matching the completed summary, and an exact evidence set.
The production boundary must call `validate_phase4_artifact`; it reloads the
passed gate, compares the candidate joint weights and metadata hashes with the
accepted routed predictions, and revalidates pure-scratch joint provenance and
learned-over-random router utility before returning the artifact metadata.

Phase 0 measurement must precede expensive training and product integration.
Implementation of the isolated benchmark, model, and loss components may happen
earlier, but no architectural success claim follows from unit tests alone. The
project does not currently have measured evidence that distinguishes a better
high-resolution architecture from a model that merely changes thresholds or
prevalence bias.

### 9.1 Implementation status (2026-09-29)

The isolated implementation now lives in `backend/core/tagger/foveatag/`. It is
not yet selectable through the production trainer or inference API. The current
status separates implemented software contracts from experiments that still
require data and GPU time:

| Phase | Implemented | Still required before exit |
|---|---|---|
| 0 | Versioned box-free partially curated photo benchmark, deterministic Danbooru ground-illustration benchmark, dense-oracle recovery metric, resolution/group/stability metrics, saved predictions, per-stage timing summaries, synthetic scratch-model GPU timing, representative data-loaded optimizer timing, fixed-resolution agreement baselines, and diagnostic NaFlex measurements. | Complete. The scratch pretraining budget is fixed below. |
| 1 | Deterministic multi-scale views, exact source transforms, expandable evidence head, fixed/NaFlex SigLIP2 token adapter, a processor bridge that preserves overview/local view axes, fixed-encoder inference and optimizer cost, masked-label training and streaming validation, domain-balanced ground/photo sampling with benchmark exclusion, contract-bound epoch resume, deterministic benchmark prediction from compact best weights, an evidence-bound all-benchmark evaluation runner, matched-FPR comparison, a baseline-preserving zero-initialized local residual, a pre-decode safety cap for the full-decode fallback, and an isolated region-readable tiled-pyramid builder/reader. The first confounded 24,000-sample pilot and the corrected residual smoke are measured in section 9.12. | Run the pre-registered corrected residual diagnostic. Dataset runs can admit an over-cap source only through a validated prebuilt pyramid bound into the manifest contract. |
| 2 | A 37.39M-parameter default scratch encoder, content-aware continuous 2-D RoPE, stride-16 local map, low-token global stage, masked RGB decoder, teacher-free losses, the diagnostic local-teacher projection, deterministic local/overview data, bounded indexed manifests, domain-balanced sampling, step-resumable AdamW state, held-out validation, separate bounded collapse-sensitive statistics for masked-local and overview features, a same-seed scratch-initialization reference, encoder-only scratch artifacts, and an explicit hash-bound Phase-2 decision artifact. | Run the pre-registered 10-epoch convergence measurement and retain the best held-out overlap/chroma checkpoint. Evaluate that exact encoder against the same-seed initialization, confirm that falling cosine loss does not coincide with vanishing channel variance, rank collapse, or near-unit mean pairwise cosine in either view, and record the Phase-2 decision. |
| 3 | End-to-end scratch tagger, masked classifier composition, density-gated prediction consistency, an adapter from the existing vocabulary/loss-mask lifecycle to deterministic FoveaTag views, explicit overview-only/placebo-local/source-native modes, a mandatory Phase-2 encoder gate, exact shared initialization plus immutable shared-data and shared-training artifacts, stateless domain-balanced sampling, step-resumable supervised training and validation, a fixed-order A/B/C runner that resumes each branch and does not advance past an incomplete branch, resolution-ladder prediction artifacts using the same view builder as training, measured model-inclusive peak VRAM, a paired report that rejects mismatched initialization, data, optimization budget, lineage, or prediction coverage before computing matched-FPR evidence, a preflighted evaluation runner that regenerates all three branches for each benchmark before writing that report, and a real-image A/B/C throughput harness that includes decode, view rendering, transfer, full forward/backward, clipping, and AdamW update. | After Phase 2 produces and accepts its best held-out encoder, run the throughput harness while no other GPU job is active and freeze the Phase-3 budget before branch A reports quality. Then run the pre-registered A/B/C comparison from the shared initialization, data, and training contracts and evaluate both benchmark manifests. Connect live-refresh and the main trainer lifecycle only after the gate passes. |
| 4 | Generic scale-space router, padding mask, NMS, dense-oracle targets, ranking/diversity losses, retained random-budget selection, box-free oracle recovery metrics, a source-native data bridge that renders fixed-count spatial/scale-stratified candidates without semantic strata while preserving the photo loss mask, a mixed-precision router trainer whose frozen stage-2 teacher encodes the overview once and evaluates independent candidates in fixed-size chunks, fixed-`K` learned-versus-random utility-recovery validation, compact resumable/router-only artifacts bound to the exact stage-2 tagger and metadata hashes without duplicating its weights, exact-offset epoch segments that checkpoint both on schedule and on requested stops, and a full dataset/validation CLI that refuses to run unless the Phase 3 decision and all hashed evidence still pass. The stage-4 joint trainer updates the tagger from only the selected fixed-`K` views, keeps a non-annealed random part of that budget, refreshes router ranking/diversity on a configurable dense-oracle interval, and chooses epoch artifacts using deterministic learned-routing validation with no random slots. Dense local pixels remain in pinned host memory during overview routing: ordinary joint steps transfer only the selected `K`, while oracle steps transfer one configured candidate chunk at a time. Its exact-resume checkpoint stores both changing modules, optimizer/scheduler state, RNG state, and epoch offset under one lineage contract. The joint experiment CLI requires the same Phase-3 hashes, candidate count, and `K` used by the accepted router artifact; refuses a router that did not beat random utility recovery; rechecks that condition after every joint epoch; and exports a combined model/router artifact only from an epoch that still passes. | After Phase 3 passes, measure router training cost and run the gate-bound router CLI; choose the standalone sample budget and joint-training dense-oracle interval from that measurement, then run both Phase-4 stages and evaluate the combined artifact on the registered benchmark slices. |
| 5 | Versioned inference/resume artifacts, exclusion of diagnostic weights from inference artifacts, coordinate-aware collation, cached-overview staged inference, optional BF16/FP16 autocast, per-stage wall time, selected source-region records, a lossless persistent tiled-pyramid source that reads only the overview and router-selected rectangles, and a resolution-ladder prediction runner that accepts only combined joint artifacts whose router still beats random at the inference mode's exact `K`. A Phase-4 comparison report binds the joint artifact to its Phase-3 source hash, rejects missing/duplicate coverage and malformed or wrong-count selected boxes, and reports routed-versus-source and routed-versus-fixed results by tag group at matched FPR. The Phase-4 evaluation runner preflights the passed Phase-3 gate, both artifacts, every manifest, and every fixed-baseline report before regenerating source and routed predictions for all registered benchmark slices. | After the go/no-go gates pass, connect pyramid-cache resolution and creation to the production manager/export path, then add API/defaults/OpenAPI/frontend surfaces and lifecycle tests. |
| 6 | Not started. | Run registered capacity and component ablations only after the smaller model is measured. |

More than one hundred focused CPU tests cover the isolated modules. After the owner's
separate training run released the GPU, the synthetic measurement below was
also run. No backend restart was performed. Quality and extrapolation
measurements for the FoveaTag controls and scratch encoder remain pending.

### 9.2 Initial synthetic GPU measurement (2026-09-29)

`gpu_benchmark.py` measured the default scratch model on one NVIDIA RTX 6000
Ada using PyTorch 2.10.0, CUDA 13.0, BF16, batch size 1, and 10,000 output tags.
Each inference result uses 5 warmups and 20 measured forwards; each training
result uses 2 warmups and 5 measured forward+BCE+backward steps. The training
number excludes optimizer update, data loading, view rendering, and router
selection, so it is not an epoch-time estimate. Peak memory likewise excludes
optimizer state. The machine-readable result is
`docs/audits/FOVEATAG_GPU_BENCHMARK_2026-09-29.json`.

| Mode | Encoded MP | Forward mean / p95 | Forward+backward mean | Peak allocated / reserved |
|---|---:|---:|---:|---:|
| fast | 0.279 | 48.0 / 54.3 ms | 106.6 ms | 628 / 732 MiB |
| balanced | 0.463 | 46.4 / 50.9 ms | 104.2 ms | 964 / 1,186 MiB |
| detail | 0.725 | 46.9 / 50.2 ms | 105.9 ms | 1,415 / 1,666 MiB |

The three latency means are too close to claim scaling behavior from this
short synthetic run; batched local views can improve device utilization as the
view count rises. Memory does increase with the view budget. These numbers show
that the default model fits with substantial margin on this GPU, but they do
not satisfy a phase gate or establish accuracy.

### 9.3 Initial data-loaded pretraining measurement (2026-09-29)

`pretraining_benchmark.py` originally measured source-image decode, two
overlapping 256-pixel local views at independently jittered fields of view,
host-to-device transfer, BF16 autocast forward/backward, and an AdamW update
with FP32 parameters and optimizer state. Ten measured steps followed three
warmups on the same RTX 6000 Ada.

| Registered slice | Batch / workers | Mean end-to-end step | Median end-to-end step | Estimated one epoch from mean | Peak reserved VRAM |
|---|---:|---:|---:|---:|---:|
| cosplay photos, 21,499 eligible images | 2 / 2 | 127.0 ms | 128.9 ms | 0.38 h | 980 MiB |
| character illustrations, 1,546,928 eligible images | 2 / 2 | 138.6 ms | 140.6 ms | 29.8 h | 980 MiB |
| character illustrations | 8 / 4 | 660.5 ms | 147.6 ms | 35.5 h | 2,804 MiB |
| character illustrations | 16 / 8 | 695.1 ms | 116.6 ms | 18.7 h | 5,246 MiB |
| character illustrations, deterministic hashed sample | 16 / 8 | 177.5 ms | 174.2 ms | 4.77 h | 5,246 MiB |

The larger-batch means contain a single roughly five-second data wait in each
ten-step window, while their medians show that GPU compute scales well. Native
image decode and view production, especially for very large source files, can
therefore dominate wall time even when average GPU work is cheap. The sample is
the first bounded database window rather than a shuffled population sample, so
the mean-derived epoch estimates are provisional and are not yet a pretraining
budget. The saved JSON reports preserve data wait, transfer, optimizer-step,
loss, and memory measurements separately. The loss change over ten updates is
only a smoke signal and is not a convergence claim.

The final row replaced that provisional estimate for the original local/local
implementation. It uses 2,048 paths selected
across the eligible ID range by a deterministic integer hash, ten warmup steps,
and fifty measured steps. Prefetch was warm before measurement: data wait was
0.25 ms mean, optimizer work was 162.2 ms mean, and end-to-end p95 was 214.1 ms.
The code audit before the long run found that this path did not exercise the
bounded overview input required by the design: both branches were local crops.
The planned objective now pairs one source-native 256-pixel local view with the
full 448-pixel overview, reconstructs the local branch, and aligns overlapping
source-coordinate cells. The table therefore remains historical Phase-0
evidence and its 47.7-hour estimate is not the budget for the corrected run.

The corrected path was then measured on the same GPU and deterministic hashed
illustration sample, with a 40 MP pre-decode cap:

| Batch / workers | Mean step | Estimated epoch | Peak reserved VRAM |
|---:|---:|---:|---:|
| 8 / 8 | 145.3 ms | 7.81 h | 5,940 MiB |
| 16 / 8 | 230.6 ms | 6.19 h | 11,544 MiB |
| 32 / 8 | 428.2 ms | 5.75 h | 22,552 MiB |

Batch 16 is the initial long-run choice. Batch 32 saves only about 7% of epoch
time while doubling reserved memory, leaving less margin for validation and
future objective state. In the repeated 50-step batch-16 run, total loss moved
from 1.0975 to 0.3449. Mean correspondence, RGB reconstruction, and chroma
losses were 0.1801, 0.3648, and 0.1131 respectively, confirming that the
overview/local correspondence path is active rather than silently producing a
zero term. This remains a connectivity and cost measurement, not convergence.

The initial teacher-free scratch pretraining budget remains **10 epochs**
(15.47 million source images, approximately 61.9 GPU-hours at batch 16).
Checkpoint and held-out overlap/chroma losses are evaluated each epoch.
Extension beyond 10 epochs requires continued held-out improvement and a gain
in the downstream random-view tag pilot; the pre-registered cap is 20 epochs
(approximately 95.4 GPU-hours). This fixes the budget before Phase 2 quality is
known while allowing a bounded continuation rather than an open-ended run.

The replacement anti-collapse run also has an early-stop policy registered
before its epoch-1 validation result. It cannot stop before five complete
epochs. After that point, two consecutive complete epochs must each improve
weighted fixed-validation loss by less than 0.5% relative to the preceding
best, and none of correspondence, RGB reconstruction, or chroma may improve by
at least 1% relative to its preceding best. The selected best encoder must, in
every domain and both views, retain channel standard deviation at least 0.5,
mean pairwise cosine at most 0.97, and effective-rank fraction at least 0.10;
effective rank may not fall more than 25% from the best healthy epoch. A health
failure is a failed/reviewed run, not a successful early stop. The resumable
checkpoint is retained so a downstream Phase-3 failure can resume the original
ten-epoch schedule. Exact semantics and the bound run-contract hash are in
`FOVEATAG_PHASE2_EARLY_STOP_POLICY_2026-09-30.json`. The current gate continues
to require ten epochs until an early-stop candidate exists and the gate is
changed to verify this policy and its complete evidence rather than merely
accepting an incomplete summary.

The registered 40 MP filter excludes 23 of 21,499 minimum-side-eligible photo
rows and 130 of 1,546,928 illustration rows in the current database. The
resulting 21,476 and 1,546,798 rows reconcile exactly with the manifest counts
before benchmark-path exclusion. The active run's immutable manifest predates
this telemetry field and is not modified in place; newly built manifests record
the minimum-side count, over-cap count, and post-cap count per domain in their
contract. They remain excluded from the immutable active run; a newly built
manifest may admit them only through the validated cache path described below.

The cache boundary is implemented without changing the active run. A normalized
absolute source path maps to a deterministic cache directory; resolution checks
the original path, byte length, and nanosecond mtime recorded at pyramid
creation. The lossless format records every tile's SHA-256 and verifies each
tile when it is read. `FoveaImagePathDataset` renders the local and overview
tensors through the same coordinate-preserving `RegionImageSource` path as
staged inference. Indexed Phase-2 manifests can admit over-cap paths only when a
relevant cache validates. Their immutable request binds the absolute cache root,
relevant-entry count, and combined cache-manifest hash; training and later
evaluation recover the root from that contract. Missing caches remain excluded,
whereas a bound cache that later fails integrity checks stops the run instead of
silently changing its effective sample set. Omitting the cache option adds no
request field, so the current long run and its resume contract remain unchanged.

### 9.4 Diagnostic NaFlex patch-budget timing (2026-09-29)

The available 97,199-tag NaFlex checkpoint was measured on eight registered cosplay
photos at batch size 1. Each budget used exactly the requested number of
effective patches. The runner includes native decode, processor work, transfer,
and BF16 forward, but excludes probability sorting and tag filtering.

| Patch budget | Decode mean | Preprocess mean | Forward mean | End-to-end mean | Peak allocated / reserved |
|---:|---:|---:|---:|---:|---:|
| 256 | 53.6 ms | 14.5 ms | 20.0 ms | 88.4 ms | 3,700 / 3,870 MiB |
| 512 | 53.2 ms | 14.7 ms | 20.3 ms | 88.5 ms | 3,704 / 4,250 MiB |
| 1,024 | 51.4 ms | 16.2 ms | 20.0 ms | 88.0 ms | 3,714 / 4,250 MiB |

This is a diagnostic of the previously underperforming NaFlex route, not the
primary quality baseline. Earlier project training found that the
fixed-resolution resize model generalized better; FoveaTag exists in part to
retain that behavior while recovering local detail. This small run does not
establish that larger patch budgets are free: the
97,199-way classification head and source decode dominate single-image wall
time, and the sample is too small for a scaling claim. It does establish the
same-checkpoint latency/VRAM reference required before quality measurements.
Loading this older checkpoint also exposed two compatibility defects now fixed:
its omitted custom-pooler dimensions are recovered from tensor shapes, and the
current Transformers NaFlex argument name `pixel_attention_mask` is used.

### 9.5 Preliminary NaFlex encoder-swap timing (2026-09-29)

The same NaFlex encoder was reused with a 512-dimensional FoveaTag
evidence head and 97,199 output tags. The overview budget was 784 patches and
each local view used 256 patches. Eight cosplay photos were measured at batch
size 1; the encoder was frozen and the newly initialized head was used only to
measure execution cost, not quality.

| Local views | Total encoded patches | View render mean | Processor mean | Forward mean | End-to-end mean | Peak reserved VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 784 | 86.0 ms | 2.2 ms | 17.4 ms | 105.9 ms | 2,430 MiB |
| 1 | 1,040 | 93.3 ms | 2.8 ms | 34.4 ms | 131.0 ms | 2,434 MiB |
| 4 | 1,808 | 112.5 ms | 4.7 ms | 37.4 ms | 155.3 ms | 2,434 MiB |

These numbers only bound the multi-view implementation cost and do not select
the Phase 1 encoder. The fixed-resolution encoder must be used for the primary
control. The four local views are encoded together, so their forward cost is not four
times the single-local cost. End-to-end latency rises by about 49 ms from the
overview-only control to four local views on this sample. Native decode and PIL
view rendering remain a larger component than the GPU forward. The quality
gate remains open until this control is trained and measured on the fixed
fine-detail benchmark.

### 9.6 Fixed partially curated photo benchmark (2026-09-29)

`weak_benchmark.py` deterministically constructed
`docs/audits/FOVEATAG_WEAK_BENCHMARK_PHOTO_V1.json` from the registered cosplay
photo dataset. The manifest contains 674 images, including 128 negative-cohort
images. Eighteen fine-detail tags are represented: eleven eye colors,
heterochromia, earrings, eyeliner, eyeshadow, lipstick, logo, and nail polish.
The builder requested colored contacts, mascara, painted nails, and text as
well, but those normalized tags were absent from this dataset and are recorded
as uncovered rather than silently substituted.

Each item carries the existing image-level tags and a deterministic
source-resolution ladder; no evidence boxes are required. The manifest uses
`evidence_mode: dense_oracle`, so later router evaluation compares selected
candidate utility against exhaustive candidate scoring at the same `K`.
Some source captions were manually corrected, while others retain prior-tagger
labels; the database does not identify the checked subset. Because correction
coverage is unknown, caption absence is only a weak negative, and tag frequency
is capped by construction, this suite is suitable for controlled supporting
comparisons but not for claiming an absolute real-world error rate.

### 9.7 Weak-label NaFlex diagnostic (2026-09-29)

The available 97,199-tag NaFlex checkpoint was evaluated on all 674 benchmark
images at every non-upscaled ladder point and at native resolution: 3,956
image-resolution conditions and 11,868 forwards across three patch budgets.
The checkpoint vocabulary contained every tag used by the manifest. Metrics
below use the checkpoint's existing threshold of 0.35; average precision is
threshold-independent.

| Patch budget | 384 fine AP / F1 | Native fine AP / F1 | Native coarse AP / F1 | Forward mean / p95 |
|---:|---:|---:|---:|---:|
| 256 | 0.6219 / 0.4025 | 0.6240 / 0.3964 | 0.8988 / 0.6266 | 20.0 / 25.1 ms |
| 512 | 0.6687 / 0.4336 | 0.6770 / 0.4196 | 0.9054 / 0.6278 | 19.9 / 24.9 ms |
| 1,024 | 0.6853 / 0.4303 | 0.7003 / 0.4202 | 0.9039 / 0.6319 | 21.0 / 24.8 ms |

These results do not establish the production baseline because the checkpoint
belongs to the NaFlex training route that previously underperformed the fixed
resize route. Increasing the patch budget improves native fine-detail macro AP by 0.0530 at
512 and 0.0763 at 1,024 relative to 256, without a coarse-tag AP regression in
this weak-label slice. Within a fixed patch budget, moving from the 384-pixel
ladder point to the native source produces much less improvement. For example,
the 1,024-patch model moves from 0.6853 to 0.7003 fine-detail AP. This supports
the premise that a bounded global token budget, rather than source decode alone,
limits recoverable detail inside NaFlex. It is not the FoveaTag acceptance
floor; section 9.8 supplies that floor.

The reports also compare every ladder score with its same-image native score.
At 384 pixels, fine-tag mean absolute score delta / threshold agreement were
0.0066 / 98.05% at 256 patches, 0.0141 / 95.83% at 512, and 0.0172 / 94.93% at
1,024. The full per-resolution, per-tag, stability, and timing results are saved
under `docs/audits/foveatag_weak_predictions/`. Decode and resize were shared
during the sweep but copied into each prediction row as standalone stage-cost
attribution; they must not be summed across rows to estimate total sweep time.

### 9.8 Primary fixed-resolution SigLIP2 baseline (2026-09-29)

After correcting the baseline choice, the strongest available fixed-resolution
checkpoint was evaluated on the same 674-image ladder. It uses
`google/siglip2-so400m-patch14-384`, 107,926 tags, and its recorded best
threshold of 0.41. Every ladder input is processed through the checkpoint's
trained fixed-384 resize contract.

| Input ladder point | Fine AP / F1 | Coarse AP / F1 | Fine score delta to native / threshold agreement |
|---:|---:|---:|---:|
| 384 | 0.7040 / 0.5641 | 0.9229 / 0.7351 | 0.0284 / 94.26% |
| 512 | 0.7086 / 0.5570 | 0.9198 / 0.7379 | 0.0212 / 95.62% |
| 768 | 0.7196 / 0.5483 | 0.9248 / 0.7362 | 0.0089 / 98.12% |
| native | **0.7209 / 0.5382** | **0.9236 / 0.7332** | 0 / 100% |

On the partially curated photo labels, the native fixed-resolution result exceeds the diagnostic NaFlex checkpoint
even at 1,024 patches: fine AP is 0.7209 versus 0.7003, fine F1 is 0.5382 versus
0.4202, coarse AP is 0.9236 versus 0.9039, and coarse F1 is 0.7332 versus
0.6319. This agrees with the owner's earlier training result and makes the
fixed-resolution checkpoint the operational photo agreement reference. It is
not by itself the Phase 3 quality gate. The values are operational model
comparisons, not a controlled architecture ablation:
the checkpoints differ in training duration, vocabulary, pooling, and training
method and some benchmark labels may descend from either model family. The
decisive matched-FPR gate is set on the ground-illustration slice; this photo
slice must be reported separately and must not regress. A gain over NaFlex
alone is insufficient.

Fixed-model forward mean / p95 was 30.3 / 37.4 ms over 3,956 conditions. The
complete weak-label report is
`docs/audits/foveatag_fixed_predictions/siglip2_fixed_report.json`.

### 9.9 Fixed-encoder Phase 1 cost (2026-09-29)

The Phase 1 multi-view control was rerun with the primary fixed-384 encoder and
107,926 output tags. Eight real photos were measured at batch size 1. The
overview-only, one-local, and four-local forward means were 30.5, 59.9, and
72.4 ms; end-to-end means including decode and PIL view construction were
113.9, 152.9, and 179.2 ms. Four-local inference reserved 2,622 MiB.

An actual BF16 optimization smoke then used four local views, a frozen encoder,
masked ASL, gradient clipping, and AdamW over the 123.95M trainable pooler,
projection, and evidence-head parameters. After two warmups, eight measured
steps averaged 93.4 ms (94.8 ms max) and reserved 4,750 MiB. Views were prepared
before timing, and the single synthetic positive label only exercises the full
107,926-way loss; the observed loss change is not a quality or convergence
result. These measurements establish that the correct fixed-encoder control
fits the project GPU. The saved reports are
`docs/audits/FOVEATAG_SIGLIP2_FIXED_CONTROL_BENCHMARK_2026-09-29.json` and
`docs/audits/FOVEATAG_SIGLIP2_FIXED_CONTROL_TRAINING_BENCHMARK_2026-09-29.json`.

The `4.39e-2` to `2.70e-2` loss in that timing smoke is not a different ASL
definition from the existing trainer. Both paths instantiate
`AsymmetricLossOptimized` with `gamma_neg=4`, `gamma_pos=1`, `clip=0.05`, and
global batch-by-tag mean reduction. The timing model discarded the trained
classifier, initialized a new 123.95M-parameter evidence head, then ran only two
warmup and eight measured updates on an artificial target containing one
positive and 107,925 negatives. At zero logits, one ASL negative contributes
about `2.45e-2`, so a loss in the `1e-2` range is expected before the random
classifier learns negative priors. It is not comparable with the `1e-4` range
from a converged SigLIP tagger. The real-data random-head pilot reached
`6.37e-4` after 6,000 optimizer steps, while the baseline-preserving residual
smoke started at `7.77e-5` because its existing classifier was already trained.
The ongoing Phase 2 values are also incomparable: they are the weighted sum of
correspondence, masked reconstruction, and chroma reconstruction losses, not
ASL. Phase 3 must compare loss only at matched initialization, targets, masks,
and training step; AP/F1 remains the model-quality criterion.

### 9.10 Danbooru ground-illustration baseline (2026-09-29)

The decisive illustration-domain slice is
`docs/audits/FOVEATAG_GROUND_BENCHMARK_DANBOORU_V1.json`. It was selected
deterministically from a bounded 200,000-row scan of registered dataset 29 and
contains 2,921 images: 512 negative-cohort images and up to 128 positives for
each of 19 covered fine-detail tags. The registered Danbooru tags are treated
as ground labels for this illustration-domain benchmark. They do not establish
photo-domain accuracy; the separate 674-photo slice remains the supporting
extrapolation and no-regression check. Colored-contact, contact-lens,
painted-nail, and text tags were requested but absent after normalization, so
the manifest records them as uncovered.

The primary fixed-384 checkpoint was evaluated at its recorded threshold of
0.41 over 16,100 non-upscaled image-resolution conditions. Every condition
still enters the checkpoint through its trained fixed-384 processor contract.

| Input ladder point | Fine AP / F1 | Coarse AP / F1 |
|---:|---:|---:|
| 384 | 0.5878 / 0.5091 | 0.7653 / 0.6525 |
| 512 | 0.5920 / 0.5128 | 0.7705 / 0.6578 |
| 768 | 0.5967 / 0.5084 | 0.7740 / 0.6591 |
| 1,024 | 0.5934 / 0.5055 | 0.7703 / 0.6556 |
| 1,536 | 0.6043 / 0.5084 | 0.7771 / 0.6560 |
| native | 0.5992 / 0.5042 | 0.7760 / 0.6584 |

The eye-color subset moves only from 0.5929 to 0.6005 macro AP between the
384 and native inputs. At the checkpoint threshold, its mean false-positive
rate rises from 0.1661 to 0.1832. `blue eyes` illustrates why later comparisons
must use matched false-positive rate: at 384 it has AP 0.6975 and recall 0.9783,
but FPR 0.3841. These figures set the fixed-model reference; they are not a
FoveaTag result. FoveaTag must improve the ground fine-detail slice at matched
FPR, then preserve the separately reported photo slice.

Fixed-model forward mean / p95 was 31.1 / 38.7 ms. The retained report is
`docs/audits/foveatag_ground_fixed_predictions/siglip2_fixed_report.json`.
The 21 MB per-condition prediction intermediate was removed after aggregation.

### 9.11 Phase 1 end-to-end connectivity smoke (2026-09-29)

The fixed-encoder control runner was connected to the existing
`TaggerDataset` vocabulary, label-mask, and image-refresh boundary. It selects
stable per-domain subsets, excludes every path in both benchmark manifests,
and uses a weighted sampler so the photo domain cannot disappear inside the
larger illustration corpus. For photos, the run applies a separate loss weight
and masks unverified fine-detail negatives while retaining observed positives.

An eight-step GPU smoke traversed 311,420 registered source samples, then used
eight selected training samples and eight validation samples with four local
views and the full 107,926-tag head. All eight optimizer steps completed.
Validation consumed 863,319 known label terms across 287 active tags and the
epoch-boundary checkpoint completed. Its F1 and threshold are connectivity
signals only: the sample is much too small for a quality claim.

The resumable `latest.pt` contains the trainable control, optimizer, scheduler,
and external-encoder provenance. The best model is stored separately as an
optimizer-free safetensors artifact, avoiding a second approximately 1.49 GB
training checkpoint. The two large smoke checkpoints were removed after the
contract was verified; the retained measurement is
`docs/audits/FOVEATAG_PHASE1_CONTROL_SMOKE_2026-09-29.json`.

### 9.12 Phase 1 fixed-encoder pilot (2026-09-29)

The first non-smoke pilot froze the existing fixed-384 SigLIP2 encoder and
trained the initially random poolers, projections, and 107,926-way evidence
head. One epoch sampled 20,000 ground-illustration and 4,000 photo examples at
an 85:15 expected domain ratio, batch size 4, four local views, 6,000 optimizer
steps, and 200 warmup steps. The 1,280-image validation split was disjoint from
training and both benchmark manifests.

Training took 3,963.0 seconds (6.06 samples/s); validation took 208.6 seconds.
The run encountered 103.5 MP and 97.4 MP sources. They did not enlarge the GPU
token sequence, which remained one 448 overview plus four 256 local canvases,
but Pillow decoded the complete sources before cropping. This measured the
storage-side failure addressed by the full-decode cap and proposed tiled
pyramid. It is not evidence for sending full-resolution token sequences to the
network.

Validation selected threshold 0.4 with macro F1 0.0274 over 9,086 active tags.
Native-resolution benchmark results were:

| Slice | Fine AP / F1 at 0.4 | Coarse AP / F1 at 0.4 | Matched-FPR threshold | Matched-FPR recall, control / baseline | Fine AP delta |
|---|---:|---:|---:|---:|---:|
| Danbooru ground illustration | 0.2337 / 0.2046 | 0.5992 / 0.4315 | 0.49 | 0.2713 / 0.7605 | -0.3655 |
| Partially curated photo | 0.2808 / 0.2068 | 0.7486 / 0.5623 | 0.62 | 0.3138 / 0.8193 | -0.4401 |

On the ground slice, mean per-image stage times were 51.5 ms decode, 35.7 ms
view rendering, 7.1 ms processor work, and 73.3 ms forward. The photo values
were 47.7, 45.4, 7.2, and 73.5 ms respectively. Aggregate reports are retained
under `docs/audits/foveatag_control_pilot_ground/` and
`docs/audits/foveatag_control_pilot_photo/`; the resumable run artifacts remain
outside git under `outputs/foveatag_phase1_control_pilot_20260929/`. The compact
machine-readable run summary is
`docs/audits/FOVEATAG_PHASE1_CONTROL_PILOT_2026-09-29.json`.

This pilot fails the quality gate but does **not** reject multi-view input. The
comparison discarded the baseline's trained classifier and replaced it with
123.95 million random trainable parameters, then showed only 24,000 images.
The measured gap therefore combines view/head effects with loss of the existing
semantic classifier. The corrected Phase 1 control must preserve the fixed-384
overview logits exactly at initialization and add the local path as a
zero-initialized residual. Only that residual comparison isolates whether local
views add fine-detail information. The pilot also used an unnormalized local
`logsumexp`; because `K` was fixed, its constant offset does not affect AP, and
threshold search absorbs it for matched-FPR metrics. The implementation now
uses `logmeanexp` so changing `K` cannot shift all logits.

The corrected residual control is now implemented. Its frozen external state is
the existing base vision encoder with the checkpoint's trained 7.96M LoRA
adaptation plus its trained baseline head; the compact artifact
contains 68.58 million trainable-state elements and initializes the local tag
projection to zero. Its overview is the unmodified source passed through the
original fixed-384 processor, not the FoveaTag 448 letterbox. An eight-step
real-data smoke completed with validation
macro F1 0.7472 at threshold 0.35, compared with 0.3019 for the earlier
random-full-head connectivity smoke. Neither eight-sample result is a quality
measurement, but the change confirms that the corrected path retains the
baseline semantic classifier instead of relearning it. The retained report is
`docs/audits/FOVEATAG_PHASE1_RESIDUAL_SMOKE_2026-09-29.json`.

The remaining Phase 1 diagnostic reuses the confounded pilot's budget so the
head correction is the only intentional experiment change: one epoch of
24,000 presented samples at batch 4 (6,000 optimizer steps), four local views,
20,000 ground-illustration and 4,000 photo training-pool entries, 1,024 and 256
validation entries, 0.85/0.15 domain sampling, photo loss weight 0.25,
`1e-4` learning rate and weight decay, 200 warmup steps, seed 20260929, and the
40 MP source-decode cap. The exact database identity, manifest hashes, label
policy, ASL parameters, optimizer budget, and input contract are now stored in
both the resumable checkpoint and compact best weights; resume refuses a
changed contract and restores the complete prior-epoch history. A single
evaluation runner preflights training completion and every external artifact
hash before using the GPU, then regenerates predictions over the complete
resolution ladders of both registered benchmark manifests and writes normal,
fine matched-FPR, and coarse matched-FPR reports with input hashes. This is a
pretrained feasibility/cost diagnostic, not a gate or initialization source
for the scratch architecture. The complete pre-run contract, including external
artifact and benchmark hashes, is retained in
`docs/audits/FOVEATAG_PHASE1_RESIDUAL_PLAN_2026-09-29.json`. The diagnostic is
launched through `core.tagger.foveatag.control_plan`, which verifies that plan's
file hashes, database identity, ASL contract, and optimizer-step arithmetic
before passing its values directly to the training or evaluation runner. This
avoids maintaining a second handwritten set of command-line values.
The plan was revalidated while Phase 2 occupied the GPU: all three external
model artifacts, the 14.00 GB database identity, both holdout manifests, both
fixed-baseline reports, and the optimizer-step arithmetic passed. Its output
directory did not yet exist, so the registered run has neither been partially
executed nor shadowed by an unregistered result.

The 68.58M figure is optimization state for the added local path, not total
model capacity. The checkpoint-specific comparison is:

| Configuration | Existing semantic state retained | Newly trainable state | Interpretation |
|---|---:|---:|---|
| Existing fixed-384 SigLIP2 LoRA tagger | -- | 132.40M (124.44M classifier + 7.96M LoRA) | Current production baseline; base encoder weights are additional frozen/external state. |
| First random-full-head control | encoder only | 123.95M | Discarded the trained 124.44M classifier, causing the confound measured above. |
| Corrected local-residual control | base encoder + 7.96M trained LoRA + 124.44M trained classifier | 68.58M | Adds capacity: 57.36M evidence head/mixer, 10.63M slot poolers, and 0.59M projection. It is not a smaller replacement tagger. |
| Full-scratch FoveaTag, 107,926 tags | none | 151.17M | Complete trainable model: 37.39M encoder, 0.59M local pooler, 0.46M projections, and 112.73M evidence head. It has no frozen external encoder. |

The local residual still gives every tag a learned 512-dimensional vector
(55.26M parameters in that matrix alone). Increasing this dimension before the
corrected pilot would add classifier capacity without showing that capacity is
the bottleneck. If local AP remains low while dense-oracle utility shows that
the selected crops contain useful evidence, compare 512, 768, and 1,024 local
dimensions at equal data and optimizer steps. For the fully scratch model, the
37.39M figure refers only to the image encoder. The exact default model with
107,926 tags has 151,170,102 trainable parameters: 37,392,032 in the encoder,
592,896 in the local slot pooler, 459,776 in the two projections, and
112,725,398 in the evidence head. The head consists of a 2,101,248-parameter
mixer and a 110,624,150-parameter 1,024-to-107,926 linear classifier including
bias. This is 18,770,102 parameters, or 14.2%, above the existing tagger's known
132.40M trainable state. The comparison remains conservative about total
capacity because the existing tagger also carries its frozen external SigLIP2
base encoder, whereas every FoveaTag parameter is trained from scratch. The
exact measurement is retained in
`docs/audits/FOVEATAG_CAPACITY_2026-09-29.json`. Phase 6 capacity growth is
therefore gated on measured under-capacity rather than the encoder-only count.

The existing fixed-384 run is also much larger than the Phase 1 diagnostic.
Its saved configuration names 16 dataset domains containing about 3.70 million
registered items, batch 32, 20 planned epochs, and dynamic tag refresh,
Danbooru injection, and vocabulary expansion. The retained checkpoint reached
epoch 13 / step 1,179,633; the best checkpoint was epoch 10 / step 949,537.
These numbers are a scale and coverage reference, not a Phase 3 budget: the
eligible fixed snapshot is smaller than the registered count, and FoveaTag has
different per-sample compute and trains every parameter. In particular, the
24,000-sample Phase 1 run must not silently become the main scratch-tagger
budget, and a two-domain gate run must be described as a subset rather than as
the same data used by the existing tagger. The exact Phase 3 budget is chosen
from measured supervised throughput before branch A reports quality, then
frozen for A/B/C by the shared training contract. The machine-readable reference
is `docs/audits/FOVEATAG_EXISTING_TAGGER_TRAINING_REFERENCE_2026-09-29.json`.

### 9.13 Phase 2 lifecycle smoke (2026-09-29)

The long-run path no longer relies on the bounded benchmark path list. It builds
JSONL path manifests with mmap offset indexes, so eight Windows data workers do
not each receive a Python copy of 1.55 million path strings. Validation paths
are selected exactly and deterministically; benchmark paths are excluded;
training samples domains at 85% illustration and 15% photo with a stateless
per-epoch mapping. Resume reuses the frozen manifests even if `datasets.db`
metadata changes, while still rejecting a changed dataset, holdout, cap, seed,
or sampling contract.

The Phase 2 loop also accepts the same thread-safe stop control used by the
later supervised and router stages. It checks the control after each optimizer
step and writes the exact epoch offset, accumulated losses, optimizer,
scheduler, and RNG state before returning `stopped_by_request`. This supplies
the backend trainer lifecycle without making FoveaTag selectable before its
quality gates pass. The loop also emits guarded `run`, `train`, and `validation`
callbacks with exact epoch, source offset, and global-step progress; a reporting
failure cannot interrupt training or checkpoint creation.

The production-contract smoke built 1,544,750 illustration and 20,293 photo
training paths plus held-out sets of 2,048 and 512. It stopped after eight
steps at source offset 128, then resumed to step 12 and offset 192. The resumed
checkpoint retained all 192 samples in its epoch loss accumulator, optimizer
state, RNG state, and scheduler position. It was 286,950,347 bytes and used
11,648 MiB peak reserved VRAM.

The registered convergence run's atomic checkpoint was also read back on CPU at
step 95,000 without interrupting the live process or initializing CUDA. Its
epoch 0 source offset was 1,520,000 and its scheduler position was exactly
95,000, matching `progress.json`. The retained epoch accumulator contained
1,519,992 decoded samples, eight deterministic skips, and 95,000 optimizer
steps. The run contract still records scratch initialization, no external
weights, batch 16,
1,546,798 presented samples per epoch, and 10 epochs; it has no pyramid-cache
field, as required for compatibility with the manifest frozen before that
optional path was implemented. The 286,950,411-byte checkpoint's cumulative
loss was 0.070776: 0.002126 correspondence, 0.061469 RGB reconstruction, and
0.014362 chroma with weight 0.5. These values are self-supervised
reconstruction and feature correspondence losses, not the globally averaged
asymmetric tag loss.

A separate one-epoch lifecycle smoke used 32 training and eight validation
samples. It completed at epoch 1 / offset 0, wrote the resumable checkpoint,
domain-specific held-out RGB/chroma/correspondence losses, and a 211-tensor,
149,589,368-byte encoder-only safetensors artifact. Its metadata records
`initialization: scratch` and an empty external-weight list. These runs prove
checkpoint and validation wiring only; the compact record is
`FOVEATAG_PHASE2_PRETRAINING_SMOKE_2026-09-29.json`. The Phase 2 convergence
gate remains open until the 10-epoch run is measured.

The first registered epoch completed 1,546,798 presented samples at step
96,675. Held-out weighted loss fell to 0.049186, but the representation gate
failed: masked-local and overview mean pairwise cosine reached 0.9983--0.9986,
while effective rank fell to 1.05--1.24 out of 384 channels. A matched
same-seed scratch evaluation measured cosine 0.8691--0.9767 and effective rank
1.85--2.00, so the trained encoder moved measurably closer to rank-one collapse
rather than merely inheriting an unusual initial scale. The run was stopped at
its exact epoch boundary before spending nine more epochs. The hash-bound
metrics and artifact identities are retained in
`FOVEATAG_PHASE2_EPOCH0_COLLAPSE_2026-09-29.json`. Phase 2 now requires an
explicit anti-collapse objective and a bounded pilot that improves both views
relative to this reference before a replacement long run is registered.

The replacement objective adds variance and covariance regularization adapted
from [VICReg](https://arxiv.org/abs/2105.04906) directly to both stride-16
feature maps. Each map is flattened across batch and space, giving 4,096 local
and 12,544 overview observations at batch 16 despite the small image batch.
The variance hinge has target standard deviation 1 and epsilon `1e-4`; the
off-diagonal covariance penalty is normalized by channel count. Weights 1.0
and 0.01 follow the smaller VICReg regularizer scale reported when adding the
terms to an existing objective. An additional 0.1-weight mean-square centering
term addresses the large common feature bias observed in this run; this term
is a FoveaTag adaptation, not part of the cited formulation. The existing
correspondence, reconstruction, and chroma weights remain 1.0, 1.0, and 0.5.
Every term is logged separately, and the Phase 2 gate now rejects checkpoints
whose run contract or validation record omits the anti-collapse terms. Because
the paper studies global embeddings and substantially larger image batches,
these weights are a bounded-pilot starting point rather than a transferred
quality claim.

A 1,000-step bounded pilot then trained on 16,000 samples and evaluated the
same fixed 512-photo and 2,048-illustration validation sets. Relative to the
same-seed scratch encoder, mean pairwise cosine fell in all four domain/view
pairs: 0.913 to 0.842 and 0.977 to 0.617 for photo local/overview, and 0.869 to
0.749 and 0.960 to 0.529 for illustration local/overview. Effective rank rose
from 1.85--2.00 to 2.98--3.53. Training sustained 62.0 samples/s and reserved
11,538 MiB, close to the rejected objective's full-epoch 63.4 samples/s. This
passes the bounded health gate for starting a replacement measurement, but is
not Phase 2 acceptance: rank remains low relative to 384 channels, and the
pilot's 100-step warmup and one-epoch cosine schedule are deliberately shorter
than the long-run contract. Hashes and component losses are recorded in
`FOVEATAG_PHASE2_ANTICOLLAPSE_PILOT_2026-09-29.json`.

The replacement 10-epoch run was then started from scratch with the registered
5,000-step warmup. Its first atomic checkpoint retained epoch 0 offset 80,000,
step 5,000, 79,999 decoded samples, one deterministic skip, and matching
optimizer/scheduler position. A fixed-validation diagnostic at that warmup
boundary measured effective rank 7.54/9.32 for photo local/overview and
10.84/12.06 for illustration local/overview; all four exceed the same-seed
scratch range of 1.85--2.00. Mean pairwise cosine was 0.647--0.776 rather than
the rejected run's approximately 0.998. The exact checkpoint is retained as a
hard-link snapshot while `latest.pt` advances, and the registered run resumes
from it. Component losses, hashes, and resume state are recorded in
`FOVEATAG_PHASE2_ANTICOLLAPSE_STEP5000_2026-09-29.json`. This authorizes
continued measurement but is not the Phase 2 acceptance decision.

The replacement run completed epoch 0 at step 96,675 after presenting
1,546,798 samples (eight deterministic decode skips). Fixed validation measured
weighted loss 0.166735. Effective rank was 103.58/127.60 for photo
masked-local/overview and 80.04/159.64 for illustration masked-local/overview,
with channel standard-deviation means 0.896--1.232. Mean pairwise cosine was
0.815--0.943. This is not the rejected objective's rank-one failure: its epoch-0
rank was 1.05--1.24 with cosine about 0.998. The run therefore continues under
the registered ten-epoch contract. The best encoder, metadata, history hashes,
component losses, and per-domain health metrics are recorded in
`FOVEATAG_PHASE2_ANTICOLLAPSE_EPOCH0_2026-09-30.json`; this remains an interim
health result rather than Phase 2 acceptance.

Epoch 1 completed at step 193,350 after a second pass over 1,546,798 presented
samples (nine deterministic decode skips). Weighted validation loss fell from
0.166735 to 0.146765, an 11.98% relative improvement. Reconstruction and chroma
losses improved by 9.21% and 12.13%; correspondence loss increased by 14.72%,
so that component remains under observation. Effective rank increased for all
four domain/view pairs to 125.92--175.91, while mean pairwise cosine fell to
0.607--0.918. This is neither a collapse nor an early-stop candidate: only two
epochs have completed and the validation improvement is far above the
registered 0.5% plateau threshold. Exact metrics, policy binding, and artifact
hashes are recorded in
`FOVEATAG_PHASE2_ANTICOLLAPSE_EPOCH1_2026-09-30.json`.

The replacement run completed all ten epochs on 2026-10-03 at step 954,024.
Best held-out weighted loss was 0.113060, versus 0.166735 after the first epoch.
It decoded 15,264,198 of 15,467,980 presented samples; 203,782 were skipped,
including 203,694 in epoch 6 during a suspected transient storage outage.
Actual optimizer steps are 12,726 below the nominal 966,750. The evidence keeps
this shortfall explicit; completing ten source passes does not imply completing
the nominal update budget.

Post-run evaluation of the exact best encoder covered all 512 photo and 2,048
illustration validation samples without skips. Photo/illustration correspondence
was 0.030928/0.029115 versus same-seed scratch 0.639283/0.465378. Effective rank
was 111.61/98.14 for masked-local and 261.04/233.70 for overview, versus scratch
1.85--2.00. Channel standard deviations were 1.02--1.28 and pairwise cosine
0.507--0.945. The explicit review accepts this representation for Phase 3;
it does not establish tag quality. `FOVEATAG_PHASE2_GATE_2026-10-03.json` binds
the evaluation, best encoder, metadata, completed summary, and fixed manifests.
JSON-restored view configuration needed tuple normalization for Pillow padding;
the repair passed 25 related CPU tests before the successful GPU evaluation.

### 9.14 Phase 3 throughput protocol (2026-09-29)

`supervised_benchmark.py` measures the three registered Phase 3 input modes in
fixed A/B/C order using the same seed, scratch model initialization, image
order, tag count, optimizer, batch size, and local-view budget. It includes
real source decode, deterministic view rendering, pinned-memory transfer, the
full tagger forward and backward passes, gradient clipping, and AdamW update.
The report separates data wait, transfer, and optimizer-step time and records
model-inclusive peak allocated and reserved VRAM. It also projects hours per
million presented samples and, when supplied, hours per proposed epoch.

The labels contain one synthetic positive per image and exist only to exercise
the full classifier and ASL tensor. Reported values are therefore named
`synthetic_asl_first` and `synthetic_asl_last`; they are neither quality
measurements nor comparable with the Phase 2 reconstruction objective. The
benchmark must run only after the current Phase 2 GPU process releases the
device. Its result fixes a feasible Phase 3 sample/step budget before branch A
produces any quality result, preventing the budget from being selected after
seeing the first outcome.

The 2026-10-03 real-image measurement used the 107,926-tag vocabulary, batch 16,
four local views, eight workers, four warmup and sixteen measured steps per
mode. Overview-only throughput was 75.95 samples/s; placebo-local and
source-native were 40.36 and 39.88 samples/s. Peak allocated VRAM was
9,819/18,204/18,202 MiB respectively. These short measurements include steady
state decode/render wait, transfers, and full AdamW updates; process startup
and long-run storage stalls are not covered.

`FOVEATAG_PHASE3_ABC_PLAN_2026-10-03.json` registers a two-domain architecture
comparison before branch quality is observed: 200,000 illustration and 18,000
photo training examples, 2,048/512 fixed validation examples, ten epochs of
240,000 presented samples each, and 150,000 nominal updates per branch. Domain
sampling weights are 0.85/0.15; photo loss weight is 0.25 with the existing
fine-detail missing-negative mask. Each branch starts from the accepted Phase-2
encoder and the same random head and updates every parameter. A consistent
SQLite backup freezes the database at
`outputs/foveatag_phase3_abc_20261003/datasets_snapshot.db`. The comparison runs
in A/B/C order with resumable checkpoints every 1,000 updates. Projected pure
training time is about 42 hours total, plus validation and startup. This is a
two-domain causal experiment; broad-domain replacement coverage remains
required before claiming a replacement for the existing 16-domain tagger.

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
