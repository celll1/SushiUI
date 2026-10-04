# Ming frontend static review and owner acceptance

Date: 2026-10-02. Branch: `codex/ming-image-design`.

This follow-up closes the identified source/wiring findings for owner UI review.
It does not assert TypeScript compiler success, rendered layout correctness, or
real-model behavior. No build/type-check, live API, provider request, frontend
server, or GPU was used. Existing CPU backend evidence remains scoped as recorded
in [the feature audit](MING_IMAGE_01_PARITY_AUDIT.md).

## Reviewed contracts and operation paths

| Area | Static review / resulting behavior |
|---|---|
| Public types and defaults | Model profile, layer count, partition, result groups, structural census, global descriptor, conditioner LR/groups, diagnostics and native export fields are declared in `api.ts`. Native new defaults come from startup schema; Design/Layer overlays have separate identities. Four image transports serialize layer/partition fields. |
| Model and profile changes | Generation panels resolve the current model before dispatch. Partition values clear when leaving Ming. Studio compares fresh profile as well as architecture/variant; Design/Layer defaults and recipes have distinct identities. Generation graphics consume loaded model metadata; training uses inspected checkpoint metadata. |
| Layer input restrictions | Txt2img requires a reference; inpaint/outpaint explain and refuse Layer before rewriting or enqueueing. Img2img supplies its primary image and optional references. Studio Layer requires an unmasked image input and refuses masked/inpaint requests before job creation. |
| Partition and loop wiring | Inpaint queue allowlist and txt2img/img2img/inpaint loop builders retain Ming partition. Shared model-environment inheritance includes layer count and partition. Graph totals/tooltips count all target frames; Ming guide/style exclusion applies to any enabled partition mode. The existing shared planner still supplies snapping and geometry. |
| Prompt enhancement | Typed native modes, image conversion, shared persisted settings, cancellation and stale-response guards; per-panel datalist IDs. Provider/default failures stay visible; stale replies cannot overwrite newer provider/settings. Manual prompt application remains explicit. |
| Reference/style controls | Schema defaults, native-only selection, RGBA input transport, per-entry windows/ranges and combine mode, unsupported independent guidance clearing, normal-attention and partition limitations. Backend validation remains authoritative for invalid external/persisted entries. |
| Output ownership | Queue snapshots preserve native ordered groups and gallery labels. Layer primary is layer 1; auxiliary canvas remains a separate output. Studio excludes the actual primary filename when importing other outputs, rather than dropping the first frame; this avoids primary duplication and lost canvas. |
| Gallery transfers | Valid saved layer count and partition restore into generation params. Actual native role/group and actual partition metadata are displayed separately from requested options. Studio recipes retain profile/layer count and server image metadata. |
| Training configuration | Full conditioner groups and per-group LRs, caption adapter, ordinary LoRA/full capabilities, frozen-conditioner prefetch guard, global/partition coupling, U-Net inheritance/configuration and parameter graphics, YAML/pass-through keys, sample layer count, native guidance/null/query-chunk controls. Submit validation and backend refusals remain in place. |
| Precision, export and diagnostics | ConvRot controls use schema defaults and communicate CUDA/BF16/block-swap restrictions; incompatible cached policies remain backend-gated. Native-package export uses a fresh destination and model-identity guards; stopped full-run export uses detected architecture. Debug frame roles/sign/weights and metric UI are typed. |

## Findings fixed in this follow-up

1. Missing Ming partition in inpaint queue and loop continuation paths.
2. Missing saved Layer count on Gallery-to-generation transfers.
3. Studio output import assumed the first frame was primary, duplicating layer 1
   and dropping the auxiliary canvas.
4. Studio's fresh-model comparison omitted the native profile.
5. Layer unsupported tasks/missing references could reach rewriting/queue/jobs
   before a clear profile-specific UI refusal.
6. Region tooltips counted only one target frame; Ming's partition exclusion text
   incorrectly suggested a single-region enabled plan could use guide/style.

The previous prompt settings race and repeated datalist IDs were fixed in
`1bdcd394`; this review includes their lifecycle and request interfaces.

## Static evidence and its limits

33 changed frontend source files were included in local import-path
resolution: 420 relative/alias import paths resolve. `git diff --check` passes.
This is source linkage evidence, not module export checking or TypeScript
semantic verification. No new API or backend changes were needed.

No identified critical source/wiring finding remains open in this review.
Compiler and browser evidence remain missing; source review cannot replace them.

## Owner confirmation checklist

Use the feature branch frontend and a matching backend when available. Model
execution checks may wait for the occupied API/GPU to be released.

1. Run the owner build/type check once. Open Design and Layer and confirm model
   structure, parameter totals, saved U-Net/global graphics and training model
   selection; switching profiles changes the correct defaults.
2. For Layer, verify reference-free txt2img and inpaint/outpaint refuse clearly.
   Confirm Studio requires unmasked img2img input and detects a profile switch.
3. With resources available, generate ordered Layer output: canvas plus selected
   layers, primary layer 1, previews, downloads, shared gallery and Studio all
   retain every file without duplication. Transfer it back and verify layer count.
4. Set partition once and confirm it remains in inpaint and loop continuation;
   switch away from Ming and confirm it clears. Confirm guide/style exclusion
   and all-frame token totals match displayed configuration.
5. Change prompt providers/settings quickly, preview/cancel/unmount and switch
   panels. Confirm no old rewrite/settings/model list is applied to the new
   selection. No provider call occurs merely from opening the controls.
6. Inspect training full/LoRA controls, resume/YAML-restored settings, U-Net
   inheritance, conditioner groups/LRs, sample layer count, export and diagnostics.
   Confirm error messages remain readable and a corrected request can be retried.

## Source scope

- [frontend/src/app/generate/page.tsx](../../frontend/src/app/generate/page.tsx)
- [frontend/src/app/settings/page.tsx](../../frontend/src/app/settings/page.tsx)
- [frontend/src/components/common/ControlNetSelector.tsx](../../frontend/src/components/common/ControlNetSelector.tsx)
- [frontend/src/components/common/DiTRewiringSummary.tsx](../../frontend/src/components/common/DiTRewiringSummary.tsx)
- [frontend/src/components/common/FloatingGallery.tsx](../../frontend/src/components/common/FloatingGallery.tsx)
- [frontend/src/components/common/MingGlobalSummary.tsx](../../frontend/src/components/common/MingGlobalSummary.tsx)
- [frontend/src/components/common/MingLayerControls.tsx](../../frontend/src/components/common/MingLayerControls.tsx)
- [frontend/src/components/common/MingOutputGroups.tsx](../../frontend/src/components/common/MingOutputGroups.tsx)
- [frontend/src/components/common/MingPartitionControls.tsx](../../frontend/src/components/common/MingPartitionControls.tsx)
- [frontend/src/components/common/MingPromptUpsample.tsx](../../frontend/src/components/common/MingPromptUpsample.tsx)
- [frontend/src/components/common/ModelParameterStructure.tsx](../../frontend/src/components/common/ModelParameterStructure.tsx)
- [frontend/src/components/common/ModelSelector.tsx](../../frontend/src/components/common/ModelSelector.tsx)
- [frontend/src/components/common/QuantizedExportSection.tsx](../../frontend/src/components/common/QuantizedExportSection.tsx)
- [frontend/src/components/common/QwenPartitionControls.tsx](../../frontend/src/components/common/QwenPartitionControls.tsx)
- [frontend/src/components/generation/GenerationQueueProcessor.tsx](../../frontend/src/components/generation/GenerationQueueProcessor.tsx)
- [frontend/src/components/generation/Img2ImgPanel.tsx](../../frontend/src/components/generation/Img2ImgPanel.tsx)
- [frontend/src/components/generation/InpaintPanel.tsx](../../frontend/src/components/generation/InpaintPanel.tsx)
- [frontend/src/components/generation/OutpaintPanel.tsx](../../frontend/src/components/generation/OutpaintPanel.tsx)
- [frontend/src/components/generation/Txt2ImgPanel.tsx](../../frontend/src/components/generation/Txt2ImgPanel.tsx)
- [frontend/src/components/studio/StudioWorkspace.tsx](../../frontend/src/components/studio/StudioWorkspace.tsx)
- [frontend/src/components/training/DiTRewiringEditor.tsx](../../frontend/src/components/training/DiTRewiringEditor.tsx)
- [frontend/src/components/training/MingConditionerTraining.tsx](../../frontend/src/components/training/MingConditionerTraining.tsx)
- [frontend/src/components/training/QwenTrainingObjective.tsx](../../frontend/src/components/training/QwenTrainingObjective.tsx)
- [frontend/src/components/training/TrainingConfig.tsx](../../frontend/src/components/training/TrainingConfig.tsx)
- [frontend/src/components/training/TrainingMonitor.tsx](../../frontend/src/components/training/TrainingMonitor.tsx)
- [frontend/src/components/training/trainingConfigDefinitions.tsx](../../frontend/src/components/training/trainingConfigDefinitions.tsx)
- [frontend/src/components/training/trainingParams.ts](../../frontend/src/components/training/trainingParams.ts)
- [frontend/src/components/viewer/ImageGrid.tsx](../../frontend/src/components/viewer/ImageGrid.tsx)
- [frontend/src/contexts/GenerationQueueContext.tsx](../../frontend/src/contexts/GenerationQueueContext.tsx)
- [frontend/src/utils/api.ts](../../frontend/src/utils/api.ts)
- [frontend/src/utils/archConstraints.ts](../../frontend/src/utils/archConstraints.ts)
- [frontend/src/utils/loopGenerationInheritance.ts](../../frontend/src/utils/loopGenerationInheritance.ts)
- [frontend/src/utils/mingPromptUpsample.ts](../../frontend/src/utils/mingPromptUpsample.ts)

## Official-weight live follow-up (2026-10-04)

Both official ConvRot packages are now selectable and loadable on main. Generate
and ordered Layer outputs were exercised in the logged-in browser; see
[Ming deployment evidence](MING_IMAGE_01_DEPLOYMENT.md) for precision findings,
primary-output fixes, and the remaining quality/owner acceptance boundaries.

## Preview output selection (2026-10-04)

Moved Ming output groups from settings into Preview for txt2img, img2img,
inpaint and outpaint. Selection updates the main image and its group seed;
existing viewer, Send image and Studio consumers use that selected URL.
Downloads remain attached to each RGBA output. Unrelated or cleared previews
hide stale groups; active generation disables selection. Thumbnail groups are
height-limited and scrollable. No frontend build/type-check was run by the agent.

Main browser verification used existing completed Layer results without another
GPU generation: Layer 2 selected `_2.png`, then Layer 1 selected `_1.png`, with
matching active indicators. Preview placement was inspected at the current narrow
viewport. Evidence: `local/deployment/ming_layer_ui.png` in the main checkout.
