# Generation queue processor

The generation queue's **state** lives in
`frontend/src/contexts/GenerationQueueContext.tsx` (mounted at the root
layout). Its **dispatch loop** lives in
`frontend/src/components/generation/GenerationQueueProcessor.tsx`, a headless
component mounted once next to `{children}` inside the provider.

Before this split, each of the five generation panels owned a `processQueue`
tied to its own mount. `/generate` truly unmounts a panel on tab switch, so an
item whose type was claimed only by the panel the user just left had no
dispatcher left and stalled forever. (Txt2Img and Img2Img both claimed
`img2img`, so switching between just those two happened to keep working.)

## Division of labour

| Concern | Owner |
|---|---|
| Queue array, `currentItem`, `startNextInQueue`, `keep_models_hot`, drift-pause gate | `GenerationQueueContext` |
| Choosing and running the next item, patching the next loop step, advancing a video chain, alerts | `GenerationQueueProcessor` |
| Building requests (`handleAddToQueue`, `addLoopStepsToQueue*`), rendering, "generate forever" | the panels |

The processor is a React component, not a module singleton and not part of the
provider: merging it into the provider would give `GenerationQueueContext` a
runtime import of `videoChain.ts`, which today imports `QueueItem` type-only
precisely to avoid that cycle. As a component it just uses the hooks
(`useGenerationQueue`, `useStartup`, `useActiveTraining`) it needs.

## Frozen-at-enqueue state

The processor has no panel to read UI state from, so anything a request
depends on is frozen onto the `QueueItem` when it is enqueued:

- `params.developer_mode`, and the collapsed-Advanced-CFG resets
  (`cfg_schedule_type` / `cfg_rescale_snr_alpha` /
  `dynamic_threshold_percentile`) — via each panel's `freezeDispatchState`
- `params.ref_images`
- `useTrainingModel` / `trainingRunId` / `savePreviewToGallery` — the first two
  were already per-item; `savePreviewToGallery` used to be read live, so a
  queued preview now keeps the setting it was queued with
- `loopStepConfig` — the `sizeMode` / `scale` / `useMainControlNets` /
  `controlnets` of THAT step, read when its predecessor finishes
- `panel` — the enqueuing panel. `img2img`, `ref2vid` and `chain_vid` have two
  possible origins, so the type does not identify whose display a result
  belongs to.

This also fixes a latent bug: editing the loop config mid-run used to retarget
a loop that was already in flight.

## Panels as renderers

A panel decides "is the running item mine" with
`(currentItem.panel ?? typeToPanel(currentItem.type)) === "<its id>"`, then:

- renders `progressSnapshot` (fed by the provider's own global WebSocket
  subscription, which keeps running while every panel is unmounted);
- clears its display once per new owned item id (`clearedForItemRef`);
- applies `completedResults[panel]` when it does not own the running item —
  which is also how a result produced while the panel was unmounted arrives;
- keeps a WebSocket subscription only for the CFG-metrics accumulation, the
  one thing `progressSnapshot` does not carry (it holds the latest tick, not
  the accumulated array).

Two things only a panel can do are routed back to it:

- **restore-on-cancel** — only the panel holds the image the cancelled run
  replaced, and only the dispatcher can tell a user cancel from a failure;
  hence `lastFailure: { panel, itemId, cancelled, revision }`.
- **InpaintPanel's own session gallery** — must skip a result the server kept
  no record for; hence `ephemeral` on the published result.

## Traps

- **Alert ordering.** `alert()` blocks the JS thread. Every failure path
  releases the busy guard, fails the item, performs any cascade-cancel and
  re-schedules the queue *before* alerting. `runVideoItem`'s `onFailure`
  callback therefore returns the alert to show rather than showing it.
- **The model hold gate** is expressed through `startNextInQueue`'s
  `allowedTypes`: with no model loaded the processor claims only
  `NO_MODEL_REQUIRED` types. `upscale` is in that set because it can run on a
  spandrel checkpoint alone, and UpscalePanel never had the gate.
- **The drift pause** is a three-way contract: `startNextInQueue` refuses the
  paused group, the processor's auto-start effect excludes it from "has
  pending work" (otherwise it busy-loops re-entering an item that cannot
  start), and `resolveChainDriftPause` — still panel-rendered, since the state
  is queue-side — clears it, at which point the effect resumes the queue by
  itself.
- **`lastFailure` is only published for image runs.** Restore-on-cancel is an
  image contract; a cancelled video/audio run must not resurrect a stale still
  into the slot the user was making a clip in.
- **A drift pause is only *visible* on the txt2img and img2img tabs**, which
  are the two panels that render `ChainDriftPauseDialog`. The queue is held
  correctly whatever tab is open, but a user sitting on Outpaint/Inpaint/
  Upscale sees a stopped queue with no prompt until they switch back.
- **"Generate forever" stays panel-side** by design. It is an *enqueue*
  concern and needs the live panel params the processor deliberately cannot
  see. Do not "fix" it into the processor.
