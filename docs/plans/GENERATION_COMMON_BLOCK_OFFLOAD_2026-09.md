# Generation common block offload

## Goal

Cover every generation architecture with the shared immutable transfer engine
without pretending that every model has a `transformer_blocks` list. Existing
explicit-loop users keep their current `FrozenSequentialTransferEngine` path;
the missing architectures use hook-driven conductors over architecture-owned
execution units.

## Architecture units

| Architecture | Generation units |
|---|---|
| SD1.5 / SDXL | U-Net `down_blocks`, `mid_block`, then `up_blocks` |
| Krea 2 | `transformer.transformer_blocks` |
| ACE-Step 1.5 | `dit.decoder.layers` |
| SenseNova U1.5 | `(understanding|generation, language_model.model.layers[i])` |
| MiniMax Music 3 | AR language-model decoder blocks and flow-transformer `transformer_blocks`, staged independently |

MiniMax Music 3 has no training route but is included because this plan covers
the authoritative generation registry. Existing Z-Image, FLUX.2, Anima, Lens,
Ideogram 4, MiniT2I, LTX-2.3, and MiniMax-H3 already use the common immutable
engine and need no second wrapper.

## Implementation sequence

1. Add a hook-driven frozen-module conductor backed by
   `FrozenSequentialTransferEngine`. It owns persistent CPU dtype planes,
   fixed GPU slots, event-only consumer waits, cross-iteration prefetch,
   exception-safe post hooks, and deterministic teardown.
2. Add a branch-aware frozen conductor for SenseNova so a prefix pass does not
   transfer the unused generation half and a denoise pass does not transfer the
   unused understanding half. Mixed-token calls acquire both keys and therefore
   require at least two slots.
3. Wire Krea 2, ACE-Step, SenseNova, and both MiniMax Music 3 stages after any
   per-request adapter/quantization mutation and before their repeated model
   loop. Tear down before adapter restoration or whole-component CPU moves.
4. Wire SD1.5/SDXL over top-level U-Net execution stages. Unlike the DiT paths,
   these bundles include convolution and normalization parameters, not only
   `Linear.weight`; `blocks_to_swap` is clamped to the real stage count.
5. Remove generation capability refusals, expose the existing image controls,
   add the same offload fields to audio request defaults/API/UI, and keep
   `param_defaults.py` plus `openapi.yaml` authoritative.

## Safety boundaries

- Generation conductors are forward-only and never write device data back.
- A tensor may belong to only one execution unit; shared ownership fails before
  hooks are installed.
- Setup occurs after LoRA/runtime quantization so the immutable master layout
  matches the actual request graph.
- Teardown precedes LoRA unload and component-level `.to("cpu")` normalization.
- Keep-hot may not claim a split component as wholly GPU-resident.
- SenseNova MoT phase eviction and branch block offload are mutually exclusive
  for one request until disjoint ownership is implemented on generation too.
- FBCache/Spectrum/style paths that skip or replay blocks must either use the
  order-agnostic hook policy or refuse the combination explicitly.

## Deferred verification

Implementation proceeds without runtime or GPU validation at the owner's
request. Static import, per-route smoke, same-seed output parity, exception
cleanup, LoRA apply/unload, transfer counters, peak VRAM, host pinned memory,
and warmed iteration latency remain release gates and will be recorded in the
unified offload validation audit.
