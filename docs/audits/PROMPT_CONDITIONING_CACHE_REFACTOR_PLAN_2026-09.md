# Prompt-conditioning cache refactor plan (2026-09)

## Goal

Make repeated-prompt generation skip both text-encoder execution and GPU
staging where conditioning identity can be stated completely. Preserve output
tensors, prompt semantics and component-residency behavior. A cache is enabled
only after one architecture-owned function controls every encoder call for the
request.

## Shared contract

- Store detached tensor trees on CPU in the existing bounded eight-entry LRU.
- Isolate entries by the live encoder object; a cache entry must not retain a
  model after unload.
- Return independent copies on every hit.
- Include model/checkpoint, adapter and quantization identity; tokenizer and
  prompt-parser settings; every encoded positive, negative and NAG prompt;
  sequence/grid dimensions; dtype and execution device.
- Perform lookup before GPU staging. A hit that skips staging must not mark a
  CPU component as resident.
- Keep vision/reference-derived conditioning outside the text-only cache unless
  its full content identity is available without duplicating expensive work.

## Architecture boundaries

| Architecture | Refactor boundary | Required identity | Disposition |
|---|---|---|---|
| Krea2 | Existing `_krea2_encode` | Complete today | Already implemented |
| MiniT2I | Existing `_minit2i_encode` | Complete today, including cleaned NegPip text | Already implemented |
| Flux2 | New wrapper around positive, CFG-negative and NAG encodes | Model key, three prompts, CFG/NAG gates, max length, hidden-state layers, tokenizer | Implemented |
| Anima | New wrapper around positive, CFG-unconditional and NAG encodes | Model key, three prompts, CFG/NAG gates, both tokenizers, dtype | Implemented |
| Lens | New wrapper around main and NAG encodes | Model key, prompts, NAG gate, tokenizer, max length, dtype | Implemented |
| Ideogram 4 | Merge main and NAG encoding into one owner | Model key, cleaned prompts, grid, max length, tokenizer, dtype | Implemented |
| Z-Image | New wrapper around CFG and NAG encodes | Model key, prompt lists, CFG/NAG gates, max length, tokenizer, FP8 mode | Implemented |
| SD1.5 / SDXL | Cache weighted/chunked text conditioning before vision tokens | Live encoders, tokenizer pair, parser/chunk policy, custom-TE bridge, prompts, dtype/device | Implemented; mutable PEFT state bypasses the cache |
| SenseNova | Prefix construction returns live per-layer KV state, not text embeddings | Resolution, branch topology, references, transformer state and KV-streaming mode | Rejected; the returned state is mutated and cleared by denoising |
| LTX-2.3 | Diffusers pipeline owns prompt encoding and callback state | Pipeline-private prompt/encoder state | Do not duplicate upstream internals |
| ACE-Step / MiniMax Music 3 | Conditioning is interleaved with architecture-specific language/audio state | Long-form timeline and pipeline state | Reject as a generic image prompt-cache target |
| MiniMax-H3 | Specialized projected prompt cache | Encoder/projection paths, prompt and DiT width | Retain existing implementation |

## Commit sequence

1. Strengthen the shared cache lifecycle and key helpers, with ownership,
   eviction and mutation tests.
2. Refactor and cache Flux2 conditioning.
3. Refactor and cache Anima and Lens conditioning as separate commits.
4. Refactor and cache Ideogram 4 and Z-Image as separate commits.
5. Add the SD1.5/SDXL eligibility/key boundary and cache only the proven-safe
   base-conditioning path.
6. Re-audit SenseNova; either implement its separable text-only case or record
   the concrete dependency that prevents it.
7. Update the generation-efficiency audit, remove stale measurement wording,
   and run the combined CPU/static regression suite plus CUDA-free imports.

## Proof requirements

- A hit performs no encoder forward and no new GPU stage.
- A miss returns the same tensor tree as the uncached function.
- Changing any relevant key input forces a miss.
- Mutating a returned tensor cannot alter a later hit.
- Model unload/reload and encoder replacement cannot reuse stale entries.
- Keep-hot bookkeeping remains truthful on both hit and miss paths.

Real-model timing and broad architecture operation are accepted through user
feedback; automated tests retain the static, ownership and equivalence
contracts above.

## SenseNova re-audit

`sensenova_pipeline_ops.encode_prompt()` does not expose a separable text
embedding. It runs the transformer prefix forward and returns up to three
per-layer KV caches. Their token indexes include output resolution; their
branch topology depends on both CFG scales; reference images contribute pixel
embeddings; and the forward observes the request's applied LoRA state.

The caches are then expanded or adopted by the selected KV ring streamer.
Denoising writes their flash-cache tails and cleanup frees those buffers.
Treating that object as immutable CPU prompt data would require copying a large
multi-layer cache, rebuilding streamer ownership, and proving that no denoise
mutation survives. That moves substantially more memory than ordinary prompt
embeddings and does not preserve the current lifecycle. Tokenization alone is
separable but too small to justify another cache and invalidation contract.
