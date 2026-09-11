# Adapter subsystem contract

SushiUI represents an adapter family as two independent axes:
`adapter_algorithm` (`lora`, `loha`, or `lokr`) and `weight_decompose`. The
pair names ordinary LoRA/LoHa/LoKr or the decomposed DoRA/DoHa/DoKr forms.

The authoritative generation and training capability tables are
`ENABLED_ADAPTER_PAIRS` and `TRAINABLE_ADAPTER_PAIRS` in
`backend/core/adapters/capability.py`. Documentation and architecture handlers
must not maintain copies of those tables.

## Shipped capability

| Architectures | Generation | Training |
|---|---|---|
| SD1.5, SDXL | LoRA | LoRA |
| Z-Image, Lens, MiniT2I | LoRA, LoHa, LoKr, DoRA | LoRA, LoHa, LoKr, DoRA |
| Anima, Ideogram 4, Krea 2, FLUX.2, LTX-2.3, ACE-Step | LoRA, LoHa, LoKr | LoRA, LoHa, LoKr |
| MiniMax-H3, SenseNova | LoRA, LoHa, LoKr | LoRA |

DoHa and DoKr are refused on every architecture. A generation capability means
that a checkpoint can be loaded and applied; a training capability additionally
means that the trainer can construct, save, resume, and reload it through that
architecture's generation path.

ACE-Step's LoHa/LoKr support belongs to its sd-scripts codec path. Its PEFT path
accepts ordinary LoRA keys only.

## 1. Adapter specification

`AdapterSpec` is the normalized description shared by training and generation.
It carries algorithm, decomposition, rank/alpha, format, target metadata, and
algorithm-specific options. `AdapterCodec` detects and normalizes SushiUI,
Kohya/LyCORIS, and diffusers/PEFT checkpoints before a session mutates a model.

The pair, not a family nickname, is the stable identity. This prevents a
decomposition flag from being silently lost when a configuration or checkpoint
round-trips through an older surface.

## 2. Layer algebra

`backend/core/adapters/layers.py` owns the trainable LoRA, LoHa, LoKr, and DoRA
branches. `backend/core/adapters/reference.py` supplies independent dense
oracles used by the test suite.

LoHa and LoKr are additive and can execute over supported quantized base
Linears without reconstructing the base weight. DoRA rescales the direction of
`base + delta`; it therefore requires a usable dense base-weight norm. The
implementation accepts the supported row-magnitude layouts and refuses an
ambiguous magnitude axis rather than applying a numerically different adapter.

## 3. Target topology

Architecture code supplies target discovery and component ownership through
`backend/core/adapters/targets.py`. The shared layer implementation does not
guess fused-QKV slices, MoT halves, component lifetime, or block-swap ownership.

MiniMax-H3's fused-QKV topology and SenseNova's two MoT halves are why their
LoHa/LoKr generation rows do not imply training rows. SD1.5 and SDXL remain on
the diffusers loader, which does not preserve every extended tensor family.

## 4. Checkpoint codecs and sessions

`AdapterTensorGroup` validates that every logical target has one complete,
unambiguous tensor group before mutation. Split fused-QKV groups are handled by
the architecture-owned topology and refuse decomposed or Tucker forms where a
mathematically exact split is unavailable.

`AdapterSession` owns application order and rollback. It resolves all files,
validates groups and capabilities, then installs branches atomically. A failed
apply leaves no partial stack. Unload restores the original modules in reverse
order; callers must use the session rather than mutate adapter layers directly.

## Block swap and quantized bases

`BLOCK_SWAP_ADAPTER_ORDER` records whether an architecture installs adapters
before, after, or inside its block partition. The capability layer uses this to
refuse a branch that would remain on the host and to report advisory cases where
the adapter stays resident while its base streams.

Additive adapters preserve the quantized base forward. Weight decomposition is
refused where it would require dequantizing the base every call or abandoning
the architecture's fused quantized GEMM. Refusal text comes from
`backend/core/adapters/capability.py` and is part of the public capability
payload.

## Phase 4: execution backends

The execution-backend registry is shipped, but no fused backend is registered
as a default. `reference` is the correctness path. `auto` may select only a
backend whose registered domain covers the adapter family, device, dtype,
shape, and requested operations; otherwise it uses the reference path.

Backend selection is per compatible region, not per file. A backend must pass
forward, gradient, stacking, save/reload, device/dtype, and failure-fallback
gates before registration. A benchmark or kernel import by itself is not a
capability.

## API and persistence

- API defaults live only in `backend/api/param_defaults.py`.
- `openapi.yaml`, configuration generation, database persistence, and frontend
  forms carry both axes.
- Generation validates against `ENABLED_ADAPTER_PAIRS`; training validates
  against `TRAINABLE_ADAPTER_PAIRS` before model load.
- Checkpoints record enough metadata to recover the pair without relying on a
  filename.

## Maintenance checklist

1. Change one capability row; do not add an architecture-side mirror.
2. Add a real architecture round trip for every newly opened pair.
3. Verify block-swap placement and quantized-base behavior for that row.
4. Exercise stacked apply, rollback, unload, save, and resume.
5. Keep unsupported pairs as explicit refusals until all gates pass.
