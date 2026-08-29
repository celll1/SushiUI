# MiniMax-H3 hybrid-loader contract

The hybrid loader constructs one MiniMax-H3 transformer from an `fl2va` base and
a compatible `ref2va` overlay. It does not load a ComfyUI node or bypass SushiUI's
checkpoint conversion path.

## Merge recipe

- The base is `fl2va`; the overlay is `ref2va`.
- Selected block AdaLN projection tensors come from the overlay. The default
  inclusive block range is owned by `backend/api/param_defaults.py` and the
  loader validates it against checkpoint geometry.
- Final AdaLN overlay is independently opt-in.
- Weight and all associated quantization sidecars are selected atomically from
  the same source file.
- Both files must have compatible key sets, shapes, pruned geometry, and
  quantization layout. Validation occurs before tensor materialization.
- The validated file digest is checked again at load time so replacing either
  checkpoint between preflight and load is refused.

The resulting model reports `variant: hybrid` and carries base, overlay, recipe,
compatibility digest, and quantization provenance. Use the hybrid digest—not the
base model hash—to distinguish two recipes built on one base.

## Capability boundary

Hybrid is an explicit variant, not an alias for either source partition. Current
API route gates conservatively expose only the workflows that have their own
hybrid acceptance. Reference-conditioned workflows are not enabled merely
because some AdaLN tensors came from `ref2va`. Applying a LoRA is allowed with an
explicit unmeasured-composition warning.

The API contract is described by `POST /models/load`,
`GET /models/minimax-h3/hybrid-overlays`, and their schemas in `openapi.yaml`.
The loader and route gates are authoritative if this summary drifts.

## Provenance

The tensor-selection concept was adapted from the MIT-licensed
`scottmudge/ComfyUI_MinimaxH3HybridLoader`; SushiUI owns the preflight, raw tensor
reader, conversion, lifecycle, API, and capability gates. Required upstream
notice verification is tracked in `docs/legal/THIRD_PARTY_PROVENANCE.md`.
