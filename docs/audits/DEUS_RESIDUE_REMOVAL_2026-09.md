# DEUS residue removal plan (2026-09)

## Goal

Remove the retired DEUS architecture from executable paths and public
contracts without changing behavior for any architecture in `ModelType` or
`ARCH_REGISTRY`.

## Boundary

The loader no longer detects or loads DEUS, so every `is_deus` runtime branch
is unreachable. Cleanup is split into independently reviewable units:

1. Remove `is_deus` propagation, two-pass sampling branches and preview-decoder
   compatibility arguments from generation code.
2. Remove DEUS from training defaults, frontend descriptions, OpenAPI and
   architecture documentation.
3. Update source-sensitive tests and verify that no model-specific DEUS token
   remains.

The Anima tokenizer's ordinary-language `Deus` token is model data and remains.
The `.gitignore` entry for the private `DEUS_TRAINING_IMPLEMENTATION_PLAN.md`
also remains so a local historical note cannot become tracked accidentally.
Path-redaction fixtures may keep arbitrary feature names only when they still
test a live warning string; otherwise they are updated with the warning.

## Verification

- Compile every changed Python file and import changed backend modules with
  CUDA initialization stubbed.
- Run focused generation, preview, defaults, schema and path-redaction tests
  without loading model weights.
- Check `openapi.yaml`, backend defaults and frontend consumers together.
- Search tracked files after cleanup and classify every remaining case.
