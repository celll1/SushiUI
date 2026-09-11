# Test suite rationalization plan

Date: 2026-09-11

## Objective

Reduce test runtime, brittleness, and maintenance cost without preserving tests
for their count or coverage contribution. A test earns its place only when its
removal would make a realistic product defect materially harder to detect.

The initial static inventory contains 398 Python files under `backend/tests`,
7,005 test functions or methods, and about 157,000 lines. First-party frontend
behavior tests are absent; several backend tests inspect TypeScript source text
instead.

## Decision rules

Delete a test when it only:

- checks source spelling, statement order, a private method, or an import shape;
- executes a copied old implementation or a mock that validates the test itself;
- rechecks Python, PyTorch, Pydantic, or another dependency's documented behavior;
- duplicates a stronger test of the same externally visible contract; or
- divides one equivalence class into cases that cannot expose different defects.

Consolidate tests when one table-driven or registry-driven test can retain the
same failure localization. Preserve independent numerical oracles, persistence
round trips, corruption handling, concurrency behavior, public API contracts,
and architecture-specific tensor layouts.

## Execution units

1. Remove source-only UI tests and self-validating negative controls that have
   no production execution path.
2. Replace source/AST wiring assertions with behavior tests at the nearest
   stable boundary.
3. Consolidate repeated parameter-threading tests into schema-driven contracts.
4. Consolidate architecture-neutral adapter, CFG-null, REPA, sample-generation,
   VAE-swap, and optimizer tests around their registries or shared engines.
5. Reduce excessive LR-scheduler, VAE-refusal, video-chain, MiniMax-H3,
   MiniMax-Music3, and SenseNova case matrices by equivalence class.
6. Move manual GPU probes and benchmarks out of test discovery naming; retain
   only documented probes that measure behavior unavailable to CPU tests.

Each unit is committed separately. A deletion unit must record the concrete
remaining test that detects the relevant defect, or state that the removed test
had no product-behavior assertion.

## Verification

- Run the smallest affected CPU-only test set after each unit.
- Compile and import changed backend modules if production code changes.
- Do not run GPU/model-loading tests while an owner training run is active.
- At the end, collect the remaining suite and compare collection errors, not
  coverage percentage or raw test count.

## Deferred work

Browser-level tests are needed before source-text checks of navigation,
generation persistence, and training-form restoration can be replaced with
real frontend behavior coverage. Their absence does not justify retaining tests
that only prove particular TypeScript strings exist.
