# Code comment hygiene plan (2026-09)

## Goal

Reduce maintenance noise without changing runtime or test behaviour. Code must
carry the implementation; comments are retained only where they explain why an
apparently simpler alternative is unsafe, record a measured constraint, or
state a compatibility contract that the code cannot express.

## Scope

The audit covers tracked Python, TypeScript, JavaScript, C++ and CUDA sources.
Generated declarations, vendored code and third-party sources are excluded.

The work is split into independently reviewable commits:

1. Remove disabled legacy implementations and debug snippets.
2. Remove procedural narration from backend inference and API code.
3. Remove procedural narration from training code.
4. Remove procedural narration from frontend code.
5. Reduce test comments to the behaviour each test verifies.

## Rules

- Delete comments that merely repeat the following statement or function name.
- Delete commented-out code; Git history is its archive.
- Delete investigation history, mutation-test narratives and decorative
  separators.
- Keep concise explanations of non-obvious ordering, shape, numerical, memory,
  concurrency and backward-compatibility constraints.
- Keep public API contracts when a type or signature does not express them.
- Move durable design rationale to the existing design document rather than
  reproducing it beside an implementation.
- Do not rename symbols, reorder statements or rewrite executable expressions
  as part of this pass.

## Verification

- Inspect each diff with whitespace ignored to ensure only comments, docstrings
  and adjacent blank lines changed.
- Compile every changed Python file and import changed backend modules with CUDA
  initialisation stubbed where the module reaches the trainer or pipeline stack.
- Run focused tests for source-sensitive files and the affected subsystem.
- Do not run frontend build or type-check commands; repository policy reserves
  those for the owner.

## Audit baseline

The initial heuristic scan covered 735 tracked production source files and 388
test files. It found 2,749 procedural-comment candidates, 1,772 decorative
separator candidates, 552 possibly trivial docstrings, 43 disabled debug lines,
and 119 test docstrings containing mutation-history narration. These are review
queues, not deletion targets: every candidate is judged in context.
