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

## Outcome

The five implementation units removed 6,234 lines from 415 file revisions:

| Commit | Unit | Result |
|---|---|---|
| `5cc12f20` | Legacy/debug | Removed retired implementations and disabled debug snippets from 8 files. |
| `8eae417e` | Backend | Removed 1,435 lines of API, inference and pipeline narration from 48 files. |
| `9022243c` | Training | Removed 1,355 lines of training narration from 86 files. |
| `23587b8a` | Frontend | Removed 652 lines of component and request-flow narration from 70 files. |
| `5d356653` | Tests | Removed 2,488 lines of section banners, procedural narration and mutation-investigation history from 203 files. |

Public API docstrings were retained because FastAPI can expose them through the
generated schema. Comments explaining numerical domains, tensor shapes,
ordering, ownership, synchronization, memory lifetime, compatibility or a
known unsafe simplification were also retained. A final conservative scan still
identified 938 lines by lexical shape; contextual review rejected that batch
because it mixed decorative headings with those load-bearing explanations.
Lexical candidate counts are therefore not a completion target.

## Verification result

- Every changed Python production file compiled, and changed backend modules
  imported with CUDA initialization stubbed; CUDA remained uninitialized.
- Every changed frontend line was verified as a removed full-line comment.
  Build and type-check commands were intentionally not run under repository
  policy.
- All 203 test-file revisions were compared to their parent after removing
  docstrings from both syntax trees. Their executable ASTs were identical.
- A focused CPU-only run completed 266 tests successfully. Three failures were
  unrelated existing expectations: one configuration-key census mismatch, one
  Lion extension link collision, and one obsolete SenseNova block-swap refusal
  expectation.
- `git diff --check` passed for every accepted unit.
