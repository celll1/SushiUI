# Documentation hygiene audit and cleanup plan

Date: 2026-09-11

## Scope and privacy decision

This audit covers the current tracked tree and its boundary with ignored local
working material. It does not rewrite published history and does not decide the
outstanding third-party redistribution questions in `docs/legal/`.

The current tree contains no detected private key, service-token signature,
credential assignment, authenticated URL, email address, or Windows user
profile path. Repository-owner names and generic model-drive examples do not
justify history rewriting. Current documentation should nevertheless use
portable placeholders, and hygiene tooling must not embed a maintainer's local
user name.

## Publication rule

Tracked documentation retains only material needed to understand, operate,
verify, or legally redistribute the implementation that exists now:

- current behavior and maintenance contracts;
- decisions that still constrain that behavior;
- completed, reproducible evidence supporting a current claim; and
- third-party provenance and required notices.

The ignored `local/` tree owns material that does not need to ship:

- unimplemented, rejected, or deferred product ideas;
- completed implementation plans and chronological work logs;
- source-reading notes and clean-room separation records; and
- machine-specific raw measurements.

When a mixed document contains both kinds, preserve the original under
`local/docs/archive/` and rewrite the tracked document around the shipped
contract. A tracked document must not depend on that local copy.

## Findings

1. `tools/doc_hygiene.py` embeds a local user name and checks only a narrow set
   of path forms.
2. Five tracked files under `docs/plans/` and five binary figures under
   `docs/rope_analysis/` sat outside the published taxonomy and were archived.
3. The documentation map omits standalone tracked documents and presents most
   paths as code spans rather than navigable links.
4. Completed plans, superseded audits, current contracts, and open ideas are
   mixed across `docs/guides/` and `docs/audits/`.
5. Raw GPU-result JSON contains machine-specific checkpoint paths even though
   the durable conclusions are already summarized in tracked audits.
6. Large mixed guides retain deferred feature designs and references to ignored
   scratchpad evidence, allowing their public and private halves to drift.

## Execution units

1. Generalize the privacy checks and sanitize current machine-specific paths.
2. Archive standalone plans, superseded work logs, deferred validation plans,
   and machine-specific raw measurements under ignored `local/` directories.
3. Split mixed guides: keep shipped behavior in tracked documents and archive
   unimplemented alternatives locally.
4. Rebuild the documentation map around the resulting ownership boundaries and
   make its entries navigable.
5. Strengthen the hygiene check for the full taxonomy, index coverage, tracked
   targets, credential signatures, and structured result files.

Each unit is committed separately. Moving a file to `local/` preserves the
working copy but intentionally records its removal from the public tree.

## Verification

- Search tracked content for credentials, authenticated URLs, private user
  paths, and references to ignored working material without printing values.
- Require every file under `docs/` to belong to the declared taxonomy.
- Require every relative documentation target to exist in a fresh clone.
- Confirm `docs/guides/DOC_MAP.md` covers every maintained document either
  directly or through an explicit sub-index.
- Keep the worktree's ignored archive present after its tracked source is
  removed.
