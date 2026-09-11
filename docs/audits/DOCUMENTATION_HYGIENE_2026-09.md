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

## Local account identifier history rewrite

One historical revision of the hygiene checker embedded the repository owner's
local account identifier as a literal detection pattern. Removing it from the
current tree is insufficient because the blob remains addressable in Git
history.

The rewrite procedure is deliberately separate from ordinary document cleanup:

1. create and verify an ignored Git bundle containing every current ref;
2. identify every local and remote-tracking ref that contains the affected
   revision;
3. rewrite only the offending source literal without changing unrelated file
   contents;
4. verify every rewritten ref, compare the current tree with its pre-rewrite
   tree, and search all reachable revisions for the identifier;
5. retain the private bundle until the repository owner confirms the rewritten
   remote history is healthy.

The backup necessarily contains the removed identifier and must not be
published. Rewriting the local refs does not update the hosted repository;
force-pushing affected branches remains an explicit owner action.

Status: completed locally on 2026-09-11. The `flux2` branch, its
remote-tracking ref, one detached worktree, and five Codex checkpoint-tree refs
were rewritten. The current tree remained byte-identical, the detached
worktree's two pre-existing uncommitted files were hash-verified before and
after, and no affected commit or blob remains reachable from local refs. The
verified ignored bundle is retained; the hosted `flux2` branch still requires
an explicit lease-protected force-push.

Post-rewrite integrity audit: the affected interval is a linear sequence of
441 commits from the unchanged base `5e6351e0`. Every old commit maps to exactly
one new commit with the same parent relationship, author and committer
identity, timestamps, encoding, and message. Their combined metadata stream has
SHA-256 `9B5093DA04A6A6D0F32409BD83F4D73DBE201A2762053B77954383563CF672F6`
on both sides. Comparing every complete tree produced 435 changed entries over
432 snapshots; every entry was one of the three predeclared old-blob to
redacted-blob substitutions, with unchanged file mode and path. No other blob,
file, or topology change was found. The old and rewritten remote-tracking tips
also match the same mapping.
