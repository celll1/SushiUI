# Documentation policy

This directory contains the documentation that is safe and useful to ship with
SushiUI.  A tracked document must help a fresh clone understand, operate,
verify, or legally redistribute the current codebase.

## Tracked documentation

| Location | Purpose |
|---|---|
| `docs/guides/` | Current procedures, architecture facts, and maintained implementation contracts. |
| `docs/reference/` | Stable reference material derived from the current implementation. |
| `docs/decisions/` | Decisions whose rationale still constrains the current implementation. |
| `docs/audits/` | Completed audits with a stated scope, evidence, and date. |
| `docs/legal/` | Third-party provenance, notices, and license copies. |
| Code-adjacent Markdown | Documentation whose usefulness depends on living beside the code it describes. |

`docs/guides/DOC_MAP.md` is the detailed index. `AGENTS.md` is the short task
router for coding agents; it is not a second documentation index.

## Non-tracked working material

The following material belongs under `local/`, whose contents are ignored by
Git except for `local/README.md`:

| Location | Purpose |
|---|---|
| `local/strategy/` | Roadmaps, future product direction, and implementation proposals. |
| `local/research/` | Paper notes, exploratory comparisons, and source excerpts. |
| `local/measurements/` | Raw benchmark output and machine-specific measurements. |
| `local/docs/` | Drafts and historical documents retained only for local reference. |

Throwaway scripts, generated samples, and transient investigations belong in
`scratchpad/`. A local document may be absent from another clone and must never
be the only source of a current behavior contract.

When a design ships, extract the enduring behavior and constraints into a
tracked guide, reference, decision, or code-adjacent document. Keep the
roadmap, abandoned alternatives, and chronological work log local.

## Content rules

- Describe the current implementation in the present tense. Mark measured,
  inferred, proposed, and unverified claims explicitly.
- Use repository-relative paths. In examples use placeholders such as
  `<REPO_ROOT>`, `<MODEL_ROOT>`, and `<OUTPUT_ROOT>`; do not record a person's
  drive letters, user directory, host name, or environment layout.
- Never include credentials, tokens, private keys, cookies, or real secret
  values. Examples must use conspicuously synthetic placeholders.
- Paraphrase external material and link or identify the source. Do not copy
  paper prose, source comments, or implementation text into a design note.
- Vendored or adapted code must have a verifiable upstream repository, version
  or commit when available, upstream license, modification status, and any
  required notice or license copy recorded in `docs/legal/`.
- Do not assert a license when it has not been verified. Record the uncertainty
  and block redistribution of the affected material until it is resolved.
- Do not create a project-level license by inference. The repository owner must
  make that licensing decision explicitly.

## Review checklist

Before committing a documentation change:

1. Confirm that the document belongs in the tracked taxonomy above.
2. Check that links and paths resolve in a fresh clone.
3. Search the changed text for personal paths and credential-like values.
4. Check quoted or adapted material against the provenance ledger.
5. Update `docs/guides/DOC_MAP.md` when a tracked document is added, removed,
   renamed, or changes purpose materially.
