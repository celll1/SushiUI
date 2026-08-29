# local/ — machine-local working area (NOT version controlled)

Everything in this directory except this README is excluded from version
control (`.gitignore`: `local/*`). It is scratch space for one machine: ad-hoc
scripts, debugging utilities, experiment leftovers, and reports that are not
part of the product.

The directory used to be called `scripts/` (with a sibling root `tests/`).
Both names implied "this is where the project's scripts and tests live", which
was wrong in both directions — the tracked ones live elsewhere — so they were
merged here under a name that states what the directory actually is.

## Where the tracked equivalents live

| If you are looking for | It is in |
|---|---|
| One-time DB migrations and backfills | `backend/migrations/` |
| The backend test suite | `backend/tests/` |
| Schema/DB migration procedure | `docs/guides/DATABASE_MIGRATION.md` |

A script that the repository genuinely needs — one that another clone would
have to run, or that documentation refers to — does not belong here. Put it in
`backend/migrations/` (or the appropriate tracked location) so it ships with
the code.

## Layout

- `local/strategy/` — roadmaps, future direction and implementation proposals
- `local/research/` — paper notes, exploratory comparisons and source excerpts
- `local/measurements/` — raw benchmarks and machine-specific measurements
- `local/docs/` — drafts and historical documents retained for local reference
- `local/tests/` — scratch tests and debugging utilities

Tracked documentation policy and the public taxonomy live in
`docs/README.md`. A current behavior contract must not exist only under
`local/`; extract it into a tracked guide, reference, decision, audit, or
code-adjacent document first.

Note that several filename patterns are ignored repository-wide regardless of
directory (`test_*.py`, `verify_*.py`, `check_*.py`, `debug_*.py`,
`*_backup.py`). Tracked files matching those patterns need an explicit
negation in `.gitignore`; see the existing exceptions there for examples.
