# Training run storage v2

Status: v2 diffusion/VAE metric storage and terminal-run migration are
implemented. The legacy store remains supported; central purge and tagger-run
detail stores are not yet implemented.

## Decision

`training.db` becomes the lightweight catalogue and control-plane database.
Detailed, run-owned state moves to `<output_dir>/training_run.db` for new runs.
The list endpoint must never open a run database. Detail endpoints resolve the
run's declared store and preserve their existing response contracts.

This is a storage-boundary change, not the assumed fix for the current Training
page delay. On 2026-09-10 the existing 39-row summary SQL measured a 0.63 ms
median over 20 read-only connections, while direct requests to the running
`GET /api/v1/training/runs` endpoint completed in 7--9 ms after warm-up. The
frontend-visible delay therefore needs its own measurement through the browser
and proxy path.

The storage change is still justified. The measured `training.db` was
497,500,160 bytes and held 2,066,635 diffusion-training metric rows, 252,426
tagger metric rows, and 101,944 diffusion metric rows whose parent run no longer
existed. A single ever-growing history file makes retention, backup, deletion,
and run portability unnecessarily broad operations.

## Ownership boundary

The catalogue keeps only data needed to list, select, start, stop, and locate a
run:

- stable integer ID and UUID, name, method, architecture summary;
- status, phase, progress, current/total step, and latest scalar summary;
- output directory, short error summary, and lifecycle timestamps;
- detail-store kind, schema version, and availability/migration state.

The run database owns:

- the immutable configuration and dataset snapshots used by the run;
- per-step metrics, extra metrics, resume segments, and validation metrics;
- checkpoint and sample metadata;
- structured warnings and diagnostic events.

Weights, samples, logs, TensorBoard events, and diagnostic sidecars remain
ordinary files. Paths inside the output directory are stored relative to it so
the directory remains relocatable.

## Store identity and schema

The catalogue gains nullable store-discriminator columns. A missing discriminator
means the legacy central store, so an untouched database remains readable:

- `detail_store`: `central_v1` or `run_db_v2`;
- `detail_schema_version`;
- `detail_state`: `pending`, `ready`, `unavailable`, or `migrating`;
- `detail_db_name`, normally `training_run.db`.

Every run database contains a one-row identity table with the catalogue run UUID,
catalogue integer ID at creation, and schema version. A resolver refuses a file
whose UUID does not match; an accidentally copied database must never be shown as
another run.

Run-database migrations use their own schema version and run only when that one
run is opened. The list endpoint neither discovers files nor runs migrations.

## Read and write rules

`GET /training/runs` reads the catalogue only. Existing detail, metrics,
checkpoint, and sample endpoints call one store resolver:

1. `central_v1` reads the current tables in `training.db`.
2. `run_db_v2` reads `<output_dir>/<detail_db_name>` after identity validation.
3. While legacy rows are retained, a missing or invalid v2 file may explicitly
   fall back to legacy data and returns an availability warning; it must not
   silently return an empty history.

The trainer writes detailed rows only to its selected store. It mirrors current
step, progress, phase, latest loss/rate, and status to the catalogue through the
existing asynchronous progress path. Catalogue state is authoritative for
control and listing; the run database is authoritative for detail. Operations
across the two files are idempotent rather than pretending to be one atomic
transaction.

Active run databases use WAL only on a local filesystem with reliable shared
locking. A non-local or unsupported output filesystem uses a local active store
with an explicit completed-run snapshot policy; it must not be assumed safe for
SQLite WAL merely because a path is writable.

## Compatibility and migration

Migration is opt-in per terminal run. A running or starting run stays on its
current store until it stops.

1. Add the nullable catalogue columns and the resolver without changing any
   row's behavior.
2. Exercise v2 with dual-write for new runs, keeping legacy detailed rows usable
   by the previous application version during the rollback window.
3. Stop central detail writes for newly created v2 runs after parity tests pass.
4. For a terminal legacy run, copy into `training_run.db.tmp`, validate identity,
   row counts, step bounds, representative values, and `integrity_check`, close
   it, then atomically rename it in the output directory.
5. Flip the catalogue discriminator only after the final file is durable. A
   crash before that point leaves the legacy reader authoritative.
6. Retain central rows for a defined rollback period. Purging them is a separate,
   explicit transaction. File compaction is a later maintenance action performed
   only with training stopped.

Copying, verification, discriminator flip, purge, and compaction are distinct
operations. The migration command is restartable and records enough state to
distinguish an abandoned temporary copy from an authoritative run database.

An old `training.db` therefore continues to work unchanged with new code. During
the dual-write window, rolling back the application also keeps new runs readable.
After central rows for a v2 run are purged, rollback to a version that has no v2
reader is no longer promised.

## Deletion and retention

Before v2 migration, run deletion must explicitly delete dependent metric rows
in the same central transaction. SQLite foreign-key enforcement is enabled on
every connection, but application-level deletion remains explicit for tagger
metrics whose string run key is not currently a foreign key.

Deleting a catalogue entry and deleting its output artifacts are different
operations. The existing delete endpoint continues to remove the registered run
record; it does not silently erase the output directory or its run database.
A future destructive artifact-delete operation requires a separate explicit API.

Dense history retention is applied per run, never by rewriting every run during
page load. Any future downsampling must preserve resume boundaries, extrema, and
the original database or an explicit archive until the user accepts the loss of
raw points.

## Verification gates

The implementation must cover:

- legacy-only, v2-only, dual-written, missing-v2, corrupt-v2, and wrong-identity
  fixtures through the unchanged API response contracts;
- interruption before and after every migration state transition;
- resume-from-earlier-step cleanup and same-step partial metric updates;
- concurrent trainer writes and API reads on the run database;
- explicit dependent-row deletion for diffusion and tagger runs;
- an unavailable or moved output directory without slowing or breaking lists;
- a synthetic catalogue whose run databases contain millions of metrics, proving
  the list path opens no detail database;
- CPU-only compile/import tests. No CUDA or VRAM test is required for this change.

## Delivery units

1. Measure the complete browser/proxy/list path and add a regression benchmark.
2. Correct central deletion and connection-integrity behavior.
3. Add store metadata and a legacy-compatible resolver.
4. Add the run-database schema and dual-write implementation for new runs.
5. Route detail APIs through the resolver without changing their payloads.
6. Add the restartable terminal-run migration and verification command.
7. End dual-write after parity validation, then expose explicit legacy purge.
8. Add cursor pagination and incremental frontend loading independently of the
   storage migration if browser measurement shows it is needed.

## Operator boundary

New diffusion and VAE runs create `training_run.db` automatically. During the
rollback window their metrics are also retained centrally. Existing runs are
not moved automatically. After the updated backend has initialized the nullable
catalogue columns, inspect candidates without writing:

```powershell
venv\Scripts\python.exe backend\migrations\migrate_training_run_details.py --all-terminal
```

Migrate selected terminal runs explicitly:

```powershell
venv\Scripts\python.exe backend\migrations\migrate_training_run_details.py --run-id 123 --apply
```

The command copies and verifies data but does not purge central rows or compact
`training.db`. Those remain separate future operations so migration itself is
rollback-safe.
