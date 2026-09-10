# Run 127 schema-safe resume fix

## Failure

The training child queried `TrainingRun` before reconciling the central
`training.db` schema. A backend kept alive across the detail-store deployment
therefore spawned new code against the old table and failed on the missing
`training_runs.detail_store` column before checkpoint discovery or model load.

## Change

1. Add an idempotent training-only schema reconciliation entry point.
2. Call it in `train_runner` before opening its first ORM session.
3. Fail with the original migration error instead of issuing the same broken ORM
   query again from the exception handler.
4. Test an old SQLite schema through migration and the startup ordering through
   source inspection.

The child may add missing nullable columns but does not migrate historical
metrics into per-run databases. Existing central rows therefore retain the
documented `central_v1` compatibility behavior until the explicit detail-store
migration is run.
