"""Transactional cleanup of rows owned by a training run."""

from sqlalchemy.orm import Session

from .models import StyleTrainingMetrics, TaggerTrainingMetrics, TrainingMetrics


def delete_training_run_record(db: Session, run) -> None:
    """Delete one diffusion/VAE run and its central detail rows.

    Keep this explicit even with SQLite foreign keys enabled: old connections
    and imported databases may not have enforced them, and metrics are not an
    ORM relationship on ``TrainingRun``.
    """
    db.query(TrainingMetrics).filter(
        TrainingMetrics.run_id == run.id
    ).delete(synchronize_session=False)
    db.delete(run)


def delete_tagger_training_run_record(db: Session, run) -> None:
    """Delete one tagger run and metrics keyed by its string run ID."""
    db.query(TaggerTrainingMetrics).filter(
        TaggerTrainingMetrics.run_id == run.run_id
    ).delete(synchronize_session=False)
    db.delete(run)


def delete_style_training_run_record(db: Session, run) -> None:
    """Delete one style run's catalogue row and any central metrics under its run ID.

    Its per-run detail database and artifacts in the output directory are kept.
    """
    db.query(StyleTrainingMetrics).filter(
        StyleTrainingMetrics.run_id == run.run_id
    ).delete(synchronize_session=False)
    db.delete(run)
