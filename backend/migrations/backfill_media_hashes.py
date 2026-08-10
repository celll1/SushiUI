"""One-time backfill: fill `image_hash` for existing video/audio gallery rows.

Background
----------
Every video/audio `create_db_image_record` call site used to hardcode
`image_hash=""` (see backend/api/routes.py, the 8 media generation routes).
That made a video/audio row unable to ever be the TARGET of a
`source_image_hash` link -- clicking a gallery item's "source" link, when the
source was a video, could never resolve.

The routes now hash the saved MASTER file's bytes (sha256) instead. This
script backfills that same hash for rows that were written before the fix,
so old rows become resolvable through `GET /images/by-hash/{hash}` too.

Scope (READ CAREFULLY BEFORE RUNNING WITHOUT --dry-run)
--------------------------------------------------------
- Only touches rows where `image_hash` is the EMPTY STRING (`''`) -- the
  video/audio sentinel the old code always wrote.
- NEVER touches rows where `image_hash IS NULL`. Those are unrelated,
  pre-existing legacy STILL-IMAGE rows from a filesystem scan (~5872 of
  them); NULL and '' are different states with different histories and this
  script must not conflate them.
- Only considers rows whose filename extension is a media type this backend
  actually produces for video/audio: .mp4, .mkv, .webm (video), .flac, .wav
  (audio). Anything else with image_hash='' (should not currently exist, but
  is deliberately left alone if it ever does) is skipped and reported.
- Idempotent: re-running after a successful (non-dry-run) pass finds nothing
  left to do, because every backfilled row's image_hash is no longer ''.
- Skips (and reports) rows whose file is missing under outputs/.
- Does NOT touch `source_image_hash`. ~33 existing outpaint_vid/inpaint_vid
  rows carry a legacy frame-0 PIXEL hash there (from before the source side
  was also switched to file-bytes hashing) that this backfill cannot make
  resolvable -- ffmpeg's decode of an already-lossy re-encode is not
  reversible back to the original upload's bytes. These are reported in the
  summary as "unrecoverable legacy source hashes"; nothing is done for them.

Usage
-----
    venv/Scripts/python.exe scripts/db/backfill_media_hashes.py --dry-run
    venv/Scripts/python.exe scripts/db/backfill_media_hashes.py

Always run --dry-run first and inspect the report before writing.
"""

import argparse
import hashlib
import os
import sqlite3
import sys

# Repo root: scripts/db/this_file.py -> scripts/db -> scripts -> repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GALLERY_DB_PATH = os.path.join(_REPO_ROOT, "gallery.db")
_OUTPUTS_DIR = os.path.join(_REPO_ROOT, "outputs")

# Extensions this backend's video/audio routes actually write (see
# backend/utils/video_utils.py::save_video_with_metadata and
# backend/utils/audio_utils.py::save_audio_with_metadata). .webm/.wav are
# included for forward-compatibility even though no current row uses them.
_MEDIA_EXTENSIONS = {".mp4", ".mkv", ".webm", ".flac", ".wav"}

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _sha256_file(path: str) -> str:
    """Same algorithm as backend/utils/image_utils.py::calculate_file_hash,
    reimplemented here so this script has no dependency on importing the
    backend package (which pulls in heavyweight startup side effects)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing to gallery.db.",
    )
    parser.add_argument(
        "--db-path",
        default=_GALLERY_DB_PATH,
        help=f"Path to gallery.db (default: {_GALLERY_DB_PATH}).",
    )
    parser.add_argument(
        "--outputs-dir",
        default=_OUTPUTS_DIR,
        help=f"Path to the outputs directory (default: {_OUTPUTS_DIR}).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"[Backfill] ERROR: gallery.db not found at {args.db_path}")
        return 1

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, filename, parameters FROM generated_images WHERE image_hash = ''"
        ).fetchall()

        candidates = []
        skipped_wrong_ext = []
        for row in rows:
            _, ext = os.path.splitext(row["filename"])
            if ext.lower() in _MEDIA_EXTENSIONS:
                candidates.append(row)
            else:
                skipped_wrong_ext.append(row)

        updated = []
        missing_files = []
        for row in candidates:
            file_path = os.path.join(args.outputs_dir, row["filename"])
            if not os.path.exists(file_path):
                missing_files.append(row["filename"])
                continue
            file_hash = _sha256_file(file_path)
            updated.append((row["id"], row["filename"], file_hash))

        if not args.dry_run:
            for image_id, _filename, file_hash in updated:
                conn.execute(
                    "UPDATE generated_images SET image_hash = ? WHERE id = ?",
                    (file_hash, image_id),
                )
            conn.commit()

        # Informational only: legacy frame-0 PIXEL source_image_hash rows this
        # backfill cannot make resolvable (see module docstring). Not touched.
        legacy_source_rows = conn.execute(
            """
            SELECT id, generation_type FROM generated_images
            WHERE generation_type IN ('outpaint_vid', 'inpaint_vid')
              AND source_image_hash IS NOT NULL
              AND source_image_hash != ''
            """
        ).fetchall()
        # hash_kind='file_bytes' rows (post-fix) ARE resolvable-by-source
        # already; only count ones that predate the fix.
        legacy_source_ids = []
        for r in legacy_source_rows:
            params_row = conn.execute(
                "SELECT parameters FROM generated_images WHERE id = ?", (r["id"],)
            ).fetchone()
            raw = params_row["parameters"] if params_row else None
            has_file_bytes_kind = raw is not None and '"hash_kind": "file_bytes"' in raw
            if not has_file_bytes_kind:
                legacy_source_ids.append(r["id"])

    finally:
        conn.close()

    print("=" * 70)
    print(f"[Backfill] Mode: {'DRY RUN (no writes)' if args.dry_run else 'WRITE'}")
    print(f"[Backfill] gallery.db: {args.db_path}")
    print(f"[Backfill] outputs dir: {args.outputs_dir}")
    print("-" * 70)
    print(f"[Backfill] Rows with image_hash='': {len(rows)}")
    print(f"[Backfill]   -> media-extension candidates: {len(candidates)}")
    print(f"[Backfill]   -> skipped (non-media extension, left untouched): {len(skipped_wrong_ext)}")
    for r in skipped_wrong_ext:
        print(f"[Backfill]      id={r['id']} filename={r['filename']!r}")
    print(f"[Backfill]   -> missing file on disk (left untouched): {len(missing_files)}")
    for fn in missing_files:
        print(f"[Backfill]      {fn}")
    print(f"[Backfill]   -> {'would update' if args.dry_run else 'updated'}: {len(updated)}")
    for image_id, filename, file_hash in updated:
        print(f"[Backfill]      id={image_id} filename={filename!r} -> image_hash={file_hash}")
    print("-" * 70)
    print(
        f"[Backfill] NOTE: {len(legacy_source_ids)} outpaint_vid/inpaint_vid row(s) carry a "
        "pre-fix frame-0 PIXEL source_image_hash that this backfill does NOT and cannot "
        "repair (ffmpeg's decode of an already-lossy re-encode does not reverse back to the "
        "original upload's bytes). These rows' \"source\" link will keep 404ing even after "
        "this backfill runs. This is accepted/intentional -- see the module docstring."
    )
    if legacy_source_ids:
        print(f"[Backfill]   affected ids: {legacy_source_ids}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
