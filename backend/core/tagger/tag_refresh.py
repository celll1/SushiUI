"""Live tag-refresh for tagger training (mid-epoch, zero iter-loop overhead).

The dataset is built once at training start and each DataLoader worker holds a
frozen pickled snapshot of its ``_samples``. If the user edits tags in the UI
during training (which writes ``DatasetCaption`` rows in datasets.db), those
edits would otherwise not appear until the run is restarted.

This module closes that gap without touching the GPU/iter critical path:

  * ``TagRefreshDetector`` — a background thread (no GPU) that polls datasets.db
    for captions whose ``updated_at`` advanced since training start, rebuilds the
    effective tag list for the affected samples, and publishes a cumulative
    ``{sample_idx: [tag, ...]}`` override map.

  * ``TagRefreshStore`` — the IPC channel. Overrides are written to a payload
    file (atomic replace) and a monotonically-increasing *generation* counter is
    written to a tiny mmap'd file. Workers read the 8-byte generation with a pure
    memory read on every ``__getitem__`` (no syscall, off the GPU critical path
    because workers run ahead via prefetch) and only re-read the payload file when
    the generation changed — i.e. almost never.

The detection query lives entirely on the background thread, so it never slows a
training iteration. A cheap datasets.db file-mtime gate skips the SQL scan when
nothing has been written at all.
"""
from __future__ import annotations

import mmap
import os
import pickle
import sqlite3
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

_GEN_BYTES = 8  # little-endian uint64 generation counter


# ----------------------------------------------------------------------------
# IPC store (writer side, main process)
# ----------------------------------------------------------------------------

class TagRefreshStore:
    """Generation-gated override channel shared with DataLoader workers.

    Files live under *base_dir*:
      - ``<prefix>_gen.bin``       : 8-byte little-endian generation counter (mmap)
      - ``<prefix>_overrides.pkl`` : pickled ``{sample_idx: [tag, ...]}`` map
    """

    def __init__(self, base_dir: str, prefix: str = "tag_refresh") -> None:
        os.makedirs(base_dir, exist_ok=True)
        self.gen_path     = os.path.join(base_dir, f"{prefix}_gen.bin")
        self.payload_path = os.path.join(base_dir, f"{prefix}_overrides.pkl")
        self._generation  = 0
        # Initialise payload (empty) then the generation file, so a worker that
        # attaches and sees generation 0 always finds a readable payload.
        self._write_payload({})
        with open(self.gen_path, "wb") as fh:
            fh.write((0).to_bytes(_GEN_BYTES, "little"))
        self._fh = open(self.gen_path, "r+b")
        self._mm = mmap.mmap(self._fh.fileno(), _GEN_BYTES)

    def _write_payload(self, overrides: Dict[int, List[str]]) -> None:
        tmp = self.payload_path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(overrides, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.payload_path)  # atomic on the same filesystem

    def publish(self, overrides: Dict[int, List[str]]) -> int:
        """Write the full cumulative override map and bump the generation.

        Payload is written *before* the generation bump so a worker that observes
        the new generation always finds the matching (complete) payload.
        """
        self._write_payload(overrides)
        self._generation += 1
        self._mm[0:_GEN_BYTES] = self._generation.to_bytes(_GEN_BYTES, "little")
        self._mm.flush()
        return self._generation

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass

    def cleanup_files(self) -> None:
        for p in (self.gen_path, self.payload_path, self.payload_path + ".tmp"):
            try:
                os.remove(p)
            except OSError:
                pass


# ----------------------------------------------------------------------------
# Worker-side reader (called from TaggerDataset.__getitem__)
# ----------------------------------------------------------------------------

class TagRefreshReader:
    """Per-worker reader. Cheap generation check + lazy payload reload.

    Construct once per worker with the two file paths; call ``override(idx)`` from
    ``__getitem__``. Steady-state cost is one mmap memory read + one dict lookup.
    """

    def __init__(self, gen_path: str, payload_path: str) -> None:
        self.gen_path     = gen_path
        self.payload_path = payload_path
        self._fh = None
        self._mm: Optional[mmap.mmap] = None
        self._local_gen = -1
        self._overrides: Dict[int, List[str]] = {}
        self._disabled = False

    def _ensure_mm(self) -> bool:
        if self._mm is not None:
            return True
        if self._disabled:
            return False
        try:
            self._fh = open(self.gen_path, "rb")
            self._mm = mmap.mmap(self._fh.fileno(), _GEN_BYTES, access=mmap.ACCESS_READ)
            return True
        except OSError:
            # File not ready yet — try again on a later __getitem__.
            return False

    def _reload(self, gen: int) -> None:
        try:
            with open(self.payload_path, "rb") as fh:
                self._overrides = pickle.load(fh)
            self._local_gen = gen
        except (OSError, pickle.UnpicklingError, EOFError):
            # Mid-write or missing — keep the old map and retry on the next call
            # (generation still differs, so we will try again).
            pass

    def override(self, idx: int) -> Optional[List[str]]:
        """Return replacement tags for *idx*, or None when unchanged."""
        if not self._ensure_mm():
            return None
        gen = int.from_bytes(self._mm[0:_GEN_BYTES], "little")
        if gen != self._local_gen:
            self._reload(gen)
        return self._overrides.get(idx)


# ----------------------------------------------------------------------------
# Detector (background thread, main process)
# ----------------------------------------------------------------------------

class TagRefreshDetector:
    """Polls datasets.db for edited captions and publishes overrides.

    Parameters
    ----------
    db_path        : datasets.db filesystem path
    dataset_ids    : training dataset ids (scope of the edit query)
    item_ids       : numpy int64 array aligned to dataset ``_samples`` (sample idx
                     -> item_id), used to map edited item_ids back to sample idx
    caption_types  : optional caption_type filter (same as the dataset build)
    comma_resolver / alias_resolver : same resolvers used to build the vocabulary,
                     so refreshed tags canonicalise identically
    store          : TagRefreshStore to publish through
    interval       : poll period in seconds
    """

    def __init__(
        self,
        db_path: str,
        dataset_ids: List[int],
        item_ids,                       # np.ndarray[int64]
        caption_types: Optional[List[str]],
        comma_resolver,
        alias_resolver,
        store: TagRefreshStore,
        interval: float = 60.0,
    ) -> None:
        self.db_path        = db_path
        self.dataset_ids    = list(dataset_ids)
        self._item_ids      = item_ids
        self.caption_types  = caption_types
        self._comma_resolver = comma_resolver
        self._alias_resolver = alias_resolver
        self.store          = store
        self.interval       = max(5.0, float(interval))

        self._overrides: Dict[int, List[str]] = {}  # cumulative
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Only edits AFTER training start count.
        self._last_seen = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
        self._db_mtime  = self._safe_mtime()
        # Map item_id -> sample idx, built lazily once (numpy, no giant dict).
        self._id_to_idx: Optional[Dict[int, int]] = None

    def _safe_mtime(self) -> float:
        # datasets.db runs in WAL mode: edits land in datasets.db-wal and the
        # main file's mtime does NOT advance until a checkpoint. Gating on the
        # main file alone would miss (or badly delay) edits, so take the newest
        # mtime across the db and its -wal / -shm sidecars. Still just a few
        # cheap stat() syscalls.
        newest = 0.0
        for suffix in ("", "-wal", "-shm"):
            try:
                m = os.path.getmtime(self.db_path + suffix)
                if m > newest:
                    newest = m
            except OSError:
                pass
        return newest

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="TagRefreshDetector", daemon=True)
        self._thread.start()
        print(f"[TagRefresh] Detector started (interval={self.interval:.0f}s, "
              f"datasets={self.dataset_ids})")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.store.close()

    # -- internals ---------------------------------------------------------

    def _idx_for_item_ids(self, changed_ids: List[int]) -> Dict[int, int]:
        """Return {item_id: sample_idx} for the changed ids using numpy (no giant
        Python dict). One O(N) pass over the aligned item-id array per cycle."""
        import numpy as np
        ids = self._item_ids
        if ids is None or len(ids) == 0 or not changed_ids:
            return {}
        changed = np.asarray(list(set(changed_ids)), dtype=np.int64)
        mask = np.isin(ids, changed)
        pos = np.nonzero(mask)[0]
        return {int(ids[p]): int(p) for p in pos}

    def _ensure_index(self) -> None:
        """Create an index on dataset_captions.updated_at if it is missing.

        The poll filters on ``updated_at``, which is otherwise unindexed — so each
        time the change-gate trips it would full-scan the (potentially multi-
        million-row) dataset_captions table. ``updated_at`` is a high-cardinality
        timestamp, so an index turns ``updated_at > ?`` into a cheap range seek
        with no risk of the low-cardinality index pathology. Built once in the
        background thread, so it never blocks training startup.
        """
        try:
            con = sqlite3.connect(self.db_path, timeout=30.0)
            try:
                con.execute("PRAGMA busy_timeout=30000")
                con.execute(
                    "CREATE INDEX IF NOT EXISTS ix_dataset_captions_updated_at "
                    "ON dataset_captions(updated_at)"
                )
                con.commit()
            finally:
                con.close()
            print("[TagRefresh] ensured index ix_dataset_captions_updated_at")
        except Exception as e:
            print(f"[TagRefresh] index ensure skipped ({e}); "
                  f"updated_at scans will be full table scans")

    def _run(self) -> None:
        # One-time: make the per-poll updated_at filter cheap on large datasets.
        self._ensure_index()
        while not self._stop.wait(self.interval):
            try:
                self._poll_once()
            except Exception as e:
                print(f"[TagRefresh] poll error: {e}")

    def _poll_once(self) -> None:
        # Cheap gate: if datasets.db has not been written since the last cycle,
        # nothing could have changed — skip the SQL scan entirely.
        mtime = self._safe_mtime()
        if mtime <= self._db_mtime:
            return
        cycle_start = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")

        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=10.0)
        try:
            con.execute("PRAGMA busy_timeout=10000")
            cur = con.cursor()
            ph = ",".join("?" for _ in self.dataset_ids)
            # 1) which items had a tags-caption edited since last_seen?
            cur.execute(
                f"""SELECT DISTINCT c.item_id
                      FROM dataset_captions c
                      JOIN dataset_items i ON c.item_id = i.id
                     WHERE i.dataset_id IN ({ph})
                       AND c.is_tags_format = 1
                       AND c.updated_at > ?""",
                (*self.dataset_ids, self._last_seen),
            )
            changed_ids = [r[0] for r in cur.fetchall()]
            if not changed_ids:
                self._db_mtime = mtime
                self._last_seen = cycle_start
                return

            id_to_idx = self._idx_for_item_ids(changed_ids)
            if not id_to_idx:
                # Edited items are not in the training sample set (e.g. items that
                # had no tags at build time) — nothing to override.
                self._db_mtime = mtime
                self._last_seen = cycle_start
                return

            # 2) rebuild the FULL effective tag list for each changed item
            #    (an item may carry several tags-captions; mirror _build_samples).
            target_ids = list(id_to_idx.keys())
            from core.tagger.tagger_dataset import resolve_caption_tags
            n_applied = 0
            CH = 400
            per_item: Dict[int, List[str]] = {i: [] for i in target_ids}
            for s in range(0, len(target_ids), CH):
                chunk = target_ids[s:s + CH]
                iph = ",".join("?" for _ in chunk)
                q = (f"SELECT item_id, tag_data, content FROM dataset_captions "
                     f"WHERE item_id IN ({iph}) AND is_tags_format = 1")
                params: list = list(chunk)
                if self.caption_types:
                    q += " AND caption_type IN (" + ",".join("?" for _ in self.caption_types) + ")"
                    params += list(self.caption_types)
                cur.execute(q, params)
                for item_id, tag_data, content in cur.fetchall():
                    per_item[item_id].extend(
                        resolve_caption_tags(tag_data, content,
                                             self._comma_resolver, self._alias_resolver)
                    )
            for item_id, tags in per_item.items():
                idx = id_to_idx.get(item_id)
                if idx is not None:
                    self._overrides[idx] = tags
                    n_applied += 1
        finally:
            con.close()

        self.store.publish(self._overrides)
        self._db_mtime = mtime
        self._last_seen = cycle_start
        print(f"[TagRefresh] published {n_applied} edited sample(s) "
              f"(total overrides: {len(self._overrides)})")
