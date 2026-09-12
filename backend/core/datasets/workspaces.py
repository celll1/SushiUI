"""Bounded, process-local roots for folder-backed dataset workspaces."""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class _Workspace:
    root: str
    touched_at: float


class DatasetWorkspaceRegistry:
    """Keep opaque workspace IDs separate from filesystem paths."""

    def __init__(self, *, max_entries: int = 64, ttl_seconds: float = 12 * 60 * 60):
        if max_entries < 1 or ttl_seconds <= 0:
            raise ValueError("Workspace limits must be positive")
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, _Workspace] = OrderedDict()
        self._legacy_id: str | None = None
        self._lock = threading.Lock()

    def create(self, root: str) -> tuple[str, str]:
        normalized = os.path.abspath(root)
        if not os.path.isdir(normalized):
            raise ValueError("Invalid directory")
        now = time.monotonic()
        workspace_id = secrets.token_urlsafe(24)
        with self._lock:
            self._prune(now)
            self._entries[workspace_id] = _Workspace(normalized, now)
            self._entries.move_to_end(workspace_id)
            while len(self._entries) > self._max_entries:
                evicted_id, _ = self._entries.popitem(last=False)
                if evicted_id == self._legacy_id:
                    self._legacy_id = None
        return workspace_id, normalized

    def set_legacy(self, workspace_id: str) -> None:
        with self._lock:
            if workspace_id not in self._entries:
                raise KeyError("Unknown workspace")
            self._legacy_id = workspace_id

    def root(self, workspace_id: str | None) -> str:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            resolved_id = workspace_id or self._legacy_id or ""
            entry = self._entries.get(resolved_id)
            if entry is None:
                raise KeyError("Unknown or expired workspace")
            entry.touched_at = now
            self._entries.move_to_end(resolved_id)
            return entry.root

    def resolve(self, workspace_id: str | None, rel_path: str) -> str:
        root = self.root(workspace_id)
        drive, _ = os.path.splitdrive(rel_path)
        if drive or os.path.isabs(rel_path) or rel_path.startswith(("/", "\\")):
            raise PermissionError("Absolute paths are not allowed")
        normalized = os.path.normpath(rel_path)
        if normalized == ".." or normalized.startswith(("../", "..\\")):
            raise PermissionError("Path traversal not allowed")
        candidate = os.path.join(root, normalized)
        real_root = os.path.realpath(root)
        real_candidate = os.path.realpath(candidate)
        try:
            contained = os.path.commonpath((real_root, real_candidate)) == real_root
        except ValueError:
            contained = False
        if not contained:
            raise PermissionError("Path outside allowed directory")
        return candidate

    def _prune(self, now: float) -> None:
        expired = [
            workspace_id
            for workspace_id, entry in self._entries.items()
            if now - entry.touched_at > self._ttl_seconds
        ]
        for workspace_id in expired:
            del self._entries[workspace_id]
            if workspace_id == self._legacy_id:
                self._legacy_id = None


dataset_workspaces = DatasetWorkspaceRegistry()
