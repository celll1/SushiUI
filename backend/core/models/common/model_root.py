"""The optional external model tree this installation reads components from.

Most component lookups take an explicit path. A few offline-friendly fallbacks
additionally probe a conventional layout (``<root>/<arch>/...``) so that a
sibling component can be found without naming it in the request. That root is
one machine's storage layout, so it is configuration:

    SUSHIUI_MODEL_ROOT=/path/holding/minit2i,minimax_h3,sdxl,...

Unset means "no external tree": every such fallback is simply skipped, and the
caller falls back to what it did before -- an explicit path, the repository's
own ``models/`` directory, or the hub.
"""

from __future__ import annotations

import os

MODEL_ROOT_ENV = "SUSHIUI_MODEL_ROOT"


def external_model_root() -> str:
    """The configured external model tree, or an empty string when unset."""
    return os.environ.get(MODEL_ROOT_ENV, "").strip()


def external_model_path(*parts: str) -> str:
    """A path under the external model tree, or "" when no tree is configured."""
    root = external_model_root()
    return os.path.join(root, *parts) if root else ""
