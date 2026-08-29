"""Where the tests that read real model files look for them.

A handful of tests assert against the locally staged model trees (header-only
reads, no tensor bytes) rather than only against synthetic fixtures, because
several defects lived exactly in the gap between the two. Those trees are far
too large to live in the repository and sit at a different place on every
machine, so their location is configuration, not something to hard-code. It is
the same setting the runtime uses (`core.models.common.model_root`):

    SUSHIUI_MODEL_ROOT=/path/holding/minimax_h3,minimax-music3,sdxl,...

If the variable is unset, the first line of the untracked `local/model_root.txt`
is used instead, so a checkout can pin its own tree without exporting anything.
With neither present, `model_path()` returns a path that cannot exist, so the
callers take the same skip they take on a machine that has not staged the tree.
"""

import os

MODEL_ROOT_ENV = "SUSHIUI_MODEL_ROOT"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT_FILE = os.path.join(_REPO_ROOT, "local", "model_root.txt")

# Deliberately not a valid path: it makes the callers' skip messages say why
# they skipped instead of naming an empty or arbitrary directory.
_UNSET = "<%s unset>" % MODEL_ROOT_ENV


def model_root() -> str:
    """The configured model tree root, or a non-existent placeholder."""
    value = os.environ.get(MODEL_ROOT_ENV, "").strip()
    if value:
        return value
    try:
        with open(_ROOT_FILE, "r", encoding="utf-8") as handle:
            configured = handle.read().strip()
    except OSError:
        return _UNSET
    return configured or _UNSET


def model_path(*parts: str) -> str:
    """Join `parts` onto the configured model tree root."""
    return os.path.join(model_root(), *parts)
