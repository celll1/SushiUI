"""`GET /models` must expose BOTH DiT partitions when a `model_dirs` entry
points directly at the H3 tree root.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_model_listing_test.py -v

WHY THIS FILE EXISTS
--------------------
Every other architecture in this repo is added to Settings -> Directories as
its OWN root (`M:\\model\\sdxl`, `M:\\model\\krea2`, `M:\\model\\anima`, ...),
with the individual checkpoints as its direct children. A user who follows
that same convention for MiniMax-H3 and adds `M:\\model\\minimax_h3` itself
hits a shape the scanner's H3 expansion did not cover: it only looked for a
tree ONE LEVEL BELOW the configured directory (`models_dir/<item>/diffusion_
models/`), so `os.listdir(models_dir)` fell through to the per-child loop and
enumerated the tree's own components (`official/`, `text_encoders/`, `vae/`,
...) as if each were a candidate model. `official/model_index.json` declares
`MiniMaxH3ModularPipeline` and passes `is_valid_diffusers_directory`, so it
won it before the two DiT files under `diffusion_models/` were ever reached
-- leaving only "official" (a config-only, non-loadable tree) selectable.
"""

import json
import os
import struct
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import routes  # noqa: E402


def _write_fake_h3_dit(path: str, header: Optional[dict] = None) -> None:
    """A header-only safetensors file carrying the H3 key signature.

    `is_minimax_h3_safetensors` reads the JSON header only (never tensor
    bytes), so an empty/undersized data section is fine for detection. The
    variant (fl2va/ref2va) is read off the FILENAME by the loader, not
    the header, so it is not encoded here.

    `header`, when given, REPLACES the minimal default -- the hybrid preflight
    tests (`minimax_h3_hybrid_preflight_test.py`) need pairs of files that
    differ in key set, shape, dtype, geometry or quantization metadata, and
    they build those headers themselves rather than duplicating this writer.
    """
    if header is None:
        header = {
            "token_refiner.0.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 0]},
            "adaln_t_table": {"dtype": "F32", "shape": [1], "data_offsets": [0, 0]},
            "__metadata__": {"format": "pt"},
        }
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)


def _build_h3_tree(root: str) -> None:
    """The real on-disk shape: `diffusion_models/`, `official/`, `text_encoders/`, `vae/`."""
    dit_dir = os.path.join(root, "diffusion_models")
    os.makedirs(dit_dir, exist_ok=True)
    _write_fake_h3_dit(os.path.join(dit_dir, "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"))
    _write_fake_h3_dit(os.path.join(dit_dir, "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"))

    official_dir = os.path.join(root, "official")
    os.makedirs(official_dir, exist_ok=True)
    with open(os.path.join(official_dir, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump({"_class_name": "MiniMaxH3ModularPipeline"}, f)

    os.makedirs(os.path.join(root, "text_encoders"), exist_ok=True)
    os.makedirs(os.path.join(root, "vae"), exist_ok=True)


class _FakeSettingsRecord:
    def __init__(self, model_dirs):
        self.model_dirs = model_dirs


class _FakeQuery:
    def __init__(self, record):
        self._record = record

    def first(self):
        return self._record


class _FakeDB:
    """Enough of a `Session` for `get_models`: only `.query(...).first()` is used."""

    def __init__(self, model_dirs):
        self._record = _FakeSettingsRecord(model_dirs)

    def query(self, *_args, **_kwargs):
        return _FakeQuery(self._record)


def _scan(model_dirs, monkeypatch, tmp_path):
    # Isolate from the real configured models_dir and from any warm cache.
    monkeypatch.setattr(routes.settings, "models_dir", str(tmp_path / "empty_default"))
    monkeypatch.setattr(routes, "_models_cache", None)
    monkeypatch.setattr(routes, "_models_cache_timestamp", None)
    # `get_models` is a plain `def` -- 0a57444a made it sync on purpose so its
    # in-process caller could call it directly -- so this must NOT be wrapped in
    # `asyncio.run`, which raises "a coroutine was expected".
    result = routes.get_models(db=_FakeDB(model_dirs), force_rescan=True)
    return result["models"]


def test_pointing_model_dirs_at_the_h3_root_lists_both_partitions(tmp_path, monkeypatch):
    """The owner's actual configuration: `model_dirs` = the H3 tree itself."""
    h3_root = str(tmp_path / "minimax_h3")
    _build_h3_tree(h3_root)

    models = _scan([h3_root], monkeypatch, tmp_path)
    h3_models = [m for m in models if m.get("architecture") == "minimax_h3"]

    assert len(h3_models) == 2, f"expected 2 selectable H3 entries, got {h3_models}"
    variants = {m["variant"] for m in h3_models}
    assert variants == {"fl2va", "ref2va"}
    # "official" must not appear as its own selectable (non-loadable) entry.
    assert not any(m["name"].endswith("/official") or m["name"] == "official" for m in models)


def test_pointing_model_dirs_at_the_parent_still_lists_both_partitions(tmp_path, monkeypatch):
    """The other supported shape: models_dir is the PARENT, the tree is a child item."""
    parent = str(tmp_path / "parent")
    h3_root = os.path.join(parent, "minimax_h3")
    _build_h3_tree(h3_root)

    models = _scan([parent], monkeypatch, tmp_path)
    h3_models = [m for m in models if m.get("architecture") == "minimax_h3"]

    assert len(h3_models) == 2
    assert {m["variant"] for m in h3_models} == {"fl2va", "ref2va"}


def test_entries_are_distinguishable_by_name(tmp_path, monkeypatch):
    """The frontend's model list has to show two DIFFERENT, pickable names."""
    h3_root = str(tmp_path / "minimax_h3")
    _build_h3_tree(h3_root)

    models = _scan([h3_root], monkeypatch, tmp_path)
    h3_names = {m["name"] for m in models if m.get("architecture") == "minimax_h3"}
    assert len(h3_names) == 2
    assert any("fl2va" in n for n in h3_names)
    assert any("ref2va" in n for n in h3_names)


# --------------------------------------------------------------------------
# Negative control: removing the pre-pass reproduces the original defect
# --------------------------------------------------------------------------

def test_removing_the_top_level_expansion_reproduces_the_defect(tmp_path, monkeypatch):
    """Mutant: `_expand_minimax_h3_tree` always reports nothing.

    This is exactly what the pre-fix code did when `models_dir` itself was
    the H3 tree root (there was no pre-pass calling it before the per-child
    loop reached `official` first). With the mutant applied, the scan must
    fall back to the ORIGINAL symptom: only "official" is selectable, and
    neither DiT partition appears.
    """
    h3_root = str(tmp_path / "minimax_h3")
    _build_h3_tree(h3_root)

    monkeypatch.setattr(routes, "_expand_minimax_h3_tree", lambda *a, **k: [])

    models = _scan([h3_root], monkeypatch, tmp_path)
    h3_models = [m for m in models if m.get("architecture") == "minimax_h3"]

    # The mutant reproduces the reported bug: "official" is the only
    # selectable entry, and it is not a loadable checkpoint (type=diffusers,
    # no `variant`).
    assert len(h3_models) == 1
    assert h3_models[0]["name"] == "official"
    assert h3_models[0]["type"] == "diffusers"
    assert "variant" not in h3_models[0]
