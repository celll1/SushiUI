"""The dataset-side latent cache API: status, listing, and a bounded delete.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/dataset_latent_cache_api_test.py -v

Deleting a cache is irreversible and the path is assembled from a database
value plus two query parameters, so the questions here are: can a request name
a directory outside this dataset's cache root, does a delete take the
VAE-independent text embeddings with it, and does merely listing ever remove
anything.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Importing the route module must not take the GPU the owner's run holds.
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

_CUDA_LIVE_BEFORE_IMPORT = torch.cuda.is_initialized()

from fastapi import HTTPException  # noqa: E402

import api.routes as routes  # noqa: E402
from core.training import latent_cache as latent_cache_module  # noqa: E402
from core.training.train_runner import _warn_removed_cache_keys  # noqa: E402
from core.training.training_config import TrainingConfigGenerator  # noqa: E402

# Only these imports are ours to answer for. An earlier test file in the same
# session may already hold a context, and asserting the absolute state made a
# combined run die during collection instead of reporting a failure.
if not _CUDA_LIVE_BEFORE_IMPORT:
    assert not torch.cuda.is_initialized(), "importing api.routes initialised CUDA"


def _apply_criterion(rows, criterion):
    """Evaluate ``status.in_(...)`` and ``id == n`` against fake rows.

    A fake that ignored the criteria would pass a run of any status to the
    guard, which is how a `starting` run went unnoticed.
    """
    key = getattr(getattr(criterion, "left", None), "key", None)
    operator = getattr(getattr(criterion, "operator", None), "__name__", "")
    value = getattr(getattr(criterion, "right", None), "value", None)
    if key is None or operator not in ("in_op", "eq"):
        return rows
    if operator == "in_op":
        return [row for row in rows if getattr(row, key, None) in (value or ())]
    return [row for row in rows if getattr(row, key, None) == value]


class _FakeQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *criteria, **kwargs):
        if not isinstance(self._result, list):
            return self
        rows = self._result
        for criterion in criteria:
            rows = _apply_criterion(rows, criterion)
        return _FakeQuery(rows)

    def first(self):
        if isinstance(self._result, list):
            return self._result[0] if self._result else None
        return self._result

    def all(self):
        return self._result if isinstance(self._result, list) else [self._result]


class _FakeDB:
    def __init__(self, result):
        self._result = result

    def query(self, *args, **kwargs):
        return _FakeQuery(self._result)


def _dataset(unique_id="ds-uuid", dataset_id=7, total_items=3):
    return SimpleNamespace(id=dataset_id, unique_id=unique_id, total_items=total_items)


def _run(run_id=1, run_name="run112", status="running", dataset_id=None,
         dataset_configs=None):
    return SimpleNamespace(id=run_id, run_name=run_name, status=status,
                           dataset_id=dataset_id, dataset_configs=dataset_configs)


def _junction(link: Path, target: Path) -> None:
    """A Windows junction, or skip the test. Non-admins can make one, and an
    operator moving the cache to another drive does."""
    if sys.platform != "win32":
        pytest.skip("junctions are Windows-only")
    done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                          capture_output=True, text=True)
    if done.returncode != 0 or not link.exists():
        pytest.skip(f"could not create a junction: {done.stdout}{done.stderr}")


def _listing_entry(path, namespace="ns", vae_namespace="vae-aaa", entries=1, size=1):
    return {"path": path, "namespace": namespace, "vae_namespace": vae_namespace,
            "entries": entries, "bytes": size, "vae_latent_hash": None,
            "vae_family": None, "model_path": None, "created_at": None}


def _make_cache(base: Path, unique_id: str, namespace: str, vae_namespace: str,
                entries: int, size: int = 100) -> Path:
    latents = base / unique_id / namespace / vae_namespace / "latents"
    latents.mkdir(parents=True, exist_ok=True)
    for i in range(entries):
        (latents / f"{i}.pt").write_bytes(b"x" * size)
    (latents.parent / "cache_info.json").write_text(
        '{"vae_latent_hash": "abcd", "vae_family": "sdxl", "model_path": "m.safetensors"}')
    return latents.parent


def _text_embeddings(base: Path, unique_id: str, namespace: str) -> Path:
    d = base / unique_id / namespace / "text_embeddings"
    d.mkdir(parents=True, exist_ok=True)
    (d / "caption.pt").write_bytes(b"keep me")
    return d


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """A cache tree under tmp_path only — never the real cache directory."""
    base = tmp_path / "cache" / "datasets"
    base.mkdir(parents=True)
    monkeypatch.setattr(latent_cache_module, "get_cache_base_dir", lambda: str(base))
    return base


def _get(dataset, db=None):
    return asyncio.run(routes.get_dataset_latent_cache(
        dataset.id, db=_FakeDB(dataset) if db is None else db))


def _delete(dataset, runs=(), **kwargs):
    return asyncio.run(routes.delete_dataset_latent_cache(
        dataset.id, db=_FakeDB(dataset), training_db=_FakeDB(list(runs)), **kwargs))


# ---------------------------------------------------------------- status (e)

def test_status_reports_entries_and_bytes_per_namespace(cache_root):
    dataset = _dataset()
    _make_cache(cache_root, "ds-uuid", "sdxl__c4__dtfloat16", "vae-aaa", entries=2, size=100)
    _make_cache(cache_root, "ds-uuid", "sdxl__c4__dtfloat16", "vae-bbb", entries=1, size=50)
    _text_embeddings(cache_root, "ds-uuid", "sdxl__c4__dtfloat16")

    status = _get(dataset)

    assert status["item_count"] == 3
    assert status["total_entries"] == 3
    assert status["total_bytes"] == 250
    assert [n["vae_namespace"] for n in status["namespaces"]] == ["vae-aaa", "vae-bbb"]
    assert status["namespaces"][0]["entries"] == 2
    assert status["namespaces"][0]["bytes"] == 200
    assert status["namespaces"][0]["vae_family"] == "sdxl"
    assert isinstance(status["namespaces"][0]["path"], str)


def test_status_of_a_dataset_with_no_cache_is_empty_not_an_error(cache_root):
    status = _get(_dataset())
    assert status["namespaces"] == []
    assert status["total_entries"] == 0


# --------------------------------------------------------------- listing (c)

def test_listing_deletes_nothing(cache_root):
    dataset = _dataset()
    kept = _make_cache(cache_root, "ds-uuid", "sdxl__c4__dtfloat16", "vae-orphan", entries=2)

    _get(dataset)
    _get(dataset)

    assert len(list((kept / "latents").glob("*.pt"))) == 2
    assert (kept / "cache_info.json").exists()


def test_a_dry_run_reports_the_targets_without_deleting(cache_root):
    dataset = _dataset()
    target = _make_cache(cache_root, "ds-uuid", "ns", "vae-aaa", entries=2, size=10)

    result = _delete(dataset, dry_run=True)

    assert result["dry_run"] is True and result["deleted"] is False
    assert result["total_entries"] == 2 and result["total_bytes"] == 20
    assert [t["path"] for t in result["targets"]] == [str(target)]
    assert target.exists()


# ------------------------------------------------------- text embeddings (b)

def test_deleting_every_namespace_keeps_the_text_embeddings(cache_root):
    dataset = _dataset()
    vae_dir = _make_cache(cache_root, "ds-uuid", "ns", "vae-aaa", entries=2)
    embeddings = _text_embeddings(cache_root, "ds-uuid", "ns")

    result = _delete(dataset)

    assert result["deleted"] is True
    assert not vae_dir.exists()
    assert embeddings.exists() and (embeddings / "caption.pt").read_bytes() == b"keep me"


def test_one_namespace_delete_leaves_the_other_vae_alone(cache_root):
    dataset = _dataset()
    doomed = _make_cache(cache_root, "ds-uuid", "ns", "vae-aaa", entries=1)
    kept = _make_cache(cache_root, "ds-uuid", "ns", "vae-bbb", entries=1)

    _delete(dataset, namespace="ns", vae_namespace="vae-aaa")

    assert not doomed.exists()
    assert kept.exists()


# --------------------------------------------------------- path bounding (a)

@pytest.mark.parametrize(
    ("namespace", "vae_namespace"),
    [
        ("..", "vae-aaa"),
        ("ns", ".."),
        ("../..", "../.."),
        ("", ""),
        ("ns", ""),
        ("/etc", "vae-aaa"),
        ("C:\\Windows", "vae-aaa"),
        ("ns", "../../../vae-aaa"),
    ],
)
def test_a_target_that_is_not_in_the_listing_is_refused(cache_root, namespace, vae_namespace):
    """The query string selects from the listing; it never builds a path."""
    dataset = _dataset()
    outside = cache_root.parent / "outside"
    outside.mkdir()
    (outside / "keep.pt").write_bytes(b"x")
    kept = _make_cache(cache_root, "ds-uuid", "ns", "vae-aaa", entries=1)

    with pytest.raises(HTTPException) as excinfo:
        _delete(dataset, namespace=namespace, vae_namespace=vae_namespace)

    assert excinfo.value.status_code == 400
    assert kept.exists()
    assert (outside / "keep.pt").exists()


def test_one_half_of_the_pair_is_refused(cache_root):
    with pytest.raises(HTTPException) as excinfo:
        _delete(_dataset(), namespace="ns")
    assert excinfo.value.status_code == 400
    assert "together" in excinfo.value.detail


@pytest.mark.parametrize("unique_id", ["", "..", "../evil", "a/b", "a\\b", "C:evil"])
def test_an_unusable_dataset_unique_id_never_reaches_the_filesystem(cache_root, unique_id):
    outside = cache_root.parent / "outside"
    outside.mkdir()
    (outside / "keep.pt").write_bytes(b"x")

    with pytest.raises(HTTPException) as excinfo:
        _delete(_dataset(unique_id=unique_id))

    assert excinfo.value.status_code == 500
    assert (outside / "keep.pt").exists()


def test_a_listed_path_outside_the_dataset_root_is_still_refused(cache_root, monkeypatch):
    """Second guard: even if the enumerator returned a foreign path, the
    resolved target must sit under {base}/{dataset_unique_id}."""
    dataset = _dataset()
    outside = cache_root.parent / "outside"
    (outside / "latents").mkdir(parents=True)
    (outside / "latents" / "0.pt").write_bytes(b"x")

    monkeypatch.setattr(latent_cache_module, "list_vae_namespaces",
                        lambda *a, **k: [_listing_entry(outside)])

    with pytest.raises(HTTPException) as excinfo:
        _delete(dataset)

    assert excinfo.value.status_code == 500
    assert (outside / "latents" / "0.pt").exists()


def test_the_dataset_root_itself_is_not_a_deletable_target(cache_root, monkeypatch):
    dataset = _dataset()
    root = cache_root / "ds-uuid"
    root.mkdir(parents=True)
    monkeypatch.setattr(latent_cache_module, "list_vae_namespaces",
                        lambda *a, **k: [_listing_entry(root, entries=0, size=0)])

    with pytest.raises(HTTPException) as excinfo:
        _delete(dataset)

    assert excinfo.value.status_code == 500
    assert root.exists()


# ------------------------------------------------------ the active-run guard

@pytest.mark.parametrize("status", ["running", "paused", "starting"])
@pytest.mark.parametrize("attach", ["dataset_configs", "string_dataset_id", "legacy"])
def test_an_unfinished_run_on_this_dataset_blocks_the_delete(cache_root, status, attach):
    """`starting` is the one that matters: the status becomes `running` only
    once a training step is parsed, so the whole pre-encode phase — hours on a
    large dataset — is `starting`, and deleting under it makes the next
    save_latent fail with FileNotFoundError and takes the run down."""
    dataset = _dataset()
    kept = _make_cache(cache_root, "ds-uuid", "ns", "vae-aaa", entries=1)
    attached = {
        "dataset_configs": dict(dataset_configs=[{"dataset_id": dataset.id}]),
        "string_dataset_id": dict(dataset_configs=[{"dataset_id": str(dataset.id)}]),
        "legacy": dict(dataset_id=dataset.id),
    }[attach]
    run = _run(status=status, **attached)

    with pytest.raises(HTTPException) as excinfo:
        _delete(dataset, runs=[run])
    assert excinfo.value.status_code == 409

    preview = _delete(dataset, runs=[run], dry_run=True)
    assert preview["active_runs"] == ["run112"]
    assert kept.exists()


def test_a_finished_run_does_not_block_the_delete(cache_root):
    dataset = _dataset()
    doomed = _make_cache(cache_root, "ds-uuid", "ns", "vae-aaa", entries=1)
    run = _run(status="completed", dataset_configs=[{"dataset_id": dataset.id}])

    assert _delete(dataset, runs=[run])["deleted"] is True
    assert not doomed.exists()


def test_a_run_on_another_dataset_does_not_block_the_delete(cache_root):
    dataset = _dataset()
    doomed = _make_cache(cache_root, "ds-uuid", "ns", "vae-aaa", entries=1)
    run = _run(status="starting", dataset_configs=[{"dataset_id": dataset.id + 1}])

    assert _delete(dataset, runs=[run])["deleted"] is True
    assert not doomed.exists()


def test_a_live_training_process_blocks_a_run_whose_row_says_otherwise(cache_root, monkeypatch):
    """The DB status is not liveness: a child is registered before its status
    is written, which is the window between this check and the rmtree."""
    dataset = _dataset()
    kept = _make_cache(cache_root, "ds-uuid", "ns", "vae-aaa", entries=1)
    run = _run(run_id=42, status="failed", dataset_configs=[{"dataset_id": dataset.id}])
    monkeypatch.setitem(routes.training_process_manager.processes, 42,
                        SimpleNamespace(process=None))

    with pytest.raises(HTTPException) as excinfo:
        _delete(dataset, runs=[run])

    assert excinfo.value.status_code == 409
    assert kept.exists()


# ------------------------------------------------- batch is all-or-nothing (P3)

def test_a_batch_with_one_refused_target_deletes_nothing(cache_root, monkeypatch):
    dataset = _dataset()
    good = _make_cache(cache_root, "ds-uuid", "ns", "vae-aaa", entries=2)
    outside = cache_root.parent / "outside"
    (outside / "latents").mkdir(parents=True)
    (outside / "latents" / "0.pt").write_bytes(b"x")
    monkeypatch.setattr(latent_cache_module, "list_vae_namespaces", lambda *a, **k: [
        _listing_entry(good, vae_namespace="vae-aaa", entries=2),
        _listing_entry(outside, vae_namespace="vae-bad"),
    ])

    with pytest.raises(HTTPException) as excinfo:
        _delete(dataset)

    assert excinfo.value.status_code == 500
    assert "nothing was removed" in excinfo.value.detail
    assert len(list((good / "latents").glob("*.pt"))) == 2
    assert (outside / "latents" / "0.pt").exists()


def test_a_failed_rmtree_reports_how_far_the_batch_got(cache_root, monkeypatch):
    dataset = _dataset()
    first = _make_cache(cache_root, "ds-uuid", "ns", "vae-aaa", entries=1)
    second = _make_cache(cache_root, "ds-uuid", "ns", "vae-bbb", entries=1)
    real_rmtree = shutil.rmtree

    def flaky(path, *args, **kwargs):
        if Path(path).name == "vae-bbb":
            raise OSError("device or resource busy")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", flaky)

    with pytest.raises(HTTPException) as excinfo:
        _delete(dataset)

    detail = excinfo.value.detail
    assert excinfo.value.status_code == 500
    assert "Deleted 1 of 2" in detail
    assert str(first) in detail and str(second) in detail
    assert not first.exists() and second.exists()


def test_deleting_a_dataset_with_no_cache_is_a_reported_no_op(cache_root):
    result = _delete(_dataset())

    assert result["deleted"] is True
    assert result["targets"] == []
    assert result["total_entries"] == 0 and result["total_bytes"] == 0
    assert result["active_runs"] == []


# ------------------------------------------------------------- junctions (P2)

def test_a_vae_junction_onto_its_own_architecture_namespace_is_refused(cache_root):
    """Measured before the fix: this deleted the architecture namespace, text
    embeddings included, because containment alone accepted the resolved path."""
    dataset = _dataset()
    kept = _make_cache(cache_root, "ds-uuid", "ns", "vae-aaa", entries=1)
    embeddings = _text_embeddings(cache_root, "ds-uuid", "ns")
    arch_dir = cache_root / "ds-uuid" / "ns"
    _junction(arch_dir / "vae-evil", arch_dir)

    with pytest.raises(HTTPException) as excinfo:
        _delete(dataset)

    assert excinfo.value.status_code == 500
    assert (embeddings / "caption.pt").read_bytes() == b"keep me"
    assert (kept / "latents").exists()


def test_a_vae_junction_pointing_outside_the_cache_is_refused(cache_root):
    dataset = _dataset()
    outside = cache_root.parent / "outside"
    (outside / "latents").mkdir(parents=True)
    (outside / "latents" / "0.pt").write_bytes(b"x")
    arch_dir = cache_root / "ds-uuid" / "ns"
    arch_dir.mkdir(parents=True)
    _junction(arch_dir / "vae-out", outside)

    with pytest.raises(HTTPException) as excinfo:
        _delete(dataset)

    assert excinfo.value.status_code == 500
    assert (outside / "latents" / "0.pt").exists()


def test_a_junction_inside_a_real_namespace_is_removed_without_its_target(cache_root):
    dataset = _dataset()
    vae_dir = _make_cache(cache_root, "ds-uuid", "ns", "vae-aaa", entries=1)
    outside = cache_root.parent / "outside"
    outside.mkdir()
    (outside / "keep.pt").write_bytes(b"x")
    _junction(vae_dir / "elsewhere", outside)

    result = _delete(dataset)

    assert result["deleted"] is True and not vae_dir.exists()
    assert (outside / "keep.pt").read_bytes() == b"x"


# ----------------------------------------------------- removed run keys (d)

def test_an_old_config_carrying_the_removed_keys_warns_once(capsys):
    _warn_removed_cache_keys({"datasets": [
        {"folder_path": "a", "cache_latents_to_disk": True, "force_recache": True},
        {"folder_path": "b", "force_recache": True},
    ]})

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "cache_latents_to_disk" in lines[0] and "force_recache" in lines[0]
    assert "latent_encoding_mode" in lines[0]


def test_a_config_without_the_removed_keys_is_silent(capsys):
    _warn_removed_cache_keys({"datasets": [{"folder_path": "a"}]})
    _warn_removed_cache_keys({"datasets": [{"cache_latents_to_disk": False,
                                            "force_recache": False}]})
    _warn_removed_cache_keys({})
    assert capsys.readouterr().out == ""


def test_the_request_schema_no_longer_carries_the_removed_keys():
    fields = routes.TrainingRunCreateRequest.model_fields
    assert "force_recache" not in fields
    assert "cache_latents_to_disk" not in fields
    assert "latent_encoding_mode" in fields


def test_a_generator_given_the_removed_keys_writes_neither(tmp_path):
    import yaml

    for generator in (TrainingConfigGenerator.generate_lora_config,
                      TrainingConfigGenerator.generate_full_finetune_config,
                      TrainingConfigGenerator.generate_controlnet_config,
                      TrainingConfigGenerator.generate_vae_config):
        text = generator(
            {"total_steps": 1, "cache_latents_to_disk": True, "force_recache": True},
            run_name="cache-removal",
            base_model_path="model.safetensors",
            output_dir=str(tmp_path),
            dataset_path="dataset",
        )
        for dataset in yaml.safe_load(text)["config"]["process"][0]["datasets"]:
            assert "cache_latents_to_disk" not in dataset
            assert "force_recache" not in dataset
