"""Dispatch contracts for API and training-triggered dataset scans."""

import asyncio

import torch

torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 9)
torch.cuda._lazy_init = lambda *args, **kwargs: None
torch._C._cuda_init = lambda *args, **kwargs: None


def test_api_scan_disables_internal_cancellation(monkeypatch):
    from api import routes
    from core.datasets import scanning

    seen = {}

    async def fake_scan(dataset_id, db, *, incremental, should_cancel, progress):
        seen.update(
            dataset_id=dataset_id,
            db=db,
            incremental=incremental,
            should_cancel=should_cancel,
            progress=progress,
        )
        return {"ok": True}

    monkeypatch.setattr(scanning, "scan_dataset_index", fake_scan)
    session = object()

    assert asyncio.run(routes.scan_dataset(7, session, incremental=True)) == {"ok": True}
    assert seen == {
        "dataset_id": 7,
        "db": session,
        "incremental": True,
        "should_cancel": None,
        "progress": routes.manager.send_progress_sync,
    }


def test_training_rescan_forwards_cancellation(monkeypatch):
    from core.datasets import scanning
    from core.training.dataset_drift import rescan_dataset_inline

    seen = {}

    async def fake_scan(dataset_id, db, *, incremental, should_cancel, progress):
        seen.update(
            dataset_id=dataset_id,
            db=db,
            incremental=incremental,
            should_cancel=should_cancel,
            progress=progress,
        )
        return {"items_found": 0}

    monkeypatch.setattr(scanning, "scan_dataset_index", fake_scan)
    session = object()
    cancel = lambda: False

    messages = []
    assert asyncio.run(rescan_dataset_inline(
        9,
        session,
        progress_callback=messages.append,
        should_cancel=cancel,
    )) == {"items_found": 0}
    assert seen == {
        "dataset_id": 9,
        "db": session,
        "incremental": True,
        "should_cancel": cancel,
        "progress": seen["progress"],
    }
    seen["progress"](1, 2, "working")
    assert messages == ["Rescanning dataset 9...", "working"]
