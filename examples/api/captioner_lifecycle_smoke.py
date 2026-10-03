"""Opt-in localhost caption lifecycle smoke with real one-layer SigLIP2 weights.

Creates a named fixture dataset/run; keeps their files and receipts for inspection.
Never starts servers, downloads weights, or changes existing dataset captions.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import urlparse
from uuid import uuid4

import requests


def call(base, method, path, **kwargs):
    response = requests.request(method, base + path, timeout=120, **kwargs)
    if not response.ok:
        raise RuntimeError(f"{method} {path}: {response.status_code} {response.text[:1200]}")
    return response.json()


def wait_for(base, path, timeout):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = call(base, "GET", path)
        if last["status"] in {"completed", "failed", "stopped", "interrupted"}:
            if last["status"] != "completed":
                raise RuntimeError(json.dumps(last, ensure_ascii=False))
            return last
        time.sleep(2)
    raise TimeoutError(f"Live job {path} remains active: {last}")


def fixture(root):
    import torch
    from PIL import Image
    from tokenizers import Tokenizer, models
    from transformers import (PreTrainedTokenizerFast, Siglip2Config, Siglip2ImageProcessor,
                              Siglip2Model, Siglip2Processor)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
    from core.tagger.siglip2_tagger_model import SigLIP2TaggerModel
    from core.captioner.storage import write_json

    torch.manual_seed(7)
    torch.set_num_threads(1)
    encoder_root = root / "encoder"
    full = Siglip2Model(Siglip2Config(
        vision_config={"hidden_size": 1152, "intermediate_size": 48, "num_hidden_layers": 1,
                       "num_attention_heads": 3, "patch_size": 16, "num_patches": 4},
        text_config={"hidden_size": 24, "intermediate_size": 48, "num_hidden_layers": 1,
                     "num_attention_heads": 3, "vocab_size": 4, "max_position_embeddings": 64,
                     "bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 1}))
    full.save_pretrained(encoder_root)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(
        {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}, unk_token="[UNK]")),
        unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]")
    Siglip2Processor(Siglip2ImageProcessor(), tokenizer).save_pretrained(encoder_root)
    tagger = SigLIP2TaggerModel(2, full.vision_model)
    checkpoint = tagger.save_checkpoint(str(root), "fixture", metadata={
        "num_tags": 2, "vision_encoder_repo": str(encoder_root), "is_naflex": True})
    vocabulary = root / "fixture_vocabulary.json"
    write_json(vocabulary, {"idx_to_tag": {"0": "red", "1": "blue"},
                            "tag_to_category": {"red": "General", "blue": "General"}})
    images = root / "images"
    images.mkdir()
    generator = torch.Generator().manual_seed(17)
    for index in range(8):
        pixels = torch.randint(0, 256, (32, 32, 3), generator=generator, dtype=torch.uint8)
        Image.fromarray(pixels.numpy()).save(images / f"sample_{index}.png")
        write_json(images / f"sample_{index}.json", {"tags": "red, blue", "caption": "A colored pattern."})
    return checkpoint, str(vocabulary), images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--no-dry-run", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda:0"])
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    address = urlparse(base)
    if address.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Fixture paths and writes are restricted to a localhost backend")
    if not args.no_dry_run:
        print(json.dumps({"dry_run": True, "base_url": base, "writes": [
            {"method": "POST", "path": "/datasets", "json": {"name": "<unique smoke name>", "path": "<fixture images>"}},
            {"method": "POST", "path": "/datasets/<id>/scan"},
            {"method": "PATCH", "path": "/datasets/<id>/items/<item>/captions", "json": "fixture tags + explicit tag_data"},
            {"method": "POST", "path": "/captioner-training/runs", "json": "API defaults + explicit tiny fixture overrides"},
            {"method": "POST", "path": "/captioner-training/runs/<id>/start", "json": {"resume": False}},
            {"method": "POST", "path": "/captioner-training/runs/<id>/start", "json": {"resume": True, "checkpoint": "<first saved checkpoint>"}},
            {"method": "POST", "path": "/captioner-training/runs/<id>/export"},
            {"method": "POST", "path": "/tagger/siglip2/load", "json": "fixture weights/vocabulary only"},
            {"method": "POST", "path": "/captioner/load", "json": {"path": "<export>"}},
            {"method": "POST", "path": "/captioner/predict-item", "json": "fixture item + source modes"},
            {"method": "POST", "path": "/captioner-training/runs/<id>/sample", "json": "fixture item"},
            {"method": "POST", "path": "/captioner/save", "json": "reviewed fixture caption + revision/hash"},
            {"method": "POST", "path": "/captioner/batches", "json": "fixture items + pinned export"},
            {"method": "POST", "path": "/captioner/batches/<id>/start"},
            {"method": "POST", "path": "/captioner/batches/<id>/items/<item>/save", "json": "reviewed fixture caption + revision/hash"},
            {"method": "POST", "path": "/captioner/unload"},
            {"method": "POST", "path": "/tagger/siglip2/unload"}],
            "note": "IDs/paths are resolved from creation responses; no files or HTTP writes in dry-run."}, indent=2))
        return

    if call(base, "GET", "/tagger/siglip2/status")["loaded"] or call(base, "GET", "/captioner/status")["loaded"]:
        raise RuntimeError("Unload tagger/captioner explicitly before this smoke; existing models are not replaced")
    if any(run["status"] in {"starting", "running", "stopping"}
           for run in call(base, "GET", "/captioner-training/runs")["runs"]):
        raise RuntimeError("A caption training run is active")
    output_root = Path(__file__).resolve().parents[2] / "outputs"
    output_root.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="captioner-live-", dir=output_root))
    print(f"Fixture retained at {root}", flush=True)
    checkpoint, vocab, images = fixture(root)
    dataset = call(base, "POST", "/datasets", json={"name": root.name, "path": str(images), "recursive": False})
    dataset_id = dataset["id"]
    call(base, "POST", f"/datasets/{dataset_id}/scan")
    items = call(base, "GET", f"/datasets/{dataset_id}/items")["items"]
    assert len(items) == 8
    for item in items:
        response = call(base, "PATCH", f"/datasets/{dataset_id}/items/{item['id']}/captions", json={
            "caption_type": "tags", "source_field": "tags", "content": "red, blue", "persist_sidecar": False,
            "tag_data": [{"tag": "red", "category": "General"}, {"tag": "blue", "category": "General"}]})
        assert response["caption"]["is_tags_format"]
    cfg = call(base, "GET", "/schema/captioner-training-defaults")
    cfg.update(hidden_dim=12, heads=3, layers=1, ffn_dim=24, visual_queries=2,
               max_length=40, vocab_size=280, tokenizer_min_frequency=1, batch_size=2, epochs=1,
               val_ratio=0.0, test_ratio=0.0, near_duplicate_distance=0, max_num_patches=4,
               max_concepts=2, save_every_steps=1, use_tag_aliases=False)
    if args.device:
        cfg["device"] = args.device
    rd = call(base, "GET", "/schema/captioner-run-defaults")
    run = call(base, "POST", "/captioner-training/runs", json={"run_name": root.name,
        "output_dir": str(root / "run"), "tagger": {"checkpoint_path": checkpoint, "vocab_path": vocab,
        "vision_encoder_path": rd["vision_encoder_path"], "lora_rank": rd["lora_rank"], "lora_alpha": rd["lora_alpha"]},
        "dataset_configs": [{"dataset_id": dataset_id, "input_tag_caption_types": ["tags"],
                             "target_caption_types": ["caption"], "filters": {}}], "config": cfg})
    key = run["run_id"]
    print(f"dataset={dataset_id} run={key}", flush=True)
    run_path = f"/captioner-training/runs/{key}"
    call(base, "POST", run_path + "/start", json={"resume": False})
    run = wait_for(base, run_path, args.timeout)
    assert run["progress"]["step"] == 4
    snapshot = json.loads((root / "run" / "snapshot.json").read_text(encoding="utf-8"))
    assert all(record["tags"] == ["red", "blue"] for record in snapshot["records"])
    final_weights = (Path(run["checkpoints"][-1]) / "model.safetensors").read_bytes()
    first = Path(run["checkpoints"][0]).name
    call(base, "POST", run_path + "/start", json={"resume": True, "checkpoint": first})
    resumed = wait_for(base, run_path, args.timeout)
    assert resumed["resume_seq"] == 2 and resumed["progress"]["step"] == 4
    assert (Path(resumed["checkpoints"][-1]) / "model.safetensors").read_bytes() == final_weights
    bundle = call(base, "POST", run_path + "/export", json={"checkpoint": ""})["path"]
    assert not (Path(bundle) / "resume.pt").exists()
    loaded_fixture = False
    try:
        call(base, "POST", "/tagger/siglip2/load", json={"checkpoint_path": checkpoint, "vocab_path": vocab})
        loaded_fixture = True
        loaded = call(base, "POST", "/captioner/load", json={"path": bundle})
        cd = call(base, "GET", "/schema/captioner-defaults")
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        options = {"device": device, "max_new_tokens": min(8, loaded["max_new_tokens"]), "tag_source": "predicted", "tags": []}
        item = items[0]
        preview = None
        for source in ("predicted", "provided", "merged"):
            preview = call(base, "POST", "/captioner/predict-item", json={**options, "tag_source": source,
                "tags": [] if source == "predicted" else ["red", "unknown fixture tag"],
                "dataset_id": dataset_id, "item_id": item["id"], "request_id": str(uuid4())})
            assert preview["token_count"] <= options["max_new_tokens"]
            assert preview["termination"] in {"eos", "length_limit"}
        sample = call(base, "POST", run_path + "/sample", json={**options, "checkpoint": "", "dataset_id": dataset_id,
                      "item_id": item["id"], "request_id": str(uuid4())})
        assert sample["run_id"] == key and call(base, "GET", "/captioner/status")["path"] == bundle
        payload = {"dataset_id": dataset_id, "item_id": item["id"], "dataset_revision": preview["dataset_revision"],
                   "image_hash": preview["image_hash"], "caption": "Reviewed fixture pattern.", "caption_type": cd["save_caption_type"]}
        saved = call(base, "POST", "/captioner/save", json=payload)
        assert saved["dataset_revision"] == preview["dataset_revision"] + 1
        stale = requests.post(base + "/captioner/save", json=payload, timeout=10)
        assert stale.status_code == 409
        job = call(base, "POST", "/captioner/batches", json={**options, "bundle_path": bundle,
                   "dataset_id": dataset_id, "item_ids": [row["id"] for row in items[:2]]})
        batch_path = f"/captioner/batches/{job['job_id']}"
        call(base, "POST", batch_path + "/start", json={})
        complete = wait_for(base, batch_path, args.timeout)
        assert complete["counts"]["completed"] == 2
        rows = call(base, "GET", batch_path + "/results")["items"]
        row = rows[1]
        batch_payload = {"dataset_revision": row["preview"]["dataset_revision"], "image_hash": row["preview"]["image_hash"],
                         "caption": "Reviewed batch fixture.", "caption_type": cd["save_caption_type"]}
        receipt = call(base, "POST", batch_path + f"/items/{row['item_id']}/save", json=batch_payload)
        assert call(base, "POST", batch_path + f"/items/{row['item_id']}/save", json=batch_payload) == receipt
        saved_item = next(item for item in items if item["id"] == row["item_id"])
        sidecar = Path(saved_item["image_path"]).with_suffix(".json")
        # Verify the API write rather than interpreting generated random-model text.
        persisted = json.loads(sidecar.read_text(encoding="utf-8"))
        assert persisted["tags"] == "red, blue" and persisted["caption"] == "A colored pattern."
        assert persisted[cd["save_caption_type"]] == batch_payload["caption"]
        call(base, "POST", f"/datasets/{dataset_id}/scan")
        indexed = call(base, "GET", f"/datasets/{dataset_id}/items/{row['item_id']}")["captions"]
        natural = next(caption for caption in indexed if caption["caption_type"] == cd["save_caption_type"])
        assert not natural["is_tags_format"] and natural["content"] == batch_payload["caption"]
        assert next(caption for caption in indexed if caption["caption_type"] == "tags")["content"] == "red, blue"
        summary = {"passed": True, "dataset_id": dataset_id, "run_id": key, "batch_id": job["job_id"],
                   "fixture_root": str(root), "bundle": bundle, "training_device": cfg["device"], "inference_device": device,
                   "resume_weights_sha256": hashlib.sha256(final_weights).hexdigest(),
                   "checks": ["real encoder load", "training", "exact resume", "export", "three sources", "sample isolation",
                              "explicit save", "stale revision refusal", "batch", "idempotent receipt", "sidecar preservation", "rescan"]}
        from core.captioner.storage import write_json
        write_json(root / "result.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        current = call(base, "GET", "/captioner/status")
        if current.get("path") == bundle:
            call(base, "POST", "/captioner/unload")
        if loaded_fixture and call(base, "GET", "/tagger/siglip2/status").get("checkpoint_path") == checkpoint:
            call(base, "POST", "/tagger/siglip2/unload")


if __name__ == "__main__":
    main()
