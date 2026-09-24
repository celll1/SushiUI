"""Style remote-source integration with generated pixels and a fake API client."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style_training_fixture  # noqa: F401 - disables CUDA initialization during imports

from core.style_encoder.style_dataset import TRAIN
from core.style_encoder.style_trainer import StyleTrainingConfig, prepare_style_data, run_style_training
from style_training_fixture import build_corpus, config, environment, image_processor


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.forbid = False

    def _call(self):
        assert not self.forbid, "a frozen source must not use the API on resume"
        self.calls += 1

    def fetch_artist_tags(self, _min_count, _page):
        self._call()
        return [{"name": "artist_remote", "category": 1, "post_count": 100}]

    def fetch_posts(self, _name, *, page, strict):
        self._call()
        return [{"id": i, "tag_string_artist": "artist_remote", "tag_string_general": "1girl"}
                for i in (901, 902, 903)]

    def download_inmemory(self, post):
        self._call()
        image = Image.new("RGB", (8, 8), (post["id"] % 255, 15, 25))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), "png", []


def test_remote_items_train_only_and_resume_from_cache(tmp_path):
    factory, ids, _ = build_corpus(tmp_path, artists=4, per_artist=3)
    cfg = StyleTrainingConfig.from_dict(config(
        ids, style_danbooru_enable=True, style_danbooru_max_artists=1,
        style_danbooru_max_posts_per_artist=3, style_danbooru_max_tag_pages=2,
        style_val_artist_ratio=0.0, style_val_image_ratio=0.5))
    client = FakeClient()
    out = tmp_path / "out"
    first = prepare_style_data(cfg, environment(factory), image_processor(), np.zeros(0, dtype=np.int64),
                               output_dir=str(out), danbooru_client=client)
    remote = first.index.item_ids < 0
    assert remote.sum() == 3
    assert np.all(first.index.item_split[remote] == TRAIN)
    first_calls = client.calls
    client.forbid = True
    second = prepare_style_data(cfg, environment(factory), image_processor(), first.index.sampled_keys(),
                                output_dir=str(out), danbooru_client=client)
    assert client.calls == first_calls
    assert second.index.index_hash == first.index.index_hash


def test_cpu_training_uses_frozen_remote_source(tmp_path):
    factory, ids, _ = build_corpus(tmp_path, artists=4, per_artist=3)
    raw = config(ids, style_danbooru_enable=True, style_danbooru_max_artists=1,
                 style_danbooru_max_posts_per_artist=3, style_danbooru_max_tag_pages=2,
                 style_val_artist_ratio=0.0, style_val_image_ratio=0.0,
                 gradient_checkpointing=False, num_workers=0)
    cfg = StyleTrainingConfig.from_dict(raw)
    out = tmp_path / "out"
    fake = FakeClient()
    prepare_style_data(cfg, environment(factory), image_processor(), np.zeros(0, dtype=np.int64),
                       output_dir=str(out), danbooru_client=fake)
    fake.forbid = True
    result = run_style_training("synthetic", raw, str(out), environment=environment(factory))
    assert result["global_step"] == result["total_steps"] > 0
    assert fake.calls > 0


def test_resume_rejects_changed_remote_labels(tmp_path):
    factory, ids, _ = build_corpus(tmp_path, artists=4, per_artist=3)
    raw = config(ids, style_danbooru_enable=True, style_danbooru_max_artists=1,
                 style_danbooru_max_posts_per_artist=3, style_danbooru_max_tag_pages=2,
                 style_val_artist_ratio=0.0, style_val_image_ratio=0.0,
                 gradient_checkpointing=False, num_workers=0, save_every_n_steps=1)
    cfg = StyleTrainingConfig.from_dict(raw)
    out = tmp_path / "out"
    prepare_style_data(cfg, environment(factory), image_processor(), np.zeros(0, dtype=np.int64),
                       output_dir=str(out), danbooru_client=FakeClient())
    holder = []

    def stop_after_first(_run, event, data):
        if event == "step" and data["step"] == 1:
            holder[0].stop()

    stopped = run_style_training("synthetic", raw, str(out), stop_after_first, holder, environment(factory))
    assert stopped["stopped"] is True
    resumed = run_style_training("synthetic", raw, str(out), environment=environment(factory))
    assert resumed["global_step"] == resumed["total_steps"] > stopped["global_step"]
    manifest_path = out / "style_danbooru_cache" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["items"][0]["tag_data"].append({"tag": "new_general", "category": "General"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="manifest changed"):
        run_style_training("synthetic", raw, str(out), environment=environment(factory))
