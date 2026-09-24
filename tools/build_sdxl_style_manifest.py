"""Build the local dataset-39 SDXL style-pair manifest from the pinned style split."""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

WORKTREE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKTREE / "backend"))

from api.param_defaults import STYLE_TRAINING_DEFAULTS
from core.style_encoder.style_dataset import SplitParams, build_style_index
from core.style_encoder.style_ingest import StyleDatasetConfig, StyleIngest, build_label_resolver
from database.models import DatasetCaption


def main() -> None:
    defaults = STYLE_TRAINING_DEFAULTS
    engine = create_engine("sqlite://", creator=lambda: sqlite3.connect(
        f"file:{(REPO / 'datasets.db').as_posix()}?mode=ro", uri=True,
    ))
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        resolver = build_label_resolver(
            str(REPO), defaults["style_use_tag_aliases"], defaults["style_use_gelbooru_categories"],
            defaults["style_signature_tags"], defaults["style_non_copyright_tags"],
        )
        ingest = StyleIngest(session, (StyleDatasetConfig(39, ()),), resolver)
        split = SplitParams.create(0.05, 0.02, 4, 0)
        index = build_style_index(ingest.records(), split, np.zeros(0, dtype=np.int64))
        expected = "9df70dad080a8b3dfa3eb5b000371830"
        if index.index_hash != expected:
            raise ValueError(f"dataset 39 split changed: {index.index_hash} != {expected}")
        paths = ingest.paths.take(index.item_ids)
        captions = {}
        for start in range(0, len(index.item_ids), 500):
            ids = index.item_ids[start:start + 500].tolist()
            rows = session.query(DatasetCaption).filter(
                DatasetCaption.item_id.in_(ids), DatasetCaption.is_tags_format == True,
            ).order_by(DatasetCaption.item_id, DatasetCaption.id)
            for caption in rows:
                if caption.item_id in captions:
                    continue
                tags = json.loads(caption.tag_data) if caption.tag_data else []
                names = [entry["tag"] for entry in tags if entry.get("category") != "Artist"]
                if not names:
                    names = [part.strip() for part in caption.content.split(",") if part.strip()]
                captions[caption.item_id] = ", ".join(names[:70])

        output = WORKTREE / "outputs" / "sdxl_style_ip_dataset39" / "manifest.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        counts = {0: 0, 1: 0, 2: 0}
        with output.open("w", encoding="utf-8") as handle:
            for row, item_id in enumerate(index.item_ids):
                caption = captions.get(int(item_id))
                if not caption:
                    continue
                split_id = int(index.item_split[row])
                counts[split_id] += 1
                entry = {
                    "row": row, "item_key": int(index.item_keys[row]), "path": paths.path(row),
                    "caption": caption, "artist": int(index.item_artist[row]), "split": split_id,
                    "eligible": bool(index.artist_eligible[index.item_artist[row]]),
                    "copyrights": index.item_copyrights[
                        index.item_copyright_indptr[row]:index.item_copyright_indptr[row + 1]
                    ].tolist(),
                    "characters": index.item_characters[
                        index.item_character_indptr[row]:index.item_character_indptr[row + 1]
                    ].tolist(),
                }
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(json.dumps({"manifest": str(output), "index_hash": index.index_hash,
                          "train": counts[0], "eligible_train": index.stats.train_items,
                          "val_image": counts[1], "val_artist": counts[2]}))
    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    main()
