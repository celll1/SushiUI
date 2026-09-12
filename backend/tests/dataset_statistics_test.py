from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from core.datasets import statistics
from database.models import Dataset, DatasetBase, DatasetCaption, DatasetItem


def test_tag_statistics_streams_once_and_preserves_category_priority(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'datasets.db'}")
    DatasetBase.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    dataset = Dataset(name="stats", path=str(tmp_path))
    other = Dataset(name="other", path=str(tmp_path / "other"))
    db.add_all([dataset, other])
    db.flush()
    item = DatasetItem(dataset_id=dataset.id, base_name="one", image_path="one.png")
    other_item = DatasetItem(dataset_id=other.id, base_name="two", image_path="two.png")
    db.add_all([item, other_item])
    db.flush()
    db.add_all([
        DatasetCaption(
            item_id=item.id,
            caption_type="tags",
            content="ignored",
            tag_data='[{"tag":"known","category":"Character"},{"tag":"lookup"}]',
        ),
        DatasetCaption(item_id=item.id, caption_type="tags", content="plain, known"),
        DatasetCaption(item_id=item.id, caption_type="tags", content="fallback", tag_data="{}"),
        DatasetCaption(item_id=item.id, caption_type="natural_language", content="not_a_tag"),
        DatasetCaption(item_id=other_item.id, caption_type="tags", content="other_tag"),
    ])
    db.commit()

    monkeypatch.setattr(statistics.taglist_cache, "initialize", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        statistics.taglist_cache,
        "get_categories_batch",
        lambda tags: {tag: {"lookup": "General", "plain": "Artist"}.get(tag, "Unknown") for tag in tags},
    )
    caption_selects = 0

    def count_caption_selects(_conn, _cursor, statement, _parameters, _context, _many):
        nonlocal caption_selects
        if statement.lstrip().upper().startswith("SELECT") and "dataset_captions" in statement:
            caption_selects += 1

    event.listen(engine, "before_cursor_execute", count_caption_selects)
    result = statistics.compute_tag_statistics(dataset.id, db, root_dir=str(tmp_path))

    assert result == {
        "known": {"count": 2, "category": "Character"},
        "lookup": {"count": 1, "category": "General"},
        "plain": {"count": 1, "category": "Artist"},
        "fallback": {"count": 1, "category": "Unknown"},
    }
    assert caption_selects == 1
