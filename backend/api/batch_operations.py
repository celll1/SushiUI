"""
Batch operations for dataset items (tagger inference, tag reordering, tag replacement)
"""
from typing import Callable, List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import asyncio
import json

from core.datasets.captions import update_caption
from core.datasets.revisions import bump_dataset_revision
from utils.taglist_cache import taglist_cache
from config.settings import settings
from api.param_defaults import DATASET_DEFAULTS


class BatchSelection(BaseModel):
    mode: Literal["query"]
    search: Optional[str] = None
    tags: Optional[str] = None
    excluded_ids: List[int] = Field(default_factory=list)

class BatchTaggerRequest(BaseModel):
    item_ids: List[int]
    selection: Optional[BatchSelection] = DATASET_DEFAULTS["batch_selection"]
    operation_id: Optional[str] = Field(default=None, max_length=128)
    gen_threshold: float = 0.45
    char_threshold: float = 0.45
    thresholds: Optional[Dict[str, float]] = None
    model_version: str = "cl_tagger_1_02"
    remove_below_threshold: bool = True
    merge_with_existing: bool = True

class BatchReorderTagsRequest(BaseModel):
    item_ids: List[int]
    selection: Optional[BatchSelection] = DATASET_DEFAULTS["batch_selection"]
    category_order: List[str]
    operation_id: Optional[str] = Field(default=None, max_length=128)

class BatchReplaceTagRequest(BaseModel):
    item_ids: List[int]
    selection: Optional[BatchSelection] = DATASET_DEFAULTS["batch_selection"]
    from_tag: str
    to_tag: str
    normalize_match: bool = True  # Use normalized matching (whitespace, underscores)
    operation_id: Optional[str] = Field(default=None, max_length=128)

class BatchBackfillTagDataRequest(BaseModel):
    dataset_id: int
    batch_size: int = 1000  # Number of captions to process per commit

class BatchOperationResponse(BaseModel):
    status: str
    processed_count: int
    updated_count: int
    skipped_count: int
    failed_count: int
    message: str
    operation_id: Optional[str] = None

def _tag_data(tags: List[str], existing_json: Optional[str] = None) -> List[Dict[str, str]]:
    taglist_cache.initialize(settings.root_dir)
    existing: Dict[str, str] = {}
    if existing_json:
        try:
            existing = {
                item["tag"]: item.get("category", "Unknown")
                for item in json.loads(existing_json)
                if isinstance(item, dict) and item.get("tag")
            }
        except (TypeError, ValueError):
            existing = {}
    missing = [tag for tag in tags if tag not in existing]
    resolved = taglist_cache.get_categories_batch(missing) if missing else {}
    return [
        {"tag": tag, "category": existing.get(tag, resolved.get(tag, "Unknown"))}
        for tag in tags
    ]


def _persist_tags(item, caption, tags: List[str], db, *, source: Optional[str] = None) -> None:
    update_caption(
        db,
        item_id=item.id,
        caption_type="tags",
        content=", ".join(tags),
        tag_data=_tag_data(tags, caption.tag_data if caption else None),
        caption_id=caption.id if caption else None,
        source_field=caption.source_field if caption else None,
        persist_sidecar=True,
        dataset_id=item.dataset_id,
        source=source,
    )


def normalize_tag_for_matching(tag: str) -> str:
    """
    Normalize tag for matching: lowercase, replace underscores with spaces
    """
    return tag.lower().replace('_', ' ').strip()


async def batch_tagger_inference(
    request: BatchTaggerRequest,
    db,
    send_progress_callback,
    *,
    dataset_id: int,
    should_cancel: Callable[[], bool],
) -> BatchOperationResponse:
    """
    Run tagger inference on multiple items
    """
    from database.models import DatasetItem, DatasetCaption
    from core.extensions.tagger_manager import tagger_manager
    from PIL import Image

    total = len(request.item_ids)

    # Validate input
    if total == 0:
        return BatchOperationResponse(
            status="completed",
            processed_count=0,
            updated_count=0,
            skipped_count=0,
            failed_count=0,
            message="No items selected"
        )

    processed = 0
    updated = 0
    skipped = 0
    failed = 0

    send_progress_callback(0, total, "Starting batch tagger inference...")

    # Load tagger model if not loaded
    if not tagger_manager.loaded:
        send_progress_callback(0, total, "Loading tagger model...")
        tagger_manager.load_model(
            use_gpu=True,
            use_huggingface=True,
            model_version=request.model_version
        )

    for idx, item_id in enumerate(request.item_ids):
        if should_cancel():
            send_progress_callback(processed, total, "Batch operation cancelled")
            break

        try:
            item = db.query(DatasetItem).filter(
                DatasetItem.id == item_id,
                DatasetItem.dataset_id == dataset_id,
            ).first()
            if not item:
                skipped += 1
                processed += 1
                continue

            send_progress_callback(
                processed,
                total,
                f"Processing {item.base_name} ({processed + 1}/{total})"
            )

            try:
                image = Image.open(item.image_path)
                # Ensure image is in RGB mode
                if image.mode not in ('RGB', 'RGBA'):
                    image = image.convert('RGB')
            except Exception as img_error:
                print(f"[BatchTagger] Failed to load image {item.image_path}: {img_error}")
                failed += 1
                processed += 1
                continue

            predictions = tagger_manager.predict(
                image,
                gen_threshold=request.gen_threshold,
                char_threshold=request.char_threshold,
                model_version=request.model_version or "cl_tagger_1_02",
                auto_unload=False,  # Don't unload during batch processing
                thresholds=request.thresholds or {}
            )

            tags_caption = db.query(DatasetCaption).filter(
                DatasetCaption.item_id == item.id,
                DatasetCaption.caption_type == "tags"
            ).first()

            existing_tags_set = set()
            if tags_caption and request.merge_with_existing:
                existing_tags_set = set(t.strip() for t in tags_caption.content.split(',') if t.strip())

            # Collect predicted tags with their scores
            predicted_tags = {}  # tag -> score
            for category, category_predictions in predictions.items():
                for tag, score in category_predictions:
                    predicted_tags[tag] = score

            final_tags = []

            if request.merge_with_existing and tags_caption:
                # Merge mode: Keep existing tags + add new predictions
                # Remove existing tags that are now below threshold
                for existing_tag in existing_tags_set:
                    # If tag is in predictions and above threshold, keep it
                    if existing_tag in predicted_tags:
                        if predicted_tags[existing_tag] >= request.gen_threshold:
                            final_tags.append(existing_tag)
                    else:
                        # Tag not in predictions, keep it (user might have added manually)
                        final_tags.append(existing_tag)

                # Add new predicted tags (not in existing)
                for tag, score in predicted_tags.items():
                    if tag not in existing_tags_set:
                        final_tags.append(tag)
            else:
                # Replace mode or no existing tags: Use only predictions
                final_tags = list(predicted_tags.keys())

            _persist_tags(
                item,
                tags_caption,
                final_tags,
                db,
                source="tagger_batch" if tags_caption is None else None,
            )
            updated += 1

        except Exception as e:
            print(f"[BatchTagger] Failed to process item {item_id}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        processed += 1
        send_progress_callback(processed, total, f"Processed {processed}/{total} items")

    # Unload tagger model to free VRAM/memory
    if tagger_manager.loaded:
        print("[BatchTagger] Unloading tagger model to free VRAM")
        tagger_manager.unload_model()

    cancelled = should_cancel()
    status = "cancelled" if cancelled else "completed"
    message = f"Batch tagger: {updated} updated, {skipped} skipped, {failed} failed"
    if cancelled:
        message += " (cancelled)"

    send_progress_callback(total, total, message)

    return BatchOperationResponse(
        status=status,
        processed_count=processed,
        updated_count=updated,
        skipped_count=skipped,
        failed_count=failed,
        message=message
    )



async def batch_reorder_tags(
    request: BatchReorderTagsRequest,
    db,
    send_progress_callback,
    *,
    dataset_id: int,
    should_cancel: Callable[[], bool],
) -> BatchOperationResponse:
    """
    Reorder tags by category for multiple items
    """
    from database.models import DatasetItem, DatasetCaption

    total = len(request.item_ids)

    # Validate input
    if total == 0:
        return BatchOperationResponse(
            status="completed",
            processed_count=0,
            updated_count=0,
            skipped_count=0,
            failed_count=0,
            message="No items selected"
        )

    processed = 0
    updated = 0
    skipped = 0
    failed = 0

    send_progress_callback(0, total, "Starting batch tag reordering...")

    taglist_cache.initialize(settings.root_dir)

    for idx, item_id in enumerate(request.item_ids):
        if should_cancel():
            send_progress_callback(processed, total, "Batch operation cancelled")
            break

        try:
            item = db.query(DatasetItem).filter(
                DatasetItem.id == item_id,
                DatasetItem.dataset_id == dataset_id,
            ).first()
            if not item:
                skipped += 1
                processed += 1
                continue

            tags_caption = db.query(DatasetCaption).filter(
                DatasetCaption.item_id == item.id,
                DatasetCaption.caption_type == "tags"
            ).first()

            if not tags_caption:
                skipped += 1
                processed += 1
                continue

            send_progress_callback(
                processed,
                total,
                f"Reordering {item.base_name} ({processed + 1}/{total})"
            )

            tags = [t.strip() for t in tags_caption.content.split(',') if t.strip()]

            # Categorize tags using cache (O(1) lookup per tag)
            categorized = {cat: [] for cat in request.category_order}
            categorized['Unknown'] = []

            for tag in tags:
                category = taglist_cache.get_category(tag)
                if category in categorized:
                    categorized[category].append(tag)
                else:
                    categorized['Unknown'].append(tag)

            # Rebuild tag list in category order
            reordered_tags = []
            for category in request.category_order:
                reordered_tags.extend(categorized[category])
            reordered_tags.extend(categorized['Unknown'])

            new_content = ', '.join(reordered_tags)
            if new_content != tags_caption.content:
                _persist_tags(item, tags_caption, reordered_tags, db)
                updated += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"[BatchReorder] Failed to process item {item_id}: {e}")
            failed += 1

        processed += 1
        send_progress_callback(processed, total, f"Processed {processed}/{total} items")

    cancelled = should_cancel()
    status = "cancelled" if cancelled else "completed"
    message = f"Batch reorder: {updated} updated, {skipped} skipped, {failed} failed"
    if cancelled:
        message += " (cancelled)"

    send_progress_callback(total, total, message)

    return BatchOperationResponse(
        status=status,
        processed_count=processed,
        updated_count=updated,
        skipped_count=skipped,
        failed_count=failed,
        message=message
    )



async def batch_replace_tag(
    request: BatchReplaceTagRequest,
    db,
    send_progress_callback,
    *,
    dataset_id: int,
    should_cancel: Callable[[], bool],
) -> BatchOperationResponse:
    """
    Replace a specific tag with another tag for multiple items
    """
    from database.models import DatasetItem, DatasetCaption

    total = len(request.item_ids)

    # Validate input
    if total == 0:
        return BatchOperationResponse(
            status="completed",
            processed_count=0,
            updated_count=0,
            skipped_count=0,
            failed_count=0,
            message="No items selected"
        )

    processed = 0
    updated = 0
    skipped = 0
    failed = 0

    from_tag_normalized = normalize_tag_for_matching(request.from_tag) if request.normalize_match else request.from_tag

    send_progress_callback(0, total, f"Starting batch tag replacement: '{request.from_tag}' → '{request.to_tag}'...")

    for idx, item_id in enumerate(request.item_ids):
        if should_cancel():
            send_progress_callback(processed, total, "Batch operation cancelled")
            break

        try:
            item = db.query(DatasetItem).filter(
                DatasetItem.id == item_id,
                DatasetItem.dataset_id == dataset_id,
            ).first()
            if not item:
                skipped += 1
                processed += 1
                continue

            tags_caption = db.query(DatasetCaption).filter(
                DatasetCaption.item_id == item.id,
                DatasetCaption.caption_type == "tags"
            ).first()

            if not tags_caption:
                skipped += 1
                processed += 1
                continue

            tags = [t.strip() for t in tags_caption.content.split(',') if t.strip()]

            # Replace tag
            replaced = False
            new_tags = []
            for tag in tags:
                if request.normalize_match:
                    if normalize_tag_for_matching(tag) == from_tag_normalized:
                        new_tags.append(request.to_tag)
                        replaced = True
                    else:
                        new_tags.append(tag)
                else:
                    if tag == request.from_tag:
                        new_tags.append(request.to_tag)
                        replaced = True
                    else:
                        new_tags.append(tag)

            if replaced:
                send_progress_callback(
                    processed,
                    total,
                    f"Replacing in {item.base_name} ({processed + 1}/{total})"
                )

                _persist_tags(item, tags_caption, new_tags, db)
                updated += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"[BatchReplace] Failed to process item {item_id}: {e}")
            failed += 1

        processed += 1
        if processed % 10 == 0 or processed == total:
            send_progress_callback(processed, total, f"Processed {processed}/{total} items")

    cancelled = should_cancel()
    status = "cancelled" if cancelled else "completed"
    message = f"Batch replace: {updated} updated, {skipped} skipped, {failed} failed"
    if cancelled:
        message += " (cancelled)"

    send_progress_callback(total, total, message)

    return BatchOperationResponse(
        status=status,
        processed_count=processed,
        updated_count=updated,
        skipped_count=skipped,
        failed_count=failed,
        message=message
    )



async def batch_backfill_tag_data(
    request: "BatchBackfillTagDataRequest",
    db,
    send_progress_callback=None,
    *,
    should_cancel: Callable[[], bool] = lambda: False,
) -> BatchOperationResponse:
    """
    Populate tag_data JSON for all is_tags_format=True captions that currently
    have tag_data=NULL (i.e. captions created by bulk import / scan).

    Uses taglist_cache.get_categories_batch() for batch category lookup.
    Processes in batches to avoid excessive memory usage.
    """
    from database.models import Dataset, DatasetCaption, DatasetItem

    def send_progress(current, total, message):
        if send_progress_callback:
            send_progress_callback(current, total, message)

    taglist_cache.initialize(settings.root_dir)

    # Count captions that need backfilling for this dataset
    item_ids_subq = (
        db.query(DatasetItem.id)
        .filter(DatasetItem.dataset_id == request.dataset_id)
    )

    total = (
        db.query(DatasetCaption)
        .filter(
            DatasetCaption.item_id.in_(item_ids_subq),
            DatasetCaption.is_tags_format == True,
            DatasetCaption.tag_data == None,
        )
        .count()
    )

    if total == 0:
        msg = "All tag_data already populated, nothing to backfill."
        send_progress(0, 0, msg)
        return BatchOperationResponse(
            status="completed",
            processed_count=0,
            updated_count=0,
            skipped_count=0,
            failed_count=0,
            message=msg,
        )

    send_progress(0, total, f"Backfilling tag_data for {total} captions...")

    processed = 0
    updated = 0
    failed = 0
    batch_size = request.batch_size
    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()

    while True:
        if should_cancel():
            break

        batch = (
            db.query(DatasetCaption)
            .filter(
                DatasetCaption.item_id.in_(item_ids_subq),
                DatasetCaption.is_tags_format == True,
                DatasetCaption.tag_data == None,
            )
            .limit(batch_size)
            .all()
        )
        if not batch:
            break

        # Collect all unique tag strings in this batch for one bulk lookup
        all_tags: List[str] = []
        caption_tags: Dict[int, List[str]] = {}
        for caption in batch:
            if caption.content:
                tags = [t.strip() for t in caption.content.split(",") if t.strip()]
                caption_tags[caption.id] = tags
                all_tags.extend(tags)
            else:
                caption_tags[caption.id] = []

        # Batch-resolve categories
        categories = taglist_cache.get_categories_batch(list(set(all_tags))) if all_tags else {}

        for caption in batch:
            try:
                tags = caption_tags.get(caption.id, [])
                if not tags:
                    caption.tag_data = "[]"
                else:
                    tag_data = [
                        {"tag": t, "category": categories.get(t, "General")}
                        for t in tags
                    ]
                    caption.tag_data = json.dumps(tag_data, ensure_ascii=False)
                updated += 1
            except Exception as e:
                print(f"[BackfillTagData] Failed caption {caption.id}: {e}")
                failed += 1

            processed += 1

        if dataset is not None:
            bump_dataset_revision(dataset)
        db.commit()
        send_progress(processed, total, f"Backfilled {processed}/{total} captions...")
        await asyncio.sleep(0)

    cancelled = should_cancel()
    status = "cancelled" if cancelled else "completed"
    message = f"Backfill tag_data: {updated} updated, {failed} failed"
    if cancelled:
        message += " (cancelled)"

    send_progress(total, total, message)

    return BatchOperationResponse(
        status=status,
        processed_count=processed,
        updated_count=updated,
        skipped_count=0,
        failed_count=failed,
        message=message,
    )
