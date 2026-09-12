"""
Batch operations for dataset items (tagger inference, tag reordering, tag replacement)
"""
from typing import Callable, List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import asyncio
from datetime import datetime

from core.datasets.sidecars import write_indexed_caption
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



async def save_item_to_txt_json(item, db):
    """Persist the indexed tags caption through the canonical sidecar writer."""
    from database.models import DatasetCaption

    tags_caption = db.query(DatasetCaption).filter(
        DatasetCaption.item_id == item.id,
        DatasetCaption.caption_type == "tags"
    ).first()

    if not tags_caption:
        return None

    return await asyncio.to_thread(
        write_indexed_caption,
        item.image_path,
        tags_caption.content,
        caption_type=tags_caption.caption_type,
        source_field=tags_caption.source_field,
    )


async def update_tag_statistics(dataset_id: int, db):
    """
    Update tag statistics for a dataset with category information
    (MIGRATED TO USE TaglistCache singleton - Phase 3)
    """
    from database.models import Dataset, DatasetCaption
    from sqlalchemy import func

    taglist_cache.initialize(settings.root_dir)

    captions = db.query(DatasetCaption).join(
        DatasetCaption.item
    ).filter(
        DatasetCaption.item.has(dataset_id=dataset_id),
        DatasetCaption.caption_type == "tags"
    ).all()

    # Count tags
    tag_counts = {}
    for caption in captions:
        tags = [t.strip() for t in caption.content.split(',') if t.strip()]
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    all_tag_names = list(tag_counts.keys())
    tag_categories = taglist_cache.get_categories_batch(all_tag_names)

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset:
        tag_statistics = {}
        for tag, count in tag_counts.items():
            category = tag_categories.get(tag, "General")
            tag_statistics[tag] = {
                "count": count,
                "category": category
            }
        dataset.tag_statistics = tag_statistics
        db.commit()
        print(f"[BatchOps] Updated tag statistics: {len(tag_statistics)} unique tags (via TaglistCache)")


def normalize_tag_for_matching(tag: str) -> str:
    """
    Normalize tag for matching: lowercase, replace underscores with spaces
    """
    return tag.lower().replace('_', ' ').strip()


def get_tag_category(tag: str, tag_suggestions_context) -> str:
    """
    Get tag category using taglist
    Returns "General" if not found
    """
    try:
        # This is a placeholder - actual implementation should use tagSuggestions
        # For now, return "General" as default
        return "General"
    except:
        return "General"



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

            content = ', '.join(final_tags)

            if tags_caption:
                tags_caption.content = content
                tags_caption.updated_at = datetime.utcnow()
            else:
                tags_caption = DatasetCaption(
                    item_id=item.id,
                    caption_type="tags",
                    content=content,
                    field_category="training",
                    is_tags_format=True,
                    source="tagger_batch"
                )
                db.add(tags_caption)

            db.commit()

            await save_item_to_txt_json(item, db)

            updated += 1

        except Exception as e:
            print(f"[BatchTagger] Failed to process item {item_id}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        processed += 1
        send_progress_callback(processed, total, f"Processed {processed}/{total} items")

    if updated > 0:
        send_progress_callback(total, total, "Updating tag statistics...")
        await update_tag_statistics(dataset_id, db)

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
    from datetime import datetime

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
                tags_caption.content = new_content
                tags_caption.updated_at = datetime.utcnow()
                db.commit()

                await save_item_to_txt_json(item, db)

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
    from datetime import datetime

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

                tags_caption.content = ', '.join(new_tags)
                tags_caption.updated_at = datetime.utcnow()
                db.commit()

                await save_item_to_txt_json(item, db)

                updated += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"[BatchReplace] Failed to process item {item_id}: {e}")
            failed += 1

        processed += 1
        if processed % 10 == 0 or processed == total:
            send_progress_callback(processed, total, f"Processed {processed}/{total} items")

    if updated > 0:
        send_progress_callback(total, total, "Updating tag statistics...")
        await update_tag_statistics(dataset_id, db)

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
    import json
    from database.models import DatasetCaption, DatasetItem

    def send_progress(current, total, message):
        if send_progress_callback:
            send_progress_callback(current, total, message)

    taglist_cache.initialize(settings.root_dir)

    # Count captions that need backfilling for this dataset
    item_ids_subq = (
        db.query(DatasetItem.id)
        .filter(DatasetItem.dataset_id == request.dataset_id)
        .subquery()
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
