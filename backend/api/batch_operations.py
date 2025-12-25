"""
Batch operations for dataset items (tagger inference, tag reordering, tag replacement)
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import asyncio
from pathlib import Path

from utils.taglist_loader import load_all_tags

# Global cancellation flag for batch operations
_batch_operation_cancelled = False

def reset_cancellation_flag():
    """Reset the global cancellation flag"""
    global _batch_operation_cancelled
    _batch_operation_cancelled = False

def cancel_batch_operation():
    """Set the global cancellation flag"""
    global _batch_operation_cancelled
    _batch_operation_cancelled = True

def is_batch_operation_cancelled() -> bool:
    """Check if batch operation is cancelled"""
    global _batch_operation_cancelled
    return _batch_operation_cancelled


# ============================================================
# Pydantic Models
# ============================================================

class BatchTaggerRequest(BaseModel):
    item_ids: List[int]
    gen_threshold: float = 0.45
    char_threshold: float = 0.45
    thresholds: Optional[Dict[str, float]] = None
    model_version: str = "cl_tagger_1_02"
    remove_below_threshold: bool = True
    merge_with_existing: bool = True

class BatchReorderTagsRequest(BaseModel):
    item_ids: List[int]
    category_order: List[str]

class BatchReplaceTagRequest(BaseModel):
    item_ids: List[int]
    from_tag: str
    to_tag: str
    normalize_match: bool = True  # Use normalized matching (whitespace, underscores)

class BatchOperationResponse(BaseModel):
    status: str
    processed_count: int
    updated_count: int
    skipped_count: int
    failed_count: int
    message: str


# ============================================================
# Helper Functions
# ============================================================

async def save_item_to_txt_json(item, db):
    """
    Save item captions to txt/json file
    """
    from database.models import DatasetCaption

    # Get tags caption
    tags_caption = db.query(DatasetCaption).filter(
        DatasetCaption.item_id == item.id,
        DatasetCaption.caption_type == "tags"
    ).first()

    if not tags_caption:
        return

    # Get base path (image path without extension)
    image_path = Path(item.image_path)
    base_path = image_path.parent / image_path.stem

    # Save to .txt file
    txt_path = base_path.with_suffix('.txt')
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(tags_caption.content)
    except Exception as e:
        print(f"[BatchOps] Failed to save txt file {txt_path}: {e}")


async def update_tag_statistics(dataset_id: int, db):
    """
    Update tag statistics for a dataset
    """
    from database.models import Dataset, DatasetCaption
    from sqlalchemy import func

    # Get all tags captions
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

    # Update dataset tag_statistics
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset:
        dataset.tag_statistics = {tag: {"count": count} for tag, count in tag_counts.items()}
        db.commit()


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


# ============================================================
# Batch Tagger Inference
# ============================================================

async def batch_tagger_inference(
    request: BatchTaggerRequest,
    db,
    send_progress_callback
) -> BatchOperationResponse:
    """
    Run tagger inference on multiple items
    """
    from database.models import DatasetItem, DatasetCaption
    from core.extensions.tagger_manager import tagger_manager
    from PIL import Image
    import base64
    import io

    reset_cancellation_flag()

    total = len(request.item_ids)
    processed = 0
    updated = 0
    skipped = 0
    failed = 0

    send_progress_callback(0, total, "Starting batch tagger inference...")

    # Load tagger model if not loaded
    if not tagger_manager.is_loaded():
        send_progress_callback(0, total, "Loading tagger model...")
        tagger_manager.load_model(
            use_gpu=True,
            use_huggingface=True,
            model_version=request.model_version
        )

    for idx, item_id in enumerate(request.item_ids):
        if is_batch_operation_cancelled():
            send_progress_callback(processed, total, "Batch operation cancelled")
            break

        try:
            # Get item
            item = db.query(DatasetItem).filter(DatasetItem.id == item_id).first()
            if not item:
                skipped += 1
                processed += 1
                continue

            send_progress_callback(
                processed,
                total,
                f"Processing {item.base_name} ({processed + 1}/{total})"
            )

            # Load image and convert to base64
            image = Image.open(item.image_path)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Run tagger inference
            result = tagger_manager.predict(
                image_base64,
                gen_threshold=request.gen_threshold,
                char_threshold=request.char_threshold,
                thresholds=request.thresholds or {}
            )

            # Get existing tags caption
            tags_caption = db.query(DatasetCaption).filter(
                DatasetCaption.item_id == item.id,
                DatasetCaption.caption_type == "tags"
            ).first()

            existing_tags = []
            if tags_caption and request.merge_with_existing:
                existing_tags = [t.strip() for t in tags_caption.content.split(',') if t.strip()]

            # Merge predicted tags
            new_tags = []

            # Add existing tags first (if merging)
            if request.merge_with_existing:
                new_tags.extend(existing_tags)

            # Add predicted tags (avoiding duplicates)
            for category, predictions in result['predictions'].items():
                for tag, score in predictions:
                    if tag not in new_tags:
                        new_tags.append(tag)

            # Update or create caption
            content = ', '.join(new_tags)

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

            # Save to txt file
            await save_item_to_txt_json(item, db)

            updated += 1

        except Exception as e:
            print(f"[BatchTagger] Failed to process item {item_id}: {e}")
            failed += 1

        processed += 1
        send_progress_callback(processed, total, f"Processed {processed}/{total} items")

    # Update tag statistics
    if updated > 0:
        send_progress_callback(total, total, "Updating tag statistics...")
        # Get dataset_id from first item
        first_item = db.query(DatasetItem).filter(DatasetItem.id == request.item_ids[0]).first()
        if first_item:
            await update_tag_statistics(first_item.dataset_id, db)

    cancelled = is_batch_operation_cancelled()
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


# ============================================================
# Batch Tag Reordering
# ============================================================

async def batch_reorder_tags(
    request: BatchReorderTagsRequest,
    db,
    send_progress_callback
) -> BatchOperationResponse:
    """
    Reorder tags by category for multiple items
    """
    from database.models import DatasetItem, DatasetCaption
    from datetime import datetime

    reset_cancellation_flag()

    total = len(request.item_ids)
    processed = 0
    updated = 0
    skipped = 0
    failed = 0

    send_progress_callback(0, total, "Starting batch tag reordering...")

    # Load taglist for category detection
    all_tags = load_all_tags()
    tag_to_category = {}
    for tag_entry in all_tags:
        tag_to_category[tag_entry['tag']] = tag_entry['category']

    for idx, item_id in enumerate(request.item_ids):
        if is_batch_operation_cancelled():
            send_progress_callback(processed, total, "Batch operation cancelled")
            break

        try:
            # Get item
            item = db.query(DatasetItem).filter(DatasetItem.id == item_id).first()
            if not item:
                skipped += 1
                processed += 1
                continue

            # Get tags caption
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

            # Parse tags
            tags = [t.strip() for t in tags_caption.content.split(',') if t.strip()]

            # Categorize tags
            categorized = {cat: [] for cat in request.category_order}
            categorized['Unknown'] = []

            for tag in tags:
                category = tag_to_category.get(tag, 'General')
                if category in categorized:
                    categorized[category].append(tag)
                else:
                    categorized['Unknown'].append(tag)

            # Rebuild tag list in category order
            reordered_tags = []
            for category in request.category_order:
                reordered_tags.extend(categorized[category])
            # Add unknown tags at the end
            reordered_tags.extend(categorized['Unknown'])

            # Update caption
            new_content = ', '.join(reordered_tags)
            if new_content != tags_caption.content:
                tags_caption.content = new_content
                tags_caption.updated_at = datetime.utcnow()
                db.commit()

                # Save to txt file
                await save_item_to_txt_json(item, db)

                updated += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"[BatchReorder] Failed to process item {item_id}: {e}")
            failed += 1

        processed += 1
        send_progress_callback(processed, total, f"Processed {processed}/{total} items")

    cancelled = is_batch_operation_cancelled()
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


# ============================================================
# Batch Tag Replacement
# ============================================================

async def batch_replace_tag(
    request: BatchReplaceTagRequest,
    db,
    send_progress_callback
) -> BatchOperationResponse:
    """
    Replace a specific tag with another tag for multiple items
    """
    from database.models import DatasetItem, DatasetCaption
    from datetime import datetime

    reset_cancellation_flag()

    total = len(request.item_ids)
    processed = 0
    updated = 0
    skipped = 0
    failed = 0

    from_tag_normalized = normalize_tag_for_matching(request.from_tag) if request.normalize_match else request.from_tag

    send_progress_callback(0, total, f"Starting batch tag replacement: '{request.from_tag}' → '{request.to_tag}'...")

    for idx, item_id in enumerate(request.item_ids):
        if is_batch_operation_cancelled():
            send_progress_callback(processed, total, "Batch operation cancelled")
            break

        try:
            # Get item
            item = db.query(DatasetItem).filter(DatasetItem.id == item_id).first()
            if not item:
                skipped += 1
                processed += 1
                continue

            # Get tags caption
            tags_caption = db.query(DatasetCaption).filter(
                DatasetCaption.item_id == item.id,
                DatasetCaption.caption_type == "tags"
            ).first()

            if not tags_caption:
                skipped += 1
                processed += 1
                continue

            # Parse tags
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

                # Update caption
                tags_caption.content = ', '.join(new_tags)
                tags_caption.updated_at = datetime.utcnow()
                db.commit()

                # Save to txt file
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

    # Update tag statistics
    if updated > 0:
        send_progress_callback(total, total, "Updating tag statistics...")
        first_item = db.query(DatasetItem).filter(DatasetItem.id == request.item_ids[0]).first()
        if first_item:
            await update_tag_statistics(first_item.dataset_id, db)

    cancelled = is_batch_operation_cancelled()
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
