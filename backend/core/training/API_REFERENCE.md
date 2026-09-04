# SushiUI Training Framework - API Reference

**Last Updated**: 2026-09-04

This document provides a comprehensive API reference for all components, classes, methods, and functions in the `backend/core/training` module.

---

## Table of Contents

1. [LatentCache](#latentcache)
2. [BucketManager](#bucketmanager)
3. [VideoBucketManager](#videobucketmanager)
4. [TagGroupManager](#taggroupmanager)
5. [TrainingConfigGenerator](#trainingconfiggenerator)
6. [Utility Functions](#utility-functions)
7. [BaseTrainer](#basetrainer)

---

## LatentCache

**File**: `latent_cache.py`

Manages disk cache for VAE latents and optionally text embeddings to reduce VRAM usage during training.

### Cache Directory Structure

```
cache/datasets/{dataset_unique_id}/
├── latents/
│   ├── {image_hash}.pt
│   └── ...
├── text_embeddings/  (optional)
│   ├── {caption_hash}_clip1.pt
│   ├── {caption_hash}_clip2.pt  (SDXL only)
│   ├── {caption_hash}_pooled.pt (SDXL only)
│   └── ...
└── cache_info.json
```

### Constructor

```python
LatentCache(
    dataset_unique_id: str,
    base_cache_dir: str = None,
    namespace: str = None
)
```

**Parameters**:
- `dataset_unique_id` (str): Dataset unique ID (UUID)
- `base_cache_dir` (str, optional): Base directory for cache. Defaults to user settings (`cache/datasets`)
- `namespace` (str, optional): Architecture/VAE identity component (see
  `build_cache_namespace`). With it the cache lives at
  `{base}/{dataset_id}/{namespace}/`; without it the legacy
  `{base}/{dataset_id}/` layout is used, and those entries are unlabeled and
  must not be shared across architectures.

**Behavior**:
- Automatically creates cache directories (`latents/`, `text_embeddings/`)
- If `base_cache_dir` is None, fetches from `UserSettings.cache_dir` via database

**Example**:
```python
from core.training.latent_cache import LatentCache

cache = LatentCache(dataset_unique_id="a1b2c3d4-...")
# Cache dir: cache/datasets/a1b2c3d4-.../
```

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `dataset_unique_id` | str | Dataset UUID |
| `cache_dir` | Path | Cache root directory |
| `latents_dir` | Path | Latent tensors directory |
| `embeddings_dir` | Path | Text embeddings directory |
| `cache_info_path` | Path | Path to `cache_info.json` |

### Static Methods

#### `compute_image_hash`

```python
@staticmethod
def compute_image_hash(image_path: str, width: int, height: int) -> str
```

Compute hash for image cache key (includes dimensions for bucketing).

**Parameters**:
- `image_path` (str): Path to image
- `width` (int): Target width
- `height` (int): Target height

**Returns**:
- `str`: MD5 hash string

**Example**:
```python
hash_key = LatentCache.compute_image_hash("dataset/img001.png", 1024, 1024)
# Returns: "a1b2c3d4e5f6..."
```

#### `compute_caption_hash`

```python
@staticmethod
def compute_caption_hash(caption: str) -> str
```

Compute hash for caption cache key.

**Parameters**:
- `caption` (str): Text caption

**Returns**:
- `str`: MD5 hash string

### Instance Methods

#### `save_latent`

```python
def save_latent(
    self,
    image_path: str,
    width: int,
    height: int,
    latents: torch.Tensor,
    skip_existing: bool = True
) -> bool
```

Save VAE latents to cache.

**Parameters**:
- `image_path` (str): Source image path
- `width` (int): Target width
- `height` (int): Target height
- `latents` (torch.Tensor): Latent tensor `[1, C, H/8, W/8]` where C=4 (SD/SDXL) or C=16 (Z-Image)
- `skip_existing` (bool): If True, skip if cache file already exists (default: True)

**Returns**:
- `bool`: True if saved (new file), False if skipped (existing file)

**Cache File Format**:
```python
{
    'latents': torch.Tensor,  # [1, C, H/8, W/8]
    'image_path': str,
    'width': int,
    'height': int,
    'created_at': str  # ISO 8601 timestamp
}
```

**Example**:
```python
was_saved = cache.save_latent(
    image_path="dataset/img001.png",
    width=1024,
    height=1024,
    latents=latent_tensor,
    skip_existing=True
)
```

#### `has_latent`

```python
def has_latent(
    self,
    image_path: str,
    width: int,
    height: int,
) -> bool
```

Check if latent exists in cache WITHOUT loading it.

**Parameters**:
- `image_path` (str): Source image path
- `width` (int): Target width
- `height` (int): Target height

**Returns**:
- `bool`: True if latent is cached, False otherwise

**Example**:
```python
if cache.has_latent("dataset/img001.png", 1024, 1024):
    latent = cache.load_latent("dataset/img001.png", 1024, 1024)
else:
    # Encode and save new latent
    latent = encode_image(...)
    cache.save_latent("dataset/img001.png", 1024, 1024, latent)
```

#### `load_latent`

```python
def load_latent(
    self,
    image_path: str,
    width: int,
    height: int,
    device: str = 'cuda'
) -> Optional[torch.Tensor]
```

Load VAE latents from cache.

**Parameters**:
- `image_path` (str): Source image path
- `width` (int): Target width
- `height` (int): Target height
- `device` (str): Device to load tensor to (default: 'cuda')

**Returns**:
- `torch.Tensor | None`: Latent tensor or None if not cached

**Example**:
```python
latent = cache.load_latent("dataset/img001.png", 1024, 1024, device='cuda')
if latent is None:
    # Cache miss - need to encode
    latent = vae.encode(image)
```

#### `save_text_embeddings`

```python
def save_text_embeddings(
    self,
    caption: str,
    text_embeddings: torch.Tensor,
    pooled_embeddings: Optional[torch.Tensor] = None,
    text_embeddings_2: Optional[torch.Tensor] = None
)
```

Save text embeddings to cache.

**Parameters**:
- `caption` (str): Text caption
- `text_embeddings` (torch.Tensor): Text embeddings from first encoder `[1, 77, 768]` (SD1.5/SDXL CLIP-L)
- `pooled_embeddings` (torch.Tensor, optional): Pooled embeddings (SDXL only) `[1, 1280]`
- `text_embeddings_2` (torch.Tensor, optional): Text embeddings from second encoder (SDXL only) `[1, 77, 1280]`

**Saved Files**:
- `{caption_hash}_clip1.pt`: First encoder embeddings (always)
- `{caption_hash}_pooled.pt`: Pooled embeddings (SDXL only)
- `{caption_hash}_clip2.pt`: Second encoder embeddings (SDXL only)

**Example**:
```python
# SD1.5
cache.save_text_embeddings(
    caption="1girl, anime",
    text_embeddings=clip_embeddings
)

# SDXL
cache.save_text_embeddings(
    caption="1girl, anime",
    text_embeddings=clip_l_embeddings,
    pooled_embeddings=pooled,
    text_embeddings_2=clip_g_embeddings
)
```

#### `load_text_embeddings`

```python
def load_text_embeddings(
    self,
    caption: str,
    is_sdxl: bool = False,
    device: str = 'cuda'
) -> Optional[Tuple[torch.Tensor, ...]]
```

Load text embeddings from cache.

**Parameters**:
- `caption` (str): Text caption
- `is_sdxl` (bool): Whether to load SDXL embeddings (includes pooled and clip2)
- `device` (str): Device to load tensors to

**Returns**:
- For SD1.5: `(text_embeddings,)` - Tuple with 1 element
- For SDXL: `(text_embeddings, pooled_embeddings)` - Tuple with 2 elements
- `None` if not cached

**Example**:
```python
# SD1.5
embeddings = cache.load_text_embeddings("1girl", is_sdxl=False)
if embeddings:
    text_emb = embeddings[0]

# SDXL
embeddings = cache.load_text_embeddings("1girl", is_sdxl=True)
if embeddings:
    text_emb, pooled_emb = embeddings
```

#### `save_cache_info`

```python
def save_cache_info(
    self,
    model_path: str,
    model_type: str,
    item_count: int,
    training_dtype: str = 'unknown'
)
```

Save cache metadata to `cache_info.json`.

**Parameters**:
- `model_path` (str): Path to base model
- `model_type` (str): Model type ('sdxl', 'sd15', 'sd', 'zimage', 'z-image')
- `item_count` (int): Number of items in dataset
- `training_dtype` (str): Training dtype (e.g., 'bfloat16', 'float16', 'float32')

**Metadata Format**:
```json
{
    "dataset_unique_id": "a1b2c3d4-...",
    "model_path": "models/model.safetensors",
    "model_type": "sdxl",
    "training_dtype": "bfloat16",
    "created_at": "2025-12-15T10:30:00.000Z",
    "item_count": 150
}
```

**Example**:
```python
cache.save_cache_info(
    model_path="models/sdxl_base.safetensors",
    model_type="sdxl",
    item_count=150,
    training_dtype="bfloat16"
)
```

#### `load_cache_info`

```python
def load_cache_info(self) -> Optional[Dict]
```

Load cache metadata from `cache_info.json`.

**Returns**:
- `dict | None`: Cache info dict or None if not exists

**Example**:
```python
info = cache.load_cache_info()
if info:
    print(f"Model: {info['model_path']}")
    print(f"Type: {info['model_type']}")
```

#### `is_valid`

```python
def is_valid(
    self,
    model_path: str,
    model_type: str,
    training_dtype: str = 'unknown'
) -> bool
```

Check if cache is valid for current model.

**Parameters**:
- `model_path` (str): Current model path
- `model_type` (str): Current model type
- `training_dtype` (str): Current training dtype

**Returns**:
- `bool`: True if cache is valid

**Validation Checks**:
1. `cache_info.json` exists
2. Model path matches (normalized absolute paths)
3. Model type matches
4. Training dtype matches (if not 'unknown')

**Important Notes**:
- **This method also checks if cache exists** (returns False if `cache_info.json` is missing)
- No separate `exists()` method is needed

**Example**:
```python
dtype_str = str(training_dtype).replace('torch.', '')  # 'torch.bfloat16' -> 'bfloat16'
if cache.is_valid(model_path, model_type, dtype_str):
    print("Cache is valid, reusing...")
else:
    print("Cache invalid, regenerating...")
```

#### `validate_cache_format`

```python
def validate_cache_format(
    self,
    expected_channels: int = 4,
    sample_count: int = 5
) -> bool
```

Validate cache format by randomly sampling cached latents.

**Parameters**:
- `expected_channels` (int): Expected number of latent channels (4 for SD/SDXL, 16 for Z-Image)
- `sample_count` (int): Number of random samples to check

**Returns**:
- `bool`: True if cache format is valid, False otherwise

**Validation Checks**:
- At least 1 cached latent file exists
- Random sample can be loaded
- Latent is 4D tensor `[B, C, H, W]`
- Channel count matches `expected_channels`

**Example**:
```python
expected_channels = 16 if is_zimage else 4
if cache.validate_cache_format(expected_channels=expected_channels, sample_count=5):
    print("Cache format is valid")
else:
    print("Cache format validation failed, regenerating...")
```

#### `clear`

```python
def clear(self)
```

Clear all cached data (removes all files, recreates directories).

**Example**:
```python
cache.clear()
```

---

## BucketManager

**File**: `bucketing.py`

Manages aspect ratio bucketing for training datasets with multiple resolution support.

### Constructor

```python
BucketManager(
    base_resolutions: List[int],
    divisibility: int = 8,
    strategy: Literal["resize", "crop", "random_crop"] = "resize",
    multi_resolution_mode: Literal["max", "random"] = "max",
    separate_by_reference: bool = False
)
```

**Parameters**:
- `base_resolutions` (List[int]): List of base resolutions (e.g., `[512, 768, 1024]`)
- `divisibility` (int): All dimensions must be divisible by this (default: 8 for VAE)
- `strategy` (Literal): How to handle oversized images:
  - `"resize"`: Resize image to fit bucket
  - `"crop"`: Center crop to bucket size
  - `"random_crop"`: Random crop to bucket size
- `multi_resolution_mode` (Literal): How to assign images when multiple resolutions specified:
  - `"max"`: Use largest resolution that fits the image (minimizes cropping, default)
  - `"random"`: Randomly select from available resolutions
- `separate_by_reference` (bool): Key buckets by `(resolution, has_reference)` so
  a batch is uniform in reference status as well as size (default: False)

**Example**:
```python
from core.training.bucketing import BucketManager

bucket_manager = BucketManager(
    base_resolutions=[1024],
    divisibility=8,
    strategy="resize",
    multi_resolution_mode="max"
)
```

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `base_resolutions` | List[int] | Sorted list of base resolutions |
| `divisibility` | int | Dimension divisibility requirement |
| `strategy` | str | Image handling strategy |
| `multi_resolution_mode` | str | Multi-resolution assignment mode |
| `bucket_lists` | Dict[int, List[BucketResolution]] | Pre-generated bucket lists per resolution |
| `buckets` | Dict[BucketResolution, List[Dict]] | Images assigned to each bucket |

### Instance Methods

#### `assign_image_to_bucket`

```python
def assign_image_to_bucket(
    self,
    image_path: str,
    width: int,
    height: int,
    caption: str = "",
    target_resolution: Optional[int] = None,
    dataset_unique_id: Optional[str] = None,
    has_reference: bool = False,
    reference_images: Optional[list] = None,
    forced_bucket: Optional[BucketResolution] = None,
) -> Tuple[BucketKey, Dict]
```

Assign an image to the best bucket.

**Parameters**:
- `image_path` (str): Path to image file
- `width` (int): Image width
- `height` (int): Image height
- `caption` (str): Image caption (default: "")
- `target_resolution` (int, optional): Specific resolution to use (or None for auto)
- `dataset_unique_id` (str, optional): Dataset UUID (for cache management)
- `has_reference` (bool): Item carries reference images (default: False)
- `reference_images` (list, optional): Reference image entries for the item
- `forced_bucket` (BucketResolution, optional): Bypass selection and use this bucket

**Returns**:
- `Tuple[BucketKey, Dict]`: (bucket key, image_info). The key is a
  `BucketResolution`, or `(BucketResolution, has_reference)` when the manager
  was built with `separate_by_reference=True`.

**Image Info Dict**:
```python
{
    "image_path": str,
    "caption": str,
    "original_width": int,
    "original_height": int,
    "bucket_width": int,
    "bucket_height": int,
    "target_resolution": int,
    "dataset_unique_id": str  # (if provided)
}
```

**Example**:
```python
bucket, image_info = bucket_manager.assign_image_to_bucket(
    image_path="dataset/img001.png",
    width=1200,
    height=800,
    caption="1girl, anime",
    dataset_unique_id="a1b2c3d4-..."
)

print(f"Assigned to bucket: {bucket.width}x{bucket.height}")
print(f"Original size: {image_info['original_width']}x{image_info['original_height']}")
```

#### `get_bucket_counts`

```python
def get_bucket_counts(self) -> Dict[str, int]
```

Get count of images in each bucket.

**Returns**:
- `Dict[str, int]`: Mapping of `"{width}x{height}"` → count

**Example**:
```python
counts = bucket_manager.get_bucket_counts()
# {'1024x1024': 50, '1024x768': 30, '768x1024': 20}
```

#### `get_all_items`

```python
def get_all_items(self) -> List[Dict]
```

Get all items across all buckets (shuffled).

**Returns**:
- `List[Dict]`: List of all image info dicts (shuffled)

**Example**:
```python
all_items = bucket_manager.get_all_items()
for item in all_items:
    print(item["image_path"], item["bucket_width"], item["bucket_height"])
```

#### `get_items_by_bucket`

```python
def get_items_by_bucket(self) -> Dict[BucketKey, List[Dict]]
```

Get items grouped by bucket.

**Returns**:
- `Dict[BucketResolution, List[Dict]]`: Copy of buckets dict

**Example**:
```python
by_bucket = bucket_manager.get_items_by_bucket()
for bucket, items in by_bucket.items():
    print(f"Bucket {bucket.width}x{bucket.height}: {len(items)} images")
```

#### `shuffle_buckets`

```python
def shuffle_buckets(self)
```

Shuffle items within each bucket (in-place).

**Example**:
```python
bucket_manager.shuffle_buckets()
```

#### `build_batch_indices`

```python
def build_batch_indices(self, batch_size: int) -> List[List[Dict]]
```

Build batch indices for training.

**Parameters**:
- `batch_size` (int): Number of items per batch

**Returns**:
- `List[List[Dict]]`: List of batches, where each batch is a list of image info dicts

**Behavior**:
- Groups items from the same bucket into batches of `batch_size`
- Ensures all items in a batch have the same resolution
- Shuffles the batches (not the items within batches)

**Example**:
```python
batches = bucket_manager.build_batch_indices(batch_size=4)
for batch in batches:
    print(f"Batch: {len(batch)} images, resolution: {batch[0]['bucket_width']}x{batch[0]['bucket_height']}")
```

### Helper Functions

#### `get_bucket_sizes`

```python
def get_bucket_sizes(resolution: int = 512, divisibility: int = 8) -> List[BucketResolution]
```

Generate bucket sizes for a given base resolution.

**Parameters**:
- `resolution` (int): Base resolution (e.g., 512, 768, 1024)
- `divisibility` (int): All dimensions must be divisible by this

**Returns**:
- `List[BucketResolution]`: List of bucket resolutions scaled from SDXL base (1024x1024)

**Example**:
```python
from core.training.bucketing import get_bucket_sizes

buckets_512 = get_bucket_sizes(resolution=512, divisibility=8)
# Returns scaled versions of SDXL buckets
```

#### `get_bucket_for_image_size`

```python
def get_bucket_for_image_size(
    width: int,
    height: int,
    bucket_list: Optional[List[BucketResolution]] = None,
    resolution: Optional[int] = None,
    divisibility: int = 8
) -> BucketResolution
```

Find the best bucket for an image size.

**Parameters**:
- `width` (int): Image width
- `height` (int): Image height
- `bucket_list` (List[BucketResolution], optional): Pre-generated bucket list
- `resolution` (int, optional): Base resolution if bucket_list not provided
- `divisibility` (int): Dimension divisibility requirement

**Returns**:
- `BucketResolution`: Best matching bucket resolution

**Example**:
```python
from core.training.bucketing import get_bucket_for_image_size

bucket = get_bucket_for_image_size(1200, 800, resolution=1024)
print(f"Best bucket: {bucket.width}x{bucket.height}")
```

---

## VideoBucketManager

**File**: `bucketing.py`

Temporal bucketing for video clips. A standalone sibling of `BucketManager`: it
keys buckets by the pair `(spatial_bucket, clip_length)`, so a batch drawn from
one bucket is uniform in both the spatial size and the frame count, which is
what lets the 5D latents stack. It reuses a `BucketManager` internally for the
spatial bucket lists only and never mutates the image path.

### Constructor

```python
VideoBucketManager(
    base_resolutions: List[int],
    divisibility: Optional[int] = None,
    allowed_clip_lengths: Optional[List[int]] = None,
    stride: int = 1,
    multi_resolution_mode: Literal["max", "random"] = "max",
    temporal_spec: Optional[TemporalSpec] = None,
)
```

**Parameters**:
- `base_resolutions` (List[int]): Base resolutions for the spatial buckets
- `divisibility` (int, optional): Spatial divisibility; defaults to the
  `temporal_spec`'s `pixel_align`, or the LTX spatial divisibility when no spec
  is given. Video callers pass 32.
- `allowed_clip_lengths` (List[int], optional): Candidate frame counts. Filtered
  through `is_valid_clip_length` against `temporal_spec`; defaults to the spec's
  `default_clip_lengths`, or `DEFAULT_CLIP_LENGTHS` when no spec is given.
- `stride` (int): Frame stride used when picking a clip length (minimum 1)
- `multi_resolution_mode` (Literal): Passed through to the internal `BucketManager`
- `temporal_spec` (TemporalSpec, optional): Per-architecture temporal rules.
  `None` selects the LTX-2.3 index-sampled rule this class shipped with
  (`core.models.components.wiring.TemporalSpec`).

### Instance Methods

| Method | Description |
|---|---|
| `select_spatial_bucket(width, height, target_resolution=None)` | Spatial `BucketResolution` for a clip; no state mutation |
| `pick_clip_length(num_frames, stride=None, source_fps=None)` | Clip length from this manager's allowed set, stride and temporal spec |
| `assign_video_to_bucket(video_path, width, height, num_frames, caption="", stride=None, fps=None, target_resolution=None, dataset_unique_id=None)` | Assigns an item and returns `((BucketResolution, clip_length), video_info)`; the chosen bucket and clip length are in the info dict so the caller can build the clip cache key from the actual window used |
| `clip_cache_params(...)` | Keys matching `LatentCache.compute_clip_hash` / `save_clip_latent` |
| `get_bucket_counts()`, `get_items_by_bucket()`, `shuffle_buckets()`, `build_batch_indices(batch_size)` | As on `BucketManager`, but keyed by `(BucketResolution, clip_length)` |

Module-level helpers for the same rules: `is_valid_clip_length`, `clip_span`,
`pick_clip_length`, `get_video_spatial_bucket`, `clip_cache_key_extras`.

---

## TagGroupManager

**File**: `tag_group_utils.py`

Manages tag groups for caption processing (categorization, shuffle, dropout).

### Constructor

```python
TagGroupManager(tag_group_dir: str = "taglist", enable_gelbooru: bool = False)
```

**Parameters**:
- `tag_group_dir` (str): Directory containing tag group JSON files (default: "taglist")
- `enable_gelbooru` (bool): Also load the gelbooru supplement from `taglist_gel/`
  (training only, to reduce "Unknown" tags; larger vocabulary, more noise)

**Behavior**:
- Automatically loads tag groups from JSON files
- If relative path, resolves from project root
- Adds hardcoded Rating and Quality tags

**Example**:
```python
from core.training.tag_group_utils import TagGroupManager

tag_manager = TagGroupManager(tag_group_dir="taglist")
```

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `tag_group_dir` | Path | Directory containing JSON files |
| `tag_groups` | Dict[str, Set[str]] | Mapping of group name → set of tags |
| `_tag_to_group_cache` | Dict[str, str] | Mapping of normalized tag → group name |

### Hardcoded Tag Groups

**Rating Tags**:
```python
{'general', 'sensitive', 'questionable', 'explicit',
 'rating:general', 'rating:sensitive', 'rating:questionable', 'rating:explicit'}
```

**Quality Tags**:
```python
{'best quality', 'high quality', 'great quality', 'normal quality',
 'low quality', 'worst quality', 'masterpiece', 'amazing quality'}
```

### Instance Methods

#### `load_tag_groups`

```python
def load_tag_groups(self)
```

Load tag groups from JSON files (called automatically in `__init__`).

**Expected JSON Format**:
```json
{
    "tag_name_1": count,
    "tag_name_2": count,
    ...
}
```

**Loaded Groups** (from `taglist/` directory):
- `Character.json` → Character tags
- `General.json` → General tags
- `Copyright.json` → Copyright tags
- `Artist.json` → Artist tags
- `Meta.json` → Meta tags

#### `get_tag_group`

```python
def get_tag_group(self, tag: str) -> Optional[str]
```

Get group name for a tag.

**Parameters**:
- `tag` (str): Tag string (case-insensitive, handles underscores/escapes)

**Returns**:
- `str | None`: Group name or None if not found

**Example**:
```python
group = tag_manager.get_tag_group("hakurei_reimu")
# Returns: "Character"

group = tag_manager.get_tag_group("long hair")
# Returns: "General"

group = tag_manager.get_tag_group("masterpiece")
# Returns: "Quality"
```

#### `is_person_count_tag`

```python
def is_person_count_tag(self, tag: str) -> bool
```

Check if tag is a person count tag.

**Parameters**:
- `tag` (str): Tag string

**Returns**:
- `bool`: True if tag is a person count tag

**Person Count Tags**:
```python
{'no_humans', 'solo', 'group', 'still_life', 'multiple_girls', 'multiple_boys',
 '1girl', '2girls', '3girls', ..., '1boy', '2boys', ..., '1other', '2others', ...}
```

**Example**:
```python
is_person_count = tag_manager.is_person_count_tag("1girl")
# Returns: True

is_person_count = tag_manager.is_person_count_tag("long hair")
# Returns: False
```

#### `categorize_tags`

```python
def categorize_tags(self, tags: List[str]) -> Dict[str, List[str]]
```

Categorize tags by group.

**Parameters**:
- `tags` (List[str]): List of tags

**Returns**:
- `Dict[str, List[str]]`: Mapping of group name → list of tags

**Example**:
```python
tags = ["hakurei_reimu", "long hair", "masterpiece", "touhou"]
categorized = tag_manager.categorize_tags(tags)
# {
#     "Character": ["hakurei_reimu"],
#     "General": ["long hair"],
#     "Quality": ["masterpiece"],
#     "Copyright": ["touhou"]
# }
```

#### `shuffle_by_groups`

```python
def shuffle_by_groups(
    self,
    tokens: List[str],
    groups_to_shuffle: List[str],
    keep_first_n: int = 0,
    exclude_person_count: bool = False,
    shuffle_together: bool = False,
    rng: Optional[random.Random] = None,
) -> List[str]
```

Shuffle tokens by tag groups.

**Parameters**:
- `tokens` (List[str]): List of tokens (comma-separated tags)
- `groups_to_shuffle` (List[str]): List of group names to shuffle (e.g., `["Character", "General"]`)
- `keep_first_n` (int): Number of first tokens to keep unshuffled (default: 0)
- `exclude_person_count` (bool): Exclude person count tags from General group shuffling (default: False)
- `shuffle_together` (bool): Shuffle all selected groups together vs within each group (default: False)
- `rng` (random.Random, optional): Random number generator for reproducibility

**Returns**:
- `List[str]`: Shuffled token list

**Example**:
```python
tokens = ["1girl", "hakurei_reimu", "long hair", "red bow", "touhou"]
shuffled = tag_manager.shuffle_by_groups(
    tokens=tokens,
    groups_to_shuffle=["General"],
    keep_first_n=1,
    exclude_person_count=True
)
# Possible result: ["1girl", "hakurei_reimu", "red bow", "long hair", "touhou"]
# Note: "1girl" kept (keep_first_n=1), "hakurei_reimu" and "touhou" not shuffled (not in General)
```

### Helper Functions

#### `normalize_tag_for_matching`

```python
def normalize_tag_for_matching(tag: str) -> str
```

Normalize tag for matching purposes (lowercase, remove escapes, underscores → spaces).

**Parameters**:
- `tag` (str): Tag string

**Returns**:
- `str`: Normalized tag

**Normalization Rules**:
- Remove backslash escapes: `\\` → ` `
- Replace underscores with spaces: `_` → ` `
- Lowercase

**Example**:
```python
from core.training.tag_group_utils import normalize_tag_for_matching

normalized = normalize_tag_for_matching("hakurei_reimu")
# Returns: "hakurei reimu"

normalized = normalize_tag_for_matching("djibril_\\(makai_tenshi_djibril\\)")
# Returns: "djibril (makai tenshi djibril)"
```

#### `normalize_tag_for_output`

```python
def normalize_tag_for_output(tag: str) -> str
```

Normalize tag for output (standardize to escaped parentheses format).

**Target Format**: `"tag \\(qualifier\\)"`

**Parameters**:
- `tag` (str): Tag string

**Returns**:
- `str`: Normalized tag for output

**Example**:
```python
from core.training.tag_group_utils import normalize_tag_for_output

output = normalize_tag_for_output("djibril_(makai_tenshi_djibril)")
# Returns: "djibril \\(makai tenshi djibril\\)"

output = normalize_tag_for_output("long_hair")
# Returns: "long hair"
```

#### `get_tag_group_manager`

```python
def get_tag_group_manager(
    tag_group_dir: str = "taglist",
    enable_gelbooru: bool = False
) -> TagGroupManager
```

Get or create tag group manager (cached globally).

**Parameters**:
- `tag_group_dir` (str): Directory containing tag group JSON files
- `enable_gelbooru` (bool): Enable the gelbooru supplement; once enabled it stays
  enabled for the session

**Returns**:
- `TagGroupManager`: Cached instance

**Example**:
```python
from core.training.tag_group_utils import get_tag_group_manager

tag_manager = get_tag_group_manager()
```

---

## TrainingConfigGenerator

**File**: `training_config.py`

Generates YAML configuration files for ai-toolkit-based training.

### Static Methods

#### `generate_lora_config`

```python
@staticmethod
def generate_lora_config(
    p: Optional[Dict[str, Any]] = None,
    *,
    run_name: str,
    base_model_path: str,
    output_dir: str,
    dataset_path: str = "",
    dataset_configs: Optional[List[Dict[str, Any]]] = None,
    sample_prompts: Optional[list] = None,
    caption_processing: Optional[Dict[str, Any]] = None,
    **legacy_kwargs: Any,
) -> str
```

Generate LoRA training configuration YAML.

**Parameters**:
- `p` (Dict, optional): The training parameters dict — normally
  `TrainingRunCreateRequest.model_dump()`. Every knob not named below
  (`total_steps`/`epochs`, `batch_size`, `learning_rate`, `optimizer`,
  `lora_rank`, `lora_alpha`, bucketing, dtypes, sample settings,
  `resume_from_checkpoint`, ...) is read out of this dict. It also carries
  `_explicit_fields` (the route passes `request.model_fields_set`), which the
  sample-default resolver uses to tell a caller-supplied value from a Pydantic
  default.
- `run_name` (str, keyword-only): Training run identifier
- `base_model_path` (str, keyword-only): Path to base model
- `output_dir` (str, keyword-only): Output directory for checkpoints
- `dataset_path` (str, keyword-only): Deprecated, use `dataset_configs`
- `dataset_configs` (List[Dict], keyword-only): List of dataset configurations
- `sample_prompts` (list, keyword-only): Sample prompts
- `caption_processing` (Dict, keyword-only): Caption processing config from the
  database. Not written to the YAML — the trainer reads
  `Dataset.caption_processing` at training time.
- `**legacy_kwargs`: absorbs the old kwargs-style call. They are **merged into
  `p`, not ignored and not warned about**: with no `p`, `p = legacy_kwargs`;
  with both, `p = {**p, **legacy_kwargs}`, so a legacy kwarg overrides the same
  key in `p`. An old-shape call therefore still works.

Either `total_steps` or `epochs` must be present in the resulting dict, and not
both; otherwise `ValueError`.

**Returns**:
- `str`: YAML configuration string

**Example**:
```python
from core.training.training_config import TrainingConfigGenerator

yaml_config = TrainingConfigGenerator.generate_lora_config(
    {
        "epochs": 10,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "lora_rank": 16,
        "enable_bucketing": True,
        "base_resolutions": [1024],
    },
    run_name="lora_training_001",
    base_model_path="models/sdxl_base.safetensors",
    output_dir="training/lora_001",
    dataset_configs=[{"path": "datasets/my_dataset"}],
)
```

#### The `network` block (adapter algebra)

`generate_lora_config` writes the run's adapter algebra into
`config.process[0].network`:

```yaml
network:
  type: lora            # "relora" after generate_relora_config rewrites it
  adapter_algorithm: lora   # lora | loha | lokr
  weight_decompose: false   # DoRA/DoHa/DoKr magnitude vector
  linear: 16                # from lora_rank
  linear_alpha: 16          # from lora_alpha
  lora_dtype: fp32
  adapter_config: {}        # algebra-specific options
```

Defaults live only in `backend/api/param_defaults.py` (`TRAINING_DEFAULTS`):
`adapter_algorithm: "lora"`, `weight_decompose: False`, `adapter_config: {}`,
`lora_rank: 16`, `lora_alpha: 16`, `lora_dtype: "fp32"`. A YAML written before
these keys existed — or any run that omits them — normalizes to ordinary LoRA
without weight decomposition (`TrainingAdapterSpec`, `adapters/base_adapter.py`),
and an ordinary-LoRA checkpoint stays byte-identical to what every architecture
already writes: `TrainingAdapterSpec.metadata()` emits the `sushi.adapter.*`
block only for a non-ordinary algebra.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `adapter_algorithm` | `"lora"` \| `"loha"` \| `"lokr"` | `"lora"` | The algebra the branch is built from. Any other value is refused by name. |
| `weight_decompose` | bool | `false` | Adds one `dora_scale` per target on top of the algebra (DoRA / DoHa / DoKr). |
| `adapter_config` | dict | `{}` | Algebra-specific options. **API-only — no UI control writes it.** |

**Which architecture may train which pair** is decided by
`TRAINABLE_ADAPTER_PAIRS` in `backend/core/adapters/capability.py` and by
nothing else. It is a *different table* from the generation one
(`ENABLED_ADAPTER_PAIRS`) in the same file: an architecture can load and
generate an algebra it cannot train. The per-architecture summary is the table
in `docs/guides/MODEL_FACTS.md` ("Adapter families per architecture"); the
subsystem's durable note is `docs/guides/LYCORIS_ADAPTER_DESIGN.md`.

`adapter_config` options, per algebra (`FRESH_BRANCH_OPTIONS` in
`core/adapters/layers.py`); an option the algebra does not have is refused by
name rather than ignored:

- `lora`: none.
- `loha`: `use_scalar` only — and a `use_scalar` layer **cannot be trained**
  (`validate_adapter_options` refuses it: the exporter folds `scalar` into the
  first factor and every reader forces `scalar := 1`, so resuming the saved
  file would rebuild a different layer). So LoHa has no usable option today.
- `lokr`: `factor` (Kronecker factorization applied to both dimensions,
  `-1` = auto) and `decompose_both` (stores `w1` low-rank as well, and requires
  `lora_rank > 0`); plus the same refused `use_scalar`.

**Refusals.** `_assert_adapter_algebra_contract` (`train_runner.py`) checks all
of these from the config, before the checkpoint loads;
`require_trainable_algebra` in `lora_trainer.py` is the backstop for a caller
that skipped the preflight. An ordinary-LoRA run is unaffected by all but the
first.

- An `adapter_config` key the algorithm does not accept — including a leftover
  `factor` on an ordinary LoRA run, which is validated even though nothing
  would read it.
- A non-ordinary algebra with `network.type != "lora"`: ReLoRA takes ordinary
  LoRA only, because its merge / reinitialize and optimizer reset are not
  defined for a Hadamard or Kronecker factorization, nor for a magnitude vector
  whose meaning depends on the base it was merged into.
- A pair the detected architecture's training row does not carry
  (`capability.require(..., AXIS_TRAINING)`), with that architecture's own
  reason.
- `blocks_to_swap > 0` with any non-ordinary algebra: the block offloader
  selects modules whose class name ends in `Linear`, and a LyCORIS branch's
  factors (and `dora_scale`) are bare parameters, so the branch is invisible to
  the swap. Set `blocks_to_swap: 0`, or train an ordinary LoRA.
- `weight_decompose` together with `fp8_base_dtype`: a decomposed branch reads
  the base weight's direction and norm every forward, and that setting
  quantizes the base before the adapter is injected.

#### `generate_full_finetune_config`

```python
@staticmethod
def generate_full_finetune_config(
    p: Optional[Dict[str, Any]] = None,
    *,
    run_name: str,
    base_model_path: str,
    output_dir: str,
    dataset_path: str = "",
    dataset_configs: Optional[List[Dict[str, Any]]] = None,
    sample_prompts: Optional[list] = None,
    caption_processing: Optional[Dict[str, Any]] = None,
    **legacy_kwargs: Any,
) -> str
```

Generate full fine-tuning configuration YAML (`network.type: full_finetune`).
Same call shape and same `legacy_kwargs` merge as `generate_lora_config`.
Differences in what it writes:

- `learning_rate` default `1e-6` (vs `1e-4` for LoRA)
- `train_text_encoder` default from
  `param_defaults.resolve_full_finetune_train_text_encoder`
- `max_step_saves_to_keep` default 3
- `noise_process` default `"add_noise"`, `strict_validation` default `True`
- Component LRs emitted only if not None (LoRA always emits them)
- Bucketing emitted only if `enable_bucketing`

**Returns**:
- `str`: YAML configuration string

**Example**:
```python
yaml_config = TrainingConfigGenerator.generate_full_finetune_config(
    {
        "epochs": 5,
        "batch_size": 2,
        "learning_rate": 1e-6,
        "train_unet": True,
        "train_text_encoder": True,
    },
    run_name="full_finetune_001",
    base_model_path="models/sdxl_base.safetensors",
    output_dir="training/full_001",
    dataset_configs=[{"path": "datasets/my_dataset"}],
)
```

#### `generate_relora_config`, `generate_controlnet_config`, `generate_vae_config`

Same call shape as `generate_lora_config` (`p` positional, the same seven
keyword-only arguments, the same `legacy_kwargs` merge). `generate_relora_config`
builds the LoRA config and rewrites `network.type` to `relora`;
`generate_controlnet_config` hardcodes ControlNet-only training;
`generate_vae_config` writes `network.type: vae_decoder` with no `sample`
section — see `docs/guides/VAE_TRAINING.md` and each method's docstring in
`training_config.py`.

#### `save_config`

```python
@staticmethod
def save_config(config_yaml: str, output_path: str) -> None
```

Save YAML configuration to file.

**Parameters**:
- `config_yaml` (str): YAML configuration string
- `output_path` (str): Path to save the config file

**Example**:
```python
TrainingConfigGenerator.save_config(yaml_config, "training/config.yaml")
```

---

## Utility Functions

**File**: `training_utils.py`

### `get_training_base_dir`

```python
def get_training_base_dir() -> str
```

Get the base training directory from user settings.

**Returns**:
- `str`: Base training directory path (default: "training")

**Behavior**:
- Queries `UserSettings.training_dir` from database
- Falls back to `"training"` if not configured

**Example**:
```python
from core.training.training_utils import get_training_base_dir

training_dir = get_training_base_dir()
# Returns: "training" or user-configured path
```

---

## BaseTrainer

**File**: `base_trainer.py`

Abstract base class for all trainers (LoRA, Full Parameter).

### Constructor

`BaseTrainer.__init__` takes **45 keyword parameters** (plus `self`); only
`model_path` and `output_dir` are required. They are not reproduced here — read
the signature in `base_trainer.py` (`def __init__` on `BaseTrainer`), which
carries the per-argument docstring, and
`backend/core/training/TRAINING_PARAMS_GUIDE.md` for what the YAML keys behind
them mean. Defaults visible to the API come from
`backend/api/param_defaults.py` (`TRAINING_DEFAULTS`), the single source of
truth for them; the literal defaults in this signature are the in-process
fallbacks and are not restated in this document.

Grouped by concern:

| Concern | Parameters |
|---|---|
| Model and run identity | `model_path`, `output_dir`, `run_name`, `run_id` (DB run id for metrics logging), `resume_from_checkpoint`, `train_config` |
| Precision | `weight_dtype`, `training_dtype`, `output_dtype`, `vae_dtype`, `mixed_precision` |
| Learning rates | `learning_rate`, `unet_lr`, `text_encoder_lr`, `text_encoder_1_lr`, `text_encoder_2_lr`, `image_encoder_lr` (each falls back to `learning_rate`, and the numbered text-encoder rates fall back to `text_encoder_lr` first) |
| Optimizer options | `num_optimizer_groups`, `optimizer_cautious`, `optimizer_beta1`, `optimizer_beta2`, `optimizer_epsilon`, `optimizer_weight_decay`, `optimizer_schedule_free`, `optimizer_warmup_steps`, `optimizer_schedule_free_r`, `optimizer_schedule_free_weight_lr_power`, `optimizer_use_radam`, `optimizer_stochastic_rounding`. There is no `optimizer_is_paged`: paging is selected by the optimizer type name (`paged_adamw`, `paged_adamw8bit`, `paged_lion8bit`). The Schedule-Free and stochastic-rounding options apply to the ring-buffer optimizers only (`optimizers/RINGBUFFER_OPTIMIZERS.md`) |
| Memory | `blocks_to_swap`, `use_pinned_memory` (`backend/core/memory_management/BLOCK_SWAP.md`), `activation_dispatch_enable`, `activation_dispatch_margin_gb`, `activation_dispatch_seed_coef`, `activation_dispatch_residual_frac`, `activation_dispatch_threshold_mb` |
| Attention | `attention_backend`, `attention_impl`, `use_flash_attention` (deprecated compat boolean; re-derived as `attention_backend != 'native'` when a backend is set) |
| Loss | `min_snr_gamma`, `reconstruction_loss_weight` |
| Prompt chunking (SD/SDXL long prompts) | `prompt_chunking_mode`, `max_prompt_chunks` |
| Device and debug | `device`, `debug_vram` |

Everything else an architecture needs is read out of the `train_config` dict
stored as `self.config`, not from a dedicated parameter. For example:

- `audio_loss_weight` (float, via `train_config`): MiniMax-H3 only — weight of the
  audio half of its joint objective, `loss = video_mean + audio_loss_weight *
  audio_mean` with each modality's velocity MSE averaged over tokens, channels and
  samples before weighting (default: 1.0, from `TRAINING_DEFAULTS`). `0.0` trains on
  the video half only. Every other architecture leaves it unread.

**Video datasets and `latent_encoding_mode`**: a cached video-clip latent is keyed by
its WINDOW (`compute_clip_hash`), so `pre_encoded_cache` encodes and then reuses ONE
fixed (centred) window per video for the whole run — it gives no temporal augmentation.
`swap_onthefly` and `onthefly_gpu` sample a fresh random window each time a clip is
encoded. This is inherent to addressing a disk cache by window, not a limitation of a
particular architecture.

**Important Notes**:
- This is an **abstract class** - use `LoRATrainer` or `FullParameterTrainer` instead
- Detects the model type and dispatches to the matching architecture handler
  (`backend/core/training/arch/__init__.py`, `ARCH_REGISTRY`)
- Loads model components separately (VAE, U-Net/Transformer, Text Encoders, Scheduler)

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model_path` | str | Path to base model |
| `output_dir` | Path | Output directory |
| `run_name` | str | Training run name |
| `learning_rate` | float | Base learning rate |
| `device` | torch.device | Training device |
| `unet_lr` | float | U-Net learning rate |
| `text_encoder_lr` | float | Text encoder learning rate |
| `weight_dtype` | torch.dtype | Model weight dtype |
| `training_dtype` | torch.dtype | Training dtype |
| `output_dtype` | torch.dtype | Output dtype |
| `vae_dtype` | torch.dtype | VAE dtype |
| `mixed_precision` | bool | Mixed precision enabled |
| `debug_vram` | bool | VRAM debugging enabled |
| `use_flash_attention` | bool | Flash Attention enabled |
| `min_snr_gamma` | float | Min-SNR gamma value |
| `audio_loss_weight` | float | MiniMax-H3 joint video+audio objective weight |
| `is_zimage` | bool | True if Z-Image model |
| `is_minimax_h3` | bool | True if MiniMax-H3 model (LoRA only; full FT is refused) |
| `is_sdxl` | bool | True if SDXL model |
| `vae` | AutoencoderKL | VAE model |
| `unet` | UNet2DConditionModel \| None | U-Net (SD1.5/SDXL only) |
| `transformer` | BatchedZImageWrapper \| None | Transformer (Z-Image only) |
| `text_encoder` | CLIPTextModel | Text encoder (CLIP-L) |
| `text_encoder_2` | CLIPTextModelWithProjection \| None | Text encoder 2 (SDXL only) |
| `tokenizer` | CLIPTokenizer | Tokenizer |
| `tokenizer_2` | CLIPTokenizer \| None | Tokenizer 2 (SDXL only) |
| `noise_scheduler` | DDPMScheduler | Noise scheduler |
| `writer` | SummaryWriter | TensorBoard writer |

### Abstract Methods

Subclasses must implement:

```python
@abstractmethod
def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
    """Setup trainable parameters; returns optimizer parameter groups."""

@abstractmethod
def save_checkpoint(self, step: int, epoch: int):
    """Save checkpoint (LoRA weights, full model, etc.)"""

@abstractmethod
def load_checkpoint(self, checkpoint_path: str) -> int:
    """Load a checkpoint; returns the step to resume from."""
```

### Public Methods

#### `train`

Main training loop. It takes **64 parameters** (plus `self`); only `datasets` is
required. As with the constructor, read the signature in `base_trainer.py`
(`def train` on `BaseTrainer`) rather than a copy here. All 25 sampling
parameters default from `_TRAINING_DEFAULTS`, i.e.
`TRAINING_DEFAULTS` in `backend/api/param_defaults.py`.

Grouped by concern:

| Concern | Parameters |
|---|---|
| Data and length | `datasets` (list of dataset objects from the database), `num_epochs`, `total_steps` (when set, overrides `num_epochs`), `batch_size` |
| Checkpointing | `save_every_n_steps`, `max_step_saves_to_keep`, `max_optimizer_saves_to_keep`, `resume_from_checkpoint` |
| Sampling during training | `sample_every_n_steps`, `sample_prompts` (`Optional[List[Dict[str, str]]]` — entries are `{positive, negative, condition_image_path?}`), `sample_guidance_scale`, `sample_steps`, `sample_width`, `sample_height`, `sample_seed`, `sample_sampler`, `sample_schedule_type`, the `sample_cfg_schedule_*` and `sample_dynamic_threshold_*` group, the `sample_nag_*` group, and `sensenova_sample_timestep_shift` / `sensenova_sample_img_cfg_scale` / `sensenova_sample_cfg_norm` |
| Optimization | `optimizer_type`, `lr_scheduler_type`, `gradient_accumulation_steps`, `max_grad_norm`, `timestep_sampling_config`, `priority_training` |
| Bucketing | `enable_bucketing`, `base_resolutions`, `bucket_strategy` (`"resize"`, `"crop"`, `"random_crop"`), `multi_resolution_mode` (`"max"`, `"random"`) |
| Encoding residency | `text_encoding_mode`, `text_encoding_swap_interval`, `text_encoding_prefetch_depth`, `latent_encoding_mode`, `latent_encoding_swap_interval`, `force_recache` |
| Reference / vision encoder | `use_reference_images`, `train_vision_encoder`, `vision_encoder_path`, `vision_encoder_lr`, `gradient_routing_ve` |
| Callbacks and instrumentation | `progress_callback`, `update_total_steps_callback`, `run_id`, `debug_latents`, `debug_latents_every`, `param_tracking`, `param_tracking_interval` |
| Accepted but unused | `multi_noise_timesteps`, `multi_noise_mode`, `trajectory_blend_alpha` — multi-noise timesteps are disabled; these are kept for call-site compatibility |

`0` means "never" for every optional periodic action. `gradient_accumulation_steps`
is not optional, so `0` folds to `1` rather than disabling the optimizer step.

`text_encoding_mode` (Z-Image and other swap-capable architectures) accepts
`"swap_onthefly"` (swap text encoder and transformer, encode on the fly),
`"pre_encoded_cache"` (disk cache), and `"onthefly_gpu"` (encode on GPU without
a cache).

**Example**:
```python
trainer = LoRATrainer(
    model_path="models/sdxl_base.safetensors",
    output_dir="training/lora_001",
    learning_rate=1e-4
)

trainer.train(
    datasets=training_datasets,
    num_epochs=10,
    batch_size=4,
    enable_bucketing=True,
    base_resolutions=[1024],
    optimizer_type="adamw8bit"
)
```

---

## Common Usage Patterns

### Pattern 1: Basic LoRA Training with Latent Caching

```python
from core.training.lora_trainer import LoRATrainer
from core.training.latent_cache import LatentCache
from database import get_db
from database.models import Dataset

# Load datasets from database
db = next(get_db())
datasets = db.query(Dataset).filter(Dataset.id.in_([1, 2, 3])).all()

# Create trainer
trainer = LoRATrainer(
    model_path="models/sdxl_base.safetensors",
    output_dir="training/lora_001",
    learning_rate=1e-4,
    lora_rank=16,
    lora_alpha=16
)

# Train (latent caching is handled automatically)
trainer.train(
    datasets=datasets,
    num_epochs=10,
    batch_size=4,
    enable_bucketing=True,
    base_resolutions=[1024],
    optimizer_type="adamw8bit"
)
```

### Pattern 2: Manual Latent Cache Validation

```python
from core.training.latent_cache import LatentCache

cache = LatentCache(dataset_unique_id="a1b2c3d4-...")

# Check if cache is valid
dtype_str = "bfloat16"
if cache.is_valid("models/sdxl_base.safetensors", "sdxl", dtype_str):
    print("Cache is valid, reusing...")
else:
    print("Cache invalid, regenerating...")

    # Validate cache format
    expected_channels = 4  # SD/SDXL
    if not cache.validate_cache_format(expected_channels=expected_channels):
        print("Cache format validation failed!")
```

### Pattern 3: Bucketing with Multiple Resolutions

```python
from core.training.bucketing import BucketManager

bucket_manager = BucketManager(
    base_resolutions=[512, 768, 1024],
    divisibility=8,
    strategy="resize",
    multi_resolution_mode="max"
)

# Assign images to buckets
for image_path, width, height, caption in image_data:
    bucket, image_info = bucket_manager.assign_image_to_bucket(
        image_path=image_path,
        width=width,
        height=height,
        caption=caption,
        dataset_unique_id=dataset.unique_id
    )

# Build batches
batches = bucket_manager.build_batch_indices(batch_size=4)
```

### Pattern 4: Tag Group-Based Caption Processing

```python
from core.training.tag_group_utils import get_tag_group_manager

tag_manager = get_tag_group_manager()

# Categorize tags
tags = ["hakurei_reimu", "long hair", "masterpiece", "touhou"]
categorized = tag_manager.categorize_tags(tags)

# Shuffle General tags while keeping first token and excluding person count
tokens = ["1girl", "hakurei_reimu", "long hair", "red bow", "touhou"]
shuffled = tag_manager.shuffle_by_groups(
    tokens=tokens,
    groups_to_shuffle=["General"],
    keep_first_n=1,
    exclude_person_count=True
)
```

---

## Error Handling

### Common Errors and Solutions

#### Error: `LatentCache` has no attribute `exists()`

**Problem**: Calling non-existent method `cache.exists()`.

**Solution**: Use `cache.is_valid()` instead, which checks both existence and validity.

```python
# ❌ Wrong
if cache.exists():
    ...

# ✅ Correct
dtype_str = str(training_dtype).replace('torch.', '')
if cache.is_valid(model_path, model_type, dtype_str):
    ...
```

#### Error: `BucketManager` unexpected keyword argument `min_resolution`

**Problem**: Passing `min_resolution`, `max_resolution`, `step_resolution` instead of `base_resolutions`.

**Solution**: Pass `base_resolutions` as a list.

```python
# ❌ Wrong
bucket_manager = BucketManager(
    min_resolution=512,
    max_resolution=1024,
    step_resolution=256
)

# ✅ Correct
bucket_manager = BucketManager(
    base_resolutions=[512, 768, 1024]
)
```

---

## Version History

- **2026-09-04**: Re-verified `BaseTrainer` against the code
  - `BaseTrainer.__init__` (45 parameters) and `BaseTrainer.train` (64) are now
    documented as grouped summaries pointing at the signatures, not as copies
  - Removed `sample_prompt: str`, which does not exist; the real parameter is
    `sample_prompts: Optional[List[Dict[str, str]]]`
  - Corrected the abstract-method block (`setup_trainable_parameters` returns
    parameter groups; `load_checkpoint` was missing)
  - Added `VideoBucketManager`
- **2025-12-15**: Updated to match current implementation
  - Added `LatentCache.has_latent()` method documentation
  - Removed obsolete error handling for `has_latent()` (method now exists)
  - Confirmed `BaseTrainer.train()` signature matches implementation
  - Bucketing is now automatically applied before latent cache generation
- **2025-12-15**: Initial comprehensive API reference created
  - Covers all classes, methods, and functions in `backend/core/training/`
  - Includes usage patterns and error handling guide

---

## Related Documentation

- [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md) - Model architecture specifications (SD1.5, SDXL, Z-Image)
- [../../../AGENTS.md](../../../AGENTS.md) - Task router and development guidelines
