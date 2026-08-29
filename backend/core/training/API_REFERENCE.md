# SushiUI Training Framework - API Reference

**Last Updated**: 2025-12-15

This document provides a comprehensive API reference for all components, classes, methods, and functions in the `backend/core/training` module.

---

## Table of Contents

1. [LatentCache](#latentcache)
2. [BucketManager](#bucketmanager)
3. [TagGroupManager](#taggroupmanager)
4. [TrainingConfigGenerator](#trainingconfiggenerator)
5. [Utility Functions](#utility-functions)
6. [BaseTrainer](#basetrainer)

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
LatentCache(dataset_unique_id: str, base_cache_dir: str = None)
```

**Parameters**:
- `dataset_unique_id` (str): Dataset unique ID (UUID)
- `base_cache_dir` (str, optional): Base directory for cache. Defaults to user settings (`cache/datasets`)

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
    multi_resolution_mode: Literal["max", "random"] = "max"
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
    dataset_unique_id: Optional[str] = None
) -> Tuple[BucketResolution, Dict]
```

Assign an image to the best bucket.

**Parameters**:
- `image_path` (str): Path to image file
- `width` (int): Image width
- `height` (int): Image height
- `caption` (str): Image caption (default: "")
- `target_resolution` (int, optional): Specific resolution to use (or None for auto)
- `dataset_unique_id` (str, optional): Dataset UUID (for cache management)

**Returns**:
- `Tuple[BucketResolution, Dict]`: (bucket_resolution, image_info)

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
def get_items_by_bucket(self) -> Dict[BucketResolution, List[Dict]]
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

## TagGroupManager

**File**: `tag_group_utils.py`

Manages tag groups for caption processing (categorization, shuffle, dropout).

### Constructor

```python
TagGroupManager(tag_group_dir: str = "taglist")
```

**Parameters**:
- `tag_group_dir` (str): Directory containing tag group JSON files (default: "taglist")

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
def get_tag_group_manager(tag_group_dir: str = "taglist") -> TagGroupManager
```

Get or create tag group manager (cached globally).

**Parameters**:
- `tag_group_dir` (str): Directory containing tag group JSON files

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
    run_name: str,
    dataset_path: str,
    base_model_path: str,
    output_dir: str,
    dataset_configs: Optional[List[Dict[str, Any]]] = None,
    total_steps: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    lr_scheduler: str = "constant",
    optimizer: str = "adamw8bit",
    lora_rank: int = 16,
    lora_alpha: int = 16,
    save_every: int = 100,
    save_every_unit: str = "steps",
    sample_every: int = 100,
    sample_prompts: Optional[list] = None,
    debug_latents: bool = False,
    debug_latents_every: int = 50,
    enable_bucketing: bool = False,
    base_resolutions: Optional[list] = None,
    bucket_strategy: str = "resize",
    multi_resolution_mode: str = "max",
    train_unet: bool = True,
    train_text_encoder: bool = False,
    unet_lr: Optional[float] = None,
    text_encoder_lr: Optional[float] = None,
    text_encoder_1_lr: Optional[float] = None,
    text_encoder_2_lr: Optional[float] = None,
    cache_latents_to_disk: bool = False,
    weight_dtype: str = "fp16",
    training_dtype: str = "fp16",
    output_dtype: str = "fp32",
    vae_dtype: str = "fp16",
    mixed_precision: bool = True,
    use_flash_attention: bool = False,
    min_snr_gamma: float = 5.0,
    sample_width: int = 1024,
    sample_height: int = 1024,
    sample_steps: int = 28,
    sample_cfg_scale: float = 7.0,
    sample_sampler: str = "euler",
    sample_seed: int = 42,
    resume_from_checkpoint: Optional[str] = None,
    caption_processing: Optional[Dict[str, Any]] = None,
) -> str
```

Generate LoRA training configuration YAML.

**Key Parameters**:
- `run_name` (str): Training run identifier
- `dataset_path` (str): Path to dataset directory (deprecated, use `dataset_configs`)
- `base_model_path` (str): Path to base model
- `output_dir` (str): Output directory for checkpoints
- `dataset_configs` (List[Dict], optional): List of dataset configurations (new, multi-dataset support)
- `total_steps` (int, optional): Total training steps (mutually exclusive with `epochs`)
- `epochs` (int, optional): Number of epochs (mutually exclusive with `total_steps`)
- `lora_rank` (int): LoRA rank (default: 16)
- `lora_alpha` (int): LoRA alpha (default: 16)
- `optimizer` (str): Optimizer type (e.g., `"adamw8bit"`, `"adamw"`, `"sgd"`)

**Returns**:
- `str`: YAML configuration string

**Example**:
```python
from core.training.training_config import TrainingConfigGenerator

yaml_config = TrainingConfigGenerator.generate_lora_config(
    run_name="lora_training_001",
    dataset_path="datasets/my_dataset",
    base_model_path="models/sdxl_base.safetensors",
    output_dir="training/lora_001",
    epochs=10,
    batch_size=4,
    learning_rate=1e-4,
    lora_rank=16,
    enable_bucketing=True,
    base_resolutions=[1024]
)
```

#### `generate_full_finetune_config`

```python
@staticmethod
def generate_full_finetune_config(
    run_name: str,
    dataset_path: str,
    base_model_path: str,
    output_dir: str,
    dataset_configs: Optional[List[Dict[str, Any]]] = None,
    total_steps: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: int = 1,
    learning_rate: float = 1e-6,
    lr_scheduler: str = "constant",
    optimizer: str = "adamw8bit",
    save_every: int = 100,
    save_every_unit: str = "steps",
    sample_every: int = 100,
    sample_prompts: Optional[list] = None,
    debug_latents: bool = False,
    debug_latents_every: int = 50,
    enable_bucketing: bool = False,
    base_resolutions: Optional[List[int]] = None,
    bucket_strategy: str = "resize",
    multi_resolution_mode: str = "max",
    train_unet: bool = True,
    train_text_encoder: bool = True,
    unet_lr: Optional[float] = None,
    text_encoder_lr: Optional[float] = None,
    text_encoder_1_lr: Optional[float] = None,
    text_encoder_2_lr: Optional[float] = None,
    cache_latents_to_disk: bool = False,
    weight_dtype: str = "fp16",
    training_dtype: str = "fp16",
    output_dtype: str = "fp32",
    vae_dtype: str = "fp16",
    mixed_precision: bool = True,
    use_flash_attention: bool = False,
    min_snr_gamma: float = 5.0,
    sample_width: int = 1024,
    sample_height: int = 1024,
    sample_steps: int = 28,
    sample_cfg_scale: float = 7.0,
    sample_sampler: str = "euler",
    sample_seed: int = -1,
    resume_from_checkpoint: Optional[str] = None,
    caption_processing: Optional[dict] = None,
) -> str
```

Generate full fine-tuning configuration YAML (similar to `generate_lora_config` but with `network.type: full_finetune`).

**Returns**:
- `str`: YAML configuration string

**Example**:
```python
yaml_config = TrainingConfigGenerator.generate_full_finetune_config(
    run_name="full_finetune_001",
    dataset_path="datasets/my_dataset",
    base_model_path="models/sdxl_base.safetensors",
    output_dir="training/full_001",
    epochs=5,
    batch_size=2,
    learning_rate=1e-6,  # Lower LR for full fine-tune
    train_unet=True,
    train_text_encoder=True
)
```

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

```python
BaseTrainer(
    model_path: str,
    output_dir: str,
    run_name: str = None,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    unet_lr: Optional[float] = None,
    text_encoder_lr: Optional[float] = None,
    weight_dtype: str = "fp16",
    training_dtype: str = "fp16",
    output_dtype: str = "fp32",
    vae_dtype: str = "fp16",
    mixed_precision: bool = True,
    debug_vram: bool = False,
    use_flash_attention: bool = False,
    min_snr_gamma: float = 5.0,
)
```

**Parameters**:
- `model_path` (str): Path to base model (safetensors or diffusers directory)
- `output_dir` (str): Output directory for checkpoints
- `run_name` (str, optional): Training run identifier (defaults to output_dir name)
- `learning_rate` (float): Learning rate (default: 1e-4)
- `device` (str): Device to use ("cuda" or "cpu", default: "cuda")
- `unet_lr` (float, optional): U-Net learning rate (defaults to `learning_rate`)
- `text_encoder_lr` (float, optional): Text encoder learning rate (defaults to `learning_rate`)
- `weight_dtype` (str): Model weight dtype ("fp16", "bf16", "fp32", default: "fp16")
- `training_dtype` (str): Training/activation dtype (default: "fp16")
- `output_dtype` (str): Output latent dtype (default: "fp32")
- `vae_dtype` (str): VAE-specific dtype (default: "fp16")
- `mixed_precision` (bool): Enable autocast for mixed precision (default: True)
- `debug_vram` (bool): Enable VRAM debugging logs (default: False)
- `use_flash_attention` (bool): Enable Flash Attention (default: False)
- `min_snr_gamma` (float): Min-SNR gamma value for loss weighting (default: 5.0)
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
- Automatically detects model type (SD1.5, SDXL, Z-Image)
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
def setup_trainable_parameters(self) -> Tuple[torch.nn.Module, ...]:
    """Setup trainable parameters (LoRA modules, full model, etc.)"""
    pass

@abstractmethod
def save_checkpoint(self, step: int, epoch: int) -> str:
    """Save checkpoint (LoRA weights, full model, etc.)"""
    pass
```

### Public Methods

#### `train`

```python
def train(
    self,
    datasets: List[Any],
    num_epochs: int = 10,
    batch_size: int = 1,
    save_every_n_steps: int = 500,
    sample_every_n_steps: int = 500,
    sample_prompt: str = "a beautiful landscape",
    optimizer_type: str = "adamw",
    lr_scheduler_type: str = "constant",
    enable_bucketing: bool = True,
    base_resolutions: Optional[List[int]] = None,
    bucket_strategy: str = "resize",
    multi_resolution_mode: str = "max",
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    debug_latents: bool = False,
    debug_latents_every: int = 50,
    progress_callback: Optional[Callable] = None,
)
```

Main training loop.

**Parameters**:
- `datasets` (List[Any]): List of dataset objects (from database)
- `num_epochs` (int): Number of epochs (default: 10)
- `batch_size` (int): Batch size (default: 1)
- `save_every_n_steps` (int): Save checkpoint every N steps (default: 500)
- `sample_every_n_steps` (int): Generate sample every N steps (default: 500)
- `sample_prompt` (str): Prompt for sample generation (default: "a beautiful landscape")
- `optimizer_type` (str): Optimizer type ("adamw", "adamw8bit", "sgd", etc.)
- `lr_scheduler_type` (str): LR scheduler type ("constant", "cosine", "linear", etc.)
- `enable_bucketing` (bool): Enable aspect ratio bucketing (default: True)
- `base_resolutions` (List[int], optional): List of base resolutions for bucketing (e.g., `[1024]`)
- `bucket_strategy` (str): Bucketing strategy ("resize", "crop", "random_crop")
- `multi_resolution_mode` (str): Multi-resolution mode ("max", "random")
- `gradient_accumulation_steps` (int): Gradient accumulation steps (default: 1)
- `max_grad_norm` (float): Max gradient norm for clipping (default: 1.0)
- `debug_latents` (bool): Enable debug latent saving (default: False)
- `debug_latents_every` (int): Save debug latents every N steps (default: 50)
- `progress_callback` (Callable, optional): Progress callback function

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
