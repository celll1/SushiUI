# Aesthetic Scorer - Usage Guide

Complete guide for generating, scoring, and training the aesthetic quality model.

## Quick Start

### 0. Configure Storage Paths (Optional)

By default, data is saved to `subapps/aesthetic_scorer/data/`. To use a custom path (e.g., on a larger drive):

```bash
# Get current settings
curl http://localhost:8001/api/settings

# Update storage paths
curl -X PUT http://localhost:8001/api/settings \
  -H "Content-Type: application/json" \
  -d '{
    "latents_dir": "E:/aesthetic_scorer_data/latents",
    "images_dir": "E:/aesthetic_scorer_data/images",
    "models_dir": "E:/aesthetic_scorer_data/models"
  }'
```

**Note**: Directories will be created automatically if they don't exist.

### 1. Generate Latent Dataset

First, generate predicted latents from your SushiUI training dataset:

```bash
cd subapps/aesthetic_scorer

# Start backend server
python -m backend.main --host 127.0.0.1 --port 8001

# In another terminal, generate latents via API
curl -X POST http://localhost:8001/api/generate_latents \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": 1,
    "model_path": "M:/models/zimage_v1.safetensors",
    "num_samples": 1000,
    "timestep_range": [0.0, 1.0],
    "shuffle": true,
    "output_dir": "E:/aesthetic_scorer_data/latents"
  }'

# Or use default path from settings (omit output_dir)
```

**Response**:
```json
{
  "num_generated": 1000,
  "total_size_gb": 1.95,
  "dataset_name": "my_training_dataset"
}
```

### 2. Decode Latents to Images

Decode latents to PNG images for visual inspection:

```bash
# Get list of record IDs
curl http://localhost:8001/api/latents?limit=100 | jq '.records[] | .id'

# Decode a batch
curl -X POST http://localhost:8001/api/decode_latents \
  -H "Content-Type: application/json" \
  -d '{
    "record_ids": [1, 2, 3, 4, 5],
    "vae_path": "M:/models/zimage_v1.safetensors"
  }'
```

### 3. Score Latents (UI)

Start the frontend to manually score latents:

```bash
# Backend (if not already running)
python -m backend.main --port 8001

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:3001` and score latents:

**Keyboard Shortcuts**:
- `1-9`: Quick score (0.1-0.9)
- `0`: Worst score (1.0)
- `Space`: Submit and next
- `←/→`: Navigate

**Scoring Guidelines**:
- **0.0-0.3**: Excellent (very close to ground truth)
- **0.3-0.5**: Good (minor artifacts)
- **0.5-0.7**: Acceptable (noticeable degradation)
- **0.7-0.9**: Poor (overbaked/artifacts)
- **0.9-1.0**: Very poor (unusable)

### 4. Train Aesthetic Model

Once you have scored enough samples (recommended: 500+), train the model:

```bash
curl -X POST http://localhost:8001/api/train_model \
  -H "Content-Type: application/json" \
  -d '{
    "architecture": "LatentCNN",
    "learning_rate": 0.0001,
    "num_epochs": 50,
    "batch_size": 16,
    "val_split": 0.1,
    "model_name": "aesthetic_v1"
  }'
```

**Response**:
```json
{
  "model_id": 1,
  "model_path": "subapps/aesthetic_scorer/models/aesthetic_v1_best.safetensors",
  "train_loss": 0.0234,
  "val_loss": 0.0289
}
```

### 5. Integrate into SushiUI Training

Update your training configuration to include aesthetic loss:

```yaml
# training_config.yaml
aesthetic_loss_weight: 0.1  # Regularization weight (0.05-0.2 recommended)
aesthetic_model_path: "subapps/aesthetic_scorer/models/aesthetic_v1_best.safetensors"
aesthetic_architecture: "LatentCNN"
```

Or via Python API:

```python
from backend.core.training.full_parameter_trainer import FullParameterTrainer

trainer = FullParameterTrainer(
    model_path="M:/models/zimage_v1.safetensors",
    output_dir="training/my_run",
    # ... other parameters ...
    aesthetic_loss_weight=0.1,
    aesthetic_model_path="subapps/aesthetic_scorer/models/aesthetic_v1_best.safetensors",
    aesthetic_architecture="LatentCNN",
)
```

---

## API Reference

### Settings

**GET** `/api/settings`

Get current application settings.

**PUT** `/api/settings`

```json
{
  "latents_dir": "E:/aesthetic_scorer_data/latents",
  "images_dir": "E:/aesthetic_scorer_data/images",
  "models_dir": "E:/aesthetic_scorer_data/models",
  "default_timestep_range_min": 0.0,
  "default_timestep_range_max": 1.0
}
```

### Generate Latents

**POST** `/api/generate_latents`

```json
{
  "dataset_id": 1,
  "model_path": "path/to/model.safetensors",
  "num_samples": 1000,
  "timestep_range": [0.0, 1.0],
  "shuffle": true,
  "output_dir": "E:/custom/path/latents"
}
```

**Note**: `output_dir` is optional. If omitted, uses path from settings.

### Get Latent Records

**GET** `/api/latents?skip=0&limit=100&unscored_only=true`

### Score Latent

**POST** `/api/latents/{record_id}/score`

```json
{
  "score": 0.35
}
```

### Get Statistics

**GET** `/api/latents/stats`

```json
{
  "total": 1000,
  "scored": 250,
  "unscored": 750,
  "scored_percentage": 25.0
}
```

### Train Model

**POST** `/api/train_model`

```json
{
  "architecture": "LatentCNN",
  "learning_rate": 0.0001,
  "num_epochs": 50,
  "batch_size": 16,
  "val_split": 0.1,
  "model_name": "aesthetic_v1"
}
```

### Get Models

**GET** `/api/models`

### Activate Model

**POST** `/api/models/{model_id}/activate`

---

## Advanced Usage

### Batch Generation with Script

Create a Python script for batch generation:

```python
# generate_dataset.py
import sys
sys.path.append("backend")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import DatasetBase, Dataset
from subapps.aesthetic_scorer.backend.core.latent_generator import LatentGenerator
from pathlib import Path

# Connect to SushiUI datasets.db
engine = create_engine("sqlite:///backend/datasets.db")
Session = sessionmaker(bind=engine)
session = Session()

# Initialize generator
generator = LatentGenerator(
    model_path="M:/models/zimage_v1.safetensors",
    device="cuda",
    weight_dtype="fp16",
)

# Generate latents
records = generator.generate_latents_from_dataset(
    sushiui_db_session=session,
    dataset_id=1,
    num_samples=5000,
    timestep_range=(0.0, 1.0),
    output_dir=Path("subapps/aesthetic_scorer/data/latents"),
    save_mode="minimal",
    shuffle=True,
)

print(f"Generated {len(records)} samples")
```

Run:
```bash
cd subapps/aesthetic_scorer
python generate_dataset.py
```

### Custom Model Architecture

Use LatentTransformer for higher accuracy (at cost of more parameters):

```python
{
  "architecture": "LatentTransformer",
  "learning_rate": 0.0001,
  "num_epochs": 100,
  "batch_size": 8,
  "val_split": 0.1,
  "model_name": "aesthetic_transformer_v1"
}
```

**Comparison**:
- **LatentCNN**: ~50K parameters, <5MB VRAM, fast inference
- **LatentTransformer**: ~500K parameters, ~20MB VRAM, higher accuracy

---

## Troubleshooting

### "Not enough scored samples"

You need at least 10 scored samples to train. Recommended: 500+ for good generalization.

### Decoded images not showing

Run decode latents API:
```bash
curl -X POST http://localhost:8001/api/decode_latents \
  -H "Content-Type: application/json" \
  -d '{"record_ids": [1,2,3], "vae_path": "path/to/model.safetensors"}'
```

### High validation loss

- Increase number of scored samples
- Check scoring consistency (are similar latents scored similarly?)
- Try higher learning rate or more epochs
- Use LatentTransformer architecture

### VRAM issues during generation

Reduce batch size in `latent_generator.py` or use CPU:
```python
generator = LatentGenerator(
    model_path="...",
    device="cpu",  # Use CPU instead of CUDA
)
```

---

## File Structure Reference

```
aesthetic_scorer/
├── backend/
│   ├── main.py                     # FastAPI app entry point
│   ├── api/routes.py               # API endpoints
│   ├── core/
│   │   ├── latent_generator.py    # Generate latents from datasets
│   │   ├── aesthetic_model.py     # LatentCNN / LatentTransformer
│   │   └── aesthetic_trainer.py   # Model training
│   └── database/
│       └── models.py               # SQLAlchemy models
├── frontend/
│   └── src/app/page.tsx           # Scoring UI
├── data/
│   ├── latents/                   # Generated .pt files
│   └── images/                    # Decoded PNG images
├── models/                        # Trained aesthetic models
└── aesthetic_scorer.db            # SQLite database
```

---

## Tips for Best Results

1. **Diverse Timesteps**: Generate latents across full range (0.0-1.0)
2. **Consistent Scoring**: Review your scores periodically to maintain consistency
3. **Sufficient Samples**: More scored samples = better model generalization
4. **Regularization Weight**: Start with 0.1, adjust based on results (0.05-0.2 typical)
5. **Iterative Training**: Train multiple versions as you score more samples

---

## Integration Example

Full example of training with aesthetic loss:

```python
from backend.core.training.full_parameter_trainer import FullParameterTrainer

trainer = FullParameterTrainer(
    model_path="M:/models/zimage_v1.safetensors",
    output_dir="training/my_aesthetic_run",
    run_name="aesthetic_test",
    learning_rate=1e-5,

    # Aesthetic loss (NEW)
    aesthetic_loss_weight=0.1,
    aesthetic_model_path="subapps/aesthetic_scorer/models/aesthetic_v1_best.safetensors",
    aesthetic_architecture="LatentCNN",

    # Standard parameters
    weight_dtype="fp16",
    training_dtype="fp16",
    mixed_precision=True,
    min_snr_gamma=5.0,
)

# Train with dataset
trainer.train(
    dataset_ids=[1],
    num_epochs=10,
    batch_size=2,
    # ...
)
```

**Expected output**:
```
[AestheticLoss] Loaded LatentCNN from subapps/aesthetic_scorer/models/aesthetic_v1_best.safetensors
[AestheticLoss] Parameters: 52,129 (~203.6 KB)
[AestheticLoss] Model frozen (no gradient)
[Trainer] Aesthetic loss enabled (weight=0.1)

[Epoch 1] mse_loss=0.0234, aesthetic_loss=0.3512, total_loss=0.0585
...
```
