# Aesthetic Scorer

Lightweight tool for scoring predicted latent quality to prevent "overbaked" images in diffusion model training.

## Overview

This tool helps improve diffusion model training by:
1. Generating large datasets of predicted latents from training data
2. Manual scoring of latent quality (0=best, 1=worst)
3. Training a lightweight neural network to predict quality scores
4. Using the trained model as a regularization loss during training

## Architecture

- **Backend**: FastAPI (Python)
- **Frontend**: Next.js + React (TypeScript)
- **Database**: SQLite
- **Model**: Ultra-lightweight CNN (~50K parameters, <10MB VRAM overhead)

## Directory Structure

```
aesthetic_scorer/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── api/                 # API endpoints
│   ├── core/                # Core logic (generator, model, trainer)
│   ├── database/            # SQLAlchemy models
│   └── utils/               # Utilities
├── frontend/
│   └── src/                 # Next.js application
├── data/
│   ├── latents/             # Generated .pt files
│   └── images/              # VAE decoded images (cache)
├── models/                  # Trained aesthetic models
└── aesthetic_scorer.db      # SQLite database
```

## Usage

### 1. Generate Latent Dataset

```bash
cd subapps/aesthetic_scorer
python -m backend.main generate --dataset-id 1 --num-samples 1000
```

### 2. Score Latents (UI)

```bash
# Start backend
python -m backend.main serve --port 8001

# Start frontend (separate terminal)
cd frontend
npm run dev
```

Navigate to `http://localhost:3000` and score latents manually.

### 3. Train Aesthetic Model

```bash
python -m backend.main train --epochs 50 --batch-size 16
```

### 4. Integrate into SushiUI Training

Add to training config:
```yaml
aesthetic_loss_weight: 0.1
aesthetic_model_path: "subapps/aesthetic_scorer/models/aesthetic_v50.safetensors"
```

## Data Format

Generated .pt files (minimal mode):
```python
{
    'latents': torch.Tensor,          # [1, 16, H, W] Ground truth
    'predicted_latent': torch.Tensor, # [1, 16, H, W] Model prediction
    'timestep': float,                # 0.0-1.0
    'recon_loss': float,              # MSE(predicted_latent, latents)
    'caption': str,                   # Text prompt
    'scheduler_type': str,            # "FlowMatching"
}
```

## Model Architecture

```
Input: [B, 16, H, W] (predicted latent)
  ↓
Conv2d(16→32, stride=2) → ReLU
  ↓
Conv2d(32→64, stride=2) → ReLU
  ↓
Conv2d(64→128, stride=2) → ReLU
  ↓
AdaptiveAvgPool2d(1, 1)
  ↓
Linear(128→1) → Sigmoid
  ↓
Output: [B, 1] (score: 0=best, 1=worst)
```

**Parameters**: ~50K (~200KB)
**VRAM overhead**: <10MB

## Integration with SushiUI

The trained aesthetic model is frozen during main training and used only for loss calculation:

```python
total_loss = mse_loss + aesthetic_weight * aesthetic_loss
```

This acts as a regularization term to prevent overbaking.
