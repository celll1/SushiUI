# Implementation Plan: Original Diffusion Model Architecture

## Overview

Complete ground-up implementation of a new diffusion model architecture combining:
- **SigLIP-2** (Text + Image Encoders): Multi-modal conditioning without token limits
- **DiC-inspired U-Net**: Sparse skip connections + improved Conv blocks
- **RoPE positional encoding**: From Z-Image
- **FLUX VAE**: 16-channel latents (vs SDXL's 4-channel)

**Target Model Size**: 2.5 - 4B parameters

---

## 1. Component Specifications

### 1.1 Text Encoder: SigLIP-2 SO-400M

**Model**: `google/siglip2-so400m-patch16-naflex`

**Specifications**:
- Hidden size: 1152
- Layers: 27
- Attention heads: 16
- Intermediate size: 4304
- Max position embeddings: 64
- Vocab size: 256,000
- **No token limit** (vs CLIP's 77 tokens)

**Output**:
- Text embeddings: `[batch, seq_len, 1152]`
- Pooled output: `[batch, 1152]`

**Usage**:
- Always present in all generation tasks (t2i, i2i, inpaint)
- No chunking needed (unlimited tokens)

---

### 1.2 Image Encoder: SigLIP-2 SO-400M (Vision)

**Same model as Text Encoder** (dual encoder architecture)

**Specifications**:
- Hidden size: 1152
- Layers: 27
- Attention heads: 16
- Intermediate size: 4304
- Patch size: 16
- Projection dim: Shared with text

**Output**:
- Image embeddings: `[batch, num_patches, 1152]`
- Pooled output: `[batch, 1152]`

**Usage**:
- **Optional**: Present in i2i, controlnet, IP-Adapter tasks
- **Not present**: In pure t2i tasks
- **Variable count**: Support 0, 1, or multiple input images

**U-Net Integration Strategy**:
```python
# Handle variable image inputs
if image_embeddings is None:
    # T2I mode: Use learned null embeddings
    image_cond = null_image_embedding.expand(batch, -1, -1)
elif len(image_embeddings) == 1:
    # Single image mode
    image_cond = image_embeddings[0]
else:
    # Multi-image mode: Average or attention pooling
    image_cond = attention_pool(image_embeddings)
```

---

### 1.3 U-Net: DiC-Inspired SDXL-Based Architecture

#### Base Structure: SDXL U-Net

**Down Blocks**:
- Down 0: 320 channels, 2 ResNet blocks, no attention
- Down 1: 640 channels, 2 ResNet blocks, 2 Transformer blocks
- Down 2: 1280 channels, 2 ResNet blocks, 10 Transformer blocks

**Mid Block**:
- 1280 channels, 1 ResNet, 10 Transformer blocks, 1 ResNet

**Up Blocks**:
- Up 0: 1280 channels, 2 ResNet blocks, 10 Transformer blocks
- Up 1: 640 channels, 2 ResNet blocks, 2 Transformer blocks
- Up 2: 320 channels, 2 ResNet blocks, no attention

#### DiC Improvements

**1. Sparse Skip Connections** (from DiC paper):
```python
# Instead of skip at every block:
# SDXL: skip every 2-3 blocks
# DiC: skip every N blocks (strided skip)

# Implementation:
skip_stride = 3  # Skip every 3 blocks
for i, (down_block, up_block) in enumerate(zip(down_blocks, up_blocks)):
    if i % skip_stride == 0:
        # Apply skip connection
        up_input = torch.cat([up_input, down_output], dim=1)
    else:
        # No skip (sparse)
        pass
```

**Benefits**:
- Reduces computational overhead (~20-30%)
- Improves scalability
- Better gradient flow

**2. Improved Conv Blocks** (from DiC paper):
```python
class ImprovedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Simple 3x3 stride-1 conv (hardware optimized)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # GELU instead of SiLU
        self.act = nn.GELU()
        # Stage-specific embeddings
        self.emb_proj = nn.Linear(time_emb_dim, out_channels)
        # Conditional gating
        self.gate = nn.Linear(out_channels, out_channels)

    def forward(self, x, time_emb, cond):
        # Conv
        h = self.conv(x)
        # Add time embedding
        h = h + self.emb_proj(time_emb)[:, :, None, None]
        # Activation
        h = self.act(h)
        # Conditional gating
        gate = torch.sigmoid(self.gate(cond))
        h = h * gate[:, :, None, None]
        return h
```

**3. RoPE Positional Encoding** (from Z-Image):
```python
# Replace learned positional embeddings with RoPE
class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.theta = theta

    def forward(self, q, k, v, height, width):
        # Apply RoPE to Q and K
        q = apply_rotary_emb(q, height, width, self.theta)
        k = apply_rotary_emb(k, height, width, self.theta)
        # Standard attention
        return scaled_dot_product_attention(q, k, v)

def apply_rotary_emb(x, height, width, theta=10000.0):
    # 2D rotary embeddings for spatial positions
    # Based on Z-Image implementation
    freqs_h = compute_freqs(height, theta)
    freqs_w = compute_freqs(width, theta)
    # Apply rotation
    return rotate_tensor(x, freqs_h, freqs_w)
```

#### Removed SDXL Features

- ❌ `crop_coords_top_left`: Not needed (RoPE handles position)
- ❌ `add_time_ids`: Simplified conditioning
- ❌ Micro-conditioning: Focus on core diffusion

#### Architecture Size Calculation

**Target: 2.5 - 4B parameters**

**Component Breakdown**:
- Text Encoder (frozen): ~400M (SigLIP-2)
- Image Encoder (frozen): ~400M (SigLIP-2)
- U-Net (trainable): **2.0 - 3.0B**
- VAE (frozen): ~80M (FLUX VAE)

**U-Net Scaling Options**:

| Variant | Channels | Transformer Layers | Params | Target Use |
|---------|----------|-------------------|--------|------------|
| Small   | [320, 640, 1280] | [0, 2, 10] (SDXL) | ~2.0B | Fast inference |
| Medium  | [384, 768, 1536] | [0, 3, 12] | ~2.8B | Balanced |
| Large   | [448, 896, 1792] | [0, 4, 14] | ~3.5B | High quality |

**Recommendation**: Start with **Medium** (2.8B) for balance

---

### 1.4 VAE: FLUX VAE

**Model**: `black-forest-labs/FLUX.1-dev` VAE

**Specifications**:
- Latent channels: **16** (vs SDXL's 4)
- Scaling factor: 0.3611
- Latent resolution: H/8 × W/8
- Architecture: Autoencoder with improved reconstruction

**Key Differences from SDXL**:
```python
# SDXL VAE
latent_shape = (batch, 4, height // 8, width // 8)

# FLUX VAE
latent_shape = (batch, 16, height // 8, width // 8)
```

**U-Net Input Adaptation**:
```python
# U-Net input layer must accept 16 channels
self.conv_in = nn.Conv2d(16, model_channels, 3, padding=1)  # Not 4!
```

**Training Implications**:
- 4x more latent data per pixel
- Better detail preservation
- Higher VRAM usage for latents

---

## 2. Model Architecture Design

### 2.1 Overall Pipeline

```
Text Input → SigLIP-2 Text Encoder → text_embeddings [B, L, 1152]
                                    → pooled_text [B, 1152]

Image Input(s) → SigLIP-2 Image Encoder → image_embeddings [B, N, P, 1152]
   (optional)                            → pooled_image [B, 1152]

Input Image → FLUX VAE Encoder → latents [B, 16, H/8, W/8]

Conditioning:
  - Cross-attention: text_embeddings (always present)
  - Cross-attention: image_embeddings (optional, variable count)
  - Add embeddings: pooled_text + pooled_image (if present)

Latents + Conditioning → DiC U-Net → denoised_latents [B, 16, H/8, W/8]

Denoised Latents → FLUX VAE Decoder → output_image [B, 3, H, W]
```

### 2.2 U-Net Detailed Architecture

#### Input Processing

```python
class UNet2DConditionModel(nn.Module):
    def __init__(
        self,
        in_channels=16,  # FLUX VAE latent channels
        model_channels=384,  # Base channel count (Medium variant)
        out_channels=16,
        num_res_blocks=2,
        attention_resolutions=[2, 4],  # Apply attention at 1/2, 1/4 resolution
        channel_mult=[1, 2, 4],  # [384, 768, 1536]
        transformer_depth=[0, 3, 12],  # Transformer blocks per level
        context_dim=1152,  # SigLIP-2 embedding dim
        use_sparse_skip=True,  # DiC sparse skip connections
        skip_stride=3,
    ):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.GELU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )

        # Pooled embedding projection (text + optional image)
        self.pooled_embed = nn.Linear(1152, model_channels * 4)

        # Input conv (16 channels from FLUX VAE)
        self.conv_in = ImprovedConvBlock(in_channels, model_channels)
```

#### Down Blocks with Sparse Skip

```python
        self.down_blocks = nn.ModuleList()
        self.skip_connections = []  # Track which blocks have skip

        current_channels = model_channels
        for level, (mult, depth) in enumerate(zip(channel_mult, transformer_depth)):
            out_channels = model_channels * mult

            # ResNet blocks
            resnet_blocks = nn.ModuleList([
                ImprovedResNetBlock(current_channels, out_channels)
                for _ in range(num_res_blocks)
            ])

            # Transformer blocks (if depth > 0)
            transformer_blocks = None
            if depth > 0:
                transformer_blocks = nn.ModuleList([
                    RoPETransformerBlock(
                        dim=out_channels,
                        num_heads=out_channels // 64,
                        context_dim=context_dim,  # Cross-attention to text/image
                    )
                    for _ in range(depth)
                ])

            # Downsample
            downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

            self.down_blocks.append(nn.ModuleDict({
                'resnets': resnet_blocks,
                'transformers': transformer_blocks,
                'downsample': downsample,
            }))

            # Sparse skip connection decision
            has_skip = (level % skip_stride == 0) if use_sparse_skip else True
            self.skip_connections.append(has_skip)

            current_channels = out_channels
```

#### Mid Block

```python
        # Mid block (no skip connections)
        self.mid_block = nn.ModuleDict({
            'resnet_1': ImprovedResNetBlock(current_channels, current_channels),
            'transformers': nn.ModuleList([
                RoPETransformerBlock(current_channels, current_channels // 64, context_dim)
                for _ in range(transformer_depth[-1])  # Same depth as deepest down block
            ]),
            'resnet_2': ImprovedResNetBlock(current_channels, current_channels),
        })
```

#### Up Blocks with Sparse Skip

```python
        self.up_blocks = nn.ModuleList()

        for level, (mult, depth) in reversed(list(enumerate(zip(channel_mult, transformer_depth)))):
            out_channels = model_channels * mult

            # Calculate input channels considering sparse skip
            if self.skip_connections[level]:
                # Skip connection present
                in_channels = current_channels + out_channels
            else:
                # No skip
                in_channels = current_channels

            # ResNet blocks
            resnet_blocks = nn.ModuleList([
                ImprovedResNetBlock(in_channels if i == 0 else out_channels, out_channels)
                for i in range(num_res_blocks + 1)  # +1 for skip connection block
            ])

            # Transformer blocks
            transformer_blocks = None
            if depth > 0:
                transformer_blocks = nn.ModuleList([
                    RoPETransformerBlock(out_channels, out_channels // 64, context_dim)
                    for _ in range(depth)
                ])

            # Upsample
            upsample = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)

            self.up_blocks.append(nn.ModuleDict({
                'resnets': resnet_blocks,
                'transformers': transformer_blocks,
                'upsample': upsample,
            }))

            current_channels = out_channels
```

#### Output Layer

```python
        # Output conv (16 channels to match FLUX VAE)
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.GELU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )
```

#### Forward Pass

```python
    def forward(
        self,
        sample,  # [B, 16, H/8, W/8] FLUX VAE latents
        timestep,  # [B] or scalar
        encoder_hidden_states,  # [B, L, 1152] text embeddings (required)
        pooled_embeddings,  # [B, 1152] pooled text (required)
        image_encoder_hidden_states=None,  # [B, N*P, 1152] image embeddings (optional)
        image_pooled_embeddings=None,  # [B, 1152] pooled image (optional)
    ):
        # Time embedding
        t_emb = self.time_embed(timestep_embedding(timestep))

        # Pooled embedding (text + optional image)
        pooled = pooled_embeddings
        if image_pooled_embeddings is not None:
            pooled = pooled + image_pooled_embeddings  # Simple addition
        pooled_emb = self.pooled_embed(pooled)

        # Combine embeddings for conditioning
        emb = t_emb + pooled_emb

        # Prepare context for cross-attention
        context = encoder_hidden_states
        if image_encoder_hidden_states is not None:
            # Concatenate text and image contexts
            context = torch.cat([context, image_encoder_hidden_states], dim=1)

        # Input
        h = self.conv_in(sample, emb)

        # Down
        down_block_res_samples = []
        for i, block in enumerate(self.down_blocks):
            # ResNet blocks
            for resnet in block['resnets']:
                h = resnet(h, emb)

            # Transformer blocks
            if block['transformers'] is not None:
                height, width = h.shape[2], h.shape[3]
                for transformer in block['transformers']:
                    h = transformer(h, context, height, width)  # RoPE needs H, W

            # Save for skip connection (if applicable)
            if self.skip_connections[i]:
                down_block_res_samples.append(h)

            # Downsample
            h = block['downsample'](h)

        # Mid
        h = self.mid_block['resnet_1'](h, emb)
        height, width = h.shape[2], h.shape[3]
        for transformer in self.mid_block['transformers']:
            h = transformer(h, context, height, width)
        h = self.mid_block['resnet_2'](h, emb)

        # Up
        for i, block in enumerate(self.up_blocks):
            level = len(self.up_blocks) - 1 - i

            # Apply skip connection if present
            if self.skip_connections[level]:
                skip_sample = down_block_res_samples.pop()
                h = torch.cat([h, skip_sample], dim=1)

            # ResNet blocks
            for resnet in block['resnets']:
                h = resnet(h, emb)

            # Transformer blocks
            if block['transformers'] is not None:
                height, width = h.shape[2], h.shape[3]
                for transformer in block['transformers']:
                    h = transformer(h, context, height, width)

            # Upsample
            if i < len(self.up_blocks) - 1:  # No upsample on last block
                h = block['upsample'](h)

        # Output
        h = self.conv_out(h)

        return h
```

---

### 2.3 Conditioning Strategy

#### Text Conditioning (Always Present)

```python
# SigLIP-2 Text Encoder output
text_outputs = text_encoder(input_ids)
text_embeddings = text_outputs.last_hidden_state  # [B, L, 1152]
pooled_text = text_outputs.pooler_output  # [B, 1152]

# Usage in U-Net:
# - Cross-attention context: text_embeddings
# - Add embedding: pooled_text
```

#### Image Conditioning (Optional, Variable Count)

```python
# Handle 0, 1, or N input images
if reference_images is None or len(reference_images) == 0:
    # T2I mode: Use learned null embeddings
    image_embeddings = None
    pooled_image = None
elif len(reference_images) == 1:
    # Single image mode
    image_outputs = image_encoder(reference_images[0])
    image_embeddings = image_outputs.last_hidden_state  # [B, P, 1152]
    pooled_image = image_outputs.pooler_output  # [B, 1152]
else:
    # Multi-image mode: Process each and pool
    all_embeddings = []
    all_pooled = []
    for img in reference_images:
        outputs = image_encoder(img)
        all_embeddings.append(outputs.last_hidden_state)
        all_pooled.append(outputs.pooler_output)

    # Average pooling across images
    image_embeddings = torch.cat(all_embeddings, dim=1)  # [B, N*P, 1152]
    pooled_image = torch.stack(all_pooled).mean(dim=0)  # [B, 1152]

# Usage in U-Net:
# - Cross-attention context: concatenate with text_embeddings
# - Add embedding: pooled_text + pooled_image
```

---

### 2.4 RoPE Implementation (from Z-Image)

```python
def compute_frequencies(seq_len, dim, theta=10000.0):
    """
    Compute rotary frequencies for 1D sequence.

    Args:
        seq_len: Sequence length
        dim: Embedding dimension
        theta: Base for frequency computation

    Returns:
        Frequencies tensor [seq_len, dim//2]
    """
    position = torch.arange(seq_len, dtype=torch.float32)
    freqs = theta ** (-torch.arange(0, dim, 2).float() / dim)
    emb = position[:, None] * freqs[None, :]  # [seq_len, dim//2]
    return emb

def compute_2d_frequencies(height, width, dim, theta=10000.0):
    """
    Compute 2D rotary frequencies for spatial positions.

    Args:
        height: Spatial height
        width: Spatial width
        dim: Embedding dimension (must be divisible by 4)
        theta: Base for frequency computation

    Returns:
        Frequencies tensor [height, width, dim//2]
    """
    assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D RoPE"

    # Split dimension: half for height, half for width
    dim_h = dim // 2
    dim_w = dim // 2

    # Compute frequencies for height and width separately
    freqs_h = compute_frequencies(height, dim_h, theta)  # [H, dim_h//2]
    freqs_w = compute_frequencies(width, dim_w, theta)  # [W, dim_w//2]

    # Broadcast to 2D grid
    freqs_h = freqs_h[:, None, :].expand(-1, width, -1)  # [H, W, dim_h//2]
    freqs_w = freqs_w[None, :, :].expand(height, -1, -1)  # [H, W, dim_w//2]

    # Concatenate
    freqs_2d = torch.cat([freqs_h, freqs_w], dim=-1)  # [H, W, dim//2]

    return freqs_2d

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_emb_2d(x, freqs):
    """
    Apply 2D rotary embeddings.

    Args:
        x: Input tensor [B, H, W, num_heads, head_dim]
        freqs: Frequencies [H, W, head_dim//2]

    Returns:
        Rotated tensor [B, H, W, num_heads, head_dim]
    """
    # Expand frequencies for batch and heads
    freqs = freqs[:, :, None, :]  # [H, W, 1, head_dim//2]

    # Compute cos and sin
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # Apply rotation
    # x: [B, H, W, num_heads, head_dim]
    # Split head_dim in half
    x_rot = (x * cos) + (rotate_half(x) * sin)

    return x_rot

class RoPEAttention2D(nn.Module):
    """
    Multi-head attention with 2D RoPE positional encoding.
    Based on Z-Image implementation.
    """
    def __init__(self, dim, num_heads, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.theta = theta

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        # Cache for frequencies (recomputed if spatial size changes)
        self.register_buffer('_cached_freqs', None, persistent=False)
        self._cached_size = (None, None)

    def get_frequencies(self, height, width, device):
        """Get or compute 2D frequencies for given spatial size."""
        if self._cached_size != (height, width):
            freqs = compute_2d_frequencies(height, width, self.head_dim, self.theta)
            self._cached_freqs = freqs.to(device)
            self._cached_size = (height, width)
        return self._cached_freqs

    def forward(self, x, height, width):
        """
        Args:
            x: Input tensor [B, H*W, dim]
            height: Spatial height
            width: Spatial width

        Returns:
            Output tensor [B, H*W, dim]
        """
        B, N, C = x.shape
        assert N == height * width, f"Sequence length {N} != height {height} * width {width}"

        # Project to Q, K, V
        q = self.to_q(x)  # [B, N, dim]
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape to multi-head format
        q = q.reshape(B, height, width, self.num_heads, self.head_dim)
        k = k.reshape(B, height, width, self.num_heads, self.head_dim)
        v = v.reshape(B, height, width, self.num_heads, self.head_dim)

        # Apply RoPE to Q and K
        freqs = self.get_frequencies(height, width, x.device)
        q = apply_rotary_emb_2d(q, freqs)
        k = apply_rotary_emb_2d(k, freqs)

        # Reshape for attention: [B, num_heads, H*W, head_dim]
        q = q.reshape(B, height * width, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, height * width, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, height * width, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)

        # Apply attention to V
        out = attn @ v  # [B, num_heads, H*W, head_dim]

        # Reshape back
        out = out.transpose(1, 2).reshape(B, height * width, C)

        # Output projection
        out = self.to_out(out)

        return out
```

---

## 3. Training Pipeline

### 3.1 Data Pipeline

```python
class MultiModalDiffusionDataset(Dataset):
    """
    Dataset for training with optional image conditioning.

    Supports:
    - T2I: text only
    - I2I: text + single reference image
    - Multi-image: text + multiple reference images
    """
    def __init__(
        self,
        image_paths,
        captions,
        reference_images=None,  # Optional: list of lists of reference images
        resolution=1024,
        vae_scale_factor=8,
    ):
        self.image_paths = image_paths
        self.captions = captions
        self.reference_images = reference_images
        self.resolution = resolution
        self.vae_scale_factor = vae_scale_factor

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Target image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)

        # Caption (text conditioning)
        caption = self.captions[idx]

        # Reference images (optional image conditioning)
        ref_images = None
        if self.reference_images is not None and self.reference_images[idx]:
            ref_images = [
                self.transform(Image.open(path).convert('RGB'))
                for path in self.reference_images[idx]
            ]

        return {
            'image': image,
            'caption': caption,
            'reference_images': ref_images,
        }
```

### 3.2 Training Loop

```python
def train_step(batch, unet, vae, text_encoder, image_encoder, noise_scheduler):
    """
    Single training step.

    Args:
        batch: Dict with 'image', 'caption', 'reference_images'
        unet: U-Net model (trainable)
        vae: FLUX VAE (frozen)
        text_encoder: SigLIP-2 text encoder (frozen)
        image_encoder: SigLIP-2 image encoder (frozen)
        noise_scheduler: Diffusion scheduler (e.g., DDPMScheduler)
    """
    # 1. Encode target image to latents (FLUX VAE, 16 channels)
    with torch.no_grad():
        latents = vae.encode(batch['image']).latent_dist.sample()
        latents = latents * vae.config.scaling_factor  # 0.3611

    # 2. Encode text (SigLIP-2, always present)
    with torch.no_grad():
        text_outputs = text_encoder(batch['caption'])
        text_embeddings = text_outputs.last_hidden_state  # [B, L, 1152]
        pooled_text = text_outputs.pooler_output  # [B, 1152]

    # 3. Encode reference images (SigLIP-2, optional)
    image_embeddings = None
    pooled_image = None
    if batch['reference_images'] is not None:
        with torch.no_grad():
            # Handle variable number of reference images
            if isinstance(batch['reference_images'], list):
                # Multiple images per sample
                all_emb = []
                all_pooled = []
                for ref_imgs in batch['reference_images']:
                    for img in ref_imgs:
                        outputs = image_encoder(img.unsqueeze(0))
                        all_emb.append(outputs.last_hidden_state)
                        all_pooled.append(outputs.pooler_output)
                image_embeddings = torch.cat(all_emb, dim=1)
                pooled_image = torch.stack(all_pooled).mean(dim=0)
            else:
                # Single image
                outputs = image_encoder(batch['reference_images'])
                image_embeddings = outputs.last_hidden_state
                pooled_image = outputs.pooler_output

    # 4. Add noise to latents
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, noise_scheduler.num_train_timesteps,
        (latents.shape[0],), device=latents.device
    )
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # 5. Predict noise with U-Net
    noise_pred = unet(
        sample=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=text_embeddings,
        pooled_embeddings=pooled_text,
        image_encoder_hidden_states=image_embeddings,
        image_pooled_embeddings=pooled_image,
    )

    # 6. Compute loss
    loss = F.mse_loss(noise_pred, noise)

    return loss
```

### 3.3 Training Configuration

```yaml
# training_config.yaml

model:
  variant: medium  # small, medium, large
  in_channels: 16  # FLUX VAE latents
  model_channels: 384  # Medium variant
  channel_mult: [1, 2, 4]  # [384, 768, 1536]
  transformer_depth: [0, 3, 12]
  use_sparse_skip: true
  skip_stride: 3
  rope_theta: 10000.0

training:
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch size: 32
  learning_rate: 1e-4
  lr_scheduler: cosine
  warmup_steps: 1000
  max_train_steps: 100000
  mixed_precision: bf16
  gradient_checkpointing: true

  # Optimizer
  optimizer: adamw8bit_ringbuffer
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_weight_decay: 0.01
  adam_epsilon: 1e-8

  # EMA
  use_ema: true
  ema_decay: 0.9999

data:
  resolution: 1024
  center_crop: true
  random_flip: true

  # Multi-modal support
  text_dropout: 0.1  # 10% unconditional (CFG training)
  image_dropout: 0.5  # 50% without image conditioning (T2I focus)

vae:
  model: black-forest-labs/FLUX.1-dev
  scaling_factor: 0.3611
  latent_channels: 16

text_encoder:
  model: google/siglip2-so400m-patch16-naflex
  freeze: true
  max_length: null  # No limit

image_encoder:
  model: google/siglip2-so400m-patch16-naflex
  freeze: true
  max_images: 4  # Support up to 4 reference images
```

---

## 4. Implementation Plan

### Phase 1: Core Components (Week 1-2)

#### 1.1 Encoder Integration
- [ ] Load SigLIP-2 model from HuggingFace
- [ ] Test text encoding (verify no token limit)
- [ ] Test image encoding (single and multiple images)
- [ ] Implement conditioning strategy (variable image count)

#### 1.2 VAE Integration
- [ ] Load FLUX VAE from HuggingFace
- [ ] Test encoding/decoding (verify 16-channel latents)
- [ ] Measure VRAM usage vs SDXL VAE

#### 1.3 RoPE Implementation
- [ ] Implement 2D frequency computation
- [ ] Implement rotary embedding application
- [ ] Create RoPEAttention2D module
- [ ] Unit test: verify position awareness

---

### Phase 2: U-Net Architecture (Week 3-4)

#### 2.1 Basic Blocks
- [ ] ImprovedConvBlock (DiC-style, GELU, conditional gating)
- [ ] ImprovedResNetBlock (with stage-specific embeddings)
- [ ] RoPETransformerBlock (cross-attention + RoPE self-attention)

#### 2.2 U-Net Skeleton
- [ ] Input/output layers (16-channel support)
- [ ] Time embedding (+ pooled text/image embedding)
- [ ] Down blocks (with sparse skip tracking)
- [ ] Mid block
- [ ] Up blocks (with sparse skip application)

#### 2.3 Sparse Skip Connections
- [ ] Implement skip stride logic
- [ ] Test: verify gradient flow with/without skips
- [ ] Profile: measure VRAM/compute savings

#### 2.4 Multi-Modal Conditioning
- [ ] Text cross-attention (always present)
- [ ] Image cross-attention (optional, variable count)
- [ ] Pooled embedding addition (text + image)
- [ ] Test: t2i, i2i, multi-image modes

---

### Phase 3: Training Pipeline (Week 5-6)

#### 3.1 Data Pipeline
- [ ] MultiModalDiffusionDataset class
- [ ] DataLoader with collation (handle variable image count)
- [ ] Test: sample batches with 0/1/N reference images

#### 3.2 Training Loop
- [ ] Noise scheduling (DDPM or Flow Matching)
- [ ] Encoder frozen inference
- [ ] U-Net forward pass
- [ ] Loss computation (MSE or v-prediction)
- [ ] Gradient accumulation

#### 3.3 Optimization
- [ ] AdamW8bit ring buffer optimizer
- [ ] Gradient checkpointing
- [ ] Mixed precision (BF16)
- [ ] EMA model

#### 3.4 Monitoring
- [ ] Loss/LR logging
- [ ] Sample generation (fixed prompts)
- [ ] VRAM tracking
- [ ] Checkpoint saving

---

### Phase 4: Inference & Evaluation (Week 7)

#### 4.1 Inference Pipeline
- [ ] Sampling loop (DDPM, DPM-Solver, etc.)
- [ ] CFG support (text + image)
- [ ] Multi-image conditioning UI

#### 4.2 Frontend Integration
- [ ] Update model loader for new architecture
- [ ] Add image conditioning UI (upload reference images)
- [ ] Update generation panels

#### 4.3 Evaluation
- [ ] FID on validation set
- [ ] CLIP score
- [ ] Human evaluation (quality, adherence)

---

## 5. File Structure

```
backend/
├── core/
│   ├── models/
│   │   ├── unet_dic.py              # DiC-inspired U-Net
│   │   ├── attention_rope.py        # RoPE attention modules
│   │   ├── conv_blocks.py           # Improved conv/resnet blocks
│   │   ├── conditioning.py          # Multi-modal conditioning
│   │   └── siglip2_encoder.py       # SigLIP-2 wrapper
│   ├── pipelines/
│   │   ├── pipeline_multimodal.py   # Inference pipeline
│   │   └── pipeline_training.py     # Training pipeline
│   ├── training/
│   │   ├── trainer_multimodal.py    # Training loop
│   │   └── data_multimodal.py       # Dataset class
│   └── utils/
│       ├── rope_utils.py            # RoPE helper functions
│       └── conditioning_utils.py    # Conditioning helpers
```

---

## 6. Parameter Count Estimation

### Medium Variant (Target: 2.8B)

**U-Net Breakdown**:

| Component | Channels | Params | Notes |
|-----------|----------|--------|-------|
| Input Conv | 16 → 384 | ~6K | 3x3 conv |
| Time Embedding | | ~1.2M | Linear layers |
| Down Block 0 (no attn) | 384 | ~10M | 2 ResNet blocks |
| Down Block 1 (attn) | 768 | ~450M | 2 ResNet + 3 Transformers |
| Down Block 2 (attn) | 1536 | ~1.8B | 2 ResNet + 12 Transformers |
| Mid Block | 1536 | ~900M | 2 ResNet + 12 Transformers |
| Up Block 0 (attn) | 1536 | ~1.8B | 3 ResNet + 12 Transformers |
| Up Block 1 (attn) | 768 | ~450M | 3 ResNet + 3 Transformers |
| Up Block 2 (no attn) | 384 | ~12M | 3 ResNet blocks |
| Output Conv | 384 → 16 | ~6K | 3x3 conv |

**Total U-Net**: ~2.8B parameters

**Sparse Skip Savings**: ~20-30% compute reduction (not param reduction)

**Full Model**:
- SigLIP-2 Text Encoder: 400M (frozen)
- SigLIP-2 Image Encoder: 400M (frozen)
- U-Net: 2.8B (trainable)
- FLUX VAE: 80M (frozen)
- **Total**: ~3.7B parameters, **2.8B trainable**

---

## 7. Expected Challenges & Solutions

### Challenge 1: VRAM Usage (16-channel latents)

**Problem**: FLUX VAE latents are 4x larger than SDXL (16ch vs 4ch)

**Solutions**:
- Gradient checkpointing (enabled by default)
- Batch size reduction (8 → 4 if needed)
- Latent caching (pre-encode training images)

**Estimated VRAM (BF16 training)**:
- Model (2.8B): ~6 GB
- Optimizer state (AdamW8bit): ~3 GB
- Latents (1024x1024x16): ~0.5 GB/sample
- Activations (grad checkpoint): ~4 GB
- **Total**: ~15-18 GB for batch size 4

**Feasible on**: 24GB VRAM (RTX 4090, A5000)

---

### Challenge 2: Multi-Image Conditioning Complexity

**Problem**: Variable number of reference images (0, 1, N)

**Solutions**:
- Use masking for attention (pad to max count)
- Average pooling for simplicity (initial version)
- Cross-attention pooling (advanced version)

**Implementation**:
```python
# Simple averaging (Phase 1)
if len(image_embeddings) > 1:
    pooled_image = torch.stack(image_embeddings).mean(dim=0)

# Advanced: learnable attention pooling (Phase 2)
pooled_image = attention_pool(image_embeddings, query=text_pooled)
```

---

### Challenge 3: Training Stability (Large Model)

**Problem**: 2.8B parameters may be unstable with default hyperparams

**Solutions**:
- Lower learning rate: 1e-4 → 5e-5
- Gradient clipping: max_norm = 1.0
- EMA for stable checkpoints
- Warmup steps: 1000

**Monitoring**:
- Watch for NaN losses (reduce LR if occurs)
- Track gradient norms per layer
- Use mixed precision carefully (BF16 recommended)

---

## 8. Success Metrics

### Training Metrics

- [ ] **Loss convergence**: Stable loss < 0.1 (MSE noise prediction)
- [ ] **No NaN/Inf**: Clean training for 10k+ steps
- [ ] **EMA divergence**: < 5% difference from main model

### Quality Metrics

- [ ] **FID**: < 20 on validation set (competitive with SDXL)
- [ ] **CLIP Score**: > 0.3 (text-image alignment)
- [ ] **Visual inspection**: Coherent 1024x1024 images

### Inference Metrics

- [ ] **Speed**: < 3s per image @ 28 steps (RTX 4090)
- [ ] **VRAM**: < 12 GB @ 1024x1024 inference
- [ ] **CFG**: Stable guidance scale 7.0

---

## 9. Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Encoders + VAE | 1-2 weeks | Working SigLIP-2 + FLUX VAE integration |
| Phase 2: U-Net | 2 weeks | Complete U-Net architecture with RoPE + DiC |
| Phase 3: Training | 2 weeks | Full training pipeline + optimizer |
| Phase 4: Inference | 1 week | Frontend integration + evaluation |
| **Total** | **6-7 weeks** | Production-ready model |

---

## 10. References

### Papers
- **DiC**: [Diffusion in Convergence](https://arxiv.org/abs/2501.00603)
- **SigLIP-2**: [Google Blog](https://huggingface.co/google/siglip2-so400m-patch16-naflex)
- **Z-Image**: RoPE positional encoding
- **FLUX**: VAE architecture

### Code References
- SDXL U-Net: `diffusers.models.unet_2d_condition`
- RoPE: Z-Image implementation
- DiC: Official repo (if available)

### Model Checkpoints
- SigLIP-2: `google/siglip2-so400m-patch16-naflex`
- FLUX VAE: `black-forest-labs/FLUX.1-dev`

---

## Appendix: Alternative Architectures

### Option A: Smaller Model (2.0B)

**Changes**:
- Reduce channels: [320, 640, 1280] (SDXL standard)
- Reduce transformer depth: [0, 2, 10]
- Same sparse skip stride: 3

**Benefits**: Faster training, lower VRAM

**Trade-offs**: Potentially lower quality

---

### Option B: Larger Model (3.5B)

**Changes**:
- Increase channels: [448, 896, 1792]
- Increase transformer depth: [0, 4, 14]
- Same sparse skip stride: 3

**Benefits**: Higher capacity, better quality

**Trade-offs**: Slower, higher VRAM (needs 48GB+ for training)

---

### Option C: DiT-style (Full Transformer)

**Changes**:
- Replace ResNet blocks with Transformer blocks
- Isotropic architecture (no downsampling)
- Patchify input latents

**Benefits**: Better scalability, simpler architecture

**Trade-offs**: Slower inference, higher compute

**Recommendation**: Stick with U-Net for now (better proven for diffusion)
