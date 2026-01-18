# Memory Management for VRAM-Efficient Training

This module provides advanced memory management for training large transformer models with limited VRAM.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    VRAM-Efficient Training                      │
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │  Block Offloading    │    │  Fused Optimizer Groups      │  │
│  │  (CPU ↔ GPU swap)    │    │  (Per-group update)          │  │
│  └──────────┬───────────┘    └────────┬─────────────────────┘  │
│             │                         │                        │
│             └─────────┬───────────────┘                        │
│                       │                                        │
│             ┌─────────▼────────────┐                          │
│             │ FusedBlockSwapTrainer │                         │
│             │ (Complete Integration)│                         │
│             └─────────┬─────────────┘                         │
│                       │                                        │
│        ┌──────────────┴──────────────┐                        │
│        │                             │                        │
│  ┌─────▼──────────┐       ┌──────────▼─────────┐            │
│  │ Ring Buffer    │       │ Layer Offload       │            │
│  │ Allocator      │       │ Conductor           │            │
│  │ (Custom Alloc) │       │ (Hook-based swap)   │            │
│  └────────────────┘       └─────────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Ring Buffer Allocator (`ring_buffer_allocator.py`)

Custom memory allocator to avoid PyTorch caching allocator fragmentation.

**Classes**:
- `RingBufferAllocator`: Main allocator with lazy buffer initialization
- `TensorAllocator`: Bidirectional allocation (forward: left-to-right, backward: right-to-left)
- `DynamicActivationAllocator`: Growing allocator for gradient checkpointing activations

**Key Features**:
- Eliminates memory fragmentation during layer swapping
- Bidirectional ring buffer strategy
- Pinned memory support for faster CPU↔GPU transfer
- Lazy allocation to minimize memory overhead

**Usage**:
```python
allocator = RingBufferAllocator(device=torch.device('cpu'))
allocator.initialize(layers=transformer.layers, target_bytes=8*1024**3)

# Get allocator for specific layer
layer_alloc = allocator.get_layer_allocator(layer_idx=5, forward=True)
tensor = layer_alloc.allocate(template=original_tensor)
```

### 2. Tensor Utilities (`tensor_utils.py`)

Helper functions for tensor manipulation and device transfer.

**Key Functions**:
- `extract_tensors()`: Extract all tensors from nested structures
- `replace_tensor_data()`: In-place tensor data replacement (preserves gradient tracking)
- `move_tensors_to_device()`: Device transfer with custom allocator support
- `async_copy_to_device()`: Non-blocking copy with CUDA events
- `cuda_stream_context()`: CUDA stream management context manager

**Usage**:
```python
# Async copy with event
event = async_copy_to_device(source=cpu_tensor, target=gpu_tensor, stream=my_stream)

# Wait for completion
wait_for_event(event, stream=compute_stream)
```

### 3. Layer Offload Strategy (`layer_offload_strategy.py`)

Computes optimal loading/offloading schedule for transformer layers.

**Class**: `LayerOffloadStrategy`

**Key Features**:
- Resident vs offloadable layer calculation
- Forward/backward schedule generation
- Prefetching support for next layer
- Memory-efficient scheduling

**Usage**:
```python
strategy = LayerOffloadStrategy(
    num_layers=28,
    blocks_to_swap=22,
    device=torch.device('cuda:0')
)

# Get forward pass schedule
forward_schedule = strategy.get_forward_schedule()

# Check if layer should be prefetched
next_layer = strategy.should_prefetch(current_layer=5, direction='forward')
```

### 4. Layer Offload Conductor (`layer_offload_conductor.py`)

Orchestrates layer loading/offloading with async transfers and custom allocators.

**Class**: `LayerOffloadConductor`

**Key Features**:
- Async CPU↔GPU transfer with dedicated CUDA streams
- Custom memory allocators to avoid fragmentation
- Activation offloading for gradient checkpointing
- Hook-based integration with transformer layers
- Prefetching for performance optimization

**Usage**:
```python
conductor = LayerOffloadConductor(
    layers=transformer.layers,
    blocks_to_swap=22,
    device=torch.device('cuda:0'),
    use_pinned_memory=True,
    enable_activation_offload=True
)

# Register hooks for automatic offloading
conductor.register_hooks()

# Forward pass (handled by hooks)
output = conductor.forward_layer(layer_idx=5, hidden_states=x)

# Cleanup
conductor.cleanup()
```

### 5. Block Offloading (`block_offloading.py`)

Production-ready block offloader for inference and training.

**Class**: `TransformerBlockOffloader`

**Key Features**:
- Weight-only offloading (Linear/Conv weights on CPU, buffers on GPU)
- Forward-only strategy (keeps first N blocks on GPU permanently)
- Async weight swapping with staging buffers
- Backward hooks for training-time block swapping
- CUDA streams and events for async operations

**Usage**:
```python
offloader = TransformerBlockOffloader(
    blocks=transformer.layers,
    blocks_to_swap=22,
    device=torch.device('cuda:0'),
    use_pinned_memory=True,
    supports_backward=True  # For training
)

# Prepare block devices
offloader.prepare_block_devices_before_forward()

# Register backward hooks
offloader.register_backward_hooks()

# Forward pass with automatic swapping
for i, layer in enumerate(transformer.layers):
    offloader.wait_for_block(i)
    hidden_states = layer(hidden_states)
    offloader.submit_move_blocks_forward(i)
```

### 6. Fused Block Swap Trainer (`fused_block_swap.py`)

Complete VRAM-efficient training system combining Block Swap + Fused Optimizer Groups.

**Class**: `FusedBlockSwapTrainer`

**Key Features**:
- Integrates Block Swap and Fused Optimizer Groups
- Minimal VRAM usage for full fine-tuning large models
- Training-time block swapping with backward hooks
- Memory statistics and monitoring

**Usage**:
```python
trainer = FusedBlockSwapTrainer(
    transformer=model,
    blocks_to_swap=22,
    optimizer_groups=optimizers,
    device=torch.device('cuda:0'),
    use_pinned_memory=True,
    max_grad_norm=1.0
)

# Prepare for training
trainer.prepare()

# Training loop
for batch in dataloader:
    trainer.train_step_begin()

    # Forward pass
    output = trainer.forward_with_block_swap(batch)
    loss = compute_loss(output, targets)

    # Backward (optimizer step handled by hooks)
    loss.backward()

    trainer.train_step_end()

# Cleanup
trainer.cleanup()
```

## Implementation Strategy

### Block Swap

**Goal**: Reduce VRAM usage by moving transformer layers between CPU and GPU.

**Forward Pass**:
1. Load layer N from CPU to GPU
2. Execute layer N
3. Offload layer N from GPU to CPU
4. Prefetch layer N+1 (async)

**Backward Pass**:
1. Load layer N from CPU to GPU (reverse order)
2. Execute layer N backward
3. Offload layer N from GPU to CPU

**Key Insight**: Only 1-2 layers need to be on GPU at any time.

### Fused Optimizer Groups

**Goal**: Enable optimizer updates with Block Swap.

**Problem**: Standard optimizer.step() requires all parameters on GPU simultaneously.

**Solution**: Divide parameters into groups, update each group when gradients are ready.

**Implementation**:
1. Divide parameters into N groups (recommended: 4-10)
2. Create N optimizer instances (one per group)
3. Register post-accumulate-grad hooks
4. Hook calls optimizer.step() when all parameters in group have gradients

**Key Insight**: Per-group updates avoid keeping all parameters on GPU.

## Memory Reduction

**Example**: Z-Image Transformer (28 layers, 8.4B parameters)

| Configuration | VRAM Usage | Reduction |
|---------------|------------|-----------|
| No optimization | 39-44 GB | - |
| Gradient checkpointing only | 20-25 GB | ~45% |
| + Block Swap (22 blocks) | 10-12 GB | ~75% |
| + Fused Optimizer Groups | 8-10 GB | ~80% |

**Trade-offs**:
- Training speed: ~30-40% slower (due to CPU↔GPU transfer)
- Implementation complexity: Higher
- Compatibility: Requires PyTorch 2.1+ for hooks

## PyTorch Requirements

- **Minimum**: PyTorch 2.0.0
- **Recommended**: PyTorch 2.1.0+ (for `register_post_accumulate_grad_hook`)

**Required features**:
- `torch.cuda.Stream`: CUDA stream support
- `torch.cuda.Event`: CUDA event support
- `Tensor.register_post_accumulate_grad_hook`: Per-parameter gradient hooks (2.1+)
- `pin_memory()`: Pinned memory allocation

## Best Practices

### Block Swap Configuration

1. **Start with blocks_to_swap = 0** (baseline)
2. **Increase gradually**: 6 → 12 → 18 → 22
3. **Monitor VRAM**: Use `print_memory_stats()` to check reduction
4. **Balance speed vs VRAM**: More swapping = less VRAM, slower training

### Fused Optimizer Groups

1. **num_optimizer_groups**: 4-10 (recommended)
2. **Too few**: Less VRAM reduction
3. **Too many**: Higher overhead, slower training
4. **Incompatible with**: 8-bit optimizers (AdamW8bit, Lion8bit)

### Pinned Memory

- **Enable** (`use_pinned_memory=True`): Faster CPU↔GPU transfer
- **Disable** (`use_pinned_memory=False`): Lower CPU memory usage
- **Recommended**: Enable unless CPU RAM is limited (<32GB)

### Gradient Checkpointing

- **Always enable** for large models
- Reduces activation memory by ~50%
- Combined with Block Swap: ~80% total VRAM reduction
- Trade-off: ~30% slower training (recomputation during backward)

## Troubleshooting

### Error: "CUDA out of memory"

**Cause**: Too many layers on GPU simultaneously.

**Solution**:
1. Increase `blocks_to_swap` (e.g., 22 → 24)
2. Enable gradient checkpointing
3. Reduce batch size

### Error: "register_post_accumulate_grad_hook not found"

**Cause**: PyTorch version < 2.1.0

**Solution**: Upgrade PyTorch to 2.1.0+

### Performance: Training very slow

**Cause**: Too much CPU↔GPU transfer overhead.

**Solution**:
1. Decrease `blocks_to_swap` (less swapping)
2. Enable pinned memory (`use_pinned_memory=True`)
3. Use faster CPU (Ryzen 9 / i9)

### Error: "8-bit optimizer incompatible with Block Swap"

**Cause**: 8-bit optimizers cannot handle CPU parameters.

**Solution**:
1. Use standard optimizer (AdamW, Lion, SGD)
2. Or disable Block Swap (`blocks_to_swap=0`)
3. Or use Adafactor with fused backward pass (no optimizer groups)

## Future Extensions

### Activation Offloading

- **Goal**: Offload activations to CPU during forward pass
- **Benefit**: Further VRAM reduction (~50% additional)
- **Status**: Implemented in `LayerOffloadConductor` (`enable_activation_offload=True`)

### Adaptive Block Swap

- **Goal**: Dynamically adjust blocks_to_swap based on VRAM usage
- **Benefit**: Automatic optimization without manual tuning
- **Status**: Planned

### Multi-GPU Support

- **Goal**: Distribute layers across multiple GPUs
- **Benefit**: Train larger models with 2-4 GPUs
- **Status**: Planned

## References

- **musubi-tuner**: Weight-only offloading strategy
- **PyTorch Documentation**: CUDA streams, events, pinned memory
- **Diffusers**: Gradient checkpointing implementation

## License

This implementation is original code developed for SushiUI.
