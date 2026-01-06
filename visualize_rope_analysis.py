"""
RoPE 2D Analysis and Visualization

Analyzes current RoPE implementation and visualizes:
1. Frequency distribution
2. Resolution extrapolation behavior
3. Proposed improvements for resolution-adaptive RoPE
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Create output directory
output_dir = Path("docs/rope_analysis")
output_dir.mkdir(parents=True, exist_ok=True)


def current_rope_implementation(H, W, dim=320, base=10000):
    """Current RoPE implementation from DEUS"""
    # Frequency bands
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # Position indices
    pos_h = torch.arange(H, dtype=torch.float32)
    pos_w = torch.arange(W, dtype=torch.float32)

    # Compute sinusoidal embeddings
    freqs_h = torch.einsum("i,j->ij", pos_h, inv_freq)  # [H, dim//2]
    emb_h = torch.cat([freqs_h.sin(), freqs_h.cos()], dim=-1)  # [H, dim]

    freqs_w = torch.einsum("i,j->ij", pos_w, inv_freq)  # [W, dim//2]
    emb_w = torch.cat([freqs_w.sin(), freqs_w.cos()], dim=-1)  # [W, dim]

    # Expand to 2D grid
    emb_h = emb_h.unsqueeze(1).expand(-1, W, -1)  # [H, W, dim]
    emb_w = emb_w.unsqueeze(0).expand(H, -1, -1)  # [H, W, dim]

    # Combine (simple addition)
    emb_2d = emb_h + emb_w  # [H, W, dim]

    return emb_2d, inv_freq


def resolution_adaptive_rope(H, W, dim=320, base=10000, train_resolution=128):
    """
    Resolution-adaptive RoPE with frequency scaling

    Key idea: Scale frequencies based on resolution ratio to maintain
    consistent positional information across different resolutions.

    Args:
        H, W: Target resolution (in latent space)
        dim: Embedding dimension
        base: Base for frequency calculation
        train_resolution: Training resolution (in latent space)
    """
    # Resolution scaling factor
    scale_h = H / train_resolution
    scale_w = W / train_resolution

    # Adaptive frequency bands (scale inversely with resolution)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # Position indices (normalized by scale)
    pos_h = torch.arange(H, dtype=torch.float32) / scale_h
    pos_w = torch.arange(W, dtype=torch.float32) / scale_w

    # Compute sinusoidal embeddings
    freqs_h = torch.einsum("i,j->ij", pos_h, inv_freq)  # [H, dim//2]
    emb_h = torch.cat([freqs_h.sin(), freqs_h.cos()], dim=-1)  # [H, dim]

    freqs_w = torch.einsum("i,j->ij", pos_w, inv_freq)  # [W, dim//2]
    emb_w = torch.cat([freqs_w.sin(), freqs_w.cos()], dim=-1)  # [W, dim]

    # Expand to 2D grid
    emb_h = emb_h.unsqueeze(1).expand(-1, W, -1)  # [H, W, dim]
    emb_w = emb_w.unsqueeze(0).expand(H, -1, -1)  # [H, W, dim]

    # Combine
    emb_2d = emb_h + emb_w  # [H, W, dim]

    return emb_2d, inv_freq, scale_h, scale_w


def visualize_frequency_distribution():
    """Visualize frequency distribution of RoPE"""
    dim = 320
    base = 10000

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    frequencies = 1.0 / inv_freq.numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Frequency distribution
    ax = axes[0, 0]
    ax.plot(frequencies, marker='o', markersize=3)
    ax.set_xlabel('Frequency Index')
    ax.set_ylabel('Frequency (cycles per position)')
    ax.set_title('RoPE Frequency Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 2. Wavelength distribution
    ax = axes[0, 1]
    wavelengths = 1.0 / frequencies
    ax.plot(wavelengths, marker='o', markersize=3, color='orange')
    ax.set_xlabel('Frequency Index')
    ax.set_ylabel('Wavelength (positions per cycle)')
    ax.set_title('RoPE Wavelength Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3. Coverage at different resolutions
    ax = axes[1, 0]
    resolutions = [32, 64, 128, 192, 256]
    colors = plt.cm.viridis(np.linspace(0, 1, len(resolutions)))

    for res, color in zip(resolutions, colors):
        # Number of complete cycles at this resolution
        cycles = res / wavelengths
        ax.plot(cycles, label=f'{res}x{res} latent', alpha=0.7, color=color)

    ax.set_xlabel('Frequency Index')
    ax.set_ylabel('Number of Complete Cycles')
    ax.set_title('Frequency Coverage at Different Resolutions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 4. Low vs high frequency coverage
    ax = axes[1, 1]
    low_freq_threshold = 10  # cycles per 128 positions
    high_freq_threshold = 100

    cycles_128 = 128 / wavelengths
    low_freq_count = (cycles_128 < low_freq_threshold).sum()
    mid_freq_count = ((cycles_128 >= low_freq_threshold) & (cycles_128 < high_freq_threshold)).sum()
    high_freq_count = (cycles_128 >= high_freq_threshold).sum()

    bars = ax.bar(['Low\n(<10 cycles)', 'Mid\n(10-100 cycles)', 'High\n(>100 cycles)'],
                  [low_freq_count, mid_freq_count, high_freq_count],
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_ylabel('Number of Frequency Bands')
    ax.set_title('Frequency Band Distribution\n(at 128x128 latent resolution)')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "rope_frequency_analysis.png", dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'rope_frequency_analysis.png'}")
    plt.close()


def visualize_resolution_extrapolation():
    """Visualize how RoPE behaves at different resolutions"""
    dim = 64  # Reduced for visualization
    train_res = 128
    test_resolutions = [64, 128, 192, 256]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for idx, res in enumerate(test_resolutions):
        # Current implementation
        emb_current, _ = current_rope_implementation(res, res, dim=dim)

        # Adaptive implementation
        emb_adaptive, _, scale_h, scale_w = resolution_adaptive_rope(
            res, res, dim=dim, train_resolution=train_res
        )

        # Visualize first 3 dimensions (channels)
        emb_current_vis = emb_current[:, :, :3].mean(dim=-1).numpy()
        emb_adaptive_vis = emb_adaptive[:, :, :3].mean(dim=-1).numpy()

        # Current implementation
        ax = axes[0, idx]
        im = ax.imshow(emb_current_vis, cmap='RdBu', vmin=-2, vmax=2)
        ax.set_title(f'Current\n{res}x{res} (×{res/train_res:.1f})')
        ax.axis('off')

        # Adaptive implementation
        ax = axes[1, idx]
        im = ax.imshow(emb_adaptive_vis, cmap='RdBu', vmin=-2, vmax=2)
        ax.set_title(f'Adaptive\n{res}x{res} (scale={scale_h:.2f})')
        ax.axis('off')

    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)

    fig.suptitle('RoPE Behavior at Different Resolutions\n(First 3 channels averaged)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "rope_resolution_extrapolation.png", dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'rope_resolution_extrapolation.png'}")
    plt.close()


def visualize_2d_rope_patterns():
    """Visualize 2D RoPE patterns at different frequencies"""
    H, W = 128, 128
    dim = 320

    emb_2d, inv_freq = current_rope_implementation(H, W, dim=dim)

    # Select specific frequency bands to visualize
    freq_indices = [0, 10, 50, 150]  # Low to high frequency

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for idx, freq_idx in enumerate(freq_indices):
        # Sin component
        ax = axes[0, idx]
        pattern = emb_2d[:, :, freq_idx].numpy()
        im = ax.imshow(pattern, cmap='RdBu', vmin=-2, vmax=2)
        wavelength = 1.0 / (1.0 / inv_freq[freq_idx//2].item())
        ax.set_title(f'Freq {freq_idx}: Sin\nλ={wavelength:.1f} positions')
        ax.axis('off')

        # Cos component
        ax = axes[1, idx]
        pattern = emb_2d[:, :, freq_idx + 1].numpy()
        im = ax.imshow(pattern, cmap='RdBu', vmin=-2, vmax=2)
        ax.set_title(f'Freq {freq_idx}: Cos\nλ={wavelength:.1f} positions')
        ax.axis('off')

    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)
    fig.suptitle('2D RoPE Patterns at Different Frequencies\n(128x128 latent space)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "rope_2d_patterns.png", dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'rope_2d_patterns.png'}")
    plt.close()


def compare_addition_vs_concatenation():
    """Compare addition vs concatenation for combining H/W embeddings"""
    H, W = 128, 128
    dim = 64  # Smaller for visualization

    emb_2d_add, inv_freq = current_rope_implementation(H, W, dim=dim)

    # Alternative: concatenation
    pos_h = torch.arange(H, dtype=torch.float32)
    pos_w = torch.arange(W, dtype=torch.float32)

    freqs_h = torch.einsum("i,j->ij", pos_h, inv_freq)
    emb_h = torch.cat([freqs_h.sin(), freqs_h.cos()], dim=-1)

    freqs_w = torch.einsum("i,j->ij", pos_w, inv_freq)
    emb_w = torch.cat([freqs_w.sin(), freqs_w.cos()], dim=-1)

    # Concatenate instead of add
    emb_h_exp = emb_h.unsqueeze(1).expand(-1, W, -1)
    emb_w_exp = emb_w.unsqueeze(0).expand(H, -1, -1)
    emb_2d_concat = torch.cat([emb_h_exp, emb_w_exp], dim=-1)  # [H, W, 2*dim]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # H component only
    ax = axes[0]
    im = ax.imshow(emb_h_exp[:, :, 0].numpy(), cmap='RdBu', vmin=-2, vmax=2)
    ax.set_title('Height Component\n(varies along Y)')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')

    # W component only
    ax = axes[1]
    im = ax.imshow(emb_w_exp[:, :, 0].numpy(), cmap='RdBu', vmin=-2, vmax=2)
    ax.set_title('Width Component\n(varies along X)')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')

    # Combined (addition)
    ax = axes[2]
    im = ax.imshow(emb_2d_add[:, :, 0].numpy(), cmap='RdBu', vmin=-2, vmax=2)
    ax.set_title('Combined (Addition)\n(varies along both)')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')

    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)
    fig.suptitle('2D RoPE: Height + Width Combination (First Channel)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "rope_2d_combination.png", dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'rope_2d_combination.png'}")
    plt.close()


def analyze_resolution_consistency():
    """Analyze consistency of embeddings across resolutions"""
    dim = 64
    train_res = 128
    test_resolutions = [64, 96, 128, 160, 192, 224, 256]

    # Sample a fixed position (center) at different resolutions
    center_embeddings_current = []
    center_embeddings_adaptive = []

    for res in test_resolutions:
        center_h = res // 2
        center_w = res // 2

        # Current
        emb_current, _ = current_rope_implementation(res, res, dim=dim)
        center_embeddings_current.append(emb_current[center_h, center_w, :10].numpy())

        # Adaptive
        emb_adaptive, _, _, _ = resolution_adaptive_rope(res, res, dim=dim, train_resolution=train_res)
        center_embeddings_adaptive.append(emb_adaptive[center_h, center_w, :10].numpy())

    center_embeddings_current = np.array(center_embeddings_current)
    center_embeddings_adaptive = np.array(center_embeddings_adaptive)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Current implementation
    ax = axes[0]
    im = ax.imshow(center_embeddings_current.T, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Embedding Dimension')
    ax.set_title('Current RoPE\n(Center Position Embedding)')
    ax.set_xticks(range(len(test_resolutions)))
    ax.set_xticklabels([str(r) for r in test_resolutions])

    # Adaptive implementation
    ax = axes[1]
    im = ax.imshow(center_embeddings_adaptive.T, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Embedding Dimension')
    ax.set_title('Adaptive RoPE\n(Center Position Embedding - Normalized)')
    ax.set_xticks(range(len(test_resolutions)))
    ax.set_xticklabels([str(r) for r in test_resolutions])

    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)
    fig.suptitle('Resolution Consistency Analysis\n(First 10 embedding dimensions at center position)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "rope_resolution_consistency.png", dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'rope_resolution_consistency.png'}")
    plt.close()


if __name__ == "__main__":
    print("=" * 80)
    print("RoPE 2D Analysis and Visualization")
    print("=" * 80)
    print()

    print("Generating visualizations...")
    print()

    visualize_frequency_distribution()
    visualize_resolution_extrapolation()
    visualize_2d_rope_patterns()
    compare_addition_vs_concatenation()
    analyze_resolution_consistency()

    print()
    print("=" * 80)
    print("Analysis complete!")
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 80)
