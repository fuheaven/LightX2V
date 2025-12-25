"""
Rotary Position Embedding (RoPE) implementation for Enflame GCU.

This module provides utility functions for RoPE computation compatible with GCU platform.
Key compatibility notes:
- GCU does not support float64, all computations use float32
- GCU does not support int64/uint64, use int32 where needed

Reference: scripts/enflame/wan2.1/wan/modules/model.py
"""

import torch
import torch.cuda.amp as amp


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    """
    Generate RoPE frequency parameters.

    Args:
        max_seq_len: Maximum sequence length
        dim: Dimension of the embedding (must be even)
        theta: Base frequency parameter (default: 10000)

    Returns:
        freqs: Complex frequency tensor [max_seq_len, dim // 2]
    """
    assert dim % 2 == 0, "dim must be even for RoPE"

    # Use float32 instead of float64 (GCU does not support float64)
    freqs = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float32),
        1.0 / torch.pow(
            theta, torch.arange(0, dim, 2, dtype=torch.float32).div(dim)
        ),
    )

    # Generate complex frequencies using polar form
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    Apply rotary position embedding to input tensor.

    Args:
        x: Input tensor [B, L, N, C] where C is even
        grid_sizes: Grid sizes tensor [B, 3] containing (F, H, W) for each sample
        freqs: Frequency tensor from rope_params, shape [max_seq_len, dim // 2]

    Returns:
        output: Output tensor [B, L, N, C] with RoPE applied
    """
    n, c = x.size(2), x.size(3) // 2

    # Split frequencies into three parts (for F, H, W dimensions)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # Loop over samples in batch
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # Convert input to complex representation (GCU compatibility: use float32)
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2)
        )

        # Construct frequency tensor for this sample
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # Apply rotary embedding: multiply complex numbers
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)

        # Concatenate with remaining sequence (if any)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # Append to collection
        output.append(x_i)

    return torch.stack(output).float()

