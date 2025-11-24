# Copyright (c) UnieAI. All rights reserved.
"""Attention reduce utilities optimized with Triton.

This module implements utilities for reducing partial attention outputs,
typically used in chunked or split attention for long sequences.

Based on TurboMind's implementation but using Triton for cross-platform support.
"""
import torch
import triton
import triton.language as tl
from typing import Optional


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 128}, num_warps=8),
    ],
    key=['head_dim'],
)
@triton.jit
def _reduce_partial_attention_kernel(
    output_ptr,
    partial_outputs_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    num_queries,
    num_heads,
    head_dim: tl.constexpr,
    num_splits,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Reduce partial attention outputs from chunked computation.

    This kernel merges partial attention results computed over different chunks
    of the key-value sequence.

    Formula:
        1. Find global max: M_global = max(M_i) for all splits i
        2. Compute correction factors: scale_i = exp(M_i - M_global)
        3. Reduce: O = sum(O_i * scale_i * L_i) / sum(scale_i * L_i)

    Args:
        output_ptr: Output attention [num_queries, num_heads, head_dim]
        partial_outputs_ptr: Partial outputs [num_queries, num_heads, num_splits, head_dim]
        partial_max_ptr: Partial max values [num_queries, num_heads, num_splits]
        partial_sum_ptr: Partial sum values [num_queries, num_heads, num_splits]
        num_queries: Number of queries
        num_heads: Number of attention heads
        head_dim: Head dimension
        num_splits: Number of splits
        BLOCK_SIZE_D: Block size for head dimension
    """
    query_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    d_block_idx = tl.program_id(2)

    if query_idx >= num_queries or head_idx >= num_heads:
        return

    # Compute offsets for head dimension
    d_start = d_block_idx * BLOCK_SIZE_D
    d_offs = d_start + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offs < head_dim

    # Find global maximum across all splits
    global_max = float('-inf')
    for split_idx in range(num_splits):
        max_offset = (query_idx * num_heads + head_idx) * num_splits + split_idx
        partial_max = tl.load(partial_max_ptr + max_offset)
        global_max = tl.maximum(global_max, partial_max)

    # Compute scaling factors and accumulate normalized sum
    total_sum = 0.0
    for split_idx in range(num_splits):
        idx_base = (query_idx * num_heads + head_idx) * num_splits + split_idx
        partial_max = tl.load(partial_max_ptr + idx_base)
        partial_sum = tl.load(partial_sum_ptr + idx_base)

        # Compute correction factor
        scale = tl.exp(partial_max - global_max)
        total_sum += scale * partial_sum

    # Accumulate partial outputs with proper scaling
    accu = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
    for split_idx in range(num_splits):
        idx_base = (query_idx * num_heads + head_idx) * num_splits + split_idx

        # Load partial values
        partial_max = tl.load(partial_max_ptr + idx_base)
        partial_sum = tl.load(partial_sum_ptr + idx_base)

        # Load partial output
        partial_offset = idx_base * head_dim + d_offs
        partial_out = tl.load(partial_outputs_ptr + partial_offset, mask=d_mask, other=0.0)

        # Compute contribution
        scale = tl.exp(partial_max - global_max)
        contribution = partial_out * scale * partial_sum

        # Accumulate
        accu += contribution

    # Normalize by total sum
    output = accu / tl.maximum(total_sum, 1e-9)

    # Store final output
    output_offset = (query_idx * num_heads + head_idx) * head_dim + d_offs
    tl.store(output_ptr + output_offset, output, mask=d_mask)


def reduce_partial_attention(
    partial_outputs: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
) -> torch.Tensor:
    """Reduce partial attention outputs from chunked computation.

    Args:
        partial_outputs: Partial outputs [num_queries, num_heads, num_splits, head_dim]
        partial_max: Partial max values [num_queries, num_heads, num_splits]
        partial_sum: Partial sum (denominator) values [num_queries, num_heads, num_splits]

    Returns:
        Reduced attention output [num_queries, num_heads, head_dim]
    """
    num_queries, num_heads, num_splits, head_dim = partial_outputs.shape

    # Output buffer
    output = torch.empty(
        (num_queries, num_heads, head_dim),
        dtype=partial_outputs.dtype,
        device=partial_outputs.device
    )

    # Launch kernel
    BLOCK_SIZE_D = 64 if head_dim <= 64 else 128
    grid = (num_queries, num_heads, triton.cdiv(head_dim, BLOCK_SIZE_D))

    _reduce_partial_attention_kernel[grid](
        output,
        partial_outputs,
        partial_max,
        partial_sum,
        num_queries,
        num_heads,
        head_dim,
        num_splits,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return output


@triton.jit
def _reduce_sum_kernel(
    output_ptr,
    input_ptr,
    reduction_size,
    other_size,
    reduce_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Generic reduction sum kernel.

    Args:
        output_ptr: Output tensor
        input_ptr: Input tensor
        reduction_size: Size of reduction dimension
        other_size: Size of other dimensions (combined)
        reduce_dim: Which dimension to reduce
        BLOCK_SIZE: Block size
    """
    idx = tl.program_id(0)

    if idx >= other_size:
        return

    # Accumulate sum
    accu = 0.0
    for r_start in range(0, reduction_size, BLOCK_SIZE):
        r_offs = r_start + tl.arange(0, BLOCK_SIZE)
        r_mask = r_offs < reduction_size

        # Calculate input offset
        offset = idx * reduction_size + r_offs
        vals = tl.load(input_ptr + offset, mask=r_mask, other=0.0)

        # Accumulate
        accu += tl.sum(vals, axis=0)

    # Store result
    tl.store(output_ptr + idx, accu)


def reduce_sum(
    input: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Reduce tensor by summing along a dimension.

    Args:
        input: Input tensor
        dim: Dimension to reduce

    Returns:
        Reduced tensor
    """
    # Flatten to 2D for reduction
    shape = input.shape
    if dim < 0:
        dim = len(shape) + dim

    # Calculate sizes
    reduction_size = shape[dim]
    before_size = 1
    for i in range(dim):
        before_size *= shape[i]
    after_size = 1
    for i in range(dim + 1, len(shape)):
        after_size *= shape[i]

    other_size = before_size * after_size

    # Output shape
    output_shape = list(shape)
    output_shape.pop(dim)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (other_size,)

    # Reshape for kernel
    input_2d = input.reshape(other_size, reduction_size)
    output_1d = output.reshape(other_size)

    _reduce_sum_kernel[grid](
        output_1d,
        input_2d,
        reduction_size,
        other_size,
        dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
