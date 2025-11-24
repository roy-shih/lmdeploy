# Copyright (c) UnieAI. All rights reserved.
"""KV Cache Quantization kernels (INT4/INT8) optimized with Triton.

This module implements quantization for KV cache to reduce memory consumption:
- INT8: 50% memory savings
- INT4: 75% memory savings

Based on TurboMind's implementation but using Triton for cross-platform support.
"""
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['hidden_dim'],
)
@triton.jit
def _quantize_kv_int8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    zeropoint_ptr,
    num_tokens,
    hidden_dim: tl.constexpr,
    stride_n: tl.constexpr,
    stride_d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Quantize KV cache to INT8 per-token.

    Formula: quant = clamp(round(x / scale) + zeropoint, 0, 255)
    """
    token_id = tl.program_id(0)

    if token_id >= num_tokens:
        return

    # Load input for this token
    d_offs = tl.arange(0, BLOCK_SIZE)
    d_mask = d_offs < hidden_dim

    # Calculate min/max for this token
    min_val = float('inf')
    max_val = float('-inf')

    for d_start in range(0, hidden_dim, BLOCK_SIZE):
        d_block_offs = d_start + d_offs
        d_block_mask = d_block_offs < hidden_dim

        input_offset = token_id * stride_n + d_block_offs * stride_d
        vals = tl.load(input_ptr + input_offset, mask=d_block_mask, other=0.0)

        # Update min/max
        block_min = tl.min(vals, 0)
        block_max = tl.max(vals, 0)
        min_val = tl.minimum(min_val, block_min)
        max_val = tl.maximum(max_val, block_max)

    # Calculate quantization parameters
    # INT8 range: [0, 255]
    scale = (max_val - min_val) / 255.0
    scale = tl.maximum(scale, 1e-8)  # Avoid division by zero
    zeropoint = -tl.math.round(min_val / scale)
    zeropoint = tl.clamp(zeropoint, 0.0, 255.0)

    # Store quantization parameters
    tl.store(scale_ptr + token_id, scale)
    tl.store(zeropoint_ptr + token_id, zeropoint)

    # Quantize and store
    for d_start in range(0, hidden_dim, BLOCK_SIZE):
        d_block_offs = d_start + d_offs
        d_block_mask = d_block_offs < hidden_dim

        input_offset = token_id * stride_n + d_block_offs * stride_d
        vals = tl.load(input_ptr + input_offset, mask=d_block_mask, other=0.0)

        # Quantize: round(x / scale) + zeropoint
        quant_vals = tl.math.round(vals / scale) + zeropoint
        quant_vals = tl.clamp(quant_vals, 0.0, 255.0)

        # Convert to uint8
        quant_vals_u8 = quant_vals.to(tl.uint8)

        output_offset = token_id * hidden_dim + d_block_offs
        tl.store(output_ptr + output_offset, quant_vals_u8, mask=d_block_mask)


@triton.jit
def _dequantize_kv_int8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    zeropoint_ptr,
    num_tokens,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Dequantize INT8 KV cache back to FP16/BF16.

    Formula: dequant = (quant - zeropoint) * scale
    """
    token_id = tl.program_id(0)
    d_block_id = tl.program_id(1)

    if token_id >= num_tokens:
        return

    # Load quantization parameters
    scale = tl.load(scale_ptr + token_id)
    zeropoint = tl.load(zeropoint_ptr + token_id)

    # Load quantized values
    d_start = d_block_id * BLOCK_SIZE
    d_offs = d_start + tl.arange(0, BLOCK_SIZE)
    d_mask = d_offs < hidden_dim

    input_offset = token_id * hidden_dim + d_offs
    quant_vals = tl.load(input_ptr + input_offset, mask=d_mask, other=0)

    # Dequantize
    quant_vals_f32 = quant_vals.to(tl.float32)
    dequant_vals = (quant_vals_f32 - zeropoint) * scale

    # Store
    output_offset = token_id * hidden_dim + d_offs
    tl.store(output_ptr + output_offset, dequant_vals, mask=d_mask)


def quantize_kv_int8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize KV cache to INT8 with per-token scaling.

    Args:
        input: Input tensor of shape [num_tokens, hidden_dim]

    Returns:
        Tuple of (quantized, scales, zeropoints)
        - quantized: INT8 tensor of shape [num_tokens, hidden_dim]
        - scales: FP32 tensor of shape [num_tokens]
        - zeropoints: FP32 tensor of shape [num_tokens]
    """
    num_tokens, hidden_dim = input.shape

    # Output buffers
    output = torch.empty(num_tokens, hidden_dim, dtype=torch.uint8, device=input.device)
    scales = torch.empty(num_tokens, dtype=torch.float32, device=input.device)
    zeropoints = torch.empty(num_tokens, dtype=torch.float32, device=input.device)

    # Launch kernel
    grid = (num_tokens,)
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 256)

    _quantize_kv_int8_kernel[grid](
        input,
        output,
        scales,
        zeropoints,
        num_tokens,
        hidden_dim,
        input.stride(0),
        input.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output, scales, zeropoints


def dequantize_kv_int8(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    zeropoints: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize INT8 KV cache back to FP16/BF16.

    Args:
        quantized: INT8 tensor of shape [num_tokens, hidden_dim]
        scales: FP32 tensor of shape [num_tokens]
        zeropoints: FP32 tensor of shape [num_tokens]
        dtype: Output dtype (default: torch.float16)

    Returns:
        Dequantized tensor of shape [num_tokens, hidden_dim]
    """
    num_tokens, hidden_dim = quantized.shape

    # Output buffer
    output = torch.empty(num_tokens, hidden_dim, dtype=dtype, device=quantized.device)

    # Launch kernel
    BLOCK_SIZE = 256
    grid = (num_tokens, triton.cdiv(hidden_dim, BLOCK_SIZE))

    _dequantize_kv_int8_kernel[grid](
        quantized,
        output,
        scales,
        zeropoints,
        num_tokens,
        hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# ===== INT4 Quantization =====

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['hidden_dim'],
)
@triton.jit
def _quantize_kv_int4_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    zeropoint_ptr,
    num_tokens,
    hidden_dim: tl.constexpr,
    stride_n: tl.constexpr,
    stride_d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Quantize KV cache to INT4 per-token (2 values packed per byte).

    INT4 range: [0, 15]
    Packing: low 4 bits = first value, high 4 bits = second value
    """
    token_id = tl.program_id(0)

    if token_id >= num_tokens:
        return

    # Calculate min/max for this token
    min_val = float('inf')
    max_val = float('-inf')

    d_offs = tl.arange(0, BLOCK_SIZE)
    for d_start in range(0, hidden_dim, BLOCK_SIZE):
        d_block_offs = d_start + d_offs
        d_block_mask = d_block_offs < hidden_dim

        input_offset = token_id * stride_n + d_block_offs * stride_d
        vals = tl.load(input_ptr + input_offset, mask=d_block_mask, other=0.0)

        block_min = tl.min(vals, 0)
        block_max = tl.max(vals, 0)
        min_val = tl.minimum(min_val, block_min)
        max_val = tl.maximum(max_val, block_max)

    # Calculate quantization parameters for INT4 [0, 15]
    scale = (max_val - min_val) / 15.0
    scale = tl.maximum(scale, 1e-8)
    zeropoint = -tl.math.round(min_val / scale)
    zeropoint = tl.clamp(zeropoint, 0.0, 15.0)

    # Store quantization parameters
    tl.store(scale_ptr + token_id, scale)
    tl.store(zeropoint_ptr + token_id, zeropoint)

    # Quantize and pack (2 values per byte)
    output_dim = (hidden_dim + 1) // 2  # Packed dimension

    for d_start in range(0, hidden_dim, BLOCK_SIZE):
        d_block_offs = d_start + d_offs
        d_block_mask = d_block_offs < hidden_dim

        input_offset = token_id * stride_n + d_block_offs * stride_d
        vals = tl.load(input_ptr + input_offset, mask=d_block_mask, other=0.0)

        # Quantize
        quant_vals = tl.math.round(vals / scale) + zeropoint
        quant_vals = tl.clamp(quant_vals, 0.0, 15.0)
        quant_vals_u8 = quant_vals.to(tl.uint8)

        # Pack pairs of INT4 values into bytes
        # Low 4 bits: val[i*2], High 4 bits: val[i*2+1]
        for i in range(BLOCK_SIZE // 2):
            idx = d_start + i * 2
            if idx + 1 < hidden_dim:
                val0 = tl.load(input_ptr + token_id * stride_n + idx * stride_d)
                val1 = tl.load(input_ptr + token_id * stride_n + (idx + 1) * stride_d)

                quant0 = tl.clamp(tl.math.round(val0 / scale) + zeropoint, 0.0, 15.0).to(tl.uint8)
                quant1 = tl.clamp(tl.math.round(val1 / scale) + zeropoint, 0.0, 15.0).to(tl.uint8)

                # Pack: low 4 bits = quant0, high 4 bits = quant1
                packed = (quant1 << 4) | quant0

                output_offset = token_id * output_dim + idx // 2
                tl.store(output_ptr + output_offset, packed)


def quantize_kv_int4(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize KV cache to INT4 (75% memory savings).

    Args:
        input: Input tensor of shape [num_tokens, hidden_dim]

    Returns:
        Tuple of (quantized, scales, zeropoints)
        - quantized: Packed INT4 tensor [num_tokens, (hidden_dim+1)//2]
        - scales: FP32 tensor of shape [num_tokens]
        - zeropoints: FP32 tensor of shape [num_tokens]
    """
    num_tokens, hidden_dim = input.shape
    packed_dim = (hidden_dim + 1) // 2

    # Output buffers
    output = torch.empty(num_tokens, packed_dim, dtype=torch.uint8, device=input.device)
    scales = torch.empty(num_tokens, dtype=torch.float32, device=input.device)
    zeropoints = torch.empty(num_tokens, dtype=torch.float32, device=input.device)

    # Launch kernel
    grid = (num_tokens,)
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 256)

    _quantize_kv_int4_kernel[grid](
        input,
        output,
        scales,
        zeropoints,
        num_tokens,
        hidden_dim,
        input.stride(0),
        input.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output, scales, zeropoints


@triton.jit
def _dequantize_kv_int4_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    zeropoint_ptr,
    num_tokens,
    hidden_dim: tl.constexpr,
    packed_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Dequantize packed INT4 KV cache."""
    token_id = tl.program_id(0)
    d_block_id = tl.program_id(1)

    if token_id >= num_tokens:
        return

    # Load quantization parameters
    scale = tl.load(scale_ptr + token_id)
    zeropoint = tl.load(zeropoint_ptr + token_id)

    # Dequantize packed values
    d_start = d_block_id * BLOCK_SIZE
    d_offs = d_start + tl.arange(0, BLOCK_SIZE)

    for i in range(BLOCK_SIZE):
        d_idx = d_start + i
        if d_idx < hidden_dim:
            packed_idx = d_idx // 2
            is_high_bits = d_idx % 2

            # Load packed byte
            packed = tl.load(input_ptr + token_id * packed_dim + packed_idx)

            # Extract 4 bits
            if is_high_bits == 0:
                quant_val = packed & 0x0F  # Low 4 bits
            else:
                quant_val = (packed >> 4) & 0x0F  # High 4 bits

            # Dequantize
            quant_val_f32 = quant_val.to(tl.float32)
            dequant_val = (quant_val_f32 - zeropoint) * scale

            # Store
            tl.store(output_ptr + token_id * hidden_dim + d_idx, dequant_val)


def dequantize_kv_int4(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    zeropoints: torch.Tensor,
    hidden_dim: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize packed INT4 KV cache.

    Args:
        quantized: Packed INT4 tensor [num_tokens, (hidden_dim+1)//2]
        scales: FP32 tensor [num_tokens]
        zeropoints: FP32 tensor [num_tokens]
        hidden_dim: Original hidden dimension
        dtype: Output dtype

    Returns:
        Dequantized tensor [num_tokens, hidden_dim]
    """
    num_tokens, packed_dim = quantized.shape

    # Output buffer
    output = torch.empty(num_tokens, hidden_dim, dtype=dtype, device=quantized.device)

    # Launch kernel
    BLOCK_SIZE = 256
    grid = (num_tokens, triton.cdiv(hidden_dim, BLOCK_SIZE))

    _dequantize_kv_int4_kernel[grid](
        quantized,
        output,
        scales,
        zeropoints,
        num_tokens,
        hidden_dim,
        packed_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
