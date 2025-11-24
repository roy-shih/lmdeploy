# Copyright (c) UnieAI. All rights reserved.
"""Type conversion kernels optimized with Triton.

This module implements efficient type conversions between various data types:
- FP32 <-> FP16
- FP32 <-> BF16
- FP16 <-> BF16
- FP16/BF16/FP32 <-> INT8
- INT4 packing/unpacking
- Vectorized conversions for high performance

Based on TurboMind's implementation but using Triton for cross-platform support.
"""
import torch
import triton
import triton.language as tl
from typing import Optional


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _cast_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Generic type casting kernel.

    Converts elements from input dtype to output dtype.

    Args:
        input_ptr: Input tensor
        output_ptr: Output tensor
        n_elements: Number of elements
        BLOCK_SIZE: Block size for vectorization
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Store output (type conversion happens automatically)
    tl.store(output_ptr + offsets, input_vals, mask=mask)


def cast_dtype(
    input: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Convert tensor to different dtype.

    Args:
        input: Input tensor
        output_dtype: Target dtype (torch.float16, torch.bfloat16, torch.float32, etc.)

    Returns:
        Converted tensor
    """
    n_elements = input.numel()

    # Output buffer
    output = torch.empty(input.shape, dtype=output_dtype, device=input.device)

    # Launch kernel
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _cast_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _quantize_int8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    n_elements,
    per_token_quant: tl.constexpr,
    num_tokens,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Quantize FP16/BF16/FP32 to INT8.

    Args:
        input_ptr: Input tensor (float)
        output_ptr: Output tensor (int8)
        scale_ptr: Quantization scales
        n_elements: Total number of elements
        per_token_quant: If True, quantize per token; else per tensor
        num_tokens: Number of tokens (if per_token_quant)
        hidden_dim: Hidden dimension (if per_token_quant)
        BLOCK_SIZE: Block size
    """
    pid = tl.program_id(0)

    if per_token_quant:
        # Per-token quantization
        token_id = pid
        if token_id >= num_tokens:
            return

        # Find max absolute value for this token
        max_val = 0.0
        for d_start in range(0, hidden_dim, BLOCK_SIZE):
            d_offs = d_start + tl.arange(0, BLOCK_SIZE)
            d_mask = d_offs < hidden_dim
            offset = token_id * hidden_dim + d_offs
            vals = tl.load(input_ptr + offset, mask=d_mask, other=0.0)
            abs_vals = tl.abs(vals)
            block_max = tl.max(abs_vals, axis=0)
            max_val = tl.maximum(max_val, block_max)

        # Calculate scale
        scale = max_val / 127.0
        scale = tl.maximum(scale, 1e-8)  # Avoid division by zero

        # Store scale
        tl.store(scale_ptr + token_id, scale)

        # Quantize
        for d_start in range(0, hidden_dim, BLOCK_SIZE):
            d_offs = d_start + tl.arange(0, BLOCK_SIZE)
            d_mask = d_offs < hidden_dim
            offset = token_id * hidden_dim + d_offs
            vals = tl.load(input_ptr + offset, mask=d_mask, other=0.0)

            # Quantize to INT8: round(x / scale)
            quant_vals = tl.math.round(vals / scale)
            quant_vals = tl.clamp(quant_vals, -128.0, 127.0)
            quant_vals_i8 = quant_vals.to(tl.int8)

            tl.store(output_ptr + offset, quant_vals_i8, mask=d_mask)
    else:
        # Per-tensor quantization (simplified - use torch for global reduction)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load scale (pre-computed)
        scale = tl.load(scale_ptr)

        # Load and quantize
        vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        quant_vals = tl.math.round(vals / scale)
        quant_vals = tl.clamp(quant_vals, -128.0, 127.0)
        quant_vals_i8 = quant_vals.to(tl.int8)

        tl.store(output_ptr + offsets, quant_vals_i8, mask=mask)


def quantize_to_int8(
    input: torch.Tensor,
    per_token: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8.

    Args:
        input: Input tensor of shape [num_tokens, hidden_dim] or [n]
        per_token: If True, quantize per token; else per tensor

    Returns:
        Tuple of (quantized_tensor, scales)
    """
    if per_token:
        assert input.ndim == 2, "Per-token quantization requires 2D tensor"
        num_tokens, hidden_dim = input.shape
        n_elements = num_tokens * hidden_dim

        # Output buffers
        output = torch.empty(input.shape, dtype=torch.int8, device=input.device)
        scales = torch.empty(num_tokens, dtype=torch.float32, device=input.device)

        # Launch kernel (one thread block per token)
        grid = (num_tokens,)
        BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 1024)

        _quantize_int8_kernel[grid](
            input,
            output,
            scales,
            n_elements,
            per_token,
            num_tokens,
            hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Per-tensor quantization
        n_elements = input.numel()

        # Compute global scale
        max_val = input.abs().max()
        scale = (max_val / 127.0).clamp(min=1e-8)
        scales = torch.tensor([scale.item()], dtype=torch.float32, device=input.device)

        # Output buffer
        output = torch.empty(input.shape, dtype=torch.int8, device=input.device)

        # Launch kernel
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        _quantize_int8_kernel[grid](
            input,
            output,
            scales,
            n_elements,
            per_token,
            0,  # Unused
            0,  # Unused
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output, scales


@triton.jit
def _dequantize_int8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    n_elements,
    per_token_quant: tl.constexpr,
    num_tokens,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Dequantize INT8 to FP16/BF16/FP32.

    Args:
        input_ptr: Input tensor (int8)
        output_ptr: Output tensor (float)
        scale_ptr: Quantization scales
        n_elements: Total number of elements
        per_token_quant: If True, dequantize per token; else per tensor
        num_tokens: Number of tokens (if per_token_quant)
        hidden_dim: Hidden dimension (if per_token_quant)
        BLOCK_SIZE: Block size
    """
    pid = tl.program_id(0)

    if per_token_quant:
        token_id = pid // triton.cdiv(hidden_dim, BLOCK_SIZE)
        block_id = pid % triton.cdiv(hidden_dim, BLOCK_SIZE)

        if token_id >= num_tokens:
            return

        # Load scale
        scale = tl.load(scale_ptr + token_id)

        # Dequantize block
        d_start = block_id * BLOCK_SIZE
        d_offs = d_start + tl.arange(0, BLOCK_SIZE)
        d_mask = d_offs < hidden_dim

        offset = token_id * hidden_dim + d_offs
        quant_vals = tl.load(input_ptr + offset, mask=d_mask, other=0)

        # Dequantize: x * scale
        quant_vals_f32 = quant_vals.to(tl.float32)
        dequant_vals = quant_vals_f32 * scale

        tl.store(output_ptr + offset, dequant_vals, mask=d_mask)
    else:
        # Per-tensor dequantization
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load scale
        scale = tl.load(scale_ptr)

        # Dequantize
        quant_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
        quant_vals_f32 = quant_vals.to(tl.float32)
        dequant_vals = quant_vals_f32 * scale

        tl.store(output_ptr + offsets, dequant_vals, mask=mask)


def dequantize_from_int8(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
    per_token: bool = False,
) -> torch.Tensor:
    """Dequantize INT8 tensor.

    Args:
        quantized: Quantized INT8 tensor
        scales: Quantization scales
        output_dtype: Output dtype (default: torch.float16)
        per_token: If True, dequantize per token; else per tensor

    Returns:
        Dequantized tensor
    """
    n_elements = quantized.numel()

    # Output buffer
    output = torch.empty(quantized.shape, dtype=output_dtype, device=quantized.device)

    if per_token:
        assert quantized.ndim == 2, "Per-token dequantization requires 2D tensor"
        num_tokens, hidden_dim = quantized.shape

        BLOCK_SIZE = 256
        grid = (num_tokens * triton.cdiv(hidden_dim, BLOCK_SIZE),)

        _dequantize_int8_kernel[grid](
            quantized,
            output,
            scales,
            n_elements,
            per_token,
            num_tokens,
            hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        _dequantize_int8_kernel[grid](
            quantized,
            output,
            scales,
            n_elements,
            per_token,
            0,
            0,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output


@triton.jit
def _pack_int4_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Pack INT8 values into INT4 (2 values per byte).

    Args:
        input_ptr: Input INT8 tensor (values in range [0, 15])
        output_ptr: Output packed INT4 tensor
        n_elements: Number of elements in input
        BLOCK_SIZE: Block size
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Process pairs of elements
    # Each thread handles 2 input values -> 1 output byte
    input_offsets = offsets * 2
    mask = input_offsets < n_elements

    # Load two values
    val0 = tl.load(input_ptr + input_offsets, mask=mask, other=0).to(tl.uint8)
    val1 = tl.load(input_ptr + input_offsets + 1, mask=mask, other=0).to(tl.uint8)

    # Clamp to [0, 15]
    val0 = tl.minimum(val0, 15)
    val1 = tl.minimum(val1, 15)

    # Pack: low 4 bits = val0, high 4 bits = val1
    packed = (val1 << 4) | val0

    # Store packed byte
    tl.store(output_ptr + offsets, packed, mask=mask)


def pack_to_int4(input: torch.Tensor) -> torch.Tensor:
    """Pack INT8 tensor to INT4 (2 values per byte).

    Args:
        input: Input INT8 tensor with values in range [0, 15]

    Returns:
        Packed tensor with shape [(numel+1)//2]
    """
    n_elements = input.numel()
    packed_size = (n_elements + 1) // 2

    # Output buffer
    output = torch.empty(packed_size, dtype=torch.uint8, device=input.device)

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(packed_size, BLOCK_SIZE),)

    _pack_int4_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


@triton.jit
def _unpack_int4_kernel(
    input_ptr,
    output_ptr,
    n_output_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Unpack INT4 to INT8 (2 values per byte).

    Args:
        input_ptr: Input packed INT4 tensor
        output_ptr: Output INT8 tensor
        n_output_elements: Number of elements in output
        BLOCK_SIZE: Block size
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_output_elements

    # Calculate packed byte index
    packed_idx = offsets // 2
    is_high = offsets % 2

    # Load packed byte
    packed = tl.load(input_ptr + packed_idx, mask=mask, other=0)

    # Extract 4 bits
    val = tl.where(is_high == 0, packed & 0x0F, (packed >> 4) & 0x0F)

    # Store
    tl.store(output_ptr + offsets, val.to(tl.int8), mask=mask)


def unpack_from_int4(
    packed: torch.Tensor,
    n_elements: int,
) -> torch.Tensor:
    """Unpack INT4 tensor to INT8.

    Args:
        packed: Packed INT4 tensor
        n_elements: Number of elements to unpack

    Returns:
        Unpacked INT8 tensor
    """
    # Output buffer
    output = torch.empty(n_elements, dtype=torch.int8, device=packed.device)

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _unpack_int4_kernel[grid](
        packed,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


@triton.jit
def _fuse_scales_zeros_kernel(
    scales_ptr,
    zeros_ptr,
    fused_ptr,
    n_elements,
    has_zeros: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse scales and zeros for quantization.

    Formula: fused[i*2] = scale[i], fused[i*2+1] = -zero[i] * scale[i]

    Args:
        scales_ptr: Scales tensor
        zeros_ptr: Zeros tensor (can be None)
        fused_ptr: Fused output [n_elements * 2]
        n_elements: Number of scale/zero pairs
        has_zeros: Whether zeros tensor is provided
        BLOCK_SIZE: Block size
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load scales
    scales = tl.load(scales_ptr + offsets, mask=mask, other=0.0)

    # Load zeros if provided
    if has_zeros:
        zeros = tl.load(zeros_ptr + offsets, mask=mask, other=0.0)
    else:
        zeros = tl.zeros_like(scales)

    # Store fused: [scale, -zero * scale, scale, -zero * scale, ...]
    output_offsets_0 = offsets * 2
    output_offsets_1 = offsets * 2 + 1

    tl.store(fused_ptr + output_offsets_0, scales, mask=mask)
    tl.store(fused_ptr + output_offsets_1, -zeros * scales, mask=mask)


def fuse_scales_and_zeros(
    scales: torch.Tensor,
    zeros: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fuse scales and zeros for efficient quantization.

    Args:
        scales: Scales tensor
        zeros: Optional zeros tensor

    Returns:
        Fused tensor of shape [numel * 2]
    """
    n_elements = scales.numel()

    # Output buffer
    fused = torch.empty(n_elements * 2, dtype=scales.dtype, device=scales.device)

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _fuse_scales_zeros_kernel[grid](
        scales,
        zeros if zeros is not None else scales,  # Dummy pointer
        fused,
        n_elements,
        zeros is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return fused
