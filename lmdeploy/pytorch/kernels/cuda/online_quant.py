# Copyright (c) UnieAI. All rights reserved.
"""Online Activation Quantization kernels for W8A8 inference.

Implements dynamic quantization of activations during inference for
INT8 matrix multiplication (W8A8).
"""
import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'K'],
)
@triton.jit
def _online_quant_activation_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    M,
    K,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_om: tl.constexpr,
    stride_ok: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Quantize activations to INT8 with per-token scaling.

    For each token (row), compute:
    - abs_max = max(abs(input[i]))
    - scale = abs_max / 127
    - output[i] = clamp(round(input[i] / scale), -128, 127)
    """
    m_id = tl.program_id(0)
    m_start = m_id * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    # Find abs_max for each row
    abs_max_vals = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load input block
        input_ptrs = input_ptr + m_offs[:, None] * stride_im + k_offs[None, :] * stride_ik
        mask = m_mask[:, None] & k_mask[None, :]
        vals = tl.load(input_ptrs, mask=mask, other=0.0)

        # Update abs_max for each row
        abs_vals = tl.abs(vals)
        row_max = tl.max(abs_vals, axis=1)
        abs_max_vals = tl.maximum(abs_max_vals, row_max)

    # Compute scales (per-token)
    scales = abs_max_vals / 127.0
    scales = tl.maximum(scales, 1e-8)  # Avoid division by zero

    # Store scales
    tl.store(scale_ptr + m_offs, scales, mask=m_mask)

    # Quantize and store
    for k_start in tl.range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load input
        input_ptrs = input_ptr + m_offs[:, None] * stride_im + k_offs[None, :] * stride_ik
        mask = m_mask[:, None] & k_mask[None, :]
        vals = tl.load(input_ptrs, mask=mask, other=0.0)

        # Quantize: round(x / scale)
        quant_vals = tl.math.round(vals / scales[:, None])
        quant_vals = tl.clamp(quant_vals, -128.0, 127.0)
        quant_vals_i8 = quant_vals.to(tl.int8)

        # Store
        output_ptrs = output_ptr + m_offs[:, None] * stride_om + k_offs[None, :] * stride_ok
        tl.store(output_ptrs, quant_vals_i8, mask=mask)


def online_quant_activation(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Online quantize activations to INT8 with per-token scaling.

    Args:
        input: FP16/BF16 tensor of shape [M, K]

    Returns:
        Tuple of (quantized, scales)
        - quantized: INT8 tensor of shape [M, K]
        - scales: FP32 tensor of shape [M]
    """
    M, K = input.shape

    # Output buffers
    output = torch.empty(M, K, dtype=torch.int8, device=input.device)
    scales = torch.empty(M, dtype=torch.float32, device=input.device)

    # Launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)

    _online_quant_activation_kernel[grid](
        input,
        output,
        scales,
        M,
        K,
        input.stride(0),
        input.stride(1),
        output.stride(0),
        output.stride(1),
    )

    return output, scales


@triton.jit
def _dequant_activation_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    M,
    K,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_om: tl.constexpr,
    stride_ok: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Dequantize INT8 activations back to FP16/BF16."""
    m_id = tl.program_id(0)
    k_id = tl.program_id(1)

    m_start = m_id * BLOCK_M
    k_start = k_id * BLOCK_K

    m_offs = m_start + tl.arange(0, BLOCK_M)
    k_offs = k_start + tl.arange(0, BLOCK_K)

    m_mask = m_offs < M
    k_mask = k_offs < K

    # Load scales
    scales = tl.load(scale_ptr + m_offs, mask=m_mask, other=1.0)

    # Load quantized values
    input_ptrs = input_ptr + m_offs[:, None] * stride_im + k_offs[None, :] * stride_ik
    mask = m_mask[:, None] & k_mask[None, :]
    quant_vals = tl.load(input_ptrs, mask=mask, other=0)

    # Dequantize
    quant_vals_f32 = quant_vals.to(tl.float32)
    dequant_vals = quant_vals_f32 * scales[:, None]

    # Store
    output_ptrs = output_ptr + m_offs[:, None] * stride_om + k_offs[None, :] * stride_ok
    tl.store(output_ptrs, dequant_vals, mask=mask)


def dequant_activation(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize INT8 activations.

    Args:
        quantized: INT8 tensor [M, K]
        scales: FP32 tensor [M]
        dtype: Output dtype

    Returns:
        Dequantized tensor [M, K]
    """
    M, K = quantized.shape

    output = torch.empty(M, K, dtype=dtype, device=quantized.device)

    BLOCK_M, BLOCK_K = 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

    _dequant_activation_kernel[grid](
        quantized,
        scales,
        output,
        M,
        K,
        quantized.stride(0),
        quantized.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )

    return output


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _w8a8_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """W8A8 INT8 GEMM with per-tensor scaling.

    C = (A_int8 @ B_int8) * (scale_A * scale_B)
    """
    m_id = tl.program_id(0)
    n_id = tl.program_id(1)

    m_offs = m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = n_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load scales
    a_scales = tl.load(a_scale_ptr + m_offs, mask=m_offs < M, other=1.0)
    b_scale = tl.load(b_scale_ptr)  # Assuming per-tensor scale for weights

    # Accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

    # Matrix multiplication
    for k_start in tl.range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # Load A (activations)
        a_ptrs = a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.int8)

        # Load B (weights)
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn
        b_mask = (k_offs[:, None] < K) & (n_offs[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.int8)

        # Accumulate
        acc += tl.dot(a.to(tl.int32), b.to(tl.int32), allow_tf32=False)

    # Scale and store
    acc_f32 = acc.to(tl.float32)
    c = acc_f32 * a_scales[:, None] * b_scale

    # Store result
    c_ptrs = c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    c_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w8a8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """W8A8 INT8 matrix multiplication.

    Args:
        a: INT8 activations [M, K]
        b: INT8 weights [K, N]
        a_scales: Per-token scales [M]
        b_scale: Per-tensor weight scale (scalar)

    Returns:
        Output tensor [M, N] in FP16/BF16
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

    # Output buffer
    c = torch.empty(M, N, dtype=torch.float16, device=a.device)

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    _w8a8_gemm_kernel[grid](
        a, b, c,
        a_scales, b_scale,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c
