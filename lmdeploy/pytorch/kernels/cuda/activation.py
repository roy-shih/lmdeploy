# Copyright (c) UnieAI. All rights reserved.
"""
Fused activation(gate) * up kernel (production version)

- gate_up: [M, 2N] where gate = [:, :N], up = [:, N:]
- out    : [M, N]  = act(gate) * up

支援兩種 activation:
- SiLU: act = x * sigmoid(x)
- GELU: 常見 tanh 近似公式
"""

import torch
import triton
import triton.language as tl
from packaging import version
from typing import Optional

from .utils import get_device_props

TRITON_VERSION = version.parse(triton.__version__)

if TRITON_VERSION >= version.parse("3.0.0"):
    fast_expf = tl.math.exp
else:
    fast_expf = tl.math.fast_expf

# Activation types (compile-time constants)
ACT_SILU = 0
ACT_GELU = 1


@triton.jit
def _act_and_mul_kernel(
    gateup_ptr,
    out_ptr,
    N: tl.constexpr,
    M,
    stride_gum: tl.constexpr,
    stride_gun: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACT: tl.constexpr,
):
    """
    Fused activation(gate) * up kernel.

    Layout:
        gate_up: [M, 2N] where gate = [:, :N], up = [:, N:]
        out    : [M, N]

    Optimizations:
        - 使用 base pointer 給 up（只算一次 N * stride_gun）
        - 預先計算 pointer step（跨 row 時只做一次加法）
        - 向量化 load/store + 適當 mask
        - fast_expf / tanh 近似
    """
    # block index along N dimension
    n_block_id = tl.program_id(0)
    # starting row index for this program along M dimension
    m_id_start = tl.program_id(1)
    m_id_stride = tl.num_programs(1)

    # feature indices handled by this block
    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # mask for boundary
    if N % BLOCK_SIZE_N == 0:
        mask = None
    else:
        mask = offs_n < N

    # base pointer of up (整體往後 N 個 element)
    up_base = gateup_ptr + N * stride_gun

    # initial pointers for this program
    gate_ptrs = gateup_ptr + m_id_start * stride_gum + offs_n * stride_gun
    up_ptrs = up_base + m_id_start * stride_gum + offs_n * stride_gun
    out_ptrs = out_ptr + m_id_start * stride_om + offs_n * stride_on

    # precomputed step for strided loop over M
    ptr_step = m_id_stride * stride_gum
    out_ptr_step = m_id_stride * stride_om

    # strided loop over rows (M dimension)
    for _ in tl.range(m_id_start, M, m_id_stride):
        if mask is None:
            gate = tl.load(gate_ptrs)
            up = tl.load(up_ptrs)
        else:
            gate = tl.load(gate_ptrs, mask=mask, other=0.0)
            up = tl.load(up_ptrs, mask=mask, other=0.0)

        # fp32 for activation
        gate_f32 = gate.to(tl.float32)
        up_f32 = up.to(tl.float32)

        # activation
        if ACT == ACT_SILU:
            # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            x = gate_f32
            activated = x / (1.0 + fast_expf(-x))
        else:
            # GELU(x) ≈ x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
            x = gate_f32
            x_cubed = x * x * x
            tanh_arg = 0.7978845608028654 * (x + 0.044715 * x_cubed)
            activated = x * 0.5 * (1.0 + tl.math.tanh(tanh_arg))

        out_val = activated * up_f32

        if mask is None:
            tl.store(out_ptrs, out_val)
        else:
            tl.store(out_ptrs, out_val, mask=mask)

        # move pointers to next row (stride over M)
        gate_ptrs += ptr_step
        up_ptrs += ptr_step
        out_ptrs += out_ptr_step


def _launch_act_and_mul(
    gate_up: torch.Tensor,
    out: Optional[torch.Tensor],
    act: int,
) -> torch.Tensor:
    """Shared launcher for SiLU / GELU fused kernels."""
    assert gate_up.dim() == 2, "gate_up must be 2D [M, 2N]"
    assert gate_up.is_cuda, "Triton kernel only supports CUDA tensors"

    M, twoN = gate_up.shape
    assert twoN % 2 == 0, "gate_up last dim must be even (2 * N)"
    N = twoN // 2

    # contiguous is friendlier for Triton, strides still respected
    gate_up = gate_up.contiguous()

    if out is None:
        out = gate_up.new_empty((M, N))
    else:
        assert out.shape == (M, N), f"out shape {out.shape} != ({M}, {N})"
        out = out.contiguous()

    # block size: next power-of-2 of N, up to 512
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, 512)

    # simple heuristic for warps
    if BLOCK_SIZE_N >= 256:
        num_warps = 8
    elif BLOCK_SIZE_N >= 128:
        num_warps = 4
    else:
        num_warps = 2

    # memory-bound elementwise: 2 stages 通常就夠好
    num_stages = 2

    props = get_device_props(gate_up.device.index)
    num_sm = props["multi_processor_count"]
    warps_per_sm = props["warps_per_sm"]

    grid_size0 = triton.cdiv(N, BLOCK_SIZE_N)  # blocks along N
    grid_size1 = min(M, num_sm * warps_per_sm // num_warps)  # programs along M

    assert grid_size0 < 65536 and grid_size1 < 65536, "Grid size overflow"
    grid = (grid_size0, grid_size1)

    _act_and_mul_kernel[grid](
        gate_up,
        out,
        N,
        M,
        stride_gum=gate_up.stride(0),
        stride_gun=gate_up.stride(1),
        stride_om=out.stride(0),
        stride_on=out.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        ACT=act,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out


def silu_and_mul(
    gate_up: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused SiLU(gate) * up.

    Args:
        gate_up: [M, 2N]，前半為 gate，後半為 up
        out    : [M, N]，可選；若為 None 則內部分配

    Returns:
        out: [M, N] = SiLU(gate) * up
    """
    return _launch_act_and_mul(gate_up, out, ACT_SILU)


def gelu_and_mul(
    gate_up: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused GELU(gate) * up.

    Args:
        gate_up: [M, 2N]，前半為 gate，後半為 up
        out    : [M, N]，可選；若為 None 則內部分配

    Returns:
        out: [M, N] = GELU(gate) * up
    """
    return _launch_act_and_mul(gate_up, out, ACT_GELU)