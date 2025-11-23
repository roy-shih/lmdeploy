#!/usr/bin/env python3
"""
示例：如何将 TurboMind CUDA kernel 移植到 Triton 跨平台实现

这个示例展示了完整的移植流程：
1. 分析原始 CUDA kernel
2. 编写等价的 Triton kernel
3. 性能对比和验证
4. 跨平台部署
"""

import torch
import triton
import triton.language as tl
import time
from typing import Optional


# ============================================================================
# 示例 1: 简单的 Element-wise Kernel - GeGLU
# ============================================================================

# 原始 CUDA 伪代码（类似 TurboMind 的 activation.cu）:
"""
template<class T>
__global__ void GeGLUKernel(T* gate_up, int64_t stride, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        T gate = gate_up[idx];
        T up = gate_up[idx + N];

        // GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        T x = gate;
        T x3 = x * x * x;
        T tanh_arg = 0.79788456f * (x + 0.044715f * x3);
        T gelu = 0.5f * x * (1.0f + tanhf(tanh_arg));

        gate_up[idx] = gelu * up;
    }
}
"""

# Triton 移植实现
@triton.jit
def geglu_kernel(
    gate_up_ptr,
    output_ptr,
    N: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    GeGLU: GELU(gate) * up

    输入: gate_up [M, 2*N] - 前半部分是 gate，后半部分是 up
    输出: output [M, N]
    """
    # 1. 获取当前 block 的 program ID
    pid = tl.program_id(0)

    # 2. 计算此 block 处理的元素范围
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # 3. 加载 gate 和 up
    gate = tl.load(gate_up_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(gate_up_ptr + offsets + stride, mask=mask, other=0.0)

    # 4. GELU activation (跨平台实现)
    x = gate.to(tl.float32)
    x3 = x * x * x
    tanh_arg = 0.79788456 * (x + 0.044715 * x3)

    # Triton 的 tanh 是跨平台的
    gelu = 0.5 * x * (1.0 + tl.math.tanh(tanh_arg))

    # 5. 乘以 up
    up_f32 = up.to(tl.float32)
    result = gelu * up_f32

    # 6. 存储结果
    tl.store(output_ptr + offsets, result.to(gate.dtype), mask=mask)


def geglu_triton(gate_up: torch.Tensor) -> torch.Tensor:
    """
    Triton 版本的 GeGLU（跨平台）

    支持的平台：
    - NVIDIA GPU (CUDA)
    - AMD GPU (ROCm)
    - Intel GPU (XPU, 实验性)
    """
    assert gate_up.dim() == 2
    M, two_N = gate_up.shape
    N = two_N // 2

    output = torch.empty(M, N, device=gate_up.device, dtype=gate_up.dtype)

    # 对每一行并行处理
    for row_idx in range(M):
        row_gate_up = gate_up[row_idx]
        row_output = output[row_idx]

        BLOCK_SIZE = triton.next_power_of_2(N)
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)

        grid = (triton.cdiv(N, BLOCK_SIZE),)

        geglu_kernel[grid](
            row_gate_up,
            row_output,
            N=N,
            stride=N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    return output


def geglu_pytorch(gate_up: torch.Tensor) -> torch.Tensor:
    """PyTorch 原生实现（参考）"""
    gate, up = gate_up.chunk(2, dim=-1)
    return torch.nn.functional.gelu(gate) * up


# ============================================================================
# 示例 2: 带 Reduction 的 Kernel - LayerNorm
# ============================================================================

@triton.jit
def layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,  # 可选：存储 mean 用于调试
    rstd_ptr,  # 可选：存储 rstd 用于调试
    M,  # batch size
    N: tl.constexpr,  # hidden dim
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    跨平台的 LayerNorm 实现

    对应 TurboMind 的 rms_norm.cu，但这里实现的是完整的 LayerNorm：
    y = (x - mean) / sqrt(var + eps) * weight + bias
    """
    # 每个 program 处理一行
    row_idx = tl.program_id(0)

    if row_idx >= M:
        return

    # 计算当前行的起始位置
    row_start = row_idx * N

    # 加载整行数据（如果 N 很大，需要分块处理）
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Step 1: 计算 mean
    x = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0)
    x_sum = tl.sum(x, axis=0)
    mean = x_sum / N

    # Step 2: 计算 variance
    x_centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(variance + eps)

    # Step 3: 归一化
    x_normed = x_centered * rstd

    # Step 4: 应用 affine transformation
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    output = x_normed * weight + bias

    # Step 5: 存储结果
    tl.store(output_ptr + row_start + offsets, output, mask=mask)

    # 可选：存储统计信息
    if mean_ptr is not None:
        if tl.program_id(0) < M and offsets[0] == 0:
            tl.store(mean_ptr + row_idx, mean)
    if rstd_ptr is not None:
        if tl.program_id(0) < M and offsets[0] == 0:
            tl.store(rstd_ptr + row_idx, rstd)


# ============================================================================
# 示例 3: 融合 Kernel - Bias + Residual + RMSNorm
# ============================================================================

@triton.jit
def fused_bias_residual_rmsnorm_kernel(
    input_ptr,
    bias_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    out_residual_ptr,
    M,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合操作：Bias + Residual + RMSNorm

    等价于：
    1. x = input + bias
    2. x = x + residual
    3. output = RMSNorm(x) * weight

    这种融合减少了 3 次内存往返！
    """
    row_idx = tl.program_id(0)

    if row_idx >= M:
        return

    row_start = row_idx * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load all inputs in one go
    x = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + row_start + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)

    # Fused computation (全部在寄存器中完成)
    x = x + bias  # 1. 加 bias
    x = x + residual  # 2. 加 residual

    # 保存新的 residual
    tl.store(out_residual_ptr + row_start + offsets, x, mask=mask)

    # 3. RMSNorm
    x_f32 = x.to(tl.float32)
    square_sum = tl.sum(x_f32 * x_f32, axis=0)
    rms = tl.sqrt(square_sum / N + eps)
    x_normed = x_f32 / rms

    # 4. 乘以权重
    output = x_normed * weight.to(tl.float32)

    # Store output
    tl.store(output_ptr + row_start + offsets, output.to(x.dtype), mask=mask)


def fused_bias_residual_rmsnorm(
    input: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    跨平台的融合 Bias + Residual + RMSNorm

    返回：(output, new_residual)
    """
    M, N = input.shape

    output = torch.empty_like(input)
    out_residual = torch.empty_like(input)

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    fused_bias_residual_rmsnorm_kernel[grid](
        input, bias, residual, weight, output, out_residual,
        M, N, eps, BLOCK_SIZE,
        num_warps=4,
    )

    return output, out_residual


# ============================================================================
# 测试和性能对比
# ============================================================================

def test_correctness():
    """测试 Triton kernel 的正确性"""
    print("=" * 80)
    print("正确性测试")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test GeGLU
    print("\n[1] 测试 GeGLU...")
    gate_up = torch.randn(32, 8192, device=device, dtype=torch.float16)

    out_pytorch = geglu_pytorch(gate_up)
    if device.type == 'cuda':
        out_triton = geglu_triton(gate_up)

        # 验证结果
        max_diff = (out_pytorch - out_triton).abs().max().item()
        mean_diff = (out_pytorch - out_triton).abs().mean().item()

        print(f"   Max diff: {max_diff:.6f}")
        print(f"   Mean diff: {mean_diff:.6f}")
        print(f"   ✓ PASS" if max_diff < 1e-2 else "   ✗ FAIL")
    else:
        print("   CUDA not available, skipping Triton test")

    print("\n测试完成！")


def benchmark_performance():
    """性能对比"""
    print("\n" + "=" * 80)
    print("性能测试")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device = torch.device('cuda')

    # Benchmark GeGLU
    print("\n[GeGLU Performance]")
    sizes = [(32, 4096), (64, 4096), (128, 4096), (32, 8192)]

    for batch_size, dim in sizes:
        gate_up = torch.randn(batch_size, dim * 2, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(10):
            _ = geglu_pytorch(gate_up)
            _ = geglu_triton(gate_up)

        # PyTorch benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = geglu_pytorch(gate_up)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / 100 * 1000

        # Triton benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = geglu_triton(gate_up)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / 100 * 1000

        speedup = pytorch_time / triton_time
        print(f"  Shape [{batch_size:3d}, {dim:5d}*2]: "
              f"PyTorch {pytorch_time:6.3f}ms | "
              f"Triton {triton_time:6.3f}ms | "
              f"Speedup {speedup:5.2f}x")


def demonstrate_cross_platform():
    """展示跨平台能力"""
    print("\n" + "=" * 80)
    print("跨平台演示")
    print("=" * 80)

    # 检测可用设备
    available_devices = []

    if torch.cuda.is_available():
        available_devices.append('cuda')
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")

    # 注意：这些需要相应的 PyTorch 版本和硬件
    # if torch.backends.mps.is_available():
    #     available_devices.append('mps')
    #     print("✓ MPS (Apple Silicon) available")

    # if hasattr(torch, 'xpu') and torch.xpu.is_available():
    #     available_devices.append('xpu')
    #     print("✓ Intel XPU available")

    if not available_devices:
        print("No GPU devices available")
        return

    print(f"\n可以在以下平台运行 Triton kernels：")
    for device in available_devices:
        print(f"  - {device}")

    print("\n跨平台使用示例：")
    print("""
    # 代码无需修改，自动适配平台
    input = torch.randn(32, 4096, device='cuda')  # 或 'xpu', 'mps'
    output = geglu_triton(input)  # Triton 自动编译为目标平台
    """)


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  TurboMind → Triton Kernel 移植示例                                      ║
    ║                                                                          ║
    ║  本示例展示如何将 CUDA kernels 移植到 Triton 跨平台实现                 ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # 运行测试
    test_correctness()
    benchmark_performance()
    demonstrate_cross_platform()

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
    ✅ Triton kernels 可以达到接近手写 CUDA 的性能
    ✅ 代码量减少 50%+，开发效率大幅提升
    ✅ 自动支持多个平台（CUDA, ROCm, XPU）
    ✅ 内建 auto-tuning，自动优化性能

    建议：
    1. 新的 kernels 直接用 Triton 编写
    2. 现有的简单 CUDA kernels 可以移植到 Triton
    3. 复杂的 GEMM 操作使用 PyTorch 原生或 CUTLASS
    """)


if __name__ == '__main__':
    main()
