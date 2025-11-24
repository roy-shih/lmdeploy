#!/usr/bin/env python3
"""Test script for GELU and Mul kernel."""
import torch
import torch.nn.functional as F
import sys

# Add lmdeploy to path
sys.path.insert(0, '/home/user/lmdeploy')

from lmdeploy.pytorch.kernels.cuda.activation import gelu_and_mul


def test_gelu_and_mul_correctness():
    """Test correctness of gelu_and_mul kernel."""
    print("=" * 80)
    print("Testing GELU and Mul Kernel Correctness")
    print("=" * 80)

    # Test configurations
    configs = [
        (128, 256),    # Small
        (1024, 2048),  # Medium
        (4096, 4096),  # Large
    ]

    for M, N in configs:
        print(f"\nTesting shape: M={M}, N={N}")

        # Create input (gate_up has shape [M, 2*N])
        gate_up = torch.randn(M, 2 * N, device='cuda', dtype=torch.float16)
        gate, up = gate_up.chunk(2, dim=-1)

        # Reference implementation (PyTorch)
        with torch.no_grad():
            expected = F.gelu(gate.to(torch.float32)) * up.to(torch.float32)
            expected = expected.to(torch.float16)

        # Triton kernel
        with torch.no_grad():
            actual = gelu_and_mul(gate_up)

        # Check correctness
        max_diff = (actual - expected).abs().max().item()
        mean_diff = (actual - expected).abs().mean().item()
        relative_error = ((actual - expected).abs() / (expected.abs() + 1e-5)).mean().item()

        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        print(f"  Relative error: {relative_error:.6f}")

        # Check if close enough
        try:
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
            print(f"  ✅ PASSED")
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            return False

    print("\n" + "=" * 80)
    print("✅ All correctness tests PASSED!")
    print("=" * 80)
    return True


def benchmark_gelu_and_mul():
    """Benchmark gelu_and_mul kernel."""
    print("\n" + "=" * 80)
    print("Benchmarking GELU and Mul Kernel")
    print("=" * 80)

    # Benchmark configuration
    M, N = 4096, 4096
    num_warmup = 10
    num_iters = 100

    print(f"\nShape: M={M}, N={N}")
    print(f"Warmup iterations: {num_warmup}")
    print(f"Benchmark iterations: {num_iters}")

    gate_up = torch.randn(M, 2 * N, device='cuda', dtype=torch.float16)
    gate, up = gate_up.chunk(2, dim=-1)

    # Warmup
    for _ in range(num_warmup):
        _ = F.gelu(gate) * up
        _ = gelu_and_mul(gate_up)

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        _ = F.gelu(gate) * up
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / num_iters

    # Benchmark Triton
    start.record()
    for _ in range(num_iters):
        _ = gelu_and_mul(gate_up)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / num_iters

    print(f"\nResults:")
    print(f"  PyTorch (unfused):  {pytorch_time:.3f} ms")
    print(f"  Triton (fused):     {triton_time:.3f} ms")
    print(f"  Speedup:            {pytorch_time / triton_time:.2f}x")

    if triton_time < pytorch_time:
        print(f"  ✅ Triton is {pytorch_time / triton_time:.2f}x faster!")
    else:
        print(f"  ⚠️  PyTorch is {triton_time / pytorch_time:.2f}x faster")

    print("=" * 80)


if __name__ == '__main__':
    # Test correctness
    success = test_gelu_and_mul_correctness()

    if success:
        # Benchmark performance
        try:
            benchmark_gelu_and_mul()
        except Exception as e:
            print(f"\n⚠️  Benchmark failed: {e}")
            print("But correctness tests passed, so the kernel is functional!")
