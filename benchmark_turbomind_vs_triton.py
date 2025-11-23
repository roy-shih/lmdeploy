#!/usr/bin/env python3
"""Benchmark TurboMind CUDA kernels vs PyTorch Triton kernels.

This script compares the performance of:
1. TurboMind's native CUDA kernels
2. PyTorch Engine's Triton kernels
3. PyTorch native operations
"""

import torch
import time
import argparse
from typing import Callable, Dict, List
import numpy as np


def benchmark_function(
    func: Callable,
    *args,
    warmup: int = 10,
    iterations: int = 100,
    **kwargs
) -> Dict[str, float]:
    """Benchmark a function and return timing statistics."""
    # Warmup
    for _ in range(warmup):
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
    }


def benchmark_silu_and_mul(batch_size: int = 32, seq_len: int = 2048, dim: int = 4096):
    """Benchmark SiluAndMul implementations."""
    print(f"\n{'='*80}")
    print(f"Benchmarking SiluAndMul: batch_size={batch_size}, seq_len={seq_len}, dim={dim}")
    print(f"{'='*80}")

    # Prepare input
    device = torch.device('cuda')
    gate_up = torch.randn(batch_size * seq_len, dim * 2, device=device, dtype=torch.float16)

    results = {}

    # 1. PyTorch native implementation
    def pytorch_native(gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    print("\n[1/3] Benchmarking PyTorch native...")
    results['PyTorch Native'] = benchmark_function(pytorch_native, gate_up)

    # 2. PyTorch Triton implementation
    try:
        from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul
        print("[2/3] Benchmarking PyTorch Triton...")
        results['Triton (PyTorch Engine)'] = benchmark_function(silu_and_mul, gate_up)
    except Exception as e:
        print(f"[2/3] Triton implementation not available: {e}")

    # 3. TurboMind implementation
    try:
        from lmdeploy.turbomind import TurboMind
        print("[3/3] TurboMind implementation requires full model context")
        print("       Skipping direct kernel comparison (use full inference benchmark)")
    except Exception as e:
        print(f"[3/3] TurboMind not available: {e}")

    # Print results
    print(f"\n{'Results':-^80}")
    print(f"{'Implementation':<30} {'Mean (ms)':>12} {'Std (ms)':>12} {'Speedup':>12}")
    print(f"{'-'*80}")

    baseline = results['PyTorch Native']['mean']
    for impl, stats in results.items():
        speedup = baseline / stats['mean']
        print(f"{impl:<30} {stats['mean']:>12.4f} {stats['std']:>12.4f} {speedup:>12.2f}x")


def benchmark_rms_norm(batch_size: int = 32, seq_len: int = 2048, hidden_dim: int = 4096):
    """Benchmark RMSNorm implementations."""
    print(f"\n{'='*80}")
    print(f"Benchmarking RMSNorm: batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
    print(f"{'='*80}")

    device = torch.device('cuda')
    x = torch.randn(batch_size * seq_len, hidden_dim, device=device, dtype=torch.float16)
    weight = torch.randn(hidden_dim, device=device, dtype=torch.float16)
    eps = 1e-6

    results = {}

    # 1. PyTorch native implementation
    def pytorch_native(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        return weight * x_normed

    print("\n[1/3] Benchmarking PyTorch native...")
    results['PyTorch Native'] = benchmark_function(pytorch_native, x, weight, eps)

    # 2. PyTorch Triton implementation
    try:
        from lmdeploy.pytorch.kernels.cuda.rms_norm import rms_norm
        print("[2/3] Benchmarking PyTorch Triton...")
        results['Triton (PyTorch Engine)'] = benchmark_function(rms_norm, x, weight, eps)
    except Exception as e:
        print(f"[2/3] Triton implementation not available: {e}")

    # 3. PyTorch native with residual
    def pytorch_native_residual(x, weight, eps, residual):
        new_x = x + residual
        variance = new_x.pow(2).mean(-1, keepdim=True)
        x_normed = new_x * torch.rsqrt(variance + eps)
        return weight * x_normed, new_x

    residual = torch.randn_like(x)
    print("[3/3] Benchmarking PyTorch native (with residual)...")
    results['PyTorch Native + Residual'] = benchmark_function(
        pytorch_native_residual, x, weight, eps, residual
    )

    # 4. Triton with residual
    try:
        from lmdeploy.pytorch.kernels.cuda.rms_norm import rms_norm
        print("[4/4] Benchmarking Triton (with residual)...")
        out_residual = torch.empty_like(x)
        results['Triton + Residual'] = benchmark_function(
            rms_norm, x, weight, eps, residual=residual, out_residual=out_residual
        )
    except Exception as e:
        print(f"[4/4] Triton residual fusion not available: {e}")

    # Print results
    print(f"\n{'Results':-^80}")
    print(f"{'Implementation':<30} {'Mean (ms)':>12} {'Std (ms)':>12} {'Speedup':>12}")
    print(f"{'-'*80}")

    baseline = results['PyTorch Native']['mean']
    for impl, stats in results.items():
        speedup = baseline / stats['mean']
        print(f"{impl:<30} {stats['mean']:>12.4f} {stats['std']:>12.4f} {speedup:>12.2f}x")


def benchmark_memory_bandwidth():
    """Calculate theoretical memory bandwidth utilization."""
    print(f"\n{'='*80}")
    print("Memory Bandwidth Analysis")
    print(f"{'='*80}")

    device = torch.device('cuda')

    # Get GPU memory bandwidth (this is approximate)
    # For accurate results, use nvidia-smi or CUDA runtime API
    gpu_name = torch.cuda.get_device_name(device)
    print(f"\nGPU: {gpu_name}")

    # Theoretical bandwidth for common GPUs (GB/s)
    bandwidth_map = {
        'A100': 1555,  # A100 PCIe 40GB
        'A100-SXM4': 2039,  # A100 SXM4 80GB
        'H100': 3350,  # H100 SXM5
        'V100': 900,   # V100 PCIe 32GB
        'A6000': 768,  # RTX A6000
        '4090': 1008,  # RTX 4090
    }

    theoretical_bw = None
    for key, bw in bandwidth_map.items():
        if key in gpu_name:
            theoretical_bw = bw
            break

    if theoretical_bw:
        print(f"Theoretical Memory Bandwidth: {theoretical_bw} GB/s")
    else:
        print("Theoretical bandwidth unknown for this GPU")
        theoretical_bw = 1000  # Default estimate

    # Calculate actual bandwidth for a simple kernel
    print("\nMeasuring actual memory bandwidth...")
    size = 1024 * 1024 * 256  # 256M elements
    x = torch.randn(size, device=device, dtype=torch.float32)
    y = torch.empty_like(x)

    # Simple copy kernel
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        y.copy_(x)
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = (end - start) / 100
    bytes_transferred = size * 4 * 2  # read + write, 4 bytes per float32
    actual_bw = (bytes_transferred / elapsed) / 1e9

    print(f"Measured Memory Bandwidth: {actual_bw:.2f} GB/s")
    print(f"Efficiency: {(actual_bw / theoretical_bw * 100):.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Benchmark TurboMind vs Triton kernels')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--hidden-dim', type=int, default=4096, help='Hidden dimension')
    parser.add_argument('--test', choices=['all', 'silu', 'rms_norm', 'bandwidth'],
                        default='all', help='Which test to run')

    args = parser.parse_args()

    print(f"\n{'LMDeploy Kernel Benchmark':=^80}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")

    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available. This benchmark requires CUDA.")
        return

    if args.test in ['all', 'silu']:
        benchmark_silu_and_mul(args.batch_size, args.seq_len, args.hidden_dim)

    if args.test in ['all', 'rms_norm']:
        benchmark_rms_norm(args.batch_size, args.seq_len, args.hidden_dim)

    if args.test in ['all', 'bandwidth']:
        benchmark_memory_bandwidth()

    print(f"\n{'Benchmark Complete':=^80}\n")


if __name__ == '__main__':
    main()
