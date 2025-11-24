# TurboMind Kernel Inventory & Triton Porting Status

This document tracks the status of porting TurboMind's CUDA kernels to Triton for the PyTorch engine.

## Overview

*   **TurboMind Kernels (CUDA)**: Located in `src/turbomind/kernels`. Total files: ~90+.
*   **PyTorch Kernels (Triton)**: Located in `lmdeploy/pytorch/kernels/cuda`. Total files: ~20+.

## Kernel Mapping & Status

| Category | TurboMind Kernel (CUDA) | Functionality | Triton Port (PyTorch Engine) | Checked | Status / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Attention** | `attention/attention.cu` | Core FlashAttention implementation | `flashattention.py` | | ✅ **Done** (Supports FlashAttention-2) |
| | `attention/decoding.cu` | Decoding phase attention (paged) | `pagedattention.py` | | ✅ **Done** (PagedAttention) |
| | `attention/kv_cache_utils_v2.cu` | KV Cache management | `fill_kv_cache.py`, `flatten_kv_cache.py` | | ✅ **Done** |
| | `attention/flash_mla.py` (CUDA) | Multi-Head Latent Attention | `flash_mla.py` | | ⚠️ **Partial** (New architecture support) |
| **Normalization** | `norm/rms_norm.cu` | RMSNorm | `rms_norm.py` | ✅ | ✅ **Done** (Added Bias+Residual fusion) |
| **Activation** | `activation_kernels.cu` | Gelu, Silu, Relu | `activation.py` | ✅ | ✅ **Done** (Aligned with TurboMind) |
| **GEMM / Linear** | `gemm/*` | Matrix Multiplication (Int8/Int4) | `w8a8_triton_kernels.py`, `awq_kernels.py` | | ✅ **Done** (AWQ, W8A8 supported) |
| **MoE** | `gemm/fused_moe_gemm_kernels.cu` | Mixture of Experts Fused GEMM | `fused_moe.py`, `w8a8_fused_moe.py` | | ✅ **Done** |
| **Sampling** | `sampling_kernels.cu` | Top-K, Top-P sampling | `multinomial_sampling.py` | | ⚠️ **Partial** (Triton sampling is complex) |
| | `sampling_penalty_kernels.cu` | Repetition Penalty | `sampling_penalty.py` | ✅ | ✅ **Done** (Triton kernel with fallback) |
| **Positional Emb** | `rotary_embedding.cu` (implied) | Rotary Positional Embedding | `apply_rotary_pos_emb.py` | ✅ | ⚠️ **Partial** (Missing Llama 3 / Yarn support) |
| **Quantization** | `quantization.cu` | Weight-only quantization utils | `w8a8_kernels.py` | | ✅ **Done** |
| **Misc** | `apply_token_bitmask_inplace_cuda.cu` | Masking operations | N/A | | ❌ **Missing** |
| | `ban_bad_words.cu` | Generation constraints | `ban_bad_words.py` | ✅ | ✅ **Done** (Single-token banning) |
| | `stop_criteria_kernels.cu` | Stop criteria check | `stop_criteria.py` | ✅ | ✅ **Done** (Length + Stop words) |

## High-Priority Porting Candidates

Based on the analysis, the following kernels offer the best return on investment for porting:

1.  **Fused RMSNorm (Bias + Residual)**:
    *   **TurboMind**: `BiasResidualRMSNormKernel` in `rms_norm.cu`.
    *   **Current Triton**: `rms_norm.py` only supports Residual + RMSNorm.
    *   **Benefit**: Reduces memory bandwidth usage by fusing bias addition.

2.  **Sampling Kernels (Penalty, Top-K/P)**:
    *   **TurboMind**: Highly optimized fused sampling kernels.
    *   **Current Triton**: Often relies on standard PyTorch ops which can be slower due to kernel launch overhead.
    *   **Benefit**: End-to-end latency reduction, especially for small batch sizes.

3.  **Fused Linear + Activation (SwiGLU)**:
    *   **TurboMind**: Often fused in `LlamaFfnLayer`.
    *   **Current Triton**: `activation.py` exists, but fusing it with the preceding Linear (GEMM) output de-quantization would be faster.

## Next Steps

1.  Implement **Fused RMSNorm** (Bias + Residual) using the provided Triton draft.
2.  Investigate **Sampling Kernel** porting feasibility in Triton (challenging due to sort/scan requirements).
3.  Benchmark existing Triton kernels against TurboMind CUDA kernels to identify performance gaps.
