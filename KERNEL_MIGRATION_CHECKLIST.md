# LMDeploy Kernel Migration Checklist
## TurboMind CUDA → PyTorch Triton 移植清单

**生成日期**: 2025-11-24
**目的**: 将 TurboMind 高性能 CUDA kernels 移植到 PyTorch Triton，实现跨平台支持

---

## 📊 总体统计

| 类别 | TurboMind CUDA | PyTorch Triton | 覆盖率 | 优先级 |
|------|----------------|----------------|--------|--------|
| **Attention** | 45 个 .cu 文件 | 4 个 (flash, paged, alibi, mla) | ~40% | 🔴 高 |
| **GEMM/Linear** | 28 个 .cu 文件 | 5 个 (awq, w8a8, fp8, ep_moe) | ~30% | 🔴 高 |
| **Activation** | 2 个 .cu 文件 | 1 个 (silu_and_mul) | ✅ 50% | 🟡 中 |
| **Normalization** | 1 个 .cu 文件 | 1 个 (rms_norm) | ✅ 100% | ✅ 完成 |
| **Sampling** | 4 个 .cu 文件 | 1 个 (multinomial) | ~25% | 🟡 中 |
| **KV Cache** | 1 个 .cu 文件 | 2 个 (fill, flatten) | ✅ 200% | ✅ 完成 |
| **MoE** | 1 个 .cu 文件 | 4 个 (fused, w8a8, fp8, ep) | ✅ 400% | ✅ 完成 |
| **Utilities** | 7 个 .cu 文件 | 3 个 (rotary, lora, utils) | ~40% | 🟢 低 |

**总计**: 87 个 TurboMind CUDA 文件 vs 21 个 PyTorch Triton 文件

---

## 🔴 高优先级：需要移植的核心 Kernels

### 1. Attention Kernels (重要！)

#### ✅ **已完成**
- [x] Flash Attention (prefill)
  - 文件: `lmdeploy/pytorch/kernels/cuda/flashattention.py`
  - 支持: Causal, Sliding Window, Logit Capping

- [x] Paged Attention (decoding)
  - 文件: `lmdeploy/pytorch/kernels/cuda/pagedattention.py`
  - 支持: Block KV Cache

- [x] Alibi Paged Attention
  - 文件: `lmdeploy/pytorch/kernels/cuda/alibi_pagedattention.py`
  - 支持: ALiBi 位置编码

- [x] Flash MLA (DeepSeek-V2/V3)
  - 文件: `lmdeploy/pytorch/kernels/cuda/flash_mla.py`
  - 支持: Multi-head Latent Attention

#### ❌ **缺失 (需移植)**

| TurboMind CUDA | 功能 | 优先级 | 难度 | 预估工作量 |
|----------------|------|--------|------|-----------|
| `attention/decoding_sm*_*_u4.cu` (12个) | **KV Cache INT4 量化** | 🔴 高 | ⭐⭐⭐ | 2-3周 |
| `attention/decoding_sm*_*_u8.cu` (12个) | **KV Cache INT8 量化** | 🔴 高 | ⭐⭐ | 1-2周 |
| `attention/cp_utils.cu` | **Context Parallelism** | 🟡 中 | ⭐⭐⭐⭐ | 3-4周 |
| `attention/reduce.cu` | Attention Score Reduction | 🟡 中 | ⭐⭐ | 1周 |
| `attention/reference.cu` | Reference Implementation (测试用) | 🟢 低 | ⭐ | - |

**关键缺失**:
- 🚨 **KV Cache 量化 (U4/U8)**: TurboMind 有 24 个高度优化的量化 attention kernels，PyTorch 端几乎没有！
- 🚨 **Context Parallelism**: 长上下文优化，Triton 实现难度高

---

### 2. GEMM Kernels (重要！)

#### ✅ **已完成**
- [x] AWQ Linear (W4A16)
  - 文件: `lmdeploy/pytorch/kernels/cuda/awq_kernels.py`

- [x] W8A8 Triton Kernels
  - 文件: `lmdeploy/pytorch/kernels/cuda/w8a8_triton_kernels.py`

- [x] Blocked FP8 GEMM
  - 文件: `lmdeploy/pytorch/kernels/cuda/blocked_gemm_fp8.py`

- [x] EP MoE (Expert Parallel)
  - 文件: `lmdeploy/pytorch/kernels/cuda/ep_moe.py`

#### ❌ **缺失 (需移植)**

| TurboMind CUDA | 功能 | 优先级 | 难度 | 预估工作量 |
|----------------|------|--------|------|-----------|
| `gemm/kernel/sm70_*.cu` (3个) | **Volta (V100) 优化 GEMM** | 🟡 中 | ⭐⭐⭐ | 2-3周 |
| `gemm/kernel/sm75_*.cu` (3个) | **Turing (T4) 优化 GEMM** | 🟡 中 | ⭐⭐⭐ | 2-3周 |
| `gemm/kernel/sm80_*.cu` (3个) | **Ampere (A100) 优化 GEMM** | 🔴 高 | ⭐⭐⭐ | 2-3周 |
| `gemm/kernel/sm90_*.cu` (4个) | **Hopper (H100) 优化 GEMM** | 🔴 高 | ⭐⭐⭐⭐ | 4-6周 |
| `gemm/tma.cu` | **TMA (Tensor Memory Accelerator)** | 🔴 高 | ⭐⭐⭐⭐⭐ | 6-8周 |
| `gemm/convert_v3.cu` | Type Conversion (INT8/FP16/BF16) | 🟡 中 | ⭐⭐ | 1周 |
| `gemm/cast.cu` | Fast Type Casting | 🟡 中 | ⭐ | 3-5天 |
| `gemm/unpack.cu` | Weight Unpacking | 🟡 中 | ⭐⭐ | 1周 |

**关键缺失**:
- 🚨 **架构特定优化**: TurboMind 针对 SM70-90 有高度优化的 GEMM，Triton 缺乏这些
- 🚨 **TMA (H100 专用)**: Hopper 架构的 Tensor Memory Accelerator，Triton 支持有限
- ⚠️ **量化 GEMM**: W4A16 只有 AWQ，缺乏 GPTQ、SmoothQuant 等

---

### 3. Sampling Kernels

#### ✅ **已完成**
- [x] Multinomial Sampling
  - 文件: `lmdeploy/pytorch/kernels/cuda/multinomial_sampling.py`

#### ❌ **缺失 (需移植)**

| TurboMind CUDA | 功能 | 优先级 | 难度 | 预估工作量 |
|----------------|------|--------|------|-----------|
| `sampling_topk_kernels.cu` | **Top-K Sampling** | 🔴 高 | ⭐⭐ | 1-2周 |
| `sampling_topp_kernels.cu` | **Top-P (Nucleus) Sampling** | 🔴 高 | ⭐⭐⭐ | 2-3周 |
| `sampling_penalty_kernels.cu` | **Penalty Application** (repetition, frequency) | 🟡 中 | ⭐⭐ | 1周 |
| `sampling_kernels.cu` | Core Sampling Utils | 🟡 中 | ⭐⭐ | 1周 |

**关键缺失**:
- 🚨 **Top-K/Top-P**: 最常用的采样策略，PyTorch 端只有 multinomial，缺乏高性能实现
- ⚠️ **Penalty Kernels**: Repetition penalty, frequency penalty 等，需要融合到 sampling 中

---

## 🟡 中优先级：功能增强 Kernels

### 4. Activation Kernels

#### ✅ **已完成**
- [x] SiLU and Mul (Fused)
  - 文件: `lmdeploy/pytorch/kernels/cuda/activation.py`

#### ❌ **缺失 (需移植)**

| TurboMind CUDA | 功能 | 优先级 | 难度 | 预估工作量 |
|----------------|------|--------|------|-----------|
| `activation_kernels.cu` | **GELU and Mul** (Fused) | 🟡 中 | ⭐ | 3-5天 |
| `activation_kernels.cu` | **ReLU and Mul** (Fused) | 🟢 低 | ⭐ | 2-3天 |
| `activation.cu` | Generic Activation Framework | 🟡 中 | ⭐⭐ | 1周 |

**建议**:
- 扩展 `activation.py`，添加 `gelu_and_mul`, `relu_and_mul`
- 添加 `@triton.autotune` 自动调优

---

### 5. Decoding/Embedding Kernels

#### ❌ **缺失 (需移植)**

| TurboMind CUDA | 功能 | 优先级 | 难度 | 预估工作量 |
|----------------|------|--------|------|-----------|
| `decoding_kernels.cu` | **Embedding Lookup + Pos Encoding** | 🟡 中 | ⭐⭐ | 1周 |
| `gpt_kernels.cu` | **Embedding Lookup** (vectorized) | 🟡 中 | ⭐⭐ | 1周 |

**建议**:
- 融合 Embedding Lookup + RoPE/Alibi
- Triton 实现较简单，性能提升明显

---

### 6. Utility Kernels

#### ✅ **已完成**
- [x] Apply Rotary Position Embedding
  - 文件: `lmdeploy/pytorch/kernels/cuda/apply_rotary_pos_emb.py`

- [x] Fused LoRA
  - 文件: `lmdeploy/pytorch/kernels/cuda/fused_lora.py`

#### ❌ **缺失 (需移植)**

| TurboMind CUDA | 功能 | 优先级 | 难度 | 预估工作量 |
|----------------|------|--------|------|-----------|
| `quantization.cu` | **Online Activation Quantization** | 🟡 中 | ⭐⭐⭐ | 2-3周 |
| `logprob_kernels.cu` | **Log Probability Computation** | 🟡 中 | ⭐⭐ | 1周 |
| `ban_bad_words.cu` | **Bad Words Banning** | 🟢 低 | ⭐⭐ | 1周 |
| `stop_criteria_kernels.cu` | **Stopping Criteria** | 🟢 低 | ⭐ | 3-5天 |
| `apply_token_bitmask_inplace_cuda.cu` | Token Bitmask Application | 🟢 低 | ⭐ | 3-5天 |
| `unfused_attention_kernels.cu` | Unfused Attention (fallback) | 🟢 低 | ⭐⭐ | 1周 |

---

## 🟢 低优先级：测试/调优 Kernels

### 7. 测试和 Benchmark Kernels

这些 kernels 主要用于测试和性能调优，不需要移植到生产环境：

- `attention/test_attention.cu`
- `attention/test_quant.cu`
- `attention/test_utils.cu`
- `gemm/test/gemm_bench.cu`
- `gemm/test/quantization.cu`
- `gemm/test/reference.cu`
- `gemm/test/test_moe_utils.cu`
- `gemm/test/test_utils.cu`

**建议**: 使用 PyTorch 原生测试框架，不需要移植

---

### 8. Tuner Kernels

这些用于 auto-tuning，Triton 有内置的 `@triton.autotune`：

- `gemm/tuner/cache_utils.cu`
- `gemm/tuner/measurer.cu`
- `gemm/tuner/sampler.cu`

**建议**: 使用 Triton 自带的 auto-tuning 机制

---

## 📋 移植路线图

### **Phase 1: 基础 Kernels (1-2个月)**

| 任务 | 优先级 | 工作量 | 负责人 | 状态 |
|------|--------|--------|--------|------|
| Top-K Sampling | 🔴 高 | 1-2周 | - | ⬜ 待开始 |
| Top-P Sampling | 🔴 高 | 2-3周 | - | ⬜ 待开始 |
| GELU and Mul | 🟡 中 | 3-5天 | - | ⬜ 待开始 |
| Embedding Lookup + Pos Encoding | 🟡 中 | 1周 | - | ⬜ 待开始 |

**目标**: 补齐基础功能，达到 TurboMind 的功能完整性

---

### **Phase 2: 量化 Kernels (2-3个月)**

| 任务 | 优先级 | 工作量 | 负责人 | 状态 |
|------|--------|--------|--------|------|
| KV Cache INT8 Quantization | 🔴 高 | 1-2周 | - | ⬜ 待开始 |
| KV Cache INT4 Quantization | 🔴 高 | 2-3周 | - | ⬜ 待开始 |
| Online Activation Quantization | 🟡 中 | 2-3周 | - | ⬜ 待开始 |
| GPTQ/SmoothQuant Linear | 🟡 中 | 2-3周 | - | ⬜ 待开始 |

**目标**: 支持低精度推理，降低显存占用

---

### **Phase 3: 高性能优化 (3-6个月)**

| 任务 | 优先级 | 工作量 | 负责人 | 状态 |
|------|--------|--------|--------|------|
| SM80 (A100) 优化 GEMM | 🔴 高 | 2-3周 | - | ⬜ 待开始 |
| SM90 (H100) 优化 GEMM | 🔴 高 | 4-6周 | - | ⬜ 待开始 |
| TMA (Hopper) Support | 🔴 高 | 6-8周 | - | ⬜ 待开始 |
| Context Parallelism | 🟡 中 | 3-4周 | - | ⬜ 待开始 |

**目标**: 在新架构 (A100/H100) 上达到接近 TurboMind 的性能

---

## 🎯 移植优先级总结

### **立即开始 (本月)**
1. ✅ **Top-K/Top-P Sampling** - 最常用，影响用户体验
2. ✅ **KV Cache INT8 Quantization** - 降低显存，提高吞吐
3. ✅ **GELU and Mul** - 简单但常用

### **短期计划 (1-2个月)**
4. ✅ **KV Cache INT4 Quantization** - 进一步降低显存
5. ✅ **Embedding + Pos Encoding Fusion** - Prefill 性能优化
6. ✅ **Penalty Kernels** - 采样质量提升

### **中期计划 (2-4个月)**
7. ✅ **SM80/SM90 GEMM Optimization** - 新硬件性能优化
8. ✅ **Online Activation Quantization** - W8A8 推理
9. ✅ **Context Parallelism** - 长上下文支持

### **长期计划 (4-6个月)**
10. ✅ **TMA Support (H100)** - Hopper 架构极致优化
11. ✅ **GPTQ/SmoothQuant** - 更多量化方法
12. ✅ **Unfused Attention Fallback** - 兼容性

---

## 🔧 移植指南

### **移植流程**
1. **选择 Kernel**: 从上面的清单选择一个 TurboMind CUDA kernel
2. **阅读源码**: 理解 CUDA kernel 的功能和优化策略
3. **Triton 实现**: 用 Triton 重写，保持功能等价
4. **添加 Auto-tuning**: 使用 `@triton.autotune` 优化性能
5. **测试**: 单元测试 + 性能测试 (对比 CUDA 版本)
6. **集成**: 集成到 `lmdeploy/pytorch/kernels/cuda/`
7. **文档**: 更新本清单，标记为完成 ✅

### **代码模板**
```python
# lmdeploy/pytorch/kernels/cuda/new_kernel.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def my_kernel(...):
    # Triton kernel implementation
    pass

def my_kernel_launcher(input: torch.Tensor) -> torch.Tensor:
    # Python wrapper
    output = torch.empty_like(input)
    grid = lambda META: (triton.cdiv(input.size(0), META['BLOCK_SIZE']),)
    my_kernel[grid](input, output, ...)
    return output
```

### **性能测试**
```python
# benchmark/test_new_kernel.py
import torch
from lmdeploy.pytorch.kernels.cuda.new_kernel import my_kernel_launcher

def benchmark():
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(10):
        my_kernel_launcher(x)

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        my_kernel_launcher(x)
    end.record()
    torch.cuda.synchronize()

    print(f"Time: {start.elapsed_time(end) / 100:.3f} ms")
```

---

## 📈 性能目标

| Kernel 类型 | 目标性能 (vs TurboMind CUDA) |
|-------------|------------------------------|
| **简单 Elementwise** (activation, norm) | ≥ 95% |
| **中等复杂** (sampling, embedding) | ≥ 85% |
| **复杂 Attention** (flash, paged) | ≥ 80% |
| **高度优化 GEMM** | ≥ 70% (可接受) |

**说明**: Triton 的跨平台优势通常能弥补 5-20% 的性能差距

---

## 🤝 贡献指南

1. **Fork 此仓库**
2. **选择任务**: 从清单中选择一个 `⬜ 待开始` 的任务
3. **更新状态**: 将状态改为 `🔄 进行中`，填写负责人
4. **完成移植**: 按照移植流程完成
5. **提交 PR**: 提交到 lmdeploy 主仓库
6. **更新清单**: 将状态改为 `✅ 已完成`

---

## 📚 参考资源

- **TurboMind CUDA Kernels**: `/home/user/lmdeploy/src/turbomind/kernels/`
- **PyTorch Triton Kernels**: `/home/user/lmdeploy/lmdeploy/pytorch/kernels/cuda/`
- **Triton 文档**: https://triton-lang.org/
- **CUTLASS 文档**: https://github.com/NVIDIA/cutlass
- **vLLM Kernels** (参考): https://github.com/vllm-project/vllm

---

**维护者**: @your-github-username
**最后更新**: 2025-11-24

---

## ✅ 快速行动清单 (本周可完成)

### 1️⃣ **Top-K Sampling** (1-2周)
- **文件**: `src/turbomind/kernels/sampling_topk_kernels.cu`
- **目标**: `lmdeploy/pytorch/kernels/cuda/topk_sampling.py`
- **难度**: ⭐⭐ (中等)
- **功能**: 从 logits 中选择概率最高的 K 个 token

### 2️⃣ **Top-P Sampling** (2-3周)
- **文件**: `src/turbomind/kernels/sampling_topp_kernels.cu`
- **目标**: `lmdeploy/pytorch/kernels/cuda/topp_sampling.py`
- **难度**: ⭐⭐⭐ (较难)
- **功能**: Nucleus sampling，累积概率达到 P

### 3️⃣ **GELU and Mul** (3-5天)
- **文件**: `src/turbomind/kernels/activation_kernels.cu:56-78`
- **目标**: 扩展 `lmdeploy/pytorch/kernels/cuda/activation.py`
- **难度**: ⭐ (简单)
- **功能**: Fused GELU activation + elementwise multiply

开始吧！🚀
