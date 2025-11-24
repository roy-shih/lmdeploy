# Kernel 移植快速参考表

## 🎯 优先级排序 (按重要性)

| # | Kernel 名称 | TurboMind 源文件 | 目标文件 | 优先级 | 难度 | 工作量 | 状态 |
|---|------------|------------------|----------|--------|------|--------|------|
| 1 | **Top-K Sampling** | `sampling_topk_kernels.cu` | `cuda/topk_sampling.py` | 🔴 | ⭐⭐ | 1-2周 | ✅ |
| 2 | **Top-P Sampling** | `sampling_topp_kernels.cu` | `cuda/topp_sampling.py` | 🔴 | ⭐⭐⭐ | 2-3周 | ✅ |
| 3 | **KV Cache INT8 Quant** | `attention/decoding_*_u8.cu` (12个) | `cuda/kv_cache_quant_int8.py` | 🔴 | ⭐⭐ | 1-2周 | ⬜ |
| 4 | **KV Cache INT4 Quant** | `attention/decoding_*_u4.cu` (12个) | `cuda/kv_cache_quant_int4.py` | 🔴 | ⭐⭐⭐ | 2-3周 | ⬜ |
| 5 | **SM80 GEMM (A100)** | `gemm/kernel/sm80_*.cu` (3个) | `cuda/gemm_sm80.py` | 🔴 | ⭐⭐⭐ | 2-3周 | ⬜ |
| 6 | **SM90 GEMM (H100)** | `gemm/kernel/sm90_*.cu` (4个) | `cuda/gemm_sm90.py` | 🔴 | ⭐⭐⭐⭐ | 4-6周 | ⬜ |
| 7 | **TMA (H100)** | `gemm/tma.cu` | `cuda/tma.py` | 🔴 | ⭐⭐⭐⭐⭐ | 6-8周 | ⬜ |
| 8 | **GELU and Mul** | `activation_kernels.cu` | `cuda/activation.py` (扩展) | 🟡 | ⭐ | 3-5天 | ✅ |
| 9 | **Penalty Kernels** | `sampling_penalty_kernels.cu` | `cuda/sampling_penalty.py` | 🟡 | ⭐⭐ | 1周 | ⬜ |
| 10 | **Embedding + Pos Enc** | `decoding_kernels.cu` | `cuda/embedding_lookup.py` | 🟡 | ⭐⭐ | 1周 | ✅ |
| 11 | **Context Parallelism** | `attention/cp_utils.cu` | `cuda/context_parallel.py` | 🟡 | ⭐⭐⭐⭐ | 3-4周 | ⬜ |
| 12 | **Online Quant** | `quantization.cu` | `cuda/online_quant.py` | 🟡 | ⭐⭐⭐ | 2-3周 | ⬜ |
| 13 | **Type Convert** | `gemm/convert_v3.cu` | `cuda/type_convert.py` | 🟡 | ⭐⭐ | 1周 | ⬜ |
| 14 | **Log Probability** | `logprob_kernels.cu` | `cuda/logprob.py` | 🟡 | ⭐⭐ | 1周 | ⬜ |
| 15 | **Attention Reduce** | `attention/reduce.cu` | `cuda/attention_reduce.py` | 🟡 | ⭐⭐ | 1周 | ⬜ |
| 16 | **Bad Words Ban** | `ban_bad_words.cu` | `cuda/ban_bad_words.py` | 🟢 | ⭐⭐ | 1周 | ⬜ |
| 17 | **Stop Criteria** | `stop_criteria_kernels.cu` | `cuda/stop_criteria.py` | 🟢 | ⭐ | 3-5天 | ⬜ |

**图例**:
- 优先级: 🔴 高 | 🟡 中 | 🟢 低
- 难度: ⭐ 简单 → ⭐⭐⭐⭐⭐ 极难
- 状态: ⬜ 待开始 | 🔄 进行中 | ✅ 已完成

---

## 📅 建议时间线

### **Week 1-2: 快速胜利**
- [ ] #8 GELU and Mul (3-5天) ← **从这个开始！**
- [ ] #17 Stop Criteria (3-5天)

### **Week 3-6: 核心功能**
- [ ] #1 Top-K Sampling (1-2周)
- [ ] #9 Penalty Kernels (1周)
- [ ] #10 Embedding + Pos Enc (1周)

### **Week 7-10: 量化支持**
- [ ] #3 KV Cache INT8 (1-2周)
- [ ] #13 Type Convert (1周)
- [ ] #14 Log Probability (1周)

### **Week 11-16: 高性能优化**
- [ ] #2 Top-P Sampling (2-3周)
- [ ] #4 KV Cache INT4 (2-3周)
- [ ] #5 SM80 GEMM (2-3周)

### **Month 4-6: 高级功能**
- [ ] #11 Context Parallelism (3-4周)
- [ ] #6 SM90 GEMM (4-6周)
- [ ] #7 TMA (6-8周)

---

## 🚀 立即开始：第一个任务

### **任务 #8: GELU and Mul** (最简单，3-5天)

#### **步骤**:
1. **阅读源码**:
   ```bash
   cat src/turbomind/kernels/activation_kernels.cu | grep -A 30 "GeluActivation"
   ```

2. **创建文件**:
   ```bash
   # 编辑现有文件
   vim lmdeploy/pytorch/kernels/cuda/activation.py
   ```

3. **添加 Triton kernel**:
   ```python
   @triton.jit
   def _gelu_and_mul_kernel(gateup_ptr, out_ptr, N, M, ...):
       # 1. Load gate and up
       gate = tl.load(...)
       up = tl.load(...)

       # 2. GELU activation
       # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
       x = gate.to(tl.float32)
       x3 = x * x * x
       tanh_arg = 0.7978845608 * (x + 0.044715 * x3)
       gelu = 0.5 * x * (1.0 + tl.math.tanh(tanh_arg))

       # 3. Fused multiply
       out = gelu * up

       # 4. Store
       tl.store(out_ptr, out)
   ```

4. **测试**:
   ```python
   # 对比 PyTorch 原生 GELU
   import torch.nn.functional as F

   gate_up = torch.randn(1024, 2048, device='cuda', dtype=torch.float16)
   gate, up = gate_up.chunk(2, dim=-1)

   # PyTorch reference
   expected = F.gelu(gate) * up

   # Triton kernel
   actual = gelu_and_mul(gate_up)

   # Check
   torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
   ```

5. **Benchmark**:
   ```python
   # 对比性能
   %timeit F.gelu(gate) * up  # PyTorch
   %timeit gelu_and_mul(gate_up)  # Triton
   ```

---

## 📊 已完成的 Kernels (参考)

| Kernel | PyTorch 文件 | 说明 |
|--------|--------------|------|
| ✅ SiLU and Mul | `cuda/activation.py` | Fused SiLU + multiply |
| ✅ Flash Attention | `cuda/flashattention.py` | Prefill attention |
| ✅ Paged Attention | `cuda/pagedattention.py` | Decoding attention |
| ✅ Alibi Paged Attn | `cuda/alibi_pagedattention.py` | ALiBi 位置编码 |
| ✅ Flash MLA | `cuda/flash_mla.py` | DeepSeek-V2/V3 |
| ✅ Fused MoE | `cuda/fused_moe.py` | MoE routing + GEMM |
| ✅ W8A8 MoE | `cuda/w8a8_fused_moe.py` | INT8 MoE |
| ✅ FP8 MoE | `cuda/blocked_fp8_fused_moe.py` | FP8 MoE |
| ✅ EP MoE | `cuda/ep_moe.py` | Expert Parallel MoE |
| ✅ RMS Norm | `cuda/rms_norm.py` | Root Mean Square Norm |
| ✅ RoPE | `cuda/apply_rotary_pos_emb.py` | Rotary Position Embedding |
| ✅ AWQ Linear | `cuda/awq_kernels.py` | W4A16 量化 |
| ✅ W8A8 Linear | `cuda/w8a8_triton_kernels.py` | INT8 量化 |
| ✅ FP8 GEMM | `cuda/blocked_gemm_fp8.py` | FP8 GEMM |
| ✅ Fill KV Cache | `cuda/fill_kv_cache.py` | KV cache 填充 |
| ✅ Flatten KV Cache | `cuda/flatten_kv_cache.py` | KV cache 重组 |
| ✅ Fused LoRA | `cuda/fused_lora.py` | LoRA adapter |
| ✅ Multinomial | `cuda/multinomial_sampling.py` | 多项式采样 |

这些已完成的 kernels 可以作为你移植新 kernels 的参考！

---

## 💡 移植技巧

### 1. **从简单的开始**
- Elementwise operations (activation, norm) 最容易
- 避免一开始就挑战 GEMM 或复杂 attention

### 2. **复用现有代码**
- 参考已完成的 kernels (`activation.py`, `fused_moe.py`)
- 学习 `@triton.autotune` 的使用

### 3. **测试驱动开发**
- 先写测试，确保功能正确
- 再优化性能，对比 TurboMind

### 4. **性能分析**
```python
# 使用 Triton profiler
import triton.profiler as profiler

with profiler.profile():
    my_kernel_launcher(x)

# 查看 kernel 执行时间
profiler.report()
```

### 5. **调试技巧**
```python
# Triton 支持打印调试
@triton.jit
def my_kernel(...):
    if tl.program_id(0) == 0:  # 只在第一个 block 打印
        tl.device_print("x =", x)
```

---

**开始你的第一个 Kernel 移植吧！🎉**

选择 **#8 GELU and Mul**，3-5 天完成，建立信心！
