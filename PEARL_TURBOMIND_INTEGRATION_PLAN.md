# PEARL Integration Plan for TurboMind Engine

## 执行摘要 (Executive Summary)

本文档分析将 nano-PEARL (Parallel Speculative Decoding) 整合到 lmdeploy 的 TurboMind engine 的可行性和实施计划。

**关键发现:**
- ✅ nano-PEARL 目前已在 PyTorch backend 实现完成
- ⚠️ TurboMind 是独立的 C++/CUDA 引擎，目前**没有**推测解码实现
- 🎯 整合具有可行性，但需要重大架构改动

---

## 1. nano-PEARL 技术概述

### 1.1 核心特性

nano-PEARL 是一种并行推测解码技术，具有以下创新：

1. **GPU 解耦** (GPU Decoupling)
   - Draft 模型和 Target 模型运行在不同 GPU 组
   - Draft devices: `[0]` (单独的 GPU)
   - Target devices: `[1, 2, 3]` (多 GPU 并行)

2. **并行执行** (Parallel Execution)
   - 使用 CUDA Streams 实现真正的并行
   - Draft 和 Target 模型异步执行，无需相互等待
   - `draft_stream` 和 `target_stream` 独立运行

3. **自适应 Draft Length** (Adaptive Gamma)
   - 根据 batch size 动态调整 gamma 值
   - 使用 AIMD (Additive Increase, Multiplicative Decrease) 算法
   - Gamma LUT (Lookup Table): `{4: 2, 8: 3, 16: 3, 32: 4, 64: 5}`

4. **Pre-verification**
   - Target 模型可以提前开始验证第一个 token
   - 减少延迟，提高吞吐量

### 1.2 性能优势

相比 EAGLE3 speculative decoding:
- Batch 16-32: ~1.2× 加速
- Batch 32-64: ~1.3× 加速
- Batch 64+: ~1.4-1.5× 加速

**最佳使用场景**: 高并发、大 batch size (≥32) 的生产环境

### 1.3 当前实现位置

```
lmdeploy/pytorch/spec_decode/proposers/pearl.py
```

**关键类:**
```python
class PEARLProposer(BaseSpecProposer):
    - build_model()          # 在 draft GPU 上构建模型
    - draft_tokens_parallel() # 并行生成 draft tokens
    - get_adaptive_gamma()    # 自适应 gamma 调整
    - update_gamma()          # AIMD 算法更新
```

---

## 2. TurboMind Engine 架构分析

### 2.1 核心组件

TurboMind 是一个 C++/CUDA 实现的高性能推理引擎：

```
src/turbomind/
├── engine/          # 引擎核心
│   ├── gateway.h/cc        # 请求路由和队列管理
│   ├── request.h           # 请求抽象
│   └── request_queue.h/cc  # 请求队列
├── models/
│   └── llama/              # Llama 模型实现
│       ├── LlamaBatch.cc   # Batch 处理逻辑
│       ├── LlamaV2.cc      # 主模型实现
│       ├── SequenceManager.cc  # 序列和缓存管理
│       └── unified_decoder.cc  # 统一的 decoder
├── kernels/         # CUDA kernels
└── layers/          # 网络层实现
```

### 2.2 关键发现

1. **没有现有的推测解码实现**
   - SequenceManager 中的 `Verify()` 是用于验证缓存块，不是推测解码
   - 没有 draft model 或 proposer 的概念

2. **批处理架构**
   - `LlamaBatch` 负责处理批量请求
   - `SequenceManager` 管理 KV cache 和序列状态
   - `Gateway` 负责请求路由和队列管理

3. **多 GPU 支持**
   - 已有 Tensor Parallelism (TP) 支持
   - `Gateway` 可以路由请求到不同 rank (GPU组)

---

## 3. 整合挑战分析

### 3.1 主要挑战

#### Challenge 1: 语言和框架差异
- **PEARL**: Python + PyTorch (动态图)
- **TurboMind**: C++ + CUDA (静态编译)

**影响**:
- 无法直接复用 Python 代码
- 需要重新实现所有逻辑

#### Challenge 2: GPU 管理机制不同
- **PEARL**: 使用 PyTorch CUDA Streams
  ```python
  self.draft_stream = torch.cuda.Stream(device=self.draft_device)
  with torch.cuda.stream(self.draft_stream):
      outputs = self._forward(...)
  ```

- **TurboMind**: 使用原生 CUDA API
  ```cpp
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  ```

**影响**: 需要重新设计流管理机制

#### Challenge 3: 模型加载和推理接口
- **PEARL**: 依赖 PyTorch 的 `torch.nn.Module`
- **TurboMind**: 使用自定义的权重加载和 kernel 调用

**影响**: Draft model 需要完全重新实现加载逻辑

#### Challenge 4: 缺少推测解码基础设施
TurboMind 需要新增：
1. Draft model 管理 (单独的模型实例)
2. Token verification 逻辑 (比较 draft vs target)
3. Speculative batch 管理 (扩展的 batch with draft tokens)
4. Acceptance/Rejection 处理

### 3.2 技术可行性评估

| 组件 | 可行性 | 难度 | 工作量估计 |
|------|--------|------|-----------|
| 多 GPU 管理 | ✅ 高 | 中 | 2-3 周 |
| CUDA Streams 并行 | ✅ 高 | 中 | 1-2 周 |
| Draft Model 加载 | ⚠️ 中 | 高 | 3-4 周 |
| Verification 逻辑 | ✅ 高 | 低 | 1 周 |
| Adaptive Gamma | ✅ 高 | 中 | 1-2 周 |
| Batch 管理扩展 | ⚠️ 中 | 高 | 3-4 周 |
| **总计** | - | - | **11-16 周** |

---

## 4. 整合方案设计

### 4.1 整体架构

```
┌─────────────────────────────────────────────────┐
│            TurboMind Gateway                    │
│         (Request Routing & Queue)               │
└────────────────┬────────────────────────────────┘
                 │
                 ├──────────────────┬──────────────────┐
                 │                  │                  │
          ┌──────▼──────┐    ┌──────▼──────┐   ┌──────▼──────┐
          │  Draft GPU  │    │ Target GPU 1│   │ Target GPU 2│
          │   (Rank 0)  │    │  (Rank 1)   │   │  (Rank 2)   │
          │             │    │             │   │             │
          │ Draft Model │    │Target Model │   │Target Model │
          │  (8B/13B)   │    │   (70B TP)  │   │   (70B TP)  │
          └──────┬──────┘    └──────┬──────┘   └──────┬──────┘
                 │                  │                  │
                 │  CUDA Stream 1   │  CUDA Stream 2   │
                 └──────────────────┴──────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ PEARLCoordinator  │
                    │  - Draft tokens   │
                    │  - Verify tokens  │
                    │  - Update gamma   │
                    └───────────────────┘
```

### 4.2 核心组件设计

#### 4.2.1 PEARLCoordinator (新增)

```cpp
// src/turbomind/engine/pearl_coordinator.h

namespace turbomind {

struct PEARLConfig {
    std::vector<int> draft_devices;  // e.g., {0}
    std::vector<int> target_devices; // e.g., {1, 2, 3}
    int gamma = -1;                   // -1 = adaptive
    bool enable_adaptive = true;
    bool use_parallel_streams = true;
};

class PEARLCoordinator {
public:
    PEARLCoordinator(const PEARLConfig& config,
                     std::shared_ptr<LlamaV2> draft_model,
                     std::shared_ptr<LlamaV2> target_model);

    // 核心接口
    struct DraftResult {
        std::vector<int> draft_token_ids;  // [batch_size, gamma]
        int num_tokens;
    };

    // 并行生成 draft tokens
    DraftResult GenerateDraftTokens(const ModelInputs& inputs,
                                     int batch_size,
                                     cudaStream_t draft_stream);

    // 验证 draft tokens vs target outputs
    struct VerifyResult {
        std::vector<int> accepted_tokens;  // [batch_size, num_accepted]
        int num_accepted;
        float acceptance_rate;
    };

    VerifyResult VerifyTokens(const DraftResult& draft,
                              const ModelOutputs& target_output);

    // 自适应 gamma 更新
    void UpdateGamma(int batch_size, float acceptance_rate);

    int GetAdaptiveGamma(int batch_size) const;

private:
    PEARLConfig config_;
    std::shared_ptr<LlamaV2> draft_model_;
    std::shared_ptr<LlamaV2> target_model_;

    // CUDA streams for parallel execution
    cudaStream_t draft_stream_;
    cudaStream_t target_stream_;

    // Adaptive gamma lookup table
    std::unordered_map<int, int> gamma_lut_;  // {batch_size: gamma}
};

}  // namespace turbomind
```

#### 4.2.2 修改 LlamaBatch (现有)

```cpp
// src/turbomind/models/llama/LlamaBatch.h

class LlamaBatch {
public:
    // 新增: 推测解码批处理
    struct SpeculativeBatch {
        std::vector<int> input_ids;      // 原始输入
        std::vector<int> draft_token_ids; // Draft tokens [batch, gamma]
        int gamma;                        // Draft length
    };

    // 新增接口
    void PrepareSpeculativeBatch(const SpeculativeBatch& spec_batch);

    // 现有接口保持兼容
    void Forward(...);

private:
    std::unique_ptr<PEARLCoordinator> pearl_coordinator_;
};
```

#### 4.2.3 修改 Gateway (现有)

```cpp
// src/turbomind/engine/gateway.h

class Gateway {
public:
    // 新增: 支持将请求路由到不同的设备组
    void SetPEARLRouting(const std::vector<int>& draft_ranks,
                        const std::vector<int>& target_ranks);

    // 新增: 为 PEARL 创建专用队列
    void CreatePEARLQueues(const PEARLConfig& config);

    // 现有接口保持不变
    void push(std::shared_ptr<Request> r);
    void pop(...);
};
```

### 4.3 数据流程

```
1. Request arrives at Gateway
   │
   ├─> Route to Draft GPU (Rank 0)
   │   └─> Draft Model generates gamma tokens asynchronously
   │       └─> DraftResult ready
   │
   └─> Route to Target GPUs (Rank 1,2,3)
       └─> Prepare speculative batch:
           - Original input
           - Appended draft tokens [gamma tokens]
       └─> Target Model forward pass (with draft tokens)
       └─> Verify draft tokens vs target logits
           ├─> Accept: Use draft tokens
           └─> Reject: Use target token, discard remaining drafts
```

---

## 5. 实施计划

### Phase 1: 基础设施准备 (3-4 周)

**任务:**
1. ✅ 分析现有代码 (已完成)
2. 创建 `PEARLCoordinator` 类框架
3. 实现多 GPU 设备管理
4. 实现 CUDA Streams 并行机制

**交付物:**
- `src/turbomind/engine/pearl_coordinator.h/cc`
- CUDA streams 测试通过

### Phase 2: Draft Model 集成 (3-4 周)

**任务:**
1. 在 TurboMind 中加载第二个模型实例 (draft model)
2. 实现 draft model 在独立 GPU 上的推理
3. 实现 `GenerateDraftTokens()` 接口

**交付物:**
- Draft model 可以在独立 GPU 上运行
- 能够生成 draft tokens

### Phase 3: Verification 逻辑 (2 周)

**任务:**
1. 实现 token verification kernel
2. 实现 acceptance/rejection 逻辑
3. 集成到 `LlamaBatch`

**交付物:**
- `VerifyTokens()` 功能完成
- 单元测试通过

### Phase 4: Adaptive Gamma (1-2 周)

**任务:**
1. 实现 AIMD 算法
2. 实现 gamma lookup table
3. 实现 profiling 功能

**交付物:**
- 自适应 gamma 调整功能
- Profiling 工具

### Phase 5: 端到端整合 (2-3 周)

**任务:**
1. 修改 Gateway 路由逻辑
2. 整合所有组件
3. 性能测试和调优

**交付物:**
- 完整的 PEARL 功能
- 性能报告

### Phase 6: 测试和优化 (2 周)

**任务:**
1. 功能测试
2. 性能基准测试
3. 文档编写

**交付物:**
- 测试报告
- 用户文档
- API 文档

---

## 6. 风险评估

### 6.1 高风险项

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| Draft model 加载失败 | 高 | 中 | 早期验证，使用简化模型测试 |
| CUDA Streams 同步问题 | 高 | 中 | 完善的同步机制，充分测试 |
| 性能不达预期 | 中 | 中 | 分阶段性能测试，及时调整 |
| 内存占用过高 | 中 | 低 | 实现内存共享机制 |

### 6.2 技术债务

1. **代码重复**: Draft 和 Target 可能有很多相同逻辑
   - 解决: 提取公共接口

2. **配置复杂度**: PEARL 引入新的配置参数
   - 解决: 提供默认配置和自动检测

3. **维护成本**: 需要同时维护两个模型实例
   - 解决: 统一模型管理接口

---

## 7. 性能预期

基于 PyTorch backend 的 PEARL 性能数据，预计 TurboMind 版本：

| Batch Size | 预期加速比 | TurboMind Baseline | 预期 TPS |
|-----------|-----------|-------------------|---------|
| 32        | 1.2×      | 2000 tok/s        | 2400 tok/s |
| 64        | 1.3×      | 2500 tok/s        | 3250 tok/s |
| 128       | 1.4×      | 3000 tok/s        | 4200 tok/s |

**关键指标:**
- Acceptance Rate: 目标 > 60%
- MAT (Mean Accepted Tokens): 目标 > 2.5
- Latency Overhead: 目标 < 10%

---

## 8. 替代方案

### 8.1 方案 A: 仅在 PyTorch Backend 使用 PEARL

**优点:**
- ✅ 已经实现，无需额外开发
- ✅ 维护成本低

**缺点:**
- ❌ TurboMind 用户无法受益
- ❌ 性能可能不如原生 C++/CUDA 实现

### 8.2 方案 B: 混合方案 (PyTorch Draft + TurboMind Target)

**优点:**
- ✅ 减少开发工作量 (draft model 用 PyTorch)
- ✅ 可以快速验证概念

**缺点:**
- ⚠️ Python/C++ 通信开销
- ⚠️ 部署复杂度增加

### 8.3 方案 C: 完全重写 (推荐)

**优点:**
- ✅ 最佳性能
- ✅ 统一的技术栈
- ✅ 更好的可维护性

**缺点:**
- ❌ 开发周期长 (11-16 周)
- ❌ 需要大量测试

---

## 9. 结论和建议

### 9.1 可行性评估

**结论**: PEARL 整合到 TurboMind **技术上可行**，但需要**重大工程投入**。

### 9.2 建议行动

#### 短期 (1-2 个月):
1. ✅ **保持 PyTorch Backend 的 PEARL 实现**
   - 继续优化和测试
   - 收集用户反馈

2. 🔬 **进行概念验证 (PoC)**
   - 在 TurboMind 中实现简化版 PEARL
   - 验证核心技术点 (多 GPU, CUDA Streams)
   - 评估实际性能提升

#### 中期 (3-6 个月):
3. 🚀 **如果 PoC 成功，启动完整实施**
   - 按照 Phase 1-6 执行
   - 分阶段交付和测试

#### 长期 (6+ 个月):
4. 🔧 **持续优化**
   - Kernel 融合优化
   - 内存优化
   - 支持更多模型架构

### 9.3 立即可行的步骤

**NOW (本周):**
```bash
# 1. 创建 PoC 分支
git checkout -b feature/pearl-turbomind-poc

# 2. 创建基础文件结构
mkdir -p src/turbomind/engine/pearl/
touch src/turbomind/engine/pearl/coordinator.h
touch src/turbomind/engine/pearl/coordinator.cc

# 3. 实现最小可行版本
# - 加载两个模型实例
# - 简单的 draft + verify 逻辑
# - 单个 CUDA stream 测试
```

**NEXT (下周):**
- 实现 CUDA Streams 并行
- 测试多 GPU 通信
- 性能基准测试

---

## 10. 参考资料

### 10.1 相关文档

- [PEARL 论文](https://arxiv.org/abs/2408.11850)
- [nano-PEARL GitHub](https://github.com/smart-lty/nano-PEARL)
- [lmdeploy Speculative Decoding 文档](https://lmdeploy.readthedocs.io/en/latest/advance/spec_decoding.html)

### 10.2 代码参考

**PyTorch PEARL 实现:**
- `lmdeploy/pytorch/spec_decode/proposers/pearl.py`
- `lmdeploy/pytorch/spec_decode/spec_agent.py`

**TurboMind 核心代码:**
- `src/turbomind/models/llama/LlamaBatch.cc`
- `src/turbomind/engine/gateway.h`
- `src/turbomind/models/llama/SequenceManager.cc`

---

## 附录 A: 关键代码片段

### A.1 PyTorch PEARL Draft Generation

```python
@record_function('pearl_draft_forward')
def draft_tokens_parallel(self,
                         previous_output: Dict[str, torch.Tensor],
                         model_inputs: ModelInputs,
                         extra_inputs: ExtraInputs,
                         cache_engine: CacheEngine,
                         num_tokens: int = None) -> torch.Tensor:
    """Generate draft tokens using parallel streams."""
    if num_tokens is None:
        batch_size = model_inputs.input_ids.size(0)
        num_tokens = self.get_adaptive_gamma(batch_size)

    draft_tokens = []
    current_inputs = model_inputs

    # Use draft stream if available
    stream_ctx = (torch.cuda.stream(self.draft_stream)
                 if self.draft_stream else torch.cuda.default_stream())

    with stream_ctx:
        # Generate tokens...
        for step in range(num_tokens):
            outputs = self._forward(current_inputs, cache_engine)
            draft_token_ids = self.get_outputs(outputs, ...)
            draft_tokens.append(draft_token_ids)

    return torch.cat(draft_tokens, dim=1)
```

### A.2 建议的 TurboMind 接口

```cpp
// 伪代码示例
class PEARLCoordinator {
public:
    DraftResult GenerateDraftTokens(const ModelInputs& inputs,
                                     int batch_size,
                                     cudaStream_t draft_stream) {
        DraftResult result;
        result.num_tokens = GetAdaptiveGamma(batch_size);

        // Asynchronous draft generation on draft_stream
        for (int i = 0; i < result.num_tokens; ++i) {
            auto outputs = draft_model_->Forward(inputs, draft_stream);
            result.draft_token_ids.push_back(GetGreedyToken(outputs));
        }

        // Synchronize
        cudaStreamSynchronize(draft_stream);
        return result;
    }
};
```

---

**文档版本**: v1.0
**创建日期**: 2025-12-14
**作者**: Claude (AI Assistant)
**状态**: 提案 (Proposal)
