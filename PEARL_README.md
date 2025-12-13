# PEARL (Parallel Speculative Decoding) 使用指南

## 🎯 什麼是 PEARL？

PEARL (Parallel spEculative decoding with Adaptive dRaft Length) 是一種先進的推測解碼技術，相比傳統的 Speculative Decoding，PEARL 通過以下創新實現更高的吞吐量：

1. **Draft-Target GPU 解耦**: Draft 模型和 Target 模型運行在不同的 GPU 組
2. **並行執行**: Draft 和 Target 模型真正並行運行，消除相互等待
3. **Pre-verify**: Target 模型可以提前開始驗證第一個 token
4. **Adaptive Draft Length**: 根據 batch size 動態調整 gamma 值

### 性能優勢

相比 EAGLE3：
- Batch 16-32: ~1.2× 加速
- Batch 32-64: ~1.3× 加速  
- Batch 64+: ~1.4-1.5× 加速

**最適用場景**: 高並發、大 batch size (≥32) 的生產環境

---

## 🚀 快速開始

### 前置要求

- 至少 2 張 GPU (1 張用於 draft，≥1 張用於 target)
- lmdeploy (已安裝 PEARL 支持)
- 一個 draft model (例如 Llama-3.1-8B)
- 一個 target model (例如 Llama-3.1-70B)

### 示例 1: Python API

```python
from lmdeploy import pipeline
from lmdeploy.pytorch.config import PytorchEngineConfig, PEARLConfig

# 配置 PEARL
pearl_config = PEARLConfig(
    method='pearl',
    model='meta-llama/Llama-3.1-8B-Instruct',  # draft model
    draft_devices=[0],          # Draft 在 GPU 0
    target_devices=[1, 2, 3],   # Target 在 GPU 1-3
    gamma=-1,                   # 自動調整
    num_speculative_tokens=3,   # fallback gamma
)

# 創建 pipeline
pipe = pipeline(
    'meta-llama/Llama-3.1-70B-Instruct',  # target model
    backend_config=PytorchEngineConfig(
        max_batch_size=128,
        tp=3,  # Target model TP=3
    ),
    speculative_config=pearl_config
)

# 批次推理
prompts = [
    "解釋量子計算",
    "什麼是機器學習？",
    # ... 更多 prompts
]

responses = pipe(prompts)
for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}")
    print(f"A: {response.text}\n")
```

### 示例 2: 命令行 (API Server)

```bash
lmdeploy serve api_server \
  meta-llama/Llama-3.1-70B-Instruct \
  --backend pytorch \
  --server-port 24545 \
  --tp 3 \
  --max-batch-size 128 \
  --speculative-algorithm pearl \
  --speculative-draft-model meta-llama/Llama-3.1-8B-Instruct \
  --pearl-draft-devices 0 \
  --pearl-target-devices 1 2 3 \
  --pearl-gamma -1 \
  --enable-metrics
```

**客戶端使用**:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:24545/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```

---

## ⚙️ 配置參數詳解

### PEARLConfig 參數

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `method` | str | 'pearl' | 固定為 'pearl' |
| `model` | str | required | Draft 模型路徑 |
| `draft_devices` | List[int] | [0] | Draft 模型使用的 GPU IDs |
| `target_devices` | List[int] | [1] | Target 模型使用的 GPU IDs |
| `gamma` | int | -1 | Draft tokens 數量，-1 = 自動 |
| `num_speculative_tokens` | int | 3 | Gamma 的 fallback 值 |
| `enable_pre_verify` | bool | True | 啟用 pre-verification |
| `enable_post_verify` | bool | True | 啟用 post-verification |
| `enable_adaptive_gamma` | bool | True | 動態調整 gamma |
| `use_parallel_streams` | bool | True | 使用 CUDA streams 並行 |
| `draft_temperature` | float | 0.0 | Draft 採樣溫度 (0=greedy) |

### 命令行參數

```bash
# 基礎參數
--speculative-algorithm pearl          # 選擇 PEARL 算法
--speculative-draft-model <path>       # Draft 模型路徑
--speculative-num-draft-tokens <int>   # Fallback gamma

# PEARL 專用參數
--pearl-draft-devices <gpu_ids>        # Draft GPU，例: 0
--pearl-target-devices <gpu_ids>       # Target GPUs，例: 1 2 3
--pearl-gamma <int>                    # -1 = auto，或指定固定值
--pearl-disable-adaptive               # 停用自適應 gamma
```

---

## 🎚️ 性能調優

### 1. GPU 分配策略

**推薦配置**:
```python
# 小 draft model (8B) + 大 target model (70B)
draft_devices=[0]          # 1 GPU for draft
target_devices=[1, 2, 3]   # 3 GPU for target (TP=3)

# 中等 draft (13B) + 大 target (70B)
draft_devices=[0, 1]       # 2 GPU for draft (TP=2)
target_devices=[2, 3, 4, 5] # 4 GPU for target (TP=4)
```

**原則**:
- Draft model 越小，分配越少 GPU
- Target model 越大，TP 越高
- Draft 和 Target 不能共享 GPU

### 2. Gamma 調優

**自動模式 (推薦)**:
```python
gamma=-1  # 啟動時自動 profiling
```

**手動指定**:
```python
# 根據 batch size 選擇
gamma=2   # batch size 1-8
gamma=3   # batch size 8-16
gamma=4   # batch size 16-32
gamma=5   # batch size 32+
```

### 3. Batch Size 配置

PEARL 在大 batch 下效果最好：

```python
--max-batch-size 128  # 推薦：64-128
```

### 4. 監控性能

```python
# 啟用 metrics
--enable-metrics

# 查看指標
curl http://localhost:24545/metrics
```

關鍵指標：
- `spec_decode_acceptance_rate`: 接受率 (越高越好)
- `throughput_tokens_per_sec`: 吞吐量
- `mean_acceptance_tokens`: 平均接受 tokens 數 (MAT)

---

## 📊 與其他算法對比

| 特性 | EAGLE3 | PEARL |
|------|--------|-------|
| GPU 策略 | 同一 GPU 組 | **分離 GPU 組** |
| 執行方式 | 順序 (Draft → Verify) | **並行** |
| Draft Length | 固定 | **自適應** |
| 最佳 Batch | 4-32 | **32-128** |
| GPU 需求 | 最少 |  ≥2 張 |
| 加速比 (BS=64) | 2.0-2.5× | **2.5-3.0×** |
| 延遲 | 低 | 中等 |

**選擇建議**:
- **EAGLE3**: 小 batch (1-32), 延遲敏感, GPU 有限
- **PEARL**: 大 batch (32+), 追求吞吐量, 多 GPU

---

## 🐛 常見問題

### Q1: GPU 分配錯誤

```
ValueError: draft_devices and target_devices must not overlap
```

**解決**: 確保 draft 和 target 使用不同的 GPU
```python
draft_devices=[0]      # ✓
target_devices=[1, 2]  # ✓

# 錯誤示例
draft_devices=[0, 1]   # ✗
target_devices=[1, 2]  # ✗ (GPU 1 重複)
```

### Q2: OOM (顯存不足)

```
CUDA out of memory
```

**解決方案**:
1. 減小 batch size: `--max-batch-size 64`
2. 減小 gamma: `--pearl-gamma 2`
3. 使用更小的 draft model
4. 減少 target model TP

### Q3: 性能不如 EAGLE3

**可能原因**:
- Batch size 太小 (< 16)
- GPU 數量不足
- Draft model 質量差

**解決**:
```bash
# 確保大 batch
--max-batch-size 128

# 監控接受率
curl http://localhost:24545/metrics | grep acceptance_rate
# 應該 > 0.6
```

### Q4: Gamma 自動調整失敗

```
WARNING: Auto-profiling gamma failed
```

**解決**: 手動指定 gamma
```python
gamma=3  # 不使用 -1
```

---

## 📈 完整示例：生產部署

### 場景：高並發 API 服務

**需求**:
- Target: Llama-3.1-70B
- Draft: Llama-3.1-8B  
- 預期並發: 100+ QPS
- 可用 GPU: 4 張 A100

**部署配置**:

```bash
#!/bin/bash
# deploy_pearl.sh

lmdeploy serve api_server \
  meta-llama/Llama-3.1-70B-Instruct \
  --backend pytorch \
  --server-port 24545 \
  --server-name 0.0.0.0 \
  \
  `# Target model 配置` \
  --tp 3 \
  --max-batch-size 128 \
  --cache-max-entry-count 0.9 \
  \
  `# PEARL 配置` \
  --speculative-algorithm pearl \
  --speculative-draft-model meta-llama/Llama-3.1-8B-Instruct \
  --pearl-draft-devices 0 \
  --pearl-target-devices 1 2 3 \
  --pearl-gamma -1 \
  \
  `# 監控和日誌` \
  --enable-metrics \
  --log-level INFO \
  \
  `# 可選：前綴快取` \
  --enable-prefix-caching
```

**性能預期**:
- 吞吐量: ~3000-3500 tok/s (batch 128)
- 延遲: ~300-500ms (P99)
- 接受率: ~70%

---

## 🧪 測試和驗證

### 1. 單 GPU 測試（驗證功能）

```python
# test_pearl_basic.py
from lmdeploy import pipeline
from lmdeploy.pytorch.config import PytorchEngineConfig, PEARLConfig

pearl_config = PEARLConfig(
    method='pearl',
    model='TinyLlama/TinyLlama-1.1B',  # 小模型測試
    draft_devices=[0],
    target_devices=[0],  # 測試時可以同 GPU
    gamma=2,
)

pipe = pipeline(
    'TinyLlama/TinyLlama-1.1B',
    backend_config=PytorchEngineConfig(max_batch_size=4),
    speculative_config=pearl_config
)

result = pipe("Hello, how are you?")
print(result.text)
```

### 2. Benchmark 對比

```bash
# benchmark_pearl.sh
python -m lmdeploy.cli benchmark \
  meta-llama/Llama-3.1-70B-Instruct \
  --backend pytorch \
  --dataset ShareGPT \
  --num-prompts 100 \
  --batch-size 32 \
  --speculative-algorithm pearl \
  --speculative-draft-model meta-llama/Llama-3.1-8B-Instruct \
  --pearl-draft-devices 0 \
  --pearl-target-devices 1 2 3
```

---

## 📚 更多資源

- [PEARL 論文](https://arxiv.org/abs/2408.11850)
- [nano-PEARL GitHub](https://github.com/smart-lty/nano-PEARL)
- [lmdeploy Speculative Decoding 文檔](https://lmdeploy.readthedocs.io/en/latest/advance/spec_decoding.html)

---

## 🔧 開發和調試

### 啟用 Debug 日誌

```bash
--log-level DEBUG
```

### 檢查 PEARL 運行狀態

```python
import logging
logging.basicConfig(level=logging.INFO)

# 查看日誌輸出
# [PEARL] Initializing...
# [PEARL] Draft on cuda:0, Target on cuda:1,2,3
# [PEARL] Auto-profiling gamma...
# [PEARL] Gamma LUT: {4: 2, 8: 3, 16: 3, 32: 4, 64: 5}
```

---

## ✅ 總結

PEARL 是高並發場景下的最佳選擇：

**何時使用 PEARL**:
- ✅ Batch size ≥ 32
- ✅ 有 ≥2 張 GPU
- ✅ 追求極致吞吐量
- ✅ 能容忍稍高延遲

**何時使用 EAGLE3**:
- ✅ Batch size < 32
- ✅ GPU 有限
- ✅ 需要低延遲
- ✅ 在線服務

立即開始使用 PEARL 提升你的推理吞吐量！🚀
