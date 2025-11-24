# Copyright (c) UnieAI. All rights reserved.
"""
Top-P (Nucleus) sampling kernels (Triton + PyTorch fallback).

設計重點：
- 小 vocab（預設 <= 4096）用 Triton 單 block kernel，計算 + sort + cumsum 一次搞定。
- 大 vocab / 非 CUDA 環境自動 fallback 到 PyTorch 版本，確保正確性。
- Triton kernel 完全避免 runtime Python if，全部用向量化 & tl.where。
"""

import torch
import triton
import triton.language as tl
from typing import Optional


# ============================================================
# Triton kernels: small-vocab 單 block 實作
# ============================================================

@triton.jit
def _topp_sampling_small_kernel(
    logits_ptr,      # float*  [batch, vocab]
    output_ptr,      # int32*  [batch]
    seeds_ptr,       # int32*  [batch]
    offsets_ptr,     # int32*  [batch]
    p_ptr,           # float*  [batch]
    stride_batch,    # int
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    單 block Top-P sampling kernel（每個 program 處理一個 batch row）。
    限制：vocab_size <= BLOCK_SIZE 且 BLOCK_SIZE 不宜太大（預設上限 4096）。
    """
    pid = tl.program_id(0)
    row_start = pid * stride_batch

    # 讀 top-p 值與 RNG 狀態
    p_val = tl.load(p_ptr + pid)
    seed = tl.load(seeds_ptr + pid).to(tl.int32)
    offset = tl.load(offsets_ptr + pid).to(tl.int32)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size

    # 讀 logits（超出 vocab 的位置用 -inf padding）
    logits = tl.load(
        logits_ptr + row_start + offs,
        mask=mask,
        other=float("-inf"),
    )
    logits_f32 = logits.to(tl.float32)

    # ===== softmax → probs =====
    max_logit = tl.max(logits_f32, axis=0)
    logits_shift = logits_f32 - max_logit
    exp_logits = tl.exp(logits_shift)
    exp_logits = tl.where(mask, exp_logits, 0.0)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp  # [BLOCK_SIZE]

    # ===== 在 sorted domain 找 nucleus =====
    # sort 機率（由大到小）
    sorted_probs = tl.sort(probs, descending=True)  # [BLOCK_SIZE]

    # 累積機率
    cumsum_sorted = tl.cumsum(sorted_probs, axis=0)

    # 找出 nucleus mask
    p_vec = tl.full([BLOCK_SIZE], p_val, dtype=probs.dtype)
    nucleus_mask_sorted = cumsum_sorted <= p_vec
    # 至少保留一個 token
    first = offs == 0
    nucleus_mask_sorted = nucleus_mask_sorted | first

    # nucleus 內機率重新 normalize
    nucleus_probs_sorted = tl.where(nucleus_mask_sorted, sorted_probs, 0.0)
    nucleus_total = tl.sum(nucleus_probs_sorted, axis=0)
    nucleus_probs_sorted = nucleus_probs_sorted / nucleus_total

    # ===== 在 sorted domain 進行抽樣 =====
    rand = tl.rand(seed, offset)  # scalar ∈ (0, 1)
    rand_vec = tl.full([BLOCK_SIZE], rand, dtype=probs.dtype)

    cdf = tl.cumsum(nucleus_probs_sorted, axis=0)
    take = cdf >= rand_vec
    big = vocab_size  # sentinel index

    idx_candidates_sorted = tl.where(take, offs, big)
    sampled_pos_sorted = tl.min(idx_candidates_sorted, axis=0)  # sorted domain index

    # 用「原始 sorted_probs」中對應的機率值當 key（不要用 renorm 後的 nucleus_probs_sorted）
    selected_prob = sorted_probs[sampled_pos_sorted]

    # ===== 映射回原本 vocab index（approx by prob） =====
    selected_prob_vec = tl.full([BLOCK_SIZE], selected_prob, dtype=probs.dtype)
    diff = tl.abs(probs - selected_prob_vec)

    # 固定 epsilon（實務上已足夠；機率常 < 1）
    eps = 1e-6
    eps_vec = tl.full([BLOCK_SIZE], eps, dtype=probs.dtype)

    is_match = (diff <= eps_vec) & mask

    # 如果因為數值誤差完全沒有 match，fallback 到 argmax(probs)
    match_count = tl.sum(is_match.to(tl.int32), axis=0)
    has_match = match_count > 0

    max_prob = tl.max(probs, axis=0)
    max_prob_vec = tl.full([BLOCK_SIZE], max_prob, dtype=probs.dtype)
    fallback_match = (probs == max_prob_vec) & mask

    # has_match 為 scalar bool，可以在 tl.where 中 broadcast
    final_match = tl.where(has_match, is_match, fallback_match)

    idx_candidates_final = tl.where(final_match, offs, big)
    sampled_idx = tl.min(idx_candidates_final, axis=0)

    tl.store(output_ptr + pid, sampled_idx.to(tl.int32))


@triton.jit
def _topp_filter_small_kernel(
    logits_ptr,      # float* [batch, vocab]
    p_ptr,           # float* [batch]
    out_ptr,         # float* [batch, vocab]
    stride_batch,    # int
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    單 block Top-P filter kernel：
    - 保留 nucleus 內的 logits
    - nucleus 外的 logits 設為 -inf
    """
    pid = tl.program_id(0)
    row_start = pid * stride_batch

    p_val = tl.load(p_ptr + pid)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size

    # 讀 logits
    logits = tl.load(
        logits_ptr + row_start + offs,
        mask=mask,
        other=float("-inf"),
    )
    logits_f32 = logits.to(tl.float32)

    # softmax
    max_logit = tl.max(logits_f32, axis=0)
    logits_shift = logits_f32 - max_logit
    exp_logits = tl.exp(logits_shift)
    exp_logits = tl.where(mask, exp_logits, 0.0)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp  # [BLOCK_SIZE]

    # 在 sorted domain 找 nucleus 長度
    sorted_probs = tl.sort(probs, descending=True)
    cumsum_sorted = tl.cumsum(sorted_probs, axis=0)

    p_vec = tl.full([BLOCK_SIZE], p_val, dtype=probs.dtype)
    nucleus_mask_sorted = cumsum_sorted <= p_vec
    nucleus_mask_sorted = nucleus_mask_sorted | (offs == 0)

    # nucleus 長度（至少 1）
    nucleus_len = tl.sum(nucleus_mask_sorted.to(tl.int32), axis=0)
    one_i32 = tl.full((), 1, dtype=tl.int32)
    nucleus_len = tl.maximum(nucleus_len, one_i32)

    # cutoff index = nucleus_len - 1
    cutoff_idx = nucleus_len - one_i32  # scalar int32
    cutoff_prob = sorted_probs[cutoff_idx]

    # 把 cutoff_prob 略微往下調，避免邊界浮點誤差
    eps = 1e-6
    cutoff_vec = tl.full([BLOCK_SIZE], cutoff_prob - eps, dtype=probs.dtype)
    nucleus_mask_orig = (probs >= cutoff_vec) & mask

    minus_inf_vec = tl.full([BLOCK_SIZE], float("-inf"), dtype=logits.dtype)
    filtered_logits = tl.where(nucleus_mask_orig, logits, minus_inf_vec)

    tl.store(out_ptr + row_start + offs, filtered_logits, mask=mask)


# ============================================================
# 對外 API：自動選擇 Triton / PyTorch 路徑
# ============================================================

def topp_sampling(
    logits: torch.Tensor,
    p: torch.Tensor,
    seeds: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Top-P (nucleus) 抽樣，高效版本。

    Args:
        logits:  [batch_size, vocab_size]，float16/float32 皆可
        p:       [batch_size]，每個樣本自己的 top-p，0 < p <= 1
        seeds:   [batch_size]，int32（建議），Triton stateless RNG seed
        offsets: [batch_size]，int32，Triton stateless RNG offset

    Returns:
        [batch_size] int64，抽出的 token ids
    """
    assert logits.dim() == 2, "logits 必須是 [batch, vocab]"
    batch_size, vocab_size = logits.shape

    device = logits.device
    logits = logits.contiguous()

    # Triton kernel 設定：單 block 吃完整 vocab
    BLOCK_SIZE = triton.next_power_of_2(vocab_size)
    MAX_BLOCK_SIZE = 4096  # heuristic，可依硬體調整

    use_triton = (
        logits.is_cuda
        and BLOCK_SIZE <= MAX_BLOCK_SIZE
        and p.device == device
        and seeds.device == device
        and offsets.device == device
    )

    if use_triton:
        out = torch.empty(batch_size, device=device, dtype=torch.int32)
        grid = (batch_size,)

        _topp_sampling_small_kernel[grid](
            logits,
            out,
            seeds,
            offsets,
            p,
            logits.stride(0),
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out.to(torch.long)

    # vocab 過大或不在 CUDA：走 PyTorch 參考實作
    return torch_topp_sampling(logits, p)


def topp_filter(
    logits: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    """
    Top-P filter 高效版本：
    - nucleus 內 logits 保留
    - nucleus 外 logits 設為 -inf
    """
    assert logits.dim() == 2, "logits 必須是 [batch, vocab]"
    batch_size, vocab_size = logits.shape

    device = logits.device
    logits = logits.contiguous()

    BLOCK_SIZE = triton.next_power_of_2(vocab_size)
    MAX_BLOCK_SIZE = 4096

    use_triton = (
        logits.is_cuda
        and BLOCK_SIZE <= MAX_BLOCK_SIZE
        and p.device == device
    )

    if use_triton:
        out = torch.empty_like(logits)
        grid = (batch_size,)

        _topp_filter_small_kernel[grid](
            logits,
            p,
            out,
            logits.stride(0),
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out

    # fallback: PyTorch 版本
    return torch_topp_filter(logits, p)


# ============================================================
# PyTorch 參考實作（正確性基準）
# ============================================================

def torch_topp_sampling(
    logits: torch.Tensor,
    p: torch.Tensor,
    temperature: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    純 PyTorch Top-P sampling，邏輯清楚、易於驗證。
    可用來當 Triton 版本的 correctness baseline。

    Args:
        logits:      [B, V]
        p:           [B]
        temperature: [B]，可選，若提供則 logits / temperature

    Returns:
        [B] int64，抽樣出的 token ids
    """
    if temperature is not None:
        logits = logits / temperature.unsqueeze(-1)

    probs = torch.softmax(logits, dim=-1)           # [B, V]
    sorted_probs, sorted_indices = torch.sort(
        probs, dim=-1, descending=True
    )                                               # [B, V]

    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    p_expanded = p.unsqueeze(-1)
    nucleus_mask = cumsum_probs <= p_expanded
    # 確保至少留一個 token
    nucleus_mask[:, 0] = True

    nucleus_probs = sorted_probs * nucleus_mask
    nucleus_probs = nucleus_probs / nucleus_probs.sum(dim=-1, keepdim=True)

    sampled_positions = torch.multinomial(nucleus_probs, num_samples=1).squeeze(-1)
    sampled_ids = torch.gather(
        sorted_indices, 1, sampled_positions.unsqueeze(-1)
    ).squeeze(-1)

    return sampled_ids


def torch_topp_filter(
    logits: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    """
    純 PyTorch Top-P filter：
    - nucleus 內 logits 保留
    - nucleus 外 logits 設為 -inf
    """
    probs = torch.softmax(logits, dim=-1)  # [B, V]
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    p_expanded = p.unsqueeze(-1)
    nucleus_mask_sorted = cumsum_probs <= p_expanded
    nucleus_mask_sorted[:, 0] = True  # 至少一個 token

    # nucleus 長度、cutoff prob
    nucleus_len = nucleus_mask_sorted.sum(dim=-1)               # [B]
    cutoff_idx = (nucleus_len - 1).clamp_min(0)                 # [B]
    cutoff_prob = sorted_probs.gather(1, cutoff_idx.unsqueeze(-1))  # [B,1]

    # 回到原本 domain：prob >= cutoff 的 token 都視為 nucleus 內
    mask_nucleus = probs >= cutoff_prob
    filtered_logits = logits.masked_fill(~mask_nucleus, float("-inf"))
    return filtered_logits


# ============================================================
# 簡單測試工具（選擇性使用）
# ============================================================

def _test_filter_correctness(
    batch_size=8, vocab_size=1024, num_tests=20, device: Optional[torch.device] = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mismatches = 0
    for _ in range(num_tests):
        logits = torch.randn(batch_size, vocab_size, device=device)
        p = torch.rand(batch_size, device=device) * 0.5 + 0.5  # [0.5, 1.0]

        triton_filtered = topp_filter(logits, p)
        torch_filtered = torch_topp_filter(logits, p)

        triton_nucleus = (triton_filtered > -float("inf"))
        torch_nucleus = (torch_filtered > -float("inf"))

        if not torch.allclose(triton_nucleus.float(), torch_nucleus.float()):
            mismatches += 1

    print(f"[filter] matches: {num_tests - mismatches}/{num_tests}")
    return mismatches == 0


def _test_sampling_sanity(
    batch_size=8, vocab_size=1024, num_samples=1000, device: Optional[torch.device] = None
):
    """
    粗略測 sampling 是否合理（不是嚴格統計檢定，只是 sanity check）。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logits = torch.randn(batch_size, vocab_size, device=device)
    p = torch.full((batch_size,), 0.9, device=device)

    # 固定一組 seed/offset（示範）
    seeds = torch.randint(0, 2**30, (batch_size,), device=device, dtype=torch.int32)
    offsets = torch.arange(batch_size, device=device, dtype=torch.int32)

    # 估計每個 batch 的 top-1 hit rate
    triton_counts = torch.zeros(batch_size, vocab_size, device=device, dtype=torch.int64)
    torch_counts = torch.zeros(batch_size, vocab_size, device=device, dtype=torch.int64)

    for i in range(num_samples):
        # 為了公平，比較時可以改變 offset 但 seed 固定
        triton_ids = topp_sampling(logits, p, seeds, offsets + i)
        torch_ids = torch_topp_sampling(logits, p)

        triton_counts[torch.arange(batch_size, device=device), triton_ids] += 1
        torch_counts[torch.arange(batch_size, device=device), torch_ids] += 1

    # 只看每個 row 的 top-1 與 top-5 index 是否大致一致（粗比）
    triton_top1 = triton_counts.argmax(dim=-1)
    torch_top1 = torch_counts.argmax(dim=-1)
    same_top1 = (triton_top1 == torch_top1).float().mean().item()

    print(f"[sampling] top-1 argmax一致率 ≈ {same_top1 * 100:.1f}%（只是 sanity，不是嚴格檢定）")


if __name__ == "__main__":
    print("=== Testing Top-P kernels ===")
    ok = _test_filter_correctness()
    print("filter correctness:", ok)
    _test_sampling_sanity()
