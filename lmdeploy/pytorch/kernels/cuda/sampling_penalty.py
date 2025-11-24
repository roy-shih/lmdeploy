# Copyright (c) UnieAI.
import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def build_repetition_mask_kernel(
    input_ids_ptr,
    visited_ptr,
    batch_size,
    seq_len,
    vocab_size,
    stride_is,    # input_ids.stride(0): seq
    stride_ib,    # input_ids.stride(1): batch
    stride_vb,    # visited.stride(0): batch
    stride_vv,    # visited.stride(1): vocab
    BLOCK_SEQ: tl.constexpr,
):
    """
    Phase 1: 建立 visited mask: [batch, vocab]
    input_ids: [seq_len, batch_size]
    visited:   [batch_size, vocab_size]  (uint8 0/1)
    """
    b_id = tl.program_id(0)
    seq_block = tl.program_id(1)

    offs_s = seq_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    mask_s = offs_s < seq_len

    inp_ptr = input_ids_ptr + offs_s * stride_is + b_id * stride_ib
    token_ids = tl.load(inp_ptr, mask=mask_s, other=0)
    token_ids = tl.astype(token_ids, tl.int32)

    valid = mask_s & (token_ids >= 0) & (token_ids < vocab_size)

    visited_row_ptr = visited_ptr + b_id * stride_vb
    tl.store(visited_row_ptr + token_ids, 1, mask=valid)


@triton.jit
def apply_repetition_penalty_kernel(
    logits_ptr,
    visited_ptr,
    penalties_ptr,
    batch_size,
    vocab_size,
    stride_lb,  # logits.stride(0)
    stride_lv,  # logits.stride(1)
    stride_vb,  # visited.stride(0)
    stride_vv,  # visited.stride(1)
    penalty_type: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Phase 2: 對 visited==1 的 vocab 位置套用 penalty。
    logits, visited: [batch, vocab]
    penalty_type: 0=none, 1=additive, 2=multiplicative
    """
    b_id = tl.program_id(0)
    v_block = tl.program_id(1)

    offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < vocab_size

    logits_row_ptr = logits_ptr + b_id * stride_lb + offs_v * stride_lv
    visited_row_ptr = visited_ptr + b_id * stride_vb + offs_v * stride_vv

    logits = tl.load(logits_row_ptr, mask=mask_v, other=0.0)
    visited = tl.load(visited_row_ptr, mask=mask_v, other=0)
    visited = tl.astype(visited, tl.int32)
    has_seen = visited > 0

    penalty = tl.load(penalties_ptr + b_id)

    if penalty_type == 1:
        penalized = logits - penalty
        new_logits = tl.where(has_seen, penalized, logits)
    elif penalty_type == 2:
        penalized = tl.where(logits < 0.0, logits * penalty, logits / penalty)
        new_logits = tl.where(has_seen, penalized, logits)
    else:
        new_logits = logits

    tl.store(logits_row_ptr, new_logits, mask=mask_v)


def apply_repetition_penalty(
    logits: Tensor,
    input_ids: Tensor,
    penalties: Tensor,
    penalty_type: str = "multiplicative",
) -> Tensor:
    """
    ROCm 友善版 Repetition Penalty。
    logits:    [batch, vocab]
    input_ids: [seq, batch] 或 [batch, seq]
    penalties: [batch]
    """
    assert logits.dim() == 2
    B, V = logits.shape

    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D, got {input_ids.shape}")

    if input_ids.shape[0] == B:
        input_ids = input_ids.t().contiguous()
    elif input_ids.shape[1] == B:
        input_ids = input_ids.contiguous()
    else:
        raise ValueError(
            f"input_ids shape {input_ids.shape} not compatible with batch={B}"
        )

    S = input_ids.shape[0]

    if penalties.shape[0] != B:
        raise ValueError(
            f"penalties shape[0]={penalties.shape[0]} != batch_size={B}"
        )
    penalties = penalties.to(dtype=logits.dtype, device=logits.device).contiguous()

    visited = torch.zeros(
        (B, V), dtype=torch.uint8, device=logits.device
    ).contiguous()

    penalty_type_map = {
        "none": 0,
        "additive": 1,
        "multiplicative": 2,
    }
    p_type = penalty_type_map.get(penalty_type.lower(), 2)

    # Phase 1
    BLOCK_SEQ = 128  # 2 * wavefront (64 * 2)，ROCm 友善
    grid1 = (B, triton.cdiv(S, BLOCK_SEQ))

    build_repetition_mask_kernel[grid1](
        input_ids,
        visited,
        batch_size=B,
        seq_len=S,
        vocab_size=V,
        stride_is=input_ids.stride(0),
        stride_ib=input_ids.stride(1),
        stride_vb=visited.stride(0),
        stride_vv=visited.stride(1),
        BLOCK_SEQ=BLOCK_SEQ,
        num_warps=2,
    )

    # Phase 2
    if not logits.is_contiguous():
        logits = logits.contiguous()

    BLOCK_V = 128
    grid2 = (B, triton.cdiv(V, BLOCK_V))

    apply_repetition_penalty_kernel[grid2](
        logits,
        visited,
        penalties,
        batch_size=B,
        vocab_size=V,
        stride_lb=logits.stride(0),
        stride_lv=logits.stride(1),
        stride_vb=visited.stride(0),
        stride_vv=visited.stride(1),
        penalty_type=p_type,
        BLOCK_V=BLOCK_V,
        num_warps=2,
    )

    return logits
