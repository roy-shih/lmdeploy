# Copyright (c) UnieAI.
import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def length_criterion_kernel(
    finished_ptr,
    sequence_lengths_ptr,
    max_lengths_ptr,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    finished[i] |= sequence_lengths[i] >= max_lengths[i]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < batch_size

    finished = tl.load(finished_ptr + offs, mask=mask, other=0)
    finished = tl.astype(finished, tl.int1)

    seq_lens = tl.load(sequence_lengths_ptr + offs, mask=mask, other=0)
    max_lens = tl.load(max_lengths_ptr + offs, mask=mask, other=0)

    reached = seq_lens >= max_lens
    new_finished = finished | reached

    tl.store(finished_ptr + offs, new_finished, mask=mask)


@triton.jit(do_not_specialize=("current_step",))
def stop_words_criterion_kernel(
    finished_ptr,        # [batch]
    output_ids_ptr,      # [max_seq_len, batch]
    stop_words_ptr,      # [batch, max_stop_words]
    stop_words_len_ptr,  # [batch]
    current_step,
    batch_size,
    max_stop_words: tl.constexpr,
    stride_os,
    stride_ob,
    stride_sw_b,
    stride_sw_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    單 token stop word 檢查：
      finished[i] |= (output_ids[current_step, i] in stop_words[i, :len])
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < batch_size

    finished = tl.load(finished_ptr + offs, mask=mask, other=0)
    finished = tl.astype(finished, tl.int1)

    last_ptr = output_ids_ptr + current_step * stride_os + offs * stride_ob
    last_token = tl.load(last_ptr, mask=mask, other=-1)

    n_sw = tl.load(stop_words_len_ptr + offs, mask=mask, other=0)

    is_stop = tl.zeros([BLOCK_SIZE], dtype=tl.int1)

    # 建議 max_stop_words 不要太大（例如 <= 16），避免 ROCm 編譯超慢
    for sw_idx in tl.static_range(max_stop_words):
        sw_ptr = stop_words_ptr + offs * stride_sw_b + sw_idx * stride_sw_w
        sw = tl.load(sw_ptr, mask=mask, other=-1)
        valid = (sw_idx < n_sw) & mask
        matches = (last_token == sw) & valid
        is_stop = is_stop | matches

    new_finished = finished | is_stop
    tl.store(finished_ptr + offs, new_finished, mask=mask)


def check_length_criterion(
    finished: Tensor,
    sequence_lengths: Tensor,
    max_lengths: Tensor,
) -> Tensor:
    """
    長度 stop criterion：finished[i] |= length[i] >= max_length[i]
    """
    if finished.dim() != 1:
        raise ValueError(f"finished must be 1D, got {finished.shape}")
    B = finished.shape[0]

    if sequence_lengths.shape != (B,):
        raise ValueError("sequence_lengths must be [batch]")
    if max_lengths.shape != (B,):
        raise ValueError("max_lengths must be [batch]")

    finished = finished.contiguous()
    sequence_lengths = sequence_lengths.to(device=finished.device).contiguous()
    max_lengths = max_lengths.to(device=finished.device).contiguous()

    BLOCK = 128  # 2 * wavefront
    grid = (triton.cdiv(B, BLOCK),)

    length_criterion_kernel[grid](
        finished,
        sequence_lengths,
        max_lengths,
        batch_size=B,
        BLOCK_SIZE=BLOCK,
        num_warps=2,
    )
    return finished


def check_stop_words_criterion(
    finished: Tensor,
    output_ids: Tensor,
    stop_words: Tensor,
    stop_words_len: Tensor,
    current_step: int,
) -> Tensor:
    """
    單 token stop words 檢查。
    output_ids: [max_seq_len, batch]
    stop_words: [batch, max_stop_words]
    stop_words_len: [batch]
    """
    if finished.dim() != 1:
        raise ValueError(f"finished must be 1D, got {finished.shape}")
    B = finished.shape[0]

    if output_ids.dim() != 2 or output_ids.shape[1] != B:
        raise ValueError(f"output_ids must be [max_seq_len, batch], got {output_ids.shape}")
    if stop_words.dim() != 2 or stop_words.shape[0] != B:
        raise ValueError(f"stop_words must be [batch, max_stop_words], got {stop_words.shape}")
    if stop_words_len.shape != (B,):
        raise ValueError(f"stop_words_len must be [batch], got {stop_words_len.shape}")

    S = output_ids.shape[0]
    max_sw = stop_words.shape[1]
    if not (0 <= current_step < S):
        raise ValueError(f"current_step={current_step} out of range [0, {S})")

    finished = finished.contiguous()
    output_ids = output_ids.to(device=finished.device).contiguous()
    stop_words = stop_words.to(device=finished.device).contiguous()
    stop_words_len = stop_words_len.to(device=finished.device).contiguous()

    BLOCK = 128
    grid = (triton.cdiv(B, BLOCK),)

    stop_words_criterion_kernel[grid](
        finished,
        output_ids,
        stop_words,
        stop_words_len,
        current_step=current_step,
        batch_size=B,
        max_stop_words=max_sw,
        stride_os=output_ids.stride(0),
        stride_ob=output_ids.stride(1),
        stride_sw_b=stop_words.stride(0),
        stride_sw_w=stop_words.stride(1),
        BLOCK_SIZE=BLOCK,
        num_warps=2,
    )
    return finished
