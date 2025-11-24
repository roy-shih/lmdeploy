# Copyright (c) UnieAI.
import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def length_criterion_kernel(
    finished_ptr,           # [batch_size] bool / uint8
    sequence_lengths_ptr,   # [batch_size] int32
    max_lengths_ptr,        # [batch_size] int32
    batch_size,             # runtime scalar
    BLOCK_SIZE: tl.constexpr,
):
    """
    Check if sequences have reached their maximum length.

    Logic:
        finished[i] |= sequence_lengths[i] >= max_lengths[i]
    """
    pid = tl.program_id(0)

    # 每個 program 負責一段 batch index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < batch_size

    # load
    finished = tl.load(finished_ptr + offs, mask=mask, other=0)
    finished = tl.astype(finished, tl.int1)

    seq_lens = tl.load(sequence_lengths_ptr + offs, mask=mask, other=0)
    max_lens = tl.load(max_lengths_ptr + offs, mask=mask, other=0)

    reached_max = seq_lens >= max_lens  # int1
    new_finished = finished | reached_max

    # store 回去（PyTorch bool / uint8 都可以接）
    tl.store(finished_ptr + offs, new_finished, mask=mask)


@triton.jit(do_not_specialize=('current_step',))
def stop_words_criterion_kernel(
    finished_ptr,        # [batch_size] bool / uint8
    output_ids_ptr,      # [max_seq_len, batch_size] int32
    stop_words_ptr,      # [batch_size, max_stop_words] int32
    stop_words_len_ptr,  # [batch_size] int32 - actual number of stop words per batch
    current_step,        # runtime scalar
    batch_size,          # runtime scalar
    max_stop_words: tl.constexpr,
    stride_os,           # output_ids.stride(0)  (seq)
    stride_ob,           # output_ids.stride(1)  (batch)
    stride_sw_b,         # stop_words.stride(0)  (batch)
    stride_sw_w,         # stop_words.stride(1)  (stop_word index)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Check if the last generated token matches any stop word (single-token only).

    finished[i] |= (output_ids[current_step, i] in stop_words[i, :stop_words_len[i]])
    """
    pid = tl.program_id(0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < batch_size

    # finished: [batch]
    finished = tl.load(finished_ptr + offs, mask=mask, other=0)
    finished = tl.astype(finished, tl.int1)

    # 取每個 batch 在 current_step 的最後一個 token
    last_token_ptr = output_ids_ptr + current_step * stride_os + offs * stride_ob
    last_token = tl.load(last_token_ptr, mask=mask, other=-1)

    # 有效 stop words 個數
    num_stop_words = tl.load(stop_words_len_ptr + offs, mask=mask, other=0)

    # 累積：是否命中任何 stop word
    is_stop_word = tl.zeros([BLOCK_SIZE], dtype=tl.int1)

    # 對於每一個 stop word index sw_idx（0..max_stop_words-1）做向量化 check
    for sw_idx in tl.static_range(max_stop_words):
        # stop_words: [batch, max_stop_words]
        # index = base + batch * stride_sw_b + sw_idx * stride_sw_w
        sw_ptr = stop_words_ptr + offs * stride_sw_b + sw_idx * stride_sw_w
        stop_word = tl.load(sw_ptr, mask=mask, other=-1)

        # 當前這個 sw_idx 對於每個 batch 是否有效
        # sw_idx < num_stop_words[b] 且在 batch mask 內
        valid_sw = (sw_idx < num_stop_words) & mask

        # token 相等且此 stop word 有效
        matches = (last_token == stop_word) & valid_sw

        is_stop_word = is_stop_word | matches

    new_finished = finished | is_stop_word

    tl.store(finished_ptr + offs, new_finished, mask=mask)


def check_length_criterion(
    finished: Tensor,
    sequence_lengths: Tensor,
    max_lengths: Tensor,
) -> Tensor:
    """
    Check if sequences have reached their maximum length.

    Args:
        finished: [batch_size] bool tensor marking finished sequences (modified in-place)
        sequence_lengths: [batch_size] current sequence lengths (int32/long)
        max_lengths: [batch_size] maximum allowed lengths (int32/long)

    Returns:
        Updated finished tensor (same object as input)
    """
    if finished.dim() != 1:
        raise ValueError(f"finished must be 1D [batch], got {finished.shape}")

    batch_size = finished.shape[0]

    if sequence_lengths.shape != (batch_size,):
        raise ValueError(
            f"sequence_lengths must be [batch_size], got {sequence_lengths.shape}"
        )
    if max_lengths.shape != (batch_size,):
        raise ValueError(
            f"max_lengths must be [batch_size], got {max_lengths.shape}"
        )

    # 保證 contiguous
    finished = finished.contiguous()
    sequence_lengths = sequence_lengths.to(device=finished.device).contiguous()
    max_lengths = max_lengths.to(device=finished.device).contiguous()

    BLOCK_SIZE = 128
    grid = (triton.cdiv(batch_size, BLOCK_SIZE),)

    length_criterion_kernel[grid](
        finished,
        sequence_lengths,
        max_lengths,
        batch_size=batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
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
    Check if last generated token matches any stop word (single-token only).

    Args:
        finished: [batch_size] bool tensor (modified in-place)
        output_ids: [max_seq_len, batch_size] generated token IDs (int32/long)
        stop_words: [batch_size, max_stop_words] stop word IDs (-1 for padding)
        stop_words_len: [batch_size] number of valid stop words per batch
        current_step: current generation step (0-based)

    Returns:
        Updated finished tensor (same object as input)
    """
    if finished.dim() != 1:
        raise ValueError(f"finished must be 1D [batch], got {finished.shape}")
    batch_size = finished.shape[0]

    if output_ids.dim() != 2:
        raise ValueError(f"output_ids must be 2D [max_seq_len, batch], got {output_ids.shape}")
    if output_ids.shape[1] != batch_size:
        raise ValueError(
            f"output_ids.shape[1] ({output_ids.shape[1]}) != batch_size ({batch_size})"
        )

    if stop_words.dim() != 2:
        raise ValueError(f"stop_words must be 2D [batch, max_stop_words], got {stop_words.shape}")
    if stop_words.shape[0] != batch_size:
        raise ValueError(
            f"stop_words.shape[0] ({stop_words.shape[0]}) != batch_size ({batch_size})"
        )

    if stop_words_len.shape != (batch_size,):
        raise ValueError(
            f"stop_words_len must be [batch_size], got {stop_words_len.shape}"
        )

    max_seq_len = output_ids.shape[0]
    max_stop_words = stop_words.shape[1]

    if not (0 <= current_step < max_seq_len):
        raise ValueError(
            f"current_step={current_step} out of range [0, {max_seq_len})"
        )

    # 保證 contiguous & device 一致
    finished = finished.contiguous()
    output_ids = output_ids.to(device=finished.device).contiguous()
    stop_words = stop_words.to(device=finished.device).contiguous()
    stop_words_len = stop_words_len.to(device=finished.device).contiguous()

    BLOCK_SIZE = 128
    grid = (triton.cdiv(batch_size, BLOCK_SIZE),)

    stop_words_criterion_kernel[grid](
        finished,
        output_ids,
        stop_words,
        stop_words_len,
        current_step=current_step,
        batch_size=batch_size,
        max_stop_words=max_stop_words,
        stride_os=output_ids.stride(0),
        stride_ob=output_ids.stride(1),
        stride_sw_b=stop_words.stride(0),
        stride_sw_w=stop_words.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return finished
