# Copyright (c) UnieAI.
import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rms_forward(x_fp32, w, eps: tl.constexpr, N_COLS: tl.constexpr):
    sq = x_fp32 * x_fp32
    var = tl.sum(sq, axis=0) / N_COLS
    inv = tl.math.rsqrt(var + eps)
    y = x_fp32 * inv
    y = y * w
    return y


@triton.jit
def rms_norm_kernel(
    X, W, OUT,
    stride_x,       # stride over dim
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Pure RMSNorm: OUT = RMSNorm(X) * W
    X/OUT layout: flatten to [N_ROWS, N_COLS]
    每個 program 處理一整 row
    """
    row_id = tl.program_id(0)

    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS

    x_ptr = X + row_id * stride_x + offs
    out_ptr = OUT + row_id * stride_x + offs

    x = tl.load(x_ptr, mask=mask, other=0.0)
    x_fp32 = tl.astype(x, tl.float32)

    w = tl.load(W + offs, mask=mask, other=0.0)
    w = tl.astype(w, tl.float32)

    y_fp32 = _rms_forward(x_fp32, w, eps, N_COLS)
    y = tl.astype(y_fp32, X.dtype.element_ty)

    tl.store(out_ptr, y, mask=mask)


@triton.jit
def add_rms_norm_kernel(
    X, W, RES, OUT, OUT_RES,
    stride_x,
    stride_res,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Residual + RMSNorm:
      new_res = X + RES
      OUT     = RMSNorm(new_res) * W
      OUT_RES = new_res
    """
    row_id = tl.program_id(0)

    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS

    x_ptr = X + row_id * stride_x + offs
    r_ptr = RES + row_id * stride_res + offs
    out_ptr = OUT + row_id * stride_x + offs
    out_res_ptr = OUT_RES + row_id * stride_res + offs

    x = tl.load(x_ptr, mask=mask, other=0.0)
    r = tl.load(r_ptr, mask=mask, other=0.0)

    x_fp32 = tl.astype(x, tl.float32)
    r_fp32 = tl.astype(r, tl.float32)

    new_res_fp32 = x_fp32 + r_fp32
    new_res = tl.astype(new_res_fp32, X.dtype.element_ty)
    tl.store(out_res_ptr, new_res, mask=mask)

    w = tl.load(W + offs, mask=mask, other=0.0)
    w = tl.astype(w, tl.float32)

    y_fp32 = _rms_forward(new_res_fp32, w, eps, N_COLS)
    y = tl.astype(y_fp32, X.dtype.element_ty)

    tl.store(out_ptr, y, mask=mask)


@triton.jit
def bias_residual_rms_norm_kernel(
    X, W, BIAS, RES, OUT,
    stride_x,
    stride_res,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Bias + Residual + RMSNorm:
      new_res = X + BIAS + RES
      RES     = new_res  (in-place 更新 residual)
      OUT     = RMSNorm(new_res) * W
    """
    row_id = tl.program_id(0)

    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS

    x_ptr = X + row_id * stride_x + offs
    r_ptr = RES + row_id * stride_res + offs
    out_ptr = OUT + row_id * stride_x + offs

    x = tl.load(x_ptr, mask=mask, other=0.0)
    r = tl.load(r_ptr, mask=mask, other=0.0)
    b = tl.load(BIAS + offs, mask=mask, other=0.0)

    x_fp32 = tl.astype(x, tl.float32)
    r_fp32 = tl.astype(r, tl.float32)
    b_fp32 = tl.astype(b, tl.float32)

    new_res_fp32 = x_fp32 + b_fp32 + r_fp32
    new_res = tl.astype(new_res_fp32, X.dtype.element_ty)

    # in-place 更新 residual
    tl.store(r_ptr, new_res, mask=mask)

    w = tl.load(W + offs, mask=mask, other=0.0)
    w = tl.astype(w, tl.float32)

    y_fp32 = _rms_forward(new_res_fp32, w, eps, N_COLS)
    y = tl.astype(y_fp32, X.dtype.element_ty)

    tl.store(out_ptr, y, mask=mask)


def rms_norm(
    x: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
    residual: Tensor = None,
    bias: Tensor = None,
    out: Tensor = None,
    out_residual: Tensor = None,
):
    """
    ROCm 友善版 RMSNorm + (Bias)Residual fusion
    支援 layout: [..., seq_len, hidden_size]，最後一維是 hidden_size
    
    Args:
        x: input tensor
        weight: RMSNorm weight
        eps: epsilon for numerical stability
        residual: optional residual tensor
        bias: optional bias tensor
        out: optional output tensor
        out_residual: optional output residual tensor
    
    Returns:
        out or (out, out_residual)
    """
    if not x.is_contiguous():
        x = x.contiguous()

    B = x.numel() // x.size(-1)
    D = x.size(-1)

    assert weight.shape[0] == D
    W = weight

    # 展平成 [B, D]
    x_flat = x.view(B, D)
    stride_x = x_flat.stride(0)

    # BLOCK 設成 64/128/256（適合 ROCm wavefront）
    BLOCK = min(triton.next_power_of_2(D), 256)

    if out is None:
        out = torch.empty_like(x)
    out_flat = out.view(B, D)

    grid = (B,)

    if residual is None:
        # pure RMSNorm
        rms_norm_kernel[grid](
            x_flat,
            W,
            out_flat,
            stride_x=stride_x,
            eps=eps,
            N_COLS=D,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out

    # residual 存在
    residual_flat = residual
    if residual_flat.dim() > 2:
        residual_flat = residual_flat.view(B, D)
    if not residual_flat.is_contiguous():
        residual_flat = residual_flat.contiguous()
    stride_res = residual_flat.stride(0)

    if bias is not None:
        # Bias + Residual + RMSNorm
        bias_residual_rms_norm_kernel[grid](
            x_flat,
            W,
            bias,
            residual_flat,
            out_flat,
            stride_x=stride_x,
            stride_res=stride_res,
            eps=eps,
            N_COLS=D,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out

    # Residual + RMSNorm
    if out_residual is None:
        out_residual = torch.empty_like(x)
    out_res_flat = out_residual.view(B, D)

    add_rms_norm_kernel[grid](
        x_flat,
        W,
        residual_flat,
        out_flat,
        out_res_flat,
        stride_x=stride_x,
        stride_res=stride_res,
        eps=eps,
        N_COLS=D,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out, out_residual