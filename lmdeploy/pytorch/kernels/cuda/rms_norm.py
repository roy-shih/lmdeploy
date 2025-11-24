# Copyright (c) UnieAI.
import torch
import triton
import triton.language as tl
from torch import Tensor


# -------------------------------
# Utility: RMSNorm compute (FP32 accumulate)
# -------------------------------
@triton.jit
def _rms_forward(x_fp32, w, eps: tl.constexpr, N_COLS: tl.constexpr):
    sq = x_fp32 * x_fp32
    var = tl.sum(sq, axis=0) / N_COLS
    inv = tl.math.rsqrt(var + eps)
    y = x_fp32 * inv
    y = y * w.to(tl.float32)
    return y


# -------------------------------
# Pure RMSNorm
# -------------------------------
@triton.jit
def rms_norm_kernel(
    X, W, OUT,
    seq_len,
    stride,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS

    # each pid handles a row
    row_ptr = X + pid * stride + offs
    w = tl.load(W + offs, mask=mask)

    x = tl.load(row_ptr, mask=mask).to(tl.float32)

    y = _rms_forward(x, w, eps, N_COLS)

    out_ptr = OUT + pid * stride + offs
    tl.store(out_ptr, y.to(X.dtype.element_ty), mask=mask)


# -------------------------------
# Residual + RMSNorm
# -------------------------------
@triton.jit
def add_rms_norm_kernel(
    X, W, RES, OUT, OUT_RES,
    seq_len,
    x_stride,
    res_stride,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS

    w = tl.load(W + offs, mask=mask)

    x_ptr = X + pid * x_stride + offs
    r_ptr = RES + pid * res_stride + offs
    out_ptr = OUT + pid * x_stride + offs
    out_res_ptr = OUT_RES + pid * res_stride + offs

    x = tl.load(x_ptr, mask=mask).to(tl.float32)
    r = tl.load(r_ptr, mask=mask).to(tl.float32)

    new = x + r
    tl.store(out_res_ptr, new.to(X.dtype.element_ty), mask=mask)

    y = _rms_forward(new, w, eps, N_COLS)
    tl.store(out_ptr, y.to(X.dtype.element_ty), mask=mask)


# -------------------------------
# Bias + Residual + RMSNorm  (TurboMind exact)
# new_res = x + bias + res
# y = RMSNorm(new_res)
# -------------------------------
@triton.jit
def bias_residual_rms_norm_kernel(
    X, W, BIAS, RES, OUT,
    seq_len,
    x_stride,
    res_stride,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Bias + Residual + RMSNorm Kernel.
    
    Reference Implementation: src/turbomind/kernels/norm/rms_norm.cu (BiasResidualRMSNormKernel)
    
    Why this fusion is important:
    1. Memory Bandwidth Optimization: 
       TurboMind fuses Bias Add, Residual Add, and RMSNorm into a single kernel.
       Without fusion: 
         - Read Input, Read Bias -> Add -> Write Temp (Memory R/W)
         - Read Temp, Read Residual -> Add -> Write Residual (Memory R/W)
         - Read Residual -> RMSNorm -> Write Output (Memory R/W)
       With fusion:
         - Read Input, Bias, Residual -> Compute All -> Write Residual, Write Output
       This significantly reduces global memory traffic, which is the bottleneck for normalization layers.
       
    2. Logic Alignment:
       We strictly follow TurboMind's logic:
       - New_Residual = Input + Bias + Old_Residual
       - Output = RMSNorm(New_Residual)
       - Store New_Residual (for next layer's skip connection)
       - Store Output (for next layer's input)
    """
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS

    w = tl.load(W + offs, mask=mask)
    b = tl.load(BIAS + offs, mask=mask).to(tl.float32)

    x_ptr = X + pid * x_stride + offs
    r_ptr = RES + pid * res_stride + offs
    out_ptr = OUT + pid * x_stride + offs

    x = tl.load(x_ptr, mask=mask).to(tl.float32)
    r = tl.load(r_ptr, mask=mask).to(tl.float32)

    new = x + b + r

    # update residual in-place
    tl.store(r_ptr, new.to(X.dtype.element_ty), mask=mask)

    # RMSNorm
    y = _rms_forward(new, w, eps, N_COLS)
    tl.store(out_ptr, y.to(X.dtype.element_ty), mask=mask)


# -------------------------------
# Host Function
# -------------------------------
def rms_norm(x: Tensor, weight: Tensor,
             eps=1e-6, residual=None, bias=None,
             out=None, out_residual=None):
    """RMS Normalization with optional bias and residual fusion."""
    B, S, D = x.shape
    device = x.device

    BLOCK = min(triton.next_power_of_2(D), 2048)

    if out is None:
        out = torch.empty_like(x)

    grid = (B * S,)

    if residual is None:
        # Pure RMSNorm
        rms_norm_kernel[grid](
            x, weight, out,
            seq_len=B*S,
            stride=x.stride(1),
            eps=eps,
            N_COLS=D,
            BLOCK=BLOCK,
        )
        return out

    # residual exists
    if bias is not None:
        # Bias+Residual+RMSNorm
        bias_residual_rms_norm_kernel[grid](
            x, weight, bias, residual, out,
            seq_len=B*S,
            x_stride=x.stride(1),
            res_stride=residual.stride(1),
            eps=eps,
            N_COLS=D,
            BLOCK=BLOCK,
        )
        return out

    # Only residual + rms norm
    if out_residual is None:
        out_residual = torch.empty_like(x)

    add_rms_norm_kernel[grid](
        x, weight, residual, out, out_residual,
        seq_len=B*S,
        x_stride=x.stride(1),
        res_stride=residual.stride(1),
        eps=eps,
        N_COLS=D,
        BLOCK=BLOCK,
    )
    return out, out_residual