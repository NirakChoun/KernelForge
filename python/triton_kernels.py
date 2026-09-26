"""Triton kernels for Stages 7, 8, and 10. All FP32.

Every kernel is autotuned; the config lists below are the search spaces recorded in the
stage docs. Matmul uses input_precision="ieee" so tl.dot computes in FP32 like the
Stage 3 CUDA kernels and cuBLAS (Triton's default for FP32 dot is TF32).
"""
import triton
import triton.language as tl


def _add_configs():
    return [triton.Config({"BLOCK": b}, num_warps=w) for b in (1024, 2048, 4096, 8192) for w in (4, 8)]


@triton.autotune(configs=_add_configs(), key=["n"])
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def _row_configs():
    return [triton.Config({}, num_warps=w) for w in (1, 2, 4, 8, 16, 32)]


@triton.autotune(configs=_row_configs(), key=["n_cols"])
@triton.jit
def softmax_kernel(out_ptr, in_ptr, n_cols, BLOCK: tl.constexpr):
    # One program per row; the whole row is held in registers (BLOCK = next power of two).
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * n_cols + cols, num / den, mask=mask)


@triton.autotune(configs=_row_configs(), key=["n_cols"])
@triton.jit
def rmsnorm_kernel(out_ptr, in_ptr, w_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * n_cols + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    ms = tl.sum(x * x, axis=0) / n_cols
    r = 1.0 / tl.sqrt(ms + eps)
    tl.store(out_ptr + row * n_cols + cols, x * r * w, mask=mask)


MATMUL_SPACE = [
    # (BM, BN, BK, num_warps, num_stages)
    (128, 128, 32, 8, 3), (128, 128, 16, 8, 4), (128, 128, 32, 4, 3), (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4), (64, 64, 32, 4, 4), (128, 256, 16, 8, 3), (256, 128, 16, 8, 3),
    (64, 64, 64, 4, 3), (128, 64, 64, 8, 3), (64, 256, 32, 8, 3), (32, 64, 32, 4, 5),
]


def _matmul_configs():
    return [triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP_M": 8}, num_warps=w, num_stages=s)
            for bm, bn, bk, w, s in MATMUL_SPACE]


@triton.autotune(configs=_matmul_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  GROUP_M: tl.constexpr):
    # Grouped program ordering: GROUP_M row-blocks of C are visited together so their A
    # rows and B columns are reused from L2 (Triton tutorial pattern).
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_in_group = GROUP_M * num_pid_n
    group = pid // num_in_group
    first_m = group * GROUP_M
    group_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % num_in_group) % group_m)
    pid_n = (pid % num_in_group) // group_m
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        k_left = K - k * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
