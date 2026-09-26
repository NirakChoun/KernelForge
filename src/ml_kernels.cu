// Stage 8: row-wise softmax and RMSNorm (FP32), one block per row, built as a shared
// library (libkf_ml.so) and called from python/stage8_ml.py through ctypes so that CUDA,
// Triton, and PyTorch versions are timed and checked by the same harness.
//
// Both kernels read a row twice: pass 1 computes the row statistic (online max and sum of
// exponentials for softmax; sum of squares for RMSNorm), pass 2 reads the row again and
// writes the result. The second read of a row that was just read is expected to come
// from L2. float4 accesses are used when the row length is a multiple of 4.
#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr unsigned kFullMask = 0xffffffffu;

// Block size: one thread per 16 elements (4 float4 loads), between 32 and 1024 threads.
int threads_for(int cols) {
  int t = 32;
  while (t < 1024 && t * 16 < cols) t *= 2;
  return t;
}

struct MaxSum {
  float m, s;  // running max and sum of exp(x - m)
};

__device__ __forceinline__ MaxSum combine(MaxSum a, MaxSum b) {
  const float m = fmaxf(a.m, b.m);
  if (m == -INFINITY) return {m, 0.0f};
  return {m, a.s * expf(a.m - m) + b.s * expf(b.m - m)};
}

__device__ __forceinline__ MaxSum add_value(MaxSum a, float x) {
  if (x > a.m) return {x, a.s * expf(a.m - x) + 1.0f};
  return {a.m, a.s + expf(x - a.m)};
}

// Block-wide reduction; result broadcast to every thread. identity fills the lanes of
// warp 0 beyond the number of warps.
template <typename T, typename Op>
__device__ T block_reduce(T v, Op op, T identity, T* smem) {
  for (int off = 16; off > 0; off >>= 1) {
    T o;
    if constexpr (sizeof(T) == sizeof(MaxSum)) {
      o.m = __shfl_xor_sync(kFullMask, v.m, off);
      o.s = __shfl_xor_sync(kFullMask, v.s, off);
    } else {
      o = __shfl_xor_sync(kFullMask, v, off);
    }
    v = op(v, o);
  }
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = blockDim.x >> 5;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < nwarps ? smem[lane] : identity;
    for (int off = 16; off > 0; off >>= 1) {
      T o;
      if constexpr (sizeof(T) == sizeof(MaxSum)) {
        o.m = __shfl_xor_sync(kFullMask, v.m, off);
        o.s = __shfl_xor_sync(kFullMask, v.s, off);
      } else {
        o = __shfl_xor_sync(kFullMask, v, off);
      }
      v = op(v, o);
    }
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}

template <bool Vec>
__global__ void softmax_rows(const float* __restrict__ x, float* __restrict__ y, int cols) {
  __shared__ MaxSum red[32];
  const float* xr = x + static_cast<size_t>(blockIdx.x) * cols;
  float* yr = y + static_cast<size_t>(blockIdx.x) * cols;
  MaxSum ms{-INFINITY, 0.0f};
  if (Vec) {
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    for (int i = threadIdx.x; i < cols / 4; i += blockDim.x) {
      const float4 v = x4[i];
      ms = add_value(add_value(add_value(add_value(ms, v.x), v.y), v.z), v.w);
    }
  } else {
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ms = add_value(ms, xr[i]);
  }
  ms = block_reduce(ms, [](MaxSum a, MaxSum b) { return combine(a, b); }, MaxSum{-INFINITY, 0.0f}, red);
  const float inv = 1.0f / ms.s;
  if (Vec) {
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    float4* y4 = reinterpret_cast<float4*>(yr);
    for (int i = threadIdx.x; i < cols / 4; i += blockDim.x) {
      const float4 v = x4[i];
      y4[i] = make_float4(expf(v.x - ms.m) * inv, expf(v.y - ms.m) * inv, expf(v.z - ms.m) * inv,
                          expf(v.w - ms.m) * inv);
    }
  } else {
    for (int i = threadIdx.x; i < cols; i += blockDim.x) yr[i] = expf(xr[i] - ms.m) * inv;
  }
}

template <bool Vec>
__global__ void rmsnorm_rows(const float* __restrict__ x, const float* __restrict__ w,
                             float* __restrict__ y, int cols, float eps) {
  __shared__ float red[32];
  const float* xr = x + static_cast<size_t>(blockIdx.x) * cols;
  float* yr = y + static_cast<size_t>(blockIdx.x) * cols;
  float ss = 0.0f;
  if (Vec) {
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    for (int i = threadIdx.x; i < cols / 4; i += blockDim.x) {
      const float4 v = x4[i];
      ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
  } else {
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += xr[i] * xr[i];
  }
  ss = block_reduce(ss, [](float a, float b) { return a + b; }, 0.0f, red);
  const float r = rsqrtf(ss / cols + eps);
  if (Vec) {
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* y4 = reinterpret_cast<float4*>(yr);
    for (int i = threadIdx.x; i < cols / 4; i += blockDim.x) {
      const float4 v = x4[i], g = w4[i];
      y4[i] = make_float4(v.x * r * g.x, v.y * r * g.y, v.z * r * g.z, v.w * r * g.w);
    }
  } else {
    for (int i = threadIdx.x; i < cols; i += blockDim.x) yr[i] = xr[i] * r * w[i];
  }
}

template <typename K>
int attrs(K k, int* regs, int* local, int* smem) {
  cudaFuncAttributes a{};
  if (cudaFuncGetAttributes(&a, k) != cudaSuccess) return -1;
  *regs = a.numRegs;
  *local = static_cast<int>(a.localSizeBytes);
  *smem = static_cast<int>(a.sharedSizeBytes);
  return 0;
}

}  // namespace

extern "C" {

// Returns the threads per block used for a row length (for reporting).
int kf_ml_threads(int cols) { return threads_for(cols); }

// Returns 0 on success, otherwise a cudaError_t value from the launch.
int kf_softmax(const float* x, float* y, int rows, int cols, cudaStream_t s) {
  const int t = threads_for(cols);
  if (cols % 4 == 0) softmax_rows<true><<<rows, t, 0, s>>>(x, y, cols);
  else softmax_rows<false><<<rows, t, 0, s>>>(x, y, cols);
  return static_cast<int>(cudaGetLastError());
}

int kf_rmsnorm(const float* x, const float* w, float* y, int rows, int cols, float eps,
               cudaStream_t s) {
  const int t = threads_for(cols);
  if (cols % 4 == 0) rmsnorm_rows<true><<<rows, t, 0, s>>>(x, w, y, cols, eps);
  else rmsnorm_rows<false><<<rows, t, 0, s>>>(x, w, y, cols, eps);
  return static_cast<int>(cudaGetLastError());
}

// which: 0 softmax vec, 1 softmax scalar, 2 rmsnorm vec, 3 rmsnorm scalar.
int kf_ml_attrs(int which, int* regs, int* local, int* smem) {
  switch (which) {
    case 0: return attrs(softmax_rows<true>, regs, local, smem);
    case 1: return attrs(softmax_rows<false>, regs, local, smem);
    case 2: return attrs(rmsnorm_rows<true>, regs, local, smem);
    case 3: return attrs(rmsnorm_rows<false>, regs, local, smem);
    default: return -1;
  }
}

// Resident blocks per SM from the CUDA occupancy API at the block size used for cols.
int kf_ml_occupancy(int which, int cols, int* blocks_per_sm) {
  const int t = threads_for(cols);
  cudaError_t e;
  switch (which) {
    case 0: e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, softmax_rows<true>, t, 0); break;
    case 1: e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, softmax_rows<false>, t, 0); break;
    case 2: e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, rmsnorm_rows<true>, t, 0); break;
    case 3: e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, rmsnorm_rows<false>, t, 0); break;
    default: return -1;
  }
  return static_cast<int>(e);
}

}  // extern "C"
