// Stage 2: float sum reduction. One version per --version; see docs/stage2.md.
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include "kf/bench.cuh"

constexpr int kBlock = 256;

// v1: one atomicAdd per element, all on the same address.
__global__ void reduce_atomic(const float* __restrict__ in, float* __restrict__ out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out, in[i]);
}

// v2: shared-memory tree, interleaved addressing. Active threads are those with
// tid % (2 * stride) == 0, so every warp stays partly active until the last steps.
__global__ void reduce_interleaved(const float* __restrict__ in, float* __restrict__ out, size_t n) {
  __shared__ float s[kBlock];
  const unsigned tid = threadIdx.x;
  const size_t i = static_cast<size_t>(blockIdx.x) * kBlock + tid;
  s[tid] = i < n ? in[i] : 0.0f;
  __syncthreads();
  for (unsigned stride = 1; stride < kBlock; stride *= 2) {
    if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// v3: sequential addressing. Active threads are the contiguous range tid < stride.
__global__ void reduce_sequential(const float* __restrict__ in, float* __restrict__ out, size_t n) {
  __shared__ float s[kBlock];
  const unsigned tid = threadIdx.x;
  const size_t i = static_cast<size_t>(blockIdx.x) * kBlock + tid;
  s[tid] = i < n ? in[i] : 0.0f;
  __syncthreads();
  for (unsigned stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// v4: v3, but each thread adds two elements while loading, so a block covers 2 * kBlock.
__global__ void reduce_first_add(const float* __restrict__ in, float* __restrict__ out, size_t n) {
  __shared__ float s[kBlock];
  const unsigned tid = threadIdx.x;
  const size_t i = static_cast<size_t>(blockIdx.x) * (2 * kBlock) + tid;
  s[tid] = (i < n ? in[i] : 0.0f) + (i + kBlock < n ? in[i + kBlock] : 0.0f);
  __syncthreads();
  for (unsigned stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

constexpr unsigned kFullMask = 0xffffffffu;

// v5: v4, but the last 64 values are reduced by one warp with shuffles, without
// shared memory or __syncthreads.
__global__ void reduce_warp_shuffle(const float* __restrict__ in, float* __restrict__ out, size_t n) {
  __shared__ float s[kBlock];
  const unsigned tid = threadIdx.x;
  const size_t i = static_cast<size_t>(blockIdx.x) * (2 * kBlock) + tid;
  s[tid] = (i < n ? in[i] : 0.0f) + (i + kBlock < n ? in[i + kBlock] : 0.0f);
  __syncthreads();
  for (unsigned stride = kBlock / 2; stride > 32; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid < 32) {
    float v = s[tid] + s[tid + 32];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(kFullMask, v, off);
    if (tid == 0) out[blockIdx.x] = v;
  }
}

// Block sum with warp shuffles; result valid in thread 0.
__device__ float block_sum(float v) {
  __shared__ float warp_sums[kBlock / 32];
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(kFullMask, v, off);
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < kBlock / 32 ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(kFullMask, v, off);
  }
  return v;
}

// v6: fixed grid sized to fill the GPU; each thread accumulates many elements with a
// grid-stride loop, then one block sum. A second single-block launch sums the partials.
__global__ void reduce_grid_stride(const float* __restrict__ in, float* __restrict__ out, size_t n) {
  float v = 0.0f;
  const size_t step = static_cast<size_t>(gridDim.x) * kBlock;
  for (size_t i = static_cast<size_t>(blockIdx.x) * kBlock + threadIdx.x; i < n; i += step) v += in[i];
  v = block_sum(v);
  if (threadIdx.x == 0) out[blockIdx.x] = v;
}

struct Plan {
  std::string name, variant = "-", grid = "-", block = "-";
  kf::KernelInfo ki;
  std::function<void()> run;
};

// Repeats a block-per-tile kernel until one value remains. Each pass turns m values
// into ceil(m / per_block) partial sums, alternating between two scratch buffers; the
// last pass writes the result.
template <typename K>
static void multipass(K kernel, size_t per_block, const float* in, size_t n, float* a, float* b,
                      float* result) {
  const float* src = in;
  size_t m = n;
  float* dst = a;
  while (true) {
    const size_t blocks = (m + per_block - 1) / per_block;
    float* out = blocks == 1 ? result : dst;
    kernel<<<static_cast<unsigned>(blocks), kBlock>>>(src, out, m);
    if (blocks == 1) break;
    src = out;
    m = blocks;
    dst = dst == a ? b : a;
  }
}

static int count_passes(size_t n, size_t per_block) {
  int p = 1;
  for (size_t m = (n + per_block - 1) / per_block; m > 1; m = (m + per_block - 1) / per_block) ++p;
  return p;
}

template <typename K>
static Plan tiled_plan(const char* name, K kernel, size_t per_block, const float* in, size_t n,
                       float* a, float* b, float* result) {
  Plan p;
  p.name = name;
  p.variant = "passes=" + std::to_string(count_passes(n, per_block));
  p.grid = std::to_string((n + per_block - 1) / per_block);
  p.block = std::to_string(kBlock);
  p.ki = kf::kernel_info(kernel, kBlock);
  p.run = [=] { multipass(kernel, per_block, in, n, a, b, result); };
  return p;
}

static Plan make_plan(int version, const float* in, size_t n, [[maybe_unused]] float* a,
                      [[maybe_unused]] float* b, float* result) {
  switch (version) {
    case 1: {
      Plan p;
      p.name = "v1_atomic";
      const unsigned grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
      p.grid = std::to_string(grid);
      p.block = std::to_string(kBlock);
      p.ki = kf::kernel_info(reduce_atomic, kBlock);
      // The zeroing of the result is part of each timed launch.
      p.run = [=] {
        CUDA_CHECK(cudaMemsetAsync(result, 0, sizeof(float)));
        reduce_atomic<<<grid, kBlock>>>(in, result, n);
      };
      return p;
    }
    case 2: return tiled_plan("v2_interleaved", reduce_interleaved, kBlock, in, n, a, b, result);
    case 3: return tiled_plan("v3_sequential", reduce_sequential, kBlock, in, n, a, b, result);
    case 4: return tiled_plan("v4_first_add", reduce_first_add, 2 * kBlock, in, n, a, b, result);
    case 5: return tiled_plan("v5_warp_shuffle", reduce_warp_shuffle, 2 * kBlock, in, n, a, b, result);
    case 6: {
      Plan p;
      p.name = "v6_grid_stride";
      p.ki = kf::kernel_info(reduce_grid_stride, kBlock);
      int dev = 0, sms = 0;
      CUDA_CHECK(cudaGetDevice(&dev));
      CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
      // One full wave of resident blocks, fewer if n is small.
      const size_t full = static_cast<size_t>(p.ki.blocks_per_sm) * sms;
      const unsigned grid = static_cast<unsigned>(std::min(full, (n + kBlock - 1) / kBlock));
      p.variant = "passes=2";
      p.grid = std::to_string(grid);
      p.block = std::to_string(kBlock);
      p.run = [=] {
        reduce_grid_stride<<<grid, kBlock>>>(in, a, n);
        reduce_grid_stride<<<1, kBlock>>>(a, result, grid);
      };
      return p;
    }
    default:
      std::fprintf(stderr, "unknown --version %d\n", version);
      std::exit(2);
  }
}

static float run_once(const Plan& p, const float* result) {
  p.run();
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());
  float got = 0;
  CUDA_CHECK(cudaMemcpy(&got, result, sizeof(float), cudaMemcpyDeviceToHost));
  return got;
}

int main(int argc, char** argv) {
  const kf::BenchArgs args =
      kf::parse_bench_args(argc, argv, "results/stage2/reduce.csv", "--version V");
  int version = 0;
  for (size_t i = 0; i < args.rest.size(); ++i) {
    if (args.rest[i] == "--version" && i + 1 < args.rest.size()) version = std::stoi(args.rest[++i]);
    else {
      std::fprintf(stderr, "unknown argument: %s\n", args.rest[i].c_str());
      return 2;
    }
  }
  const size_t n = args.n;
  if (n > (size_t(1) << 31)) {
    std::fprintf(stderr, "n must be <= 2^31\n");
    return 2;
  }
  const size_t bytes = n * sizeof(float);
  const size_t partials = (n + kBlock - 1) / kBlock + 1;
  float *d_in, *d_a, *d_b, *d_result;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_a, partials * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, partials * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
  const Plan plan = make_plan(version, d_in, n, d_a, d_b, d_result);
  std::printf("version=%s variant=%s n=%zu grid=%s block=%s\n", plan.name.c_str(),
              plan.variant.c_str(), n, plan.grid.c_str(), plan.block.c_str());
  std::vector<float> h(n);
  std::mt19937 rng(12345);

  // Test 1, exact: values in {-1, 0, +1} with fewer than 2^24 nonzeros. Every partial
  // sum in any order is an integer below 2^24 and exactly representable, so any
  // correct summation order must return the exact integer sum.
  const double p_nonzero = std::min(1.0, double(1 << 23) / n);
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  int64_t exact = 0, nonzero = 0;
  for (size_t i = 0; i < n; ++i) {
    float v = 0.0f;
    if (coin(rng) < p_nonzero) v = (rng() & 1) ? 1.0f : -1.0f;
    h[i] = v;
    exact += static_cast<int64_t>(v);
    nonzero += v != 0.0f;
  }
  if (nonzero >= (1 << 24)) {
    std::fprintf(stderr, "exact test setup error: %lld nonzeros\n", (long long)nonzero);
    return 1;
  }
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));
  const float got_exact = run_once(plan, d_result);
  const bool pass_exact = static_cast<double>(got_exact) == static_cast<double>(exact);
  std::printf("correctness exact: %s (got %.1f expected %lld, nonzeros %lld)\n",
              pass_exact ? "PASS" : "FAIL", got_exact, (long long)exact, (long long)nonzero);

  // Test 2, tolerance: uniform [-1, 1). Reference is the double sum. Bound
  // 2 * ceil(log2 n) * 2^-24 * sum|x| covers tree orders of depth log2 n with margin;
  // v1's sequential atomic order has a larger worst-case bound and is held to the
  // same limit empirically.
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  double ref = 0.0, abs_sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    h[i] = dist(rng);
    ref += h[i];
    abs_sum += std::fabs(h[i]);
  }
  const double tol = 2.0 * std::ceil(std::log2(double(n) > 1 ? double(n) : 2.0)) * std::ldexp(1.0, -24) * abs_sum;
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));
  const float got = run_once(plan, d_result);
  const double err = std::fabs(static_cast<double>(got) - ref);
  const bool pass_tol = err <= tol;
  std::printf("correctness tolerance: %s (got %.6f ref %.6f abs_err %.3g tol %.3g)\n",
              pass_tol ? "PASS" : "FAIL", got, ref, err, tol);

  const bool pass = pass_exact && pass_tol;
  if (pass) {
    kf::Record r;
    r.kernel = plan.name;
    r.variant = plan.variant;
    r.n = n;
    r.bytes = static_cast<double>(bytes);  // input read once; partial sums not counted
    r.flops = static_cast<double>(n - 1);
    r.grid = plan.grid;
    r.block = plan.block;
    r.ki = plan.ki;
    r.s = kf::bench(plan.run, args);
    kf::report(r, args);
  }
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_result));
  return pass ? 0 : 1;
}
