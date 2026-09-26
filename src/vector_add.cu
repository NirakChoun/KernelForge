#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "kf/cuda_check.cuh"
#include "kf/timer.cuh"

__global__ void vector_add(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ c, size_t n) {
  // size_t index: n up to 2 GiB of data later in Stage 0 overflows int arithmetic.
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <n>\n", argv[0]);
    return 2;
  }
  const size_t n = std::strtoull(argv[1], nullptr, 10);
  if (n == 0) {
    std::fprintf(stderr, "n must be > 0\n");
    return 2;
  }
  constexpr int kBlock = 256;
  constexpr int kWarmup = 10;
  constexpr int kReps = 100;
  const unsigned grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
  const size_t bytes = n * sizeof(float);

  std::vector<float> h_a(n), h_b(n), h_c(n);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) {
    h_a[i] = dist(rng);
    h_b[i] = dist(rng);
  }

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
  // Sentinel fill so an untouched output element fails the check.
  CUDA_CHECK(cudaMemset(d_c, 0xFF, bytes));

  vector_add<<<grid, kBlock>>>(d_a, d_b, d_c, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  // A single float add is correctly rounded on both sides, so results should match
  // exactly; the tolerance only guards against compiler contraction differences.
  size_t errors = 0;
  double max_err = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const float ref = h_a[i] + h_b[i];
    const double err = std::fabs(static_cast<double>(h_c[i]) - ref);
    if (!(err <= 1e-6 * std::fabs(ref) + 1e-7)) {  // !(<=) also catches NaN
      if (errors < 5)
        std::fprintf(stderr, "mismatch at %zu: got %.9g expected %.9g\n", i, h_c[i], ref);
      ++errors;
    }
    if (err > max_err) max_err = err;
  }
  const bool pass = errors == 0;
  std::printf("n=%zu block=%d grid=%u\n", n, kBlock, grid);
  std::printf("correctness: %s (errors=%zu, max_abs_err=%.3g)\n", pass ? "PASS" : "FAIL",
              errors, max_err);

  if (pass) {
    auto times = kf::time_gpu([&] { vector_add<<<grid, kBlock>>>(d_a, d_b, d_c, n); },
                              kWarmup, kReps);
    const kf::Stats s = kf::summarize(times);
    // Two reads and one write per element.
    const double gbs = 3.0 * bytes / (s.median_ms * 1e-3) / 1e9;
    std::printf("warmup=%d reps=%d median_ms=%.6f min_ms=%.6f stddev_ms=%.6f bandwidth_GBps=%.1f\n",
                kWarmup, kReps, s.median_ms, s.min_ms, s.stddev_ms, gbs);
  }

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return pass ? 0 : 1;
}
