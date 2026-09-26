#include <random>
#include <vector>
#include "kf/bench.cuh"
#include "kf/check.hpp"

__global__ void vector_add(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ c, size_t n) {
  // size_t index: n up to 2 GiB of data overflows int arithmetic.
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
  const kf::BenchArgs args = kf::parse_bench_args(argc, argv, "results/stage0/vector_add.csv");
  if (!args.rest.empty()) {
    std::fprintf(stderr, "unknown argument: %s\n", args.rest[0].c_str());
    return 2;
  }
  const size_t n = args.n;
  constexpr int kBlock = 256;
  const unsigned grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
  const size_t bytes = n * sizeof(float);

  std::vector<float> h_a(n), h_b(n), h_c(n), ref(n);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) {
    h_a[i] = dist(rng);
    h_b[i] = dist(rng);
    ref[i] = h_a[i] + h_b[i];
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
  const kf::CheckResult c = kf::check_close(h_c, ref, 1e-6, 1e-7, "vector_add");
  std::printf("n=%zu block=%d grid=%u\n", n, kBlock, grid);
  kf::print_check("vector_add", c);

  if (c.pass()) {
    kf::Record r;
    r.kernel = "vector_add";
    r.n = n;
    r.bytes = 3.0 * bytes;  // two reads and one write per element
    r.grid = std::to_string(grid);
    r.block = std::to_string(kBlock);
    r.ki = kf::kernel_info(vector_add, kBlock);
    r.s = kf::bench([&] { vector_add<<<grid, kBlock>>>(d_a, d_b, d_c, n); }, args);
    kf::report(r, args);
  }

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return c.pass() ? 0 : 1;
}
