#include <random>
#include <vector>
#include "kf/bench.cuh"
#include "kf/check.hpp"

// y = a * x + y, in place.
__global__ void saxpy(float a, const float* __restrict__ x, float* __restrict__ y, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main(int argc, char** argv) {
  const kf::BenchArgs args = kf::parse_bench_args(argc, argv, "results/stage0/saxpy.csv");
  if (!args.rest.empty()) {
    std::fprintf(stderr, "unknown argument: %s\n", args.rest[0].c_str());
    return 2;
  }
  const size_t n = args.n;
  constexpr int kBlock = 256;
  constexpr float kA = 1.5f;
  const unsigned grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
  const size_t bytes = n * sizeof(float);

  std::vector<float> h_x(n), h_y(n), h_out(n);
  std::vector<double> ref(n);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) {
    h_x[i] = dist(rng);
    h_y[i] = dist(rng);
    ref[i] = double(kA) * h_x[i] + h_y[i];  // exact in double
  }

  float *d_x, *d_y;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

  saxpy<<<grid, kBlock>>>(kA, d_x, d_y, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  // The GPU result is either one FMA rounding or two separate roundings of the exact
  // value, so it is within 2 ulp of |a*x| + |y|, which is at most 2.5.
  const kf::CheckResult c = kf::check_close(h_out, ref, 0.0, 2.5 * 2 * 1.2e-7, "saxpy");
  std::printf("n=%zu block=%d grid=%u a=%g\n", n, kBlock, grid, kA);
  kf::print_check("saxpy", c);

  if (c.pass()) {
    kf::Record r;
    r.kernel = "saxpy";
    r.n = n;
    r.bytes = 3.0 * bytes;  // read x, read y, write y
    r.flops = 2.0 * n;
    r.grid = std::to_string(grid);
    r.block = std::to_string(kBlock);
    r.ki = kf::kernel_info(saxpy, kBlock);
    // Repeated in-place updates grow y by at most 1.5 per launch; stays finite.
    r.s = kf::bench([&] { saxpy<<<grid, kBlock>>>(kA, d_x, d_y, n); }, args);
    kf::report(r, args);
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return c.pass() ? 0 : 1;
}
