#include <cstring>
#include <string>
#include <vector>
#include "kf/bench.cuh"
#include "kf/check.hpp"

__global__ void copy_kernel(const float* __restrict__ src, float* __restrict__ dst, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

// Modes: --mode kernel (default) times copy_kernel; --mode memcpy times
// cudaMemcpyAsync device-to-device, which uses the driver's own copy path.
int main(int argc, char** argv) {
  const kf::BenchArgs args =
      kf::parse_bench_args(argc, argv, "results/stage0/copy.csv", "[--mode kernel|memcpy]");
  std::string mode = "kernel";
  for (size_t i = 0; i < args.rest.size(); ++i) {
    if (args.rest[i] == "--mode" && i + 1 < args.rest.size()) mode = args.rest[++i];
    else {
      std::fprintf(stderr, "unknown argument: %s\n", args.rest[i].c_str());
      return 2;
    }
  }
  if (mode != "kernel" && mode != "memcpy") {
    std::fprintf(stderr, "--mode must be kernel or memcpy\n");
    return 2;
  }
  const size_t n = args.n;
  constexpr int kBlock = 256;
  const unsigned grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
  const size_t bytes = n * sizeof(float);

  // Distinct values per index so a shifted or partial copy is detected.
  std::vector<float> h_src(n), h_dst(n);
  for (size_t i = 0; i < n; ++i) h_src[i] = static_cast<float>(i % 16777216) + 0.25f;

  float *d_src, *d_dst;
  CUDA_CHECK(cudaMalloc(&d_src, bytes));
  CUDA_CHECK(cudaMalloc(&d_dst, bytes));
  CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_dst, 0xFF, bytes));

  auto run = [&] {
    if (mode == "kernel") copy_kernel<<<grid, kBlock>>>(d_src, d_dst, n);
    else CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, 0));
  };
  run();
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, bytes, cudaMemcpyDeviceToHost));

  const kf::CheckResult c = kf::check_close(h_dst, h_src, 0.0, 0.0, mode.c_str());
  std::printf("n=%zu mode=%s\n", n, mode.c_str());
  kf::print_check(mode.c_str(), c);

  if (c.pass()) {
    kf::Record r;
    r.kernel = mode == "kernel" ? "copy_kernel" : "memcpy_d2d";
    r.n = n;
    r.bytes = 2.0 * bytes;  // one read and one write per element
    if (mode == "kernel") {
      r.grid = std::to_string(grid);
      r.block = std::to_string(kBlock);
      r.ki = kf::kernel_info(copy_kernel, kBlock);
    } else {
      r.grid = r.block = "-";
    }
    r.s = kf::bench(run, args);
    kf::report(r, args);
  }

  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));
  return c.pass() ? 0 : 1;
}
